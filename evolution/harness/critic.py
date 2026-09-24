"""Advisory visual failure analysis; never a reviewer or promotion gate."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from racap.backends.llm import ask
from racap.backends.vlm import encode_png

from .prompting import read_prompt
from .schema import CriticReport

CRITIC_ROLE_PROMPT = read_prompt("critic.md")


def _compact(value: object, *, depth: int = 0) -> object:
    """Bound verbose arrays while preserving causal tool and evaluator fields."""
    # Candidate-owned reports add one wrapper named after the newly evolved
    # mechanism.  Eight levels retain small pose / verification payloads below
    # that wrapper while the explicit tensor-key filter still removes images.
    if depth > 8:
        return "<nested>"
    if isinstance(value, dict):
        return {
            str(key): _compact(item, depth=depth + 1)
            for key, item in value.items()
            if str(key)
            not in {
                # Camera tensors are already represented by the separately
                # attached contact sheet.  Keeping them in a tool report can
                # turn one critic request into hundreds of megabytes without
                # adding any causal information.
                "images",
                "rgb",
                "depth",
                "segmentation",
                "native_scene_poses_initial",
                "native_scene_poses_final",
                "destination_region_boundary_pixels",
            }
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 24:
            return [*(_compact(item, depth=depth + 1) for item in value[:12]), f"<{len(value)-12} more>"]
        return [_compact(item, depth=depth + 1) for item in value]
    if isinstance(value, np.ndarray):
        return _compact(value.tolist(), depth=depth + 1)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compact_human_trace(trace: str, *, max_chars: int = 80_000) -> str:
    """Keep the readable action trace while removing duplicated raw reports.

    ``trace.md`` contains a concise observation and key-evidence section, then
    repeats the complete JSON tool report in an HTML details block.  A public
    observation may itself contain camera tensors, so that duplicate block can
    exceed the model gateway request limit.  The structured causal packet
    already carries a redacted copy of the report; the contact sheet carries
    the pixels.
    """
    compact = re.sub(
        r"<details><summary>Full tool report</summary>.*?</details>",
        "<details><summary>Full tool report omitted; see causal packet</summary></details>",
        trace,
        flags=re.DOTALL,
    )
    if len(compact) <= max_chars:
        return compact
    half = max_chars // 2
    omitted = len(compact) - (2 * half)
    return (
        compact[:half]
        + f"\n\n[... {omitted} trace characters omitted ...]\n\n"
        + compact[-half:]
    )


def _causal_packet(trajectory: dict) -> dict:
    """Align agent choices, physical reports and evaluator-only aftermath.

    Native diagnostics are supplied only to the offline critic. Candidate code
    still receives the public runtime and can never query these fields online.
    """
    steps = []
    for raw in trajectory.get("steps") or []:
        if not isinstance(raw, dict):
            continue
        report = raw.get("report")
        action = raw.get("action")
        # Mature ReAct steps use the standard action/report schema.  An evolved
        # candidate may instead emit {"new_mechanism": {...}} so its public
        # evidence remains self-describing.  Preserve that payload rather than
        # silently presenting the visual critic with an empty tool report.
        if not isinstance(report, dict) or not report:
            custom = {
                str(key): value
                for key, value in raw.items()
                if str(key) not in {"turn", "thought", "action", "args", "observation", "report"}
            }
            if custom:
                report = custom
                if action is None and len(custom) == 1:
                    action = next(iter(custom))
        if not isinstance(report, dict):
            report = {"value": report}
        steps.append(
            {
                "turn": raw.get("turn"),
                "thought": raw.get("thought"),
                "action": action,
                "args": raw.get("args"),
                "observation": raw.get("observation"),
                "report": report,
            }
        )
    return _compact(
        {
            "instruction": trajectory.get("instruction"),
            "native_success": trajectory.get("native_success"),
            "agent_success": trajectory.get("agent_success"),
            "stop_reason": trajectory.get("stopped") or trajectory.get("error"),
            "simulator_steps": trajectory.get("simulator_steps"),
            "tool_steps": steps,
            "native_predicates": trajectory.get("native_predicates"),
            "native_predicate_diagnostics": trajectory.get("native_predicate_diagnostics"),
            "public_runtime_calls": trajectory.get("evaluator_runtime_calls")
            or trajectory.get("public_runtime_calls"),
            "llm_calls": trajectory.get("evaluator_llm_calls"),
        }
    )


def _event_frame_candidates(value: object) -> set[int]:
    """Collect evaluator-owned frame markers without assuming one trace schema."""
    found: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if isinstance(item, int) and (
                "frame" in normalized or normalized in {"start", "end"}
            ):
                found.add(item)
            elif isinstance(item, (dict, list, tuple)):
                found.update(_event_frame_candidates(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_event_frame_candidates(item))
    return found


def _sample_video(
    path: Path, count: int = 12, trajectory: dict | None = None
) -> tuple[np.ndarray, tuple[int, ...]]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        length = reader.count_frames()
        uniform = set(
            np.linspace(0, max(0, length - 1), count).astype(int).tolist()
        )
        event_aligned: set[int] = set()
        for marker in _event_frame_candidates(trajectory or {}):
            for offset in (-2, 0, 2):
                event_aligned.add(max(0, min(length - 1, marker + offset)))
        # Preserve global context, then add action boundaries.  If a trace has
        # hundreds of markers, subsample them deterministically rather than
        # silently dropping the end of the rollout.
        if len(event_aligned) > count:
            ordered = sorted(event_aligned)
            positions = np.linspace(0, len(ordered) - 1, count).astype(int)
            event_aligned = {ordered[index] for index in positions}
        indices = tuple(sorted(uniform | event_aligned))
        return np.stack([reader.get_data(index) for index in indices]), indices
    finally:
        reader.close()


def _contact_sheet(frames: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    if not len(frames):
        raise ValueError("cannot build a contact sheet without frames")
    thumbs: list[Image.Image] = []
    for frame, index in zip(frames, indices):
        image = Image.fromarray(np.asarray(frame).astype("uint8")).convert("RGB")
        image.thumbnail((320, 240))
        canvas = Image.new("RGB", (320, 268), "white")
        canvas.paste(image, ((320 - image.width) // 2, 24))
        ImageDraw.Draw(canvas).text((8, 5), f"frame {index}", fill="black")
        thumbs.append(canvas)
    columns = min(4, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 320, rows * 268), (225, 225, 225))
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % columns) * 320, (index // columns) * 268))
    return np.asarray(sheet)


def _json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("critic response contained no JSON object")


class VisualCritic:
    """Turn video + human trace into a structured, explicitly uncertain diagnosis."""

    def __init__(
        self,
        model: str,
        *,
        query: Callable[..., str] = ask,
        frame_count: int = 12,
        parallelism: int = 4,
    ):
        self.model = model
        self.query = query
        self.frame_count = frame_count
        self.parallelism = max(1, int(parallelism))

    def analyze(self, episode_dir: Path) -> CriticReport:
        trajectory = json.loads((episode_dir / "trajectory.json").read_text(encoding="utf-8"))
        trace_path = episode_dir / "trace.md"
        trace = (
            _compact_human_trace(trace_path.read_text(encoding="utf-8"))
            if trace_path.is_file()
            else ""
        )
        video = episode_dir / "rollout.mp4"
        images: list[str] = []
        indices: tuple[int, ...] = ()
        if video.is_file():
            frames, indices = _sample_video(video, self.frame_count, trajectory)
            images.append(encode_png(_contact_sheet(frames, indices)))
        causal_packet = _causal_packet(trajectory)
        prompt = f"""
{CRITIC_ROLE_PROMPT}

Instruction: {trajectory.get("instruction", "")}
Native success after episode: {bool(trajectory.get("native_success"))}
Agent claimed success: {bool(trajectory.get("agent_success"))}
Stop reason: {trajectory.get("stopped") or trajectory.get("error") or "-"}
Available sampled frames: {list(indices)}

Action-aligned causal packet. Native predicate diagnostics are offline
evaluator evidence only; do not recommend candidate access to them at runtime:
{json.dumps(causal_packet, indent=2, ensure_ascii=False)}

Human-readable trace:
{trace}

Return one JSON object:
{{
  "failure_phase": "...",
  "first_divergence": "earliest action/frame where behavior departs from the intended mechanism",
  "failure_mechanism": "grounding|contact|grasp|transit|placement_geometry|release|articulation_path|verification|planning|budget|unknown",
  "phase_evidence": [
    {{"phase": "...", "observation": "...", "source": "frame|tool_report|motion|native_aftermath"}}
  ],
  "causal_timeline": ["expected state", "first visible divergence", "downstream consequence"],
  "visual_evidence": ["..."],
  "hypothesis": "...",
  "alternative_hypotheses": ["plausible alternative and why it is weaker"],
  "confidence": 0.0,
  "counterfactual": "what observation would falsify this diagnosis",
  "recommended_observation": "one cheapest next observation that separates the hypotheses",
  "recommended_instrumentation": ["public report field needed to disambiguate execution next time"],
  "generality_scope": "task family affected",
  "suggested_layer": "policy_api|agent|memory|perception|unknown",
  "experience_lesson": {{
    "condition": "when this mechanism applies; no task ids or coordinates",
    "antipattern": "what failed strategy to avoid",
    "remedy": "an adjustable, code-level strategy to try",
    "applicable_tags": ["skill", "geometry", "failure mode"]
  }}
}}
""".strip()
        response = self.query(
            "Use visual and trajectory evidence to diagnose robot rollout failures.",
            prompt,
            images=images,
            model=self.model,
            max_tokens=4096,
            temperature=0.0,
            attempts=3,
            cache=True,
        )
        value = _json_object(response)
        evidence = value.get("visual_evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
        timeline = value.get("causal_timeline") or []
        if isinstance(timeline, str):
            timeline = [timeline]
        alternatives = value.get("alternative_hypotheses") or []
        if isinstance(alternatives, str):
            alternatives = [alternatives]
        lesson = value.get("experience_lesson") or {}
        tags = lesson.get("applicable_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        phase_evidence = value.get("phase_evidence") or []
        if isinstance(phase_evidence, dict):
            phase_evidence = [phase_evidence]
        instrumentation = value.get("recommended_instrumentation") or []
        if isinstance(instrumentation, str):
            instrumentation = [instrumentation]
        return CriticReport(
            episode_key=str(trajectory.get("key", episode_dir.name)),
            failure_phase=str(value.get("failure_phase", "unknown")),
            visual_evidence=tuple(str(item) for item in evidence),
            hypothesis=str(value.get("hypothesis", "")),
            confidence=confidence,
            counterfactual=str(value.get("counterfactual", "")),
            generality_scope=str(value.get("generality_scope", "unknown")),
            suggested_layer=str(value.get("suggested_layer", "unknown")),
            source_frames=indices,
            causal_timeline=tuple(str(item) for item in timeline),
            alternative_hypotheses=tuple(str(item) for item in alternatives),
            recommended_observation=str(value.get("recommended_observation", "")),
            lesson_condition=str(lesson.get("condition", "")),
            lesson_antipattern=str(lesson.get("antipattern", "")),
            lesson_remedy=str(lesson.get("remedy", "")),
            applicable_tags=tuple(str(item) for item in tags),
            raw_response=response,
            first_divergence=str(value.get("first_divergence", "")),
            failure_mechanism=str(value.get("failure_mechanism", "unknown")),
            phase_evidence=tuple(
                {str(key): _compact(item) for key, item in evidence_item.items()}
                for evidence_item in phase_evidence
                if isinstance(evidence_item, dict)
            ),
            recommended_instrumentation=tuple(str(item) for item in instrumentation),
        )

    def analyze_failures(
        self, output_dir: Path, limit: int | None = None
    ) -> tuple[CriticReport, ...]:
        records = [
            json.loads(line)
            for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        failures = [row for row in records if not bool(row.get("native_success"))]
        if limit is not None:
            failures = failures[:limit]
        episode_dirs: list[Path] = []
        for row in failures:
            artifacts = row.get("artifacts") or {}
            episode_dir = Path(artifacts.get("episode_dir", output_dir / "episodes" / "missing"))
            if episode_dir.is_dir():
                episode_dirs.append(episode_dir)
        if not episode_dirs:
            return ()
        # Each report is advisory and independent.  Parallel network requests
        # preserve input order through ``map`` while avoiding minutes of idle
        # serial latency on a cohort of visually similar failures.
        with ThreadPoolExecutor(
            max_workers=min(self.parallelism, len(episode_dirs)),
            thread_name_prefix="racap-critic",
        ) as pool:
            return tuple(pool.map(self.analyze, episode_dirs))

    def compare_changed(
        self, paired_delta: dict, limit: int | None = 6
    ) -> tuple[dict, ...]:
        """Explain new wins and regressions from paired visual evidence.

        This is deliberately post-evaluation advice for the next iteration; it
        cannot approve a patch or alter the native promotion decision.
        """
        changed = list(paired_delta.get("changed_episodes") or [])
        if limit is not None:
            changed = changed[:limit]
        reports: list[dict] = []
        for item in changed:
            old_value = (item.get("champion") or {}).get("episode_dir")
            new_value = (item.get("candidate") or {}).get("episode_dir")
            if not old_value or not new_value:
                continue
            old_dir = Path(old_value)
            new_dir = Path(new_value)
            if not old_dir.is_dir() or not new_dir.is_dir():
                continue
            images: list[str] = []
            frame_manifest: dict[str, list[int]] = {}
            traces: dict[str, str] = {}
            for name, directory in (("champion", old_dir), ("candidate", new_dir)):
                trajectory_path = directory / "trajectory.json"
                trajectory = (
                    json.loads(trajectory_path.read_text(encoding="utf-8"))
                    if trajectory_path.is_file()
                    else {}
                )
                video = directory / "rollout.mp4"
                if video.is_file():
                    frames, indices = _sample_video(video, self.frame_count, trajectory)
                    images.append(encode_png(_contact_sheet(frames, indices)))
                    frame_manifest[name] = list(indices)
                trace_path = directory / "trace.md"
                traces[name] = (
                    trace_path.read_text(encoding="utf-8") if trace_path.is_file() else ""
                )
            prompt = f"""
{CRITIC_ROLE_PROMPT}

This is a paired comparison for {item.get('key')}: {item.get('change')}.
The first image is the champion contact sheet; the second is the candidate.
Frame manifest: {json.dumps(frame_manifest)}

Champion trace:
{traces.get('champion', '')}

Candidate trace:
{traces.get('candidate', '')}

Explain the earliest visible behavioral difference. Separate changes caused by
the patch from stochastic or visually unsupported explanations. Return JSON:
{{
  "episode_key": "{item.get('key')}",
  "change": "{item.get('change')}",
  "first_divergence": "...",
  "champion_evidence": ["..."],
  "candidate_evidence": ["..."],
  "causal_explanation": "...",
  "confidence": 0.0,
  "alternative_explanation": "...",
  "next_experiment": "how to test whether this mechanism generalizes"
}}
""".strip()
            response = self.query(
                "Compare paired robot rollouts and identify the causal behavior change.",
                prompt,
                images=images,
                model=self.model,
                max_tokens=4096,
                temperature=0.0,
                attempts=3,
                cache=True,
            )
            value = _json_object(response)
            value["source_frame_manifest"] = frame_manifest
            value["raw_response"] = response
            reports.append(value)
        return tuple(reports)
