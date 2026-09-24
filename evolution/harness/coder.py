"""Coding-agent prompt construction and proposal parsing."""

from __future__ import annotations

import json
from functools import lru_cache
import os
import re
from pathlib import Path
from typing import Callable

from racap.backends.llm import ask
from racap.backends.vlm import encode_png

from .constitution import BOUNDARY_RUBRIC, CONSTITUTION
from .critic import _contact_sheet, _sample_video
from .prompting import read_prompt
from .public_runtime import PUBLIC_RUNTIME_SPEC
from .schema import CriticReport, Metrics, Proposal, StageSpec


class ProposalError(RuntimeError):
    pass


CODER_ROLE_PROMPT = read_prompt("coder.md")
EXPERIMENT_PROTOCOL = read_prompt("experiment_protocol.md")


def _decode_whole_file_escapes(content: str) -> str:
    """Recover a model response that JSON-escaped a file twice.

    The signature is deliberately narrow: a source file has no real newline
    but contains several ``\\n`` sequences.  The small JSON-style decoder keeps
    non-ASCII text intact and correctly distinguishes ``\\n`` (newline) from
    ``\\\\n`` (a literal backslash-n inside the generated source).
    """
    if "\n" in content or content.count("\\n") < 3:
        return content
    decoded: list[str] = []
    index = 0
    escapes = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "'": "'", "/": "/"}
    while index < len(content):
        if content[index] != "\\":
            decoded.append(content[index])
            index += 1
            continue
        end = index
        while end < len(content) and content[end] == "\\":
            end += 1
        count = end - index
        decoded.append("\\" * (count // 2))
        if count % 2 == 0 or end >= len(content):
            if count % 2 and end >= len(content):
                decoded.append("\\")
            index = end
            continue
        marker = content[end]
        replacement = escapes.get(marker)
        if replacement is None:
            decoded.extend(("\\", marker))
        else:
            decoded.append(replacement)
        index = end + 1
    result = "".join(decoded)
    return result if result.count("\n") >= 3 and "\x00" not in result else content


def _extract_json(text: str) -> dict:
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ProposalError("coding model did not return a JSON object")


def parse_proposal(text: str) -> Proposal:
    value = _extract_json(text)
    patch = str(value.get("patch") or "").strip()
    raw_files = value.get("files") or {}
    if not isinstance(raw_files, dict):
        raise ProposalError("proposal.files must be an object mapping paths to contents or null")
    files: dict[str, str | None] = {}
    for raw_path, content in raw_files.items():
        path = Path(str(raw_path))
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] not in {"solution", "memory", "tests"}
        ):
            raise ProposalError(f"proposal.files contains unsafe candidate path: {raw_path!r}")
        if content is not None and not isinstance(content, str):
            raise ProposalError(f"proposal.files[{raw_path!r}] must be text or null")
        files[path.as_posix()] = (
            _decode_whole_file_escapes(content) if isinstance(content, str) else content
        )
    if patch and patch.lstrip().startswith("*** Begin Patch"):
        # Hosted coding models frequently use the Codex context-patch format
        # even when the JSON schema requests a Git diff. WorkspaceManager has
        # a path-safe exact-context adapter for this format. Route the payload
        # through the files channel using a virtual, never-written key so a
        # mechanically valid repair can reach compilation and runtime.
        virtual = "solution/.racap-context-patch"
        suffix = 1
        while virtual in files:
            virtual = f"solution/.racap-context-patch-{suffix}"
            suffix += 1
        files[virtual] = patch
        patch = ""
    if patch and not patch.startswith("diff --git"):
        # Some coding responses provide complete-file edits plus a prose or
        # fenced pseudo-patch. The file map is already path-validated and is a
        # complete implementation channel, so an unusable redundant patch
        # should not discard otherwise executable code.
        if files:
            patch = ""
        else:
            raise ProposalError("proposal.patch must be a git unified diff when supplied")
    if not patch and not files:
        raise ProposalError("proposal requires either patch or complete-file edits")

    # Experiment-card prose is useful for lineage analysis, but it is not an
    # executable safety property. Hosted coding models occasionally return a
    # complete implementation while omitting one of these descriptive fields.
    # Do not discard runnable code before syntax/tests/runtime can judge it;
    # preserve the omission explicitly in the ledger via conservative defaults.
    target = str(value.get("target_failure_cluster") or "").strip()
    mechanism = str(value.get("mechanism") or "").strip()

    def _metadata(key: str, fallback: str) -> str:
        result = str(value.get(key) or "").strip()
        return result or fallback

    title = _metadata(
        "title",
        f"Executable candidate for {target}" if target else "Executable candidate",
    )
    hypothesis = _metadata(
        "hypothesis",
        mechanism
        or "The supplied implementation may improve the observed failure cohort; "
        "paired runtime evaluation is the deciding evidence.",
    )
    predicted_effect = _metadata(
        "predicted_effect",
        "Unspecified by the coding response; measure native-success delta on the "
        "pre-registered paired cohort.",
    )
    risk = _metadata(
        "risk",
        "Unspecified by the coding response; retain only if paired evaluation "
        "shows a strict native-success improvement.",
    )

    def _strings(key: str) -> tuple[str, ...]:
        raw = value.get(key) or []
        if isinstance(raw, str):
            raw = [raw]
        return tuple(str(item) for item in raw)

    return Proposal(
        title=title,
        hypothesis=hypothesis,
        predicted_effect=predicted_effect,
        risk=risk,
        patch=patch,
        target_failure_cluster=str(value.get("target_failure_cluster", "")),
        evidence=_strings("evidence"),
        mechanism=str(value.get("mechanism", "")),
        expected_wins=_strings("expected_wins"),
        regression_risks=_strings("regression_risks"),
        falsification_test=str(value.get("falsification_test", "")),
        raw_response=text,
        files=files,
    )


def _source_snapshot(repository: Path) -> str:
    chunks: list[str] = []
    for path in sorted(repository.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(repository)
        if relative.parts[0] not in {"solution", "memory", "tests"} and relative.name not in {
            "README.md",
            "RUNTIME_SDK.md",
        }:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        chunks.append(f"\n===== {relative} =====\n{content}")
    return "".join(chunks)


def _critic_view(report: CriticReport) -> dict:
    """Keep structured evidence; omit the duplicated provider response."""
    return {
        key: value
        for key, value in report.to_dict().items()
        if key != "raw_response"
    }


_RAW_OBSERVATION_KEYS = {
    "images",
    "rgb",
    "depth",
    "segmentation",
    "native_scene_poses_initial",
    "native_scene_poses_final",
}


def _nested_runtime_errors(
    value: object, *, path: str = "", depth: int = 0
) -> list[str]:
    """Extract deterministic software failures without traversing pixels."""
    if depth > 12:
        return []
    if isinstance(value, dict):
        found: list[str] = []
        for key, item in value.items():
            name = str(key)
            if name in _RAW_OBSERVATION_KEYS:
                continue
            child_path = f"{path}.{name}" if path else name
            if name.lower() in {"error", "exception"} and item:
                found.append(f"{child_path}: {item}")
            elif isinstance(item, (dict, list, tuple)):
                found.extend(
                    _nested_runtime_errors(item, path=child_path, depth=depth + 1)
                )
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for index, item in enumerate(value):
            if isinstance(item, (dict, list, tuple)):
                found.extend(
                    _nested_runtime_errors(
                        item, path=f"{path}[{index}]", depth=depth + 1
                    )
                )
        return found
    return []


@lru_cache(maxsize=32)
def _runtime_diagnostics(records_path: str, limit: int = 16) -> tuple[dict, ...]:
    """Read evaluator-owned nested exceptions from a completed rollout."""
    path = Path(records_path)
    if not path.is_file():
        return ()
    diagnostics: list[dict] = []
    seen: set[tuple[str, str]] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                episode = str(row.get("key") or "unknown")
                for error in _nested_runtime_errors(
                    row.get("steps") or [], path="steps"
                ):
                    key = (episode, error)
                    if key in seen:
                        continue
                    seen.add(key)
                    diagnostics.append({"episode": episode, "error": error})
                    if len(diagnostics) >= limit:
                        return tuple(diagnostics)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ()
    return tuple(diagnostics)


def _recent_experiment_history(history: list[dict], limit: int = 8) -> list[dict]:
    """Keep behavior evidence while leaving request failures in the event log.

    A run of provider or response-format failures never produced a controller
    candidate. Including those rows as the entire recent context can evict all
    paired rollout evidence and mislead the next coding request.
    """
    evidence = [
        record for record in history if record.get("status") != "proposal_failed"
    ]
    recent: list[dict] = []
    for record in evidence[-limit:]:
        item = dict(record)
        metrics = item.get("candidate_metrics") or {}
        records_path = metrics.get("records_path") if isinstance(metrics, dict) else None
        if records_path:
            diagnostics = _runtime_diagnostics(str(records_path))
            if diagnostics:
                item["runtime_diagnostics"] = list(diagnostics)
        recent.append(item)
    return recent


def _champion_visuals(
    champion: Metrics, *, limit: int = 4, frame_count: int = 9
) -> tuple[list[str], list[dict]]:
    """Give the coder representative first-hand visuals plus all critic prose.

    The visual critic already analyzes up to eight failures at twelve frames
    each.  Reattaching every full contact sheet to the code-generation request
    duplicates several megabytes of evidence and can make hosted endpoints
    time out before producing a patch.  Four deterministic raw examples keep
    the coder visually grounded while the structured critic reports preserve
    coverage of the remaining failure cohort.
    """
    records_path = Path(champion.records_path)
    if not records_path.is_file():
        return [], []
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    images: list[str] = []
    manifest: list[dict] = []
    for row in rows:
        if bool(row.get("native_success")):
            continue
        episode_value = (row.get("artifacts") or {}).get("episode_dir")
        if not episode_value:
            continue
        episode_dir = Path(episode_value)
        video = episode_dir / "rollout.mp4"
        if not video.is_file():
            continue
        try:
            trajectory_path = episode_dir / "trajectory.json"
            trajectory = (
                json.loads(trajectory_path.read_text(encoding="utf-8"))
                if trajectory_path.is_file()
                else {}
            )
            frames, indices = _sample_video(video, frame_count, trajectory)
        except Exception:
            # Visuals are advisory. A corrupt recording must not block a
            # syntactically valid evolution iteration; the trace/critic remain.
            continue
        images.append(encode_png(_contact_sheet(frames, indices)))
        manifest.append(
            {
                "image_index": len(images) - 1,
                "episode_key": row.get("key"),
                "frames": list(indices),
                "episode_dir": str(episode_dir),
            }
        )
        if len(images) >= limit:
            break
    return images, manifest


class CodingAgent:
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 65536,
        query: Callable[..., str] = ask,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.query = query

    def propose(
        self,
        repository: Path,
        stage: StageSpec,
        champion: Metrics,
        critics: tuple[CriticReport, ...],
        history: list[dict],
        capability_memory: dict | None = None,
    ) -> Proposal:
        images, visual_manifest = _champion_visuals(champion)
        prompt = f"""
{CODER_ROLE_PROMPT}

{EXPERIMENT_PROTOCOL}

{CONSTITUTION}

Current curriculum stage:
{json.dumps(stage.to_dict(), indent=2, ensure_ascii=False)}

Boundary metrics (directional design targets, not hard gates):
{json.dumps(BOUNDARY_RUBRIC, indent=2, ensure_ascii=False)}

Stable non-privileged runtime SDK. This substrate is fixed and mechanism-neutral;
use it to implement new physical APIs instead of assuming missing backend calls:
{json.dumps(PUBLIC_RUNTIME_SPEC, indent=2, ensure_ascii=False)}

Champion metrics:
{json.dumps(champion.to_dict(), indent=2, ensure_ascii=False)}

Advisory visual critic reports:
{json.dumps([_critic_view(report) for report in critics], indent=2, ensure_ascii=False)}

First-hand champion failure contact sheets attached to this request:
{json.dumps(visual_manifest, indent=2, ensure_ascii=False)}
Use these images to verify or reject critic hypotheses. Image indices follow
the attachment order; frame labels inside each sheet are rollout frame numbers.

Recent experiment history:
{json.dumps(_recent_experiment_history(history), indent=2, ensure_ascii=False)}

Evaluator-owned capability map and experiment ledger summary:
{json.dumps(capability_memory or {}, indent=2, ensure_ascii=False)}

Current solution source:
{_source_snapshot(repository)}

Return exactly one JSON object with keys:
- title, hypothesis, predicted_effect, risk (required)
- target_failure_cluster, evidence, mechanism, expected_wins,
  regression_risks, falsification_test (required by the experiment protocol;
  they are recorded for causal analysis, not used as promotion gates).
- files: an object mapping candidate-relative paths to complete UTF-8 contents,
  or null to delete a file. Prefer this robust format for generated code.
- patch: optional complete git unified diff. Use it only when it is less error-prone
  than complete-file edits.

Candidate paths may live only under solution/, memory/ and tests/. You may add,
remove, split or merge Policy APIs and agent modules, update memory, and add
tests. There is no line-count or patch-size limit. Keep the public entry point
solution.controller.run_episode(runtime, episode).

The champion's tests are editable behavioral evidence. Preserve true public
interfaces, safety properties, and unrelated regressions. When this proposal
intentionally supersedes an incidental negative routing assertion left by an
older capability (such as "this instruction must use fallback" solely because
no dedicated route existed), update only that stale assertion to the new
explicit contract and add an entry-point positive test. Do not delete or
broadly weaken tests merely to make the guard green.

IMPORTANT serialization rule: every `files` value is normally written
byte-for-byte as the complete destination file. Never put `*** Begin Patch`,
`*** Update File`, `@@`, or a git diff inside a `files` value. If returning a
patch, place one valid unified diff beginning with `diff --git` in `patch`.
""".strip()
        response = self.query(
            "Write a generalizable robot-controller improvement backed by rollout evidence.",
            prompt,
            images=images,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.2,
            attempts=3,
            cache=False,
            timeout_s=float(os.environ.get("RACAP_CODER_TIMEOUT_S", "600")),
        )
        return parse_proposal(response)

    def repair(
        self,
        repository: Path,
        stage: StageSpec,
        proposal: Proposal,
        diagnostics: dict,
        *,
        capability_memory: dict | None = None,
    ) -> Proposal:
        """Let the same implementation agent debug an invalid candidate.

        This is not a reviewer: no second model approves the design. The coder
        receives mechanical apply/compile/test diagnostics and emits the next
        repair relative to the cumulative failed candidate snapshot.
        """
        prompt = f"""
{CODER_ROLE_PROMPT}

The previous implementation did not reach simulator evaluation. Preserve its
evidence-backed hypothesis, but repair the implementation using the exact
diagnostics below. The source snapshot is the cumulative failed candidate, not
the original champion. Return only the edits needed on top of that snapshot;
do not repeat or remove already-correct parts of the causal mechanism.

Stage:
{json.dumps(stage.to_dict(), indent=2, ensure_ascii=False)}

Stable public runtime SDK:
{json.dumps(PUBLIC_RUNTIME_SPEC, indent=2, ensure_ascii=False)}

Previous experiment card:
{json.dumps({key: value for key, value in proposal.to_dict().items() if key not in {'patch', 'files', 'raw_response'}}, indent=2, ensure_ascii=False)}

Mechanical diagnostics:
{json.dumps(diagnostics, indent=2, ensure_ascii=False)}

Capability memory:
{json.dumps(capability_memory or {}, indent=2, ensure_ascii=False)}

Cumulative candidate source snapshot:
{_source_snapshot(repository)}

Return one JSON object with the same experiment-card fields and either complete
`files` replacements or an optional valid `patch`. Prefer complete files. Do
not delete or broadly weaken tests or remove the original causal mechanism
merely to compile. If the diagnostics expose an older incidental negative
routing assertion that this experiment intentionally supersedes, update that
narrow assertion to the new contract while preserving unrelated regressions.
Every `files` value must be the complete destination file, never patch syntax.
""".strip()
        response = self.query(
            "Repair a robot-controller implementation from concrete build diagnostics.",
            prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.1,
            attempts=3,
            cache=False,
            timeout_s=float(os.environ.get("RACAP_CODER_TIMEOUT_S", "600")),
        )
        return parse_proposal(response)
