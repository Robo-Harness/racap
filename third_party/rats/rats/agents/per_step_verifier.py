"""Post-execution per-step verification.

The verifier asks a VLM whether one plan step succeeded, given three things:
the task description in natural language, the executable code that ran for
that step (sliced out of the policy via ``step_context`` markers), and a
short list of the critical runtime values that code produced.  Everything
else — heavy event payloads, raw poses, internal attribution metadata — is
written to ``verifier_input.json`` for offline inspection but kept out of
the prompt so the VLM gets a clean, targeted question.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rats.agents.base_agent import query_llm_json, video_file_to_data_url


def extract_exec_history_api_events(
    exec_history: Any,
    *,
    policy_step_index: int | None = None,
) -> list[dict[str, Any]]:
    """Pull ``timeline_kind == "api"`` entries off an exec_history.

    LIBERO (and any env that doesn't wrap APIs with ``_wrap_api_function``)
    never populates ``info["api_call_trace"]``, so PerStepVerifier was
    rendering "(no top-level API calls recorded)" on every step prompt
    even when SAM3 / contact_graspnet / goto_pose had clearly run. Those
    runtime events DO exist in the execution_logger's timeline as
    ``timeline_kind == "api"`` markers stamped with ``policy_step_index``;
    this helper turns them into the same flat dict shape the verifier's
    evidence pipeline already consumes (``function_name`` / ``event`` /
    ``api_name`` for the prompt header, ``return_summary`` for the
    rendered detail line, ``policy_step_index`` / ``policy_step_id`` for
    the deterministic step-routing in ``_marked_event_step_index``).

    When ``policy_step_index`` is given (multiturn-reset path verifies one
    plan step at a time), only events from that step are returned. When
    ``None`` (legacy single-shot path verifies the full plan in one call),
    all api events flow through and the verifier's existing index-aware
    assigner routes them per step.
    """
    steps = list(getattr(exec_history, "steps", []) or [])
    out: list[dict[str, Any]] = []
    for s in steps:
        if str(getattr(s, "timeline_kind", "") or "") != "api":
            continue
        psi_raw = getattr(s, "policy_step_index", None)
        try:
            psi = int(psi_raw) if psi_raw is not None else None
        except (TypeError, ValueError):
            psi = None
        if policy_step_index is not None:
            if psi is None or psi != int(policy_step_index):
                continue
        tool = str(getattr(s, "tool_name", "") or "")
        label = str(getattr(s, "timeline_label", "") or tool)
        text = str(getattr(s, "text", "") or "")
        out.append({
            "function_name": tool,
            "event": tool,
            "api_name": label,
            # ``return_summary`` is in PerStepVerifier._CRITICAL_KEYS, so
            # the rendered prompt line becomes
            #   "- SAM3 Text Segmentation: return_summary=sam3_segment(...)"
            # instead of just the bare name.
            "return_summary": text[:600],
            "policy_step_index": psi,
            "policy_step_id": str(getattr(s, "policy_step_id", "") or ""),
            "frame_start": getattr(s, "frame_start", None),
            "frame_end": getattr(s, "frame_end", None),
            "evidence_source": "execution_history",
        })
    return out


@dataclass(frozen=True)
class PerStepVerifierConfig:
    enabled: bool = True
    include_privileged_state: bool = False
    save_artifacts: bool = True
    max_events_per_step: int = 96
    max_state_entries: int = 12
    model: str = "google/gemini-3.1-pro-preview"
    max_tokens: int = 1600
    max_images: int = 32
    # When True, the VLM prompt also asks for `edit_scale` +
    # `corrective_action` so the same call produces the next-retry
    # directive that multiturn-reset mode feeds back to the writer.
    # Default off: legacy callers (failure_diagnoser, feedback_generator)
    # don't consume these fields, so we save the output tokens in those
    # runs. Enabled by LifelongLoop when multiturn_reset_mode is on.
    include_retry_directive: bool = False


class IntermediateOutputAssignmentAgent:
    """Collect primitive artifacts with metadata, then assign them to plan steps.

    This is deliberately deterministic/local: it is an "agent" boundary in the
    data pipeline, not a policy-code reader.  It sees only primitive/helper
    diagnostic output metadata and per-step goals.
    """

    PATH_KEYS = {
        "overlay_path",
        "point_overlay_path",
        "segmentation_overlay_path",
        "image_overlay_path",
        "pointcloud_viz_path",
        "visualization_path",
        "annotated_path",
        "crop_path",
        "raw_npz_path",
        "raw_json_path",
        "image_path",
        "video_path",
    }

    def collect(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for idx, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            paths = self._paths_from_event(event)
            output_type = str(event.get("output_type") or self._infer_output_type(event, paths))
            description = str(event.get("description") or self._default_description(output_type))
            has_semantic_payload = output_type != "primitive_result" or any(
                key in event for key in ("verified", "confidence", "point", "world_point", "normalized_vector")
            )
            if not paths and not has_semantic_payload:
                continue
            outputs.append(
                {
                    "output_id": f"out_{len(outputs) + 1:03d}",
                    "output_type": output_type,
                    "description": description,
                    "source_event": str(event.get("event") or event.get("function_name") or "event"),
                    "source_index": idx,
                    "evidence_source": event.get("evidence_source"),
                    "paths": paths,
                    "metadata": self._metadata_from_event(event),
                }
            )
        return outputs

    def assign(
        self,
        steps: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        *,
        max_outputs_per_step: int = 24,
    ) -> dict[int, list[dict[str, Any]]]:
        if not steps:
            return {}
        if len(steps) == 1:
            return {0: outputs[:max_outputs_per_step]}
        step_tokens = [self._tokens(step.get("goal", "")) for step in steps]
        step_lookup = self._step_lookup(steps)
        assignments: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(steps))}
        for output in outputs:
            marker_idx = self._marked_step_index(output, step_lookup, len(steps))
            semantic = self._semantic_tokens(output)
            text_tokens = self._tokens(json.dumps(output, sort_keys=True)[:3000])
            scores = [
                len(tokens & text_tokens) + 3 * len(tokens & semantic) + self._category_boost(tokens, semantic)
                for tokens in step_tokens
            ]
            if marker_idx is not None:
                best = marker_idx
                method = "runtime_step_marker"
                confidence = 1.0
            else:
                best = max(range(len(scores)), key=lambda i: scores[i])
                method = "semantic_output_alignment"
                confidence = round(min(0.95, 0.35 + 0.1 * max(scores[best], 1)), 3)
            if marker_idx is None and scores[best] <= 0:
                # Fallback to source temporal order without inspecting policy code.
                source_idx = int(output.get("source_index") or 0)
                best = min(len(steps) - 1, int(source_idx * len(steps) / max(1, len(outputs))))
                method = "temporal_output_fallback"
            assigned = dict(output)
            assigned["assignment"] = {
                "method": method,
                "step_index": best,
                "score": int(scores[best]),
                "confidence": confidence,
            }
            if len(assignments[best]) < max_outputs_per_step:
                assignments[best].append(assigned)
        return assignments

    @classmethod
    def _step_lookup(cls, steps: list[dict[str, Any]]) -> dict[str, int]:
        lookup: dict[str, int] = {}
        for idx, step in enumerate(steps):
            for value in (
                step.get("step_id"),
                step.get("id"),
                f"step-{idx + 1}",
                f"step_{idx + 1}",
                str(idx),
                str(idx + 1),
            ):
                if value is not None:
                    lookup[cls._normalize_id(value)] = idx
        return lookup

    @classmethod
    def _marked_step_index(cls, output: dict[str, Any], step_lookup: dict[str, int], step_count: int) -> int | None:
        metadata = output.get("metadata") if isinstance(output.get("metadata"), dict) else {}
        for key in ("policy_step_id", "step_id"):
            value = output.get(key) or metadata.get(key)
            if value is not None:
                idx = step_lookup.get(cls._normalize_id(value))
                if idx is not None:
                    return idx
        marker_index = output.get("policy_step_index", metadata.get("policy_step_index"))
        try:
            if marker_index is not None:
                idx = int(marker_index)
                if 0 <= idx < step_count:
                    return idx
        except Exception:
            pass
        return None

    def _paths_from_event(self, event: dict[str, Any]) -> list[dict[str, str]]:
        paths: list[dict[str, str]] = []
        for key, value in event.items():
            if not isinstance(value, str) or not value:
                continue
            lk = key.lower()
            looks_like_path = (
                lk in self.PATH_KEYS
                or lk.endswith("_path")
                or lk.endswith("_file")
                or "artifact" in lk
                or "visualization" in lk
                or value.endswith((".json", ".npz", ".png", ".jpg", ".jpeg", ".mp4"))
            )
            if looks_like_path:
                paths.append({"role": key, "path": value})
        return paths

    @staticmethod
    def _metadata_from_event(event: dict[str, Any]) -> dict[str, Any]:
        skip_suffixes = ("_path", "_file")
        skip = {"rgb", "image", "mask", "depth"}
        metadata: dict[str, Any] = {}
        for key, value in event.items():
            lk = str(key).lower()
            if lk in skip or lk.endswith(skip_suffixes):
                continue
            if lk in {"event", "t", "evidence_source", "api_name", "diagnostic_index", "description", "output_type"}:
                metadata[key] = value
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
            elif isinstance(value, (list, tuple, dict)):
                metadata[key] = value
        return metadata

    @staticmethod
    def _infer_output_type(event: dict[str, Any], paths: list[dict[str, str]]) -> str:
        name = str(event.get("event") or event.get("function_name") or "").lower()
        roles = " ".join(path.get("role", "") for path in paths).lower()
        if "molmo" in name or "point_overlay" in roles:
            return "molmo_point_image_overlay"
        if "sam3" in name or "segment" in name or "segmentation" in roles:
            return "segmentation_mask_overlay"
        if "mask_to_world" in name or "pointcloud" in roles:
            return "world_point_cloud_visualization"
        if "pull_direction" in name or "direction" in roles:
            return "pull_direction_visualization"
        if "verify" in name:
            return "verify_object_style_primitive_result"
        return "primitive_result"

    @staticmethod
    def _default_description(output_type: str) -> str:
        return {
            "molmo_point_image_overlay": "Image-space point selected by Molmo overlaid on the actual camera image.",
            "segmentation_mask_overlay": "Segmentation mask result overlaid on the actual camera image.",
            "world_point_cloud_visualization": "World point-cloud output and corresponding image-space source overlay.",
            "pull_direction_visualization": "Projected direction arrow overlaid on the actual camera image.",
            "verify_object_style_primitive_result": "Object/predicate verification result with the images shown to the verifier.",
        }.get(output_type, "Primitive/helper intermediate output.")

    @classmethod
    def _semantic_tokens(cls, output: dict[str, Any]) -> set[str]:
        text = " ".join(
            str(output.get(key) or "")
            for key in ("output_type", "description", "source_event")
        ).lower()
        tokens = cls._tokens(text)
        if any(t in text for t in ("molmo", "point", "segment", "mask", "verify", "identity")):
            tokens.update({"find", "locate", "perceive", "identify", "segment", "object", "point"})
        if any(t in text for t in ("world", "pointcloud", "3d", "coordinate", "pose")):
            tokens.update({"world", "3d", "coordinate", "point", "estimate", "position"})
        if any(t in text for t in ("direction", "pull", "push", "slide", "open")):
            tokens.update({"direction", "pull", "push", "slide", "open"})
        return tokens

    @staticmethod
    def _category_boost(step_tokens: set[str], semantic: set[str]) -> int:
        boosts = 0
        if step_tokens & {"find", "locate", "detect", "segment", "identify", "look", "perceive"} and semantic & {
            "find",
            "locate",
            "segment",
            "identify",
        }:
            boosts += 3
        if step_tokens & {"world", "3d", "coordinate", "estimate", "position"} and semantic & {"world", "3d", "coordinate"}:
            boosts += 3
        if step_tokens & {"pull", "push", "slide", "open", "direction"} and semantic & {"pull", "push", "slide", "direction"}:
            boosts += 3
        return boosts

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop = {"the", "a", "an", "to", "of", "and", "or", "in", "on", "with", "for", "then", "step"}
        return {t for t in re.findall(r"[a-z0-9_]+", str(text).lower().replace("_", " ")) if len(t) > 1 and t not in stop}

    @staticmethod
    def _normalize_id(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())


class PerStepVerifier:
    """Minimal per-step verifier over per-step goals and output evidence only."""

    schema_version = "rats_per_step_verification_v1"

    def __init__(
        self,
        *,
        enabled: bool = True,
        include_privileged_state: bool = False,
        save_artifacts: bool = True,
        max_events_per_step: int = 96,
        max_state_entries: int = 12,
        model: str | None = "google/gemini-3.1-pro-preview",
        max_tokens: int = 1600,
        max_images: int = 32,
        include_retry_directive: bool = False,
    ) -> None:
        self.config = PerStepVerifierConfig(
            enabled=bool(enabled),
            include_privileged_state=bool(include_privileged_state),
            save_artifacts=bool(save_artifacts),
            max_events_per_step=int(max_events_per_step),
            max_state_entries=int(max_state_entries),
            model=str(model or "google/gemini-3.1-pro-preview"),
            max_tokens=int(max_tokens or 1600),
            max_images=int(max_images or 32),
            include_retry_directive=bool(include_retry_directive),
        )
        self._output_assignment_agent = IntermediateOutputAssignmentAgent()

    def verify_attempt(
        self,
        execution_result: dict[str, Any],
        *,
        plan: dict[str, Any] | None,
        step: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        iteration: int | None = None,
        attempt: int | None = None,
        attempt_in_iter: int | None = None,
        turn_in_attempt: int | None = None,
        code: str | None = None,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False, "schema_version": self.schema_version}

        steps = [self._normalize_step(step, 0)] if step else self._normalize_steps(plan)
        policy_code = code if code is not None else execution_result.get("policy_code") or ""
        step_code_blocks = self._step_code_blocks(str(policy_code), steps)
        evidence = self._collect_attempt_evidence(execution_result)
        assignments = self._assign_events_to_steps(steps, evidence)
        all_events = self._all_evidence_events(evidence)
        intermediate_outputs = self._output_assignment_agent.collect(all_events)
        output_assignments = self._output_assignment_agent.assign(
            steps,
            intermediate_outputs,
            max_outputs_per_step=self.config.max_events_per_step,
        )
        evidence["intermediate_outputs"] = intermediate_outputs

        base_dir = Path(output_dir) if output_dir is not None else None
        if base_dir is not None and self.config.save_artifacts:
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                base_dir = None

        full_trace = evidence.get("api_call_trace", [])
        diagnostic_events = evidence.get("diagnostic_events", [])
        summary: dict[str, Any] = {
            "enabled": True,
            "schema_version": self.schema_version,
            "verifier_backend": "vlm",
            "verifier_model": self.config.model,
            "evidence_boundary": {
                "policy_code_included": bool(policy_code),
                "policy_result_included": False,
                "privileged_state_included": self.config.include_privileged_state,
            },
            "iteration": iteration,
            "attempt": attempt,
            "attempt_in_iteration": attempt_in_iter,
            "turn_in_attempt": turn_in_attempt,
            "step_count": len(steps),
            "api_event_count": len(full_trace),
            "diagnostic_event_count": len(diagnostic_events),
            "intermediate_output_count": len(intermediate_outputs),
            "intermediate_output_assignment_agent": "IntermediateOutputAssignmentAgent",
            "steps": [],
            "artifact_dir": str(base_dir) if base_dir is not None else None,
        }

        if base_dir is not None and self.config.save_artifacts:
            self._safe_artifact_write(
                "trace_full.jsonl",
                lambda: self._write_jsonl(
                    base_dir / "trace_full.jsonl",
                    self._redact_prompt_value(full_trace, "events") or [],
                ),
                summary,
            )
            self._safe_artifact_write(
                "trace_summary.json",
                lambda: self._write_json(
                    base_dir / "trace_summary.json",
                    {
                        "schema_version": self.schema_version,
                        "api_event_count": len(full_trace),
                        "diagnostic_event_count": len(diagnostic_events),
                        "functions": self._count_by(full_trace, "function_name"),
                        "diagnostic_events": self._count_by(diagnostic_events, "event"),
                        "intermediate_output_count": len(intermediate_outputs),
                        "intermediate_output_types": self._count_by(intermediate_outputs, "output_type"),
                        "intermediate_output_assignment_agent": "IntermediateOutputAssignmentAgent",
                        "verifier_backend": "vlm",
                        "verifier_model": self.config.model,
                        "evidence_boundary": summary["evidence_boundary"],
                    },
                ),
                summary,
            )
            self._safe_artifact_write(
                "intermediate_outputs_full.json",
                lambda: self._write_json(
                    base_dir / "intermediate_outputs_full.json",
                    self._redact_prompt_value(intermediate_outputs, "intermediate_outputs") or [],
                ),
                summary,
            )

        for idx, normalized_step in enumerate(steps):
            step_outputs = output_assignments.get(idx, [])
            step_dir = (
                base_dir / f"step_{idx + 1:02d}_{self._slug(normalized_step['goal'])}"
                if base_dir is not None and self.config.save_artifacts
                else None
            )
            package: dict[str, Any] = {}
            verdict: dict[str, Any]
            step_artifact_errors: list[str] = []
            try:
                package = self._build_step_input(
                    normalized_step,
                    evidence,
                    assignments.get(idx, []),
                    step_outputs,
                    step_code=step_code_blocks.get(idx, ""),
                    full_policy_code=str(policy_code or ""),
                )
                if step_dir is not None:
                    try:
                        step_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        step_artifact_errors.append(f"mkdir: {type(exc).__name__}: {exc}")
                        step_dir = None
                    if step_dir is not None:
                        try:
                            package["saved_artifacts"] = self._save_step_media(
                                step_dir,
                                execution_result,
                                package.get("events", []),
                                normalized_step,
                            )
                        except Exception as exc:
                            step_artifact_errors.append(f"save_step_media: {type(exc).__name__}: {exc}")
                            package["saved_artifacts"] = {"frames": [], "errors": list(step_artifact_errors)}
                package = dict(package)
                package["motion_frame_segment"] = self._step_frame_segment_metadata(
                    execution_result,
                    normalized_step,
                )
                verdict = self._verify_step_package(package)
            except Exception as exc:
                verdict = self._json_safe(
                    {
                        "schema_version": self.schema_version,
                        "verifier_backend": "vlm",
                        "verifier_model": self.config.model,
                        "success": False,
                        "status": "failed",
                        "confidence": 0.0,
                        "reason": f"Per-step verifier crashed for this step: {type(exc).__name__}: {exc}",
                        "evidence_keys": [],
                        "checks": {"per_step_internal_error": f"{type(exc).__name__}: {exc}"},
                        "image_manifest": [],
                        "vlm_prompt": "",
                    }
                )
                step_artifact_errors.append(f"verify_step: {type(exc).__name__}: {exc}")
                package = package or {
                    "schema_version": self.schema_version,
                    "step": normalized_step,
                    "per_step_goal": normalized_step["goal"],
                    "step_code": step_code_blocks.get(idx, ""),
                    "events": assignments.get(idx, []),
                    "intermediate_outputs": step_outputs,
                    "artifact_errors": list(step_artifact_errors),
                }
            if step_artifact_errors:
                package.setdefault("artifact_errors", []).extend(step_artifact_errors)
                verdict.setdefault("artifact_errors", []).extend(step_artifact_errors)
                summary.setdefault("artifact_errors", []).extend(
                    f"{normalized_step['step_id']}: {err}" for err in step_artifact_errors
                )
            if step_dir is not None:
                self._safe_artifact_write(
                    f"{step_dir.name}/verifier_input.json",
                    lambda step_dir=step_dir, package=package: self._write_json(
                        step_dir / "verifier_input.json",
                        self._prompt_safe_package(package),
                    ),
                    summary,
                )
                self._safe_artifact_write(
                    f"{step_dir.name}/verifier_output.json",
                    lambda step_dir=step_dir, verdict=verdict: self._write_json(step_dir / "verifier_output.json", verdict),
                    summary,
                )
                self._safe_artifact_write(
                    f"{step_dir.name}/step_trace.json",
                    lambda step_dir=step_dir, package=package: self._write_json(
                        step_dir / "step_trace.json",
                        self._redact_prompt_value(package.get("events", []), "events") or [],
                    ),
                    summary,
                )
                self._safe_artifact_write(
                    f"{step_dir.name}/intermediate_outputs.json",
                    lambda step_dir=step_dir, package=package: self._write_json(
                        step_dir / "intermediate_outputs.json",
                        self._redact_prompt_value(package.get("intermediate_outputs", []), "intermediate_outputs") or [],
                    ),
                    summary,
                )
                self._safe_artifact_write(
                    f"{step_dir.name}/vlm_prompt.txt",
                    lambda step_dir=step_dir, verdict=verdict: (step_dir / "vlm_prompt.txt").write_text(str(verdict.get("vlm_prompt") or "")),
                    summary,
                )
                self._safe_artifact_write(
                    f"{step_dir.name}/vlm_media_manifest.json",
                    lambda step_dir=step_dir, verdict=verdict: self._write_json(
                        step_dir / "vlm_media_manifest.json",
                        verdict.get("media_manifest", verdict.get("image_manifest", [])),
                    ),
                    summary,
                )

            step_entry = {
                "step_id": normalized_step["step_id"],
                "goal": normalized_step["goal"],
                "success": verdict.get("success"),
                "status": verdict.get("status"),
                "confidence": verdict.get("confidence"),
                "reason": verdict.get("reason"),
                "evidence_keys": verdict.get("evidence_keys", []),
                "artifact_dir": str(step_dir) if step_dir is not None else None,
                "attribution_confidence": package.get("attribution", {}).get("confidence"),
                "intermediate_output_count": len(step_outputs),
                "artifact_errors": step_artifact_errors,
            }
            # Only the multiturn-reset path consumes these; including them
            # by default would change legacy iteration JSONs for no gain.
            if self.config.include_retry_directive:
                step_entry["visual_evidence"] = verdict.get("visual_evidence", "")
                step_entry["unsatisfied_conditions"] = verdict.get("unsatisfied_conditions", [])
                step_entry["edit_scale"] = verdict.get("edit_scale", "")
                step_entry["corrective_action"] = verdict.get("corrective_action", "")
            summary["steps"].append(step_entry)

        summary["summary_text"] = self._format_summary_text(summary)
        return self._json_safe(summary)

    def _normalize_steps(self, plan: dict[str, Any] | None) -> list[dict[str, Any]]:
        raw_steps = []
        if isinstance(plan, dict):
            raw_steps = plan.get("steps") or []
        if not raw_steps:
            return [{"step_id": "step_1", "index": 0, "goal": "Complete the attempted robot step."}]
        return [self._normalize_step(s, i) for i, s in enumerate(raw_steps)]

    @staticmethod
    def _normalize_step(step: dict[str, Any] | None, index: int) -> dict[str, Any]:
        step = step or {}
        sid = step.get("step_id") or step.get("id") or f"step_{index + 1}"
        goal = step.get("description") or step.get("goal") or step.get("text") or str(step)
        return {"step_id": str(sid), "index": index, "goal": str(goal or sid)}

    def _collect_attempt_evidence(self, execution_result: dict[str, Any]) -> dict[str, Any]:
        artifacts = execution_result.get("artifacts") if isinstance(execution_result, dict) else {}
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        info = artifacts.get("info") if isinstance(artifacts.get("info"), dict) else {}
        api_call_trace = info.get("api_call_trace") if isinstance(info.get("api_call_trace"), list) else []
        # Supplement with API-kind events from the execution_history
        # timeline. Required for LIBERO, which doesn't bind primitives
        # through ``_wrap_api_function`` and so leaves ``api_call_trace``
        # empty even when SAM3 / goto_pose / etc. clearly ran. Each
        # timeline event already carries ``policy_step_index`` so the
        # existing ``_marked_event_step_index`` router can place it on
        # the correct plan step without re-doing token-overlap matching.
        timeline_events = (
            execution_result.get("api_timeline_events")
            if isinstance(execution_result.get("api_timeline_events"), list)
            else []
        )
        if timeline_events:
            # Deduplicate by (function_name, policy_step_index, frame_start) so
            # MolmoSpaces (which would set both api_call_trace AND the timeline)
            # doesn't render the same event twice.
            seen: set[tuple[Any, Any, Any]] = set()
            for ev in api_call_trace:
                if isinstance(ev, dict):
                    key = (
                        ev.get("function_name") or ev.get("event"),
                        ev.get("policy_step_index"),
                        ev.get("frame_start"),
                    )
                    seen.add(key)
            merged = list(api_call_trace)
            for ev in timeline_events:
                if not isinstance(ev, dict):
                    continue
                key = (
                    ev.get("function_name") or ev.get("event"),
                    ev.get("policy_step_index"),
                    ev.get("frame_start"),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(ev)
            api_call_trace = merged
        diagnostic_events = self._flatten_api_diagnostics(info.get("api_diagnostics"))
        state: dict[str, Any] = {}
        if self.config.include_privileged_state:
            state = {
                "grounded_state": artifacts.get("grounded_state") if isinstance(artifacts.get("grounded_state"), dict) else {},
                "state_trace": artifacts.get("state_trace") if isinstance(artifacts.get("state_trace"), dict) else {},
            }
        return {
            "api_call_trace": self._json_safe(api_call_trace),
            "diagnostic_events": self._json_safe(diagnostic_events),
            "api_diagnostics_summary": str(info.get("api_diagnostics_summary") or ""),
            "state": self._json_safe(state),
            "trajectory": {
                "frame_count": execution_result.get("trajectory_frame_count"),
                "has_sampled_frames": bool(execution_result.get("trajectory_frames")),
                "has_vlm_frames": bool(execution_result.get("vlm_verifier_frames")),
                "has_before_after": bool(execution_result.get("before_frame") is not None or execution_result.get("after_frame") is not None),
                "has_wrist_before_after": bool(execution_result.get("before_wrist_frame") is not None or execution_result.get("after_wrist_frame") is not None),
            },
            "execution_status": {
                "sandbox_success": bool(execution_result.get("success")),
                "stderr_present": bool(str(execution_result.get("stderr") or "").strip()),
                "timeout": bool(execution_result.get("timeout")),
                "terminated": bool(execution_result.get("terminated")),
                "truncated": bool(execution_result.get("truncated")),
            },
        }

    @staticmethod
    def _flatten_api_diagnostics(api_diagnostics: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not isinstance(api_diagnostics, dict):
            return out
        for api_name, payload in api_diagnostics.items():
            if not isinstance(payload, dict):
                continue
            for idx, event in enumerate(payload.get("events") or []):
                if isinstance(event, dict):
                    e = dict(event)
                    e.setdefault("api_name", str(api_name))
                    e.setdefault("diagnostic_index", idx)
                    out.append(e)
        return out

    @staticmethod
    def _all_evidence_events(evidence: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for event in evidence.get("api_call_trace", []):
            if isinstance(event, dict):
                e = dict(event)
                e.setdefault("evidence_source", "api_call_trace")
                events.append(e)
        for event in evidence.get("diagnostic_events", []):
            if isinstance(event, dict):
                e = dict(event)
                e.setdefault("evidence_source", "api_diagnostics")
                events.append(e)
        return events

    def _assign_events_to_steps(
        self,
        steps: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> dict[int, list[dict[str, Any]]]:
        events = self._all_evidence_events(evidence)
        if not steps:
            return {}
        if len(steps) == 1:
            return {0: events[: self.config.max_events_per_step]}

        assignments: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(steps))}
        step_tokens = [self._tokens(s["goal"]) for s in steps]
        step_lookup = self._step_lookup(steps)
        for event_idx, event in enumerate(events):
            event_text = json.dumps(self._json_safe(event), sort_keys=True)[:4000]
            event_tokens = self._tokens(event_text)
            semantic = self._event_semantic_tokens(event)
            scores = [len(tokens & event_tokens) + len(tokens & semantic) * 2 for tokens in step_tokens]
            marker_idx = self._marked_event_step_index(event, step_lookup, len(steps))
            if marker_idx is not None:
                best = marker_idx
                confidence = 1.0
                method = "runtime_step_marker"
            else:
                best = max(range(len(scores)), key=lambda i: scores[i])
                confidence = 0.2
                method = "semantic_event_alignment"
                if scores[best] > 0:
                    confidence = min(0.95, 0.35 + 0.12 * scores[best])
                else:
                    # Ordered fallback: preserve temporal information without using
                    # policy line numbers or policy comments.
                    best = min(len(steps) - 1, int(event_idx * len(steps) / max(1, len(events))))
                    method = "temporal_event_fallback"
            e = dict(event)
            e["attribution"] = {
                "method": method,
                "step_index": best,
                "score": scores[best],
                "confidence": confidence,
            }
            if len(assignments[best]) < self.config.max_events_per_step:
                assignments[best].append(e)
        return assignments

    @classmethod
    def _step_lookup(cls, steps: list[dict[str, Any]]) -> dict[str, int]:
        lookup: dict[str, int] = {}
        for idx, step in enumerate(steps):
            for value in (
                step.get("step_id"),
                step.get("id"),
                f"step-{idx + 1}",
                f"step_{idx + 1}",
                str(idx),
                str(idx + 1),
            ):
                if value is not None:
                    lookup[cls._normalize_id(value)] = idx
        return lookup

    @classmethod
    def _marked_event_step_index(
        cls,
        event: dict[str, Any],
        step_lookup: dict[str, int],
        step_count: int,
    ) -> int | None:
        for key in ("policy_step_id", "step_id"):
            value = event.get(key)
            if value is not None:
                idx = step_lookup.get(cls._normalize_id(value))
                if idx is not None:
                    return idx
        nested = event.get("policy_step")
        if isinstance(nested, dict):
            value = nested.get("policy_step_id") or nested.get("step_id")
            if value is not None:
                idx = step_lookup.get(cls._normalize_id(value))
                if idx is not None:
                    return idx
            marker_index = nested.get("policy_step_index")
        else:
            marker_index = event.get("policy_step_index")
        try:
            if marker_index is not None:
                idx = int(marker_index)
                if 0 <= idx < step_count:
                    return idx
        except Exception:
            pass
        return None

    def _build_step_input(
        self,
        step: dict[str, Any],
        evidence: dict[str, Any],
        events: list[dict[str, Any]],
        intermediate_outputs: list[dict[str, Any]] | None = None,
        step_code: str = "",
        full_policy_code: str = "",
    ) -> dict[str, Any]:
        target_state = self._state_relevant_to_goal(step["goal"], evidence.get("state") or {})
        intermediate_outputs = intermediate_outputs or []
        artifact_refs = self._artifact_refs(events, intermediate_outputs)
        confidence = 0.0
        if events:
            confidence = sum(float((e.get("attribution") or {}).get("confidence", 0.35)) for e in events) / len(events)
        excluded = {
            "policy_result_dict": "excluded_by_design",
            "privileged_state": "excluded_by_default",
        }
        if not step_code:
            excluded["policy_code"] = "no_step_code_extracted"
        package = {
            "schema_version": self.schema_version,
            "verifier_scope": "single_step_post_execution",
            "excluded_inputs": excluded,
            "step": step,
            "per_step_goal": step["goal"],
            "step_code": step_code or "",
            "full_policy_code": full_policy_code or "",
            "events": events,
            "intermediate_outputs": intermediate_outputs,
            "artifact_refs": artifact_refs,
            "api_diagnostics_summary": evidence.get("api_diagnostics_summary", ""),
            "trajectory": evidence.get("trajectory", {}),
            "execution_status": evidence.get("execution_status", {}),
            "attribution": {
                "method": "posthoc_semantic_event_alignment",
                "confidence": round(confidence, 3),
                "event_count": len(events),
                "intermediate_output_assignment_method": "intermediate_output_assignment_agent",
                "intermediate_output_count": len(intermediate_outputs),
            },
        }
        if self.config.include_privileged_state:
            package["privileged_state"] = target_state
            package["excluded_inputs"].pop("privileged_state", None)
        return self._json_safe(package)

    def _verify_step_package(self, package: dict[str, Any]) -> dict[str, Any]:
        images, videos, media_manifest = self._vlm_media_for_step(package)
        system_prompt, user_prompt = self._vlm_prompts_for_step(package, media_manifest)
        if not images and not videos:
            return self._json_safe(
                {
                    "schema_version": self.schema_version,
                    "verifier_backend": "vlm",
                    "verifier_model": self.config.model,
                    "success": False,
                    "status": "failed",
                    "confidence": 0.0,
                    "reason": "VLM per-step verifier could not run because no step media or visual artifact images were available.",
                    "evidence_keys": [],
                    "checks": {"vlm_error": "no_visual_inputs"},
                    "image_manifest": media_manifest,
                    "media_manifest": media_manifest,
                    "vlm_prompt": user_prompt,
                }
            )

        try:
            parsed = query_llm_json(
                system_prompt,
                user_prompt,
                images=images,
                videos=videos,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            return self._json_safe(
                {
                    "schema_version": self.schema_version,
                    "verifier_backend": "vlm",
                    "verifier_model": self.config.model,
                    "success": False,
                    "status": "failed",
                    "confidence": 0.0,
                    "reason": f"VLM per-step verifier call failed: {type(exc).__name__}: {exc}",
                    "evidence_keys": [],
                    "checks": {"vlm_error": f"{type(exc).__name__}: {exc}"},
                    "image_manifest": media_manifest,
                    "media_manifest": media_manifest,
                    "vlm_prompt": user_prompt,
                }
            )

        if not isinstance(parsed, dict):
            parsed = {}
        success = bool(parsed.get("success", parsed.get("step_success", parsed.get("visual_success", False))))
        status = str(parsed.get("status") or ("succeeded" if success else "failed")).strip().lower()
        if status not in {"succeeded", "failed", "ambiguous"}:
            status = "succeeded" if success else "failed"
        confidence = self._safe_float(parsed.get("confidence"), 0.0)
        confidence = max(0.0, min(1.0, confidence))
        reason = str(
            parsed.get("reason")
            or parsed.get("failure_reason")
            or parsed.get("evidence")
            or ("VLM judged the step succeeded." if success else "VLM did not find enough evidence that the step succeeded.")
        ).strip()
        evidence_keys = parsed.get("evidence_keys")
        if not isinstance(evidence_keys, list):
            evidence_keys = []
        checks = parsed.get("checks") if isinstance(parsed.get("checks"), dict) else {}
        checks.update(
            {
                "vlm_image_count": len(images),
                "vlm_video_count": len(videos),
                "vlm_model": self.config.model,
                "raw_status": parsed.get("status"),
            }
        )
        verdict: dict[str, Any] = {
            "schema_version": self.schema_version,
            "verifier_backend": "vlm",
            "verifier_model": self.config.model,
            "success": success,
            "status": status,
            "confidence": round(confidence, 3),
            "reason": reason,
            "evidence_keys": [str(x) for x in evidence_keys if x],
            "satisfied_conditions": parsed.get("satisfied_conditions") or [],
            "unsatisfied_conditions": parsed.get("unsatisfied_conditions") or [],
            "visual_evidence": str(parsed.get("visual_evidence") or parsed.get("evidence") or "").strip(),
            "checks": checks,
            "image_manifest": media_manifest,
            "media_manifest": media_manifest,
            "vlm_prompt": user_prompt,
            "raw_vlm_output": parsed,
        }

        # Optional next-retry directive — only emitted when the prompt asked
        # for it (multiturn-reset mode). Legacy callers don't read these
        # fields, so we save the output tokens by not requesting them.
        if self.config.include_retry_directive:
            edit_scale_raw = str(parsed.get("edit_scale") or "").strip().lower()
            if status == "succeeded":
                edit_scale_out = "n/a"
            elif edit_scale_raw in ("argument_level", "rewrite_needed"):
                edit_scale_out = edit_scale_raw
            elif edit_scale_raw == "n/a":
                # VLM bailed on the field even though status != succeeded —
                # default to argument_level so the writer at least gets a hint.
                edit_scale_out = "argument_level"
            else:
                edit_scale_out = ""  # unknown / not emitted
            verdict["edit_scale"] = edit_scale_out
            verdict["corrective_action"] = str(parsed.get("corrective_action") or "").strip()

        return self._json_safe(verdict)

    @staticmethod
    def _has_perception_success(
        events: list[dict[str, Any]],
        intermediate_outputs: list[dict[str, Any]] | None = None,
    ) -> bool:
        for e in events:
            name = str(e.get("event") or e.get("function_name") or "")
            if name == "molmo_point_prompt" and e.get("point") is not None:
                return True
            if name in {"camera_segmented_pointcloud", "language_pointcloud_complete", "object_search_success"}:
                count = e.get("selected_point_count", e.get("fused_point_count", e.get("point_count", 0)))
                try:
                    if int(count or 0) > 0:
                        return True
                except Exception:
                    pass
            if name in {"get_object_pose_success", "sample_grasp_pose_success"}:
                return True
        for output in intermediate_outputs or []:
            output_type = str(output.get("output_type") or "")
            paths = output.get("paths") if isinstance(output.get("paths"), list) else []
            metadata = output.get("metadata") if isinstance(output.get("metadata"), dict) else {}
            if output_type in {
                "molmo_point_image_overlay",
                "segmentation_mask_overlay",
                "world_point_cloud_visualization",
                "verify_object_identity_result",
                "verify_object_style_primitive_result",
            } and (paths or metadata.get("verified") is True or metadata.get("point") is not None):
                return True
        return False

    @staticmethod
    def _has_grasp_success(events: list[dict[str, Any]]) -> bool:
        for e in events:
            name = str(e.get("event") or e.get("function_name") or "")
            if name in {"grasp_plan_pointclouds", "sample_grasp_pose_success", "grasp_selection"}:
                if e.get("selected") is True or e.get("grasp_position") is not None:
                    return True
                try:
                    if int(e.get("candidate_count") or 0) > 0:
                        return True
                except Exception:
                    pass
        return False

    @staticmethod
    def _has_motion_success(events: list[dict[str, Any]]) -> bool:
        for e in events:
            name = str(e.get("event") or e.get("function_name") or "")
            if name == "move_to_joints_result" and str(e.get("status", "")).lower() == "success":
                return True
            if name == "goto_pose_tcp_delta":
                try:
                    if float(e.get("position_delta_norm")) < 0.12:
                        return True
                except Exception:
                    return True
            if name in {"goto_pose", "move_to_joints"} and str(e.get("status", "")).lower() in {"ok", "success"}:
                return True
        return False

    @staticmethod
    def _has_gripper_action(events: list[dict[str, Any]], goal_tokens: set[str]) -> bool:
        desired = None
        if "open" in goal_tokens or "release" in goal_tokens or "drop" in goal_tokens or "place" in goal_tokens:
            desired = "open"
        if "close" in goal_tokens or "grasp" in goal_tokens or "grab" in goal_tokens:
            desired = "close"
        for e in events:
            name = str(e.get("event") or e.get("function_name") or "")
            if name == "gripper_action":
                if desired is None or str(e.get("action")) == desired:
                    return True
            if desired and name == f"{desired}_gripper":
                return True
        return False

    def _state_relevant_to_goal(self, goal: str, state: dict[str, Any]) -> dict[str, Any]:
        if not state:
            return {}
        grounded = state.get("grounded_state") if isinstance(state.get("grounded_state"), dict) else {}
        before = grounded.get("before") if isinstance(grounded.get("before"), dict) else {}
        after = grounded.get("after") if isinstance(grounded.get("after"), dict) else {}
        tokens = self._tokens(goal)
        before_entries = self._matching_inventory_entries(before.get("inventory"), tokens)
        after_entries = self._matching_inventory_entries(after.get("inventory"), tokens)
        dense = state.get("state_trace") if isinstance(state.get("state_trace"), dict) else {}
        samples = dense.get("samples") if isinstance(dense.get("samples"), list) else []
        return {
            "source": "execution_result.artifacts.grounded_state/state_trace",
            "before_robot": before.get("robot") or {},
            "after_robot": after.get("robot") or {},
            "target_candidates_before": before_entries[: self.config.max_state_entries],
            "target_candidates_after": after_entries[: self.config.max_state_entries],
            "dense_state_sample_count": len(samples),
            "dense_state_samples_excerpt": samples[:2] + samples[-2:] if len(samples) > 4 else samples,
        }

    def _movement_measurements(self, state: dict[str, Any]) -> dict[str, Any]:
        before_robot = state.get("before_robot") if isinstance(state.get("before_robot"), dict) else {}
        after_robot = state.get("after_robot") if isinstance(state.get("after_robot"), dict) else {}
        before_target = (state.get("target_candidates_before") or [{}])[0] if isinstance(state.get("target_candidates_before"), list) else {}
        after_target = (state.get("target_candidates_after") or [{}])[0] if isinstance(state.get("target_candidates_after"), list) else {}
        return {
            "robot_motion_m": self._distance(self._position(before_robot), self._position(after_robot)),
            "target_motion_m": self._distance(self._position(before_target), self._position(after_target)),
            "joint_delta": self._joint_delta(before_target, after_target),
        }

    def _matching_inventory_entries(self, inventory: Any, tokens: set[str]) -> list[dict[str, Any]]:
        entries = self._inventory_entries(inventory)
        scored: list[tuple[int, dict[str, Any]]] = []
        for entry in entries:
            text = json.dumps(self._json_safe(entry), sort_keys=True)[:2000]
            score = len(tokens & self._tokens(text))
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._compact_state_entry(e) for _, e in scored]

    def _inventory_entries(self, value: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if any(k in value for k in ("internal_name", "category", "name", "object_id", "joint_state", "position")):
                out.append(value)
            for child in value.values():
                out.extend(self._inventory_entries(child))
        elif isinstance(value, list):
            for child in value:
                out.extend(self._inventory_entries(child))
        return out

    def _compact_state_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        keep = {}
        for key, value in entry.items():
            lk = str(key).lower()
            if any(token in lk for token in ("name", "category", "position", "center", "joint", "open", "state", "room", "receptacle", "artic")):
                keep[str(key)] = value
        return self._json_safe(keep or entry)

    @staticmethod
    def _position(entry: Any) -> list[float] | None:
        if not isinstance(entry, dict):
            return None
        for key in ("position", "pos", "center", "centroid", "world_position", "robot_cartesian_pos"):
            value = entry.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    return [float(value[0]), float(value[1]), float(value[2])]
                except Exception:
                    pass
        return None

    def _joint_delta(self, before: Any, after: Any) -> float | None:
        b_vals = self._numeric_state_values(before)
        a_vals = self._numeric_state_values(after)
        keys = sorted(set(b_vals) & set(a_vals))
        deltas = [abs(a_vals[k] - b_vals[k]) for k in keys if math.isfinite(a_vals[k]) and math.isfinite(b_vals[k])]
        return max(deltas) if deltas else None

    def _numeric_state_values(self, value: Any, prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                lk = str(key).lower()
                if isinstance(child, (int, float)) and any(t in lk for t in ("joint", "qpos", "open", "state", "value")):
                    out[path] = float(child)
                elif isinstance(child, (dict, list, tuple)):
                    out.update(self._numeric_state_values(child, path))
        elif isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                path = f"{prefix}[{idx}]"
                if isinstance(child, (dict, list, tuple)):
                    out.update(self._numeric_state_values(child, path))
        return out

    @staticmethod
    def _distance(a: list[float] | None, b: list[float] | None) -> float | None:
        if a is None or b is None:
            return None
        try:
            return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3]))))
        except Exception:
            return None

    @staticmethod
    def _artifact_refs(
        events: list[dict[str, Any]],
        intermediate_outputs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for e in events:
            for key, value in e.items():
                if not isinstance(value, str):
                    continue
                lk = key.lower()
                if lk.endswith("_path") or lk.endswith("_file") or "artifact" in lk or "visualization" in lk or value.endswith((".json", ".npz", ".png", ".jpg", ".mp4")):
                    refs.append({"event": e.get("event") or e.get("function_name"), "key": key, "path": value})
        for output in intermediate_outputs or []:
            for path_ref in output.get("paths") or []:
                if not isinstance(path_ref, dict):
                    continue
                path = path_ref.get("path")
                if isinstance(path, str) and path:
                    refs.append(
                        {
                            "event": output.get("source_event"),
                            "output_id": output.get("output_id"),
                            "output_type": output.get("output_type"),
                            "key": path_ref.get("role"),
                            "path": path,
                        }
                    )
        return refs

    def _save_step_media(
        self,
        step_dir: Path,
        execution_result: dict[str, Any],
        events: list[dict[str, Any]] | None = None,
        step: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        media_dir = step_dir / "frames"
        media_dir.mkdir(parents=True, exist_ok=True)
        frame_window = self._frame_window_for_events(events or [])
        saved: dict[str, Any] = {
            "frames": [],
            "videos": [],
            "frame_window": frame_window,
            "scope_note": (
                "Attempt-level before/after frames are global. Trajectory frames are "
                "saved as full per-step/turn videos for VLM input; still frames "
                "are limited to before/after context and artifact previews."
            ),
        }
        try:
            import imageio.v2 as imageio
        except Exception:
            try:
                import imageio  # type: ignore
            except Exception as exc:
                return {"frames": [], "error": f"imageio_unavailable: {exc}"}
        candidates: list[tuple[str, Any]] = []
        segment = self._step_frame_segment(execution_result, step or {})
        if segment:
            frames = list(segment.get("sampled_frames") or [])
            saved["exact_step_segment"] = {
                k: v
                for k, v in segment.items()
                if k not in {"sampled_frames"}
            }
            if frames:
                candidates.append(("step_before", frames[0]))
                # Perception-only steps don't tick simulator frames between
                # the marker enter/exit, so the segment collapses to a single
                # frame: step_before == step_after, and a 1-frame mp4 is
                # uninformative. Only emit step_after + step_motion_video
                # when the step actually spans more than one frame.
                if len(frames) > 1:
                    candidates.append(("step_after", frames[-1]))
                    self._write_video_artifact(
                        media_dir,
                        "step_motion_video",
                        frames,
                        saved,
                    )
        trajectory_frames = list(execution_result.get("trajectory_frames") or [])
        full_video_frames = list(
            execution_result.get("trajectory_video_frames")
            or execution_result.get("full_trajectory_frames")
            or []
        )
        if segment:
            pass
        elif frame_window.get("has_frame_window"):
            self._write_video_artifact(
                media_dir,
                "turn_motion_video",
                full_video_frames or trajectory_frames,
                saved,
            )
        elif events:
            self._write_video_artifact(
                media_dir,
                "turn_motion_video",
                full_video_frames or trajectory_frames,
                saved,
            )
        for label, frame in candidates:
            if frame is None:
                continue
            path = media_dir / f"{label}.png"
            try:
                imageio.imwrite(str(path), np.asarray(frame))
                saved["frames"].append({"label": label, "path": str(path)})
            except Exception as exc:
                saved.setdefault("errors", []).append({"label": label, "error": str(exc)})
        return saved

    @staticmethod
    def _write_video_artifact(
        media_dir: Path,
        label: str,
        frames: list[Any],
        saved: dict[str, Any],
        *,
        fps: int = 20,
    ) -> None:
        if not frames:
            return
        path = media_dir / f"{label}.mp4"
        try:
            import imageio.v2 as imageio
        except Exception:
            try:
                import imageio  # type: ignore
            except Exception as exc:
                saved.setdefault("errors", []).append(
                    {"label": label, "error": f"imageio_unavailable: {exc}"}
                )
                return
        try:
            arrs = [np.asarray(frame) for frame in frames if frame is not None]
            if not arrs:
                return
            imageio.mimsave(str(path), arrs, fps=fps)
            saved.setdefault("videos", []).append(
                {
                    "label": label,
                    "path": str(path),
                    "frame_count": len(arrs),
                    "fps": fps,
                }
            )
        except Exception as exc:
            saved.setdefault("errors", []).append({"label": label, "error": str(exc)})

    def _step_frame_segment_metadata(
        self,
        execution_result: dict[str, Any],
        step: dict[str, Any],
    ) -> dict[str, Any]:
        segment = self._step_frame_segment(execution_result, step)
        if not segment:
            return {}
        return self._json_safe({k: v for k, v in segment.items() if k != "sampled_frames"})

    def _step_frame_segment(
        self,
        execution_result: dict[str, Any],
        step: dict[str, Any],
    ) -> dict[str, Any] | None:
        segments = execution_result.get("step_frame_segments")
        if not isinstance(segments, list):
            return None
        keys = {
            self._normalize_id(step.get("step_id")),
            self._normalize_id(step.get("id")),
            self._normalize_id(f"step-{int(step.get('index', 0)) + 1}"),
            self._normalize_id(f"step_{int(step.get('index', 0)) + 1}"),
        }
        step_index = step.get("index")
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            seg_id = self._normalize_id(segment.get("policy_step_id"))
            if seg_id and seg_id in keys:
                return segment
            try:
                if step_index is not None and segment.get("policy_step_index") is not None:
                    if int(segment.get("policy_step_index")) == int(step_index):
                        return segment
            except Exception:
                pass
        return None

    @staticmethod
    def _frame_window_for_events(events: list[dict[str, Any]]) -> dict[str, Any]:
        starts: list[int] = []
        ends: list[int] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            for key in ("frame_start", "start_frame", "trajectory_start_frame"):
                value = event.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    starts.append(int(value))
            for key in ("frame_end", "end_frame", "trajectory_end_frame"):
                value = event.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    ends.append(int(value))
        return {
            "has_frame_window": bool(starts or ends),
            "start_frame": min(starts) if starts else None,
            "end_frame": max(ends) if ends else None,
        }

    def _vlm_media_for_step(self, package: dict[str, Any]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Return image/video data URLs plus a path-free label manifest.

        The prompt manifest intentionally omits filesystem paths: those paths
        are meaningful to RATS but invisible to the remote VLM.  Labels and
        media indices are stable across per-step verification and feedback.
        """
        image_paths: list[tuple[str, str]] = []
        video_paths: list[dict[str, Any]] = []
        saved = package.get("saved_artifacts") if isinstance(package.get("saved_artifacts"), dict) else {}
        for frame in saved.get("frames") or []:
            if not isinstance(frame, dict):
                continue
            path = frame.get("path")
            label = frame.get("label") or "step_frame"
            if str(label).startswith(("trajectory_", "step_trajectory_")):
                continue
            if isinstance(path, str) and self._is_image_path(path):
                image_paths.append((str(label), path))
        for video in saved.get("videos") or []:
            if not isinstance(video, dict):
                continue
            path = video.get("path")
            if isinstance(path, str) and self._is_video_path(path):
                video_paths.append(video)

        # Re-include ``artifact_refs`` images as main-function-visible
        # intermediate values (e.g. sam3 mask overlays, vlm_verify
        # annotated images, molmo point-prompt overlays). An earlier
        # commit (1d73a4cb) dropped them treating them as "API-internal";
        # that was wrong for steps like "localize the potato + verify
        # mask", where the per-step verifier was left with only the
        # before-step frame and could not judge whether the right object
        # was localized. The main function called ``segment_sam3_text_
        # prompt`` and ``vlm_verify``; the resulting mask + annotated
        # image ARE the variables the main code received and acted on.
        # Cap the count so a single step doesn't dump a hundred frames
        # into the verifier prompt.
        artifact_refs = package.get("artifact_refs") or []
        max_artifact_images = 8
        artifact_count = 0
        for ref in artifact_refs:
            if not isinstance(ref, dict):
                continue
            path = ref.get("path")
            if not isinstance(path, str) or not self._is_image_path(path):
                continue
            label_parts = [
                str(x)
                for x in (
                    ref.get("output_type"),
                    ref.get("event") or ref.get("function_name"),
                    ref.get("key"),
                )
                if x
            ]
            label = ".".join(label_parts) or "intermediate_artifact"
            image_paths.append((label, path))
            artifact_count += 1
            if artifact_count >= max_artifact_images:
                break

        images: list[str] = []
        videos: list[str] = []
        manifest: list[dict[str, Any]] = []
        seen: set[str] = set()
        media_index = 0
        for video in video_paths:
            path = str(video.get("path") or "")
            if path in seen:
                continue
            seen.add(path)
            url = video_file_to_data_url(path)
            if not url:
                continue
            media_index += 1
            manifest.append(
                {
                    "media_index": media_index,
                    "type": "video",
                    "label": video.get("label") or "motion_video",
                    "frame_count": video.get("frame_count"),
                    "fps": video.get("fps"),
                }
            )
            videos.append(url)
        for label, path in image_paths:
            if path in seen:
                continue
            seen.add(path)
            url = self._image_file_to_data_url(path)
            if not url:
                continue
            media_index += 1
            manifest.append({"media_index": media_index, "type": "image", "label": label})
            images.append(url)
            if len(images) >= self.config.max_images:
                break
        return images, videos, manifest

    @staticmethod
    def _is_image_path(path: str) -> bool:
        return str(path).lower().endswith((".png", ".jpg", ".jpeg"))

    @staticmethod
    def _is_video_path(path: str) -> bool:
        return str(path).lower().endswith((".mp4", ".m4v", ".mov", ".webm"))

    @staticmethod
    def _image_file_to_data_url(path: str) -> str | None:
        try:
            import base64
            import io

            from PIL import Image

            p = Path(path)
            if not p.exists() or not p.is_file():
                return None
            with Image.open(p) as img:
                img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return None

    def _vlm_prompts_for_step(
        self,
        package: dict[str, Any],
        media_manifest: list[dict[str, Any]],
    ) -> tuple[str, str]:
        system_prompt = (
            "You are a strict visual verifier for one robot policy step. "
            "You are given (1) the natural-language goal for this single "
            "step, (2) the full policy code (so you can see this step in "
            "the context of what comes before and after — focus on the "
            "with step_context(...) block whose step_id / step_index "
            "matches this step), (3) the main-function API calls the "
            "policy made for this step with their high-level return "
            "summaries, and (4) attached media. The media includes the "
            "step's motion video (primary visual evidence for what the "
            "robot actually did) AND intermediate perception artifacts "
            "produced by the API calls during this step — e.g. the "
            "segmentation mask overlay returned by sam3, the annotated "
            "image returned by vlm_verify with the candidate mask / "
            "marker drawn on it, point-prompt overlays from Molmo, etc. "
            "Use these intermediates to judge whether the right object "
            "was localized / the verify pass landed on the intended "
            "thing; use the motion video to judge whether the robot "
            "executed the intended motion. The code shows intent, not "
            "runtime values. Do not assume an API name implies success — "
            "check the visual evidence. If the evidence is insufficient "
            "or ambiguous, return success=false. Respond only with JSON."
        )

        goal = str(package.get("per_step_goal") or "").strip() or "(no goal recorded)"
        step = package.get("step") if isinstance(package.get("step"), dict) else {}
        step_id = str(step.get("step_id") or "").strip()
        step_index = step.get("index")
        # Render only top-level events (the main function's own API calls).
        # ``intermediate_outputs`` are API-internal trace artifacts (Molmo
        # overlays, sub-API verify_object frames, etc.) and they were
        # confusing the verifier — the sub-artifact dumps had values the
        # main function never saw. So we drop intermediate_outputs from
        # the prompt; the video carries the ground-truth visual signal.
        critical_values = self._render_critical_values(
            package.get("events") or [],
            [],
        )
        full_code = str(package.get("full_policy_code") or "").strip()
        media_lines = self._render_media_manifest(media_manifest)

        step_locator_lines = []
        if step_id:
            step_locator_lines.append(f"step_id = {step_id!r}")
        if step_index is not None:
            step_locator_lines.append(f"step_index = {step_index}")
        step_locator = ", ".join(step_locator_lines) or "(no step locator)"

        sections = [
            "TASK (natural-language goal for this single step):",
            goal,
            "",
            f"STEP LOCATOR (the with step_context(...) block in the policy code below that matches this step):",
            step_locator,
            "",
            "FULL POLICY CODE (the code shows intent, not runtime values; use the attached video as ground truth):",
            f"```python\n{full_code}\n```" if full_code else "(no policy code captured)",
            "",
            "MAIN-FUNCTION API CALLS DURING THIS STEP:",
            critical_values or "(no top-level API calls recorded)",
            "",
            "ATTACHED MEDIA (in the order they appear):",
            media_lines or "(no media attached)",
            "",
            "Return JSON with exactly these keys:",
            "{",
            '  "success": boolean,',
            '  "status": "succeeded" | "failed" | "ambiguous",',
            '  "confidence": number between 0 and 1,',
            '  "reason": short explanation of why the step did or did not succeed,',
            '  "visual_evidence": short description of what the images show,',
            '  "satisfied_conditions": [strings],',
            '  "unsatisfied_conditions": [strings],',
            '  "evidence_keys": [strings naming the images/values you relied on],',
            '  "checks": {"object_visible": boolean|null, "robot_moved": boolean|null, "target_effect_visible": boolean|null}'
            + ("," if self.config.include_retry_directive else ""),
        ]
        # Optional retry-directive fields. Only multiturn-reset mode consumes
        # them; legacy callers (failure_diagnoser, feedback_generator) ignore
        # them, so we save the ~150 output tokens per call when off.
        if self.config.include_retry_directive:
            sections.extend([
                '  "edit_scale": "argument_level" | "rewrite_needed" | "n/a",  // n/a when status=succeeded',
                '  "corrective_action": "one short paragraph telling the policy writer exactly what to change in the next attempt at THIS step (named argument value, primitive substitution, missing precondition call). Empty string when status=succeeded. Do NOT rewrite code yourself."',
                "}",
                "",
                "Guidance for edit_scale (only when status != succeeded):",
                "- argument_level = keep the same primitives and call sequence; "
                "only specific named arguments need to change (offsets, "
                "quaternions, prompt strings, IK seeds).",
                "- rewrite_needed = different primitive or call sequence; the "
                "current approach is structurally wrong for this step. Prefer "
                "this when the failure points to category errors (wrong object "
                "grasped, wrong handle, navigation_error) or when an argument "
                "tune cannot plausibly fix the observed visual outcome.",
            ])
        else:
            sections.append("}")
        return system_prompt, "\n".join(sections)

    @staticmethod
    def _render_media_manifest(manifest: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for entry in manifest or []:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("media_index")
            kind = entry.get("type") or "media"
            label = entry.get("label") or kind
            bits = [f"#{idx} {kind}: {label}"]
            if kind == "video":
                fc = entry.get("frame_count")
                fps = entry.get("fps")
                if fc is not None or fps is not None:
                    bits.append(f"({fc} frames @ {fps} fps)")
            lines.append("  - " + " ".join(bits))
        return "\n".join(lines)

    # Keys that carry actually useful information for a verifier and are
    # safe to surface verbatim — booleans, scalars, short status strings,
    # small 2-element image coordinates, etc.  Dense numeric payloads
    # (point clouds, poses, joint vectors) are filtered out by
    # ``_redact_prompt_value``-style logic in ``_critical_value`` below.
    _CRITICAL_KEYS: tuple[str, ...] = (
        "status",
        "action",
        "verified",
        "confidence",
        "prompt",
        "point",
        "selected",
        "candidate_count",
        "selected_point_count",
        "fused_point_count",
        "point_count",
        "position_delta_norm",
        "reason",
        "reasoning",
        "return_summary",
        "args_summary",
    )

    def _render_critical_values(
        self,
        events: list[dict[str, Any]],
        intermediate_outputs: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []
        for event in events or []:
            if not isinstance(event, dict):
                continue
            name = str(event.get("event") or event.get("function_name") or event.get("api_name") or "event")
            pairs = self._critical_pairs(event)
            if pairs:
                lines.append(f"  - {name}: {pairs}")
            else:
                lines.append(f"  - {name}")
        for output in intermediate_outputs or []:
            if not isinstance(output, dict):
                continue
            output_type = str(output.get("output_type") or "intermediate_output")
            description = str(output.get("description") or "").strip()
            metadata = output.get("metadata") if isinstance(output.get("metadata"), dict) else {}
            pairs = self._critical_pairs(metadata)
            head = f"  - [{output_type}]"
            if description:
                head += f" {description}"
            if pairs:
                head += f" — {pairs}"
            lines.append(head)
        return "\n".join(lines)

    def _critical_pairs(self, source: dict[str, Any]) -> str:
        pairs: list[str] = []
        for key in self._CRITICAL_KEYS:
            if key not in source:
                continue
            value = self._critical_value(key, source.get(key))
            if value is None:
                continue
            pairs.append(f"{key}={value}")
        return ", ".join(pairs)

    @staticmethod
    def _critical_value(key: str, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                return None
            if isinstance(value, float):
                return f"{value:.3f}".rstrip("0").rstrip(".") or "0"
            return str(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            return text[:160] + ("…" if len(text) > 160 else "")
        if isinstance(value, (list, tuple)):
            # Only surface compact coordinate-like payloads (image points,
            # 2-3 element scalars).  Dense vectors are noise to a VLM.
            if 1 <= len(value) <= 3 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
                for v in value
            ):
                return "[" + ", ".join(
                    (f"{float(v):.2f}".rstrip("0").rstrip(".") or "0")
                    if isinstance(v, float)
                    else str(v)
                    for v in value
                ) + "]"
            return None
        return None

    def _step_code_blocks(
        self,
        code: str,
        steps: list[dict[str, Any]],
    ) -> dict[int, str]:
        """Split policy code into per-step blocks using ``step_context`` markers.

        Looks for ``with step_context("step_id", "description", step_index=N):``
        and slices the body of each ``with`` statement back to its owning step.
        Matches by ``step_index`` when present, falling back to ``step_id``.
        Returns ``{step_index: code_string}`` (dedented) — empty when the
        policy did not use the runtime markers.
        """
        if not code or not steps:
            return {}
        marker_re = re.compile(
            r"^(?P<indent>[ \t]*)with\s+step_context\(\s*"
            r"(?P<args>[^)]*)\)\s*:\s*$",
            re.MULTILINE,
        )
        id_re = re.compile(r"""['\"]([^'\"]+)['\"]""")
        index_re = re.compile(r"step_index\s*=\s*(\d+)")
        step_id_to_index: dict[str, int] = {}
        for step in steps:
            sid = self._normalize_id(step.get("step_id"))
            if sid:
                step_id_to_index[sid] = int(step.get("index") or 0)

        matches = list(marker_re.finditer(code))
        if not matches:
            return {}
        blocks: dict[int, str] = {}
        for idx, m in enumerate(matches):
            args = m.group("args") or ""
            indent = m.group("indent") or ""
            step_index: int | None = None
            mi = index_re.search(args)
            if mi:
                try:
                    step_index = int(mi.group(1))
                except Exception:
                    step_index = None
            if step_index is None:
                mid = id_re.search(args)
                if mid:
                    step_index = step_id_to_index.get(self._normalize_id(mid.group(1)))
            if step_index is None:
                continue
            body_start = code.find("\n", m.end())
            if body_start < 0:
                continue
            body_start += 1
            # Walk lines until we hit one whose non-blank indent is ≤ the
            # with line's own indent — that's where the with-block actually
            # ends. The previous logic used ``matches[idx + 1].start()``,
            # which silently swept up top-level code or the next step's
            # ``# Step N: ...`` intro comment between blocks; combined with
            # the dedent's min-indent heuristic, those leaked lines pinned
            # prefix_len to 0 and left the real body indented at +4.
            with_indent_len = len(indent.expandtabs(4))
            i = body_start
            body_end = len(code)
            while i < len(code):
                nl = code.find("\n", i)
                line = code[i:nl] if nl >= 0 else code[i:]
                stripped = line.strip()
                if stripped:
                    leading = len(line) - len(line.lstrip(" \t"))
                    leading_expanded = len((line[:leading]).expandtabs(4))
                    if leading_expanded <= with_indent_len:
                        body_end = i
                        break
                if nl < 0:
                    break
                i = nl + 1
            body = code[body_start:body_end]
            dedented = self._dedent_body(body, indent)
            if dedented.strip():
                # If multiple ``with step_context`` blocks reference the same
                # step (uncommon, but legal), concatenate them in source order.
                if step_index in blocks:
                    blocks[step_index] = blocks[step_index].rstrip() + "\n\n" + dedented
                else:
                    blocks[step_index] = dedented
        return blocks

    @staticmethod
    def _dedent_body(body: str, with_indent: str) -> str:
        # Body lines are indented one level deeper than the ``with`` line.
        # Strip a consistent leading-whitespace prefix without touching blank
        # lines or content of inner blocks.
        lines = body.splitlines()
        non_blank = [ln for ln in lines if ln.strip()]
        if not non_blank:
            return ""
        prefix_len = len(with_indent) + 4  # assume 4-space indent inside with
        # Fallback: detect minimum indent of non-blank lines.
        observed = min(len(ln) - len(ln.lstrip(" \t")) for ln in non_blank)
        prefix_len = min(prefix_len, observed) if observed > len(with_indent) else observed
        out: list[str] = []
        for ln in lines:
            if not ln.strip():
                out.append("")
                continue
            out.append(ln[prefix_len:] if ln[:prefix_len].isspace() or ln[:prefix_len] == "" else ln.lstrip())
        return "\n".join(out).strip("\n")

    def _prompt_safe_package(self, package: dict[str, Any]) -> dict[str, Any]:
        """Compact verifier package for prompt use without image blobs/code."""
        keep = {
            "schema_version",
            "verifier_scope",
            "excluded_inputs",
            "step",
            "per_step_goal",
            "step_code",
            "events",
            "intermediate_outputs",
            "artifact_refs",
            "api_diagnostics_summary",
            "privileged_state",
            "trajectory",
            "execution_status",
            "attribution",
            "motion_frame_segment",
            "saved_artifacts",
        }
        compact = {k: package.get(k) for k in keep if k in package}
        raw_step_code = compact.pop("step_code", None)
        safe = self._redact_prompt_value(self._json_safe(compact)) or {}
        if isinstance(raw_step_code, str) and raw_step_code:
            safe["step_code"] = raw_step_code
        text = json.dumps(safe, sort_keys=True)
        if len(text) <= 18000:
            return safe
        # Keep the most important fields intact and trim verbose event payloads.
        safe["events"] = self._compact_events(package.get("events") or [])
        safe["intermediate_outputs"] = self._compact_events(package.get("intermediate_outputs") or [])
        if isinstance(raw_step_code, str) and raw_step_code:
            safe["step_code"] = raw_step_code
        return safe

    def _redact_prompt_value(self, value: Any, key: str = "") -> Any:
        """Remove local paths, privileged/raw state, and dense numeric payloads.

        This intentionally deletes fields instead of replacing them with
        ``[omitted]`` placeholders so the VLM prompt contains only meaningful
        symbolic/runtime evidence plus the attached media manifest.
        """
        lk = key.lower()
        rawish = any(
            token in lk
            for token in (
                "pointcloud",
                "point_cloud",
                "joints",
                "joint",
                "qpos",
                "pose",
                "pose_mat",
                "intrinsics",
                "extrinsics",
                "depth",
                "centroid",
                "center",
                "world_point",
                "world_points",
                "normalized_vector",
                "raw",
                "grounded_state",
                "state_trace",
            )
        )
        pathish = (
            lk in {"path", "paths"}
            or lk.endswith("_path")
            or lk.endswith("_file")
            or lk.endswith("_dir")
            or "raw_json_path" in lk
            or "npz" in lk
        )
        if pathish or rawish:
            return None
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for child_key, child in value.items():
                redacted = self._redact_prompt_value(child, str(child_key))
                if redacted not in (None, {}, []):
                    out[str(child_key)] = redacted
            return out
        if isinstance(value, (list, tuple)):
            if not value:
                return []
            if lk in {"events", "intermediate_outputs", "artifact_refs", "steps"}:
                return [
                    redacted
                    for v in value
                    if (redacted := self._redact_prompt_value(v, key)) not in (None, {}, [])
                ]
            numeric_count = sum(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
            if numeric_count >= 3 and numeric_count >= max(3, len(value) // 2):
                return None
            if len(value) > 12 and any(isinstance(x, (list, tuple, dict)) for x in value):
                # Dense state traces / point samples are not useful to a VLM
                # prompt compared with the attached video.
                return None
            return [
                redacted
                for v in value
                if (redacted := self._redact_prompt_value(v, key)) not in (None, {}, [])
            ]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if any(token in lk for token in ("joint", "pose", "position", "center", "centroid", "point", "depth", "intrinsic", "delta", "dist")):
                return None
        if isinstance(value, str):
            text = re.sub(
                r"(?i)\b(?:raw|path|file|artifact|json|npz)=((?:/|outputs/|rats/outputs/|\S*/outputs/)[^\s,;]+)",
                "",
                value,
            )
            text = re.sub(
                r"(?i)(?:/workspace/[^\s,;]+|/tmp/[^\s,;]+|outputs/[^\s,;]+|rats/outputs/[^\s,;]+)",
                "",
                text,
            )
            text = re.sub(
                r"\[(?:\s*-?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*,){2,}\s*-?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*\]",
                "",
                text,
            )
            text = re.sub(r"\b\w+=($|\s)", " ", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
            return text or None
        return value

    def _compact_events(self, events: list[Any]) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        keep = {
            "event",
            "function_name",
            "api_name",
            "status",
            "action",
            "output_type",
            "description",
            "policy_step_id",
            "policy_step_index",
            "policy_step_goal",
            "evidence_source",
            "verified",
            "confidence",
            "reasoning",
            "return_summary",
            "assignment",
            "metadata",
        }
        for event in events[: self.config.max_events_per_step]:
            if not isinstance(event, dict):
                continue
            compacted.append(
                self._redact_prompt_value(
                    {k: self._json_safe(v) for k, v in event.items() if k in keep}
                )
            )
        return compacted

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
            return out if math.isfinite(out) else default
        except Exception:
            return default

    def _format_summary_text(self, summary: dict[str, Any]) -> str:
        lines = ["Per-step verification (policy code/RESULT excluded):"]
        for step in summary.get("steps", []):
            mark = "✓" if step.get("success") else "✗"
            lines.append(
                f"- {mark} {step.get('step_id')}: {step.get('status')} "
                f"conf={step.get('confidence')} — {step.get('reason')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _event_key(event: dict[str, Any]) -> str:
        return str(event.get("event") or event.get("function_name") or event.get("api_name") or "event")

    @staticmethod
    def _event_semantic_tokens(event: dict[str, Any]) -> set[str]:
        name = str(event.get("event") or event.get("function_name") or "").lower()
        tokens: set[str] = set(re.findall(r"[a-z0-9]+", name))
        if any(t in name for t in ("molmo", "sam3", "pointcloud", "segment", "pose")):
            tokens.update({"find", "locate", "perceive", "object", "point"})
        if any(t in name for t in ("grasp", "gripper")):
            tokens.update({"grasp", "pick", "hold", "close", "open", "release"})
        if any(t in name for t in ("goto", "move", "joint", "ik", "reach")):
            tokens.update({"move", "reach", "approach"})
        return tokens

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop = {"the", "a", "an", "to", "of", "and", "or", "in", "on", "with", "for", "then", "step"}
        return {t for t in re.findall(r"[a-z0-9_]+", str(text).lower().replace("_", " ")) if len(t) > 1 and t not in stop}

    @staticmethod
    def _normalize_id(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    @staticmethod
    def _slug(text: str, *, max_len: int = 48) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text).lower()).strip("_")
        return (slug or "step")[:max_len].strip("_") or "step"

    @staticmethod
    def _count_by(events: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in events:
            value = str(event.get(key) or event.get("event") or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(self._json_safe(payload), indent=2, sort_keys=True))

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(self._json_safe(row), sort_keys=True) + "\n")

    @staticmethod
    def _safe_artifact_write(label: str, fn, summary: dict[str, Any]) -> None:
        """Persist optional verifier artifacts without aborting verification.

        Run directories in remote/NFS-backed workspaces can occasionally raise
        transient EIO errors while writing images/json.  Per-step verification is
        feedback, not the execution authority, so a single artifact write must
        not collapse the whole per-step payload into ``enabled=false``.
        """
        try:
            fn()
        except Exception as exc:
            summary.setdefault("artifact_errors", []).append(
                f"{label}: {type(exc).__name__}: {exc}"
            )

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return self._json_safe(value.item())
            if value.size > 128:
                arr = value.astype(float, copy=False) if np.issubdtype(value.dtype, np.number) else value
                summary: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                if np.issubdtype(value.dtype, np.number):
                    finite = arr[np.isfinite(arr)]
                    if finite.size:
                        summary.update({"min": float(np.min(finite)), "max": float(np.max(finite)), "mean": float(np.mean(finite))})
                return summary
            return [self._json_safe(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return self._json_safe(value.item())
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return repr(value)
