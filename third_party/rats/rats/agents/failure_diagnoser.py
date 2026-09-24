"""Failure Diagnoser: LLM vision agent for analyzing execution failures.

Compares first and last frame observations. Produces structured feedback
for the Policy Writer. THIS VISUAL FEEDBACK LOOP IS THE PRIMARY LEARNING SIGNAL.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from rats.agents.base_agent import image_to_data_url, query_llm_text

logger = logging.getLogger("rats.failure_diagnoser")


class FailureDiagnoser:
    DEFAULT_MODEL = "google/gemini-3.1-pro-preview"

    _TIMEOUT_MARKERS = (
        "TimeoutError",
        "Timed out waiting for mlspaces server response",
        "Execution timed out after",
        "timed out waiting",
    )

    def __init__(self, model: str | None = None) -> None:
        resolved = (
            model
            if model is not None
            else os.getenv("RATS_DIAGNOSER_MODEL", self.DEFAULT_MODEL)
        )
        resolved = str(resolved or "").strip()
        self.model = (
            None
            if resolved.lower() in {"", "default", "global", "inherit"}
            else resolved
        )

    def _query_llm_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        max_tokens: int,
    ) -> str:
        kwargs: dict[str, Any] = {
            "images": images,
            "max_tokens": max_tokens,
        }
        if videos:
            kwargs["videos"] = videos
        if self.model:
            kwargs["model"] = self.model
        try:
            return query_llm_text(system_prompt, user_prompt, **kwargs)
        except TypeError as exc:
            # A few unit tests monkeypatch query_llm_text with a narrower
            # signature. Production query_llm_text supports model=.
            if "unexpected keyword argument 'model'" not in str(exc):
                raise
            kwargs.pop("model", None)
            return query_llm_text(system_prompt, user_prompt, **kwargs)

    def _get_api_diagnostics_summary(self, execution_result: dict[str, Any]) -> str:
        artifacts = execution_result.get("artifacts", {}) or {}
        info = artifacts.get("info", {}) or {}
        summary = info.get("api_diagnostics_summary", "")
        if isinstance(summary, str):
            return summary
        return ""

    def _get_api_diagnostics(self, execution_result: dict[str, Any]) -> dict[str, Any]:
        artifacts = execution_result.get("artifacts", {}) or {}
        info = artifacts.get("info", {}) or {}
        diagnostics = info.get("api_diagnostics", {})
        return diagnostics if isinstance(diagnostics, dict) else {}

    def _get_per_step_verification_summary(self, execution_result: dict[str, Any]) -> str:
        """Return verifier-only per-step summary, excluding policy code/RESULT."""
        artifacts = execution_result.get("artifacts", {}) or {}
        payload = artifacts.get("per_step_verification")
        if not isinstance(payload, dict) or not payload.get("enabled", False):
            return ""
        text = str(payload.get("summary_text") or "").strip()
        if text:
            return text[:3000]
        lines: list[str] = []
        for step in payload.get("steps") or []:
            if not isinstance(step, dict):
                continue
            mark = "success" if step.get("success") else "fail"
            lines.append(
                f"- {step.get('step_id')}: {mark} ({step.get('status')}, "
                f"conf={step.get('confidence')}) — {step.get('reason')}"
            )
        return "\n".join(lines)[:3000]

    def _iter_api_diagnostic_events(
        self,
        execution_result: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        diagnostics = self._get_api_diagnostics(execution_result)
        events: list[tuple[str, dict[str, Any]]] = []
        for source, payload in diagnostics.items():
            if isinstance(payload, dict):
                raw_events = payload.get("events", []) or []
            elif isinstance(payload, list):
                raw_events = payload
            else:
                continue
            for event in raw_events:
                if isinstance(event, dict):
                    events.append((str(source), event))
        return events

    def _format_diagnostic_value(self, value: Any, *, max_chars: int = 140) -> str:
        if value is None:
            return "None"
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, (int, bool)):
            return str(value)
        if isinstance(value, str):
            return value if len(value) <= max_chars else value[: max_chars - 3] + "..."
        if isinstance(value, (list, tuple)):
            if value and all(isinstance(v, (int, float)) for v in value):
                txt = "[" + ", ".join(f"{float(v):.4g}" for v in value[:8]) + "]"
                if len(value) > 8:
                    txt = txt[:-1] + ", ...]"
                return txt
            txt = str(value)
            return txt if len(txt) <= max_chars else txt[: max_chars - 3] + "..."
        if isinstance(value, dict):
            txt = str(value)
            return txt if len(txt) <= max_chars else txt[: max_chars - 3] + "..."
        txt = str(value)
        return txt if len(txt) <= max_chars else txt[: max_chars - 3] + "..."

    def _resolve_local_image_path(self, path_value: Any) -> Path | None:
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        raw = Path(path_value).expanduser()
        candidates = [raw]
        if not raw.is_absolute():
            candidates.extend([
                Path.cwd() / raw,
                Path(__file__).resolve().parent.parent.parent / raw,
            ])
        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate
            except OSError:
                continue
        return None

    def _image_path_to_data_url(self, path_value: Any) -> str | None:
        path = self._resolve_local_image_path(path_value)
        if path is None or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            return None
        try:
            import numpy as np
            from PIL import Image

            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB"))
            return image_to_data_url(arr)
        except Exception:
            return None

    def _numeric_path_candidates(self, event: dict[str, Any]) -> list[tuple[str, Path]]:
        paths: list[tuple[str, Path]] = []
        for key, value in event.items():
            if not (
                key in {"raw_npz_path", "raw_json_path", "raw_file_path"}
                or key.endswith("_npz")
                or key.endswith("_npz_path")
                or key.endswith("_json_path")
            ):
                continue
            path = self._resolve_local_image_path(value)
            if path is None:
                continue
            if path.suffix.lower() in {".npz", ".npy", ".json", ".jsonl", ".txt", ".csv"}:
                paths.append((key, path))
        return paths

    def _summarize_npz_file(
        self,
        path: Path,
        *,
        max_arrays: int = 8,
        max_values: int = 24,
    ) -> str:
        try:
            import numpy as np

            if path.suffix.lower() == ".npy":
                arr = np.asarray(np.load(path, allow_pickle=False))
                flat = arr.reshape(-1) if arr.ndim > 0 else arr.reshape(1)
                sample = [v.item() for v in flat[:max_values]]
                return (
                    f"array: shape={tuple(arr.shape)} dtype={arr.dtype} "
                    f"sample={self._format_diagnostic_value(sample, max_chars=220)}"
                )

            with np.load(path, allow_pickle=False) as data:
                parts = []
                for name in list(data.files)[:max_arrays]:
                    arr = np.asarray(data[name])
                    flat = arr.reshape(-1) if arr.ndim > 0 else arr.reshape(1)
                    sample_values = flat[:max_values]
                    if arr.dtype.kind in {"U", "S", "O"}:
                        sample = [str(v) for v in sample_values[:8]]
                    else:
                        sample = [
                            float(v) if isinstance(v.item(), float) else v.item()
                            for v in sample_values
                        ]
                    parts.append(
                        f"{name}: shape={tuple(arr.shape)} dtype={arr.dtype} "
                        f"sample={self._format_diagnostic_value(sample, max_chars=220)}"
                    )
                if len(data.files) > max_arrays:
                    parts.append(f"... {len(data.files) - max_arrays} more arrays")
                return "\n    ".join(parts)
        except Exception as exc:
            return f"(could not read npz: {exc})"

    def _summarize_text_numeric_file(self, path: Path, *, max_chars: int = 5000) -> str:
        try:
            text = path.read_text(errors="replace")
        except Exception as exc:
            return f"(could not read text/json: {exc})"
        text = text.strip()
        # if len(text) > max_chars:
        #     return text[:max_chars] + "\n    ... (truncated)"
        return text

    def _collect_raw_numeric_artifacts(
        self,
        execution_result: dict[str, Any],
        *,
        events: list[tuple[str, dict[str, Any]]] | None = None,
        limit: int = 8,
        max_arrays: int = 8,
        max_values: int = 24,
        max_text_chars: int = 5000,
    ) -> str:
        blocks: list[str] = []
        seen_paths: set[str] = set()
        event_items = events if events is not None else self._iter_api_diagnostic_events(execution_result)
        for source, event in event_items:
            event_name = str(event.get("event", "diagnostic"))
            for key, path in self._numeric_path_candidates(event):
                path_key = str(path.resolve())
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                suffix = path.suffix.lower()
                if suffix in {".npz", ".npy"}:
                    summary = self._summarize_npz_file(
                        path,
                        max_arrays=max_arrays,
                        max_values=max_values,
                    )
                else:
                    summary = self._summarize_text_numeric_file(
                        path,
                        max_chars=max_text_chars,
                    )
                blocks.append(
                    f"- {source}.{event_name} {key}: {path}\n"
                    f"    {summary}"
                )
                if len(blocks) >= limit:
                    return "\n".join(blocks)
        return "\n".join(blocks)

    def _collect_diagnostic_visual_artifacts(
        self,
        execution_result: dict[str, Any],
        *,
        events: list[tuple[str, dict[str, Any]]] | None = None,
        limit: int = 6,
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Load pointcloud/grasp visualization images recorded by the API."""
        labels: list[dict[str, str]] = []
        images: list[str] = []
        seen_paths: set[str] = set()
        event_items = events if events is not None else self._iter_api_diagnostic_events(execution_result)
        for source, event in event_items:
            if len(images) >= limit:
                break
            event_name = str(event.get("event", "diagnostic"))
            for key, value in event.items():
                if not key.endswith("_path") and key not in {"visualization_path", "viz_path"}:
                    continue
                path = self._resolve_local_image_path(value)
                if path is None or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                path_key = str(path.resolve())
                if path_key in seen_paths:
                    continue
                url = self._image_path_to_data_url(str(path))
                if not url:
                    continue
                seen_paths.add(path_key)
                labels.append({
                    "source": source,
                    "event": event_name,
                    "key": key,
                    "path": str(path),
                    "camera": str(event.get("camera", "") or ""),
                    "label": str(event.get("label", event.get("prompt", "")) or ""),
                })
                images.append(url)
                if len(images) >= limit:
                    break
        return labels, images

    def _diagnostic_context_package(
        self,
        execution_result: dict[str, Any],
        *,
        focus_text: str = "",
        artifact_labels: list[dict[str, str]] | None = None,
        artifact_image_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        """Package non-privileged diagnostic artifacts for policy retries."""
        selected_events = self._select_relevant_diagnostic_events(
            execution_result,
            focus_text=focus_text,
            limit=8,
        )
        artifact_summary = self._diagnostic_artifact_summary(
            execution_result,
            events=selected_events,
        )
        raw_numeric_artifacts = self._collect_raw_numeric_artifacts(
            execution_result,
            events=selected_events,
            limit=3,
            max_arrays=6,
            max_values=32,
            max_text_chars=4000,
        )
        staleness = self._pointcloud_staleness_warning(execution_result)
        labels = artifact_labels
        image_urls = artifact_image_urls
        if labels is None or image_urls is None:
            labels, image_urls = self._collect_diagnostic_visual_artifacts(
                execution_result,
                events=selected_events,
                limit=2,
            )
        else:
            labels, image_urls = self._filter_relevant_artifact_images(
                labels,
                image_urls,
                selected_events=selected_events,
                limit=2,
            )

        package: dict[str, Any] = {}
        if selected_events:
            package["selection"] = (
                "Only the top relevance-ranked diagnostic events/raw files/"
                "images are forwarded to the policy writer to keep retry "
                "prompts small."
            )
        if artifact_summary:
            package["artifact_summary"] = artifact_summary
        if raw_numeric_artifacts:
            package["raw_numeric_artifacts"] = raw_numeric_artifacts
        if staleness:
            package["pointcloud_staleness"] = staleness
        if labels and image_urls:
            package["artifact_images"] = [
                {
                    "source": str(label.get("source", "")),
                    "event": str(label.get("event", "")),
                    "key": str(label.get("key", "")),
                    "path": str(label.get("path", "")),
                    "camera": str(label.get("camera", "")),
                    "label": str(label.get("label", "")),
                    "data_url": url,
                }
                for label, url in zip(labels, image_urls, strict=False)
            ]
        return package

    def _diagnostic_focus_terms(self, focus_text: str) -> set[str]:
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "into",
            "onto", "was", "were", "are", "not", "none", "true", "false",
            "step", "failed", "failure", "policy", "feedback", "object",
            "point", "points", "pointcloud", "cloud", "fresh", "use", "using",
        }
        terms = set()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", focus_text.lower()):
            token = token.strip("_'-")
            if token and token not in stop:
                terms.add(token)
        return terms

    def _diagnostic_event_blob(self, source: str, event: dict[str, Any]) -> str:
        keys = (
            "event", "prompt", "label", "object_name", "camera", "source",
            "status", "exception_type", "exception_message", "selected_strategy",
            "visualization_path", "raw_npz_path", "raw_json_path",
        )
        parts = [source]
        for key in keys:
            value = event.get(key)
            if value is not None:
                parts.append(str(value))
        return " ".join(parts).lower()

    def _diagnostic_event_relevance_score(
        self,
        source: str,
        event: dict[str, Any],
        focus_terms: set[str],
        focus_lower: str,
    ) -> int:
        blob = self._diagnostic_event_blob(source, event)
        event_name = str(event.get("event", "")).lower()
        score = 0

        matched_terms = {term for term in focus_terms if term in blob}
        score += min(len(matched_terms), 6) * 3

        if any(word in blob for word in ("error", "exception", "fail", "timeout")):
            score += 8
        if event.get("exception_type") or event.get("exception_message"):
            score += 8
        if str(event.get("status", "")).lower() not in {"", "success", "ok"}:
            score += 5

        for count_key in (
            "selected_point_count", "fused_point_count", "segment_point_count",
            "point_count", "candidate_count",
        ):
            try:
                if count_key in event and float(event[count_key]) <= 0:
                    score += 7
            except (TypeError, ValueError):
                pass

        if "grasp" in focus_lower and "grasp" in event_name:
            score += 6
        if any(term in focus_lower for term in ("pointcloud", "point cloud", "points", "segmentation", "mask")):
            if "pointcloud" in event_name or "segment" in event_name:
                score += 6
        if any(term in focus_lower for term in ("motion", "goto", "ik", "timeout", "collision", "stuck")):
            if event_name in {
                "goto_pose_target", "move_to_joints_target",
                "move_to_joints_result", "solve_ik_frame_transform",
                "gripper_action",
            }:
                score += 5

        if self._numeric_path_candidates(event):
            score += 2
        if any(
            key.endswith("_path") or key in {"visualization_path", "viz_path"}
            for key in event
        ):
            score += 1
        return score

    def _select_relevant_diagnostic_events(
        self,
        execution_result: dict[str, Any],
        *,
        focus_text: str = "",
        limit: int = 8,
    ) -> list[tuple[str, dict[str, Any]]]:
        events = self._iter_api_diagnostic_events(execution_result)
        if not events:
            return []

        focus_lower = focus_text.lower()
        focus_terms = self._diagnostic_focus_terms(focus_text)
        scored = [
            (
                self._diagnostic_event_relevance_score(
                    source,
                    event,
                    focus_terms,
                    focus_lower,
                ),
                idx,
                source,
                event,
            )
            for idx, (source, event) in enumerate(events)
        ]

        positive = [(score, idx, source, event) for score, idx, source, event in scored if score > 0]
        if positive:
            max_score = max(score for score, _idx, _source, _event in positive)
            if max_score >= 15:
                threshold = max(8, max_score - 12)
                narrowed = [
                    item for item in positive
                    if item[0] >= threshold
                ]
                if narrowed:
                    positive = narrowed
        if not positive:
            positive = [
                (score, idx, source, event)
                for score, idx, source, event in scored
                if self._numeric_path_candidates(event)
                or any(
                    key.endswith("_path") or key in {"visualization_path", "viz_path"}
                    for key in event
                )
            ]
        if not positive:
            positive = scored[-limit:]

        top = sorted(positive, key=lambda item: (item[0], item[1]), reverse=True)[:limit]
        selected_indices: set[int] = set()
        for score, idx, _source, _event in top:
            selected_indices.add(idx)

        if len(selected_indices) > limit:
            ranked_indices = [
                idx
                for _score, idx, _source, _event in sorted(
                    scored,
                    key=lambda item: (item[0], item[1]),
                    reverse=True,
                )
                if idx in selected_indices
            ][:limit]
            selected_indices = set(ranked_indices)

        return [
            (source, event)
            for idx, (source, event) in enumerate(events)
            if idx in selected_indices
        ]

    def _filter_relevant_artifact_images(
        self,
        labels: list[dict[str, str]],
        image_urls: list[str],
        *,
        selected_events: list[tuple[str, dict[str, Any]]],
        limit: int = 2,
    ) -> tuple[list[dict[str, str]], list[str]]:
        selected_keys = {
            (
                source,
                str(event.get("event", "")),
                str(event.get("camera", "") or ""),
                str(event.get("label", event.get("prompt", "")) or ""),
            )
            for source, event in selected_events
        }

        kept_labels: list[dict[str, str]] = []
        kept_urls: list[str] = []
        for label, url in zip(labels, image_urls, strict=False):
            key = (
                str(label.get("source", "")),
                str(label.get("event", "")),
                str(label.get("camera", "")),
                str(label.get("label", "")),
            )
            if key not in selected_keys:
                continue
            kept_labels.append(label)
            kept_urls.append(url)
            if len(kept_urls) >= limit:
                break

        if kept_urls:
            return kept_labels, kept_urls
        return labels[:limit], image_urls[:limit]

    def _has_structured_diagnostic_artifacts(
        self,
        execution_result: dict[str, Any],
    ) -> bool:
        """True when runtime diagnostics contain event/raw/image artifacts."""
        return bool(self._iter_api_diagnostic_events(execution_result))

    def _stderr_suggests_perception_or_motion_issue(self, stderr: str) -> bool:
        text = stderr.lower()
        markers = (
            "pointcloud",
            "point cloud",
            "no valid points",
            "object not found",
            "not found in",
            "segmentation",
            "segment_sam",
            "sam3",
            "molmo",
            "mask",
            "depth",
            "grasp",
            "ik",
            "inverse kinematics",
            "goto_pose",
            "move_to_joints",
            "object search",
        )
        return any(marker in text for marker in markers)

    def _extract_runtime_error_context(
        self,
        execution_result: dict[str, Any],
        code: str,
    ) -> str:
        stderr = str(execution_result.get("stderr", "") or "")
        lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
        last_line = lines[-1] if lines else "runtime error flag set without stderr"

        blocks = [
            "The policy raised a runtime error. Treat the exception as a "
            "symptom to explain using code + diagnostic artifacts, not as a "
            "complete diagnosis by itself.",
            f"Runtime error signal: {last_line}",
        ]

        code_lines = code.splitlines()
        policy_frames: list[tuple[int, str, str]] = []
        for match in re.finditer(
            r'File "<string>", line (\d+), in ([^\n]+)\n\s*([^\n]*)',
            stderr,
        ):
            line_no = int(match.group(1))
            fn_name = match.group(2).strip()
            traceback_code = match.group(3).strip()
            source_line = (
                code_lines[line_no - 1].strip()
                if 0 < line_no <= len(code_lines)
                else traceback_code
            )
            policy_frames.append((line_no, fn_name, source_line))

        if policy_frames:
            blocks.append("Policy-code traceback frames:")
            for line_no, fn_name, source_line in policy_frames[-4:]:
                blocks.append(f"  - line {line_no} in {fn_name}: {source_line}")
        else:
            blocks.append("No <string> policy traceback frame was available.")

        diagnostics = self._get_api_diagnostics_summary(execution_result)
        if diagnostics:
            blocks.append(f"Runtime diagnostics: {diagnostics}")

        return "\n".join(blocks)

    def _diagnostic_artifact_summary(
        self,
        execution_result: dict[str, Any],
        *,
        events: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> str:
        events = events if events is not None else self._iter_api_diagnostic_events(execution_result)
        if not events:
            return ""

        lines: list[str] = []
        interesting = {
            "camera_segmented_pointcloud",
            "language_pointcloud_complete",
            "object_search_view",
            "object_search_success",
            "get_object_pose_success",
            "grasp_plan_pointclouds",
            "sample_grasp_pose_success",
            "grasp_selection",
            "goto_pose_target",
            "goto_pose_raw_artifact",
            "solve_ik_frame_transform",
            "move_to_joints_target",
            "move_to_joints_result",
            "gripper_action",
        }
        field_order = {
            "camera_segmented_pointcloud": [
                "camera", "prompt", "point", "selected_strategy",
                "selected_score", "selected_point_count",
                "object_centroid_world", "visualization_path", "raw_npz_path",
            ],
            "language_pointcloud_complete": [
                "prompt", "valid_cameras", "fused_source", "fused_point_count",
                "fused_centroid_world", "agentview_score", "wrist_score",
                "per_camera_visualizations", "per_camera_raw_npz",
            ],
            "object_search_view": ["prompt", "view_index", "position"],
            "object_search_success": ["prompt", "source", "point_count", "centroid"],
            "get_object_pose_success": ["object_name", "position", "quaternion_wxyz"],
            "grasp_plan_pointclouds": [
                "label", "segment_point_count", "segment_centroid_world",
                "candidate_count", "best_score", "top_scores",
                "top_candidate_grasps", "visualization_path", "raw_npz_path",
            ],
            "sample_grasp_pose_success": [
                "object_name", "candidate_count", "best_index", "best_score",
                "top_scores", "grasp_position", "grasp_quaternion_wxyz",
            ],
            "grasp_selection": [
                "selector", "candidate_count", "selected", "selected_index",
                "selected_score", "reason", "selected_position_world",
                "target_approach_direction_world", "selected_alignment",
            ],
            "goto_pose_target": [
                "position_world", "quaternion_world_wxyz", "z_approach",
                "approach_position_world",
            ],
            "goto_pose_raw_artifact": ["raw_json_path"],
            "solve_ik_frame_transform": [
                "position_world", "position_base", "quaternion_world_wxyz",
            ],
            "move_to_joints_target": ["target_joints"],
            "move_to_joints_result": [
                "status", "target_joints", "before_cartesian_pos",
                "after_cartesian_pos", "elapsed_s", "exception_type",
                "exception_message", "raw_json_path",
            ],
            "gripper_action": [
                "action", "before_cartesian_pos", "after_cartesian_pos",
                "elapsed_s", "raw_json_path",
            ],
        }

        for source, event in events[-32:]:
            event_name = str(event.get("event", ""))
            if event_name not in interesting:
                continue
            fields = []
            for key in field_order.get(event_name, []):
                if key in event:
                    fields.append(f"{key}={self._format_diagnostic_value(event.get(key))}")
            if fields:
                lines.append(f"- {source}.{event_name}: " + "; ".join(fields))

        return "\n".join(lines[-24:])

    def _pointcloud_staleness_warning(
        self,
        execution_result: dict[str, Any],
    ) -> str:
        """Warn when saved pointclouds may predate object-moving robot motion.

        Pointcloud/grasp artifacts are snapshots produced by policy API calls
        during execution. If the arm/gripper moved after the latest snapshot,
        especially near the segmented target, the saved numeric centroid/grasp
        candidates may no longer describe the current scene. The VLM should
        then recommend recomputing perception, not blindly reusing old points.
        """
        events = self._iter_api_diagnostic_events(execution_result)
        if not events:
            return ""

        pointcloud_events = {
            "camera_segmented_pointcloud",
            "language_pointcloud_complete",
            "grasp_plan_pointclouds",
            "sample_grasp_pose_pointclouds",
            "sample_grasp_pose_success",
            "grasp_selection",
        }
        motion_events = {
            "goto_pose_target",
            "move_to_joints_target",
            "move_to_joints_result",
            "gripper_action",
        }

        latest_pc_idx: int | None = None
        latest_pc: tuple[str, dict[str, Any]] | None = None
        for idx, (source, event) in enumerate(events):
            event_name = str(event.get("event", ""))
            if event_name in pointcloud_events or "pointcloud" in event_name:
                latest_pc_idx = idx
                latest_pc = (source, event)

        if latest_pc_idx is None or latest_pc is None:
            return ""

        motions_after: list[tuple[int, str, dict[str, Any]]] = []
        for idx, (source, event) in enumerate(
            events[latest_pc_idx + 1 :],
            start=latest_pc_idx + 1,
        ):
            event_name = str(event.get("event", ""))
            if event_name in motion_events:
                motions_after.append((idx, source, event))

        if not motions_after:
            return ""

        pc_source, pc_event = latest_pc
        pc_name = str(pc_event.get("event", "pointcloud"))
        pc_label = (
            pc_event.get("prompt")
            or pc_event.get("label")
            or pc_event.get("object_name")
            or "unknown object"
        )
        pc_centroid = (
            pc_event.get("fused_centroid_world")
            or pc_event.get("object_centroid_world")
            or pc_event.get("segment_centroid_world")
            or pc_event.get("selected_position_world")
            or pc_event.get("grasp_position")
        )

        motion_lines = []
        for _idx, source, event in motions_after[-6:]:
            event_name = str(event.get("event", "motion"))
            fields: list[str] = []
            for key in (
                "action",
                "status",
                "position_world",
                "approach_position_world",
                "before_cartesian_pos",
                "after_cartesian_pos",
                "target_joints",
            ):
                if key in event:
                    fields.append(
                        f"{key}={self._format_diagnostic_value(event.get(key), max_chars=100)}"
                    )
            suffix = "; ".join(fields) if fields else "(no details)"
            motion_lines.append(f"  - {source}.{event_name}: {suffix}")

        centroid_txt = (
            self._format_diagnostic_value(pc_centroid)
            if pc_centroid is not None
            else "unknown"
        )
        return (
            "Latest saved pointcloud/grasp snapshot: "
            f"{pc_source}.{pc_name} for label={self._format_diagnostic_value(pc_label)} "
            f"centroid_or_selected_point={centroid_txt}.\n"
            "Robot arm/gripper motion was logged AFTER that snapshot, so the "
            "object may have been pushed, dragged, lifted, dropped, or occluded "
            "after the pointcloud was computed:\n"
            + "\n".join(motion_lines)
            + "\nBefore telling the policy writer to reuse old pointcloud "
            "coordinates, old centroids, or old grasp candidates, inspect the "
            "filmstrip/wrist view for object displacement. If contact or "
            "displacement is possible or uncertain, instruct the next policy to "
            "recompute perception from a fresh observation/current agentview + "
            "wrist view after that motion, then reuse only the high-level "
            "successful strategy or object feature."
        )

    def _guard_stale_pointcloud_feedback(
        self,
        diagnosis: dict[str, Any],
        execution_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Append a safety correction if feedback leans on stale pointclouds."""
        warning = self._pointcloud_staleness_warning(execution_result)
        if not warning:
            return diagnosis

        feedback = str(diagnosis.get("policy_feedback", "") or "")
        if not feedback:
            return diagnosis

        feedback_lower = feedback.lower()
        mentions_pointcloud = bool(
            re.search(
                r"point\s*cloud|pointcloud|centroid|grasp candidate|candidate grasp|"
                r"selected grasp|old points|previous points",
                feedback_lower,
            )
        )
        recommends_reuse = bool(
            re.search(
                r"\b(reuse|use|keep|same|previous|prior|old|cached|saved)\b",
                feedback_lower,
            )
        )
        already_guarded = "fresh observation" in feedback_lower or "recompute" in feedback_lower
        if not (mentions_pointcloud and recommends_reuse) or already_guarded:
            return diagnosis

        guarded = dict(diagnosis)
        guarded["policy_feedback"] = (
            feedback.rstrip()
            + " Staleness guard: the saved pointcloud/centroid/grasp candidates "
            "are only snapshots from before later arm/gripper motion. Do not "
            "reuse their numeric coordinates blindly; first recompute the "
            "object pointcloud from a fresh observation/current agentview+wrist "
            "view, then reuse only the successful strategy or object feature."
        )
        if guarded.get("failure_reason") == feedback:
            guarded["failure_reason"] = guarded["policy_feedback"]
        return guarded

    def _has_visual_context(self, execution_result: dict[str, Any]) -> bool:
        return bool(
            execution_result.get("trajectory_video_data_url")
            or execution_result.get("trajectory_frames")
            or execution_result.get("before_frame") is not None
            or execution_result.get("after_frame") is not None
        )

    def _is_timeout_failure(self, execution_result: dict[str, Any]) -> bool:
        artifacts = execution_result.get("artifacts", {}) or {}
        if execution_result.get("timeout") or artifacts.get("timeout"):
            return True
        text = "\n".join(
            str(execution_result.get(k, "") or "")
            for k in ("stderr", "timeout_message")
        )
        return any(marker in text for marker in self._TIMEOUT_MARKERS)

    def _timeout_budget_string(self, execution_result: dict[str, Any]) -> str:
        artifacts = execution_result.get("artifacts", {}) or {}
        timeout_s = (
            execution_result.get("timeout_seconds")
            or artifacts.get("timeout_seconds")
        )
        if not timeout_s:
            text = "\n".join(
                str(execution_result.get(k, "") or "")
                for k in ("stderr", "timeout_message")
            )
            match = re.search(r"(?:after|for)\s+(\d+(?:\.\d+)?)s", text)
            timeout_s = match.group(1) if match else None
        return f"{timeout_s}s" if timeout_s else "the execution budget"

    def _extract_timeout_context(
        self,
        execution_result: dict[str, Any],
        code: str,
    ) -> str:
        """Summarize where a timeout surfaced so vision can tie motion to code."""
        stderr = str(execution_result.get("stderr", "") or "")
        timeout_message = str(execution_result.get("timeout_message", "") or "")
        text = stderr or timeout_message
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        last_line = lines[-1] if lines else "timeout flag was set without stderr"
        budget_str = self._timeout_budget_string(execution_result)

        code_lines = code.splitlines()
        policy_frames: list[tuple[int, str, str]] = []
        for match in re.finditer(
            r'File "<string>", line (\d+), in ([^\n]+)\n\s*([^\n]*)',
            stderr,
        ):
            line_no = int(match.group(1))
            fn_name = match.group(2).strip()
            traceback_code = match.group(3).strip()
            source_line = (
                code_lines[line_no - 1].strip()
                if 0 < line_no <= len(code_lines)
                else traceback_code
            )
            policy_frames.append((line_no, fn_name, source_line))

        stack_frames: list[str] = []
        for path, line_no, fn_name in re.findall(
            r'File "([^"]+)", line (\d+), in ([^\n]+)',
            stderr,
        ):
            if path == "<string>":
                continue
            if "/rats/" in path:
                short_path = path.split("/rats/")[-1]
                stack_frames.append(f"{fn_name.strip()} ({short_path}:{line_no})")

        blocks = [
            "The attempt timed out. Use this as visual evidence of where the "
            "robot/code got stuck; do not collapse the diagnosis to a generic "
            "'shorten the policy' message if the filmstrip shows a physical stall.",
            f"Timeout budget/signal: {budget_str}; {last_line}",
        ]

        if policy_frames:
            blocks.append("Policy-code traceback frames:")
            for line_no, fn_name, source_line in policy_frames[-4:]:
                blocks.append(f"  - line {line_no} in {fn_name}: {source_line}")
            line_no = policy_frames[-1][0]
            start = max(1, line_no - 3)
            end = min(len(code_lines), line_no + 3)
            snippet = "\n".join(
                f"{idx:>4}: {code_lines[idx - 1]}"
                for idx in range(start, end + 1)
            )
            if snippet:
                blocks.append("Code around the most specific timeout frame:")
                blocks.append(snippet)
        else:
            blocks.append(
                "No <string> policy traceback frame was available; the timeout "
                "was raised by the outer executor alarm or before Python "
                "reported the active policy line."
            )

        if stack_frames:
            blocks.append("Downstream simulator/API stack after the policy call:")
            blocks.extend(f"  - {frame}" for frame in stack_frames[-6:])

        diagnostics = self._get_api_diagnostics_summary(execution_result)
        if diagnostics:
            blocks.append(f"Runtime diagnostics: {diagnostics}")

        return "\n".join(blocks)

    def _timeout_fallback_response(
        self,
        execution_result: dict[str, Any],
        code: str,
    ) -> dict[str, Any]:
        budget_str = self._timeout_budget_string(execution_result)
        context = self._extract_timeout_context(execution_result, code)
        result = {
            "visual_success": False,
            "failed_step": "timeout",
            "failure_reason": f"Policy exceeded {budget_str} wall-clock budget.",
            "policy_feedback": (
                f"Previous attempt timed out after {budget_str}. No usable "
                "trajectory frames were available, so infer from the timeout "
                "traceback and runtime diagnostics: "
                f"{context} "
                "Shorten the wall-clock path by removing redundant hover/"
                "approach poses, hidden retry loops, and unnecessary scans; "
                "do NOT call set_arm_speed() to slow the robot down further. "
                "Gate each major phase on perception/runtime feedback "
                "(for example wrist-camera checks, inspect_at_wrist(...), or "
                "object-pose availability) and return as soon as the visible "
                "outcome is satisfied."
            ),
            "confidence": 0.75,
            "failure_mode": "timeout",
            "visual_predicate_status": [],
        }
        diagnostic_context = self._diagnostic_context_package(
            execution_result,
            focus_text=(
                result["policy_feedback"]
                + "\n"
                + str(execution_result.get("stderr", "") or "")
                + "\n"
                + code
            ),
        )
        if diagnostic_context:
            result["diagnostic_context"] = diagnostic_context
        return result

    # NOTE — 2-stage main-loop diagnoser (deferred refactor)
    #
    # The subagent diagnose path already does a cheap tier-1 outcome call
    # (`mode="subagent_outcome"`, ~2k token budget, 2 frames) before the
    # expensive tier-2 critique (`mode=None`, 8k tokens, full video). The
    # main-loop path here jumps straight to tier-2, which is why the
    # diagnoser dominates token cost on every retry.
    #
    # Proper 2-stage refactor steps:
    #   1. Add a `mode="main_outcome"` that mirrors `subagent_outcome` —
    #      reset + terminal frames + minimal prompt, returns
    #      `{visual_success, failure_mode, evidence}`.
    #   2. In lifelong_loop.py's diagnose call site, run main_outcome
    #      first. If `visual_success=True` AND verifier disagrees, surface
    #      the conflict to the user (verifier_challenge path); if
    #      visual_success matches the verifier verdict, return early
    #      without spending the 8k-token tier-2 call.
    #   3. Only run the full tier-2 critique when both verifier and
    #      tier-1 visual judge agree on failure.
    #
    # Not implemented here yet — the max_tokens bump + reasoning-loop
    # watchdog from earlier commits already prevent the worst of the
    # main-loop diagnoser pain in the LIBERO smoke run. Track this as a
    # follow-up; cost savings from skipping unnecessary tier-2 calls would
    # be substantial (≈70% of input tokens in the smoke run were
    # diagnoser+verifier on Gemini, which is also the path with 0% cache
    # hits).

    def diagnose(
        self,
        execution_result: dict[str, Any],
        scene_context: dict[str, Any],
        *,
        plan: dict[str, Any] | None = None,
        code: str = "",
        goal_predicates: list[Any] | None = None,
        affordance_hints: dict[str, Any] | None = None,
        prior_attempts: list[dict[str, Any]] | None = None,
        mode: str | None = None,
        tier1_evidence: str = "",
        task_in_progress: bool = False,
        verifier_challenge: str = "",
    ) -> dict[str, Any]:
        """Analyze execution failure using trajectory vision + code.

        Args:
            execution_result: From Executor. Preferably includes
                trajectory_video_data_url. May also include trajectory_frames
                (evenly sampled filmstrip) + before/after/wrist fallbacks.
            scene_context: Scene info with goal conditions.
            plan: Planner output (steps) — used to explain code intent.
            code: Executed policy code — for code-vs-visual cross-check.
            goal_predicates: BDDL goal predicates — per-predicate visual
                verdict is produced for each.
            affordance_hints: Optional dict of object → feature hints
                (e.g. {"porcelain_mug": "grip handle on side"}), written by
                task_proposer from its own priors (no in-context examples).
            prior_attempts: Earlier attempts in THIS iteration, each a dict
                with {attempt_idx, code, policy_feedback, failure_mode,
                visual_predicate_status, last_frame}. The diagnoser uses
                them to avoid re-attributing failure to sub-behaviors that
                already worked (e.g. if grasp succeeded in attempt 0, don't
                flag grasp as the failure again in attempt 1 unless there
                is new visual evidence of regression).
            mode: Optional diagnosis mode.
                - ``None``: default full-task filmstrip prompt.
                - ``"subagent_outcome"``: Tier-1 outcome call for a
                  sub-agent probe. Uses only reset + terminal frames
                  (trajectory_frames[0] and [-1]) and a minimal prompt
                  that returns ``visual_success`` + ``evidence`` +
                  ``failure_mode`` only — no spatial critique.
                - ``"subagent_critique"``: Tier-2 critique call. Given
                  that Tier-1 already decided failure, uses the full
                  filmstrip and prior Tier-1 evidence to produce the
                  spatial delta the next attempt should apply.
                - ``"subagent"``: legacy alias for ``"subagent_critique"``.
                  Kept so older call sites that pre-date the two-tier
                  split keep working.
            verifier_challenge: Optional retry-time challenge from the
                independent verifier when the verifier rejected the task but
                the initial diagnosis said no corrective action remained.

        Returns:
            Dict with:
              - visual_success: bool
              - failed_step: str | None
              - failure_reason: str
              - policy_feedback: str (concrete suggestion for Policy Writer)
              - confidence: float
              - failure_mode: str (from taxonomy)
              - visual_predicate_status: list of
                  {predicate, visually_satisfied, evidence}
        """
        # Fast path: execution clearly succeeded
        if not verifier_challenge and execution_result.get("success") and (
            execution_result.get("task_completed")
            or (execution_result.get("reward", 0) or 0) >= 0.99
        ):
            return {
                "visual_success": True,
                "failed_step": None,
                "failure_reason": "",
                "policy_feedback": "No corrective action needed.",
                "confidence": 1.0,
                "failure_mode": "none",
                "visual_predicate_status": [],
            }

        # Timeout path: prefer visual diagnosis when frames exist. Timeouts are
        # often physical failures (wedged against a wall, wrong approach angle,
        # repeated blocked goto_pose calls), so the policy writer needs the
        # partial trajectory plus the traceback/code line that was active.
        if self._is_timeout_failure(execution_result):
            timeout_context = self._extract_timeout_context(execution_result, code)
            enriched_result = dict(execution_result)
            enriched_result["_timeout_context"] = timeout_context
            if self._has_visual_context(enriched_result):
                return self._diagnose_with_vision(
                    enriched_result,
                    scene_context,
                    plan=plan,
                    code=code,
                    goal_predicates=goal_predicates,
                    affordance_hints=affordance_hints,
                    prior_attempts=prior_attempts,
                    mode=mode,
                    tier1_evidence=tier1_evidence,
                    task_in_progress=task_in_progress,
                    verifier_challenge=verifier_challenge,
                )
            return self._timeout_fallback_response(execution_result, code)

        # Check for stderr (code bugs)
        stderr = execution_result.get("stderr", "")
        if stderr:
            if (
                self._has_structured_diagnostic_artifacts(execution_result)
                and self._stderr_suggests_perception_or_motion_issue(str(stderr))
            ):
                runtime_error_context = self._extract_runtime_error_context(
                    execution_result,
                    code,
                )
                enriched_result = dict(execution_result)
                enriched_result["_runtime_error_context"] = runtime_error_context
                return self._diagnose_with_vision(
                    enriched_result,
                    scene_context,
                    plan=plan,
                    code=code,
                    goal_predicates=goal_predicates,
                    affordance_hints=affordance_hints,
                    prior_attempts=prior_attempts,
                    mode=mode,
                    tier1_evidence=tier1_evidence,
                    task_in_progress=task_in_progress,
                    verifier_challenge=verifier_challenge,
                )
            last_line = stderr.strip().splitlines()[-1] if stderr.strip() else stderr
            diagnostics = self._get_api_diagnostics_summary(execution_result)
            extra = f" Perception/runtime diagnostics: {diagnostics}." if diagnostics else ""
            result = {
                "visual_success": False,
                "failed_step": "runtime",
                "failure_reason": last_line,
                "policy_feedback": (
                    f"Fix runtime error: {last_line}.{extra} "
                    "Avoid dynamic execution or hidden retry loops."
                ).replace("..", "."),
                "confidence": 0.9,
                "failure_mode": "code_bug",
                "visual_predicate_status": [],
            }
            diagnostic_context = self._diagnostic_context_package(
                execution_result,
                focus_text=(
                    result["policy_feedback"]
                    + "\n"
                    + str(execution_result.get("stderr", "") or "")
                    + "\n"
                    + code
                ),
            )
            if diagnostic_context:
                result["diagnostic_context"] = diagnostic_context
            return result

        # Check for explicit step failure markers in stdout
        stdout = execution_result.get("stdout", "")
        if "STEP_FAILED:" in stdout:
            failed_step = stdout.split("STEP_FAILED:")[-1].strip().splitlines()[0]
            return {
                "visual_success": False,
                "failed_step": failed_step,
                "failure_reason": f"Step '{failed_step}' reported failure",
                "policy_feedback": f"Revise the approach for '{failed_step}'. Use deterministic primitives.",
                "confidence": 0.8,
                "failure_mode": "partial_completion",
                "visual_predicate_status": [],
            }

        # Vision-based diagnosis — prefer full trajectory filmstrip, fall back
        # to before/after pair.
        has_trajectory = bool(
            execution_result.get("trajectory_video_data_url")
            or execution_result.get("trajectory_frames")
        )
        before_frame = execution_result.get("before_frame")
        after_frame = execution_result.get("after_frame")
        if has_trajectory or before_frame is not None or after_frame is not None:
            return self._diagnose_with_vision(
                execution_result,
                scene_context,
                plan=plan,
                code=code,
                goal_predicates=goal_predicates,
                affordance_hints=affordance_hints,
                prior_attempts=prior_attempts,
                mode=mode,
                tier1_evidence=tier1_evidence,
                task_in_progress=task_in_progress,
                verifier_challenge=verifier_challenge,
            )

        # Fallback: no frames, no stderr
        goal = scene_context.get("goal_conditions_nl", "goal condition")
        reward = execution_result.get("reward", 0) or 0
        diagnostics = self._get_api_diagnostics_summary(execution_result)
        diag_suffix = f" Diagnostics: {diagnostics}" if diagnostics else ""
        return {
            "visual_success": False,
            "failed_step": "goal-check",
            "failure_reason": f"Execution ended without satisfying: {goal}",
            "policy_feedback": f"Re-plan the interaction sequence. Current reward: {reward:.2f}.{diag_suffix}",
            "confidence": 0.4,
            "failure_mode": "nothing_happened" if reward == 0 else "partial_completion",
            "visual_predicate_status": [],
        }

    def _diagnose_with_vision(
        self,
        execution_result: dict[str, Any],
        scene_context: dict[str, Any],
        *,
        plan: dict[str, Any] | None = None,
        code: str = "",
        goal_predicates: list[Any] | None = None,
        affordance_hints: dict[str, Any] | None = None,
        prior_attempts: list[dict[str, Any]] | None = None,
        mode: str | None = None,
        tier1_evidence: str = "",
        task_in_progress: bool = False,
        verifier_challenge: str = "",
    ) -> dict[str, Any]:
        """Use LLM vision to analyze a trajectory filmstrip + code intent.

        Inputs to the VLM:
          - N filmstrip frames (time-ordered, start → end of episode)
          - AFTER wrist frame (gripper-down close-up)
          - Goal language + BDDL predicates (when available)
          - Plan steps + executed code (for code-vs-visual cross-check)
          - Affordance hints (from task_proposer, on-model object-feature cues)

        Output is JSON with per-predicate visual verdicts AND an overall
        policy_feedback string. See prompts/failure_diagnoser.txt for schema.
        """
        # Sub-agent probes run as a two-tier diagnosis:
        #   Tier 1 (``subagent_outcome``): reset+terminal only, minimal
        #       prompt, one question — did the subgoal happen?
        #   Tier 2 (``subagent_critique``): full motion filmstrip, Tier-1
        #       evidence inlined, produces the spatial delta.
        # ``subagent`` is kept as a legacy alias for ``subagent_critique``
        # so older call sites that pre-date the split still work.
        if mode == "subagent_outcome":
            prompt_path = "rats/prompts/failure_diagnoser_subagent_outcome.txt"
        elif mode in ("subagent_critique", "subagent"):
            prompt_path = "rats/prompts/failure_diagnoser_subagent.txt"
        else:
            prompt_path = "rats/prompts/failure_diagnoser.txt"
        prompt_template = Path(prompt_path).read_text()
        goal = scene_context.get("goal_conditions_nl", "unknown goal")
        diagnostics = self._get_api_diagnostics_summary(execution_result)
        diagnostic_artifacts_txt = self._diagnostic_artifact_summary(execution_result)
        raw_numeric_artifacts_txt = self._collect_raw_numeric_artifacts(execution_result)
        pointcloud_staleness_txt = self._pointcloud_staleness_warning(execution_result)

        # Build plan / code / predicate / affordance context blocks.
        plan_steps_txt = "(no plan provided)"
        if plan and plan.get("steps"):
            plan_steps_txt = "\n".join(
                f"  {s.get('id') or s.get('step_id') or '?'}: {s.get('description','')}"
                for s in plan["steps"][:12]
            )

        code_snip = code 
        # if len(code) < 10000 else code[:9700] + "\n# ... (truncated)"

        # `goal_predicates` is kept in the signature for compatibility
        # but deliberately NOT rendered into the prompt — passing BDDL's
        # symbolic checklist would leak the verifier's exact success
        # criterion (a baseline non-priv agent doesn't have this). The
        # diagnoser judges per plan step instead, from the agent's own
        # natural-language plan descriptions.
        _ = goal_predicates  # intentionally unused

        aff_txt = "(no affordance hints provided)"
        if affordance_hints:
            aff_txt = "\n".join(
                f"  - {k}: {v}" for k, v in affordance_hints.items()
            )

        # Prior attempts in this iteration. Each entry carries the earlier
        # attempt's code tail + the diagnoser's own prior feedback + visual
        # per-predicate verdict. This lets the VLM anchor on sub-behaviors
        # that already succeeded instead of re-attributing failure every
        # attempt (e.g. grasp works in attempt 0 → attempt 1 should not
        # claim grasp failed unless there is fresh visual evidence).
        prior_txt = "(no prior attempts — this is attempt 0)"
        prior_frame_urls: list[str] = []
        if prior_attempts:
            blocks = []
            # FIX (truncation audit): same critique as failure_memory /
            # library dedup / subagent retry directive. Was emitting
            # `policy_feedback[:400]`, only the first 4 visual-predicate
            # entries (and each truncated to 60 chars), and only the last
            # 12 non-empty code lines. The diagnoser is the most
            # information-bound caller in the loop and arbitrary slices
            # here destroy exactly the signal we need it to weigh against
            # the current attempt. Bound is now the per-iter retry budget
            # (`prior_attempts[-4:]` keeps token cost capped at 4
            # attempts × full content) and the model's 65k input window.
            for pa in prior_attempts[-4:]:
                idx = pa.get("attempt_idx", "?")
                pfb = str(pa.get("policy_feedback", "") or "")
                fmode = str(pa.get("failure_mode", "") or "")
                vps = pa.get("visual_predicate_status", []) or []
                vps_txt = ", ".join(
                    f"{'✓' if e.get('visually_satisfied') else '✗'}"
                    f"{e.get('step_id') or e.get('description') or '?'}"
                    for e in vps
                ) or "(none)"
                pcode = str(pa.get("code", "") or "")
                code_block = pcode if pcode else "(no code)"
                blocks.append(
                    f"  attempt {idx}:\n"
                    f"    diagnoser_said (failure_mode={fmode}): {pfb}\n"
                    f"    visual_predicates: {vps_txt}\n"
                    f"    code:\n```python\n{code_block}\n```"
                )
                # Final frame of the prior attempt, for visual comparison.
                last_f = pa.get("last_frame")
                if last_f is not None:
                    u = image_to_data_url(last_f)
                    if u:
                        prior_frame_urls.append(u)
            prior_txt = "\n".join(blocks)

        user_prompt = prompt_template.replace(
            "{goal}", str(goal or "")
        ).replace(
            "{plan_steps}", plan_steps_txt
        ).replace(
            "{code}", code_snip
        ).replace(
            "{affordance_hints}", aff_txt
        ).replace(
            "{prior_attempts}", prior_txt
        ).replace(
            # FIX (truncation audit): emit full stdout/stderr. The
            # diagnoser is the caller that most needs the runtime
            # output — argument-level fixes often live in the last
            # python traceback (which is more than 500 chars) or in a
            # series of "Connection refused" lines (which the [:1000]
            # slice happily decapitated). The executor/sandbox already
            # caps output upstream, so we don't unbound disk usage.
            "{stdout}", str(execution_result.get("stdout", "") or "")
        ).replace(
            "{stderr}", str(execution_result.get("stderr", "") or "")
        ).replace(
            "{tier1_evidence}", tier1_evidence or "(none provided)"
        )
        if diagnostics:
            user_prompt += f"\nRUNTIME DIAGNOSTICS: {diagnostics}\n"
        per_step_summary = self._get_per_step_verification_summary(execution_result)
        if per_step_summary:
            user_prompt += (
                "\nPER-STEP VERIFICATION SUMMARY:\n"
                "This post-execution verifier used only each plan-step goal "
                "and runtime output artifacts/logs. It did not inspect policy "
                "code and did not trust the policy RESULT dict. Treat it as "
                "evidence about which physical sub-goal likely failed, not as "
                "a substitute for visual diagnosis.\n"
                f"{per_step_summary}\n"
            )
        if diagnostic_artifacts_txt:
            user_prompt += (
                "\nPERCEPTION / GRASP / MOTION DIAGNOSTICS:\n"
                "These are structured runtime facts from the robot API. Use "
                "them to locate the task object in 3D, compare agentview vs "
                "wrist pointclouds, see which grasp candidates existed and "
                "which grasp was selected, and connect executed goto_pose / "
                "IK targets to the visual trajectory. Treat them as hints, "
                "not as privileged ground-truth success labels.\n"
                f"{diagnostic_artifacts_txt}\n"
            )
        if raw_numeric_artifacts_txt:
            user_prompt += (
                "\nRAW NUMERIC ARTIFACT FILES:\n"
                "The full raw artifacts are saved on disk at the listed paths. "
                "Below are compact excerpts loaded from those files so you can "
                "reason over actual numbers, not just rendered images. NPZ "
                "pointcloud files include full/segmented world-frame points, "
                "camera intrinsics/extrinsics, masks, and robot pose where "
                "available. Grasp NPZ files include all candidate transforms "
                "and scores. Motion JSON files include target joints/poses and "
                "observed before/after/error robot states.\n"
                f"{raw_numeric_artifacts_txt}\n"
            )
        if pointcloud_staleness_txt:
            user_prompt += (
                "\nPOINTCLOUD STALENESS CHECK:\n"
                "Pointclouds/grasp candidates are time-local snapshots from "
                "when the policy called perception. They are not persistent "
                "ground truth after the arm contacts or moves objects.\n"
                f"{pointcloud_staleness_txt}\n"
            )
        timeout_context = str(execution_result.get("_timeout_context", "") or "")
        if timeout_context:
            user_prompt += (
                "\nTIMEOUT CONTEXT:\n"
                f"{timeout_context}\n\n"
                "TIMEOUT DIAGNOSIS INSTRUCTIONS:\n"
                "- The trajectory ended because execution timed out. Use the "
                "filmstrip to identify whether the robot was physically stuck, "
                "pushing against an obstacle/fixture, approaching from a bad "
                "angle, repeatedly retrying, or simply executing too many "
                "unnecessary motions.\n"
                "- Tie the visual stall to the listed policy-code line or "
                "primitive when possible. In policy_feedback, name the motion "
                "primitive/code block most likely responsible and describe the "
                "cleaner movement to generate next.\n"
                "- If the task outcome is visually complete despite the "
                "timeout, still mention that the code timed out after "
                "completion and advise returning immediately after verifying "
                "the satisfied state.\n"
            )
        runtime_error_context = str(execution_result.get("_runtime_error_context", "") or "")
        if runtime_error_context:
            user_prompt += (
                "\nRUNTIME ERROR CONTEXT:\n"
                f"{runtime_error_context}\n\n"
                "RUNTIME ERROR DIAGNOSIS INSTRUCTIONS:\n"
                "- Do not stop at 'code bug'. Explain why the error happened "
                "using the pointcloud/grasp/motion artifacts, raw numeric "
                "excerpts, code, and images. For example, if the error is "
                "'object not found' or 'no valid points', determine whether "
                "the object was outside view, occluded, moved after an older "
                "pointcloud, segmented as the wrong object, filtered by depth, "
                "or targeted with the wrong text prompt.\n"
                "- In policy_feedback, give the writer the concrete perception "
                "or motion change needed next, not merely the exception text.\n"
            )
        if verifier_challenge:
            user_prompt += (
                "\nVERIFIER CHALLENGE:\n"
                "An independent task verifier did NOT accept the previous "
                "turn as successful, but the initial failure diagnosis "
                "reported no remaining corrective action. Re-evaluate the "
                "images, code, and plan under the assumption that the task "
                "is not accepted as complete. Do NOT answer "
                "visual_success=true, failure_mode=none, or 'No corrective "
                "action needed' unless you can cite clear visual evidence "
                "that the verifier is likely wrong. If uncertain, choose "
                "the most plausible remaining failure mode and give a "
                "concrete next corrective action. Do not recommend a "
                "no-op/RESULT-only turn.\n"
                f"{verifier_challenge.strip()}\n"
            )

        # Note: the turn-mode clause is appended AFTER `n_film` is
        # computed below, so it can reference the actual frame count.
        # Don't move it up here — earlier code referenced n_film before
        # assignment, which raised UnboundLocalError mid-iteration and
        # killed the loop on every iteration that hit it.

        # Vision inputs — order matters for the prompt.
        # Preferred: time-ordered execution video + wrist / diagnostic images.
        # Fallback: sampled filmstrip or before/after agentview + wrist.
        videos: list[str] = []
        images: list[str] = []
        diagnoser_input_videos: list[dict[str, Any]] = []
        diagnoser_input_images: list[dict[str, Any]] = []

        def add_input_video(
            data_url: str | None,
            *,
            kind: str,
            label: str,
            **metadata: Any,
        ) -> bool:
            if not data_url:
                return False
            videos.append(data_url)
            record = {
                "video_index": len(videos),
                "kind": kind,
                "label": label,
                "data_url": data_url,
            }
            for key, value in metadata.items():
                if value not in (None, ""):
                    record[key] = str(value)
            diagnoser_input_videos.append(record)
            return True

        def add_input_image(
            data_url: str | None,
            *,
            kind: str,
            label: str,
            **metadata: Any,
        ) -> bool:
            if not data_url:
                return False
            images.append(data_url)
            record = {
                "image_index": len(images),
                "kind": kind,
                "label": label,
                "data_url": data_url,
            }
            for key, value in metadata.items():
                if value not in (None, ""):
                    record[key] = str(value)
            diagnoser_input_images.append(record)
            return True

        visual_input_mode = os.environ.get(
            "RATS_DIAGNOSER_VISUAL_INPUT",
            "video",
        ).strip().lower()
        force_frame_inputs = visual_input_mode in {
            "frame",
            "frames",
            "filmstrip",
            "image",
            "images",
            "still",
            "stills",
        }
        trajectory_video_url = execution_result.get("trajectory_video_data_url")
        trajectory_video_frame_count = (
            execution_result.get("trajectory_video_frame_count")
            or execution_result.get("trajectory_frame_count")
        )
        trajectory_video_fps = execution_result.get("trajectory_video_fps")
        has_trajectory_video = bool(
            mode != "subagent_outcome"
            and not force_frame_inputs
            and isinstance(trajectory_video_url, str)
            and trajectory_video_url.strip()
            and add_input_video(
                trajectory_video_url.strip(),
                kind="trajectory_video",
                label="Agent-view execution trajectory video",
                frame_count=trajectory_video_frame_count,
                fps=trajectory_video_fps,
            )
        )

        filmstrip = execution_result.get("trajectory_frames") or []
        if has_trajectory_video:
            n_film = 0
        elif filmstrip:
            n_film = 0
            for frame_idx, f in enumerate(filmstrip, start=1):
                u = image_to_data_url(f)
                if add_input_image(
                    u,
                    kind="trajectory_frame",
                    label=f"Trajectory frame {frame_idx}/{len(filmstrip)}",
                    frame_index=frame_idx,
                    frame_count=len(filmstrip),
                ):
                    n_film += 1
        else:
            n_film = 0
            before_url = image_to_data_url(execution_result.get("before_frame"))
            after_url = image_to_data_url(execution_result.get("after_frame"))
            add_input_image(
                before_url,
                kind="before_frame",
                label="Before agent-view frame",
            )
            add_input_image(
                after_url,
                kind="after_frame",
                label="After agent-view frame",
            )

        wrist_url = image_to_data_url(execution_result.get("after_wrist_frame"))
        has_wrist = bool(wrist_url)
        add_input_image(
            wrist_url,
            kind="after_wrist_frame",
            label="After wrist-camera frame",
        )

        artifact_labels, artifact_image_urls = self._collect_diagnostic_visual_artifacts(
            execution_result,
        )
        base_visual_image_count = len(images)
        diagnostic_image_layout = ""
        if artifact_image_urls:
            first_artifact_idx = len(images) + 1
            artifact_lines = []
            for label, artifact_url in zip(artifact_labels, artifact_image_urls, strict=False):
                image_idx = len(images) + 1
                bits = [
                    f"image {image_idx}",
                    label.get("event", "diagnostic"),
                ]
                if label.get("camera"):
                    bits.append(f"camera={label['camera']}")
                if label.get("label"):
                    bits.append(f"label={label['label']}")
                bits.append(f"path={label.get('path', '')}")
                artifact_lines.append("  - " + "; ".join(bits))
                add_input_image(
                    artifact_url,
                    kind="diagnostic_artifact",
                    label="; ".join(bits[1:]),
                    source=label.get("source", ""),
                    event=label.get("event", ""),
                    key=label.get("key", ""),
                    path=label.get("path", ""),
                    camera=label.get("camera", ""),
                    artifact_label=label.get("label", ""),
                )
            last_artifact_idx = len(images)
            user_prompt += (
                "\nDIAGNOSTIC ARTIFACT IMAGES:\n"
                "After the trajectory video or trajectory/wrist images, "
                "additional images show "
                "saved pointcloud/grasp overlays from the run:\n"
                + "\n".join(artifact_lines)
                + "\n"
            )
            if has_trajectory_video:
                diagnostic_image_layout = (
                    f" Diagnostic artifact images {first_artifact_idx}.."
                    f"{last_artifact_idx} follow the trajectory video and any "
                    "wrist/terminal images; they visualize segmented "
                    "pointclouds and grasp candidates in world coordinates."
                )
            elif base_visual_image_count:
                diagnostic_image_layout = (
                    f" Diagnostic artifact images {first_artifact_idx}.."
                    f"{last_artifact_idx} follow the trajectory/wrist images; "
                    "they visualize segmented pointclouds and grasp candidates "
                    "in world coordinates."
                )
            else:
                diagnostic_image_layout = (
                    f" Diagnostic artifact images {first_artifact_idx}.."
                    f"{last_artifact_idx} are the only images; they visualize "
                    "segmented pointclouds and grasp candidates in world "
                    "coordinates."
                )

        if task_in_progress:
            # Turn mode: env state from earlier turns persists. Tell the
            # diagnoser to recommend the NEXT step rather than a from-
            # scratch re-attempt — sub-actions whose effects are visible
            # in the LAST filmstrip frame should not be redone.
            if has_trajectory_video:
                user_prompt += (
                    "\nTURN MODE: This turn ran on a NON-RESET environment. "
                    "The first moment of the trajectory video already reflects "
                    "the cumulative effects of all earlier turns within this "
                    "attempt (objects moved, joints actuated, gripper state). "
                    "Frame the policy_feedback as 'what should happen NEXT "
                    "given the current state' — do NOT propose redoing "
                    "sub-actions whose effects are still visible near the end "
                    "of the video. If the task is already satisfied by the "
                    "end of the video, set visual_success=true.\n"
                )
            else:
                terminal_frame_n = n_film if n_film else 'N'
                user_prompt += (
                    "\nTURN MODE: This turn ran on a NON-RESET environment. "
                    "The scene at frame 1 already reflects the cumulative "
                    "effects of all earlier turns within this attempt (objects "
                    "moved, joints actuated, gripper state). Frame the "
                    "policy_feedback as 'what should happen NEXT given the "
                    "current state' — do NOT propose redoing sub-actions whose "
                    "effects are still visible in the last 2-3 filmstrip "
                    f"frames. If the task is already satisfied at frame "
                    f"{terminal_frame_n}, set visual_success=true.\n"
                )

        if has_trajectory_video:
            frame_count_txt = (
                f" ({trajectory_video_frame_count} recorded frames"
                + (f" at {trajectory_video_fps} fps" if trajectory_video_fps else "")
                + ")"
                if trajectory_video_frame_count
                else ""
            )
            image_layout_txt = (
                f"You have 1 agent-view execution video{frame_count_txt}. "
                "The video starts at this turn's initial state and ends at "
                "the terminal state after the policy stopped. Use the video "
                "for motion timing, approach direction, gripper contact, "
                "object displacement, stalls, and whether the arm reversed "
                "course."
                + (" Image 1 is the wrist camera at terminal state." if has_wrist else "")
                + diagnostic_image_layout
            )
        elif mode == "subagent_outcome" and n_film >= 2:
            # Tier 1 — outcome call. Caller passed exactly reset + terminal
            # (trajectory_frames[0] and [-1]); no motion frames. Prompt
            # narrows the task to a single outcome verdict so the VLM
            # doesn't get pulled into the weeds of mid-trajectory
            # corrections and call a succeeded-but-ugly grasp "failure".
            image_layout_txt = (
                "You have 2 agent-view frames: image 1 = RESET state "
                "(before the code ran), image 2 = TERMINAL state (after "
                "the code finished)."
                + (" Image 3 is the wrist camera at terminal state." if has_wrist else "")
                + " Decide outcome from image 2 vs image 1 only; do not "
                "attempt to infer the trajectory — a separate critique "
                "call handles that."
                + diagnostic_image_layout
            )
        elif mode in ("subagent_critique", "subagent") and n_film >= 2:
            # Tier 2 — critique call. Tier 1 already decided the attempt
            # failed; our job here is the spatial delta. Image 1 is
            # RESET, image n_film is TERMINAL, the rest are evenly-
            # sampled motion frames the critic reasons over.
            image_layout_txt = (
                f"You have {n_film} time-ordered agent-view frames: image 1 "
                f"= RESET state (before the code ran), image {n_film} = "
                f"TERMINAL state (after the code finished), images 2.."
                f"{n_film - 1} are evenly sampled motion frames showing the "
                f"approach trajectory."
                + (f" Image {n_film + 1} is the wrist camera at terminal state." if has_wrist else "")
                + " Tier-1 has already judged the outcome; focus on the "
                "motion frames to diagnose the spatial error (which axis "
                "was off, by roughly how much, in which direction)."
                + diagnostic_image_layout
            )
        elif n_film > 0:
            image_layout_txt = (
                f"You have {n_film} time-ordered filmstrip frames followed by "
                f"{'1 wrist-camera frame' if has_wrist else '0 wrist frames'}. "
                f"Frame 1 = START of episode, frame {n_film} = END. "
                f"The filmstrip lets you reason about motion (approach direction, "
                f"timing of close_gripper, whether the arm reversed course, etc.)."
                + diagnostic_image_layout
            )
        elif base_visual_image_count == 0 and artifact_image_urls:
            image_layout_txt = (
                "No trajectory or terminal camera frames were available. "
                f"You have {len(artifact_image_urls)} diagnostic artifact "
                "image(s) showing saved pointcloud/grasp overlays from the "
                "failed run. Use them with the raw numeric artifacts, stderr, "
                "and code to diagnose the runtime/perception failure."
            )
        elif base_visual_image_count == 0:
            image_layout_txt = (
                "No trajectory, terminal camera, or diagnostic visualization "
                "images were available. Use the raw numeric artifacts, "
                "runtime diagnostics, stderr, code, and plan to diagnose the "
                "runtime/perception failure."
            )
        else:
            image_layout_txt = (
                f"You have BEFORE agent-view (image 1), AFTER agent-view (image 2)"
                + (", and AFTER wrist camera (image 3)." if has_wrist else ".")
                + diagnostic_image_layout
            )

        system_prompt = (
            "You are a robotic execution failure diagnoser. Analyze what "
            "actually happened during the trajectory by combining:\n"
            "  1. VISION — the trajectory video or image sequence shows the full motion.\n"
            "  2. CODE — the Python policy that ran. Compare its intent to "
            "what the visual media show.\n"
            "  3. PLAN — the intended steps.\n"
            "  4. AFFORDANCE HINTS — cues about which object features the "
            "robot should have targeted (e.g. handle on a mug, door-pull on "
            "a microwave).\n"
            "  5. BDDL PREDICATES — per-predicate goal conditions. You will "
            "output a visual verdict for EACH predicate.\n\n"
            f"{image_layout_txt}\n\n"
            "The wrist camera (when present) is sent as a still image after "
            "any trajectory video/agent-view frames and before diagnostic "
            "artifact images. It looks straight DOWN from the gripper, so it is the "
            "best source for grasp alignment judgements — whether the "
            "gripper closed on the target or on empty space, whether the "
            "object is still held.\n\n"
            "The diagnostic pointcloud/grasp artifact images (when present) "
            "show where perception believed the task object was, which "
            "agentview/wrist segmentations contributed, and where top grasp "
            "candidates were relative to the scene/robot. Use them to "
            "explain wrong-object localization, bad approach angle, or "
            "collision/timeout stalls more concretely.\n\n"
            "The raw numeric artifact excerpts (when present) are loaded "
            "from the saved pointcloud, grasp, and robot-motion files. Use "
            "their actual coordinates, candidate scores, selected indices, "
            "and before/after robot poses to ground feedback in measurable "
            "spatial errors rather than vague motion advice.\n\n"
            "When the code and visual media disagree (e.g. code called close_gripper "
            "but visually the gripper never touched the object), flag that "
            "specifically — it often reveals a mis-localized grasp point or a "
            "targeting error against the wrong object feature."
        )

        try:
            # gpt-5.4 is a reasoning model: max_tokens caps reasoning +
            # output combined. With a ~7.5k-token prompt plus filmstrip,
            # the Gemini-3.1-pro-preview path routinely spends 5000+ tokens
            # on hidden reasoning. Pinned to 65536 — the cross-provider
            # ceiling (Gemini-3 Pro = 65536, Claude 4.x = 64k, gpt-5 = 128k)
            # — so reasoning + response always have room regardless of mode.
            max_tokens = 65536
            response = self._query_llm_text(
                system_prompt,
                user_prompt,
                images=images,
                videos=videos,
                max_tokens=max_tokens,
            )
            # Retry-on-empty: Gemini-3.1-pro periodically (~20% of calls
            # in libero_main_10iter) finishes with content="" and
            # finish_reason="stop" after burning 1000-6000 reasoning
            # tokens looping on the same observation. The model
            # self-terminates well below the token cap, so bumping
            # max_tokens does nothing; re-issuing the call from scratch
            # un-sticks roughly half of these because the sampler takes
            # a different reasoning path. Bounded to one retry per call
            # so a persistently-stuck case can't burn the whole budget.
            if not (response or "").strip():
                logger.warning(
                    f"FailureDiagnoser: empty content from "
                    f"{self.model or 'default model'} "
                    f"(likely Gemini reasoning-loop with finish_reason=stop); "
                    f"retrying once"
                )
                response = self._query_llm_text(
                    system_prompt,
                    user_prompt,
                    images=images,
                    videos=videos,
                    max_tokens=max_tokens,
                )
                if not (response or "").strip():
                    logger.warning(
                        "FailureDiagnoser: retry also returned empty "
                        "content; falling through to default parser. "
                        "Diagnosis will likely be 'nothing_happened' for "
                        "this attempt."
                    )
            parsed = self._parse_diagnosis_response(
                response, goal_predicates or [], mode=mode,
            )
            guarded = self._guard_stale_pointcloud_feedback(parsed, execution_result)
            diagnostic_context = self._diagnostic_context_package(
                execution_result,
                focus_text="\n".join([
                    str(guarded.get("policy_feedback", "") or ""),
                    str(guarded.get("failure_mode", "") or ""),
                    str(guarded.get("failed_step", "") or ""),
                    str(execution_result.get("stderr", "") or ""),
                    str(execution_result.get("stdout", "") or ""),
                    code,
                ]),
                artifact_labels=artifact_labels,
                artifact_image_urls=artifact_image_urls,
            )
            if diagnostic_context:
                guarded["diagnostic_context"] = diagnostic_context
            if diagnoser_input_videos:
                guarded["diagnoser_input_videos"] = diagnoser_input_videos
            if diagnoser_input_images:
                guarded["diagnoser_input_images"] = diagnoser_input_images
            return guarded
        except Exception as e:
            result = {
                "visual_success": False,
                "failed_step": "diagnosis_error",
                "failure_reason": str(e),
                "policy_feedback": "Vision diagnosis failed. Retry with different approach.",
                "confidence": 0.3,
                "failure_mode": "nothing_happened",
                "visual_predicate_status": [],
            }
            if diagnoser_input_images:
                result["diagnoser_input_images"] = diagnoser_input_images
            return result

    @staticmethod
    def _normalize_edit_scale(raw: Any, feedback: str) -> str | None:
        """Normalize edit_scale field to 'argument_level' | 'rewrite_needed' | None.

        Prefers the structured field; falls back to scanning the
        policy_feedback prose for the labels the older prompt asked the
        LLM to prefix. Returns None when the value is missing/unknown so
        the policy writer's retry banner stays silent rather than
        misleading the next attempt.
        """
        if isinstance(raw, str):
            token = raw.strip().lower().replace("-", "_")
            if token in ("argument_level", "argument", "args", "arg_level"):
                return "argument_level"
            if token in ("rewrite_needed", "rewrite", "full_rewrite", "rewrite_required"):
                return "rewrite_needed"
        # Prose-prefix fallback for older runs / cached responses.
        text = (feedback or "").lower()
        if "argument-level" in text or "argument_level" in text:
            return "argument_level"
        if "rewrite-needed" in text or "rewrite_needed" in text:
            return "rewrite_needed"
        return None

    def _parse_diagnosis_response(
        self,
        response: str,
        goal_predicates: list[Any],
        *,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Parse JSON response, fall back to legacy KEY=VALUE if needed."""
        import json as _json
        import re

        # Try JSON first (the new prompt asks for a fenced JSON block).
        parsed_json: dict[str, Any] | None = None
        # Extract fenced block if present
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        candidate = match.group(1) if match else response.strip()
        # Also try a bare top-level object
        if not match:
            obj_match = re.search(r"\{.*\}", response, re.DOTALL)
            if obj_match:
                candidate = obj_match.group(0)
        try:
            parsed_json = _json.loads(candidate)
        except Exception:
            parsed_json = None

        # Tier-1 outcome schema is flat: {visual_success, evidence,
        # failure_mode, confidence}. Synthesize a single-entry VPS so
        # callers that key off ``visual_predicate_status[0]`` (the whole
        # subagent loop does this) keep working without a branch.
        if mode == "subagent_outcome" and isinstance(parsed_json, dict):
            success = bool(parsed_json.get("visual_success", False))
            evidence = str(parsed_json.get("evidence", "") or "")
            failure_mode = str(
                parsed_json.get("failure_mode", "nothing_happened")
                or "nothing_happened"
            )
            confidence = float(parsed_json.get("confidence", 0.7) or 0.7)
            return {
                "visual_success": success,
                "failed_step": None if success else "step-1",
                "failure_reason": "" if success else evidence,
                "policy_feedback": "",  # critique handled by Tier 2
                "confidence": confidence,
                "failure_mode": "none" if success else (failure_mode or "nothing_happened"),
                "visual_predicate_status": [{
                    "step_id": "step-1",
                    "description": "",
                    "visually_satisfied": success,
                    "evidence": evidence,
                }],
                "plan_issue": False,
                "plan_issue_reason": "",
                "subagent_skill_target": None,
                "subagent_skill_target_reason": "",
            }

        if isinstance(parsed_json, dict):
            success = bool(parsed_json.get("visual_success", False))
            failed_step = parsed_json.get("failed_step") or None
            if failed_step in ("none", ""):
                failed_step = None
            feedback = str(parsed_json.get("policy_feedback", "") or "")
            failure_mode = str(
                parsed_json.get("failure_mode", "nothing_happened") or "nothing_happened"
            )
            # VPS is now one entry per plan step (not per BDDL predicate).
            # Schema: {step_id, description, visually_satisfied, evidence}.
            # Mechanism A keys off step_id for preserved-code extraction.
            raw_vps = parsed_json.get("visual_predicate_status", []) or []
            vps: list[dict[str, Any]] = []
            for entry in raw_vps:
                if not isinstance(entry, dict):
                    continue
                step_id = entry.get("step_id")
                if step_id in (None, "", "null"):
                    step_id = None
                else:
                    step_id = str(step_id)
                vps.append({
                    "step_id": step_id,
                    "description": str(entry.get("description", "") or ""),
                    "visually_satisfied": bool(entry.get("visually_satisfied", False)),
                    "evidence": str(entry.get("evidence", "") or ""),
                })
            confidence = float(parsed_json.get("confidence", 0.7) or 0.7)
            # plan_issue signals a structural plan-level failure (wrong
            # ordering, infeasible prerequisite, missing step) as opposed
            # to a code-level fix. When true, the feedback generator
            # routes into a replan path instead of just regenerating
            # code under the same plan.
            plan_issue = bool(parsed_json.get("plan_issue", False))
            plan_issue_reason = str(parsed_json.get("plan_issue_reason", "") or "")
            # Sub-agent isolation target. When non-null, the outer loop
            # spawns a SubAgent focused on practicing this specific
            # sub-behavior in isolation (env reset to initial state, own
            # retry budget, success judged visually on a single-step plan).
            subagent_target = parsed_json.get("subagent_skill_target")
            if subagent_target in (None, "", "null"):
                subagent_target = None
            else:
                subagent_target = str(subagent_target)
            subagent_reason = str(
                parsed_json.get("subagent_skill_target_reason", "") or ""
            )
            # Parallel-sub-agent approaches: 2-3 distinct strategies the
            # orchestrator should run concurrently. Empty list means
            # fall back to a single-approach sequential sub-agent. Each
            # entry must be a non-empty short string; we silently drop
            # anything else so a malformed entry doesn't kill the whole
            # diagnosis.
            raw_approaches = parsed_json.get("subagent_approaches", []) or []
            subagent_approaches: list[str] = []
            if isinstance(raw_approaches, list) and subagent_target is not None:
                for a in raw_approaches:
                    if not isinstance(a, str):
                        continue
                    a = a.strip()
                    if a:
                        subagent_approaches.append(a[:300])
                # Cap at 3 — beyond that the parallel cost outweighs the
                # diversity gain (vision servers serialize anyway).
                subagent_approaches = subagent_approaches[:3]
            # Tier-2 critique optionally flags disagreement with Tier-1.
            # Irrelevant for other modes but harmless to surface.
            tier1_disagreement = bool(parsed_json.get("tier1_disagreement", False))
            # Structured retry-edit scale. Drives the policy writer's
            # retry banner so it doesn't have to scrape the prose prefix
            # out of policy_feedback. Falls back to prefix-parsing on
            # older runs / cached responses; None means "no banner".
            edit_scale = self._normalize_edit_scale(
                parsed_json.get("edit_scale"), feedback
            )
            return {
                "visual_success": success,
                "failed_step": failed_step,
                "failure_reason": "" if success else feedback,
                "policy_feedback": feedback or response.strip(),
                "confidence": confidence,
                "failure_mode": "none" if success else (failure_mode or "nothing_happened"),
                "visual_predicate_status": vps,
                "plan_issue": plan_issue,
                "plan_issue_reason": plan_issue_reason,
                "subagent_skill_target": subagent_target,
                "subagent_skill_target_reason": subagent_reason,
                "subagent_approaches": subagent_approaches,
                "tier1_disagreement": tier1_disagreement,
                "edit_scale": edit_scale,
            }

        # Legacy fallback: KEY=VALUE lines.
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        parsed: dict[str, str] = {}
        for line in lines:
            if "=" in line:
                key, value = line.split("=", 1)
                parsed[key.strip().upper()] = value.strip()

        success = parsed.get("SUCCESS", "no").lower() in ("yes", "true")
        failed_step = parsed.get("FAILED_STEP")
        if failed_step in (None, "none", ""):
            failed_step = None
        feedback = parsed.get("FEEDBACK", response.strip())

        failure_mode = "nothing_happened"
        feedback_lower = feedback.lower()
        if "grasp" in feedback_lower:
            failure_mode = "grasp_failure"
        elif "navigat" in feedback_lower or "position" in feedback_lower:
            failure_mode = "navigation_error"
        elif "wrong object" in feedback_lower:
            failure_mode = "wrong_object"
        elif "collis" in feedback_lower:
            failure_mode = "collision"
        elif "partial" in feedback_lower:
            failure_mode = "partial_completion"

        return {
            "visual_success": success,
            "failed_step": failed_step,
            "failure_reason": "" if success else feedback,
            "policy_feedback": feedback,
            "confidence": 0.7,
            "failure_mode": "none" if success else failure_mode,
            "visual_predicate_status": [],
            "edit_scale": self._normalize_edit_scale(None, feedback),
        }
