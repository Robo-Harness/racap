"""Human-facing artifacts for recorded ReAct evaluation rollouts."""

from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_RAW_SENSOR_KEYS = {
    "images",
    "rgb",
    "depth",
    "segmentation",
    "pointcloud",
    "point_cloud",
    "camera_image",
    "camera_images",
}


def _nested_shape(value: Any, *, depth: int = 0) -> list[int]:
    """Infer a rectangular tensor shape without copying a nested pixel list."""
    if depth >= 8:
        return []
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.shape]
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        return [len(value), *_nested_shape(value[0], depth=depth + 1)]
    return []


def _sensor_manifest(value: Any) -> dict[str, Any]:
    """Describe a sensory payload while omitting redundant raw pixels.

    Rollout videos retain the actual RGB stream.  Keeping every RGB/depth array
    again inside every tool report made a single human-readable trajectory tens
    of megabytes and obscured the action/state evidence that matters for
    debugging.  The manifest preserves channel names, shapes and dtypes.
    """
    if isinstance(value, dict):
        return {
            "raw_sensor_payload_omitted": True,
            "channels": {
                str(key): _sensor_manifest(item) for key, item in value.items()
            },
        }
    shape = _nested_shape(value)
    dtype = str(value.dtype) if isinstance(value, np.ndarray) else "json_numeric"
    count = 1
    for dimension in shape:
        count *= max(0, int(dimension))
    return {
        "raw_sensor_payload_omitted": True,
        "shape": shape,
        "dtype": dtype,
        "elements": count,
    }


def compact_trajectory_row(value: Any, *, _key: str = "") -> Any:
    """Return a JSON-safe trajectory with raw sensory tensors summarized.

    This function changes only persisted artifacts after policy execution.  It
    does not alter observations visible to the controller or evaluator oracle.
    A generic large numeric-list fallback also protects future camera schemas
    whose field names are not yet known.
    """
    key = _key.lower()
    if key in _RAW_SENSOR_KEYS or key.endswith(("_rgb", "_image", "_images")):
        return _sensor_manifest(value)
    if isinstance(value, np.ndarray):
        if value.size > 4096:
            return _sensor_manifest(value)
        return value.tolist()
    if isinstance(value, dict):
        return {
            str(child_key): compact_trajectory_row(child, _key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        shape = _nested_shape(value)
        count = 1
        for dimension in shape:
            count *= max(0, int(dimension))
        if count > 4096 and len(shape) >= 2:
            return _sensor_manifest(value)
        return [compact_trajectory_row(item, _key=_key) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def episode_slug(row: dict[str, Any]) -> str:
    return f"t{int(row['task_id']):03d}_seed{int(str(row['key']).rsplit('seed', 1)[-1])}"


def _compact(value: Any, limit: int = 180) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _card(size: tuple[int, int], lines: Iterable[str]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    width, height = size
    image = Image.new("RGB", size, (18, 21, 27))
    draw = ImageDraw.Draw(image)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    try:
        title_font = ImageFont.truetype(str(bold_path), 30)
        body_font = ImageFont.truetype(str(font_path), 22)
    except OSError:
        title_font = body_font = ImageFont.load_default()

    y = 56
    for index, source in enumerate(lines):
        font = title_font if index == 0 else body_font
        wrap_width = max(24, int(width / (18 if index == 0 else 13)))
        for line in textwrap.wrap(str(source), width=wrap_width) or [""]:
            draw.text((54, y), line, fill=(245, 247, 250), font=font)
            bbox = draw.textbbox((54, y), line, font=font)
            y = bbox[3] + 12
        y += 8
    return np.asarray(image)


def _write_video(
    path: Path,
    row: dict[str, Any],
    frames: list[np.ndarray],
    *,
    fps: int,
) -> None:
    import imageio.v2 as imageio

    if frames:
        height, width = np.asarray(frames[0]).shape[:2]
    else:
        width, height = 800, 512
    size = (width, height)
    result = "PASS" if row.get("native_success") else "FAIL"
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=2)

    def hold(card: np.ndarray, seconds: float) -> None:
        for _ in range(max(1, round(fps * seconds))):
            writer.append_data(card)

    try:
        hold(
            _card(
                size,
                [
                    f"Task {int(row['task_id']):03d} | {result}",
                    str(row.get("instruction", "")),
                    f"Latest ReAct rollout | {row.get('turn_budget', 27)} turns | "
                    f"pickplace {row.get('pickplace_budget', 8)} / "
                    f"push {row.get('push_budget', 3)} / "
                    f"insert {row.get('insert_budget', 2)}"
                    + (
                        f" / state {row.get('state_budget', 2)} / "
                        f"stack {row.get('stack_budget', 2)}"
                        if row.get("full_task")
                        else ""
                    ),
                ],
            ),
            1.5,
        )

        cursor = 0
        for step in row.get("steps") or []:
            frame_range = (step.get("report") or {}).get("video_frame_range")
            if not (isinstance(frame_range, list) and len(frame_range) == 2):
                continue
            start = max(cursor, min(len(frames), int(frame_range[0])))
            end = max(start, min(len(frames), int(frame_range[1])))
            for frame in frames[cursor:start]:
                writer.append_data(np.asarray(frame).astype("uint8"))
            hold(
                _card(
                    size,
                    [
                        f"Turn {int(step.get('turn', 0)) + 1}: {step.get('action', '')}",
                        f"Thought: {step.get('thought', '')}",
                        f"Args: {_compact(step.get('args') or {}, 260)}",
                    ],
                ),
                0.55,
            )
            for frame in frames[start:end]:
                writer.append_data(np.asarray(frame).astype("uint8"))
            cursor = end
        for frame in frames[cursor:]:
            writer.append_data(np.asarray(frame).astype("uint8"))

        hold(
            _card(
                size,
                [
                    f"Final: {result}",
                    f"Agent claimed: {'done' if row.get('agent_success') else 'not done'} | "
                    f"turns: {row.get('turns', 0)}",
                    f"Stop reason: {row.get('stopped') or row.get('error') or '-'}",
                ],
            ),
            1.5,
        )
    finally:
        writer.close()


_REPORT_KEYS = (
    "failure_mode",
    "grounding_failure",
    "failed_label",
    "params",
    "pick_pose",
    "pick_extent",
    "destination_pose",
    "destination_kind",
    "destination_extent",
    "destination_bbox",
    "grasp_strategy",
    "ladder",
    "carry_below_hand",
    "carry_offset",
    "release_xy",
    "release_z",
    "object_footprint_size",
    "opening_polygon_vertices",
    "feasible_centre_xy",
    "feasible_clearance_cm",
    "planned_object_yaw_deg",
    "planned_rotation_deg",
    "held_visual_measurement",
    "held_visual_feedback",
    "held_footprint_size",
    "held_object_yaw_deg",
    "held_clearance_before_cm",
    "visual_servo_translation_cm",
    "visual_servo_rotation_deg",
    "requested_insertion_depth_cm",
    "achieved_insertion_depth_cm",
    "go_home_hook",
    "reliable",
    "offset",
    "horizontal_error",
    "height_error",
    "suggested_nudge",
    "inventory",
    "verified_destination_aliases",
    "alias_evidence",
    "video_frame_range",
)


def _trace_markdown(row: dict[str, Any]) -> str:
    result = "PASS" if row.get("native_success") else "FAIL"
    lines = [
        f"# Task {int(row['task_id']):03d} — {result}",
        "",
        f"- Instruction: `{row.get('instruction', '')}`",
        f"- Pick: `{row.get('pick_label', '')}`",
        f"- Destination: `{row.get('destination', '')}`",
        f"- Native simulator result: **{result}**",
        f"- Agent called done: **{'yes' if row.get('agent_success') else 'no'}**",
        f"- Turns: **{row.get('turns', 0)}**",
        f"- Runtime: **{row.get('seconds', 0)} s**",
        f"- Stop reason: {row.get('stopped') or row.get('error') or '-'}",
        "- [Simulation video](rollout.mp4)",
        "- [Raw trajectory JSON](trajectory.json)",
        "",
    ]
    all_reflections = list(row.get("reflections") or [])
    if row.get("full_task") and all_reflections:
        lines.extend(
            [
                "## Task decomposition",
                "",
                "```json",
                str(all_reflections[0]).removeprefix("TASK PLAN\n").strip(),
                "```",
                "",
                "## Agent trajectory",
                "",
            ]
        )
        reflections = iter(())
    else:
        reflections = iter(all_reflections)
        lines.extend(["## Agent trajectory", ""])
    for step in row.get("steps") or []:
        action = str(step.get("action", ""))
        lines.extend(
            [
                f"### Turn {int(step.get('turn', 0)) + 1} — `{action}`",
                "",
                f"**Thought:** {step.get('thought') or '-'}",
                "",
                f"**Call:** `{action}({_compact(step.get('args') or {}, 1000)})`",
                "",
                f"**Observation:** {step.get('observation') or '-'}",
                "",
            ]
        )
        report = step.get("report") or {}
        evidence = {key: report[key] for key in _REPORT_KEYS if key in report}
        if evidence:
            lines.extend(["**Key evidence:**", ""])
            for key, value in evidence.items():
                lines.append(f"- `{key}`: `{_compact(value, 700)}`")
            lines.append("")
        if action == "pickplace":
            try:
                reflection = next(reflections)
            except StopIteration:
                reflection = ""
            if reflection:
                try:
                    reflection = json.dumps(json.loads(reflection), indent=2, ensure_ascii=False)
                except (TypeError, json.JSONDecodeError):
                    pass
                lines.extend(
                    [
                        "**Visual reflection:**",
                        "",
                        "```json",
                        str(reflection).strip(),
                        "```",
                        "",
                    ]
                )
        if report:
            lines.extend(
                [
                    "<details><summary>Full tool report</summary>",
                    "",
                    "```json",
                    json.dumps(report, indent=2, ensure_ascii=False, default=str),
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )
    if row.get("full_task") and len(all_reflections) > 1:
        lines.extend(["## Visual reflections", ""])
        for index, reflection in enumerate(all_reflections[1:], start=1):
            lines.extend(
                [
                    f"### Reflection {index}",
                    "",
                    "```json",
                    str(reflection).strip(),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_episode_artifacts(
    root: Path,
    row: dict[str, Any],
    frames: list[np.ndarray],
    *,
    fps: int,
) -> dict[str, Any]:
    compact_row = compact_trajectory_row(row)
    episode_dir = root / "episodes" / episode_slug(row)
    episode_dir.mkdir(parents=True, exist_ok=True)
    video_path = episode_dir / "rollout.mp4"
    trace_path = episode_dir / "trace.md"
    raw_path = episode_dir / "trajectory.json"
    _write_video(video_path, compact_row, frames, fps=fps)
    trace_path.write_text(_trace_markdown(compact_row), encoding="utf-8")
    raw_path.write_text(
        json.dumps(compact_row, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return {
        "episode_dir": str(episode_dir),
        "video": str(video_path),
        "trace": str(trace_path),
        "trajectory": str(raw_path),
        "video_frames": len(frames),
    }


def write_rollout_index(root: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["task_id"]), str(row["key"])))
    native = sum(bool(row.get("native_success")) for row in ordered)
    markdown = [
        "# Recorded ReAct rollouts",
        "",
        f"Recorded policy evaluation. Native result: "
        f"**{native}/{len(ordered)} ({native / len(ordered):.1%})**.",
        "",
        "Each video contains the continuous simulator motion plus cards marking "
        "the ReAct tool turns. `trace.md` contains thoughts, calls, observations, "
        "visual reflections, measurements, and the final stop reason.",
        "",
        "| Task | Native | Agent done | Agreement | Turns | Instruction | Video | Trace |",
        "|---:|:---:|:---:|:---:|---:|---|---|---|",
    ]
    cards = []
    for row in ordered:
        slug = episode_slug(row)
        rel = f"episodes/{slug}"
        result = "PASS" if row.get("native_success") else "FAIL"
        claimed = "yes" if row.get("agent_success") else "no"
        agrees = bool(row.get("native_success")) == bool(row.get("agent_success"))
        agreement = "yes" if agrees else "**NO**"
        instruction = str(row.get("instruction", "")).replace("|", "\\|")
        markdown.append(
            f"| {int(row['task_id'])} | {result} | {claimed} | {agreement} | "
            f"{row.get('turns', 0)} | "
            f"{instruction} | "
            f"[MP4]({rel}/rollout.mp4) | [trajectory]({rel}/trace.md) |"
        )
        css = "pass" if row.get("native_success") else "fail"
        mismatch = (
            '<strong class="mismatch">Agent/native mismatch</strong> · ' if not agrees else ""
        )
        cards.append(f"""
        <article class="{css}">
          <h2>Task {int(row["task_id"]):03d} · {result}</h2>
          <p>{html.escape(str(row.get("instruction", "")))}</p>
          <video controls preload="metadata" src="{rel}/rollout.mp4"></video>
          <p>{mismatch}agent done: {claimed} · {row.get("turns", 0)} turns · <a href="{rel}/trace.md">readable trace</a>
          · <a href="{rel}/trajectory.json">raw JSON</a></p>
        </article>""")
    (root / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (root / "index.html").write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Recorded ReAct rollouts</title>
<style>
body {{ font: 15px system-ui,sans-serif; margin: 24px; background:#11151b; color:#edf2f7 }}
.summary {{ position:sticky; top:0; background:#11151bee; padding:10px 0; z-index:2 }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:18px }}
article {{ border:1px solid #344050; border-left:6px solid; border-radius:8px; padding:14px; background:#1a2029 }}
article.pass {{ border-left-color:#38a169 }} article.fail {{ border-left-color:#e53e3e }}
video {{ width:100%; background:#000 }} a {{ color:#63b3ed }} h2 {{ margin:0 0 6px }}
.mismatch {{ color:#f6ad55 }}
</style></head><body>
<div class="summary"><h1>Recorded ReAct rollouts</h1>
<p>Native success: <strong>{native}/{len(ordered)} ({native / len(ordered):.1%})</strong>. Green is pass; red is fail.</p></div>
<main class="grid">{"".join(cards)}</main></body></html>
""",
        encoding="utf-8",
    )
