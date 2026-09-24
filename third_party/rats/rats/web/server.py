"""FastAPI server for CaP-X interactive web UI."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request as UrlRequest

import tyro
import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from rats.envs.configs.instantiate import instantiate
from rats.envs.runner import _start_api_servers
from rats.utils.launch_utils import _load_config
from rats.web.async_trial_runner import LaunchArgsCompat, run_trial_async
from rats.web.models import (
    CodeExecutionResultEvent,
    CodeExecutionStartEvent,
    ConfigListResponse,
    DecisionType,
    EnvironmentInitEvent,
    ExecutionStepEvent,
    ImageAnalysisEvent,
    InjectPromptCommand,
    LoadConfigRequest,
    LoadConfigResponse,
    ModelResponseEvent,
    SessionState,
    SessionStatusResponse,
    StartTrialRequest,
    StartTrialResponse,
    StateUpdateEvent,
    StopCommand,
    StopTrialRequest,
    StopTrialResponse,
    TrialCompleteEvent,
)
from rats.web.session_manager import Session, get_session_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Viser reverse-proxy helpers
# ---------------------------------------------------------------------------
_VISER_PORTS = list(range(8080, 8090))
_viser_port_cache: int | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _truncate_text(value: Any, *, max_chars: int = 80_000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n... truncated for Web UI display ({len(text)} total characters) ..."
    )


def _json_block(value: Any, *, max_chars: int = 16_000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, indent=2, default=str)
    return f"```json\n{_truncate_text(text, max_chars=max_chars)}\n```"


def _read_json_file(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _resolve_history_run_dir(run_dir: str | Path) -> Path:
    """Resolve a historical run dir while keeping file access inside the repo."""
    raw = Path(run_dir).expanduser()
    if not raw.is_absolute():
        raw = _project_root() / raw
    resolved = raw.resolve()
    root = _project_root().resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="Run directory must be inside this repository")
    if not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"Run directory not found: {run_dir}")
    if not list(resolved.glob("iteration_*.json")):
        raise HTTPException(status_code=400, detail=f"No iteration_*.json files found in: {run_dir}")
    return resolved


def _safe_history_artifact(run_dir: Path, rel_path: str) -> Path:
    candidate = (run_dir / rel_path).resolve()
    if candidate != run_dir and run_dir not in candidate.parents:
        raise HTTPException(status_code=400, detail="Artifact path escapes run directory")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {rel_path}")
    return candidate


def _history_run_summary(run_dir: Path) -> dict[str, Any]:
    iteration_paths = sorted(run_dir.glob("iteration_*.json"))
    mtimes = [p.stat().st_mtime for p in iteration_paths]
    for extra in ("report.md", "final_summary.json", "lifelong_summary.json"):
        p = run_dir / extra
        if p.exists():
            mtimes.append(p.stat().st_mtime)

    summary = _read_json_file(run_dir / "lifelong_summary.json", {}) or _read_json_file(
        run_dir / "final_summary.json", {}
    ) or {}
    rel = run_dir.relative_to(_project_root()) if _project_root() in run_dir.parents else run_dir
    return {
        "path": str(rel),
        "name": run_dir.name,
        "iteration_count": len(iteration_paths),
        "has_report": (run_dir / "report.md").exists(),
        "mtime": max(mtimes) if mtimes else run_dir.stat().st_mtime,
        "summary": {
            key: summary.get(key)
            for key in (
                "total_iterations",
                "successful_iterations",
                "failed_iterations",
                "final_skill_library_size",
                "learned_skills",
            )
            if key in summary
        },
    }


def _attempt_indices(iter_data: dict[str, Any]) -> list[int]:
    total = iter_data.get("total_attempts")
    if isinstance(total, int) and total > 0:
        return list(range(total))
    indices: set[int] = set()
    for key in iter_data:
        match = re.fullmatch(r"code_attempt_(\d+)", key)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _attempt_thumbnail_base64(run_dir: Path, iteration: int, attempt: int) -> str | None:
    candidates = [
        run_dir / "report_assets" / f"iter{iteration:03d}_attempt{attempt}.jpg",
        run_dir / f"iter{iteration:03d}_attempt{attempt}_before.png",
        run_dir / f"iter{iteration:03d}_attempt{attempt}_after.png",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            try:
                return base64.b64encode(path.read_bytes()).decode("utf-8")
            except Exception:
                return None
    return None


def _image_mime_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _history_image_file_to_data_url(path: Path) -> str | None:
    try:
        mime = _image_mime_from_path(path)
        payload = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{payload}"
    except Exception:
        return None


def _resolve_history_image_file(run_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [run_dir / raw]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved != run_dir and run_dir not in resolved.parents:
            continue
        if resolved.is_file():
            return resolved
    return None


def _history_artifact_relpath(run_dir: Path, value: Any) -> str | None:
    """Return a run-dir-relative artifact path from stored absolute/relative paths."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [run_dir / raw, _project_root() / raw]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            run_resolved = run_dir.resolve()
        except OSError:
            continue
        if resolved == run_resolved or run_resolved not in resolved.parents:
            continue
        if resolved.is_file():
            return str(resolved.relative_to(run_resolved))
    return None


def _first_history_artifact_relpath(run_dir: Path, *values: Any) -> str | None:
    for value in values:
        rel = _history_artifact_relpath(run_dir, value)
        if rel:
            return rel
    return None


def _strip_inline_image_data(value: Any) -> Any:
    if isinstance(value, dict):
        stripped: dict[str, Any] = {}
        for key, item in value.items():
            if key == "data_url" and isinstance(item, str) and item.startswith("data:image/"):
                stripped[key] = "<image data displayed below>"
            else:
                stripped[str(key)] = _strip_inline_image_data(item)
        return stripped
    if isinstance(value, list):
        return [_strip_inline_image_data(item) for item in value]
    return value


def _history_diagnosis_images(run_dir: Path, diagnosis: dict[str, Any]) -> tuple[list[str], str]:
    raw_records = diagnosis.get("diagnoser_input_images") or []
    if not isinstance(raw_records, list):
        return [], ""

    images: list[str] = []
    lines: list[str] = []
    for idx, raw in enumerate(raw_records, start=1):
        if isinstance(raw, dict):
            data_url = raw.get("data_url")
            image_url = data_url if isinstance(data_url, str) and data_url.startswith("data:image/") else None
            if image_url is None:
                for key in ("relative_file", "file", "path"):
                    image_path = _resolve_history_image_file(run_dir, raw.get(key))
                    if image_path is not None:
                        image_url = _history_image_file_to_data_url(image_path)
                        break
            if image_url is None:
                continue
            image_index = raw.get("image_index") or idx
            label = str(raw.get("label") or raw.get("kind") or f"image {image_index}")
            kind = str(raw.get("kind", "") or "")
        elif isinstance(raw, str) and raw.startswith("data:image/"):
            image_url = raw
            image_index = idx
            label = f"image {idx}"
            kind = ""
        else:
            continue

        images.append(image_url)
        bits = [f"image {image_index}", label]
        if kind:
            bits.append(f"kind={kind}")
        lines.append("- " + "; ".join(bit for bit in bits if bit))

    if not images:
        return [], ""
    return images, "Images sent to failure diagnoser:\n" + "\n".join(lines)


def _format_task_details(iter_data: dict[str, Any]) -> str:
    iteration = iter_data.get("iteration", "?")
    task = iter_data.get("task_proposal") or {}
    scene = iter_data.get("scene_context") or {}
    goal = task.get("goal_conditions") or task.get("goal") or task.get("language")
    lines = [
        f"## Iteration {iteration}",
        f"- Activity: `{task.get('activity_name') or task.get('language') or 'unknown'}`",
        f"- Scene: `{scene.get('scene_model') or task.get('scene_model') or 'unknown'}`",
        f"- Success: `{iter_data.get('success')}`",
        f"- Attempts: `{iter_data.get('total_attempts', len(_attempt_indices(iter_data)))}`",
    ]
    if goal:
        lines.append(f"- Goal: {goal}")
    if task.get("reasoning"):
        lines.extend(["", "### Proposer Reasoning", _truncate_text(task["reasoning"], max_chars=6_000)])
    return "\n".join(lines)


def _format_plan(iter_data: dict[str, Any]) -> str:
    plan = iter_data.get("plan") or {}
    lines = [f"## Iteration {iter_data.get('iteration', '?')} Plan"]
    steps = plan.get("steps") or []
    if isinstance(steps, list) and steps:
        for step in steps:
            sid = step.get("id", step.get("step_id", "?"))
            desc = step.get("description", "")
            skills = ", ".join(step.get("relevant_skills", []) or [])
            lines.append(f"- **{sid}**: {desc}" + (f"  \n  skills: `{skills}`" if skills else ""))
    else:
        lines.append(_json_block(plan, max_chars=20_000))
    return "\n".join(lines)


def _build_history_events(run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Convert saved iteration artifacts into the live Web UI event protocol."""
    session_id = "history"
    events: list[str] = []
    block_index = 0
    iteration_paths = sorted(run_dir.glob("iteration_*.json"))
    root = _project_root()
    rel_run = str(run_dir.relative_to(root)) if root in run_dir.parents else str(run_dir)

    def add(event: Any) -> None:
        events.append(event.model_dump_json())

    add(
        EnvironmentInitEvent(
            session_id=session_id,
            status="description_complete",
            message="Historical RATS run loaded",
            description_content=(
                f"Run directory: `{rel_run}`\n\n"
                f"Iterations: `{len(iteration_paths)}`\n\n"
                f"Report: `{rel_run}/report.md`"
            ),
        )
    )

    successful = 0
    failed = 0
    for iter_path in iteration_paths:
        data = _read_json_file(iter_path, {}) or {}
        iteration = int(data.get("iteration") or int(iter_path.stem.rsplit("_", 1)[-1]))
        if data.get("success"):
            successful += 1
        else:
            failed += 1

        add(
            EnvironmentInitEvent(
                session_id=session_id,
                status="description_complete",
                message=f"Iteration {iteration}: task selected",
                description_content=_format_task_details(data),
            )
        )
        if data.get("plan"):
            add(
                ModelResponseEvent(
                    session_id=session_id,
                    content=_format_plan(data),
                    reasoning=None,
                    code_blocks=[],
                    decision=DecisionType.INITIAL,
                )
            )

        for attempt in _attempt_indices(data):
            code = data.get(f"code_attempt_{attempt}")
            if not isinstance(code, str) or not code.strip():
                continue
            display_code = _truncate_text(code, max_chars=80_000)
            execution_raw = data.get(f"execution_attempt_{attempt}") or {}
            execution = execution_raw if isinstance(execution_raw, dict) else {}
            execution_legacy_text = "" if isinstance(execution_raw, dict) else str(execution_raw)
            verification_raw = data.get(f"verification_attempt_{attempt}") or {}
            verification = verification_raw if isinstance(verification_raw, dict) else {}
            diagnosis_raw = data.get(f"diagnosis_attempt_{attempt}") or {}
            diagnosis = diagnosis_raw if isinstance(diagnosis_raw, dict) else {}
            quality_raw = data.get(f"quality_attempt_{attempt}") or {}
            quality = quality_raw if isinstance(quality_raw, dict) else {}
            self_check_raw = data.get(f"policy_self_check_attempt_{attempt}") or {}
            self_check = self_check_raw if isinstance(self_check_raw, dict) else {}
            result_success = bool(
                verification.get("success")
                if isinstance(verification, dict) and "success" in verification
                else execution.get("success")
            )

            add(
                ModelResponseEvent(
                    session_id=session_id,
                    content=f"## Iteration {iteration} Attempt {attempt + 1}\n\n```python\n{display_code}\n```",
                    reasoning=None,
                    code_blocks=[display_code],
                    decision=DecisionType.INITIAL if attempt == 0 else DecisionType.REGENERATE,
                )
            )
            add(
                CodeExecutionStartEvent(
                    session_id=session_id,
                    block_index=block_index,
                    code=display_code,
                )
            )

            step_index = 0
            thumb = _attempt_thumbnail_base64(run_dir, iteration, attempt)
            attempt_artifacts = (
                execution.get("attempt_artifacts") if isinstance(execution, dict) else {}
            ) or {}
            video_candidates = sorted(run_dir.glob(f"iter{iteration:03d}_attempt{attempt}*.mp4"))
            video_candidates.extend(
                sorted(run_dir.glob(f"iter{iteration:03d}_attempt{attempt}_*/combined.mp4"))
            )
            video_line = ""
            rel_video = _first_history_artifact_relpath(
                run_dir,
                attempt_artifacts.get("combined_video_path"),
                attempt_artifacts.get("video_path"),
            )
            if rel_video is None and video_candidates:
                rel_video = str(video_candidates[0].relative_to(run_dir))
            if rel_video:
                video_line = f"\n\nVideo: `{rel_video}`"
            viser_artifacts = (
                execution.get("viser_recording")
                if isinstance(execution, dict)
                else None
            ) or attempt_artifacts.get("viser_recording") or {}
            rel_viser = _first_history_artifact_relpath(
                run_dir,
                viser_artifacts.get("video_path") if isinstance(viser_artifacts, dict) else None,
            )
            if rel_viser:
                video_line += f"\n\nViser playback: `{rel_viser}`"
            add(
                ExecutionStepEvent(
                    session_id=session_id,
                    block_index=block_index,
                    step_index=step_index,
                    tool_name="Recorded Execution",
                    text=(
                        f"Success: `{execution.get('success')}`\n\n"
                        f"Reward: `{execution.get('reward')}`\n\n"
                        f"Task completed: `{execution.get('task_completed')}`"
                        f"{video_line}\n\n"
                        f"{_json_block(execution.get('user_result') if execution else execution_legacy_text, max_chars=12_000)}"
                    ),
                    images=[thumb] if thumb else [],
                    highlight=not result_success,
                )
            )
            step_index += 1

            if quality:
                add(
                    ExecutionStepEvent(
                        session_id=session_id,
                        block_index=block_index,
                        step_index=step_index,
                        tool_name="Policy Quality",
                        text=_json_block(quality, max_chars=12_000),
                        images=[],
                        highlight=not bool(quality.get("approved", True)),
                    )
                )
                step_index += 1
            if self_check:
                add(
                    ExecutionStepEvent(
                        session_id=session_id,
                        block_index=block_index,
                        step_index=step_index,
                        tool_name="Runtime Self Check",
                        text=_json_block(self_check, max_chars=12_000),
                        images=[],
                        highlight=not bool(self_check.get("passed", True)),
                    )
                )
                step_index += 1
            if verification:
                add(
                    ExecutionStepEvent(
                        session_id=session_id,
                        block_index=block_index,
                        step_index=step_index,
                        tool_name="Verification",
                        text=_json_block(verification, max_chars=24_000),
                        images=[],
                        highlight=not bool(verification.get("success")),
                    )
                )
                step_index += 1
            if diagnosis:
                diagnosis_images, diagnosis_image_text = _history_diagnosis_images(
                    run_dir,
                    diagnosis,
                )
                diagnosis_text = _json_block(
                    _strip_inline_image_data(diagnosis),
                    max_chars=24_000,
                )
                if diagnosis_image_text:
                    diagnosis_text = f"{diagnosis_text}\n\n{diagnosis_image_text}"
                add(
                    ExecutionStepEvent(
                        session_id=session_id,
                        block_index=block_index,
                        step_index=step_index,
                        tool_name="Diagnosis",
                        text=diagnosis_text,
                        images=diagnosis_images,
                        highlight=True,
                    )
                )
                add(
                    ImageAnalysisEvent(
                        session_id=session_id,
                        analysis_type="state_comparison",
                        content=diagnosis_text,
                        model_used=None,
                    )
                )

            add(
                CodeExecutionResultEvent(
                    session_id=session_id,
                    block_index=block_index,
                    success=result_success,
                    stdout=str(execution.get("stdout") or ""),
                    stderr=str(execution.get("stderr") or execution.get("stderr_snippet") or ""),
                    reward=float(execution.get("reward") or 0.0),
                    task_completed=execution.get("task_completed"),
                )
            )
            block_index += 1

    summary = {
        "total_iterations": len(iteration_paths),
        "successful_iterations": successful,
        "failed_iterations": failed,
        "num_code_blocks": block_index,
    }
    add(
        TrialCompleteEvent(
            session_id=session_id,
            success=failed == 0,
            total_reward=float(successful),
            task_completed=None,
            num_regenerations=0,
            num_code_blocks=block_index,
            summary=(
                f"Historical run: {rel_run}\n"
                f"Iterations: {len(iteration_paths)}\n"
                f"Successful: {successful}\n"
                f"Failed: {failed}\n"
                f"Code attempts: {block_index}"
            ),
        )
    )
    return events, summary


def _replay_debug_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RATS Replay Debug</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; height: 100vh; overflow: hidden; background: #0d1117; color: #e6edf3; }
    header { height: 52px; display: flex; align-items: center; justify-content: space-between; padding: 0 16px; border-bottom: 1px solid #30363d; background: #161b22; }
    h1 { margin: 0; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; }
    main { height: calc(100vh - 53px); display: grid; grid-template-columns: minmax(460px, 58%) 1fr; }
    #left { display: grid; grid-template-rows: auto 1fr; min-width: 0; border-right: 1px solid #30363d; }
    #controls { padding: 12px 14px; border-bottom: 1px solid #30363d; background: #111820; display: grid; gap: 10px; }
    #buttons { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    button { background: #238636; color: white; border: 1px solid #2ea043; border-radius: 6px; padding: 7px 11px; cursor: pointer; font-size: 12px; font-weight: 650; }
    button.secondary { background: #21262d; border-color: #30363d; color: #e6edf3; }
    button.warn { background: #da3633; border-color: #f85149; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    input { width: 58px; background: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 5px; padding: 6px 8px; }
    #status { color: #8b949e; font-size: 12px; line-height: 1.45; white-space: pre-wrap; }
    #feed { overflow: auto; padding: 14px; }
    #viser { width: 100%; height: 100%; border: 0; background: #010409; }
    .event { border: 1px solid #30363d; border-left: 3px solid #3fb950; border-radius: 6px; background: #161b22; padding: 10px 12px; margin-bottom: 10px; }
    .event.warn { border-left-color: #f85149; }
    .event.run { border-left-color: #d29922; }
    .meta { display: flex; gap: 8px; align-items: center; color: #8b949e; font-size: 11px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .04em; }
    .title { color: #e6edf3; font-weight: 650; margin-bottom: 6px; }
    pre { white-space: pre-wrap; overflow: auto; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 12px; line-height: 1.45; max-height: 520px; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    img { max-width: 240px; max-height: 180px; object-fit: contain; border: 1px solid #30363d; border-radius: 4px; margin: 6px 8px 0 0; }
    .grid { display: flex; flex-wrap: wrap; }
    .small { color: #8b949e; font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>RATS Replay Debug</h1>
    <span id="wsStatus" class="small">connecting</span>
  </header>
  <main>
    <section id="left">
      <div id="controls">
        <div id="buttons">
          <button onclick="sendCommand('next')">Run Next</button>
          <button class="secondary" onclick="sendRunTo()">Run To Step</button>
          <input id="targetStep" type="number" min="0" value="1" title="0 means setup; 1 means first policy step" />
          <button class="secondary" onclick="sendCommand('run_all')">Run All</button>
          <button class="warn" onclick="sendCommand('stop')">Stop</button>
          <button class="secondary" onclick="reloadViser()">Reload View</button>
          <button class="secondary" onclick="location.reload()">Reconnect</button>
        </div>
        <div id="status">loading replay status...</div>
      </div>
      <div id="feed"></div>
    </section>
    <iframe id="viser" src="/viser-proxy/"></iframe>
  </main>
  <script>
    const feed = document.getElementById('feed');
    const statusEl = document.getElementById('status');
    const wsStatus = document.getElementById('wsStatus');
    const viserFrame = document.getElementById('viser');
    const blocks = new Map();
    function ts(t) { try { return new Date(t).toLocaleTimeString(); } catch { return ''; } }
    function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    function imgSrc(s) { return String(s || '').startsWith('data:') ? s : 'data:image/jpeg;base64,' + s; }
    function reloadViser() {
      viserFrame.src = `/viser-proxy/?_=${Date.now()}`;
    }
    function add(kind, title, body, opts={}) {
      const el = document.createElement('div');
      el.className = 'event ' + (opts.warn ? 'warn' : opts.run ? 'run' : '');
      el.innerHTML = `<div class="meta"><span>${esc(kind)}</span><span>${esc(opts.time || '')}</span></div><div class="title">${esc(title)}</div>${body || ''}`;
      feed.appendChild(el);
      feed.scrollTop = feed.scrollHeight;
      return el;
    }
    function renderEvent(ev) {
      if (ev.type === 'state_update') { wsStatus.textContent = ev.state; return; }
      if (ev.type === 'environment_init') {
        add('status', ev.message || ev.status, ev.description_content ? `<pre>${esc(ev.description_content)}</pre>` : '', {time: ts(ev.timestamp), run: ev.status === 'starting'});
      } else if (ev.type === 'model_response' || ev.type === 'model_streaming_end') {
        let code = (ev.code_blocks || []).map(c => `<pre><code>${esc(c)}</code></pre>`).join('');
        add('code', ev.decision || 'response', `<pre>${esc(ev.content || '')}</pre>${code}`, {time: ts(ev.timestamp)});
      } else if (ev.type === 'code_execution_start') {
        const el = add('run', `Block ${(ev.block_index || 0) + 1} running`, `<pre><code>${esc(ev.code || '')}</code></pre><div class="steps"></div>`, {time: ts(ev.timestamp), run: true});
        blocks.set(ev.block_index, el);
      } else if (ev.type === 'execution_step') {
        const parent = blocks.get(ev.block_index) || add('run', `Block ${(ev.block_index || 0) + 1}`, '<div class="steps"></div>', {time: ts(ev.timestamp)});
        const steps = parent.querySelector('.steps');
        let step = steps.querySelector(`[data-step="${ev.step_index}"]`);
        const imgs = (ev.images || []).map(i => `<img src="${imgSrc(i)}" />`).join('');
        const html = `<div class="small"><b>${esc(ev.tool_name)}</b> step ${(ev.step_index || 0) + 1}</div><pre>${esc(ev.text || '')}</pre><div class="grid">${imgs}</div>`;
        if (!step) { step = document.createElement('div'); step.dataset.step = ev.step_index; steps.appendChild(step); }
        step.innerHTML = html;
      } else if (ev.type === 'code_execution_result') {
        const parent = blocks.get(ev.block_index);
        if (parent) parent.classList.toggle('warn', !ev.success);
        add('result', ev.success ? 'Execution completed' : 'Execution failed', `<div class="small">reward=${esc(ev.reward)} task_completed=${esc(ev.task_completed)}</div>${ev.stdout ? `<pre>${esc(ev.stdout)}</pre>` : ''}${ev.stderr ? `<pre>${esc(ev.stderr)}</pre>` : ''}`, {time: ts(ev.timestamp), warn: !ev.success});
      } else if (ev.type === 'visual_feedback') {
        add('image', ev.description || 'Visual feedback', `<img src="${imgSrc(ev.image_base64)}" />`, {time: ts(ev.timestamp)});
      } else if (ev.type === 'trial_complete') {
        add('complete', ev.success ? 'Replay complete' : 'Replay finished', `<pre>${esc(ev.summary || '')}</pre>`, {time: ts(ev.timestamp), warn: !ev.success});
      } else if (ev.type === 'error') {
        add('error', ev.message || 'error', '', {time: ts(ev.timestamp), warn: true});
      }
    }
    async function sendCommand(command, payload={}) {
      const previousStatus = statusEl.textContent;
      statusEl.textContent = `sending command: ${command}...\n\n${previousStatus}`;
      try {
        const resp = await fetch('/api/replay/command', {
          method: 'POST',
          headers: {'content-type': 'application/json'},
          body: JSON.stringify({command, ...payload}),
        });
        const text = await resp.text();
        let data = {};
        try { data = text ? JSON.parse(text) : {}; } catch { data = {raw: text}; }
        if (!resp.ok || data.ok === false) {
          throw new Error(data.error || data.detail || text || `HTTP ${resp.status}`);
        }
        await refreshStatus();
      } catch (e) {
        const message = e && e.message ? e.message : String(e);
        statusEl.textContent = `command failed: ${message}\n\n${previousStatus}`;
        add('error', 'Replay command failed', `<pre>${esc(message)}</pre>`, {warn: true});
      }
    }
    function sendRunTo() {
      const target = Number(document.getElementById('targetStep').value || 0);
      sendCommand('run_to', {target});
    }
    async function refreshStatus() {
      const s = await fetch('/api/replay/status').then(r => r.json()).catch(e => ({available:false, error:String(e)}));
      if (!s.available) { statusEl.textContent = 'No replay controller attached.'; return; }
      const labels = (s.blocks || []).map((b, i) => `${i + 1}. ${b}`).join('\n');
      const current =
        s.phase === 'waiting'
          ? `waiting for command at stage ${s.current_stage} ${s.current_label || ''}`
          : s.phase === 'running'
            ? `running stage ${s.current_stage} ${s.current_label || ''}`
            : `${s.phase} stage ${s.current_stage} ${s.current_label || ''}`;
      statusEl.textContent =
        `${current}\n` +
        `completed stage: ${s.completed_stage}\n` +
        `stopped: ${s.stopped}  done: ${s.done}\n\n` +
        `Stages: 0. setup\n${labels}`;
    }
    async function connect() {
      const active = await fetch('/api/active-session').then(r => r.json()).catch(() => ({}));
      if (!active.session_id) {
        wsStatus.textContent = 'waiting for replay session';
        setTimeout(connect, 1000);
        return;
      }
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${proto}//${location.host}/ws/${active.session_id}`);
      ws.onopen = () => wsStatus.textContent = 'connected';
      ws.onclose = () => { wsStatus.textContent = 'disconnected'; setTimeout(connect, 1000); };
      ws.onerror = () => wsStatus.textContent = 'websocket error';
      ws.onmessage = msg => { try { renderEvent(JSON.parse(msg.data)); } catch (e) { console.error(e); } };
    }
    connect();
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>"""



def _port_from_viser_server(viser_server: Any | None) -> int | None:
    websock = getattr(viser_server, "_websock_server", None)
    port = getattr(websock, "_port", None)
    return port if isinstance(port, int) else None


def _session_viser_port(session: Any | None) -> int | None:
    """Best-effort extraction of the Viser port attached to a session env."""
    for attr in ("rats_viser_server", "viser_server"):
        port = _port_from_viser_server(getattr(session, attr, None) if session is not None else None)
        if port is not None:
            return port
    env = getattr(session, "env", None) if session is not None else None
    candidates = [
        env,
        getattr(env, "low_level_env", None),
        getattr(getattr(env, "low_level_env", None), "low_level_env", None),
    ]
    for obj in candidates:
        port = _port_from_viser_server(getattr(obj, "viser_server", None))
        if port is not None:
            return port
    return None


def _find_viser_port(preferred_port: int | None = None) -> int | None:
    """Probe candidate ports to find a running Viser server (cached)."""
    global _viser_port_cache
    if preferred_port is not None:
        try:
            urlopen(f"http://localhost:{preferred_port}/", timeout=1)
            _viser_port_cache = preferred_port
            return preferred_port
        except Exception:
            pass
    # Try cached port first
    if _viser_port_cache is not None:
        try:
            urlopen(f"http://localhost:{_viser_port_cache}/", timeout=1)
            return _viser_port_cache
        except Exception:
            _viser_port_cache = None
    for port in _VISER_PORTS:
        try:
            urlopen(f"http://localhost:{port}/", timeout=1)
            _viser_port_cache = port
            return port
        except Exception:
            continue
    return None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="CaP-X Interactive Web UI",
        description="Real-time interactive interface for CaP-X robot code execution",
        version="1.0.0",
    )

    # CORS for frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========================================================================
    # REST API Endpoints
    # ========================================================================

    @app.get("/api/default-config")
    async def get_default_config():
        """Return the default config path passed via launch.py (if any).

        When a config_path was provided on the CLI, ``auto_start`` is set to
        ``True`` so the frontend can kick off the trial immediately after load.
        """
        default_path = getattr(app.state, "default_config_path", None)
        return {
            "config_path": default_path,
            "auto_start": default_path is not None,
        }

    @app.get("/api/configs", response_model=ConfigListResponse)
    async def list_configs():
        """List available YAML config files from all environment directories."""
        configs_root = Path("env_configs")
        if not configs_root.exists():
            return ConfigListResponse(configs=[])

        configs = []
        for yaml_file in configs_root.rglob("*.yaml"):
            # Skip hillclimb subdirectories (internal experiment configs)
            if "hillclimb" in yaml_file.parts:
                continue
            configs.append(str(yaml_file.relative_to(".")))

        configs.sort()
        return ConfigListResponse(configs=configs)

    @app.post("/api/load-config", response_model=LoadConfigResponse)
    async def load_config(request: LoadConfigRequest):
        """Load and validate a YAML config file."""
        config_path = request.config_path

        if not Path(config_path).exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

        try:
            # Create a minimal args object for _load_config
            @dataclass
            class MinimalArgs:
                config_path: str
                server_url: str = "http://127.0.0.1:8110/chat/completions"
                model: str = "google/gemini-3.1-pro-preview"
                temperature: float = 1.0
                max_tokens: int = 20480
                reasoning_effort: str = "medium"
                api_key: str | None = None
                use_visual_feedback: bool | None = None
                use_img_differencing: bool | None = None
                visual_differencing_model: str | None = "google/gemini-3.1-pro-preview"
                visual_differencing_model_server_url: str | None = "http://127.0.0.1:8110/chat/completions"
                visual_differencing_model_api_key: str | None = None
                total_trials: int | None = None
                num_workers: int | None = None
                record_video: bool | None = None
                output_dir: str | None = None
                debug: bool = False
                use_oracle_code: bool | None = None
                use_parallel_ensemble: bool | None = None
                use_video_differencing: bool | None = None
                use_wrist_camera: bool | None = None
                use_multimodel: bool | None = None
                web_ui: bool | None = None
                web_ui_port: int | None = None

            args = MinimalArgs(config_path=config_path)
            env_factory, config, _ = await asyncio.to_thread(_load_config, args)

            # Extract task prompt (no length limit - UI handles display)
            task_prompt = env_factory.get("cfg", {}).get("prompt", "")

            return LoadConfigResponse(
                status="loaded",
                config_summary={
                    "record_video": config.get("record_video"),
                    "output_dir": config.get("output_dir"),
                    "use_visual_feedback": config.get("use_visual_feedback"),
                    "use_img_differencing": config.get("use_img_differencing"),
                    "total_trials": config.get("total_trials"),
                },
                task_prompt=task_prompt,
            )
        except Exception as e:
            logger.exception(f"Failed to load config: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/start-trial", response_model=StartTrialResponse)
    async def start_trial(request: StartTrialRequest):
        """Start a new trial execution."""
        config_path = request.config_path

        if not Path(config_path).exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

        manager = get_session_manager()

        # Create new session
        session = await manager.create_session()

        try:
            # Build args for config loading
            @dataclass
            class LoadArgs:
                config_path: str
                server_url: str
                model: str
                temperature: float
                max_tokens: int
                reasoning_effort: str = "medium"
                api_key: str | None = None
                use_visual_feedback: bool | None = None
                use_img_differencing: bool | None = None
                visual_differencing_model: str | None = None
                visual_differencing_model_server_url: str | None = None
                visual_differencing_model_api_key: str | None = None
                total_trials: int | None = 1
                num_workers: int | None = 1
                record_video: bool | None = None
                output_dir: str | None = None
                debug: bool = False
                use_oracle_code: bool | None = None
                use_parallel_ensemble: bool | None = None
                use_video_differencing: bool | None = None
                use_wrist_camera: bool | None = None
                use_multimodel: bool | None = None
                web_ui: bool | None = None
                web_ui_port: int | None = None

            load_args = LoadArgs(
                config_path=config_path,
                server_url=request.server_url,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_visual_feedback=request.use_visual_feedback,
                use_img_differencing=request.use_img_differencing,
                visual_differencing_model=request.visual_differencing_model,
                visual_differencing_model_server_url=request.visual_differencing_model_server_url,
            )
            env_factory, config, api_servers = await asyncio.to_thread(_load_config, load_args)
            session.api_server_procs = await asyncio.to_thread(_start_api_servers, api_servers)

            # Process output_dir like launch.py does - add model name to path and create directory
            if config.get("output_dir"):
                from datetime import datetime
                # Add model name and timestamp to output path
                base_dir = config["output_dir"]
                model_slug = request.model.replace("/", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_out_dir = f"{base_dir}/{model_slug}/{timestamp}"
                Path(new_out_dir).mkdir(parents=True, exist_ok=True)
                config["output_dir"] = new_out_dir
                logger.info(f"Output directory set to: {new_out_dir}")

            # Store in session
            session.config_path = config_path
            session.config = config
            session.env_factory = env_factory
            session.state = SessionState.LOADING_CONFIG

            # Build launch args for trial runner
            trial_args = LaunchArgsCompat(
                model=request.model,
                server_url=request.server_url,
                api_key=None,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                reasoning_effort="medium",
                debug=False,
                visual_differencing_model=request.visual_differencing_model,
                visual_differencing_model_server_url=request.visual_differencing_model_server_url,
                visual_differencing_model_api_key=None,
            )

            # Set the initial settings on session (can be changed during trial)
            session.await_user_input_each_turn = request.await_user_input_each_turn
            session.execution_timeout = request.execution_timeout

            # Start trial in background task
            session.task = asyncio.create_task(
                run_trial_async(
                    session=session,
                    args=trial_args,
                )
            )

            return StartTrialResponse(
                session_id=session.session_id,
                status="started",
            )

        except Exception as e:
            logger.exception(f"Failed to start trial: {e}")
            await manager.remove_session(session.session_id)
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/stop", response_model=StopTrialResponse)
    async def stop_trial(request: StopTrialRequest):
        """Stop a running trial."""
        manager = get_session_manager()
        success = await manager.stop_session(request.session_id)

        if not success:
            raise HTTPException(status_code=404, detail="Session not found or not running")

        return StopTrialResponse(status="stopped")

    @app.post("/api/inject-prompt")
    async def inject_prompt(session_id: str, text: str):
        """Inject user prompt text into a running session."""
        manager = get_session_manager()
        success = await manager.inject_prompt(session_id, text)

        if not success:
            raise HTTPException(status_code=400, detail="Cannot inject: session not awaiting input")

        return {"status": "injected"}

    @app.get("/api/session/{session_id}", response_model=SessionStatusResponse)
    async def get_session_status(session_id: str):
        """Get the status of a session."""
        manager = get_session_manager()
        session = await manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionStatusResponse(
            session_id=session.session_id,
            state=session.state,
            current_block_index=session.current_block_index,
            total_code_blocks=session.total_code_blocks,
            num_regenerations=session.num_regenerations,
        )

    @app.get("/api/sessions")
    async def list_sessions():
        """List all active sessions."""
        manager = get_session_manager()
        return {"sessions": manager.list_sessions()}

    @app.get("/api/active-session")
    async def get_active_session():
        """Get the currently active session (if any).

        Useful for reconnecting after page refresh.
        """
        manager = get_session_manager()
        session = manager.get_active_session()
        if session:
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "config_path": session.config_path,
                "history_run": session.config.get("history_run"),
            }
        return {"session_id": None}

    @app.get("/api/history-runs")
    async def list_history_runs(limit: int = 100):
        """List saved RATS output directories that can be loaded into the Web UI."""
        outputs_root = _project_root() / "outputs"
        if not outputs_root.exists():
            return {"runs": []}

        max_depth = 4
        seen: set[Path] = set()
        candidate_dirs: list[Path] = []
        stack: list[tuple[Path, int]] = [(outputs_root, 0)]
        while stack:
            current, depth = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if any(current.glob("iteration_*.json")):
                candidate_dirs.append(current)
                # A directory with iteration artifacts is itself a run root;
                # do not crawl into its bulky report/video/artifact children.
                continue
            if depth >= max_depth:
                continue
            try:
                children = [child for child in current.iterdir() if child.is_dir()]
            except OSError:
                continue
            stack.extend((child, depth + 1) for child in children)

        runs: list[dict[str, Any]] = []
        for run_dir in candidate_dirs:
            if any(item.get("path") == str(run_dir.relative_to(_project_root())) for item in runs):
                continue
            try:
                runs.append(_history_run_summary(run_dir))
            except Exception:
                logger.debug("Skipping invalid history run directory: %s", run_dir, exc_info=True)

        runs.sort(key=lambda item: float(item.get("mtime") or 0), reverse=True)
        return {"runs": runs[: max(1, min(int(limit), 500))]}

    @app.post("/api/history-runs/load")
    async def load_history_run(request: Request):
        """Create a read-only session from saved RATS iteration artifacts."""
        payload = await request.json()
        run_dir = _resolve_history_run_dir(str(payload.get("run_dir") or ""))
        events, summary = _build_history_events(run_dir)

        manager = get_session_manager()
        session = await manager.create_session()
        rel_run = str(run_dir.relative_to(_project_root())) if _project_root() in run_dir.parents else str(run_dir)
        session.config_path = None
        session.config = {
            "rats_history": True,
            "history_run": {
                "path": rel_run,
                "name": run_dir.name,
                "report_url": f"/api/history-runs/report?run_dir={rel_run}",
                "artifact_url": f"/api/history-runs/artifact?run_dir={rel_run}&path=",
                "summary": summary,
            },
        }
        session.state = SessionState.COMPLETE
        session.event_history = events
        session.total_code_blocks = int(summary.get("num_code_blocks") or 0)
        session.current_block_index = max(0, session.total_code_blocks - 1)
        return {
            "session_id": session.session_id,
            "status": "loaded",
            "run": session.config["history_run"],
        }

    @app.get("/api/history-runs/report")
    async def get_history_run_report(run_dir: str):
        """Serve report.md for a historical run, rendering it if needed."""
        resolved = _resolve_history_run_dir(run_dir)
        report = resolved / "report.md"
        if not report.exists():
            try:
                from scripts.render_run_md import render

                report.write_text(await asyncio.to_thread(render, resolved))
            except Exception as exc:
                raise HTTPException(status_code=404, detail=f"report.md not found and render failed: {exc}") from exc
        return Response(report.read_text(), media_type="text/markdown")

    @app.get("/api/history-runs/artifact")
    async def get_history_run_artifact(run_dir: str, path: str):
        """Serve a file under a historical run directory for report assets."""
        resolved = _resolve_history_run_dir(run_dir)
        artifact = _safe_history_artifact(resolved, path)
        return FileResponse(artifact)

    @app.get("/api/replay/status")
    async def replay_status():
        """Return status for the optional RATS replay step controller."""
        controller = getattr(app.state, "rats_replay_controller", None)
        if controller is None:
            return {"available": False}
        status_fn = getattr(controller, "status", None)
        if not callable(status_fn):
            return {"available": False}
        status = await asyncio.to_thread(status_fn)
        if isinstance(status, dict):
            status.setdefault("available", True)
            return status
        return {"available": True, "status": status}

    @app.post("/api/replay/command")
    async def replay_command(request: Request):
        """Send a control command to the optional RATS replay step controller."""
        controller = getattr(app.state, "rats_replay_controller", None)
        if controller is None:
            raise HTTPException(status_code=404, detail="No replay controller attached")
        payload = await request.json()
        command = str(payload.get("command", "")).strip()
        command_fn = getattr(controller, "command", None)
        if not callable(command_fn):
            raise HTTPException(status_code=400, detail="Replay controller does not accept commands")
        result = await asyncio.to_thread(command_fn, command, payload)
        return result if isinstance(result, dict) else {"status": result}

    @app.get("/replay")
    async def replay_debug_page():
        """Small RATS replay UI: code/output/control on the left, Viser on the right."""
        return HTMLResponse(_replay_debug_html())

    # ========================================================================
    # WebSocket Endpoint
    # ========================================================================

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket connection for real-time trial updates."""
        manager = get_session_manager()
        session = await manager.get_session(session_id)

        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept()
        session.websockets.append(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

        # Replay event history so reconnecting clients see all past messages
        if session.event_history:
            logger.info(f"Replaying {len(session.event_history)} events for session {session_id}")
            for event_json in session.event_history:
                try:
                    await websocket.send_text(event_json)
                except Exception:
                    break

        # Send current state
        await websocket.send_text(
            StateUpdateEvent(
                session_id=session_id,
                state=session.state,
            ).model_dump_json()
        )

        try:
            while True:
                # Receive commands from client
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    msg_type = message.get("type")

                    if msg_type == "stop":
                        logger.info(f"Stop command received for session {session_id}")
                        await manager.stop_session(session_id)

                    elif msg_type == "inject_prompt":
                        text = message.get("text", "")
                        if text:
                            await manager.inject_prompt(session_id, text)
                            logger.info(f"Injected prompt: {text[:50]}...")

                    elif msg_type == "resume":
                        # Put empty string to unblock the queue wait
                        if session.state == SessionState.AWAITING_USER_INPUT:
                            await session.user_injection_queue.put("")

                    elif msg_type == "update_settings":
                        # Update session settings dynamically during a trial
                        if "await_user_input_each_turn" in message:
                            session.await_user_input_each_turn = message["await_user_input_each_turn"]
                            logger.info(f"Updated await_user_input_each_turn to {session.await_user_input_each_turn}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {data}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        finally:
            if websocket in session.websockets:
                session.websockets.remove(websocket)
            # Check if session should be cleaned up
            await manager.on_websocket_disconnect(session_id)

    # ========================================================================
    # Viser reverse proxy  (avoids port-forwarding issues)
    # ========================================================================

    async def _proxy_viser_http(path: str = "", query: str = "") -> Response:
        """Forward an HTTP request to the local Viser server."""
        manager = get_session_manager()
        preferred_port = _session_viser_port(manager.get_active_session())
        port = await asyncio.to_thread(_find_viser_port, preferred_port)
        if port is None:
            return Response(
                content="Viser server not available — is the trial running?",
                status_code=503,
            )
        url = f"http://localhost:{port}/{path}"
        if query:
            url += f"?{query}"
        try:
            req = UrlRequest(url)
            resp = await asyncio.to_thread(urlopen, req, None, 5)
            content = await asyncio.to_thread(resp.read)
            content_type = resp.headers.get(
                "Content-Type", "application/octet-stream"
            )
            return Response(content=content, media_type=content_type)
        except Exception as exc:
            return Response(content=f"Proxy error: {exc}", status_code=502)

    @app.api_route("/viser-proxy", methods=["GET", "HEAD"])
    async def proxy_viser_root(request: Request):
        """Proxy the Viser root page (no trailing slash)."""
        return await _proxy_viser_http("", str(request.url.query))

    @app.api_route("/viser-proxy/{path:path}", methods=["GET", "HEAD"])
    async def proxy_viser_path(request: Request, path: str):
        """Proxy Viser sub-paths (assets, hdri, etc.)."""
        return await _proxy_viser_http(path, str(request.url.query))

    @app.websocket("/viser-proxy")
    async def proxy_viser_ws(websocket: WebSocket):
        """Reverse-proxy WebSocket connections to the local Viser server.

        The Viser client constructs the WS URL from ``window.location.href``
        (replacing http→ws and stripping the trailing slash), so the iframe at
        ``/viser-proxy/`` connects to ``ws://host/viser-proxy``.
        """
        manager = get_session_manager()
        preferred_port = _session_viser_port(manager.get_active_session())
        port = await asyncio.to_thread(_find_viser_port, preferred_port)
        if port is None:
            await websocket.close(code=1013, reason="Viser not running")
            return

        # Forward subprotocols (Viser expects "viser-v<VERSION>")
        proto_header = websocket.headers.get("sec-websocket-protocol", "")
        client_protocols = [
            s.strip() for s in proto_header.split(",") if s.strip()
        ]

        import websockets
        from websockets.asyncio.client import connect as ws_connect

        try:
            viser_ws = await ws_connect(
                f"ws://localhost:{port}/",
                subprotocols=(
                    [websockets.Subprotocol(p) for p in client_protocols]
                    if client_protocols
                    else None
                ),
                compression=None,
                max_size=2**24,  # 16 MiB — Viser can send large scene blobs
            )
        except Exception as exc:
            logger.warning(f"Could not connect to Viser WS: {exc}")
            await websocket.close(code=1013, reason=str(exc))
            return

        # Accept the browser connection with Viser's selected subprotocol
        await websocket.accept(subprotocol=viser_ws.subprotocol)

        async def _client_to_viser() -> None:
            try:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in msg and msg["bytes"]:
                        await viser_ws.send(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        await viser_ws.send(msg["text"])
            except (WebSocketDisconnect, Exception):
                pass

        async def _viser_to_client() -> None:
            try:
                async for data in viser_ws:
                    if isinstance(data, bytes):
                        await websocket.send_bytes(data)
                    else:
                        await websocket.send_text(data)
            except Exception:
                pass

        try:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_client_to_viser()),
                    asyncio.create_task(_viser_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
        except Exception as exc:
            logger.warning(f"Viser WS proxy error: {exc}")
        finally:
            await viser_ws.close()
            try:
                await websocket.close()
            except Exception:
                pass

    # ========================================================================
    # Static file serving for frontend (production)
    # ========================================================================

    # Check if built frontend exists
    frontend_dist = Path(__file__).parent.parent.parent / "web-ui" / "dist"
    if frontend_dist.exists() and os.getenv("RATS_WEB_UI_FORCE_FALLBACK", "0") != "1":
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @app.get("/")
        async def serve_frontend():
            return FileResponse(frontend_dist / "index.html")

        @app.get("/{path:path}")
        async def serve_frontend_routes(path: str):
            # Try to serve static file, otherwise serve index.html for SPA routing
            file_path = frontend_dist / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dist / "index.html")

    return app


# ============================================================================
# CLI Entry Point
# ============================================================================


@dataclass
class ServerArgs:
    """Command-line arguments for the web server."""

    host: str = "0.0.0.0"
    """Host to bind the server to."""

    port: int = 8200
    """Port to run the server on."""

    reload: bool = False
    """Enable auto-reload for development."""


def main(args: ServerArgs | None = None) -> None:
    """Run the web server."""
    if args is None:
        args = tyro.cli(ServerArgs)

    app = create_app()

    logger.info(f"Starting CaP-X Interactive Web UI on http://{args.host}:{args.port}")
    logger.info("Frontend dev server should be running on http://localhost:5173")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
