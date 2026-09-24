"""RATS live debugger backed by the existing CaP-X Web UI protocol.

The CaP-X web UI already knows how to render model responses, code blocks,
execution steps with images, visual feedback frames, and a Viser iframe.  This
module lets ``scripts/run_rats.py`` publish the RATS lifelong loop into that
same protocol instead of requiring the CaP-X single-trial runner to own the
experiment.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import shutil
import subprocess
import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from rats.web.models import (
    CodeExecutionResultEvent,
    CodeExecutionStartEvent,
    DecisionType,
    EnvironmentInitEvent,
    ErrorEvent,
    ExecutionStepEvent,
    ImageAnalysisEvent,
    ModelResponseEvent,
    SessionState,
    StateUpdateEvent,
    TrialCompleteEvent,
    ViserReloadEvent,
    VisualFeedbackEvent,
    WSEventBase,
)

logger = logging.getLogger("rats.web_debug")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _frontend_needs_build(webui_dir: Path) -> bool:
    dist = webui_dir / "dist"
    index = dist / "index.html"
    if not index.exists():
        return True
    dist_mtime = index.stat().st_mtime
    for src in (webui_dir / "src").rglob("*"):
        if src.is_file() and src.stat().st_mtime > dist_mtime:
            return True
    pkg = webui_dir / "package.json"
    return pkg.exists() and pkg.stat().st_mtime > dist_mtime


def _ensure_frontend_built() -> None:
    """Best-effort build of ``web-ui/dist`` using the local npm when needed."""
    root = _project_root()
    webui_dir = root / "web-ui"
    if not webui_dir.exists() or not _frontend_needs_build(webui_dir):
        return
    if os.getenv("RATS_WEB_UI_AUTO_BUILD", "0") != "1":
        logger.info(
            "web-ui/dist is missing or stale; using lightweight fallback "
            "(set RATS_WEB_UI_AUTO_BUILD=1 to build the React UI automatically)"
        )
        return

    npm = shutil.which("npm")
    if npm is None:
        logger.warning("npm not found; using the built-in lightweight RATS Web UI fallback")
        return

    logger.info("Building web-ui frontend for RATS debug server")
    node_modules = webui_dir / "node_modules"
    needs_install = (
        not node_modules.exists()
        or not (node_modules / ".bin" / "tsc").exists()
        or not (node_modules / ".bin" / "vite").exists()
    )
    if needs_install:
        try:
            subprocess.check_call([npm, "install", "--no-audit", "--no-fund"], cwd=webui_dir, timeout=180)
        except Exception as exc:
            logger.warning("npm install failed or timed out; using RATS Web UI fallback: %s", exc)
            return
    try:
        subprocess.check_call([npm, "run", "build"], cwd=webui_dir, timeout=120)
    except Exception as exc:
        logger.warning("web-ui build failed; using RATS Web UI fallback: %s", exc)


def _fallback_html() -> str:
    """A dependency-free UI used when ``web-ui/dist`` is not built."""
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RATS Live Debug</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #0d1117; color: #e6edf3; height: 100vh; overflow: hidden; }
    header { height: 52px; display: flex; align-items: center; justify-content: space-between; padding: 0 18px; border-bottom: 1px solid #30363d; background: #161b22; }
    h1 { font-size: 14px; margin: 0; letter-spacing: .08em; text-transform: uppercase; }
    #status { font-size: 12px; color: #8b949e; }
    main { height: calc(100vh - 53px); display: grid; grid-template-columns: minmax(420px, 58%) 1fr; }
    #feed { overflow: auto; padding: 14px; border-right: 1px solid #30363d; }
    #viser { width: 100%; height: 100%; border: 0; background: #010409; }
    .event { border-left: 3px solid #3fb950; background: #161b22; border: 1px solid #30363d; border-left-color: #3fb950; border-radius: 6px; margin-bottom: 10px; padding: 10px 12px; }
    .event.warn { border-left-color: #f85149; }
    .event.run { border-left-color: #d29922; }
    .meta { display: flex; gap: 8px; align-items: center; color: #8b949e; font-size: 11px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .04em; }
    .title { color: #e6edf3; font-weight: 650; }
    pre { white-space: pre-wrap; overflow: auto; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 12px; line-height: 1.45; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    img { max-width: 240px; max-height: 180px; object-fit: contain; border: 1px solid #30363d; border-radius: 4px; margin: 6px 8px 0 0; }
    .grid { display: flex; flex-wrap: wrap; }
    .small { color: #8b949e; font-size: 12px; }
    button { background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 5px; padding: 5px 9px; cursor: pointer; }
    button:hover { background: #30363d; }
  </style>
</head>
<body>
  <header>
    <h1>RATS Live Debug</h1>
    <div>
      <button onclick="location.reload()">Reconnect</button>
      <span id="status">connecting</span>
    </div>
  </header>
  <main>
    <section id="feed"></section>
    <iframe id="viser" src="/viser-proxy/"></iframe>
  </main>
  <script>
    const feed = document.getElementById('feed');
    const statusEl = document.getElementById('status');
    const viserFrame = document.getElementById('viser');
    const blocks = new Map();
    function ts(t) { try { return new Date(t).toLocaleTimeString(); } catch { return ''; } }
    function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    function imgSrc(s) { return String(s || '').startsWith('data:') ? s : 'data:image/jpeg;base64,' + s; }
    function fmtRange(ev) {
      if (typeof ev.start_s !== 'number') return '';
      const f = x => `${Math.floor(x / 60)}:${String((x % 60).toFixed(1)).padStart(4, '0')}`;
      if (typeof ev.end_s === 'number' && ev.end_s !== ev.start_s) return ` · video ${f(ev.start_s)}–${f(ev.end_s)}`;
      return ` · video ${f(ev.start_s)}`;
    }
    function reloadViser() {
      if (viserFrame) viserFrame.src = `/viser-proxy/?_=${Date.now()}`;
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
      if (ev.type === 'viser_reload') {
        statusEl.textContent = ev.port ? `visualization switching to ${ev.port}` : 'visualization reconnecting';
        reloadViser();
        return;
      }
      if (ev.type === 'state_update') { statusEl.textContent = ev.state; return; }
      if (ev.type === 'environment_init') {
        add('status', ev.message || ev.status, ev.description_content ? `<pre>${esc(ev.description_content)}</pre>` : '', {time: ts(ev.timestamp), run: ev.status === 'starting'});
      } else if (ev.type === 'model_response' || ev.type === 'model_streaming_end') {
        let code = (ev.code_blocks || []).map(c => `<pre><code>${esc(c)}</code></pre>`).join('');
        add('model', ev.decision || 'response', `<pre>${esc(ev.content || '')}</pre>${code}`, {time: ts(ev.timestamp)});
      } else if (ev.type === 'code_execution_start') {
        const el = add('code', `Block ${(ev.block_index || 0) + 1} running`, `<pre><code>${esc(ev.code || '')}</code></pre><div class="steps"></div>`, {time: ts(ev.timestamp), run: true});
        blocks.set(ev.block_index, el);
      } else if (ev.type === 'execution_step') {
        const parent = blocks.get(ev.block_index) || add('code', `Block ${(ev.block_index || 0) + 1}`, '<div class="steps"></div>', {time: ts(ev.timestamp)});
        const steps = parent.querySelector('.steps');
        let step = steps.querySelector(`[data-step="${ev.step_index}"]`);
        const imgs = (ev.images || []).map(i => `<img src="${imgSrc(i)}" />`).join('');
        const label = ev.timeline_label ? ` · ${esc(ev.timeline_label)}` : '';
        const html = `<div class="small"><b>${esc(ev.tool_name)}</b> step ${(ev.step_index || 0) + 1}${label}${fmtRange(ev)}</div><pre>${esc(ev.text || '')}</pre><div class="grid">${imgs}</div>`;
        if (!step) {
          step = document.createElement('div');
          step.dataset.step = ev.step_index;
          steps.appendChild(step);
        }
        step.innerHTML = html;
      } else if (ev.type === 'code_execution_result') {
        const parent = blocks.get(ev.block_index);
        if (parent) parent.classList.toggle('warn', !ev.success);
        add('result', ev.success ? 'Execution completed' : 'Execution failed', `<div class="small">reward=${esc(ev.reward)} task_completed=${esc(ev.task_completed)}</div>${ev.stdout ? `<pre>${esc(ev.stdout)}</pre>` : ''}${ev.stderr ? `<pre>${esc(ev.stderr)}</pre>` : ''}`, {time: ts(ev.timestamp), warn: !ev.success});
      } else if (ev.type === 'visual_feedback') {
        add('image', ev.description || 'Visual feedback', `<img src="${imgSrc(ev.image_base64)}" />`, {time: ts(ev.timestamp)});
      } else if (ev.type === 'image_analysis') {
        add('analysis', ev.analysis_type || 'analysis', `<pre>${esc(ev.content || '')}</pre>`, {time: ts(ev.timestamp)});
      } else if (ev.type === 'trial_complete') {
        add('complete', ev.success ? 'Run complete' : 'Run finished with failures', `<pre>${esc(ev.summary || '')}</pre>`, {time: ts(ev.timestamp), warn: !ev.success});
      } else if (ev.type === 'error') {
        add('error', ev.message || 'error', '', {time: ts(ev.timestamp), warn: true});
      }
    }
    async function connect() {
      const active = await fetch('/api/active-session').then(r => r.json()).catch(() => ({}));
      if (!active.session_id) {
        statusEl.textContent = 'waiting for RATS session';
        setTimeout(connect, 1500);
        return;
      }
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${proto}//${location.host}/ws/${active.session_id}`);
      ws.onopen = () => statusEl.textContent = 'connected';
      ws.onclose = () => { statusEl.textContent = 'disconnected'; setTimeout(connect, 1500); };
      ws.onerror = () => statusEl.textContent = 'websocket error';
      ws.onmessage = msg => { try { renderEvent(JSON.parse(msg.data)); } catch (e) { console.error(e); } };
    }
    connect();
  </script>
</body>
</html>"""


def _image_to_base64(image: np.ndarray | Image.Image | str) -> str:
    """Encode an image as a plain base64 string for Web UI image components."""
    if isinstance(image, str):
        if image.startswith("data:"):
            return image.split(",", 1)[-1]
        if len(image) > 1000:
            return image
        return base64.b64encode(Path(image).read_bytes()).decode("utf-8")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_to_data_url(image: np.ndarray | Image.Image | str) -> str:
    return f"data:image/jpeg;base64,{_image_to_base64(image)}"


def _compact_code_for_web(code: str, *, max_chars: int = 60_000, max_lines: int = 1_200) -> str:
    """Keep WebSocket/UI payloads bounded without changing executed code."""
    original_chars = len(code)
    original_lines = code.count("\n") + 1 if code else 0
    lines = code.splitlines()
    truncated = False

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        code = "\n".join(lines)
        truncated = True
    if len(code) > max_chars:
        code = code[:max_chars]
        truncated = True

    if not truncated:
        return code
    return (
        "# Display truncated for Web UI; execution used the complete code.\n"
        f"# Original size: {original_lines} lines, {original_chars} characters.\n\n"
        f"{code}\n\n"
        "# ... truncated ..."
    )


def _render_env_frame(env: Any) -> Any:
    render = getattr(env, "render", None)
    if callable(render):
        try:
            return render(mode="rgb_array")
        except TypeError:
            try:
                return render()
            except Exception:
                return None
        except Exception:
            return None
    low = getattr(env, "low_level_env", None)
    render = getattr(low, "render", None)
    if callable(render):
        try:
            return render(mode="rgb_array")
        except TypeError:
            try:
                return render()
            except Exception:
                return None
        except Exception:
            return None
    return None


def _env_viser_port(env: Any) -> int | None:
    candidates = [
        env,
        getattr(env, "low_level_env", None),
        getattr(getattr(env, "low_level_env", None), "low_level_env", None),
    ]
    for obj in candidates:
        viser_server = getattr(obj, "viser_server", None)
        websock = getattr(viser_server, "_websock_server", None)
        port = getattr(websock, "_port", None)
        if isinstance(port, int):
            return port
    return None


class RatsWebDebugger:
    """Publish a running RATS experiment into the CaP-X Web UI."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        config_path: str | None = None,
        model: str | None = None,
        env_type: str | None = None,
        replay_controller: Any | None = None,
        host: str = "0.0.0.0",
        port: int = 8200,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config_path = config_path if config_path and Path(config_path).exists() else None
        self.model = model or ""
        self.env_type = env_type or ""
        self.replay_controller = replay_controller
        self.host = host
        self.port = int(port)
        self.url = f"http://localhost:{self.port}"

        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._session: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._block_index = 0
        self._step_index_by_block: dict[int, int] = {}
        self._logger_step_index_maps: dict[int, dict[int, int]] = {}
        self._molmospaces_publisher: Any | None = None
        if self.env_type == "molmospaces":
            try:
                from rats.loop.molmospaces_viser_publisher import MolmoSpacesViserPublisher

                self._molmospaces_publisher = MolmoSpacesViserPublisher()
            except Exception as exc:
                logger.warning("MolmoSpaces Viser publisher unavailable: %s", exc)

    def _active_viser_port(self, env: Any | None = None) -> int | None:
        publisher = getattr(self, "_molmospaces_publisher", None)
        if publisher is not None:
            port = getattr(publisher, "port", None)
            if isinstance(port, int):
                return port
        return _env_viser_port(env) if env is not None else None

    @property
    def session_id(self) -> str | None:
        return getattr(self._session, "session_id", None)

    def start(self, env: Any | None = None) -> None:
        """Start the FastAPI server in a background thread."""
        if self._thread is not None:
            return

        _ensure_frontend_built()
        self._thread = threading.Thread(target=self._thread_main, name="rats-web-ui", daemon=True)
        self._thread.start()
        try:
            startup_timeout = float(os.getenv("RATS_WEB_UI_STARTUP_TIMEOUT", "120"))
        except ValueError:
            startup_timeout = 120.0
        ready = self._ready.wait(timeout=max(1.0, startup_timeout))
        if self._startup_error is not None:
            raise RuntimeError(f"RATS Web UI failed to start: {self._startup_error}") from self._startup_error
        if not ready:
            thread_alive = bool(self._thread and self._thread.is_alive())
            raise RuntimeError(
                "RATS Web UI startup timed out before creating a session "
                f"after {startup_timeout:.1f}s "
                f"(thread_alive={thread_alive}, port={self.port}). "
                "This is usually slow first-time FastAPI/CaP-X import or a "
                "blocked web server startup. Retry with "
                "`RATS_WEB_UI_STARTUP_TIMEOUT=180` or run without `--web-ui` "
                "to continue the benchmark while still saving per-step artifacts."
            )
        if self._session is None:
            thread_alive = bool(self._thread and self._thread.is_alive())
            raise RuntimeError(
                "RATS Web UI signaled ready but did not create a session "
                f"(thread_alive={thread_alive}, port={self.port})."
            )

        if env is not None:
            self.set_env(env)
        self.set_state(SessionState.RUNNING)
        self.status(
            "RATS Web UI attached",
            details=(
                f"Output directory: `{self.output_dir}`\n\n"
                f"Model: `{self.model or 'default'}`\n\n"
                f"Env type: `{self.env_type or 'unknown'}`\n\n"
                "This session mirrors `scripts/run_rats.py`; the Start Trial "
                "button is for the CaP-X single-trial runner and is not needed here."
            ),
        )

    def _thread_main(self) -> None:
        try:
            import uvicorn
            from rats.web.server import create_app
            from rats.web.session_manager import get_session_manager
            from fastapi.responses import HTMLResponse

            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)

            async def serve() -> None:
                webui_dir = _project_root() / "web-ui"
                frontend_dist = webui_dir / "dist"
                use_fallback = not frontend_dist.exists() or _frontend_needs_build(webui_dir)
                old_force_fallback = os.environ.get("RATS_WEB_UI_FORCE_FALLBACK")
                if use_fallback:
                    os.environ["RATS_WEB_UI_FORCE_FALLBACK"] = "1"
                try:
                    app = create_app()
                finally:
                    if old_force_fallback is None:
                        os.environ.pop("RATS_WEB_UI_FORCE_FALLBACK", None)
                    else:
                        os.environ["RATS_WEB_UI_FORCE_FALLBACK"] = old_force_fallback
                if self.replay_controller is not None:
                    app.state.rats_replay_controller = self.replay_controller
                if use_fallback:
                    @app.get("/")
                    async def rats_debug_fallback() -> HTMLResponse:
                        return HTMLResponse(_fallback_html())
                # Do not auto-start a CaP-X trial; RATS is the active session.
                app.state.default_config_path = None
                manager = get_session_manager()
                session = await manager.create_session()
                session.config_path = self.config_path
                session.config = {
                    "rats_web_debug": True,
                    "output_dir": str(self.output_dir),
                    "model": self.model,
                    "env_type": self.env_type,
                }
                session.state = SessionState.RUNNING
                # Keep a task present so the existing Stop websocket command
                # sets session.cancel_event. RATS checks that flag at loop
                # boundaries and can also be interrupted during env execution.
                session.task = asyncio.create_task(session.cancel_event.wait())
                self._session = session
                self._ready.set()

                config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
                self._server = uvicorn.Server(config)
                await self._server.serve()

            loop.run_until_complete(serve())
        except BaseException as exc:  # pragma: no cover - startup failure path
            self._startup_error = exc
            self._ready.set()
            logger.exception("RATS Web UI server failed")

    def _emit(self, event: WSEventBase) -> None:
        if self._loop is None or self._session is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._session.emit(event), self._loop)
        except RuntimeError:
            return

    def _set_session_attr(self, **attrs: Any) -> None:
        if self._loop is None or self._session is None:
            return

        async def update() -> None:
            for key, value in attrs.items():
                setattr(self._session, key, value)

        try:
            asyncio.run_coroutine_threadsafe(update(), self._loop)
        except RuntimeError:
            return

    def set_env(self, env: Any) -> None:
        """Attach the active RATS env and enable API/WebUI visualization logs."""
        publisher = getattr(self, "_molmospaces_publisher", None)
        if publisher is not None:
            try:
                publisher.start()
            except Exception:
                pass
        if self._session is not None:
            self._session.env = env
            if publisher is not None:
                self._session.rats_viser_server = getattr(publisher, "server", None)
        self._set_session_attr(
            env=env,
            rats_viser_server=(getattr(publisher, "server", None) if publisher is not None else None),
        )
        for api in getattr(env, "_apis", {}).values():
            enable = getattr(api, "enable_webui", None)
            if callable(enable):
                try:
                    enable(True)
                except Exception:
                    pass
            if publisher is not None:
                try:
                    setattr(api, "_viser_publisher", publisher)
                except Exception:
                    pass
        low_level_env = getattr(env, "low_level_env", None)
        if publisher is not None and low_level_env is not None:
            with suppress(Exception):
                low_level_env._viser_publisher = publisher
        if publisher is not None:
            try:
                publisher.publish_env(env, reason="env_attached")
            except Exception:
                pass
        self._emit(
            ViserReloadEvent(
                session_id=self.session_id or "rats",
                port=self._active_viser_port(env),
            )
        )

    def publish_env(self, env: Any, *, reason: str = "update") -> None:
        """Best-effort push of the current env observation into owned Viser."""
        publisher = getattr(self, "_molmospaces_publisher", None)
        if publisher is None:
            return
        try:
            if publisher.publish_env(env, reason=reason):
                self._set_session_attr(rats_viser_server=getattr(publisher, "server", None))
        except Exception:
            pass

    def mark_viser_recording(self) -> int | None:
        """Return the current MolmoSpaces Viser snapshot index, if available."""
        publisher = getattr(self, "_molmospaces_publisher", None)
        count_fn = getattr(publisher, "recorded_frame_count", None)
        if not callable(count_fn):
            return None
        try:
            return int(count_fn())
        except Exception:
            return None

    def export_viser_recording(
        self,
        output_dir: Path | str,
        *,
        frame_start: int | None = None,
        frame_end: int | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> dict[str, str] | None:
        """Persist a Viser-style reconstruction for a run or attempt.

        The returned paths are absolute/relative exactly as written by the
        publisher.  Callers store them in iteration JSON so the history WebUI
        and reports can link directly to the playback.
        """
        publisher = getattr(self, "_molmospaces_publisher", None)
        export_fn = getattr(publisher, "export_recording", None)
        if not callable(export_fn):
            return None
        try:
            video_path = export_fn(
                output_dir,
                frame_start=frame_start,
                frame_end=frame_end,
                metadata_extra=metadata_extra,
            )
        except Exception as exc:
            logger.debug("MolmoSpaces Viser recording export failed: %s", exc)
            return None
        if video_path is None:
            return None
        video = Path(video_path)
        out_dir = video.parent
        result = {
            "video_path": str(video),
            "metadata_path": str(out_dir / "viser_visualization_metadata.json"),
        }
        manifest = out_dir / "viser_frames_manifest.json"
        frames_npz = out_dir / "viser_frames.npz"
        if manifest.exists():
            result["frames_manifest_path"] = str(manifest)
        if frames_npz.exists():
            result["frames_npz_path"] = str(frames_npz)
        return result

    def set_state(self, state: SessionState) -> None:
        if self._session is not None:
            self._session.state = state
        self._emit(StateUpdateEvent(session_id=self.session_id or "rats", state=state))

    def stop_requested(self) -> bool:
        return bool(getattr(getattr(self, "_session", None), "cancel_event", None) and self._session.cancel_event.is_set())

    def raise_if_stopped(self) -> None:
        if self.stop_requested():
            raise KeyboardInterrupt("RATS Web UI stop requested")

    @contextmanager
    def execution_interrupt_scope(self) -> Iterator[None]:
        """Let the Web UI Stop button interrupt a blocking env execution."""
        if self._session is None:
            yield
            return
        self._set_session_attr(execution_thread_id=threading.get_ident())
        try:
            yield
        finally:
            self._set_session_attr(execution_thread_id=None)

    def status(self, message: str, *, details: str | None = None, running: bool = False) -> None:
        self._emit(
            EnvironmentInitEvent(
                session_id=self.session_id or "rats",
                status="starting" if running else "description_complete",
                message=message,
                description_content=details,
            )
        )

    def visual_feedback(self, image: Any, description: str) -> None:
        if image is None:
            return
        try:
            data_url = _image_to_data_url(image)
        except Exception:
            return
        self._emit(
            VisualFeedbackEvent(
                session_id=self.session_id or "rats",
                image_base64=data_url,
                description=description,
            )
        )

    def capture_env_frame(self, env: Any, description: str) -> None:
        self.visual_feedback(_render_env_frame(env), description)

    def task_proposed(self, iteration: int, task: dict[str, Any], scene_context: dict[str, Any]) -> None:
        goal = task.get("goal_conditions") or task.get("language") or task.get("activity_name", "")
        details = [
            f"## Iteration {iteration} Task",
            f"- Activity: `{task.get('activity_name', 'unknown')}`",
            f"- Scene: `{scene_context.get('scene_model', 'unknown')}`",
            f"- Goal: {goal}",
        ]
        if task.get("reasoning"):
            details.extend(["", "### Proposer Reasoning", str(task.get("reasoning", ""))])
        if task.get("expected_new_skills"):
            details.extend(["", "### Expected New Skills", ", ".join(map(str, task.get("expected_new_skills", [])))])
        self.status(f"Iteration {iteration}: task selected", details="\n".join(details))

    def plan_ready(self, iteration: int, plan: dict[str, Any]) -> None:
        lines = [f"## Iteration {iteration} Plan"]
        for step in plan.get("steps", []) or []:
            sid = step.get("id", step.get("step_id", "?"))
            desc = step.get("description", "")
            skills = ", ".join(step.get("relevant_skills", []) or [])
            lines.append(f"- **{sid}**: {desc}" + (f"  \n  skills: `{skills}`" if skills else ""))
        self._emit(
            ModelResponseEvent(
                session_id=self.session_id or "rats",
                content="\n".join(lines),
                reasoning=None,
                code_blocks=[],
                decision=DecisionType.INITIAL,
            )
        )

    def start_attempt(
        self,
        *,
        iteration: int,
        attempt: int,
        code: str,
        label: str = "Policy Attempt",
        display_code: str | None = None,
    ) -> int:
        block_idx = self._block_index
        self._block_index += 1
        self._step_index_by_block[block_idx] = 0
        self._logger_step_index_maps[block_idx] = {}
        self._set_session_attr(current_block_index=block_idx, total_code_blocks=self._block_index)

        decision = DecisionType.INITIAL if attempt == 0 else DecisionType.REGENERATE
        ui_code = _compact_code_for_web(display_code if display_code is not None else code)
        self._emit(
            ModelResponseEvent(
                session_id=self.session_id or "rats",
                content=f"## {label} {attempt + 1}\n\n```python\n{ui_code}\n```",
                reasoning=None,
                code_blocks=[ui_code],
                decision=decision,
            )
        )
        self._emit(
            CodeExecutionStartEvent(
                session_id=self.session_id or "rats",
                block_index=block_idx,
                code=ui_code,
            )
        )
        self.step(
            block_idx,
            "RATS Attempt",
            f"Iteration {iteration}, attempt {attempt + 1}",
            highlight=True,
        )
        return block_idx

    def _next_step_index(self, block_idx: int) -> int:
        idx = self._step_index_by_block.get(block_idx, 0)
        self._step_index_by_block[block_idx] = idx + 1
        return idx

    def step(
        self,
        block_idx: int | None,
        tool_name: str,
        text: str,
        *,
        images: Any | list[Any] | None = None,
        highlight: bool = False,
        frame_start: int | None = None,
        frame_end: int | None = None,
        start_s: float | None = None,
        end_s: float | None = None,
        timeline_kind: str | None = None,
        timeline_label: str | None = None,
    ) -> None:
        if block_idx is None:
            self.status(tool_name, details=text)
            return
        encoded: list[str] = []
        if images is not None:
            values = images if isinstance(images, list) else [images]
            for img in values:
                if img is None:
                    continue
                try:
                    encoded.append(_image_to_base64(img))
                except Exception:
                    pass
        self._emit(
            ExecutionStepEvent(
                session_id=self.session_id or "rats",
                block_index=block_idx,
                step_index=self._next_step_index(block_idx),
                tool_name=tool_name,
                text=text,
                images=encoded,
                highlight=highlight,
                frame_start=frame_start,
                frame_end=frame_end,
                start_s=start_s,
                end_s=end_s,
                timeline_kind=timeline_kind,
                timeline_label=timeline_label,
            )
        )

    def execution_step_callback(self, block_idx: int):
        """Return a callback compatible with ``execution_logger``."""
        def emit_step(step: Any) -> None:
            index_map = self._logger_step_index_maps.setdefault(block_idx, {})
            raw_idx = int(getattr(step, "step_index", 0))
            if raw_idx not in index_map:
                index_map[raw_idx] = self._next_step_index(block_idx)
            self._emit(
                ExecutionStepEvent(
                    session_id=self.session_id or "rats",
                    block_index=block_idx,
                    step_index=index_map[raw_idx],
                    tool_name=getattr(step, "tool_name", "Execution Step"),
                    text=getattr(step, "text", ""),
                    images=list(getattr(step, "images", []) or []),
                    highlight=bool(getattr(step, "highlight", False)),
                    frame_start=getattr(step, "frame_start", None),
                    frame_end=getattr(step, "frame_end", None),
                    start_s=getattr(step, "start_s", None),
                    end_s=getattr(step, "end_s", None),
                    timeline_kind=getattr(step, "timeline_kind", None),
                    timeline_label=getattr(step, "timeline_label", None),
                )
            )
        return emit_step

    def finish_attempt(
        self,
        block_idx: int | None,
        execution_result: dict[str, Any],
        *,
        success: bool | None = None,
    ) -> None:
        if block_idx is None:
            return
        before = execution_result.get("before_frame")
        after = execution_result.get("after_frame")
        imgs = [img for img in (before, after) if img is not None]
        if imgs:
            self.step(block_idx, "Robot Observation", "Before/after frames from this attempt.", images=imgs)
        self._emit(
            CodeExecutionResultEvent(
                session_id=self.session_id or "rats",
                block_index=block_idx,
                success=bool(execution_result.get("success") if success is None else success),
                stdout=execution_result.get("stdout", "") or "",
                stderr=execution_result.get("stderr", "") or "",
                reward=float(execution_result.get("reward") or 0.0),
                task_completed=execution_result.get("task_completed"),
            )
        )

    def verification(self, block_idx: int | None, verification: dict[str, Any]) -> None:
        status = "SUCCESS" if verification.get("success") else "FAILED"
        lines = [
            f"Verification: **{status}**",
            f"Reward: `{verification.get('reward')}`",
            f"Task completed: `{verification.get('task_completed')}`",
        ]
        if verification.get("state_hint"):
            lines.extend(["", str(verification.get("state_hint"))])
        llm = verification.get("llm_analysis") or {}
        if llm.get("fix_suggestion"):
            lines.extend(["", "Fix suggestion:", str(llm.get("fix_suggestion"))])
        self.step(block_idx, "Verification", "\n".join(lines), highlight=not bool(verification.get("success")))

    def diagnosis(self, block_idx: int | None, diagnosis: dict[str, Any]) -> None:
        lines = [
            f"Failure mode: `{diagnosis.get('failure_mode', 'unknown')}`",
            "",
            str(diagnosis.get("policy_feedback", "") or diagnosis.get("diagnosis_summary", "")),
        ]
        self.step(block_idx, "Diagnosis", "\n".join(lines), highlight=True)
        if diagnosis.get("diagnosis_summary") or diagnosis.get("policy_feedback"):
            self._emit(
                ImageAnalysisEvent(
                    session_id=self.session_id or "rats",
                    analysis_type="state_comparison",
                    content="\n".join(lines),
                    model_used=self.model or None,
                )
            )

    def complete(self, summary: dict[str, Any]) -> None:
        publisher = getattr(self, "_molmospaces_publisher", None)
        if publisher is not None:
            try:
                video_path = publisher.export_recording(
                    self.output_dir,
                    export_frame_archive=False,
                )
                if video_path is not None:
                    logger.info("MolmoSpaces Viser recording saved to %s", video_path)
            except Exception as exc:
                logger.debug("MolmoSpaces Viser recording export failed: %s", exc)
        self.set_state(SessionState.COMPLETE)
        self._emit(
            TrialCompleteEvent(
                session_id=self.session_id or "rats",
                success=bool(summary.get("failed_iterations", 1) == 0),
                total_reward=float(summary.get("successful_iterations", 0)),
                task_completed=None,
                num_regenerations=0,
                num_code_blocks=self._block_index,
                summary=(
                    f"Total iterations: {summary.get('total_iterations')}\n"
                    f"Successful: {summary.get('successful_iterations')}\n"
                    f"Failed: {summary.get('failed_iterations')}\n"
                    f"Final skill library size: {summary.get('final_skill_library_size')}\n"
                    f"Learned skills: {summary.get('learned_skills')}"
                ),
            )
        )

    def error(self, message: str) -> None:
        self.set_state(SessionState.ERROR)
        self._emit(
            ErrorEvent(
                session_id=self.session_id or "rats",
                message=message,
                recoverable=False,
            )
        )
