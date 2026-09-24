#!/usr/bin/env python3
"""MolmoSpaces bridge server for two-process RATS operation.

Runs inside the ``mlspaces`` conda environment and exposes the real
MolmoSpacesBridge over a length-prefixed msgpack-over-TCP protocol so that
the capx RATS process (which may live in an incompatible venv) can drive
the MuJoCo simulator without importing ``molmo_spaces`` itself.

Usage::

    conda run -n mlspaces python scripts/mlspaces_server.py --port 9100
    conda run -n mlspaces python scripts/mlspaces_server.py \\
        --port 9100 --task-type pick --scene-dataset procthor-10k \\
        --output-dir outputs/run_name
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import socket
import struct
import sys
import traceback
from pathlib import Path
from typing import Any, TextIO

import msgpack
import msgpack_numpy as m
import numpy as np

m.patch()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Shared asset cache. Set MLSPACES_CACHE_DIR in your env to override it.
os.environ.setdefault("MLSPACES_CACHE_DIR", str(PROJECT_ROOT / "rats-cache" / "molmospaces"))
os.environ.setdefault(
    "MLSPACES_ASSETS_DIR",
    str(PROJECT_ROOT / "rats" / "third_party" / "molmospaces" / "assets"),
)

logger = logging.getLogger("mlspaces_server")

logging.getLogger("molmo_spaces.env.sensors").setLevel(logging.ERROR)

_BRIDGE_MODULE_NAME = "_rats_molmospaces_bridge_impl"
_BRIDGE_INIT_KEYS = (
    "task_type",
    "scene_dataset",
    "data_split",
    "house_index",
    "benchmark_dir",
    "episode_index",
    "max_steps",
    "seed",
    "canonical_task_id",
    "reset_physical_state",
    "randomize_agentview",
    "use_recorded_cameras",
    "pickup_types",
    "require_grasp_files",
    "pin_pickup_obj_name",
    "front_facing_robot_placement",
    "candidate_house_indices",
)


class _TeeStream:
    """Write a text stream to both the original stream and a log file."""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        written = self._stream.write(data)
        self._log_file.write(data)
        return written

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    @property
    def encoding(self) -> str | None:
        return self._stream.encoding

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def resolve_log_file(log_file: str | None, output_dir: str | None) -> Path | None:
    """Resolve optional server log destination.

    ``--log-file`` is explicit and wins. ``--output-dir`` is a convenience for
    RATS runs: it writes ``mlspaces_server.log`` beside ``rats.log`` in the run
    output folder.
    """

    if log_file:
        return Path(log_file)
    if output_dir:
        return Path(output_dir) / "mlspaces_server.log"
    return None


def install_stream_log_tee(log_file: Path) -> TextIO:
    """Mirror server terminal output to ``log_file`` while keeping it visible.

    This captures Python logging, prints/warnings from MolmoSpaces, and
    uncaught tracebacks because both stdout and stderr are teed before logging
    is configured.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = log_file.open("w", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, fh)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, fh)  # type: ignore[assignment]
    return fh


def _load_molmospaces_bridge_class():
    """Load ``MolmoSpacesBridge`` from file without importing ``rats.envs.simulators``.

    A normal ``from rats.envs.simulators.molmospaces_bridge import ...`` runs
    ``rats/envs/simulators/__init__.py``, which eagerly imports optional simulators
    (robosuite, LIBERO, …) and spams tracebacks when those extras are missing.
    """
    path = PROJECT_ROOT / "rats" / "envs" / "simulators" / "molmospaces_bridge.py"
    spec = importlib.util.spec_from_file_location(_BRIDGE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MolmoSpaces bridge from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_BRIDGE_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod.MolmoSpacesBridge


# ------------------------------------------------------------------
# Wire protocol (matches rats/utils/msgpack_server_client_utils.py)
# ------------------------------------------------------------------

def _encode(obj: dict) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def _decode(raw: bytes) -> dict:
    return msgpack.unpackb(raw, raw=False)


def _send(sock: socket.socket, obj: dict) -> None:
    payload = _encode(obj)
    header = struct.pack("!I", len(payload))
    sock.sendall(header + payload)


def _recv(sock: socket.socket) -> dict:
    header = _recvall(sock, 4)
    if header is None:
        raise ConnectionError("Client disconnected")
    (msg_len,) = struct.unpack("!I", header)
    payload = _recvall(sock, msg_len)
    if payload is None:
        raise ConnectionError("Client disconnected mid-message")
    return _decode(payload)


def _recvall(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ------------------------------------------------------------------
# RPC dispatch
# ------------------------------------------------------------------

class BridgeServer:
    """Single-client synchronous RPC server wrapping MolmoSpacesBridge."""

    def __init__(self, host: str, port: int, bridge_kwargs: dict[str, Any]) -> None:
        self._host = host
        self._port = port
        self._bridge_kwargs = bridge_kwargs
        self._bridge = None
        self._active_bridge_kwargs: dict[str, Any] | None = None

    def _compute_effective_bridge_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        kw = dict(self._bridge_kwargs)
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    kw[key] = value
        return {key: kw[key] for key in _BRIDGE_INIT_KEYS if key in kw}

    @staticmethod
    def _bridge_ctor_kwargs(effective_kwargs: dict[str, Any]) -> dict[str, Any]:
        return dict(effective_kwargs)

    def _close_bridge(self) -> None:
        if self._bridge is None:
            self._active_bridge_kwargs = None
            return
        try:
            self._bridge.close()
        except Exception:
            logger.warning("Failed closing existing MolmoSpacesBridge cleanly", exc_info=True)
        finally:
            self._bridge = None
            self._active_bridge_kwargs = None

    def _ensure_bridge(
        self,
        kwargs: dict[str, Any] | None = None,
        *,
        allow_recreate: bool = False,
    ) -> None:
        effective_kwargs = self._compute_effective_bridge_kwargs(kwargs)
        if self._bridge is not None:
            # `init` RPCs (allow_recreate=True) always rebuild — short-circuiting
            # on matching kwargs leaves sampler RNG state, _task_counter, and
            # pinned task identity carried over from the previous client run,
            # so two consecutive runs with the same seed would pick different
            # objects. Non-init calls (reset/step/etc.) still reuse.
            if allow_recreate:
                if effective_kwargs != self._active_bridge_kwargs:
                    logger.info(
                        "Recreating MolmoSpacesBridge because init kwargs changed: old=%s new=%s",
                        self._active_bridge_kwargs,
                        effective_kwargs,
                    )
                else:
                    logger.info(
                        "Recreating MolmoSpacesBridge on init RPC to reset sampler state"
                    )
                self._close_bridge()
            else:
                return
        MolmoSpacesBridge = _load_molmospaces_bridge_class()

        ctor_kwargs = self._bridge_ctor_kwargs(effective_kwargs)
        logger.info("Creating MolmoSpacesBridge with %s", ctor_kwargs)
        self._bridge = MolmoSpacesBridge(**ctor_kwargs)
        self._active_bridge_kwargs = effective_kwargs
        logger.info("MolmoSpacesBridge ready")

    def _dispatch(self, request: dict) -> dict:
        method = request.get("method", "")
        kwargs = request.get("kwargs", {})

        try:
            if method == "init":
                self._ensure_bridge(kwargs, allow_recreate=True)
                return {"status": "ok", "result": None}

            self._ensure_bridge()

            if method == "reset":
                obs, info = self._bridge.reset(**kwargs)
                return {"status": "ok", "result": {"obs": obs, "info": info}}

            elif method == "step":
                obs, reward, terminated, truncated, info = self._bridge.step(kwargs.get("action", {}))
                return {"status": "ok", "result": {
                    "obs": obs,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                }}

            elif method == "build_observation":
                obs = self._bridge.build_observation()
                return {"status": "ok", "result": obs}

            elif method == "judge_success":
                return {"status": "ok", "result": bool(self._bridge.judge_success())}

            elif method == "get_reward":
                return {"status": "ok", "result": float(self._bridge.get_reward())}

            elif method == "get_info":
                return {"status": "ok", "result": self._bridge.get_info()}

            elif method == "render":
                camera = kwargs.get("camera_name", "agentview")
                frame = self._bridge.render(camera)
                return {"status": "ok", "result": frame}

            elif method == "render_wrist":
                frame = self._bridge.render_wrist()
                return {"status": "ok", "result": frame}

            elif method == "get_task_description":
                return {"status": "ok", "result": self._bridge.get_task_description()}

            elif method == "list_task_descriptors":
                return {"status": "ok", "result": self._bridge.list_task_descriptors()}

            elif method == "get_task_metadata":
                return {"status": "ok", "result": self._bridge.get_task_metadata()}

            elif method == "get_anchored_task_target":
                return {"status": "ok", "result": self._bridge.get_anchored_task_target()}

            elif method == "anchor_to_pickup":
                return {
                    "status": "ok",
                    "result": self._bridge.anchor_to_pickup(
                        kwargs.get("target_internal_name", ""),
                    ),
                }

            elif method == "describe_scene_obstacles":
                return {
                    "status": "ok",
                    "result": self._bridge.describe_scene_obstacles(**kwargs),
                }

            elif method == "get_task_descriptor":
                return {
                    "status": "ok",
                    "result": self._bridge.get_task_descriptor(
                        kwargs.get("canonical_task_id"),
                    ),
                }

            elif method == "resample_task":
                self._bridge.resample_task(**kwargs)
                return {"status": "ok", "result": None}

            elif method == "get_move_group_info":
                group_name = kwargs.get("group_name", "arm")
                mg = self._bridge.robot_view.get_move_group(group_name)
                info: dict[str, Any] = {"joint_pos": np.array(mg.joint_pos, dtype=np.float64)}
                leaf_frame = getattr(mg, "leaf_frame_to_world", None)
                if leaf_frame is not None:
                    info["leaf_frame_to_world"] = np.asarray(leaf_frame, dtype=np.float64)
                if group_name == "gripper":
                    info["inter_finger_dist"] = float(mg.inter_finger_dist)
                    info["inter_finger_dist_range"] = [
                        float(mg.inter_finger_dist_range[0]),
                        float(mg.inter_finger_dist_range[1]),
                    ]
                return {"status": "ok", "result": info}

            elif method == "close":
                self._close_bridge()
                return {"status": "ok", "result": None}

            elif method == "ping":
                return {"status": "ok", "result": "pong"}

            elif method == "describe_scene_inventory":
                return {"status": "ok", "result": self._bridge.describe_scene_inventory()}

            elif method == "describe_contact_pairs":
                return {
                    "status": "ok",
                    "result": self._bridge.describe_contact_pairs(
                        max_pairs=int(kwargs.get("max_pairs", 64)),
                    ),
                }

            elif method == "describe_object_relation":
                return {
                    "status": "ok",
                    "result": self._bridge.describe_object_relation(
                        str(kwargs.get("pickup_obj_name") or ""),
                        str(kwargs.get("receptacle_name") or ""),
                    ),
                }

            elif method == "set_task_from_spec":
                return {
                    "status": "ok",
                    "result": self._bridge.set_task_from_spec(**kwargs),
                }

            elif method == "request_new_house":
                return {
                    "status": "ok",
                    "result": self._bridge.request_new_house(**kwargs),
                }

            else:
                return {"status": "error", "message": f"Unknown method: {method}"}

        except Exception as e:
            logger.error("Error handling %s: %s", method, e, exc_info=True)
            return {"status": "error", "message": f"{type(e).__name__}: {e}"}

    def serve_forever(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self._host, self._port))
        srv.listen(1)
        logger.info("MolmoSpaces bridge server listening on %s:%d", self._host, self._port)

        while True:
            conn, addr = srv.accept()
            logger.info("Client connected from %s", addr)
            try:
                self._handle_client(conn)
            except ConnectionError:
                logger.info("Client disconnected")
            except Exception:
                logger.error("Session error:\n%s", traceback.format_exc())
            finally:
                conn.close()

    def _handle_client(self, conn: socket.socket) -> None:
        while True:
            request = _recv(conn)
            response = self._dispatch(request)
            _send(conn, response)


def main() -> None:
    parser = argparse.ArgumentParser(description="MolmoSpaces bridge RPC server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--task-type", default="pick", choices=["pick", "pick_and_place", "open", "close"])
    parser.add_argument("--scene-dataset", default="procthor-10k")
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--house-index", type=int, default=None)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--use-recorded-cameras",
        action="store_true",
        help=(
            "Use the camera definitions recorded in the benchmark JSON instead "
            "of the RATS eval-camera system. Requires --benchmark-dir."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Mirror the MolmoSpaces server terminal output to this file. "
            "Use this when starting the server in a separate terminal for "
            "post-run debugging."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Convenience alias for --log-file <output-dir>/mlspaces_server.log; "
            "pass the same run directory used by scripts/run_rats.py so server "
            "logs are saved beside rats.log."
        ),
    )
    parser.add_argument(
        "--lazy-init",
        action="store_true",
        help="Don't create the bridge until the first RPC ``init`` call. "
        "Useful when the client wants to pass constructor args at runtime.",
    )
    args = parser.parse_args()

    log_fh = None
    server_log_path = resolve_log_file(args.log_file, args.output_dir)
    if server_log_path is not None:
        log_fh = install_stream_log_tee(server_log_path)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if server_log_path is not None:
        logger.info("Mirroring MolmoSpaces server terminal output to %s", server_log_path)

    bridge_kwargs: dict[str, Any] = {
        "task_type": args.task_type,
        "scene_dataset": args.scene_dataset,
        "data_split": args.data_split,
        "max_steps": args.max_steps,
    }
    if args.house_index is not None:
        bridge_kwargs["house_index"] = args.house_index
    if args.benchmark_dir is not None:
        bridge_kwargs["benchmark_dir"] = args.benchmark_dir
    if args.seed is not None:
        bridge_kwargs["seed"] = args.seed
    if args.use_recorded_cameras:
        bridge_kwargs["use_recorded_cameras"] = True

    server = BridgeServer(args.host, args.port, bridge_kwargs)

    if not args.lazy_init:
        logger.info("Eagerly initializing MolmoSpacesBridge ...")
        server._ensure_bridge()

    try:
        server.serve_forever()
    finally:
        if log_fh is not None:
            if isinstance(sys.stdout, _TeeStream):
                sys.stdout = sys.stdout._stream  # type: ignore[assignment]
            if isinstance(sys.stderr, _TeeStream):
                sys.stderr = sys.stderr._stream  # type: ignore[assignment]
            log_fh.close()


if __name__ == "__main__":
    main()
