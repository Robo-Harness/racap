"""Remote MolmoSpaces bridge client for two-process RATS operation.

Provides the same call surface as ``MolmoSpacesBridge`` (and the same duck-type
interface expected by ``FrankaMolmoSpacesEnv``) but delegates every call over
length-prefixed msgpack-over-TCP to an ``mlspaces_server.py`` running in a
separate conda/venv.

This module intentionally has **zero** ``molmo_spaces`` imports so it can run
inside the capx venv which does not (and cannot) install the MolmoSpaces stack.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import time
from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np

m.patch()

logger = logging.getLogger("rats.molmospaces_remote")


# RPCs that are safe to retry transparently after a reconnect: they do
# not advance the simulator and are idempotent server-side. Anything not
# in this set (`step`, `reset`, `set_task_from_spec`, `request_new_house`,
# `resample_task`, `init`, `close`) needs explicit handling because a
# silent retry could double-step / re-init mid-iteration.
_IDEMPOTENT_RPCS: frozenset[str] = frozenset({
    "ping",
    "build_observation",
    "judge_success",
    "get_reward",
    "get_info",
    "render",
    "render_wrist",
    "get_task_description",
    "list_task_descriptors",
    "get_task_metadata",
    "get_task_descriptor",
    "describe_scene_inventory",
    "describe_contact_pairs",
    "describe_object_relation",
    "get_move_group_info",
})


# ------------------------------------------------------------------
# Wire protocol (mirrors mlspaces_server.py / msgpack_server_client_utils.py)
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
        raise ConnectionError("Server disconnected")
    (msg_len,) = struct.unpack("!I", header)
    payload = _recvall(sock, msg_len)
    if payload is None:
        raise ConnectionError("Server disconnected mid-message")
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
# Move-group proxy (duck-types the properties read by FrankaMolmoSpacesEnv)
# ------------------------------------------------------------------

class _MoveGroupProxy:
    """Thin proxy returned by ``RemoteMolmoSpacesBridge.robot_view.get_move_group()``.

    ``FrankaMolmoSpacesEnv`` reads ``mg.joint_pos``, ``mg.inter_finger_dist``,
    ``mg.inter_finger_dist_range``, and ``mg.leaf_frame_to_world`` on the arm
    and gripper move-groups.  This proxy fetches them from the server on demand.
    """

    def __init__(self, info: dict[str, Any]) -> None:
        self._info = info

    @property
    def joint_pos(self) -> np.ndarray:
        return np.asarray(self._info["joint_pos"], dtype=np.float64)

    @property
    def leaf_frame_to_world(self) -> np.ndarray:
        mat = self._info.get("leaf_frame_to_world")
        if mat is None:
            return np.eye(4, dtype=np.float64)
        return np.asarray(mat, dtype=np.float64).reshape(4, 4)

    @property
    def inter_finger_dist(self) -> float:
        return float(self._info.get("inter_finger_dist", 0.0))

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        r = self._info.get("inter_finger_dist_range", [0.0, 1.0])
        return (float(r[0]), float(r[1]))


class _RobotViewProxy:
    """Duck-types ``bridge.robot_view`` so callers can do
    ``bridge.robot_view.get_move_group("arm")`` transparently.
    """

    def __init__(self, rpc_fn) -> None:
        self._rpc = rpc_fn

    def get_move_group(self, group_name: str) -> _MoveGroupProxy:
        info = self._rpc("get_move_group_info", group_name=group_name)
        return _MoveGroupProxy(info)


# ------------------------------------------------------------------
# Remote bridge
# ------------------------------------------------------------------

class RemoteMolmoSpacesBridge:
    """RPC client that mirrors the ``MolmoSpacesBridge`` call surface.

    Connects to ``mlspaces_server.py`` and serialises every method call
    as a msgpack request.  The server owns the actual ``molmo_spaces``
    simulator and sends back results (including numpy arrays).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9100,
        *,
        connect_timeout: float = 30.0,
        rpc_timeout: float | None = None,
        init_kwargs: dict[str, Any] | None = None,
        max_reconnect_attempts: int | None = None,
        reconnect_backoff_seconds: float | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._connect_timeout = connect_timeout
        self._rpc_timeout = self._resolve_rpc_timeout(rpc_timeout)
        # Saved for explicit init() calls and optional recovery.  Do NOT replay
        # this by default after every reconnect: this repo's mlspaces_server
        # preserves the bridge object across client disconnects, while an init
        # RPC always rebuilds the sampler.  Replaying init after a mid-attempt
        # render/build_observation timeout therefore resets the server to the
        # seed task and makes later turns observe a different environment.
        self._init_kwargs: dict[str, Any] | None = (
            dict(init_kwargs) if init_kwargs else None
        )
        self._max_reconnect_attempts = self._resolve_int_env(
            max_reconnect_attempts,
            env="RATS_MOLMOSPACES_MAX_RECONNECT",
            default=5,
        )
        self._reconnect_backoff = self._resolve_float_env(
            reconnect_backoff_seconds,
            env="RATS_MOLMOSPACES_RECONNECT_BACKOFF",
            default=1.0,
        )
        self._replay_init_on_reconnect = self._resolve_bool_env(
            env="RATS_MOLMOSPACES_REPLAY_INIT_ON_RECONNECT",
            default=False,
        )

        self._connect()

        if self._init_kwargs:
            self.init(**self._init_kwargs)

        self.robot_view = _RobotViewProxy(self._call)

    # -- connection management ------------------------------------------

    @staticmethod
    def _resolve_rpc_timeout(explicit: float | None) -> float | None:
        if explicit is not None:
            return explicit
        raw = os.environ.get("RATS_MOLMOSPACES_RPC_TIMEOUT", "120")
        if raw.strip().lower() in {"", "none", "0", "false", "off"}:
            return None
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                "Invalid RATS_MOLMOSPACES_RPC_TIMEOUT=%r; using 120s",
                raw,
            )
            return 120.0

    @staticmethod
    def _resolve_int_env(explicit: int | None, *, env: str, default: int) -> int:
        if explicit is not None:
            return int(explicit)
        raw = os.environ.get(env, "")
        if not raw.strip():
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using %d", env, raw, default)
            return default

    @staticmethod
    def _resolve_float_env(explicit: float | None, *, env: str, default: float) -> float:
        if explicit is not None:
            return float(explicit)
        raw = os.environ.get(env, "")
        if not raw.strip():
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using %.2fs", env, raw, default)
            return default

    @staticmethod
    def _resolve_bool_env(*, env: str, default: bool) -> bool:
        raw = os.environ.get(env, "")
        if not raw.strip():
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    def _connect(self) -> None:
        logger.info("Connecting to mlspaces server at %s:%d ...", self._host, self._port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout)
        sock.connect((self._host, self._port))
        sock.settimeout(self._rpc_timeout)
        self._sock = sock
        logger.info("Connected to mlspaces server")

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _reconnect(self) -> None:
        """Close and re-open the socket to drain any pending wire data.

        The server's `_handle_client` loop re-enters `accept()` on disconnect
        and preserves the bridge state, so a fresh TCP connection gives us a
        clean request/response alignment without losing session state.
        """
        self._close_socket()
        self._connect()

    def _ensure_connected(self) -> None:
        """Reconnect with backoff if the socket is currently closed.

        Used by ``_call`` when a previous RPC tore the socket down (timeout,
        broken pipe, server disconnect) so the next request can transparently
        re-establish the wire instead of raising "Not connected" until the
        outer loop happens to call ``reset()``.

        Important: the bundled ``scripts/mlspaces_server.py`` preserves bridge
        state across client disconnects.  Re-issuing ``init`` after a reconnect
        rebuilds the sampler and can change the task/house in the middle of a
        multi-turn attempt.  Only replay init when the opt-in env var
        RATS_MOLMOSPACES_REPLAY_INIT_ON_RECONNECT is set.
        """
        if self._sock is not None:
            return
        last_exc: BaseException | None = None
        for attempt in range(1, self._max_reconnect_attempts + 1):
            try:
                self._connect()
            except (OSError, socket.timeout) as exc:
                last_exc = exc
                delay = self._reconnect_backoff * (2 ** (attempt - 1))
                logger.warning(
                    "mlspaces reconnect attempt %d/%d failed (%s: %s); "
                    "retrying in %.1fs",
                    attempt,
                    self._max_reconnect_attempts,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                self._close_socket()
                if attempt < self._max_reconnect_attempts:
                    time.sleep(delay)
                continue
            # Optional compatibility path for older server implementations
            # that drop bridge state on disconnect.  Disabled by default
            # because init is destructive for the bundled server.
            if self._replay_init_on_reconnect and self._init_kwargs:
                try:
                    _send(self._sock, {"method": "init", "kwargs": self._init_kwargs})
                    resp = _recv(self._sock)
                    if isinstance(resp, dict) and resp.get("status") == "error":
                        raise RuntimeError(
                            f"mlspaces re-init failed: {resp.get('message')}"
                        )
                except (OSError, socket.timeout, ConnectionError, RuntimeError) as exc:
                    last_exc = exc
                    logger.warning(
                        "mlspaces re-init after reconnect failed (%s: %s); "
                        "dropping socket and retrying",
                        type(exc).__name__, exc,
                    )
                    self._close_socket()
                    continue
            logger.info(
                "Auto-reconnected to mlspaces server (attempt %d/%d)",
                attempt, self._max_reconnect_attempts,
            )
            return
        # Exhausted reconnect attempts.
        raise ConnectionError(
            f"Failed to reconnect to mlspaces server at "
            f"{self._host}:{self._port} after "
            f"{self._max_reconnect_attempts} attempts"
        ) from last_exc

    def _call(self, method: str, **kwargs: Any) -> Any:
        """Send an RPC and return the result (or raise on error).

        Auto-reconnect:
          * Before sending, if the socket is closed, transparently
            reconnect with exponential backoff.  ``init`` replay is opt-in via
            RATS_MOLMOSPACES_REPLAY_INIT_ON_RECONNECT because it is destructive
            for the bundled server.
          * If the request itself dies on the wire (broken pipe, server
            close, timeout), close the socket and — for IDEMPOTENT RPCs
            only — silently retry once on a fresh connection. Stateful
            RPCs (`step`, `reset`, etc.) raise on first failure: silent
            retry could double-step or re-init mid-iteration.
        """
        # Ensure we have a live socket; this is a no-op when already connected.
        try:
            self._ensure_connected()
        except ConnectionError:
            raise

        try:
            _send(self._sock, {"method": method, "kwargs": kwargs})  # type: ignore[arg-type]
            resp = _recv(self._sock)  # type: ignore[arg-type]
        except socket.timeout as exc:
            self._close_socket()
            if method in _IDEMPOTENT_RPCS:
                logger.warning(
                    "mlspaces RPC %s timed out after %.1fs; "
                    "auto-reconnecting and retrying once",
                    method, self._rpc_timeout or float("inf"),
                )
                self._ensure_connected()
                _send(self._sock, {"method": method, "kwargs": kwargs})  # type: ignore[arg-type]
                resp = _recv(self._sock)  # type: ignore[arg-type]
            else:
                raise TimeoutError(
                    f"Timed out waiting for mlspaces server response to {method} "
                    f"after {self._rpc_timeout}s"
                ) from exc
        except (ConnectionError, OSError) as exc:
            self._close_socket()
            if method in _IDEMPOTENT_RPCS:
                logger.warning(
                    "mlspaces RPC %s lost connection (%s: %s); "
                    "auto-reconnecting and retrying once",
                    method, type(exc).__name__, exc,
                )
                self._ensure_connected()
                _send(self._sock, {"method": method, "kwargs": kwargs})  # type: ignore[arg-type]
                resp = _recv(self._sock)  # type: ignore[arg-type]
            else:
                raise
        if resp.get("status") == "error":
            raise RuntimeError(f"mlspaces server error ({method}): {resp.get('message')}")
        return resp.get("result")

    @staticmethod
    def _validate_observation_payload(observation: dict[str, Any]) -> dict[str, Any]:
        robot_base_pose = observation.get("robot_base_pose")
        pose_arr = np.asarray(robot_base_pose, dtype=np.float64)
        if pose_arr.shape != (7,):
            raise RuntimeError(
                f"mlspaces server returned invalid robot_base_pose shape: {pose_arr.shape}",
            )
        if not np.all(np.isfinite(pose_arr)):
            raise RuntimeError("mlspaces server returned non-finite robot_base_pose")
        observation["robot_base_pose"] = pose_arr
        return observation

    # -- MolmoSpacesBridge-compatible public API -------------------------

    def init(self, **kwargs: Any) -> None:
        # Persist the latest init kwargs for explicit compatibility recovery.
        # Replaying them after reconnect is opt-in because the bundled server
        # preserves bridge state across client disconnects and init rebuilds
        # the sampler.
        if kwargs:
            self._init_kwargs = dict(kwargs)
        self._call("init", **kwargs)

    def ping(self) -> str:
        return str(self._call("ping"))

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        # Reconnect before each reset: if a previous RPC left the wire
        # desynced (client and server disagreeing on which response belongs
        # to which request), a fresh socket drains the backlog.
        self._reconnect()
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        result = self._call("reset", **kwargs)
        return self._validate_observation_payload(result["obs"]), result["info"]

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        result = self._call("step", action=action)
        return (
            result["obs"],
            float(result["reward"]),
            bool(result["terminated"]),
            bool(result["truncated"]),
            result["info"],
        )

    def build_observation(self) -> dict[str, Any]:
        return self._validate_observation_payload(self._call("build_observation"))

    def judge_success(self) -> bool:
        return bool(self._call("judge_success"))

    def get_reward(self) -> float:
        return float(self._call("get_reward"))

    def get_info(self) -> dict[str, Any]:
        try:
            result = self._call("get_info")
        except Exception:
            return {}
        return dict(result or {})

    def render(self, camera_name: str = "agentview") -> np.ndarray:
        return np.asarray(self._call("render", camera_name=camera_name))

    def render_wrist(self) -> np.ndarray:
        return np.asarray(self._call("render_wrist"))

    def get_task_description(self) -> str:
        return str(self._call("get_task_description"))

    def list_task_descriptors(self) -> list[dict[str, Any]]:
        result = self._call("list_task_descriptors")
        return list(result or [])

    def get_task_metadata(self) -> dict[str, Any]:
        result = self._call("get_task_metadata")
        return dict(result or {})

    def get_anchored_task_target(self) -> dict[str, Any]:
        result = self._call("get_anchored_task_target")
        return dict(result or {})

    def anchor_to_pickup(self, target_internal_name: str) -> dict[str, Any]:
        """RPC shim for MolmoSpacesBridge.anchor_to_pickup."""
        result = self._call(
            "anchor_to_pickup", target_internal_name=str(target_internal_name)
        )
        return dict(result or {})

    def describe_scene_obstacles(
        self,
        *,
        exclude_internal_names: list[str] | None = None,
        max_distance_m: float = 3.0,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"max_distance_m": float(max_distance_m)}
        if exclude_internal_names is not None:
            kwargs["exclude_internal_names"] = list(exclude_internal_names)
        result = self._call("describe_scene_obstacles", **kwargs)
        return list(result or [])

    def get_task_descriptor(self, canonical_task_id: str | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if canonical_task_id is not None:
            kwargs["canonical_task_id"] = canonical_task_id
        result = self._call("get_task_descriptor", **kwargs)
        return dict(result or {})

    def resample_task(
        self,
        *,
        house_index: int | None = None,
        canonical_task_id: str | None = None,
        episode_index: int | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if house_index is not None:
            kwargs["house_index"] = house_index
        if canonical_task_id is not None:
            kwargs["canonical_task_id"] = canonical_task_id
        if episode_index is not None:
            kwargs["episode_index"] = episode_index
        if self._init_kwargs is not None:
            if house_index is not None:
                self._init_kwargs["house_index"] = int(house_index)
            if canonical_task_id is not None:
                self._init_kwargs["canonical_task_id"] = canonical_task_id
            if episode_index is not None:
                self._init_kwargs["episode_index"] = int(episode_index)
        self._call("resample_task", **kwargs)

    # ------------------------------------------------------------------
    # Open-mode task proposer plumbing (mirrors MolmoSpacesBridge)
    # ------------------------------------------------------------------

    def describe_scene_inventory(self) -> dict[str, Any]:
        """Snapshot of the active house's pickables / receptacles / articulations.

        Forwarded to ``MolmoSpacesBridge.describe_scene_inventory`` on
        the server side. Used by the rats novel-task proposer to drive
        the LLM's choice of ``(task_type, target)`` triple from real
        in-scene contents instead of a static catalog.
        """
        result = self._call("describe_scene_inventory")
        return dict(result or {})

    def describe_contact_pairs(self, *, max_pairs: int = 64) -> dict[str, Any]:
        """Snapshot of robot end-effector MuJoCo contact pairs on the server."""
        result = self._call("describe_contact_pairs", max_pairs=int(max_pairs))
        return dict(result or {})

    def describe_object_relation(self, pickup_obj_name: str, receptacle_name: str) -> dict[str, Any]:
        """Privileged support/contact relation for an arbitrary object pair."""
        result = self._call(
            "describe_object_relation",
            pickup_obj_name=str(pickup_obj_name or ""),
            receptacle_name=str(receptacle_name or ""),
        )
        return dict(result or {})

    def set_task_from_spec(
        self,
        *,
        task_type: str,
        target_internal_name: str | None = None,
        place_receptacle_internal_name: str | None = None,
        joint_internal_name: str | None = None,
        joint_index: int | None = None,
    ) -> dict[str, Any]:
        """Instantiate a task from an LLM-chosen spec; see bridge docstring."""
        kwargs: dict[str, Any] = {"task_type": task_type}
        if target_internal_name is not None:
            kwargs["target_internal_name"] = target_internal_name
        if place_receptacle_internal_name is not None:
            kwargs["place_receptacle_internal_name"] = place_receptacle_internal_name
        if joint_internal_name is not None:
            kwargs["joint_internal_name"] = joint_internal_name
        if joint_index is not None:
            kwargs["joint_index"] = int(joint_index)
        result = self._call("set_task_from_spec", **kwargs)
        if self._init_kwargs is not None:
            self._init_kwargs["task_type"] = task_type
        return dict(result or {})

    def request_new_house(
        self,
        house_index: int | None = None,
        *,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """Switch the sampler to a different house; see bridge docstring.

        ``task_type`` is accepted for API symmetry. The usual
        ``FrankaMolmoSpacesEnv`` caller reinitializes the remote bridge via an
        ``init`` RPC before invoking this method.
        """
        kwargs: dict[str, Any] = {}
        if house_index is not None:
            kwargs["house_index"] = int(house_index)
        if task_type is not None:
            kwargs["task_type"] = str(task_type)
        result = self._call("request_new_house", **kwargs)
        if self._init_kwargs is not None:
            if house_index is not None:
                self._init_kwargs["house_index"] = int(house_index)
            if task_type is not None:
                self._init_kwargs["task_type"] = str(task_type)
        return dict(result or {})

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._call("close")
        except Exception:
            pass
        self._close_socket()


__all__ = ["RemoteMolmoSpacesBridge"]
