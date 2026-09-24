from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import requests
from scipy.spatial.transform import Rotation as SciRotation

from rats.utils.serve_utils import post_with_retries

DEFAULT_URL = os.environ.get("PYROKI_SERVICE_URL", "http://127.0.0.1:8116").rstrip("/")


def init_pyroki(
    server_url: str = DEFAULT_URL,
) -> None:
    """
    A *drop-in replacement* for your original init_pyroki(), but instead of
    loading PyRoKi locally, it forwards IK + planning requests to the remote server.

    Downstream code calling ik_solve_fn() or plan_fn() works identically.
    """

    # Normalize trailing slash
    server_url = server_url.rstrip("/")

    # =====================================================
    # IK SOLVER WRAPPER
    # =====================================================
    def ik_solve_fn(
        target_pose_wxyz_xyz: np.ndarray,
        prev_cfg: np.ndarray | None = None,
        obstacles: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Same signature as the local solver, forwarding to /ik over HTTP.

        ``obstacles`` (optional) routes through the server's collision-
        aware IK (``solve_ik_with_collision``). Each entry follows
        the ObstacleEntry schema (``type``: halfspace / sphere /
        capsule / box + the type-specific fields). When omitted/None,
        behavior is unchanged — basic non-collision IK.
        """

        payload: dict[str, Any] = {
            "target_pose_wxyz_xyz": target_pose_wxyz_xyz.tolist(),
            "prev_cfg": prev_cfg.tolist() if prev_cfg is not None else None,
        }
        if obstacles:
            payload["obstacles"] = obstacles

        data = post_with_retries(f"{server_url}/ik", payload, timeout_seconds=15.0)
        joints = np.asarray(data["joint_positions"], dtype=np.float32)

        return joints

    return ik_solve_fn

    # =====================================================
    # PLANNING WRAPPER
    # =====================================================

def init_pyroki_trajopt(
    server_url: str = DEFAULT_URL,
    timeout_seconds: float | None = None,
) -> None:
    """
    A *drop-in replacement* for your original init_pyroki_trajopt(), but instead of
    loading PyRoKi locally, it forwards IK + planning requests to the remote server.
    """

    timeout_s = float(
        timeout_seconds
        if timeout_seconds is not None
        else os.environ.get("PYROKI_TRAJOPT_TIMEOUT_SECONDS", "45.0")
    )

    def trajopt_plan_fn(
        start_pose_wxyz_xyz: np.ndarray,
        end_pose_wxyz_xyz: np.ndarray,
        obstacles: list[dict[str, Any]] | None = None,
        timesteps: int = 20,
        dt: float = 0.02,
        start_cfg: np.ndarray | None = None,
        end_cfg: np.ndarray | None = None,
    ) -> np.ndarray:
        """Same signature as the local planner, forwarding to /plan over HTTP.

        When ``obstacles`` is non-empty, the server routes through
        ``solve_trajopt`` (full collision-aware trajectory optimization);
        otherwise the legacy linear-IK planner is used.

        ``start_cfg`` / ``end_cfg``: optional 7-DOF joint configs. When
        provided, the server skips its internal IK re-solve and uses these
        directly, preventing the optimizer from drifting to a different
        joint-space branch.
        """
        payload: dict[str, Any] = {
            "start_pose_wxyz_xyz": start_pose_wxyz_xyz.tolist(),
            "end_pose_wxyz_xyz": end_pose_wxyz_xyz.tolist(),
            "timesteps": int(timesteps),
            "dt": float(dt),
        }
        if obstacles:
            payload["obstacles"] = obstacles
        if start_cfg is not None:
            payload["start_cfg"] = np.asarray(start_cfg, dtype=np.float64).tolist()
        if end_cfg is not None:
            payload["end_cfg"] = np.asarray(end_cfg, dtype=np.float64).tolist()
        try:
            data = post_with_retries(
                f"{server_url}/plan",
                payload,
                timeout_seconds=timeout_s,
                max_retries=1,
            )
        except Exception as exc:
            obstacle_count = len(obstacles or [])
            raise RuntimeError(
                "PyRoki trajectory planning failed or timed out "
                f"after {timeout_s:.1f}s "
                f"(obstacle_count={obstacle_count}, timesteps={int(timesteps)}, dt={float(dt)}). "
                "This usually means collision-free trajopt did not return before the per-move "
                "planner timeout; check goto_pose_collision_aware_trajectory_error artifacts."
            ) from exc
        waypoints = np.asarray(data["waypoints"], dtype=np.float32)
        return waypoints

    trajopt_plan_fn.timeout_seconds = timeout_s  # type: ignore[attr-defined]
    return trajopt_plan_fn

    # def plan_fn(
    #     q_start: np.ndarray,
    #     q_goal: np.ndarray,         # ignored — kept only for API-compat
    #     obstacles: list[dict[str, Any]],
    # ) -> dict[str, Any]:
    #     """
    #     Matches your original signature even though the PyRoKi server ignores `q_goal`.

    #     Returns:
    #         { "waypoints": [...], "dt": float }
    #     """

    #     payload = {
    #         "q_start": np.asarray(q_start, dtype=np.float64).tolist(),
    #         "obstacles": obstacles,
    #     }

    #     data = _post_with_retries(f"{server_url}/plan", payload)

    #     return {
    #         "waypoints": data["waypoints"],
    #         "dt": data["dt"],
    #     }

    # # =====================================================
    # # TRAJECTORY EXECUTION (stub)
    # # =====================================================
    # def exec_traj_fn(traj: dict[str, Any]) -> bool:
    #     return True

    # =====================================================
    # Register identical API hooks
    # =====================================================
    # register_ik(ik_solve_fn)
    # register_motion_planner(plan_fn, exec_traj_fn)
