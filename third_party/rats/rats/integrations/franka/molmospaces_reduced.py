from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import pathlib
import time
from typing import Any

import numpy as np
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from rats.envs.base import BaseEnv
from rats.integrations.base_api import ApiBase
from rats.integrations.franka.common import (
    apply_tcp_offset,
    close_gripper as _close_gripper,
    extract_arm_joints,
    open_gripper as _open_gripper,
    solve_ik_with_convergence,
)
from rats.integrations.franka.molmospaces_frames import (
    grasp_site_quat_to_panda_hand_quat,
    validate_robot_base_pose,
    world_pose_to_robot_base_frame,
)
from rats.integrations.franka.perception_mixin import FrankaPerceptionMixin

logger = logging.getLogger(__name__)


def _collect_ik_obstacles(env: BaseEnv) -> list[dict] | None:
    """Pull live scene obstacles for collision-aware IK, in pyroki's frame.

    Walks through ``low_level_env`` to reach the bridge's
    ``describe_scene_obstacles`` accessor, excludes the anchored task
    target so the planner doesn't refuse to approach the object we
    want to interact with, and returns the obstacle list in the
    pyroki ObstacleEntry schema (halfspace/sphere/capsule/box).

    Pyroki loads the URDF with ``panda_link0`` at its origin, so all
    IK targets and obstacle geometry must be expressed in robot-base
    frame. ``describe_scene_obstacles`` returns positions in MuJoCo
    *world* frame (procthor scenes place the robot 3-10 m from world
    origin). Without conversion, pyroki treats those world-frame
    positions as if they were panda_link0-frame coordinates: obstacles
    end up several meters from the (pyroki) origin, outside the arm's
    reach envelope, so ``solve_ik_with_collision`` silently never
    triggers any avoidance.

    This function therefore translates+rotates each obstacle's
    position from world frame into robot-base frame using the live
    ``robot_base_pose``. Box extents are left untouched (acceptable
    when the base is near-upright; procthor robots have only a small
    yaw, no roll/pitch, so the world-axis-aligned AABB is still a
    good approximation in base frame).

    Returns ``None`` if anything along the chain is missing or raises
    so the caller can transparently fall back to non-collision IK.
    """
    low_level = getattr(env, "low_level_env", env)
    get_obstacles = getattr(low_level, "describe_scene_obstacles", None)
    if not callable(get_obstacles):
        return None
    excluded: list[str] = []
    anchor_fn = getattr(low_level, "get_anchored_task_target", None)
    if callable(anchor_fn):
        try:
            anchor = anchor_fn() or {}
        except Exception:
            anchor = {}
        for k in ("pickup_obj_name", "place_receptacle_name", "joint_name"):
            v = anchor.get(k)
            if v:
                excluded.append(str(v))
    try:
        obstacles = get_obstacles(
            exclude_internal_names=excluded or None,
            max_distance_m=3.0,
        )
    except Exception as exc:
        logger.debug("describe_scene_obstacles raised: %s; using non-collision IK", exc)
        return None
    if not obstacles:
        return None

    # Convert each obstacle's position from world frame to robot-base
    # frame. We read the live robot_base_pose so the conversion tracks
    # the robot wherever the bridge places it in the scene.
    try:
        from rats.integrations.franka.molmospaces_frames import (
            validate_robot_base_pose,
            _pose7d_to_matrix_wxyz,
        )
    except Exception:
        # If the helper imports fail, returning world-frame obstacles is
        # still safer than crashing: pyroki will simply not avoid them
        # (the historical pre-fix behavior).
        return obstacles or None
    obs_dict = None
    try:
        obs_dict = low_level.get_observation()
    except Exception:
        try:
            obs_dict = env.get_observation()
        except Exception:
            return obstacles or None
    base_pose = obs_dict.get("robot_base_pose") if isinstance(obs_dict, dict) else None
    if base_pose is None:
        return obstacles or None
    try:
        base_pose = validate_robot_base_pose(base_pose)
        world_from_base = _pose7d_to_matrix_wxyz(base_pose)
        base_from_world = np.linalg.inv(world_from_base)
    except Exception:
        return obstacles or None

    def _to_base(point: list[float] | np.ndarray) -> list[float]:
        p_world = np.asarray(point, dtype=np.float64).reshape(3)
        p_h = np.append(p_world, 1.0)
        return (base_from_world @ p_h)[:3].tolist()

    R_base_from_world = base_from_world[:3, :3]
    converted: list[dict] = []
    for o in obstacles:
        t = o.get("type")
        try:
            if t == "box":
                converted.append({
                    **o,
                    "position": _to_base(o.get("position", [0.0, 0.0, 0.0])),
                })
            elif t == "sphere":
                converted.append({
                    **o,
                    "center": _to_base(o.get("center", [0.0, 0.0, 0.0])),
                })
            elif t == "capsule":
                converted.append({
                    **o,
                    "position": _to_base(o.get("position", [0.0, 0.0, 0.0])),
                })
            elif t == "halfspace":
                normal_world = np.asarray(o.get("normal", [0.0, 0.0, 1.0]), dtype=np.float64).reshape(3)
                normal_base = (R_base_from_world @ normal_world).tolist()
                converted.append({
                    **o,
                    "point": _to_base(o.get("point", [0.0, 0.0, 0.0])),
                    "normal": normal_base,
                })
            else:
                converted.append(o)
        except Exception:
            continue
    return converted or None


class FrankaMolmoSpacesApiReduced(FrankaPerceptionMixin, ApiBase):
    """Reduced non-privileged API for MolmoSpaces.

    This mirrors the structure of the working LIBERO reduced API so policies can
    explicitly decompose perception and motion instead of relying only on
    higher-level helpers such as ``sample_grasp_pose``.
    """

    # _TCP_OFFSET = -0.155 m is inherited from FrankaPerceptionMixin (the
    # geometric distance from panda_hand to the Robotiq grasp_site reported
    # by the franka_droid sim; see the mixin docstring for the derivation).
    _capx_only_reduced_api_message_printed = False

    def __init__(
        self,
        env: BaseEnv,
        use_sam3: bool = True,
        *,
        enable_augmented_helpers: bool = False,
        enable_arm_speed: bool = False,
        enable_filter_noise: bool = False,
        grasp_backend: str = "graspnet",
        enable_collision_aware_motion: bool | None = None,
    ) -> None:
        ApiBase.__init__(self, env)
        self.camera_name = "agentview"
        self.wrist_camera_name = "robot0_eye_in_hand"
        self.output_dir = pathlib.Path(".")
        self._enable_augmented_helpers = bool(enable_augmented_helpers)
        self._enable_arm_speed = bool(enable_arm_speed)
        self._enable_filter_noise = bool(enable_filter_noise)
        if enable_collision_aware_motion is None:
            enable_collision_aware_motion = os.environ.get(
                "MOLMOSPACES_COLLISION_AWARE_MOTION", ""
            ).lower() in ("1", "true", "yes")
        self._enable_collision_aware_motion = bool(enable_collision_aware_motion)
        self._init_perception(use_sam3=use_sam3, grasp_backend=grasp_backend)

        # Optional warmup: keep best-effort so construction never fails when
        # the PyRoKi server is absent in unit tests or lightweight environments.
        try:
            self.ik_solve_fn(
                target_pose_wxyz_xyz=np.array([1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.5]),
                prev_cfg=None,
            )
        except Exception:
            pass

    def functions(self) -> dict[str, Any]:
        fns = {
            "get_observation": self.get_observation,
            "segment_sam3_text_prompt": self.segment_sam3_text_prompt,
            "segment_sam3_point_prompt": self.segment_sam3_point_prompt,
            "point_prompt_molmo": self.point_prompt_molmo,
            "plan_grasp": self.plan_grasp,
            "plan_grasp_from_point_clouds": self.plan_grasp_from_point_clouds,
            "get_oriented_bounding_box_from_3d_points": self.get_oriented_bounding_box_from_3d_points,
            "solve_ik": self.solve_ik,
            "move_to_joints": self.move_to_joints,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "goto_home_joint_position": self.goto_home_joint_position,
            "subsample_point_cloud": self.subsample_point_cloud,
        }
        capx_only = bool(getattr(self._env, "capx_only", False))
        if self._enable_filter_noise and not capx_only:
            fns["filter_noise"] = self.filter_noise
        if self._enable_augmented_helpers and not capx_only:
            fns["get_object_3d_points_and_masks_from_language"] = (
                self.get_object_3d_points_and_masks_from_language
            )
            fns["fuse_object_world_points"] = self.fuse_object_world_points
            fns["search_and_locate_object"] = self.search_and_locate_object
            fns["get_object_pose"] = self.get_object_pose
            fns["sample_grasp_pose"] = self.sample_grasp_pose
        if self._enable_arm_speed and not capx_only:
            fns["set_arm_speed"] = self.set_arm_speed
        if capx_only and (
            self._enable_augmented_helpers or self._enable_arm_speed
        ):
            if not type(self)._capx_only_reduced_api_message_printed:
                print("using cap-x apis only for reduced api set")
                type(self)._capx_only_reduced_api_message_printed = True
        return {name: self._webui_timeline_wrapper(name, fn) for name, fn in fns.items()}

    def _webui_timeline_wrapper(self, name: str, fn: Any) -> Any:
        """Wrap exposed primitives with WebUI timeline logging without changing docs."""
        try:
            signature = inspect.signature(fn)
        except Exception:
            signature = None

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not getattr(self, "_webui_enabled", False):
                return fn(*args, **kwargs)
            self._log_step(
                name,
                f"Calling MolmoSpaces API `{name}`.",
                highlight=False,
            )
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                self._log_step_update(
                    text=f"Raised `{type(exc).__name__}`: {exc}",
                )
                raise
            self._log_step_update(text="Completed.")
            publisher = getattr(self, "_viser_publisher", None)
            if publisher is not None:
                try:
                    publisher.publish_env(self._env, reason=name)
                except Exception as _pub_exc:
                    logger.debug("Viser publish_env after %s raised: %s", name, _pub_exc)
            return result

        if signature is not None:
            wrapped.__signature__ = signature  # type: ignore[attr-defined]
        return wrapped

    def set_arm_speed(
        self,
        speed: str | float = "normal",
        *,
        max_joint_step_rad: float | None = None,
        move_max_steps: int | None = None,
    ) -> dict[str, Any]:
        """RATS-only helper: set arm speed for future motion calls.

        This changes the low-level per-step joint delta used by later
        ``goto_pose(...)``, ``move_to_joints(...)``, and helper-skill motion.
        Use it at the start of a RATS attempt or before contact-rich motions.
        Numeric control is preferred when you know how cautious the motion
        should be:

        - ``set_arm_speed(0.35)``: move at 35% of the YAML speed.
        - ``set_arm_speed(max_joint_step_rad=0.02)``: direct joint-step cap
          in radians per simulator step. Smaller is slower and gentler.
        ``move_max_steps`` is accepted only for backward compatibility and is
        ignored: the YAML move budget remains authoritative so policies cannot
        stretch an attempt into a hidden long retry loop.

        Presets are also available as starting points:

        - ``set_arm_speed("very_slow")``: 25% of configured speed; safest near
          clutter, handles, and table contact.
        - ``set_arm_speed("slow")``: 50% speed; good default when objects are
          easy to push off the table.
        - ``set_arm_speed("normal")``: restore the YAML default speed.

        The setting resets to ``"normal"`` on every environment reset, so each
        RATS retry starts from the configured default unless policy code chooses
        otherwise.

        Returns:
            Dict with active ``speed`` and ``max_joint_step_rad``.
            ``move_max_steps`` reports the pinned YAML budget; any requested
            override is returned as ``ignored_move_max_steps_override``.
        """
        setter = getattr(self._env, "set_policy_motion_speed", None)
        if not callable(setter):
            raise RuntimeError("Current environment does not support policy arm speed control")
        info = setter(
            speed,
            max_joint_step_rad=max_joint_step_rad,
            move_max_steps=move_max_steps,
        )
        self._record_runtime_diagnostic("set_arm_speed", **info)
        return info

    def fuse_object_world_points(
        self,
        object_name: str,
        use_multiview: bool = True,
    ) -> np.ndarray:
        """Multi-view fused object point cloud in world frame.

        Runs Molmo + SAM3 on agentview (and the wrist camera when
        use_multiview=True), converts each segment to world coordinates, and
        fuses them: when the two views overlap (any pair within ~1cm) the union
        is returned; otherwise just the higher-confidence view's points. This
        is the same fusion logic used internally by sample_grasp_pose, exposed
        as a one-call primitive so the policy can do staged planning without
        re-implementing fusion.

        Use this instead of single-camera mask_to_world_points whenever
        possible — agentview alone often loses thin / small objects after
        DBSCAN filtering.

        Args:
            object_name: lowercase object description.
            use_multiview: include the wrist camera. Default True.

        Returns:
            (N, 3) ndarray, world frame. May be empty on segmentation failure.
        """
        result = self.get_object_3d_points_and_masks_from_language(
            object_name, use_multiview=use_multiview,
        )
        pts = result.get("points_3d")
        if pts is None:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64).reshape(-1, 3)

    def search_and_locate_object(
        self,
        object_name: str,
        *,
        max_views: int = 5,
        search_radius: float = 0.08,
        hover_height: float = 0.12,
        seed_position: np.ndarray | None = None,
        use_multiview: bool = True,
    ) -> dict[str, Any]:
        """Actively search for an object with the wrist camera, then return points.

        This is the bounded "finding object" stage for RATS policies.  It first
        tries the static/fused perception path.  If that path only finds a
        single view, or fails entirely, the robot sweeps the wrist camera through
        a small top-down pattern and retries perception after each view.  A
        previously valid static point cloud is kept as a fallback, so search
        motion cannot turn a usable detection into a hard failure.

        Args:
            object_name: Lowercase object description, e.g. ``"ladle"``.
            max_views: Maximum wrist viewpoints to visit after the initial
                perception attempt.  Use 0 to disable motion.
            search_radius: XY offset in meters for the wrist sweep pattern.
            hover_height: Height above a localized/seed object center when a
                target seed is known.
            seed_position: Optional world-frame (3,) point to search around.
                When omitted, the helper uses the initial detection centroid if
                available; otherwise it searches around the current end-effector
                pose.
            use_multiview: Whether each perception retry should use both
                agentview and wrist view.

        Returns:
            dict with ``success``, ``points_3d``/``points`` (N, 3), ``centroid``/
            ``position`` (3,), ``source`` (initial/view_i/fallback_initial),
            ``result`` (raw perception result when successful), and ``attempts``.
        """
        max_views_i = max(0, int(max_views))
        search_radius_f = max(0.0, float(search_radius))
        hover_height_f = max(0.02, float(hover_height))
        attempts: list[dict[str, Any]] = []

        self._record_runtime_diagnostic(
            "object_search_start",
            prompt=object_name,
            max_views=max_views_i,
            search_radius=search_radius_f,
            hover_height=hover_height_f,
            use_multiview=use_multiview,
        )

        def _finite_points(result: dict[str, Any]) -> np.ndarray:
            pts = np.asarray(result.get("points_3d", np.empty((0, 3))), dtype=np.float64)
            if pts.size == 0:
                return np.empty((0, 3), dtype=np.float64)
            pts = pts.reshape(-1, 3)
            return pts[np.isfinite(pts).all(axis=1)]

        def _try_perception(label: str) -> dict[str, Any] | None:
            try:
                result = self.get_object_3d_points_and_masks_from_language(
                    object_name,
                    use_multiview=use_multiview,
                )
                pts = _finite_points(result)
                success = len(pts) > 0
                attempts.append({
                    "view": label,
                    "success": success,
                    "point_count": int(len(pts)),
                    "agentview_score": result.get("agentview_score"),
                    "wrist_score": result.get("wrist_score"),
                })
                if not success:
                    return None
                centroid = np.mean(pts, axis=0)
                return {
                    "success": True,
                    "points_3d": pts,
                    "points": pts,
                    "centroid": centroid,
                    "position": centroid,
                    "source": label,
                    "result": result,
                    "attempts": attempts,
                }
            except Exception as exc:
                attempts.append({"view": label, "success": False, "error": str(exc)})
                return None

        initial = _try_perception("initial")
        if initial is not None:
            raw = initial.get("result", {})
            # If both cameras already contributed, no active search is needed.
            if raw.get("agentview_score") is not None and raw.get("wrist_score") is not None:
                self._record_runtime_diagnostic(
                    "object_search_success",
                    prompt=object_name,
                    source="initial",
                    point_count=int(len(initial["points_3d"])),
                    centroid=initial["centroid"],
                    attempts=attempts,
                )
                return initial

        if max_views_i <= 0:
            if initial is not None:
                initial["source"] = "fallback_initial"
                self._record_runtime_diagnostic(
                    "object_search_success",
                    prompt=object_name,
                    source="fallback_initial",
                    point_count=int(len(initial["points_3d"])),
                    centroid=initial["centroid"],
                    attempts=attempts,
                )
                return initial
            self._record_runtime_diagnostic(
                "object_search_failure",
                prompt=object_name,
                attempts=attempts,
                reason="initial perception failed and max_views=0",
            )
            return {
                "success": False,
                "points_3d": np.empty((0, 3), dtype=np.float64),
                "points": np.empty((0, 3), dtype=np.float64),
                "centroid": None,
                "position": None,
                "source": None,
                "result": None,
                "attempts": attempts,
            }

        if seed_position is not None:
            center = np.asarray(seed_position, dtype=np.float64).reshape(3)
            use_seed_hover = True
        elif initial is not None:
            center = np.asarray(initial["centroid"], dtype=np.float64).reshape(3)
            use_seed_hover = True
        else:
            obs = self.get_observation()
            ee = np.asarray(obs.get("robot_cartesian_pos", np.zeros(8)), dtype=np.float64).reshape(-1)
            if ee.size >= 3 and np.isfinite(ee[:3]).all():
                center = ee[:3].copy()
            else:
                center = np.array([0.35, 0.0, 0.35], dtype=np.float64)
            use_seed_hover = False

        offsets = [
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([search_radius_f, 0.0, 0.0], dtype=np.float64),
            np.array([-search_radius_f, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, search_radius_f, 0.0], dtype=np.float64),
            np.array([0.0, -search_radius_f, 0.0], dtype=np.float64),
            np.array([search_radius_f, search_radius_f, 0.0], dtype=np.float64),
            np.array([search_radius_f, -search_radius_f, 0.0], dtype=np.float64),
            np.array([-search_radius_f, search_radius_f, 0.0], dtype=np.float64),
            np.array([-search_radius_f, -search_radius_f, 0.0], dtype=np.float64),
        ]
        topdown_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

        for view_index, offset in enumerate(offsets[:max_views_i]):
            if use_seed_hover:
                view_pos = center + offset + np.array([0.0, 0.0, hover_height_f], dtype=np.float64)
            else:
                view_pos = center + offset
                view_pos[2] = max(float(view_pos[2]), hover_height_f)
            view_pos[2] = float(np.clip(view_pos[2], 0.06, 0.85))
            self._record_runtime_diagnostic(
                "object_search_view",
                prompt=object_name,
                view_index=view_index,
                position=view_pos,
            )
            try:
                self.goto_pose(view_pos, topdown_quat)
            except Exception as exc:
                attempts.append({
                    "view": f"view_{view_index}",
                    "success": False,
                    "move_error": str(exc),
                    "position": view_pos,
                })
                continue

            found = _try_perception(f"view_{view_index}")
            if found is not None:
                self._record_runtime_diagnostic(
                    "object_search_success",
                    prompt=object_name,
                    source=f"view_{view_index}",
                    point_count=int(len(found["points_3d"])),
                    centroid=found["centroid"],
                    attempts=attempts,
                )
                return found

        if initial is not None:
            initial["source"] = "fallback_initial"
            initial["attempts"] = attempts
            self._record_runtime_diagnostic(
                "object_search_success",
                prompt=object_name,
                source="fallback_initial",
                point_count=int(len(initial["points_3d"])),
                centroid=initial["centroid"],
                attempts=attempts,
            )
            return initial

        self._record_runtime_diagnostic(
            "object_search_failure",
            prompt=object_name,
            attempts=attempts,
            reason="all wrist search views failed",
        )
        return {
            "success": False,
            "points_3d": np.empty((0, 3), dtype=np.float64),
            "points": np.empty((0, 3), dtype=np.float64),
            "centroid": None,
            "position": None,
            "source": None,
            "result": None,
            "attempts": attempts,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the current observation from the environment.

        Returns:
            observation:
                A nested dictionary. Access images, intrinsics, and extrinsics
                per camera; there are NO top-level "rgb" / "depth" / "intrinsics" /
                "extrinsics" keys.

                - ["agentview"]["images"]["rgb"]: RGB as ndarray(H, W, 3), uint8.
                - ["agentview"]["images"]["depth"]: depth as ndarray(H, W), float32.
                - ["agentview"]["intrinsics"]: intrinsic matrix ndarray(3, 3), float64.
                - ["agentview"]["pose_mat"]: camera-to-world extrinsic matrix
                  ndarray(4, 4), float64. Use this as `extrinsics` for helpers like
                  mask_to_world_points / pixel_to_world_point.
                - ["robot0_eye_in_hand"]["images"]["rgb"]: wrist RGB ndarray(H, W, 3), uint8.
                - ["robot0_eye_in_hand"]["images"]["depth"]: wrist depth ndarray(H, W), float32.
                - ["robot0_eye_in_hand"]["intrinsics"]: wrist intrinsic matrix ndarray(3, 3), float64.
                - ["robot0_eye_in_hand"]["pose_mat"]: wrist camera-to-world matrix ndarray(4, 4), float64.
                - ["robot_cartesian_pos"]: end-effector pose ndarray(8,), float64.
                  [0:3] XYZ, [3:7] quaternion wxyz, [7] gripper position (0 closed → 1 open).
                - ["robot_joint_pos"]: joint positions ndarray(8,), float64. 7 arm joints + 1 normalized gripper.
                - ["robot_base_pose"]: robot base pose ndarray(7,), float64.
                  [0:3] XYZ, [3:7] quaternion wxyz.

        Example:
            obs = get_observation()
            rgb = obs["agentview"]["images"]["rgb"]
            depth = obs["agentview"]["images"]["depth"]
            K = obs["agentview"]["intrinsics"]
            T_cam_to_world = obs["agentview"]["pose_mat"]
        """
        obs = self._env.get_observation()
        for cam_name in [self.camera_name, self.wrist_camera_name]:
            if cam_name in obs and "images" in obs[cam_name]:
                depth = obs[cam_name]["images"].get("depth")
                if depth is not None and depth.ndim == 3 and depth.shape[-1] == 1:
                    obs[cam_name]["images"]["depth"] = depth.squeeze(-1)
        return obs

    def _motion_snapshot(self) -> dict[str, Any]:
        """Best-effort numeric robot state snapshot for failure diagnosis."""
        try:
            obs = self.get_observation()
        except Exception as exc:
            return {"error": str(exc)}
        if not isinstance(obs, dict):
            return {"error": "observation_unavailable"}
        snapshot: dict[str, Any] = {}
        for key in ("robot_cartesian_pos", "robot_joint_pos", "robot_base_pose"):
            if key in obs:
                snapshot[key] = np.asarray(obs[key], dtype=np.float64)
        for cam_key in (self.camera_name, self.wrist_camera_name):
            cam = obs.get(cam_key)
            if isinstance(cam, dict):
                pose_mat = cam.get("pose_mat")
                if pose_mat is not None:
                    snapshot[f"{cam_key}_pose_mat"] = np.asarray(pose_mat, dtype=np.float64)
        for attr in ("get_video_frame_count",):
            fn = getattr(self._env, attr, None)
            if callable(fn):
                try:
                    snapshot[attr] = int(fn())
                except Exception:
                    pass
        for attr in ("_sim_step_count",):
            if hasattr(self._env, attr):
                try:
                    snapshot[attr] = int(getattr(self._env, attr))
                except Exception:
                    pass
        return snapshot

    def _save_motion_raw(self, stem: str, payload: dict[str, Any]) -> str | None:
        path_fn = getattr(self, "_json_output_path", None)
        if not callable(path_fn):
            return None
        out_path = path_fn(stem)
        if out_path is None:
            return None
        try:
            serializable = self._jsonify_diagnostic_value(payload)
            with open(out_path, "w") as f:
                json.dump(serializable, f, indent=2)
            return str(out_path)
        except Exception as exc:
            logger.warning("Failed to save motion raw artifact %s: %s", stem, exc)
            return None

    @staticmethod
    def _normalize_quaternion_wxyz_for_debug(quaternion_wxyz: np.ndarray) -> np.ndarray:
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-8 or not np.isfinite(norm):
            return quat
        return quat / norm

    @classmethod
    def _pose_frame_debug_payload(
        cls,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> dict[str, Any]:
        """Return a serializable pose frame: position, quaternion, matrix, and signed axes."""
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = cls._normalize_quaternion_wxyz_for_debug(quaternion_wxyz)
        quat_norm = float(np.linalg.norm(quat))
        quat_valid = bool(quat_norm > 1e-8 and np.isfinite(quat_norm))
        if quat_valid:
            rotmat = SciRotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        else:
            rotmat = np.eye(3, dtype=np.float64)
        axes = {
            "x_plus": rotmat[:, 0],
            "x_minus": -rotmat[:, 0],
            "y_plus": rotmat[:, 1],
            "y_minus": -rotmat[:, 1],
            "z_plus": rotmat[:, 2],
            "z_minus": -rotmat[:, 2],
        }
        return {
            "position": pos,
            "quaternion_wxyz": quat,
            "quaternion_norm": quat_norm,
            "quaternion_valid": quat_valid,
            "rotation_matrix": rotmat,
            "axes": axes,
        }

    def _agentview_debug_context(self) -> dict[str, Any] | None:
        """Capture current agentview image and projection matrices for debug plots."""
        try:
            obs = self.get_observation()
            cam = obs.get(self.camera_name) if isinstance(obs, dict) else None
            if not isinstance(cam, dict):
                return None
            images = cam.get("images")
            if not isinstance(images, dict):
                return None
            return {
                "camera_name": self.camera_name,
                "rgb": np.asarray(images.get("rgb"), dtype=np.uint8)[..., :3].copy(),
                "intrinsics": np.asarray(cam["intrinsics"], dtype=np.float64),
                "extrinsics": np.asarray(cam["pose_mat"], dtype=np.float64),
            }
        except Exception:
            return None

    @staticmethod
    def _quaternion_angle_error_deg_wxyz(
        reference_wxyz: np.ndarray,
        measured_wxyz: np.ndarray,
    ) -> float | None:
        try:
            ref = FrankaMolmoSpacesApiReduced._normalize_quaternion_wxyz_for_debug(
                reference_wxyz
            )
            meas = FrankaMolmoSpacesApiReduced._normalize_quaternion_wxyz_for_debug(
                measured_wxyz
            )
            if not np.isfinite(ref).all() or not np.isfinite(meas).all():
                return None
            if float(np.linalg.norm(ref)) <= 1e-8 or float(np.linalg.norm(meas)) <= 1e-8:
                return None
            ref_rot = SciRotation.from_quat([ref[1], ref[2], ref[3], ref[0]])
            meas_rot = SciRotation.from_quat([meas[1], meas[2], meas[3], meas[0]])
            return float((ref_rot.inv() * meas_rot).magnitude() * 180.0 / np.pi)
        except Exception:
            return None

    @classmethod
    def _ee_delta_from_goto_world(
        cls,
        target_position_world: np.ndarray,
        target_quaternion_wxyz: np.ndarray,
        snapshot: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Compare simulator-reported leaf pose against the commanded world pose."""
        if not isinstance(snapshot, dict):
            return None
        try:
            ee = np.asarray(snapshot.get("robot_cartesian_pos"), dtype=np.float64).reshape(-1)
            if ee.size < 7 or not np.isfinite(ee[:7]).all():
                return None
            target_pos = np.asarray(target_position_world, dtype=np.float64).reshape(3)
            target_quat = np.asarray(target_quaternion_wxyz, dtype=np.float64).reshape(4)
            delta = ee[:3] - target_pos
            payload: dict[str, Any] = {
                "ee_position_world": ee[:3],
                "target_position_world": target_pos,
                "position_error_vector_world": delta,
                "position_error_norm_m": float(np.linalg.norm(delta)),
                "orientation_error_deg": cls._quaternion_angle_error_deg_wxyz(
                    target_quat,
                    ee[3:7],
                ),
            }
            if ee.size >= 8:
                payload["gripper_position"] = float(ee[7])
            return payload
        except Exception:
            return None

    @staticmethod
    def _pose_delta_debug(
        expected_position: np.ndarray,
        expected_quaternion_wxyz: np.ndarray | None,
        measured_position: np.ndarray,
        measured_quaternion_wxyz: np.ndarray | None,
    ) -> dict[str, Any]:
        expected = np.asarray(expected_position, dtype=np.float64).reshape(3)
        measured = np.asarray(measured_position, dtype=np.float64).reshape(3)
        delta = measured - expected
        payload: dict[str, Any] = {
            "expected_position": expected,
            "measured_position": measured,
            "position_error_vector": delta,
            "position_error_norm_m": float(np.linalg.norm(delta)),
        }
        if expected_quaternion_wxyz is not None and measured_quaternion_wxyz is not None:
            payload["orientation_error_deg"] = (
                FrankaMolmoSpacesApiReduced._quaternion_angle_error_deg_wxyz(
                    np.asarray(expected_quaternion_wxyz, dtype=np.float64).reshape(4),
                    np.asarray(measured_quaternion_wxyz, dtype=np.float64).reshape(4),
                )
            )
        return payload

    @staticmethod
    def _robot_base_pose_to_transform(
        robot_base_pose_world: Any,
    ) -> tuple[np.ndarray, np.ndarray, SciRotation] | None:
        try:
            pose = np.asarray(robot_base_pose_world, dtype=np.float64).reshape(7)
            quat = FrankaMolmoSpacesApiReduced._normalize_quaternion_wxyz_for_debug(pose[3:])
            base_rot = SciRotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
            return base_rot.as_matrix(), pose[:3], base_rot
        except Exception:
            return None

    @classmethod
    def _pose_base_to_world_debug(
        cls,
        position_base: np.ndarray,
        quaternion_base_wxyz: np.ndarray,
        robot_base_pose_world: Any,
    ) -> dict[str, Any] | None:
        tf = cls._robot_base_pose_to_transform(robot_base_pose_world)
        if tf is None:
            return None
        base_rot_mat, base_pos, base_rot = tf
        pos_base = np.asarray(position_base, dtype=np.float64).reshape(3)
        quat_base = cls._normalize_quaternion_wxyz_for_debug(quaternion_base_wxyz)
        link_rot = SciRotation.from_quat([quat_base[1], quat_base[2], quat_base[3], quat_base[0]])
        world_rot = base_rot * link_rot
        quat_xyzw = world_rot.as_quat()
        return cls._pose_frame_debug_payload(
            base_rot_mat @ pos_base + base_pos,
            np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64),
        )

    @classmethod
    def _pose_world_to_base_debug(
        cls,
        position_world: np.ndarray,
        quaternion_world_wxyz: np.ndarray,
        robot_base_pose_world: Any,
    ) -> dict[str, Any] | None:
        tf = cls._robot_base_pose_to_transform(robot_base_pose_world)
        if tf is None:
            return None
        base_rot_mat, base_pos, base_rot = tf
        pos_world = np.asarray(position_world, dtype=np.float64).reshape(3)
        quat_world = cls._normalize_quaternion_wxyz_for_debug(quaternion_world_wxyz)
        world_rot = SciRotation.from_quat([quat_world[1], quat_world[2], quat_world[3], quat_world[0]])
        base_frame_rot = base_rot.inv() * world_rot
        quat_xyzw = base_frame_rot.as_quat()
        return cls._pose_frame_debug_payload(
            base_rot_mat.T @ (pos_world - base_pos),
            np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64),
        )

    def _pyroki_hand_target_to_grasp_site_debug(
        self,
        position_base: np.ndarray,
        quaternion_base_wxyz: np.ndarray,
    ) -> dict[str, Any]:
        """Predict the MolmoSpaces grasp_site pose implied by a PyRoki panda_hand target."""
        hand_pos = np.asarray(position_base, dtype=np.float64).reshape(3)
        hand_quat = self._normalize_quaternion_wxyz_for_debug(quaternion_base_wxyz)
        hand_rot = SciRotation.from_quat([hand_quat[1], hand_quat[2], hand_quat[3], hand_quat[0]])
        grasp_rot = hand_rot * SciRotation.from_euler("z", np.pi / 4)
        grasp_quat_xyzw = grasp_rot.as_quat()
        grasp_quat_wxyz = np.array(
            [grasp_quat_xyzw[3], grasp_quat_xyzw[0], grasp_quat_xyzw[1], grasp_quat_xyzw[2]],
            dtype=np.float64,
        )
        return {
            "status": "ok",
            "position": hand_pos - hand_rot.apply(self._TCP_OFFSET),
            "quaternion_wxyz": grasp_quat_wxyz,
            "input_pyroki_hand_position_base": hand_pos,
            "input_pyroki_hand_quaternion_base_wxyz": hand_quat,
            "tcp_offset_local": self._TCP_OFFSET,
        }

    def _save_pose_frame_debug_artifacts(
        self,
        stem: str,
        payload: dict[str, Any],
        *,
        agentview_context: dict[str, Any] | None = None,
    ) -> dict[str, str | None]:
        """Save JSON plus a 3D/agentview pose-frame visualization."""
        raw_json_path = self._save_motion_raw(stem, payload)
        image_path = self._save_pose_frame_debug_plot(
            stem,
            payload,
            agentview_context=agentview_context,
        )
        return {"raw_json_path": raw_json_path, "image_path": image_path}

    def _save_pose_frame_debug_plot(
        self,
        stem: str,
        payload: dict[str, Any],
        *,
        agentview_context: dict[str, Any] | None = None,
    ) -> str | None:
        out_path_fn = getattr(self, "_viz_output_path", None)
        if not callable(out_path_fn):
            return None
        out_path = out_path_fn(stem)
        if out_path is None:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except Exception as exc:
            logger.debug("Pose-frame debug plot disabled: %s", exc)
            return None

        def _robot_base_transform() -> tuple[np.ndarray, np.ndarray] | None:
            try:
                pose = np.asarray(payload["robot_base_pose_world"], dtype=np.float64).reshape(7)
                quat = self._normalize_quaternion_wxyz_for_debug(pose[3:])
                rot = SciRotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                return rot, pose[:3]
            except Exception:
                return None

        base_tf = _robot_base_transform()

        def _pose_entry(name: str, entry: Any) -> tuple[str, np.ndarray, dict[str, np.ndarray]] | None:
            if not isinstance(entry, dict):
                return None
            try:
                pos = np.asarray(entry["position"], dtype=np.float64).reshape(3)
                axes_raw = entry["axes"]
                axes = {
                    key: np.asarray(axes_raw[key], dtype=np.float64).reshape(3)
                    for key in ("x_plus", "x_minus", "y_plus", "y_minus", "z_plus", "z_minus")
                }
                return name, pos, axes
            except Exception:
                return None

        def _base_entry_to_world(
            entry: tuple[str, np.ndarray, dict[str, np.ndarray]] | None,
            name: str,
        ) -> tuple[str, np.ndarray, dict[str, np.ndarray]] | None:
            if entry is None or base_tf is None:
                return None
            _, pos, axes = entry
            rot, trans = base_tf
            return name, rot @ pos + trans, {key: rot @ val for key, val in axes.items()}

        def _world_entry_to_base(
            entry: tuple[str, np.ndarray, dict[str, np.ndarray]] | None,
            name: str,
        ) -> tuple[str, np.ndarray, dict[str, np.ndarray]] | None:
            if entry is None or base_tf is None:
                return None
            _, pos, axes = entry
            rot, trans = base_tf
            inv_rot = rot.T
            return name, inv_rot @ (pos - trans), {key: inv_rot @ val for key, val in axes.items()}

        def _ee_point(label: str, pose: Any) -> tuple[str, np.ndarray] | None:
            try:
                arr = np.asarray(pose, dtype=np.float64).reshape(-1)
                if arr.size >= 3 and np.isfinite(arr[:3]).all():
                    return label, arr[:3]
            except Exception:
                pass
            return None

        def _world_point_to_base(point: tuple[str, np.ndarray] | None) -> tuple[str, np.ndarray] | None:
            if point is None or base_tf is None:
                return None
            name, pos = point
            rot, trans = base_tf
            return f"{name}->base", rot.T @ (pos - trans)

        def _first_attempt_entry(key: str, name: str) -> tuple[str, np.ndarray, dict[str, np.ndarray]] | None:
            for attempt in payload.get("pyroki_attempts", []) or []:
                if isinstance(attempt, dict):
                    entry = _pose_entry(name, attempt.get(key))
                    if entry is not None:
                        return entry
            return None

        goto_world = _pose_entry("goto_pose_world", payload.get("goto_pose_world"))
        target_base = _pose_entry("target_base", payload.get("target_base"))
        target_base_clipped = _pose_entry("target_base_clipped", payload.get("target_base_clipped"))
        pyroki_base = _first_attempt_entry("pyroki_target_base", "pyroki_requested")
        predicted_leaf_base = _first_attempt_entry(
            "predicted_grasp_site_from_pyroki_base",
            "predicted_grasp_site_from_pyroki",
        )
        predicted_leaf_world = _first_attempt_entry(
            "predicted_grasp_site_from_pyroki_world",
            "predicted_grasp_site_from_pyroki->world",
        )
        before_world = _ee_point("ee_before", (payload.get("before") or {}).get("robot_cartesian_pos"))
        after_world = _ee_point("ee_after", (payload.get("after") or {}).get("robot_cartesian_pos"))

        world_entries = [entry for entry in (
            goto_world,
            _base_entry_to_world(target_base, "target_base->world"),
            _base_entry_to_world(target_base_clipped, "target_base_clipped->world"),
            _base_entry_to_world(pyroki_base, "pyroki_requested->world"),
            predicted_leaf_world,
        ) if entry is not None]
        base_entries = [entry for entry in (
            _world_entry_to_base(goto_world, "goto_pose_world->base"),
            target_base,
            target_base_clipped,
            pyroki_base,
            predicted_leaf_base,
        ) if entry is not None]
        world_points = [point for point in (before_world, after_world) if point is not None]
        base_points = [point for point in (
            _world_point_to_base(before_world),
            _world_point_to_base(after_world),
        ) if point is not None]

        point_colors = {
            "goto_pose_world": (255, 96, 32),
            "target_base->world": (50, 160, 255),
            "target_base_clipped->world": (40, 210, 220),
            "pyroki_requested->world": (180, 80, 255),
            "predicted_grasp_site_from_pyroki->world": (255, 210, 30),
            "ee_before": (20, 20, 20),
            "ee_after": (255, 40, 220),
        }

        def _agentview_target_overlay() -> np.ndarray | None:
            if not isinstance(agentview_context, dict):
                return None
            try:
                rgb = np.asarray(agentview_context["rgb"], dtype=np.uint8)[..., :3]
                image = Image.fromarray(rgb.copy())
                draw = ImageDraw.Draw(image)
                intrinsics = np.asarray(agentview_context["intrinsics"], dtype=np.float64)
                extrinsics = np.asarray(agentview_context["extrinsics"], dtype=np.float64)
                overlay_items: list[tuple[str, np.ndarray]] = [
                    (name, pos) for name, pos, _axes in world_entries
                ] + world_points
                legend_items: list[tuple[str, tuple[int, int, int]]] = []
                for name, position_world in overlay_items:
                    point_px = self._project_world_to_pixel_best_effort(
                        position_world,
                        intrinsics,
                        extrinsics,
                    )
                    if point_px is None:
                        continue
                    x, y = point_px
                    xi = int(round(float(x)))
                    yi = int(round(float(y)))
                    if xi < 0 or yi < 0 or xi >= image.width or yi >= image.height:
                        continue
                    color = point_colors.get(name, (255, 255, 0))
                    radius = 9 if not name.startswith("ee_") else 7
                    draw.ellipse(
                        [xi - radius, yi - radius, xi + radius, yi + radius],
                        fill=color,
                        outline=(255, 255, 255),
                        width=2,
                    )
                    draw.line([xi - 16, yi, xi + 16, yi], fill=(255, 255, 255), width=1)
                    draw.line([xi, yi - 16, xi, yi + 16], fill=(255, 255, 255), width=1)
                    if (name, color) not in legend_items:
                        legend_items.append((name, color))
                if legend_items:
                    box_h = min(image.height - 8, 16 + 18 * len(legend_items))
                    draw.rectangle([4, 4, min(image.width - 4, 560), box_h], fill=(0, 0, 0))
                    for idx, (name, color) in enumerate(legend_items):
                        y = 12 + 18 * idx
                        draw.rectangle([12, y, 24, y + 10], fill=color, outline=(255, 255, 255))
                        draw.text((30, y - 2), name, fill=(255, 255, 255))
                return np.asarray(image)
            except Exception:
                return None

        try:
            overlay = _agentview_target_overlay()
            if overlay is None:
                fig = plt.figure(figsize=(13, 6.5))
                world_ax = fig.add_subplot(121, projection="3d")
                base_ax = fig.add_subplot(122, projection="3d")
                image_ax = None
            else:
                fig = plt.figure(figsize=(17, 6.5))
                world_ax = fig.add_subplot(131, projection="3d")
                base_ax = fig.add_subplot(132, projection="3d")
                image_ax = fig.add_subplot(133)
            axis_scale = 0.08
            colors = {
                "x_plus": "red",
                "x_minus": "lightcoral",
                "y_plus": "green",
                "y_minus": "lightgreen",
                "z_plus": "blue",
                "z_minus": "lightskyblue",
            }
            axis_handles = [
                Line2D([0], [0], color=colors["x_plus"], lw=2, label="+X axis"),
                Line2D([0], [0], color=colors["x_minus"], lw=2, alpha=0.55, label="-X axis"),
                Line2D([0], [0], color=colors["y_plus"], lw=2, label="+Y axis"),
                Line2D([0], [0], color=colors["y_minus"], lw=2, alpha=0.55, label="-Y axis"),
                Line2D([0], [0], color=colors["z_plus"], lw=2, label="+Z axis"),
                Line2D([0], [0], color=colors["z_minus"], lw=2, alpha=0.55, label="-Z axis"),
            ]

            def _draw_pose_panel(
                ax: Any,
                entries: list[tuple[str, np.ndarray, dict[str, np.ndarray]]],
                points: list[tuple[str, np.ndarray]],
                title: str,
            ) -> bool:
                plotted_points: list[np.ndarray] = []
                for name, pos, axes in entries:
                    origin = pos
                    plotted_points.append(origin)
                    ax.scatter([origin[0]], [origin[1]], [origin[2]], s=36, label=name)
                    ax.text(origin[0], origin[1], origin[2], f" {name}", fontsize=7)
                    for axis_key, direction in axes.items():
                        delta = direction * axis_scale
                        ax.quiver(
                            origin[0],
                            origin[1],
                            origin[2],
                            delta[0],
                            delta[1],
                            delta[2],
                            color=colors[axis_key],
                            linewidth=1.4 if axis_key.endswith("plus") else 0.9,
                            alpha=0.95 if axis_key.endswith("plus") else 0.55,
                        )
                        plotted_points.append(origin + delta)
                for label, point in points:
                    color = "black" if label.startswith("ee_before") else "magenta"
                    plotted_points.append(point)
                    ax.scatter([point[0]], [point[1]], [point[2]], c=color, s=42, marker="^", label=label)
                if not plotted_points:
                    return False
                pts = np.asarray(plotted_points, dtype=np.float64).reshape(-1, 3)
                center = np.mean(pts, axis=0)
                span = float(np.max(np.ptp(pts, axis=0)))
                span = max(span, 0.25)
                half = span * 0.6
                ax.set_xlim(center[0] - half, center[0] + half)
                ax.set_ylim(center[1] - half, center[1] + half)
                ax.set_zlim(center[2] - half, center[2] + half)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_zlabel("")
                ax.set_title(title)
                handles, handle_labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles + axis_handles,
                    handle_labels + [h.get_label() for h in axis_handles],
                    loc="upper right",
                    fontsize=6,
                )
                return True

            drew_world = _draw_pose_panel(world_ax, world_entries, world_points, f"{payload.get('event', stem)} world view")
            drew_base = _draw_pose_panel(base_ax, base_entries, base_points, "robot base view")
            if not drew_world and not drew_base:
                plt.close(fig)
                return None
            if image_ax is not None:
                image_ax.imshow(overlay)
                image_ax.set_axis_off()
                image_ax.set_title("agentview projected world markers")
            plt.tight_layout()
            plt.savefig(out_path, dpi=130)
            plt.close(fig)
            return str(out_path)
        except Exception as exc:
            logger.warning("Failed to save pose-frame debug plot %s: %s", stem, exc)
            return None

    def _push_motion_debug_context(self, **context: Any) -> Any:
        previous = getattr(self._env, "_active_motion_debug_context", None)
        merged = dict(previous or {})
        for key, value in context.items():
            if value is not None:
                merged[key] = value
        try:
            setattr(self._env, "_active_motion_debug_context", merged)
        except Exception:
            pass
        return previous

    def _restore_motion_debug_context(self, previous: Any) -> None:
        try:
            if previous is None:
                setattr(self._env, "_active_motion_debug_context", {})
            else:
                setattr(self._env, "_active_motion_debug_context", previous)
        except Exception:
            pass

    @staticmethod
    def _motion_debug_from_exception(exc: Exception) -> Any:
        return getattr(exc, "motion_debug", None) or getattr(exc, "diagnostics", None)

    def plan_grasp(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan single-view grasp candidates with the configured grasp backend.

        Returns transforms in the CAMERA frame, with +0.1034 m TCP offset
        along the gripper approach axis already applied.
        """
        backend_name = self._grasp_backend_display_name()
        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
            depth_arr = depth_arr[:, :, 0]

        seg_arr = np.asarray(segmentation)
        if seg_arr.ndim == 3 and seg_arr.shape[-1] == 1:
            seg_arr = seg_arr[:, :, 0]
        seg_arr = (seg_arr > 0).astype(np.int32, copy=False)

        try:
            grasp_sample, grasp_scores, _ = self.grasp_net_plan_fn(
                depth_arr,
                np.asarray(intrinsics, dtype=np.float64),
                seg_arr,
                1,
            )
        except Exception as exc:
            service_diagnostics = getattr(self.grasp_net_plan_fn, "last_diagnostics", {})
            log_path = self._append_graspnet_log(
                {
                    "event": "plan_grasp",
                    "backend": backend_name,
                    "depth_shape": list(depth_arr.shape),
                    "segmentation_pixel_count": int(np.count_nonzero(seg_arr)),
                    "service_diagnostics": service_diagnostics,
                    "error": str(exc),
                }
            )
            self._record_runtime_diagnostic(
                "grasp_plan_single_view",
                candidate_count=0,
                backend=backend_name,
                segmentation_pixel_count=int(np.count_nonzero(seg_arr)),
                graspnet_log_path=log_path,
                service_diagnostics=service_diagnostics,
                error=str(exc),
            )
            try:
                from rats.utils.execution_logger import log_step

                log_step(
                    f"{backend_name} Diagnostics",
                    (
                        f"{backend_name} single-view call failed: {exc}. "
                        f"log={log_path or 'not saved'}"
                    ),
                    highlight=True,
                    timeline_kind="perception",
                    timeline_label="graspnet_error",
                )
            except Exception:
                pass
            raise
        service_diagnostics = getattr(self.grasp_net_plan_fn, "last_diagnostics", {})
        scores_arr = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
        candidate_count = int(min(len(grasp_sample), scores_arr.size))

        obs = {}
        try:
            obs = self.get_observation()
        except Exception:
            pass
        camera_name = getattr(self, "camera_name", "agentview")
        cam = obs.get(camera_name) if isinstance(obs, dict) else None
        rgb = None
        extrinsics = None
        if isinstance(cam, dict):
            images = cam.get("images")
            if isinstance(images, dict):
                rgb = images.get("rgb")
            extrinsics = cam.get("pose_mat")

        if candidate_count == 0:
            overlay_path = self._save_grasp_candidate_overlay(
                label="single_view",
                camera_name=camera_name,
                rgb=rgb,
                intrinsics=np.asarray(intrinsics, dtype=np.float64),
                extrinsics=extrinsics,
                grasp_tfs=np.empty((0, 4, 4), dtype=np.float64),
                grasp_scores=np.empty((0,), dtype=np.float64),
                frame="camera",
            )
            log_path = self._append_graspnet_log(
                {
                    "event": "plan_grasp",
                    "backend": backend_name,
                    "candidate_count": 0,
                    "depth_shape": list(depth_arr.shape),
                    "segmentation_pixel_count": int(np.count_nonzero(seg_arr)),
                    "agentview_overlay_path": overlay_path,
                    "service_diagnostics": service_diagnostics,
                    "reason": "no_grasp_candidates",
                }
            )
            self._record_runtime_diagnostic(
                "grasp_plan_single_view",
                candidate_count=0,
                backend=backend_name,
                segmentation_pixel_count=int(np.count_nonzero(seg_arr)),
                agentview_overlay_path=overlay_path,
                graspnet_log_path=log_path,
                service_diagnostics=service_diagnostics,
                reason="no_grasp_candidates",
            )
            try:
                from rats.utils.execution_logger import log_step

                log_step(
                    f"{backend_name} Diagnostics",
                    (
                        f"{backend_name} returned zero single-view candidates. "
                        f"seg_pixels={int(np.count_nonzero(seg_arr))} "
                        f"log={log_path or 'not saved'}"
                    ),
                    highlight=True,
                    timeline_kind="perception",
                    timeline_label="graspnet_zero",
                )
            except Exception:
                pass
            raise AssertionError("No grasp candidates found")

        grasp_sample_tf = (
            vtf.SE3.from_matrix(grasp_sample)
            @ vtf.SE3.from_translation(np.array([0.0, 0.0, 0.1034]))
        ).as_matrix()
        top_order = np.argsort(-scores_arr[:candidate_count])[: min(3, candidate_count)]
        top_scores = [float(scores_arr[i]) for i in top_order]
        overlay_path = self._save_grasp_candidate_overlay(
            label="single_view",
            camera_name=camera_name,
            rgb=rgb,
            intrinsics=np.asarray(intrinsics, dtype=np.float64),
            extrinsics=extrinsics,
            grasp_tfs=grasp_sample_tf,
            grasp_scores=grasp_scores,
            frame="camera",
        )
        log_path = self._append_graspnet_log(
            {
                "event": "plan_grasp",
                "backend": backend_name,
                "candidate_count": candidate_count,
                "local_z_offset_m": 0.1034,
                "depth_shape": list(depth_arr.shape),
                "segmentation_pixel_count": int(np.count_nonzero(seg_arr)),
                "top_scores": top_scores,
                "agentview_overlay_path": overlay_path,
                "service_diagnostics": service_diagnostics,
            }
        )
        self._record_runtime_diagnostic(
            "grasp_plan_single_view",
            candidate_count=candidate_count,
            backend=backend_name,
            segmentation_pixel_count=int(np.count_nonzero(seg_arr)),
            top_scores=top_scores,
            agentview_overlay_path=overlay_path,
            graspnet_log_path=log_path,
            service_diagnostics=service_diagnostics,
        )
        return grasp_sample_tf, grasp_scores

    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Solve IK for the Franka arm with simple orientation fallbacks."""
        pos_world = np.asarray(position, dtype=np.float64).reshape(3)
        quat_world_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        before = self._motion_snapshot()
        agentview_context = self._agentview_debug_context()
        frame_debug: dict[str, Any] = {
            "event": "solve_ik_frame_debug",
            "output_type": "pose_frame_debug",
            "description": (
                "Frame-boundary artifact for solve_ik. goto_pose_world is the "
                "policy/world-frame grasp_site pose; target_base is after "
                "world->robot-base conversion; pyroki_attempts are the exact "
                "base-frame panda_hand targets sent to PyRoki after TCP and "
                "grasp_site->panda_hand orientation alignment."
            ),
            "goto_pose_world": self._pose_frame_debug_payload(pos_world, quat_world_wxyz),
            "before": before,
            "pyroki_attempts": [],
        }
        self._publish_robot_reach_target(
            pos_world,
            quat_world_wxyz,
            label=str(getattr(self, "_solve_ik_target_label", "solve_ik_target")),
        )
        try:
            robot_base_pose = validate_robot_base_pose(
                self.get_observation().get("robot_base_pose"),
            )
            pos, quat_wxyz = world_pose_to_robot_base_frame(
                pos_world,
                quat_world_wxyz,
                robot_base_pose,
            )
            self._record_runtime_diagnostic(
                "solve_ik_frame_transform",
                position_world=pos_world,
                quaternion_world_wxyz=quat_world_wxyz,
                robot_base_pose=robot_base_pose,
                position_base=pos,
                quaternion_base_wxyz=quat_wxyz,
            )
            frame_debug["robot_base_pose_world"] = robot_base_pose
            frame_debug["target_base"] = self._pose_frame_debug_payload(pos, quat_wxyz)
        except Exception as exc:
            frame_debug.update({
                "status": "error",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            })
            self._latest_solve_ik_frame_debug = dict(frame_debug)
            artifacts = self._save_pose_frame_debug_artifacts(
                "solve_ik_frame_transform_error",
                frame_debug,
                agentview_context=agentview_context,
            )
            self._record_runtime_diagnostic(
                "missing_or_invalid_robot_base_pose",
                reason=str(exc),
                robot_base_pose=self.get_observation().get("robot_base_pose"),
                raw_json_path=artifacts.get("raw_json_path"),
                image_path=artifacts.get("image_path"),
                output_type="pose_frame_debug",
            )
            raise ValueError(f"Invalid robot_base_pose for MolmoSpaces solve_ik: {exc}") from exc

        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)
        frame_debug["target_base_clipped"] = self._pose_frame_debug_payload(pos, quat_wxyz)

        # The LLM-facing API interprets quaternion_wxyz as the desired grasp_site
        # orientation (consistent with how robot_cartesian_pos[3:7] is reported),
        # but pyroki IK targets the panda_hand link, which is rotated +45° about
        # local Z from the grasp_site once the franka_droid + Robotiq mounting
        # chain is composed (see molmospaces_frames.grasp_site_quat_to_panda_hand_quat
        # for the derivation). Pre-rotate every IK target — both the user's
        # requested orientation and the canonical fallback orientations
        # (top-down / 45-tilt / side-approach are likewise named from the
        # gripper perspective and should reach grasp_site at those poses) — by
        # Rz(-45°) so the achieved grasp_site lands where the API name promises.
        # Position offset (apply_tcp_offset) is invariant under Rz: TCP_OFFSET
        # is along local +Z, which Rz preserves.
        orientations = [
            ("requested", grasp_site_quat_to_panda_hand_quat(quat_wxyz)),
            ("top-down", grasp_site_quat_to_panda_hand_quat(
                np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64))),
            ("45-tilt", grasp_site_quat_to_panda_hand_quat(
                np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float64))),
            ("side-approach", grasp_site_quat_to_panda_hand_quat(
                np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float64))),
        ]

        # Collision-aware IK. Pull live scene obstacles via the bridge so
        # pyroki's `solve_ik_with_collision` returns a configuration that
        # already avoids self-collision + the world geometry (chairs,
        # tables, drawer fronts, etc.). The bridge-anchored task target
        # is excluded so the planner doesn't refuse to approach the
        # object we actually want to interact with. When the env doesn't
        # expose `describe_scene_obstacles` or the call fails, behavior
        # falls back to the non-collision IK path transparently —
        # `ik_solve_fn` interprets `obstacles=None` as "basic IK".
        obstacles = _collect_ik_obstacles(self._env) if self._enable_collision_aware_motion else None

        for label, quat in orientations:
            off_pos = (
                offset_pos
                if label == "requested"
                else apply_tcp_offset(pos, quat, self._TCP_OFFSET)
            )
            pyroki_quat = self._normalize_quaternion_wxyz_for_debug(quat)
            attempt_debug: dict[str, Any] = {
                "label": label,
                "pyroki_target_frame": "robot_base",
                "pyroki_target_link_name": "panda_hand",
                "pyroki_target_base": self._pose_frame_debug_payload(off_pos, pyroki_quat),
                "target_base_clipped_grasp_site": self._pose_frame_debug_payload(pos, quat_wxyz),
                "target_grasp_site_quaternion_base_wxyz": quat_wxyz,
                "pyroki_quaternion_base_wxyz": pyroki_quat,
                "tcp_offset_local": self._TCP_OFFSET,
                "obstacle_count": len(obstacles or []),
            }
            predicted_grasp_site = self._pyroki_hand_target_to_grasp_site_debug(
                off_pos,
                pyroki_quat,
            )
            attempt_debug["predicted_grasp_site_from_pyroki"] = predicted_grasp_site
            if predicted_grasp_site.get("status") == "ok":
                predicted_base = self._pose_frame_debug_payload(
                    predicted_grasp_site["position"],
                    predicted_grasp_site["quaternion_wxyz"],
                )
                attempt_debug["predicted_grasp_site_from_pyroki_base"] = predicted_base
                predicted_world = self._pose_base_to_world_debug(
                    predicted_base["position"],
                    predicted_base["quaternion_wxyz"],
                    robot_base_pose,
                )
                if predicted_world is not None:
                    attempt_debug["predicted_grasp_site_from_pyroki_world"] = predicted_world
                attempt_debug["predicted_grasp_site_from_pyroki_vs_target_base_clipped"] = (
                    self._pose_delta_debug(
                        pos,
                        quat_wxyz,
                        predicted_base["position"],
                        predicted_base.get("quaternion_wxyz"),
                    )
                )
            frame_debug["pyroki_attempts"].append(attempt_debug)
            self._record_runtime_diagnostic(
                "pyroki_ik_input",
                label=label,
                target_position_base=off_pos,
                target_wxyz_base=pyroki_quat,
                target_grasp_site_position_base=pos,
                target_grasp_site_wxyz_base=quat_wxyz,
                tcp_offset_local=self._TCP_OFFSET,
                obstacle_count=len(obstacles or []),
            )
            try:
                self.cfg = solve_ik_with_convergence(
                    self.ik_solve_fn,
                    pyroki_quat,
                    off_pos,
                    self.cfg,
                    obstacles=obstacles,
                )
                ik_solution = extract_arm_joints(self.cfg)
                attempt_debug["status"] = "ok"
                attempt_debug["ik_solution"] = ik_solution
                frame_debug.update({
                    "status": "ok",
                    "selected_orientation_label": label,
                    "ik_solution": ik_solution,
                })
                self._latest_solve_ik_frame_debug = dict(frame_debug)
                artifacts = self._save_pose_frame_debug_artifacts(
                    "solve_ik_frame_debug",
                    frame_debug,
                    agentview_context=agentview_context,
                )
                self._record_runtime_diagnostic(
                    "solve_ik_frame_debug_artifact",
                    raw_json_path=artifacts.get("raw_json_path"),
                    image_path=artifacts.get("image_path"),
                    selected_orientation_label=label,
                    output_type="pose_frame_debug",
                )
                if label != "requested":
                    logger.info("MolmoSpaces reduced IK solved with fallback orientation: %s", label)
                return ik_solution
            except Exception as exc:
                attempt_debug["status"] = "error"
                attempt_debug["exception_type"] = type(exc).__name__
                attempt_debug["exception_message"] = str(exc)
                logger.warning("MolmoSpaces reduced IK failed with orientation '%s'", label)
                continue

        frame_debug.update({
            "status": "error",
            "exception_type": "RuntimeError",
            "exception_message": f"IK failed for position {pos} with all orientation fallbacks",
        })
        self._latest_solve_ik_frame_debug = dict(frame_debug)
        artifacts = self._save_pose_frame_debug_artifacts(
            "solve_ik_frame_debug_error",
            frame_debug,
            agentview_context=agentview_context,
        )
        self._record_runtime_diagnostic(
            "solve_ik_frame_debug_artifact",
            raw_json_path=artifacts.get("raw_json_path"),
            image_path=artifacts.get("image_path"),
            output_type="pose_frame_debug",
            status="error",
        )
        raise RuntimeError(f"IK failed for position {pos} with all orientation fallbacks")

    def _publish_robot_reach_target(
        self,
        position_world: np.ndarray,
        quaternion_world_wxyz: np.ndarray | None = None,
        *,
        label: str = "reach_target",
    ) -> None:
        """Best-effort WebUI/Viser marker for the commanded reach target."""
        publisher = getattr(self, "_viser_publisher", None)
        published = False
        if publisher is not None:
            try:
                published = bool(
                    publisher.publish_robot_target(
                        position_world,
                        quaternion_wxyz=quaternion_world_wxyz,
                        label=label,
                    )
                )
            except Exception:
                published = False
        camera_overlays = self._save_reach_target_camera_overlays(
            position_world,
            label=label,
        )
        self._record_runtime_diagnostic(
            "robot_reach_target",
            position_world=np.asarray(position_world, dtype=np.float64).reshape(3),
            quaternion_world_wxyz=(
                np.asarray(quaternion_world_wxyz, dtype=np.float64).reshape(4)
                if quaternion_world_wxyz is not None
                else None
            ),
            label=label,
            viser_published=published,
            camera_overlays=camera_overlays,
            output_type="goto_pose_camera_overlay",
            description=(
                "Camera RGB overlay showing the commanded world-frame reach "
                "target projected into available camera views."
            ),
        )

    def _save_reach_target_camera_overlays(
        self,
        position_world: np.ndarray,
        *,
        label: str,
    ) -> dict[str, str]:
        """Save RGB overlays of a commanded reach target in camera views."""
        try:
            pos = np.asarray(position_world, dtype=np.float64).reshape(3)
        except Exception:
            return {}
        try:
            obs = self.get_observation()
        except Exception:
            return {}
        if not isinstance(obs, dict):
            return {}

        saved: dict[str, str] = {}
        for camera_name in (self.camera_name, self.wrist_camera_name):
            cam = obs.get(camera_name)
            if not isinstance(cam, dict):
                continue
            try:
                rgb = np.asarray(cam["images"]["rgb"], dtype=np.uint8)[..., :3]
                intrinsics = np.asarray(cam["intrinsics"], dtype=np.float64)
                extrinsics = np.asarray(cam["pose_mat"], dtype=np.float64)
            except Exception:
                continue
            overlay = self._reach_target_camera_overlay(
                rgb,
                pos,
                intrinsics,
                extrinsics,
                camera_name=camera_name,
                label=label,
            )
            out_path = self._save_rgb_artifact(
                f"{label}_{camera_name}_target_overlay",
                overlay,
            )
            if out_path:
                saved[camera_name] = out_path
        return saved

    def _reach_target_camera_overlay(
        self,
        rgb: np.ndarray,
        position_world: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        *,
        camera_name: str,
        label: str,
    ) -> np.ndarray | None:
        """Draw the projected reach target on a camera RGB image."""
        point_px = self._project_world_to_pixel_best_effort(
            position_world,
            intrinsics,
            extrinsics,
        )
        if point_px is None:
            return None
        base = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
        image = Image.fromarray(base)
        draw = ImageDraw.Draw(image)
        x, y = point_px
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= image.width or yi >= image.height:
            return None
        draw.rectangle(
            [4, 4, min(image.width - 4, 560), min(image.height - 1, 30)],
            fill=(0, 0, 0),
        )
        draw.text(
            (8, 10),
            f"{camera_name}: {label} target",
            fill=(255, 255, 255),
        )
        radius = 13
        draw.ellipse(
            [xi - radius, yi - radius, xi + radius, yi + radius],
            outline=(255, 255, 255),
            width=4,
        )
        draw.ellipse(
            [xi - 7, yi - 7, xi + 7, yi + 7],
            fill=(255, 80, 32),
            outline=(255, 255, 0),
            width=2,
        )
        draw.line([xi - 22, yi, xi + 22, yi], fill=(255, 255, 255), width=2)
        draw.line([xi, yi - 22, xi, yi + 22], fill=(255, 255, 255), width=2)
        draw.text((min(image.width - 1, xi + 16), max(0, yi - 16)), label, fill=(255, 255, 255))
        return np.asarray(image)

    def move_to_joints(self, joints: np.ndarray) -> None:
        """Move to a target joint configuration in a blocking manner."""
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._record_runtime_diagnostic(
            "move_to_joints_target",
            target_joints=target,
        )
        before = self._motion_snapshot()
        t0 = time.time()
        previous_motion_context = self._push_motion_debug_context(
            target_joints=target,
            ik_solution=target,
        )
        try:
            self._env.move_to_joints_blocking(target)
        except Exception as exc:
            after = self._motion_snapshot()
            motion_debug = self._motion_debug_from_exception(exc)
            raw_path = self._save_motion_raw(
                "move_to_joints_error",
                {
                    "event": "move_to_joints_result",
                    "status": "error",
                    "target_joints": target,
                    "before": before,
                    "after_or_error_state": after,
                    "elapsed_s": time.time() - t0,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "motion_debug": motion_debug,
                },
            )
            self._record_runtime_diagnostic(
                "move_to_joints_result",
                status="error",
                target_joints=target,
                before_cartesian_pos=before.get("robot_cartesian_pos"),
                after_cartesian_pos=after.get("robot_cartesian_pos"),
                elapsed_s=time.time() - t0,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                motion_debug=motion_debug,
                raw_json_path=raw_path,
            )
            raise
        finally:
            self._restore_motion_debug_context(previous_motion_context)
        after = self._motion_snapshot()
        raw_path = self._save_motion_raw(
            "move_to_joints_success",
            {
                "event": "move_to_joints_result",
                "status": "success",
                "target_joints": target,
                "before": before,
                "after": after,
                "elapsed_s": time.time() - t0,
            },
        )
        self._record_runtime_diagnostic(
            "move_to_joints_result",
            status="success",
            target_joints=target,
            before_cartesian_pos=before.get("robot_cartesian_pos"),
            after_cartesian_pos=after.get("robot_cartesian_pos"),
            elapsed_s=time.time() - t0,
            raw_json_path=raw_path,
        )

    def _snapshot_grasp_site_pose_to_pyroki_hand_pose(
        self,
        snapshot: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return current panda_hand pose in PyRoki robot-base frame.

        MolmoSpaces reports ``robot_cartesian_pos`` as the gripper
        ``grasp_site`` pose in world frame, while PyRoki plans for the
        ``panda_hand`` link in robot-base frame. This mirrors the same
        frame/TCP conversion used by ``solve_ik`` so trajectory planning
        starts from the pose the simulator is actually reporting.
        """
        ee = np.asarray(snapshot.get("robot_cartesian_pos"), dtype=np.float64).reshape(-1)
        if ee.size < 7:
            raise ValueError("robot_cartesian_pos is missing or shorter than xyz+wxyz")
        robot_base_pose = validate_robot_base_pose(snapshot.get("robot_base_pose"))
        pos_base, quat_base_wxyz = world_pose_to_robot_base_frame(
            ee[:3],
            ee[3:7],
            robot_base_pose,
        )
        hand_quat_wxyz = grasp_site_quat_to_panda_hand_quat(quat_base_wxyz)
        hand_pos_base = apply_tcp_offset(pos_base, quat_base_wxyz, self._TCP_OFFSET)
        return hand_pos_base, hand_quat_wxyz

    @staticmethod
    def _selected_pyroki_target_from_solve_debug(
        solve_frame_debug: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if not isinstance(solve_frame_debug, dict):
            raise ValueError("solve_ik frame debug is unavailable")
        selected_label = str(solve_frame_debug.get("selected_orientation_label") or "")
        attempts = solve_frame_debug.get("pyroki_attempts") or []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            if attempt.get("status") != "ok" and attempt.get("label") != selected_label:
                continue
            pose = attempt.get("pyroki_target_base") or {}
            pos = np.asarray(pose.get("position"), dtype=np.float64).reshape(3)
            quat = np.asarray(pose.get("quaternion_wxyz"), dtype=np.float64).reshape(4)
            return pos, quat, str(attempt.get("label") or selected_label or "selected")
        raise ValueError("selected PyRoki target was not found in solve_ik frame debug")

    @staticmethod
    def _coerce_arm_trajectory(trajectory: np.ndarray) -> np.ndarray:
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2:
            raise ValueError(f"trajectory must be 2-D, got shape {traj.shape}")
        if traj.shape[1] == 7:
            return traj
        if traj.shape[1] > 7:
            return np.vstack([extract_arm_joints(row) for row in traj])
        raise ValueError(f"trajectory waypoints must have at least 7 joints, got shape {traj.shape}")

    def _move_along_collision_aware_trajectory(
        self,
        trajectory: np.ndarray,
        *,
        raw_trajectory: np.ndarray,
        target_pose: np.ndarray,
        target_quaternion_wxyz: np.ndarray,
        ik_solution: np.ndarray,
        obstacle_count: int,
    ) -> dict[str, Any]:
        """Execute PyRoki waypoints with short blocking moves."""
        arm_traj = self._coerce_arm_trajectory(trajectory)
        if len(arm_traj) == 0:
            raise ValueError("trajectory has no waypoints")

        t0 = time.time()
        executed = 0
        previous_motion_context = self._push_motion_debug_context(
            target_pose=target_pose,
            target_quaternion_wxyz=target_quaternion_wxyz,
            ik_solution=ik_solution,
            trajectory_waypoint_count=int(len(arm_traj)),
            obstacle_count=int(obstacle_count),
        )
        try:
            start_index = 1 if len(arm_traj) > 1 else 0
            for waypoint in arm_traj[start_index:]:
                self._env.move_to_joints_blocking(
                    waypoint,
                    tolerance=0.025,
                    max_steps=15,
                )
                executed += 1
        finally:
            self._restore_motion_debug_context(previous_motion_context)

        if raw_trajectory.shape[1] > 7:
            self.cfg = np.asarray(raw_trajectory[-1], dtype=np.float64)
        else:
            self.cfg = np.asarray(arm_traj[-1], dtype=np.float64)

        summary = {
            "status": "ok",
            "waypoint_count": int(len(arm_traj)),
            "executed_waypoint_count": int(executed),
            "obstacle_count": int(obstacle_count),
            "elapsed_s": time.time() - t0,
        }
        raw_path = self._save_motion_raw(
            "goto_pose_collision_aware_trajectory",
            {
                "event": "goto_pose_collision_aware_trajectory",
                **summary,
                "trajectory": arm_traj,
            },
        )
        if raw_path:
            summary["raw_json_path"] = raw_path
        self._record_runtime_diagnostic(
            "goto_pose_collision_aware_trajectory",
            **summary,
        )
        return summary

    def open_gripper(self) -> None:
        """Open the gripper fully."""
        before = self._motion_snapshot()
        t0 = time.time()
        _open_gripper(self._env, steps=30)
        after = self._motion_snapshot()
        raw_path = self._save_motion_raw(
            "open_gripper",
            {
                "event": "gripper_action",
                "action": "open",
                "before": before,
                "after": after,
                "elapsed_s": time.time() - t0,
            },
        )
        self._record_runtime_diagnostic(
            "gripper_action",
            action="open",
            before_cartesian_pos=before.get("robot_cartesian_pos"),
            after_cartesian_pos=after.get("robot_cartesian_pos"),
            elapsed_s=time.time() - t0,
            raw_json_path=raw_path,
        )

    def close_gripper(self) -> None:
        """Close the gripper fully."""
        before = self._motion_snapshot()
        t0 = time.time()
        _close_gripper(self._env, steps=30)
        after = self._motion_snapshot()
        raw_path = self._save_motion_raw(
            "close_gripper",
            {
                "event": "gripper_action",
                "action": "close",
                "before": before,
                "after": after,
                "elapsed_s": time.time() - t0,
            },
        )
        self._record_runtime_diagnostic(
            "gripper_action",
            action="close",
            before_cartesian_pos=before.get("robot_cartesian_pos"),
            after_cartesian_pos=after.get("robot_cartesian_pos"),
            elapsed_s=time.time() - t0,
            raw_json_path=raw_path,
        )

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> None:
        """Go to a WORLD-frame pose via collision-aware trajectory planning.

        Approach/retreat waypoints must be commanded explicitly with
        separate ``goto_pose`` calls. MolmoSpaces intentionally does not
        expose an implicit approach-offset parameter because the old
        implementation offset along world +Z, which was misleading for
        side grasps and articulated handles.

        ``solve_ik`` still selects a reachable panda_hand target and
        fallback orientation, but execution prefers PyRoki's ``/plan``
        endpoint when scene obstacles are available. That endpoint
        optimizes a full joint trajectory with swept robot/world
        collision costs; if no obstacles are available, behavior falls
        back to the historical single ``move_to_joints`` target move.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        before = self._motion_snapshot()
        agentview_context = self._agentview_debug_context()
        frame_debug: dict[str, Any] = {
            "event": "goto_pose_frame_debug",
            "output_type": "goto_pose_pose_frame_debug",
            "description": (
                "Per-movement goto_pose artifact. Includes the policy/world "
                "input pose, signed +X/-X/+Y/-Y/+Z/-Z axis directions, "
                "IK target frames, and before/after simulator leaf snapshots."
            ),
            "goto_pose_world": self._pose_frame_debug_payload(pos, quat),
            "before": before,
        }
        self._record_runtime_diagnostic(
            "goto_pose_target",
            position_world=pos,
            quaternion_world_wxyz=quat,
            axes_world=frame_debug["goto_pose_world"]["axes"],
        )
        goto_raw_path = self._save_motion_raw(
            "goto_pose_target",
            {
                "event": "goto_pose_target",
                "position_world": pos,
                "quaternion_world_wxyz": quat,
                "axes_world": frame_debug["goto_pose_world"]["axes"],
                "before": before,
            },
        )
        if goto_raw_path:
            self._record_runtime_diagnostic(
                "goto_pose_raw_artifact",
                raw_json_path=goto_raw_path,
            )

        previous_motion_context = None
        motion_context_pushed = False
        try:
            self._solve_ik_target_label = "goto_pose_target"
            ik_solution = self.solve_ik(pos, quat)
            solve_frame_debug = getattr(self, "_latest_solve_ik_frame_debug", None)
            if isinstance(solve_frame_debug, dict):
                for key in (
                    "robot_base_pose_world",
                    "target_base",
                    "target_base_clipped",
                    "pyroki_attempts",
                    "selected_orientation_label",
                ):
                    if key in solve_frame_debug:
                        frame_debug[key] = solve_frame_debug[key]
                frame_debug["solve_ik_frame_debug"] = solve_frame_debug
            frame_debug["ik_solution"] = ik_solution
            previous_motion_context = self._push_motion_debug_context(
                target_pose=pos,
                target_quaternion_wxyz=quat,
                ik_solution=ik_solution,
            )
            motion_context_pushed = True
            obstacles = _collect_ik_obstacles(self._env) if self._enable_collision_aware_motion else None
            frame_debug["trajectory_obstacle_count"] = len(obstacles or [])
            if obstacles:
                start_hand_pos, start_hand_quat = self._snapshot_grasp_site_pose_to_pyroki_hand_pose(before)
                end_hand_pos, end_hand_quat, end_label = self._selected_pyroki_target_from_solve_debug(
                    solve_frame_debug if isinstance(solve_frame_debug, dict) else None
                )
                start_pose_wxyz_xyz = np.concatenate([start_hand_quat, start_hand_pos])
                end_pose_wxyz_xyz = np.concatenate([end_hand_quat, end_hand_pos])
                frame_debug["trajectory_planning"] = {
                    "status": "requested",
                    "planner": "pyroki_solve_trajopt",
                    "target_link_name": "panda_hand",
                    "selected_orientation_label": end_label,
                    "start_pyroki_target_base": self._pose_frame_debug_payload(
                        start_hand_pos,
                        start_hand_quat,
                    ),
                    "end_pyroki_target_base": self._pose_frame_debug_payload(
                        end_hand_pos,
                        end_hand_quat,
                    ),
                    "obstacle_count": len(obstacles),
                }
                plan_t0 = time.time()
                plan_timeout_s = getattr(self.trajopt_plan_fn, "timeout_seconds", None)
                if plan_timeout_s is not None:
                    frame_debug["trajectory_planning"]["timeout_seconds"] = float(plan_timeout_s)
                start_joints = None
                if "robot_joint_pos" in before:
                    _sj = np.asarray(before["robot_joint_pos"], dtype=np.float64)
                    if _sj.size >= 7:
                        start_joints = _sj[:7]
                try:
                    raw_traj = np.asarray(
                        self.trajopt_plan_fn(
                            start_pose_wxyz_xyz,
                            end_pose_wxyz_xyz,
                            obstacles=obstacles,
                            timesteps=24,
                            dt=0.04,
                            start_cfg=start_joints,
                            end_cfg=np.asarray(ik_solution, dtype=np.float64),
                        ),
                        dtype=np.float64,
                    )
                except Exception as plan_exc:
                    elapsed_s = time.time() - plan_t0
                    frame_debug["trajectory_planning"].update({
                        "status": "error",
                        "phase": "pyroki_plan",
                        "elapsed_s": elapsed_s,
                        "exception_type": type(plan_exc).__name__,
                        "exception_message": str(plan_exc),
                    })
                    raw_path = self._save_motion_raw(
                        "goto_pose_collision_aware_trajectory_error",
                        {
                            "event": "goto_pose_collision_aware_trajectory_error",
                            "status": "error",
                            "phase": "pyroki_plan",
                            "planner": "pyroki_solve_trajopt",
                            "elapsed_s": elapsed_s,
                            "timeout_seconds": plan_timeout_s,
                            "obstacle_count": len(obstacles),
                            "start_pose_wxyz_xyz": start_pose_wxyz_xyz,
                            "end_pose_wxyz_xyz": end_pose_wxyz_xyz,
                            "selected_orientation_label": end_label,
                            "exception_type": type(plan_exc).__name__,
                            "exception_message": str(plan_exc),
                        },
                    )
                    if raw_path:
                        frame_debug["trajectory_planning"]["raw_json_path"] = raw_path
                    self._record_runtime_diagnostic(
                        "goto_pose_collision_aware_trajectory_error",
                        status="error",
                        phase="pyroki_plan",
                        planner="pyroki_solve_trajopt",
                        elapsed_s=elapsed_s,
                        timeout_seconds=plan_timeout_s,
                        obstacle_count=len(obstacles),
                        selected_orientation_label=end_label,
                        exception_type=type(plan_exc).__name__,
                        exception_message=str(plan_exc),
                        raw_json_path=raw_path,
                    )
                    raise RuntimeError(
                        "Collision-aware PyRoki trajectory planning failed before execution; "
                        "see goto_pose_collision_aware_trajectory_error and "
                        "goto_pose_frame_debug_error artifacts for planner diagnostics."
                    ) from plan_exc
                frame_debug["trajectory_planning"].update({
                    "plan_elapsed_s": time.time() - plan_t0,
                })
                arm_traj = self._coerce_arm_trajectory(raw_traj)
                trajectory_summary = self._move_along_collision_aware_trajectory(
                    arm_traj,
                    raw_trajectory=raw_traj,
                    target_pose=pos,
                    target_quaternion_wxyz=quat,
                    ik_solution=ik_solution,
                    obstacle_count=len(obstacles),
                )
                frame_debug["trajectory_planning"].update({
                    **trajectory_summary,
                    "raw_waypoint_shape": list(raw_traj.shape),
                    "arm_waypoint_shape": list(arm_traj.shape),
                })
            else:
                frame_debug["trajectory_planning"] = {
                    "status": "skipped",
                    "reason": "no_scene_obstacles_available",
                }
                self.move_to_joints(ik_solution)
        except Exception as exc:
            # IK exhaustion raises before move_to_joints runs, so no
            # ``bridge.step`` is called and the frame buffer only holds the
            # initial reset frame. Force a snapshot of the current sim state
            # so the diagnoser video shows where the arm was when IK gave up.
            self._capture_failure_frame()
            after = self._motion_snapshot()
            frame_debug.update({
                "status": "error",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "after": after,
            })
            ee_delta = self._ee_delta_from_goto_world(pos, quat, after)
            if ee_delta is not None:
                frame_debug["ee_after_delta_from_goto_world"] = ee_delta
            artifacts = self._save_pose_frame_debug_artifacts(
                "goto_pose_frame_debug_error",
                frame_debug,
                agentview_context=agentview_context,
            )
            self._record_runtime_diagnostic(
                "goto_pose_frame_debug_artifact",
                raw_json_path=artifacts.get("raw_json_path"),
                image_path=artifacts.get("image_path"),
                output_type="goto_pose_pose_frame_debug",
                status="error",
            )
            self._publish_gripper_markers_from_frame_debug(frame_debug)
            raise
        finally:
            if motion_context_pushed:
                self._restore_motion_debug_context(previous_motion_context)
            try:
                delattr(self, "_solve_ik_target_label")
            except AttributeError:
                pass
        after = self._motion_snapshot()
        frame_debug.update({
            "status": "ok",
            "after": after,
        })
        ee_delta = self._ee_delta_from_goto_world(pos, quat, after)
        if ee_delta is not None:
            frame_debug["ee_after_delta_from_goto_world"] = ee_delta
        artifacts = self._save_pose_frame_debug_artifacts(
            "goto_pose_frame_debug",
            frame_debug,
            agentview_context=agentview_context,
        )
        self._record_runtime_diagnostic(
            "goto_pose_frame_debug_artifact",
            raw_json_path=artifacts.get("raw_json_path"),
            image_path=artifacts.get("image_path"),
            output_type="goto_pose_pose_frame_debug",
            status="ok",
        )
        self._publish_gripper_markers_from_frame_debug(frame_debug)

    def _publish_gripper_markers_from_frame_debug(
        self,
        frame_debug: dict[str, Any],
    ) -> None:
        """Best-effort publish gripper fork markers + standalone artifact."""
        publisher = getattr(self, "_viser_publisher", None)
        if publisher is None:
            return
        try:
            publisher.publish_pose_debug_gripper_markers(
                frame_debug,
                robot_base_pose=frame_debug.get("robot_base_pose_world"),
            )
            out_path_fn = getattr(self, "_viz_output_path", None)
            if callable(out_path_fn):
                out_path = out_path_fn("gripper_markers")
                if out_path is not None:
                    import pathlib
                    out_dir = pathlib.Path(out_path).parent
                    publisher.export_gripper_markers_artifact(
                        out_dir,
                        frame_debug,
                        robot_base_pose=frame_debug.get("robot_base_pose_world"),
                    )
        except Exception:
            pass

    def _capture_failure_frame(self) -> None:
        record_frame = getattr(self._env, "_record_frame", None)
        if record_frame is None or not getattr(self._env, "_record_frames", False):
            return
        try:
            record_frame()
        except Exception:
            logger.debug("Failure-frame capture raised; continuing", exc_info=True)

    def goto_home_joint_position(self) -> None:
        """Return the arm to its reset joint configuration."""
        home = getattr(self._env, "home_joint_position", None)
        if home is None:
            raise RuntimeError("Home joint position is unavailable in the current environment.")
        self._env.move_to_joints_blocking(np.asarray(home, dtype=np.float64).reshape(7))
