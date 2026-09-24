"""Shared perception stack for Franka robot APIs.

Provides Molmo point-prompting, SAM3 segmentation, Contact-GraspNet grasp
planning, and pyroki IK solving. These capabilities are robot-agnostic (they
only need RGB/depth images and camera parameters) and can be reused across
different low-level environments (Libero, MolmoSpaces, real robot, etc.).
"""

from __future__ import annotations

import copy
import json
import logging
import pathlib
import re
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import viser.transforms as vtf
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN

from rats.integrations.vision.graspgen import init_graspgen, init_graspgen_point_clouds
from rats.integrations.vision.graspnet import init_contact_graspnet, init_contact_graspnet_point_clouds
from rats.integrations.vision.molmo import init_molmo
from rats.integrations.vision.sam2 import init_sam2_point_prompt
from rats.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt
from rats.integrations.franka.common import (
    get_oriented_bounding_box_from_3d_points as _get_obb,
)
from rats.utils.depth_utils import depth_to_pointcloud

logger = logging.getLogger(__name__)


class FrankaPerceptionMixin:
    """Mixin providing perception capabilities for Franka robot APIs.

    Subclasses must set ``self._env``, ``self.camera_name``, and
    ``self.wrist_camera_name`` before using any methods from this mixin.

    Note on _TCP_OFFSET: this mixin is currently only used by MolmoSpaces
    APIs (``FrankaMolmoSpacesControlApi``, ``FrankaMolmoSpacesApiReduced``),
    whose simulator runs ``franka_droid`` + Robotiq 2F-85. The
    ``grasp_site`` measured from the franka_droid MJCF is at z=0.155 m in
    the Robotiq base frame (= the panda_hand origin in panda_description,
    which is what pyroki IK targets). The default below matches that. If a
    future subclass mounts a different gripper, it must override
    ``_TCP_OFFSET`` accordingly.
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.155], dtype=np.float64)

    # Hard floors for SAM3 segmentation confidence. Below the
    # text-prompt floor the mask is essentially a phantom: SAM3 latches
    # onto random texture when the queried object is not in view, and
    # the resulting point cloud poisons every downstream waypoint
    # (observed score=0.012 / 0.046 returned a uniformly-distributed
    # 10k-point junk cloud). We accept point-prompt masks at lower
    # scores because Molmo first localized a real pixel — if the
    # prompt landed inside the object the mask is usually fine even
    # when SAM3's confidence is conservative.
    _SAM3_TEXT_PROMPT_SCORE_FLOOR: float = 0.10
    _SAM3_POINT_PROMPT_SCORE_FLOOR: float = 0.05

    def _init_perception(self, use_sam3: bool = True, grasp_backend: str = "graspnet") -> None:
        """Initialize perception models. Call from subclass ``__init__``."""
        self.use_sam3 = use_sam3
        if self.use_sam3:
            self.sam3_seg_fn = init_sam3()
            self.sam3_point_prompt_fn = init_sam3_point_prompt()
        else:
            self.sam2_point_prompt_fn = init_sam2_point_prompt()
        self.molmo_point_fn = init_molmo()
        self.grasp_backend = grasp_backend
        if grasp_backend == "graspgen":
            self.grasp_net_plan_fn = init_graspgen()
            self.grasp_net_plan_point_clouds_fn = init_graspgen_point_clouds()
        elif grasp_backend == "graspnet":
            self.grasp_net_plan_fn = init_contact_graspnet()
            self.grasp_net_plan_point_clouds_fn = init_contact_graspnet_point_clouds()
        else:
            raise ValueError(f"Unsupported grasp backend: {grasp_backend}")

        from rats.integrations.motion.pyroki import init_pyroki, init_pyroki_trajopt
        self.ik_solve_fn = init_pyroki()
        self.trajopt_plan_fn = init_pyroki_trajopt()

        from rats.integrations.motion import pyroki_snippets as pks
        from rats.integrations.motion.pyroki_context import get_pyroki_context

        ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        self._robot = ctx.robot
        self._target_link_name = ctx.target_link_name
        self._pks = pks

        self.cfg = None
        self._curobo_world_config = None
        self.clear_runtime_diagnostics()

    def clear_runtime_diagnostics(self) -> None:
        """Clear step-level perception/runtime diagnostics collected by this API."""
        self._runtime_diagnostics: dict[str, Any] = {
            "events": [],
            "last_error": None,
        }
        self._mask_debug_metadata: dict[int, dict[str, Any]] = {}
        self._molmo_point_debug_metadata: dict[tuple[int, int], dict[str, Any]] = {}
        self._latest_pixel_world_context: dict[str, Any] = {}

    def get_runtime_diagnostics(self) -> dict[str, Any]:
        """Return a deep copy of the current step-level diagnostics."""
        return copy.deepcopy(self._runtime_diagnostics)

    def get_runtime_diagnostics_summary(self) -> str:
        """Return a concise human-readable summary of recent perception events."""
        if not hasattr(self, "_runtime_diagnostics"):
            return ""
        events = self._runtime_diagnostics.get("events", [])
        parts: list[str] = []
        for event in events[-8:]:
            event_name = event.get("event", "")
            if event_name == "molmo_point_prompt":
                prompt = event.get("prompt", "?")
                point = event.get("point")
                parts.append(f"molmo[{prompt}]={point}")
            elif event_name == "camera_segmented_pointcloud":
                cam = event.get("camera", "?")
                strategy = event.get("selected_strategy", "?")
                pts = event.get("selected_point_count", 0)
                score = event.get("selected_score")
                parts.append(
                    f"{cam}:{strategy} pts={pts} score={score} raw={event.get('raw_npz_path')}"
                )
            elif event_name == "language_pointcloud_low_score_rejected":
                cam = event.get("camera", "?")
                strategy = event.get("selected_strategy", "?")
                score = event.get("selected_score")
                floor = event.get("score_floor")
                parts.append(f"{cam}:reject {strategy} score={score}<floor={floor}")
            elif event_name == "language_pointcloud_camera_skipped":
                cam = event.get("camera", "?")
                skip_reason = event.get("skip_reason", "unknown")
                parts.append(f"{cam}:skipped={skip_reason}")
            elif event_name == "language_pointcloud_partial_views":
                valid = event.get("valid_cameras", [])
                skipped = event.get("skipped_cameras", [])
                parts.append(f"partial_views valid={valid} skipped={skipped}")
            elif event_name == "object_search_view":
                parts.append(f"search_view={event.get('view_index')} pos={event.get('position')}")
            elif event_name == "object_search_success":
                parts.append(f"search_success={event.get('source')} pts={event.get('point_count')}")
            elif event_name == "get_object_pose_filtered":
                parts.append(f"pose_pts={event.get('filtered_point_count', 0)}")
            elif event_name == "grasp_plan_pointclouds":
                parts.append(
                    f"grasps={event.get('candidate_count', 0)} "
                    f"best={event.get('best_score')} raw={event.get('raw_npz_path')} "
                    f"overlay={event.get('agentview_overlay_path')}"
                )
            elif event_name == "grasp_plan_single_view":
                parts.append(
                    f"single_view_grasps={event.get('candidate_count', 0)} "
                    f"overlay={event.get('agentview_overlay_path')} "
                    f"log={event.get('graspnet_log_path')}"
                )
            elif event_name == "grasp_selection":
                parts.append(
                    f"{event.get('selector', 'select_grasp')} "
                    f"selected={event.get('selected')} idx={event.get('selected_index')} "
                    f"score={event.get('selected_score')}"
                )
            elif event_name == "goto_pose_target":
                parts.append(f"goto_pose={event.get('position_world')}")
            elif event_name == "move_to_joints_target":
                parts.append("move_to_joints")
            elif event_name == "move_to_joints_result":
                parts.append(
                    f"move_result={event.get('status')} "
                    f"after={event.get('after_cartesian_pos')} raw={event.get('raw_json_path')}"
                )
            elif event_name == "gripper_action":
                parts.append(
                    f"gripper={event.get('action')} "
                    f"after={event.get('after_cartesian_pos')}"
                )
            elif event_name == "sample_grasp_pose_failure":
                parts.append(f"grasp_fail={event.get('reason', 'unknown')}")
            elif event_name == "language_pointcloud_complete":
                parts.append(
                    f"fused_pts={event.get('fused_point_count', 0)} "
                    f"centroid={event.get('fused_centroid_world')}"
                )
            elif event_name == "sample_grasp_pose_success":
                parts.append(
                    f"sample_grasp idx={event.get('best_index')} "
                    f"pos={event.get('grasp_position')}"
                )
        last_error = self._runtime_diagnostics.get("last_error")
        if last_error:
            parts.append(f"error={last_error}")
        return "; ".join(part for part in parts if part)

    def _jsonify_diagnostic_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return self._jsonify_diagnostic_value(value.item())
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [self._jsonify_diagnostic_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._jsonify_diagnostic_value(v) for k, v in value.items()}
        return value

    def _record_runtime_diagnostic(self, event: str, **fields: Any) -> None:
        if not hasattr(self, "_runtime_diagnostics"):
            self.clear_runtime_diagnostics()
        payload = {"event": event, "t": round(time.time(), 3)}
        try:
            from rats.utils.execution_logger import get_current_step_context

            policy_step = get_current_step_context()
        except Exception:
            policy_step = None
        if isinstance(policy_step, dict) and policy_step:
            for key in (
                "policy_step_id",
                "policy_step_index",
                "policy_step_goal",
                "policy_step_marker_index",
                "policy_step_start_frame",
                "policy_step_start_s",
            ):
                if key in policy_step:
                    payload[key] = policy_step.get(key)
        payload.update({k: self._jsonify_diagnostic_value(v) for k, v in fields.items()})
        events = self._runtime_diagnostics.setdefault("events", [])
        events.append(payload)
        if len(events) > 64:
            del events[:-64]
        error = payload.get("error") or payload.get("reason")
        if isinstance(error, str) and error:
            self._runtime_diagnostics["last_error"] = error

    # ------------------------------------------------------------------
    # Debug visualization helpers
    # ------------------------------------------------------------------

    def _viz_output_dir(self) -> pathlib.Path | None:
        out_dir = getattr(self, "output_dir", None)
        if out_dir is None:
            return None
        try:
            viz_dir = pathlib.Path(out_dir) / "pointcloud_viz"
            viz_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        return viz_dir

    def _viz_output_path(self, stem: str) -> pathlib.Path | None:
        viz_dir = self._viz_output_dir()
        if viz_dir is None:
            return None
        if not hasattr(self, "_viz_counter"):
            self._viz_counter = 0
        self._viz_counter += 1
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "viz"
        return viz_dir / f"{self._viz_counter:04d}_{safe_stem}.png"

    def _npz_output_path(self, stem: str) -> pathlib.Path | None:
        viz_dir = self._viz_output_dir()
        if viz_dir is None:
            return None
        counter = getattr(self, "_viz_counter", 0)
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "viz"
        return viz_dir / f"{counter:04d}_{safe_stem}.npz"

    def _json_output_path(self, stem: str) -> pathlib.Path | None:
        viz_dir = self._viz_output_dir()
        if viz_dir is None:
            return None
        if not hasattr(self, "_json_counter"):
            self._json_counter = 0
        self._json_counter += 1
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "raw"
        return viz_dir / f"{self._json_counter:04d}_{safe_stem}.json"

    def _graspnet_log_path(self) -> pathlib.Path | None:
        out_dir = getattr(self, "output_dir", None)
        if out_dir is None:
            return None
        try:
            log_dir = pathlib.Path(out_dir) / "graspnet_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            return log_dir / "graspnet_diagnostics.jsonl"
        except Exception:
            return None

    def _grasp_backend_display_name(self) -> str:
        backend = str(getattr(self, "grasp_backend", "graspnet")).lower()
        if backend == "graspgen":
            return "GraspGen"
        if backend == "graspnet":
            return "Contact-GraspNet"
        return backend or "grasp planner"

    def _append_graspnet_log(self, payload: dict[str, Any]) -> str | None:
        log_path = self._graspnet_log_path()
        if log_path is None:
            return None
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self._jsonify_diagnostic_value(payload),
        }
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            return str(log_path)
        except Exception as exc:
            logger.warning("Failed to append GraspNet diagnostics log: %s", exc)
            return None

    def _save_rgb_artifact(self, stem: str, image: np.ndarray | None) -> str | None:
        """Persist an RGB debug image into the run artifact directory."""
        if image is None:
            return None
        out_path = self._viz_output_path(stem)
        if out_path is None:
            return None
        try:
            arr = np.asarray(image)
            if arr.ndim != 3:
                return None
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr[..., :3]).save(out_path)
            return str(out_path)
        except Exception as exc:
            logger.warning("Failed to save RGB artifact %s: %s", stem, exc)
            return None

    @staticmethod
    def _project_world_to_pixel_best_effort(
        world_point: np.ndarray | list[float] | tuple[float, ...],
        intrinsics: np.ndarray | None,
        extrinsics: np.ndarray | None,
    ) -> tuple[float, float] | None:
        """Project a world-frame point into an image using camera intrinsics/extrinsics."""
        if intrinsics is None or extrinsics is None:
            return None
        try:
            K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
            T = np.asarray(extrinsics, dtype=np.float64).reshape(4, 4)
            wp = np.asarray(world_point, dtype=np.float64).reshape(3)
            p_world = np.append(wp, 1.0)
            p_cam = np.linalg.inv(T) @ p_world
            z = float(p_cam[2])
            if not np.isfinite(z) or abs(z) < 1e-9:
                return None
            u = float(K[0, 0] * p_cam[0] / z + K[0, 2])
            v = float(K[1, 1] * p_cam[1] / z + K[1, 2])
            if np.isfinite([u, v]).all():
                return (u, v)
        except Exception:
            return None
        return None

    @staticmethod
    def _project_world_points_to_pixels_best_effort(
        world_points: np.ndarray | None,
        intrinsics: np.ndarray | None,
        extrinsics: np.ndarray | None,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Project world-frame points into bounded integer image pixels.

        ``extrinsics`` is expected to be camera-to-world, matching
        ``_project_world_to_pixel_best_effort`` and the MolmoSpaces observation
        convention used by ``mask_to_world_points``.
        """
        if world_points is None or intrinsics is None or extrinsics is None:
            return np.empty((0, 2), dtype=np.int32)
        try:
            pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
            if len(pts) == 0:
                return np.empty((0, 2), dtype=np.int32)
            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            if len(pts) == 0:
                return np.empty((0, 2), dtype=np.int32)
            K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
            T = np.asarray(extrinsics, dtype=np.float64).reshape(4, 4)
            pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
            pts_cam = (np.linalg.inv(T) @ pts_h.T).T[:, :3]
            z = pts_cam[:, 2]
            valid_z = np.isfinite(z) & (z > 1e-9)
            pts_cam = pts_cam[valid_z]
            z = z[valid_z]
            if len(pts_cam) == 0:
                return np.empty((0, 2), dtype=np.int32)
            u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
            v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
            h, w = int(image_shape[0]), int(image_shape[1])
            valid_uv = (
                np.isfinite(u)
                & np.isfinite(v)
                & (u >= 0)
                & (u < w)
                & (v >= 0)
                & (v < h)
            )
            if not np.any(valid_uv):
                return np.empty((0, 2), dtype=np.int32)
            px = np.stack([np.rint(u[valid_uv]), np.rint(v[valid_uv])], axis=1).astype(
                np.int32
            )
            px[:, 0] = np.clip(px[:, 0], 0, w - 1)
            px[:, 1] = np.clip(px[:, 1], 0, h - 1)
            return px
        except Exception:
            return np.empty((0, 2), dtype=np.int32)

    def _mask_world_image_overlay(
        self,
        rgb: np.ndarray | None,
        mask: np.ndarray,
        *,
        label: str,
        world_points: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return an image-space overlay that ties a mask to its world-point output."""
        if rgb is None:
            return None
        try:
            base = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
            mask_bool = np.asarray(mask).astype(bool)
            if mask_bool.shape != base.shape[:2]:
                return None
            overlay = base.copy()
            overlay[mask_bool] = np.array([255, 64, 32], dtype=np.uint8)
            blended = (0.55 * base + 0.45 * overlay).astype(np.uint8)
            image = Image.fromarray(blended)
            draw = ImageDraw.Draw(image)
            ys, xs = np.where(mask_bool)
            if len(xs) > 0:
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                cx, cy = float(xs.mean()), float(ys.mean())
                draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)
                draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(0, 255, 255))
                draw.text((max(0, x0), max(0, y0 - 16)), f"{label} mask -> world pts", fill=(255, 255, 255))
            n = int(len(world_points)) if world_points is not None else 0
            draw.rectangle([4, 4, min(image.width - 4, 560), min(image.height - 1, 30)], fill=(0, 0, 0))
            draw.text((8, 10), f"World point cloud source overlay: {label} ({n} pts)", fill=(255, 255, 255))
            return np.asarray(image)
        except Exception:
            return None

    def _pull_direction_overlay_from_context(
        self,
        direction: np.ndarray,
        *,
        label: str = "pull_direction",
        scale_m: float = 0.08,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Draw a projected world-frame direction arrow on the latest relevant camera image."""
        context = getattr(self, "_latest_pixel_world_context", {}) or {}
        rgb = context.get("rgb")
        origin_world = context.get("world_point")
        intrinsics = context.get("intrinsics")
        extrinsics = context.get("extrinsics")
        metadata: dict[str, Any] = {
            "origin_world": origin_world,
            "camera": context.get("camera"),
            "label": context.get("label") or context.get("prompt"),
        }
        if label == "pull_direction" and metadata.get("label"):
            label = str(metadata["label"])
        if rgb is None or origin_world is None or intrinsics is None or extrinsics is None:
            return None, metadata
        try:
            vec = np.asarray(direction, dtype=np.float64).reshape(3)
            if not np.isfinite(vec).all() or np.linalg.norm(vec) < 1e-9:
                return None, metadata
            origin = np.asarray(origin_world, dtype=np.float64).reshape(3)
            tip = origin + vec / np.linalg.norm(vec) * float(scale_m)
            start_px = self._project_world_to_pixel_best_effort(origin, intrinsics, extrinsics)
            tip_px = self._project_world_to_pixel_best_effort(tip, intrinsics, extrinsics)
            metadata.update({"tip_world": tip, "start_pixel": start_px, "tip_pixel": tip_px})
            if start_px is None or tip_px is None:
                return None, metadata
            base = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
            image = Image.fromarray(base)
            draw = ImageDraw.Draw(image)
            sx, sy = start_px
            tx, ty = tip_px
            draw.rectangle([4, 4, min(image.width - 4, 560), min(image.height - 1, 30)], fill=(0, 0, 0))
            draw.text((8, 10), f"{label}: projected world direction", fill=(255, 255, 255))
            draw.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], fill=(255, 64, 32), outline=(255, 255, 255), width=2)
            draw.line([sx, sy, tx, ty], fill=(64, 255, 64), width=5)
            # Simple arrow head in image space.
            angle = np.arctan2(ty - sy, tx - sx)
            # Arrowhead endpoints must be behind the projected tip along the
            # image-space direction.  Using ~pi-sized deltas here flips the
            # head forward past the tip, which makes the arrow look reversed.
            for delta in (0.55, -0.55):
                hx = tx - 16 * np.cos(angle + delta)
                hy = ty - 16 * np.sin(angle + delta)
                draw.line([tx, ty, hx, hy], fill=(64, 255, 64), width=4)
            draw.text((tx + 8, ty + 8), "direction", fill=(255, 255, 255))
            return np.asarray(image), metadata
        except Exception:
            return None, metadata

    def _camera_pointcloud_overlay(
        self,
        rgb: np.ndarray | None,
        pts_full: np.ndarray,
        pts_segment: np.ndarray,
        intrinsics: np.ndarray | None,
        extrinsics: np.ndarray | None,
        *,
        camera_name: str,
        prompt: str,
        score: float | None,
    ) -> np.ndarray | None:
        """Overlay projected world point clouds directly on the camera image."""
        if rgb is None or intrinsics is None or extrinsics is None:
            return None
        try:
            base = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
            h, w = base.shape[:2]
            full = (
                np.asarray(pts_full, dtype=np.float64).reshape(-1, 3)
                if pts_full is not None
                else np.empty((0, 3))
            )
            seg = (
                np.asarray(pts_segment, dtype=np.float64).reshape(-1, 3)
                if pts_segment is not None
                else np.empty((0, 3))
            )
            if len(full) > 12000:
                idx = np.linspace(0, len(full) - 1, 12000, dtype=np.int64)
                full = full[idx]
            if len(seg) > 5000:
                idx = np.linspace(0, len(seg) - 1, 5000, dtype=np.int64)
                seg = seg[idx]
            full_px = self._project_world_points_to_pixels_best_effort(
                full, intrinsics, extrinsics, (h, w)
            )
            seg_px = self._project_world_points_to_pixels_best_effort(
                seg, intrinsics, extrinsics, (h, w)
            )
            if len(full_px) == 0 and len(seg_px) == 0:
                return None

            overlay = base.copy()
            if len(full_px) > 0:
                overlay[full_px[:, 1], full_px[:, 0]] = np.array(
                    [80, 190, 255], dtype=np.uint8
                )
            blended = (0.72 * base + 0.28 * overlay).astype(np.uint8)
            image = Image.fromarray(blended)
            draw = ImageDraw.Draw(image)
            for x, y in seg_px:
                draw.ellipse(
                    [int(x) - 2, int(y) - 2, int(x) + 2, int(y) + 2],
                    fill=(255, 64, 32),
                    outline=(255, 255, 0),
                )
            if len(seg_px) > 0:
                x0, y0 = seg_px.min(axis=0)
                x1, y1 = seg_px.max(axis=0)
                draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=(255, 255, 0), width=2)
                cx, cy = seg_px.mean(axis=0)
                draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(0, 255, 255))
            score_str = f"{float(score):.2f}" if score is not None else "n/a"
            draw.rectangle(
                [4, 4, min(image.width - 4, 620), min(image.height - 1, 32)],
                fill=(0, 0, 0),
            )
            draw.text(
                (8, 10),
                (
                    f"{camera_name} projected point cloud: {prompt} | "
                    f"seg={len(seg_px)} scene={len(full_px)} score={score_str}"
                ),
                fill=(255, 255, 255),
            )
            return np.asarray(image)
        except Exception:
            return None

    def _grasp_candidate_camera_overlay(
        self,
        rgb: np.ndarray | None,
        intrinsics: np.ndarray | None,
        extrinsics: np.ndarray | None,
        grasp_tfs: np.ndarray,
        grasp_scores: np.ndarray,
        *,
        label: str,
        camera_name: str,
        frame: str = "world",
        top_k: int = 3,
        selected_tf: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Draw selected/top grasp candidates as fork-shaped gripper markers on a camera RGB image."""
        if rgb is None or intrinsics is None:
            return None
        try:
            base = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
            image = Image.fromarray(base)
            draw = ImageDraw.Draw(image)
            h, w = base.shape[:2]
            tfs = np.asarray(grasp_tfs, dtype=np.float64).reshape(-1, 4, 4)
            scores_arr = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
            n_candidates = int(min(len(tfs), len(scores_arr)))
            k = int(min(top_k, n_candidates))

            draw.rectangle(
                [4, 4, min(image.width - 4, 760), min(image.height - 1, 34)],
                fill=(0, 0, 0),
            )
            draw.text(
                (8, 11),
                (
                    f"{camera_name} grasp candidates: {label or '-'} | "
                    f"candidates={n_candidates} showing top-{k}"
                ),
                fill=(255, 255, 255),
            )
            if k <= 0:
                draw.text((8, 40), "GraspNet returned zero candidates", fill=(255, 96, 96))
                return np.asarray(image)

            order = np.argsort(-scores_arr[:n_candidates])[:k]
            colors = [
                (255, 64, 192),   # top-1: magenta
                (255, 214, 64),   # top-2: yellow
                (64, 224, 255),   # top-3: cyan
            ]

            finger_length = 0.04
            finger_spread = 0.025
            stem_length = 0.05

            def _project(pt: np.ndarray) -> tuple[float, float] | None:
                if frame == "camera":
                    return self._project_camera_point_to_pixel_best_effort(pt, intrinsics)
                return self._project_world_to_pixel_best_effort(pt, intrinsics, extrinsics)

            for rank, idx in enumerate(order):
                tf = tfs[idx]
                pos = tf[:3, 3]
                local_x = tf[:3, 0]
                local_z = tf[:3, 2]

                stem_tip = pos - local_z * stem_length
                finger_base_left = pos - local_x * finger_spread
                finger_base_right = pos + local_x * finger_spread
                finger_tip_left = finger_base_left + local_z * finger_length
                finger_tip_right = finger_base_right + local_z * finger_length

                segments = [
                    (stem_tip, pos),
                    (pos, finger_base_left),
                    (pos, finger_base_right),
                    (finger_base_left, finger_tip_left),
                    (finger_base_right, finger_tip_right),
                ]

                color = colors[min(rank, len(colors) - 1)]
                width = 4 if rank == 0 else 3

                origin_px = _project(pos)
                if origin_px is None:
                    continue
                ox, oy = origin_px
                if ox < 0 or ox >= w or oy < 0 or oy >= h:
                    continue

                for start_pt, end_pt in segments:
                    start_px = _project(start_pt)
                    end_px = _project(end_pt)
                    if start_px is not None and end_px is not None:
                        draw.line(
                            [start_px[0], start_px[1], end_px[0], end_px[1]],
                            fill=color,
                            width=width,
                        )

                draw.ellipse(
                    [ox - 4, oy - 4, ox + 4, oy + 4],
                    fill=color,
                )
                draw.text(
                    (min(w - 70, int(ox) + 11), max(36, int(oy) - 13)),
                    f"#{rank + 1} {float(scores_arr[idx]):.2f}",
                    fill=color,
                )

            if selected_tf is not None:
                sel = np.asarray(selected_tf, dtype=np.float64).reshape(4, 4)
                sel_pos = sel[:3, 3]
                sel_x = sel[:3, 0]
                sel_z = sel[:3, 2]
                sel_stem_tip = sel_pos - sel_z * stem_length
                sel_fbl = sel_pos - sel_x * finger_spread
                sel_fbr = sel_pos + sel_x * finger_spread
                sel_ftl = sel_fbl + sel_z * finger_length
                sel_ftr = sel_fbr + sel_z * finger_length
                sel_segments = [
                    (sel_stem_tip, sel_pos),
                    (sel_pos, sel_fbl),
                    (sel_pos, sel_fbr),
                    (sel_fbl, sel_ftl),
                    (sel_fbr, sel_ftr),
                ]
                sel_origin_px = _project(sel_pos)
                if sel_origin_px is not None:
                    for start_pt, end_pt in sel_segments:
                        start_px = _project(start_pt)
                        end_px = _project(end_pt)
                        if start_px is not None and end_px is not None:
                            draw.line(
                                [start_px[0], start_px[1], end_px[0], end_px[1]],
                                fill=(255, 255, 255),
                                width=5,
                            )
                    sx, sy = sel_origin_px
                    draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(255, 255, 255))
                    draw.text(
                        (min(w - 90, int(sx) + 11), max(36, int(sy) - 13)),
                        "SELECTED",
                        fill=(255, 255, 255),
                    )

            return np.asarray(image)
        except Exception:
            return None

    @staticmethod
    def _project_camera_point_to_pixel_best_effort(
        camera_point: np.ndarray | list[float] | tuple[float, ...],
        intrinsics: np.ndarray | None,
    ) -> tuple[float, float] | None:
        if intrinsics is None:
            return None
        try:
            K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
            p = np.asarray(camera_point, dtype=np.float64).reshape(3)
            z = float(p[2])
            if not np.isfinite(z) or z <= 1e-9:
                return None
            u = float(K[0, 0] * p[0] / z + K[0, 2])
            v = float(K[1, 1] * p[1] / z + K[1, 2])
            if np.isfinite([u, v]).all():
                return (u, v)
        except Exception:
            return None
        return None

    def _save_grasp_candidate_overlay(
        self,
        *,
        label: str,
        camera_name: str,
        rgb: np.ndarray | None,
        intrinsics: np.ndarray | None,
        extrinsics: np.ndarray | None,
        grasp_tfs: np.ndarray,
        grasp_scores: np.ndarray,
        frame: str,
        selected_tf: np.ndarray | None = None,
    ) -> str | None:
        overlay = self._grasp_candidate_camera_overlay(
            rgb,
            intrinsics,
            extrinsics,
            grasp_tfs,
            grasp_scores,
            label=label,
            camera_name=camera_name,
            frame=frame,
            selected_tf=selected_tf,
        )
        if overlay is None:
            return None
        out_path = self._viz_output_path(f"grasp_candidates_{camera_name}_{label}")
        if out_path is None:
            return None
        try:
            Image.fromarray(overlay[..., :3]).save(out_path)
            logger.info("Saved grasp candidate camera overlay to %s", out_path)
            try:
                from rats.utils.execution_logger import log_step

                log_step(
                    "Grasp Candidate Visualization",
                    (
                        f"{camera_name} overlay for '{label or '-'}'. "
                        "Top-1 is magenta; additional top candidates are yellow/cyan."
                    ),
                    images=[overlay],
                    timeline_kind="perception",
                    timeline_label="grasp_candidates",
                )
            except Exception:
                pass
            return str(out_path)
        except Exception as exc:
            logger.warning("Failed to save grasp candidate overlay: %s", exc)
            return None

    @staticmethod
    def _set_axes_equal_3d(ax) -> None:
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        origin = limits.mean(axis=1)
        half = max(limits[:, 1] - limits[:, 0]) / 2.0
        if half <= 0 or not np.isfinite(half):
            return
        ax.set_xlim3d([origin[0] - half, origin[0] + half])
        ax.set_ylim3d([origin[1] - half, origin[1] + half])
        ax.set_zlim3d([origin[2] - half, origin[2] + half])

    def _save_per_camera_pointcloud_viz(
        self,
        camera_name: str,
        pts_full: np.ndarray,
        pts_segment: np.ndarray,
        prompt: str,
        score: float | None,
        extra_arrays: dict[str, np.ndarray] | None = None,
    ) -> dict[str, str | None]:
        out_path = self._viz_output_path(f"pc_{camera_name}_{prompt}")
        if out_path is None:
            return {"image_path": None, "raw_npz_path": None}
        npz_path = self._npz_output_path(f"pc_{camera_name}_{prompt}")
        npz_path_str = str(npz_path) if npz_path is not None else None
        if npz_path is not None:
            try:
                payload: dict[str, np.ndarray] = {
                    "points_full_world": np.asarray(pts_full, dtype=np.float64),
                    "points_segment_world": np.asarray(pts_segment, dtype=np.float64),
                    "prompt": np.array(prompt),
                    "score": np.array(float(score) if score is not None else np.nan),
                    "camera_name": np.array(camera_name),
                }
                if extra_arrays:
                    for k, v in extra_arrays.items():
                        if v is None:
                            continue
                        payload[k] = np.asarray(v)
                np.savez_compressed(npz_path, **payload)
            except Exception as exc:
                logger.warning("Failed to save point cloud npz for %s: %s", camera_name, exc)
        if extra_arrays:
            camera_overlay = self._camera_pointcloud_overlay(
                extra_arrays.get("rgb"),
                pts_full,
                pts_segment,
                extra_arrays.get("intrinsics"),
                extra_arrays.get("extrinsics_cam_to_world"),
                camera_name=camera_name,
                prompt=prompt,
                score=score,
            )
            if camera_overlay is not None:
                try:
                    Image.fromarray(camera_overlay[..., :3]).save(out_path)
                    logger.info("Saved projected camera point cloud overlay to %s", out_path)
                    return {"image_path": str(out_path), "raw_npz_path": npz_path_str}
                except Exception as exc:
                    logger.warning(
                        "Failed to save projected point cloud overlay for %s: %s",
                        camera_name,
                        exc,
                    )
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning("Point cloud viz disabled (matplotlib unavailable): %s", exc)
            return {"image_path": None, "raw_npz_path": npz_path_str}
        try:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            if pts_full is not None and len(pts_full) > 0:
                finite = np.isfinite(pts_full).all(axis=1)
                pts = pts_full[finite]
                if len(pts) > 8000:
                    pts = pts[np.random.choice(len(pts), 8000, replace=False)]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="lightgray", s=0.5, alpha=0.5, label="scene")
            if pts_segment is not None and len(pts_segment) > 0:
                finite = np.isfinite(pts_segment).all(axis=1)
                seg = pts_segment[finite]
                if len(seg) > 3000:
                    seg = seg[np.random.choice(len(seg), 3000, replace=False)]
                ax.scatter(
                    seg[:, 0], seg[:, 1], seg[:, 2],
                    c="red", s=2.0, label=f"{prompt} (n={len(pts_segment)})",
                )
            if extra_arrays:
                base = extra_arrays.get("robot_base_pose")
                if base is not None and np.asarray(base).size >= 3:
                    bp = np.asarray(base).reshape(-1)[:3]
                    ax.scatter([bp[0]], [bp[1]], [bp[2]], c="green", s=60, marker="s", label="robot_base")
                ee = extra_arrays.get("robot_cartesian_pos")
                if ee is not None and np.asarray(ee).size >= 3:
                    ep = np.asarray(ee).reshape(-1)[:3]
                    ax.scatter([ep[0]], [ep[1]], [ep[2]], c="blue", s=60, marker="^", label="grasp_site")
                cam_ext = extra_arrays.get("extrinsics_cam_to_world")
                if cam_ext is not None and np.asarray(cam_ext).shape == (4, 4):
                    cp = np.asarray(cam_ext)[:3, 3]
                    ax.scatter([cp[0]], [cp[1]], [cp[2]], c="purple", s=60, marker="*", label="camera")
            score_str = f"{float(score):.2f}" if score is not None else "n/a"
            ax.set_title(f"{camera_name}: '{prompt}' score={score_str}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend(loc="upper right", fontsize=8)
            self._set_axes_equal_3d(ax)
            plt.tight_layout()
            plt.savefig(out_path, dpi=120)
            plt.close(fig)
            logger.info("Saved point cloud viz to %s", out_path)
            return {"image_path": str(out_path), "raw_npz_path": npz_path_str}
        except Exception as exc:
            logger.warning("Failed to save point cloud viz for %s: %s", camera_name, exc)
            return {"image_path": None, "raw_npz_path": npz_path_str}

    def _save_grasp_viz(
        self,
        pc_full: np.ndarray,
        grasp_tfs: np.ndarray,
        grasp_scores: np.ndarray,
        label: str = "",
        top_k: int = 10,
        extra_arrays: dict[str, np.ndarray] | None = None,
    ) -> dict[str, str | None]:
        stem = f"grasps_{label}" if label else "grasps"
        out_path = self._viz_output_path(stem)
        if out_path is None:
            return {"image_path": None, "raw_npz_path": None}
        npz_path = self._npz_output_path(stem)
        npz_path_str = str(npz_path) if npz_path is not None else None
        if npz_path is not None:
            try:
                payload: dict[str, np.ndarray] = {
                    "pc_full_world": np.asarray(pc_full, dtype=np.float64),
                    "grasp_tfs_world": np.asarray(grasp_tfs, dtype=np.float64),
                    "grasp_scores": np.asarray(grasp_scores, dtype=np.float64).reshape(-1),
                    "label": np.array(label),
                }
                if extra_arrays:
                    for k, v in extra_arrays.items():
                        payload[k] = np.asarray(v)
                np.savez_compressed(npz_path, **payload)
            except Exception as exc:
                logger.warning("Failed to save grasp npz: %s", exc)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning("Grasp viz disabled (matplotlib unavailable): %s", exc)
            return {"image_path": None, "raw_npz_path": npz_path_str}
        try:
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d")
            if pc_full is not None and len(pc_full) > 0:
                finite = np.isfinite(pc_full).all(axis=1)
                scene = pc_full[finite]
                if len(scene) > 6000:
                    scene = scene[np.random.choice(len(scene), 6000, replace=False)]
                ax.scatter(scene[:, 0], scene[:, 1], scene[:, 2], c="lightgray", s=0.5, alpha=0.4, label="scene")

            if extra_arrays:
                seg = extra_arrays.get("pc_segment_world")
                if seg is not None:
                    seg_arr = np.asarray(seg)
                    if seg_arr.ndim == 2 and seg_arr.shape[1] == 3 and len(seg_arr) > 0:
                        finite = np.isfinite(seg_arr).all(axis=1)
                        seg_arr = seg_arr[finite]
                        if len(seg_arr) > 3000:
                            seg_arr = seg_arr[np.random.choice(len(seg_arr), 3000, replace=False)]
                        ax.scatter(
                            seg_arr[:, 0], seg_arr[:, 1], seg_arr[:, 2],
                            c="red", s=2.0, label=f"segment (n={len(seg)})",
                        )
                base = extra_arrays.get("robot_base_pose")
                if base is not None and np.asarray(base).size >= 3:
                    bp = np.asarray(base).reshape(-1)[:3]
                    ax.scatter([bp[0]], [bp[1]], [bp[2]], c="green", s=60, marker="s", label="robot_base")
                ee = extra_arrays.get("robot_cartesian_pos")
                if ee is not None and np.asarray(ee).size >= 3:
                    ep = np.asarray(ee).reshape(-1)[:3]
                    ax.scatter([ep[0]], [ep[1]], [ep[2]], c="blue", s=60, marker="^", label="grasp_site")

            scores_arr = np.asarray(grasp_scores).reshape(-1)
            n_candidates = int(min(len(grasp_tfs), len(scores_arr)))
            k = int(min(top_k, n_candidates))
            axis_len = 0.03
            if k > 0:
                order = np.argsort(-scores_arr[:n_candidates])[:k]
                for rank, idx in enumerate(order):
                    tf = np.asarray(grasp_tfs[idx], dtype=np.float64)
                    if tf.shape != (4, 4):
                        continue
                    origin = tf[:3, 3]
                    x_axis = tf[:3, 0] * axis_len
                    y_axis = tf[:3, 1] * axis_len
                    z_axis = tf[:3, 2] * axis_len
                    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color="red", linewidth=1.3)
                    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color="green", linewidth=1.3)
                    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color="blue", linewidth=1.3)
                    ax.scatter([origin[0]], [origin[1]], [origin[2]], c="black", s=10)
                    ax.text(
                        origin[0], origin[1], origin[2],
                        f"  {rank + 1}:{float(scores_arr[idx]):.2f}",
                        fontsize=7,
                    )
            ax.set_title(f"Grasp candidates top-{k}{(': ' + label) if label else ''}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend(loc="upper right", fontsize=8)
            self._set_axes_equal_3d(ax)
            plt.tight_layout()
            plt.savefig(out_path, dpi=120)
            plt.close(fig)
            logger.info("Saved grasp viz to %s", out_path)
            return {"image_path": str(out_path), "raw_npz_path": npz_path_str}
        except Exception as exc:
            logger.warning("Failed to save grasp viz: %s", exc)
            return {"image_path": None, "raw_npz_path": npz_path_str}

    def _remember_sam3_masks(
        self,
        *,
        kind: str,
        label: str,
        results: list[dict[str, Any]],
        rgb: np.ndarray | None = None,
        overlay_path: str | None = None,
    ) -> None:
        """Cache mask provenance so mask_to_world_points can label 3D points."""
        if not hasattr(self, "_mask_debug_metadata"):
            self._mask_debug_metadata: dict[int, dict[str, Any]] = {}
        rgb_copy = None
        if rgb is not None:
            try:
                rgb_copy = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
            except Exception:
                rgb_copy = None
        for rank, result in enumerate(results):
            mask = result.get("mask") if isinstance(result, dict) else None
            if mask is None:
                continue
            self._mask_debug_metadata[id(mask)] = {
                "kind": kind,
                "label": label,
                "score": float(result.get("score", 0.0)),
                "rank": rank,
                "mask_pixels": int(np.count_nonzero(mask)),
                "rgb": rgb_copy,
                "segmentation_overlay_path": overlay_path,
            }

    def _label_for_point_prompt(self, point_coords: tuple[float, float]) -> str:
        """Return Molmo text label for a point prompt when known."""
        point_map = getattr(self, "_molmo_point_debug_metadata", {})
        try:
            key = (int(round(float(point_coords[0]))), int(round(float(point_coords[1]))))
        except Exception:
            key = None
        if key is not None and key in point_map:
            metadata = point_map[key]
            if isinstance(metadata, dict):
                return str(metadata.get("label") or metadata.get("prompt") or key)
            return str(metadata)
        return f"point({float(point_coords[0]):.1f}, {float(point_coords[1]):.1f})"

    @staticmethod
    def _molmo_point_overlay(
        rgb: np.ndarray,
        result: dict[str, Any],
        *,
        title: str,
    ) -> np.ndarray | None:
        """Return a 2D RGB overlay showing Molmo's raw point-prompt output."""
        if rgb is None or np.asarray(rgb).ndim != 3:
            return None
        base = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
        image = Image.fromarray(base)
        draw = ImageDraw.Draw(image)
        if image.width >= 12 and image.height >= 16:
            draw.rectangle([4, 4, min(image.width - 4, 560), min(image.height - 1, 28)], fill=(0, 0, 0))
            draw.text((8, 9), title[:95], fill=(255, 255, 255))
        for label, point in (result or {}).items():
            if point is None:
                continue
            try:
                x = int(round(float(point[0])))
                y = int(round(float(point[1])))
            except Exception:
                continue
            if x < 0 or y < 0 or x >= image.width or y >= image.height:
                continue
            outer = 13
            inner = 7
            draw.ellipse(
                [x - outer, y - outer, x + outer, y + outer],
                outline=(255, 255, 255),
                width=4,
            )
            draw.ellipse(
                [x - inner, y - inner, x + inner, y + inner],
                fill=(240, 82, 156),
            )
            draw.line([x - 18, y, x + 18, y], fill=(255, 255, 255), width=2)
            draw.line([x, y - 18, x, y + 18], fill=(255, 255, 255), width=2)
            draw.text((x + 16, max(0, y - 16)), str(label)[:50], fill=(255, 255, 255))
        return np.asarray(image)

    def _camera_context_for_rgb(
        self,
        rgb_query: np.ndarray,
    ) -> tuple[str | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Best-effort match of a point-prompt RGB array to the current cameras."""
        try:
            obs = self.get_observation()
        except Exception:
            return None, None, None, None
        query = np.asarray(rgb_query)
        for cam_name in (getattr(self, "camera_name", "agentview"), getattr(self, "wrist_camera_name", "robot0_eye_in_hand")):
            try:
                cam = obs[cam_name]
                rgb = np.asarray(cam["images"]["rgb"])
            except Exception:
                continue
            if rgb.shape != query.shape:
                continue
            same = rgb is rgb_query
            if not same:
                try:
                    same = bool(np.array_equal(rgb, query))
                except Exception:
                    same = False
            if same:
                return (
                    cam_name,
                    np.asarray(cam["images"]["depth"]),
                    np.asarray(cam["intrinsics"], dtype=np.float64),
                    np.asarray(cam["pose_mat"], dtype=np.float64),
                )
        return None, None, None, None

    @staticmethod
    def _project_pixel_to_world_best_effort(
        point: tuple[Any, Any],
        depth: np.ndarray | None,
        intrinsics: np.ndarray | None,
        extrinsics: np.ndarray | None,
    ) -> np.ndarray | None:
        """Project a Molmo image point to world coordinates when RGBD context exists."""
        if depth is None or intrinsics is None or extrinsics is None:
            return None
        try:
            u = int(round(float(point[0])))
            v = int(round(float(point[1])))
            depth_img = np.asarray(depth, dtype=np.float64)
            if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                depth_img = depth_img[..., 0]
            if depth_img.ndim != 2:
                return None
            h, w = depth_img.shape
            if u < 0 or v < 0 or u >= w or v >= h:
                return None
            z = float(depth_img[v, u])
            if not np.isfinite(z) or z <= 0:
                y0, y1 = max(0, v - 2), min(h, v + 3)
                x0, x1 = max(0, u - 2), min(w, u + 3)
                patch = depth_img[y0:y1, x0:x1]
                valid = patch[np.isfinite(patch) & (patch > 0)]
                if len(valid) == 0:
                    return None
                z = float(np.median(valid))
            K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
            T = np.asarray(extrinsics, dtype=np.float64).reshape(4, 4)
            x_cam = (float(u) - K[0, 2]) * z / K[0, 0]
            y_cam = (float(v) - K[1, 2]) * z / K[1, 1]
            p_cam = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
            world = (T @ p_cam)[:3]
            if np.isfinite(world).all():
                return world
        except Exception:
            return None
        return None

    @staticmethod
    def _sam3_mask_overlay(
        rgb: np.ndarray,
        results: list[dict[str, Any]],
        *,
        title: str,
        max_masks: int = 5,
    ) -> np.ndarray | None:
        """Return an RGB image with SAM3 masks/boxes overlaid for WebUI."""
        if rgb is None or np.asarray(rgb).ndim != 3:
            return None
        if not results:
            return np.asarray(rgb, dtype=np.uint8)
        base = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
        overlay = base.copy()
        colors = np.asarray([
            (255, 64, 64),
            (64, 180, 255),
            (255, 210, 64),
            (120, 255, 120),
            (220, 120, 255),
        ], dtype=np.float32)
        ranked = sorted(
            results,
            key=lambda result: float(result.get("score", 0.0)),
            reverse=True,
        )[:max_masks]
        for idx, result in enumerate(ranked):
            mask = np.asarray(result.get("mask", []))
            if mask.ndim != 2 or mask.shape[:2] != base.shape[:2]:
                continue
            active = mask > 0
            if not np.any(active):
                continue
            color = colors[idx % len(colors)]
            overlay[active] = (
                0.55 * overlay[active].astype(np.float32)
                + 0.45 * color
            ).astype(np.uint8)

        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        draw.rectangle([4, 4, min(image.width - 4, 520), 26], fill=(0, 0, 0))
        draw.text((8, 8), title[:90], fill=(255, 255, 255))
        for idx, result in enumerate(ranked):
            color = tuple(int(c) for c in colors[idx % len(colors)])
            box = result.get("box")
            score = float(result.get("score", 0.0))
            if box is not None and len(box) >= 4:
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1 + 2, max(0, y1 - 12)), f"#{idx+1} {score:.2f}", fill=color)
            else:
                ys, xs = np.where(np.asarray(result.get("mask", [])) > 0)
                if len(xs) > 0:
                    draw.text((float(xs.mean()), float(ys.mean())), f"#{idx+1} {score:.2f}", fill=color)
        return np.asarray(image)

    # ------------------------------------------------------------------
    # Perception primitives
    # ------------------------------------------------------------------

    def point_prompt_molmo(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> dict[str, tuple[int | None, int | None]]:
        """Use Molmo to point to a coordinate in the image based on a text prompt.

        Args:
            image: RGB image array of shape (H, W, 3), dtype uint8.
            text_prompt: The text prompt to point to.

        Returns:
            dict mapping text_prompt to (x, y) pixel coordinates; (None, None) on failure.
        """
        # FIX (self-check/Step-5 doubling): same rationale as
        # verify_object_identity — Molmo is an LLM (vLLM-served VLM on
        # port 8122) and the lifelong loop's Step-4b/Step-5 dual-execute
        # invokes it twice per attempt with bit-identical inputs. The
        # primitive-cache scope (set up around the self-check + exec
        # pair in lifelong_loop) makes the second call return the first
        # call's result without re-querying the VLM.
        try:
            from rats.agents.policy_primitive_cache import (
                lookup as _cache_lookup,
                store as _cache_store,
                make_key as _cache_key,
            )
            _has_cache = True
        except ImportError:
            _has_cache = False
        # origin/main added a uint8 cast to defend against the
        # callers occasionally passing float/bgr-style arrays. Apply
        # that BEFORE hashing so the cache key is stable across
        # equivalent inputs.
        rgb = np.asarray(image, dtype=np.uint8)
        if _has_cache:
            cache_key = _cache_key("point_prompt_molmo", rgb, text_prompt)
            hit, cached = _cache_lookup(cache_key)
            if hit:
                return cached
        result = self.molmo_point_fn(Image.fromarray(rgb), objects=[text_prompt])
        point = result.get(text_prompt, (None, None))
        try:
            valid = bool(point and point[0] is not None and point[1] is not None)
        except Exception:
            valid = False
        camera_name = None
        world_point = None
        overlay = None
        overlay_path = None
        if valid:
            if not hasattr(self, "_molmo_point_debug_metadata"):
                self._molmo_point_debug_metadata: dict[tuple[int, int], dict[str, Any]] = {}
            key = (int(round(float(point[0]))), int(round(float(point[1]))))
            camera_name, depth, intrinsics, extrinsics = self._camera_context_for_rgb(rgb)
            world_point = self._project_pixel_to_world_best_effort(
                point, depth, intrinsics, extrinsics,
            )
            overlay = self._molmo_point_overlay(
                rgb,
                result,
                title=f"Molmo point prompt: {text_prompt}",
            )
            overlay_path = self._save_rgb_artifact(f"molmo_point_{text_prompt}", overlay)
            metadata = {
                "label": str(text_prompt),
                "prompt": str(text_prompt),
                "point": [int(key[0]), int(key[1])],
                "camera": camera_name,
                "world_point": world_point,
                "overlay_path": overlay_path,
                "rgb": rgb.copy(),
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
            }
            self._molmo_point_debug_metadata[key] = metadata
            self._latest_pixel_world_context = metadata
            self._log_step_update(
                text=(
                    f"Molmo point for `{text_prompt}`: {point}"
                    + (f" from `{camera_name}`" if camera_name else "")
                    + (
                        f"; projected_world={np.asarray(world_point).round(4).tolist()}"
                        if world_point is not None else ""
                    )
                ),
                images=overlay,
            )
        else:
            self._log_step_update(text=f"Molmo returned no valid point for `{text_prompt}`.")
        publisher = getattr(self, "_viser_publisher", None)
        viser_published = False
        if publisher is not None and valid:
            try:
                viser_published = bool(
                    publisher.publish_molmo_point(
                        rgb,
                        result,
                        prompt=text_prompt,
                        camera_name=camera_name,
                        world_point=world_point,
                    )
                )
            except Exception:
                viser_published = False
        self._record_runtime_diagnostic(
            "molmo_point_prompt",
            prompt=text_prompt,
            point=point,
            valid=valid,
            camera=camera_name,
            world_point=world_point,
            overlay_path=overlay_path,
            output_type="molmo_point_image_overlay",
            description="RGB image with Molmo's selected point overlaid on the camera image.",
            viser_published=viser_published,
            image_shape=list(rgb.shape),
        )
        if _has_cache:
            _cache_store(cache_key, result)
        return result

    def segment_sam3_point_prompt(
        self,
        rgb: np.ndarray,
        point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation conditioned on a point prompt.

        Args:
            rgb: RGB image array of shape (H, W, 3), dtype uint8.
            point_coords: (x, y) pixel coordinates.

        Returns:
            List of mask dicts with "mask" and "score" keys.
        """
        results = self.sam3_point_prompt_fn(Image.fromarray(rgb), point_coords)
        top = max(results, key=lambda r: r["score"]) if results else None
        label = self._label_for_point_prompt(point_coords)
        vis = self._sam3_mask_overlay(
            rgb,
            results,
            title=(
                f"SAM3 point prompt for {label} "
                f"@ ({float(point_coords[0]):.1f}, {float(point_coords[1]):.1f})"
            ),
        )
        overlay_path = self._save_rgb_artifact(f"sam3_point_{label}", vis)
        self._remember_sam3_masks(
            kind="point",
            label=label,
            results=results,
            rgb=rgb,
            overlay_path=overlay_path,
        )
        self._log_step_update(
            text=(
                f"Returned {len(results)} mask(s); "
                f"top_score={float(top['score']):.3f}; "
                f"top_pixels={int(np.count_nonzero(top['mask']))}"
                if top is not None
                else "SAM3 returned no point-prompt masks."
            ),
            images=vis,
        )
        self._record_runtime_diagnostic(
            "sam3_point_segment",
            point_coords=point_coords,
            num_masks=len(results),
            top_score=float(top["score"]) if top is not None else None,
            top_mask_pixels=int(np.count_nonzero(top["mask"])) if top is not None else 0,
            overlay_path=overlay_path,
            output_type="segmentation_mask_overlay",
            description="RGB image with SAM3 point-prompt segmentation masks overlaid.",
        )
        return results

    def segment_sam3_text_prompt(
        self,
        rgb: np.ndarray,
        text_prompt: str,
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation conditioned on a text prompt.

        Args:
            rgb: RGB image array of shape (H, W, 3), dtype uint8.
            text_prompt: Text prompt for segmentation.

        Returns:
            List of mask dicts with "mask", "box", and "score" keys.
        """
        results = self.sam3_seg_fn(rgb, text_prompt=text_prompt)
        if len(results) == 0:
            print(f"[segment_sam3_text_prompt] SAM3 returned no results for prompt: '{text_prompt}'")
            self._log_step_update(text=f"SAM3 returned no masks for `{text_prompt}`.")
            self._record_runtime_diagnostic(
                "sam3_text_segment",
                prompt=text_prompt,
                num_masks=0,
                top_score=None,
                top_mask_pixels=0,
            )
            return []
        top = max(results, key=lambda r: r["score"])
        vis = self._sam3_mask_overlay(
            rgb,
            results,
            title=f"SAM3 text prompt: {text_prompt}",
        )
        overlay_path = self._save_rgb_artifact(f"sam3_text_{text_prompt}", vis)
        self._remember_sam3_masks(
            kind="text",
            label=str(text_prompt),
            results=results,
            rgb=rgb,
            overlay_path=overlay_path,
        )
        self._log_step_update(
            text=(
                f"Returned {len(results)} mask(s) for `{text_prompt}`; "
                f"top_score={float(top['score']):.3f}; "
                f"top_pixels={int(np.count_nonzero(top['mask']))}"
            ),
            images=vis,
        )
        self._record_runtime_diagnostic(
            "sam3_text_segment",
            prompt=text_prompt,
            num_masks=len(results),
            top_score=float(top["score"]),
            top_mask_pixels=int(np.count_nonzero(top["mask"])),
            overlay_path=overlay_path,
            output_type="segmentation_mask_overlay",
            description="RGB image with SAM3 text-prompt segmentation masks overlaid.",
        )
        return results

    def _segment_object_from_language(
        self, image: Image.Image, object_name: str
    ) -> tuple[np.ndarray | None, tuple[int, int] | None, list[float] | None]:
        """Use SAM3 (or Molmo + SAM2) to return a binary mask for a language-described object."""
        rgb_arr = np.asarray(image)
        if self.use_sam3:
            results = self.segment_sam3_text_prompt(rgb_arr, object_name)
            if len(results) == 0:
                dets = self.point_prompt_molmo(rgb_arr, object_name)
                point = dets.get(object_name)
                if point is None or any(coord is None for coord in point):
                    return None, None, None
                point_coords = (float(point[0]), float(point[1]))
                results = self.segment_sam3_point_prompt(rgb_arr, point_coords=point_coords)

            if len(results) == 0:
                return None, None, None

            scores = [result["score"] for result in results]
            best_result = results[np.argmax(scores)]
            mask_bool = best_result["mask"].astype(bool)
            if not hasattr(self, "_mask_debug_metadata"):
                self._mask_debug_metadata = {}
            best_meta = getattr(self, "_mask_debug_metadata", {}).get(id(best_result.get("mask")), {})
            self._mask_debug_metadata[id(mask_bool)] = {
                **best_meta,
                "kind": best_meta.get("kind", "language"),
                "label": str(object_name),
                "score": float(best_result.get("score", 0.0)),
                "mask_pixels": int(np.count_nonzero(mask_bool)),
                "rgb": rgb_arr.copy(),
            }

            ys, xs = np.where(mask_bool)
            if len(xs) > 0 and len(ys) > 0:
                point_xy = (int(xs.mean()), int(ys.mean()))
            else:
                point_xy = None

            return mask_bool, point_xy, scores
        else:
            dets = self.point_prompt_molmo(rgb_arr, object_name)
            point = dets.get(object_name)
            if point is None or any(coord is None for coord in point):
                return None, None, None
            point_coords = (float(point[0]), float(point[1]))
            scores, masks = self.sam2_point_prompt_fn(image, point_coords=point_coords)
            if len(masks) == 0:
                raise ValueError(f"SAM2 returned no masks for '{object_name}'")

            best_mask = np.asarray(masks[0])
            best_mask = np.squeeze(best_mask)
            if best_mask.ndim != 2:
                raise ValueError(f"SAM2 mask must be 2D, got shape {best_mask.shape}")

            mask_bool = best_mask.astype(bool)
            results = [
                {"mask": np.asarray(mask).squeeze(), "score": float(scores[idx]) if idx < len(scores) else 0.0}
                for idx, mask in enumerate(masks)
            ]
            overlay = self._sam3_mask_overlay(
                rgb_arr,
                results,
                title=f"SAM2 point prompt fallback: {object_name}",
            )
            overlay_path = self._save_rgb_artifact(f"sam2_point_{object_name}", overlay)
            self._remember_sam3_masks(
                kind="sam2_point",
                label=str(object_name),
                results=results,
                rgb=rgb_arr,
                overlay_path=overlay_path,
            )
            self._record_runtime_diagnostic(
                "sam2_point_segment",
                prompt=object_name,
                point_coords=point_coords,
                num_masks=len(results),
                top_score=float(scores[0]) if len(scores) > 0 else None,
                top_mask_pixels=int(np.count_nonzero(mask_bool)),
                overlay_path=overlay_path,
                output_type="segmentation_mask_overlay",
                description="RGB image with SAM2 fallback point-prompt segmentation mask overlaid.",
            )
            if not hasattr(self, "_mask_debug_metadata"):
                self._mask_debug_metadata = {}
            self._mask_debug_metadata[id(mask_bool)] = {
                "kind": "sam2_point",
                "label": str(object_name),
                "score": float(scores[0]) if len(scores) > 0 else 0.0,
                "rank": 0,
                "mask_pixels": int(np.count_nonzero(mask_bool)),
                "rgb": rgb_arr.copy(),
                "segmentation_overlay_path": overlay_path,
            }
            point_xy = (int(round(point_coords[0])), int(round(point_coords[1])))
            return mask_bool, point_xy, scores

    # ------------------------------------------------------------------
    # 3D perception
    # ------------------------------------------------------------------

    def get_object_3d_points_and_masks_from_language(
        self,
        text_prompt: str,
        use_multiview: bool = True,
    ) -> dict[str, Any]:
        """Segment an object using text prompt across one or more views.

        Each camera is evaluated independently.  When ``use_multiview=True`` a
        bad wrist view no longer invalidates a good agentview detection (or vice
        versa): low-confidence / missing / empty per-camera detections are
        skipped, and fusion proceeds with any remaining valid point cloud.  A
        ``ValueError`` is raised only when no requested camera produced usable
        3D points.

        Args:
            text_prompt: Text description of the object to segment.
            use_multiview: If True, also tries the wrist camera.

        Returns:
            dict with agentview_mask, wrist_mask, fused points_3d, per-view
            point clouds, and per-view SAM3 scores.  Missing views are returned
            as ``None`` (mask/score) or an empty ``(0, 3)`` point cloud.
        """
        obs = self.get_observation()

        cameras = [self.camera_name]
        if use_multiview:
            cameras.append(self.wrist_camera_name)
        self._record_runtime_diagnostic(
            "language_pointcloud_start",
            prompt=text_prompt,
            use_multiview=use_multiview,
            cameras=cameras,
        )

        camera_data: dict[str, dict[str, Any]] = {}
        camera_failures: list[dict[str, Any]] = []

        def _skip_camera(cam_name: str, skip_reason: str, **fields: Any) -> None:
            payload = {
                "camera": cam_name,
                "prompt": text_prompt,
                "skip_reason": skip_reason,
            }
            payload.update(fields)
            camera_failures.append(payload)
            self._record_runtime_diagnostic(
                "language_pointcloud_camera_skipped",
                **payload,
            )

        for cam_name in cameras:
            try:
                cam_obs = obs[cam_name]
                rgb = cam_obs["images"]["rgb"]
                depth = cam_obs["images"]["depth"]
                intrinsics = cam_obs["intrinsics"]
                extrinsics = cam_obs["pose_mat"]
            except Exception as exc:
                _skip_camera(cam_name, "missing_camera_observation", exception=str(exc))
                continue

            points = self.point_prompt_molmo(rgb, text_prompt)
            point = points.get(text_prompt, (None, None))
            point_valid = bool(point[0] is not None and point[1] is not None)

            masks = []
            point_mask_candidates = 0
            text_mask_candidates = 0
            selected_strategy = None
            if point_valid:
                masks = self.segment_sam3_point_prompt(rgb, point)
                point_mask_candidates = len(masks)
                if masks:
                    selected_strategy = "point_prompt"
            if not masks:
                masks = self.segment_sam3_text_prompt(rgb, text_prompt)
                text_mask_candidates = len(masks)
                if masks:
                    selected_strategy = "text_prompt"
            if not masks:
                _skip_camera(
                    cam_name,
                    "segmentation_failed",
                    point=point,
                    point_valid=point_valid,
                    point_mask_candidates=point_mask_candidates,
                    text_mask_candidates=text_mask_candidates,
                )
                continue

            mask_data = max(masks, key=lambda x: x["score"])
            mask = mask_data["mask"]
            score = float(mask_data["score"])
            score_floor = (
                self._SAM3_POINT_PROMPT_SCORE_FLOOR
                if selected_strategy == "point_prompt"
                else self._SAM3_TEXT_PROMPT_SCORE_FLOOR
            )
            if score < score_floor:
                self._record_runtime_diagnostic(
                    "language_pointcloud_low_score_rejected",
                    camera=cam_name,
                    prompt=text_prompt,
                    point=point,
                    selected_strategy=selected_strategy,
                    selected_score=score,
                    score_floor=score_floor,
                )
                _skip_camera(
                    cam_name,
                    "low_score",
                    point=point,
                    selected_strategy=selected_strategy,
                    selected_score=score,
                    score_floor=score_floor,
                )
                continue

            pts_camera = depth_to_pointcloud(depth, intrinsics, subsample_factor=1)
            pts_homogeneous = np.concatenate([pts_camera, np.ones((len(pts_camera), 1))], axis=1)
            pts_world = (extrinsics @ pts_homogeneous.T).T[:, :3]

            mask_flat = np.asarray(mask).flatten()
            if len(pts_world) != len(mask_flat):
                min_len = min(len(pts_world), len(mask_flat))
                pts_3d = pts_world[:min_len][mask_flat[:min_len]]
            else:
                pts_3d = pts_world[mask_flat]

            if len(pts_3d) == 0:
                _skip_camera(
                    cam_name,
                    "empty_pointcloud",
                    point=point,
                    selected_strategy=selected_strategy,
                    selected_score=score,
                    raw_point_count=int(len(pts_world)),
                    selected_mask_pixels=int(np.count_nonzero(mask)),
                )
                continue

            object_centroid_world = np.mean(pts_3d, axis=0)
            pointcloud_artifact = self._save_per_camera_pointcloud_viz(
                cam_name, pts_world, pts_3d, text_prompt, float(score),
                extra_arrays={
                    "rgb": np.asarray(rgb),
                    "depth": np.asarray(depth),
                    "intrinsics": np.asarray(intrinsics, dtype=np.float64),
                    "extrinsics_cam_to_world": np.asarray(extrinsics, dtype=np.float64),
                    "mask": np.asarray(mask),
                    "robot_base_pose": np.asarray(
                        obs.get("robot_base_pose", np.zeros(7)), dtype=np.float64
                    ),
                    "robot_cartesian_pos": np.asarray(
                        obs.get("robot_cartesian_pos", np.zeros(8)), dtype=np.float64
                    ),
                },
            )
            if not isinstance(pointcloud_artifact, dict):
                pointcloud_artifact = {}
            pointcloud_viz_path = pointcloud_artifact.get("image_path")
            pointcloud_raw_path = pointcloud_artifact.get("raw_npz_path")
            mask_meta = getattr(self, "_mask_debug_metadata", {}).get(id(mask), {})
            source_overlay_path = mask_meta.get("segmentation_overlay_path")
            image_overlay = self._mask_world_image_overlay(
                rgb,
                mask,
                label=text_prompt,
                world_points=pts_3d,
            )
            image_overlay_path = self._save_rgb_artifact(
                f"language_world_points_overlay_{cam_name}_{text_prompt}",
                image_overlay,
            )
            self._record_runtime_diagnostic(
                "camera_segmented_pointcloud",
                camera=cam_name,
                prompt=text_prompt,
                point=point,
                point_valid=point_valid,
                point_mask_candidates=point_mask_candidates,
                text_mask_candidates=text_mask_candidates,
                selected_strategy=selected_strategy,
                selected_score=float(score),
                selected_mask_pixels=int(np.count_nonzero(mask)),
                raw_point_count=int(len(pts_world)),
                selected_point_count=int(len(pts_3d)),
                object_centroid_world=object_centroid_world,
                image_overlay_path=image_overlay_path,
                segmentation_overlay_path=source_overlay_path,
                visualization_path=pointcloud_viz_path,
                raw_npz_path=pointcloud_raw_path,
                output_type="world_point_cloud_visualization",
                description=(
                    "Per-camera world point cloud from language segmentation, plus "
                    "the actual camera-image overlay showing source pixels."
                ),
            )

            camera_data[cam_name] = {
                "mask": mask,
                "score": score,
                "points_3d": np.asarray(pts_3d, dtype=np.float64).reshape(-1, 3),
                "visualization_path": pointcloud_viz_path,
                "raw_npz_path": pointcloud_raw_path,
                "image_overlay_path": image_overlay_path,
                "segmentation_overlay_path": source_overlay_path,
            }

        if not camera_data:
            reason = (
                f"No valid point cloud for '{text_prompt}' from requested cameras "
                f"{cameras}. Per-camera failures: {camera_failures}"
            )
            self._record_runtime_diagnostic(
                "language_pointcloud_failure",
                prompt=text_prompt,
                cameras=cameras,
                camera_failures=camera_failures,
                reason=reason,
            )
            raise ValueError(reason)

        if camera_failures:
            self._record_runtime_diagnostic(
                "language_pointcloud_partial_views",
                prompt=text_prompt,
                valid_cameras=list(camera_data.keys()),
                skipped_cameras=[failure.get("camera") for failure in camera_failures],
                camera_failures=camera_failures,
            )

        agent_data = camera_data.get(self.camera_name)
        wrist_data = camera_data.get(self.wrist_camera_name) if use_multiview else None

        agent_pts_3d = (
            np.asarray(agent_data["points_3d"], dtype=np.float64).reshape(-1, 3)
            if agent_data is not None
            else np.empty((0, 3), dtype=np.float64)
        )
        wrist_pts_3d = (
            np.asarray(wrist_data["points_3d"], dtype=np.float64).reshape(-1, 3)
            if wrist_data is not None
            else None
        )
        wrist_mask = wrist_data["mask"] if wrist_data is not None else None
        wrist_score = wrist_data["score"] if wrist_data is not None else None

        points_3d: np.ndarray
        fused_source = "agentview"
        if wrist_data is not None and wrist_pts_3d is not None and len(wrist_pts_3d) > 0 and len(agent_pts_3d) > 0:
            distances = np.linalg.norm(
                agent_pts_3d[:, np.newaxis, :] - wrist_pts_3d[np.newaxis, :, :],
                axis=2,
            )
            min_distances = np.min(distances, axis=1)
            threshold = 0.01
            if min_distances.min() < threshold:
                points_3d = np.concatenate([agent_pts_3d, wrist_pts_3d])
                fused_source = "agentview+wrist"
            elif float(wrist_score) > float(agent_data["score"]):
                points_3d = wrist_pts_3d
                fused_source = "wrist_higher_score"
            else:
                points_3d = agent_pts_3d
                fused_source = "agentview_higher_score"
        elif len(agent_pts_3d) > 0:
            points_3d = agent_pts_3d
            fused_source = "agentview_only" if use_multiview else "agentview"
        elif wrist_pts_3d is not None and len(wrist_pts_3d) > 0:
            points_3d = wrist_pts_3d
            fused_source = "wrist_only"
        else:
            points_3d = np.empty((0, 3), dtype=np.float64)
            fused_source = "none"

        self._record_runtime_diagnostic(
            "language_pointcloud_complete",
            prompt=text_prompt,
            agentview_point_count=int(len(agent_pts_3d)),
            wrist_point_count=int(len(wrist_pts_3d)) if wrist_pts_3d is not None else 0,
            fused_point_count=int(len(points_3d)),
            fused_source=fused_source,
            fused_centroid_world=(
                np.mean(points_3d, axis=0)
                if len(points_3d) > 0
                else None
            ),
            agentview_score=(
                float(agent_data["score"])
                if agent_data is not None and agent_data["score"] is not None
                else None
            ),
            wrist_score=float(wrist_score) if wrist_score is not None else None,
            valid_cameras=list(camera_data.keys()),
            per_camera_visualizations={
                cam: data.get("visualization_path")
                for cam, data in camera_data.items()
                if data.get("visualization_path")
            },
            per_camera_image_overlays={
                cam: data.get("image_overlay_path")
                for cam, data in camera_data.items()
                if data.get("image_overlay_path")
            },
            per_camera_segmentation_overlays={
                cam: data.get("segmentation_overlay_path")
                for cam, data in camera_data.items()
                if data.get("segmentation_overlay_path")
            },
            per_camera_raw_npz={
                cam: data.get("raw_npz_path")
                for cam, data in camera_data.items()
                if data.get("raw_npz_path")
            },
        )

        return {
            "agentview_mask": agent_data["mask"] if agent_data is not None else None,
            "wrist_mask": wrist_mask,
            "points_3d": np.asarray(points_3d, dtype=np.float64).reshape(-1, 3),
            "agentview_points_3d": agent_pts_3d,
            "wrist_points_3d": np.asarray(wrist_pts_3d, dtype=np.float64).reshape(-1, 3) if wrist_pts_3d is not None else None,
            "agentview_score": agent_data["score"] if agent_data is not None else None,
            "wrist_score": wrist_score,
        }

    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        """Get the oriented bounding box from 3D points.

        The returned rotation R is canonicalized to remove PCA sign-flip
        ambiguity so repeated calls on the same (or similar) point cloud
        give consistent orientation, not random 180° flips:

          1) Z axis points down in world (so top-down grasps are default).
          2) X axis lies in the world +X half-plane (with +Y tiebreaker
             when X is exactly along world ±Y).

        See ``_canonicalize_obb_rotation`` for the full derivation. Both
        the LLM-direct callers (registered as a function via the API's
        ``functions()`` map) and the internal caller in ``get_object_pose``
        receive the canonicalized rotation.

        Args:
            points: (N, 3) float64 points.

        Returns:
            dict with center, extent, R (canonicalized).
        """
        obb = _get_obb(points)
        obb["R"] = self._canonicalize_obb_rotation(np.asarray(obb["R"]))
        return obb

    @staticmethod
    def _canonicalize_obb_rotation(R: np.ndarray) -> np.ndarray:
        """Remove PCA sign-flip ambiguity from an OBB rotation matrix.

        Open3D's ``get_oriented_bounding_box`` adds Gaussian noise to the
        points before PCA (see common.py:340), and that noise determines
        the sign Open3D picks for each eigenvector. The result: repeat
        calls on the same point cloud return rotations that can differ
        by 180° about any axis. We pin two axes deterministically; the
        third follows from the right-hand rule.

        Step 1 — Z points down. When the raw OBB Z has a positive world Z
        component, flip about Y (swap X and Z signs). Default grasps
        approach from above.

        Step 2 — X positive in the world XY plane. Without this, an
        elongated object's long axis randomly flips 180° about Z between
        calls. We constrain the OBB X axis to satisfy ``R[0,0] > 0``, or
        when it's exactly along world ±Y (``R[0,0] ≈ 0``), to
        ``R[1,0] > 0`` as a tiebreaker. Flipping about Z swaps X and Y
        signs simultaneously, preserving handedness and keeping Z down.
        """
        Rm = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()
        if Rm[2, 2] > 0:
            Rm = Rm @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
        rx, ry = float(Rm[0, 0]), float(Rm[1, 0])
        if rx < -1e-9 or (abs(rx) <= 1e-9 and ry < 0):
            Rm = Rm @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        return Rm

    def filter_noise(
        self, points: np.ndarray, colors: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Filter noise from a point cloud using DBSCAN.

        Args:
            points: (N, 3) points.
            colors: optional (N, 3) colors.

        Returns:
            Filtered (points, colors).
        """
        eps = 0.005
        min_samples = 10
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        filtered = points[labels != -1]
        filtered_colors = colors[labels != -1] if colors is not None else None
        return filtered, filtered_colors

    def subsample_point_cloud(self, pc: np.ndarray, max_points: int = 10000) -> np.ndarray:
        """Randomly subsample a point cloud as finite (N, 3) xyz rows."""
        arr = np.asarray(pc)
        if arr.ndim >= 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3].reshape(-1, 3)
        elif arr.ndim == 2 and arr.shape[1] == 3:
            pass
        elif arr.ndim == 2 and arr.shape[0] == 3:
            arr = arr.T
        else:
            arr = arr.reshape(-1, 3)

        arr = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(arr).all(axis=1)
        arr = arr[finite]
        if len(arr) > max_points:
            return arr[np.random.choice(len(arr), max_points, replace=False)]
        return arr

    # ------------------------------------------------------------------
    # Grasp planning
    # ------------------------------------------------------------------

    def plan_grasp_from_point_clouds(
        self,
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
        label: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan grasp candidates using the configured grasp backend.

        Args:
            pc_full: (N, 3) full scene point cloud (world frame).
            pc_segment: (M, 3) segmented object point cloud (world frame).
            label: optional object label used for viz filenames / logs.

        Returns:
            (grasp_sample_tf, grasp_scores) with +0.1034m TCP offset along
            the gripper approach axis already applied. The returned
            transforms are in the SAME frame as the inputs (world frame if
            you passed world points). Apply the repo's standard 90-degree
            grasp-yaw convention before converting these matrices into
            final pos/quat commands.

        Why 0.1034 m (was 0.12, inherited from LIBERO):
            ContactGraspNet's gripper control points are documented in
            ``panda_gripper_coords.yml``:

                gripper_*_center_flat: [_, _, 0.1034]
                gripper_*_tip_flat:    [_, _, 0.1122]

            i.e. the finger PAD CENTER sits at local +Z = 0.1034 m from
            the grasp-frame origin (which corresponds to the panda_hand
            link). Translating the returned grasp by exactly +0.1034 m
            along the local approach axis shifts ``g.translation`` so
            that after the system's internal TCP cancellation in
            ``goto_pose`` the Robotiq ``grasp_site`` lands at exactly the
            point ContactGraspNet predicted as the finger pad contact.

            LIBERO uses 0.12 m — works there because robosuite's Panda
            gripper has finger compliance and LIBERO objects are
            small/squishy. On MolmoSpaces' rigid procthor scenes the
            extra ~17 mm pushes the gripper through the table on thin
            tabletop objects (~1-3 cm tall tools), causing
            ``MolmoSpacesMotionAbort: no measurable progress`` and the
            "approach succeeded but close_gripper never fired" failure
            mode (see 2026-05-21 ep8 trial). Reducing to the
            geometrically-correct 0.1034 m fixes the table penetration
            without giving up much grip depth.
        """
        backend_name = self._grasp_backend_display_name()
        try:
            grasp_sample, grasp_scores, _ = self.grasp_net_plan_point_clouds_fn(
                pc_full, pc_segment, segmap_id=1
            )
        except Exception as exc:
            service_diagnostics = copy.deepcopy(
                getattr(self.grasp_net_plan_point_clouds_fn, "last_diagnostics", {})
            )
            log_path = self._append_graspnet_log(
                {
                    "event": "plan_grasp_from_point_clouds",
                    "label": label,
                    "backend": backend_name,
                    "full_point_count": int(len(pc_full)),
                    "segment_point_count": int(len(pc_segment)),
                    "service_diagnostics": service_diagnostics,
                    "error": str(exc),
                }
            )
            self._record_runtime_diagnostic(
                "grasp_plan_pointclouds",
                label=label,
                full_point_count=int(len(pc_full)),
                segment_point_count=int(len(pc_segment)),
                candidate_count=0,
                backend=backend_name,
                graspnet_log_path=log_path,
                service_diagnostics=service_diagnostics,
                error=str(exc),
            )
            try:
                from rats.utils.execution_logger import log_step

                log_step(
                    f"{backend_name} Diagnostics",
                    (
                        f"{backend_name} call failed for '{label or '-'}': {exc}. "
                        f"log={log_path or 'not saved'}"
                    ),
                    highlight=True,
                    timeline_kind="perception",
                    timeline_label="graspnet_error",
                )
            except Exception:
                pass
            raise
        service_diagnostics = copy.deepcopy(
            getattr(self.grasp_net_plan_point_clouds_fn, "last_diagnostics", {})
        )
        scores_arr = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
        candidate_count = int(min(len(grasp_sample), scores_arr.size))

        try:
            obs = self.get_observation()
        except Exception:
            obs = {}

        agent_cam_name = getattr(self, "camera_name", "agentview")
        agent_cam = obs.get(agent_cam_name) if isinstance(obs, dict) else None
        agent_rgb = None
        agent_intrinsics = None
        agent_extrinsics = None
        if isinstance(agent_cam, dict):
            images = agent_cam.get("images")
            if isinstance(images, dict):
                agent_rgb = images.get("rgb")
            agent_intrinsics = agent_cam.get("intrinsics")
            agent_extrinsics = agent_cam.get("pose_mat")

        if candidate_count == 0:
            overlay_path = self._save_grasp_candidate_overlay(
                label=label,
                camera_name=agent_cam_name,
                rgb=agent_rgb,
                intrinsics=agent_intrinsics,
                extrinsics=agent_extrinsics,
                grasp_tfs=np.empty((0, 4, 4), dtype=np.float64),
                grasp_scores=np.empty((0,), dtype=np.float64),
                frame="world",
            )
            log_path = self._append_graspnet_log(
                {
                    "event": "plan_grasp_from_point_clouds",
                    "label": label,
                    "backend": backend_name,
                    "full_point_count": int(len(pc_full)),
                    "segment_point_count": int(len(pc_segment)),
                    "candidate_count": 0,
                    "service_diagnostics": service_diagnostics,
                    "agentview_overlay_path": overlay_path,
                    "reason": "no_grasp_candidates",
                }
            )
            self._record_runtime_diagnostic(
                "grasp_plan_pointclouds",
                label=label,
                full_point_count=int(len(pc_full)),
                segment_point_count=int(len(pc_segment)),
                backend=backend_name,
                segment_centroid_world=(
                    np.mean(pc_segment, axis=0) if len(pc_segment) > 0 else None
                ),
                candidate_count=0,
                top_scores=[],
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
                        f"{backend_name} returned zero candidates for '{label or '-'}'. "
                        f"full_pts={len(pc_full)} segment_pts={len(pc_segment)} "
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
        top_scores = np.sort(scores_arr)[::-1][:3]
        top_scores_list = [float(s) for s in top_scores]
        top_order = np.argsort(-scores_arr)[: min(5, scores_arr.size)]
        top_candidates = []
        for idx in top_order:
            grasp_tf = np.asarray(grasp_sample_tf[idx], dtype=np.float64)
            top_candidates.append({
                "index": int(idx),
                "score": float(scores_arr[idx]),
                "position_world": grasp_tf[:3, 3],
                "approach_axis_world": grasp_tf[:3, 2],
            })
        logger.info(
            "plan_grasp_from_point_clouds[%s]: candidates=%d top3_scores=%s full_pts=%d segment_pts=%d",
            label or "-",
            int(scores_arr.size),
            [round(s, 3) for s in top_scores_list],
            int(len(pc_full)),
            int(len(pc_segment)),
        )

        extras: dict[str, np.ndarray] = {
            "pc_segment_world": np.asarray(pc_segment, dtype=np.float64),
            "robot_base_pose": np.asarray(
                obs.get("robot_base_pose", np.zeros(7)) if isinstance(obs, dict) else np.zeros(7),
                dtype=np.float64,
            ),
            "robot_cartesian_pos": np.asarray(
                obs.get("robot_cartesian_pos", np.zeros(8)) if isinstance(obs, dict) else np.zeros(8),
                dtype=np.float64,
            ),
        }
        for cam_key in (self.camera_name, self.wrist_camera_name):
            cam = obs.get(cam_key) if isinstance(obs, dict) else None
            pose_mat = cam.get("pose_mat") if isinstance(cam, dict) else None
            if pose_mat is not None:
                extras[f"{cam_key}_extrinsics_cam_to_world"] = np.asarray(pose_mat, dtype=np.float64)
        grasp_artifact = self._save_grasp_viz(
            pc_full, grasp_sample_tf, grasp_scores, label=label, extra_arrays=extras,
        )
        grasp_viz_path = grasp_artifact.get("image_path")
        grasp_raw_path = grasp_artifact.get("raw_npz_path")
        overlay_path = self._save_grasp_candidate_overlay(
            label=label,
            camera_name=agent_cam_name,
            rgb=agent_rgb,
            intrinsics=agent_intrinsics,
            extrinsics=agent_extrinsics,
            grasp_tfs=grasp_sample_tf,
            grasp_scores=grasp_scores,
            frame="world",
        )
        graspnet_log_path = self._append_graspnet_log(
            {
                "event": "plan_grasp_from_point_clouds",
                "label": label,
                "backend": backend_name,
                "full_point_count": int(len(pc_full)),
                "segment_point_count": int(len(pc_segment)),
                "candidate_count": int(len(grasp_sample)),
                "local_z_offset_m": 0.1034,
                "top_scores": top_scores_list,
                "visualization_path": grasp_viz_path,
                "agentview_overlay_path": overlay_path,
                "raw_npz_path": grasp_raw_path,
                "service_diagnostics": service_diagnostics,
            }
        )
        self._record_runtime_diagnostic(
            "grasp_plan_pointclouds",
            label=label,
            full_point_count=int(len(pc_full)),
            segment_point_count=int(len(pc_segment)),
            backend=backend_name,
            segment_centroid_world=(
                np.mean(pc_segment, axis=0) if len(pc_segment) > 0 else None
            ),
            candidate_count=int(len(grasp_sample)),
            local_z_offset_m=0.1034,
            best_score=float(np.max(grasp_scores)) if len(grasp_scores) > 0 else None,
            top_scores=top_scores_list,
            top_candidate_grasps=top_candidates,
            visualization_path=grasp_viz_path,
            agentview_overlay_path=overlay_path,
            raw_npz_path=grasp_raw_path,
            graspnet_log_path=graspnet_log_path,
            service_diagnostics=service_diagnostics,
        )
        return grasp_sample_tf, grasp_scores

    # ------------------------------------------------------------------
    # High-level object perception
    # ------------------------------------------------------------------

    def get_object_pose(
        self, object_name: str, use_multiview: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Get the pose of an object from a natural language description.

        Args:
            object_name: object name in lowercase.
            use_multiview: use wrist camera as well.

        Returns:
            (position (3,), quaternion_wxyz (4,)) or (None, None).
        """
        start_time = time.time()
        self._record_runtime_diagnostic(
            "get_object_pose_start",
            object_name=object_name,
            use_multiview=use_multiview,
        )

        result = self.get_object_3d_points_and_masks_from_language(
            object_name, use_multiview=use_multiview
        )
        points_3d = result["points_3d"]

        if len(points_3d) == 0:
            self._record_runtime_diagnostic(
                "get_object_pose_failure",
                object_name=object_name,
                reason="no_segment_points",
            )
            return None, None

        raw_point_count = int(len(points_3d))
        points_3d, _ = self.filter_noise(points_3d)
        if len(points_3d) == 0:
            self._record_runtime_diagnostic(
                "get_object_pose_failure",
                object_name=object_name,
                reason="all_points_filtered",
                raw_point_count=raw_point_count,
            )
            return None, None
        self._record_runtime_diagnostic(
            "get_object_pose_filtered",
            object_name=object_name,
            raw_point_count=raw_point_count,
            filtered_point_count=int(len(points_3d)),
        )

        # The wrapper now canonicalizes for us, so R is already in the
        # canonical form (Z down, X positive in world XY plane). See
        # `_canonicalize_obb_rotation` for the full rule.
        obb = self.get_oriented_bounding_box_from_3d_points(points_3d)

        position = np.array(obb["center"])
        R = np.array(obb["R"])

        quaternion_wxyz = vtf.SO3.from_matrix(R).wxyz
        self._record_runtime_diagnostic(
            "get_object_pose_success",
            object_name=object_name,
            position=np.asarray(position),
            quaternion_wxyz=np.asarray(quaternion_wxyz),
        )
        print(f"get_object_pose in {time.time() - start_time} seconds")
        return position, quaternion_wxyz

    def sample_grasp_pose(
        self, object_name: str, use_multiview: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object from a natural language description.

        Args:
            object_name: object name in lowercase.
            use_multiview: use wrist camera as well.

        Returns:
            (position (3,), quaternion_wxyz (4,)).
        """
        start_time = time.time()
        self._record_runtime_diagnostic(
            "sample_grasp_pose_start",
            object_name=object_name,
            use_multiview=use_multiview,
        )

        result = self.get_object_3d_points_and_masks_from_language(
            object_name, use_multiview=use_multiview
        )
        pc_segment = result["points_3d"]

        if len(pc_segment) == 0:
            self._record_runtime_diagnostic(
                "sample_grasp_pose_failure",
                object_name=object_name,
                reason="no_segment_points",
            )
            raise ValueError(f"Could not segment object '{object_name}'")

        obs = self.get_observation()
        pc_full_parts = []
        for cam_name in [self.camera_name, self.wrist_camera_name]:
            depth = obs[cam_name]["images"]["depth"]
            intrinsics = obs[cam_name]["intrinsics"]
            extrinsics = obs[cam_name]["pose_mat"]
            pts_camera = depth_to_pointcloud(depth, intrinsics, subsample_factor=1)
            pts_homogeneous = np.concatenate(
                [pts_camera, np.ones((len(pts_camera), 1))], axis=1
            )
            pts_world = (extrinsics @ pts_homogeneous.T).T[:, :3]
            pc_full_parts.append(pts_world)
        pc_full = np.concatenate(pc_full_parts)

        raw_segment_count = int(len(pc_segment))
        pc_segment, _ = self.filter_noise(pc_segment)
        if len(pc_segment) == 0:
            self._record_runtime_diagnostic(
                "sample_grasp_pose_failure",
                object_name=object_name,
                reason="all_segment_points_filtered",
                raw_segment_count=raw_segment_count,
                full_point_count=int(len(pc_full)),
            )
            raise ValueError(f"No valid points after filtering for '{object_name}'")
        self._record_runtime_diagnostic(
            "sample_grasp_pose_pointclouds",
            object_name=object_name,
            raw_segment_count=raw_segment_count,
            filtered_segment_count=int(len(pc_segment)),
            full_point_count=int(len(pc_full)),
        )

        grasp_sample_tf, grasp_scores = self.plan_grasp_from_point_clouds(
            pc_full, pc_segment, label=object_name,
        )

        best_idx = grasp_scores.argmax()
        best_grasp = vtf.SE3.from_matrix(grasp_sample_tf[best_idx])
        best_grasp = best_grasp @ vtf.SE3.from_rotation(
            rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2)
        )
        scores_arr = np.asarray(grasp_scores).reshape(-1)
        top_k = min(3, scores_arr.size)
        top_order = np.argsort(-scores_arr)[:top_k]
        top_scores = [float(scores_arr[i]) for i in top_order]
        best_pos = np.asarray(best_grasp.wxyz_xyz[-3:], dtype=np.float64)
        best_quat = np.asarray(best_grasp.wxyz_xyz[:4], dtype=np.float64)
        logger.info(
            "sample_grasp_pose[%s]: candidates=%d top3_scores=%s best_idx=%d best_pos=%s best_quat_wxyz=%s",
            object_name,
            int(scores_arr.size),
            [round(s, 3) for s in top_scores],
            int(best_idx),
            np.round(best_pos, 4).tolist(),
            np.round(best_quat, 4).tolist(),
        )
        self._record_runtime_diagnostic(
            "sample_grasp_pose_success",
            object_name=object_name,
            candidate_count=int(scores_arr.size),
            best_index=int(best_idx),
            best_score=float(scores_arr[best_idx]),
            top_scores=top_scores,
            grasp_position=best_pos,
            grasp_quaternion_wxyz=best_quat,
        )

        print(f"sample_grasp_pose in {time.time() - start_time} seconds")
        return best_grasp.wxyz_xyz[-3:], best_grasp.wxyz_xyz[:4]
