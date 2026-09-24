from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as vtf

from rats.envs.base import BaseEnv
from rats.integrations.franka.molmospaces_reduced import FrankaMolmoSpacesApiReduced


class FrankaMolmoSpacesApiReducedSkillLibrary(FrankaMolmoSpacesApiReduced):
    """MolmoSpaces reduced API plus reusable geometry/perception helper skills.

    This mirrors the LIBERO reduced skill-library surface so the planner can
    compose robust routines from smaller pieces instead of overusing a single
    high-level grasp helper.
    """

    _capx_only_added_skills_message_printed = False

    def __init__(
        self,
        env: BaseEnv,
        enable_augmented_helpers: bool = True,
        enable_arm_speed: bool = False,
        enable_filter_noise: bool = False,
        enable_raw_language_pointcloud_helpers: bool = False,
        enable_grasp_selection_helpers: bool = False,
        enable_wrist_closeloop: bool = False,
        grasp_backend: str = "graspnet",
    ) -> None:
        super().__init__(
            env,
            enable_augmented_helpers=enable_augmented_helpers,
            enable_arm_speed=enable_arm_speed,
            enable_filter_noise=enable_filter_noise,
            grasp_backend=grasp_backend,
        )
        self._enable_augmented_helpers = bool(enable_augmented_helpers)
        self._enable_raw_language_pointcloud_helpers = bool(
            enable_raw_language_pointcloud_helpers
        )
        self._enable_grasp_selection_helpers = bool(enable_grasp_selection_helpers)
        self._enable_wrist_closeloop = bool(enable_wrist_closeloop)

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        if not getattr(self, "_enable_raw_language_pointcloud_helpers", False):
            # Keep the default prompt away from these opaque high-level helpers:
            # they select the highest-score SAM3 mask internally and bypass
            # explicit vlm_verify-based mask validation.
            fns.pop("get_object_3d_points_and_masks_from_language", None)
            fns.pop("fuse_object_world_points", None)
            fns.pop("search_and_locate_object", None)
            fns.pop("get_object_pose", None)
            fns.pop("sample_grasp_pose", None)
        fns["rotation_matrix_to_quaternion"] = self.rotation_matrix_to_quaternion
        fns["decompose_transform"] = self.decompose_transform
        fns["depth_to_point_cloud"] = self.depth_to_point_cloud
        fns["mask_to_world_points"] = self.mask_to_world_points
        fns["pixel_to_world_point"] = self.pixel_to_world_point
        fns["transform_points"] = self.transform_points
        fns["interpolate_segment"] = self.interpolate_segment
        fns["normalize_vector"] = self.normalize_vector
        fns["select_top_down_grasp"] = self.select_top_down_grasp
        if self._enable_augmented_helpers and not getattr(self._env, "capx_only", False):
            if getattr(self, "_enable_grasp_selection_helpers", False):
                fns["select_horizontal_grasp"] = self.select_horizontal_grasp
                fns["select_grasp_along_direction"] = self.select_grasp_along_direction
            fns["vlm_verify"] = self.vlm_verify
            fns["verify_object_identity"] = self.verify_object_identity
            fns["inspect_at_wrist"] = self.inspect_at_wrist
            if self._enable_wrist_closeloop:
                fns["grasp_with_wrist_closeloop"] = self.grasp_with_wrist_closeloop
        elif getattr(self._env, "capx_only", False) and self._enable_augmented_helpers:
            if not type(self)._capx_only_added_skills_message_printed:
                print("Using cap-x apis only for added skills")
                type(self)._capx_only_added_skills_message_printed = True
        return fns

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert a 3x3 rotation matrix to a [w, x, y, z] quaternion."""
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        return np.array([w, x, y, z], dtype=np.float64)

    def decompose_transform(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decompose a 4x4 transform into position and [w, x, y, z] quaternion."""
        position = T[:3, 3]
        quat = self.rotation_matrix_to_quaternion(T[:3, :3])
        return position, quat

    def depth_to_point_cloud(self, depth_img: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """Convert a depth image to an organized point cloud in camera frame."""
        depth = depth_img[:, :, 0] if depth_img.ndim == 3 else depth_img
        h, w = depth.shape
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        z = depth
        x = (x_grid - cx) * z / fx
        y = (y_grid - cy) * z / fy
        return np.dstack((x, y, z))

    def mask_to_world_points(
        self, mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """Convert masked depth pixels into 3D world points."""
        mask_meta = getattr(self, "_mask_debug_metadata", {}).get(id(mask), {})
        label = str(mask_meta.get("label") or "current_mask_object")
        score = mask_meta.get("score")
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            self._log_step_update(text=f"No positive mask pixels for `{label}`.")
            return np.empty((0, 3))

        depth_img = depth[:, :, 0] if depth.ndim == 3 else depth
        z_vals = depth_img[ys, xs]
        valid = z_vals > 0
        ys = ys[valid]
        xs = xs[valid]
        z = z_vals[valid]

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        x_cam = (xs - cx) * z / fx
        y_cam = (ys - cy) * z / fy
        points_cam = np.stack([x_cam, y_cam, z], axis=-1)
        points_cam_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world_hom = (extrinsics @ points_cam_hom.T).T
        points_world = points_world_hom[:, :3]

        segmentation_overlay_path = mask_meta.get("segmentation_overlay_path")
        rgb_for_context = mask_meta.get("rgb")
        if rgb_for_context is not None and len(points_world) > 0:
            # `normalize_vector()` renders the pull/push direction overlay from
            # the latest image/world context.  Molmo point-prompt paths populate
            # that context via `pixel_to_world_point`; SAM3 text-mask paths do
            # not have a single pixel, so seed the context here from the mask
            # centroid / world-point centroid.  This keeps the generated policy
            # simple (`mask_to_world_points(...)` followed by
            # `normalize_vector(...)`) while still producing an actual
            # per-step-verifier image artifact for the interaction direction.
            try:
                self._latest_pixel_world_context = {
                    "label": label,
                    "prompt": label,
                    "point": [int(round(float(xs.mean()))), int(round(float(ys.mean())))],
                    "camera": "agentview",
                    "world_point": np.asarray(points_world, dtype=np.float64).mean(axis=0),
                    "rgb": np.asarray(rgb_for_context, dtype=np.uint8)[..., :3].copy(),
                    "intrinsics": np.asarray(intrinsics, dtype=np.float64),
                    "extrinsics": np.asarray(extrinsics, dtype=np.float64),
                    "segmentation_overlay_path": segmentation_overlay_path,
                }
            except Exception:
                pass

        viz_text = (
            f"Projected {len(points_world)} world point(s) for `{label}`"
            + (f" (SAM3 score={float(score):.3f})." if score is not None else ".")
        )
        image_path = None
        image_overlay_path = None
        raw_npz_path = None
        overlay = self._mask_world_image_overlay(
            mask_meta.get("rgb"),
            mask,
            label=label,
            world_points=points_world,
        )
        image_overlay_path = self._save_rgb_artifact(f"world_points_overlay_{label}", overlay)
        try:
            full_cam = self.depth_to_point_cloud(depth_img, intrinsics).reshape(-1, 3)
            valid_full = np.isfinite(full_cam).all(axis=1) & (full_cam[:, 2] > 0)
            full_cam = full_cam[valid_full]
            full_world_h = np.hstack([full_cam, np.ones((len(full_cam), 1))])
            full_world = (extrinsics @ full_world_h.T).T[:, :3]
            full_world_for_viz = full_world
            if len(full_world_for_viz) > 20000:
                idx = np.random.choice(len(full_world_for_viz), 20000, replace=False)
                full_world_for_viz = full_world_for_viz[idx]
            points_world_for_viz = points_world
            if len(points_world_for_viz) > 20000:
                idx = np.random.choice(len(points_world_for_viz), 20000, replace=False)
                points_world_for_viz = points_world_for_viz[idx]
            artifact = self._save_per_camera_pointcloud_viz(
                "agentview",
                full_world_for_viz,
                points_world_for_viz,
                prompt=label,
                score=(float(score) if score is not None else None),
                extra_arrays={
                    "rgb": np.asarray(rgb_for_context, dtype=np.uint8)[..., :3].copy()
                    if rgb_for_context is not None
                    else None,
                    "intrinsics": np.asarray(intrinsics, dtype=np.float64),
                    "extrinsics_cam_to_world": np.asarray(extrinsics, dtype=np.float64),
                    "mask": np.asarray(mask),
                },
            )
            image_path = artifact.get("image_path")
            raw_npz_path = artifact.get("raw_npz_path")
        except Exception as exc:
            self._record_runtime_diagnostic(
                "mask_to_world_points_viz_failed",
                label=label,
                error=f"{type(exc).__name__}: {exc}",
            )

        self._record_runtime_diagnostic(
            "mask_to_world_points",
            label=label,
            score=(float(score) if score is not None else None),
            mask_pixels=int(len(ys)),
            valid_depth_pixels=int(len(points_world)),
            image_overlay_path=image_overlay_path,
            segmentation_overlay_path=segmentation_overlay_path,
            pointcloud_viz_path=image_path,
            raw_npz_path=raw_npz_path,
            output_type="world_point_cloud_visualization",
            description=(
                "World-frame point cloud from a segmentation mask, plus an actual "
                "RGB image overlay showing the mask pixels that produced it."
            ),
        )
        self._log_step_update(
            text=viz_text
            + (f"\nPoint-cloud artifact: `{raw_npz_path}`" if raw_npz_path else ""),
            images=[p for p in (image_overlay_path, image_path) if p],
        )

        publisher = getattr(self, "_viser_publisher", None)
        if publisher is not None:
            try:
                publisher.publish_object_points(
                    points_world,
                    label=label,
                    score=(float(score) if score is not None else None),
                )
            except Exception:
                pass

        return points_world

    def pixel_to_world_point(
        self, u: int, v: int, z: float, intrinsics: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """Project a single depth pixel into world coordinates."""
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        p_cam = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
        world = (extrinsics @ p_cam)[:3]
        key = (int(round(float(u))), int(round(float(v))))
        point_meta = getattr(self, "_molmo_point_debug_metadata", {}).get(key, {})
        if isinstance(point_meta, dict) and point_meta:
            self._latest_pixel_world_context = dict(point_meta)
            self._latest_pixel_world_context["world_point"] = world
            self._latest_pixel_world_context["intrinsics"] = np.asarray(intrinsics, dtype=np.float64)
            self._latest_pixel_world_context["extrinsics"] = np.asarray(extrinsics, dtype=np.float64)
        self._record_runtime_diagnostic(
            "pixel_to_world_point",
            pixel=[int(u), int(v)],
            depth=float(z),
            world_point=world,
            point_overlay_path=point_meta.get("overlay_path") if isinstance(point_meta, dict) else None,
            prompt=point_meta.get("prompt") if isinstance(point_meta, dict) else None,
            camera=point_meta.get("camera") if isinstance(point_meta, dict) else None,
            output_type="world_point_from_image_point",
            description="World-frame point produced from an image point, with Molmo point overlay when available.",
        )
        return world

    def transform_points(self, points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Apply a homogeneous transform to a set of 3D points."""
        original_shape = points.shape
        pts = points.reshape(-1, 3)
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1))))
        transformed = (transform_matrix @ pts_hom.T).T
        return transformed[:, :3].reshape(original_shape)

    def interpolate_segment(
        self, p1: np.ndarray, p2: np.ndarray, step: float = 0.03
    ) -> list[np.ndarray]:
        """Generate evenly spaced waypoints along a line segment."""
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6:
            return [p1]
        num_points = int(np.ceil(dist / step))
        return [p1 + (p2 - p1) * t for t in np.linspace(0, 1, num_points + 1)]

    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length when possible."""
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            self._record_runtime_diagnostic(
                "normalize_vector",
                vector=v,
                norm=float(norm),
                normalized_vector=v,
                output_type="vector_normalization",
                description="Degenerate vector normalization; vector was too small to normalize.",
            )
            return v
        normalized = v / norm
        overlay, overlay_meta = self._pull_direction_overlay_from_context(normalized)
        overlay_path = self._save_rgb_artifact("pull_direction_overlay", overlay)
        self._record_runtime_diagnostic(
            "pull_direction_estimate",
            vector=v,
            norm=float(norm),
            normalized_vector=normalized,
            overlay_path=overlay_path,
            output_type="pull_direction_visualization",
            description="Projected pull/interaction direction arrow overlaid on the relevant camera image.",
            **overlay_meta,
        )
        return normalized

    def select_top_down_grasp(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        world_from_grasp_frame: np.ndarray | None = None,
        vertical_threshold: float = 0.8,
        apply_grasp_yaw: bool = True,
    ) -> tuple[np.ndarray | None, float]:
        """Pick the highest-scoring near-top-down grasp candidate.

        Args:
            grasps: (K, 4, 4) candidate grasp transforms.
            scores: (K,) candidate scores.
            world_from_grasp_frame: 4x4 transform that takes each grasp into
                world frame. Default None means identity, which is correct when
                the grasps are ALREADY in world frame (e.g. returned from
                plan_grasp_from_point_clouds with world-frame point clouds).
                Pass an explicit camera-to-world matrix only if the grasps were
                produced in camera frame (e.g. from plan_grasp(depth, intr, seg)).
            vertical_threshold: minimum -Z alignment between gripper approach
                and world Z to call a grasp "top-down". 0.8 ≈ 37° of vertical.
            apply_grasp_yaw: if True (default), post-multiply the selected grasp
                by a 90° yaw to match the standard gripper convention
                (fingers close ACROSS the approach axis).
                Disable only if you plan to apply the yaw yourself.

        Returns:
            (selected_grasp_4x4 in world frame | None, selected_score).
        """
        if world_from_grasp_frame is None:
            world_from_grasp_frame = np.eye(4, dtype=np.float64)

        best_grasp = None
        best_score = -np.float64("inf")
        best_index = None
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        for i, g in enumerate(grasps):
            g_world = world_from_grasp_frame @ g
            gripper_approach = g_world[:3, :3][:, 2]
            alignment = -np.dot(gripper_approach, world_z)
            if alignment > vertical_threshold and scores[i] > best_score:
                best_score = float(scores[i])
                best_grasp = g_world
                best_index = int(i)

        best_grasp_pre_yaw = best_grasp.copy() if best_grasp is not None else None

        if best_grasp is not None and apply_grasp_yaw:
            yawed = vtf.SE3.from_matrix(best_grasp) @ vtf.SE3.from_rotation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2)
            )
            best_grasp = yawed.as_matrix()

        selection_overlay_path = None
        try:
            obs = self.get_observation()
            cam_name = getattr(self, "camera_name", "agentview")
            cam = obs.get(cam_name, {})
            images = cam.get("images", {}) if isinstance(cam, dict) else {}
            world_grasps = np.asarray([world_from_grasp_frame @ g for g in grasps])
            overlay_scores = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
            if best_index is not None and best_index < overlay_scores.size:
                overlay_scores[best_index] = (
                    float(np.nanmax(overlay_scores)) + 1.0
                    if overlay_scores.size > 0
                    else 1.0
                )
            selection_overlay_path = self._save_grasp_candidate_overlay(
                label="selected_top_down_grasp",
                camera_name=cam_name,
                rgb=images.get("rgb") if isinstance(images, dict) else None,
                intrinsics=cam.get("intrinsics") if isinstance(cam, dict) else None,
                extrinsics=cam.get("pose_mat") if isinstance(cam, dict) else None,
                grasp_tfs=world_grasps,
                grasp_scores=overlay_scores,
                frame="world",
                selected_tf=best_grasp_pre_yaw,
            )
        except Exception:
            selection_overlay_path = None

        self._record_runtime_diagnostic(
            "grasp_selection",
            selector="select_top_down_grasp",
            candidate_count=int(len(grasps)),
            selected=best_grasp is not None,
            selected_index=best_index,
            selected_score=best_score if best_grasp is not None else None,
            selected_position_world=best_grasp[:3, 3] if best_grasp is not None else None,
            agentview_overlay_path=selection_overlay_path,
            vertical_threshold=float(vertical_threshold),
            apply_grasp_yaw=bool(apply_grasp_yaw),
        )
        return best_grasp, best_score

    def select_horizontal_grasp(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        world_from_grasp_frame: np.ndarray | None = None,
        horizontal_threshold: float = 0.7,
        apply_grasp_yaw: bool = True,
    ) -> tuple[np.ndarray | None, float]:
        """Pick the highest-scoring near-horizontal grasp candidate.

        Complement to ``select_top_down_grasp``. Use this for articulated
        handles (drawer, cabinet, oven, fridge, microwave, doorway) whose
        manipulation axis is roughly parallel to the ground — a top-down
        approach collides with the furniture face before reaching the handle.

        "Horizontal" means the gripper approach axis has a small vertical
        component. Selection rule: among grasps with ``|approach_z| <
        (1 - horizontal_threshold)``, keep the highest-scoring one. The
        default ``horizontal_threshold=0.7`` accepts approach axes within
        ~acos(0.7)≈45° of the floor plane.

        Args:
            grasps: (K, 4, 4) candidate grasp transforms.
            scores: (K,) candidate scores.
            world_from_grasp_frame: camera→world transform if grasps are in
                camera frame. Leave None when grasps are already world-frame
                (e.g. from ``plan_grasp_from_point_clouds`` with world-frame
                point clouds).
            horizontal_threshold: minimum horizontality (1.0 = perfectly flat,
                0.0 = any). Grasps with |approach_z| > 1 - threshold fail.
            apply_grasp_yaw: same semantics as ``select_top_down_grasp`` —
                post-multiply by a 90° yaw so fingers close across the
                approach axis.

        Returns:
            (selected_grasp_4x4 in world frame | None, selected_score).
            None when no candidate clears the horizontality bar; callers
            should fall back to ``argmax(scores)`` or
            ``select_grasp_along_direction``.
        """
        if world_from_grasp_frame is None:
            world_from_grasp_frame = np.eye(4, dtype=np.float64)

        z_cutoff = 1.0 - float(horizontal_threshold)
        best_grasp = None
        best_score = -np.float64("inf")
        best_index = None

        for i, g in enumerate(grasps):
            g_world = world_from_grasp_frame @ g
            gripper_approach = g_world[:3, :3][:, 2]
            if abs(float(gripper_approach[2])) < z_cutoff and scores[i] > best_score:
                best_score = float(scores[i])
                best_grasp = g_world
                best_index = int(i)

        if best_grasp is not None and apply_grasp_yaw:
            yawed = vtf.SE3.from_matrix(best_grasp) @ vtf.SE3.from_rotation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2)
            )
            best_grasp = yawed.as_matrix()
        self._record_runtime_diagnostic(
            "grasp_selection",
            selector="select_horizontal_grasp",
            candidate_count=int(len(grasps)),
            selected=best_grasp is not None,
            selected_index=best_index,
            selected_score=best_score if best_grasp is not None else None,
            selected_position_world=best_grasp[:3, 3] if best_grasp is not None else None,
            horizontal_threshold=float(horizontal_threshold),
            apply_grasp_yaw=bool(apply_grasp_yaw),
        )
        return best_grasp, best_score

    def select_grasp_along_direction(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        approach_direction: np.ndarray,
        world_from_grasp_frame: np.ndarray | None = None,
        alignment_threshold: float = 0.6,
        score_weight: float = 1.0,
        apply_grasp_yaw: bool = True,
    ) -> tuple[np.ndarray | None, float]:
        """Pick the grasp whose approach axis best aligns with a target direction.

        This is the physically-meaningful selector for articulated opens: the
        gripper must approach the handle along the ``-joint_axis`` direction
        (i.e. from the side the joint opens toward). For a drawer that slides
        along world +X the robot is standing at +X, so the gripper should
        reach toward -X — pass ``approach_direction=np.array([-1, 0, 0])``.

        Ranking: among grasps with ``dot(approach, approach_direction) >
        alignment_threshold``, return the one that maximises
        ``alignment + score_weight * normalized_score``. Scores are minmax-
        normalised across candidates so the weighting is scale-independent.

        Args:
            grasps: (K, 4, 4) candidate grasp transforms.
            scores: (K,) candidate scores.
            approach_direction: (3,) world-frame unit vector the gripper
                approach axis should point along. Will be normalised if not
                already unit length.
            world_from_grasp_frame: camera→world transform if grasps are in
                camera frame. Leave None when grasps are already world-frame.
            alignment_threshold: minimum cosine alignment required.
                0.6 ≈ 53° cone, 0.3 ≈ 73° cone.
            score_weight: weight on the GraspNet score relative to the
                alignment term. Set to 0.0 for purely geometric selection.
            apply_grasp_yaw: post-multiply by a 90° yaw so fingers close
                across the approach axis.

        Returns:
            (selected_grasp_4x4 in world frame | None, combined_selector_score).
        """
        if world_from_grasp_frame is None:
            world_from_grasp_frame = np.eye(4, dtype=np.float64)

        direction = np.asarray(approach_direction, dtype=np.float64).reshape(3)
        direction = self.normalize_vector(direction)
        if np.linalg.norm(direction) < 1e-6:
            self._record_runtime_diagnostic(
                "grasp_selection",
                selector="select_grasp_along_direction",
                candidate_count=int(len(grasps)),
                selected=False,
                reason="invalid_approach_direction",
                target_approach_direction_world=direction,
            )
            return None, float("-inf")

        scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores_arr.size == 0:
            self._record_runtime_diagnostic(
                "grasp_selection",
                selector="select_grasp_along_direction",
                candidate_count=int(len(grasps)),
                selected=False,
                reason="empty_scores",
                target_approach_direction_world=direction,
            )
            return None, float("-inf")
        s_min = float(scores_arr.min())
        s_range = float(scores_arr.max() - s_min)
        if s_range < 1e-9:
            norm_scores = np.zeros_like(scores_arr)
        else:
            norm_scores = (scores_arr - s_min) / s_range

        best_grasp = None
        best_combined = -np.float64("inf")
        best_index = None
        best_alignment = None
        for i, g in enumerate(grasps):
            g_world = world_from_grasp_frame @ g
            gripper_approach = g_world[:3, :3][:, 2]
            alignment = float(np.dot(gripper_approach, direction))
            if alignment < alignment_threshold:
                continue
            combined = alignment + float(score_weight) * float(norm_scores[i])
            if combined > best_combined:
                best_combined = combined
                best_grasp = g_world
                best_index = int(i)
                best_alignment = alignment

        best_grasp_pre_yaw = best_grasp.copy() if best_grasp is not None else None

        if best_grasp is not None and apply_grasp_yaw:
            yawed = vtf.SE3.from_matrix(best_grasp) @ vtf.SE3.from_rotation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2)
            )
            best_grasp = yawed.as_matrix()
        selection_overlay_path = None
        try:
            obs = self.get_observation()
            cam_name = getattr(self, "camera_name", "agentview")
            cam = obs.get(cam_name, {})
            images = cam.get("images", {}) if isinstance(cam, dict) else {}
            world_grasps = np.asarray([world_from_grasp_frame @ g for g in grasps])
            overlay_scores = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
            if best_index is not None and best_index < overlay_scores.size:
                overlay_scores[best_index] = (
                    float(np.nanmax(overlay_scores)) + 1.0
                    if overlay_scores.size > 0
                    else 1.0
                )
            selection_overlay_path = self._save_grasp_candidate_overlay(
                label="selected_direction_grasp",
                camera_name=cam_name,
                rgb=images.get("rgb") if isinstance(images, dict) else None,
                intrinsics=cam.get("intrinsics") if isinstance(cam, dict) else None,
                extrinsics=cam.get("pose_mat") if isinstance(cam, dict) else None,
                grasp_tfs=world_grasps,
                grasp_scores=overlay_scores,
                frame="world",
                selected_tf=best_grasp_pre_yaw,
            )
        except Exception:
            selection_overlay_path = None
        self._record_runtime_diagnostic(
            "grasp_selection",
            selector="select_grasp_along_direction",
            candidate_count=int(len(grasps)),
            selected=best_grasp is not None,
            selected_index=best_index,
            selected_score=best_combined if best_grasp is not None else None,
            selected_position_world=best_grasp[:3, 3] if best_grasp is not None else None,
            agentview_overlay_path=selection_overlay_path,
            target_approach_direction_world=direction,
            selected_alignment=best_alignment,
            alignment_threshold=float(alignment_threshold),
            score_weight=float(score_weight),
            apply_grasp_yaw=bool(apply_grasp_yaw),
        )
        return best_grasp, best_combined

    # ------------------------------------------------------------------
    # Active perception + VLM verification (ported from libero_reduced_skill_library).
    # All goto_pose inputs stay WORLD frame; MolmoSpaces solve_ik converts
    # to the robot base frame internally.
    # ------------------------------------------------------------------

    def inspect_at_wrist(
        self,
        target_world_pos: np.ndarray,
        hover_height: float = 0.10,
    ) -> dict[str, Any]:
        """Move the wrist camera above a 3D target and return the close-up view.

        Use this when agentview perception (Molmo / SAM3) is uncertain about
        an object's identity or exact location. A 10 cm hover gives a much
        sharper close-up than the static agentview.

        Args:
            target_world_pos: (3,) world-frame position of the target object.
            hover_height: Vertical offset above the target (meters).

        Returns:
            dict with keys:
              - "wrist":     {"rgb", "depth", "intrinsics", "pose_mat"}
              - "agentview": same shape, captured at the same time.
              - "moved_to":  (3,) hover pose actually requested.
              - "ok":        bool; False if goto_pose raised.
        """
        target = np.asarray(target_world_pos, dtype=np.float64).reshape(3)
        topdown_quat = self.rotation_matrix_to_quaternion(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=np.float64,
            )
        )
        hover_pos = target + np.array([0.0, 0.0, max(0.02, float(hover_height))])
        ok = True
        try:
            self.goto_pose(hover_pos, topdown_quat)
        except Exception as e:
            ok = False
            print(f"[inspect_at_wrist] goto_pose failed: {e}")

        obs = self.get_observation()
        agent = obs.get(self.camera_name, {}) or {}
        wrist = obs.get(self.wrist_camera_name, {}) or {}

        def _pack(cam_obs: dict[str, Any]) -> dict[str, Any]:
            imgs = cam_obs.get("images", {}) or {}
            return {
                "rgb": imgs.get("rgb"),
                "depth": imgs.get("depth"),
                "intrinsics": cam_obs.get("intrinsics"),
                "pose_mat": cam_obs.get("pose_mat"),
            }

        return {
            "ok": ok,
            "moved_to": hover_pos,
            "wrist": _pack(wrist),
            "agentview": _pack(agent),
        }

    def verify_object_identity(
        self,
        rgb: np.ndarray,
        target_pixel: tuple[int, int],
        expected_object: str,
        crop_radius: int = 60,
    ) -> dict[str, Any]:
        """VLM second-look: does the marked pixel actually show `expected_object`?

        Use BEFORE grasping when the scene contains visually similar items.

        Args:
            rgb: (H, W, 3) uint8 RGB image.
            target_pixel: (x, y) pixel from your perception output.
            expected_object: Plain-language name (e.g. "red mug").
            crop_radius: Half-width of the square crop shown to the VLM.

        Returns:
            {"verified": bool, "confidence": float, "actual": str, "reasoning": str}.
        """
        from rats.agents.base_agent import image_to_data_url, query_llm_text

        if rgb is None or len(rgb.shape) != 3:
            self._record_runtime_diagnostic(
                "verify_object_identity",
                expected_object=expected_object,
                target_pixel=target_pixel,
                verified=False,
                confidence=0.0,
                actual="",
                reasoning="no image",
                output_type="verify_object_identity_result",
                description="VLM verification result for whether a marked point shows the expected object.",
            )
            return {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": "no image",
            }

        h, w = rgb.shape[:2]
        x, y = int(target_pixel[0]), int(target_pixel[1])
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        x0 = max(0, x - crop_radius)
        y0 = max(0, y - crop_radius)
        x1 = min(w, x + crop_radius)
        y1 = min(h, y + crop_radius)
        crop = rgb[y0:y1, x0:x1].copy()

        annotated = rgb.copy()
        try:
            from PIL import Image as _PILImage, ImageDraw as _PILDraw
            _img = _PILImage.fromarray(annotated)
            _draw = _PILDraw.Draw(_img)
            _draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(0, 255, 0), width=3)
            _draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 255, 0), width=2)
            annotated = np.array(_img)
        except Exception:
            pass

        annotated_path = self._save_rgb_artifact(f"verify_object_identity_{expected_object}_scene", annotated)
        crop_path = self._save_rgb_artifact(f"verify_object_identity_{expected_object}_crop", crop)
        crop_url = image_to_data_url(crop)
        annotated_url = image_to_data_url(annotated)
        if not crop_url or not annotated_url:
            self._record_runtime_diagnostic(
                "verify_object_identity",
                expected_object=expected_object,
                target_pixel=[x, y],
                crop_radius=int(crop_radius),
                annotated_path=annotated_path,
                crop_path=crop_path,
                verified=False,
                confidence=0.0,
                actual="",
                reasoning="encode_failed",
                output_type="verify_object_identity_result",
                description="VLM verification result for whether a marked point shows the expected object.",
            )
            return {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": "encode_failed",
            }

        system = (
            "You are a vision-grounding verifier for a robot manipulation system. "
            "You will be given two images: (1) the full scene with a green dot/box "
            "marking a candidate point, and (2) a tight crop around that point. "
            "Decide whether the marked object is what the robot intends to grasp. "
            "Reply ONLY with a compact JSON object."
        )
        user = (
            f"Robot intends to grasp: '{expected_object}'.\n"
            "Image 1 (full scene with green marker) and Image 2 (crop) are below.\n"
            "Reply with JSON exactly of the form: "
            '{"verified": <true|false>, "confidence": <0-1 float>, '
            '"actual": "<what is at the marker>", "reasoning": "<one short sentence>"}.'
        )
        try:
            raw = query_llm_text(
                system, user,
                images=[annotated_url, crop_url],
                model="google/gemini-3.1-pro-preview",
                temperature=0.0,
                max_tokens=8096,
                reasoning_effort="low",
                json_mode=True,
            )
            import json as _json
            from rats.agents.base_agent import strip_markdown_json_fences
            # Gemini-3.1-pro emits JSON wrapped in ``​```json … ```​``
            # markdown fences despite json_mode=True (614/621 calls on the
            # v7 audit). Strip the wrapper before json.loads.
            parsed = _json.loads(strip_markdown_json_fences(raw))
            result_payload = {
                "verified": bool(parsed.get("verified", False)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "actual": str(parsed.get("actual", "")),
                "reasoning": str(parsed.get("reasoning", "")),
            }
            self._record_runtime_diagnostic(
                "verify_object_identity",
                expected_object=expected_object,
                target_pixel=[x, y],
                crop_radius=int(crop_radius),
                annotated_path=annotated_path,
                crop_path=crop_path,
                output_type="verify_object_identity_result",
                description="VLM verification result for whether a marked point shows the expected object.",
                **result_payload,
            )
            return result_payload
        except Exception as e:
            result_payload = {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": f"vlm_error: {e}",
            }
            self._record_runtime_diagnostic(
                "verify_object_identity",
                expected_object=expected_object,
                target_pixel=[x, y],
                crop_radius=int(crop_radius),
                annotated_path=annotated_path,
                crop_path=crop_path,
                output_type="verify_object_identity_result",
                description="VLM verification result for whether a marked point shows the expected object.",
                **result_payload,
            )
            return result_payload

    def vlm_verify(
        self,
        rgb: np.ndarray,
        target_description: str,
        mask: Any | None = None,
        *,
        question: str | None = None,
        target_pixel: tuple[int, int] | None = None,
        crop_padding: int = 32,
    ) -> dict[str, Any]:
        """General VLM verification helper for MolmoSpaces policy code.

        SAM3 mask validation: validate a candidate SAM3 segmentation before
        converting it to world points. Pass the image, target object text, and
        the SAM3 mask; the VLM sees both the full image with the mask
        highlighted and the same full image with no mask overlay. Use the first
        verified mask rather than blindly trusting SAM3 rank 1.

        Args:
            rgb: (H, W, 3) uint8 RGB image.
            target_description: Object or predicate to verify, e.g. "oven handle".
            mask: SAM3 mask array or SAM3 result dict containing "mask". Either
                this OR ``target_pixel`` is required — without one, the VLM
                receives two identical unannotated images and the call is
                refused.
            question: Optional custom verification question. When omitted, asks
                whether the highlighted mask corresponds to target_description.
            target_pixel: (x, y) point to mark when no mask is provided. Either
                this OR ``mask`` is required.
            crop_padding: Pixel padding around mask/point crop.

        Returns:
            {"verified": bool, "confidence": float, "actual": str,
             "reasoning": str}. If neither ``mask`` nor ``target_pixel`` is
            supplied the call short-circuits with ``verified=False`` —
            localize the target with SAM3 or Molmo first.

        Example:
            >>> masks = segment_sam3_text_prompt(rgb, "oven handle")
            >>> for cand in masks[:3]:
            ...     v = vlm_verify(rgb, "oven handle", mask=cand["mask"])
            ...     if v["verified"]:
            ...         mask = cand["mask"]
            ...         break
        """
        from rats.agents.base_agent import image_to_data_url, query_llm_text
        import json as _json

        if rgb is None or not hasattr(rgb, "shape") or len(rgb.shape) != 3:
            self._record_runtime_diagnostic(
                "vlm_verify",
                target_description=target_description,
                verified=False,
                confidence=0.0,
                actual="",
                reasoning="no image",
                output_type="verify_object_style_primitive_result",
                description="VLM verification result for a visual predicate, mask, or marked point.",
            )
            return {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": "no image",
            }

        raw_mask_probe = mask.get("mask") if isinstance(mask, dict) else mask
        if raw_mask_probe is None and target_pixel is None:
            reasoning = (
                "no mask or target_pixel supplied; vlm_verify requires a SAM3 mask "
                "or a target pixel to ground the highlighted region — localize first "
                "(e.g. segment_sam3_text_prompt or point_prompt_molmo) and pass the "
                "result via mask=... or target_pixel=..."
            )
            self._record_runtime_diagnostic(
                "vlm_verify",
                target_description=target_description,
                verified=False,
                confidence=0.0,
                actual="",
                reasoning=reasoning,
                output_type="verify_object_style_primitive_result",
                description="VLM verification result for a visual predicate, mask, or marked point.",
            )
            return {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": reasoning,
            }

        image = np.asarray(rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        h, w = image.shape[:2]

        raw_mask = mask.get("mask") if isinstance(mask, dict) else mask
        mask_bool: np.ndarray | None = None
        bbox: tuple[int, int, int, int] | None = None
        if raw_mask is not None:
            candidate = np.asarray(raw_mask)
            if candidate.ndim == 3:
                candidate = candidate.squeeze()
            if candidate.shape != (h, w):
                self._record_runtime_diagnostic(
                    "vlm_verify",
                    target_description=target_description,
                    verified=False,
                    confidence=0.0,
                    actual="",
                    reasoning=f"mask_shape_mismatch: expected {(h, w)}, got {candidate.shape}",
                    output_type="verify_object_style_primitive_result",
                    description="VLM verification result for a visual predicate, mask, or marked point.",
                )
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "actual": "",
                    "reasoning": f"mask_shape_mismatch: expected {(h, w)}, got {candidate.shape}",
                }
            mask_bool = candidate.astype(bool)
            ys, xs = np.where(mask_bool)
            if len(xs) == 0:
                self._record_runtime_diagnostic(
                    "vlm_verify",
                    target_description=target_description,
                    verified=False,
                    confidence=0.0,
                    actual="",
                    reasoning="empty_mask",
                    output_type="verify_object_style_primitive_result",
                    description="VLM verification result for a visual predicate, mask, or marked point.",
                )
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "actual": "",
                    "reasoning": "empty_mask",
                }
            pad = max(0, int(crop_padding))
            x0 = max(0, int(xs.min()) - pad)
            y0 = max(0, int(ys.min()) - pad)
            x1 = min(w, int(xs.max()) + pad + 1)
            y1 = min(h, int(ys.max()) + pad + 1)
            bbox = (x0, y0, x1, y1)
        elif target_pixel is not None:
            x = max(0, min(w - 1, int(target_pixel[0])))
            y = max(0, min(h - 1, int(target_pixel[1])))
            pad = max(1, int(crop_padding))
            bbox = (max(0, x - pad), max(0, y - pad), min(w, x + pad + 1), min(h, y + pad + 1))

        annotated = image.copy()
        try:
            from PIL import Image as _PILImage, ImageDraw as _PILDraw

            if mask_bool is not None:
                overlay = annotated.copy()
                overlay[mask_bool] = np.array([255, 48, 32], dtype=np.uint8)
                annotated = (0.55 * annotated + 0.45 * overlay).astype(np.uint8)
            pil = _PILImage.fromarray(annotated)
            draw = _PILDraw.Draw(pil)
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 255, 0), width=3)
            if target_pixel is not None:
                x = max(0, min(w - 1, int(target_pixel[0])))
                y = max(0, min(h - 1, int(target_pixel[1])))
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(0, 255, 0), width=3)
            annotated = np.array(pil)
        except Exception:
            pass

        annotated_path = self._save_rgb_artifact(f"vlm_verify_{target_description}_scene", annotated)
        unmasked_path = self._save_rgb_artifact(f"vlm_verify_{target_description}_unmasked_scene", image)
        crop_path = None
        images = [image_to_data_url(annotated), image_to_data_url(image)]
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            crop = annotated[y0:y1, x0:x1].copy()
            crop_path = self._save_rgb_artifact(f"vlm_verify_{target_description}_crop", crop)
        images = [u for u in images if u]
        if not images:
            self._record_runtime_diagnostic(
                "vlm_verify",
                target_description=target_description,
                target_pixel=target_pixel,
                bbox=bbox,
                annotated_path=annotated_path,
                unmasked_path=unmasked_path,
                crop_path=crop_path,
                verified=False,
                confidence=0.0,
                actual="",
                reasoning="encode_failed",
                output_type="verify_object_style_primitive_result",
                description="VLM verification result for a visual predicate, mask, or marked point.",
            )
            return {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": "encode_failed",
            }

        if question:
            task = question
        elif mask_bool is not None:
            task = (
                f"Does the red highlighted segmentation mask correspond to the target "
                f"object: '{target_description}'?"
            )
        else:
            task = f"Verify this visual predicate: '{target_description}'."

        system = (
            "You are a strict vision verifier for a robot manipulation policy. "
            "If a segmentation mask is highlighted, judge the highlighted region, "
            "not just whether the target exists somewhere in the image. Mark true "
            "only when the visual evidence directly supports the requested target. "
            "Reply ONLY with compact JSON."
        )
        user = (
            f"{task}\n\n"
            "Image 1 is the full scene with the candidate region highlighted "
            "in red/yellow. Image 2 is the same full scene with no mask or "
            "highlight overlay; use it to identify the actual object appearance "
            "inside the highlighted region.\n\n"
            "Reply with JSON exactly of the form: "
            '{"verified": <true|false>, "confidence": <0-1 float>, '
            '"actual": "<what the highlighted region/predicate shows>", '
            '"reasoning": "<one short sentence>"}.'
        )
        try:
            raw = query_llm_text(
                system,
                user,
                images=images,
                model="google/gemini-3.1-pro-preview",
                temperature=0.0,
                max_tokens=8096,
                reasoning_effort="low",
                json_mode=True,
            )
            from rats.agents.base_agent import strip_markdown_json_fences
            # Gemini-3.1-pro emits JSON wrapped in markdown fences despite
            # json_mode=True (614/621 calls on the v7 audit). Strip the
            # wrapper before json.loads.
            parsed = _json.loads(strip_markdown_json_fences(raw))
            result_payload = {
                "verified": bool(parsed.get("verified", False)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "actual": str(parsed.get("actual", "")),
                "reasoning": str(parsed.get("reasoning", "")),
            }
            self._record_runtime_diagnostic(
                "vlm_verify",
                target_description=target_description,
                target_pixel=target_pixel,
                bbox=bbox,
                annotated_path=annotated_path,
                unmasked_path=unmasked_path,
                crop_path=crop_path,
                has_mask=mask_bool is not None,
                output_type="verify_object_style_primitive_result",
                description="VLM verification result for a visual predicate, mask, or marked point.",
                **result_payload,
            )
            return result_payload
        except Exception as e:
            result_payload = {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": f"vlm_error: {type(e).__name__}",
            }
            self._record_runtime_diagnostic(
                "vlm_verify",
                target_description=target_description,
                target_pixel=target_pixel,
                bbox=bbox,
                annotated_path=annotated_path,
                crop_path=crop_path,
                has_mask=mask_bool is not None,
                output_type="verify_object_style_primitive_result",
                description="VLM verification result for a visual predicate, mask, or marked point.",
                **result_payload,
            )
            return result_payload

    def grasp_with_wrist_closeloop(
        self,
        object_name: str,
        verify_label: str | None = None,
        max_retries: int = 2,
        approach_height: float = 0.10,
    ) -> dict[str, Any]:
        """Top-down grasp with wrist-camera refinement and post-lift verification.

        Pipeline:
          1) Coarse agentview localize: Molmo point → SAM3 mask → centroid → depth
             → world point.
          2) Move wrist camera to hover above the coarse estimate and re-localize
             in the wrist view.
          3) Optional VLM identity check on the wrist crop if `verify_label` is set.
          4) open_gripper → explicit hover → explicit descend → close_gripper → lift.
          5) Post-lift wrist-view check: re-run Molmo; if the object is still in
             frame the grasp is confirmed. Otherwise re-open, shift target 1 cm
             laterally, retry up to `max_retries` times.

        Args:
            object_name: Molmo prompt (e.g. "red mug").
            verify_label: Optional tight label for the VLM identity check.
            max_retries: Extra post-lift retries with lateral perturbation.
            approach_height: Pre-grasp hover height (meters).

        Returns:
            {"success": bool, "position": list|None, "attempts": int,
             "failure_mode": str, "reasoning": str}.
        """
        topdown_quat = self.rotation_matrix_to_quaternion(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=np.float64,
            )
        )

        def _pick_first_xy(pts: Any) -> tuple[int, int] | None:
            if not pts:
                return None
            if isinstance(pts, dict):
                iterable = pts.values()
            elif isinstance(pts, (list, tuple)):
                iterable = pts
            else:
                return None
            for v in iterable:
                if (
                    isinstance(v, (tuple, list))
                    and len(v) >= 2
                    and v[0] is not None
                    and v[1] is not None
                ):
                    return int(v[0]), int(v[1])
            return None

        def _mask_centroid_world(
            rgb: np.ndarray,
            depth: np.ndarray,
            K: np.ndarray,
            T: np.ndarray,
            px: int,
            py: int,
        ) -> tuple[np.ndarray, int] | None:
            masks = self.segment_sam3_point_prompt(rgb, (float(px), float(py)))
            if not masks:
                return None
            best = max(masks, key=lambda m: float(m.get("score", 0.0)))
            mask = best["mask"].astype(bool)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return None
            u = int(np.round(xs.mean()))
            v = int(np.round(ys.mean()))
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            valid = mask & (depth > 0.02) & (depth < 5.0)
            z = float(np.median(depth[valid])) if valid.any() else float(depth[v, u])
            return self.pixel_to_world_point(u, v, z, K, T), int(mask.sum())

        # Stage 1: coarse agentview localization.
        try:
            obs = self.get_observation()
            agent = obs[self.camera_name]
            a_rgb = agent["images"]["rgb"]
            a_depth = agent["images"]["depth"]
            a_K = agent["intrinsics"]
            a_T = agent["pose_mat"]
        except Exception as e:
            return {
                "success": False,
                "position": None,
                "attempts": 0,
                "failure_mode": "observation_error",
                "reasoning": f"get_observation failed: {e}",
            }
        a_pts = self.point_prompt_molmo(a_rgb, object_name)
        a_xy = _pick_first_xy(a_pts)
        if a_xy is None:
            return {
                "success": False,
                "position": None,
                "attempts": 0,
                "failure_mode": "molmo_none",
                "reasoning": f"Molmo returned no point for '{object_name}' in agentview",
            }
        coarse = _mask_centroid_world(a_rgb, a_depth, a_K, a_T, a_xy[0], a_xy[1])
        if coarse is None:
            return {
                "success": False,
                "position": None,
                "attempts": 0,
                "failure_mode": "sam3_none",
                "reasoning": f"SAM3 segmentation failed on agentview for '{object_name}'",
            }
        coarse_pos, _ = coarse
        fine_pos = coarse_pos

        # Stage 2: wrist-camera refinement.
        inspect = self.inspect_at_wrist(coarse_pos, hover_height=approach_height)
        if inspect.get("ok", False):
            wrist = inspect.get("wrist", {})
            w_rgb = wrist.get("rgb")
            w_depth = wrist.get("depth")
            w_K = wrist.get("intrinsics")
            w_T = wrist.get("pose_mat")
            if (
                w_rgb is not None
                and w_depth is not None
                and w_K is not None
                and w_T is not None
            ):
                w_pts = self.point_prompt_molmo(w_rgb, object_name)
                w_xy = _pick_first_xy(w_pts)
                if w_xy is not None:
                    if verify_label:
                        verdict = self.verify_object_identity(
                            w_rgb, (w_xy[0], w_xy[1]), verify_label, crop_radius=80,
                        )
                        if not verdict.get("verified", False):
                            return {
                                "success": False,
                                "position": fine_pos.tolist()
                                if isinstance(fine_pos, np.ndarray)
                                else fine_pos,
                                "attempts": 0,
                                "failure_mode": "identity_mismatch",
                                "reasoning": (
                                    f"wrist VLM said '{verdict.get('actual','')}' "
                                    f"not '{verify_label}'"
                                ),
                            }
                    w_refined = _mask_centroid_world(
                        w_rgb, w_depth, w_K, w_T, w_xy[0], w_xy[1],
                    )
                    if w_refined is not None:
                        fine_pos = w_refined[0]

        # Stage 3: close-loop grasp with retries.
        last_failure = "ok"
        for attempt_idx in range(max_retries + 1):
            perturb = np.zeros(3, dtype=np.float64)
            if attempt_idx > 0:
                angle = attempt_idx * (2.0 * np.pi / max(1, (max_retries + 1)))
                perturb = 0.01 * np.array([np.cos(angle), np.sin(angle), 0.0])
            target_pos = np.asarray(fine_pos, dtype=np.float64) + perturb
            try:
                self.open_gripper()
                self.goto_pose(
                    target_pos + np.array([0.0, 0.0, approach_height]), topdown_quat,
                )
                self.goto_pose(target_pos, topdown_quat)
                self.close_gripper()
                lift_pos = target_pos + np.array([0.0, 0.0, 0.05])
                self.goto_pose(lift_pos, topdown_quat)
            except Exception as e:
                last_failure = "ik_fail"
                print(
                    f"[grasp_with_wrist_closeloop] IK/motion error on attempt "
                    f"{attempt_idx}: {e}"
                )
                continue

            try:
                post_obs = self.get_observation()
                post_w_rgb = post_obs[self.wrist_camera_name]["images"]["rgb"]
                held_pts = self.point_prompt_molmo(post_w_rgb, object_name)
                if _pick_first_xy(held_pts) is not None:
                    return {
                        "success": True,
                        "position": fine_pos.tolist()
                        if isinstance(fine_pos, np.ndarray)
                        else fine_pos,
                        "attempts": attempt_idx + 1,
                        "failure_mode": "ok",
                        "reasoning": "post-lift wrist view shows target; grasp confirmed",
                    }
            except Exception as e:
                print(f"[grasp_with_wrist_closeloop] post-lift check error: {e}")
            last_failure = "post_grasp_empty_jaws"

        return {
            "success": False,
            "position": fine_pos.tolist() if isinstance(fine_pos, np.ndarray) else fine_pos,
            "attempts": max_retries + 1,
            "failure_mode": last_failure,
            "reasoning": f"{max_retries + 1} close-loop grasp iterations; last={last_failure}",
        }
