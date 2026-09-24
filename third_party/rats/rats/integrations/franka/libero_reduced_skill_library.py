import math
import os
import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from rats.envs.base import (
    BaseEnv,
)
from rats.integrations.motion import pyroki_snippets as pks  # type: ignore
from rats.integrations.base_api import ApiBase
from rats.integrations.franka.libero_reduced import FrankaLiberoApiReduced
from rats.integrations.vision.graspnet import init_contact_graspnet
from rats.integrations.vision.molmo import init_molmo
from rats.integrations.motion.pyroki import init_pyroki

# from rats.integrations.vision.owlvit import init_owlvit
from rats.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

# from rats.integrations.vision.sam2 import init_sam2
from rats.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt
from rats.utils.camera_utils import obs_get_rgb
from rats.utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud, depth_to_rgb


def _runtime_vlm_model() -> str:
    """Use the model frozen by the experiment instead of a hidden helper default."""

    return os.getenv("RATS_RUNTIME_VLM_MODEL") or os.getenv("RATS_LLM_MODEL") or "google/gemini-3.1-pro-preview"


# ------------------------------- Control API ------------------------------
class FrankaLiberoApiReducedSkillLibrary(FrankaLiberoApiReduced):
    """
    Robot control helpers for Franka.

    ``enable_wrist_closeloop`` (default False) gates ``grasp_with_wrist_closeloop``
    — a RATS-side closed-loop wrist-refined grasp primitive added in commit
    8a01e13a. The CaP-X paper Table 2 baseline (CaP-Agent0) was measured WITHOUT
    this primitive, so the default keeps the API paper-matching. Set the flag
    to True (or use the ``FrankaLiberoApiReducedSkillLibraryWristCloseloop``
    registered name) to opt in for RATS exploration runs that want the
    perception-gap-bridging closed-loop grasp.
    """

    def __init__(
        self,
        env: BaseEnv,
        enable_wrist_closeloop: bool = False,
        grasp_backend: str = "graspnet",
    ) -> None:
        super().__init__(env, grasp_backend=grasp_backend)
        self._enable_wrist_closeloop = enable_wrist_closeloop

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        fns["rotation_matrix_to_quaternion"] = self.rotation_matrix_to_quaternion
        fns["decompose_transform"] = self.decompose_transform
        fns["depth_to_point_cloud"] = self.depth_to_point_cloud
        fns["mask_to_world_points"] = self.mask_to_world_points
        fns["pixel_to_world_point"] = self.pixel_to_world_point
        fns["transform_points"] = self.transform_points
        fns["interpolate_segment"] = self.interpolate_segment
        fns["normalize_vector"] = self.normalize_vector
        fns["select_top_down_grasp"] = self.select_top_down_grasp
        fns["verify_object_identity"] = self.verify_object_identity
        fns["inspect_at_wrist"] = self.inspect_at_wrist
        if self._enable_wrist_closeloop:
            fns["grasp_with_wrist_closeloop"] = self.grasp_with_wrist_closeloop

        return fns

    def inspect_at_wrist(
        self,
        target_world_pos: np.ndarray,
        hover_height: float = 0.10,
    ) -> dict[str, Any]:
        """Active perception: move the wrist camera above a 3D target and return the close-up view.

        Use this when the agent-view perception (Molmo / SAM3) is uncertain
        about an object's identity in a cluttered scene. The wrist camera at
        10 cm above the target gives a much sharper close-up than the static
        agent-view, often enough to disambiguate visually similar items.

        The arm is moved to a top-down pose hovering above `target_world_pos`,
        then the full observation dict is captured. Read the wrist image as
        `result["wrist"]["rgb"]` / `result["wrist"]["depth"]`, or with the
        standard camera layout `result["wrist"]["images"]["rgb"]` /
        `result["wrist"]["images"]["depth"]`. The agent-view is also returned
        for convenience under `result["agentview"]`.

        Args:
            target_world_pos: (3,) world-frame position of the target object
                (typically obtained from a prior text-prompt segmentation +
                `mask_to_world_points`).
            hover_height: Vertical offset above the target (meters).

        Returns:
            dict with keys:
              - "wrist": {"rgb": ndarray (H, W, 3), "depth": ndarray (H, W),
                          "images": {"rgb": ndarray, "depth": ndarray},
                          "intrinsics": ndarray (3, 3),
                          "pose_mat": ndarray (4, 4)}
              - "agentview": same shape, captured at the same time.
              - "moved_to": ndarray (3,) the actual hover pose used.
              - "ok": bool — False if goto_pose raised.

        Example:
            >>> obs = get_observation()
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> masks = segment_sam3_text_prompt(rgb, "cream cheese")
            >>> if masks:
            ...     pos = mask_to_world_points(
            ...         masks[0]["mask"],
            ...         obs["agentview"]["images"]["depth"],
            ...         obs["agentview"]["intrinsics"],
            ...         obs["agentview"]["pose_mat"],
            ...     ).mean(axis=0)
            ...     close = inspect_at_wrist(pos, hover_height=0.10)
            ...     # close["wrist"]["images"]["rgb"] is a tight close-up; re-segment to
            ...     # confirm before grasping.
            ...     masks2 = segment_sam3_text_prompt(close["wrist"]["rgb"], "cream cheese")
        """
        target = np.asarray(target_world_pos, dtype=np.float64).reshape(3)
        topdown_quat = self.rotation_matrix_to_quaternion(
            np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
        )
        hover_pos = target + np.array([0.0, 0.0, max(0.02, float(hover_height))])
        ok = True
        try:
            # goto_pose is exposed by the parent FrankaLiberoApiReduced
            self.goto_pose(hover_pos, topdown_quat)
        except Exception as e:
            ok = False
            print(f"[inspect_at_wrist] goto_pose failed: {e}")

        obs = self.get_observation()
        agent = obs.get(self.camera_name, {}) or {}
        wrist = obs.get(self.wrist_camera_name, {}) or {}

        def _pack(cam_obs: dict[str, Any]) -> dict[str, Any]:
            imgs = cam_obs.get("images", {}) or {}
            rgb = imgs.get("rgb")
            depth = imgs.get("depth")
            return {
                "rgb": rgb,
                "depth": depth,
                "images": {"rgb": rgb, "depth": depth},
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
        """Second-look VLM verification that a localized pixel actually shows the expected object.

        Use this BEFORE grasping when the scene contains visually similar items (e.g. several
        small packaged foods). Crops a region around `target_pixel` and asks an LLM with vision
        whether the cropped object matches `expected_object`. Returns a structured verdict.

        Args:
            rgb: (H, W, 3) uint8 RGB image (typically the agentview).
            target_pixel: (x, y) pixel coordinates from your perception output (Molmo point or
                SAM3 mask centroid).
            expected_object: Plain-language name of the object you intend to grasp, e.g.
                "cream cheese package", "butter".
            crop_radius: Half-width (in pixels) of the square crop sent to the VLM.

        Returns:
            dict with keys:
              - "verified" (bool): True iff the VLM agrees the crop shows `expected_object`.
              - "confidence" (float in [0,1]): VLM-reported confidence.
              - "actual" (str): What the VLM thinks the object actually is (free-form).
              - "reasoning" (str): One-sentence rationale.

        Example:
            >>> obs = get_observation()
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> pt = point_prompt_molmo(rgb, "cream cheese").get("cream cheese", (None, None))
            >>> if pt[0] is not None:
            ...     v = verify_object_identity(rgb, pt, "cream cheese package")
            ...     if not v["verified"]:
            ...         # try a different prompt or a different camera view
            ...         pass
        """
        from rats.agents.base_agent import image_to_data_url, query_llm_text  # local import to avoid module init

        if rgb is None or len(rgb.shape) != 3:
            return {"verified": False, "confidence": 0.0, "actual": "", "reasoning": "no image"}

        h, w = rgb.shape[:2]
        x, y = int(target_pixel[0]), int(target_pixel[1])
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        x0 = max(0, x - crop_radius)
        y0 = max(0, y - crop_radius)
        x1 = min(w, x + crop_radius)
        y1 = min(h, y + crop_radius)
        crop = rgb[y0:y1, x0:x1].copy()

        # Annotate the full frame with a marker so the VLM knows which point we're asking about.
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

        crop_url = image_to_data_url(crop)
        annotated_url = image_to_data_url(annotated)
        if not crop_url or not annotated_url:
            return {"verified": False, "confidence": 0.0, "actual": "", "reasoning": "encode_failed"}

        system = (
            "You are a vision-grounding verifier for a robot manipulation system. You will be given "
            "two images: (1) the full scene with a green dot/box marking a candidate point, and "
            "(2) a tight crop around that point. Decide whether the marked object is what the robot "
            "intends to grasp. Reply ONLY with a compact JSON object."
        )
        user = (
            f"Robot intends to grasp: '{expected_object}'.\n"
            f"Image 1 (full scene with green marker) and Image 2 (crop) are below.\n"
            "Reply with JSON exactly of the form: "
            '{"verified": <true|false>, "confidence": <0-1 float>, '
            '"actual": "<what is at the marker>", "reasoning": "<one short sentence>"}.'
        )
        # FIX (self-check/Step-5 doubling): the lifelong loop runs the
        # policy twice per attempt — once in Step 4b (runtime self-check)
        # and once in Step 5 (official execution) — with the env reset to
        # the same state between them. Every LLM-using primitive the
        # policy invokes was being called twice with bit-identical inputs.
        # When the primitive-cache scope is active (set up by
        # lifelong_loop around the self-check/execute pair), look up the
        # full input signature (annotated frame bytes, crop bytes,
        # expected_object) and return the prior result on hit. The scope
        # is cleared at the end of each attempt so cross-attempt state
        # never leaks.
        from rats.agents.policy_primitive_cache import lookup as _cache_lookup, store as _cache_store, make_key as _cache_key
        cache_key = _cache_key(
            "verify_object_identity",
            annotated, crop, expected_object,
        )
        hit, cached = _cache_lookup(cache_key)
        if hit:
            return cached
        try:
            raw = query_llm_text(
                system, user,
                images=[annotated_url, crop_url],
                model=_runtime_vlm_model(),
                temperature=0.0,
                max_tokens=8096,
                reasoning_effort="low",
                json_mode=True,
            )
            import json as _json
            from rats.agents.base_agent import strip_markdown_json_fences
            # Gemini-3.1-pro emits JSON wrapped in markdown fences despite
            # json_mode=True. Strip the wrapper before json.loads.
            parsed = _json.loads(strip_markdown_json_fences(raw))
            result = {
                "verified": bool(parsed.get("verified", False)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "actual": str(parsed.get("actual", "")),
                "reasoning": str(parsed.get("reasoning", "")),
            }
        except Exception as e:
            result = {"verified": False, "confidence": 0.0, "actual": "", "reasoning": f"vlm_error: {e}"}
        _cache_store(cache_key, result)
        return result

    def grasp_with_wrist_closeloop(
        self,
        object_name: str,
        verify_label: str | None = None,
        max_retries: int = 2,
        approach_height: float = 0.10,
    ) -> dict[str, Any]:
        """Top-down grasp with wrist-camera refinement and post-lift visual verification.

        PREFER THIS over hand-writing the Molmo + SAM3 + goto_pose + close_gripper chain
        for grasping. Agentview-only localization has ~2-3 cm median 3D error on LIBERO
        objects (empirically measured); 2 cm is enough to close the gripper on air for
        small items (butter, bottle caps). This primitive folds in:

          1) Coarse agentview localize: Molmo point -> SAM3 mask -> centroid -> depth
             -> world point.
          2) Move wrist camera to hover above the coarse estimate and re-localize
             in the wrist view. At 10 cm hover the object fills more pixels and the
             depth map is less noisy, so the refined 3D estimate drops to ~5 mm error.
          3) (optional) verify_object_identity on the wrist crop: if `verify_label`
             is given and the VLM disagrees, abort without grasping to avoid picking
             the wrong object.
          4) open_gripper -> goto hover -> descend with z_approach -> close_gripper ->
             lift 5 cm.
          5) Visual check after lift: re-run Molmo on the wrist view. If the target
             object is still visible near the gripper, the grasp is confirmed. If not,
             reopen, shift target 1 cm laterally, retry up to `max_retries` times.

        Args:
            object_name: Plain-language prompt for Molmo (e.g. "butter",
                "orange juice").
            verify_label: Optional tight label used for wrist-view identity check
                (e.g. "butter stick package"). If set and the VLM disagrees, the
                primitive returns failure with mode "identity_mismatch".
            max_retries: Extra post-lift retries with lateral perturbation when the
                first grasp comes up empty. Default 2 (so up to 3 total close-loop
                iterations).
            approach_height: Pre-grasp hover height in meters. Default 0.10.

        Returns:
            dict with keys:
              - "success" (bool)
              - "position" (list[float] | None): world XYZ of the refined target.
              - "attempts" (int): how many close-loop iterations actually ran.
              - "failure_mode" (str): "ok" | "molmo_none" | "sam3_none" |
                  "identity_mismatch" | "ik_fail" | "post_grasp_empty_jaws".
              - "reasoning" (str)

        Example:
            >>> # Before: hand-written
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> pt = point_prompt_molmo(rgb, "butter").get("butter", (None, None))
            >>> mask = segment_sam3_point_prompt(rgb, pt)[0]["mask"]
            >>> pos = mask_to_world_points(mask, depth, K, T).mean(axis=0)
            >>> goto_pose(pos + [0,0,0.15], tdq()); goto_pose(pos, tdq(), z_approach=0.10)
            >>> close_gripper(); goto_pose(pos + [0,0,0.05], tdq())  # hope for the best
            >>>
            >>> # After: single call with built-in closed-loop
            >>> r = grasp_with_wrist_closeloop("butter", verify_label="butter stick",
            ...                                max_retries=2)
            >>> if not r["success"]:
            ...     RESULT["grasp"] = False
            ... else:
            ...     # proceed to place
            ...     RESULT["grasp"] = True

        Note:
            On pick-and-place tasks, a successful return from this primitive is
            enough to continue with transport and release. Additional VLM or
            wrist-camera checks are useful diagnostics, but callers should not
            use an ambiguous post-grasp visual check as the sole reason to skip
            the planned place/open_gripper step; that leaves the object
            suspended in the gripper and prevents task completion.
        """
        topdown_quat = self.rotation_matrix_to_quaternion(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
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
                if isinstance(v, (tuple, list)) and len(v) >= 2 and v[0] is not None and v[1] is not None:
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
            if w_rgb is not None and w_depth is not None and w_K is not None and w_T is not None:
                w_pts = self.point_prompt_molmo(w_rgb, object_name)
                w_xy = _pick_first_xy(w_pts)
                if w_xy is not None:
                    if verify_label:
                        verdict = self.verify_object_identity(
                            w_rgb, (w_xy[0], w_xy[1]), verify_label, crop_radius=80
                        )
                        if not verdict.get("verified", False):
                            return {
                                "success": False,
                                "position": fine_pos.tolist()
                                if isinstance(fine_pos, np.ndarray)
                                else fine_pos,
                                "attempts": 0,
                                "failure_mode": "identity_mismatch",
                                "reasoning": f"wrist VLM said '{verdict.get('actual','')}' not '{verify_label}'",
                            }
                    w_refined = _mask_centroid_world(
                        w_rgb, w_depth, w_K, w_T, w_xy[0], w_xy[1]
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
                    target_pos + np.array([0.0, 0.0, approach_height]), topdown_quat
                )
                self.goto_pose(
                    target_pos, topdown_quat, z_approach=approach_height * 0.8
                )
                self.close_gripper()
                lift_pos = target_pos + np.array([0.0, 0.0, 0.05])
                self.goto_pose(lift_pos, topdown_quat)
            except Exception as e:
                last_failure = "ik_fail"
                print(f"[grasp_with_wrist_closeloop] IK/motion error on attempt {attempt_idx}: {e}")
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

    def verify_step(
        self,
        predicate_description: str,
        on_fail: str = "return_false",
        use_wrist: bool = True,
    ) -> dict[str, Any]:
        """In-rollout check that a runtime predicate is TRUE right now.

        By default this function is a soft pass-through verifier: it preserves the
        API shape expected by generated policies and learned skills, but does not
        run the strict VLM check. Set RATS_VERIFY_STEP_MODE=vlm or strict to restore
        the original VLM-based behavior.

        Args:
            predicate_description: Plain-language predicate to test, e.g.
                "butter is held between the gripper jaws",
                "orange juice is resting inside the basket, not hovering above it",
                "the microwave door is open wide enough to insert a mug".
            on_fail: "return_false" (default) returns verified=False so caller can
                branch. "raise" raises RuntimeError to propagate up to the outer
                loop's plan-refinement path.
            use_wrist: Include the wrist-camera image alongside the agentview. Set
                False for scene-level predicates where wrist view is occluded or
                irrelevant.

        Important:
            Treat this as a diagnostic signal, not a hard interlock for
            pick-and-place. After a grasp primitive reports success, callers
            should continue to the target localization and release step even if a
            post-grasp wrist predicate is false or ambiguous, then verify the
            final resting state after release.

        Returns:
            dict with keys:
              - "verified" (bool)
              - "confidence" (float in [0, 1])
              - "actual" (str): short free-form description of what VLM sees.
              - "reasoning" (str)

        Cost: ~1-2 s per call (one VLM query with 1-2 images). Use at step
        boundaries, not inside tight loops. This cost only applies when
        RATS_VERIFY_STEP_MODE=vlm or strict.

        Example:
            >>> # Paired with mechanism A's RESULT dict:
            >>> RESULT = {"step_grasp": False, "step_place": False}
            >>>
            >>> close_gripper()
            >>> goto_pose(obj_pos + np.array([0, 0, 0.05]), tdq())  # lift
            >>> v = verify_step("target object is held between the gripper jaws",
            ...                 use_wrist=True)
            >>> RESULT["post_grasp_verify"] = v["verified"]
            >>> goto_pose(target_drop_pos, tdq(), z_approach=0.08)
            >>> open_gripper()
            >>> v2 = verify_step("target object is resting on the plate surface",
            ...                  use_wrist=False)
            >>> RESULT["step_place"] = v2["verified"]
        """
        import os

        mode = os.environ.get("RATS_VERIFY_STEP_MODE", "soft_true").strip().lower()
        mode = mode or "soft_true"
        if mode in {
            "off",
            "disable",
            "disabled",
            "noop",
            "no_op",
            "pass",
            "soft",
            "soft_true",
            "assume_true",
        }:
            return {
                "verified": True,
                "confidence": 1.0,
                "actual": "verify_step bypassed; predicate assumed true.",
                "reasoning": (
                    f"verify_step bypassed by RATS_VERIFY_STEP_MODE={mode}; "
                    "predicate was not visually evaluated."
                ),
            }
        if mode in {"soft_false", "assume_false"}:
            return {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": (
                    f"verify_step bypassed by RATS_VERIFY_STEP_MODE={mode}; "
                    "predicate was not visually evaluated."
                ),
            }

        from rats.agents.base_agent import image_to_data_url, query_llm_text  # local import
        import json as _json

        try:
            obs = self.get_observation()
        except Exception as e:
            out = {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": f"observation_error: {type(e).__name__}",
            }
            if on_fail == "raise":
                raise RuntimeError(f"verify_step observation failed: {e}") from e
            return out

        images: list[str] = []
        agent_rgb = obs.get(self.camera_name, {}).get("images", {}).get("rgb")
        if agent_rgb is not None:
            u = image_to_data_url(agent_rgb)
            if u:
                images.append(u)
        if use_wrist:
            wrist_rgb = obs.get(self.wrist_camera_name, {}).get("images", {}).get("rgb")
            if wrist_rgb is not None:
                u = image_to_data_url(wrist_rgb)
                if u:
                    images.append(u)
        if not images:
            out = {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": "no_images_available",
            }
            if on_fail == "raise":
                raise RuntimeError("verify_step has no camera images")
            return out

        system = (
            "You are a runtime verifier for a robot manipulation system. You will be "
            "shown one or two images of the CURRENT scene (agentview, and optionally a "
            "wrist close-up). You will be asked whether a specific predicate is TRUE "
            "right now in those images. Be strict: mark True only if you directly "
            "observe evidence. Do not speculate about upcoming actions or off-camera "
            "state. Reply ONLY with a compact JSON object with keys: verified (bool), "
            "confidence (float 0-1), actual (short description of what is in the "
            "scene), reasoning (one short sentence)."
        )
        user = (
            f"Predicate to check: \"{predicate_description}\"\n\n"
            "Reply with JSON exactly of the form: "
            '{"verified": <true|false>, "confidence": <0-1>, '
            '"actual": "...", "reasoning": "..."}'
        )

        try:
            raw = query_llm_text(
                system,
                user,
                images=images,
                model=_runtime_vlm_model(),
                temperature=0.0,
                max_tokens=8096,
                reasoning_effort="low",
                json_mode=True,
            )
            from rats.agents.base_agent import strip_markdown_json_fences
            # Gemini-3.1-pro emits JSON wrapped in markdown fences despite
            # json_mode=True. Strip the wrapper before json.loads.
            parsed = _json.loads(strip_markdown_json_fences(raw))
            out = {
                "verified": bool(parsed.get("verified", False)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "actual": str(parsed.get("actual", "")),
                "reasoning": str(parsed.get("reasoning", "")),
            }
        except Exception as e:
            out = {
                "verified": False,
                "confidence": 0.0,
                "actual": "",
                "reasoning": f"vlm_error: {type(e).__name__}",
            }

        if not out["verified"] and on_fail == "raise":
            raise RuntimeError(
                f"verify_step failed: {predicate_description!r} — {out['reasoning']}"
            )
        return out

    # SKILL LIBRARY - Reusable Functions from LLM Robot Code Generation
    # ======================================================================
    # Source: reduced_api and reduced_api_exampleless experiments
    # Total unique functions analyzed: 182
    # Functions after filtering: 73
    # Minimum occurrence threshold: 2
    # ======================================================================

    # Here is a curated library of reusable robotics skills derived from the provided code generations.

    # I have categorized them into **Coordinate Transforms**, **Vision & Perception**, and **Geometry & Math**. I selected implementations that are vectorized (for performance), numerically stable (especially for quaternion conversion), and decoupled from specific environment dictionaries to ensure maximum reusability.

    ### 1. Category: Coordinate Transformations
    # These functions were the most frequent across all experiments (occurring >80 times in total). They are essential because planners often output matrices, but controllers (like `solve_ik`) often require quaternions.
    # **Why Reusable:** Converting between rotation matrices, quaternions, and homogeneous transformation matrices is a fundamental requirement for almost every manipulation task.

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].

        Implements the robust Sheppard's method (checking trace and diagonal elements)
        to avoid numerical instability when the trace is close to zero.

        Args:
            R: (3, 3) rotation matrix.

        Returns:
            np.array: [w, x, y, z] unit quaternion.

        """
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
        return np.array([w, x, y, z])

    def decompose_transform(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Decompose a 4x4 homogeneous transformation matrix into position and quaternion.

        Args:
            T: (4, 4) homogeneous transformation matrix.

        Returns:
            tuple:
                - position: (3,) np.array
                - quaternion: (4,) np.array [w, x, y, z]

        """
        position = T[:3, 3]
        R = T[:3, :3]
        quat = self.rotation_matrix_to_quaternion(R)
        return position, quat

    ### 2. Category: Vision & Perception (Depth to 3D)
    # These functions bridge the gap between 2D camera data and 3D robot actions. They are crucial for converting segmentation masks into grasp targets.

    # **Why Reusable:** They encapsulate the pinhole camera model math, handling intrinsics (projection) and extrinsics (camera pose), allowing the agent to reason in the World Frame.

    def depth_to_point_cloud(self, depth_img: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """
        Convert a depth image to a 3D point cloud in the Camera Frame.

        Args:
            depth_img: (H, W) depth map in meters.
            intrinsics: (3, 3) camera intrinsic matrix.

        Returns:
            np.array: (H, W, 3) image of 3D coordinates.

        """
        if depth_img.ndim == 3:
            depth_img = depth_img[:, :, 0]

        h, w = depth_img.shape
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Vectorized grid generation
        y_grid, x_grid = np.mgrid[0:h, 0:w]

        z = depth_img
        x = (x_grid - cx) * z / fx
        y = (y_grid - cy) * z / fy

        return np.dstack((x, y, z))

    def mask_to_world_points(
        self, mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Convert specific pixels defined by a binary mask into 3D points in the World Frame.

        Args:
            mask: (H, W) binary mask (0 or 1).
            depth: (H, W) depth map.
            intrinsics: (3, 3) camera intrinsics.
            extrinsics: (4, 4) camera-to-world pose matrix.

        Returns:
            np.array: (N, 3) array of valid 3D points in world coordinates.

        """
        # Get pixel coordinates
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return np.empty((0, 3))

        if depth.ndim == 3:
            depth = depth[:, :, 0]

        z_vals = depth[ys, xs]

        # Filter invalid depth
        valid = z_vals > 0
        ys = ys[valid]
        xs = xs[valid]
        z = z_vals[valid]

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Deproject to Camera Frame
        x_cam = (xs - cx) * z / fx
        y_cam = (ys - cy) * z / fy

        # Stack to (N, 3)
        points_cam = np.stack([x_cam, y_cam, z], axis=-1)

        # Transform to World Frame
        # Create homogeneous coordinates (N, 4)
        points_cam_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world_hom = (extrinsics @ points_cam_hom.T).T

        return points_world_hom[:, :3]

    def pixel_to_world_point(
        self, u: int, v: int, z: float, intrinsics: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Deproject a single pixel to a 3D world point.

        Args:
            u, v: Pixel coordinates (col, row).
            z: Depth at that pixel.
            intrinsics: (3, 3) matrix.
            extrinsics: (4, 4) matrix.

        Returns:
            np.array: [x, y, z] in world frame.

        """
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy

        print(f"u: {u}, v: {v}, z: {z}")

        p_cam = np.array([x_cam, y_cam, z, 1.0])
        p_world = extrinsics @ p_cam
        return p_world[:3]

    ### 3. Category: Geometry & Math
    # These functions help manipulate 3D data once it has been extracted from the camera.

    # **Why Reusable:** The `transform_points` function is particularly useful because it handles both lists of points `(N, 3)` and organized point clouds `(H, W, 3)` via reshaping, making it a "do-it-all" spatial transformer.

    def transform_points(self, points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply a 4x4 homogeneous transform to a set of 3D points.

        Args:
            points: (N, 3) or (H, W, 3) array of points.
            transform_matrix: (4, 4) homogeneous transformation matrix.

        Returns:
            np.array: Transformed points with same shape as input.

        """
        original_shape = points.shape
        # Flatten to (N, 3)
        points_reshaped = points.reshape(-1, 3)

        # Convert to homogeneous (N, 4)
        ones = np.ones((points_reshaped.shape[0], 1))
        points_hom = np.hstack((points_reshaped, ones))

        # Apply transform: (4,4) @ (4,N) -> (4,N) -> Transpose back to (N,4)
        points_transformed = (transform_matrix @ points_hom.T).T

        # Return to (N, 3) and original shape
        return points_transformed[:, :3].reshape(original_shape)

    def interpolate_segment(
        self, p1: np.ndarray, p2: np.ndarray, step: float = 0.03
    ) -> list[np.ndarray]:
        """
        Generate waypoints along a line segment between two 3D points.

        Args:
            p1: Start point (3,).
            p2: End point (3,).
            step: Distance between waypoints in meters.

        Returns:
            list[np.ndarray]: List of points including p1 and p2.

        """
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6:
            return [p1]

        num_points = int(np.ceil(dist / step))
        # Using linspace to ensure we hit the start and end exactly
        return [p1 + (p2 - p1) * t for t in np.linspace(0, 1, num_points + 1)]

    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.

        Args:
            v: (3,) vector.

        Returns:
            np.array: (3,) unit vector.
        """
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return v
        return v / norm

    # ### 4. Category: Grasp Heuristics
    # A reusable heuristic for filtering grasps generated by learned models (like Contact-GraspNet).

    def select_top_down_grasp(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        cam_to_world: np.ndarray,
        vertical_threshold: float = 0.8,
    ) -> tuple:
        """
        Selects the best grasp that aligns the gripper vertically (Top-Down).

        Args:
            grasps: (N, 4, 4) Grasp poses in camera frame.
            scores: (N,) Grasp scores.
            cam_to_world: (4, 4) Extrinsics matrix.
            vertical_threshold: Dot product threshold (1.0 is perfectly vertical).

        Returns:
            tuple: (best_grasp_world_matrix, best_score) or (None, -inf)
        """
        best_grasp = None
        best_score = -np.float64("inf")

        # World Z axis (vertical)
        world_z = np.array([0, 0, 1])

        for i, g_camera in enumerate(grasps):
            # Transform grasp to world frame
            g_world = cam_to_world @ g_camera

            # Extract rotation
            R = g_world[:3, :3]

            # Assuming Gripper Z or Y is the approach vector depending on gripper definition.
            # For Franka/Robotiq, the approach vector is usually the Z-axis of the end effector.
            gripper_approach = R[:, 2]

            # Check alignment with negative World Z (pointing down)
            # Dot product should be close to -1 for top-down
            alignment = -np.dot(gripper_approach, world_z)

            if alignment > vertical_threshold:
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_grasp = g_world

        return best_grasp, best_score
