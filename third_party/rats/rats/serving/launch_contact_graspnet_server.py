import asyncio
import base64
import contextlib
import functools
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import tyro
import uvicorn
import viser.transforms as vtf
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rats.utils.graspnet_utils import camera_so3_looking_at_origin, sample_hemisphere_viewpoint, sample_random_camera_viewpoint, sample_hemisphere_viewpoints_evenly

from rats.utils.graspnet_utils import (
    camera_so3_looking_at_origin,
    sample_cone_viewpoints_evenly,
    sample_hemisphere_viewpoint,
    sample_hemisphere_viewpoints_evenly,
    sample_random_camera_viewpoint,
)

# --- Service Configuration ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State ---
_GRASP_ESTIMATOR: Any | None = None
_DEVICE: str = "cuda"

# Semaphore to serialize GPU access (prevents OOM from concurrent inference)
_GPU_SEMAPHORE = asyncio.Semaphore(1)


async def _run_on_gpu(fn, *args, **kwargs):
    """Run a blocking GPU function without blocking the event loop."""
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def recursive_key_value_assign(d, ks, v):
    """
    Recursive value assignment to a nested dict
    """
    if len(ks) > 1:
        recursive_key_value_assign(d[ks[0]], ks[1:], v)
    elif len(ks) == 1:
        d[ks[0]] = v


def load_contact_graspnet_config(
    checkpoint_dir, batch_size=None, max_epoch=None, data_path=None, arg_configs=None
):
    arg_configs = arg_configs or []

    config_path = os.path.join(checkpoint_dir, "config.yaml")
    # If not found in checkpoint dir, look relative to this file's location
    if not os.path.exists(config_path):
        # Fallback to looking in the vendor directory if we can find it
        pass

    with open(config_path) as f:
        global_config = yaml.safe_load(f)

    for conf in arg_configs:
        k_str, v = conf.split(":")
        with contextlib.suppress(Exception):
            v = eval(v)
        ks = [int(k) if k.isdigit() else k for k in k_str.split(".")]

        recursive_key_value_assign(global_config, ks, v)

    if batch_size is not None:
        global_config["OPTIMIZER"]["batch_size"] = int(batch_size)
    if max_epoch is not None:
        global_config["OPTIMIZER"]["max_epoch"] = int(max_epoch)
    if data_path is not None:
        global_config["DATA"]["data_path"] = data_path

    global_config["DATA"]["classes"] = None

    return global_config


# --- Serialization Helpers ---


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    try:
        data = base64.b64decode(b64_str)
        with io.BytesIO(data) as f:
            return np.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid numpy data: {e}")


# --- API Models ---


class PlanRequest(BaseModel):
    depth_base64: str
    cam_K_base64: str
    segmap_base64: str
    segmap_id: int
    local_regions: bool = True
    filter_grasps: bool = True
    skip_border_objects: bool = False
    z_range: list[float] | None = None
    forward_passes: int = 1
    max_retries: int = 7


class PlanPointCloudsRequest(BaseModel):
    pc_full_base64: str
    pc_segment_base64: str
    segmap_id: int = 1
    local_regions: bool = True
    filter_grasps: bool = True
    forward_passes: int = 1
    max_retries: int = 7


class PlanEvenlyRequest(PlanRequest):
    num_viewpoints: int = 10
    max_angle_deg: float = 45.0


class PlanResponse(BaseModel):
    grasps_base64: str
    scores_base64: str
    contact_pts_base64: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def _point_summary(points: Any) -> dict[str, Any]:
    try:
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    except Exception:
        return {"count": 0}
    out: dict[str, Any] = {"count": int(len(arr))}
    if len(arr) == 0:
        return out
    finite = np.isfinite(arr).all(axis=1)
    out["finite_count"] = int(np.count_nonzero(finite))
    arr = arr[finite]
    if len(arr) == 0:
        return out
    out.update(
        {
            "min": np.round(np.min(arr, axis=0), 6).tolist(),
            "max": np.round(np.max(arr, axis=0), 6).tolist(),
            "centroid": np.round(np.mean(arr, axis=0), 6).tolist(),
        }
    )
    return out


def _score_summary(scores: Any) -> dict[str, Any]:
    try:
        arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    except Exception:
        return {"count": 0}
    out: dict[str, Any] = {"count": int(len(arr))}
    if len(arr) == 0:
        return out
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return out
    out.update(
        {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "top3": [float(x) for x in np.sort(finite)[::-1][:3]],
        }
    )
    return out


def _normalize_point_cloud(points: Any, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize service input to ContactGraspNet's expected (N, 3) layout."""
    arr = np.asarray(points)
    diagnostics: dict[str, Any] = {
        "input_shape": list(arr.shape),
        "input_dtype": str(arr.dtype),
    }

    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"{name} must contain xyz triples; got {arr.shape}")
        arr = arr.reshape(-1, 3)
        diagnostics["reshape"] = "flat_xyz_triples"
    elif arr.ndim == 2:
        if arr.shape[1] == 3:
            pass
        elif arr.shape[0] == 3:
            arr = arr.T
            diagnostics["reshape"] = "transposed_3_by_n"
        else:
            raise ValueError(f"{name} must have shape (N, 3); got {arr.shape}")
    elif arr.ndim >= 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3].reshape(-1, 3)
        diagnostics["reshape"] = "image_xyz"
    else:
        raise ValueError(f"{name} must have an xyz last axis; got {arr.shape}")

    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr).all(axis=1)
    dropped = int(arr.shape[0] - np.count_nonzero(finite))
    if dropped:
        arr = arr[finite]
        diagnostics["dropped_nonfinite"] = dropped

    diagnostics["shape"] = list(arr.shape)
    diagnostics["count"] = int(arr.shape[0])
    return np.ascontiguousarray(arr), diagnostics


def _do_plan(req: PlanRequest) -> PlanResponse:
    """Blocking grasp planning (runs on GPU thread)."""
    depth = _base64_to_numpy(req.depth_base64)
    cam_K = _base64_to_numpy(req.cam_K_base64)
    segmap = _base64_to_numpy(req.segmap_base64)
    diagnostics: dict[str, Any] = {
        "endpoint": "plan",
        "depth_shape": list(np.shape(depth)),
        "depth_valid_count": int(np.count_nonzero(np.isfinite(depth) & (np.asarray(depth) > 0))),
        "segmap_shape": list(np.shape(segmap)),
        "segmap_id": int(req.segmap_id),
        "segmap_pixel_count": int(np.count_nonzero(np.asarray(segmap) == req.segmap_id)),
        "local_regions": bool(req.local_regions),
        "filter_grasps": bool(req.filter_grasps),
        "skip_border_objects": bool(req.skip_border_objects),
        "forward_passes": int(req.forward_passes),
        "max_retries": int(req.max_retries),
        "attempts": [],
    }

    # Original logic from grasp_graspnet.py's plan function
    current_retries: int = 0
    max_retries = req.max_retries
    segmap_id = req.segmap_id
    z_range = req.z_range

    if z_range is None:
        z_range = [0.2, 2.0]

    # extract_point_clouds
    pc_full, pc_segments, pc_colors = _GRASP_ESTIMATOR.extract_point_clouds(
        depth,
        cam_K,
        segmap=segmap,
        segmap_id=segmap_id,
        skip_border_objects=req.skip_border_objects,
        z_range=z_range,
    )
    diagnostics["z_range"] = list(z_range)
    diagnostics["pc_full"] = _point_summary(pc_full)
    diagnostics["pc_segment"] = _point_summary(pc_segments.get(segmap_id, np.empty((0, 3))))

    pred_grasps_cam, scores, contact_pts, _ = _GRASP_ESTIMATOR.predict_scene_grasps(
        pc_full,
        pc_segments=pc_segments,
        local_regions=req.local_regions,
        filter_grasps=req.filter_grasps,
        forward_passes=req.forward_passes,
    )
    initial_count = len(pred_grasps_cam.get(segmap_id, []))
    diagnostics["attempts"].append(
        {
            "kind": "initial",
            "candidate_count": int(initial_count),
            "scores": _score_summary(scores.get(segmap_id, [])),
        }
    )

    while len(pred_grasps_cam.get(segmap_id, [])) == 0 and current_retries < max_retries:
        # Determine target point for camera to look at
        if segmap_id in pc_segments and len(pc_segments[segmap_id]) > 0:
            target_centroid = np.mean(pc_segments[segmap_id], axis=0)
        else:
            target_centroid = np.mean(pc_full, axis=0)

        position, wxyz = sample_random_camera_viewpoint(target_centroid, xy_extent_meters=0.25)
        tf_wc = vtf.SE3(wxyz_xyz=np.concatenate([wxyz, position]))

        pc_full_h = np.hstack([pc_full, np.ones((pc_full.shape[0], 1))])  # N x 4
        pc_segments_h = {}
        for seg_id in pc_segments:
            pc_segments_h[seg_id] = np.hstack(
                [pc_segments[seg_id], np.ones((pc_segments[seg_id].shape[0], 1))]
            )  # N x 4

        # Transform world -> camera
        tf_cw_matrix = tf_wc.inverse().as_matrix()
        pc_full_cam = (tf_cw_matrix @ pc_full_h.T).T[:, :3]
        pc_segments_cam = {}
        for seg_id in pc_segments:
            pc_segments_cam[seg_id] = (tf_cw_matrix @ pc_segments_h[seg_id].T).T[
                :, :3
            ]

        pred_grasps_cam_new, scores_new, contact_pts_new, _ = (
            _GRASP_ESTIMATOR.predict_scene_grasps(
                pc_full_cam,
                pc_segments=pc_segments_cam,
                local_regions=req.local_regions,
                filter_grasps=req.filter_grasps,
                forward_passes=req.forward_passes,
            )
        )
        retry_count = len(pred_grasps_cam_new.get(segmap_id, []))
        diagnostics["attempts"].append(
            {
                "kind": "retry",
                "retry_index": int(current_retries + 1),
                "candidate_count": int(retry_count),
                "camera_position": np.round(position, 6).tolist(),
                "target_centroid": np.round(target_centroid, 6).tolist(),
                "scores": _score_summary(scores_new.get(segmap_id, [])),
            }
        )

        # Transform results back to Original Frame
        tf_wc_matrix = tf_wc.as_matrix()
        if segmap_id in pred_grasps_cam_new and len(pred_grasps_cam_new[segmap_id]) > 0:
            # Transform grasps (N, 4, 4)
            grasps_cam = pred_grasps_cam_new[segmap_id]
            grasps_world = np.matmul(tf_wc_matrix, grasps_cam)
            pred_grasps_cam[segmap_id] = grasps_world

            # Transform contact points (N, 3)
            pts_cam = contact_pts_new[segmap_id]
            pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
            pts_world = (tf_wc_matrix @ pts_cam_h.T).T[:, :3]
            contact_pts[segmap_id] = pts_world

            # Scores are invariant
            scores[segmap_id] = scores_new[segmap_id]

        current_retries += 1

    if current_retries >= max_retries:
        grasps_out = np.array([])
        scores_out = np.array([])
        contact_pts_out = np.array([])
        diagnostics["failure_reason"] = "max_retries_exhausted"
    else:
        grasps_result = pred_grasps_cam[segmap_id]
        scores_result = scores[segmap_id]
        contact_pts_result = contact_pts[segmap_id]

        if isinstance(grasps_result, list):
            grasps_out = np.array(grasps_result)
        else:
            grasps_out = grasps_result

        if isinstance(scores_result, list):
            scores_out = np.array(scores_result)
        else:
            scores_out = scores_result

        if isinstance(contact_pts_result, list):
            contact_pts_out = np.array(contact_pts_result)
        else:
            contact_pts_out = contact_pts_result
    diagnostics["retry_count"] = int(current_retries)
    diagnostics["final_candidate_count"] = int(len(grasps_out))
    diagnostics["final_scores"] = _score_summary(scores_out)
    if len(grasps_out) == 0:
        logger.warning(
            "GraspNet /plan produced zero candidates: seg_pixels=%s pc_segment=%s retries=%s",
            diagnostics.get("segmap_pixel_count"),
            diagnostics.get("pc_segment", {}).get("count"),
            diagnostics.get("retry_count"),
        )
    else:
        logger.info(
            "GraspNet /plan candidates=%d retries=%d top3=%s",
            diagnostics["final_candidate_count"],
            diagnostics["retry_count"],
            diagnostics.get("final_scores", {}).get("top3"),
        )

    return PlanResponse(
        grasps_base64=_numpy_to_base64(grasps_out),
        scores_base64=_numpy_to_base64(scores_out),
        contact_pts_base64=_numpy_to_base64(contact_pts_out),
        diagnostics=diagnostics,
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_endpoint(req: PlanRequest):
    if _GRASP_ESTIMATOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        return await _run_on_gpu(_do_plan, req)
    except Exception as e:
        logger.error(f"Grasp planning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Grasp planning failed: {e}")


def _do_plan_point_clouds(req: PlanPointCloudsRequest) -> PlanResponse:
    """Blocking grasp planning from pre-computed point clouds (runs on GPU thread)."""
    pc_full_raw = _base64_to_numpy(req.pc_full_base64)
    pc_segment_raw = _base64_to_numpy(req.pc_segment_base64)
    pc_full, pc_full_normalization = _normalize_point_cloud(pc_full_raw, "pc_full")
    pc_segment, pc_segment_normalization = _normalize_point_cloud(
        pc_segment_raw, "pc_segment"
    )
    segmap_id = req.segmap_id
    pc_segments = {segmap_id: pc_segment}
    diagnostics: dict[str, Any] = {
        "endpoint": "plan_point_clouds",
        "segmap_id": int(segmap_id),
        "normalization": {
            "pc_full": pc_full_normalization,
            "pc_segment": pc_segment_normalization,
        },
        "pc_full": _point_summary(pc_full),
        "pc_segment": _point_summary(pc_segment),
        "local_regions": bool(req.local_regions),
        "filter_grasps": bool(req.filter_grasps),
        "forward_passes": int(req.forward_passes),
        "max_retries": int(req.max_retries),
        "attempts": [],
    }

    current_retries: int = 0
    max_retries = req.max_retries

    pred_grasps_cam, scores, contact_pts, _ = _GRASP_ESTIMATOR.predict_scene_grasps(
        pc_full,
        pc_segments=pc_segments,
        local_regions=req.local_regions,
        filter_grasps=req.filter_grasps,
        forward_passes=req.forward_passes,
    )
    diagnostics["attempts"].append(
        {
            "kind": "initial",
            "candidate_count": int(len(pred_grasps_cam.get(segmap_id, []))),
            "scores": _score_summary(scores.get(segmap_id, [])),
        }
    )

    while len(pred_grasps_cam.get(segmap_id, [])) == 0 and current_retries < max_retries:
        if segmap_id in pc_segments and len(pc_segments[segmap_id]) > 0:
            target_centroid = np.mean(pc_segments[segmap_id], axis=0)
        else:
            target_centroid = np.mean(pc_full, axis=0)

        position, wxyz = sample_random_camera_viewpoint(target_centroid, xy_extent_meters=0.25)
        tf_wc = vtf.SE3(wxyz_xyz=np.concatenate([wxyz, position]))

        pc_full_h = np.hstack([pc_full, np.ones((pc_full.shape[0], 1))])
        pc_segments_h = {}
        for seg_id in pc_segments:
            pc_segments_h[seg_id] = np.hstack(
                [pc_segments[seg_id], np.ones((pc_segments[seg_id].shape[0], 1))]
            )

        tf_cw_matrix = tf_wc.inverse().as_matrix()
        pc_full_cam = (tf_cw_matrix @ pc_full_h.T).T[:, :3]
        pc_segments_cam = {}
        for seg_id in pc_segments:
            pc_segments_cam[seg_id] = (tf_cw_matrix @ pc_segments_h[seg_id].T).T[:, :3]

        pred_grasps_cam_new, scores_new, contact_pts_new, _ = (
            _GRASP_ESTIMATOR.predict_scene_grasps(
                pc_full_cam,
                pc_segments=pc_segments_cam,
                local_regions=req.local_regions,
                filter_grasps=req.filter_grasps,
                forward_passes=req.forward_passes,
            )
        )
        diagnostics["attempts"].append(
            {
                "kind": "retry",
                "retry_index": int(current_retries + 1),
                "candidate_count": int(len(pred_grasps_cam_new.get(segmap_id, []))),
                "camera_position": np.round(position, 6).tolist(),
                "target_centroid": np.round(target_centroid, 6).tolist(),
                "scores": _score_summary(scores_new.get(segmap_id, [])),
            }
        )

        tf_wc_matrix = tf_wc.as_matrix()
        if segmap_id in pred_grasps_cam_new and len(pred_grasps_cam_new[segmap_id]) > 0:
            grasps_cam = pred_grasps_cam_new[segmap_id]
            grasps_world = np.matmul(tf_wc_matrix, grasps_cam)
            pred_grasps_cam[segmap_id] = grasps_world

            pts_cam = contact_pts_new[segmap_id]
            pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
            pts_world = (tf_wc_matrix @ pts_cam_h.T).T[:, :3]
            contact_pts[segmap_id] = pts_world

            scores[segmap_id] = scores_new[segmap_id]

        current_retries += 1

    if current_retries >= max_retries:
        grasps_out = np.array([])
        scores_out = np.array([])
        contact_pts_out = np.array([])
        diagnostics["failure_reason"] = "max_retries_exhausted"
    else:
        grasps_result = pred_grasps_cam[segmap_id]
        scores_result = scores[segmap_id]
        contact_pts_result = contact_pts[segmap_id]

        grasps_out = np.array(grasps_result) if isinstance(grasps_result, list) else grasps_result
        scores_out = (
            np.array(scores_result) if isinstance(scores_result, list) else scores_result
        )
        contact_pts_out = (
            np.array(contact_pts_result)
            if isinstance(contact_pts_result, list)
            else contact_pts_result
        )
    diagnostics["retry_count"] = int(current_retries)
    diagnostics["final_candidate_count"] = int(len(grasps_out))
    diagnostics["final_scores"] = _score_summary(scores_out)
    if len(grasps_out) == 0:
        logger.warning(
            "GraspNet /plan_point_clouds produced zero candidates: pc_full=%s pc_segment=%s retries=%s",
            diagnostics.get("pc_full", {}).get("count"),
            diagnostics.get("pc_segment", {}).get("count"),
            diagnostics.get("retry_count"),
        )
    else:
        logger.info(
            "GraspNet /plan_point_clouds candidates=%d retries=%d top3=%s",
            diagnostics["final_candidate_count"],
            diagnostics["retry_count"],
            diagnostics.get("final_scores", {}).get("top3"),
        )

    return PlanResponse(
        grasps_base64=_numpy_to_base64(grasps_out),
        scores_base64=_numpy_to_base64(scores_out),
        contact_pts_base64=_numpy_to_base64(contact_pts_out),
        diagnostics=diagnostics,
    )


@app.post("/plan_point_clouds", response_model=PlanResponse)
async def plan_point_clouds_endpoint(req: PlanPointCloudsRequest):
    if _GRASP_ESTIMATOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        return await _run_on_gpu(_do_plan_point_clouds, req)
    except Exception as e:
        logger.error(f"Point cloud grasp planning failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Point cloud grasp planning failed: {e}"
        )


def _do_plan_evenly(req: PlanEvenlyRequest) -> PlanResponse:
    """Blocking evenly-sampled grasp planning (runs on GPU thread)."""
    depth = _base64_to_numpy(req.depth_base64)
    cam_K = _base64_to_numpy(req.cam_K_base64)
    segmap = _base64_to_numpy(req.segmap_base64)

    segmap_id = req.segmap_id
    z_range = req.z_range

    if z_range is None:
        z_range = [0.2, 2.0]

    # extract_point_clouds
    pc_full, pc_segments, pc_colors = _GRASP_ESTIMATOR.extract_point_clouds(
        depth,
        cam_K,
        segmap=segmap,
        segmap_id=segmap_id,
        skip_border_objects=req.skip_border_objects,
        z_range=z_range,
    )

    # Determine target point for camera to look at
    if segmap_id in pc_segments and len(pc_segments[segmap_id]) > 0:
        target_centroid = np.mean(pc_segments[segmap_id], axis=0)
    else:
        target_centroid = np.mean(pc_full, axis=0)

    # Sample N viewpoints evenly
    positions, wxyzs = sample_cone_viewpoints_evenly(
        target_centroid,
        current_camera_position=np.array([0.0, 0.0, 0.0]),
        num_samples=req.num_viewpoints,
        max_angle_deg=req.max_angle_deg,
    )

    # We will accumulate results across all viewpoints
    aggregated_grasps = []
    aggregated_scores = []
    aggregated_contact_pts = []

    # Prepare homogenous coords for transformation
    pc_full_h = np.hstack([pc_full, np.ones((pc_full.shape[0], 1))])
    pc_segments_h = {}
    for seg_id in pc_segments:
        pc_segments_h[seg_id] = np.hstack(
            [pc_segments[seg_id], np.ones((pc_segments[seg_id].shape[0], 1))]
        )

    for i in range(req.num_viewpoints):
        position = positions[i]
        wxyz = wxyzs[i]

        # Camera pose in World (Original) frame
        tf_wc = vtf.SE3(wxyz_xyz=np.concatenate([wxyz, position]))

        # Transform World -> Camera
        tf_cw_matrix = tf_wc.inverse().as_matrix()

        pc_full_cam = (tf_cw_matrix @ pc_full_h.T).T[:, :3]
        pc_segments_cam = {}
        for seg_id in pc_segments:
            pc_segments_cam[seg_id] = (tf_cw_matrix @ pc_segments_h[seg_id].T).T[:, :3]

        # Predict grasps in this camera frame
        pred_grasps_cam, scores, contact_pts, _ = _GRASP_ESTIMATOR.predict_scene_grasps(
            pc_full_cam,
            pc_segments=pc_segments_cam,
            local_regions=req.local_regions,
            filter_grasps=req.filter_grasps,
            forward_passes=req.forward_passes,
        )

        if segmap_id in pred_grasps_cam and len(pred_grasps_cam[segmap_id]) > 0:
            # Transform grasps back to World frame
            tf_wc_matrix = tf_wc.as_matrix()

            # Grasps: (N, 4, 4)
            grasps_cam = pred_grasps_cam[segmap_id]
            grasps_world = np.matmul(tf_wc_matrix, grasps_cam)

            # Contact points: (N, 3)
            pts_cam = contact_pts[segmap_id]
            pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
            pts_world = (tf_wc_matrix @ pts_cam_h.T).T[:, :3]

            scores_val = scores[segmap_id]

            aggregated_grasps.append(grasps_world)
            aggregated_scores.append(scores_val)
            aggregated_contact_pts.append(pts_world)

    # Concatenate all results
    if len(aggregated_grasps) > 0:
        grasps_out = np.concatenate(aggregated_grasps, axis=0)
        scores_out = np.concatenate(aggregated_scores, axis=0)
        contact_pts_out = np.concatenate(aggregated_contact_pts, axis=0)
    else:
        grasps_out = np.array([])
        scores_out = np.array([])
        contact_pts_out = np.array([])

    return PlanResponse(
        grasps_base64=_numpy_to_base64(grasps_out),
        scores_base64=_numpy_to_base64(scores_out),
        contact_pts_base64=_numpy_to_base64(contact_pts_out),
    )


@app.post("/plan_evenly", response_model=PlanResponse)
async def plan_evenly_endpoint(req: PlanEvenlyRequest):
    if _GRASP_ESTIMATOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        return await _run_on_gpu(_do_plan_evenly, req)
    except Exception as e:
        logger.error(f"Evenly-sampled grasp planning failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Evenly-sampled grasp planning failed: {e}"
        )


def main(device: str = "cuda", port: int = 8115, host: str = "127.0.0.1"):
    global _GRASP_ESTIMATOR, _DEVICE
    _DEVICE = device

    # --- Setup Paths & Import ---

    here = os.path.dirname(os.path.abspath(__file__))
    # Assume rats/serving -> go up to capx -> go to third_party
    vendor_root = os.environ.get("CONTACT_GRASPNET_ROOT") or os.path.normpath(
        os.path.join(here, "..", "third_party", "contact_graspnet_pytorch")
    )

    pointnet_root = os.path.join(vendor_root, "Pointnet_Pointnet2_pytorch")
    if pointnet_root not in sys.path:
        sys.path.append(pointnet_root)
    sys.path.append(vendor_root)

    try:
        from contact_graspnet_pytorch.checkpoints import CheckpointIO
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    except ImportError:
        logger.error(
            f"Could not import contact_graspnet_pytorch. Verified vendor_root: {vendor_root}"
        )
        raise

    model_checkpoint_dir = os.path.join(vendor_root, "checkpoints/contact_graspnet/checkpoints")

    logger.info(f"Loading GraspNet config from {model_checkpoint_dir}")
    global_config = load_contact_graspnet_config(Path(model_checkpoint_dir).parent)

    logger.info("Building GraspNet model...")
    _GRASP_ESTIMATOR = GraspEstimator(global_config)

    logger.info(f"Loading weights from {model_checkpoint_dir}")
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=_GRASP_ESTIMATOR.model)
    try:
        checkpoint_io.load("model.pt")
    except FileExistsError:
        logger.warning("No model checkpoint found")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")

    logger.info(f"GraspNet Service initialized on {device}. Starting Uvicorn...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
