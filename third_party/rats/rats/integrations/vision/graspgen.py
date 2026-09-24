from __future__ import annotations

import logging
import os
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()

logger = logging.getLogger(__name__)

SERVICE_HOST = os.environ.get("GRASPGEN_HOST", "127.0.0.1")
SERVICE_PORT = int(os.environ.get("GRASPGEN_PORT", "5556"))
SERVICE_TIMEOUT_MS = int(os.environ.get("GRASPGEN_TIMEOUT_MS", "60000"))
COLLISION_FILTER_ENABLED = (
    os.environ.get("GRASPGEN_FILTER_COLLISIONS", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
COLLISION_THRESHOLD = float(os.environ.get("GRASPGEN_COLLISION_THRESHOLD", "0.02"))
COLLISION_MAX_SCENE_POINTS = int(
    os.environ.get("GRASPGEN_COLLISION_MAX_SCENE_POINTS", "8192")
)
COLLISION_OBJECT_EXCLUSION_RADIUS = float(
    os.environ.get("GRASPGEN_COLLISION_OBJECT_EXCLUSION_RADIUS", "0.005")
)
COLLISION_NUM_SAMPLES = int(os.environ.get("GRASPGEN_COLLISION_NUM_SAMPLES", "2000"))


def _depth_to_pointcloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
        depth_arr = depth_arr[:, :, 0]
    h, w = depth_arr.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth_arr.reshape(-1)
    x = (xs.reshape(-1) - K[0, 2]) * z / K[0, 0]
    y = (ys.reshape(-1) - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    return pts[z > 0].astype(np.float32, copy=False)


def _segmented_depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    segmentation: np.ndarray,
    segmap_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
        depth_arr = depth_arr[:, :, 0]
    seg_arr = np.asarray(segmentation)
    if seg_arr.ndim == 3 and seg_arr.shape[-1] == 1:
        seg_arr = seg_arr[:, :, 0]
    if seg_arr.dtype == bool:
        seg_arr = seg_arr.astype(np.int32)
    if seg_arr.shape != depth_arr.shape:
        raise ValueError(
            f"segmentation shape {seg_arr.shape} must match depth shape {depth_arr.shape}"
        )

    h, w = depth_arr.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth_arr.reshape(-1)
    x = (xs.reshape(-1) - K[0, 2]) * z / K[0, 0]
    y = (ys.reshape(-1) - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    valid = z > 0
    seg_flat = seg_arr.reshape(-1)
    pc_full = pts[valid].astype(np.float32, copy=False)
    pc_segment = pts[valid & (seg_flat == segmap_id)].astype(np.float32, copy=False)
    return pc_full, pc_segment


def _point_summary(points: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(points, dtype=np.float64)
    summary: dict[str, Any] = {"count": int(len(arr))}
    if arr.size == 0:
        return summary
    finite = arr[np.isfinite(arr).all(axis=1)]
    summary["finite_count"] = int(len(finite))
    if len(finite) > 0:
        summary["min_xyz"] = finite.min(axis=0).tolist()
        summary["max_xyz"] = finite.max(axis=0).tolist()
        summary["centroid_xyz"] = finite.mean(axis=0).tolist()
    return summary


def _score_summary(scores: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    summary: dict[str, Any] = {"count": int(arr.size)}
    finite = arr[np.isfinite(arr)]
    summary["finite_count"] = int(finite.size)
    if finite.size > 0:
        summary["min"] = float(np.min(finite))
        summary["max"] = float(np.max(finite))
        summary["mean"] = float(np.mean(finite))
        summary["top3"] = [float(v) for v in np.sort(finite)[::-1][:3]]
    return summary


class _GraspGenZmqClient:
    def __init__(
        self,
        host: str = SERVICE_HOST,
        port: int = SERVICE_PORT,
        timeout_ms: int = SERVICE_TIMEOUT_MS,
    ) -> None:
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = int(timeout_ms)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "pyzmq is required for the GraspGen client. Install the rats "
                "environment after pulling the graspgen branch."
            ) from exc

        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(self._addr)
            sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = sock.recv()
            response = msgpack.unpackb(raw, raw=False)
        except zmq.error.Again as exc:
            raise RuntimeError(
                f"Timed out communicating with GraspGen service at {self._addr}"
            ) from exc
        finally:
            sock.close()
            ctx.term()

        if "error" in response:
            raise RuntimeError(f"GraspGen service error: {response['error']}")
        return response

    def health_check(self) -> bool:
        try:
            return self._request({"action": "health"}).get("status") == "ok"
        except Exception:
            return False

    def infer(
        self,
        point_cloud: np.ndarray,
        *,
        scene_point_cloud: np.ndarray | None = None,
        filter_collisions: bool = False,
        collision_threshold: float = COLLISION_THRESHOLD,
        max_scene_points: int = COLLISION_MAX_SCENE_POINTS,
        object_exclusion_radius: float = COLLISION_OBJECT_EXCLUSION_RADIUS,
        num_collision_samples: int = COLLISION_NUM_SAMPLES,
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = 100,
        min_grasps: int = 40,
        max_tries: int = 6,
        remove_outliers: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        pc = np.asarray(point_cloud, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"point_cloud must be (N, 3), got {pc.shape}")
        if len(pc) == 0:
            return (
                np.empty((0, 4, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"collision": {"requested": bool(filter_collisions), "enabled": False}},
            )

        payload: dict[str, Any] = {
            "action": "infer",
            "point_cloud": pc,
            "grasp_threshold": float(grasp_threshold),
            "num_grasps": int(num_grasps),
            "topk_num_grasps": int(topk_num_grasps),
            "min_grasps": int(min_grasps),
            "max_tries": int(max_tries),
            "remove_outliers": bool(remove_outliers),
        }
        scene_pc = None
        if scene_point_cloud is not None:
            scene_pc = np.asarray(scene_point_cloud, dtype=np.float32)
            if scene_pc.ndim != 2 or scene_pc.shape[1] != 3:
                raise ValueError(
                    f"scene_point_cloud must be (N, 3), got {scene_pc.shape}"
                )
        if filter_collisions and scene_pc is not None and len(scene_pc) > 0:
            payload.update(
                {
                    "scene_point_cloud": scene_pc,
                    "filter_collisions": True,
                    "collision_threshold": float(collision_threshold),
                    "max_scene_points": int(max_scene_points),
                    "object_exclusion_radius": float(object_exclusion_radius),
                    "num_collision_samples": int(num_collision_samples),
                }
            )
        else:
            payload["filter_collisions"] = False

        response = self._request(payload)
        grasps = np.asarray(response["grasps"], dtype=np.float32)
        scores = np.asarray(response["confidences"], dtype=np.float32)
        return grasps, scores, response


def _client() -> _GraspGenZmqClient:
    return _GraspGenZmqClient()


def init_graspgen(device: str = "cuda", checkpoint_path: str | None = None) -> Any:
    """Initialize a GraspGen client with the Contact-GraspNet single-view shape.

    ``device`` and ``checkpoint_path`` are accepted for compatibility; model
    ownership lives in the external GraspGen ZMQ server.
    """

    _ = (device, checkpoint_path)

    def plan(
        depth: np.ndarray,
        cam_K: np.ndarray,
        segmap: np.ndarray,
        segmap_id: int,
        local_regions: bool = True,
        filter_grasps: bool = True,
        skip_border_objects: bool = False,
        z_range: list[float] | None = None,
        forward_passes: int = 2,
        max_retries: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _ = (local_regions, skip_border_objects, z_range, forward_passes)
        pc_full, pc_segment = _segmented_depth_to_pointcloud(
            depth,
            np.asarray(cam_K, dtype=np.float64),
            segmap,
            int(segmap_id),
        )
        collision_requested = bool(filter_grasps and COLLISION_FILTER_ENABLED)
        try:
            grasps, scores, response = _client().infer(
                pc_segment,
                scene_point_cloud=pc_full,
                filter_collisions=collision_requested,
                max_tries=max_retries,
            )
        except Exception as exc:
            plan.last_diagnostics = {
                "backend": "graspgen",
                "endpoint": "plan",
                "service_addr": f"tcp://{SERVICE_HOST}:{SERVICE_PORT}",
                "full_point_cloud": _point_summary(pc_full),
                "segment_point_cloud": _point_summary(pc_segment),
                "max_retries": int(max_retries),
                "collision_filter_requested": collision_requested,
                "error": str(exc),
            }
            raise
        plan.last_diagnostics = {
            "backend": "graspgen",
            "endpoint": "plan",
            "service_addr": f"tcp://{SERVICE_HOST}:{SERVICE_PORT}",
            "full_point_cloud": _point_summary(pc_full),
            "segment_point_cloud": _point_summary(pc_segment),
            "max_retries": int(max_retries),
            "collision_filter_requested": collision_requested,
            "service_collision": response.get("collision", {}),
            "service_timing": response.get("timing", {}),
            "final_candidate_count": int(len(grasps)),
            "final_scores": _score_summary(scores),
        }
        return grasps, scores, np.empty((0, 3), dtype=np.float32)

    plan.last_diagnostics = {}
    return plan


def init_graspgen_point_clouds() -> Any:
    """Initialize a GraspGen point-cloud planner.

    The signature mirrors ``init_contact_graspnet_point_clouds`` so existing
    reduced APIs can opt into GraspGen without changing policy-facing helper
    names. GraspGen consumes the segmented object cloud for generation and
    uses ``pc_full`` as optional scene context for terminal gripper collision
    filtering.
    """

    def plan_point_clouds(
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
        segmap_id: int = 1,
        local_regions: bool = True,
        filter_grasps: bool = True,
        forward_passes: int = 2,
        max_retries: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _ = (segmap_id, local_regions, forward_passes)
        collision_requested = bool(filter_grasps and COLLISION_FILTER_ENABLED)
        try:
            grasps, scores, response = _client().infer(
                pc_segment,
                scene_point_cloud=pc_full,
                filter_collisions=collision_requested,
                max_tries=max_retries,
            )
        except Exception as exc:
            plan_point_clouds.last_diagnostics = {
                "backend": "graspgen",
                "endpoint": "plan_point_clouds",
                "service_addr": f"tcp://{SERVICE_HOST}:{SERVICE_PORT}",
                "full_point_cloud": _point_summary(pc_full),
                "segment_point_cloud": _point_summary(pc_segment),
                "max_retries": int(max_retries),
                "collision_filter_requested": collision_requested,
                "error": str(exc),
            }
            raise
        plan_point_clouds.last_diagnostics = {
            "backend": "graspgen",
            "endpoint": "plan_point_clouds",
            "service_addr": f"tcp://{SERVICE_HOST}:{SERVICE_PORT}",
            "full_point_cloud": _point_summary(pc_full),
            "segment_point_cloud": _point_summary(pc_segment),
            "max_retries": int(max_retries),
            "collision_filter_requested": collision_requested,
            "service_collision": response.get("collision", {}),
            "service_timing": response.get("timing", {}),
            "final_candidate_count": int(len(grasps)),
            "final_scores": _score_summary(scores),
        }
        return grasps, scores, np.empty((0, 3), dtype=np.float32)

    plan_point_clouds.last_diagnostics = {}
    return plan_point_clouds
