"""Optional lightweight client for an external NVIDIA GraspGen server.

The simulator process deliberately does not import GraspGen's pinned CUDA /
PyTorch stack.  A ZMQ server owns the model and this client sends only the
visually segmented world-frame point cloud.  If no endpoint is configured or
the server is unavailable, callers receive ``None`` and retain their existing
Contact-GraspNet fallback.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def infer_grasps(
    point_cloud: np.ndarray,
    *,
    endpoint: str | None = None,
    timeout_ms: int = 30_000,
    num_grasps: int = 160,
    topk: int = 80,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return world-frame GraspGen poses/scores, or ``None`` on fallback."""
    address = str(endpoint or os.getenv("RACAP_GRASPGEN_ENDPOINT", "")).strip()
    if not address:
        return None
    if not address.startswith("tcp://"):
        address = f"tcp://{address}"
    points = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 30:
        return None
    try:
        import msgpack
        import msgpack_numpy
        import zmq

        msgpack_numpy.patch()
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(address)
        payload: dict[str, Any] = {
            "action": "infer",
            "point_cloud": points,
            "grasp_threshold": -1.0,
            "num_grasps": int(num_grasps),
            "topk_num_grasps": int(topk),
            "min_grasps": min(30, int(topk)),
            "max_tries": 2,
            "remove_outliers": True,
        }
        socket.send(msgpack.packb(payload, use_bin_type=True))
        response = msgpack.unpackb(socket.recv(), raw=False)
        socket.close()
        if not isinstance(response, dict) or response.get("error"):
            return None
        grasps = np.asarray(response.get("grasps"), dtype=float)
        scores = np.asarray(response.get("confidences"), dtype=float).reshape(-1)
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
            return None
        count = min(len(grasps), len(scores))
        finite = np.isfinite(grasps[:count]).all(axis=(1, 2)) & np.isfinite(scores[:count])
        return grasps[:count][finite], scores[:count][finite]
    except Exception:
        return None
