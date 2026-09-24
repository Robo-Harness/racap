"""Frame capture utilities for the executor."""

from __future__ import annotations

from typing import Any

import numpy as np


def capture_frame(env: Any) -> np.ndarray | None:
    """Capture current RGB frame from environment."""
    render_fn = getattr(env, "render", None)
    if callable(render_fn):
        try:
            frame = render_fn(mode="rgb_array")
            if isinstance(frame, np.ndarray):
                return frame
        except (TypeError, Exception):
            pass
    return None


def get_observation_images(env: Any) -> dict[str, np.ndarray]:
    """Get observation images from environment cameras."""
    images = {}
    low_level = getattr(env, "low_level_env", env)
    get_obs_fn = getattr(low_level, "get_observation", None)
    if callable(get_obs_fn):
        try:
            obs = get_obs_fn()
            for cam_key, cam_data in obs.items():
                if isinstance(cam_data, dict) and "images" in cam_data:
                    rgb = cam_data["images"].get("rgb")
                    if rgb is not None:
                        if hasattr(rgb, "cpu"):
                            rgb = rgb.cpu().numpy()
                        images[cam_key] = np.asarray(rgb)
        except Exception:
            pass
    return images
