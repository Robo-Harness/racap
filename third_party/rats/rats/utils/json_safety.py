"""Small JSON-safety helpers for run artifacts."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger("rats.json_safety")

# Maximum nesting depth before json_safe replaces the value with a stub.
# Python's default recursion limit is 1000 and each json_safe step uses ~2
# frames (function call + dict/list comprehension), so empirically anything
# over ~200 levels is already pathological and will blow the stack inside
# json.dumps a few calls later. 64 is generous for any legitimate nested
# structure we have in iteration_data (plans, diagnoses, retry packages).
_MAX_DEPTH = 64

# Maximum ndarray-like .size before json_safe replaces the array with a
# shape/dtype stub instead of expanding via .tolist(). Image and depth
# tensors (e.g. 800x512x3 RGB = 1.2M values) silently bloated iteration
# JSONs to ~1 GB each via the policy_self_check artifacts.observation
# subtree. 8192 elements is well above legitimate uses (joint vectors,
# 4x4 poses, ~hundreds of sampled keypoints) but kills image-class arrays.
_MAX_NDARRAY_ELEMENTS = 8192


def json_safe(value: Any) -> Any:
    """Recursively convert common scientific/debug values to strict JSON.

    Unlike ``json.dump(default=str)``, this preserves nested dict/list shape so
    final summary and iteration artifacts do not collapse into opaque strings
    when a single numpy scalar, Path, NaN, or ndarray appears inside them.

    Cycle-safe: tracks visited container ids so a self-referential structure
    (which previously raised RecursionError mid-save and lost the whole
    iteration's JSON record) gets the cycle leaf replaced with a stub string
    instead of crashing the persistence layer. Also depth-bounded so
    pathologically deep but acyclic structures still get truncated rather
    than exploding the stack inside json.dumps below.

    On cycle or depth-cap hits, logs the path to the offender so the
    upstream cause can be tracked down rather than papered over forever.
    """
    return _json_safe_impl(value, _seen=set(), _path=(), _depth=0)


def _json_safe_impl(
    value: Any,
    *,
    _seen: set[int],
    _path: tuple[str, ...],
    _depth: int,
) -> Any:
    if _depth > _MAX_DEPTH:
        path = "/".join(_path) or "<root>"
        logger.warning(
            f"json_safe depth cap hit at {path} (depth={_depth}) — "
            f"replacing value of type {type(value).__name__} with stub. "
            f"This usually means a cycle or pathologically nested dict "
            f"reached the serializer; trace the path above to find it."
        )
        return f"<json_safe:max_depth_at_{path}>"

    if isinstance(value, (dict, list, tuple, set)):
        vid = id(value)
        if vid in _seen:
            path = "/".join(_path) or "<root>"
            logger.warning(
                f"json_safe cycle detected at {path} — value of type "
                f"{type(value).__name__} reappeared along its own ancestry. "
                f"Replacing the recursion edge with a stub. Source of the "
                f"cycle should be fixed upstream rather than relying on this "
                f"guard."
            )
            return f"<json_safe:cycle_at_{path}>"
        _seen = _seen | {vid}

    if isinstance(value, dict):
        return {
            str(k): _json_safe_impl(
                v, _seen=_seen, _path=_path + (str(k),), _depth=_depth + 1,
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe_impl(
                v, _seen=_seen, _path=_path + (f"[{i}]",), _depth=_depth + 1,
            )
            for i, v in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _json_safe_impl(
                value.item(), _seen=_seen, _path=_path, _depth=_depth + 1,
            )
        except Exception:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            size = getattr(value, "size", None)
            if isinstance(size, int) and size > _MAX_NDARRAY_ELEMENTS:
                shape = getattr(value, "shape", None)
                dtype = getattr(value, "dtype", None)
                return (
                    f"<ndarray:shape={tuple(shape) if shape is not None else '?'},"
                    f"dtype={dtype if dtype is not None else '?'},size={size}>"
                )
            return _json_safe_impl(
                value.tolist(), _seen=_seen, _path=_path, _depth=_depth + 1,
            )
        except Exception:
            pass
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return repr(value)
