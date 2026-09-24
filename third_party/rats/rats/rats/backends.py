from __future__ import annotations

import base64
import io
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from rats.llm.client import ModelQueryArgs, query_model


def extract_python_code(content: str) -> str:
    fence = "```python"
    if fence in content:
        after = content.split(fence, 1)[1]
        if "```" in after:
            return after.split("```", 1)[0].strip()
    return content.strip()


def image_to_data_url(image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_python_or_plain(content: str) -> str:
    code = extract_python_code(content)
    return code or content.strip()


def make_policy_query_backend(
    args: ModelQueryArgs,
    *,
    query_fn: Callable[[ModelQueryArgs, list[dict[str, Any]]], dict[str, Any]] = query_model,
) -> Callable[[list[dict[str, Any]]], str]:
    def _backend(prompt: list[dict[str, Any]]) -> str:
        result = query_fn(args, prompt)
        content = result.get("content", "")
        return extract_python_or_plain(content)

    return _backend


def make_diagnoser_query_backend(
    args: ModelQueryArgs,
    *,
    query_fn: Callable[[ModelQueryArgs, list[dict[str, Any]]], dict[str, Any]] = query_model,
) -> Callable[[list[dict[str, Any]]], str]:
    def _backend(prompt: list[dict[str, Any]]) -> str:
        result = query_fn(args, prompt)
        return str(result.get("content", "")).strip()

    return _backend
