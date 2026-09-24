"""Client helpers for the local Gemini ADC proxy.

The proxy in ``scripts/gemini_proxy.py`` speaks the Gemini Developer API
wire format, while the rest of this repo mostly builds OpenAI-style chat
messages.  This module keeps that translation in one place so RATS and CaP-X
launch paths can prefer the local proxy without duplicating payload glue.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_GEMINI_PROXY_URL = "http://127.0.0.1:8112"
_HEALTH_CACHE: dict[str, tuple[float, bool]] = {}
_HEALTH_TTL_SECONDS = 30.0
_FORCE_PROXY_ENV_KEYS = (
    "RATS_USE_GEMINI_PROXY",
    "CAPX_USE_GEMINI_PROXY",
    "GEMINI_USE_PROXY",
)


class GeminiProxyError(RuntimeError):
    """Raised when the local Gemini proxy cannot satisfy a request."""


class GeminiProxyUnsupported(GeminiProxyError):
    """Raised when an OpenAI-style prompt cannot be translated safely."""


@dataclass
class GeminiProxyResult:
    content: str
    reasoning: str | None
    raw: dict[str, Any]
    payload: dict[str, Any]
    model_name: str
    elapsed_s: float


def gemini_proxy_base_url() -> str:
    """Return the configured Gemini proxy base URL."""
    return (
        os.environ.get("RATS_GEMINI_PROXY_URL")
        or os.environ.get("CAPX_GEMINI_PROXY_URL")
        or os.environ.get("GEMINI_PROXY_URL")
        or DEFAULT_GEMINI_PROXY_URL
    ).rstrip("/")


def gemini_proxy_enabled() -> bool:
    """Return False only when the caller explicitly disables proxy routing."""
    val = (
        os.environ.get("RATS_GEMINI_PROXY")
        or os.environ.get("CAPX_GEMINI_PROXY")
        or os.environ.get("GEMINI_PROXY")
        or "1"
    )
    return str(val).strip().lower() not in {"0", "false", "no", "off"}


def gemini_proxy_forced() -> bool:
    """Return True when legacy env vars explicitly force proxy routing."""
    for key in _FORCE_PROXY_ENV_KEYS:
        raw = os.environ.get(key)
        if raw is None:
            continue
        if str(raw).strip().lower() not in {"", "0", "false", "no", "off"}:
            return True
    return False


def is_google_gemini_model(model: str | None) -> bool:
    """Return True for models that should prefer the local Gemini proxy."""
    if not model:
        return False
    value = str(model).strip().lower()
    return value.startswith("google/gemini") or value.startswith("gemini")


def gemini_proxy_model_name(model: str) -> str:
    """Convert repo model ids into Gemini Developer API model names."""
    value = str(model or "").strip()
    if value.startswith("google/"):
        value = value.split("/", 1)[1]
    # Accept the shorter alias users often type.
    if value == "gemini-3.1-pro":
        return "gemini-3.1-pro-preview"
    return value


def gemini_proxy_available(base_url: str | None = None, *, timeout: float = 0.5) -> bool:
    """Fast health check with a short TTL to avoid probing on every call."""
    if not gemini_proxy_enabled():
        return False
    if gemini_proxy_forced():
        return True
    url = (base_url or gemini_proxy_base_url()).rstrip("/")
    now = time.time()
    cached = _HEALTH_CACHE.get(url)
    if cached and now - cached[0] < _HEALTH_TTL_SECONDS:
        return cached[1]
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        ok = 200 <= response.status_code < 300
    except requests.RequestException:
        ok = False
    _HEALTH_CACHE[url] = (now, ok)
    return ok


def should_try_gemini_proxy(model: str | None, base_url: str | None = None) -> bool:
    """Return True when a model should be routed to the local proxy first."""
    return is_google_gemini_model(model) and gemini_proxy_available(base_url)


def _data_url_to_inline_data(url: str) -> dict[str, Any]:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise GeminiProxyUnsupported("Gemini proxy only supports inline data: media URLs")
    try:
        header, payload = url.split(",", 1)
    except ValueError as exc:
        raise GeminiProxyUnsupported("Malformed data URL") from exc
    mime = header[5:].split(";", 1)[0] if header.startswith("data:") else ""
    if not mime:
        raise GeminiProxyUnsupported("Data URL is missing a MIME type")
    try:
        # Validate early so fallback paths can be used for bad media.
        base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise GeminiProxyUnsupported("Data URL payload is not valid base64") from exc
    return {"inlineData": {"mimeType": mime, "data": payload}}


def _content_blocks_to_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            parts.append({"text": block})
            continue
        if not isinstance(block, dict):
            parts.append({"text": str(block)})
            continue

        block_type = block.get("type")
        if block_type in {"text", "input_text"}:
            parts.append({"text": str(block.get("text", ""))})
        elif block_type in {"image_url", "input_image"}:
            image_url = block.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url", "")
            else:
                url = image_url or block.get("url", "")
            parts.append(_data_url_to_inline_data(str(url)))
        elif block_type == "video_url":
            video_url = block.get("video_url")
            if isinstance(video_url, dict):
                url = video_url.get("url", "")
            else:
                url = video_url or block.get("url", "")
            parts.append(_data_url_to_inline_data(str(url)))
        else:
            raise GeminiProxyUnsupported(f"Unsupported message block type: {block_type!r}")
    return parts


def openai_messages_to_gemini_payload(
    messages: list[dict[str, Any]],
    *,
    model_name: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Translate OpenAI chat messages into Gemini ``generateContent`` JSON."""
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role") or "user")
        parts = _content_blocks_to_parts(message.get("content", ""))
        if role == "system":
            system_parts.extend(parts)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": parts})

    if not contents:
        raise GeminiProxyUnsupported("Gemini proxy requires at least one non-system message")

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    return payload


def messages_to_gemini_payload(
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    json_mode: bool = False,
    model_name: str = "gemini-3.1-pro-preview",
) -> dict[str, Any]:
    """Backward-compatible alias for OpenAI-message Gemini payload conversion."""
    return openai_messages_to_gemini_payload(
        messages,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )


def _extract_text(body: dict[str, Any]) -> str:
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return ""
    return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))


def query_gemini_proxy(
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool = False,
    base_url: str | None = None,
    timeout: float = 200.0,
) -> GeminiProxyResult:
    """Query the local Gemini proxy with an OpenAI-style message list."""
    url = (base_url or gemini_proxy_base_url()).rstrip("/")
    model_name = gemini_proxy_model_name(model)
    payload = openai_messages_to_gemini_payload(
        messages,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )

    start = time.time()
    response = requests.post(
        f"{url}/v1beta/models/{model_name}:generateContent",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    elapsed = time.time() - start
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        raise GeminiProxyError(f"{exc}; response body: {body or '<empty>'}") from exc

    body = response.json()
    return GeminiProxyResult(
        content=_extract_text(body),
        reasoning=None,
        raw=body,
        payload=payload,
        model_name=model_name,
        elapsed_s=elapsed,
    )
