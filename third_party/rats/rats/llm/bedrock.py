"""Amazon Bedrock Converse API helpers.

This module keeps Bedrock-specific model aliases, OpenAI-chat-message
conversion, and HTTP request handling in one place so the RATS and CaP-X LLM
clients can share the same backend without duplicating provider glue.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from urllib.parse import quote

import requests


DEFAULT_BEDROCK_REGION = "us-east-1"
DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
OPUS_4_7_BEDROCK_MODEL_ID = "us.anthropic.claude-opus-4-7"

BEDROCK_MODEL_ALIASES = {
    "sonnet-4.6": DEFAULT_BEDROCK_MODEL_ID,
    "claude-sonnet-4.6": DEFAULT_BEDROCK_MODEL_ID,
    "claude-sonnet-4-6": DEFAULT_BEDROCK_MODEL_ID,
    "anthropic/claude-sonnet-4.6": DEFAULT_BEDROCK_MODEL_ID,
    "anthropic/claude-sonnet-4-6": DEFAULT_BEDROCK_MODEL_ID,
    # Opus 4.7 — used as the Gemini-vision fallback in
    # agents/base_agent.py:query_with_fallback. Routed through Bedrock
    # so it has a separate quota from any OpenRouter Anthropic key.
    "opus-4.7": OPUS_4_7_BEDROCK_MODEL_ID,
    "claude-opus-4.7": OPUS_4_7_BEDROCK_MODEL_ID,
    "claude-opus-4-7": OPUS_4_7_BEDROCK_MODEL_ID,
    "anthropic/claude-opus-4.7": OPUS_4_7_BEDROCK_MODEL_ID,
    "anthropic/claude-opus-4-7": OPUS_4_7_BEDROCK_MODEL_ID,
}

BEDROCK_MODELS = [
    "bedrock/us.anthropic.claude-sonnet-4-6",
    "bedrock/sonnet-4.6",
    "bedrock/claude-sonnet-4.6",
    "bedrock/claude-sonnet-4-6",
    "bedrock/us.anthropic.claude-opus-4-7",
    "bedrock/opus-4.7",
    "bedrock/claude-opus-4.7",
    "bedrock/claude-opus-4-7",
]

_IMAGE_MIME_TO_FORMAT = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
}

_VIDEO_MIME_TO_FORMAT = {
    "video/mp4": "mp4",
    "video/mpeg": "mpeg",
    "video/mpg": "mpg",
    "video/quicktime": "mov",
    "video/mov": "mov",
    "video/webm": "webm",
}


def is_bedrock_model(model: str | None) -> bool:
    """Return True when a model should be routed through Amazon Bedrock."""
    return str(model or "").strip().lower().startswith("bedrock/")


def canonicalize_bedrock_model(model: str) -> str:
    """Normalize Bedrock shorthand into a `bedrock/<model-id>` name."""
    raw = str(model or "").strip()
    if not raw:
        return f"bedrock/{DEFAULT_BEDROCK_MODEL_ID}"
    if raw.lower().startswith("bedrock/"):
        name = raw.split("/", 1)[1].strip()
    else:
        name = raw
    model_id = BEDROCK_MODEL_ALIASES.get(name.lower(), name)
    return f"bedrock/{model_id}"


def bedrock_model_id(model: str) -> str:
    """Return the Bedrock model ID from a `bedrock/...` model name."""
    canonical = canonicalize_bedrock_model(model)
    return canonical.split("/", 1)[1]


def bedrock_region() -> str:
    """Return the configured Bedrock region, defaulting to us-east-1."""
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("BEDROCK_REGION")
        or DEFAULT_BEDROCK_REGION
    ).strip()


def bedrock_api_key(api_key: str | None = None) -> str:
    """Return the Bedrock API key from an explicit override or env var."""
    key = (
        api_key
        or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or os.environ.get("BEDROCK_API_KEY")
        or ""
    )
    return key.strip()


def _bedrock_converse_url(model_id: str, region: str) -> str:
    encoded_model_id = quote(model_id, safe="")
    return f"https://bedrock-runtime.{region}.amazonaws.com/model/{encoded_model_id}/converse"


def _extract_data_url(url: str) -> tuple[str, str]:
    match = re.match(r"^data:([^;,]+)(;base64)?,(.*)$", url, flags=re.DOTALL)
    if not match:
        raise ValueError("Bedrock media inputs must be data URLs, not remote URLs")
    mime = match.group(1).lower()
    is_base64 = bool(match.group(2))
    data = match.group(3)
    if not is_base64:
        raise ValueError("Bedrock media data URLs must be base64 encoded")
    return mime, data


def _media_block_from_data_url(url: str) -> dict[str, Any]:
    mime, data = _extract_data_url(url)
    if mime in _IMAGE_MIME_TO_FORMAT:
        return {
            "image": {
                "format": _IMAGE_MIME_TO_FORMAT[mime],
                "source": {"bytes": data},
            }
        }
    if mime in _VIDEO_MIME_TO_FORMAT:
        return {
            "video": {
                "format": _VIDEO_MIME_TO_FORMAT[mime],
                "source": {"bytes": data},
            }
        }
    raise ValueError(f"Unsupported Bedrock media MIME type: {mime}")


def _openai_content_to_bedrock_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return [{"text": str(content)}] if content is not None else []

    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            if item:
                blocks.append({"text": item})
            continue
        if not isinstance(item, dict):
            blocks.append({"text": str(item)})
            continue

        block_type = item.get("type")
        if block_type in ("text", "input_text") or "text" in item:
            text = item.get("text", "")
            if text:
                blocks.append({"text": str(text)})
            continue

        if block_type in ("image_url", "input_image") or "image_url" in item:
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url", "")
            else:
                url = image_url or item.get("url", "")
            if url:
                blocks.append(_media_block_from_data_url(str(url)))
            continue

        # Preserve unknown structured content as text instead of silently
        # dropping potentially important prompt context.
        blocks.append({"text": json.dumps(item, ensure_ascii=False)})
    return blocks


def openai_messages_to_bedrock(
    messages: list[dict[str, Any]],
    *,
    json_mode: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Convert OpenAI-style chat messages into Bedrock Converse input."""
    system_blocks: list[dict[str, str]] = []
    bedrock_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = str(msg.get("role", "user")).lower()
        content = msg.get("content", "")

        if role == "system":
            for block in _openai_content_to_bedrock_blocks(content):
                if "text" in block and block["text"]:
                    system_blocks.append({"text": str(block["text"])})
            continue

        bedrock_role = "assistant" if role == "assistant" else "user"
        blocks = _openai_content_to_bedrock_blocks(content)
        if blocks:
            bedrock_messages.append({"role": bedrock_role, "content": blocks})

    if json_mode:
        system_blocks.append(
            {
                "text": (
                    "Return only valid JSON. Do not include markdown fences, "
                    "preamble, or trailing commentary."
                )
            }
        )

    if not bedrock_messages:
        bedrock_messages.append({"role": "user", "content": [{"text": ""}]})

    return system_blocks, bedrock_messages


def _extract_bedrock_text(body: dict[str, Any]) -> tuple[str, str | None]:
    try:
        content_blocks = body["output"]["message"].get("content", [])
    except KeyError as exc:
        raise RuntimeError(f"Unexpected Bedrock response: {body}") from exc

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            text_parts.append(str(block.get("text", "")))
        reasoning = block.get("reasoningContent")
        if isinstance(reasoning, dict):
            reasoning_text = reasoning.get("reasoningText")
            if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
                reasoning_parts.append(str(reasoning_text["text"]))

    reasoning = "\n".join(part for part in reasoning_parts if part).strip() or None
    return "".join(text_parts), reasoning


def query_bedrock_converse(
    model: str,
    messages: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    json_mode: bool = False,
    timeout: int = 200,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call Amazon Bedrock Converse and return project-standard content fields."""
    token = bedrock_api_key(api_key)
    if not token:
        raise RuntimeError(
            "Bedrock model requested but AWS_BEARER_TOKEN_BEDROCK is not set. "
            "Set AWS_BEARER_TOKEN_BEDROCK to the Bedrock API key."
        )

    model_id = bedrock_model_id(model)
    region = bedrock_region()
    system, bedrock_messages = openai_messages_to_bedrock(messages, json_mode=json_mode)
    inference_config: dict[str, Any] = {"maxTokens": max_tokens}
    # Claude Opus 4.x (and other extended-thinking models) deprecated the
    # `temperature` parameter — Bedrock returns 400 if it's passed. Same
    # pattern as gpt-5.x on OpenAI. Detect by model ID containing "opus-4"
    # and skip; older Sonnet 4.6 still accepts it.
    if "opus-4" not in model_id.lower():
        inference_config["temperature"] = temperature
    payload: dict[str, Any] = {
        "messages": bedrock_messages,
        "inferenceConfig": inference_config,
    }
    if system:
        payload["system"] = system

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    url = _bedrock_converse_url(model_id, region)

    start = time.time()
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    retry_count = 0
    while response.status_code in (404, 429, 500, 502, 503, 504) and retry_count < max_retries:
        retry_count += 1
        wait = 10 + retry_count * 5
        print(
            f"[bedrock] Retry {retry_count}: status {response.status_code}, "
            f"waiting {wait}s..."
        )
        time.sleep(wait)
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    elapsed = time.time() - start

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body_text = response.text.strip()
        if len(body_text) > 2000:
            body_text = body_text[:2000] + "... (truncated)"
        raise RuntimeError(
            f"{exc}; Bedrock response body: {body_text or '<empty>'}"
        ) from exc

    body = response.json()
    content, reasoning = _extract_bedrock_text(body)
    return {
        "content": content,
        "reasoning": reasoning,
        "usage": body.get("usage", {}),
        "metrics": body.get("metrics", {}),
        "raw": body,
        "model_id": model_id,
        "region": region,
        "elapsed": elapsed,
    }
