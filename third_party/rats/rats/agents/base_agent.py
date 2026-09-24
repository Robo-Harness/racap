"""Thin wrapper around the configured LLM for all RATS agents.

Every agent in the system uses this shared interface to query the LLM.
Supports text + image inputs and structured JSON output parsing.

Routing:
  - If OPENAI_API_KEY is set: routes GPT models directly to OpenAI API.
  - If model starts with bedrock/: routes to Amazon Bedrock using
    AWS_BEARER_TOKEN_BEDROCK.
  - Else if .openrouterkey exists: routes via OpenRouter proxy.
  - Model name comes from call-site override, RATS_LLM_MODEL, or rats/config/default.yaml.
"""

from __future__ import annotations

import base64
import inspect
import itertools
import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import requests

from rats.control_flow import RATSProviderAbort
from rats.llm.gemini_proxy import (
    GeminiProxyError,
    GeminiProxyUnsupported,
    gemini_proxy_forced,
    query_gemini_proxy,
    should_try_gemini_proxy,
)
from rats.llm.bedrock import (
    canonicalize_bedrock_model,
    is_bedrock_model,
    query_bedrock_converse,
)

logger = logging.getLogger("rats.base_agent")


def should_use_gemini_proxy(model: str | None) -> bool:
    """Compatibility wrapper for tests and older callers."""
    return should_try_gemini_proxy(model)


# Per-call agent-IO logging. When RATS_AGENT_IO_DIR is set, every query_llm call
# writes a self-contained JSON record (and unpacks any base64 image payloads)
# so the user can audit exactly what each agent saw.
_IO_LOCK = threading.Lock()
_IO_COUNTER = itertools.count(1)


def _agent_io_dir() -> Path | None:
    val = os.environ.get("RATS_AGENT_IO_DIR", "").strip()
    if not val:
        return None
    p = Path(val).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _infer_caller() -> str:
    """Walk the call stack and return the first frame outside base_agent.py."""
    here = Path(__file__).resolve()
    try:
        for frame_info in inspect.stack()[1:]:
            fpath = Path(frame_info.filename).resolve()
            if fpath == here:
                continue
            # Prefer the module file's stem (e.g. "planner") + function name.
            return f"{fpath.stem}.{frame_info.function}"
    except Exception:
        pass
    return "unknown"


def _summarize_media_url(
    url: str,
    out_dir: Path,
    prefix: str,
    idx: int,
    *,
    kind: str = "image",
) -> dict[str, Any]:
    """For a data: URL, dump raw bytes to disk and return a placeholder dict.
    For an http(s) URL, return the URL itself.

    Agent-IO records should preserve exactly which media blocks were sent
    without inlining multi-megabyte base64 payloads into JSON.
    """
    if isinstance(url, str) and url.startswith("data:"):
        try:
            header, payload = url.split(",", 1)
            mime = header[5:].split(";", 1)[0] if header.startswith("data:") else "application/octet-stream"
            raw = base64.b64decode(payload)
            ext = mime.split("/", 1)[-1] or "bin"
            # Normalize common extensions.
            ext = {"jpeg": "jpg", "x-icon": "ico", "mp4;codecs=avc1": "mp4"}.get(ext, ext)
            ext = re.sub(r"[^a-zA-Z0-9]+", "_", ext).strip("_") or "bin"
            short = "vid" if kind == "video" else "img"
            fname = f"{prefix}_{short}_{idx}.{ext}"
            (out_dir / fname).write_bytes(raw)
            return {f"{kind}_ref": fname, "mime": mime, "bytes": len(raw)}
        except Exception as exc:
            return {f"{kind}_ref": None, "error": f"decode_failed: {exc}", "head": url[:80]}
    return {f"{kind}_url": str(url)[:500]}


def _summarize_image_url(url: str, out_dir: Path, prefix: str, idx: int) -> dict[str, Any]:
    return _summarize_media_url(url, out_dir, prefix, idx, kind="image")


def _detect_reasoning_loop(reasoning_text: str) -> tuple[bool, int, str]:
    """Detect thinking-model reasoning that is stuck in a paragraph loop.

    Gemini-3.1-pro-preview occasionally locks into rephrasing the same
    section header over and over (observed 7x in the LIBERO smoke run's
    failure_diagnoser call 0020 — 61s wall, no JSON output). We can't
    abort the call mid-stream from here, but we CAN detect after the fact
    and let callers treat the response as if it errored, rather than
    parsing scratch text into structured fields.

    Heuristic: split by markdown section headers (`**Header**`) and flag
    when one header dominates the trace. Two thresholds to avoid false
    positives on normal reasoning that revisits the same theme:
      - the header repeats ≥5 times absolutely, OR
      - the header repeats ≥4 times AND accounts for ≥40% of all headers.

    Returns (looped, max_count, repeated_header).
    """
    if not reasoning_text or len(reasoning_text) < 200:
        return False, 0, ""
    import re as _re
    headers = _re.findall(r"\*\*[^*\n]{3,80}\*\*", reasoning_text)
    if len(headers) < 4:
        return False, 0, ""
    from collections import Counter as _Counter
    counts = _Counter(headers)
    top_header, top_count = counts.most_common(1)[0]
    if top_count >= 5:
        return True, top_count, top_header
    if top_count >= 4 and top_count / len(headers) >= 0.4:
        return True, top_count, top_header
    return False, 0, ""


def _redact_messages(messages: list[dict[str, Any]], out_dir: Path, prefix: str) -> list[dict[str, Any]]:
    """Return a deep-redacted copy of messages where image data URLs are replaced
    with on-disk references. Text content is preserved verbatim."""
    redacted: list[dict[str, Any]] = []
    img_idx = 0
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            redacted.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            redacted.append({"role": role, "content": content})
            continue
        new_blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                new_blocks.append({"raw": str(block)[:500]})
                continue
            btype = block.get("type")
            if btype == "text":
                new_blocks.append({"type": "text", "text": block.get("text", "")})
            elif btype == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                media_kind = "video" if isinstance(url, str) and url.startswith("data:video/") else "image"
                summary = _summarize_media_url(url, out_dir, prefix, img_idx, kind=media_kind)
                img_idx += 1
                new_blocks.append({"type": "image_url", **summary})
            elif btype == "video_url":
                url = (block.get("video_url") or {}).get("url", "")
                summary = _summarize_media_url(url, out_dir, prefix, img_idx, kind="video")
                img_idx += 1
                new_blocks.append({"type": "video_url", **summary})
            else:
                new_blocks.append({k: v for k, v in block.items()})
        redacted.append({"role": role, "content": new_blocks})
    return redacted


_IO_NOTICE_PRINTED = False


def _write_io_record(
    *,
    messages: list[dict[str, Any]],
    model: str,
    payload: dict[str, Any],
    response_body: dict[str, Any] | None,
    response_content: str,
    response_reasoning: str | None,
    elapsed_s: float,
    error: str | None = None,
) -> None:
    out_dir = _agent_io_dir()
    if out_dir is None:
        return
    global _IO_NOTICE_PRINTED
    if not _IO_NOTICE_PRINTED:
        _IO_NOTICE_PRINTED = True
        print(f"[base_agent] agent IO logging enabled -> {out_dir}")
    with _IO_LOCK:
        idx = next(_IO_COUNTER)
    caller = _infer_caller()
    # Stamp PID into the filename so parallel subagent subprocesses (whose
    # _IO_COUNTER resets to 1) don't silently overwrite each other's records
    # when they happen to make the same kind of LLM call at the same index.
    prefix = f"{idx:04d}_p{os.getpid()}_{caller.replace('/', '_')}"
    try:
        redacted_messages = _redact_messages(messages, out_dir, prefix)
    except Exception as exc:
        redacted_messages = [{"role": "error", "content": f"redact failed: {exc}"}]
    record: dict[str, Any] = {
        "index": idx,
        "caller": caller,
        "model": model,
        "actual_model": (response_body or {}).get("model", ""),
        "request_id": (response_body or {}).get("id", ""),
        "episode_key": os.getenv("RATS_EPISODE_KEY", ""),
        "timestamp": time.time(),
        "elapsed_s": round(elapsed_s, 3),
        "request": {
            "messages": redacted_messages,
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens") or payload.get("max_completion_tokens"),
            "reasoning_effort": payload.get("reasoning_effort"),
            "response_format": payload.get("response_format"),
        },
        "response": {
            "content": response_content,
            "reasoning": response_reasoning,
            "usage": (response_body or {}).get("usage"),
            "finish_reason": ((response_body or {}).get("choices") or [{}])[0].get("finish_reason"),
        },
    }
    if error:
        record["error"] = error
    try:
        (out_dir / f"{prefix}.json").write_text(json.dumps(record, indent=2, default=str))
    except Exception as exc:
        # Logging must never break the run.
        print(f"[base_agent] WARN: failed to write IO record {prefix}: {exc}")


def _truncate_http_body(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "... (truncated)"


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_URL = os.environ.get(
    "RATS_OPENROUTER_URL",
    "http://localhost:8110/chat/completions",
)
VAPI_URL = os.environ.get(
    "RATS_VAPI_URL",
    "",
)
# Google's OpenAI-compatible Gemini endpoint: accepts the same chat/completions
# payload shape we already build, so no payload reshape is needed.
GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# Google's OpenAI-compatible endpoint currently accepts the same Gemini model
# ids we use internally. Keep this dict for explicit future aliases only; do
# not remap current preview ids to retired predecessor ids.
_GOOGLE_DIRECT_MODEL_REMAP: dict[str, str] = {
}

# Project root for finding .openrouterkey and config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load defaults from rats/config/default.yaml
_CONFIG: dict[str, Any] = {}
_config_path = _PROJECT_ROOT / "rats" / "config" / "default.yaml"
if _config_path.exists():
    import yaml
    with _config_path.open() as _f:
        _CONFIG = yaml.safe_load(_f) or {}

DEFAULT_MODEL = _CONFIG.get("llm_model", "openai/gpt-5.5")
DEFAULT_MAX_TOKENS = int(_CONFIG.get("llm_max_tokens", 8192))
DEFAULT_TEMPERATURE = float(_CONFIG.get("llm_temperature", 0.2))
DEFAULT_REASONING_EFFORT = _CONFIG.get("llm_reasoning_effort", "medium")


def canonicalize_model_name(model: str) -> str:
    """Normalize common shorthand into the provider-qualified model id.

    Examples:
      - gpt5.5 -> openai/gpt-5.5
      - gpt-5.5 -> openai/gpt-5.5
      - openai/gpt5.5 -> openai/gpt-5.5

    Non-GPT provider-qualified names are left unchanged.
    """
    model = str(model or "").strip()
    if not model:
        return DEFAULT_MODEL

    provider = ""
    name = model
    if "/" in model:
        provider, name = model.split("/", 1)

    # Accept shorthand such as "gpt5.5".
    name = re.sub(r"^gpt(?=\d)", "gpt-", name, flags=re.IGNORECASE)
    # Accept old / short Gemini names and route them to the current published
    # preview id. ``gemini-3-pro-preview`` is shut down on Google's direct API.
    if provider.lower() == "google" and name.lower() in {
        "gemini-3.1-pro",
        "gemini-3-pro",
        "gemini-3-pro-preview",
    }:
        name = "gemini-3.1-pro-preview"
    elif (
        provider.lower() == "openrouter"
        and name.lower() in {
            "google/gemini-3.1-pro",
            "google/gemini-3-pro",
            "google/gemini-3-pro-preview",
        }
    ):
        name = "google/gemini-3.1-pro-preview"
    elif not provider and name.lower() in {
        "gemini-3.1-pro",
        "gemini-3-pro",
        "gemini-3-pro-preview",
    }:
        return "google/gemini-3.1-pro-preview"
    # Bare Gemini ids users naturally type — promote to ``google/...`` so the
    # routing layer hits Google's direct API (and applies the model-id remap
    # via _GOOGLE_DIRECT_MODEL_REMAP). Without this, a bare ``gemini-3-pro-
    # preview`` fell through to direct OpenAI routing and burned an entire
    # 50-iter run (200/200 empty policy_writer responses) before the user
    # noticed missing videos.
    if not provider and name.lower() in (
        "gemini-3-pro-preview",
        "gemini-3.1-pro-preview",
        "gemini-3-pro",
    ):
        return "google/gemini-3.1-pro-preview"

    if provider.lower() == "bedrock":
        return canonicalize_bedrock_model(model)
    if provider:
        return f"{provider}/{name}"
    if name.lower().startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai/{name}"
    return name


def get_default_model() -> str:
    """Return the current default model, honoring runtime env overrides."""
    return canonicalize_model_name(os.environ.get("RATS_LLM_MODEL") or DEFAULT_MODEL)


def _get_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    key_path = _PROJECT_ROOT / ".openrouterkey"
    if not key_path.exists():
        return ""
    try:
        for raw in key_path.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        return ""
    return ""


def _get_gemini_key() -> str:
    """Return the Google Gemini API key, preferring GEMINI_API_KEY then GOOGLE_API_KEY."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.environ.get(var, "").strip()
        if key:
            return key
    return ""


def _get_api_config(model: str | None = None) -> tuple[str, dict[str, str]]:
    """Determine API URL and headers based on available credentials.

    Returns:
        (api_url, headers_dict)
    """
    vapi_key = os.environ.get("RATS_VAPI_KEY", "").strip()
    if vapi_key:
        from urllib.parse import urlsplit

        endpoint = urlsplit(VAPI_URL)
        local_http = endpoint.scheme == "http" and endpoint.hostname in ("127.0.0.1", "localhost", "::1")
        if (
            not endpoint.hostname or (endpoint.scheme != "https" and not local_http)
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
        ):
            raise RuntimeError("Configure an explicit credential-free HTTPS RATS_VAPI_URL (HTTP is allowed only on loopback).")
        return VAPI_URL, {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {vapi_key}",
        }

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openrouter_key = _get_openrouter_key()
    provider = ""
    if model and "/" in model:
        provider = model.split("/", 1)[0].strip().lower()

    # Explicit non-OpenAI provider prefixes (e.g. google/..., anthropic/...)
    # route through OpenRouter. Prefer the official API when an API key is
    # available; otherwise preserve the local proxy fallback used by older runs.
    # Previously any OPENAI_API_KEY in the environment forced direct OpenAI
    # routing, which made `--model google/...` fail even when OpenRouter was
    # configured.
    key_path = _PROJECT_ROOT / ".openrouterkey"
    if provider and provider != "openai":
        # google/... → Google's OpenAI-compatible endpoint when GEMINI_API_KEY
        # (or GOOGLE_API_KEY) is set. Takes precedence over OpenRouter so users
        # can A/B by setting / unsetting the env var. The model id is rewritten
        # at the call site (model_name stripping below) because Google's direct
        # API uses different ids than OpenRouter's `google/...` namespace.
        if provider == "google":
            gemini_key = _get_gemini_key()
            if gemini_key:
                return GOOGLE_API_URL, {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {gemini_key}",
                }
        if openrouter_key:
            return OPENROUTER_API_URL, {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openrouter_key}",
            }
        if key_path.exists():
            return OPENROUTER_URL, {"Content-Type": "application/json"}
        raise RuntimeError(
            f"Model {model!r} uses provider prefix {provider!r}, but direct "
            "OpenAI routing only supports OpenAI models. Set OPENROUTER_API_KEY "
            "or ensure .openrouterkey exists and the OpenRouter proxy is "
            "running, or use an openai/... model."
        )

    # Fail loud if the caller asked for a Gemini-shaped model but the only
    # remaining route is direct OpenAI. Previously this silently sent
    # ``gemini-*`` ids to OpenAI's /v1/chat/completions, which replied
    # ``model_not_found`` per request — the run logged warnings but kept
    # producing empty policy_writer drafts for 200 attempts before the user
    # spotted the missing videos. Better to abort the launch.
    if model and "gemini" in model.lower():
        raise RuntimeError(
            f"Model {model!r} looks like a Gemini id but no Gemini route is "
            "available: GEMINI_API_KEY / GOOGLE_API_KEY is unset (or use the "
            "``google/<id>`` prefix), and the Gemini proxy on "
            f"{os.environ.get('RATS_GEMINI_PROXY_URL') or 'http://127.0.0.1:8112'} "
            "is unreachable. Don't silently fall through to OpenAI."
        )

    if openai_key:
        return OPENAI_API_URL, {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}",
        }

    # Try OpenRouter proxy (no auth header needed, proxy handles it)
    if key_path.exists():
        return OPENROUTER_URL, {"Content-Type": "application/json"}

    raise RuntimeError(
        "No LLM credentials found. Set OPENAI_API_KEY env var or ensure "
        "OPENROUTER_API_KEY is set, or ensure .openrouterkey file exists and "
        "OpenRouter proxy is running."
    )


def _strip_videos_from_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of messages with any embedded mp4/video data URLs removed.

    Some fallback models (e.g. OpenAI GPT) don't accept video data URLs the
    way OpenRouter+Gemini does. Keeping the call alive with images + text
    only is strictly better than dropping the whole prompt. Conservative:
    only strips content blocks whose `image_url.url` contains "video/" or
    "data:video"; image blocks pass through untouched.
    """
    stripped: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            stripped.append(msg)
            continue
        new_blocks: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                if isinstance(url, str) and (
                    "data:video" in url[:32] or url.startswith("data:video")
                ):
                    continue
            new_blocks.append(block)
        new_msg = dict(msg)
        new_msg["content"] = new_blocks
        stripped.append(new_msg)
    return stripped


def _video_keyframes_for_image_only_models(
    messages: list[dict[str, Any]], max_frames: int = 3
) -> list[dict[str, Any]]:
    """Replace embedded videos with uniformly sampled PNG keyframes.

    The chat-completions route used by GPT-5.5 accepts images but rejects an
    mp4 carried in an ``image_url`` block.  Dropping the video altogether
    makes visual verification systematically weaker, so preserve its temporal
    evidence as first/middle/last frames instead.  On a decode error we fall
    back to the established video-stripping behavior.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue
        blocks: list[Any] = []
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "image_url"):
                blocks.append(block)
                continue
            url = (block.get("image_url") or {}).get("url", "")
            if not (isinstance(url, str) and url.startswith("data:video/")):
                blocks.append(block)
                continue
            try:
                import cv2

                _, payload = url.split(",", 1)
                raw = base64.b64decode(payload)
                with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
                    handle.write(raw)
                    handle.flush()
                    capture = cv2.VideoCapture(handle.name)
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    if frame_count <= 0:
                        capture.release()
                        continue
                    indices = sorted(
                        set(
                            int(round(i * (frame_count - 1) / max(1, max_frames - 1)))
                            for i in range(min(max_frames, frame_count))
                        )
                    )
                    frames: list[str] = []
                    for index in indices:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                        ok, frame = capture.read()
                        if not ok:
                            continue
                        encoded, png = cv2.imencode(".png", frame)
                        if encoded:
                            frames.append(
                                "data:image/png;base64,"
                                + base64.b64encode(png.tobytes()).decode("ascii")
                            )
                    capture.release()
                if frames:
                    blocks.append(
                        {
                            "type": "text",
                            "text": "Video keyframes follow in chronological order.",
                        }
                    )
                    blocks.extend(
                        {"type": "image_url", "image_url": {"url": frame_url}}
                        for frame_url in frames
                    )
            except Exception as exc:
                logger.warning("could not extract video keyframes: %s", exc)
        new_message = dict(message)
        new_message["content"] = blocks
        converted.append(new_message)
    return converted


# Fallback chain triggered when the primary model errors or returns empty
# content. Order matters: try cheaper / faster / shared-quota models first
# (Bedrock Opus 4.7 on a separate AWS quota), then a final OpenAI fallback
# that doesn't share OpenRouter's quota at all.
#
# Only the Gemini-3.1-pro path needs a chain — other models are either the
# fallback target themselves (gpt-5.5 has nowhere else to fall) or aren't
# load-bearing enough to justify a per-call extra LLM hop.
#
# Behavior: see query_llm_with_fallback below. Set env RATS_LLM_FALLBACK=0
# to disable at runtime without code changes.
FALLBACK_CHAINS: dict[str, list[str]] = {
    "google/gemini-3.1-pro-preview": [
        "bedrock/claude-opus-4-7",
        "openai/gpt-5.5",
    ],
}


def _abort_for_provider_error(status_code: int, body_text: str, model: str) -> None:
    sentinel = os.getenv("RATS_ABORT_SENTINEL", "").strip()
    if not sentinel:
        return
    lowered = body_text.lower()
    markers = (
        "insufficient_user_quota",
        "insufficient quota",
        "insufficient balance",
        "balance is insufficient",
        "need pre-deduct",
        "no available channel",
        "no available route",
    )
    if not any(marker in lowered for marker in markers):
        return
    path = Path(sentinel).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "time": time.time(),
                "reason": "provider_or_quota_unavailable",
                "status_code": status_code,
                "model": model,
                "episode_key": os.getenv("RATS_EPISODE_KEY", ""),
                "body_preview": body_text[:1000],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    raise RATSProviderAbort(f"provider unavailable for {model}; checkpoint written to {path}")


def _resolve_fallback_chain(model: str | None) -> list[str]:
    """Look up the fallback chain for a primary model.

    Returns [] when fallback is disabled (env) or no chain is configured.
    """
    if os.getenv("RATS_LLM_FALLBACK", "1") == "0":
        return []
    canonical = canonicalize_model_name(model) if model else get_default_model()
    return list(FALLBACK_CHAINS.get(canonical, []))


def query_llm_with_fallback(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Wrap query_llm with a per-model fallback chain.

    Fallback is triggered on:
      - Any exception raised by query_llm (HTTPError, RuntimeError, etc.).
        Covers OpenRouter 403 (quota), 429 (rate limit), and 5xx after the
        internal retry loop in query_llm has already exhausted its budget.
      - Empty content. The stuck-loop watchdog inside query_llm sets
        content="" when it detects a Gemini reasoning loop with header
        repetition; that case should also fall through to the next model.

    Non-fallback errors that should NOT trigger the chain (bad request,
    unsupported model) currently DO trigger fallback for simplicity — the
    next model would just raise the same kind of error and we'd surface
    that one instead. Acceptable trade-off given the rarity of malformed
    requests in production.

    On every fallback hop, video data URLs are stripped from the messages
    because most non-Gemini multimodal models reject mp4 inputs. Image
    blocks pass through, so the verifier still has visual context.
    """
    primary = canonicalize_model_name(model) if model else get_default_model()
    chain = [primary] + _resolve_fallback_chain(primary)
    primary_messages = (
        _video_keyframes_for_image_only_models(messages)
        if primary.startswith("openai/")
        else messages
    )
    if len(chain) == 1:
        return query_llm(
            primary_messages, model=primary, temperature=temperature,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
            json_mode=json_mode,
        )

    last_exc: Exception | None = None
    for idx, try_model in enumerate(chain):
        msgs = primary_messages if idx == 0 else _video_keyframes_for_image_only_models(messages)
        try:
            result = query_llm(
                msgs, model=try_model, temperature=temperature,
                max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                json_mode=json_mode,
            )
            content = (result.get("content") or "").strip()
            if content:
                if idx > 0:
                    print(
                        f"[base_agent] FALLBACK succeeded on {try_model} "
                        f"(primary {primary} failed)"
                    )
                return result
            # Empty content. If this was the last hop, return it; otherwise
            # try the next model in the chain.
            if idx == len(chain) - 1:
                return result
            print(
                f"[base_agent] Empty content from {try_model}; "
                f"trying next fallback in chain"
            )
        except Exception as exc:
            last_exc = exc
            if idx == len(chain) - 1:
                raise
            print(
                f"[base_agent] {try_model} raised "
                f"{type(exc).__name__}: {str(exc)[:200]}; "
                f"trying next fallback in chain"
            )
    # Defensive: every chain entry should either return or raise. If we
    # somehow get here, re-raise the last exception or return an empty
    # response stub.
    if last_exc is not None:
        raise last_exc
    return {"content": "", "reasoning": None}


def query_llm(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Query the LLM and return parsed response.

    Args:
        messages: OpenAI-format message list.
        model: Model identifier (e.g. 'openai/gpt-5.5').
        temperature: Sampling temperature.
        max_tokens: Max completion tokens.
        reasoning_effort: Reasoning effort for capable models.
        json_mode: If True, request JSON output format.

    Returns:
        Dict with 'content' (str) and 'reasoning' (str | None).
    """
    model = canonicalize_model_name(model) if model else get_default_model()
    if is_bedrock_model(model):
        result = query_bedrock_converse(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        usage = result.get("usage", {})
        prompt_tokens = int(
            usage.get("inputTokens")
            or usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        cached_tokens = int(
            usage.get("cacheReadInputTokens")
            or usage.get("cache_read_input_tokens")
            or usage.get("cached_tokens")
            or 0
        )
        cache_pct = (100.0 * cached_tokens / prompt_tokens) if prompt_tokens else 0.0
        print(
            f"[base_agent] LLM query took {result.get('elapsed', 0.0):.1f}s "
            f"(model={result.get('model_id', model)}, provider=bedrock, "
            f"region={result.get('region')}) prompt={prompt_tokens} "
            f"cached={cached_tokens} ({cache_pct:.0f}%)"
        )
        return {
            "content": str(result.get("content") or ""),
            "reasoning": result.get("reasoning"),
        }

    # Routing priority for Gemini:
    #   1. google.genai / Vertex AI when the SDK is installed and ADC is
    #      configured with the caller's own credentials.
    #   2. A caller-provisioned loopback gemini_proxy at 127.0.0.1:8112
    #      when reachable.
    #   3. Existing Google-OpenAI-compatible / OpenRouter routing below.
    from rats.llm import genai_backend

    if (
        not gemini_proxy_forced()
        and genai_backend.is_gemini_model(model)
        and genai_backend.is_available()
    ):
        start = time.time()
        try:
            result = genai_backend.query(
                messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
            )
        except Exception as exc:
            elapsed = time.time() - start
            _write_io_record(
                messages=messages, model=model, payload={"backend": "genai"},
                response_body=None, response_content="", response_reasoning=None,
                elapsed_s=elapsed, error=f"genai backend failed: {exc}",
            )
            raise
        elapsed = result.get("elapsed", time.time() - start)
        model_name = result.get("model", model)
        content = str(result.get("content") or "")
        usage = result.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
        print(
            f"[base_agent] LLM query took {elapsed:.1f}s "
            f"(model={model_name}, provider=vertex) "
            f"prompt={prompt_tokens} completion={completion_tokens} "
            f"thoughts={reasoning_tokens}"
        )
        _write_io_record(
            messages=messages, model=model_name,
            payload={"backend": "genai", "max_tokens": max_tokens,
                     "temperature": temperature, "json_mode": json_mode},
            response_body={"usage": usage}, response_content=content,
            response_reasoning=None, elapsed_s=elapsed,
        )
        return {"content": content, "reasoning": None}

    direct_google_available = model.startswith("google/") and bool(_get_gemini_key())
    if not direct_google_available and should_use_gemini_proxy(model):
        try:
            result = query_gemini_proxy(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            usage = result.raw.get("usageMetadata", {})
            prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
            total_tokens = int(usage.get("totalTokenCount", 0) or 0)
            print(
                f"[base_agent] LLM query took {result.elapsed_s:.1f}s "
                f"(model={result.model_name}, provider=gemini-proxy) "
                f"prompt={prompt_tokens} total={total_tokens}"
            )
            response_body = {
                "usage": usage,
                "choices": [
                    {
                        "finish_reason": (
                            (result.raw.get("candidates") or [{}])[0].get("finishReason")
                        )
                    }
                ],
                "gemini": result.raw,
            }
            _write_io_record(
                messages=messages,
                model=result.model_name,
                payload=result.payload,
                response_body=response_body,
                response_content=result.content,
                response_reasoning=None,
                elapsed_s=result.elapsed_s,
            )
            return {"content": result.content, "reasoning": result.reasoning}
        except GeminiProxyUnsupported as exc:
            print(f"[base_agent] Gemini proxy cannot handle this prompt; falling back: {exc}")
        except GeminiProxyError as exc:
            print(f"[base_agent] Gemini proxy failed; falling back: {exc}")

    api_url, headers = _get_api_config(model)

    # For direct OpenAI, strip only the explicit OpenAI provider prefix.
    if api_url in (OPENAI_API_URL, VAPI_URL) and model.startswith("openai/"):
        model_name = model.split("/", 1)[-1] if "/" in model else model
    elif api_url == GOOGLE_API_URL and model.startswith("google/"):
        bare = model.split("/", 1)[-1]
        model_name = _GOOGLE_DIRECT_MODEL_REMAP.get(bare, bare)
    elif api_url in (OPENROUTER_API_URL, OPENROUTER_URL) and model.startswith(
        "openrouter/"
    ):
        model_name = model.split("/", 1)[-1]
    else:
        model_name = model

    # Build payload based on target API
    # For reasoning models, max_completion_tokens covers BOTH reasoning + output.
    if api_url == OPENAI_API_URL:
        payload: dict[str, Any] = {
            "model": model_name,
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_tokens,
            "messages": messages,
        }
    else:
        payload = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }

    if json_mode and (
        api_url == OPENAI_API_URL
        or api_url == GOOGLE_API_URL
        or os.getenv("RATS_OPENROUTER_RESPONSE_FORMAT", "0") == "1"
    ):
        payload["response_format"] = {"type": "json_object"}

    # NOTE — Gemini prompt-cache investigation (deferred follow-up)
    #
    # In the LIBERO smoke run, 57 Gemini calls had
    # `prompt_tokens_details.cached_tokens=0 AND cache_write_tokens=0`
    # i.e. cache wasn't being read OR written. OpenAI calls in the same
    # run hit ~33% token cache rate. Two likely causes:
    #   1. The local OpenRouter proxy at `OPENROUTER_URL` may strip
    #      `cache_control` directives on Google routes.
    #   2. We don't actually set any cache directives — OpenAI's cache is
    #      automatic for stable prefixes, but Anthropic/Google via
    #      OpenRouter typically need explicit `cache_control` markers
    #      on content blocks.
    #
    # Worth investigating: send a `cache_control={"type": "ephemeral"}`
    # marker on the system prompt block when routing to Google, and
    # confirm the proxy forwards it. Skipped here because it needs a
    # proxy-side change to verify the round-trip.

    start = time.time()
    response = requests.post(api_url, headers=headers, json=payload, timeout=200)

    # Retry on transient errors
    retry_count = 0
    while response.status_code in (404, 500, 502, 503, 504) and retry_count < 3:
        retry_count += 1
        wait = 10 + retry_count * 5
        body_preview = _truncate_http_body(response.text)
        logger.warning(
            "LLM HTTP retry %s/3: status=%s model=%s url=%s wait=%ss body=%s",
            retry_count,
            response.status_code,
            model_name,
            api_url,
            wait,
            body_preview or "<empty>",
        )
        print(
            f"[base_agent] Retry {retry_count}: status {response.status_code}, "
            f"model={model_name}, url={api_url}, body={body_preview or '<empty>'}, "
            f"waiting {wait}s..."
        )
        time.sleep(wait)
        response = requests.post(api_url, headers=headers, json=payload, timeout=200)

    elapsed = time.time() - start

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body_text = _truncate_http_body(response.text)
        logger.error(
            "LLM HTTP failure: status=%s model=%s url=%s body=%s",
            response.status_code,
            model_name,
            api_url,
            body_text or "<empty>",
        )
        # if len(body_text) > 2000:
        #     body_text = body_text[:2000] + "... (truncated)"
        _write_io_record(
            messages=messages, model=model_name, payload=payload,
            response_body=None, response_content="", response_reasoning=None,
            elapsed_s=elapsed, error=f"HTTPError: {exc}; body: {body_text}",
        )
        _abort_for_provider_error(response.status_code, body_text, model_name)
        raise RuntimeError(
            f"{exc}; response body: {body_text or '<empty>'}"
        ) from exc
    body = response.json()

    # Cache-hit telemetry: OpenAI auto-caches prefixes >=1024 tokens; the
    # discount only kicks in when a prior request had the exact same prefix
    # bytes. Logging cached/prompt tokens makes it easy to tell whether
    # prompt restructuring (stable content first) is actually paying off.
    usage = body.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    pt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(pt_details.get("cached_tokens", 0) or 0)
    cache_pct = (100.0 * cached_tokens / prompt_tokens) if prompt_tokens else 0.0
    print(
        f"[base_agent] LLM query took {elapsed:.1f}s (model={model_name}) "
        f"prompt={prompt_tokens} cached={cached_tokens} ({cache_pct:.0f}%)"
    )

    try:
        msg = body["choices"][0]["message"]
        content = msg.get("content")
        if content is None:
            content = ""
        elif isinstance(content, list):
            # Multimodal / structured content blocks
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            content = "".join(parts)
        else:
            content = str(content)
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {body}") from exc

    # If content is empty but reasoning exists, the model put code in reasoning.
    # WHY-gated: some Anthropic-routed flows legitimately put the answer in
    # `reasoning`. But for Gemini and other thinking models, `reasoning` is the
    # *thinking trace* (not the answer), and substituting it produces nonsense.
    # We only substitute when finish_reason indicates a normal stop — if the
    # call truncated or errored, the reasoning is partial scratch, never the
    # answer.
    reasoning_content = msg.get("reasoning") or msg.get("reasoning_content")
    if isinstance(reasoning_content, dict):
        reasoning_content = reasoning_content.get("text") or reasoning_content.get("content")
    if reasoning_content is not None:
        reasoning_content = str(reasoning_content)

    finish_reason = (body.get("choices") or [{}])[0].get("finish_reason")
    usage = body.get("usage", {})
    det = usage.get("completion_tokens_details", {})

    if not content.strip() and reasoning_content and reasoning_content.strip():
        if finish_reason in {"length", "error"}:
            print(
                f"[base_agent] WARNING: empty content with finish_reason={finish_reason!r}; "
                f"NOT substituting reasoning ({len(reasoning_content)} chars of scratch). "
                f"prompt_tokens={usage.get('prompt_tokens')}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"reasoning_tokens={det.get('reasoning_tokens')}"
            )
        else:
            print(f"[base_agent] Content empty, using reasoning ({len(reasoning_content)} chars)")
            content = reasoning_content
    elif not content.strip():
        # Debug: dump the full response to diagnose empty content
        print(f"[base_agent] WARNING: empty content! prompt_tokens={usage.get('prompt_tokens')}, "
              f"completion_tokens={usage.get('completion_tokens')}, "
              f"reasoning_tokens={det.get('reasoning_tokens')}, "
              f"finish_reason={finish_reason}")

    reasoning = None
    if body.get("choices"):
        reasoning = body["choices"][0].get("message", {}).get("reasoning")

    # Stuck-loop watchdog: if reasoning shows a markdown header repeated
    # >=3 times, treat the response as if it errored — the actual content
    # is almost always partial scratch when this happens. Don't return the
    # content even if it's non-empty: a partial JSON object from inside a
    # loop is worse than no JSON at all because downstream parsing might
    # accept it.
    reasoning_str = reasoning if isinstance(reasoning, str) else None
    if reasoning_str:
        looped, top_count, top_header = _detect_reasoning_loop(reasoning_str)
        if looped:
            print(
                f"[base_agent] WARNING: reasoning loop detected — header "
                f"{top_header!r} repeated {top_count}x. Discarding content "
                f"({len(content)} chars) and treating as no-answer."
            )
            content = ""

    _write_io_record(
        messages=messages, model=model_name, payload=payload,
        response_body=body, response_content=content,
        response_reasoning=reasoning if isinstance(reasoning, str) else (
            json.dumps(reasoning) if reasoning is not None else None
        ),
        elapsed_s=elapsed,
    )

    return {"content": content, "reasoning": reasoning}


def strip_markdown_json_fences(text: str) -> str:
    """Return ``text`` with a leading/trailing markdown code fence removed.

    Gemini-3.1-pro (and occasionally other reasoning-heavy models) emits
    JSON wrapped in ``​```json ... ```​`` markdown fences even when the
    request set ``json_mode=True``. ``json.loads`` on such a string raises
    ``Expecting value: line 1 column 1 (char 0)`` — the same message as
    "empty content", which makes the failure mode confusing to diagnose.

    Audit on outputs/play_molmospaces_3iter_v7 found 614/621 (98.9%) of
    ``vlm_verify`` responses had the fenced wrapper; every one of them
    parse-failed and returned ``verified=False, reasoning="vlm_error:
    JSONDecodeError"`` to the policy code. Stripping the fence at the
    caller's parsing site fixes the false-negative cascade with no
    upstream provider change.

    Conservative behavior:
    - Only strips if the text starts with ``​``​`` (with or without a
      language tag) and ends with ``​``​``. Plain JSON passes through
      untouched.
    - Leading / trailing whitespace is trimmed both before and after the
      fence removal.
    """
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    # Drop the opening fence + optional language tag, up to the first newline.
    first_nl = s.find("\n")
    if first_nl == -1:
        # Single-line ```...``` — fall back to a regex-style strip.
        s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()
    s = s[first_nl + 1 :]
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    return s.strip()


def query_llm_text(
    system_prompt: str,
    user_prompt: str,
    *,
    images: list[str] | None = None,
    videos: list[str] | None = None,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    json_mode: bool = False,
) -> str:
    """Convenience wrapper: system + user text (+ optional image/video media) -> response text."""
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    if videos:
        for video_url in videos:
            # Match origin/main's OpenAI/OpenRouter-compatible chat schema:
            # carry video data URLs in an image_url content block. The local
            # OpenRouter proxy / upstream models that accept video via this
            # path reject explicit {"type": "video_url"} blocks.
            user_content.append({"type": "image_url", "image_url": {"url": video_url}})
    if images:
        for img_url in images:
            user_content.append({"type": "image_url", "image_url": {"url": img_url}})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    # Route through the fallback wrapper so Gemini-3.1-pro callers
    # automatically get Bedrock Opus 4.7 → OpenAI gpt-5.5 retries on
    # quota / stuck-loop / provider errors. For models without a
    # configured chain, the wrapper just delegates to query_llm.
    result = query_llm_with_fallback(
        messages, model=model, temperature=temperature,
        max_tokens=max_tokens, reasoning_effort=reasoning_effort,
        json_mode=json_mode,
    )
    text = (result.get("content") or "").strip()
    # FIX: do NOT substitute `reasoning` into `text` here. The low-level
    # query_llm at base_agent.py:430 already handles the legitimate case
    # (substitutes reasoning into content when finish_reason="stop" but
    # content is empty — for the few Anthropic-routed flows that put the
    # answer in reasoning). When that low-level branch chose NOT to
    # substitute — because finish_reason was "length" or "error", meaning
    # the reasoning is partial scratch — this wrapper was undoing the
    # gate by grabbing the scratch anyway. That's exactly how Gemini's
    # stuck-loop text in 0020_failure_diagnoser ended up flowing into
    # the policy_writer's retry prompt as if it were a real diagnosis.
    # The right thing to do is return whatever query_llm returned.
    return text


def query_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    images: list[str] | None = None,
    videos: list[str] | None = None,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict[str, Any]:
    """Query LLM and parse response as JSON."""
    text = query_llm_text(
        system_prompt, user_prompt,
        images=images, videos=videos, model=model, temperature=temperature,
        max_tokens=max_tokens, reasoning_effort=reasoning_effort, json_mode=True,
    )
    return parse_json_response(text)


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break
    return {"raw": text}



def video_file_to_data_url(path: str | Path, mime: str | None = None) -> str | None:
    """Read a video file into a base64 data URL for multimodal LLM calls."""
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        raw = p.read_bytes()
        if mime is None:
            suffix = p.suffix.lower()
            mime = "video/mp4" if suffix in {".mp4", ".m4v"} else "application/octet-stream"
        return f"data:{mime};base64," + base64.b64encode(raw).decode("utf-8")
    except Exception:
        return None

def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response (handles markdown fences)."""
    if not text or not str(text).strip():
        return ""
    text = str(text).strip()

    # Prefer a fenced ```python ... ``` block (non-greedy)
    m = re.search(r"```(?:python|py)\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner

    fence = "```python"
    if fence in text:
        after = text.split(fence, 1)[1]
        if "```" in after:
            inner = after.split("```", 1)[0].strip()
            if inner:
                return inner
        else:
            # Truncated response: no closing fence — use everything after ```python
            inner = after.strip()
            if inner:
                return inner

    # Generic ``` ... ``` (optional language line)
    m2 = re.search(r"```[^\n`]*\n(.*?)```", text, re.DOTALL)
    if m2:
        inner = m2.group(1).strip()
        if inner:
            return inner

    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            cand = parts[1].strip()
            low = cand.split("\n", 1)[0].strip().lower()
            if low in ("python", "py"):
                cand = cand.split("\n", 1)[1].strip() if "\n" in cand else ""
            if cand:
                return cand

    return text.strip()


def image_to_data_url(image) -> str | None:
    """Convert numpy array image to base64 data URL."""
    if image is None:
        return None
    import base64
    import io

    import numpy as np
    from PIL import Image

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
