"""Tiered Gemini backend: Vertex → Developer API → OpenRouter.

Each ``query()`` call walks the fallback chain in this order:

  1. **Vertex AI** (via ``google.genai`` with ``vertexai=True``) on the
     project from ``GOOGLE_CLOUD_PROJECT`` (default ``davidchan-personal``).
     Used when ADC is present. Has 5 inner retries on 429.
  2. **Gemini Developer API** (``google.genai`` with ``api_key=...``)
     when ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) is set. Separate
     quota from Vertex. 5 inner retries on 429.
  3. **OpenRouter** (direct HTTP to openrouter.ai) when
     ``OPENROUTER_API_KEY`` is set. Final fallback. 5 inner retries on
     429.

A tier is skipped if its credential is missing. A tier "fails" (and the
next tier is tried) only after all 5 retries on 429 within that tier
have been exhausted — transient 429s don't escalate. Non-429 errors
propagate immediately.

Callers detect availability via ``is_available()`` (true if ANY tier
has its credential set) and call ``query()`` with OpenAI-shaped
messages.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

_DEFAULT_PROJECT = "davidchan-personal"
_DEFAULT_LOCATION = "global"

# Boundary remap from RATS-canonical names to Vertex-accepted names.
_DEFAULT_REMAP: dict[str, str] = {}

# Cached SDK clients per tier (built lazily).
_vertex_client: Any = None
_developer_client: Any = None
_available_cached: bool | None = None

_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)

_RETRY_BACKOFFS_S = [10, 30, 60, 120, 240]  # ~7 min cap total

# OpenRouter endpoint + suggested headers.
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# Credential probes
# ---------------------------------------------------------------------------

def _adc_present() -> bool:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
        return True
    home_adc = (
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    )
    return home_adc.exists()


def _developer_api_key() -> str:
    return (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )


def _openrouter_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def is_available() -> bool:
    """Return True if google.genai is installed AND at least one tier credential is present.

    Cached after first call. Set ``RATS_DISABLE_GENAI=1`` to force the
    legacy OpenRouter-proxy path even when credentials are present.
    """
    global _available_cached
    if os.environ.get("RATS_DISABLE_GENAI", "").strip() in ("1", "true", "True"):
        return False
    if _available_cached is not None:
        return _available_cached
    # OpenRouter doesn't need google.genai, so an OpenRouter-only setup is valid.
    if _openrouter_api_key():
        _available_cached = True
        return True
    try:
        import google.genai  # noqa: F401
    except Exception:
        _available_cached = False
        return False
    if not (_developer_api_key() or _adc_present()):
        _available_cached = False
        return False
    _available_cached = True
    return True


def remap_model(name: str) -> str:
    bare = name.split("/", 1)[-1] if "/" in name else name
    return _DEFAULT_REMAP.get(bare, bare)


def is_gemini_model(name: str) -> bool:
    """True for `google/gemini-*` or bare `gemini-*`. Excludes openrouter/*."""
    if not name:
        return False
    if name.startswith("openrouter/"):
        return False
    if name.startswith("google/gemini") or name.startswith("gemini-"):
        return True
    return False


# ---------------------------------------------------------------------------
# Message conversion (OpenAI shape → google.genai Content)
# ---------------------------------------------------------------------------

def _messages_to_genai(messages: list[dict[str, Any]]) -> tuple[list[Any], str | None]:
    """Convert OpenAI-shaped messages to (contents, system_instruction)."""
    from google.genai.types import Content, Part, Blob

    system_parts: list[str] = []
    contents: list[Any] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                if content:
                    system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = str(block.get("text", ""))
                        if text:
                            system_parts.append(text)
            continue

        gen_role = "model" if role == "assistant" else "user"
        parts: list[Any] = []

        if isinstance(content, str):
            parts.append(Part(text=content))
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    parts.append(Part(text=str(block)))
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append(Part(text=str(block.get("text", ""))))
                elif btype == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    m = _DATA_URL_RE.match(url) if isinstance(url, str) else None
                    if m:
                        mime = m.group(1)
                        try:
                            data = base64.b64decode(m.group(2))
                            parts.append(
                                Part(inline_data=Blob(mime_type=mime, data=data))
                            )
                        except Exception:
                            parts.append(Part(text=f"[malformed data URL: {mime}]"))
                    else:
                        parts.append(Part(text=f"[image: {url}]"))
                else:
                    parts.append(Part(text=str(block)))
        else:
            parts.append(Part(text=str(content)))

        if parts:
            contents.append(Content(role=gen_role, parts=parts))

    system_instruction = "\n\n".join(system_parts) if system_parts else None

    if system_instruction is not None and not contents:
        contents.append(Content(role="user", parts=[Part(text=system_instruction)]))
        system_instruction = None

    return contents, system_instruction


# ---------------------------------------------------------------------------
# Client builders (lazy, cached per process)
# ---------------------------------------------------------------------------

def _get_vertex_client() -> Any:
    global _vertex_client
    if _vertex_client is not None:
        return _vertex_client
    from google import genai
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip() or _DEFAULT_PROJECT
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip() or _DEFAULT_LOCATION
    _vertex_client = genai.Client(vertexai=True, project=project, location=location)
    print(f"[genai_backend] tier 1 Vertex AI ready (project={project}, location={location})")
    return _vertex_client


def _get_developer_client() -> Any:
    global _developer_client
    if _developer_client is not None:
        return _developer_client
    from google import genai
    api_key = _developer_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    _developer_client = genai.Client(api_key=api_key)
    print("[genai_backend] tier 2 Gemini Developer API ready (GEMINI_API_KEY set)")
    return _developer_client


# ---------------------------------------------------------------------------
# 429 detection
# ---------------------------------------------------------------------------

class _QuotaExhausted(Exception):
    """All retries within a tier exhausted on 429."""


def _is_429_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in msg
        or " 429 " in f" {msg} "
        or msg.startswith("429 ")
        or "Rate limit" in msg
    )


def _retry_loop_genai(tier_label: str, call_fn) -> Any:
    """Run call_fn() with 2-retry backoff on 429 (3 total attempts).

    Default 2 retries (~40s/tier worst case) is intentionally short:
    when the tier's quota is genuinely depleted, falling through to
    the next tier fast is more valuable than waiting for slow recovery.
    Overall fallback chain (vertex → developer → openrouter) caps at
    ~80s before reaching OpenRouter which is on a separate provider.
    Override with CAPX_GEMINI_RETRY_MAX env var.
    """
    max_retries = int(os.environ.get("CAPX_GEMINI_RETRY_MAX", "2"))
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except Exception as exc:
            if not _is_429_error(exc):
                raise
            last_exc = exc
            if attempt >= max_retries:
                print(
                    f"[genai_backend] tier {tier_label}: exhausted {max_retries + 1} "
                    f"attempts on 429; falling back to next tier",
                    flush=True,
                )
                raise _QuotaExhausted(str(exc)) from exc
            wait = _RETRY_BACKOFFS_S[min(attempt, len(_RETRY_BACKOFFS_S) - 1)]
            print(
                f"[genai_backend] tier {tier_label}: 429 on attempt "
                f"{attempt + 1}/{max_retries + 1}; sleeping {wait}s before retry",
                flush=True,
            )
            time.sleep(wait)
    # unreachable; defensive
    raise last_exc or RuntimeError(f"{tier_label}: no response and no exception")


def _normalize_genai_resp(resp: Any, model_name: str, elapsed: float) -> dict[str, Any]:
    content_text = resp.text or ""
    usage_meta = getattr(resp, "usage_metadata", None)
    prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
    completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
    reasoning_tokens = int(getattr(usage_meta, "thoughts_token_count", 0) or 0)
    return {
        "content": content_text,
        "reasoning": None,
        "model": model_name,
        "elapsed": elapsed,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Tier implementations
# ---------------------------------------------------------------------------

def _try_genai_tier(
    *,
    tier_label: str,
    client: Any,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: float | None,
    json_mode: bool,
) -> dict[str, Any]:
    from google.genai.types import GenerateContentConfig
    model_name = remap_model(model)
    contents, system_instruction = _messages_to_genai(messages)
    cfg_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
    if temperature is not None:
        cfg_kwargs["temperature"] = temperature
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction
    if json_mode:
        cfg_kwargs["response_mime_type"] = "application/json"
    config = GenerateContentConfig(**cfg_kwargs)
    start = time.time()
    resp = _retry_loop_genai(
        tier_label,
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=config
        ),
    )
    return _normalize_genai_resp(resp, model_name, time.time() - start)


def _try_openrouter(
    *,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: float | None,
    json_mode: bool,
) -> dict[str, Any]:
    """Final fallback: direct OpenRouter HTTP. Reuses messages in OpenAI shape."""
    import requests  # imported lazily so non-fallback paths don't need it

    api_key = _openrouter_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    # OpenRouter accepts the OpenAI model id directly when prefixed with
    # the provider, e.g. "google/gemini-3.1-pro-preview".
    or_model = model if "/" in model else f"google/{model}"
    payload: dict[str, Any] = {
        "model": or_model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _call() -> dict[str, Any]:
        r = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=300)
        if r.status_code == 429:
            raise RuntimeError(f"429 Rate limit from OpenRouter: {r.text[:200]}")
        if r.status_code != 200:
            raise RuntimeError(
                f"OpenRouter HTTP {r.status_code}: {r.text[:300]}"
            )
        return r.json()

    start = time.time()
    data = _retry_loop_genai("openrouter", _call)
    elapsed = time.time() - start
    choices = data.get("choices") or []
    content_text = ""
    if choices:
        msg = choices[0].get("message") or {}
        content_text = msg.get("content") or ""
    usage = data.get("usage") or {}
    return {
        "content": content_text,
        "reasoning": None,
        "model": or_model,
        "elapsed": elapsed,
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "reasoning_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# Top-level fallback orchestration
# ---------------------------------------------------------------------------

def query(
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int = 8192,
    temperature: float | None = None,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Call Gemini with tiered fallback (Vertex → Developer API → OpenRouter).

    Each tier is tried in order; an entire tier "fails" only after 5
    backoff retries on 429 have all been exhausted, at which point we
    fall through to the next available tier. Non-429 errors propagate.
    """
    tiers: list[tuple[str, Any]] = []

    # Per-tier opt-out flags, useful when a tier's quota is known dead
    # and we want to skip the ~80s retry/backoff overhead. Set
    # CAPX_DISABLE_VERTEX=1 / CAPX_DISABLE_DEVELOPER=1 / CAPX_DISABLE_OPENROUTER=1
    # in the subprocess env to skip a tier.
    def _disabled(key: str) -> bool:
        return os.environ.get(key, "").strip() in ("1", "true", "True")

    # Tier 1: Vertex (ADC)
    if _adc_present() and not _disabled("CAPX_DISABLE_VERTEX"):
        try:
            import google.genai  # noqa: F401
            tiers.append(("vertex", lambda: _try_genai_tier(
                tier_label="vertex",
                client=_get_vertex_client(),
                messages=messages, model=model, max_tokens=max_tokens,
                temperature=temperature, json_mode=json_mode,
            )))
        except Exception:
            pass

    # Tier 2: Developer API
    if _developer_api_key() and not _disabled("CAPX_DISABLE_DEVELOPER"):
        try:
            import google.genai  # noqa: F401
            tiers.append(("developer", lambda: _try_genai_tier(
                tier_label="developer",
                client=_get_developer_client(),
                messages=messages, model=model, max_tokens=max_tokens,
                temperature=temperature, json_mode=json_mode,
            )))
        except Exception:
            pass

    # Tier 3: OpenRouter
    if _openrouter_api_key() and not _disabled("CAPX_DISABLE_OPENROUTER"):
        tiers.append(("openrouter", lambda: _try_openrouter(
            messages=messages, model=model, max_tokens=max_tokens,
            temperature=temperature, json_mode=json_mode,
        )))

    if not tiers:
        raise RuntimeError(
            "[genai_backend] no tier available — set ADC, GEMINI_API_KEY, "
            "or OPENROUTER_API_KEY"
        )

    last_exc: Exception | None = None
    for label, fn in tiers:
        try:
            return fn()
        except _QuotaExhausted as exc:
            last_exc = exc
            # explicit fall-through to next tier
            continue
        except Exception as exc:
            # non-429 error: don't fall back, propagate
            raise
    raise last_exc or RuntimeError("[genai_backend] all tiers exhausted")
