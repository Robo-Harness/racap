"""LLM client utilities for querying language models.

Extracted from capx/utils/launch_utils.py to separate LLM query logic
from launch/config utilities.
"""

from __future__ import annotations

import mimetypes
import concurrent.futures
import copy
import json
import os
import random
import socket
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

GPT_MODELS = [
    "gpt-5.5",
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/o4-mini",
]
VLM_MODELS = [
    "gpt-5.5",
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-5.4",
    "openai/o1",
    "openai/o4-mini",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "qwen/qwen3.5-122b-a10b",
    "moonshotai/kimi-k2",
]
CLAUDE_MODELS = ["anthropic/claude-opus-4-5", "anthropic/claude-haiku-4-5"]
OSS_MODELS = [
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "qwen/qwen3.5-122b-a10b",
    "moonshotai/kimi-k2",
]
OPENROUTER_MODELS = [
    "openrouter/google/gemini-2.5-pro-preview",
    "openrouter/google/gemini-2.5-flash-preview",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/anthropic/claude-opus-4",
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-chat-v3-0324",
    "openrouter/openai/gpt-4.1",
    "openrouter/openai/o4-mini",
    "openrouter/meta-llama/llama-4-maverick",
    "openrouter/qwen/qwen3-235b-a22b",
]
OPENROUTER_SERVER_URL = "http://localhost:8110/chat/completions"
GEMINI_PROXY_URL = "http://127.0.0.1:8112"
GEMINI_PROXY_DEFAULT_MODELS = {
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
}
_GEMINI_PROXY_HEALTH_CACHE: dict[str, bool] = {}


def _append_call_telemetry(record: dict[str, Any]) -> None:
    """Write compact cost/latency metadata without prompts or credentials."""
    configured = os.getenv("CAPX_LLM_TELEMETRY_PATH", "").strip()
    if not configured:
        return
    try:
        path = os.path.abspath(configured)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "time": time.time(),
            "pid": os.getpid(),
            "episode_key": os.getenv("CAPX_EPISODE_KEY", ""),
            **record,
        }
        line = (json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)
    except OSError:
        pass

# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------

ENSEMBLE_CONFIGS = [
    # Gemini-3-Pro only — best single model per CaP-Bench (Figure 1).
    # 3 temps for diversity; synthesis still uses Gemini-3-Pro.
    # ~45% faster than full multimodel (no Claude/GPT latency bottleneck).
    ("openai/gpt-5.4", [0.1, 0.5, 0.9]),
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def is_openrouter_model(model: str) -> bool:
    """Return True if the model should be routed through the OpenRouter proxy."""
    return model.startswith("openrouter/") or model in OPENROUTER_MODELS


def is_gemini_model(model: str) -> bool:
    """Return True for Gemini model names that can use Junyi's local proxy."""
    return model.startswith("google/gemini") or model.startswith("gemini-")


def _gemini_native_model_name(model: str) -> str:
    """Convert capx/OpenRouter-style Gemini ids to Vertex Gemini model ids."""
    if model.startswith("google/"):
        return model.split("/", 1)[1]
    return model


def _env_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_falsey(value: str | None) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "n", "off"}


def _server_url_points_to_gemini_proxy(server_url: str | None) -> bool:
    if not server_url:
        return False
    parsed = urlparse(server_url)
    return parsed.scheme in {"http", "https"} and parsed.port == 8112


def _gemini_proxy_base_url(server_url: str | None = None) -> str:
    if _server_url_points_to_gemini_proxy(server_url):
        parsed = urlparse(str(server_url))
        return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    return os.getenv("CAPX_GEMINI_PROXY_URL", GEMINI_PROXY_URL).rstrip("/")


def _gemini_proxy_model_allowed(model: str) -> bool:
    configured = os.getenv("CAPX_GEMINI_PROXY_MODELS", "").strip()
    native = _gemini_native_model_name(model)
    if configured:
        allowed = {m.strip() for m in configured.split(",") if m.strip()}
        return "*" in allowed or model in allowed or native in allowed
    return native in GEMINI_PROXY_DEFAULT_MODELS


def _hostname_allowed_for_gemini_proxy() -> bool:
    configured = os.getenv("CAPX_GEMINI_PROXY_HOSTNAMES", "").strip()
    if not configured:
        return False
    allowed = {h.strip() for h in configured.split(",") if h.strip()}
    return socket.gethostname() in allowed


def _gemini_proxy_available(base_url: str) -> bool:
    cached = _GEMINI_PROXY_HEALTH_CACHE.get(base_url)
    if cached is not None:
        return cached
    timeout = float(os.getenv("CAPX_GEMINI_PROXY_HEALTH_TIMEOUT", "0.5"))
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        ok = response.ok and response.json().get("status") == "ok"
    except Exception:
        ok = False
    _GEMINI_PROXY_HEALTH_CACHE[base_url] = ok
    return ok


def should_use_gemini_proxy(model: str, server_url: str | None = None) -> bool:
    """Decide whether a Gemini request should use Junyi's local proxy.

    Default behavior is auto-detect: only known proxy-tested Gemini models
    switch when the local proxy health endpoint is reachable. Set
    CAPX_USE_GEMINI_PROXY=1 to force it, CAPX_USE_GEMINI_PROXY=0 to disable it,
    or pass --server-url http://127.0.0.1:8112 for an explicit per-run switch.
    """
    if not is_gemini_model(model):
        return False

    mode = os.getenv("CAPX_USE_GEMINI_PROXY", "auto")
    if _env_falsey(mode):
        return False
    if _env_truthy(mode):
        return True
    if _server_url_points_to_gemini_proxy(server_url):
        return True
    if _hostname_allowed_for_gemini_proxy():
        return True
    if not _gemini_proxy_model_allowed(model):
        return False
    return _gemini_proxy_available(_gemini_proxy_base_url(server_url))


def _data_url_to_inline_data(url: str) -> dict[str, dict[str, str]]:
    header, payload = url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Gemini proxy only supports base64 data URLs")
    mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    return {"inlineData": {"mimeType": mime_type, "data": payload}}


def _url_to_gemini_media_part(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        return _data_url_to_inline_data(url)
    mime_type = mimetypes.guess_type(url)[0] or "application/octet-stream"
    return {"fileData": {"mimeType": mime_type, "fileUri": url}}


def _content_block_to_gemini_parts(block: Any) -> list[dict[str, Any]]:
    if isinstance(block, str):
        return [{"text": block}]
    if not isinstance(block, dict):
        return [{"text": str(block)}]

    block_type = block.get("type")
    if block_type in {"text", "input_text"} or "text" in block:
        return [{"text": str(block.get("text", ""))}]

    if block_type in {"image_url", "input_image"} or "image_url" in block:
        image_url = block.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not url:
            return []
        return [_url_to_gemini_media_part(str(url))]

    if block_type in {"video_url", "input_video"} or "video_url" in block:
        video_url = block.get("video_url")
        url = video_url.get("url") if isinstance(video_url, dict) else video_url
        if not url:
            return []
        return [_url_to_gemini_media_part(str(url))]

    return [{"text": json.dumps(block, default=str)}]


def _message_content_to_gemini_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for block in content:
            parts.extend(_content_block_to_gemini_parts(block))
        return parts
    return _content_block_to_gemini_parts(content)


def _messages_to_gemini_payload(
    prompt: list[dict],
    *,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, Any]] = []

    for message in prompt:
        role = str(message.get("role", "user"))
        parts = _message_content_to_gemini_parts(message.get("content", ""))
        if not parts:
            continue
        if role == "system":
            system_parts.extend(parts)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        if contents and contents[-1].get("role") == gemini_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": gemini_role, "parts": parts})

    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    return payload


def _parse_gemini_proxy_response(body: dict[str, Any]) -> dict[str, Any]:
    try:
        parts = body["candidates"][0]["content"].get("parts", [])
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini proxy response format: {body}") from exc

    content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    return {"content": content, "reasoning": None}


def query_gemini_proxy(args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]) -> dict[str, Any]:
    """Query Junyi's local Gemini proxy using Gemini Developer API format."""
    base_url = _gemini_proxy_base_url(getattr(args, "server_url", None))
    model_name = _gemini_native_model_name(args.model)
    url = f"{base_url}/v1beta/models/{model_name}:generateContent"
    payload = _messages_to_gemini_payload(
        prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    headers = {"Content-Type": "application/json"}
    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload, timeout=200)
    retry = 1
    while response.status_code in [500, 502, 503, 504] and retry <= 3:
        sleep_time = 10 + retry * 5
        print(
            f"Retry {retry}. Gemini proxy query failed with status code "
            f"{response.status_code}. Error: {response.text}. Retrying in {sleep_time} seconds..."
        )
        time.sleep(sleep_time)
        response = requests.post(url, headers=headers, json=payload, timeout=200)
        retry += 1

    end_time = time.time()
    print(f"Time taken to query Gemini proxy: {end_time - start_time:.2f} seconds")
    response.raise_for_status()
    body = response.json()
    if args.debug:
        print(json.dumps(body, indent=2))
    return _parse_gemini_proxy_response(body)


@dataclass
class ModelQueryArgs:
    """Arguments for querying a model."""

    model: str
    server_url: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    reasoning_effort: str = "medium"
    debug: bool = False


def collapse_text_image_inputs(messages: list[dict]) -> list[dict]:
    """
    Collapse a list of messages with sequential text into a single text input, images are still in the same relative position
    """
    new_prompt = []
    current_text_input = ""
    for message in messages:
        if message["type"] == "text":
            current_text_input += message["text"] + "\n"
        else:
            if current_text_input != "":
                new_prompt.append({"type": "text", "text": current_text_input})
                current_text_input = ""
            new_prompt.append(message)
    if current_text_input != "":
        new_prompt.append({"type": "text", "text": current_text_input})
    return new_prompt


def _completions_to_responses_convert_prompt(prompt: list[dict]) -> list[dict]:
    """Convert completions api format to responses api format.

    Args:
        prompt: The prompt in completions api format

    Returns:
        The prompt in responses api format

    Switch prompt structure to api responses api format e.g.:
    From
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    To
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the image in detail."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
    """

    for message in prompt:
        for content in message["content"]:
            if type(content) == str:
                continue
            if content.get("type") == "text":
                content["type"] = "input_text"
                content["text"] = content.pop("text")

            elif content.get("type") == "image_url":
                content["type"] = "input_image"
                content["image_url"] = content["image_url"]["url"]
    return prompt


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------


def query_model(args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]) -> str:
    """Query vLLM server for code generation.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
    Returns:
        Model response content
    """
    # Prefer the google.genai / Vertex AI backend for Gemini calls when the
    # SDK is installed and ADC is configured. Mirrors the main rats repo's
    # patch (commit 86a0cb9b in main/agents). Falls back silently to the
    # existing gemini_proxy / OpenRouter routing when Vertex is unavailable.
    from capx.llm import genai_backend
    if genai_backend.is_gemini_model(args.model) and genai_backend.is_available():
        result = genai_backend.query(
            prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            json_mode=False,
        )
        elapsed = result.get("elapsed", 0.0)
        print(
            f"Time taken to query model: {elapsed:.2f} seconds "
            f"(provider=vertex, model={result.get('model', args.model)})"
        )
        return {
            "content": str(result.get("content") or ""),
            "reasoning": None,
        }  # type: ignore[return-value]

    if should_use_gemini_proxy(args.model, getattr(args, "server_url", None)):
        return query_gemini_proxy(args, prompt)  # type: ignore[return-value]

    # Route OpenRouter models to the OpenRouter proxy server
    if is_openrouter_model(args.model):
        server_url = OPENROUTER_SERVER_URL
    else:
        server_url = args.server_url

    if args.model in GPT_MODELS:
        if "codex" in args.model:
            prompt = _completions_to_responses_convert_prompt(prompt)
            payload = {
                "model": args.model,
                "input": prompt,
            }
        else:
            payload = {
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "max_completion_tokens": args.max_tokens,  # Total completion tokens = reasoning + output tokens
                "messages": prompt,
            }
    elif is_openrouter_model(args.model):
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
        }
    elif args.model in OSS_MODELS:
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
        }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"
    start_time = time.time()

    # keep calling until it works
    response = requests.post(
        server_url, headers=headers, data=json.dumps(payload), timeout=200
    )
    retry = 1
    while response.status_code in [404, 500, 502, 503, 504]:
        sleep_time = 240 + random.uniform(-90, 90)
        print(f"Retry {retry}. Model query failed with status code {response.status_code}. Error: {response.text}. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
        response = requests.post(
            server_url, headers=headers, data=json.dumps(payload), timeout=200
        )
        retry += 1

    end_time = time.time()
    print(f"Time taken to query model: {end_time - start_time:.2f} seconds")
    if not response.ok:
        _append_call_telemetry(
            {
                "event": "network_error",
                "requested_model": args.model,
                "actual_model": "",
                "request_id": response.headers.get("x-request-id", ""),
                "status_code": response.status_code,
                "elapsed_s": round(end_time - start_time, 6),
                "usage": {},
            }
        )
    response.raise_for_status()
    body = response.json()
    _append_call_telemetry(
        {
            "event": "network_success",
            "requested_model": args.model,
            "actual_model": body.get("model", ""),
            "request_id": body.get("id") or response.headers.get("x-request-id", ""),
            "status_code": response.status_code,
            "elapsed_s": round(end_time - start_time, 6),
            "usage": body.get("usage") or {},
        }
    )
    out = {}
    if args.debug:
        print(json.dumps(body, indent=2))
    try:
        if args.model in GPT_MODELS and "codex" in args.model:
            out["content"] = body["output_text"]
        else:
            out["content"] = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response format: {body}") from exc
    if body.get("choices") is not None:
        out["reasoning"] = body.get("choices")[0].get("message").get("reasoning", None)
    else:
        out["reasoning"] = None
    return out  # type: ignore[return-value]


def query_model_streaming(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
) -> Iterable[dict]:
    """Query model with streaming enabled, yielding partial responses.

    Yields dictionaries with:
      - {"type": "content_delta", "content": "partial text"}
      - {"type": "reasoning_delta", "content": "partial reasoning"} (if supported)
      - {"type": "done", "content": "full content", "reasoning": "full reasoning or None"}

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation

    Yields:
        Partial response chunks as they arrive
    """
    if should_use_gemini_proxy(args.model, getattr(args, "server_url", None)):
        result = query_gemini_proxy(args, prompt)
        content = str(result.get("content", ""))
        reasoning = result.get("reasoning")
        if content:
            yield {"type": "content_delta", "content": content}
        yield {"type": "done", "content": content, "reasoning": reasoning}
        return

    if args.model in GPT_MODELS:
        payload = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
            "stream": True,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"

    full_content = ""
    full_reasoning = ""

    start_time = time.time()

    with requests.post(
        args.server_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=200,
        stream=True,
    ) as response:
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        is_sse = "text/event-stream" in content_type
        is_json = "application/json" in content_type

        # If it's a regular JSON response (server doesn't support streaming),
        # fall back to non-streaming behavior
        if is_json and not is_sse:
            print("Warning: Server returned JSON instead of SSE stream, falling back to non-streaming")
            body = response.json()
            try:
                full_content = body["choices"][0]["message"]["content"]
                full_reasoning = body.get("choices", [{}])[0].get("message", {}).get("reasoning")
                if full_reasoning:
                    print(f"Reasoning extracted ({len(full_reasoning)} chars)")
                else:
                    print("No reasoning returned by model")
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Unexpected response format: {body}") from exc

            yield {"type": "content_delta", "content": full_content}
            yield {
                "type": "done",
                "content": full_content,
                "reasoning": full_reasoning if full_reasoning else None,
            }
            end_time = time.time()
            print(f"Time taken to query model (streaming fallback): {end_time - start_time:.2f} seconds")
            return

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")

            # SSE format: "data: {...}" or "data: [DONE]"
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    # Handle content delta
                    content_delta = delta.get("content", "")
                    if content_delta:
                        full_content += content_delta
                        yield {"type": "content_delta", "content": content_delta}

                    # Handle reasoning delta (some APIs support this)
                    reasoning_delta = delta.get("reasoning", "")
                    if reasoning_delta:
                        full_reasoning += reasoning_delta
                        yield {"type": "reasoning_delta", "content": reasoning_delta}

                except json.JSONDecodeError:
                    continue
            else:
                # Try parsing as raw JSON (non-SSE format)
                try:
                    data = json.loads(line_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content_delta = delta.get("content", "")
                        if content_delta:
                            full_content += content_delta
                            yield {"type": "content_delta", "content": content_delta}
                except json.JSONDecodeError:
                    continue

    end_time = time.time()
    print(f"Time taken to query model (streaming): {end_time - start_time:.2f} seconds")
    if full_reasoning:
        print(f"Reasoning extracted ({len(full_reasoning)} chars)")
    else:
        print("No reasoning returned by model")

    yield {
        "type": "done",
        "content": full_content,
        "reasoning": full_reasoning if full_reasoning else None,
    }


def query_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    synthesis_model: str = "openai/gpt-5.4",
    is_multiturn = False
) -> dict[str, Any]:
    """Query 9 models (3 models x 3 temperatures) and synthesize final output."""

    def query_single(model: str, temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Multimodel Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    # Build all (model, temp) pairs and query in parallel
    tasks = [(m, t) for m, temps in ENSEMBLE_CONFIGS for t in temps]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, m, t): (m, t) for m, t in tasks}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Multimodel Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All ensemble queries failed")

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate ({r['model']}, temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    synth_args = ModelQueryArgs(
        model=synthesis_model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {synthesis_model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


def query_single_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    model: str,
    is_multiturn = False,
) -> dict[str, Any]:
    """Query the same model 9 times (with temperatures 0.1 to 0.9) and synthesize final output.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
        model: The model to use for both candidate generation and synthesis

    Returns:
        Dictionary containing synthesized content, reasoning, all responses, and text artifacts
    """

    def query_single(temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Single Model Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    # Query same model with 9 different temperatures (0.1 to 0.9)
    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, t): t for t in temperatures}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Single Model Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All single model ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All single model ensemble queries failed")

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate (temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else: # first generation has no REGEN/FINISH candidates
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    # Use the same model for synthesis
    synth_args = ModelQueryArgs(
        model=model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases (underscore-prefixed names)
# ---------------------------------------------------------------------------

_query_model = query_model
_query_model_streaming = query_model_streaming
_query_model_ensemble = query_model_ensemble
_query_single_model_ensemble = query_single_model_ensemble
