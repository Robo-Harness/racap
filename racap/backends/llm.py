"""Hosted-model routing, request caching, and credential-free call telemetry.

Cache keys include the complete request. Model routes and credentials are
configured through the environment; request content is not stored in telemetry.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

# Models reached through the RATs gateway rather than the relay. The gateway
# needs the prefix to pass the id through unmangled.
GATEWAY_PREFIX = "openrouter/"

_MEMORY: dict[str, str] = {}
_CALL_STATS = {
    "logical_calls": 0,
    "cache_hits": 0,
    "network_attempts": 0,
    # A retry that later succeeds is useful latency evidence, not a reason to
    # discard an episode.  These counters only advance when ``ask`` gives up
    # without returning a model answer.
    "terminal_failures": 0,
    "quota_failures": 0,
}

ONE_SHOT_CARD_ENV = "CONTROLLED_ONE_SHOT_EXPERIENCE_CARD"
MAX_ONE_SHOT_CARD_CHARS = 4_000


class LLMError(RuntimeError):
    pass


class LLMQuotaError(LLMError):
    """A provider-side balance/quota failure that retries cannot repair."""


class LLMProviderUnavailableError(LLMError):
    """The configured provider cannot route the requested model.

    This is distinct from a transient timeout: retrying the same model in a
    tight evolution loop cannot create a provider route, so callers should
    checkpoint and wait for a configuration/model change.
    """


def _append_telemetry(record: dict[str, object]) -> None:
    """Append one credential-free call record when experiment logging is enabled.

    The evaluator runs in multiple processes, so a normal ``open(..., 'a')``
    can interleave buffered writes.  One compact JSON line written through an
    ``O_APPEND`` descriptor is atomic on the local filesystem used here.  We
    intentionally retain no prompt text or image bytes in this ledger; those
    belong to the human-readable episode artifact, while this file is for
    latency, routing, token, and cost accounting.
    """
    configured = os.environ.get("RACAP_LLM_TELEMETRY_PATH", "").strip()
    if not configured:
        return
    path = Path(configured)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": time.time(),
            "pid": os.getpid(),
            "episode_key": os.environ.get("RACAP_EPISODE_KEY", ""),
            **record,
        }
        line = (json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)
    except OSError:
        # Instrumentation must never change robot behavior.
        pass


def _request_attestation(system: str, prompt: str) -> dict[str, object]:
    """Return text-free hashes proving one-trial context reached a request.

    Full request payloads may contain images and verbose task context, so the
    RACaP ledger intentionally does not retain them.  Hashes plus an exact
    substring check provide execution-level delivery evidence without storing
    prompt text.  The card truncation mirrors ``agent.experience``.
    """

    result: dict[str, object] = {
        "system_text_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "prompt_text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    configured = os.environ.get(ONE_SHOT_CARD_ENV, "").strip()
    if not configured:
        return result
    path = Path(configured).expanduser()
    try:
        card = path.read_text(encoding="utf-8").strip()[:MAX_ONE_SHOT_CARD_CHARS]
    except (OSError, UnicodeError):
        result.update(
            {
                "one_shot_card_path": path.name,
                "one_shot_card_readable": False,
                "one_shot_card_in_request": False,
            }
        )
        return result
    result.update(
        {
            "one_shot_card_path": path.name,
            "one_shot_card_readable": True,
            "one_shot_card_sha256": hashlib.sha256(card.encode()).hexdigest(),
            "one_shot_card_chars": len(card),
            "one_shot_card_in_system": bool(card and card in system),
            "one_shot_card_in_prompt": bool(card and card in prompt),
            "one_shot_card_in_request": bool(card and (card in system or card in prompt)),
        }
    )
    return result


def _abort_requested() -> str:
    configured = os.environ.get("RACAP_ABORT_SENTINEL", "").strip()
    if not configured:
        return ""
    path = Path(configured)
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("reason") or "experiment abort requested")
    except (OSError, ValueError):
        return "experiment abort requested"


def _max_network_concurrency() -> int:
    """Maximum simultaneous hosted-model requests across evaluator workers."""
    try:
        return max(1, int(os.environ.get("RACAP_VLM_MAX_CONCURRENCY", "4")))
    except ValueError:
        return 4


@contextmanager
def _network_slot():
    """Bound VLM concurrency across spawned simulator processes.

    A threading semaphore is process-local and therefore did nothing when an
    evaluation spawned 14 workers.  Advisory file locks are released by the
    kernel even when a MuJoCo worker crashes, so they provide a small robust
    cross-process pool without a coordinator service.
    """
    import fcntl

    configured = os.environ.get("RACAP_VLM_SLOT_DIR")
    directory = Path(configured) if configured else Path(tempfile.gettempdir()) / "racap-vlm-slots"
    directory.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        while True:
            for index in range(_max_network_concurrency()):
                handle = (directory / f"slot-{index}.lock").open("a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handles.append(handle)
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
                    for waiting in handles:
                        waiting.close()
                return
            for waiting in handles:
                waiting.close()
            handles.clear()
            time.sleep(0.05)
    finally:
        for handle in handles:
            handle.close()


def _is_quota_failure(status_code: int, body: str) -> bool:
    normalized = body.lower()
    markers = (
        "insufficient_user_quota",
        "insufficient quota",
        "insufficient balance",
        "balance is insufficient",
        "need pre-deduct",
    )
    return status_code in {402, 403, 429} and any(
        marker in normalized for marker in markers
    )


def _is_provider_unavailable(status_code: int, body: str) -> bool:
    normalized = body.lower()
    markers = (
        "no available channel",
        "no available route",
        "model is not supported",
        "model not supported",
        "unsupported model",
        "model_not_found",
        "暂无可用渠道",
        "不支持该模型",
    )
    return status_code in {400, 403, 404} and any(
        marker in normalized for marker in markers
    )


def cache_dir() -> Path | None:
    if os.environ.get("RACAP_LLM_CACHE", "1") == "0":
        return None
    root = os.environ.get("RACAP_LLM_CACHE_DIR")
    if not root:
        return None
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def request_key(
    model: str, system: str, prompt: str, images: list[str], max_tokens: int, temperature: float
) -> str:
    """Stable digest of everything that can change a response."""
    digest = hashlib.sha256()
    for part in (model, system, prompt, str(max_tokens), f"{temperature:.4f}"):
        digest.update(part.encode())
        digest.update(b"\x00")
    for image in images:
        digest.update(hashlib.sha256(image.encode()).digest())
    return digest.hexdigest()


def _read_cache(key: str) -> str | None:
    if key in _MEMORY:
        return _MEMORY[key]
    directory = cache_dir()
    if directory is None:
        return None
    path = directory / f"{key}.json"
    if not path.exists():
        return None
    try:
        reply = json.loads(path.read_text())["reply"]
    except (OSError, ValueError, KeyError):
        return None
    _MEMORY[key] = reply
    return reply


def cached_answer(
    system: str,
    prompt: str,
    *,
    images: list[str] | None,
    model: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str | None:
    """Read an exact cached response without issuing a network request."""
    media = list(images or [])
    return _read_cache(request_key(model, system, prompt, media, max_tokens, temperature))


def _write_cache(key: str, reply: str, model: str, prompt: str) -> None:
    _MEMORY[key] = reply
    directory = cache_dir()
    if directory is None:
        return
    payload = {
        "reply": reply,
        "model": model,
        # A prompt excerpt makes a cache entry identifiable when debugging;
        # the full prompt would balloon the directory for no benefit.
        "prompt_head": prompt[:400],
        "written_at": time.time(),
    }
    try:
        (directory / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False))
    except OSError:
        pass


def _via_gateway(
    system: str, prompt: str, images: list[str], model: str, max_tokens: int, temperature: float
) -> str:
    from rats.agents.base_agent import query_llm_text

    return query_llm_text(
        system,
        prompt,
        images=images or None,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _via_relay(
    system: str,
    prompt: str,
    images: list[str],
    model: str,
    max_tokens: int,
    temperature: float,
    *,
    timeout_s: float | None = None,
) -> str:
    import requests

    key = os.environ.get("RACAP_VAPI_KEY")
    if not key:
        raise LLMError(
            f"model {model!r} routes to the relay but RACAP_VAPI_KEY is unset; "
            "source configs/env.sh, which reads configs/local.env"
        )
    base = os.environ.get("RACAP_VAPI_BASE", "").strip().rstrip("/")
    if not base:
        raise LLMError("RACAP_VAPI_BASE is unset; configure the model endpoint in configs/local.env")
    from urllib.parse import urlsplit

    endpoint = urlsplit(base)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise LLMError("RACAP_VAPI_BASE must be an absolute HTTP(S) URL")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise LLMError("RACAP_VAPI_BASE must not contain credentials, query parameters, or fragments")
    timeout_s = timeout_s or float(os.environ.get("RACAP_VAPI_TIMEOUT_S", "180"))

    content: list[dict] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image}})

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request_attestation = _request_attestation(system, prompt)
    started = time.time()
    try:
        response = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        # A timeout is development compute even though it produces no model
        # answer. Recording it keeps the final resource comparison honest and
        # makes slow provider routes distinguishable from robot failures.
        # Persist only the exception class, never request text or credentials.
        _append_telemetry(
            {
                "event": "network_exception",
                "provider": "vapi_relay",
                "requested_model": model,
                "actual_model": "",
                "request_id": "",
                "status_code": 0,
                "elapsed_s": round(time.time() - started, 6),
                "usage": {},
                "exception_type": type(exc).__name__,
                **request_attestation,
            }
        )
        raise
    elapsed = time.time() - started
    if response.status_code != 200:
        _append_telemetry(
            {
                "event": "network_error",
                "provider": "vapi_relay",
                "requested_model": model,
                "actual_model": "",
                "request_id": response.headers.get("x-request-id", ""),
                "status_code": response.status_code,
                "elapsed_s": round(elapsed, 6),
                "usage": {},
                **request_attestation,
            }
        )
        message = f"{model} -> HTTP {response.status_code}: {response.text[:300]}"
        if _is_quota_failure(response.status_code, response.text):
            raise LLMQuotaError(message)
        if _is_provider_unavailable(response.status_code, response.text):
            raise LLMProviderUnavailableError(message)
        raise LLMError(message)
    try:
        response_body = response.json()
        _append_telemetry(
            {
                "event": "network_success",
                "provider": "vapi_relay",
                "requested_model": model,
                "actual_model": response_body.get("model", ""),
                "request_id": response_body.get("id")
                or response.headers.get("x-request-id", ""),
                "status_code": response.status_code,
                "elapsed_s": round(elapsed, 6),
                "usage": response_body.get("usage") or {},
                **request_attestation,
            }
        )
        return response_body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"{model} -> unexpected response shape: {exc}") from exc


def ask(
    system: str,
    prompt: str,
    *,
    images: list[str] | None = None,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    attempts: int = 3,
    cache: bool = True,
    timeout_s: float | None = None,
) -> str:
    """Query a model, returning the cached reply when this exact request was seen.

    Only successful replies are cached, so a transient failure does not poison
    later runs.

    Pass ``cache=False`` for calls whose value lies in varying: a coding agent
    asked twice for a proposal should be free to answer differently, and
    serving it its own previous answer would stall the search.
    """
    abort_reason = _abort_requested()
    if abort_reason:
        raise LLMQuotaError(abort_reason)

    images = list(images or [])
    _CALL_STATS["logical_calls"] += 1
    key = request_key(model, system, prompt, images, max_tokens, temperature)
    cached = _read_cache(key) if cache else None
    if cached is not None:
        _CALL_STATS["cache_hits"] += 1
        _append_telemetry(
            {
                "event": "cache_hit",
                "provider": "local_exact_cache",
                "requested_model": model,
                "actual_model": model,
                "request_id": key,
                "status_code": 200,
                "elapsed_s": 0.0,
                "usage": {},
                **_request_attestation(system, prompt),
            }
        )
        return cached

    send = _via_gateway if model.startswith(GATEWAY_PREFIX) else _via_relay
    last: Exception | None = None
    for attempt in range(attempts):
        _CALL_STATS["network_attempts"] += 1
        try:
            with _network_slot():
                if send is _via_relay:
                    reply = send(
                        system,
                        prompt,
                        images,
                        model,
                        max_tokens,
                        temperature,
                        timeout_s=timeout_s,
                    )
                else:
                    reply = send(system, prompt, images, model, max_tokens, temperature)
        except (LLMQuotaError, LLMProviderUnavailableError) as exc:
            # Retrying cannot replenish an account or create a missing model
            # route. Propagate immediately instead of manufacturing a stream
            # of rejected pseudo-iterations.
            _CALL_STATS["terminal_failures"] += 1
            if isinstance(exc, LLMQuotaError):
                _CALL_STATS["quota_failures"] += 1
            raise
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
            continue
        if reply and reply.strip():
            if cache:
                _write_cache(key, reply, model, prompt)
            return reply
        last = LLMError(f"{model} returned an empty reply")
        time.sleep(1.5 * (attempt + 1))

    _CALL_STATS["terminal_failures"] += 1
    raise LLMError(f"{model} failed after {attempts} attempts: {last}")


def reset_call_stats() -> None:
    for key in _CALL_STATS:
        _CALL_STATS[key] = 0


def call_stats() -> dict[str, int]:
    return dict(_CALL_STATS)


def cache_stats() -> dict[str, int]:
    directory = cache_dir()
    if directory is None:
        return {"entries": 0, "bytes": 0}
    entries = list(directory.glob("*.json"))
    return {
        "entries": len(entries),
        "bytes": sum(p.stat().st_size for p in entries),
    }
