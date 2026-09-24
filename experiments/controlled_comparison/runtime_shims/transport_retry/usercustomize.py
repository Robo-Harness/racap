"""Retry registered VAPI transport failures without resetting an episode.

The pinned generated-code clients retry HTTP 5xx responses but let a socket
read timeout escape.  Their outer trial runner then records a strategy failure
or, in older revisions, resets the simulator and starts the trial again.  Both
behaviours confound policy quality with transient provider transport.

This process-local ``usercustomize`` hook retries only ``POST`` calls whose URL
starts with the explicitly registered VAPI base.  It reuses the exact same
positional and keyword arguments, never touches localhost perception/control
services, and never resets or otherwise accesses the environment.  Every
transient failure and recovery is appended to a task-local JSONL ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any


RACAP_REGISTERED_TRANSPORT_RETRY_INSTALLED = False
RACAP_REGISTERED_TRANSPORT_MAX_ATTEMPTS = 0


def _request_url(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return str(args[0] if args else kwargs.get("url", ""))


def _request_body(kwargs: dict[str, Any]) -> bytes:
    if "json" in kwargs:
        try:
            return json.dumps(
                kwargs["json"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        except (TypeError, ValueError):
            return repr(kwargs["json"]).encode("utf-8", errors="replace")
    data = kwargs.get("data", b"")
    if isinstance(data, bytes):
        return data
    return str(data).encode("utf-8", errors="replace")


def _append_record(payload: dict[str, Any]) -> None:
    path = os.environ.get("RACAP_TRANSPORT_RETRY_LOG", "").strip()
    if not path:
        return
    encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _install() -> None:
    global RACAP_REGISTERED_TRANSPORT_RETRY_INSTALLED
    global RACAP_REGISTERED_TRANSPORT_MAX_ATTEMPTS

    if os.environ.get("RACAP_REGISTERED_TRANSPORT_RETRY") != "1":
        return
    registered_base = os.environ.get("RACAP_REGISTERED_VAPI_BASE", "").rstrip("/")
    if not registered_base:
        return
    maximum_attempts = int(os.environ.get("RACAP_TRANSPORT_MAX_ATTEMPTS", "3"))
    if maximum_attempts < 1:
        raise RuntimeError("RACAP_TRANSPORT_MAX_ATTEMPTS must be positive")

    import requests

    current_post = requests.post
    if getattr(current_post, "_racap_registered_transport_retry", False):
        RACAP_REGISTERED_TRANSPORT_RETRY_INSTALLED = True
        RACAP_REGISTERED_TRANSPORT_MAX_ATTEMPTS = int(
            getattr(current_post, "_racap_maximum_attempts", maximum_attempts)
        )
        return
    transient = (requests.Timeout, requests.ConnectionError)

    def registered_post(*args: Any, **kwargs: Any) -> Any:
        url = _request_url(args, kwargs)
        if not url.startswith(registered_base):
            return current_post(*args, **kwargs)
        body_sha256 = hashlib.sha256(_request_body(kwargs)).hexdigest()
        request_json = kwargs.get("json")
        model = request_json.get("model") if isinstance(request_json, dict) else None
        if model is None and "data" in kwargs:
            try:
                decoded = json.loads(kwargs["data"])
                model = decoded.get("model") if isinstance(decoded, dict) else None
            except (TypeError, ValueError, json.JSONDecodeError):
                model = None
        for attempt in range(1, maximum_attempts + 1):
            started = time.time()
            try:
                response = current_post(*args, **kwargs)
            except transient as exc:
                exhausted = attempt == maximum_attempts
                _append_record(
                    {
                        "schema_version": 1,
                        "event": "registered_vapi_transport_attempt",
                        "time_unix": time.time(),
                        "pid": os.getpid(),
                        "episode_key": os.environ.get("CAPX_EPISODE_KEY", ""),
                        "attempt": attempt,
                        "maximum_attempts": maximum_attempts,
                        "outcome": "exhausted" if exhausted else "transport_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": time.time() - started,
                        "model": model,
                        "url": url,
                        "request_body_sha256": body_sha256,
                        "prompt_or_policy_changed": False,
                        "environment_reset": False,
                    }
                )
                if exhausted:
                    raise
                continue
            if attempt > 1:
                _append_record(
                    {
                        "schema_version": 1,
                        "event": "registered_vapi_transport_attempt",
                        "time_unix": time.time(),
                        "pid": os.getpid(),
                        "episode_key": os.environ.get("CAPX_EPISODE_KEY", ""),
                        "attempt": attempt,
                        "maximum_attempts": maximum_attempts,
                        "outcome": "recovered",
                        "elapsed_seconds": time.time() - started,
                        "model": model,
                        "url": url,
                        "request_body_sha256": body_sha256,
                        "prompt_or_policy_changed": False,
                        "environment_reset": False,
                    }
                )
            return response
        raise AssertionError("unreachable registered transport retry state")

    registered_post._racap_registered_transport_retry = True  # type: ignore[attr-defined]
    registered_post._racap_maximum_attempts = maximum_attempts  # type: ignore[attr-defined]
    requests.post = registered_post
    RACAP_REGISTERED_TRANSPORT_RETRY_INSTALLED = True
    RACAP_REGISTERED_TRANSPORT_MAX_ATTEMPTS = maximum_attempts


_install()
