"""Short-lived memoization for LLM-using primitives during one attempt.

WHY this exists
---------------
The lifelong loop runs each policy TWICE per attempt:

  Step 4b: Policy Runtime Self-Check  -> executor.execute(code, env, ...)
  (env reset)
  Step 5 : Official Execution         -> executor.execute(code, env, ...)

The self-check catches Python/API runtime crashes before the official
attempt is recorded. But any LLM-using primitive the policy invokes
(e.g. ``verify_object_identity`` on libero_reduced_skill_library) gets
called twice with bit-identical inputs — the env was reset to the same
state, so the wrist crop and the prompt are the same. The second VLM
round-trip is pure waste (observed pairs ``(0005,0006)``, ``(0011,0012)``,
… in Jiaxin's ``libero_nonpriv_explore_io_v2/agent_io``: identical
text-prompt hash, identical PNG hash, identical response modulo
whitespace).

How it works
------------
A module-level dict keyed on ``(stable_image_hash, prompt_signature)``
holds the first result. While the cache scope is active, primitives
look up the key BEFORE making the LLM call; on miss they compute and
store; on hit they return the stored value. When the scope ends, the
cache is cleared.

Scope: enable AROUND a self-check + execute pair. Clear at the end of
the attempt so the next attempt starts fresh — env state has changed
(failure_memory grew, plan may have refined), so even if the wrist
crop happened to match, the policy code might be different.

Threading
---------
Each subagent runs as a separate ``subprocess``, so the cache is
naturally per-process. No locking needed.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger("rats.primitive_cache")

_enabled: bool = False
_cache: dict[tuple, Any] = {}
_hits: int = 0
_misses: int = 0


def is_enabled() -> bool:
    return _enabled


def reset_stats() -> None:
    global _hits, _misses
    _hits = 0
    _misses = 0


def stats() -> dict[str, int]:
    return {"hits": _hits, "misses": _misses, "entries": len(_cache)}


@contextmanager
def scope() -> Iterator[None]:
    """Enable the cache for the duration of the `with` block.

    Nested scopes don't reset state — the outermost scope owns lifecycle.
    """
    global _enabled
    was_enabled = _enabled
    if not was_enabled:
        _enabled = True
        reset_stats()
        _cache.clear()
    try:
        yield
    finally:
        if not was_enabled:
            s = stats()
            if s["hits"] or s["misses"]:
                logger.info(
                    "  primitive cache: %d hits, %d misses, %d entries",
                    s["hits"], s["misses"], s["entries"],
                )
            _enabled = False
            _cache.clear()
            reset_stats()


def _hash_image(arr: Any) -> str:
    """Stable hash for an image-like array, or empty string if not array-like."""
    try:
        if isinstance(arr, np.ndarray):
            return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()
    except Exception:
        pass
    return ""


def make_key(name: str, *parts: Any) -> tuple:
    """Build a stable key from a function name + arbitrary args.

    `np.ndarray` parts are hashed by content (not identity). Everything
    else is normalized to a hashable shape. Empty image hashes degrade
    to a cache-miss-always state, which is the safe default.
    """
    out: list = [name]
    for p in parts:
        if isinstance(p, np.ndarray):
            h = _hash_image(p)
            if not h:
                # Fall back to a unique sentinel so unhashable-image
                # calls never collide and never hit.
                return (name, id(p))
            out.append(("img", h))
        elif isinstance(p, (tuple, list)):
            out.append(tuple(p))
        elif isinstance(p, dict):
            out.append(tuple(sorted(p.items())))
        else:
            out.append(p)
    return tuple(out)


def lookup(key: tuple) -> tuple[bool, Any]:
    """Return (hit, value). On miss returns (False, None)."""
    global _hits, _misses
    if not _enabled:
        return False, None
    if key in _cache:
        _hits += 1
        return True, _cache[key]
    _misses += 1
    return False, None


def store(key: tuple, value: Any) -> None:
    """Store a value under the key. No-op when disabled."""
    if not _enabled:
        return
    _cache[key] = value
