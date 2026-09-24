"""Retry bank for surprise-driven task reprioritization.

A small, persistent store of diagnosable failures. Items are added when:
  - the task FAILED, AND
  - the planner predicted success_probability >= the configured threshold
    (high-surprise miss — the agent "expected to win"), AND
  - the failure_reason is in the actionable set (NOT a perception server
    outage, env_creation crash, code bug, etc.)

Items decay over time via a per-item TTL. Each iteration the loop may:
  - sample up to K items (highest surprise_score, then most recent),
  - decrement TTL for everyone,
  - drop expired items.

Sampled items become "retry-derived" candidates via a separate proposer
prompt that asks the LLM for a SIMPLIFIED nearby variant — not the exact
original task. The retry candidate inherits a small retry_bonus_score
that the curiosity scorer reads.

Storage: JSON list at ``<storage_path>``. Idempotent re-load on resume.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("rats.retry_bank")


# Failure reasons we treat as actionable — high-surprise miss here means
# there's a missing affordance, a wrong precondition, or a subskill that
# needs more practice. Worth keeping for retry.
ACTIONABLE_FAILURES = {
    "grasp_failure",
    "placement_failure",
    "wrong_sequencing",
    "missing_precondition",
    "wrong_affordance",
    "articulation_failure",
    "handle_localization_failure",
    "verifier_unsatisfied",
    "partial_success",
    "max_retries_exceeded",
}

# Failure reasons that are noise — keeping them in retry bank just makes
# the agent loop on infrastructure issues. Filtered out at insertion time.
NOISY_FAILURES = {
    "env_creation_failed",
    "invalid_scene",
    "sim_artifact",
    "perception_server_down",
    "timeout_no_movement",
    "syntax_error",
    "code_bug",
    "policy_writer_failed",
    "quality_check_failed",
    "unknown",
}


def _retry_id(task_spec: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "language": task_spec.get("language", ""),
            "objects": sorted(str(o) for o in (task_spec.get("objects") or [])),
            "fixtures": sorted(str(f) for f in (task_spec.get("fixtures") or [])),
            "goal": task_spec.get("goal") or [],
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def classify_failure_reason(failure_reason: str) -> tuple[str, bool]:
    """Return (canonical_category, is_diagnosable).

    Splits prose failure_reason strings (which often have the form
    ``"category: diagnosis text"``) on the first colon. The category is
    lowercased and matched against the actionable / noisy sets above.

    Anything not in either set falls through as actionable=False — when in
    doubt, treat as noise rather than spam the retry bank.
    """
    if not failure_reason:
        return ("unknown", False)
    head = failure_reason.split(":", 1)[0].strip().lower()
    if head in NOISY_FAILURES:
        return (head, False)
    if head in ACTIONABLE_FAILURES:
        return (head, True)
    # Heuristic substring matches for common phrasings the failure
    # diagnoser emits without our canonical category prefix.
    lowered = failure_reason.lower()
    if any(k in lowered for k in (
        "grasp", "pick up", "lift", "hold",
    )):
        return ("grasp_failure", True)
    if any(k in lowered for k in (
        "placement", "place ", "put down", "drop ",
    )):
        return ("placement_failure", True)
    if any(k in lowered for k in (
        "sequence", "order", "before", "after",
    )):
        return ("wrong_sequencing", True)
    if any(k in lowered for k in (
        "open ", "close ", "handle", "door",
    )):
        return ("articulation_failure", True)
    return (head or "unknown", False)


class RetryBank:
    """Persistent store of diagnosable failures awaiting retry.

    Not a queue — items are NOT consumed by sampling. They live until
    TTL expires, are marked resolved, or fall out of ``max_size``.
    """

    def __init__(
        self,
        storage_path: str | Path | None,
        *,
        max_size: int = 8,
        default_ttl: int = 3,
        min_predicted_success: float = 0.5,
    ) -> None:
        self._storage_path = Path(storage_path) if storage_path else None
        self.max_size = int(max_size)
        self.default_ttl = int(default_ttl)
        self.min_predicted_success = float(min_predicted_success)
        self._items: list[dict[str, Any]] = []
        self._load()

    def __len__(self) -> int:
        return len(self._items)

    def _load(self) -> None:
        if not self._storage_path or not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text())
            if isinstance(data, list):
                self._items = [d for d in data if isinstance(d, dict)]
        except Exception as e:
            logger.warning(f"RetryBank: could not load {self._storage_path}: {e}")
            self._items = []

    def save(self) -> None:
        if not self._storage_path:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(self._items, indent=2))

    def items(self) -> list[dict[str, Any]]:
        return [dict(it) for it in self._items]

    def should_add(
        self,
        *,
        success: bool,
        failure_reason: str,
        predicted_success_probability: float | None,
    ) -> tuple[bool, str]:
        """Predicate: should this failure go in the retry bank?

        Returns (decision, reason_string). The reason string is recorded
        in iteration logs so it's clear why an outcome did or didn't end
        up in the bank.
        """
        if success:
            return (False, "task succeeded")
        if predicted_success_probability is None:
            return (False, "no prediction_card.predicted_success_probability")
        if float(predicted_success_probability) < self.min_predicted_success:
            return (
                False,
                f"predicted_p ({predicted_success_probability:.2f}) below "
                f"threshold ({self.min_predicted_success:.2f})",
            )
        category, diagnosable = classify_failure_reason(failure_reason or "")
        if not diagnosable:
            return (False, f"non-diagnosable failure category: {category}")
        return (True, f"diagnosable={category}, p_hat={predicted_success_probability:.2f}")

    def add(
        self,
        *,
        task_spec: dict[str, Any],
        failure_reason: str,
        diagnosis_summary: str,
        predicted_success_probability: float,
        iteration: int,
        ttl: int | None = None,
    ) -> dict[str, Any]:
        """Insert (or refresh) a failure record.

        Surprise score is computed symmetrically: |1[success] - p_hat|.
        Since we only reach ``add`` on failure, this collapses to p_hat
        — but the symbolic form is logged so the field stays meaningful
        when we eventually start tracking success-side surprises too.
        """
        rid = _retry_id(task_spec)
        category, diagnosable = classify_failure_reason(failure_reason or "")
        surprise = max(0.0, min(1.0, float(predicted_success_probability or 0.0)))
        ttl_val = self.default_ttl if ttl is None else int(ttl)

        existing = next((it for it in self._items if it.get("retry_id") == rid), None)
        if existing is not None:
            existing["attempts"] = int(existing.get("attempts", 0)) + 1
            existing["last_seen_iter"] = int(iteration)
            existing["ttl"] = ttl_val
            existing["surprise_score"] = max(
                float(existing.get("surprise_score", 0.0) or 0.0),
                surprise,
            )
            existing["failure_reason"] = failure_reason
            existing["failure_category"] = category
            existing["diagnosable"] = diagnosable
            existing["diagnosis_summary"] = diagnosis_summary or existing.get("diagnosis_summary", "")
            self._normalize_and_save()
            return dict(existing)

        item = {
            "retry_id": rid,
            "task_spec": task_spec,
            "language": task_spec.get("language", task_spec.get("activity_name", "")),
            "objects": list(task_spec.get("objects") or []),
            "fixtures": list(task_spec.get("fixtures") or []),
            "goal": task_spec.get("goal") or [],
            "failure_reason": failure_reason,
            "failure_category": category,
            "diagnosable": diagnosable,
            "diagnosis_summary": diagnosis_summary or "",
            "predicted_success_probability": float(predicted_success_probability or 0.0),
            "surprise_score": surprise,
            "ttl": ttl_val,
            "max_ttl": ttl_val,
            "attempts": 1,
            "created_iter": int(iteration),
            "last_seen_iter": int(iteration),
            "last_sampled_iter": None,
            "created_at": time.time(),
        }
        self._items.append(item)
        self._normalize_and_save()
        return dict(item)

    def sample(self, k: int, *, iteration: int) -> list[dict[str, Any]]:
        """Take up to ``k`` items by (surprise * ttl_decay) score.

        Items are NOT removed by sampling. Each sample bumps the item's
        ``last_sampled_iter`` so log readers can spot retry-bank items
        that keep getting picked but never get attempted.
        """
        if k <= 0 or not self._items:
            return []

        def rank_key(it: dict[str, Any]) -> float:
            ttl = max(0, int(it.get("ttl", 0)))
            mttl = max(1, int(it.get("max_ttl", self.default_ttl)))
            surprise = float(it.get("surprise_score", 0.0) or 0.0)
            return surprise * (ttl / mttl)

        ranked = sorted(self._items, key=rank_key, reverse=True)
        out: list[dict[str, Any]] = []
        for it in ranked:
            if len(out) >= k:
                break
            if int(it.get("ttl", 0)) <= 0:
                continue
            it["last_sampled_iter"] = int(iteration)
            out.append(dict(it))
        if out:
            self.save()
        return out

    def decay_ttl(self) -> None:
        """Decrement TTL on every item by 1; drop expired."""
        if not self._items:
            return
        for it in self._items:
            it["ttl"] = max(0, int(it.get("ttl", 0)) - 1)
        before = len(self._items)
        self._items = [it for it in self._items if int(it.get("ttl", 0)) > 0]
        if len(self._items) < before:
            logger.info(
                "RetryBank: dropped %d expired item(s); %d remain",
                before - len(self._items), len(self._items),
            )
        self.save()

    def mark_resolved(self, retry_id: str | None) -> bool:
        if not retry_id:
            return False
        idx = next(
            (i for i, it in enumerate(self._items) if it.get("retry_id") == retry_id),
            None,
        )
        if idx is None:
            return False
        removed = self._items.pop(idx)
        logger.info(
            "RetryBank: resolved %s (was %s)",
            retry_id, removed.get("failure_category", "?"),
        )
        self.save()
        return True

    def _normalize_and_save(self) -> None:
        # Trim by surprise score so high-value items survive eviction.
        self._items.sort(
            key=lambda it: float(it.get("surprise_score", 0.0) or 0.0),
            reverse=True,
        )
        if len(self._items) > self.max_size:
            dropped = self._items[self.max_size:]
            self._items = self._items[: self.max_size]
            if dropped:
                logger.info(
                    "RetryBank: trimmed %d item(s) at max_size=%d",
                    len(dropped), self.max_size,
                )
        self.save()

    def snapshot(self, limit: int = 8) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for it in self._items[:limit]:
            out.append({
                "retry_id": it.get("retry_id"),
                "language": it.get("language"),
                "failure_category": it.get("failure_category"),
                "surprise_score": it.get("surprise_score"),
                "ttl": it.get("ttl"),
                "attempts": it.get("attempts"),
                "created_iter": it.get("created_iter"),
                "last_sampled_iter": it.get("last_sampled_iter"),
            })
        return out
