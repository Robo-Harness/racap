"""Persistent curiosity task queue for RATS.

This v1 queue is intentionally simple:

- Stores unresolved candidate tasks on disk
- Ranks them by a current_score derived from
  base_score (= novelty * learnability), surprise, and penalty
- Lets the lifelong loop insert top candidates, select one task,
  update it on failure, and rescore the whole queue after skill growth
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class TaskQueue:
    def __init__(
        self,
        storage_path: str | Path,
        *,
        max_size: int = 30,
        surprise_weight: float = 0.5,
        penalty_weight: float = 0.2,
    ) -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.surprise_weight = surprise_weight
        self.penalty_weight = penalty_weight
        self._tasks: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self._storage_path.exists():
            self._tasks = []
            return
        try:
            self._tasks = json.loads(self._storage_path.read_text())
        except Exception:
            self._tasks = []

    def save(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(self._tasks, indent=2))

    def __len__(self) -> int:
        return len(self._tasks)

    @staticmethod
    def _task_id(task_spec: dict[str, Any]) -> str:
        goal = task_spec.get("goal", []) or []
        if goal:
            stable: Any = {"goal": goal}
        else:
            stable = {"language": task_spec.get("language", "")}
        payload = json.dumps(stable, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _find_idx(self, task_id: str) -> int | None:
        for i, entry in enumerate(self._tasks):
            if entry.get("task_id") == task_id:
                return i
        return None

    def _compute_current_score(
        self,
        *,
        base_score: float,
        surprise_score: float,
        penalty: float,
    ) -> float:
        score = base_score * (1.0 + self.surprise_weight * surprise_score)
        score -= self.penalty_weight * penalty
        return round(max(0.0, score), 6)

    def _normalize_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry["task_id"] = str(entry.get("task_id") or "")
        entry["language"] = str(entry.get("language") or "")
        entry["novelty"] = float(entry.get("novelty", 0.0) or 0.0)
        entry["learnability"] = float(entry.get("learnability", 0.0) or 0.0)
        entry["base_score"] = float(entry.get("base_score", 0.0) or 0.0)
        entry["predicted_success_probability"] = float(
            entry.get("predicted_success_probability", 0.0) or 0.0
        )
        entry["prediction_reasoning"] = str(entry.get("prediction_reasoning", "") or "")
        entry["predicted_bottleneck_step"] = str(
            entry.get("predicted_bottleneck_step", "") or ""
        )
        entry["surprise_score"] = float(entry.get("surprise_score", 0.0) or 0.0)
        entry["failure_count"] = int(entry.get("failure_count", 0) or 0)
        entry["penalty"] = float(entry.get("penalty", 0.0) or 0.0)
        entry["first_proposed_iteration"] = int(
            entry.get("first_proposed_iteration", 0) or 0
        )
        entry["last_attempt_iteration"] = entry.get("last_attempt_iteration")
        entry["times_selected"] = int(entry.get("times_selected", 0) or 0)
        entry["current_score"] = self._compute_current_score(
            base_score=entry["base_score"],
            surprise_score=entry["surprise_score"],
            penalty=entry["penalty"],
        )
        return entry

    def _build_entry(
        self,
        task_spec: dict[str, Any],
        *,
        iteration: int,
    ) -> dict[str, Any]:
        novelty = float(task_spec.get("novelty", 0.0) or 0.0)
        learnability = float(task_spec.get("learnability", 0.0) or 0.0)
        base_score = float(task_spec.get("curiosity_score", novelty * learnability) or 0.0)
        entry = {
            "task_id": self._task_id(task_spec),
            "language": task_spec.get("language", task_spec.get("activity_name", "")),
            "task_spec": task_spec,
            "novelty": novelty,
            "learnability": learnability,
            "base_score": base_score,
            "predicted_success_probability": 0.0,
            "prediction_reasoning": "",
            "predicted_bottleneck_step": "",
            "surprise_score": 0.0,
            "failure_count": 0,
            "penalty": 0.0,
            "first_proposed_iteration": iteration,
            "last_attempt_iteration": None,
            "times_selected": 0,
        }
        return self._normalize_entry(entry)

    def insert_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        iteration: int,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        inserted: list[dict[str, Any]] = []
        ranked = sorted(
            candidates,
            key=lambda c: float(c.get("curiosity_score", 0.0) or 0.0),
            reverse=True,
        )[:top_k]
        for candidate in ranked:
            entry = self._build_entry(candidate, iteration=iteration)
            idx = self._find_idx(entry["task_id"])
            if idx is None:
                self._tasks.append(entry)
                inserted.append(entry)
                continue
            existing = self._tasks[idx]
            existing["task_spec"] = candidate
            existing["language"] = entry["language"]
            existing["novelty"] = entry["novelty"]
            existing["learnability"] = entry["learnability"]
            existing["base_score"] = entry["base_score"]
            self._tasks[idx] = self._normalize_entry(existing)
            inserted.append(self._tasks[idx])
        self.trim()
        self.save()
        return inserted

    def trim(self) -> None:
        self._tasks.sort(
            key=lambda t: float(t.get("current_score", 0.0) or 0.0),
            reverse=True,
        )
        if len(self._tasks) > self.max_size:
            self._tasks = self._tasks[: self.max_size]

    def select_top(self) -> dict[str, Any] | None:
        if not self._tasks:
            return None
        self.trim()
        return dict(self._tasks[0])

    def mark_selected(self, task_id: str, iteration: int) -> dict[str, Any] | None:
        idx = self._find_idx(task_id)
        if idx is None:
            return None
        entry = self._tasks[idx]
        entry["times_selected"] = int(entry.get("times_selected", 0) or 0) + 1
        entry["last_attempt_iteration"] = iteration
        self._tasks[idx] = self._normalize_entry(entry)
        self.save()
        return dict(self._tasks[idx])

    def update_prediction(self, task_id: str, prediction_card: dict[str, Any]) -> dict[str, Any] | None:
        idx = self._find_idx(task_id)
        if idx is None:
            return None
        entry = self._tasks[idx]
        prob = prediction_card.get("predicted_success_probability", 0.5)
        try:
            prob_f = max(0.0, min(1.0, float(prob)))
        except Exception:
            prob_f = 0.5
        entry["predicted_success_probability"] = prob_f
        entry["prediction_reasoning"] = str(
            prediction_card.get("prediction_reasoning", "") or ""
        )
        entry["predicted_bottleneck_step"] = str(
            prediction_card.get("predicted_bottleneck_step", "") or ""
        )
        self._tasks[idx] = self._normalize_entry(entry)
        self.save()
        return dict(self._tasks[idx])

    def record_failure(
        self,
        task_id: str,
        *,
        predicted_success_probability: float,
        iteration: int,
        penalty_increment: float = 1.0,
    ) -> dict[str, Any] | None:
        idx = self._find_idx(task_id)
        if idx is None:
            return None
        entry = self._tasks[idx]
        entry["failure_count"] = int(entry.get("failure_count", 0) or 0) + 1
        entry["penalty"] = float(entry.get("penalty", 0.0) or 0.0) + penalty_increment
        entry["surprise_score"] = max(
            0.0,
            min(1.0, float(predicted_success_probability or 0.0)),
        )
        entry["last_attempt_iteration"] = iteration
        self._tasks[idx] = self._normalize_entry(entry)
        self.trim()
        self.save()
        return dict(self._tasks[idx])

    def remove(self, task_id: str) -> dict[str, Any] | None:
        idx = self._find_idx(task_id)
        if idx is None:
            return None
        removed = self._tasks.pop(idx)
        self.save()
        return removed

    def rescore_all(
        self,
        *,
        task_proposer: Any,
        skill_context: dict[str, Any],
        reset_penalties: bool = True,
    ) -> None:
        if not self._tasks:
            return
        candidates = [dict(entry.get("task_spec") or {}) for entry in self._tasks]
        rescored = task_proposer.compute_curiosity_scores(candidates, skill_context)
        scored_by_id = {
            self._task_id(task): task
            for task in rescored
        }
        new_entries: list[dict[str, Any]] = []
        for entry in self._tasks:
            task_id = entry.get("task_id", "")
            scored = scored_by_id.get(task_id)
            if scored is not None:
                entry["task_spec"] = scored
                entry["novelty"] = float(scored.get("novelty", 0.0) or 0.0)
                entry["learnability"] = float(scored.get("learnability", 0.0) or 0.0)
                entry["base_score"] = float(scored.get("curiosity_score", 0.0) or 0.0)
                entry["language"] = scored.get(
                    "language", scored.get("activity_name", entry.get("language", ""))
                )
            if reset_penalties:
                entry["penalty"] = 0.0
            new_entries.append(self._normalize_entry(entry))
        self._tasks = new_entries
        self.trim()
        self.save()

    def top_snapshot(self, limit: int = 5) -> list[dict[str, Any]]:
        self.trim()
        out: list[dict[str, Any]] = []
        for entry in self._tasks[:limit]:
            out.append({
                "task_id": entry.get("task_id"),
                "language": entry.get("language"),
                "novelty": entry.get("novelty"),
                "learnability": entry.get("learnability"),
                "base_score": entry.get("base_score"),
                "surprise_score": entry.get("surprise_score"),
                "failure_count": entry.get("failure_count"),
                "penalty": entry.get("penalty"),
                "current_score": entry.get("current_score"),
                "first_proposed_iteration": entry.get("first_proposed_iteration"),
                "last_attempt_iteration": entry.get("last_attempt_iteration"),
                "times_selected": entry.get("times_selected"),
            })
        return out
