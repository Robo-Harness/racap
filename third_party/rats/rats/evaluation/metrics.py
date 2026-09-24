"""Metrics tracking for RATS lifelong learning loop."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class MetricsTracker:
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record_iteration(
        self,
        *,
        task: str,
        success: bool,
        retries: int,
        skills_added: int,
        reward: float,
        skills_reused: list[str] | None = None,
        skills_failed: list[str] | None = None,
        active_skill_count: int | None = None,
    ) -> None:
        self._records.append({
            "task": task,
            "success": success,
            "retries": retries,
            "skills_added": skills_added,
            "reward": reward,
            "skills_reused": skills_reused or [],
            "skills_failed": skills_failed or [],
            "active_skill_count": active_skill_count,
        })

    def get_summary(self) -> dict[str, Any]:
        if not self._records:
            return {"total": 0}

        total = len(self._records)
        successes = sum(1 for r in self._records if r["success"])
        total_retries = sum(r["retries"] for r in self._records)
        total_skills = sum(r["skills_added"] for r in self._records)
        rewards = [r["reward"] for r in self._records if r["reward"] is not None]

        # Cumulative success rate over time
        cumulative = []
        running_success = 0
        for i, r in enumerate(self._records):
            if r["success"]:
                running_success += 1
            cumulative.append(running_success / (i + 1))

        # Skill library growth. Prefer the active learned-skill count when
        # available so deprecated skills disappear from the curve.
        if any(r.get("active_skill_count") is not None for r in self._records):
            growth = [
                int(r.get("active_skill_count") or 0)
                for r in self._records
            ]
        else:
            growth = []
            running_skills = 0
            for r in self._records:
                running_skills += r["skills_added"]
                growth.append(running_skills)

        # Per-task success rate breakdown
        per_task = self._per_task_stats()

        # Retry efficiency over time (rolling window of 5)
        retry_efficiency = self._retry_efficiency(window=5)

        # Skill reuse frequency
        skill_reuse = self._skill_reuse_stats()

        return {
            "total": total,
            "single_task_success_rate": successes / total if total > 0 else 0,
            "cumulative_success_rates": cumulative,
            "skill_library_growth": growth,
            "total_skills_added": total_skills,
            "average_retries_per_success": (
                total_retries / successes if successes > 0 else float("inf")
            ),
            "average_reward": sum(rewards) / len(rewards) if rewards else 0,
            "tasks_attempted": [r["task"] for r in self._records],
            "per_task_stats": per_task,
            "retry_efficiency": retry_efficiency,
            "skill_reuse": skill_reuse,
        }

    def _per_task_stats(self) -> dict[str, Any]:
        """Per-task success rate, attempt count, avg retries."""
        task_data: dict[str, list[dict]] = defaultdict(list)
        for r in self._records:
            task_data[r["task"]].append(r)
        stats = {}
        for task, records in task_data.items():
            n = len(records)
            s = sum(1 for r in records if r["success"])
            stats[task] = {
                "attempts": n,
                "successes": s,
                "success_rate": s / n if n > 0 else 0,
                "avg_retries": sum(r["retries"] for r in records) / n if n > 0 else 0,
                "avg_reward": sum(r["reward"] for r in records if r["reward"] is not None) / n if n > 0 else 0,
            }
        return stats

    def _retry_efficiency(self, window: int = 5) -> list[float]:
        """Rolling average of retries per iteration (lower = more efficient)."""
        if not self._records:
            return []
        retries = [r["retries"] for r in self._records]
        rolling = []
        for i in range(len(retries)):
            start = max(0, i - window + 1)
            rolling.append(sum(retries[start:i + 1]) / (i - start + 1))
        return rolling

    def _skill_reuse_stats(self) -> dict[str, Any]:
        """How often learned skills are reused across tasks."""
        reuse_counts: dict[str, int] = defaultdict(int)
        total_reuses = 0
        for r in self._records:
            for skill in r.get("skills_reused", []):
                reuse_counts[skill] += 1
                total_reuses += 1
        return {
            "total_reuses": total_reuses,
            "per_skill": dict(reuse_counts),
            "failed_per_skill": dict(
                self._skill_failure_counts(),
            ),
            "iterations_with_reuse": sum(
                1 for r in self._records if r.get("skills_reused")
            ),
        }

    def _skill_failure_counts(self) -> dict[str, int]:
        """How often skills were implicated in final failed steps."""
        counts: dict[str, int] = defaultdict(int)
        for r in self._records:
            for skill in r.get("skills_failed", []):
                counts[skill] += 1
        return counts

    def get_records(self) -> list[dict[str, Any]]:
        return list(self._records)
