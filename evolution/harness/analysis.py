"""Paired rollout analysis owned by the harness, not by candidate code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Metrics


def _records(metrics: Metrics) -> dict[str, dict[str, Any]]:
    path = Path(metrics.records_path)
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {str(row.get("key")): row for row in rows}


def _episode_summary(row: dict[str, Any]) -> dict[str, Any]:
    steps = row.get("steps") or []
    return {
        "native_success": bool(row.get("native_success")),
        "agent_success": bool(row.get("agent_success")),
        "turns": row.get("turns"),
        "simulator_steps": row.get("simulator_steps"),
        "seconds": row.get("seconds"),
        "stopped": row.get("stopped") or row.get("stop_reason"),
        "error": row.get("error"),
        "actions": [step.get("action") for step in steps if isinstance(step, dict)],
        "episode_dir": (row.get("artifacts") or {}).get("episode_dir"),
    }


def compare_rollouts(champion: Metrics, candidate: Metrics) -> dict[str, Any]:
    """Return exact task-level deltas for a paired cohort.

    Native success is the only capability outcome.  Agent claims and cost are
    retained as diagnostic observations and cannot manufacture a win.
    """
    champion_wins = set(champion.successes)
    candidate_wins = set(candidate.successes)
    keys = sorted(champion_wins | set(champion.failures) | candidate_wins | set(candidate.failures))
    old_rows = _records(champion)
    new_rows = _records(candidate)
    new_wins = sorted(candidate_wins - champion_wins)
    regressions = sorted(champion_wins - candidate_wins)
    retained_wins = sorted(champion_wins & candidate_wins)
    persistent_failures = sorted(set(keys) - champion_wins - candidate_wins)
    changed = []
    for key in (*new_wins, *regressions):
        changed.append(
            {
                "key": key,
                "change": "new_win" if key in new_wins else "regression",
                "champion": _episode_summary(old_rows.get(key, {})),
                "candidate": _episode_summary(new_rows.get(key, {})),
            }
        )
    return {
        "native_delta": candidate.native_success - champion.native_success,
        "new_wins": new_wins,
        "regressions": regressions,
        "retained_wins": retained_wins,
        "persistent_failures": persistent_failures,
        "changed_episodes": changed,
        "cost_delta": {
            "mean_turns": candidate.mean_turns - champion.mean_turns,
            "mean_seconds": candidate.mean_seconds - champion.mean_seconds,
            "mean_simulator_steps": (
                candidate.mean_simulator_steps - champion.mean_simulator_steps
            ),
            "total_tool_calls": candidate.total_tool_calls - champion.total_tool_calls,
            "total_vlm_calls": candidate.total_vlm_calls - champion.total_vlm_calls,
        },
        "champion_digest": champion.digest,
        "candidate_digest": candidate.digest,
    }


def compact_delta(delta: dict[str, Any]) -> dict[str, Any]:
    """Keep state.json readable while preserving the causal signal."""
    return {
        "native_delta": delta.get("native_delta", 0),
        "new_wins": delta.get("new_wins", []),
        "regressions": delta.get("regressions", []),
        "persistent_failures": delta.get("persistent_failures", []),
        "cost_delta": delta.get("cost_delta", {}),
    }
