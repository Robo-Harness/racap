"""Minimal promotion rule: real native improvement wins."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Metrics


@dataclass(frozen=True)
class SelectionDecision:
    promote_capability: bool
    promote_efficiency: bool
    reason: str


def compare(champion: Metrics, candidate: Metrics) -> SelectionDecision:
    if champion.expected != candidate.expected:
        raise ValueError("paired comparison requires the same episode count")
    champion_cohort = set(champion.successes) | set(champion.failures)
    candidate_cohort = set(candidate.successes) | set(candidate.failures)
    if (
        len(champion_cohort) != champion.expected
        or len(candidate_cohort) != candidate.expected
        or champion_cohort != candidate_cohort
    ):
        raise ValueError("paired comparison requires the exact same episode keys")
    if candidate.native_success > champion.native_success:
        return SelectionDecision(
            True,
            False,
            f"native success improved {champion.native_success}->{candidate.native_success}",
        )
    if candidate.native_success < champion.native_success:
        return SelectionDecision(
            False,
            False,
            f"native success regressed {champion.native_success}->{candidate.native_success}",
        )
    champion_cost = (
        champion.mean_simulator_steps,
        champion.total_vlm_calls,
        champion.total_tool_calls,
    )
    candidate_cost = (
        candidate.mean_simulator_steps,
        candidate.total_vlm_calls,
        candidate.total_tool_calls,
    )
    if candidate_cost < champion_cost:
        return SelectionDecision(
            False,
            True,
            "native success tied and measured execution cost decreased",
        )
    return SelectionDecision(False, False, "no native-success improvement")
