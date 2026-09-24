"""Adaptive capability-DAG scheduling for a single global champion.

Stages are evidence distributions, not feature gates.  The scheduler decides
where the next simulator/coding iteration is most useful; it never decides
whether a candidate is promoted.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .curriculum import Curriculum
from .schema import CriticReport, Metrics, StageSpec


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", value.lower()) if len(token) > 2}


def _cluster(report: CriticReport) -> str:
    phase = report.failure_phase.strip().lower() or "unknown"
    layer = report.suggested_layer.strip().lower() or "unknown"
    scope = "_".join(sorted(_tokens(report.generality_scope))) or "unspecified"
    return f"{phase}|{layer}|{scope}"


@dataclass(frozen=True)
class SchedulerConfig:
    """Search preferences, all CLI-configurable and never promotion gates."""

    min_visits: int = 2
    patience: int = 3
    exploration: float = 0.45
    novelty_weight: float = 0.30
    revisit_weight: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageDecision:
    stage_id: str
    reason: str
    scores: dict[str, float]
    eligible: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_stage_state() -> dict[str, Any]:
    return {
        "status": "unseen",
        "visits": 0,
        "runtime_candidates": 0,
        "implementation_failures": 0,
        "baseline_evaluations": 0,
        "best_native": None,
        "latest_native": None,
        "expected": None,
        "recent_deltas": [],
        "no_gain_streak": 0,
        "last_visit_iteration": None,
        "last_improvement_iteration": None,
        "failure_clusters": {},
        "novel_clusters_last_attempt": 0,
        "revisit_requests": 0,
        "evaluation_seconds": 0.0,
        "baseline_commit": None,
        "baseline_metrics": None,
        "baseline_output_dir": None,
    }


class CurriculumScheduler:
    def __init__(self, curriculum: Curriculum, config: SchedulerConfig | None = None):
        self.curriculum = curriculum
        self.config = config or SchedulerConfig()
        self._by_id = {stage.id: stage for stage in curriculum.stages}
        self._depth = self._depths()

    def ensure_states(self, raw: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
        states = raw if isinstance(raw, dict) else {}
        for stage in self.curriculum.stages:
            current = states.setdefault(stage.id, {})
            defaults = new_stage_state()
            for key, value in defaults.items():
                current.setdefault(key, value)
        return states

    def observe_baseline(
        self,
        states: dict[str, dict[str, Any]],
        stage: StageSpec,
        metrics: Metrics,
        reports: Iterable[CriticReport],
        *,
        commit: str,
        iteration: int,
        output_dir: str,
        seconds: float,
    ) -> None:
        state = states[stage.id]
        state["status"] = "active"
        state["baseline_evaluations"] += 1
        state["latest_native"] = metrics.native_success
        best = state.get("best_native")
        state["best_native"] = metrics.native_success if best is None else max(best, metrics.native_success)
        state["expected"] = metrics.expected
        state["baseline_commit"] = commit
        state["baseline_metrics"] = metrics.to_dict()
        state["baseline_output_dir"] = output_dir
        state["evaluation_seconds"] += float(seconds)
        self._observe_clusters(state, reports, iteration=iteration, count_novel=False)

    def observe_attempt(
        self,
        states: dict[str, dict[str, Any]],
        stage: StageSpec,
        *,
        native_delta: int,
        reports: Iterable[CriticReport],
        iteration: int,
        seconds: float,
        valid_runtime: bool = True,
    ) -> None:
        state = states[stage.id]
        if not valid_runtime:
            # A malformed patch says nothing about capability saturation. Keep
            # it in lineage, but do not use it to push the curriculum forward.
            state["implementation_failures"] = int(
                state.get("implementation_failures", 0)
            ) + 1
            state["status"] = "implementation_blocked"
            return
        state["visits"] += 1
        state["runtime_candidates"] = int(state.get("runtime_candidates", 0)) + 1
        state["last_visit_iteration"] = iteration
        deltas = list(state.get("recent_deltas") or [])
        deltas.append(int(native_delta))
        state["recent_deltas"] = deltas[-12:]
        state["evaluation_seconds"] += float(seconds)
        if native_delta > 0:
            state["no_gain_streak"] = 0
            state["last_improvement_iteration"] = iteration
            state["status"] = "productive"
            latest = state.get("latest_native")
            if latest is not None:
                state["latest_native"] = int(latest) + native_delta
                best = state.get("best_native")
                state["best_native"] = max(int(best or 0), int(state["latest_native"]))
        else:
            state["no_gain_streak"] = int(state.get("no_gain_streak", 0)) + 1
            state["status"] = "active"
        novel = self._observe_clusters(state, reports, iteration=iteration, count_novel=True)
        state["novel_clusters_last_attempt"] = novel
        self._request_related_revisits(states, stage, reports)

    def choose(
        self,
        states: dict[str, dict[str, Any]],
        *,
        current_stage_id: str,
    ) -> StageDecision:
        states = self.ensure_states(states)
        eligible = self._eligible(states)
        if current_stage_id not in eligible:
            eligible.add(current_stage_id)
        scores = {stage_id: self._score(stage_id, states) for stage_id in sorted(eligible)}
        current = states[current_stage_id]
        recent = list(current.get("recent_deltas") or [])

        if int(current.get("visits", 0)) < self.config.min_visits:
            selected = current_stage_id
            reason = "continue gathering minimum within-stage evidence"
        elif recent and recent[-1] > 0:
            selected = current_stage_id
            reason = "continue exploiting a newly productive mechanism"
        elif (
            int(current.get("no_gain_streak", 0)) < self.config.patience
            and int(current.get("novel_clusters_last_attempt", 0)) > 0
        ):
            selected = current_stage_id
            reason = "continue testing a newly observed failure mechanism"
        else:
            # Curriculum coverage is itself information.  Once the current
            # distribution has met its within-stage evidence budget, probe an
            # eligible unseen node before revisiting an old one.  Otherwise a
            # shallow high-gap stage can repeatedly outscore every deeper node
            # and the capability DAG never gets observed.  This is a budget
            # allocation rule only; native-success promotion remains unchanged.
            unseen = {
                stage_id
                for stage_id in eligible
                if int(states[stage_id].get("baseline_evaluations", 0)) == 0
            }
            if unseen:
                selected = max(unseen, key=lambda key: (scores[key], key))
                reason = "probe an eligible unvisited capability distribution"
            else:
                selected = max(scores, key=lambda key: (scores[key], key))
                if selected == current_stage_id:
                    reason = "current stage still has the highest expected information-adjusted gain"
                else:
                    reason = "revisit the eligible stage with higher expected marginal value"

        previous = current_stage_id
        if selected != previous and states[previous].get("status") != "unseen":
            states[previous]["status"] = "dormant"
        if states[selected].get("status") == "unseen":
            states[selected]["status"] = "active"
        elif selected != previous:
            states[selected]["status"] = "revisit"
            states[selected]["revisit_requests"] = max(
                0, int(states[selected].get("revisit_requests", 0)) - 1
            )
        elif states[selected].get("status") == "implementation_blocked":
            states[selected]["status"] = "active"
        return StageDecision(selected, reason, scores, tuple(sorted(eligible)))

    def _eligible(self, states: dict[str, dict[str, Any]]) -> set[str]:
        visited = {
            stage_id
            for stage_id, state in states.items()
            if int(state.get("baseline_evaluations", 0)) > 0
        }
        eligible = set(visited)
        for stage in self.curriculum.stages:
            # Parents need only have been observed once; no accuracy gate exists.
            if all(parent in visited for parent in stage.parents):
                eligible.add(stage.id)
        return eligible or {self.curriculum.stages[0].id}

    def _score(self, stage_id: str, states: dict[str, dict[str, Any]]) -> float:
        state = states[stage_id]
        total_visits = sum(int(item.get("visits", 0)) for item in states.values())
        visits = int(state.get("visits", 0))
        expected = int(state.get("expected") or 0)
        latest = int(state.get("latest_native") or 0)
        gap = (expected - latest) / expected if expected else 1.0
        positive = sum(max(0, int(delta)) for delta in state.get("recent_deltas", ()))
        gain = positive / max(1, expected * max(1, visits))
        cluster_count = len(state.get("failure_clusters") or {})
        novelty = min(1.0, cluster_count / max(1, visits + 1))
        uncertainty = math.sqrt(math.log(total_visits + 2.0) / (visits + 1.0))
        revisit = min(1.0, int(state.get("revisit_requests", 0)) / 3.0)
        depth_prior = 1.0 / (1.0 + 0.12 * self._depth[stage_id])
        evaluations = int(state.get("baseline_evaluations", 0)) + visits
        mean_seconds = float(state.get("evaluation_seconds", 0.0)) / max(1, evaluations)
        observed_costs = [
            float(item.get("evaluation_seconds", 0.0))
            / max(1, int(item.get("baseline_evaluations", 0)) + int(item.get("visits", 0)))
            for item in states.values()
            if int(item.get("baseline_evaluations", 0)) + int(item.get("visits", 0)) > 0
        ]
        reference_cost = sorted(observed_costs)[len(observed_costs) // 2] if observed_costs else 1.0
        cost_factor = max(0.5, mean_seconds / max(reference_cost, 1e-6)) if evaluations else 1.0
        score = gap * (
            gain
            + self.config.novelty_weight * novelty
            + self.config.exploration * uncertainty
            + self.config.revisit_weight * revisit
        )
        return round(score * depth_prior / cost_factor, 8)

    def _observe_clusters(
        self,
        state: dict[str, Any],
        reports: Iterable[CriticReport],
        *,
        iteration: int,
        count_novel: bool,
    ) -> int:
        clusters = state.setdefault("failure_clusters", {})
        novel = 0
        for report in reports:
            key = _cluster(report)
            if key not in clusters:
                novel += 1
                clusters[key] = {
                    "count": 0,
                    "first_iteration": iteration,
                    "last_iteration": iteration,
                    "examples": [],
                }
            item = clusters[key]
            item["count"] = int(item.get("count", 0)) + 1
            item["last_iteration"] = iteration
            examples = list(item.get("examples") or [])
            if report.episode_key not in examples:
                examples.append(report.episode_key)
            item["examples"] = examples[-8:]
        return novel if count_novel else 0

    def _request_related_revisits(
        self,
        states: dict[str, dict[str, Any]],
        current: StageSpec,
        reports: Iterable[CriticReport],
    ) -> None:
        report_tokens = _tokens(
            " ".join(
                value
                for report in reports
                for value in (
                    report.failure_phase,
                    report.hypothesis,
                    report.generality_scope,
                    " ".join(report.applicable_tags),
                )
            )
        )
        if not report_tokens:
            return
        for stage in self.curriculum.stages:
            if stage.id == current.id or int(states[stage.id].get("baseline_evaluations", 0)) == 0:
                continue
            stage_tokens = _tokens(
                " ".join((stage.title, stage.capability, " ".join(stage.hints)))
            )
            if len(report_tokens & stage_tokens) >= 2:
                states[stage.id]["revisit_requests"] = int(
                    states[stage.id].get("revisit_requests", 0)
                ) + 1

    def _depths(self) -> dict[str, int]:
        depths: dict[str, int] = {}

        def depth(stage_id: str) -> int:
            if stage_id in depths:
                return depths[stage_id]
            parents = self._by_id[stage_id].parents
            depths[stage_id] = 0 if not parents else 1 + max(depth(parent) for parent in parents)
            return depths[stage_id]

        for stage_id in self._by_id:
            depth(stage_id)
        return depths
