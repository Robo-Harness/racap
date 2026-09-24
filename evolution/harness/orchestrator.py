"""End-to-end evolution loop: observe, diagnose, patch, run, select, retain."""

from __future__ import annotations

import json
import inspect
import re
import time
from pathlib import Path
from typing import Any

from racap.backends.llm import LLMProviderUnavailableError, LLMQuotaError

from .analysis import compact_delta, compare_rollouts
from .coder import CodingAgent
from .critic import VisualCritic
from .curriculum import Curriculum
from .events import EventStore
from .evidence import load_metrics
from .guards import check_candidate
from .memory import ExperienceMemory
from .runner import EvaluationRunner
from .scheduler import CurriculumScheduler, StageDecision
from .schema import CriticReport, ExperimentState, Metrics, Proposal, StageSpec
from .selection import compare
from .workspace import CandidateWorkspace, WorkspaceManager


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


class EvolutionOrchestrator:
    def __init__(
        self,
        racap_root: Path,
        experiment_root: Path,
        curriculum: Curriculum,
        runner: EvaluationRunner,
        coder: CodingAgent,
        critic: VisualCritic,
        *,
        critic_limit: int | None = 8,
        scheduler: CurriculumScheduler | None = None,
        implementation_repairs: int = 2,
        smoke_episodes: int = 1,
        seed_root: Path | None = None,
    ):
        self.racap_root = racap_root.resolve()
        self.root = experiment_root.resolve()
        self.curriculum = curriculum
        self.runner = runner
        self.coder = coder
        self.critic = critic
        self.critic_limit = critic_limit
        self.workspace = WorkspaceManager(self.root / "solution_git")
        self.events = EventStore(self.root / "lineage")
        self.memory = ExperienceMemory(self.root / "experience")
        self.scheduler = scheduler or CurriculumScheduler(curriculum)
        self.implementation_repairs = max(0, int(implementation_repairs))
        self.smoke_episodes = max(0, int(smoke_episodes))
        self.seed_root = (
            seed_root.resolve()
            if seed_root is not None
            else (self.racap_root / "policies" / "phase1").resolve()
        )

    @staticmethod
    def _smoke_failures(output_dir: Path) -> list[dict[str, Any]]:
        path = output_dir / "records.jsonl"
        if not path.is_file():
            return []
        failures: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            reasons: list[str] = []
            if row.get("error"):
                reasons.append(str(row["error"]))
            for step in row.get("steps") or []:
                observation = str(step.get("observation") or "")
                report = step.get("report") or {}
                exception = str(report.get("exception") or "")
                if observation in {"runtime_error", "unknown_skill", "missing_arguments"}:
                    reasons.append(f"{observation}: {exception or report}")
                elif "no_public" in observation or "no_public" in exception:
                    reasons.append(f"unconnected public actuator: {observation} {exception}")
            if reasons:
                failures.append(
                    {
                        "key": row.get("key"),
                        "reasons": reasons,
                        "episode_dir": (row.get("artifacts") or {}).get("episode_dir"),
                    }
                )
        return failures

    def _propose(
        self,
        repository: Path,
        stage: StageSpec,
        champion: Metrics,
        critics: tuple[CriticReport, ...],
        history: list[dict],
        capability_memory: dict,
    ):
        parameters = inspect.signature(self.coder.propose).parameters
        if "capability_memory" in parameters:
            return self.coder.propose(
                repository,
                stage,
                champion,
                critics,
                history,
                capability_memory=capability_memory,
            )
        return self.coder.propose(repository, stage, champion, critics, history)

    def _checkpoint_external_block(
        self,
        state: ExperimentState,
        stage: StageSpec,
        champion_commit: str,
        exc: Exception,
        *,
        phase: str,
        requested_iteration: int,
    ) -> None:
        """Persist a resumable stop without consuming an evolution iteration."""
        self.events.append(
            "external_blocked",
            stage=stage.id,
            phase=phase,
            requested_iteration=requested_iteration,
            retained_iteration=state.iteration,
            parent=champion_commit,
            error=f"{type(exc).__name__}: {exc}",
        )
        self.events.save_state(state)

    @staticmethod
    def _proposal_artifacts(proposal_dir: Path, name: str, proposal: Any) -> None:
        _write_json(
            proposal_dir / f"{name}.json",
            {
                key: value
                for key, value in proposal.to_dict().items()
                if key not in {"patch", "files", "raw_response"}
            },
        )
        if proposal.patch:
            (proposal_dir / f"{name}.patch").write_text(
                proposal.patch + "\n", encoding="utf-8"
            )
        if proposal.files:
            _write_json(proposal_dir / f"{name}.files.json", proposal.files)
        (proposal_dir / f"{name}.raw.txt").write_text(
            proposal.raw_response, encoding="utf-8"
        )

    def _materialize_candidate(
        self,
        *,
        state: ExperimentState,
        label: str,
        stage: StageSpec,
        champion_path: Path,
        champion_commit: str,
        proposal: Any,
        proposal_dir: Path,
        capability_memory: dict,
    ) -> tuple[Any, Any, str, Any, list[dict]]:
        """Apply, compile and test; let the same coder repair mechanical failures."""
        attempts: list[dict] = []
        current = proposal
        # Repairs are deltas over the immediately preceding implementation,
        # not independent rewrites from the champion.  A mechanical repair
        # often supplies only the missing module or corrected call site.  If
        # every repair worktree is recreated from ``champion_commit``, those
        # later files can compile while silently dropping the original causal
        # mechanism.  Keep the committed failed attempt as the next repair's
        # parent so the final runtime candidate is the cumulative program the
        # coder actually debugged.
        repair_base_commit = champion_commit
        for repair_index in range(self.implementation_repairs + 1):
            name = label if repair_index == 0 else f"{label}_repair{repair_index}"
            candidate = self.workspace.create_candidate(name, repair_base_commit)
            state.implementation_attempts += 1
            diagnostics: dict[str, Any]
            candidate_commit: str | None = None
            try:
                rejected_hunks = self.workspace.apply_proposal(
                    candidate,
                    patch=current.patch,
                    files=current.files,
                )
                candidate_commit = self.workspace.commit_candidate(
                    candidate, f"evolve({stage.id}): {current.title}"
                )
                substantive_changes = self.workspace.substantive_program_changes(
                    candidate, base_commit=champion_commit
                )
                if not substantive_changes:
                    raise RuntimeError(
                        "candidate has no non-empty solution/ or memory/ program "
                        "change; refusing to spend a simulator candidate slot"
                    )
                guard = check_candidate(candidate.path)
                diagnostics = {
                    "attempt": repair_index,
                    "candidate": name,
                    "base_commit": repair_base_commit,
                    "candidate_commit": candidate_commit,
                    "application": "partial" if rejected_hunks else "complete",
                    "rejected_hunks": list(rejected_hunks),
                    "substantive_program_changes": list(substantive_changes),
                    "guard": guard.__dict__,
                }
                attempts.append(diagnostics)
                _write_json(proposal_dir / f"implementation_{repair_index}.json", diagnostics)
                if guard.passed:
                    smoke_failures: list[dict[str, Any]] = []
                    if self.smoke_episodes:
                        cohort = tuple((stage.development or stage.active)[: self.smoke_episodes])
                        smoke = self.runner.evaluate(
                            candidate.path,
                            cohort,
                            f"{name}_smoke_{candidate_commit[:8]}_{time.time_ns()}",
                        )
                        smoke_failures = self._smoke_failures(smoke.output_dir)
                        diagnostics["smoke"] = {
                            "output_dir": str(smoke.output_dir),
                            "episodes": [episode.key for episode in cohort],
                            "mechanical_failures": smoke_failures,
                        }
                        _write_json(
                            proposal_dir / f"implementation_{repair_index}.json",
                            diagnostics,
                        )
                    if not smoke_failures:
                        return current, candidate, candidate_commit, guard, attempts
                    failure = "candidate failed runtime smoke: " + json.dumps(
                        smoke_failures, ensure_ascii=False
                    )
                else:
                    failure = "candidate failed compile/import/test contract"
            except (LLMQuotaError, LLMProviderUnavailableError) as exc:
                diagnostics = {
                    "attempt": repair_index,
                    "candidate": name,
                    "base_commit": repair_base_commit,
                    "candidate_commit": candidate_commit,
                    "application": "externally_blocked",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                attempts.append(diagnostics)
                _write_json(
                    proposal_dir / f"implementation_{repair_index}.json", diagnostics
                )
                # Preserve the committed worktree for audit/recovery. Provider
                # availability is not a code defect and must never enter the
                # mechanical repair loop.
                raise
            except Exception as exc:
                diagnostics = {
                    "attempt": repair_index,
                    "candidate": name,
                    "base_commit": repair_base_commit,
                    "candidate_commit": candidate_commit,
                    "application": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                attempts.append(diagnostics)
                _write_json(proposal_dir / f"implementation_{repair_index}.json", diagnostics)
                failure = diagnostics["error"]

            if repair_index >= self.implementation_repairs or not hasattr(self.coder, "repair"):
                self.workspace.discard(candidate)
                raise RuntimeError(failure)
            repair_repository = candidate.path if candidate_commit else champion_path
            try:
                current = self.coder.repair(
                    repair_repository,
                    stage,
                    current,
                    diagnostics,
                    capability_memory=capability_memory,
                )
            except Exception:
                # Preserve the exact broken source tree when an external model
                # call interrupts repair.  Resume auditing can account for the
                # attempt without reconstructing or mutating it.
                raise
            if candidate_commit:
                repair_base_commit = candidate_commit
            self.workspace.discard(candidate)
            self._proposal_artifacts(
                proposal_dir, f"repair_{repair_index + 1}", current
            )
        raise AssertionError("unreachable implementation repair loop")

    def _critique(self, output_dir: Path, label: str) -> tuple[CriticReport, ...]:
        try:
            reports = self.critic.analyze_failures(output_dir, self.critic_limit)
        except Exception as exc:
            self.events.append("critic_error", label=label, error=f"{type(exc).__name__}: {exc}")
            reports = ()
        _write_json(
            self.root / "critic" / f"{label}.json",
            [report.to_dict() for report in reports],
        )
        return reports

    def _load_critique(self, label: str) -> tuple[CriticReport, ...] | None:
        """Load an already-checkpointed visual critique without model calls.

        Baseline rollouts and their critic reports are immutable evidence.  A
        process restart must not ask the critic model to analyze the same
        videos again merely to reconstruct in-memory state.  ``None`` means
        the artifact is absent or invalid; an empty tuple is a valid cached
        result and must not be confused with a cache miss.
        """

        path = self.root / "critic" / f"{label}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                return None
            tuple_fields = {
                "visual_evidence",
                "source_frames",
                "causal_timeline",
                "alternative_hypotheses",
                "applicable_tags",
                "phase_evidence",
                "recommended_instrumentation",
            }
            reports = []
            for raw in payload:
                if not isinstance(raw, dict):
                    return None
                value = dict(raw)
                for key in tuple_fields:
                    if key in value:
                        value[key] = tuple(value[key] or ())
                reports.append(CriticReport(**value))
            return tuple(reports)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _critique_pairs(self, paired_delta: dict, label: str) -> tuple[dict, ...]:
        try:
            reports = self.critic.compare_changed(paired_delta, self.critic_limit)
        except Exception as exc:
            # AttributeError keeps lightweight test doubles backwards compatible.
            self.events.append(
                "paired_critic_error", label=label, error=f"{type(exc).__name__}: {exc}"
            )
            reports = ()
        _write_json(self.root / "critic" / f"{label}_paired.json", list(reports))
        return reports

    def _baseline(
        self,
        state: ExperimentState,
        stage: StageSpec,
        champion_path: Path,
        champion_commit: str,
    ) -> tuple[Metrics, tuple[CriticReport, ...]]:
        tag = (
            f"i{state.iteration:03d}_{stage.id}_champion_"
            f"{champion_commit[:8]}_{time.time_ns()}"
        )
        result = self.runner.evaluate(champion_path, stage.active, tag)
        metrics = result.metrics
        critics = self._critique(result.output_dir, tag)
        self.memory.ingest(
            critics,
            stage=stage,
            commit=champion_commit,
            iteration=state.iteration,
            rollout_label=tag,
        )
        self.scheduler.observe_baseline(
            state.stage_states,
            stage,
            metrics,
            critics,
            commit=champion_commit,
            iteration=state.iteration,
            output_dir=str(result.output_dir),
            seconds=result.seconds,
        )
        state.stage_id = stage.id
        state.champion_metrics = metrics.to_dict()
        self.events.append(
            "champion_evaluated",
            iteration=state.iteration,
            stage=stage.id,
            commit=champion_commit,
            metrics=metrics.to_dict(),
            output_dir=str(result.output_dir),
            command=list(result.command),
        )
        self.events.save_state(state)
        return metrics, critics

    def _reuse_baseline(
        self,
        state: ExperimentState,
        stage: StageSpec,
        champion_commit: str,
    ) -> tuple[Metrics, tuple[CriticReport, ...]] | None:
        """Reuse a complete evaluator-owned baseline when resuming a run.

        Resuming after an interrupted model request must not spend another full
        simulator pass on an unchanged commit and cohort.  ``load_metrics``
        revalidates the exact episode set and rejects incomplete/stale records;
        critic calls then hit the content-addressed cache for the same videos.
        """
        progress = state.stage_states.get(stage.id) or {}
        if progress.get("baseline_commit") != champion_commit:
            return None
        output_value = progress.get("baseline_output_dir")
        if not output_value:
            return None
        output_dir = Path(str(output_value))
        try:
            metrics = load_metrics(output_dir, stage.active)
        except Exception as exc:
            self.events.append(
                "baseline_reuse_rejected",
                iteration=state.iteration,
                stage=stage.id,
                commit=champion_commit,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None
        source_label = output_dir.name
        critics = self._load_critique(source_label)
        if critics is None:
            # Older runs may predate persisted critic artifacts. Analyze once
            # and store the report under the immutable rollout label so every
            # later resume is model-free.
            critics = self._critique(output_dir, source_label)
        label = f"i{state.iteration:03d}_{stage.id}_reused_{champion_commit[:8]}"
        state.stage_id = stage.id
        state.champion_metrics = metrics.to_dict()
        self.events.append(
            "champion_baseline_reused",
            iteration=state.iteration,
            stage=stage.id,
            commit=champion_commit,
            metrics=metrics.to_dict(),
            output_dir=str(output_dir),
        )
        self.events.save_state(state)
        return metrics, critics

    def _schedule_next(
        self,
        state: ExperimentState,
        current_stage_id: str,
    ) -> StageDecision:
        if state.mode == "auto":
            decision = self.scheduler.choose(
                state.stage_states, current_stage_id=current_stage_id
            )
        else:
            decision = StageDecision(
                current_stage_id,
                "stage is explicitly pinned for this run",
                {},
                (current_stage_id,),
            )
        state.next_stage_id = decision.stage_id
        self.events.append(
            "stage_scheduled",
            iteration=state.iteration,
            current_stage=current_stage_id,
            **decision.to_dict(),
        )
        snapshot = {
            "iteration": state.iteration,
            "global_champion": state.champion_commit,
            "decision": decision.to_dict(),
            "stage_states": state.stage_states,
        }
        _write_json(self.root / "scheduler" / "latest.json", snapshot)
        _write_json(
            self.root / "scheduler" / f"i{state.iteration:03d}.json", snapshot
        )
        return decision

    def _adopt_stage_baseline(
        self,
        state: ExperimentState,
        stage: StageSpec,
        *,
        commit: str,
        metrics: Metrics,
        output_dir: Path,
    ) -> None:
        progress = state.stage_states[stage.id]
        progress["baseline_commit"] = commit
        progress["baseline_metrics"] = metrics.to_dict()
        progress["baseline_output_dir"] = str(output_dir)
        progress["latest_native"] = metrics.native_success
        best = progress.get("best_native")
        progress["best_native"] = metrics.native_success if best is None else max(
            int(best), metrics.native_success
        )

    @staticmethod
    def _request_cross_stage_revalidation(
        state: ExperimentState, active_stage_id: str, champion_commit: str
    ) -> None:
        """A global code change makes other cohorts informative again."""
        for stage_id, progress in state.stage_states.items():
            if stage_id == active_stage_id:
                continue
            if int(progress.get("baseline_evaluations", 0)) == 0:
                continue
            if progress.get("baseline_commit") == champion_commit:
                continue
            progress["revisit_requests"] = int(progress.get("revisit_requests", 0)) + 1
            progress["status"] = "revisit"

    @staticmethod
    def _proposal_from_pending(value: dict[str, Any]) -> Proposal:
        def strings(key: str) -> tuple[str, ...]:
            raw = value.get(key) or []
            if isinstance(raw, str):
                raw = [raw]
            return tuple(str(item) for item in raw)

        return Proposal(
            title=str(value.get("title") or "Recovered executable candidate"),
            hypothesis=str(value.get("hypothesis") or "Recovered after interruption"),
            predicted_effect=str(value.get("predicted_effect") or "Measured by paired evaluation"),
            risk=str(value.get("risk") or "Measured by paired regression evaluation"),
            patch="",
            target_failure_cluster=str(value.get("target_failure_cluster") or ""),
            evidence=strings("evidence"),
            mechanism=str(value.get("mechanism") or ""),
            expected_wins=strings("expected_wins"),
            regression_risks=strings("regression_risks"),
            falsification_test=str(value.get("falsification_test") or ""),
        )

    def _discover_pending_candidate(
        self,
        state: ExperimentState,
        stage: StageSpec,
        champion_commit: str,
    ) -> None:
        """Recover a committed candidate created before pending checkpoints existed."""
        if state.pending_candidate:
            return
        proposals_root = self.root / "proposals"
        if not proposals_root.is_dir():
            return
        candidates: list[tuple[int, Path]] = []
        for path in proposals_root.glob("i[0-9][0-9][0-9]"):
            match = re.fullmatch(r"i(\d+)", path.name)
            if match and int(match.group(1)) > state.iteration:
                candidates.append((int(match.group(1)), path))
        for iteration, proposal_dir in sorted(candidates, reverse=True):
            guard_path = proposal_dir / "guards.json"
            proposal_path = proposal_dir / "proposal.json"
            if not guard_path.is_file() or not proposal_path.is_file():
                continue
            try:
                guard = json.loads(guard_path.read_text(encoding="utf-8"))
                proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not bool(guard.get("passed")):
                continue
            label = f"i{iteration:03d}"
            worktrees = sorted(
                (path for path in self.workspace.worktrees.glob(f"{label}*") if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for path in worktrees:
                try:
                    commit = self.workspace.head(path)
                except Exception:
                    continue
                implementation_attempts = []
                for attempt_path in sorted(proposal_dir.glob("implementation_*.json")):
                    try:
                        implementation_attempts.append(
                            json.loads(attempt_path.read_text(encoding="utf-8"))
                        )
                    except (OSError, json.JSONDecodeError):
                        pass
                recorded_attempts = sum(
                    1
                    for directory in proposals_root.glob("i[0-9][0-9][0-9]")
                    for _ in directory.glob("implementation_*.json")
                )
                state.implementation_attempts = max(
                    state.implementation_attempts, recorded_attempts
                )
                state.pending_candidate = {
                    "iteration": iteration,
                    "label": label,
                    "stage_id": stage.id,
                    "parent_commit": champion_commit,
                    "candidate_commit": commit,
                    "candidate_name": path.name,
                    "candidate_path": str(path),
                    "candidate_branch": f"candidate/{path.name}",
                    "proposal_dir": str(proposal_dir),
                    "rollout_tag": f"{label}_{commit[:8]}",
                    "proposal": proposal,
                    "implementation_attempts": implementation_attempts,
                    "retrieved_lesson_ids": [],
                }
                self.events.append(
                    "pending_candidate_recovered",
                    iteration=iteration,
                    stage=stage.id,
                    candidate=commit,
                    path=str(path),
                    rollout_tag=f"{label}_{commit[:8]}",
                )
                self.events.save_state(state)
                return

    def _finish_candidate(
        self,
        *,
        state: ExperimentState,
        stage: StageSpec,
        iteration: int,
        label: str,
        candidate: CandidateWorkspace,
        candidate_commit: str,
        proposal: Proposal,
        implementation_attempts: list[dict[str, Any]],
        retrieved_lesson_ids: tuple[str, ...],
        result: Any,
        champion_commit: str,
        champion_path: Path,
        champion_metrics: Metrics,
        critics: tuple[CriticReport, ...],
    ) -> tuple[str, Path, Metrics, tuple[CriticReport, ...]]:
        """Consume one complete paired evaluation and atomically checkpoint selection."""
        state.runtime_candidates += 1
        candidate_metrics = result.metrics
        candidate_critics = self._critique(
            result.output_dir, f"{label}_{candidate_commit[:8]}"
        )
        self.memory.ingest(
            candidate_critics,
            stage=stage,
            commit=candidate_commit,
            iteration=iteration,
            rollout_label=f"{label}_{candidate_commit[:8]}",
        )
        paired_delta = compare_rollouts(champion_metrics, candidate_metrics)
        proposal_dir = Path(str(state.pending_candidate["proposal_dir"]))
        _write_json(proposal_dir / "paired_delta.json", paired_delta)
        paired_critics = self._critique_pairs(
            paired_delta, f"{label}_{candidate_commit[:8]}"
        )
        self.memory.record_application(
            retrieved_lesson_ids,
            iteration=iteration,
            candidate=candidate_commit,
            native_delta=int(paired_delta["native_delta"]),
            new_wins=paired_delta["new_wins"],
            regressions=paired_delta["regressions"],
        )
        decision = compare(champion_metrics, candidate_metrics)
        self.scheduler.observe_attempt(
            state.stage_states,
            stage,
            native_delta=int(paired_delta["native_delta"]),
            reports=candidate_critics,
            iteration=iteration,
            seconds=result.seconds,
        )
        record = {
            "iteration": iteration,
            "stage": stage.id,
            "parent": champion_commit,
            "candidate": candidate_commit,
            "proposal": proposal.title,
            "status": (
                "capability_promoted"
                if decision.promote_capability
                else "efficiency_recorded"
                if decision.promote_efficiency
                else "retained_not_promoted"
            ),
            "decision": decision.reason,
            "champion_metrics": champion_metrics.to_dict(),
            "candidate_metrics": candidate_metrics.to_dict(),
            "experiment_card": {
                "target_failure_cluster": proposal.target_failure_cluster,
                "evidence": list(proposal.evidence),
                "mechanism": proposal.mechanism,
                "expected_wins": list(proposal.expected_wins),
                "regression_risks": list(proposal.regression_risks),
                "falsification_test": proposal.falsification_test,
            },
            "paired_delta": compact_delta(paired_delta),
            "candidate_diagnoses": [
                {
                    "episode_key": report.episode_key,
                    "failure_phase": report.failure_phase,
                    "hypothesis": report.hypothesis,
                    "counterfactual": report.counterfactual,
                    "confidence": report.confidence,
                }
                for report in candidate_critics
            ],
            "paired_visual_analysis": [
                {key: value for key, value in report.items() if key != "raw_response"}
                for report in paired_critics
            ],
            "implementation_attempts": implementation_attempts,
            "output_dir": str(result.output_dir),
            # Process wall time for the complete parallel candidate sweep.
            # Per-episode durations are also retained, but their sum is not
            # wall time when evaluator workers run concurrently.
            "evaluation_wall_seconds": float(result.seconds),
        }
        state.history.append(record)
        self.events.append("candidate_evaluated", **record)
        self.memory.record_candidate(
            stage=stage,
            iteration=iteration,
            parent=champion_commit,
            candidate=candidate_commit,
            title=proposal.title,
            mechanism=proposal.mechanism,
            status=record["status"],
            native_delta=int(paired_delta["native_delta"]),
            new_wins=paired_delta["new_wins"],
            regressions=paired_delta["regressions"],
            changed_files=self.workspace.changed_files(
                candidate, base_commit=champion_commit
            ),
        )
        if decision.promote_capability:
            champion_commit = self.workspace.promote(candidate, tag=f"champion-{label}")
            champion_path = candidate.path
            champion_metrics = candidate_metrics
            critics = candidate_critics
            state.champion_commit = champion_commit
            state.champion_metrics = champion_metrics.to_dict()
            self._adopt_stage_baseline(
                state,
                stage,
                commit=champion_commit,
                metrics=champion_metrics,
                output_dir=result.output_dir,
            )
            self._request_cross_stage_revalidation(state, stage.id, champion_commit)
            self.events.append(
                "capability_promoted",
                iteration=iteration,
                stage=stage.id,
                commit=champion_commit,
                reason=decision.reason,
            )
        elif decision.promote_efficiency:
            efficiency_commit = self.workspace.mark_efficiency(candidate)
            state.efficiency_commit = efficiency_commit
            state.efficiency_metrics = candidate_metrics.to_dict()
            self.events.append(
                "efficiency_recorded",
                iteration=iteration,
                commit=efficiency_commit,
                reason=decision.reason,
            )
        state.pending_candidate = None
        state.iteration = iteration
        self._schedule_next(state, stage.id)
        self.events.save_state(state)
        return champion_commit, champion_path, champion_metrics, critics

    def run(
        self,
        stage_id: str,
        iterations: int,
        *,
        start_stage_id: str = "s0_direct_transport",
        runtime_candidates: int | None = None,
    ) -> ExperimentState:
        auto = stage_id == "auto"
        if auto:
            self.curriculum.get(start_stage_id)
        else:
            self.curriculum.get(stage_id)
        initial_commit = self.workspace.initialize(self.seed_root)
        resumed = self.events.state_path.exists()
        if resumed:
            state = self.events.load_state()
            state.stage_states = self.scheduler.ensure_states(state.stage_states)
            state.mode = "auto" if auto else "fixed"
            if not state.start_stage_id:
                state.start_stage_id = start_stage_id if auto else stage_id
            active_stage_id = (
                state.next_stage_id or state.stage_id or state.start_stage_id
                if auto
                else stage_id
            )
            champion_commit = state.champion_commit
            if champion_commit == initial_commit:
                champion_path = self.workspace.repository
            else:
                resume = self.workspace.create_candidate(
                    f"resume_{time.time_ns()}", champion_commit
                )
                champion_path = resume.path
        else:
            active_stage_id = start_stage_id if auto else stage_id
            state = ExperimentState(
                self.root.name,
                active_stage_id,
                0,
                initial_commit,
                mode="auto" if auto else "fixed",
                start_stage_id=active_stage_id,
                next_stage_id=active_stage_id,
                stage_states=self.scheduler.ensure_states({}),
                scheduler_config=self.scheduler.config.to_dict(),
            )
            champion_commit = initial_commit
            champion_path = self.workspace.repository
            self.events.append(
                "experiment_started",
                experiment=state.experiment_id,
                mode=state.mode,
                start_stage=self.curriculum.get(active_stage_id).to_dict(),
                scheduler_config=state.scheduler_config,
                seed_commit=initial_commit,
                seed_root=str(self.seed_root),
            )
        state.scheduler_config = self.scheduler.config.to_dict()
        state.next_stage_id = active_stage_id
        stage = self.curriculum.get(active_stage_id)
        reused = (
            self._reuse_baseline(state, stage, champion_commit) if resumed else None
        )
        if reused is None:
            champion_metrics, critics = self._baseline(
                state, stage, champion_path, champion_commit
            )
        else:
            champion_metrics, critics = reused
        prepared_stage_id = stage.id

        # A candidate is checkpointed before its first simulator call. On a
        # quota/network interruption, resume that exact commit and rollout tag
        # before asking the coding model for another proposal.
        self._discover_pending_candidate(state, stage, champion_commit)
        if state.pending_candidate:
            pending = dict(state.pending_candidate)
            if pending.get("stage_id") != stage.id:
                raise RuntimeError(
                    "pending candidate stage does not match the requested resume stage"
                )
            candidate_path = Path(str(pending["candidate_path"]))
            candidate = CandidateWorkspace(
                str(pending["candidate_name"]),
                candidate_path,
                str(pending["candidate_branch"]),
                str(pending["parent_commit"]),
            )
            candidate_commit = str(pending["candidate_commit"])
            if not candidate_path.is_dir() or self.workspace.head(candidate_path) != candidate_commit:
                raise RuntimeError("pending candidate worktree or commit is missing")
            proposal = self._proposal_from_pending(dict(pending["proposal"]))
            pending_iteration = int(pending["iteration"])
            pending_label = str(pending["label"])
            substantive_changes = self.workspace.substantive_program_changes(
                candidate, base_commit=str(pending["parent_commit"])
            )
            if not substantive_changes:
                rollout_dir = self.root / "rollouts" / str(pending["rollout_tag"])
                rollout_dir.mkdir(parents=True, exist_ok=True)
                physical_records = len(
                    list((rollout_dir / "raw").rglob("record.json"))
                )
                invalidation = {
                    "schema_version": 1,
                    "status": "infrastructure_invalid_non_scoring",
                    "reason": (
                        "generated patch created only empty and/or test-only files; "
                        "no executable solution or runtime-memory content changed"
                    ),
                    "candidate": candidate_commit,
                    "parent": str(pending["parent_commit"]),
                    "rollout_tag": str(pending["rollout_tag"]),
                    "physical_episode_records_before_abort": physical_records,
                    "runtime_candidate_budget_consumed": False,
                }
                _write_json(
                    rollout_dir / "INFRASTRUCTURE_INVALID.json", invalidation
                )
                record = {
                    "iteration": pending_iteration,
                    "stage": stage.id,
                    "parent": str(pending["parent_commit"]),
                    "candidate": candidate_commit,
                    "proposal": proposal.title,
                    "status": "infrastructure_invalidated_no_substantive_program_change",
                    "decision": "not scored; simulator candidate budget unchanged",
                    "changed_files": list(
                        self.workspace.changed_files(
                            candidate, base_commit=str(pending["parent_commit"])
                        )
                    ),
                    "substantive_program_changes": [],
                    "physical_episode_records_before_abort": physical_records,
                    "implementation_attempts": list(
                        pending.get("implementation_attempts") or []
                    ),
                }
                state.history.append(record)
                self.scheduler.observe_attempt(
                    state.stage_states,
                    stage,
                    native_delta=0,
                    reports=(),
                    iteration=pending_iteration,
                    seconds=0.0,
                    valid_runtime=False,
                )
                self.events.append(
                    "candidate_infrastructure_invalidated", **record
                )
                state.pending_candidate = None
                state.iteration = pending_iteration
                self._schedule_next(state, stage.id)
                self.events.save_state(state)
            else:
                try:
                    result = self.runner.evaluate(
                        candidate.path, stage.active, str(pending["rollout_tag"])
                    )
                    (
                        champion_commit,
                        champion_path,
                        champion_metrics,
                        critics,
                    ) = self._finish_candidate(
                        state=state,
                        stage=stage,
                        iteration=pending_iteration,
                        label=pending_label,
                        candidate=candidate,
                        candidate_commit=candidate_commit,
                        proposal=proposal,
                        implementation_attempts=list(
                            pending.get("implementation_attempts") or []
                        ),
                        retrieved_lesson_ids=tuple(
                            pending.get("retrieved_lesson_ids") or []
                        ),
                        result=result,
                        champion_commit=champion_commit,
                        champion_path=champion_path,
                        champion_metrics=champion_metrics,
                        critics=critics,
                    )
                except (LLMQuotaError, LLMProviderUnavailableError) as exc:
                    self._checkpoint_external_block(
                        state,
                        stage,
                        champion_commit,
                        exc,
                        phase="pending_candidate_runtime_or_analysis",
                        requested_iteration=pending_iteration,
                    )
                    raise

        attempted_iterations = 0
        while True:
            if runtime_candidates is None:
                if attempted_iterations >= iterations:
                    break
            elif state.runtime_candidates >= runtime_candidates:
                break
            attempted_iterations += 1
            target_stage_id = state.next_stage_id or prepared_stage_id
            if target_stage_id != prepared_stage_id:
                stage = self.curriculum.get(target_stage_id)
                # A scheduler revisit of an unchanged global champion must not
                # spend another simulator pass on the same exact cohort.  The
                # evidence loader inside ``_reuse_baseline`` verifies episode
                # identity, completeness, runtime errors and the record digest
                # before reuse; stale or partial evidence still falls through
                # to a fresh evaluation.
                reused = self._reuse_baseline(state, stage, champion_commit)
                if reused is None:
                    champion_metrics, critics = self._baseline(
                        state, stage, champion_path, champion_commit
                    )
                else:
                    champion_metrics, critics = reused
                prepared_stage_id = stage.id
            iteration = state.iteration + 1
            label = f"i{iteration:03d}"
            # A host/network interruption can happen after a candidate
            # worktree is committed (and even after smoke) but before the
            # iteration result is checkpointed. Repairs use suffixed worktree
            # names (for example ``i003_repair1``), so checking only the exact
            # label can overwrite its proposal directory on resume. Preserve
            # every forensic artifact and advance to a fresh label instead.
            while True:
                interrupted_paths = sorted(
                    path
                    for path in self.workspace.worktrees.glob(f"{label}*")
                    if path.is_dir()
                )
                proposal_path = self.root / "proposals" / label
                rollout_paths = sorted(
                    path
                    for path in (self.root / "rollouts").glob(f"{label}_*")
                    if path.is_dir()
                )
                if not interrupted_paths and not proposal_path.exists() and not rollout_paths:
                    break
                commits = []
                for interrupted_path in interrupted_paths:
                    try:
                        commits.append(self.workspace.head(interrupted_path))
                    except Exception:
                        commits.append("unresolved")
                self.events.append(
                    "interrupted_candidate_preserved",
                    iteration=iteration,
                    stage=stage.id,
                    parent=champion_commit,
                    candidate=commits[-1] if commits else "unresolved",
                    worktree_paths=[str(path) for path in interrupted_paths],
                    proposal_path=str(proposal_path) if proposal_path.exists() else "",
                    rollout_paths=[str(path) for path in rollout_paths],
                    reason=(
                        "candidate artifacts exist without a checkpointed iteration; "
                        "preserved and excluded from promotion evidence"
                    ),
                )
                state.iteration = iteration
                self.events.save_state(state)
                iteration += 1
                label = f"i{iteration:03d}"
            retrieved_lessons = self.memory.retrieve(stage, critics)
            capability_memory = self.memory.snapshot(stage=stage, reports=critics)
            iteration_context = {
                "type": "evaluator_owned_iteration_context",
                "note": (
                    "Advisory experience. Evidence counts and helped/hurt counts are "
                    "correlational; inspect current rollouts before applying a lesson."
                ),
                "retrieved_lessons": list(retrieved_lessons),
                "capability_memory": capability_memory,
                "last_paired_delta": (
                    state.history[-1].get("paired_delta") if state.history else None
                ),
                "scheduler": {
                    "mode": state.mode,
                    "current_stage": stage.id,
                    "stage_progress": state.stage_states[stage.id],
                    "all_stage_status": {
                        key: value.get("status") for key, value in state.stage_states.items()
                    },
                },
            }
            try:
                proposal = self._propose(
                    champion_path,
                    stage,
                    champion_metrics,
                    critics,
                    [*state.history, iteration_context],
                    capability_memory,
                )
            except (LLMQuotaError, LLMProviderUnavailableError) as exc:
                self._checkpoint_external_block(
                    state,
                    stage,
                    champion_commit,
                    exc,
                    phase="proposal",
                    requested_iteration=iteration,
                )
                raise
            except Exception as exc:
                state.iteration = iteration
                record = {
                    "iteration": iteration,
                    "stage": stage.id,
                    "parent": champion_commit,
                    "status": "proposal_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                state.history.append(record)
                self.scheduler.observe_attempt(
                    state.stage_states,
                    stage,
                    native_delta=0,
                    reports=(),
                    iteration=iteration,
                    seconds=0.0,
                    valid_runtime=False,
                )
                self.events.append("candidate_rejected", **record)
                self._schedule_next(state, stage.id)
                self.events.save_state(state)
                continue
            proposal_dir = self.root / "proposals" / label
            self._proposal_artifacts(proposal_dir, "proposal", proposal)
            self.events.append(
                "proposal_generated",
                iteration=iteration,
                stage=stage.id,
                parent=champion_commit,
                title=proposal.title,
                hypothesis=proposal.hypothesis,
                predicted_effect=proposal.predicted_effect,
                risk=proposal.risk,
                target_failure_cluster=proposal.target_failure_cluster,
                evidence=list(proposal.evidence),
                mechanism=proposal.mechanism,
                falsification_test=proposal.falsification_test,
            )

            try:
                (
                    proposal,
                    candidate,
                    candidate_commit,
                    guard,
                    implementation_attempts,
                ) = self._materialize_candidate(
                    state=state,
                    label=label,
                    stage=stage,
                    champion_path=champion_path,
                    champion_commit=champion_commit,
                    proposal=proposal,
                    proposal_dir=proposal_dir,
                    capability_memory=capability_memory,
                )
            except (LLMQuotaError, LLMProviderUnavailableError) as exc:
                self._checkpoint_external_block(
                    state,
                    stage,
                    champion_commit,
                    exc,
                    phase="implementation_or_repair",
                    requested_iteration=iteration,
                )
                raise
            except Exception as exc:
                state.iteration = iteration
                record = {
                    "iteration": iteration,
                    "stage": stage.id,
                    "parent": champion_commit,
                    "status": "implementation_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "repair_budget": self.implementation_repairs,
                }
                state.history.append(record)
                self.scheduler.observe_attempt(
                    state.stage_states,
                    stage,
                    native_delta=0,
                    reports=(),
                    iteration=iteration,
                    seconds=0.0,
                    valid_runtime=False,
                )
                self.events.append("candidate_rejected", **record)
                self._schedule_next(state, stage.id)
                self.events.save_state(state)
                continue

            self.workspace.snapshot_diff(
                candidate,
                proposal_dir / "committed.diff",
                base_commit=champion_commit,
            )
            _write_json(proposal_dir / "guards.json", guard.__dict__)
            retrieved_lesson_ids = tuple(
                str(lesson["lesson_id"]) for lesson in retrieved_lessons
            )
            rollout_tag = f"{label}_{candidate_commit[:8]}"
            state.pending_candidate = {
                "iteration": iteration,
                "label": label,
                "stage_id": stage.id,
                "parent_commit": champion_commit,
                "candidate_commit": candidate_commit,
                "candidate_name": candidate.name,
                "candidate_path": str(candidate.path),
                "candidate_branch": candidate.branch,
                "proposal_dir": str(proposal_dir),
                "rollout_tag": rollout_tag,
                "proposal": {
                    key: value
                    for key, value in proposal.to_dict().items()
                    if key not in {"patch", "files", "raw_response"}
                },
                "implementation_attempts": implementation_attempts,
                "retrieved_lesson_ids": list(retrieved_lesson_ids),
            }
            self.events.append(
                "candidate_runtime_checkpointed",
                iteration=iteration,
                stage=stage.id,
                candidate=candidate_commit,
                rollout_tag=rollout_tag,
            )
            self.events.save_state(state)
            try:
                result = self.runner.evaluate(candidate.path, stage.active, rollout_tag)
                (
                    champion_commit,
                    champion_path,
                    champion_metrics,
                    critics,
                ) = self._finish_candidate(
                    state=state,
                    stage=stage,
                    iteration=iteration,
                    label=label,
                    candidate=candidate,
                    candidate_commit=candidate_commit,
                    proposal=proposal,
                    implementation_attempts=implementation_attempts,
                    retrieved_lesson_ids=retrieved_lesson_ids,
                    result=result,
                    champion_commit=champion_commit,
                    champion_path=champion_path,
                    champion_metrics=champion_metrics,
                    critics=critics,
                )
            except (LLMQuotaError, LLMProviderUnavailableError) as exc:
                self._checkpoint_external_block(
                    state,
                    stage,
                    champion_commit,
                    exc,
                    phase="candidate_runtime_or_analysis",
                    requested_iteration=iteration,
                )
                raise
        return state
