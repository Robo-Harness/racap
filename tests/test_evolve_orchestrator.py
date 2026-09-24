from pathlib import Path

import pytest

from evolution.harness.curriculum import default_curriculum
from evolution.harness.orchestrator import EvolutionOrchestrator
from evolution.harness.scheduler import CurriculumScheduler, SchedulerConfig
from evolution.harness.schema import EvaluationResult, Metrics, Proposal
from racap.backends.llm import LLMProviderUnavailableError, LLMQuotaError


def _metrics(success, path):
    return Metrics(
        expected=13,
        native_success=success,
        agent_claimed=success,
        mean_turns=1,
        mean_seconds=1,
        mean_simulator_steps=100,
        total_tool_calls=13,
        total_vlm_calls=0,
        successes=tuple(f"episode/{index}" for index in range(success)),
        failures=tuple(f"episode/{index}" for index in range(success, 13)),
        records_path=str(path / "records.jsonl"),
        digest=str(success),
    )


class FakeRunner:
    def __init__(self, root):
        self.root = root

    def evaluate(self, solution_root, episodes, tag):
        output = self.root / "rollouts" / tag
        output.mkdir(parents=True)
        (output / "records.jsonl").write_text("")
        learned = (
            "Learned from paired evidence"
            in (solution_root / "memory" / "experience.md").read_text()
        )
        return EvaluationResult(
            output,
            _metrics(2 if learned else 1, output),
            ("fake-eval",),
            output / "stdout.log",
            0.01,
        )


class FakeCoder:
    def propose(self, repository, stage, champion, critics, history):
        memory = (repository / "memory" / "experience.md").read_text()
        return Proposal(
            "learn",
            "paired evidence",
            "+1",
            "none",
            "",
            "fake",
            files={"memory/experience.md": memory + "\nLearned from paired evidence.\n"},
        )


class FakeCritic:
    def analyze_failures(self, output_dir, limit):
        return ()


class CountingCritic(FakeCritic):
    def __init__(self):
        self.calls = 0
        self.output_dirs = []

    def analyze_failures(self, output_dir, limit):
        self.calls += 1
        self.output_dirs.append(Path(output_dir).name)
        return ()


class NoGainRunner(FakeRunner):
    def evaluate(self, solution_root, episodes, tag):
        output = self.root / "rollouts" / tag
        output.mkdir(parents=True)
        (output / "records.jsonl").write_text("")
        return EvaluationResult(
            output,
            _metrics(1, output),
            ("fake-eval",),
            output / "stdout.log",
            0.01,
        )


class SequencedCoder:
    def __init__(self):
        self.attempt = 0

    def propose(self, repository, stage, champion, critics, history):
        self.attempt += 1
        # Use runtime memory rather than a test-only mutation so this fixture
        # remains a real candidate under the no-op fail-closed contract.
        name = f"memory/attempt_{self.attempt}.txt"
        patch = "\n".join(
            (
                f"diff --git a/{name} b/{name}",
                "new file mode 100644",
                "--- /dev/null",
                f"+++ b/{name}",
                "@@ -0,0 +1 @@",
                f"+attempt {self.attempt}",
                "",
            )
        )
        return Proposal("probe", "paired evidence", "measure", "none", patch, "fake")


class RepairingCoder:
    def propose(self, repository, stage, champion, critics, history, capability_memory=None):
        return Proposal(
            "runnable mechanism",
            "mechanical repair should preserve the idea",
            "reach runtime",
            "none",
            "",
            "initial",
            files={
                "solution/generated.py": "this is invalid python !!!\n",
                "memory/cumulative_repair_marker.txt": "initial mechanism\n",
            },
        )

    def repair(self, repository, stage, proposal, diagnostics, capability_memory=None):
        assert diagnostics["guard"]["passed"] is False
        assert (repository / "solution" / "generated.py").read_text() == (
            "this is invalid python !!!\n"
        )
        assert (
            repository / "memory" / "cumulative_repair_marker.txt"
        ).read_text() == "initial mechanism\n"
        return Proposal(
            proposal.title,
            proposal.hypothesis,
            proposal.predicted_effect,
            proposal.risk,
            "",
            "repair",
            files={"solution/generated.py": "MECHANISM = 'runnable'\n"},
        )


class EmptyProgramCoder:
    def propose(self, repository, stage, champion, critics, history):
        return Proposal(
            "empty malformed candidate",
            "patch parser lost every intended hunk",
            "must not reach runtime",
            "none",
            "",
            "empty",
            files={
                "solution/generated.py": "",
                "tests/test_generated.py": "",
            },
        )


class QuotaBlockedCoder:
    def propose(self, repository, stage, champion, critics, history):
        raise LLMQuotaError("HTTP 403: insufficient_user_quota")


class RouteBlockedCoder:
    def propose(self, repository, stage, champion, critics, history):
        raise LLMProviderUnavailableError("HTTP 403: no available channel")


class CandidateQuotaRunner:
    def __init__(self, root, *, block_candidate):
        self.root = root
        self.block_candidate = block_candidate

    def evaluate(self, solution_root, episodes, tag):
        output = self.root / "rollouts" / tag
        if "champion" not in tag and self.block_candidate:
            raise LLMQuotaError("runtime quota exhausted")
        output.mkdir(parents=True, exist_ok=True)
        (output / "records.jsonl").write_text("")
        learned = "Learned from paired evidence" in (
            solution_root / "memory" / "experience.md"
        ).read_text()
        return EvaluationResult(
            output,
            _metrics(2 if learned else 1, output),
            ("fake-eval", tag),
            output / "stdout.log",
            0.01,
        )


class MustNotProposeCoder:
    def propose(self, *args, **kwargs):
        raise AssertionError("resume must evaluate the checkpointed candidate first")


def test_quota_block_checkpoints_without_consuming_iteration(tmp_path):
    racap_root = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "evolution"
    orchestrator = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        NoGainRunner(experiment),
        QuotaBlockedCoder(),
        FakeCritic(),
    )

    with pytest.raises(LLMQuotaError):
        orchestrator.run("s0_direct_transport", 1)

    state = orchestrator.events.load_state()
    assert state.iteration == 0
    assert state.history == []
    events = (experiment / "lineage" / "events.jsonl").read_text()
    assert '"event": "external_blocked"' in events
    assert '"requested_iteration": 1' in events


def test_missing_model_route_checkpoints_without_spinning(tmp_path):
    racap_root = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "evolution"
    orchestrator = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        NoGainRunner(experiment),
        RouteBlockedCoder(),
        FakeCritic(),
    )

    with pytest.raises(LLMProviderUnavailableError):
        orchestrator.run("s0_direct_transport", 1, runtime_candidates=2)

    state = orchestrator.events.load_state()
    assert state.iteration == 0
    assert state.runtime_candidates == 0
    assert state.history == []
    events = (experiment / "lineage" / "events.jsonl").read_text()
    assert events.count('"event": "external_blocked"') == 1


def test_runtime_quota_resumes_same_commit_without_extra_candidate_budget(tmp_path):
    racap_root = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "evolution"
    first = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        CandidateQuotaRunner(experiment, block_candidate=True),
        FakeCoder(),
        FakeCritic(),
        smoke_episodes=0,
    )

    with pytest.raises(LLMQuotaError):
        first.run("s0_direct_transport", 0, runtime_candidates=1)

    checkpoint = first.events.load_state()
    assert checkpoint.runtime_candidates == 0
    assert checkpoint.iteration == 0
    assert checkpoint.pending_candidate is not None
    pending_commit = checkpoint.pending_candidate["candidate_commit"]

    resumed = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        CandidateQuotaRunner(experiment, block_candidate=False),
        MustNotProposeCoder(),
        FakeCritic(),
        smoke_episodes=0,
    ).run("s0_direct_transport", 0, runtime_candidates=1)

    assert resumed.runtime_candidates == 1
    assert resumed.iteration == 1
    assert resumed.pending_candidate is None
    assert resumed.history[-1]["candidate"] == pending_commit
    assert resumed.history[-1]["status"] == "capability_promoted"


def test_persisted_critique_loads_without_another_model_call(tmp_path):
    racap_root = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "evolution"
    first_critic = CountingCritic()
    first = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        NoGainRunner(experiment),
        FakeCoder(),
        first_critic,
    )
    assert first._critique(experiment / "rollout", "immutable_baseline") == ()
    assert first_critic.calls == 1

    resumed_critic = CountingCritic()
    resumed = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        NoGainRunner(experiment),
        FakeCoder(),
        resumed_critic,
    )
    assert resumed._load_critique("immutable_baseline") == ()
    assert resumed_critic.calls == 0


def test_end_to_end_iteration_promotes_one_native_success(tmp_path):
    racap_root = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "evolution"
    orchestrator = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        FakeRunner(experiment),
        FakeCoder(),
        FakeCritic(),
    )
    state = orchestrator.run("s0_direct_transport", 1)
    assert state.champion_metrics["native_success"] == 2
    assert state.history[-1]["status"] == "capability_promoted"
    assert (experiment / "lineage" / "events.jsonl").is_file()


def test_empty_program_candidate_does_not_consume_runtime_budget(tmp_path):
    racap_root = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "evolution"
    state = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        NoGainRunner(experiment),
        EmptyProgramCoder(),
        FakeCritic(),
        implementation_repairs=0,
    ).run("s0_direct_transport", 1)

    assert state.runtime_candidates == 0
    assert state.history[-1]["status"] == "implementation_failed"
    assert "no non-empty solution/ or memory/ program change" in state.history[-1][
        "error"
    ]


def test_switching_stage_reuses_the_global_champion_in_same_experiment(tmp_path):
    racap_root = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "evolution"
    first = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        FakeRunner(experiment),
        FakeCoder(),
        FakeCritic(),
    ).run("s0_direct_transport", 1)
    champion = first.champion_commit

    second = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        FakeRunner(experiment),
        FakeCoder(),
        FakeCritic(),
    ).run("s1_transport_react", 0)
    assert second.champion_commit == champion
    assert second.stage_id == "s1_transport_react"
    assert second.champion_metrics["native_success"] == 2
    assert second.stage_states["s0_direct_transport"]["baseline_evaluations"] >= 1
    assert second.stage_states["s1_transport_react"]["baseline_evaluations"] == 1


def test_auto_scheduler_switches_cohort_without_restarting_lineage(tmp_path):
    racap_root = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "evolution"
    curriculum = default_curriculum()
    state = EvolutionOrchestrator(
        racap_root,
        experiment,
        curriculum,
        NoGainRunner(experiment),
        SequencedCoder(),
        FakeCritic(),
        scheduler=CurriculumScheduler(
            curriculum,
            SchedulerConfig(min_visits=1, patience=0, exploration=1.0),
        ),
    ).run("auto", 2, start_stage_id="s0_direct_transport")
    assert [record["stage"] for record in state.history] == [
        "s0_direct_transport",
        "s1_transport_react",
    ]
    assert state.champion_commit == state.history[0]["parent"]
    assert state.stage_states["s0_direct_transport"]["baseline_evaluations"] == 1
    assert state.stage_states["s1_transport_react"]["baseline_evaluations"] == 1


def test_same_coder_repairs_candidate_before_runtime_and_budget_counts_runtime_only(tmp_path):
    racap_root = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "evolution"
    orchestrator = EvolutionOrchestrator(
        racap_root,
        experiment,
        default_curriculum(),
        NoGainRunner(experiment),
        RepairingCoder(),
        FakeCritic(),
        implementation_repairs=1,
    )

    state = orchestrator.run(
        "s0_direct_transport",
        0,
        runtime_candidates=1,
    )

    assert state.runtime_candidates == 1
    assert state.implementation_attempts == 2
    assert state.stage_states["s0_direct_transport"]["runtime_candidates"] == 1
    assert len(state.history[-1]["implementation_attempts"]) == 2
    first, repaired = state.history[-1]["implementation_attempts"]
    assert first["candidate_commit"]
    assert repaired["base_commit"] == first["candidate_commit"]
    repaired_tree = (
        experiment / "solution_git" / "worktrees" / repaired["candidate"]
    )
    assert (
        repaired_tree / "memory" / "cumulative_repair_marker.txt"
    ).read_text() == "initial mechanism\n"
