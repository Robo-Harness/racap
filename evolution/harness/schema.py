"""Serializable experiment types.

All records intentionally use plain JSON-compatible fields.  An experiment can
therefore be audited without importing the controller that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, order=True)
class EpisodeRef:
    suite: str
    task_id: int
    seed: int = 0
    context: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def key(self) -> str:
        explicit = self.context.get("episode_key")
        if explicit:
            return str(explicit)
        return f"{self.suite}/{self.task_id}/seed{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageSpec:
    id: str
    title: str
    capability: str
    parents: tuple[str, ...]
    development: tuple[EpisodeRef, ...]
    regression: tuple[EpisodeRef, ...] = ()
    hints: tuple[str, ...] = ()
    prompt_file: str = ""
    prompt: str = ""
    mutable_layers: tuple[str, ...] = ("policy_api", "agent", "memory")

    @property
    def active(self) -> tuple[EpisodeRef, ...]:
        seen: set[str] = set()
        result: list[EpisodeRef] = []
        for episode in (*self.development, *self.regression):
            if episode.key not in seen:
                seen.add(episode.key)
                result.append(episode)
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["active"] = [episode.to_dict() for episode in self.active]
        return value


@dataclass(frozen=True)
class Metrics:
    expected: int
    native_success: int
    agent_claimed: int
    mean_turns: float
    mean_seconds: float
    mean_simulator_steps: float
    total_tool_calls: int
    total_vlm_calls: int
    successes: tuple[str, ...]
    failures: tuple[str, ...]
    records_path: str
    digest: str

    @property
    def native_rate(self) -> float:
        return self.native_success / self.expected if self.expected else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "native_rate": self.native_rate}


@dataclass(frozen=True)
class CriticReport:
    episode_key: str
    failure_phase: str
    visual_evidence: tuple[str, ...]
    hypothesis: str
    confidence: float
    counterfactual: str
    generality_scope: str
    suggested_layer: str
    source_frames: tuple[int, ...] = ()
    causal_timeline: tuple[str, ...] = ()
    alternative_hypotheses: tuple[str, ...] = ()
    recommended_observation: str = ""
    lesson_condition: str = ""
    lesson_antipattern: str = ""
    lesson_remedy: str = ""
    applicable_tags: tuple[str, ...] = ()
    raw_response: str = ""
    first_divergence: str = ""
    failure_mechanism: str = ""
    phase_evidence: tuple[dict[str, Any], ...] = ()
    recommended_instrumentation: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Proposal:
    title: str
    hypothesis: str
    predicted_effect: str
    risk: str
    patch: str
    raw_response: str = ""
    target_failure_cluster: str = ""
    evidence: tuple[str, ...] = ()
    mechanism: str = ""
    expected_wins: tuple[str, ...] = ()
    regression_risks: tuple[str, ...] = ()
    falsification_test: str = ""
    # Complete-file edits are a robust alternative to model-generated unified
    # diffs. Values are UTF-8 contents; ``None`` deletes a candidate-owned file.
    # Keeping this field last preserves the positional constructor used by old
    # experiment fixtures.
    files: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    metrics: Metrics
    command: tuple[str, ...]
    stdout_path: Path
    seconds: float


@dataclass
class ExperimentState:
    experiment_id: str
    stage_id: str
    iteration: int
    champion_commit: str
    champion_metrics: dict[str, Any] | None = None
    efficiency_commit: str | None = None
    efficiency_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "fixed"
    start_stage_id: str = ""
    next_stage_id: str | None = None
    stage_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    scheduler_config: dict[str, Any] = field(default_factory=dict)
    # Iterations include model/network/implementation failures. These counters
    # expose the useful experimental budget separately.
    runtime_candidates: int = 0
    implementation_attempts: int = 0
    # A syntactically valid, committed candidate whose simulator evaluation or
    # visual critique has not yet been checkpointed. This makes quota/network
    # interruptions resumable without regenerating code or consuming another
    # candidate budget.
    pending_candidate: dict[str, Any] | None = None
    # Evaluator-owned protocol correction currently governing the selection
    # state.  Keeping this in the typed checkpoint prevents an amended run
    # from silently resuming under pre-amendment assumptions.
    active_protocol_amendment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
