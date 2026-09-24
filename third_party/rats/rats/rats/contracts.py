from __future__ import annotations

from typing import Protocol

from rats.rats.schemas import (
    BehaviorSceneContext,
    DiagnosisResult,
    ExecutionRecord,
    FeedbackAction,
    PlanBundle,
    PolicyDraft,
    QualityCheckResult,
    SkillSummary,
    TaskProposal,
    VerificationResult,
)


class EnvironmentCreatorContract(Protocol):
    def create_scene(self) -> tuple[object, BehaviorSceneContext]: ...
    def create_task_instance(self, proposal: TaskProposal) -> tuple[object, BehaviorSceneContext]: ...


class TaskProposerContract(Protocol):
    def propose(self, scene: BehaviorSceneContext, skill_summary: SkillSummary | None = None) -> TaskProposal: ...


class SkillLibraryContract(Protocol):
    def summarize(self) -> SkillSummary: ...
    def register(self, new_skills: list[str]) -> None: ...


class PlannerContract(Protocol):
    def plan(self, proposal: TaskProposal, skill_summary: SkillSummary | None = None) -> PlanBundle: ...


class PolicyWriterContract(Protocol):
    def write(self, plan: PlanBundle, context: BehaviorSceneContext | None = None) -> PolicyDraft: ...


class PolicyQualityCheckerContract(Protocol):
    def check(self, draft: PolicyDraft) -> QualityCheckResult: ...


class ExecutorContract(Protocol):
    def execute(self, draft: PolicyDraft, env: object, context: BehaviorSceneContext | None = None) -> ExecutionRecord: ...


class VerifierContract(Protocol):
    def verify(self, execution: ExecutionRecord, proposal: TaskProposal | None = None) -> VerificationResult: ...


class FailureDiagnoserContract(Protocol):
    def diagnose(self, execution: ExecutionRecord, context: BehaviorSceneContext | None = None) -> DiagnosisResult: ...


class FeedbackGeneratorContract(Protocol):
    def generate(
        self,
        execution: ExecutionRecord,
        verification: VerificationResult,
        diagnosis: DiagnosisResult,
    ) -> FeedbackAction: ...
