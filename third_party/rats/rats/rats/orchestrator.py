from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rats.rats.registry import RatsRegistry, build_default_registry
from rats.rats.schemas import (
    BehaviorSceneContext,
    DiagnosisResult,
    ExecutionRecord,
    FeedbackAction,
    PlanBundle,
    PolicyDraft,
    QualityCheckResult,
    TaskProposal,
    VerificationResult,
)


@dataclass(slots=True)
class OrchestratorResult:
    scene_context: BehaviorSceneContext
    proposal: TaskProposal
    plan: PlanBundle
    draft: PolicyDraft
    quality: QualityCheckResult
    execution: ExecutionRecord
    verification: VerificationResult
    diagnosis: DiagnosisResult
    feedback: FeedbackAction
    retries: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class RatsOrchestrator:
    def __init__(
        self,
        registry: RatsRegistry | None = None,
        *,
        fixed_scene_model: str | None = None,
        max_feedback_retries: int = 0,
    ) -> None:
        self.registry = registry or build_default_registry(fixed_scene_model=fixed_scene_model)
        self.max_feedback_retries = max_feedback_retries

    def run_episode(self) -> OrchestratorResult:
        env, scene_context = self.registry.environment_creator.create_scene()
        skill_summary = self.registry.skill_library.summarize()
        proposal = self.registry.task_proposer.propose(scene_context, skill_summary)
        env, scene_context = self.registry.environment_creator.create_task_instance(proposal)
        plan = self.registry.planner.plan(proposal, skill_summary)
        draft = self.registry.policy_writer.write(plan, scene_context)
        quality = self.registry.policy_quality_checker.check(draft)

        retries = 0
        history: list[dict[str, Any]] = []

        if not quality.approved:
            execution = ExecutionRecord(success=False, stdout="", stderr=quality.feedback, reward=None, task_completed=False)
            verification = self.registry.verifier.verify(execution, proposal)
            diagnosis = self.registry.failure_diagnoser.diagnose(execution, scene_context)
            feedback = FeedbackAction(action="failure", message=quality.feedback)
            return OrchestratorResult(
                scene_context=scene_context,
                proposal=proposal,
                plan=plan,
                draft=draft,
                quality=quality,
                execution=execution,
                verification=verification,
                diagnosis=diagnosis,
                feedback=feedback,
                retries=retries,
                history=history,
            )

        while True:
            execution = self.registry.executor.execute(draft, env, scene_context)
            verification = self.registry.verifier.verify(execution, proposal)
            diagnosis = self.registry.failure_diagnoser.diagnose(execution, scene_context)
            feedback = self.registry.feedback_generator.generate(execution, verification, diagnosis)
            history.append(
                {
                    "retry": retries,
                    "execution_success": execution.success,
                    "verification_success": verification.success,
                    "feedback_action": feedback.action,
                }
            )
            if feedback.action != "retry" or retries >= self.max_feedback_retries:
                return OrchestratorResult(
                    scene_context=scene_context,
                    proposal=proposal,
                    plan=plan,
                    draft=draft,
                    quality=quality,
                    execution=execution,
                    verification=verification,
                    diagnosis=diagnosis,
                    feedback=feedback,
                    retries=retries,
                    history=history,
                )
            retries += 1
            draft = self.registry.policy_writer.write(plan, scene_context)
