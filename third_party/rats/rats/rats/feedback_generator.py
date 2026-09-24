from __future__ import annotations

from rats.rats.schemas import DiagnosisResult, ExecutionRecord, FeedbackAction, VerificationResult


class FeedbackGenerator:
    def generate(
        self,
        execution: ExecutionRecord,
        verification: VerificationResult,
        diagnosis: DiagnosisResult,
    ) -> FeedbackAction:
        if verification.success:
            return FeedbackAction(action="success", message="Execution verified.", next_task_signal=True)

        if diagnosis.confidence >= 0.5 and diagnosis.policy_feedback:
            return FeedbackAction(action="retry", message=diagnosis.policy_feedback)

        if execution.success:
            return FeedbackAction(action="failure", message="Execution completed but verification failed without actionable diagnosis.")

        return FeedbackAction(action="failure", message=diagnosis.failure_reason or "execution failed")
