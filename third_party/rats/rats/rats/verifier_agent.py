from __future__ import annotations

from rats.rats.schemas import ExecutionRecord, TaskProposal, VerificationResult


class VerifierAgent:
    def verify(self, execution: ExecutionRecord, proposal: TaskProposal | None = None) -> VerificationResult:
        goals = []
        if proposal is not None:
            goals = proposal.task_summary.goal_condition_summary or [proposal.task_summary.natural_language_goal]
        if not goals:
            goals = ["task_complete"]

        reward = execution.reward if execution.reward is not None else 0.0
        task_completed = bool(execution.task_completed)
        verified = bool(execution.success) and (task_completed or reward >= 0.99)

        satisfied = goals if verified else []
        unsatisfied = [] if verified else goals
        evidence = {
            "reward": execution.reward,
            "task_completed": execution.task_completed,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
        }
        if execution.success and not verified and reward > 0:
            evidence["progress"] = reward

        return VerificationResult(
            success=verified,
            satisfied_conditions=satisfied,
            unsatisfied_conditions=unsatisfied,
            evidence=evidence,
        )
