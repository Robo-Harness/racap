from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rats.rats.backends import image_to_data_url
from rats.rats.schemas import BehaviorSceneContext, DiagnosisResult, ExecutionRecord


class FailureDiagnoser:
    def __init__(self, query_backend: Callable[[list[dict[str, Any]]], str] | None = None) -> None:
        self.query_backend = query_backend

    def build_prompt(self, execution: ExecutionRecord, context: BehaviorSceneContext | None = None) -> list[dict[str, Any]]:
        goal = context.goal_conditions_nl if context is not None and context.goal_conditions_nl else "goal condition"
        stdout = execution.stdout or ""
        before_url = image_to_data_url(execution.artifacts.get("before_frame"))
        after_url = image_to_data_url(execution.artifacts.get("after_frame"))
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": f"Task goal: {goal}"},
            {"type": "text", "text": f"Stdout:\n{stdout}"},
            {"type": "text", "text": "Compare the before and after images. Did the task appear to succeed? If not, identify the most likely failed step and one concrete corrective suggestion. Respond in 3 short lines: SUCCESS=<yes/no>; FAILED_STEP=<name>; FEEDBACK=<text>."},
        ]
        if before_url is not None:
            user_content.extend([
                {"type": "text", "text": "Before image:"},
                {"type": "image_url", "image_url": {"url": before_url}},
            ])
        if after_url is not None:
            user_content.extend([
                {"type": "text", "text": "After image:"},
                {"type": "image_url", "image_url": {"url": after_url}},
            ])
        return [
            {"role": "system", "content": [{"type": "text", "text": "You are a robotic execution failure diagnoser."}]},
            {"role": "user", "content": user_content},
        ]

    def _diagnose_locally(self, execution: ExecutionRecord, context: BehaviorSceneContext | None = None) -> DiagnosisResult:
        stdout = execution.stdout or ""
        stderr = execution.stderr or ""

        if execution.success and (execution.task_completed or (execution.reward is not None and execution.reward >= 0.99)):
            return DiagnosisResult(
                visual_success=True,
                failure_reason="",
                policy_feedback="No corrective action needed.",
                confidence=1.0,
            )

        if "STEP_FAILED:" in stdout:
            failed_step = stdout.split("STEP_FAILED:")[-1].strip().splitlines()[0]
            return DiagnosisResult(
                visual_success=False,
                failed_step=failed_step,
                failure_reason=f"Execution reported step failure at {failed_step}",
                policy_feedback=f"Revise the policy around '{failed_step}' and keep primitive usage deterministic.",
                confidence=0.8,
            )

        if stderr:
            return DiagnosisResult(
                visual_success=False,
                failed_step="runtime",
                failure_reason=stderr.splitlines()[-1] if stderr.splitlines() else stderr,
                policy_feedback="Fix the runtime error before retrying; avoid dynamic execution or hidden retry loops.",
                confidence=0.9,
            )

        goal_hint = context.goal_conditions_nl if context is not None and context.goal_conditions_nl else "goal condition"
        reward = execution.reward if execution.reward is not None else 0.0
        return DiagnosisResult(
            visual_success=False,
            failed_step="goal-check",
            failure_reason=f"Execution ended without satisfying {goal_hint}",
            policy_feedback=f"Re-plan the interaction sequence and gather stronger evidence for {goal_hint}.",
            confidence=0.6 if execution.success or reward > 0 else 0.4,
        )

    def _diagnose_with_backend(self, execution: ExecutionRecord, context: BehaviorSceneContext | None = None) -> DiagnosisResult:
        content = self.query_backend(self.build_prompt(execution, context))
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        parsed: dict[str, str] = {}
        for line in lines:
            if "=" in line:
                key, value = line.split("=", 1)
                parsed[key.strip().upper()] = value.strip()
        success = parsed.get("SUCCESS", "no").lower() in {"yes", "true"}
        failed_step = parsed.get("FAILED_STEP")
        feedback = parsed.get("FEEDBACK", content.strip())
        return DiagnosisResult(
            visual_success=success,
            failed_step=None if failed_step in {None, "none", ""} else failed_step,
            failure_reason="" if success else feedback,
            policy_feedback=feedback,
            confidence=0.7,
        )

    def diagnose(self, execution: ExecutionRecord, context: BehaviorSceneContext | None = None) -> DiagnosisResult:
        has_frames = execution.artifacts.get("before_frame") is not None or execution.artifacts.get("after_frame") is not None
        if self.query_backend is not None and has_frames:
            return self._diagnose_with_backend(execution, context)
        return self._diagnose_locally(execution, context)
