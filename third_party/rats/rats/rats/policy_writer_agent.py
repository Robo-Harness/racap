from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rats.rats.schemas import BehaviorSceneContext, PlanBundle, PolicyDraft


class PolicyWriterAgent:
    def __init__(
        self,
        query_backend: Callable[[list[dict[str, Any]]], str] | None = None,
    ) -> None:
        self.query_backend = query_backend

    def build_prompt(self, plan: PlanBundle, context: BehaviorSceneContext | None = None) -> list[dict[str, Any]]:
        goal = context.goal_conditions_nl if context is not None and context.goal_conditions_nl else plan.task_id
        scene_model = context.scene_model if context is not None else "unknown_scene"
        object_scope = context.object_scope if context is not None else {}

        step_lines = []
        for step in plan.steps:
            skills = ", ".join(step.relevant_skills) if step.relevant_skills else "none"
            step_lines.append(f"- {step.id}: {step.description} | skills: {skills} | notes: {step.notes}")

        available_functions = context.available_functions if context is not None else []
        api_docs = context.api_docs if context is not None else None
        task_prompt = context.task_prompt if context is not None else None
        fn_list = ", ".join(available_functions) if available_functions else "(none provided)"
        api_block = f"Available imported functions (use ONLY these, not env.<method>):\n{fn_list}\n\nAPI docs:\n{api_docs}\n\n" if api_docs else f"Available imported functions (use ONLY these, not env.<method>):\n{fn_list}\n\n"
        user_text = (
            f"Write deterministic Python policy code for BEHAVIOR scene {scene_model}.\n"
            f"Goal: {goal}\n"
            f"Object scope: {object_scope}\n"
            + (f"Original task prompt:\n{task_prompt}\n\n" if task_prompt else "")
            + api_block
            + f"Plan:\n" + "\n".join(step_lines) + "\n\n"
            + "Requirements:\n"
            + "- Use ONLY the already-imported primitive functions listed above.\n"
            + "- Do NOT invent methods on env or low_level_env.\n"
            + "- Do not use hidden retry loops or search helpers.\n"
            + "- Keep primitive usage deterministic.\n"
            + "- Set RESULT to a structured dict describing success/failure.\n"
            + "- Prefer explicit comments before each plan step.\n"
        )
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a deterministic policy writer for a code-as-policy robot system."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
        ]

    def _synthesize_code(self, plan: PlanBundle, context: BehaviorSceneContext | None = None) -> str:
        goal = context.goal_conditions_nl if context is not None and context.goal_conditions_nl else plan.task_id
        lines = [
            f"# Deterministic policy draft for {goal}",
            "result_steps = []",
        ]
        for step in plan.steps:
            lines.append("")
            lines.append(f"# {step.id}: {step.description}")
            if step.relevant_skills:
                skill_list = ", ".join(step.relevant_skills)
                lines.append(f"# Suggested primitives: {skill_list}")
            if step.notes:
                lines.append(f"# Notes: {step.notes}")
            lines.append(f"result_steps.append({step.id!r})")
        lines.append("")
        lines.append("RESULT = {")
        lines.append(f"    'task_id': {plan.task_id!r},")
        lines.append(f"    'goal': {goal!r},")
        lines.append("    'planned_steps': result_steps,")
        lines.append("    'status': 'not_implemented',")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def write(self, plan: PlanBundle, context: BehaviorSceneContext | None = None) -> PolicyDraft:
        prompt = self.build_prompt(plan, context)
        if self.query_backend is not None:
            content = self.query_backend(prompt)
            code = content.strip()
            reasoning = "Generated via configured query backend."
        else:
            code = self._synthesize_code(plan, context)
            reasoning = "Synthesized deterministic scaffold from plan/context without external model call."

        return PolicyDraft(
            code=code,
            reasoning=reasoning,
            skills_used=[skill for step in plan.steps for skill in step.relevant_skills],
        )
