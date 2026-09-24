from __future__ import annotations

from rats.rats.schemas import PlanBundle, PlanStep, SkillSummary, TaskProposal


class PlannerAgent:
    def plan(self, proposal: TaskProposal, skill_summary: SkillSummary | None = None) -> PlanBundle:
        promoted = skill_summary.promoted_skills if skill_summary is not None else []
        required = proposal.learning_metadata.required_primitives
        relevant = list(dict.fromkeys([*required, *promoted]))
        goal = proposal.task_summary.natural_language_goal
        goal_lower = goal.lower()
        steps: list[PlanStep] = []

        primary_objects = [
            synset for synset in proposal.task_summary.object_scope_synsets
            if not synset.startswith("agent") and "floor" not in synset
        ]
        primary_object = primary_objects[0] if primary_objects else "target object"

        steps.append(
            PlanStep(
                id="step-1",
                description=f"Observe the scene and localize {primary_object}.",
                relevant_skills=[skill for skill in relevant if skill in {"get_observation", "segment_object", "point_object", "mask_to_point_cloud", "estimate_object_pose"}],
                notes="Use deterministic perception only; no search sweeps or fallback logic.",
            )
        )

        if any(keyword in goal_lower for keyword in ("pick", "grasp", "lift", "toggle", "turn on", "turn off", "open", "close", "put", "place")):
            steps.append(
                PlanStep(
                    id="step-2",
                    description=f"Move the robot into position to interact with {primary_object}.",
                    relevant_skills=[skill for skill in relevant if skill in {"navigate_to_pose", "move_to_joint_positions", "move_end_effector_to_pose", "solve_ik"}],
                    notes="Use one-shot motion primitives and re-observe between actions in higher-level logic.",
                )
            )

        interaction_skills = [skill for skill in relevant if skill in {"plan_grasp", "choose_best_grasp", "open_gripper", "close_gripper", "object_in_hand"}]
        if interaction_skills:
            steps.append(
                PlanStep(
                    id="step-3",
                    description=f"Execute the task interaction needed to achieve: {goal}.",
                    relevant_skills=interaction_skills,
                    notes="Select one best candidate deterministically; surface failures instead of retrying inside primitives.",
                )
            )

        steps.append(
            PlanStep(
                id=f"step-{len(steps) + 1}",
                description="Check whether the goal condition appears satisfied and report the result.",
                relevant_skills=[skill for skill in relevant if skill in {"object_in_hand", "get_robot_state", "get_observation"}],
                notes="Verification remains separate from execution; collect enough evidence for the verifier.",
            )
        )

        return PlanBundle(
            task_id=f"{proposal.activity_name}:{proposal.preferred_instance_id}",
            retrieval_mode="diverse" if proposal.learning_metadata.novelty_score > 0.7 else "precise",
            steps=steps,
        )
