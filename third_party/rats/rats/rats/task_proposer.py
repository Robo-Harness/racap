from __future__ import annotations

from rats.rats.catalog import discover_r1pro_task_catalog
from rats.rats.schemas import (
    BehaviorSceneContext,
    LearningMetadata,
    SceneEvidence,
    SkillSummary,
    TaskProposal,
    TaskSummary,
)


class TaskProposer:
    def propose(self, scene: BehaviorSceneContext, skill_summary: SkillSummary | None = None) -> TaskProposal:
        promoted = skill_summary.promoted_skills if skill_summary is not None else []
        catalog = discover_r1pro_task_catalog(scene_model=scene.scene_model)

        selected = None
        if scene.activity_name is not None:
            selected = next((entry for entry in catalog if entry.activity_name == scene.activity_name), None)
        if selected is None and catalog:
            selected = catalog[0]

        activity_name = selected.activity_name if selected is not None else (scene.activity_name or "placeholder_activity")
        activity_definition_id = selected.activity_definition_id if selected is not None else (scene.activity_definition_id or 0)
        preferred_instance_id = scene.activity_instance_id if scene.activity_instance_id is not None else 0
        candidate_instance_ids = selected.candidate_instance_ids if selected is not None else [preferred_instance_id]
        natural_goal = scene.goal_conditions_nl or activity_name.replace("_", " ")
        object_scope_synsets = list(scene.object_scope.values())
        visible_objects = [key.split(".")[0] for key in object_scope_synsets][:5]

        return TaskProposal(
            scene_model=scene.scene_model,
            proposal_mode="existing_definition_single_scene",
            activity_name=activity_name,
            activity_definition_id=activity_definition_id,
            candidate_instance_ids=candidate_instance_ids,
            preferred_instance_id=preferred_instance_id,
            task_summary=TaskSummary(
                natural_language_goal=natural_goal,
                object_scope_synsets=object_scope_synsets,
                initial_condition_summary=[scene.initial_conditions_nl] if scene.initial_conditions_nl else [],
                goal_condition_summary=[scene.goal_conditions_nl] if scene.goal_conditions_nl else [natural_goal],
            ),
            scene_evidence=SceneEvidence(
                visible_objects=visible_objects,
                room_type_hypothesis=[scene.scene_model],
                compatibility_score=1.0 if selected is not None else 0.5,
                reasoning="Selected from fixed-scene BEHAVIOR catalog." if selected is not None else "Fallback to current scene context.",
            ),
            learning_metadata=LearningMetadata(
                required_primitives=promoted,
                novelty_score=0.0,
                learnability_score=1.0 if selected is not None else 0.5,
            ),
        )
