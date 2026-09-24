from __future__ import annotations

from copy import deepcopy
from typing import Any

from rats.rats.catalog import discover_r1pro_task_catalog
from rats.rats.schemas import BehaviorSceneContext, TaskProposal


class EnvironmentCreator:
    def __init__(self, fixed_scene_model: str | None = None, base_env_factory: dict[str, Any] | None = None) -> None:
        self.fixed_scene_model = fixed_scene_model or "unset_scene"
        self.base_env_factory = deepcopy(base_env_factory) if base_env_factory is not None else None

    def create_scene(self) -> tuple[object, BehaviorSceneContext]:
        catalog = discover_r1pro_task_catalog(scene_model=self.fixed_scene_model)
        activity_name = catalog[0].activity_name if catalog else None
        context = BehaviorSceneContext(scene_model=self.fixed_scene_model, activity_name=activity_name)
        return object(), context

    def create_task_instance(self, proposal: TaskProposal) -> tuple[object, BehaviorSceneContext]:
        context = BehaviorSceneContext(
            scene_model=proposal.scene_model,
            activity_name=proposal.activity_name,
            activity_definition_id=proposal.activity_definition_id,
            activity_instance_id=proposal.preferred_instance_id,
            object_scope={synset: synset for synset in proposal.task_summary.object_scope_synsets},
            goal_conditions_nl="\n".join(proposal.task_summary.goal_condition_summary),
            initial_conditions_nl="\n".join(proposal.task_summary.initial_condition_summary),
        )
        return object(), context

    def build_runtime_binding(self, proposal: TaskProposal) -> dict[str, Any]:
        return {
            "scene_model": proposal.scene_model,
            "activity_name": proposal.activity_name,
            "activity_definition_id": proposal.activity_definition_id,
            "activity_instance_id": proposal.preferred_instance_id,
        }
