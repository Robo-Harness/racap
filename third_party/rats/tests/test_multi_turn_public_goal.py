from rats.agents.multi_turn_decider import resolve_public_task_goal


def test_scene_language_precedes_opaque_catalog_id() -> None:
    goal = resolve_public_task_goal(
        {"activity_name": "libero_90_task9"},
        {"goal_conditions_nl": "put the black bowl on the plate"},
    )

    assert goal == "put the black bowl on the plate"


def test_symbolic_goal_is_not_exposed_to_runtime_decider() -> None:
    goal = resolve_public_task_goal(
        {
            "activity_name": "libero_90_task9",
            "goal_conditions": [["on", "akita_black_bowl_1", "plate_1"]],
        },
        {},
    )

    assert goal == "libero_90_task9"
