from evolution.harness.curriculum import SEALED_IDS, default_curriculum


def test_curriculum_is_a_valid_dag_without_sealed_leakage():
    curriculum = default_curriculum()
    curriculum.validate()
    development = {episode.task_id for stage in curriculum.stages for episode in stage.active}
    assert not development.intersection(SEALED_IDS)
    assert len(curriculum.sealed) == 22


def test_stage_zero_exposes_semantics_but_not_task_identity_to_controller():
    stage = default_curriculum().get("s0_direct_transport")
    assert all(set(episode.context) == {"source", "destination"} for episode in stage.development)
    assert len(stage.active) == 13
    assert stage.prompt_file == "evolution/prompts/stages/s0_direct_transport.md"
    assert "Do not add yet" not in stage.prompt
    assert "may introduce any additional abstraction" in stage.prompt


def test_every_stage_loads_a_distinct_versioned_prompt():
    stages = default_curriculum().stages
    assert all(stage.prompt and stage.prompt_file for stage in stages)
    assert len({stage.prompt_file for stage in stages}) == len(stages)
