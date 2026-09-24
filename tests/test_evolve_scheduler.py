from evolution.harness.curriculum import default_curriculum
from evolution.harness.scheduler import CurriculumScheduler, SchedulerConfig
from evolution.harness.schema import CriticReport, Metrics


def _metrics(success=2, expected=10):
    successes = tuple(f"task/{index}" for index in range(success))
    failures = tuple(f"task/{index}" for index in range(success, expected))
    return Metrics(
        expected,
        success,
        success,
        1,
        1,
        100,
        expected,
        0,
        successes,
        failures,
        "/missing/records.jsonl",
        f"{success}/{expected}",
    )


def _report():
    return CriticReport(
        "task/2",
        "grasp",
        ("frame 4: empty grasp",),
        "grasp point is above the object",
        0.8,
        "visible contact would refute this",
        "transport grasp",
        "policy_api",
        applicable_tags=("pickplace", "grasp"),
    )


def test_child_becomes_schedulable_after_parent_is_observed_without_accuracy_gate():
    curriculum = default_curriculum()
    scheduler = CurriculumScheduler(
        curriculum,
        SchedulerConfig(min_visits=0, patience=0, exploration=1.0),
    )
    states = scheduler.ensure_states({})
    stage = curriculum.get("s0_direct_transport")
    scheduler.observe_baseline(
        states,
        stage,
        _metrics(),
        (),
        commit="abc",
        iteration=0,
        output_dir="/rollout",
        seconds=1,
    )
    decision = scheduler.choose(states, current_stage_id=stage.id)
    assert "s1_transport_react" in decision.eligible
    assert decision.stage_id == "s1_transport_react"


def test_positive_gain_keeps_current_stage_productive():
    curriculum = default_curriculum()
    scheduler = CurriculumScheduler(
        curriculum,
        SchedulerConfig(min_visits=0, patience=0, exploration=1.0),
    )
    states = scheduler.ensure_states({})
    stage = curriculum.get("s0_direct_transport")
    scheduler.observe_baseline(
        states,
        stage,
        _metrics(),
        (_report(),),
        commit="abc",
        iteration=0,
        output_dir="/rollout",
        seconds=1,
    )
    scheduler.observe_attempt(
        states,
        stage,
        native_delta=1,
        reports=(),
        iteration=1,
        seconds=1,
    )
    decision = scheduler.choose(states, current_stage_id=stage.id)
    assert decision.stage_id == stage.id
    assert states[stage.id]["status"] == "productive"


def test_invalid_patch_does_not_count_as_capability_saturation():
    curriculum = default_curriculum()
    scheduler = CurriculumScheduler(curriculum)
    states = scheduler.ensure_states({})
    stage = curriculum.get("s0_direct_transport")
    scheduler.observe_attempt(
        states,
        stage,
        native_delta=0,
        reports=(),
        iteration=1,
        seconds=0,
        valid_runtime=False,
    )
    assert states[stage.id]["visits"] == 0
    assert states[stage.id]["no_gain_streak"] == 0
    assert states[stage.id]["implementation_failures"] == 1


def test_plateau_probes_eligible_unseen_node_before_requested_revisit():
    curriculum = default_curriculum()
    scheduler = CurriculumScheduler(
        curriculum,
        SchedulerConfig(min_visits=0, patience=0, exploration=0.0),
    )
    states = scheduler.ensure_states({})
    direct = curriculum.get("s0_direct_transport")
    react = curriculum.get("s1_transport_react")
    for stage in (direct, react):
        scheduler.observe_baseline(
            states,
            stage,
            _metrics(),
            (),
            commit="abc",
            iteration=0,
            output_dir=f"/rollout/{stage.id}",
            seconds=1,
        )
    states[direct.id]["revisit_requests"] = 100
    scheduler.observe_attempt(
        states,
        react,
        native_delta=0,
        reports=(),
        iteration=1,
        seconds=1,
    )

    decision = scheduler.choose(states, current_stage_id=react.id)

    assert decision.stage_id in {
        "s2_bounded_placement",
        "s3c_articulation",
        "s3d_control",
    }
    assert states[decision.stage_id]["baseline_evaluations"] == 0
    assert decision.reason == "probe an eligible unvisited capability distribution"
