from evolution.harness.curriculum import default_curriculum
from evolution.harness.memory import ExperienceMemory
from evolution.harness.schema import CriticReport


def _report(key="libero_90/9/seed0"):
    return CriticReport(
        episode_key=key,
        failure_phase="grasp",
        visual_evidence=("frame 12: fingers close above the object",),
        hypothesis="the approach point is above the graspable surface",
        confidence=0.8,
        counterfactual="a wrist frame showing contact would refute this",
        generality_scope="exposed transport of low objects",
        suggested_layer="policy_api",
        lesson_condition="a low object is localized but post-lift motion is absent",
        lesson_antipattern="reuse a mask centroid without a contact-height check",
        lesson_remedy="expose grasp height and verify contact before lift",
        applicable_tags=("pickplace", "empty_grasp"),
    )


def test_memory_persists_retrieves_and_tracks_empirical_outcome(tmp_path):
    memory = ExperienceMemory(tmp_path / "experience")
    stage = default_curriculum().get("s0_direct_transport")
    memory.ingest([_report()], stage=stage, commit="abc", iteration=0, rollout_label="base")
    lessons = memory.retrieve(stage, [_report()])
    assert len(lessons) == 1
    assert lessons[0]["evidence"][0]["episode_key"] == "libero_90/9/seed0"
    memory.record_application(
        [lessons[0]["lesson_id"]],
        iteration=1,
        candidate="def",
        native_delta=1,
        new_wins=["libero_90/9/seed0"],
        regressions=[],
    )
    updated = memory.retrieve(stage, [_report()])[0]
    assert updated["times_helped"] == 1
    assert (tmp_path / "experience" / "failure_episodes.jsonl").is_file()
    assert (tmp_path / "experience" / "applications.jsonl").is_file()
    snapshot = memory.snapshot()
    assert snapshot["failure_clusters"]
    assert snapshot["open_gaps"]
    assert (tmp_path / "experience" / "FAILURE_CLUSTERS.md").is_file()


def test_memory_records_capability_and_experiment_ledger(tmp_path):
    memory = ExperienceMemory(tmp_path / "experience")
    stage = default_curriculum().get("s0_direct_transport")
    memory.record_candidate(
        stage=stage,
        iteration=2,
        parent="abc",
        candidate="def",
        title="rim contact",
        mechanism="side contact avoids empty bowl centres",
        status="capability_promoted",
        native_delta=1,
        new_wins=["episode/1"],
        regressions=[],
        changed_files=["solution/policy_api.py"],
    )
    snapshot = memory.snapshot()
    assert snapshot["capabilities"]["pickplace contract"]["status"] == "empirically_improved"
    assert snapshot["recent_experiments"][0]["native_delta"] == 1
    assert (tmp_path / "experience" / "CAPABILITIES.md").is_file()
