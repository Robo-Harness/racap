from types import SimpleNamespace

from rats.agents.verifier import Verifier


class _FakeLiberoProblem:
    parsed_problem = {"goal_state": [["on", "black_bowl", "plate"]]}

    @staticmethod
    def _eval_predicate(_state):
        return False


class _FakeSatisfiedLiberoProblem:
    parsed_problem = {"goal_state": [["in", "ketchup", "basket"]]}

    @staticmethod
    def _eval_predicate(_state):
        return True


def test_libero_native_predicate_rejects_visual_false_positive(monkeypatch):
    """A VLM 'looks done' vote must not become a LIBERO success label."""

    verifier = Verifier(model=None)
    monkeypatch.setattr(
        verifier,
        "_run_visual_verifier",
        lambda *_args, **_kwargs: {
            "success": True,
            "confidence": 0.99,
            "observed_effect": "The bowl appears to be on the plate.",
        },
    )
    env = SimpleNamespace(
        low_level_env=SimpleNamespace(
            handle=SimpleNamespace(env=_FakeLiberoProblem()),
        ),
    )

    result = verifier.verify(
        {
            "success": True,
            "reward": 0.0,
            "task_completed": False,
            "trajectory_frames": [object(), object()],
        },
        {"activity_name": "put the black bowl on the plate"},
        env=env,
        code="",
    )

    assert result["success"] is False
    assert result["predicate_status"] == [
        {"predicate": "[on black_bowl plate]", "satisfied": False},
    ]
    assert result["evidence"]["verifier_votes"]["visual_custom"] is True
    assert result["evidence"]["outcome_authority"] == "native_predicates"
    assert result["evidence"]["native_predicate_success"] is False
    assert result["evidence"]["diagnostic_disagreement"] is True


def test_strict_benchmark_keeps_native_predicates_out_of_control_feedback(monkeypatch):
    """Exact benchmark predicates are labels/audit data, never retry context."""

    monkeypatch.setenv("RATS_VERIFIER_STRICT_BENCHMARK", "1")
    verifier = Verifier(model=None)
    monkeypatch.setattr(
        verifier,
        "_run_visual_verifier",
        lambda *_args, **_kwargs: {
            "success": False,
            "confidence": 0.9,
            "observed_effect": "The bowl remains beside the plate.",
        },
    )

    def _forbidden_native_analysis(**_kwargs):
        raise AssertionError("strict benchmark must not call predicate-aware LLM analysis")

    monkeypatch.setattr(verifier, "_llm_analyze", _forbidden_native_analysis)
    env = SimpleNamespace(
        low_level_env=SimpleNamespace(
            handle=SimpleNamespace(env=_FakeLiberoProblem()),
        ),
    )

    result = verifier.verify(
        {
            "success": True,
            "reward": 0.0,
            "task_completed": False,
            "trajectory_frames": [object(), object()],
        },
        {
            "activity_name": "libero_90_task9",
            "goal_conditions": "put the black bowl on the plate",
        },
        env=env,
        code="RESULT = {'success': True}",
        plan={"steps": [{"id": "step-1", "description": "place bowl on plate"}]},
    )

    assert result["success"] is False
    assert result["predicate_status"] == []
    assert result["satisfied_conditions"] == []
    assert result["unsatisfied_conditions"] == []
    assert result["plan_step_feedback"] == []
    assert "black_bowl" not in result["state_hint"]
    assert "plate" not in result["state_hint"]
    assert result["llm_analysis"] is None
    assert result["evidence"]["control_feedback_policy"] == "binary_native_outcome_only"
    assert result["evidence"]["privileged_feedback_redacted"] is True
    assert result["evidence"]["private_native_audit"]["predicate_status"] == [
        {"predicate": "[on black_bowl plate]", "satisfied": False},
    ]


def test_native_predicate_success_survives_post_goal_execution_error(monkeypatch):
    """A helper crash after reaching the goal must not erase native success."""

    monkeypatch.setenv("RATS_VERIFIER_STRICT_BENCHMARK", "1")
    verifier = Verifier(model=None)
    env = SimpleNamespace(
        low_level_env=SimpleNamespace(
            handle=SimpleNamespace(env=_FakeSatisfiedLiberoProblem()),
        ),
    )

    result = verifier.verify(
        {
            "success": False,
            "reward": 0.0,
            "task_completed": False,
            "stderr": "ValueError: executing action in terminated episode",
            "trajectory_frames": [object(), object()],
        },
        {
            "activity_name": "libero_90_task48",
            "goal_conditions": "pick up the ketchup and put it in the basket",
        },
        env=env,
        code="raise RuntimeError('post-goal cleanup failed')",
    )

    assert result["success"] is True
    assert result["evidence"]["outcome_authority"] == "native_predicates"
    assert result["evidence"]["native_predicate_success"] is True
    assert result["evidence"]["execution_success"] is False
    assert result["evidence"]["native_success_overrode_execution_error"] is True
    assert result["evidence"]["privileged_feedback_redacted"] is True
    assert result["state_hint"] == "Goal satisfied."
