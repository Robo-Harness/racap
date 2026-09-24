from racap.agent.react import Outcome
from racap.contracts import SkillResult

from solution import controller, policy_api


def test_phase2_exposes_all_general_policy_apis():
    assert set(policy_api.SKILLS) == {
        "pickplace",
        "insert",
        "stack",
        "push",
        "articulate",
        "actuate_control",
    }


def test_pickplace_wrapper_is_editable_and_delegates(monkeypatch):
    captured = {}

    def retained(runtime, pick, place, **kwargs):
        captured.update(runtime=runtime, pick=pick, place=place, kwargs=kwargs)
        return SkillResult(
            True,
            "placed",
            pick,
            place,
            1,
            "retained",
            report={"mechanism": "retained"},
        )

    monkeypatch.setattr(policy_api, "_pickplace", retained)
    runtime = object()
    result = policy_api.pickplace(runtime, "soup", "plate", centre=True)
    assert result.success
    assert captured["runtime"] is runtime
    assert captured["pick"] == "soup"
    assert captured["place"] == "plate"
    assert captured["kwargs"]["centre"] is True


def test_pickplace_defaults_shallow_container_to_interior_release(monkeypatch):
    captured = {}

    def retained(runtime, pick, place, **kwargs):
        captured.update(runtime=runtime, pick=pick, place=place, kwargs=kwargs)
        return SkillResult(
            True,
            "placed",
            pick,
            place,
            1,
            "retained",
            report={},
        )

    monkeypatch.setattr(policy_api, "_pickplace", retained)
    result = policy_api.pickplace(object(), "alphabet soup", "wooden tray")

    assert result.success
    assert captured["kwargs"]["destination"] == "container"
    assert captured["kwargs"]["release_on"] == "inside"
    assert captured["kwargs"]["centre"] is True
    assert captured["kwargs"]["place_margin"] >= 0.045
    assert result.report["candidate_open_container_default"] is True
    assert result.report["candidate_default_reason"] == "shallow_container_default"


def test_pickplace_converts_shallow_container_top_rim_drop_to_interior(monkeypatch):
    captured = {}

    def retained(runtime, pick, place, **kwargs):
        captured.update(kwargs=kwargs)
        return SkillResult(True, "placed", pick, place, 1, "retained", report={})

    monkeypatch.setattr(policy_api, "_pickplace", retained)
    result = policy_api.pickplace(
        object(),
        "butter",
        "wooden tray",
        destination="top",
        release_on="rim",
        centre=False,
    )

    assert captured["kwargs"]["destination"] == "container"
    assert captured["kwargs"]["release_on"] == "inside"
    assert captured["kwargs"]["centre"] is True
    assert captured["kwargs"]["place_margin"] >= 0.045
    assert result.report["candidate_open_container_default"] is True
    assert result.report["candidate_original_destination"] == "top"
    assert result.report["candidate_default_reason"] == "shallow_container_top_to_inside"


def test_pickplace_preserves_deep_basket_defaults_for_recoverability(monkeypatch):
    captured = {}

    def retained(runtime, pick, place, **kwargs):
        captured.update(kwargs=kwargs)
        return SkillResult(True, "placed", pick, place, 1, "retained", report={})

    monkeypatch.setattr(policy_api, "_pickplace", retained)
    result = policy_api.pickplace(object(), "ketchup", "basket")

    assert captured["kwargs"]["destination"] == "auto"
    assert captured["kwargs"]["release_on"] == "rim"
    assert captured["kwargs"]["centre"] is False
    assert captured["kwargs"]["place_margin"] == 0.03
    assert "candidate_open_container_default" not in result.report
    assert (
        result.report["candidate_default_reason"]
        == "generic_or_deep_container_retained_for_recoverability"
    )


def test_pickplace_preserves_deep_basket_top_controls(monkeypatch):
    captured = {}

    def retained(runtime, pick, place, **kwargs):
        captured.update(kwargs=kwargs)
        return SkillResult(True, "placed", pick, place, 1, "retained", report={})

    monkeypatch.setattr(policy_api, "_pickplace", retained)
    result = policy_api.pickplace(
        object(),
        "ketchup",
        "basket",
        destination="top",
        release_on="rim",
        centre=False,
    )

    assert captured["kwargs"]["destination"] == "top"
    assert captured["kwargs"]["release_on"] == "rim"
    assert captured["kwargs"]["centre"] is False
    assert "candidate_open_container_default" not in result.report
    assert (
        result.report["candidate_default_reason"]
        == "generic_or_deep_container_retained_for_recoverability"
    )


def test_pickplace_preserves_explicit_agent_container_controls(monkeypatch):
    captured = {}

    def retained(runtime, pick, place, **kwargs):
        captured.update(kwargs=kwargs)
        return SkillResult(True, "placed", pick, place, 1, "retained", report={})

    monkeypatch.setattr(policy_api, "_pickplace", retained)
    policy_api.pickplace(
        object(),
        "butter",
        "tray",
        destination="object",
        nudge=(0.02, 0.0),
        release_on="rim",
        centre=False,
    )

    assert captured["kwargs"]["destination"] == "object"
    assert captured["kwargs"]["release_on"] == "rim"
    assert captured["kwargs"]["centre"] is False


def test_pickplace_preserves_explicit_shallow_top_nudge(monkeypatch):
    captured = {}

    def retained(runtime, pick, place, **kwargs):
        captured.update(kwargs=kwargs)
        return SkillResult(True, "placed", pick, place, 1, "retained", report={})

    monkeypatch.setattr(policy_api, "_pickplace", retained)
    policy_api.pickplace(
        object(),
        "butter",
        "tray",
        destination="top",
        nudge=(0.01, 0.0),
        release_on="rim",
        centre=False,
    )

    assert captured["kwargs"]["destination"] == "top"
    assert captured["kwargs"]["release_on"] == "rim"
    assert captured["kwargs"]["centre"] is False


def test_direct_transport_uses_transport_react_and_candidate_session(monkeypatch):
    captured = {}
    sentinel_session = object()
    monkeypatch.setattr(controller, "_session", lambda runtime: sentinel_session)

    def fake_run(runtime, skill, episode, **kwargs):
        captured.update(runtime=runtime, skill=skill, episode=episode, kwargs=kwargs)
        return Outcome(True, 3, steps=[{"action": "pickplace"}])

    monkeypatch.setattr(controller.react, "run_episode", fake_run)
    runtime = object()
    result = controller.run_episode(
        runtime,
        {
            "instruction": "put the black bowl on the plate",
            "source": "black bowl",
            "destination": "plate",
            "model": "test-model",
            "budgets": {"pickplace": 6},
        },
    )
    assert result["success"]
    assert captured["runtime"] is runtime
    assert captured["skill"] is policy_api.pickplace
    assert captured["kwargs"]["session"] is sentinel_session
    assert captured["episode"].pick == "black bowl"
    assert captured["episode"].place == "plate"
    assert captured["episode"].max_pickplace_calls == 6
    assert captured["episode"].initial_skill == "pickplace"
    assert "system_prompt" not in captured["kwargs"]


def test_direct_transport_retrieves_only_matching_object_memory(monkeypatch):
    captured = {}
    monkeypatch.setattr(controller, "_session", lambda runtime: object())

    def fake_run(runtime, skill, episode, **kwargs):
        captured.update(episode=episode, kwargs=kwargs)
        return Outcome(True, 1, steps=[])

    monkeypatch.setattr(controller.react, "run_episode", fake_run)
    result = controller.run_episode(
        object(),
        {
            "instruction": "put the butter in the tray",
            "source": "butter",
            "destination": "tray",
        },
    )

    assert result["success"]
    prompt = captured["kwargs"]["system_prompt"]
    assert "Small upright red-orange rectangular carton" in prompt
    assert "cream-cheese package" in prompt
    assert captured["episode"].initial_skill == "pickplace"


def test_full_instruction_uses_full_react_and_restores_session(monkeypatch):
    captured = {}
    original_session = controller.full_react.Session

    def fake_run(runtime, skill, episode):
        captured.update(runtime=runtime, skill=skill, episode=episode)
        assert controller.full_react.Session is controller._session
        return Outcome(False, 4, stopped="visual retry needed")

    monkeypatch.setattr(controller.full_react, "run_full_episode", fake_run)
    result = controller.run_episode(
        object(),
        {
            "instruction": "put butter in the drawer and close it",
            "model": "test-model",
            "budgets": {"turns": 60, "state": 7},
        },
    )
    assert not result["success"]
    assert result["stopped"] == "visual retry needed"
    assert captured["skill"] is policy_api.pickplace
    assert captured["episode"].turns == 60
    assert captured["episode"].max_state_calls == 7
    assert controller.full_react.Session is original_session


def test_public_anytime_mode_enables_independent_replanning(monkeypatch):
    captured = {}

    def fake_run(runtime, skill, episode):
        captured.update(runtime=runtime, skill=skill, episode=episode)
        return Outcome(False, 2, stopped="partial completion retained")

    monkeypatch.setattr(controller.full_react, "run_full_episode", fake_run)
    result = controller.run_episode(
        object(),
        {
            "instruction": "put all tabletop objects in the basket",
            "objective_mode": "independent_anytime",
            "model": "test-model",
        },
    )

    assert result["stopped"] == "partial completion retained"
    assert captured["episode"].continue_after_subgoal_failure is True
    assert captured["episode"].max_replans == 3
