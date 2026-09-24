from racap.agent.react import Outcome
from racap.contracts import SkillResult

from solution import controller, policy_api


def test_phase1_exposes_all_general_policy_apis():
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
    result = policy_api.pickplace(runtime, "soup", "basket", centre=True)
    assert result.success
    assert captured["runtime"] is runtime
    assert captured["pick"] == "soup"
    assert captured["place"] == "basket"
    assert captured["kwargs"]["centre"] is True


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
