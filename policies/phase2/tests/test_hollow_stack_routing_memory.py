from racap.agent.react import Outcome

from solution import controller


def test_relational_hollow_stack_retrieves_routing_comparison():
    prompt = controller._instruction_strategy_memory(
        "stack the black bowl at the front on the black bowl in the middle"
    )

    assert "hollow" in prompt
    assert "similar rim or footprint size" in prompt
    assert "default stack route" in prompt
    assert "explicitly consider pickplace" in prompt
    assert "destination='top'" in prompt
    assert "centre=true" in prompt
    assert "compensate=true" in prompt
    assert "support height and concentricity" in prompt
    assert "not a forced object-name route" in prompt


def test_nonrelational_or_nonhollow_instruction_does_not_retrieve_prior():
    assert controller._instruction_strategy_memory("close the top drawer") == ""
    assert controller._instruction_strategy_memory("put the wine bottle in the drawer") == ""
    assert controller._instruction_strategy_memory("stack two blocks") == ""
    assert controller._instruction_strategy_memory("put the bowl on the plate") == ""


def test_full_react_receives_and_restores_routing_memory(monkeypatch):
    captured = {}
    original_system = getattr(controller.full_react, "SYSTEM", None)
    if not isinstance(original_system, str):
        monkeypatch.setattr(controller.full_react, "SYSTEM", "full system", raising=False)
        original_system = "full system"

    def fake_run(runtime, skill, episode):
        captured["system"] = controller.full_react.SYSTEM
        captured["episode"] = episode
        return Outcome(False, 1, stopped="test")

    monkeypatch.setattr(controller.full_react, "run_full_episode", fake_run)
    result = controller.run_episode(
        object(),
        {
            "instruction": "stack the middle black bowl on the back black bowl",
            "model": "test-model",
        },
    )

    assert result["stopped"] == "test"
    assert "Advisory tool-routing memory" in captured["system"]
    assert "explicitly consider pickplace" in captured["system"]
    assert controller.full_react.SYSTEM == original_system


def test_prompt_memory_installation_is_reversible():
    original_react = controller.react.SYSTEM
    original_full = getattr(controller.full_react, "SYSTEM", None)
    changed = controller._install_full_prompt_memory("candidate routing evidence")
    try:
        assert "candidate routing evidence" in controller.react.SYSTEM
        if isinstance(original_full, str):
            assert "candidate routing evidence" in controller.full_react.SYSTEM
    finally:
        controller._restore_prompt_memory(changed)

    assert controller.react.SYSTEM == original_react
    if isinstance(original_full, str):
        assert controller.full_react.SYSTEM == original_full
