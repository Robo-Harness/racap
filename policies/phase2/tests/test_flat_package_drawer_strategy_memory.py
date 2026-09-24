import json
from pathlib import Path

from solution import controller


def test_low_flat_package_drawer_instruction_retrieves_bounded_strategy():
    prompt = controller._instruction_strategy_memory(
        "put the butter at the back in the top drawer of the cabinet and close it"
    )

    assert "low, flat, rigid rectangular package" in prompt
    assert "one graspnet acquisition first" in prompt
    assert "verify_grasp" in prompt
    assert "invalidate the stale spatial lock" in prompt
    assert "destination='container'" in prompt
    assert "release_on='inside'" in prompt
    assert "centre=true" in prompt
    assert "compensate=true" in prompt
    assert "small deliberate nudge" in prompt
    assert "currently moved package" in prompt
    assert "not an unconditional butter or package dispatcher" in prompt


def test_unseen_alias_retrieves_same_geometry_condition():
    prompt = controller._instruction_strategy_memory(
        "place the flat packet inside the sliding compartment"
    )

    assert "current RGB-D" in prompt
    assert "graspnet acquisition first" in prompt
    assert "visible opening" in prompt
    assert "runtime agent owns" in prompt


def test_prior_is_narrow_and_does_not_change_unrelated_instructions():
    assert controller._instruction_strategy_memory("close the top drawer") == ""
    assert controller._instruction_strategy_memory("put the frying pan on the stove") == ""
    assert controller._instruction_strategy_memory("put the butter on the plate") == ""
    assert controller._instruction_strategy_memory("put the bowl in the drawer") == ""


def test_memory_keeps_visual_confirmation_and_agent_override():
    path = Path(__file__).resolve().parents[1] / "memory" / "strategy_priors.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    prior = next(
        entry
        for entry in payload["routing_priors"]
        if "flat package" in entry.get("instruction_trigger", {}).get("source_class_any", [])
    )

    assert "Confirm" in prior["visual_condition"]
    assert "Language match alone is insufficient" in prior["visual_condition"]
    assert "Reject it" in prior["uncertainty"]
    assert "runtime agent owns" in prior["uncertainty"]
    assert "never force a Policy API dispatch" in payload["policy"]
