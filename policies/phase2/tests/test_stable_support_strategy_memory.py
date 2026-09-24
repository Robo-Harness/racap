import json
from pathlib import Path

from solution import controller


def test_mug_source_retrieves_visually_gated_stable_support_prior():
    prompt = controller._strategy_memory("red coffee mug")

    assert "tall or hollow handled vessel" in prompt
    assert "small shallow support" in prompt
    assert "destination='top'" in prompt
    assert "centre=true" in prompt
    assert "compensate=true" in prompt
    assert "verify_grasp" in prompt
    assert "XY centroid proximity alone is insufficient" in prompt
    assert "re-grasp" in prompt
    assert "planar pushing" in prompt
    assert "not a hidden object-name dispatcher" in prompt


def test_coffee_cup_alias_retrieves_same_transferable_prior():
    prompt = controller._strategy_memory("yellow and white coffee cup")

    assert "stable base support" in prompt
    assert "current RGB-D" in prompt
    assert "runtime agent may choose different explicit controls" in prompt


def test_unrelated_source_does_not_receive_mug_strategy():
    assert controller._strategy_memory("black bowl") == ""


def test_strategy_memory_keeps_visual_override_and_no_forced_dispatch():
    path = Path(__file__).resolve().parents[1] / "memory" / "strategy_priors.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mug_prior = next(
        prior for prior in payload["priors"] if "mug" in prior.get("class_terms", [])
    )

    assert "Confirm both conditions from the current view" in mug_prior["visual_condition"]
    assert "Reject it" in mug_prior["uncertainty"]
    assert "runtime agent" in mug_prior["uncertainty"]
    assert "never force a Policy API dispatch" in payload["policy"]
