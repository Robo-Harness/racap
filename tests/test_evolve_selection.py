from evolution.harness.schema import Metrics
from evolution.harness.selection import compare


def _metrics(success, *, vlm=4, tools=4, turns=4):
    return Metrics(
        expected=5,
        native_success=success,
        agent_claimed=success,
        mean_turns=turns,
        mean_seconds=1,
        mean_simulator_steps=100,
        total_tool_calls=tools,
        total_vlm_calls=vlm,
        successes=tuple(f"episode/{index}" for index in range(success)),
        failures=tuple(f"episode/{index}" for index in range(success, 5)),
        records_path="records.jsonl",
        digest=str(success),
    )


def test_any_native_improvement_promotes_without_threshold():
    decision = compare(_metrics(1), _metrics(2, vlm=100, tools=100, turns=20))
    assert decision.promote_capability


def test_equal_success_lower_measured_cost_is_efficiency_only():
    decision = compare(_metrics(2), _metrics(2, vlm=3))
    assert not decision.promote_capability
    assert decision.promote_efficiency
