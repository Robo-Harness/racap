import json

import pytest

from evolution.harness.evidence import EvidenceError, load_metrics
from evolution.harness.schema import EpisodeRef


def _row(task_id, *, success, turns=2):
    return {
        "key": f"libero_90/{task_id}/seed0",
        "task_id": task_id,
        "seed": 0,
        "native_success": success,
        "agent_success": success,
        "turns": turns,
        "seconds": 1.5,
        "simulator_steps": 120,
        "evaluator_runtime_calls": {"motion_calls": 3},
        "evaluator_llm_calls": {"logical_calls": 2},
        "steps": [{"action": "pickplace", "report": {"vlm_calls": 1}}],
        "reflections": [],
        "error": "",
    }


def test_metrics_are_computed_from_episode_records(tmp_path):
    rows = [_row(1, success=True), _row(2, success=False, turns=4)]
    (tmp_path / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    metrics = load_metrics(tmp_path, (EpisodeRef("libero_90", 1), EpisodeRef("libero_90", 2)))
    assert metrics.native_success == 1
    assert metrics.mean_turns == 3
    assert metrics.total_tool_calls == 6
    assert metrics.total_vlm_calls == 4
    assert metrics.mean_simulator_steps == 120


def test_missing_episode_cannot_be_masked_by_old_data(tmp_path):
    (tmp_path / "records.jsonl").write_text(json.dumps(_row(1, success=True)) + "\n")
    with pytest.raises(EvidenceError, match="episode mismatch"):
        load_metrics(tmp_path, (EpisodeRef("libero_90", 1), EpisodeRef("libero_90", 2)))


@pytest.mark.parametrize(
    "message",
    (
        'HTTP 403: {"code":"insufficient_user_quota"}',
        "need pre-deduct $0.84, balance is insufficient",
        "provider quota exceeded",
        "model routes to the relay but RACAP_VAPI_KEY is unset",
    ),
)
def test_external_model_failure_cannot_be_counted_as_native_failure(tmp_path, message):
    row = _row(1, success=False)
    row["steps"] = [{"action": "reflect", "observation": message}]
    (tmp_path / "records.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(EvidenceError, match="external service failures"):
        load_metrics(tmp_path, (EpisodeRef("libero_90", 1),))


def test_typed_terminal_model_failure_invalidates_evidence(tmp_path):
    row = _row(1, success=False)
    row["evaluator_llm_calls"] = {
        "logical_calls": 4,
        "network_attempts": 10,
        "terminal_failures": 3,
    }
    (tmp_path / "records.jsonl").write_text(json.dumps(row) + "\n")

    with pytest.raises(EvidenceError, match="infrastructure failures"):
        load_metrics(tmp_path, (EpisodeRef("libero_90", 1),))


def test_retry_amplified_grounding_outage_invalidates_legacy_evidence(tmp_path):
    row = _row(1, success=False, turns=21)
    row["simulator_steps"] = 31
    row["evaluator_llm_calls"] = {
        "logical_calls": 84,
        "network_attempts": 210,
    }
    row["steps"] = [
        {
            "action": "pickplace",
            "report": {"failure_mode": "not_grounded", "grounding_failure": "pick"},
        }
        for _ in range(21)
    ]
    (tmp_path / "records.jsonl").write_text(json.dumps(row) + "\n")

    with pytest.raises(EvidenceError, match="grounding outage"):
        load_metrics(tmp_path, (EpisodeRef("libero_90", 1),))
