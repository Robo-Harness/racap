from __future__ import annotations

import json
from types import SimpleNamespace

from rats.envs.tasks.base import _record_experiment_native_state, _record_experiment_reset


def test_reset_telemetry_prefers_simulator_applied_init_state(monkeypatch, tmp_path):
    output = tmp_path / "resets.jsonl"
    monkeypatch.setenv("RATS_SIM_EPISODE_TELEMETRY_PATH", str(output))
    monkeypatch.setenv("RATS_EPISODE_KEY", "libero_90/9/seed0")
    env = SimpleNamespace(
        handle=SimpleNamespace(
            init_states=[object(), object(), object()],
            task_language="put the black bowl on the plate",
        ),
        _current_init_state_index=0,
        suite_name="libero_90",
        task_id=9,
    )

    _record_experiment_reset(env, seed=1)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["seed"] == 1
    assert record["init_state_index"] == 0
    assert record["task_prompt"] == "put the black bowl on the plate"


def test_reset_telemetry_keeps_generic_seed_fallback(monkeypatch, tmp_path):
    output = tmp_path / "resets.jsonl"
    monkeypatch.setenv("RATS_SIM_EPISODE_TELEMETRY_PATH", str(output))
    env = SimpleNamespace(
        handle=SimpleNamespace(init_states=[object(), object(), object()]),
    )

    _record_experiment_reset(env, seed=1)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["init_state_index"] == 1


def test_native_telemetry_records_simulator_task_identity(monkeypatch, tmp_path):
    output = tmp_path / "native.jsonl"
    monkeypatch.setenv("RATS_NATIVE_TELEMETRY_PATH", str(output))
    env = SimpleNamespace(
        handle=SimpleNamespace(suite_name="libero_90", task_id=12),
        _sim_step_count=37,
    )

    _record_experiment_native_state(env, 0.0, False, 3)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["simulator_suite"] == "libero_90"
    assert record["simulator_task_id"] == 12
    assert record["simulator_steps"] == 37
