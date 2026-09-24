from __future__ import annotations

import json

from rats.control_flow import RATSEpisodeWallTimeExceeded
from rats.loop.lifelong_loop import LifelongLoop, resolve_libero_episode_key


def test_registered_episode_key_overrides_internal_trial_label(monkeypatch):
    monkeypatch.setenv("RATS_REGISTERED_EPISODE_KEY", "libero_goal_task/2/seed0")
    assert (
        resolve_libero_episode_key(
            {"suite_name": "libero_goal_task", "task_id": 2}, iteration=1, seed_offset=0
        )
        == "libero_goal_task/2/seed0"
    )


def test_unregistered_development_key_retains_rats_iteration_seed(monkeypatch):
    monkeypatch.delenv("RATS_REGISTERED_EPISODE_KEY", raising=False)
    assert (
        resolve_libero_episode_key(
            {"suite_name": "libero_90", "task_id": 12}, iteration=2, seed_offset=0
        )
        == "libero_90/12/seed2"
    )


def test_terminal_iteration_does_not_request_a_followup_reset():
    loop = object.__new__(LifelongLoop)
    loop._iteration = 1
    loop._run_terminal_iteration = 1

    assert loop._should_reset_for_next_iteration() is False


def test_nonterminal_iteration_requests_a_followup_reset():
    loop = object.__new__(LifelongLoop)
    loop._iteration = 1
    loop._run_terminal_iteration = 2

    assert loop._should_reset_for_next_iteration() is True


def test_lifelong_loop_serializes_timeout_without_followup_reset(
    monkeypatch, tmp_path
):
    loop = object.__new__(LifelongLoop)
    loop._resumed_results = []
    loop._iteration = 0
    loop.web_debugger = None
    loop.fixed_task = True
    loop._validated_tasks = None
    loop.env_type = "libero"
    loop._skip_completed_iteration_results = {}
    loop.output_dir = tmp_path
    loop.env = object()
    loop._bootstrap_activity = "registered_task"

    saved = []
    monkeypatch.setenv("RATS_EPISODE_WALL_TIME_SECONDS", "1000")
    monkeypatch.setenv("RATS_AGENT_IO_DIR", str(tmp_path / "agent_io"))
    monkeypatch.setenv("RATS_EPISODE_KEY", "libero_90/0/seed0")
    monkeypatch.setenv("RATS_LLM_MODEL", "gpt-5.5")
    monkeypatch.setattr(
        loop,
        "_run_one_iteration",
        lambda: (_ for _ in ()).throw(
            RATSEpisodeWallTimeExceeded("Episode timed out after 1000 seconds")
        ),
    )
    monkeypatch.setattr(loop, "_save_attempt_videos", lambda **kwargs: {})
    monkeypatch.setattr(loop, "_save_iteration_result", saved.append)
    monkeypatch.setattr(loop, "_log_iteration_summary", lambda result: None)
    monkeypatch.setattr(loop, "_build_summary", lambda results: {"iterations": results})
    monkeypatch.setattr(loop, "_save_summary", lambda summary: None)

    summary = loop.run(num_iterations=1)

    assert len(saved) == 1
    assert saved[0]["timed_out"] is True
    assert saved[0]["success"] is False
    assert saved[0]["feedback_action"] == "timeout"
    assert saved[0]["elapsed_seconds"] == 1000
    assert loop._run_terminal_iteration == 1
    assert summary["iterations"] == saved
    timeout_call = tmp_path / "agent_io" / "0000_episode_deadline.json"
    assert timeout_call.is_file()
    payload = json.loads(timeout_call.read_text())
    assert payload["actual_model"] is None
    assert payload["response"]["usage"] is None
