from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

import scripts.eval_full_agent as evaluator
from scripts.eval_full_agent import (
    EpisodeWallTimeExceeded,
    InfrastructureInvariantError,
    _abort_exit_code,
    _abort_reason_for_infrastructure_failure,
    _outcome_infrastructure_failure,
    _all_recorded_keys,
    _deal,
    _episode_wall_timeout,
    _full_episode_controls,
    _historical_task_costs,
    _load_unique_records,
    _resolved_episode_grid,
    _validate_applied_init_state,
)


def test_episode_wall_timeout_cannot_be_swallowed_by_exception_retry_layers() -> None:
    assert issubclass(EpisodeWallTimeExceeded, BaseException)
    assert not issubclass(EpisodeWallTimeExceeded, Exception)


def test_run_manifest_grid_comes_from_resolved_episode_rows() -> None:
    payloads = [
        {"task_id": 5, "seed": 0},
        {"task_id": 5, "seed": 1},
        {"task_id": 5, "seed": 4},
    ]

    assert _resolved_episode_grid(payloads) == ([5], [0, 1, 4])


def test_mature_controller_receives_registered_anytime_semantics() -> None:
    context = {"5": {"objective_mode": "independent_anytime"}}

    assert _full_episode_controls(5, context) == {
        "continue_after_subgoal_failure": True,
        "max_replans": 3,
    }
    assert _full_episode_controls(4, context) == {
        "continue_after_subgoal_failure": False,
        "max_replans": 0,
    }


def test_resume_reads_consolidated_and_interrupted_worker_records(tmp_path: Path) -> None:
    consolidated = tmp_path / "records.jsonl"
    worker = tmp_path / "records.worker0.jsonl"
    consolidated.write_text('{"key":"suite/0/seed0","value":"old"}\n')
    worker.write_text('{"key":"suite/1/seed0","value":"new"}\n')

    assert _all_recorded_keys(consolidated, [worker]) == {
        "suite/0/seed0",
        "suite/1/seed0",
    }
    assert {row["key"] for row in _load_unique_records(consolidated, [worker])} == {
        "suite/0/seed0",
        "suite/1/seed0",
    }


def test_resume_rejects_conflicting_duplicate_episode_records(tmp_path: Path) -> None:
    consolidated = tmp_path / "records.jsonl"
    worker = tmp_path / "records.worker0.jsonl"
    consolidated.write_text('{"key":"suite/0/seed0","value":"old"}\n')
    worker.write_text('{"key":"suite/0/seed0","value":"different"}\n')

    with pytest.raises(InfrastructureInvariantError, match="conflicting resume"):
        _load_unique_records(consolidated, [worker])


def test_completed_resume_can_remove_absent_worker_lanes(tmp_path: Path) -> None:
    worker_lanes = [tmp_path / "records.worker0.jsonl"]

    for worker_lane in worker_lanes:
        worker_lane.unlink(missing_ok=True)

    assert not worker_lanes[0].exists()


def test_episode_wall_timeout_is_scoped_and_restores_handler(monkeypatch) -> None:
    handlers = []
    timers = []
    previous = object()

    monkeypatch.setattr(evaluator.signal, "getsignal", lambda _sig: previous)
    monkeypatch.setattr(
        evaluator.signal,
        "signal",
        lambda sig, handler: handlers.append((sig, handler)),
    )
    monkeypatch.setattr(
        evaluator.signal,
        "setitimer",
        lambda which, seconds: timers.append((which, seconds)) or (0.0, 0.0),
    )

    with pytest.raises(EpisodeWallTimeExceeded, match="17 seconds"):
        with _episode_wall_timeout(17):
            handlers[-1][1](None, None)

    assert timers == [
        (evaluator.signal.ITIMER_REAL, 17.0),
        (evaluator.signal.ITIMER_REAL, 0.0),
    ]
    assert handlers[-1] == (evaluator.signal.SIGALRM, previous)


def _episode(task_id: int, seed: int = 0) -> dict:
    return {"suite": "libero_90", "task_id": task_id, "seed": seed}


def test_deal_balances_historical_long_tail_episodes() -> None:
    values = [_episode(task_id) for task_id in range(8)]
    costs = {(task_id, 0): 10.0 for task_id in range(8)}
    costs.update({(1, 0): 100.0, (5, 0): 90.0})

    lanes = _deal(values, 2, costs)

    assert not any({row["task_id"] for row in lane} >= {1, 5} for lane in lanes)
    assert sorted(row["task_id"] for lane in lanes for row in lane) == list(range(8))


def test_historical_costs_use_median_completed_seconds(tmp_path: Path) -> None:
    for index, seconds in enumerate((10.0, 30.0, 500.0)):
        rollout = tmp_path / f"rollout-{index}"
        rollout.mkdir()
        (rollout / "summary.json").write_text(
            '{"suite":"libero_90","records":['
            f'{{"task_id":71,"seed":0,"seconds":{seconds}}}'
            "]}",
            encoding="utf-8",
        )

    assert _historical_task_costs(tmp_path, "libero_90")[(71, 0)] == 30.0


def test_queue_worker_consumes_until_sentinel(monkeypatch, tmp_path: Path) -> None:
    payload_queue: Queue = Queue()
    payload_queue.put(_episode(1))
    payload_queue.put(_episode(2))
    payload_queue.put(None)
    seen: list[int] = []

    def fake_worker(payloads, jsonl, out_dir, **kwargs):
        assert jsonl == tmp_path / "worker.jsonl"
        assert out_dir == tmp_path
        assert kwargs == {"worker_index": 0}
        seen.append(payloads[0]["task_id"])

    monkeypatch.setattr(evaluator, "_worker", fake_worker)
    evaluator._queue_worker(
        payload_queue,
        tmp_path / "worker.jsonl",
        tmp_path,
        worker_index=0,
    )

    assert seen == [1, 2]


def test_applied_init_state_is_checked_before_policy_execution() -> None:
    runtime = SimpleNamespace(
        _env=SimpleNamespace(_current_init_state_index=2),
    )

    assert _validate_applied_init_state(runtime, _episode(1, seed=2)) == 2
    with pytest.raises(InfrastructureInvariantError, match="49 != 2"):
        runtime._env._current_init_state_index = 49
        _validate_applied_init_state(runtime, _episode(1, seed=2))


def test_only_provider_failures_use_resumable_exit_code() -> None:
    assert _abort_exit_code({"reason": "quota_exhausted"}) == 75
    assert _abort_exit_code({"reason": "provider_unavailable"}) == 75
    assert _abort_exit_code({"reason": "infrastructure_invariant"}) == 1


def test_swallowed_tool_quota_failure_is_promoted_to_campaign_abort() -> None:
    outcome = SimpleNamespace(
        steps=[
            SimpleNamespace(report={}),
            SimpleNamespace(
                report={
                    "infrastructure_failure": {
                        "kind": "provider_quota",
                        "operation": "detect_many",
                    }
                }
            ),
        ]
    )

    failure = _outcome_infrastructure_failure(outcome)

    assert failure == {"kind": "provider_quota", "operation": "detect_many"}
    assert _abort_reason_for_infrastructure_failure(failure) == "quota_exhausted"


def test_swallowed_provider_failure_supports_dict_candidate_steps() -> None:
    outcome = SimpleNamespace(
        steps=[
            {
                "report": {
                    "infrastructure_failure": {
                        "kind": "provider_unavailable",
                        "operation": "verify",
                    }
                }
            }
        ]
    )

    failure = _outcome_infrastructure_failure(outcome)

    assert failure is not None
    assert _abort_reason_for_infrastructure_failure(failure) == "provider_unavailable"
