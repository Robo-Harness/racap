"""Regression tests for infrastructure aborts escaping generated policies."""

from __future__ import annotations

import time
import json

from rats.control_flow import RATSExecutionTimeout, RATSProviderAbort
from rats.envs.tasks.base import CodeExecutionEnvBase
from rats.executor.sandbox import Executor
from rats.memory.failure_memory import FailureMemory


class _GeneratedCodeEnv:
    """Tiny env that models a generated policy catching ordinary exceptions."""

    def step(self, code: str):
        exec(code, {}, {})
        return {}, 0.0, False, False, {"sandbox_rc": 0}

    def render(self, mode: str = "rgb_array"):
        return None


def test_executor_deadline_bypasses_generated_except_exception() -> None:
    code = """
while True:
    try:
        time.sleep(0.05)
    except Exception:
        continue
"""
    started = time.monotonic()
    result = Executor(timeout_seconds=1).execute(code, _GeneratedCodeEnv())
    elapsed = time.monotonic() - started

    assert result["timeout"] is True
    assert result["truncated"] is True
    assert result["timeout_seconds"] == 1
    assert elapsed < 3.0


class _ExecHarness:
    """Minimal receiver for exercising CodeExecutionEnvBase._exec_user_code."""

    _apis = {}
    _is_molmospaces_runtime = False

    def __init__(self) -> None:
        self.low_level_env = object()
        self._exec_globals = {"RESULT": None}
        self._api_call_trace = []

    def _get_observation(self):
        return {}

    def _clear_api_runtime_diagnostics(self):
        return None


def test_code_environment_reraises_hard_timeout() -> None:
    harness = _ExecHarness()
    try:
        CodeExecutionEnvBase._exec_user_code(
            harness,
            "from rats.control_flow import RATSExecutionTimeout; "
            "raise RATSExecutionTimeout('deadline')",
        )
    except RATSExecutionTimeout as exc:
        assert str(exc) == "deadline"
    else:  # pragma: no cover - explicit assertion message
        raise AssertionError("hard timeout was serialized as a policy error")


def test_code_environment_reraises_provider_abort() -> None:
    harness = _ExecHarness()
    try:
        CodeExecutionEnvBase._exec_user_code(
            harness,
            "from rats.control_flow import RATSProviderAbort; "
            "raise RATSProviderAbort('quota')",
        )
    except RATSProviderAbort as exc:
        assert str(exc) == "quota"
    else:  # pragma: no cover - explicit assertion message
        raise AssertionError("provider abort was serialized as a policy error")


def test_failure_episode_records_iteration_transaction(tmp_path) -> None:
    memory = FailureMemory(tmp_path / "failure_memory")
    episode_id = memory.record_failure(
        task_name="task",
        failure_category="timeout",
        iteration=7,
    )

    episodes = json.loads((tmp_path / "failure_memory" / "episodes.json").read_text())
    episode = next(row for row in episodes if row["episode_id"] == episode_id)
    assert episode["iteration"] == 7
