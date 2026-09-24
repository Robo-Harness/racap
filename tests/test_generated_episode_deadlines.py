from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CAPX_ROOT = ROOT / "third_party" / "rats" / "capx-baseline"
RATS_ROOT = ROOT / "third_party" / "rats"
for package_root in (str(CAPX_ROOT), str(RATS_ROOT)):
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


def test_capx_episode_deadline_escapes_generated_code_boundary() -> None:
    from capx.control_flow import CapXEpisodeWallTimeExceeded

    assert issubclass(CapXEpisodeWallTimeExceeded, BaseException)
    assert not issubclass(CapXEpisodeWallTimeExceeded, Exception)
    base_source = (
        CAPX_ROOT / "capx" / "envs" / "tasks" / "base.py"
    ).read_text(encoding="utf-8")
    runner_source = (
        CAPX_ROOT / "capx" / "envs" / "runner.py"
    ).read_text(encoding="utf-8")
    assert "except CapXControlFlowAbort:" in base_source
    assert "raise CapXEpisodeWallTimeExceeded(" in runner_source


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires Unix timers")
def test_capx_episode_deadline_repeats_when_first_signal_is_swallowed(
) -> None:
    """A third-party callback catching the first alarm must not erase the deadline."""

    from capx.control_flow import CapXEpisodeWallTimeExceeded
    from capx.deadline import arm_periodic_deadline, disarm_deadline

    deliveries = 0
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)

    def timeout_handler(signum, frame) -> None:
        nonlocal deliveries
        _ = signum, frame
        deliveries += 1
        raise CapXEpisodeWallTimeExceeded("test deadline")

    started = time.monotonic()
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        arm_periodic_deadline(0.1, 0.05)
        while deliveries < 2:
            try:
                time.sleep(0.2)
            except CapXEpisodeWallTimeExceeded:
                if deliveries == 1:
                    # Model the behavior of a third-party callback that logs
                    # and suppresses the first BaseException.
                    continue
                break
    finally:
        disarm_deadline()
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)

    assert deliveries >= 2
    assert time.monotonic() - started < 0.4
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0] == 0.0:
        assert remaining == 0.0
        assert interval == 0.0


class _SleepingEnv:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def step(self, code: str):
        _ = code
        time.sleep(self.seconds)
        return {}, 0.0, False, False, {}


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires Unix timers")
def test_rats_sandbox_does_not_mask_earlier_episode_deadline() -> None:
    from rats.control_flow import RATSEpisodeWallTimeExceeded
    from rats.executor.sandbox import Executor

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)

    def episode_deadline(signum, frame) -> None:
        _ = signum, frame
        raise RATSEpisodeWallTimeExceeded("episode deadline")

    started = time.monotonic()
    try:
        signal.signal(signal.SIGALRM, episode_deadline)
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        with pytest.raises(RATSEpisodeWallTimeExceeded, match="episode deadline"):
            Executor(timeout_seconds=5).execute("pass", _SleepingEnv(1.0))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)
    assert time.monotonic() - started < 0.8


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires Unix timers")
def test_rats_episode_deadline_repeats_when_first_signal_is_swallowed() -> None:
    from rats.control_flow import RATSEpisodeWallTimeExceeded
    from rats.deadline import (
        arm_episode_wall_deadline,
        restore_episode_wall_deadline,
    )

    state = arm_episode_wall_deadline(0.1)
    deliveries = 0
    started = time.monotonic()
    try:
        while deliveries < 2:
            try:
                time.sleep(1.2)
            except RATSEpisodeWallTimeExceeded:
                deliveries += 1
                if deliveries == 1:
                    continue
                break
    finally:
        restore_episode_wall_deadline(state)

    assert deliveries == 2
    assert time.monotonic() - started < 1.2


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires Unix timers")
def test_rats_sandbox_restores_outer_timer_minus_elapsed_time() -> None:
    from rats.executor.sandbox import Executor

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.signal(signal.SIGALRM, lambda signum, frame: None)
        signal.setitimer(signal.ITIMER_REAL, 5.0)
        Executor(timeout_seconds=1).execute("pass", _SleepingEnv(0.2))
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)
    assert 4.3 < remaining < 4.95
