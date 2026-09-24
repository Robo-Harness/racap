"""Process-local wall-time deadline helpers for controlled RATS episodes."""

from __future__ import annotations

import signal
from typing import Any

from rats.control_flow import RATSEpisodeWallTimeExceeded


def arm_episode_wall_deadline(seconds: float) -> tuple[Any, tuple[float, float]]:
    """Arm the registered policy deadline and return the prior signal state.

    The repeating signal is deliberate: a third-party callback that suppresses
    one asynchronous exception must not silently grant the policy extra time.
    ``RATSEpisodeWallTimeExceeded`` inherits from ``BaseException`` so generated
    policy code using ordinary ``except Exception`` cannot absorb the stop.
    """

    seconds = float(seconds)
    if seconds <= 0:
        raise ValueError("episode wall deadline must be positive")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def _deadline_handler(signum: int, frame: Any) -> None:
        del signum, frame
        raise RATSEpisodeWallTimeExceeded(
            f"Episode timed out after {seconds:g} seconds"
        )

    signal.signal(signal.SIGALRM, _deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
    return previous_handler, previous_timer


def restore_episode_wall_deadline(
    state: tuple[Any, tuple[float, float]] | None,
) -> None:
    """Disarm our deadline and restore any timer owned by the caller."""

    if state is None:
        return
    previous_handler, (previous_seconds, previous_interval) = state
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, previous_handler)
    if previous_seconds > 0:
        signal.setitimer(
            signal.ITIMER_REAL,
            previous_seconds,
            previous_interval,
        )
