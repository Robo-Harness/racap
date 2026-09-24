"""Process-local real-time deadline helpers for controlled CaP-X episodes."""

from __future__ import annotations

import os
import signal


# A periodic deadline survives a third-party callback suppressing one delivered
# exception.  The owning timeout boundary disarms it before artifact I/O.
TIMEOUT_REARM_SECONDS = max(
    0.05, float(os.getenv("CAPX_TIMEOUT_REARM_SECONDS", "1.0"))
)


def arm_periodic_deadline(
    initial_seconds: float,
    repeat_seconds: float = TIMEOUT_REARM_SECONDS,
) -> None:
    """Arm a repeating wall-clock signal until the owner explicitly disarms it."""

    signal.setitimer(
        signal.ITIMER_REAL,
        max(0.001, float(initial_seconds)),
        max(0.001, float(repeat_seconds)),
    )


def disarm_deadline() -> None:
    """Cancel both the initial deadline and its periodic re-arm."""

    signal.setitimer(signal.ITIMER_REAL, 0.0)


def rebase_active_deadline(initial_seconds: float) -> bool:
    """Restart an already-active timer while preserving its repeat interval."""

    remaining, repeat_interval = signal.getitimer(signal.ITIMER_REAL)
    disarm_deadline()
    if remaining <= 0.0:
        return False
    arm_periodic_deadline(
        initial_seconds,
        repeat_interval if repeat_interval > 0.0 else TIMEOUT_REARM_SECONDS,
    )
    return True
