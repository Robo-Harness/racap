"""Infrastructure control-flow signals that generated policy code must not absorb."""

from __future__ import annotations


class CapXControlFlowAbort(BaseException):
    """Base class for evaluator/runtime stops outside policy semantics."""


class CapXEpisodeWallTimeExceeded(CapXControlFlowAbort):
    """Raised when the registered continuous-episode deadline expires."""
