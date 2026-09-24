"""Non-policy control-flow signals that must escape generated user code.

Generated policies are allowed to catch ordinary :class:`Exception` values so
they can recover from a failed perception or motion primitive.  Infrastructure
events are different: a wall-clock deadline or provider-budget stop must leave
the generated program immediately.  Keeping these signals under
``BaseException`` prevents broad ``except Exception`` recovery blocks from
turning a one-shot alarm into an unbounded execution.
"""

from __future__ import annotations


class RATSControlFlowAbort(BaseException):
    """Base class for infrastructure stops that user policy code cannot handle."""


class RATSExecutionTimeout(RATSControlFlowAbort):
    """Raised when one generated-code execution exceeds its wall-clock budget."""


class RATSEpisodeWallTimeExceeded(RATSControlFlowAbort):
    """Raised when the registered continuous-episode deadline expires."""


class RATSProviderAbort(RATSControlFlowAbort):
    """Raised after a provider/quota checkpoint requests an immediate stop."""
