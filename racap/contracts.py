"""Stable contracts between evolved policy code and the runtime agent.

The prototype intentionally keeps simulator-specific objects behind this
protocol.  A learned policy can therefore be tested with a mock backend and
later bound to RATs/LIBERO, MolmoSpaces, or a real robot adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GroundedTarget:
    label: str
    pose: tuple[float, float, float]
    kind: str = "object"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    success: bool
    status: str
    pick_label: str
    place_label: str
    attempts: int
    strategy: str
    failure_mode: str = ""
    message: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    # What the skill measured while running, in the units a caller can act on:
    # where it decided things were, how high it grasped and released, which
    # strategy took hold. A controller that only sees ``failure_mode`` can
    # retry; one that sees these can retry *differently*.
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrimitiveRuntime(Protocol):
    """Primitive surface consumed by the evolved pick-place implementation."""

    def observe(self) -> dict[str, Any]: ...

    def localize(self, label: str, *, close_view: bool = False) -> GroundedTarget: ...

    def probe_bbox(self, bbox: list[int]) -> dict[str, Any]: ...

    def open_gripper(self) -> None: ...

    def grasp(self, target: GroundedTarget, *, strategy: str) -> None: ...

    def contact_grasp(
        self, target: GroundedTarget, *, strategy: str, lift: float = 0.0
    ) -> dict[str, Any] | None: ...

    def verify_grasp(self, label: str) -> bool: ...

    def place(self, target: GroundedTarget, *, strategy: str) -> None: ...

    def verify_place(self, pick_label: str, place_label: str) -> bool: ...

    def recover(self) -> None: ...
