"""In-memory backend for testing contracts without a simulator.

Useful for exercising the SkillResult schema, the retry ladder and the ReAct
orchestration in milliseconds. It proves nothing about physics; a policy that
passes here has only been shown to be well-formed.
"""

from __future__ import annotations

from typing import Any

from racap.contracts import GroundedTarget


class MockPrimitiveRuntime:
    """A scripted runtime whose difficulty is set by constructor flags.

    ``difficult`` makes the first grasp strategy fail, which exercises the
    retry ladder. ``impossible`` drops grounding confidence below the policy's
    threshold, which exercises the give-up path.
    """

    def __init__(self, *, difficult: bool = False, impossible: bool = False) -> None:
        self.difficult = difficult
        self.impossible = impossible
        self.last_strategy = ""
        self.held: str | None = None
        self.placed: dict[str, str] = {}

    def observe(self) -> dict[str, Any]:
        return {"rgb": "mock", "depth": "mock"}

    def localize(self, label: str, *, close_view: bool = False) -> GroundedTarget:
        confidence = 0.35 if self.impossible else (0.82 if close_view else 0.72)
        return GroundedTarget(label=label, pose=(0.1, 0.2, 0.3), confidence=confidence)

    def open_gripper(self) -> None:
        self.held = None

    def grasp(self, target: GroundedTarget, *, strategy: str) -> None:
        self.last_strategy = strategy
        if self.difficult and strategy == "direct_ik":
            return
        self.held = target.label

    def verify_grasp(self, label: str) -> bool:
        return self.held == label

    def place(self, target: GroundedTarget, *, strategy: str) -> None:
        if self.held is not None:
            self.placed[self.held] = target.label
            self.held = None

    def verify_place(self, pick_label: str, place_label: str) -> bool:
        return self.placed.get(pick_label) == place_label

    def recover(self) -> None:
        self.held = None


class MockSceneReasoner:
    """Stands in for the VLM. A real adapter returns the same two schemas."""

    def inventory(self, instruction: str) -> list[dict[str, str]]:
        return [
            {"label": "fork", "category": "tableware"},
            {"label": "spoon", "category": "tableware"},
            {"label": "toy cube", "category": "toy"},
            {"label": "teddy bear", "category": "toy"},
        ]

    def destination(self, item: dict[str, str], instruction: str) -> str:
        return "left plate" if item["category"] == "tableware" else "right box"
