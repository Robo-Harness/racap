"""Capability-DAG curriculum and leakage-safe task partitions."""

from __future__ import annotations

from dataclasses import dataclass

from .prompting import read_prompt
from .schema import EpisodeRef, StageSpec

SUITE = "libero_90"


def _stage_prompt(stage_id: str) -> dict[str, str]:
    relative = f"stages/{stage_id}.md"
    return {"prompt_file": f"evolution/prompts/{relative}", "prompt": read_prompt(relative)}


def _episodes(
    ids: tuple[int, ...], contexts: dict[int, dict] | None = None
) -> tuple[EpisodeRef, ...]:
    contexts = contexts or {}
    return tuple(EpisodeRef(SUITE, task_id, 0, contexts.get(task_id, {})) for task_id in ids)


DIRECT_CONTEXT = {
    46: {"source": "alphabet soup", "destination": "basket"},
    47: {"source": "cream cheese box", "destination": "basket"},
    48: {"source": "ketchup", "destination": "basket"},
    49: {"source": "tomato sauce", "destination": "basket"},
    55: {"source": "alphabet soup", "destination": "tray"},
    56: {"source": "butter", "destination": "tray"},
    57: {"source": "cream cheese", "destination": "tray"},
    58: {"source": "ketchup", "destination": "tray"},
    59: {"source": "tomato sauce", "destination": "tray"},
    9: {"source": "black bowl", "destination": "plate"},
    10: {"source": "black bowl", "destination": "top of the cabinet"},
    30: {"source": "black bowl", "destination": "plate"},
    36: {"source": "white bowl", "destination": "plate"},
}

SEALED_IDS = frozenset((33, 35, *range(50, 55), *range(60, 65), *range(78, 84), *range(86, 90)))


@dataclass(frozen=True)
class Curriculum:
    stages: tuple[StageSpec, ...]
    sealed: tuple[EpisodeRef, ...]

    def get(self, stage_id: str) -> StageSpec:
        for stage in self.stages:
            if stage.id == stage_id:
                return stage
        raise KeyError(f"unknown curriculum stage: {stage_id}")

    def validate(self) -> None:
        ids = {stage.id for stage in self.stages}
        sealed_keys = {episode.key for episode in self.sealed}
        for stage in self.stages:
            unknown = set(stage.parents) - ids
            if unknown:
                raise ValueError(f"{stage.id} has unknown parents: {sorted(unknown)}")
            leaked = {episode.key for episode in stage.active} & sealed_keys
            if leaked:
                raise ValueError(f"{stage.id} leaks sealed episodes: {sorted(leaked)}")

    def to_dict(self, *, include_sealed: bool = False) -> dict:
        result = {"stages": [stage.to_dict() for stage in self.stages]}
        if include_sealed:
            result["sealed"] = [episode.to_dict() for episode in self.sealed]
        return result


def default_curriculum() -> Curriculum:
    direct = _episodes(tuple(DIRECT_CONTEXT), DIRECT_CONTEXT)
    transport_react = _episodes((18, 19, 65, 66, 67, 68, 71, 72))
    bounded = _episodes((12, 13, 14, 15, 34, 37, 69, 70, 76, 84, 85))
    insert = _episodes((2, 24, 29, 73, 74, 75, 77, 40, 42))
    stack = _episodes((16, 17))
    articulate = _episodes((0, 22, 28, 6, 7, 11))
    control = _episodes((20, 39, 44))
    routing = _episodes((25, 26, 27, 31, 32, 41, 43))
    composition = _episodes((1, 3, 4, 5, 8, 21, 23, 45, 38))

    stages = (
        StageSpec(
            "s0_direct_transport",
            "Direct exposed transport",
            "pickplace contract",
            (),
            direct,
            hints=(
                "Object transport commonly needs localization, grasp, lift/transit, placement and retreat.",
                "Expose grasp family, destination mode and XY offset instead of hiding future strategies.",
                "Return phase-specific evidence for grounding, grasp and placement failures.",
                "The agent may add any API or runtime abstraction supported by rollout evidence.",
            ),
            **_stage_prompt("s0_direct_transport"),
        ),
        StageSpec(
            "s1_transport_react",
            "Transport recovery",
            "minimal ReAct",
            ("s0_direct_transport",),
            transport_react,
            direct,
            hints=(
                "Use visual runtime reasoning where it improves transport recovery and state decisions.",
                "The agent may retry up to the configured motion-call budget and owns stopping.",
                "Call reflection on ambiguity or failure; do not duplicate equivalent VLM checks.",
            ),
            **_stage_prompt("s1_transport_react"),
        ),
        StageSpec(
            "s2_bounded_placement",
            "Grounding and bounded placement",
            "identity + geometry",
            ("s1_transport_react",),
            bounded,
            transport_react,
            hints=(
                "Use VLM candidates followed by independent visual evidence and negative-feedback re-grounding.",
                "Reason about object footprint and destination feasible centre regions, not only box centres.",
            ),
            **_stage_prompt("s2_bounded_placement"),
        ),
        StageSpec(
            "s3a_insert",
            "Constrained insertion",
            "insert",
            ("s2_bounded_placement",),
            insert,
            bounded,
            hints=(
                "Insertion may benefit from a distinct constrained-motion contract or another evidence-supported abstraction.",
                "Expose approach direction, object yaw, insertion depth and correction controls.",
            ),
            **_stage_prompt("s3a_insert"),
        ),
        StageSpec(
            "s3b_stack",
            "Support-aware stacking",
            "stack",
            ("s2_bounded_placement",),
            stack,
            bounded,
            hints=("Model support height, footprint overlap and post-release stability.",),
            **_stage_prompt("s3b_stack"),
        ),
        StageSpec(
            "s3c_articulation",
            "Articulated mechanisms",
            "articulate",
            ("s1_transport_react",),
            articulate,
            transport_react,
            hints=(
                "Infer joint type and choose pull, push or arc motion; grasping a handle is not always required.",
                "Report measured motion as evidence, but let the visual agent decide semantic open/closed state.",
            ),
            **_stage_prompt("s3c_articulation"),
        ),
        StageSpec(
            "s3d_control",
            "Compact controls",
            "control",
            ("s1_transport_react",),
            control,
            transport_react,
            hints=(
                "Expose a bounded contact point plus line/arc path primitive for knobs, switches and doors.",
            ),
            **_stage_prompt("s3d_control"),
        ),
        StageSpec(
            "s4_tool_routing",
            "Single-goal tool routing",
            "skill selection",
            ("s3a_insert", "s3b_stack", "s3c_articulation", "s3d_control"),
            routing,
            (*insert[:3], *stack, *articulate[:2], *control[:2]),
            hints=(
                "Choose APIs from language and current visual state; API names are advisory and evolvable.",
            ),
            **_stage_prompt("s4_tool_routing"),
        ),
        StageSpec(
            "s5_causal_composition",
            "Long-horizon causal composition",
            "multi-goal planning",
            ("s4_tool_routing",),
            composition,
            routing,
            hints=(
                "Build a dependency-ordered subgoal plan, then re-observe after each physical tool call.",
                "Do not close or deactivate a receptacle before dependent placement subgoals complete.",
            ),
            **_stage_prompt("s5_causal_composition"),
        ),
        StageSpec(
            "s6_efficiency",
            "Perturbation and efficiency",
            "generalization",
            ("s5_causal_composition",),
            (),
            composition,
            hints=(
                "Preserve native success while reducing redundant VLM calls, turns and simulator steps.",
            ),
            **_stage_prompt("s6_efficiency"),
        ),
    )
    curriculum = Curriculum(stages, _episodes(tuple(sorted(SEALED_IDS))))
    curriculum.validate()
    return curriculum
