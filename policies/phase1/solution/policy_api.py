"""Frozen Policy API wrapper at the end of RACaP Phase 1.

Phase 1 establishes the non-privileged mechanisms required for reliable
transport, insertion, stacking, articulation, control actuation, and bounded
recovery. Phase 2 may edit these wrappers without exposing native predicates
or simulator state.
"""

from __future__ import annotations

from typing import Any

from racap.contracts import GroundedTarget, SkillResult
from racap.policy_api.actuate_control_api import actuate_control as _actuate_control
from racap.policy_api.articulate_api import articulate as _articulate
from racap.policy_api.insert_api import insert as _insert
from racap.policy_api.pickplace_api import pickplace as _pickplace
from racap.policy_api.push_api import push as _push
from racap.policy_api.stack_api import stack as _stack


def pickplace(
    runtime: Any,
    pick_label: str,
    place_label: str,
    *,
    grasp: str | list[str] | None = None,
    grasp_depth: float | None = None,
    destination: str = "auto",
    nudge: tuple[float, float] = (0.0, 0.0),
    place_margin: float = 0.03,
    release_on: str = "rim",
    yaw_deg: float | str | None = "auto",
    centre: bool = False,
    compensate: bool = False,
    identity_check: bool = False,
    _pick_target: GroundedTarget | None = None,
    _destination_target: GroundedTarget | None = None,
) -> SkillResult:
    """General transport with measured grounding, grasp and placement geometry."""
    return _pickplace(
        runtime,
        pick_label,
        place_label,
        grasp=grasp,
        grasp_depth=grasp_depth,
        destination=destination,
        nudge=nudge,
        place_margin=place_margin,
        release_on=release_on,
        yaw_deg=yaw_deg,
        centre=centre,
        compensate=compensate,
        identity_check=identity_check,
        _pick_target=_pick_target,
        _destination_target=_destination_target,
    )


def insert(
    runtime: Any,
    pick_label: str,
    place_label: str,
    *,
    grasp: str | list[str] | None = None,
    grasp_depth: float | None = None,
    uncertainty_m: float = 0.004,
    insertion_depth: float | None = None,
    frontload: bool | None = None,
    destination_description: str | None = None,
    _destination_target: GroundedTarget | None = None,
    _pick_target: GroundedTarget | None = None,
) -> SkillResult:
    """Footprint-aware bounded placement and contact insertion."""
    return _insert(
        runtime,
        pick_label,
        place_label,
        grasp=grasp,
        grasp_depth=grasp_depth,
        uncertainty_m=uncertainty_m,
        insertion_depth=insertion_depth,
        frontload=frontload,
        destination_description=destination_description,
        _destination_target=_destination_target,
        _pick_target=_pick_target,
    )


def stack(
    runtime: Any,
    pick_label: str,
    support_label: str,
    *,
    uncertainty_m: float = 0.002,
    grasp: str | list[str] | None = None,
) -> SkillResult:
    """Support-footprint-aware stable stacking."""
    return _stack(
        runtime,
        pick_label,
        support_label,
        uncertainty_m=uncertainty_m,
        grasp=grasp,
    )


def push(
    runtime: Any,
    pick_label: str,
    place_label: str = "",
    *,
    nudge: tuple[float, float] = (0.0, 0.0),
    expected_pose: tuple[float, float, float] | None = None,
) -> SkillResult:
    """Bounded visual planar correction after a release."""
    return _push(
        runtime,
        pick_label,
        place_label,
        nudge=nudge,
        expected_pose=expected_pose,
    )


def articulate(
    runtime: Any,
    target: str,
    *,
    goal: str,
    part: str = "",
    mechanism: str = "auto",
    amount: float | None = None,
    contact_strategy: str = "pca_axis",
    hinge_hint: list[float] | tuple[float, float] | None = None,
    handle_z_hint: float | None = None,
    hinge_radius_hint: float | None = None,
    panel_label_hint: str | None = None,
    axis_hint: list[float] | tuple[float, float] | None = None,
    opening_polygon_hint: (
        list[list[float]] | tuple[tuple[float, float], ...] | None
    ) = None,
) -> SkillResult:
    """Measured prismatic/revolute mechanism motion with contact alternatives."""
    return _articulate(
        runtime,
        target,
        goal=goal,
        part=part,
        mechanism=mechanism,
        amount=amount,
        contact_strategy=contact_strategy,
        hinge_hint=hinge_hint,
        handle_z_hint=handle_z_hint,
        hinge_radius_hint=hinge_radius_hint,
        panel_label_hint=panel_label_hint,
        axis_hint=axis_hint,
        opening_polygon_hint=opening_polygon_hint,
    )


def actuate_control(
    runtime: Any,
    target: str,
    *,
    goal: str,
    control_hint: str = "",
    mechanism: str = "auto",
    amount: float | None = None,
    contact_strategy: str = "auto",
) -> SkillResult:
    """Grounded button, switch or rotary-control contact."""
    return _actuate_control(
        runtime,
        target,
        goal=goal,
        control_hint=control_hint,
        mechanism=mechanism,
        amount=amount,
        contact_strategy=contact_strategy,
    )


SKILLS = {
    "pickplace": pickplace,
    "insert": insert,
    "stack": stack,
    "push": push,
    "articulate": articulate,
    "actuate_control": actuate_control,
}


__all__ = [*SKILLS, "SKILLS"]
