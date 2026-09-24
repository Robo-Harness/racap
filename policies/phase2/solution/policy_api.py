"""Frozen Policy API wrapper produced by RACaP Phase 2.

This layer starts from the Phase 1 mechanisms and adds the promoted,
general-purpose defaults selected during autonomous self-evolution. Native
predicates and simulator state remain unavailable.
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

# Automatic placement defaults should be conservative.  Paired rollout evidence
# showed that a centered inside release improves shallow tray placements, but the
# same automatic conversion regressed a deeper/high-sided basket by making a
# first miss non-recoverable.  The runtime agent may still explicitly request
# destination="container", release_on="inside", centre=True for any receptacle;
# this wrapper only changes generic/default-like controls for shallow tray-like
# language.  S1 evidence added one more default-like case: the visual agent can
# say destination="top" for a tray even when the instruction relation is
# containment.  With rim release, no nudge and centre=False, that behaves like
# the old unsafe rim/top drop rather than an intentional geometric override, so
# shallow receptacles convert it to the empirically improved inside placement.
_SHALLOW_CONTAINER_TERMS = frozenset(
    {
        "tray",
        "trays",
        "shallow",
        "pan",
    }
)
_GENERIC_DESTINATION_DEFAULTS = {"auto", "object", "default", ""}
_SHALLOW_TOP_DEFAULTS = {"top", "on", "surface"}


def _normalise_words(value: str) -> set[str]:
    return set(str(value or "").strip().lower().replace("_", " ").split())


def _normalise_destination(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _looks_like_shallow_container(label: str) -> bool:
    return bool(_normalise_words(label) & _SHALLOW_CONTAINER_TERMS)


def _zero_nudge(nudge: tuple[float, float]) -> bool:
    try:
        nudge_x, nudge_y = float(nudge[0]), float(nudge[1])
    except (TypeError, ValueError, IndexError):
        return False
    return abs(nudge_x) < 1e-6 and abs(nudge_y) < 1e-6


def _using_default_release_controls(
    *,
    destination: str,
    nudge: tuple[float, float],
    release_on: str,
    centre: bool,
) -> bool:
    return (
        _normalise_destination(destination) in _GENERIC_DESTINATION_DEFAULTS
        and _zero_nudge(nudge)
        and str(release_on or "").strip().lower() == "rim"
        and centre is False
    )


def _using_shallow_top_release_controls(
    *,
    destination: str,
    nudge: tuple[float, float],
    release_on: str,
    centre: bool,
) -> bool:
    """Return true for tray 'top' placements that are still physical defaults.

    This deliberately does not catch explicit offsets/nudges, centre=True, or
    non-rim releases. Those remain agent-owned strategy choices. The condition
    only broadens the shallow-container default when the supplied controls are
    otherwise indistinguishable from a generic rim drop.
    """
    return (
        _normalise_destination(destination) in _SHALLOW_TOP_DEFAULTS
        and _zero_nudge(nudge)
        and str(release_on or "").strip().lower() == "rim"
        and centre is False
    )


def _open_container_defaults(
    place_label: str,
    *,
    destination: str,
    nudge: tuple[float, float],
    place_margin: float,
    release_on: str,
    centre: bool,
) -> tuple[str, float, str, bool, bool, str]:
    """Choose interior placement only for generic shallow-container calls.

    The runtime agent owns explicit strategy choices. This wrapper only replaces
    retained rim-drop defaults for shallow receptacles where paired rollouts
    showed that rim/non-centred placement caused tray-edge failures. Deeper or
    high-sided receptacles keep retained defaults unless the agent explicitly
    asks for an inside/centred strategy, preserving recoverability observed in
    basket rollouts.
    """
    adjusted = False
    reason = "retained_defaults"
    shallow = _looks_like_shallow_container(place_label)
    default_controls = _using_default_release_controls(
        destination=destination,
        nudge=nudge,
        release_on=release_on,
        centre=centre,
    )
    shallow_top_controls = _using_shallow_top_release_controls(
        destination=destination,
        nudge=nudge,
        release_on=release_on,
        centre=centre,
    )
    if shallow and (default_controls or shallow_top_controls):
        destination = "container"
        release_on = "inside"
        centre = True
        place_margin = max(float(place_margin), 0.045)
        adjusted = True
        reason = (
            "shallow_container_top_to_inside"
            if shallow_top_controls
            else "shallow_container_default"
        )
    elif default_controls or shallow_top_controls:
        reason = "generic_or_deep_container_retained_for_recoverability"
    return destination, place_margin, release_on, centre, adjusted, reason


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
    original_destination = destination
    (
        destination,
        place_margin,
        release_on,
        centre,
        adjusted_open_container_default,
        candidate_default_reason,
    ) = _open_container_defaults(
        place_label,
        destination=destination,
        nudge=nudge,
        place_margin=place_margin,
        release_on=release_on,
        centre=centre,
    )
    result = _pickplace(
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
    try:
        report = result.report if isinstance(result.report, dict) else {}
        report.setdefault("candidate_default_reason", candidate_default_reason)
        if adjusted_open_container_default:
            report.setdefault("candidate_open_container_default", True)
            report.setdefault("candidate_original_destination", original_destination)
            report.setdefault("candidate_destination", destination)
            report.setdefault("candidate_release_on", release_on)
            report.setdefault("candidate_centre", centre)
            report.setdefault("candidate_place_margin", place_margin)
        result.report = report
    except Exception:
        pass
    return result


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
