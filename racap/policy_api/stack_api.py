"""Footprint-aware placement of one object on another support.

``pickplace`` is deliberately broad.  ``stack`` adds the mechanical contract
that matters for a support relation: the carried centre of mass must lie in a
configuration-space region that keeps its footprint over the support.  Hollow
supports are treated separately because equal-sized bowls are stable by
nesting, even though a planar footprint erosion would (correctly) be empty.

All geometry comes from RGB-D masks.  Native predicates and simulator object
poses are never read by this policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from racap.backends.vlm import PerceptionUnavailableError
from racap.contracts import GroundedTarget, SkillResult
from racap.policy_api.insert_api import (
    Footprint,
    _robust_footprint,
    feasible_placement,
)
from racap.policy_api.pickplace_api import (
    _ground_verified_pick,
    _has_transient_spatial_identity,
    _looks_hollow,
    _plausible,
    pickplace,
)

DEFAULT_STACK_UNCERTAINTY_M = 0.002


def _finite_cloud(runtime, label: str) -> np.ndarray:
    try:
        cloud = np.asarray(runtime.object_points(label), dtype=float).reshape(-1, 3)
    except Exception:
        return np.zeros((0, 3), dtype=float)
    return cloud[np.isfinite(cloud).all(axis=1)]


def _target_cloud(runtime, target: GroundedTarget, label: str) -> np.ndarray:
    """Use the verified instance box instead of re-grounding a relation label."""
    bbox = target.metadata.get("bbox")
    if bbox:
        try:
            cloud = np.asarray(runtime.probe_bbox(list(bbox))["points"], dtype=float).reshape(-1, 3)
            cloud = cloud[np.isfinite(cloud).all(axis=1)]
            if cloud.shape[0] >= 30:
                return cloud
        except Exception:
            pass
    return _finite_cloud(runtime, label)


def support_polygon(points: np.ndarray) -> np.ndarray:
    """Convex RGB-D outline of the upper load-bearing band."""
    import cv2

    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if cloud.shape[0] < 30:
        return np.zeros((0, 2), dtype=float)
    top = float(np.percentile(cloud[:, 2], 98))
    bottom = float(np.percentile(cloud[:, 2], 2))
    band = max(0.012, min(0.035, 0.30 * max(top - bottom, 0.0)))
    upper = cloud[cloud[:, 2] >= top - band, :2]
    if upper.shape[0] < 12:
        return np.zeros((0, 2), dtype=float)
    return cv2.convexHull(upper.astype("float32")).reshape(-1, 2).astype(float)


def load_bearing_footprint(
    points: np.ndarray,
) -> tuple[Footprint | None, str]:
    """Estimate the contact patch, falling back to the full outer silhouette.

    Stability is governed by the part touching the support, not the widest rim
    of a bowl or the handle of cookware.  Fit the robust low-z band, retain its
    offset from the full body centre so COM projection remains constrained,
    and use the outer footprint whenever the base is too sparse or implausible.
    """
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    outer = _robust_footprint(cloud)
    if outer is None or cloud.shape[0] < 30:
        return outer, "outer_fallback"
    low_z, high_z = np.percentile(cloud[:, 2], [2, 98])
    height = float(max(high_z - low_z, 0.0))
    if height < 0.012:
        return outer, "outer_flat"
    cutoff = float(low_z + max(0.012, min(0.035, 0.22 * height)))
    lower = cloud[cloud[:, 2] <= cutoff]
    if lower.shape[0] < 12:
        return outer, "outer_sparse_base"
    # _robust_footprint's 30-point gate protects runtime masks.  A clean base
    # band with 12--29 points still carries enough planar information; repeat
    # its samples solely to reuse the same percentile fitter.
    fit_points = lower
    if lower.shape[0] < 30:
        repeats = int(np.ceil(30 / lower.shape[0]))
        fit_points = np.tile(lower, (repeats, 1))
    base = _robust_footprint(fit_points)
    if base is None:
        return outer, "outer_unfit_base"
    outer_size = np.asarray(outer.size, dtype=float)
    base_size = np.asarray(base.size, dtype=float)
    ratios = base_size / np.maximum(outer_size, 1e-6)
    centre_offset = np.asarray(base.centre) - np.asarray(outer.centre)
    if (
        np.any(ratios < 0.10)
        or np.any(ratios > 1.10)
        or float(np.linalg.norm(centre_offset)) > 0.40 * float(max(outer_size))
    ):
        return outer, "outer_implausible_base"
    # feasible_placement translates vertices about the body's command centre.
    # Preserve an eccentric base's offset while requiring that centre (our
    # visual COM proxy) itself stay inside the support polygon.
    contact = Footprint(
        centre=np.asarray(outer.centre, dtype=float),
        vertices=np.asarray(base.vertices, dtype=float) + centre_offset,
        long_axis_rad=float(base.long_axis_rad),
        size=tuple(float(v) for v in base.size),
    )
    return contact, "low_z_contact"


def stack_plan(
    object_points: np.ndarray,
    support_points: np.ndarray,
    *,
    preferred_xy: np.ndarray,
    uncertainty_m: float = DEFAULT_STACK_UNCERTAINTY_M,
    nesting_support: bool | None = None,
) -> dict[str, Any] | None:
    """Find a stable centre/yaw, or a nesting centre for a hollow support."""
    outer = _robust_footprint(object_points)
    moved, footprint_source = load_bearing_footprint(object_points)
    polygon = support_polygon(support_points)
    if moved is None or outer is None or polygon.shape[0] < 3:
        return None
    hollow = (
        _looks_hollow(np.asarray(support_points, dtype=float))
        if nesting_support is None
        else bool(nesting_support)
    )
    if hollow:
        # Open vessels carry another vessel through rim/nesting contact.  The
        # correct invariant is concentricity, not full planar containment.
        return {
            "centre_xy": np.asarray(preferred_xy, dtype=float).reshape(2),
            "rotation_rad": 0.0,
            "clearance_m": 0.0,
            "mode": "nesting",
            "object_size": tuple(outer.size),
            "contact_size": tuple(moved.size),
            "footprint_source": footprint_source,
            "support_polygon": polygon,
        }
    plan = feasible_placement(
        polygon,
        moved,
        uncertainty_m=float(np.clip(uncertainty_m, 0.0, 0.02)),
        preferred_xy=np.asarray(preferred_xy, dtype=float),
    )
    if plan is None:
        return None
    return {
        **plan,
        "mode": "footprint_support",
        "object_size": tuple(outer.size),
        "contact_size": tuple(moved.size),
        "footprint_source": footprint_source,
        "support_polygon": polygon,
    }


def stack(
    runtime,
    pick_label: str,
    support_label: str,
    *,
    uncertainty_m: float = DEFAULT_STACK_UNCERTAINTY_M,
    grasp: str | list[str] | None = None,
) -> SkillResult:
    """Place ``pick_label`` stably on (or concentrically in) ``support_label``."""
    trace: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "params": {
            "uncertainty_m": float(np.clip(uncertainty_m, 0.0, 0.02)),
            "grasp": grasp,
        },
        "preflight_only": True,
    }

    def finish(success: bool, mode: str, message: str) -> SkillResult:
        return SkillResult(
            success=success,
            status="success" if success else "retryable_failure",
            pick_label=pick_label,
            place_label=support_label,
            attempts=1,
            strategy="stable_stack",
            failure_mode=mode,
            message=message,
            trace=trace,
            report=report,
        )

    relational = bool(
        _has_transient_spatial_identity(pick_label)
        or _has_transient_spatial_identity(support_label)
    )
    try:
        if relational:
            # Keep the complete qualifier for both roles and independently
            # veto marked candidates.  Joint category grounding plus a crop
            # check cannot distinguish three visually identical vessels.
            moved, moved_identity, _ = _ground_verified_pick(
                runtime, pick_label, trace, max_attempts=3
            )
            support, support_identity, _ = _ground_verified_pick(
                runtime, support_label, trace, max_attempts=3
            )
            report["referent_verification"] = {
                "pick": moved_identity,
                "support": support_identity,
            }
        else:
            grounded = runtime.localize_many([pick_label, support_label])
            moved = grounded.get(pick_label)
            support = grounded.get(support_label)
    except PerceptionUnavailableError:
        raise
    except Exception as exc:
        mode = "relational" if relational else "joint"
        return finish(False, "not_grounded", f"{mode} grounding raised {type(exc).__name__}")
    if not _plausible(moved):
        report.update({"grounding_failure": "pick", "failed_label": pick_label})
        return finish(False, "not_grounded", f"no usable grounding for {pick_label!r}")
    if not _plausible(support):
        report.update({"grounding_failure": "destination", "failed_label": support_label})
        return finish(False, "not_grounded", f"no usable grounding for {support_label!r}")

    object_points = _target_cloud(runtime, moved, pick_label)
    support_points = _target_cloud(runtime, support, support_label)
    moved_footprint = _robust_footprint(object_points)
    polygon = support_polygon(support_points)
    # Interior depth can disappear at a grazing camera angle.  Semantic shape
    # is an independent affordance cue: two similarly sized objects explicitly
    # named as bowls/cups are nestable even when the rear vessel's inner floor
    # is occluded.  Size agreement prevents this cue from treating a small cup
    # on a much larger serving bowl as an automatic concentric nest.
    semantic_vessel = any(
        noun in pick_label.lower() and noun in support_label.lower() for noun in ("bowl", "cup")
    )
    support_hollow_rgbd = bool(_looks_hollow(support_points))
    sizes_similar = False
    support_footprint = _robust_footprint(support_points)
    if moved_footprint is not None and support_footprint is not None:
        ratios = np.asarray(moved_footprint.size) / np.maximum(
            np.asarray(support_footprint.size), 1e-6
        )
        sizes_similar = bool(np.all((ratios >= 0.70) & (ratios <= 1.30)))
    nesting_support = support_hollow_rgbd or (semantic_vessel and sizes_similar)
    report["geometry_preflight"] = {
        "pick_pose": [round(float(v), 4) for v in moved.pose],
        "support_pose": [round(float(v), 4) for v in support.pose],
        "pick_points": int(object_points.shape[0]),
        "support_points": int(support_points.shape[0]),
        "pick_size_cm": (
            None
            if moved_footprint is None
            else [round(float(v) * 100, 1) for v in moved_footprint.size]
        ),
        "support_vertices": int(polygon.shape[0]),
        "support_hollow_rgbd": support_hollow_rgbd,
        "semantic_vessel": semantic_vessel,
        "sizes_similar": sizes_similar,
        "nesting_support": nesting_support,
    }
    plan = stack_plan(
        object_points,
        support_points,
        preferred_xy=np.asarray(support.pose[:2], dtype=float),
        uncertainty_m=uncertainty_m,
        nesting_support=nesting_support,
    )
    if plan is None:
        report["recommended_recovery"] = (
            "choose a larger support or change the carried-object presentation"
        )
        return finish(
            False,
            "no_stable_support_region",
            "the RGB-D footprint has no stable centre on this support; arm not moved",
        )

    centre = np.asarray(plan["centre_xy"], dtype=float)
    support_top = float(
        support.metadata.get(
            "top_z",
            np.percentile(support_points[:, 2], 98) if support_points.size else support.pose[2],
        )
    )
    polygon = np.asarray(plan["support_polygon"], dtype=float)
    report["stack_plan"] = {
        "mode": plan["mode"],
        "centre_xy": [round(float(v), 4) for v in centre],
        "clearance_cm": round(float(plan.get("clearance_m", 0.0)) * 100, 2),
        "rotation_deg": round(float(np.degrees(plan.get("rotation_rad", 0.0))), 1),
        "object_size_cm": [round(float(v) * 100, 1) for v in np.asarray(plan["object_size"])],
        "contact_size_cm": [round(float(v) * 100, 1) for v in np.asarray(plan["contact_size"])],
        "footprint_source": plan["footprint_source"],
        "support_vertices": int(polygon.shape[0]),
    }
    trace.append({"event": "stack_preflight", **report["stack_plan"]})

    locked = GroundedTarget(
        label=support_label,
        pose=(float(centre[0]), float(centre[1]), float(support.pose[2])),
        kind="object",
        confidence=float(support.confidence),
        metadata={
            **support.metadata,
            "top_z": support_top,
            "region_polygon_xy": polygon.tolist(),
            "region_source": "support_top_band",
        },
    )
    delegated = pickplace(
        runtime,
        pick_label,
        support_label,
        grasp=grasp,
        destination="object",
        place_margin=0.002,
        release_on="rim",
        # Centre only after the carried object is high above the destination.
        # ``_centre_over`` rewrites transient selectors to an attachment query
        # ("object held by the gripper"), so it cannot rebind to a duplicate
        # remaining on the table.
        centre=True,
        compensate=False,
        _pick_target=moved,
        _destination_target=locked,
    )
    report.update(delegated.report)
    report["preflight_only"] = False
    report["stack_plan"] = {
        **report.get("stack_plan", {}),
        "mode": plan["mode"],
        "centre_xy": [round(float(v), 4) for v in centre],
    }
    trace.extend(delegated.trace)
    return SkillResult(
        success=bool(delegated.success),
        status=delegated.status,
        pick_label=pick_label,
        place_label=support_label,
        attempts=delegated.attempts,
        strategy="stable_stack",
        failure_mode=delegated.failure_mode,
        message=delegated.message,
        trace=trace,
        report=report,
    )
