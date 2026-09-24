"""Footprint-aware insertion into a visually grounded opening.

``pickplace`` is deliberately broad: it moves an object to a point and lets
gravity finish the placement.  A narrow opening is a different manipulation
problem.  The centre of an object can lie inside a compartment while most of
its body crosses a wall, so this skill plans in the object's planar
configuration space before it moves the arm.

The semantic model selects the named opening.  Dense RGB-D supplies its world
polygon.  The moved object's visible point cloud supplies a robust footprint.
For candidate wrist yaws we translate that footprint over the opening and keep
only centres for which every footprint vertex has the requested clearance.
No simulator object poses, sites, or task predicates are read.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from racap.backends.vlm import PerceptionUnavailableError

from racap.contracts import GroundedTarget, SkillResult
from racap.policy_api.pickplace_api import (
    MAX_GRASP_REGROUND_ANCHOR_M,
    MAX_GRASP_REGROUND_STEP_M,
    MIN_CLEARANCE,
    MIN_HAND_CLEARANCE,
    TRANSIT_CLEARANCE,
    _cloud,
    _ground_destination,
    _ground_verified_pick,
    _refresh_pick_after_contact,
    _attempts,
    _appearance_label,
    _grasp_height,
    _edge_grasp_height,
    _ground_visible,
    _held_object_left_the_table,
    _ladder,
    _lowered,
    _plausible,
    _robust_object_top,
    _release_band,
    _surface_height,
)


DEFAULT_UNCERTAINTY_M = 0.004
DEFAULT_INSERTION_DEPTH_M = None
MIN_FOOTPRINT_POINTS = 30
# A one-handed tabletop transport target wider than this is usually a semantic
# mask joined to table / fixture background.  Reject it before configuration-
# space planning so the general multi-prompt / multi-view grounding ladder can
# try a cleaner observation.  This is an end-effector feasibility bound, not
# an object-category size prior.
MAX_SINGLE_GRASP_FOOTPRINT_M = 0.20
MIN_FEASIBLE_CLEARANCE_M = 0.001
RIM_HAND_CLEARANCE_M = 0.008
INSERT_FINGER_HALF_WIDTH_M = 0.006
MAX_CONTROLLED_RIM_PENETRATION_M = 0.015
FLOOR_CONTACT_MARGIN_M = 0.003
MAX_VISUAL_SERVO_TRANSLATION_M = 0.06
VISUAL_SERVO_PROBE_RAD = np.deg2rad(15.0)
MAX_VISUAL_SERVO_STEP_RAD = np.deg2rad(60.0)
MAX_VISUAL_SERVO_STEPS = 5


@dataclass(frozen=True)
class Footprint:
    """A robust planar object footprint about ``centre``."""

    centre: np.ndarray
    vertices: np.ndarray
    long_axis_rad: float
    size: tuple[float, float]


def _wrap_half_turn(angle: float) -> float:
    """Canonicalise a rectangle orientation to ``[-pi/2, pi/2)``."""
    return float((angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi)


def _robust_footprint(points: np.ndarray) -> Footprint | None:
    """Fit a percentile OBB, rejecting sparse or implausible point clouds.

    A raw convex hull faithfully preserves one-pixel segmentation leaks onto a
    table or wall.  The 4/96-percentile OBB retains the object's body while
    discarding those tails; the explicit uncertainty margin is then the only
    intentional inflation applied by the planner.
    """
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if cloud.shape[0] < MIN_FOOTPRINT_POINTS:
        return None
    xy = cloud[:, :2]
    centre = np.median(xy, axis=0)
    flat = xy - centre
    try:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axes = np.asarray(vt[:2], dtype=float)
    projected = flat @ axes.T
    low = np.percentile(projected, 4, axis=0)
    high = np.percentile(projected, 96, axis=0)
    span = high - low
    if float(np.min(span)) < 0.004 or float(np.max(span)) > 0.40:
        return None

    # Recentre the robust box itself: a partially occluded mask need not be
    # symmetric around the raw point median.
    local_centre = 0.5 * (low + high)
    centre = centre + local_centre @ axes
    half = 0.5 * span
    local = np.asarray(
        [
            [-half[0], -half[1]],
            [half[0], -half[1]],
            [half[0], half[1]],
            [-half[0], half[1]],
        ]
    )
    vertices = local @ axes
    long_index = int(np.argmax(span))
    long_axis = axes[long_index]
    angle = float(np.arctan2(long_axis[1], long_axis[0]))
    sizes = tuple(sorted((float(span[0]), float(span[1])), reverse=True))
    return Footprint(
        centre=np.asarray(centre, dtype=float),
        vertices=np.asarray(vertices, dtype=float),
        long_axis_rad=_wrap_half_turn(angle),
        size=(sizes[0], sizes[1]),
    )


def _polygon_axis(polygon: np.ndarray) -> float:
    flat = polygon - polygon.mean(axis=0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    axis = vt[0]
    return _wrap_half_turn(float(np.arctan2(axis[1], axis[0])))


def _rotation(angle: float) -> np.ndarray:
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray([[c, -s], [s, c]], dtype=float)


def feasible_placement(
    opening_polygon_xy: np.ndarray,
    footprint: Footprint,
    *,
    uncertainty_m: float = DEFAULT_UNCERTAINTY_M,
    preferred_xy: np.ndarray | None = None,
    preferred_object_axis_rad: float | None = None,
    grid_size: int = 45,
) -> dict[str, Any] | None:
    """Return the safest object centre/yaw in ``opening_polygon_xy``.

    This is a sampled Minkowski difference ``opening ⊖ footprint``.  The
    opening produced by the RGB-D cavity detector is convex, so requiring all
    four robust OBB vertices to remain inside is sufficient for containment.
    Signed point-to-boundary distance supplies both the uncertainty erosion
    and the max-clearance objective.
    """
    import cv2

    polygon = np.asarray(opening_polygon_xy, dtype=float).reshape(-1, 2)
    polygon = polygon[np.isfinite(polygon).all(axis=1)]
    if polygon.shape[0] < 3:
        return None
    contour = cv2.convexHull(polygon.astype("float32")).reshape(-1, 1, 2)
    uncertainty = float(np.clip(float(uncertainty_m), 0.0, 0.025))
    preferred = (
        np.asarray(preferred_xy, dtype=float).reshape(2)
        if preferred_xy is not None
        else polygon.mean(axis=0)
    )

    opening_axis = _polygon_axis(polygon)
    # Rectangle containment is pi-periodic.  Include exact opening alignment,
    # the perpendicular presentation, the current presentation, and a regular
    # angular sweep.  This generalises to skewed/irregular convex openings
    # without a per-container orientation rule.
    if preferred_object_axis_rad is not None:
        # Front-loaded shelves need the carried object's long axis to follow
        # the aperture depth so that an edge grasp can keep the wrist outside.
        # Still run the full containment test; this is a kinematic constraint,
        # not permission to accept a colliding placement.
        desired_axes = [float(preferred_object_axis_rad)]
    else:
        desired_axes = [
            opening_axis,
            opening_axis + 0.5 * np.pi,
            footprint.long_axis_rad,
        ]
        desired_axes.extend(np.linspace(-0.5 * np.pi, 0.5 * np.pi, 13)[:-1])
    rotations: list[float] = []
    for desired in desired_axes:
        delta = _wrap_half_turn(float(desired) - footprint.long_axis_rad)
        if not any(abs(_wrap_half_turn(delta - old)) < 1e-4 for old in rotations):
            rotations.append(delta)

    lo = polygon.min(axis=0)
    hi = polygon.max(axis=0)
    count = int(np.clip(grid_size, 15, 81))
    xs = np.linspace(float(lo[0]), float(hi[0]), count)
    ys = np.linspace(float(lo[1]), float(hi[1]), count)
    best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    feasible_count = 0
    for delta in rotations:
        rotated = footprint.vertices @ _rotation(delta).T
        # Include edge midpoints.  They are redundant for an exactly convex
        # opening but improve robustness to small contour discretisation.
        probes = np.concatenate(
            [
                rotated,
                0.5 * (rotated + np.roll(rotated, -1, axis=0)),
            ],
            axis=0,
        )
        for x in xs:
            for y in ys:
                centre = np.asarray([x, y], dtype=float)
                distances = [
                    float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))
                    for point in probes + centre
                ]
                clearance = min(distances)
                if clearance + 1e-9 < uncertainty:
                    continue
                feasible_count += 1
                proximity = float(np.linalg.norm(centre - preferred))
                # Clearance remains primary near a tight fit, but wrist yaw is
                # not free: a large in-hand rotation can make a marginal grasp
                # slip long before it reaches the opening.  Charge a small,
                # metric-equivalent manipulation cost (6 mm per radian).  A
                # rotation that is required for feasibility still wins; in a
                # roomy drawer or box, a comparable no-rotation placement is
                # preferred over gratuitously turning the carried object.
                rotation_cost = 0.006 * abs(float(delta))
                score = clearance - rotation_cost - 0.02 * proximity
                candidate = (score, clearance, -proximity, centre, rotated)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
    if best is None:
        return None
    _, clearance, _, centre, rotated = best
    delta = None
    # Recover the chosen rotation by matching the rotated vertices.  This is
    # deterministic and avoids storing numpy arrays in tuple comparisons.
    for candidate_delta in rotations:
        if np.allclose(
            footprint.vertices @ _rotation(candidate_delta).T,
            rotated,
            atol=1e-8,
        ):
            delta = candidate_delta
            break
    if delta is None:
        return None
    return {
        "centre_xy": np.asarray(centre, dtype=float),
        "rotation_rad": float(delta),
        "object_axis_rad": _wrap_half_turn(footprint.long_axis_rad + float(delta)),
        "opening_axis_rad": opening_axis,
        "clearance_m": float(clearance),
        "uncertainty_m": uncertainty,
        "feasible_samples": int(feasible_count),
        "rotated_vertices": np.asarray(rotated, dtype=float),
    }


def _broad_container_transport_is_sufficient(
    place_label: str,
    footprint: Footprint,
    plan: dict[str, Any],
) -> bool:
    """Whether a roomy top-open receptacle does not need guided insertion.

    Insert exists for tight configuration-space problems.  A roughly compact,
    bulky footprint with centimetres of clearance in a drawer/bin is the
    opposite case: pickplace's high transit and gravity-assisted release are
    more robust than descending toward an uncertain RGB-D cavity floor.  Thin
    or elongated objects remain insertions even when their centre has room,
    because their presentation and wall clearance still dominate.
    """
    lowered = " ".join(str(place_label).lower().split())
    if not re.search(r"\b(?:drawer|basket|tray|bin|box|container)\b", lowered):
        return False
    long_side, short_side = (float(v) for v in footprint.size)
    aspect = long_side / max(short_side, 1e-6)
    clearance = float(plan.get("clearance_m", 0.0))
    return bool(short_side >= 0.045 and aspect <= 2.2 and clearance >= 0.012)


def _frontload_geometry(
    opening_polygon_xy: np.ndarray,
    footprint: Footprint,
    image_down_xy: np.ndarray,
) -> dict[str, Any] | None:
    """Camera-grounded approach geometry for a front-open shelf or rack.

    The short horizontal axis of a shelf floor is its depth axis. Its sign is
    ambiguous.  The source object is normally in the free workspace in front
    of the opening, so its direction from the cavity centre is the strongest
    scene-specific sign cue. Image-down is only a fallback when the source is
    too close to, or nearly sideways from, the depth axis.  The object is
    grasped near that outward end: after yawing its long axis into the shelf,
    most of the object can cross the opening while the wrist stays outside.
    """
    polygon = np.asarray(opening_polygon_xy, dtype=float).reshape(-1, 2)
    polygon = polygon[np.isfinite(polygon).all(axis=1)]
    down = np.asarray(image_down_xy, dtype=float).reshape(2)
    if polygon.shape[0] < 3 or float(np.linalg.norm(down)) < 1e-6:
        return None
    down /= float(np.linalg.norm(down))
    long_angle = _polygon_axis(polygon)
    depth = np.asarray(
        [
            np.cos(long_angle + 0.5 * np.pi),
            np.sin(long_angle + 0.5 * np.pi),
        ]
    )
    opening_centre = np.median(polygon, axis=0)
    toward_source = np.asarray(footprint.centre, dtype=float) - opening_centre
    source_projection = float(depth @ toward_source)
    # A centimetre of signed depth separation is enough to beat the camera
    # prior.  Below that, the source may lie alongside the shelf and image
    # geometry remains the less arbitrary cue.
    if abs(source_projection) >= 0.01:
        outward = depth if source_projection >= 0.0 else -depth
        outward_source = "source_free_space"
    else:
        outward = depth if float(depth @ down) >= 0.0 else -depth
        outward_source = "camera_image_down"
    inward = -outward
    depth_projection = polygon @ inward
    depth_span = float(np.ptp(depth_projection))
    if not 0.025 <= depth_span <= 0.45:
        return None
    edge_offset = float(np.clip(0.5 * footprint.size[0] - 0.012, 0.012, 0.055))
    return {
        "outward_axis": outward,
        "inward_axis": inward,
        "object_axis_rad": _wrap_half_turn(float(np.arctan2(inward[1], inward[0]))),
        "edge_offset_m": edge_offset,
        "opening_depth_m": depth_span,
        "outward_source": outward_source,
    }


def _footprint_clearance(
    opening_polygon_xy: np.ndarray,
    footprint: Footprint,
) -> float:
    """Signed clearance of a measured footprint from an opening boundary."""
    import cv2

    polygon = np.asarray(opening_polygon_xy, dtype=float).reshape(-1, 2)
    contour = cv2.convexHull(polygon.astype("float32")).reshape(-1, 1, 2)
    vertices = footprint.vertices + footprint.centre
    probes = np.concatenate(
        [
            vertices,
            0.5 * (vertices + np.roll(vertices, -1, axis=0)),
        ],
        axis=0,
    )
    return min(
        float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))
        for point in probes
    )


def _measure_held_footprint(
    runtime,
    label: str,
    reference: Footprint,
    expected_xy: np.ndarray,
) -> tuple[Footprint | None, str]:
    """Re-measure a carried object using SAM3 and RGB-D.

    This is intentionally independent of the VLM box that selected the object
    before grasping.  Strict geometric gates keep a stale source detection or
    a mask that swallowed the gripper from steering the arm.
    """
    tracker = getattr(runtime, "track_near_hand", None)
    appearance = _appearance_label(label)
    try:
        target = (
            tracker(
                appearance,
                reference_size=reference.size,
                radius_m=0.18,
            )
            if callable(tracker)
            else _ground_visible(runtime, label)
        )
    except PerceptionUnavailableError:
        raise
    except Exception as exc:
        return None, f"grounding_error:{type(exc).__name__}"
    if not _plausible(target):
        return None, "held_object_not_grounded"
    cloud = _cloud(runtime, appearance if callable(tracker) else label)
    candidates: list[Footprint] = []
    measured = _robust_footprint(cloud)
    if measured is not None:
        candidates.append(measured)
    # A top-down grasp puts the fingers above the carried body.  Fit several
    # lower-z slices and select the component whose dimensions best agree with
    # the clean pre-grasp shape.  This removes gripper points instead of
    # weakening the yaw-confidence gate on a nearly square merged mask.
    if cloud.shape[0] >= 2 * MIN_FOOTPRINT_POINTS:
        for percentile in (50, 60, 70, 80):
            cutoff = float(np.percentile(cloud[:, 2], percentile))
            body = cloud[cloud[:, 2] <= cutoff]
            candidate = _robust_footprint(body)
            if candidate is not None:
                candidates.append(candidate)
    if candidates:

        def shape_error(candidate: Footprint) -> float:
            ratios = np.asarray(candidate.size) / np.asarray(reference.size)
            return float(np.sum(np.abs(np.log(np.clip(ratios, 0.05, 20.0)))))

        measured = min(candidates, key=shape_error)
    else:
        measured = None
    if measured is None:
        return None, "held_footprint_unavailable"
    # The fingers commonly become part of the text mask and inflate the short
    # dimension.  They do not change the object's physical shape.  Fuse the
    # stable pre-grasp dimensions with only the held centre and long-axis pose;
    # the observed long edge still has to agree and remain anisotropic.
    long_ratio = float(measured.size[0] / reference.size[0])
    aspect = float(measured.size[0] / max(measured.size[1], 1e-6))
    if not 0.65 <= long_ratio <= 1.45 or aspect < 1.20:
        measured_cm = ",".join(f"{value * 100:.1f}" for value in measured.size)
        reference_cm = ",".join(f"{value * 100:.1f}" for value in reference.size)
        return None, (
            f"held_footprint_size_mismatch:measured_cm={measured_cm};reference_cm={reference_cm}"
        )
    if float(np.linalg.norm(measured.centre - expected_xy)) > 0.12:
        return None, "held_object_position_mismatch"
    pose_delta = _wrap_half_turn(measured.long_axis_rad - reference.long_axis_rad)
    fused = Footprint(
        centre=measured.centre,
        vertices=reference.vertices @ _rotation(pose_delta).T,
        long_axis_rad=measured.long_axis_rad,
        size=reference.size,
    )
    measured_cm = ",".join(f"{value * 100:.1f}" for value in measured.size)
    return fused, f"shape_prior_with_held_pose:observed_cm={measured_cm}"


def _visual_servo_held_pose(
    runtime,
    label: str,
    reference: Footprint,
    polygon: np.ndarray,
    preferred_xy: np.ndarray,
    hand_goal: np.ndarray,
    hand_yaw: float,
    travel_z: float,
    uncertainty: float,
) -> tuple[np.ndarray, float, np.ndarray | None, dict[str, Any]]:
    """Estimate the wrist-to-object yaw response and align above the opening.

    A grasped object does not necessarily rotate one-for-one with the wrist:
    it may slip between the fingers or start with an unknown jaw-relative yaw.
    A small, collision-free probe at transit height identifies the local yaw
    response from RGB-D.  Subsequent bounded steps close the measured object
    yaw error; the final XY correction uses the newly observed body centre.
    """
    info: dict[str, Any] = {}
    current, reason = _measure_held_footprint(
        runtime, label, reference, np.asarray(hand_goal, dtype=float)
    )
    info["held_visual_measurement"] = reason
    info["held_visual_feedback"] = reason
    if current is None:
        return hand_goal, hand_yaw, None, info

    info.update(
        {
            "held_footprint_size": [round(float(v), 4) for v in current.size],
            "held_object_yaw_deg": round(float(np.degrees(current.long_axis_rad)), 1),
            "held_clearance_before_cm": round(_footprint_clearance(polygon, current) * 100, 2),
        }
    )
    original_hand = np.asarray(hand_goal, dtype=float)
    original_yaw = float(hand_yaw)
    current_hand = original_hand.copy()
    current_yaw = original_yaw
    response_gain: float | None = None
    history: list[dict[str, Any]] = []
    latest_plan: dict[str, Any] | None = None

    for index in range(MAX_VISUAL_SERVO_STEPS):
        latest_plan = feasible_placement(
            polygon,
            current,
            uncertainty_m=uncertainty,
            preferred_xy=preferred_xy,
        )
        if latest_plan is None:
            info["held_visual_feedback"] = "no_feasible_held_footprint"
            break
        object_error = float(latest_plan["rotation_rad"])
        if abs(object_error) <= np.deg2rad(5.0):
            break
        if response_gain is None:
            hand_step = VISUAL_SERVO_PROBE_RAD
        else:
            hand_step = float(
                np.clip(
                    object_error / response_gain,
                    -MAX_VISUAL_SERVO_STEP_RAD,
                    MAX_VISUAL_SERVO_STEP_RAD,
                )
            )
        previous_axis = current.long_axis_rad
        current_yaw += hand_step
        runtime.move_to([current_hand[0], current_hand[1], travel_z], yaw=current_yaw)
        measured, measured_reason = _measure_held_footprint(
            runtime, label, reference, current.centre
        )
        row: dict[str, Any] = {
            "iteration": index + 1,
            "hand_step_deg": round(float(np.degrees(hand_step)), 1),
            "object_error_before_deg": round(float(np.degrees(object_error)), 1),
            "measurement": measured_reason,
        }
        if measured is None:
            history.append(row)
            info["held_visual_feedback"] = "servo_measurement_lost"
            break
        object_step = _wrap_half_turn(measured.long_axis_rad - previous_axis)
        gain = float(object_step / hand_step) if abs(hand_step) > 1e-6 else 0.0
        row.update(
            {
                "object_step_deg": round(float(np.degrees(object_step)), 1),
                "yaw_response_gain": round(gain, 3),
            }
        )
        history.append(row)
        if 0.10 <= abs(gain) <= 3.0:
            response_gain = gain
        current = measured

    latest_plan = feasible_placement(
        polygon,
        current,
        uncertainty_m=uncertainty,
        preferred_xy=preferred_xy,
    )
    desired_xy: np.ndarray | None = None
    if latest_plan is not None:
        candidate_xy = np.asarray(latest_plan["centre_xy"], dtype=float)
        shift = candidate_xy - current.centre
        translation = float(np.linalg.norm(shift))
        residual_yaw = float(latest_plan["rotation_rad"])
        converged = bool(
            translation <= MAX_VISUAL_SERVO_TRANSLATION_M and abs(residual_yaw) <= np.deg2rad(10.0)
        )
        if converged:
            desired_xy = candidate_xy
            current_hand = current_hand + shift
            if translation > 0.001:
                runtime.move_to([current_hand[0], current_hand[1], travel_z], yaw=current_yaw)
            info.update(
                {
                    "held_visual_feedback": "corrected",
                    "visual_servo_translation_cm": round(translation * 100, 2),
                    "visual_servo_rotation_deg": round(
                        float(np.degrees(current_yaw - hand_yaw)), 1
                    ),
                    "visual_servo_residual_yaw_deg": round(float(np.degrees(residual_yaw)), 1),
                }
            )
        else:
            info["held_visual_feedback"] = (
                "translation_out_of_gate"
                if translation > MAX_VISUAL_SERVO_TRANSLATION_M
                else "yaw_residual_after_servo"
            )
            info["visual_servo_residual_yaw_deg"] = round(float(np.degrees(residual_yaw)), 1)
    if desired_xy is None and (
        not np.allclose(current_hand, original_hand) or abs(current_yaw - original_yaw) > 1e-6
    ):
        # A failed probe is diagnostic evidence, not a new open-loop command.
        # Restore the proven pre-grasp plan before descending so unavailable or
        # low-confidence held perception cannot make control worse.
        runtime.move_to([original_hand[0], original_hand[1], travel_z], yaw=original_yaw)
        current_hand = original_hand
        current_yaw = original_yaw
        info["held_visual_fallback"] = "restored_pregrasp_plan"
    info["visual_servo_history"] = history
    if response_gain is not None:
        info["visual_servo_yaw_response_gain"] = round(response_gain, 3)
    return current_hand, current_yaw, desired_xy, info


def _destination_report(goal: GroundedTarget, band: tuple[float, float]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "destination_pose": [round(float(v), 4) for v in goal.pose],
        "destination_confidence": round(float(goal.confidence), 4),
        "destination_rim_z": round(float(band[0]), 4),
        "destination_floor_z": round(float(band[1]), 4),
        "destination_kind": goal.kind,
        "destination_extent": [
            round(float(v), 4) for v in (goal.metadata.get("extent") or [0.0, 0.0, 0.0])
        ],
        "destination_synthetic": bool(goal.metadata.get("synthetic")),
    }
    if goal.metadata.get("bbox"):
        report["destination_bbox"] = [int(v) for v in goal.metadata["bbox"]]
    for source_key, report_key in (
        ("region_source", "destination_region_source"),
        ("region_polygon_xy", "destination_region_polygon_xy"),
        ("region_boundary_pixels", "destination_region_boundary_pixels"),
        ("region_safe_pixel", "destination_region_safe_pixel"),
        ("region_area_pixels", "destination_region_area_pixels"),
        ("region_clearance_m", "destination_region_clearance_m"),
        ("region_confidence", "destination_region_confidence"),
    ):
        if source_key in goal.metadata:
            report[report_key] = goal.metadata[source_key]
    return report


def insert(
    runtime,
    pick_label: str,
    place_label: str,
    *,
    grasp: str | list[str] | None = None,
    grasp_depth: float | None = None,
    uncertainty_m: float = DEFAULT_UNCERTAINTY_M,
    insertion_depth: float | None = DEFAULT_INSERTION_DEPTH_M,
    frontload: bool | None = None,
    destination_description: str | None = None,
    _destination_target: GroundedTarget | None = None,
    _pick_target: GroundedTarget | None = None,
) -> SkillResult:
    """Insert an object only after proving a collision-free planar fit.

    ``no_feasible_insertion`` and ``no_opening_geometry`` are preflight
    failures: the arm has not moved.  They deliberately tell the caller that a
    side/edge presentation is required instead of disguising an impossible
    top-down insertion as another centre-targeted pick-and-place attempt.
    """
    trace: list[dict[str, Any]] = []
    try:
        uncertainty = float(np.clip(float(uncertainty_m), 0.0, 0.025))
    except (TypeError, ValueError):
        uncertainty = DEFAULT_UNCERTAINTY_M
    requested_depth: float | None
    if insertion_depth is None:
        requested_depth = None
    else:
        try:
            requested_depth = float(np.clip(float(insertion_depth), 0.005, 0.12))
        except (TypeError, ValueError):
            requested_depth = None
    if grasp_depth is not None:
        try:
            grasp_depth = float(np.clip(float(grasp_depth), 0.005, 0.08))
        except (TypeError, ValueError):
            grasp_depth = None
    report: dict[str, Any] = {
        "params": {
            "grasp": grasp,
            "grasp_depth": grasp_depth,
            "uncertainty_m": uncertainty,
            "insertion_depth": requested_depth,
            "frontload": frontload,
            "destination_description": destination_description,
        },
        "preflight_only": True,
    }

    def finish(success: bool, mode: str = "", message: str = "", attempts: int = 1) -> SkillResult:
        return SkillResult(
            success=success,
            status="success" if success else "retryable_failure",
            pick_label=pick_label,
            place_label=place_label,
            attempts=attempts,
            strategy="footprint_insert",
            failure_mode=mode,
            message=message,
            trace=trace,
            report=report,
        )

    if _pick_target is not None and _plausible(_pick_target):
        pick = _pick_target
        identity_evidence = dict(
            pick.metadata.get("pregrasp_identity_evidence")
            or {
                "used": True,
                "matches": True,
                "answer": "episode source continuity lock",
                "view": "precontact_scene",
                "veto_enabled": False,
            }
        )
        grounding_attempts = 0
        report["pick_locked"] = True
    else:
        pick, identity_evidence, grounding_attempts = _ground_verified_pick(
            runtime,
            pick_label,
            trace,
            max_attempts=4,
            max_planar_extent=MAX_SINGLE_GRASP_FOOTPRINT_M,
        )
    if pick is None:
        report.update(
            {
                "grounding_failure": "pick",
                "failed_label": pick_label,
                "grounding_trace": list(trace),
            }
        )
        return finish(
            False,
            "not_grounded",
            f"no usable grounding for {pick_label!r}",
            attempts=grounding_attempts,
        )
    report["grounding_attempts"] = grounding_attempts
    report["pregrasp_identity_evidence"] = identity_evidence
    report["grounding_trace"] = list(trace)

    # The moved body's configuration space is part of destination grounding,
    # not merely a downstream motion-planning detail. A semantically plausible
    # box can yield a truncated cavity (too narrow) or a depth component that
    # leaks through several adjacent openings (too broad). Let those geometric
    # contradictions veto the candidate and feed its box back to the VLM while
    # the scene is still untouched.
    points = _cloud(runtime, pick_label)
    footprint = _robust_footprint(points)
    if footprint is None:
        report["recommended_recovery"] = "re-ground object or use pickplace"
        return finish(
            False,
            "no_object_footprint",
            "object footprint is too sparse or implausible; arm not moved",
        )
    if footprint.size[0] > MAX_SINGLE_GRASP_FOOTPRINT_M:
        report.update(
            {
                "object_footprint_size": [round(float(v), 4) for v in footprint.size],
                "recommended_recovery": (
                    "re-ground source from an alternate prompt or camera; "
                    "measured body exceeds single-grasp workspace"
                ),
            }
        )
        return finish(
            False,
            "implausible_object_footprint",
            "source mask is too large for a one-handed object; arm not moved",
        )

    def validate_opening(candidate: GroundedTarget) -> tuple[bool, str]:
        candidate_polygon = np.asarray(
            candidate.metadata.get("region_polygon_xy") or [], dtype=float
        ).reshape(-1, 2)
        if candidate_polygon.shape[0] < 3:
            return False, "no_opening_polygon"
        candidate_plan = feasible_placement(
            candidate_polygon,
            footprint,
            uncertainty_m=uncertainty,
            preferred_xy=np.asarray(candidate.pose[:2], dtype=float),
        )
        if candidate_plan is None:
            return False, "object_footprint_has_no_feasible_configuration"
        return True, (f"feasible_clearance_cm={float(candidate_plan['clearance_m']) * 100:.2f}")

    goal = None
    if _destination_target is not None and _plausible(_destination_target):
        goal = _destination_target
        report["destination_locked"] = True
    if goal is None:
        grounding_label = place_label
        semantic_destination = str(destination_description or "").strip()
        if bool(frontload) and semantic_destination:
            # ``under/below the shelf`` names a compartment relation, not a
            # fixture whose literal name is "under the shelf".  Retain the
            # relation in ``vertical_detail`` and give the visual grounder the
            # underlying parent fixture. This avoids self-contradictory prompts
            # such as "within under the cabinet shelf" and generalises to any
            # front-open fixture named by a vertical relation.
            fixture_phrase = (
                re.sub(
                    r"^(?:under|below|beneath|lower|bottom(?:most)?)\s+(?:the\s+)?",
                    "",
                    semantic_destination,
                    flags=re.IGNORECASE,
                ).strip()
                or semantic_destination
            )
            tier = (
                "lower"
                if any(
                    word in semantic_destination.lower() for word in ("under", "lower", "bottom")
                )
                else "upper"
            )
            vertical_detail = (
                "topmost open shelf compartment above the horizontal divider; "
                "do not select the lower compartment"
                if tier == "upper"
                else "bottommost open shelf compartment below the horizontal divider; "
                "do not select the upper compartment"
            )
            grounding_label = (
                f"empty usable interior of the {vertical_detail}, within "
                f"{fixture_phrase}, the foreground structure referred to as "
                f"{fixture_phrase}"
            )
        goal = _ground_destination(
            runtime,
            grounding_label,
            trace,
            near=pick,
            mode="inside",
            inside_validator=validate_opening,
        )
    if goal is None:
        report.update(
            {
                "grounding_failure": "destination",
                "failed_label": place_label,
                "grounding_trace": list(trace),
            }
        )
        return finish(False, "not_grounded", f"no usable opening grounding for {place_label!r}")
    band = _release_band(runtime, goal)
    report.update(_destination_report(goal, band))

    polygon = np.asarray(goal.metadata.get("region_polygon_xy") or [], dtype=float).reshape(-1, 2)
    if polygon.shape[0] < 3:
        report["recommended_recovery"] = "use pickplace; no dense opening polygon"
        return finish(
            False,
            "no_opening_geometry",
            "destination has no reliable RGB-D opening polygon; arm not moved",
        )

    lowered_place = " ".join(str(place_label).lower().split())
    frontload_requested = (
        bool(frontload)
        if frontload is not None
        else bool(
            any(word in lowered_place for word in ("shelf", "rack"))
            and "top of" not in lowered_place
        )
    )
    frontload = None
    if frontload_requested:
        try:
            frontload = _frontload_geometry(
                polygon,
                footprint,
                np.asarray(runtime.image_axes()["down"], dtype=float),
            )
        except Exception:
            frontload = None
    plan = feasible_placement(
        polygon,
        footprint,
        uncertainty_m=uncertainty,
        preferred_xy=np.asarray(goal.pose[:2], dtype=float),
        preferred_object_axis_rad=(None if frontload is None else frontload["object_axis_rad"]),
    )
    report.update(
        {
            "object_footprint_centre": [round(float(v), 4) for v in footprint.centre],
            "object_footprint_size": [round(float(v), 4) for v in footprint.size],
            "object_long_axis_deg": round(float(np.degrees(footprint.long_axis_rad)), 1),
            "opening_polygon_vertices": int(polygon.shape[0]),
        }
    )
    if plan is None:
        report["recommended_recovery"] = (
            "side_or_edge_grasp_required; top-down footprint has no feasible centre"
        )
        trace.append(
            {
                "event": "insert_preflight",
                "feasible": False,
                "uncertainty_cm": round(uncertainty * 100, 1),
            }
        )
        return finish(
            False,
            "no_feasible_insertion",
            "the top-down footprint cannot fit inside the opening; arm not moved",
        )

    desired_object_xy = np.asarray(plan["centre_xy"], dtype=float)
    rotation = float(plan["rotation_rad"])
    report.update(
        {
            "feasible_centre_xy": [round(float(v), 4) for v in desired_object_xy],
            "feasible_clearance_cm": round(float(plan["clearance_m"]) * 100, 2),
            "feasible_samples": int(plan["feasible_samples"]),
            "planned_object_yaw_deg": round(float(np.degrees(plan["object_axis_rad"])), 1),
            "planned_rotation_deg": round(float(np.degrees(rotation)), 1),
            "insertion_mode": "frontload" if frontload is not None else "vertical",
        }
    )
    if frontload is None and _broad_container_transport_is_sufficient(
        place_label,
        footprint,
        plan,
    ):
        report.update(
            {
                "recommended_recovery": "use_pickplace_for_roomy_container",
                "fit_aspect_ratio": round(
                    float(footprint.size[0] / max(footprint.size[1], 1e-6)), 2
                ),
            }
        )
        trace.append(
            {
                "event": "insert_preflight",
                "feasible": True,
                "motion_family": "broad_container_pickplace",
                "clearance_cm": report["feasible_clearance_cm"],
            }
        )
        return finish(
            False,
            "broad_container_fit",
            "opening is roomy for this compact footprint; use high-transit "
            "pickplace instead of cavity-floor insertion",
        )
    if frontload is not None:
        report["frontload_geometry"] = {
            "outward_axis": [round(float(v), 4) for v in frontload["outward_axis"]],
            "opening_depth_cm": round(float(frontload["opening_depth_m"]) * 100, 1),
            "edge_grasp_offset_cm": round(float(frontload["edge_offset_m"]) * 100, 1),
            "outward_source": frontload.get("outward_source", "unknown"),
        }
    trace.append(
        {
            "event": "insert_preflight",
            "feasible": True,
            "clearance_cm": report["feasible_clearance_cm"],
            "centre_xy": report["feasible_centre_xy"],
            "rotation_deg": report["planned_rotation_deg"],
        }
    )

    table_z = _surface_height(runtime, pick)
    top_z = float(pick.metadata.get("top_z", pick.pose[2]))
    heights = {
        "top": _grasp_height(pick, table_z, depth=grasp_depth),
        "low": table_z + float(np.clip(0.20 * max(top_z - table_z, 0.0), 0.025, 0.06)),
        "edge": _edge_grasp_height(table_z, top_z),
    }
    # Keep the presentation model explicit.  Rim/affordance contacts shift the
    # object relative to the hand in ways a pre-action footprint cannot infer.
    defaults = _attempts(points, object_height=max(top_z - table_z, 0.0))
    attempts = _ladder(grasp, defaults) if grasp is not None else defaults
    attempts = tuple(
        row
        for row in attempts
        if row[0]
        in {
            "top_down",
            "pca_axis",
            "scoop_edge",
            "tilted_edge",
            "tilted_edge_reverse",
            "graspnet",
        }
    )
    report["pick_pose"] = [round(float(v), 4) for v in pick.pose]
    report["pick_confidence"] = round(float(pick.confidence), 4)
    report["pick_n_points"] = int(pick.metadata.get("n_points", 0))
    report["pick_top_z"] = round(float(pick.metadata.get("top_z", pick.pose[2])), 4)
    if pick.metadata.get("principal_axis") is not None:
        report["pick_principal_axis"] = pick.metadata.get("principal_axis")
    if pick.metadata.get("bbox") is not None:
        report["pick_bbox"] = [int(v) for v in pick.metadata["bbox"]]
    report["pick_extent"] = [
        round(float(v), 4) for v in (pick.metadata.get("extent") or [0.0, 0.0, 0.0])
    ]
    report["ladder"] = [f"{strategy}@{where}" for strategy, where in attempts]

    def edge_target(target: GroundedTarget, body: Footprint) -> GroundedTarget:
        """Return a contact near one long edge without moving every grasp.

        An earlier implementation shifted the target for *all* front-loaded
        grasps.  That silently turned even ``top_down@top`` and ``pca_axis``
        into off-centre contacts, levering upright thin objects onto the table.
        Edge contacts are useful, but only for the strategies that explicitly
        ask to scoop or pinch an edge.
        """
        current_long = np.asarray(
            [
                np.cos(body.long_axis_rad),
                np.sin(body.long_axis_rad),
            ]
        )
        edge_xy = body.centre - current_long * float(frontload["edge_offset_m"])
        return GroundedTarget(
            label=target.label,
            pose=(float(edge_xy[0]), float(edge_xy[1]), float(target.pose[2])),
            kind=target.kind,
            confidence=target.confidence,
            metadata=dict(target.metadata),
        )

    grasp_target = pick
    if frontload is not None:
        first_edge = edge_target(pick, footprint)
        edge_xy = np.asarray(first_edge.pose[:2], dtype=float)
        report["edge_grasp_xy"] = [round(float(v), 4) for v in edge_xy]

    held = False
    used = "top_down"
    where = "top"
    height = heights["top"]
    pick_identity_anchor_xy = np.asarray(pick.pose[:2], dtype=float).copy()
    # From this point onward a failed contact can still change the scene.
    # Calling it "preflight only; scene unchanged" made the agent retry from a
    # false premise and allowed relational labels to bind to a neighbour.
    report["preflight_only"] = False

    def refresh_after_failed_contact() -> bool:
        """Reacquire the same local referent after a non-retained contact."""
        nonlocal pick, footprint, table_z, top_z, heights
        try:
            runtime.open_gripper()
            runtime.recover()
            refreshed, refresh_identity = _refresh_pick_after_contact(
                runtime, pick_label, pick, trace
            )
        except Exception as exc:
            trace.append(
                {
                    "event": "grasp_reground",
                    "accepted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return False
        step_shift = (
            float(
                np.linalg.norm(
                    np.asarray(refreshed.pose[:2], dtype=float)
                    - np.asarray(pick.pose[:2], dtype=float)
                )
            )
            if _plausible(refreshed)
            else float("inf")
        )
        anchor_shift = (
            float(
                np.linalg.norm(
                    np.asarray(refreshed.pose[:2], dtype=float) - pick_identity_anchor_xy
                )
            )
            if _plausible(refreshed)
            else float("inf")
        )
        accepted = bool(
            _plausible(refreshed)
            and step_shift <= MAX_GRASP_REGROUND_STEP_M
            and anchor_shift <= MAX_GRASP_REGROUND_ANCHOR_M
        )
        if not accepted:
            trace.append(
                {
                    "event": "grasp_reground",
                    "accepted": False,
                    "shift_cm": None if not np.isfinite(step_shift) else round(step_shift * 100, 1),
                    "anchor_shift_cm": None
                    if not np.isfinite(anchor_shift)
                    else round(anchor_shift * 100, 1),
                }
            )
            return False

        refreshed_points = _cloud(runtime, str(refreshed.metadata.get("points_label", pick_label)))
        refreshed_footprint = _robust_footprint(refreshed_points)
        if refreshed_footprint is not None:
            footprint = refreshed_footprint
        pick = refreshed
        table_z = _surface_height(runtime, pick)
        top_z, raw_top_z = _robust_object_top(pick, refreshed_points, table_z)
        heights = {
            "top": _grasp_height(pick, table_z, depth=grasp_depth, top_z=top_z),
            "low": table_z + float(np.clip(0.20 * max(top_z - table_z, 0.0), 0.025, 0.06)),
            "edge": _edge_grasp_height(table_z, top_z),
        }
        report["pick_pose"] = [round(float(v), 4) for v in pick.pose]
        report["pick_confidence"] = round(float(pick.confidence), 4)
        report["pick_n_points"] = int(pick.metadata.get("n_points", 0))
        report["pick_top_z"] = round(float(top_z), 4)
        if pick.metadata.get("principal_axis") is not None:
            report["pick_principal_axis"] = pick.metadata.get("principal_axis")
        if pick.metadata.get("bbox") is not None:
            report["pick_bbox"] = [int(v) for v in pick.metadata["bbox"]]
        report["pick_extent"] = [
            round(float(v), 4) for v in (pick.metadata.get("extent") or [0.0, 0.0, 0.0])
        ]
        report.setdefault("grasp_regroundings", []).append(
            {
                "shift_cm": round(step_shift * 100, 1),
                "anchor_shift_cm": round(anchor_shift * 100, 1),
                "pose": report["pick_pose"],
                "object_top_z": round(top_z, 4),
                "object_top_raw_z": round(raw_top_z, 4),
                "identity": refresh_identity,
            }
        )
        trace.append(
            {
                "event": "grasp_reground",
                "accepted": True,
                "shift_cm": round(step_shift * 100, 1),
                "anchor_shift_cm": round(anchor_shift * 100, 1),
            }
        )
        return True

    for index, (used, where) in enumerate(attempts):
        height = heights[where]
        use_edge = bool(
            frontload is not None
            and (
                where == "edge"
                or used
                in {
                    "scoop_edge",
                    "tilted_edge",
                    "tilted_edge_reverse",
                }
            )
        )
        grasp_target = edge_target(pick, footprint) if use_edge else pick
        try:
            runtime.open_gripper()
            runtime.grasp(_lowered(grasp_target, height), strategy=used)
        except Exception as exc:
            trace.append(
                {
                    "event": "grasp_attempt",
                    "strategy": used,
                    "at": where,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not runtime.verify_grasp(pick_label):
            trace.append(
                {
                    "event": "grasp_attempt",
                    "strategy": used,
                    "at": where,
                    "held": False,
                    "reason": "empty_close",
                }
            )
            # A non-retained contact often tips or slides a thin object.  The
            # next materially different grasp must use the new visual pose,
            # not the pre-contact coordinate.  Re-ground after clearing the
            # camera and admit only a local continuation of the original
            # referent, so a neighbouring look-alike cannot hijack the retry.
            refresh_after_failed_contact()
            continue
        if not _held_object_left_the_table(runtime, pick_label, pick):
            trace.append(
                {
                    "event": "grasp_attempt",
                    "strategy": used,
                    "at": where,
                    "held": False,
                    "reason": "false_hold_object_not_lifted",
                }
            )
            refresh_after_failed_contact()
            continue
        held = True
        break
    if not held:
        report["force_empty_after_contact"] = True
        return finish(
            False,
            "empty_grasp",
            "no geometry-preserving grasp lifted the object",
            attempts=len(attempts),
        )

    report["preflight_only"] = False
    report["grasp_strategy"] = f"{used}@{where}"
    carry = float(np.clip(height - table_z, MIN_CLEARANCE, 0.30))
    report["carry_below_hand"] = round(carry, 4)

    # The gripper closed at pick.pose, while the robust body centre may differ.
    # Rotate that measured relation with the commanded wrist yaw so the body,
    # rather than the hand, arrives at the feasible centre.
    object_from_hand = footprint.centre - np.asarray(grasp_target.pose[:2], dtype=float)
    max_carry_offset = 0.065 if frontload is not None else 0.04
    if float(np.linalg.norm(object_from_hand)) > max_carry_offset:
        object_from_hand = np.zeros(2)
        report["carry_offset_reliable"] = False
    else:
        report["carry_offset_reliable"] = True
    rotated_offset = object_from_hand @ _rotation(rotation).T
    hand_goal = desired_object_xy - rotated_offset
    report["carry_offset"] = [round(float(v), 4) for v in rotated_offset]
    report["hand_goal_xy"] = [round(float(v), 4) for v in hand_goal]

    # Absolute gripper yaw depends on the grasp strategy.  top_down starts at
    # zero; pca_axis starts perpendicular to the object's long axis.
    grasp_yaw = footprint.long_axis_rad + 0.5 * np.pi if used == "pca_axis" else 0.0
    hand_yaw = float(grasp_yaw + rotation)
    report["place_yaw_deg"] = round(float(np.degrees(hand_yaw)), 1)

    rim, floor = (float(band[0]), float(band[1]))
    cavity_depth = max(rim - floor, 0.0)
    # With an upright/thin object, releasing while its lower edge is still in
    # free space turns the residual drop into a tip-over impulse.  The default
    # is therefore contact-style insertion: descend to a small margin above
    # the measured cavity floor.  An explicit depth remains available for
    # fragile objects or uncertain floors.
    contact_depth = max(0.0, cavity_depth - FLOOR_CONTACT_MARGIN_M)
    wanted = contact_depth if requested_depth is None else min(requested_depth, contact_depth)
    if frontload is not None:
        # Enter at the shelf floor height instead of descending through the
        # shelf above.  The edge grasp keeps the wrist toward the open/front
        # side while the object centre follows the measured depth axis.
        release_z = max(
            floor + carry + FLOOR_CONTACT_MARGIN_M,
            floor + MIN_HAND_CLEARANCE,
        )
        preinsert_z = release_z
        travel_z = max(release_z + TRANSIT_CLEARANCE, rim + 0.10)
        achieved = min(
            float(frontload["opening_depth_m"]),
            float(frontload["edge_offset_m"] + 0.5 * footprint.size[0]),
        )
    else:
        # If the object has more than one finger-width of planar clearance,
        # the closed fingers can safely follow it slightly below the rim. Use
        # only the excess clearance as penetration allowance; a tight fit
        # retains the conservative above-rim hand limit. This turns a short
        # gravity drop into contact seating without assuming a container or
        # object category.
        feasible_clearance = float(plan["clearance_m"])
        controlled_rim_penetration = float(
            np.clip(
                feasible_clearance - INSERT_FINGER_HALF_WIDTH_M,
                0.0,
                MAX_CONTROLLED_RIM_PENETRATION_M,
            )
        )
        hand_rim_limit = rim + RIM_HAND_CLEARANCE_M - controlled_rim_penetration
        release_z = max(
            rim + carry - wanted,
            hand_rim_limit,
            floor + MIN_HAND_CLEARANCE,
        )
        achieved = max(0.0, rim - (release_z - carry))
        preinsert_z = max(rim + carry + 0.03, release_z + 0.03)
        travel_z = max(preinsert_z + TRANSIT_CLEARANCE, rim + carry + 0.12)
    report.update(
        {
            "requested_insertion_depth_cm": (
                "auto_floor_contact" if requested_depth is None else round(requested_depth * 100, 1)
            ),
            "achieved_insertion_depth_cm": round(achieved * 100, 1),
            "release_z": round(release_z, 4),
            "preinsert_z": round(preinsert_z, 4),
            "controlled_rim_penetration_cm": round(
                (0.0 if frontload is not None else controlled_rim_penetration) * 100, 2
            ),
        }
    )
    trace.append(
        {
            "event": "insert_plan",
            "hand_goal_xy": report["hand_goal_xy"],
            "travel_z": round(travel_z, 4),
            "preinsert_z": report["preinsert_z"],
            "release_z": report["release_z"],
            "achieved_depth_cm": report["achieved_insertion_depth_cm"],
        }
    )

    try:
        if frontload is not None:
            outward = np.asarray(frontload["outward_axis"], dtype=float)
            approach_xy = hand_goal + outward * max(0.06, 0.5 * float(frontload["opening_depth_m"]))
            report["frontload_approach_xy"] = [round(float(v), 4) for v in approach_xy]
            runtime.move_to([approach_xy[0], approach_xy[1], travel_z], yaw=hand_yaw)
            runtime.move_to([approach_xy[0], approach_xy[1], release_z])
            runtime.move_to([hand_goal[0], hand_goal[1], release_z])
            runtime.open_gripper()
            runtime.move_to([approach_xy[0], approach_xy[1], release_z])
            runtime.move_to([approach_xy[0], approach_xy[1], travel_z])
            runtime.recover()
        else:
            runtime.move_to([hand_goal[0], hand_goal[1], travel_z], yaw=hand_yaw)

            # Re-estimate the footprint after grasp and wrist rotation.  The
            # pre-grasp cloud cannot observe jaw-induced yaw error, slip, or
            # the true object-to-hand offset. A reliable held measurement
            # replans the same configuration-space problem while still above
            # the opening.
            hand_goal, hand_yaw, corrected_object_xy, visual_info = _visual_servo_held_pose(
                runtime,
                pick_label,
                footprint,
                polygon,
                np.asarray(goal.pose[:2], dtype=float),
                hand_goal,
                hand_yaw,
                travel_z,
                uncertainty,
            )
            report.update(visual_info)
            if corrected_object_xy is not None:
                desired_object_xy = corrected_object_xy
            report["hand_goal_xy"] = [round(float(v), 4) for v in hand_goal]

            runtime.move_to([hand_goal[0], hand_goal[1], preinsert_z])
            runtime.move_to([hand_goal[0], hand_goal[1], release_z])
            runtime.open_gripper()
            runtime.move_to([hand_goal[0], hand_goal[1], preinsert_z])
            runtime.move_to([hand_goal[0], hand_goal[1], travel_z])
            runtime.recover()
    except Exception as exc:
        try:
            runtime.recover()
        except Exception:
            pass
        return finish(
            False, "insertion_failed", f"insert trajectory raised {type(exc).__name__}: {exc}"
        )

    report["release_xy"] = [round(float(v), 4) for v in desired_object_xy]
    try:
        placed = bool(runtime.verify_place(pick_label, place_label))
    except Exception:
        placed = False
    trace.append({"event": "inserted", "verified": placed})
    if not placed:
        return finish(
            False,
            "verification_inconclusive",
            "guided insertion completed but placement was not confirmed",
        )
    return finish(True)
