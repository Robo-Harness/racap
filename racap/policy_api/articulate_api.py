"""Visually grounded articulation for drawers and hinged doors.

The public goal is semantic (``open`` / ``closed``); the controller below
reduces it to one of two mechanism-level constraints.  Sliding parts move
along the outward handle-to-fixture axis.  Hinged parts follow an arc about a
hinge estimated from the fixture footprint and the handle side.  Both use a
non-lifting contact grasp and verify with fresh visual state plus measured
handle displacement.  No joint position or benchmark predicate is available
to the policy.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from racap.backends.vlm import PerceptionUnavailableError
from racap.backends.llm import LLMError

from racap.contracts import GroundedTarget, SkillResult

MIN_CONTACT_POINTS = 20
DEFAULT_LINEAR_STROKE_M = 0.12
MAX_CLOSE_STROKE_M = 0.24
DEFAULT_HINGE_ANGLE_DEG = 38.0
MIN_HINGE_PROGRESS_DEG = 65.0
HOOK_OFFSET_M = 0.025
MIN_HOOK_OFFSET_M = 0.012
HOOK_APPROACH_M = 0.08
PRISMATIC_MEASUREMENT_MARGIN_M = 0.03
FACE_PUSH_APPROACH_M = 0.06
FACE_PUSH_CONTACT_GAP_M = 0.008
FINGERTIP_REACH_MIN_M = 0.065
FINGERTIP_REACH_MAX_M = 0.155
FINGERTIP_PLANAR_TOLERANCE_M = 0.025
TOP_DOWN_FINGERTIP_OFFSET_M = 0.120


def _unit(vector: np.ndarray) -> np.ndarray | None:
    value = np.asarray(vector, dtype=float).reshape(2)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-5 else None


def top_down_fingertip_contact(
    execution: Any,
    contact_z: float,
) -> bool:
    """Whether a nominally high wrist pose puts the fingers on the contact.

    LIBERO reports the Panda grip-site / wrist reference, while RGB-D measures
    the physical bar.  For a straight-down hand those are separated by roughly
    one finger length.  Comparing the two 3-D points directly therefore calls
    a good contact a 10--13 cm IK miss.  Admit it only when the horizontal
    alignment is tight and the vertical separation lies in the robot's fixed
    fingertip envelope; arbitrary high or laterally missed poses still fail.
    """
    if not isinstance(execution, dict):
        return False
    if execution.get("source") not in {"pca_axis", "top_down"}:
        return False
    try:
        commanded = np.asarray(execution["position"], dtype=float).reshape(3)
        reached = np.asarray(execution["reached_position"], dtype=float).reshape(3)
        planar_error = float(np.linalg.norm(reached[:2] - commanded[:2]))
        wrist_above_contact = float(reached[2] - float(contact_z))
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        planar_error <= FINGERTIP_PLANAR_TOLERANCE_M
        and FINGERTIP_REACH_MIN_M <= wrist_above_contact <= FINGERTIP_REACH_MAX_M
    )


def rotate_quaternion_world_z(quaternion: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate an arbitrary end-effector quaternion about world Z.

    A hinged handle rotates with the door.  Following only its XY arc while
    freezing the wrist orientation twists a retained grasp off the handle
    after a few degrees.  Left-multiplying by a world-frame Z rotation keeps
    the original contact tilt while rotating the fingers with the mechanism.
    Quaternions use the runtime's ``wxyz`` convention.
    """
    w, x, y, z = np.asarray(quaternion, dtype=float).reshape(4)
    half = 0.5 * float(angle_rad)
    cw, sz = float(np.cos(half)), float(np.sin(half))
    value = np.asarray(
        [
            cw * w - sz * z,
            cw * x - sz * y,
            cw * y + sz * x,
            cw * z + sz * w,
        ]
    )
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-9 else value


def protruding_contact(
    points: np.ndarray,
    fixture_xy: np.ndarray,
    outward: np.ndarray,
    *,
    expected_z: float | None = None,
) -> dict[str, Any] | None:
    """Fit the compact RGB-D front edge that can transmit a pull.

    Small handles are often boxed correctly by the VLM but SAM merges their
    mask into the broad drawer face.  A mask centroid then closes the fingers
    on the immovable facade.  Inside the semantic handle box, the actual handle
    is the band protruding furthest along the measured handle-to-fixture axis.
    This geometric refinement is shared by drawers and door handles and uses
    neither an asset model nor joint state.
    """
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if expected_z is not None and np.isfinite(float(expected_z)):
        same_level = cloud[np.abs(cloud[:, 2] - float(expected_z)) <= 0.035]
        if same_level.shape[0] >= MIN_CONTACT_POINTS:
            cloud = same_level
    direction = _unit(outward)
    if cloud.shape[0] < MIN_CONTACT_POINTS or direction is None:
        return None
    fixture = np.asarray(fixture_xy, dtype=float).reshape(2)
    projection = (cloud[:, :2] - fixture) @ direction
    lo, hi = np.percentile(projection, [2, 98])
    if not np.isfinite([lo, hi]).all() or hi - lo < 0.002:
        return None
    # A percentile band is less sensitive than an absolute depth threshold to
    # one flying depth pixel, while still excluding the broad recessed face.
    threshold = float(np.percentile(projection, 82))
    front = cloud[projection >= threshold]
    if front.shape[0] < MIN_CONTACT_POINTS:
        return None

    centre_xy = np.median(front[:, :2], axis=0)
    z_low, z_high = np.percentile(front[:, 2], [10, 90])
    # A handle is not a transport object to be closed below its top.  Its
    # centre line is the contact that keeps two fingertips on opposite sides.
    contact_z = float(np.median(front[:, 2]))
    flat = front[:, :2] - centre_xy
    try:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        principal = vt[0]
    except np.linalg.LinAlgError:
        principal = np.asarray([-direction[1], direction[0]])
    extent = np.percentile(front, 95, axis=0) - np.percentile(front, 5, axis=0)
    return {
        "pose": np.asarray([centre_xy[0], centre_xy[1], np.median(front[:, 2])]),
        "top_z": contact_z,
        "principal_axis": np.asarray(principal, dtype=float),
        "extent": np.asarray(extent, dtype=float),
        "n_points": int(front.shape[0]),
        "protrusion_m": float(np.median((front[:, :2] - fixture) @ direction)),
        # Retain the geometry selected by this independent RGB-D refinement.
        # Learned grasp generators should see the physical protruding band,
        # not re-run a broad semantic handle mask that can include the drawer
        # face. This stays internal metadata and is never a simulator label.
        "contact_points": np.asarray(front, dtype=float),
    }


def linear_waypoints(
    handle_xy: np.ndarray,
    fixture_xy: np.ndarray,
    *,
    goal: str,
    stroke_m: float = DEFAULT_LINEAR_STROKE_M,
    steps: int = 5,
    outward_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Waypoints along the measured outward axis and that unit axis."""
    outward = _unit(
        np.asarray(handle_xy, dtype=float) - np.asarray(fixture_xy, dtype=float)
        if outward_axis is None
        else np.asarray(outward_axis, dtype=float)
    )
    if outward is None:
        return None
    sign = 1.0 if goal == "open" else -1.0
    distances = np.linspace(0.0, sign * float(stroke_m), max(2, int(steps)) + 1)[1:]
    path = np.asarray(handle_xy, dtype=float) + distances[:, None] * outward
    return path, outward


def closing_face_push_points(
    handle_xy: np.ndarray,
    outward_axis: np.ndarray,
    final_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """External approach, first contact, and retreat for drawer closing.

    Opening needs tensile contact and may grasp or hook a handle. Closing is
    the opposite contact problem: a closed narrow gripper should remain on the
    *outside* of the drawer front and transmit compression.  Approaching from
    the outward normal prevents a top-down descent into the drawer interior;
    retreating along that same normal avoids pulling the just-closed drawer
    back open.
    """
    outward = _unit(np.asarray(outward_axis, dtype=float))
    if outward is None:
        return None
    handle = np.asarray(handle_xy, dtype=float).reshape(2)
    final = np.asarray(final_xy, dtype=float).reshape(2)
    approach = handle + outward * FACE_PUSH_APPROACH_M
    contact = handle + outward * FACE_PUSH_CONTACT_GAP_M
    retreat = final + outward * FACE_PUSH_APPROACH_M
    return approach, contact, retreat


def prismatic_stroke(
    desired: str,
    amount: float | None,
    protrusion_cm: float = 0.0,
) -> float:
    """Bounded stroke, using visible remaining travel for compression."""
    if amount is not None:
        return float(np.clip(abs(float(amount)), 0.04, MAX_CLOSE_STROKE_M))
    if desired != "closed":
        return DEFAULT_LINEAR_STROKE_M
    return float(
        np.clip(
            max(DEFAULT_LINEAR_STROKE_M, float(protrusion_cm) / 100.0),
            DEFAULT_LINEAR_STROKE_M,
            MAX_CLOSE_STROKE_M,
        )
    )


def prismatic_axis_from_footprint(
    fixture_xy: np.ndarray,
    handle_xy: np.ndarray,
    fixture_points: np.ndarray,
) -> dict[str, Any] | None:
    """Estimate a drawer rail axis from the stationary fixture footprint.

    A handle may sit well left or right of a cabinet centre, so the direct
    handle-to-centre vector contains a lateral component that makes a drawer
    bind.  Drawer travel instead follows the short horizontal principal axis
    of the enclosing cabinet.  The handle only supplies the outward sign.
    Near-square or poorly observed footprints are rejected, leaving the
    direct geometric axis as a conservative fallback.
    """
    cloud = np.asarray(fixture_points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if cloud.shape[0] < 40:
        return None
    centre = np.median(cloud[:, :2], axis=0)
    flat = cloud[:, :2] - centre
    # Trim distant depth outliers before fitting the two horizontal axes.
    radius = np.linalg.norm(flat, axis=1)
    flat = flat[radius <= np.percentile(radius, 97)]
    if flat.shape[0] < 40:
        return None
    try:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    long_axis = _unit(vt[0])
    short_axis = _unit(vt[-1])
    if long_axis is None or short_axis is None:
        return None
    long_projection = flat @ long_axis
    short_projection = flat @ short_axis
    long_span = float(np.percentile(long_projection, 95) - np.percentile(long_projection, 5))
    short_span = float(np.percentile(short_projection, 95) - np.percentile(short_projection, 5))
    if (
        not np.isfinite([long_span, short_span]).all()
        or short_span < 0.025
        or long_span < 1.12 * short_span
    ):
        return None
    raw_outward = _unit(np.asarray(handle_xy, dtype=float) - np.asarray(fixture_xy, dtype=float))
    alignment = 0.0 if raw_outward is None else abs(float(short_axis @ raw_outward))
    # The direct vector is already the lower-variance estimate only when it is
    # *nearly* collinear with the footprint depth axis. Drawer handles are
    # commonly offset several centimetres from the cabinet centre; accepting a
    # 20--30 degree diagonal as "close" produces a lateral wrench and binds the
    # slide. Use PCA to remove that offset, while rejecting a near-perpendicular
    # footprint fit.
    if raw_outward is None or alignment < 0.30 or alignment >= 0.97:
        return None
    if float(short_axis @ raw_outward) < 0.0:
        short_axis = -short_axis
    return {
        "axis": short_axis,
        "long_span_m": long_span,
        "short_span_m": short_span,
        "alignment": alignment,
    }


def prismatic_axis_from_opening_boundary(
    handle_xy: np.ndarray,
    opening_polygon_xy: np.ndarray,
) -> dict[str, Any] | None:
    """Recover the drawer normal from a previously observed open cavity.

    The cavity is observed before transport, when neither the carried object
    nor the arm hides it.  A drawer handle may be far from the cabinet centre,
    so ``handle - fixture_centre`` is generally *not* the rail direction.  For
    an open drawer the handle is outside its cavity; the vector from the
    nearest cavity-boundary point to the handle is the local outward normal.
    This construction works for rotated drawers and non-centred handles and
    does not require simulator joint state or a known world axis.
    """
    handle = np.asarray(handle_xy, dtype=float).reshape(2)
    polygon = np.asarray(opening_polygon_xy, dtype=float).reshape(-1, 2)
    polygon = polygon[np.isfinite(polygon).all(axis=1)]
    if polygon.shape[0] < 3 or not np.isfinite(handle).all():
        return None

    # A handle inside the observed opening does not define an outward side;
    # this is usually a bad/rebound grounding and must not seed kinematic
    # memory.  Standard even/odd ray casting avoids an OpenCV dependency here.
    inside = False
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        yi, yj = float(polygon[i, 1]), float(polygon[j, 1])
        xi, xj = float(polygon[i, 0]), float(polygon[j, 0])
        if (yi > handle[1]) != (yj > handle[1]) and handle[0] < (xj - xi) * (handle[1] - yi) / (
            yj - yi + 1e-12
        ) + xi:
            inside = not inside
        j = i
    if inside:
        return None

    centre = np.mean(polygon, axis=0)
    radial = _unit(handle - centre)
    best_point = None
    best_axis = None
    best_distance = float("inf")
    best_score = -float("inf")
    for i in range(polygon.shape[0]):
        start = polygon[i]
        end = polygon[(i + 1) % polygon.shape[0]]
        edge = end - start
        denom = float(edge @ edge)
        length = float(np.sqrt(denom))
        if length < 0.015:
            continue
        raw_fraction = float((handle - start) @ edge / denom)
        # Permit a little handle overhang beyond the cavity corner, but do not
        # mistake the adjacent side wall for the front face.
        if not -0.15 <= raw_fraction <= 1.15:
            continue
        point = start + raw_fraction * edge
        distance = float(np.linalg.norm(handle - point))
        normal = _unit(np.asarray([-edge[1], edge[0]], dtype=float))
        if normal is None:
            continue
        if float(normal @ (handle - point)) < 0.0:
            normal = -normal
        alignment = 0.0 if radial is None else float(normal @ radial)
        score = alignment * min(1.0, length / 0.05) - distance
        if 0.008 <= distance <= 0.30 and score > best_score:
            best_point, best_axis = point, normal
            best_distance, best_score = distance, score
    axis = best_axis
    # Sub-centimetre separation is too sensitive to mask jitter; a handle more
    # than 30 cm from the opening is almost certainly a semantic rebind.
    if axis is None or not 0.008 <= best_distance <= 0.30:
        return None
    return {
        "axis": axis,
        "nearest_boundary_xy": np.asarray(best_point, dtype=float),
        "handle_boundary_distance_m": best_distance,
    }


def prismatic_axis_from_opening_fixture(
    fixture_xy: np.ndarray,
    opening_polygon_xy: np.ndarray,
) -> dict[str, Any] | None:
    """Choose the opening edge normal that points away from the fixture.

    This is the handle-independent fallback for an open drawer.  Semantic
    handle masks sometimes rebound to the dark cavity and therefore lie
    *inside* the opening, where they cannot define an outward normal.  The
    cavity itself is still in front of the fixed cabinet body.  Projecting the
    fixture-to-cavity vector onto the observed edge normals recovers the rail
    axis without assuming a world direction or exposing simulator joints.
    """
    fixture = np.asarray(fixture_xy, dtype=float).reshape(2)
    polygon = np.asarray(opening_polygon_xy, dtype=float).reshape(-1, 2)
    polygon = polygon[np.isfinite(polygon).all(axis=1)]
    if polygon.shape[0] < 3 or not np.isfinite(fixture).all():
        return None
    centre = np.mean(polygon, axis=0)
    radial = _unit(centre - fixture)
    if radial is None:
        return None

    best_axis = None
    best_alignment = 0.0
    best_length = 0.0
    for i in range(polygon.shape[0]):
        edge = polygon[(i + 1) % polygon.shape[0]] - polygon[i]
        length = float(np.linalg.norm(edge))
        if length < 0.025:
            continue
        normal = _unit(np.asarray([-edge[1], edge[0]], dtype=float))
        if normal is None:
            continue
        if float(normal @ radial) < 0.0:
            normal = -normal
        alignment = float(normal @ radial)
        # Prefer a normal supported by a long, stable boundary.  Alignment is
        # the dominant term so a long side wall cannot beat the front edge.
        score = alignment + 0.10 * min(length / 0.10, 1.0)
        best_score = best_alignment + 0.10 * min(best_length / 0.10, 1.0)
        if score > best_score:
            best_axis = normal
            best_alignment = alignment
            best_length = length
    if best_axis is None or best_alignment < 0.65:
        return None
    return {
        "axis": best_axis,
        "opening_centre_xy": centre,
        "fixture_alignment": best_alignment,
        "supporting_edge_length_m": best_length,
    }


def prismatic_axis_from_opening_probe(
    probe_points: np.ndarray,
    opening_polygon_xy: np.ndarray,
    *,
    expected_z: float | None = None,
) -> dict[str, Any] | None:
    """Fit the outward opening normal from RGB-D inside a handle VLM box.

    SAM can attach a correct handle box to the distant background, making its
    3-D centroid useless.  The unsegmented RGB-D box still contains the real
    protruding face.  Select the opening edge with a compact band of nearby,
    same-height points outside it and orient the normal toward that band.
    """
    cloud = np.asarray(probe_points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    polygon = np.asarray(opening_polygon_xy, dtype=float).reshape(-1, 2)
    polygon = polygon[np.isfinite(polygon).all(axis=1)]
    if cloud.shape[0] < MIN_CONTACT_POINTS or polygon.shape[0] < 3:
        return None
    if expected_z is not None and np.isfinite(float(expected_z)):
        level = cloud[np.abs(cloud[:, 2] - float(expected_z)) <= 0.055]
        if level.shape[0] >= MIN_CONTACT_POINTS:
            cloud = level

    best = None
    best_score = -float("inf")
    for i in range(polygon.shape[0]):
        start = polygon[i]
        edge = polygon[(i + 1) % polygon.shape[0]] - start
        length_sq = float(edge @ edge)
        length = float(np.sqrt(length_sq))
        if length < 0.025:
            continue
        fractions = ((cloud[:, :2] - start) @ edge) / length_sq
        along = (fractions >= -0.10) & (fractions <= 1.10)
        nearest = start[None, :] + fractions[:, None] * edge[None, :]
        delta = cloud[:, :2] - nearest
        distances = np.linalg.norm(delta, axis=1)
        band = along & (distances >= 0.006) & (distances <= 0.16)
        if int(np.sum(band)) < MIN_CONTACT_POINTS:
            continue
        normal = _unit(np.asarray([-edge[1], edge[0]], dtype=float))
        if normal is None:
            continue
        signed = delta[band] @ normal
        sign = float(np.sign(np.median(signed)) or 1.0)
        normal *= sign
        outward_distance = signed * sign
        consistent = outward_distance > 0.004
        count = int(np.sum(consistent))
        if count < MIN_CONTACT_POINTS:
            continue
        median_distance = float(np.median(outward_distance[consistent]))
        spread = float(np.median(np.abs(outward_distance[consistent] - median_distance)))
        score = count - 400.0 * spread
        if score > best_score:
            best_score = score
            best = {
                "axis": normal.copy(),
                "support_points": count,
                "median_boundary_distance_m": median_distance,
                "supporting_edge_length_m": length,
            }
    return best


def relative_planar_progress(
    handle_before: np.ndarray,
    fixture_before: np.ndarray,
    handle_after: np.ndarray,
    fixture_after: np.ndarray,
    axis: np.ndarray,
) -> float:
    """Joint-like progress with rigid fixture translation cancelled out."""
    before_relative = np.asarray(handle_before, dtype=float).reshape(2) - np.asarray(
        fixture_before, dtype=float
    ).reshape(2)
    after_relative = np.asarray(handle_after, dtype=float).reshape(2) - np.asarray(
        fixture_after, dtype=float
    ).reshape(2)
    direction = _unit(np.asarray(axis, dtype=float))
    if direction is None:
        return 0.0
    return float((after_relative - before_relative) @ direction)


def prismatic_measurement_within_command(
    displacement: float,
    directed_progress: float,
    commanded_stroke: float,
) -> bool:
    """Reject a re-grounded handle outside the commanded path envelope.

    A contact may slip and therefore move less than requested, but a handle
    cannot move substantially farther than the Cartesian path just executed.
    This symmetric envelope catches detector identity jumps without assuming
    a particular drawer size or consulting simulator joint state.
    """
    values = np.asarray([displacement, directed_progress, commanded_stroke], dtype=float)
    if not np.isfinite(values).all() or commanded_stroke < 0.0:
        return False
    limit = commanded_stroke + PRISMATIC_MEASUREMENT_MARGIN_M
    return bool(0.0 <= displacement <= limit and abs(directed_progress) <= limit)


def fixture_refresh_gate(
    before_xy: np.ndarray,
    after_xy: np.ndarray,
    fixture_extent: Any,
) -> tuple[bool, float, float]:
    """Validate the static reference used for relative articulation motion."""
    before = np.asarray(before_xy, dtype=float).reshape(2)
    after = np.asarray(after_xy, dtype=float).reshape(2)
    extent = np.asarray(
        [0.0, 0.0, 0.0] if fixture_extent is None else fixture_extent,
        dtype=float,
    ).reshape(-1)
    jump = float(np.linalg.norm(after - before))
    limit = min(
        0.35,
        max(0.12, 0.50 * float(max(extent[:2], default=0.0))),
    )
    return bool(np.isfinite(jump) and jump <= limit), jump, limit


def hinge_waypoints(
    handle_xy: np.ndarray,
    hinge_xy: np.ndarray,
    fixture_xy: np.ndarray,
    *,
    goal: str,
    angle_deg: float = DEFAULT_HINGE_ANGLE_DEG,
    steps: int = 7,
) -> np.ndarray | None:
    """Choose the arc direction that moves away from/toward the fixture."""
    handle = np.asarray(handle_xy, dtype=float).reshape(2)
    pivot = np.asarray(hinge_xy, dtype=float).reshape(2)
    fixture = np.asarray(fixture_xy, dtype=float).reshape(2)
    radius = handle - pivot
    if float(np.linalg.norm(radius)) < 0.035:
        return None
    magnitude = float(np.radians(np.clip(abs(angle_deg), 20.0, 110.0)))

    def rotate(angle: float) -> np.ndarray:
        c, s = float(np.cos(angle)), float(np.sin(angle))
        return pivot + np.asarray([[c, -s], [s, c]]) @ radius

    positive = rotate(magnitude)
    negative = rotate(-magnitude)
    pos_distance = float(np.linalg.norm(positive - fixture))
    neg_distance = float(np.linalg.norm(negative - fixture))
    if goal == "open":
        signed = magnitude if pos_distance >= neg_distance else -magnitude
    else:
        signed = magnitude if pos_distance <= neg_distance else -magnitude
    return np.asarray(
        [rotate(angle) for angle in np.linspace(0.0, signed, max(2, int(steps)) + 1)[1:]]
    )


def hinge_from_moving_panel(
    panel_xy: np.ndarray,
    fixture_xy: np.ndarray,
    panel_extent: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray | None:
    """Estimate the near hinge of a visibly open broad door panel.

    Segmenting an open door together with its appliance contaminates the
    fixture PCA and can put the hinge on the far outer appliance edge.  The
    moving panel itself supplies the better constraint: its hinge is the end
    of the panel nearest the fixed body.  Approximate that endpoint by moving
    half a visible panel width from its centre toward the fixture.  This is
    deliberately asset-agnostic and rejects implausible tiny/huge panels.
    """
    panel = np.asarray(panel_xy, dtype=float).reshape(2)
    fixture = np.asarray(fixture_xy, dtype=float).reshape(2)
    toward_fixture = _unit(fixture - panel)
    extent = np.asarray(panel_extent, dtype=float).reshape(-1)
    if toward_fixture is None or extent.size < 2:
        return None
    distance = float(np.linalg.norm(fixture - panel))
    half_width = 0.5 * float(max(abs(float(v)) for v in extent[:2]))
    radius = float(min(half_width, 0.80 * distance))
    if not np.isfinite([distance, radius]).all() or distance < 0.06 or not 0.04 <= radius <= 0.28:
        return None
    return panel + toward_fixture * radius


def signed_planar_angle_deg(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    pivot_xy: np.ndarray,
) -> float | None:
    """Signed shortest angle from start to end about a measured pivot."""
    start = np.asarray(start_xy, dtype=float).reshape(2) - np.asarray(
        pivot_xy, dtype=float
    ).reshape(2)
    end = np.asarray(end_xy, dtype=float).reshape(2) - np.asarray(pivot_xy, dtype=float).reshape(2)
    if float(np.linalg.norm(start)) < 0.025 or float(np.linalg.norm(end)) < 0.025:
        return None
    cross = float(start[0] * end[1] - start[1] * end[0])
    dot = float(start @ end)
    return float(np.degrees(np.arctan2(cross, dot)))


def _usable(target: GroundedTarget | None, *, contact: bool = False) -> bool:
    if target is None or target.confidence <= 0.0:
        return False
    pose = np.asarray(target.pose, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        return False
    # RGB-D on a broad, low-texture cabinet can occasionally lift the VLM box
    # through a background depth discontinuity.  The resulting point is often
    # metres away or below the robot table, yet still has thousands of mask
    # pixels and a high semantic score.  Such a point is not merely uncertain:
    # it lies outside the Panda's single-arm manipulation workspace and must
    # never seed contact motion or persistent articulation memory.
    if float(np.linalg.norm(pose[:2])) > 1.10 or not -0.12 <= float(pose[2]) <= 0.90:
        return False
    points = int(target.metadata.get("n_points", 0))
    if points and points < (MIN_CONTACT_POINTS if contact else 20):
        return False
    extent = target.metadata.get("extent") or [0.0, 0.0, 0.0]
    if contact and max(float(v) for v in extent) > 0.35:
        return False
    return True


def _ground(runtime, label: str) -> GroundedTarget | None:
    try:
        target = runtime.localize(label, detector="vlm_bbox_sam3")
    except TypeError:
        target = runtime.localize(label)
    except PerceptionUnavailableError:
        raise
    except Exception:
        return None
    if _usable(target):
        return target
    # A grossly invalid depth lift is a view failure, not evidence that the
    # semantic target is absent.  Ask a genuinely different fixed camera once;
    # this retains visual-only grounding while preventing a bad agent-view
    # mask from poisoning all later retries.
    across_views = getattr(runtime, "localize_across_views", None)
    if across_views is not None:
        try:
            alternate = across_views(label, detector="vlm_search_bbox_sam3")
        except PerceptionUnavailableError:
            raise
        except (TypeError, ValueError):
            alternate = None
        except Exception:
            alternate = None
        if _usable(alternate):
            return alternate
    return target


def _reground_visible(
    runtime,
    label: str,
    *,
    contact: bool = False,
) -> tuple[GroundedTarget | None, str]:
    """Measure an already identified part without another semantic VLM call.

    The VLM is useful for choosing the requested handle before contact.  After
    that identity is fixed, motion verification only needs the visible mask's
    world pose.  Prefer local SAM3 and fall back to VLM grounding only when the
    local mask has no usable depth support.
    """
    try:
        target = runtime.localize(label, detector="sam3")
    except TypeError:
        target = None
    except PerceptionUnavailableError:
        raise
    except Exception:
        target = None
    if _usable(target, contact=contact):
        return target, "sam3"
    return _ground(runtime, label), "vlm_bbox_sam3_fallback"


def _handle_candidates(target: str, part: str) -> list[str]:
    def bare(value: str) -> str:
        value = " ".join(value.split()).strip()
        return re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)

    part = bare(part)
    target = bare(target)
    values = []
    if part:
        values.extend(
            [
                f"handle of the {part} of {target}",
                f"{part} handle of {target}",
                f"handle of the {part}",
            ]
        )
    values.extend([f"handle of {target}", f"{target} handle"])
    return list(dict.fromkeys(values))


def _ground_handle(
    runtime, fixture: GroundedTarget, target: str, part: str, trace: list
) -> GroundedTarget | None:
    best: GroundedTarget | None = None
    best_score = -float("inf")
    for label in _handle_candidates(target, part):
        candidate = _ground(runtime, label)
        if not _usable(candidate, contact=True):
            trace.append({"event": "ground_handle", "label": label, "usable": False})
            continue
        distance = float(
            np.linalg.norm(np.asarray(candidate.pose[:2]) - np.asarray(fixture.pose[:2]))
        )
        extent = candidate.metadata.get("extent") or [0.0, 0.0, 0.0]
        compactness = max(float(v) for v in extent)
        score = -distance - 0.25 * compactness
        trace.append(
            {
                "event": "ground_handle",
                "label": label,
                "usable": True,
                "distance_cm": round(distance * 100, 1),
            }
        )
        if distance <= 0.50 and score > best_score:
            best, best_score = candidate, score
            # The first label names both the requested part and fixture.  If
            # it produced a compact nearby box, later generic aliases add
            # latency and identity risk rather than useful evidence.
            if label == _handle_candidates(target, part)[0] and compactness <= 0.18:
                return candidate
    return best


def _panel_candidates(target: str, part: str) -> list[str]:
    """Language proposals for the externally contactable moving surface."""
    target = re.sub(r"^(?:the|a|an)\s+", "", " ".join(target.split()), flags=re.IGNORECASE)
    part = re.sub(r"^(?:the|a|an)\s+", "", " ".join(part.split()), flags=re.IGNORECASE)
    values: list[str] = []
    if part:
        values.extend(
            [
                f"external front panel of the {part} of {target}",
                f"front face of the {part} of {target}",
                f"{part} front panel",
            ]
        )
    values.extend(
        [
            f"open door panel of {target}",
            f"moving door panel of {target}",
            f"door of {target}",
        ]
    )
    return list(dict.fromkeys(values))


def _ground_moving_panel(
    runtime,
    fixture: GroundedTarget,
    target: str,
    part: str,
    trace: list,
) -> GroundedTarget | None:
    """Ground a broad moving face when a closing handle is unavailable.

    Closing transmits compression and therefore does not require a handle.  A
    panel is intentionally allowed to be broad, but must remain near the
    fixture and have enough visible extent to be a surface rather than a knob.
    """
    best: GroundedTarget | None = None
    best_score = -float("inf")
    candidates = _panel_candidates(target, part)
    specific_count = 3 if part else 0
    for index, label in enumerate(candidates):
        # A compact grounding that retained the requested moving-part phrase
        # is identity evidence.  Do not let a later, larger generic fixture
        # panel outscore it merely by surface area; generic labels are a
        # fallback only when every part-specific proposal failed.
        if part and index >= specific_count and best is not None:
            break
        candidate = _ground(runtime, label)
        if not _usable(candidate):
            trace.append({"event": "ground_panel", "label": label, "usable": False})
            continue
        extent = np.asarray(
            candidate.metadata.get("extent") or [0.0, 0.0, 0.0], dtype=float
        ).reshape(-1)
        distance = float(
            np.linalg.norm(np.asarray(candidate.pose[:2]) - np.asarray(fixture.pose[:2]))
        )
        planar_span = float(max(extent[:2], default=0.0))
        plausible_surface = bool(distance <= 0.65 and 0.06 <= planar_span <= 0.70)
        trace.append(
            {
                "event": "ground_panel",
                "label": label,
                "usable": plausible_surface,
                "distance_cm": round(distance * 100, 1),
                "planar_span_cm": round(planar_span * 100, 1),
            }
        )
        if not plausible_surface:
            continue
        # Prefer the most specific early caption, then a large load-bearing
        # face.  Distance only gates identity; an open door can be far from its
        # appliance body by design.
        score = planar_span - 0.01 * index
        if score > best_score:
            best, best_score = candidate, score
    return best


def _refine_handle_contact(
    runtime,
    fixture: GroundedTarget,
    handle: GroundedTarget,
) -> tuple[GroundedTarget, dict[str, Any]]:
    """Refine a semantic handle box to its protruding physical contact band."""
    bbox = handle.metadata.get("bbox")
    outward = _unit(
        np.asarray(handle.pose[:2], dtype=float) - np.asarray(fixture.pose[:2], dtype=float)
    )
    if not bbox or outward is None:
        return handle, {"source": "semantic_mask"}
    try:
        probed = runtime.probe_bbox(list(bbox))
        fit = protruding_contact(
            np.asarray(probed["points"], dtype=float),
            np.asarray(fixture.pose[:2], dtype=float),
            outward,
            expected_z=float(handle.pose[2]),
        )
    except Exception as exc:
        return handle, {
            "source": "semantic_mask",
            "refinement_error": f"{type(exc).__name__}:{exc}",
        }
    if fit is None:
        return handle, {"source": "semantic_mask"}
    metadata = dict(handle.metadata)
    metadata.update(
        {
            "top_z": float(fit["top_z"]),
            "principal_axis": [float(v) for v in fit["principal_axis"]],
            "contact_axis": [float(v) for v in outward],
            "extent": [float(v) for v in fit["extent"]],
            "n_points": int(fit["n_points"]),
            "contact_points": np.asarray(fit["contact_points"], dtype=float),
        }
    )
    pose = tuple(float(v) for v in fit["pose"])
    refined = GroundedTarget(
        label=handle.label,
        pose=pose,
        kind=handle.kind,
        confidence=handle.confidence,
        metadata=metadata,
    )
    return refined, {
        "source": "rgbd_protruding_band",
        "raw_pose": [round(float(v), 4) for v in handle.pose],
        "refined_pose": [round(float(v), 4) for v in pose],
        "contact_z": round(float(fit["top_z"]), 4),
        "protrusion_cm": round(float(fit["protrusion_m"]) * 100, 1),
        "points": int(fit["n_points"]),
    }


def _visual_choice(runtime, target: GroundedTarget, prompt: str, options: list[str]) -> str:
    try:
        verdict = runtime.inspect(prompt, options=options, target=target)
    except (PerceptionUnavailableError, LLMError):
        raise
    except Exception:
        return ""
    if not verdict.get("ok"):
        return ""
    return str(verdict.get("answer", "")).strip().lower()


def _mechanism(
    runtime,
    fixture: GroundedTarget,
    handle: GroundedTarget,
    target: str,
    part: str,
) -> str:
    words = f"{target} {part}".lower()
    if any(word in words for word in ("drawer", "slider", "sliding")):
        return "prismatic"
    # When a door is already open, a segmentation of the whole appliance can
    # be dominated by the door panel and a binary VLM may call it a drawer.
    # A handle more than 30 cm from the grounded fixture centre cannot be a
    # normal prismatic face contact; it is direct geometry for an extended
    # hinged link.
    handle_distance = float(
        np.linalg.norm(
            np.asarray(handle.pose[:2], dtype=float) - np.asarray(fixture.pose[:2], dtype=float)
        )
    )
    if handle_distance > 0.30:
        return "revolute"
    options = ["a sliding drawer", "a hinged door"]
    answer = _visual_choice(runtime, fixture, "the moving mechanism", options)
    return "prismatic" if answer == options[0] else "revolute"


def hinge_from_footprint(
    fixture_xy: np.ndarray,
    handle_xy: np.ndarray,
    fixture_points: np.ndarray,
) -> np.ndarray | None:
    """Estimate a side hinge from the fixture's own horizontal long axis."""
    fixture_xy = np.asarray(fixture_xy, dtype=float).reshape(2)
    handle_xy = np.asarray(handle_xy, dtype=float).reshape(2)
    cloud = np.asarray(fixture_points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if cloud.shape[0] < 30:
        return None
    flat = cloud[:, :2] - fixture_xy
    try:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    lateral = _unit(vt[0])
    if lateral is None:
        return None
    projection = flat @ lateral
    half_width = float(np.percentile(np.abs(projection), 90))
    if not 0.04 <= half_width <= 0.40:
        return None
    handle_side = float((handle_xy - fixture_xy) @ lateral)
    if abs(handle_side) < 0.01:
        return None
    # With a closed panel, the handle is inside the fixture silhouette and the
    # hinge is on the opposite edge.  With a door already extended beyond the
    # body, the hinge is instead the *near* body edge between fixture and free
    # handle.  Treating that open-door handle as a closed-panel handle doubled
    # the estimated radius and generated an arc through the appliance.
    side = np.sign(handle_side)
    if abs(handle_side) > 1.25 * half_width:
        return fixture_xy + side * lateral * half_width
    # Axis sign is arbitrary, but sign(projection) * axis is invariant to it.
    return fixture_xy - side * lateral * half_width


def _hinge_from_vision(runtime, fixture: GroundedTarget, handle: GroundedTarget) -> np.ndarray:
    fixture_xy = np.asarray(fixture.pose[:2], dtype=float)
    handle_xy = np.asarray(handle.pose[:2], dtype=float)
    try:
        cloud = np.asarray(runtime.object_points(fixture.label), dtype=float).reshape(-1, 3)
        fitted = hinge_from_footprint(fixture_xy, handle_xy, cloud)
    except Exception:
        fitted = None
    if fitted is not None:
        return fitted
    try:
        right = _unit(np.asarray(runtime.image_axes()["right"], dtype=float))
    except Exception:
        right = None
    if right is None:
        right = np.asarray([1.0, 0.0])
    signed_side = float((handle_xy - fixture_xy) @ right)
    try:
        cloud = np.asarray(runtime.object_points(fixture.label), dtype=float).reshape(-1, 3)
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        projections = (cloud[:, :2] - fixture_xy) @ right
        half_width = float(np.percentile(np.abs(projections), 90))
    except Exception:
        half_width = 0.0
    extent = fixture.metadata.get("extent") or [0.18, 0.18, 0.18]
    half_width = float(
        np.clip(max(half_width, 0.35 * max(float(v) for v in extent[:2])), 0.07, 0.25)
    )
    # The hinge lies on the side opposite the handle.
    return fixture_xy - np.sign(signed_side or 1.0) * right * half_width


def articulate(
    runtime,
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
    opening_polygon_hint: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
) -> SkillResult:
    """Open or close a visually grounded drawer/door mechanism."""
    goal = "closed" if str(goal).lower() in {"close", "closed"} else "open"
    requested_mechanism = str(mechanism).lower()
    requested_contact = (
        str(contact_strategy).lower()
        if str(contact_strategy).lower()
        in {"auto", "side_bar", "pca_axis", "top_down", "graspnet_6d", "graspgen_6d", "hook"}
        else "pca_axis"
    )
    trace: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "params": {
            "target": target,
            "part": part,
            "goal": goal,
            "mechanism": requested_mechanism,
            "amount": amount,
            "contact_strategy": requested_contact,
            "hinge_hint": hinge_hint,
            "handle_z_hint": handle_z_hint,
            "hinge_radius_hint": hinge_radius_hint,
            "panel_label_hint": panel_label_hint,
            "axis_hint": axis_hint,
            "opening_polygon_hint_vertices": (
                0 if opening_polygon_hint is None else len(opening_polygon_hint)
            ),
        },
        "preflight_only": True,
    }

    def finish(success: bool, mode: str = "", message: str = "") -> SkillResult:
        return SkillResult(
            success=success,
            status="success" if success else "retryable_failure",
            pick_label=target,
            place_label=goal,
            attempts=1,
            strategy="visual_articulation",
            failure_mode=mode,
            message=message,
            trace=trace,
            report=report,
        )

    desired = "open" if goal == "open" else "closed"
    fixture = _ground(runtime, target)
    if not _usable(fixture):
        report.update({"grounding_failure": "target", "failed_label": target})
        return finish(False, "not_grounded", f"no usable grounding for {target!r}")
    # Compression does not need force closure. Prefer the broad external
    # moving face for every close operation: it supplies a larger contact
    # patch and does not let closed fingers slip around a narrow handle after
    # a few centimetres. Opening remains a tensile problem and therefore
    # still requires a handle/hook. A handle is only the closing fallback when
    # the moving panel is not visible or cannot be grounded reliably.
    moving_panel_contact = False
    if desired == "closed":
        handle = None
        if str(panel_label_hint or "").strip():
            tracked, tracked_by = _reground_visible(runtime, str(panel_label_hint), contact=False)
            if _usable(tracked):
                extent = np.asarray(
                    tracked.metadata.get("extent") or [0.0, 0.0, 0.0],
                    dtype=float,
                ).reshape(-1)
                distance = float(
                    np.linalg.norm(
                        np.asarray(tracked.pose[:2], dtype=float)
                        - np.asarray(fixture.pose[:2], dtype=float)
                    )
                )
                span = float(max(extent[:2], default=0.0))
                if distance <= 0.65 and 0.05 <= span <= 0.70:
                    handle = tracked
                    trace.append(
                        {
                            "event": "track_panel",
                            "label": str(panel_label_hint),
                            "usable": True,
                            "source": tracked_by,
                        }
                    )
            if handle is None:
                trace.append(
                    {
                        "event": "track_panel",
                        "label": str(panel_label_hint),
                        "usable": False,
                    }
                )
        if handle is None:
            handle = _ground_moving_panel(runtime, fixture, target, part, trace)
        moving_panel_contact = handle is not None
        if handle is None:
            handle = _ground_handle(runtime, fixture, target, part, trace)
    else:
        handle = _ground_handle(runtime, fixture, target, part, trace)
    if handle is None:
        report.update(
            {
                "grounding_failure": "handle",
                "failed_label": part or target,
                "grounding_trace": trace,
            }
        )
        return finish(
            False,
            "not_grounded",
            "no handle for opening and no external moving panel for closing",
        )
    if moving_panel_contact:
        contact_fit = {
            "source": "semantic_moving_panel",
            "contact_z": round(float(handle.pose[2]), 4),
        }
    else:
        handle, contact_fit = _refine_handle_contact(runtime, fixture, handle)
    if handle_z_hint is not None and not moving_panel_contact:
        try:
            height_error = abs(float(handle.pose[2]) - float(handle_z_hint))
        except (TypeError, ValueError):
            height_error = float("inf")
        if height_error > 0.035:
            report.update(
                {
                    "grounding_failure": "handle_kinematics",
                    "failed_label": handle.label,
                    "handle_height_error_cm": round(height_error * 100, 1),
                    "grounding_trace": trace,
                }
            )
            return finish(
                False,
                "not_grounded",
                "handle grounding violates the remembered articulation height",
            )

    resolved = requested_mechanism
    if resolved not in {"prismatic", "revolute"}:
        resolved = _mechanism(runtime, fixture, handle, target, part)
    report.update(
        {
            "mechanism": resolved,
            "fixture_pose": [round(float(v), 4) for v in fixture.pose],
            "fixture_bbox": fixture.metadata.get("bbox"),
            "handle_label": handle.label,
            "handle_before": [round(float(v), 4) for v in handle.pose],
            "handle_contact": contact_fit,
            "handle_bbox": handle.metadata.get("bbox"),
            "handle_extent": handle.metadata.get("extent"),
            "moving_panel_contact": moving_panel_contact,
        }
    )

    other = "closed" if desired == "open" else "open"
    state_options = [f"the mechanism is {desired}", f"the mechanism is {other}"]
    # The outer full-agent state loop owns semantic state.  Asking an internal
    # classifier here and then asking the authoritative agent again on the
    # same scene duplicated latency and created competing verdicts.  Preserve
    # only contact geometry and metric motion in this policy report.
    before_state = ""
    report["visual_state_before"] = before_state
    report["visual_preflight_match"] = False
    # A single static view is not state-defining evidence. Door seams are
    # routinely called "open" while fully closed, and shallow open drawers can
    # be called "closed" from an oblique camera. Keep the VLM answer as a
    # diagnostic, but require contact motion / endpoint evidence before an
    # articulation skill can claim success. This rule is mechanism-generic and
    # also prevents a false preflight from skipping dependent subgoals.
    report["visual_preflight_advisory_only"] = True

    handle_xy = np.asarray(handle.pose[:2], dtype=float)
    fixture_xy = np.asarray(fixture.pose[:2], dtype=float)
    if resolved == "prismatic":
        # A fixed 12 cm stroke required three expensive contacts for a 21 cm
        # open drawer and exhausted the episode horizon before dependent
        # placement. Command the measured protrusion and let the joint stop
        # absorb the bounded excess.
        stroke = prismatic_stroke(desired, amount, float(contact_fit.get("protrusion_cm", 0.0)))
        report["planned_linear_stroke_cm"] = round(stroke * 100, 1)
        hinted_axis = None
        try:
            hinted_axis = _unit(np.asarray(axis_hint, dtype=float).reshape(2))
        except (TypeError, ValueError):
            hinted_axis = None
        axis_fit = None
        opening_axis_fit = None
        if hinted_axis is None and opening_polygon_hint is not None:
            try:
                opening_axis_fit = prismatic_axis_from_opening_boundary(
                    handle_xy, np.asarray(opening_polygon_hint, dtype=float)
                )
            except (TypeError, ValueError):
                opening_axis_fit = None
        if hinted_axis is None:
            if opening_axis_fit is None:
                try:
                    axis_fit = prismatic_axis_from_footprint(
                        fixture_xy,
                        handle_xy,
                        np.asarray(runtime.object_points(fixture.label), dtype=float),
                    )
                except Exception:
                    axis_fit = None
        planned = linear_waypoints(
            handle_xy,
            fixture_xy,
            goal=desired,
            stroke_m=stroke,
            outward_axis=(
                hinted_axis
                if hinted_axis is not None
                else opening_axis_fit["axis"]
                if opening_axis_fit is not None
                else None
                if axis_fit is None
                else axis_fit["axis"]
            ),
        )
        if planned is None:
            return finish(
                False, "no_motion_axis", "handle and fixture centres do not define an axis"
            )
        path, outward = planned
        report["motion_axis"] = [round(float(v), 4) for v in outward]
        report["motion_axis_source"] = (
            "episode_memory"
            if hinted_axis is not None
            else "persistent_opening_boundary"
            if opening_axis_fit is not None
            else "fixture_footprint_short_axis"
            if axis_fit is not None
            else "handle_to_fixture"
        )
        if opening_axis_fit is not None:
            report["motion_axis_fit"] = {
                "nearest_boundary_xy": [
                    round(float(v), 4) for v in opening_axis_fit["nearest_boundary_xy"]
                ],
                "handle_boundary_distance_cm": round(
                    float(opening_axis_fit["handle_boundary_distance_m"]) * 100, 1
                ),
            }
        if axis_fit is not None:
            report["motion_axis_fit"] = {
                "long_span_cm": round(float(axis_fit["long_span_m"]) * 100, 1),
                "short_span_cm": round(float(axis_fit["short_span_m"]) * 100, 1),
                "handle_alignment": round(float(axis_fit["alignment"]), 3),
            }
    else:
        hinted = None
        try:
            hinted = np.asarray(hinge_hint, dtype=float).reshape(2)
            radius = float(np.linalg.norm(handle_xy - hinted))
            expected_radius = radius if hinge_radius_hint is None else float(hinge_radius_hint)
            if (
                not np.isfinite(hinted).all()
                or not 0.035 <= radius <= 0.50
                or abs(radius - expected_radius) > 0.06
            ):
                hinted = None
        except (TypeError, ValueError):
            hinted = None
        panel_hinge = None
        if hinted is None and moving_panel_contact:
            panel_hinge = hinge_from_moving_panel(
                handle_xy,
                fixture_xy,
                handle.metadata.get("extent") or [0.0, 0.0, 0.0],
            )
        hinge = (
            hinted
            if hinted is not None
            else panel_hinge
            if panel_hinge is not None
            else _hinge_from_vision(runtime, fixture, handle)
        )
        report["hinge_source"] = (
            "episode_memory" if hinted is not None else "fixture_rgbd_footprint"
        )
        if hinted is None and panel_hinge is not None:
            report["hinge_source"] = "moving_panel_near_endpoint"
        default_angle = (
            MIN_HINGE_PROGRESS_DEG
            if desired == "closed" and moving_panel_contact
            else DEFAULT_HINGE_ANGLE_DEG
        )
        angle = default_angle if amount is None else float(np.clip(abs(float(amount)), 20.0, 45.0))
        path = hinge_waypoints(handle_xy, hinge, fixture_xy, goal=desired, angle_deg=angle)
        if path is None:
            return finish(False, "no_motion_axis", "handle and estimated hinge are too close")
        report["hinge_xy"] = [round(float(v), 4) for v in hinge]
        report["hinge_radius_cm"] = round(float(np.linalg.norm(handle_xy - hinge)) * 100, 1)
        planned_hinge_rotation = signed_planar_angle_deg(handle_xy, path[-1], hinge)
        report["planned_hinge_rotation_deg"] = (
            None if planned_hinge_rotation is None else round(planned_hinge_rotation, 1)
        )

    report["planned_path"] = [[round(float(v), 4) for v in point] for point in path]
    trace.append(
        {
            "event": "articulation_plan",
            "mechanism": resolved,
            "goal": desired,
            "waypoints": len(path),
        }
    )
    handle_extent = handle.metadata.get("extent") or [1.0, 1.0, 1.0]
    thin_bar = bool(
        contact_fit.get("source") == "rgbd_protruding_band"
        and min(float(v) for v in handle_extent[:2]) <= 0.035
    )
    if requested_contact == "auto":
        requested_contact = (
            # A side-on Panda pose places the wrist roughly one finger length
            # outside the bar.  At cabinets near the workspace boundary that
            # wrist pose is often unreachable even though the bar itself is
            # easily reachable from above.  PCA yaw still closes across the
            # measured bar and keeps the first contact in the arm's dexterous
            # workspace.  Side/hook contacts remain distinct recovery modes.
            "pca_axis" if resolved == "prismatic" and desired == "open" else "graspgen_6d"
        )
    report["selected_contact_strategy"] = requested_contact
    # A compact bar on a prismatic drawer does not require force closure: a
    # closed pair of fingers behind it transmits the outward pull more reliably
    # than treating a collision-induced finger gap as a retained grasp. Keep
    # learned 6-DoF grasping as the next recovery strategy for knobs, thick
    # handles, hinged arcs, or an explicit retry with another contact mode.
    hook_primary = bool(
        resolved == "prismatic" and desired == "open" and thin_bar and requested_contact == "hook"
    )
    report["hook_primary"] = hook_primary
    try:
        # Every branch below eventually shares the fallback hook geometry.
        # An explicit ``hook`` contact deliberately skips contact_grasp, so it
        # has no execution record.  Initialising this here prevents that valid
        # control path from reading an unbound local before its first motion.
        contact_execution: dict[str, Any] | None = None
        closed_face_push = bool(resolved == "prismatic" and desired == "closed")
        closed_panel_sweep = bool(
            resolved == "revolute" and desired == "closed" and moving_panel_contact
        )
        if closed_face_push:
            push_points = closing_face_push_points(
                handle_xy, np.asarray(report["motion_axis"], dtype=float), path[-1]
            )
            if push_points is None:
                return finish(False, "no_motion_axis", "no external drawer-face normal")
            approach_xy, contact_xy, retreat_xy = push_points
            contact_z = float(handle.metadata.get("top_z", handle.pose[2]))
            yaw = float(np.arctan2(outward[1], outward[0]) + 0.5 * np.pi)
            runtime.close_gripper()
            runtime.move_to(
                [
                    float(approach_xy[0]),
                    float(approach_xy[1]),
                    contact_z + HOOK_APPROACH_M,
                ],
                yaw=yaw,
            )
            runtime.move_to(
                [
                    float(approach_xy[0]),
                    float(approach_xy[1]),
                    contact_z,
                ],
                yaw=yaw,
            )
            runtime.move_to(
                [
                    float(contact_xy[0]),
                    float(contact_xy[1]),
                    contact_z,
                ],
                yaw=yaw,
            )
            report["preflight_only"] = False
            report["handle_retained"] = False
            report["contact_mode"] = "closed_finger_external_face_push"
            report["force_empty_after_contact"] = True
            report["contact_execution"] = {
                "source": "external_face_normal",
                "approach_xy": [round(float(v), 4) for v in approach_xy],
                "contact_xy": [round(float(v), 4) for v in contact_xy],
            }
            for xy in path:
                runtime.move_to([float(xy[0]), float(xy[1]), contact_z], yaw=yaw)
            runtime.move_to(
                [
                    float(retreat_xy[0]),
                    float(retreat_xy[1]),
                    contact_z,
                ],
                yaw=yaw,
            )
            runtime.open_gripper()
            runtime.move_to(
                [
                    float(retreat_xy[0]),
                    float(retreat_xy[1]),
                    contact_z + 0.10,
                ],
                yaw=yaw,
            )
            ee = np.asarray([contact_xy[0], contact_xy[1], contact_z], dtype=float)
            report["gripper_opening_after_contact"] = 0.0
            report["thin_contact_proxy"] = False
            report["gripper_opening_after_path"] = round(
                float(runtime.ee_pose().get("gripper_opening", 0.0)), 4
            )
        elif closed_panel_sweep:
            start_xy = np.asarray(handle_xy, dtype=float)
            first_tangent = _unit(np.asarray(path[0], dtype=float) - start_xy)
            if first_tangent is None:
                return finish(False, "no_motion_axis", "door arc has no contact tangent")
            approach_xy = start_xy - first_tangent * FACE_PUSH_APPROACH_M
            contact_xy = start_xy - first_tangent * FACE_PUSH_CONTACT_GAP_M
            contact_z = float(handle.metadata.get("top_z", handle.pose[2]))
            yaw = float(np.arctan2(first_tangent[1], first_tangent[0]) + 0.5 * np.pi)
            runtime.close_gripper()
            runtime.move_to(
                [
                    float(approach_xy[0]),
                    float(approach_xy[1]),
                    contact_z + HOOK_APPROACH_M,
                ],
                yaw=yaw,
            )
            runtime.move_to(
                [
                    float(approach_xy[0]),
                    float(approach_xy[1]),
                    contact_z,
                ],
                yaw=yaw,
            )
            runtime.move_to(
                [
                    float(contact_xy[0]),
                    float(contact_xy[1]),
                    contact_z,
                ],
                yaw=yaw,
            )
            report["preflight_only"] = False
            report["handle_retained"] = False
            report["contact_mode"] = "closed_finger_external_panel_sweep"
            report["force_empty_after_contact"] = True
            previous_xy = start_xy
            for xy in path:
                tangent = _unit(np.asarray(xy, dtype=float) - previous_xy)
                if tangent is not None:
                    yaw = float(np.arctan2(tangent[1], tangent[0]) + 0.5 * np.pi)
                runtime.move_to([float(xy[0]), float(xy[1]), contact_z], yaw=yaw)
                previous_xy = np.asarray(xy, dtype=float)
            final_tangent = _unit(np.asarray(path[-1]) - np.asarray(path[-2]))
            if final_tangent is None:
                final_tangent = first_tangent
            retreat_xy = np.asarray(path[-1]) - final_tangent * FACE_PUSH_APPROACH_M
            runtime.move_to(
                [
                    float(retreat_xy[0]),
                    float(retreat_xy[1]),
                    contact_z,
                ],
                yaw=yaw,
            )
            runtime.open_gripper()
            runtime.move_to(
                [
                    float(retreat_xy[0]),
                    float(retreat_xy[1]),
                    contact_z + 0.10,
                ],
                yaw=yaw,
            )
            ee = np.asarray([contact_xy[0], contact_xy[1], contact_z], dtype=float)
            report["gripper_opening_after_contact"] = 0.0
            report["thin_contact_proxy"] = False
            report["gripper_opening_after_path"] = round(
                float(runtime.ee_pose().get("gripper_opening", 0.0)), 4
            )
        elif hook_primary:
            runtime.open_gripper()
            report["preflight_only"] = False
            report["contact_execution"] = {
                "source": "geometry_first_hook",
                "reason": "thin prismatic bar does not require force closure",
            }
            report["handle_retained"] = False
            held = False
            ee = np.asarray([handle_xy[0], handle_xy[1], float(handle.pose[2])], dtype=float)
        else:
            runtime.open_gripper()
            contact_execution = runtime.contact_grasp(handle, strategy=requested_contact, lift=0.0)
            if isinstance(contact_execution, dict):
                report["contact_execution"] = contact_execution
            report["preflight_only"] = False
            finger_retention = bool(runtime.verify_grasp(""))
            learned_contact = bool(
                isinstance(contact_execution, dict)
                and contact_execution.get("source") in {"graspgen_6d", "side_bar_6d"}
                and float(contact_execution.get("pose_error_cm", 0.0)) <= 2.5
            )
            # A compact drawer bar can be thinner than the generic 2 cm
            # finger-gap threshold.  When a geometry-aligned top-down/PCA
            # contact actually reached the measured bar, the bar is between
            # the fingers even if the opening proxy looks like an empty close.
            # This is contact evidence, not a state verdict; post-motion visual
            # and metric checks still decide whether the mechanism moved.
            reached_thin_bar = bool(
                resolved == "prismatic"
                and desired == "open"
                and thin_bar
                and top_down_fingertip_contact(contact_execution, float(handle.pose[2]))
            )
            # A thin bar may leave less than the generic 2 cm finger-gap
            # threshold even under a valid model grasp. GraspGen admission has
            # already checked a high-quality candidate's signed approach and
            # fingertip contact against the refined RGB-D band. Let that
            # contact execute one measured path; post-motion handle progress,
            # not this proxy, decides whether it worked.
            # Aligned fingertips prove that the wrist reached the handle, but
            # not that a <2 cm finger gap retained a 3 cm bar.  Use the former
            # to seed a correctly referenced hook sweep; reserve ``held`` for
            # actual force-closure evidence.
            held = bool(finger_retention or learned_contact)
            report["retention_evidence"] = (
                "finger_gap"
                if finger_retention
                else "verified_side_contact"
                if learned_contact
                else "none"
            )
            if reached_thin_bar:
                report["retention_evidence"] = "geometry_aligned_top_down_fingertip_contact"
            report["handle_retained"] = held
            ee = np.asarray(runtime.ee_pose()["position"], dtype=float)
        held = bool(report.get("handle_retained", False))
        report["gripper_opening_after_contact"] = round(
            float(runtime.ee_pose().get("gripper_opening", 0.0)), 4
        )
        thin_mechanism_contact = bool(
            contact_fit.get("source") == "rgbd_protruding_band"
            and min(float(v) for v in handle_extent[:2]) <= 0.022
        )
        report["thin_contact_proxy"] = thin_mechanism_contact
        if closed_face_push or closed_panel_sweep:
            pass
        elif desired == "open" and not held:
            if resolved != "prismatic":
                runtime.open_gripper()
                return finish(
                    False,
                    "contact_failed",
                    "the hinged handle was not retained for its arc",
                )
            # A thin drawer bar may leave a small non-zero finger gap without
            # actually being retained. Treating that proxy as a grasp can make
            # the direct outward path press the bar inward. If retention is
            # not real, put the compact
            # closed fingertips behind the bar and sweep outward: the bar
            # becomes the hook. This is object-agnostic contact geometry.
            outward = np.asarray(report["motion_axis"], dtype=float)
            # Hook geometry is expressed at the fingertips, while move_to is
            # expressed at the Panda grip-site / wrist reference.  The former
            # code set the wrist to the bar's Z, placing the actual fingers a
            # full finger length below the handle.  Reuse a successfully
            # aligned top-down wrist height when available; otherwise convert
            # the RGB-D bar height through the robot's fingertip offset.
            aligned_wrist_z = None
            if isinstance(contact_execution, dict):
                try:
                    reached_xyz = np.asarray(
                        contact_execution.get("reached_position"), dtype=float
                    ).reshape(3)
                    commanded_xyz = np.asarray(
                        contact_execution.get("position"), dtype=float
                    ).reshape(3)
                    if (
                        contact_execution.get("source") in {"pca_axis", "top_down"}
                        and float(np.linalg.norm(reached_xyz[:2] - commanded_xyz[:2]))
                        <= FINGERTIP_PLANAR_TOLERANCE_M
                        and FINGERTIP_REACH_MIN_M
                        <= float(reached_xyz[2] - handle.pose[2])
                        <= FINGERTIP_REACH_MAX_M
                    ):
                        aligned_wrist_z = float(reached_xyz[2])
                except (TypeError, ValueError):
                    aligned_wrist_z = None
            ee[2] = (
                float(aligned_wrist_z)
                if aligned_wrist_z is not None
                else float(handle.pose[2]) + TOP_DOWN_FINGERTIP_OFFSET_M
            )
            planar_extent = sorted(float(v) for v in handle_extent[:2])
            hook_offset = float(np.clip(1.5 * planar_extent[0], MIN_HOOK_OFFSET_M, HOOK_OFFSET_M))
            start_xy = handle_xy - outward * hook_offset
            hook_path = path - outward[None, :] * hook_offset
            yaw = float(np.arctan2(outward[1], outward[0]) + 0.5 * np.pi)
            runtime.open_gripper()
            runtime.move_to(
                [float(start_xy[0]), float(start_xy[1]), float(ee[2] + HOOK_APPROACH_M)],
                yaw=yaw,
            )
            runtime.close_gripper()
            runtime.move_to([float(start_xy[0]), float(start_xy[1]), float(ee[2])], yaw=yaw)
            for xy in hook_path:
                runtime.move_to([float(xy[0]), float(xy[1]), float(ee[2])], yaw=yaw)
            path = hook_path
            report["contact_mode"] = "hook_pull"
            report["hook_offset_cm"] = round(hook_offset * 100, 1)
            report["hook_wrist_z"] = round(float(ee[2]), 4)
            report["hook_contact_z"] = round(float(handle.pose[2]), 4)
        else:
            report["contact_mode"] = (
                "retained_grasp"
                if held
                else "thin_contact_grasp"
                if thin_mechanism_contact
                else "contact_push"
            )
            start_quat = np.asarray(
                runtime.ee_pose().get("quat", [0.0, 1.0, 0.0, 0.0]),
                dtype=float,
            ).reshape(4)
            for xy in path:
                position = [float(xy[0]), float(xy[1]), float(ee[2])]
                if resolved == "revolute":
                    swept = signed_planar_angle_deg(handle_xy, xy, hinge)
                    quat = rotate_quaternion_world_z(
                        start_quat,
                        0.0 if swept is None else np.radians(swept),
                    )
                    runtime.move_to(position, quat=quat)
                else:
                    runtime.move_to(position)
        if not closed_face_push and not closed_panel_sweep:
            report["gripper_opening_after_path"] = round(
                float(runtime.ee_pose().get("gripper_opening", 0.0)), 4
            )
            runtime.open_gripper()
            runtime.move_to([float(path[-1, 0]), float(path[-1, 1]), float(ee[2] + 0.10)])
    except Exception as exc:
        try:
            runtime.recover()
        except Exception:
            pass
        return finish(
            False, "articulation_failed", f"contact path raised {type(exc).__name__}: {exc}"
        )

    after, after_grounder = _reground_visible(runtime, handle.label, contact=True)
    report["after_handle_grounder"] = after_grounder
    # The cabinet/appliance body is the fixed world reference of its joint.
    # Re-grounding that broad structure after the arm moves adds a semantic
    # VLM call and, under occlusion, can shift its mask centroid by tens of
    # centimetres.  World-frame RGB-D poses already share a coordinate frame,
    # so the pre-contact fixture is the more faithful motion origin.
    refreshed_fixture = fixture
    report["fixture_motion_reference"] = "precontact_static_world_pose"
    displacement = None
    directed = None
    measured_hinge_rotation = None
    if _usable(after, contact=True):
        delta = np.asarray(after.pose[:2], dtype=float) - handle_xy
        displacement = float(np.linalg.norm(delta))
        fixture_reliable, fixture_jump, fixture_limit = fixture_refresh_gate(
            fixture_xy,
            np.asarray(refreshed_fixture.pose[:2], dtype=float),
            fixture.metadata.get("extent"),
        )
        # A fixture is the reference frame of an articulation measurement.  A
        # post-action detector can jump from the cabinet to a wall or the arm;
        # subtracting that jump produces metres of fictitious drawer travel.
        # Admit normal contact-induced fixture motion, but reject a change
        # larger than both a scene-independent floor and half the measured
        # fixture footprint (with a conservative absolute cap).
        report["fixture_grounding_jump_cm"] = round(fixture_jump * 100, 1)
        report["fixture_grounding_max_jump_cm"] = round(fixture_limit * 100, 1)
        report["fixture_grounding_reliable"] = fixture_reliable
        if resolved == "prismatic":
            directed = relative_planar_progress(
                handle_xy,
                fixture_xy,
                np.asarray(after.pose[:2], dtype=float),
                np.asarray(refreshed_fixture.pose[:2], dtype=float),
                np.asarray(report["motion_axis"], dtype=float),
            )
            report["fixture_displacement_cm"] = round(
                fixture_jump * 100,
                1,
            )
        else:
            before_distance = float(np.linalg.norm(handle_xy - fixture_xy))
            after_distance = float(np.linalg.norm(np.asarray(after.pose[:2]) - fixture_xy))
            directed = after_distance - before_distance
            measured_hinge_rotation = signed_planar_angle_deg(
                handle_xy,
                np.asarray(after.pose[:2], dtype=float),
                np.asarray(report["hinge_xy"], dtype=float),
            )
        report["handle_after"] = [round(float(v), 4) for v in after.pose]
        height_reliable = abs(float(after.pose[2]) - float(handle.pose[2])) <= 0.03
        path_envelope_reliable = bool(
            resolved != "prismatic"
            or prismatic_measurement_within_command(displacement, directed, stroke)
        )
        report["motion_path_envelope_reliable"] = path_envelope_reliable
        report["motion_path_envelope_max_cm"] = (
            round((stroke + PRISMATIC_MEASUREMENT_MARGIN_M) * 100, 1)
            if resolved == "prismatic"
            else None
        )
        report["motion_measurement_reliable"] = bool(
            fixture_reliable and height_reliable and path_envelope_reliable
        )
        if not fixture_reliable:
            report["motion_measurement_rejection"] = (
                "fixture grounding moved beyond its physically plausible gate"
            )
        elif not height_reliable:
            report["motion_measurement_rejection"] = (
                "handle grounding changed height beyond its contact envelope"
            )
        elif not path_envelope_reliable:
            report["motion_measurement_rejection"] = (
                "handle grounding moved outside the commanded path envelope"
            )
    report["measured_displacement_cm"] = (
        None if displacement is None else round(displacement * 100, 1)
    )
    report["directed_progress_cm"] = None if directed is None else round(directed * 100, 1)
    report["measured_hinge_rotation_deg"] = (
        None if measured_hinge_rotation is None else round(measured_hinge_rotation, 1)
    )

    after_state = ""
    report["visual_state_after"] = after_state
    report["semantic_state_verification"] = "deferred_to_visual_agent"
    visual_ok = False
    if resolved == "prismatic":
        # A few centimetres prove only that the contact moved the mechanism,
        # not that an endpoint state was reached.  Require most of the
        # requested stroke here; Session.check_state can additionally combine
        # several partial strokes or detect a contacted endpoint stall.
        progress = None if directed is None else (directed if desired == "open" else -directed)
        travel_match = bool(
            progress is not None
            and report.get("motion_measurement_reliable", False)
            and progress >= 0.80 * stroke
        )
        # Closing is a compression task: a deliberately over-long Cartesian
        # command is absorbed by the drawer's mechanical stop.  Requiring 80%
        # of that *command* therefore rejects a correctly closed short drawer.
        # A before/after visual transition plus substantial, reliable inward
        # travel is independent endpoint evidence.  Keep the threshold tied
        # to the measured stroke (rather than a cabinet/task constant), and do
        # not admit visual-only or millimetre-scale contacts.
        compressive_endpoint = bool(
            desired == "closed"
            and before_state == state_options[1]
            and visual_ok
            and progress is not None
            and report.get("motion_measurement_reliable", False)
            and progress >= max(0.04, 0.55 * stroke)
        )
        report["compressive_endpoint_consensus"] = compressive_endpoint
        motion_ok = bool(travel_match or compressive_endpoint)
    else:
        planned_rotation = report.get("planned_hinge_rotation_deg")
        motion_ok = bool(
            measured_hinge_rotation is not None
            and report.get("motion_measurement_reliable", False)
            and planned_rotation is not None
            and np.sign(measured_hinge_rotation) == np.sign(float(planned_rotation))
            and abs(measured_hinge_rotation) >= MIN_HINGE_PROGRESS_DEG
        )
    report["visual_goal_match"] = bool(visual_ok)
    report["motion_goal_match"] = bool(motion_ok)
    # Binary appearance is useful supporting evidence, but articulated state
    # is frequently occluded and VLMs call a partly-open door "open".  Do not
    # let that weak cue overrule measured insufficient travel.
    if motion_ok:
        return finish(True, message="fresh visual/measured feedback supports the requested state")
    return finish(
        False,
        "verification_inconclusive",
        "the contact path completed but fresh visual feedback did not confirm the goal",
    )
