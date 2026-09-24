"""A small, measured image-space push for post-placement correction.

This primitive is intentionally narrower than free-form Cartesian control.
The caller says how the object should move in the current camera image; the
policy measures the object's footprint and support height, makes contact just
behind it, advances at most five centimetres, and verifies actual displacement
from a fresh visual grounding.  It never reads simulator state or predicates.
"""

from __future__ import annotations

import numpy as np

from racap.backends.vlm import PerceptionUnavailableError
from racap.contracts import SkillResult
from racap.policy_api.pickplace_api import (
    MAX_EXTENT,
    MIN_EXTENT,
    MIN_POINTS,
    _ground_visible,
    _plausible,
    _surface_height,
)

MAX_PUSH_M = 0.05
MIN_PUSH_M = 0.005
APPROACH_CLEARANCE = 0.10
CONTACT_GAP = 0.008


def _cloud(runtime, label: str) -> np.ndarray:
    try:
        points = np.asarray(runtime.object_points(label), dtype=float).reshape(-1, 3)
    except Exception:
        return np.zeros((0, 3), dtype=float)
    return points[np.isfinite(points).all(axis=1)]


def _world_delta(runtime, nudge: tuple[float, float]) -> np.ndarray:
    right_m, up_m = (float(v) for v in nudge)
    axes = runtime.image_axes()
    right = np.asarray(axes["right"], dtype=float)[:2]
    down = np.asarray(axes["down"], dtype=float)[:2]
    right /= max(float(np.linalg.norm(right)), 1e-6)
    down /= max(float(np.linalg.norm(down)), 1e-6)
    return right * right_m - down * up_m


def push(
    runtime,
    pick_label: str,
    place_label: str = "",
    *,
    nudge: tuple[float, float] = (0.0, 0.0),
    expected_pose: tuple[float, float, float] | None = None,
) -> SkillResult:
    """Nudge a visible object without re-grasping it.

    Args:
        pick_label: Object to contact.
        place_label: Destination phrase, retained for trace/readability.  It is
            not grounded again and cannot move the push target.
        nudge: Requested object displacement ``(right, up)`` in image-space
            metres.  Magnitude is clamped to 5 cm.
    """
    trace: list[dict[str, object]] = []
    report: dict[str, object] = {"params": {"nudge": list(nudge)}}

    def result(success: bool, failure_mode: str, message: str) -> SkillResult:
        return SkillResult(
            success=success,
            status="success" if success else "retryable_failure",
            pick_label=pick_label,
            place_label=place_label,
            attempts=1,
            strategy="measured_push",
            failure_mode=failure_mode,
            message=message,
            trace=trace,
            report=report,
        )

    try:
        values = tuple(float(v) for v in nudge)
        if len(values) != 2:
            raise ValueError
        desired = _world_delta(runtime, values)
    except Exception:
        return result(False, "unsafe_push", "nudge must be two image-space metres")
    distance = float(np.linalg.norm(desired))
    if distance < MIN_PUSH_M:
        return result(False, "unsafe_push", "requested displacement is too small to measure")
    if distance > MAX_PUSH_M:
        desired *= MAX_PUSH_M / distance
        distance = MAX_PUSH_M
    direction = desired / distance

    try:
        before = runtime.localize_many([pick_label]).get(pick_label)
        points = _cloud(runtime, pick_label)
    except PerceptionUnavailableError:
        raise
    except Exception as exc:
        return result(False, "not_grounded", f"grounding raised {type(exc).__name__}: {exc}")
    if expected_pose is not None:
        expected = np.asarray(expected_pose[:3], dtype=float)
        disagreement = (
            float(np.linalg.norm(np.asarray(before.pose, dtype=float) - expected))
            if before is not None
            else float("inf")
        )
        if disagreement > 0.10:
            try:
                alternate = _ground_visible(runtime, pick_label)
                alternate_points = _cloud(runtime, pick_label)
            except PerceptionUnavailableError:
                raise
            except Exception:
                alternate, alternate_points = None, np.zeros((0, 3))
            if (
                alternate is not None
                and _plausible(alternate)
                and float(np.linalg.norm(np.asarray(alternate.pose, dtype=float) - expected))
                < disagreement
            ):
                before, points = alternate, alternate_points
                disagreement = float(
                    np.linalg.norm(np.asarray(before.pose, dtype=float) - expected)
                )
        report["expected_pose"] = [round(float(v), 4) for v in expected]
        report["grounding_disagreement_cm"] = round(disagreement * 100, 1)
        if disagreement > 0.10:
            return result(
                False, "not_grounded", "fresh grounding disagrees with the last reliable check"
            )
    if not _plausible(before):
        return result(False, "not_grounded", f"no usable grounding for {pick_label!r}")
    if points.shape[0] < MIN_POINTS:
        return result(False, "not_grounded", "object point cloud is too sparse to push")

    extent = before.metadata.get("extent") or [0.0, 0.0, 0.0]
    largest = max(float(v) for v in extent)
    if not MIN_EXTENT < largest < MAX_EXTENT:
        return result(False, "unsafe_push", "object geometry is implausible")

    centre = np.asarray(before.pose[:2], dtype=float)
    projected = (points[:, :2] - centre) @ direction
    back_radius = max(0.008, -float(np.percentile(projected, 5)))
    front_radius = max(0.008, float(np.percentile(projected, 95)))
    support = _surface_height(runtime, before, fallback=float(np.percentile(points[:, 2], 3)))
    top = float(before.metadata.get("top_z", np.percentile(points[:, 2], 97)))
    height = float(np.clip(top - support, 0.01, 0.20))
    # A top-down hand has to put the *finger length*, not merely the fingertip,
    # alongside the object's wall.  Video showed that top-1cm let the closed
    # fingertips sweep above a bowl without transferring force.  Work around
    # the upper third of the visible wall, while retaining 12mm clearance from
    # the measured support so a thin object cannot drive the hand into it.
    contact_z = max(support + 0.012, top - float(np.clip(0.40 * height, 0.020, 0.032)))

    contact_xy = centre - direction * (back_radius + CONTACT_GAP)
    # A pusher that starts just behind the object should travel only far
    # enough to take up the contact gap and then carry the object by the
    # requested displacement.  Sweeping through the full object diameter
    # makes the arm collide with the support before useful lateral contact
    # (and badly over-pushes on the occasions where contact succeeds).
    finish_xy = contact_xy + direction * (CONTACT_GAP + distance)
    # The runtime yaw denotes the jaw-closing axis.  Rotating it 90 degrees
    # presents the compact closed fingertip face across the translation.  A
    # yaw aligned with translation visibly passed beside the bowl without
    # contact; the transverse pose established contact reliably.
    yaw = float(np.arctan2(direction[1], direction[0]) + 0.5 * np.pi)
    report.update(
        {
            "before_pose": [round(float(v), 4) for v in before.pose],
            "commanded_nudge": [round(float(v), 4) for v in values],
            "commanded_distance_cm": round(distance * 100, 1),
            "contact_xy": [round(float(v), 4) for v in contact_xy],
            "finish_xy": [round(float(v), 4) for v in finish_xy],
            "contact_z": round(contact_z, 4),
        }
    )
    trace.append(
        {
            "event": "push_plan",
            "distance_cm": round(distance * 100, 1),
            "object_height_cm": round(height * 100, 1),
            "back_radius_cm": round(back_radius * 100, 1),
            "front_radius_cm": round(front_radius * 100, 1),
            "contact_z": round(contact_z, 4),
        }
    )

    try:
        runtime.move_to([contact_xy[0], contact_xy[1], contact_z + APPROACH_CLEARANCE], yaw=yaw)
        # Close above the scene, then descend.  At the calibrated wall height
        # the compact, symmetric fingertip pair transfers force through the
        # intended centre line.  An open pair did make contact, but video and
        # post-action grounding showed one finger catching the bowl alone and
        # sending it 1.5 cm sideways.
        runtime.close_gripper()
        runtime.move_to([contact_xy[0], contact_xy[1], contact_z])
        runtime.move_to([finish_xy[0], finish_xy[1], contact_z])
        runtime.move_to([finish_xy[0], finish_xy[1], contact_z + APPROACH_CLEARANCE])
        runtime.recover()
    except Exception as exc:
        try:
            runtime.recover()
        except Exception:
            pass
        return result(False, "push_failed", f"push raised {type(exc).__name__}: {exc}")

    try:
        after = _ground_visible(runtime, pick_label)
    except Exception:
        after = None
    if not _plausible(after):
        report["verification_reliable"] = False
        return result(
            False,
            "verification_inconclusive",
            "push completed but the object could not be re-localized",
        )

    actual = np.asarray(after.pose[:2], dtype=float) - centre
    if float(np.linalg.norm(actual)) > max(0.12, 3.0 * distance):
        report.update(
            {
                "after_pose": [round(float(v), 4) for v in after.pose],
                "verification_reliable": False,
                "rejected_shift_cm": round(float(np.linalg.norm(actual)) * 100, 1),
            }
        )
        return result(
            False, "verification_inconclusive", "post-push grounding jumped to a different object"
        )
    along = float(actual @ direction)
    sideways = float(actual @ np.array([-direction[1], direction[0]]))
    report.update(
        {
            "after_pose": [round(float(v), 4) for v in after.pose],
            "actual_delta_xy": [round(float(v), 4) for v in actual],
            "actual_along_cm": round(along * 100, 1),
            "actual_sideways_cm": round(sideways * 100, 1),
            "verification_reliable": True,
        }
    )
    trace.append(
        {
            "event": "push_verify",
            "along_cm": round(along * 100, 1),
            "sideways_cm": round(sideways * 100, 1),
        }
    )
    if along < 0.003:
        return result(
            False, "no_motion", "contact did not move the object in the requested direction"
        )
    if abs(sideways) > max(0.012, 2.0 * along):
        return result(
            False,
            "wrong_direction",
            "contact moved the object mainly sideways; another push is unsafe",
        )
    return result(True, "", f"object moved {along * 100:.1f} cm along the requested direction")
