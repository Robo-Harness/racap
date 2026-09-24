import numpy as np

import racap.policy_api.insert_api as insert
from racap.contracts import GroundedTarget
from racap.policy_api.insert_api import (
    Footprint,
    _broad_container_transport_is_sufficient,
    _frontload_geometry,
    feasible_placement,
)


def test_roomy_compact_drawer_fit_prefers_broad_transport() -> None:
    compact = Footprint(
        centre=np.zeros(2),
        vertices=_rectangle(0.10, 0.065),
        long_axis_rad=0.0,
        size=(0.10, 0.065),
    )
    thin = Footprint(
        centre=np.zeros(2),
        vertices=_rectangle(0.10, 0.018),
        long_axis_rad=0.0,
        size=(0.10, 0.018),
    )
    plan = {"clearance_m": 0.017}

    assert _broad_container_transport_is_sufficient(
        "top drawer of cabinet",
        compact,
        plan,
    )
    assert not _broad_container_transport_is_sufficient(
        "top drawer of cabinet",
        thin,
        plan,
    )
    assert not _broad_container_transport_is_sufficient(
        "front compartment of caddy",
        compact,
        plan,
    )


def _rectangle(width: float, height: float) -> np.ndarray:
    return np.asarray(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )


def test_feasible_placement_aligns_an_elongated_footprint() -> None:
    opening = _rectangle(0.12, 0.045)
    footprint = Footprint(
        centre=np.zeros(2),
        # The object starts with its long side along world y.
        vertices=_rectangle(0.025, 0.08),
        long_axis_rad=np.pi / 2,
        size=(0.08, 0.025),
    )

    plan = feasible_placement(opening, footprint, uncertainty_m=0.003, preferred_xy=np.zeros(2))

    assert plan is not None
    assert np.linalg.norm(plan["centre_xy"]) < 0.005
    assert plan["clearance_m"] >= 0.003
    # The long object axis is rotated onto the opening's world-x long axis.
    assert abs(np.sin(plan["object_axis_rad"])) < 0.05


def test_roomy_opening_avoids_gratuitous_in_hand_rotation() -> None:
    opening = _rectangle(0.15, 0.15)
    footprint = Footprint(
        centre=np.zeros(2),
        vertices=_rectangle(0.075, 0.045),
        long_axis_rad=np.deg2rad(28.0),
        size=(0.075, 0.045),
    )

    plan = feasible_placement(opening, footprint, uncertainty_m=0.004, preferred_xy=np.zeros(2))

    assert plan is not None
    assert abs(np.degrees(plan["rotation_rad"])) < 1.0


def test_frontload_aligns_book_with_shelf_depth_and_keeps_edge_grasp_outside() -> None:
    opening = _rectangle(0.30, 0.17)
    footprint = Footprint(
        centre=np.asarray([0.0, -0.25]),
        vertices=_rectangle(0.10, 0.02),
        long_axis_rad=0.0,
        size=(0.10, 0.02),
    )
    geometry = _frontload_geometry(opening, footprint, image_down_xy=np.asarray([0.0, -1.0]))

    assert geometry is not None
    assert np.allclose(geometry["outward_axis"], [0.0, -1.0], atol=1e-6)
    assert np.allclose(geometry["inward_axis"], [0.0, 1.0], atol=1e-6)
    assert np.isclose(geometry["edge_offset_m"], 0.038)
    plan = feasible_placement(
        opening,
        footprint,
        uncertainty_m=0.004,
        preferred_object_axis_rad=geometry["object_axis_rad"],
    )
    assert plan is not None
    assert abs(abs(np.sin(plan["object_axis_rad"])) - 1.0) < 0.05


def test_frontload_source_side_overrides_misleading_camera_down() -> None:
    opening = _rectangle(0.30, 0.17)
    footprint = Footprint(
        centre=np.asarray([0.0, 0.25]),
        vertices=_rectangle(0.10, 0.02),
        long_axis_rad=0.0,
        size=(0.10, 0.02),
    )

    geometry = _frontload_geometry(opening, footprint, image_down_xy=np.asarray([0.0, -1.0]))

    assert geometry is not None
    assert np.allclose(geometry["outward_axis"], [0.0, 1.0], atol=1e-6)
    assert geometry["outward_source"] == "source_free_space"


def test_feasible_placement_rejects_an_object_larger_than_every_yaw() -> None:
    opening = _rectangle(0.08, 0.05)
    footprint = Footprint(
        centre=np.zeros(2),
        vertices=_rectangle(0.10, 0.07),
        long_axis_rad=0.0,
        size=(0.10, 0.07),
    )

    assert feasible_placement(opening, footprint, uncertainty_m=0.002) is None


def test_uncertainty_erodes_an_otherwise_exact_fit() -> None:
    opening = _rectangle(0.10, 0.06)
    footprint = Footprint(
        centre=np.zeros(2),
        vertices=_rectangle(0.096, 0.056),
        long_axis_rad=0.0,
        size=(0.096, 0.056),
    )

    assert feasible_placement(opening, footprint, uncertainty_m=0.0) is not None
    assert feasible_placement(opening, footprint, uncertainty_m=0.003) is None


def test_held_feedback_uses_shape_prior_when_gripper_inflates_short_axis(
    monkeypatch,
) -> None:
    reference = Footprint(
        centre=np.zeros(2),
        vertices=_rectangle(0.106, 0.030),
        long_axis_rad=0.0,
        size=(0.106, 0.030),
    )
    angle = np.deg2rad(25.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    rows = []
    for x in np.linspace(-0.057, 0.057, 30):
        for y in np.linspace(-0.0355, 0.0355, 20):
            xy = np.asarray([x, y]) @ rotation.T + np.asarray([0.2, 0.1])
            rows.append([xy[0], xy[1], 0.2])
    points = np.asarray(rows)
    target = GroundedTarget(
        label="book",
        pose=(0.2, 0.1, 0.2),
        confidence=1.0,
        metadata={"n_points": len(points), "extent": [0.12, 0.08, 0.02]},
    )
    monkeypatch.setattr(insert, "_ground_visible", lambda runtime, label: target)
    monkeypatch.setattr(insert, "_cloud", lambda runtime, label: points)

    measured, reason = insert._measure_held_footprint(
        object(), "book", reference, np.asarray([0.2, 0.1])
    )

    assert measured is not None
    assert measured.size == reference.size
    assert "shape_prior_with_held_pose" in reason
    assert abs(insert._wrap_half_turn(measured.long_axis_rad - angle)) < 0.05


def test_held_feedback_prefers_local_near_hand_tracker(monkeypatch) -> None:
    reference = Footprint(
        centre=np.zeros(2),
        vertices=_rectangle(0.10, 0.02),
        long_axis_rad=0.0,
        size=(0.10, 0.02),
    )
    points = np.asarray(
        [[x, y, 0.20] for x in np.linspace(0.15, 0.25, 20) for y in np.linspace(0.09, 0.11, 8)]
    )

    class Runtime:
        def __init__(self):
            self.calls = []

        def track_near_hand(self, label, **kwargs):
            self.calls.append((label, kwargs))
            return GroundedTarget(
                label=label,
                pose=(0.2, 0.1, 0.2),
                confidence=1.0,
                metadata={"n_points": len(points), "extent": [0.10, 0.02, 0.01]},
            )

    runtime = Runtime()
    cloud_labels = []

    def cloud(runtime, label):
        del runtime
        cloud_labels.append(label)
        return points

    monkeypatch.setattr(insert, "_cloud", cloud)
    monkeypatch.setattr(
        insert,
        "_ground_visible",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global grounding must not run")
        ),
    )

    measured, reason = insert._measure_held_footprint(
        runtime, "book on the right", reference, np.asarray([0.2, 0.1])
    )

    assert measured is not None, reason
    assert "shape_prior_with_held_pose" in reason
    assert runtime.calls == [("book", {"reference_size": reference.size, "radius_m": 0.18})]
    assert cloud_labels == ["book"]


def test_held_visual_servo_identifies_reversed_yaw_response(monkeypatch) -> None:
    reference = Footprint(
        centre=np.zeros(2),
        vertices=_rectangle(0.08, 0.02),
        long_axis_rad=0.0,
        size=(0.08, 0.02),
    )

    class Runtime:
        hand_yaw = 0.0
        object_yaw = 0.0

        def move_to(self, position, yaw=None):
            del position
            if yaw is not None:
                delta = yaw - self.hand_yaw
                self.object_yaw = insert._wrap_half_turn(self.object_yaw - 0.5 * delta)
                self.hand_yaw = yaw

    runtime = Runtime()

    def measure(runtime, label, reference, expected_xy):
        del label, expected_xy
        delta = insert._wrap_half_turn(runtime.object_yaw - reference.long_axis_rad)
        return Footprint(
            centre=np.zeros(2),
            vertices=reference.vertices @ insert._rotation(delta).T,
            long_axis_rad=runtime.object_yaw,
            size=reference.size,
        ), "synthetic"

    monkeypatch.setattr(insert, "_measure_held_footprint", measure)
    _, _, _, info = insert._visual_servo_held_pose(
        runtime,
        "book",
        reference,
        _rectangle(0.05, 0.12),
        np.zeros(2),
        np.zeros(2),
        0.0,
        0.4,
        0.003,
    )

    assert info["held_visual_feedback"] == "corrected"
    assert abs(info["visual_servo_yaw_response_gain"] + 0.5) < 0.05
    assert abs(info["visual_servo_residual_yaw_deg"]) <= 10.0


def test_held_feedback_removes_high_gripper_points(monkeypatch) -> None:
    reference = Footprint(
        centre=np.zeros(2),
        vertices=_rectangle(0.10, 0.025),
        long_axis_rad=0.0,
        size=(0.10, 0.025),
    )
    body = np.asarray(
        [
            [x, y, z]
            for z in np.linspace(0.10, 0.20, 8)
            for x in np.linspace(0.15, 0.25, 20)
            for y in np.linspace(0.0875, 0.1125, 8)
        ]
    )
    gripper = np.asarray(
        [
            [x, y, z]
            for z in np.linspace(0.205, 0.23, 4)
            for x in np.linspace(0.15, 0.25, 15)
            for y in np.linspace(0.06, 0.14, 12)
        ]
    )
    points = np.concatenate([body, gripper], axis=0)
    target = GroundedTarget(
        label="book",
        pose=(0.2, 0.1, 0.16),
        confidence=1.0,
        metadata={"n_points": len(points), "extent": [0.10, 0.08, 0.13]},
    )
    monkeypatch.setattr(insert, "_ground_visible", lambda runtime, label: target)
    monkeypatch.setattr(insert, "_cloud", lambda runtime, label: points)

    measured, reason = insert._measure_held_footprint(
        object(), "book", reference, np.asarray([0.2, 0.1])
    )

    assert measured is not None
    assert "shape_prior_with_held_pose" in reason
    assert abs(measured.long_axis_rad) < 0.05


class _PreflightRuntime:
    def __init__(self) -> None:
        self.grasp_calls = 0
        rows = []
        for x in np.linspace(-0.05, 0.05, 12):
            for y in np.linspace(-0.035, 0.035, 10):
                rows.append([x, y, 0.04])
        self.points = np.asarray(rows)

    def localize_many(self, labels, close_view=False):
        del close_view
        label = labels[0]
        return {
            label: GroundedTarget(
                label=label,
                pose=(0.0, 0.0, 0.04),
                confidence=1.0,
                metadata={
                    "n_points": len(self.points),
                    "extent": [0.10, 0.07, 0.02],
                    "top_z": 0.05,
                    "bbox": [10, 10, 40, 40],
                },
            )
        }

    def object_points(self, label):
        del label
        return self.points

    def grasp(self, target, *, strategy):
        del target, strategy
        self.grasp_calls += 1


def test_no_feasible_insertion_is_a_motion_free_preflight_failure(
    monkeypatch,
) -> None:
    runtime = _PreflightRuntime()
    goal = GroundedTarget(
        label="slot",
        pose=(0.2, 0.1, 0.01),
        kind="region",
        confidence=1.0,
        metadata={
            "n_points": 500,
            "extent": [0.08, 0.05, 0.06],
            "top_z": 0.07,
            "floor_z": 0.01,
            "synthetic": True,
            "region_polygon_xy": (_rectangle(0.08, 0.05) + np.asarray([0.2, 0.1])).tolist(),
        },
    )
    monkeypatch.setattr(
        insert,
        "_ground_destination",
        lambda runtime, phrase, trace, near, mode, **kwargs: goal,
    )

    result = insert.insert(runtime, "large book", "the slot")

    assert result.success is False
    assert result.failure_mode == "no_feasible_insertion"
    assert result.report["preflight_only"] is True
    assert "side_or_edge_grasp_required" in result.report["recommended_recovery"]
    assert runtime.grasp_calls == 0


def test_under_shelf_destination_grounds_parent_lower_compartment(
    monkeypatch,
) -> None:
    runtime = _PreflightRuntime()
    goal = GroundedTarget(
        label="lower shelf compartment",
        pose=(0.2, 0.1, 0.01),
        kind="region",
        confidence=1.0,
        metadata={
            "n_points": 500,
            "extent": [0.08, 0.05, 0.06],
            "top_z": 0.07,
            "floor_z": 0.01,
            "synthetic": True,
            "region_polygon_xy": (_rectangle(0.08, 0.05) + np.asarray([0.2, 0.1])).tolist(),
        },
    )
    seen: dict[str, str] = {}

    def ground(runtime, phrase, trace, near, mode, **kwargs):
        del runtime, trace, near, mode, kwargs
        seen["phrase"] = phrase
        return goal

    monkeypatch.setattr(insert, "_ground_destination", ground)

    insert.insert(
        runtime,
        "large book",
        "under the cabinet shelf",
        frontload=True,
        destination_description="under the cabinet shelf",
    )

    assert "bottommost open shelf compartment" in seen["phrase"]
    assert "within cabinet shelf" in seen["phrase"]
    assert "within under" not in seen["phrase"]
