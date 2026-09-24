from __future__ import annotations

import inspect
import numpy as np

import racap.agent.full_react as full_react
import racap.policy_api.pickplace_api as pickplace
from racap.contracts import GroundedTarget
from racap.agent.tools import Session, Step
from racap.policy_api.actuate_control_api import (
    _topdown_yaw,
    raised_control_contact,
)
from racap.policy_api.articulate_api import (
    DEFAULT_HINGE_ANGLE_DEG,
    DEFAULT_LINEAR_STROKE_M,
    fixture_refresh_gate,
    hinge_from_footprint,
    hinge_from_moving_panel,
    hinge_waypoints,
    linear_waypoints,
    closing_face_push_points,
    protruding_contact,
    prismatic_stroke,
    prismatic_axis_from_opening_boundary,
    prismatic_axis_from_opening_fixture,
    prismatic_axis_from_opening_probe,
    prismatic_axis_from_footprint,
    prismatic_measurement_within_command,
    _ground_moving_panel,
    _reground_visible,
    _usable as articulation_target_usable,
    relative_planar_progress,
    rotate_quaternion_world_z,
    signed_planar_angle_deg,
    top_down_fingertip_contact,
)
from racap.policy_api.pickplace_api import (
    _appearance_label,
    _attempts,
    _carry_offset,
    _centre_over,
    _confirm_identity,
    _edge_grasp_height,
    _grasp_height,
    _ground_verified_pick,
    _ground_destination,
    _has_transient_spatial_identity,
    _held_object_left_the_table,
    _inward_feasible_nudge,
    _low_grasp_height,
    _place_carefully,
    _planar_boundary_repair,
    _refresh_pick_after_contact,
    _robust_object_top,
    _return_unintended_grasp,
    _surface_height,
    _upper_surface,
)
from racap.policy_api.stack_api import stack_plan
from racap.backends.libero import (
    LiberoPrimitiveRuntime,
    _pca_grasp_yaw,
    _tilted_pca_quat,
    canonical_side_bar_grasp,
    select_mechanism_grasp,
)
from racap.backends.vlm import BBox, bbox_with_feedback
from racap.agent.full_react import (
    _language_allowed_skills,
    _normalise_goal,
    _order_goals,
    _plan_semantic_errors,
    _resolve_plan_pronouns,
)


def _solid_cloud(width: float, depth: float, height: float, n: int = 13) -> np.ndarray:
    xs = np.linspace(-width / 2, width / 2, n)
    ys = np.linspace(-depth / 2, depth / 2, n)
    xx, yy = np.meshgrid(xs, ys)
    top = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, height)])
    lower = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    return np.concatenate([top, lower], axis=0)


def _hollow_cloud(radius: float, height: float, n: int = 120) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rim = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, height)])
    core_angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    core = np.column_stack(
        [
            0.15 * radius * np.cos(core_angles),
            0.15 * radius * np.sin(core_angles),
            np.zeros(core_angles.size),
        ]
    )
    return np.concatenate([rim, core], axis=0)


def test_tall_hollow_vessel_prefers_learned_sidewall_contact() -> None:
    attempts = _attempts(_hollow_cloud(0.055, 0.15), object_height=0.15)

    assert attempts[:3] == (
        ("graspnet", "top"),
        ("rim", "top"),
        ("top_down", "top"),
    )


def test_shallow_hollow_vessel_keeps_analytic_rim_contact() -> None:
    attempts = _attempts(_hollow_cloud(0.055, 0.06), object_height=0.06)

    assert attempts[0] == ("rim", "top")


def test_stack_plan_erodes_support_by_carried_footprint() -> None:
    plan = stack_plan(
        _solid_cloud(0.04, 0.03, 0.04),
        _solid_cloud(0.14, 0.11, 0.03),
        preferred_xy=np.zeros(2),
        uncertainty_m=0.003,
    )
    assert plan is not None
    assert plan["mode"] == "footprint_support"
    assert np.linalg.norm(plan["centre_xy"]) < 0.01
    assert plan["clearance_m"] >= 0.003


def test_stack_plan_rejects_object_larger_than_solid_support() -> None:
    plan = stack_plan(
        _solid_cloud(0.14, 0.12, 0.04),
        _solid_cloud(0.08, 0.07, 0.03),
        preferred_xy=np.zeros(2),
        uncertainty_m=0.002,
    )
    assert plan is None


def test_stack_uses_bowl_base_not_outer_rim_as_load_bearing_footprint() -> None:
    plan = stack_plan(
        _hollow_cloud(0.055, 0.06),
        _solid_cloud(0.08, 0.08, 0.02),
        preferred_xy=np.zeros(2),
        uncertainty_m=0.002,
        nesting_support=False,
    )
    assert plan is not None
    assert plan["mode"] == "footprint_support"
    assert plan["footprint_source"] == "low_z_contact"
    assert max(plan["contact_size"]) < max(plan["object_size"])


class _TopSurfaceRuntime:
    def __init__(self) -> None:
        xs = np.linspace(0.1, 0.6, 31)
        ys = np.linspace(-0.3, 0.0, 25)
        self.points = np.asarray([[x, y, z] for z in (0.10, 0.50) for x in xs for y in ys])

    def localize_many(self, labels):
        return {
            labels[0]: GroundedTarget(
                label=labels[0],
                pose=(0.35, -0.15, 0.30),
                confidence=0.9,
                metadata={"n_points": len(self.points), "extent": [0.5, 0.3, 0.4]},
            )
        }

    def object_points(self, label):
        del label
        return self.points


def test_upper_surface_preserves_measured_support_polygon_and_extent() -> None:
    trace: list[dict] = []
    target = _upper_surface(_TopSurfaceRuntime(), "cabinet", trace)

    assert target is not None
    assert target.metadata["region_source"] == "rgbd_top_band"
    assert len(target.metadata["region_polygon_xy"]) >= 4
    assert target.metadata["extent"][0] > 0.40
    assert target.metadata["extent"][1] > 0.24


def test_equal_bowls_use_concentric_nesting_not_impossible_erosion() -> None:
    plan = stack_plan(
        _hollow_cloud(0.055, 0.06),
        _hollow_cloud(0.055, 0.06),
        preferred_xy=np.asarray([0.2, -0.1]),
    )
    assert plan is not None
    assert plan["mode"] == "nesting"
    assert np.allclose(plan["centre_xy"], [0.2, -0.1])


def test_semantic_nesting_can_recover_occluded_vessel_interior() -> None:
    # A rear bowl can expose only its rim to RGB-D, making the generic hollow
    # test inconclusive.  The caller may fuse an independent semantic vessel
    # cue while retaining the same footprint geometry.
    support = _solid_cloud(0.105, 0.10, 0.055)
    plan = stack_plan(
        _solid_cloud(0.10, 0.095, 0.05),
        support,
        preferred_xy=np.asarray([0.1, 0.2]),
        nesting_support=True,
    )
    assert plan is not None
    assert plan["mode"] == "nesting"


def test_linear_articulation_uses_handle_to_fixture_axis() -> None:
    opened = linear_waypoints(np.asarray([0.0, -0.1]), np.zeros(2), goal="open", stroke_m=0.12)
    closed = linear_waypoints(np.asarray([0.0, -0.2]), np.zeros(2), goal="closed", stroke_m=0.12)
    assert opened is not None and closed is not None
    open_path, axis = opened
    close_path, _ = closed
    assert np.allclose(axis, [0.0, -1.0])
    assert open_path[-1, 1] < -0.1
    assert close_path[-1, 1] > -0.2


def test_drawer_close_stroke_uses_bounded_visible_protrusion() -> None:
    assert np.isclose(prismatic_stroke("open", None, 21.0), 0.12)
    assert np.isclose(prismatic_stroke("closed", None, 21.0), 0.21)
    assert np.isclose(prismatic_stroke("closed", None, 80.0), 0.24)


def test_drawer_axis_uses_nearest_opening_boundary_not_cabinet_centre() -> None:
    # The handle is at the right end of the front, so handle-to-centre is
    # strongly diagonal.  The opening boundary still exposes the rail normal.
    polygon = np.asarray(
        [
            [0.55, -0.24],
            [0.71, -0.24],
            [0.70, -0.10],
            [0.55, -0.10],
        ]
    )
    fit = prismatic_axis_from_opening_boundary(np.asarray([0.708, -0.052]), polygon)
    assert fit is not None
    assert abs(float(fit["axis"][0])) < 0.05
    assert float(fit["axis"][1]) > 0.99
    assert np.isclose(fit["handle_boundary_distance_m"], 0.048, atol=0.002)


def test_drawer_axis_rejects_handle_rebound_inside_opening() -> None:
    polygon = np.asarray(
        [
            [0.55, -0.24],
            [0.71, -0.24],
            [0.70, -0.10],
            [0.55, -0.10],
        ]
    )
    assert prismatic_axis_from_opening_boundary(np.asarray([0.62, -0.17]), polygon) is None


def test_drawer_axis_falls_back_to_fixture_relative_opening_normal() -> None:
    polygon = np.asarray(
        [
            [0.55, -0.24],
            [0.71, -0.24],
            [0.70, -0.10],
            [0.55, -0.10],
        ]
    )
    fit = prismatic_axis_from_opening_fixture(np.asarray([0.70, -0.30]), polygon)
    assert fit is not None
    assert abs(float(fit["axis"][0])) < 0.05
    assert float(fit["axis"][1]) > 0.99


def test_drawer_axis_uses_rgbd_band_inside_semantic_handle_box() -> None:
    polygon = np.asarray(
        [
            [0.55, -0.24],
            [0.71, -0.24],
            [0.70, -0.10],
            [0.55, -0.10],
        ]
    )
    x = np.linspace(0.58, 0.68, 80)
    points = np.column_stack([x, np.full_like(x, -0.052), np.full_like(x, 0.185)])
    fit = prismatic_axis_from_opening_probe(points, polygon, expected_z=0.21)
    assert fit is not None
    assert abs(float(fit["axis"][0])) < 0.05
    assert float(fit["axis"][1]) > 0.99


def test_drawer_depth_bias_stays_inside_footprint_eroded_opening() -> None:
    polygon = np.asarray(
        [
            [0.55, -0.24],
            [0.71, -0.24],
            [0.70, -0.10],
            [0.55, -0.10],
        ]
    )
    object_points = _solid_cloud(0.075, 0.038, 0.017)
    object_points[:, :2] += [0.62, -0.17]
    nudge = _inward_feasible_nudge(
        np.asarray([0.62, -0.17]),
        polygon,
        np.asarray([0.0, 1.0]),
        object_points,
    )
    assert abs(float(nudge[0])) < 1e-8
    assert -0.05 < float(nudge[1]) < -0.02


def test_drawer_close_approaches_and_retreats_from_external_face() -> None:
    points = closing_face_push_points(
        np.asarray([0.0, -0.20]),
        np.asarray([0.0, -1.0]),
        np.asarray([0.0, -0.08]),
    )
    assert points is not None
    approach, contact, retreat = points
    assert approach[1] < contact[1] < -0.20
    assert retreat[1] < -0.08


def test_prismatic_progress_cancels_fixture_translation() -> None:
    # The whole cabinet moved 8 cm, while the drawer moved only 2 cm relative
    # to it.  World-handle displacement must not be reported as 10 cm of joint
    # travel.
    progress = relative_planar_progress(
        np.asarray([0.0, 0.20]),
        np.asarray([0.0, 0.00]),
        np.asarray([0.0, 0.10]),
        np.asarray([0.0, -0.08]),
        np.asarray([0.0, 1.0]),
    )
    assert np.isclose(progress, -0.02)


def test_fixture_refresh_rejects_scene_scale_grounding_jump() -> None:
    reliable, jump, limit = fixture_refresh_gate(
        np.asarray([0.4, 0.1]),
        np.asarray([2.2, -1.0]),
        [0.6, 0.3, 0.5],
    )
    assert reliable is False
    assert jump > 2.0
    assert limit == 0.30


def test_handle_contact_uses_protruding_rgbd_band_not_drawer_face_centroid() -> None:
    xs = np.linspace(-0.05, 0.05, 25)
    zs = np.linspace(0.03, 0.06, 8)
    face = np.asarray([[x, 0.10, z] for x in xs for z in zs])
    handle = np.asarray(
        [
            [x, y, z]
            for x in np.linspace(-0.035, 0.035, 18)
            for y in np.linspace(0.125, 0.135, 5)
            for z in np.linspace(0.04, 0.055, 4)
        ]
    )

    fit = protruding_contact(np.concatenate([face, handle]), np.zeros(2), np.asarray([0.0, 1.0]))

    assert fit is not None
    assert fit["pose"][1] > 0.123
    assert 0.04 <= fit["top_z"] <= 0.055
    assert abs(fit["principal_axis"][0]) > 0.8


def test_mechanism_grasp_prefers_side_approach_along_contact_axis() -> None:
    # Candidate local Z columns are their approach directions.  The highest
    # raw score is top-down and must be rejected; the second is horizontal but
    # orthogonal to the measured drawer normal; the third can transmit pull.
    top = np.eye(4)
    top[:3, 2] = [0.0, 0.0, -1.0]
    orthogonal = np.eye(4)
    orthogonal[:3, 2] = [1.0, 0.0, 0.0]
    aligned = np.eye(4)
    aligned[:3, 0] = [0.0, 0.0, 1.0]
    aligned[:3, 1] = [-1.0, 0.0, 0.0]
    aligned[:3, 2] = [0.0, -1.0, 0.0]

    selected, score = select_mechanism_grasp(
        np.asarray([top, orthogonal, aligned]),
        np.asarray([0.95, 0.90, 0.70]),
        np.eye(4),
        np.asarray([0.0, 1.0]),
    )

    assert selected is not None
    assert np.allclose(selected[:3, 2], [0.0, -1.0, 0.0])
    assert score > 0.70


def test_mechanism_grasp_rejects_horizontal_jaw_axis_on_thin_bar() -> None:
    wrong_jaws = np.eye(4)
    wrong_jaws[:3, 2] = [0.0, -1.0, 0.0]

    selected, _ = select_mechanism_grasp(
        np.asarray([wrong_jaws]),
        np.asarray([0.95]),
        np.eye(4),
        np.asarray([0.0, 1.0]),
    )

    assert selected is None


def test_canonical_side_bar_keeps_wrist_outside_and_jaws_vertical() -> None:
    contact = np.asarray([0.60, -0.20, 0.18])
    outward = np.asarray([-0.6, 0.8])

    pose = canonical_side_bar_grasp(contact, outward, finger_depth_m=0.10)

    assert pose is not None
    assert np.allclose(pose[:3, 0], [0.0, 0.0, 1.0])
    assert np.allclose(pose[:2, 2], -outward)
    assert np.allclose(pose[:3, 3], contact + np.r_[outward * 0.10, 0.0])
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0)


def test_articulation_rejects_depth_lifts_outside_robot_workspace() -> None:
    impossible = GroundedTarget(
        label="cabinet",
        pose=(-0.74, 1.36, -0.72),
        confidence=0.95,
        metadata={"n_points": 2400, "extent": [0.3, 0.2, 0.2]},
    )
    ordinary = GroundedTarget(
        label="cabinet",
        pose=(0.69, 0.31, 0.16),
        confidence=0.05,
        metadata={"n_points": 40, "extent": [0.3, 0.2, 0.2]},
    )

    assert articulation_target_usable(impossible) is False
    assert articulation_target_usable(ordinary) is True


def test_top_down_articulation_measures_fingertips_not_wrist_origin() -> None:
    execution = {
        "source": "pca_axis",
        "position": [0.6266, -0.2022, 0.1808],
        "reached_position": [0.6299, -0.1914, 0.3011],
    }

    assert top_down_fingertip_contact(execution, 0.1808) is True
    laterally_missed = dict(execution)
    laterally_missed["reached_position"] = [0.70, -0.12, 0.3011]
    assert top_down_fingertip_contact(laterally_missed, 0.1808) is False


def test_hinge_arc_selects_endpoint_by_goal_not_asset_identity() -> None:
    handle = np.asarray([-0.12, -0.05])
    hinge = np.asarray([0.10, -0.05])
    fixture = np.asarray([0.0, 0.02])
    opened = hinge_waypoints(handle, hinge, fixture, goal="open", angle_deg=75)
    closed = hinge_waypoints(handle, hinge, fixture, goal="closed", angle_deg=75)
    assert opened is not None and closed is not None
    assert np.linalg.norm(opened[-1] - fixture) >= np.linalg.norm(closed[-1] - fixture)
    assert np.linalg.norm(opened[-1] - handle) > 0.1


def test_open_panel_hinge_is_near_endpoint_toward_fixed_fixture() -> None:
    panel = np.asarray([0.42, 0.15])
    fixture = np.asarray([0.66, 0.36])

    hinge = hinge_from_moving_panel(panel, fixture, [0.42, 0.36, 0.20])

    assert hinge is not None
    assert np.isclose(np.linalg.norm(hinge - panel), 0.21, atol=1e-6)
    assert np.linalg.norm(hinge - fixture) < np.linalg.norm(panel - fixture)
    assert np.linalg.norm(hinge - panel) < 0.28


def test_measured_hinge_angle_distinguishes_partial_motion() -> None:
    pivot = np.zeros(2)
    start = np.asarray([0.2, 0.0])
    partial = np.asarray([0.2 * np.cos(np.radians(12)), 0.2 * np.sin(np.radians(12))])
    opened = np.asarray([0.0, 0.2])
    assert np.isclose(signed_planar_angle_deg(start, partial, pivot), 12.0)
    assert np.isclose(signed_planar_angle_deg(start, opened, pivot), 90.0)


def test_articulation_defaults_are_bounded_closed_loop_increments() -> None:
    assert DEFAULT_LINEAR_STROKE_M <= 0.12
    assert DEFAULT_HINGE_ANGLE_DEG <= 40.0


def test_prismatic_axis_uses_rotated_fixture_depth_not_offset_handle() -> None:
    angle = np.radians(31.0)
    long_axis = np.asarray([np.cos(angle), np.sin(angle)])
    depth_axis = np.asarray([-long_axis[1], long_axis[0]])
    fixture = np.asarray([0.55, -0.12])
    points = np.asarray(
        [
            [*(fixture + x * long_axis + y * depth_axis), 0.10]
            for x in np.linspace(-0.18, 0.18, 25)
            for y in np.linspace(-0.07, 0.07, 13)
        ]
    )
    # The handle is laterally offset along the drawer face.  Its direct
    # handle-to-centre vector is diagonal rather than parallel to the rails.
    handle = fixture + 0.11 * long_axis + 0.08 * depth_axis

    fit = prismatic_axis_from_footprint(fixture, handle, points)

    assert fit is not None
    assert abs(float(fit["axis"] @ depth_axis)) > 0.98
    assert float(fit["axis"] @ (handle - fixture)) > 0.0


def test_prismatic_axis_corrects_moderate_lateral_handle_offset() -> None:
    fixture = np.zeros(2)
    points = np.asarray(
        [[x, y, 0.1] for x in np.linspace(-0.18, 0.18, 25) for y in np.linspace(-0.07, 0.07, 13)]
    )
    # A visually modest lateral offset is still a large rail-angle error and
    # should be removed before contact motion.
    handle = np.asarray([0.035, 0.080])

    fit = prismatic_axis_from_footprint(fixture, handle, points)

    assert fit is not None
    assert abs(float(fit["axis"] @ [0.0, 1.0])) > 0.99


def test_prismatic_axis_keeps_nearly_collinear_direct_estimate() -> None:
    fixture = np.zeros(2)
    points = np.asarray(
        [[x, y, 0.1] for x in np.linspace(-0.18, 0.18, 25) for y in np.linspace(-0.07, 0.07, 13)]
    )

    assert prismatic_axis_from_footprint(fixture, np.asarray([0.005, 0.10]), points) is None


def test_identity_check_requires_an_exact_closed_set_answer() -> None:
    target = GroundedTarget(label="milk", pose=(0.4, 0.1, 0.03))

    class Runtime:
        answer = "maybe the requested package"

        def inspect(self, prompt, *, options, target):
            del prompt, options, target
            return {"ok": True, "answer": self.answer}

    runtime = Runtime()
    assert _confirm_identity(runtime, "milk", target)["matches"] is None
    runtime.answer = "a different object"
    assert _confirm_identity(runtime, "milk", target)["matches"] is False
    runtime.answer = "milk"
    assert _confirm_identity(runtime, "milk", target)["matches"] is True


def test_relational_identity_uses_marked_full_scene_verifier() -> None:
    target = GroundedTarget(
        label="the butter at the front",
        pose=(0.4, 0.1, 0.03),
        metadata={"bbox": [10, 20, 40, 60]},
    )

    class Runtime:
        inspected_crop = False

        def verify_pick_grounding(self, label, *, target):
            assert label == "the butter at the front"
            assert target.metadata["bbox"] == [10, 20, 40, 60]
            return {
                "ok": True,
                "matches": False,
                "answer": "incorrect referent",
                "view": "marked_full_scene",
            }

        def inspect(self, *args, **kwargs):
            self.inspected_crop = True
            raise AssertionError("a relation must not be verified from a crop")

    runtime = Runtime()
    verdict = _confirm_identity(runtime, "the butter at the front", target)

    assert verdict["matches"] is False
    assert verdict["relation_preserved"] is True
    assert runtime.inspected_crop is False


def test_relational_verifier_blindly_rejects_wrong_product_before_relation(
    monkeypatch,
) -> None:
    import racap.backends.llm as llm

    calls: list[dict] = []

    def fake_ask(system, prompt, *, images, model, **kwargs):
        del system, kwargs
        calls.append({"prompt": prompt, "images": images, "model": model})
        if len(calls) == 1:
            assert "butter" not in prompt.lower()
            return "dark box labelled Chocolate Pudding"
        if len(calls) == 2:
            assert images == []
            assert "binder/notebook" in prompt
            return "category mismatch"
        raise AssertionError("a category mismatch must short-circuit relation voting")

    monkeypatch.setattr(llm, "ask", fake_ask)
    monkeypatch.setenv("RACAP_VERIFIER_MODEL", "independent-verifier")

    class Runtime:
        def observe(self):
            return object()

        def _camera(self, obs, wrist=False, view=None):
            del obs, wrist, view
            rgb = np.zeros((100, 140, 3), dtype=np.uint8)
            return rgb, np.ones((100, 140)), np.eye(3), np.eye(4)

    target = GroundedTarget(
        label="the butter at the front",
        pose=(0.4, 0.1, 0.03),
        confidence=0.9,
        metadata={"bbox": [20, 25, 60, 70]},
    )
    verdict = LiberoPrimitiveRuntime.verify_pick_grounding(
        Runtime(),
        "the butter at the front",
        target=target,
        grounder_model="relation-model",
    )

    assert verdict["matches"] is False
    assert verdict["view"] == "blind_crop"
    assert verdict["verifier_model"] == "independent-verifier"
    assert "Pudding" in verdict["blind_description"]
    assert len(calls) == 2


def test_failed_contact_prefers_local_instance_tracking_over_global_vlm() -> None:
    current = GroundedTarget(
        label="the bowl at the front",
        pose=(0.4, 0.1, 0.03),
        confidence=0.9,
        metadata={
            "bbox": [20, 25, 60, 70],
            "n_points": 100,
            "extent": [0.05, 0.05, 0.03],
            "pregrasp_identity_evidence": {"matches": True},
        },
    )
    refreshed = GroundedTarget(
        label=current.label,
        pose=(0.405, 0.102, 0.03),
        confidence=0.8,
        metadata={"bbox": [22, 25, 62, 70], "n_points": 90, "extent": [0.05, 0.05, 0.03]},
    )

    class Runtime:
        def track_target(self, target, *, appearance_label):
            assert target is current
            assert appearance_label == "bowl"
            return refreshed

        def localize_many(self, *args, **kwargs):
            raise AssertionError("local success must not invoke global VLM")

    trace: list[dict] = []
    result, evidence = _refresh_pick_after_contact(Runtime(), current.label, current, trace)

    assert result is refreshed
    assert evidence["continued_by"] == "local_sam3_rgbd"
    assert evidence["global_vlm_recheck"] is False
    assert trace[-1]["event"] == "local_pick_track"


def test_vlm_negative_feedback_marks_and_excludes_rejected_box(monkeypatch) -> None:
    import racap.backends.vlm as vlm

    seen: dict[str, object] = {}

    def fake_ask(system, prompt, rgb, model, **kwargs):
        del system, model, kwargs
        seen["prompt"] = prompt
        seen["image"] = np.asarray(rgb)
        return {
            "found": True,
            "x0": 55,
            "y0": 50,
            "x1": 75,
            "y1": 80,
            "confidence": 0.9,
        }

    monkeypatch.setattr(vlm, "_ask_json", fake_ask)
    rejected = BBox(10, 15, 35, 40, 0.8)
    revised = bbox_with_feedback(
        np.zeros((100, 100, 3), dtype=np.uint8),
        "the milk carton behind the can",
        [rejected],
    )

    assert revised is not None
    assert revised.x0 >= 47 and revised.x1 <= 83  # includes bbox margin
    assert "complete description" in str(seen["prompt"])
    marked = np.asarray(seen["image"])
    assert marked[15, 10, 0] == 255
    assert marked[15, 10, 1] == 0


def test_vlm_negative_feedback_rejects_shifted_same_candidate(monkeypatch) -> None:
    import racap.backends.vlm as vlm

    monkeypatch.setattr(
        vlm,
        "_ask_json",
        lambda *args, **kwargs: {
            "found": True,
            "x0": 14,
            "y0": 17,
            "x1": 33,
            "y1": 39,
            "confidence": 0.9,
        },
    )
    revised = bbox_with_feedback(
        np.zeros((100, 100, 3), dtype=np.uint8),
        "milk",
        [BBox(10, 15, 35, 40, 0.8)],
    )

    assert revised is None


def test_pick_grounding_uses_vlm_feedback_after_identity_veto() -> None:
    from racap.policy_api.pickplace_api import pickplace

    wrong = GroundedTarget(
        label="milk",
        pose=(0.55, 0.03, 0.04),
        confidence=0.9,
        metadata={
            "bbox": [20, 20, 50, 60],
            "n_points": 200,
            "extent": [0.05, 0.04, 0.06],
            "top_z": 0.07,
        },
    )
    correct = GroundedTarget(
        label="milk",
        pose=(0.42, 0.12, 0.09),
        confidence=0.8,
        metadata={
            "bbox": [80, 20, 105, 75],
            "n_points": 220,
            "extent": [0.05, 0.04, 0.12],
            "top_z": 0.15,
        },
    )

    class Runtime:
        feedback_rejections: list[GroundedTarget] = []

        def localize_many(self, labels, close_view=False):
            del close_view
            label = labels[0]
            if label == "milk":
                return {label: wrong}
            return {
                label: GroundedTarget(
                    label=label,
                    pose=(0.0, 0.0, 0.0),
                    confidence=0.0,
                )
            }

        def localize_with_feedback(self, label, rejected):
            assert label == "milk"
            self.feedback_rejections = list(rejected)
            return correct

        def inspect(self, prompt, *, options, target):
            del prompt
            return {
                "ok": True,
                "answer": options[1] if target is wrong else options[0],
            }

        def localize(self, *args, **kwargs):
            del args, kwargs
            return GroundedTarget(
                label="missing destination",
                pose=(0.0, 0.0, 0.0),
                confidence=0.0,
            )

    runtime = Runtime()
    result = pickplace(runtime, "milk", "missing destination")

    assert result.failure_mode == "not_grounded"
    assert runtime.feedback_rejections == [wrong]
    assert result.report["pick_bbox"] == [80, 20, 105, 75]
    assert result.report["pregrasp_identity_evidence"]["matches"] is True
    assert result.report["pregrasp_identity_evidence"]["rejected_candidates"] == 1
    assert any(event.get("mode") == "vlm_negative_feedback" for event in result.trace)


def test_missing_joint_box_escalates_to_independently_verified_single_box() -> None:
    missing = GroundedTarget(
        label="book",
        pose=(0.0, 0.0, 0.0),
        confidence=0.0,
        metadata={"reason": "not_in_joint_detection"},
    )
    found = GroundedTarget(
        label="book",
        pose=(0.44, 0.12, 0.04),
        confidence=0.8,
        metadata={
            "bbox": [70, 30, 105, 90],
            "n_points": 240,
            "extent": [0.08, 0.025, 0.10],
            "top_z": 0.09,
        },
    )

    class Runtime:
        detectors: list[str] = []

        def localize_many(self, labels, close_view=False):
            del close_view
            return {labels[0]: missing}

        def localize(self, label, *, detector):
            assert label == "book"
            self.detectors.append(detector)
            return found

        def inspect(self, prompt, *, options, target):
            del prompt
            assert target is found
            return {"ok": True, "answer": options[0]}

    runtime = Runtime()
    trace: list[dict] = []
    target, evidence, attempts = _ground_verified_pick(runtime, "book", trace, max_attempts=3)

    assert target is found
    assert attempts == 2
    assert runtime.detectors == ["vlm_bbox_sam3"]
    assert evidence["matches"] is True
    assert any(event.get("mode") == "vlm_bbox_sam3" for event in trace)


def test_two_missing_boxes_escalate_to_exhaustive_visual_search() -> None:
    missing = GroundedTarget(
        label="thin object",
        pose=(0.0, 0.0, 0.0),
        confidence=0.0,
        metadata={"reason": "not_found"},
    )
    found = GroundedTarget(
        label="thin object",
        pose=(0.42, -0.03, 0.08),
        confidence=0.7,
        metadata={
            "bbox": [110, 45, 126, 115],
            "n_points": 160,
            "extent": [0.025, 0.06, 0.11],
            "top_z": 0.12,
        },
    )

    class Runtime:
        detectors: list[str] = []

        def localize_many(self, labels, close_view=False):
            del close_view
            return {labels[0]: missing}

        def localize(self, label, *, detector):
            del label
            self.detectors.append(detector)
            return found if detector == "vlm_search_bbox_sam3" else missing

        def inspect(self, prompt, *, options, target):
            del prompt, target
            return {"ok": True, "answer": options[0]}

    runtime = Runtime()
    trace: list[dict] = []
    target, evidence, attempts = _ground_verified_pick(
        runtime, "thin object", trace, max_attempts=3
    )

    assert target is found
    assert attempts == 3
    assert runtime.detectors == [
        "vlm_bbox_sam3",
        "vlm_search_bbox_sam3",
    ]
    assert evidence["matches"] is True
    assert trace[-2]["mode"] == "vlm_search_bbox_sam3"


def test_invalid_negative_feedback_candidate_does_not_end_destination_ladder(
    monkeypatch,
) -> None:
    wrong = GroundedTarget(
        label="slot",
        pose=(0.3, 0.0, 0.02),
        confidence=0.8,
        metadata={"bbox": [10, 20, 60, 90], "n_points": 200, "extent": [0.08, 0.04, 0.06]},
    )
    invalid = GroundedTarget(
        label="slot",
        pose=(0.0, 0.0, 0.0),
        confidence=0.0,
        metadata={"reason": "no_revised_bbox"},
    )
    good = GroundedTarget(
        label="slot",
        pose=(0.5, -0.1, 0.02),
        confidence=0.8,
        metadata={"bbox": [90, 20, 150, 100], "n_points": 220, "extent": [0.10, 0.05, 0.06]},
    )

    class Runtime:
        feedback_calls = 0

        def localize(self, label, *, detector):
            del label, detector
            return wrong

        def localize_with_feedback(self, label, rejected):
            del label, rejected
            self.feedback_calls += 1
            return invalid if self.feedback_calls == 1 else good

        def verify_destination_grounding(self, phrase, *, target):
            del phrase, target
            return {"ok": True, "answer": "correct destination"}

    monkeypatch.setattr(
        pickplace,
        "_inside_of",
        lambda runtime, target, trace: target,
    )
    trace: list[dict] = []
    result = _ground_destination(
        Runtime(),
        "right slot",
        trace,
        near=GroundedTarget(
            label="object",
            pose=(0.4, 0.0, 0.03),
            confidence=1.0,
            metadata={"n_points": 100, "extent": [0.05, 0.03, 0.02]},
        ),
        mode="inside",
        inside_validator=lambda target: (
            target is good,
            "good" if target is good else "bad geometry",
        ),
    )

    assert result is good
    assert any(event.get("answer") == "invalid_candidate_without_bbox" for event in trace)


def test_failed_negative_feedback_still_reaches_exhaustive_search() -> None:
    wrong = GroundedTarget(
        label="book",
        pose=(0.35, 0.1, 0.04),
        confidence=0.8,
        metadata={
            "bbox": [20, 30, 80, 100],
            "n_points": 200,
            "extent": [0.2, 0.12, 0.04],
        },
    )
    missing = GroundedTarget(
        label="book",
        pose=(0.0, 0.0, 0.0),
        confidence=0.0,
        metadata={"reason": "vlm_no_revised_bbox"},
    )
    found = GroundedTarget(
        label="book",
        pose=(0.52, -0.08, 0.07),
        confidence=0.75,
        metadata={
            "bbox": [120, 35, 142, 110],
            "n_points": 180,
            "extent": [0.03, 0.07, 0.11],
        },
    )

    class Runtime:
        feedback_calls = 0
        detectors: list[str] = []

        def localize_many(self, labels, close_view=False):
            del close_view
            return {labels[0]: wrong}

        def localize_with_feedback(self, label, rejected):
            del label
            assert rejected == [wrong]
            self.feedback_calls += 1
            return missing

        def localize(self, label, *, detector):
            del label
            self.detectors.append(detector)
            return found

        def inspect(self, prompt, *, options, target):
            del prompt
            return {
                "ok": True,
                "answer": options[1] if target is wrong else options[0],
            }

    runtime = Runtime()
    target, evidence, attempts = _ground_verified_pick(runtime, "book", [], max_attempts=3)

    assert target is found
    assert attempts == 3
    assert runtime.feedback_calls == 1
    assert runtime.detectors == ["vlm_search_bbox_sam3"]
    assert evidence["matches"] is True


def test_fourth_grounding_rung_uses_a_different_view() -> None:
    missing = GroundedTarget(
        label="thin object",
        pose=(0.0, 0.0, 0.0),
        confidence=0.0,
        metadata={"reason": "not_found"},
    )
    side = GroundedTarget(
        label="thin object",
        pose=(0.48, -0.11, 0.07),
        confidence=0.72,
        metadata={
            "bbox": [90, 40, 145, 130],
            "n_points": 220,
            "extent": [0.03, 0.08, 0.11],
            "view": "sideview",
        },
    )

    class Runtime:
        inspected_view = None

        def localize_many(self, labels, close_view=False):
            del close_view
            return {labels[0]: missing}

        def localize(self, label, *, detector):
            del label, detector
            return missing

        def localize_across_views(self, label, *, detector):
            del label
            assert detector == "vlm_point_sam3"
            return side

        def inspect(self, prompt, *, options, target, view=None):
            del prompt
            assert target is side
            self.inspected_view = view
            return {"ok": True, "answer": options[0]}

    runtime = Runtime()
    target, evidence, attempts = _ground_verified_pick(runtime, "thin object", [], max_attempts=4)

    assert target is side
    assert attempts == 4
    assert runtime.inspected_view == "sideview"
    assert evidence["view"] == "multi_view_point_sam3"


def test_grounding_size_gate_retries_a_joined_background_mask() -> None:
    joined = GroundedTarget(
        label="book",
        pose=(0.4, 0.0, 0.05),
        confidence=0.8,
        metadata={
            "bbox": [20, 20, 180, 150],
            "n_points": 800,
            "extent": [0.31, 0.14, 0.02],
        },
    )
    clean = GroundedTarget(
        label="book",
        pose=(0.5, -0.1, 0.05),
        confidence=0.75,
        metadata={
            "bbox": [100, 40, 135, 130],
            "n_points": 220,
            "extent": [0.105, 0.018, 0.14],
        },
    )

    class Runtime:
        inspections = 0

        def localize_many(self, labels, close_view=False):
            del labels, close_view
            return {"book": joined}

        def localize(self, label, *, detector):
            del label, detector
            return clean

        def inspect(self, prompt, *, options, target):
            del prompt, target
            self.inspections += 1
            return {"ok": True, "answer": options[0]}

    runtime = Runtime()
    trace: list[dict] = []
    target, evidence, attempts = _ground_verified_pick(
        runtime, "book", trace, max_attempts=3, max_planar_extent=0.20
    )

    assert target is clean
    assert attempts == 2
    assert runtime.inspections == 1
    assert evidence["matches"] is True
    assert any(event.get("event") == "grounding_rejected_geometry" for event in trace)


def test_hinge_uses_fixture_long_axis_not_camera_axis() -> None:
    angle = np.radians(32.0)
    lateral = np.asarray([np.cos(angle), np.sin(angle)])
    depth = np.asarray([-lateral[1], lateral[0]])
    fixture = np.asarray([0.4, -0.2])
    points = np.asarray(
        [
            [*(fixture + x * lateral + y * depth), 0.1]
            for x in np.linspace(-0.16, 0.16, 21)
            for y in np.linspace(-0.07, 0.07, 9)
        ]
    )
    handle = fixture + 0.12 * lateral + 0.07 * depth

    hinge = hinge_from_footprint(fixture, handle, points)

    assert hinge is not None
    # Handle is on +lateral side, so the hinge must be on the opposite side.
    assert float((hinge - fixture) @ lateral) < -0.12


def test_extended_door_uses_near_fixture_edge_as_hinge() -> None:
    fixture = np.zeros(2)
    points = np.asarray(
        [[x, y, 0.1] for x in np.linspace(-0.16, 0.16, 21) for y in np.linspace(-0.07, 0.07, 9)]
    )
    handle = np.asarray([0.48, 0.02])

    hinge = hinge_from_footprint(fixture, handle, points)

    assert hinge is not None
    assert hinge[0] > 0.10
    assert np.linalg.norm(handle - hinge) < np.linalg.norm(handle + hinge)


class _HighTransitRuntime:
    def __init__(self) -> None:
        self.moves = []

    def ee_pose(self):
        return {"position": [0.1, 0.1, 0.48]}

    def move_to(self, position, yaw=None):
        del yaw
        self.moves.append([float(v) for v in position])

    def open_gripper(self):
        pass


def test_transport_never_descends_before_horizontal_travel() -> None:
    runtime = _HighTransitRuntime()
    destination = GroundedTarget(label="plate", pose=(0.5, 0.2, 0.01))
    report = {"surface_under_object_z": 0.20}

    _place_carefully(
        runtime,
        "bowl",
        destination,
        (0.02, 0.01),
        0.04,
        None,
        [],
        nudge=np.zeros(2),
        carry_offset=np.zeros(2),
        margin=0.002,
        release_on="rim",
        centre=False,
        report=report,
    )

    assert runtime.moves[0] == [0.1, 0.1, 0.32]
    assert runtime.moves[1][:2] == [0.5, 0.2]
    assert report["transit_z"] == 0.32
    assert report["transit_source_surface_z"] == 0.20
    assert report["transit_preserved_source_clearance"] is True
    assert report["transit_vertical_lift_completed"] is True


def test_wide_measured_drawer_descends_inside_before_release() -> None:
    runtime = _HighTransitRuntime()
    destination = GroundedTarget(
        label="the top drawer of the cabinet",
        pose=(0.5, 0.2, 0.10),
        kind="region",
        metadata={
            "region_source": "rgbd_cavity",
            "region_clearance_m": 0.071,
        },
    )
    report = {"surface_under_object_z": 0.0}

    _place_carefully(
        runtime,
        "butter",
        destination,
        (0.20, 0.10),
        0.02,
        None,
        [],
        nudge=np.zeros(2),
        carry_offset=np.zeros(2),
        margin=0.01,
        release_on="rim",
        centre=False,
        report=report,
    )

    assert report["release_on_requested"] == "rim"
    assert report["release_on_effective"] == "floor"
    assert report["in_container_descent"] is True
    # First move is the vertical lift, penultimate move is the descent before
    # opening, and the last move is the retreat.
    assert runtime.moves[-2][2] == 0.13


def test_narrow_container_keeps_rim_release() -> None:
    runtime = _HighTransitRuntime()
    destination = GroundedTarget(
        label="the compartment of the caddy",
        pose=(0.5, 0.2, 0.10),
        kind="region",
        metadata={
            "region_source": "rgbd_cavity",
            "region_clearance_m": 0.035,
        },
    )
    report = {"surface_under_object_z": 0.0}

    _place_carefully(
        runtime,
        "book",
        destination,
        (0.20, 0.10),
        0.02,
        None,
        [],
        nudge=np.zeros(2),
        carry_offset=np.zeros(2),
        margin=0.01,
        release_on="rim",
        centre=False,
        report=report,
    )

    assert report["release_on_effective"] == "rim"
    assert report["in_container_descent"] is False
    assert runtime.moves[-2][2] == 0.23


def test_object_top_rejects_single_high_background_depth_ray() -> None:
    target = GroundedTarget(
        label="bowl",
        pose=(0.31, -0.07, 0.021),
        confidence=0.8,
        metadata={
            "top_z": 0.2125,
            "n_points": 200,
            "extent": [0.108, 0.091, 0.059],
        },
    )
    body = np.column_stack(
        [
            np.linspace(0.27, 0.35, 199),
            np.linspace(-0.11, -0.03, 199),
            np.linspace(-0.005, 0.052, 199),
        ]
    )
    points = np.concatenate([body, [[0.31, -0.07, 0.2125]]])

    top, raw = _robust_object_top(target, points, -0.011)

    assert raw == 0.2125
    assert 0.045 <= top <= 0.075
    # The repaired height must be the value that actually drives the hand,
    # not just a diagnostic field in the report.
    close_z = _grasp_height(target, -0.011, top_z=top)
    assert close_z < 0.075


def test_cavity_planar_fit_repairs_depth_stretched_contour() -> None:
    uu, vv = np.meshgrid(np.linspace(100, 180, 12), np.linspace(80, 140, 10))
    pixels = np.column_stack([uu.ravel(), vv.ravel()])
    world = np.column_stack(
        [
            0.0015 * pixels[:, 0] + 0.20,
            0.0014 * pixels[:, 1] - 0.30,
        ]
    )
    contour_pixels = np.asarray([[100, 80], [180, 80], [180, 140], [100, 140]])
    measured = np.asarray([[-0.30, -0.20], [0.90, -0.20], [0.90, 0.40], [-0.30, 0.40]])

    repaired, evidence = _planar_boundary_repair(world, pixels, measured, contour_pixels)

    assert evidence["used"] is True
    assert max(np.ptp(repaired, axis=0)) < 0.15


def test_appearance_identity_drops_only_preaction_spatial_qualifier() -> None:
    assert _appearance_label("the butter at the front") == "butter"
    assert _appearance_label("red mug on the right") == "red mug"
    assert _appearance_label("small black bowl") == "small black bowl"
    assert _has_transient_spatial_identity("the butter at the front") is True
    assert _has_transient_spatial_identity("small black bowl") is False


def test_compound_stack_then_place_keeps_support_out_of_pick_selector() -> None:
    instruction = "stack the left bowl on the right bowl and place them in the tray"
    goals = [
        {"skill": "pickplace", "pick": "the right bowl", "place": "the tray"},
        {"skill": "stack", "pick": "the left bowl", "support": "the right bowl"},
    ]

    assert _plan_semantic_errors(instruction, goals) == []


def test_surface_height_rejects_floor_mode_far_below_small_object() -> None:
    class Runtime:
        def probe_pixels(self, pixels):
            del pixels
            return np.asarray([[0.0, 0.0, -0.435]] * 24)

    target = GroundedTarget(
        label="butter",
        pose=(0.656, 0.21, 0.0),
        confidence=0.8,
        metadata={
            "bbox": [210, 210, 240, 230],
            "top_z": 0.0056,
            "extent": [0.075, 0.041, 0.0178],
        },
    )

    surface = _surface_height(Runtime(), target)

    assert np.isclose(surface, -0.0089)


def test_surface_height_accepts_support_below_incomplete_vertical_slab_cloud() -> None:
    """A top-facing mask need not span a tall object's full vertical body."""

    class Runtime:
        def probe_pixels(self, pixels):
            del pixels
            # A coherent tabletop mode 13.5 cm below the top-pages cloud.
            return np.asarray([[0.0, 0.0, -0.0306]] * 24)

    target = GroundedTarget(
        label="middle upright slab",
        pose=(0.593, -0.148, 0.1052),
        confidence=0.8,
        metadata={
            "bbox": [255, 272, 332, 410],
            "top_z": 0.1101,
            # RGB-D only saw the top-facing strip, not the vertical body.
            "extent": [0.1039, 0.0216, 0.0092],
        },
    )

    surface = _surface_height(Runtime(), target)

    assert np.isclose(surface, -0.0306)


def test_planar_slide_is_not_mistaken_for_a_lifted_grasp() -> None:
    before = GroundedTarget(
        label="butter",
        pose=(0.6, 0.2, 0.0),
        confidence=0.9,
        metadata={"n_points": 200, "extent": [0.08, 0.04, 0.02]},
    )

    class Runtime:
        def localize_many(self, labels):
            return {
                labels[0]: GroundedTarget(
                    label=labels[0],
                    pose=(0.5, 0.2, 0.002),
                    confidence=0.9,
                    metadata={"n_points": 200, "extent": [0.08, 0.04, 0.02]},
                )
            }

        def localize(self, *args, **kwargs):
            del args, kwargs
            return GroundedTarget(
                label="butter",
                pose=(0.5, 0.2, 0.002),
                confidence=0.9,
                metadata={"n_points": 200, "extent": [0.08, 0.04, 0.02]},
            )

        def ee_pose(self):
            return {"position": [0.5, 0.2, 0.12]}

    assert _held_object_left_the_table(Runtime(), "butter", before) is False


def test_transient_selector_at_original_scene_pose_defers_to_wrist() -> None:
    before = GroundedTarget(
        label="the butter at the back",
        pose=(0.57, 0.21, 0.002),
        confidence=0.9,
        metadata={"n_points": 300, "extent": [0.08, 0.04, 0.02]},
    )

    class Runtime:
        def localize_many(self, labels):
            return {
                labels[0]: GroundedTarget(
                    label=labels[0],
                    pose=(0.571, 0.209, 0.003),
                    confidence=0.9,
                    metadata={"n_points": 280, "extent": [0.08, 0.04, 0.02]},
                )
            }

        def localize(self, label, **kwargs):
            if kwargs.get("close_view"):
                return GroundedTarget(
                    label=label,
                    pose=(0.57, 0.21, 0.14),
                    confidence=0.9,
                    metadata={"n_points": 200, "extent": [0.08, 0.04, 0.02]},
                )
            return before

        def ee_pose(self):
            return {"position": [0.57, 0.21, 0.15]}

    assert _held_object_left_the_table(Runtime(), "the butter at the back", before) is True


def test_transient_local_track_can_confirm_the_held_instance() -> None:
    before = GroundedTarget(
        label="book in the middle",
        pose=(0.59, -0.15, 0.10),
        confidence=0.9,
        metadata={
            "n_points": 800,
            "extent": [0.104, 0.022, 0.01],
            "bbox": [255, 272, 332, 410],
        },
    )

    class Runtime:
        def track_target(self, target, **kwargs):
            del target, kwargs
            return GroundedTarget(
                label="book in the middle",
                pose=(0.60, -0.14, 0.19),
                confidence=0.8,
                metadata={"n_points": 700, "extent": [0.10, 0.03, 0.05]},
            )

        def ee_pose(self):
            return {"position": [0.59, -0.15, 0.35]}

    assert _held_object_left_the_table(Runtime(), "book in the middle", before) is True


def test_broad_object_scene_rise_requires_wrist_held_confirmation() -> None:
    before = GroundedTarget(
        label="frying pan",
        pose=(0.60, 0.05, 0.02),
        confidence=0.9,
        metadata={"n_points": 1000, "extent": [0.32, 0.18, 0.06]},
    )

    class Runtime:
        def localize(self, label, **kwargs):
            if kwargs.get("close_view"):
                return GroundedTarget(
                    label=label,
                    pose=(0.20, -0.20, 0.04),
                    confidence=0.9,
                    metadata={"n_points": 800, "extent": [0.32, 0.18, 0.12]},
                )
            # The pan was tipped, so its centroid rose without being carried.
            return GroundedTarget(
                label=label,
                pose=(0.61, 0.05, 0.07),
                confidence=0.9,
                metadata={"n_points": 900, "extent": [0.32, 0.18, 0.12]},
            )

        def ee_pose(self):
            return {"position": [0.60, 0.05, 0.25]}

    assert _held_object_left_the_table(Runtime(), "frying pan", before) is False


def test_transient_selector_rebind_requires_wrist_held_confirmation() -> None:
    before = GroundedTarget(
        label="the butter at the front",
        pose=(0.70, 0.06, 0.008),
        confidence=0.9,
        metadata={"n_points": 300, "extent": [0.08, 0.04, 0.02]},
    )

    class Runtime:
        def localize_many(self, labels):
            return {
                labels[0]: GroundedTarget(
                    label=labels[0],
                    pose=(0.57, 0.21, 0.002),
                    confidence=0.9,
                    metadata={"n_points": 280, "extent": [0.08, 0.04, 0.02]},
                )
            }

    # A distant scene rebind can mean either that the intended object was
    # lifted or that a failed grasp merely displaced it. Without a wrist-view
    # object near the hand, it is not positive held-object evidence.
    assert _held_object_left_the_table(Runtime(), "the butter at the front", before) is False


def test_low_grasp_height_stays_inside_thin_object_body() -> None:
    surface_z, top_z = -0.0092, 0.0056
    low = _low_grasp_height(surface_z, top_z)

    assert surface_z + 0.007 < low < top_z
    # Preserve the established 2.5 cm low contact for a normal-height vessel.
    assert np.isclose(_low_grasp_height(-0.01, 0.05), 0.015)


def test_flush_thin_elongated_object_tries_nondestructive_pinch_first() -> None:
    points = _solid_cloud(0.075, 0.038, 0.017)
    attempts = _attempts(points, object_height=0.017)
    assert attempts[:3] == (
        ("top_down", "top"),
        ("scoop_edge", "edge"),
        ("tilted_edge", "edge"),
    )
    assert np.isclose(_edge_grasp_height(-0.011, 0.006), 0.0065)


def test_libero_runtime_long_chain_horizon_defaults_to_8000() -> None:
    parameter = inspect.signature(LiberoPrimitiveRuntime).parameters["max_steps"]
    assert parameter.default == 8000


def test_libero_runtime_maps_public_seed_to_one_based_simulator_seed() -> None:
    applied: list[int] = []

    class FakeEnv:
        def reset(self, *, seed: int) -> None:
            applied.append(seed)

    runtime = LiberoPrimitiveRuntime.__new__(LiberoPrimitiveRuntime)
    runtime.seed = 0
    runtime.public_seed_indexed = True
    runtime._env = FakeEnv()
    runtime.reset_cameras = lambda: None
    runtime._points_cache = {}
    runtime._grasp_origin = {}

    runtime.reset()
    runtime.reset(seed=3)

    assert applied == [1, 4]


def test_prismatic_measurement_rejects_motion_beyond_commanded_path() -> None:
    assert prismatic_measurement_within_command(0.14, 0.13, 0.12) is True
    assert prismatic_measurement_within_command(0.199, -0.184, 0.12) is False


def test_topdown_quaternion_yaw_roundtrip() -> None:
    yaw = np.radians(67.0)
    quat = np.asarray([0.0, np.cos(yaw / 2), np.sin(yaw / 2), 0.0])
    assert np.isclose(_topdown_yaw(quat), yaw)


def test_hinge_arc_rotates_wrist_about_world_z_without_losing_tilt() -> None:
    start = np.asarray([0.0, 1.0, 0.0, 0.0])
    rotated = rotate_quaternion_world_z(start, np.pi / 2)

    assert np.allclose(rotated, [0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-7)
    assert np.isclose(np.linalg.norm(rotated), 1.0)


def test_pca_yaw_aligns_finger_motion_with_short_axis() -> None:
    long_axis = np.asarray([1.0, 0.0])
    yaw = _pca_grasp_yaw(long_axis)
    # Canonical top-down fingers translate along -Y.  Under world-Z yaw the
    # closing direction is (sin(yaw), -cos(yaw)), which must be perpendicular
    # to the measured long footprint axis.
    closing = np.asarray([np.sin(yaw), -np.cos(yaw)])
    assert abs(float(closing @ long_axis)) < 1e-7
    assert abs(float(closing @ np.asarray([0.0, 1.0]))) > 0.99


def test_tilted_pca_quat_is_sign_invariant_and_lowers_one_finger() -> None:
    expected = np.asarray(
        [
            -np.sin(np.deg2rad(12.5)),
            np.cos(np.deg2rad(12.5)),
            0.0,
            0.0,
        ]
    )
    assert np.allclose(_tilted_pca_quat([1.0, 0.0], angle_deg=25.0), expected, atol=1e-6)
    assert np.allclose(_tilted_pca_quat([-1.0, 0.0], angle_deg=25.0), expected, atol=1e-6)


def test_control_contact_separates_raised_knob_from_appliance_plate() -> None:
    xs = np.linspace(-0.06, 0.06, 25)
    ys = np.linspace(-0.06, 0.06, 25)
    plate = np.asarray([[x, y, 0.015] for x in xs for y in ys])
    knob = np.asarray(
        [
            [0.025 + x, -0.01 + y, z]
            for x in np.linspace(-0.012, 0.012, 8)
            for y in np.linspace(-0.009, 0.009, 7)
            for z in np.linspace(0.025, 0.075, 10)
        ]
    )

    fit = raised_control_contact(np.concatenate([plate, knob]), np.asarray([0.025, -0.01]))

    assert fit is not None
    assert np.linalg.norm(fit["pose"][:2] - [0.025, -0.01]) < 0.005
    assert fit["top_z"] > 0.065
    assert max(fit["extent"][:2]) < 0.04


class _StateCheckRuntime:
    def localize(self, label, **kwargs):
        return GroundedTarget(
            label=label,
            pose=(0.5, 0.2, 0.03),
            metadata={"n_points": 1200, "extent": [0.2, 0.2, 0.03]},
        )

    def inspect(self, prompt, *, options, target):
        return {"ok": True, "answer": options[1]}


def test_state_check_can_use_directional_motion_when_visual_state_is_invisible() -> None:
    session = Session(_StateCheckRuntime(), object())
    session.steps.append(
        Step(
            action="actuate_control",
            args={"target": "stove", "goal": "on"},
            report={"motion_goal_match": True},
        )
    )
    check = Step(action="check_state", args={"target": "stove", "goal": "on"})

    session._do_check_state(check)

    assert check.success is True
    assert check.report["visual_goal_match"] is False
    assert check.report["evidence"] == "measured_motion"


def test_state_check_defers_semantics_and_preserves_zero_motion() -> None:
    runtime = _StateCheckRuntime()
    session = Session(runtime, object())
    session.steps.append(
        Step(
            action="articulate",
            args={"target": "cabinet", "goal": "open"},
            report={
                "motion_goal_match": False,
                "measured_displacement_cm": 0.0,
            },
        )
    )
    check = Step(action="check_state", args={"target": "cabinet", "goal": "open"})
    runtime.inspect = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("check_state must not invoke a semantic VLM")
    )

    session._do_check_state(check)

    assert check.success is False
    assert check.report["visual_goal_match"] is False
    assert check.report["motion_contradicts_visual"] is True
    assert check.report["semantic_state_verification"] == "deferred_to_visual_agent"


def test_later_articulation_preflight_failure_cannot_erase_no_motion() -> None:
    runtime = _StateCheckRuntime()
    runtime.inspect = lambda prompt, *, options, target: {"ok": True, "answer": options[0]}
    session = Session(runtime, object())
    session.steps.extend(
        [
            Step(
                action="articulate",
                args={"target": "microwave", "goal": "open"},
                report={
                    "mechanism": "revolute",
                    "motion_goal_match": False,
                    "planned_hinge_rotation_deg": -38.0,
                    "measured_hinge_rotation_deg": -5.0,
                    "measured_displacement_cm": 1.0,
                    "motion_measurement_reliable": True,
                },
            ),
            Step(
                action="articulate",
                args={"target": "microwave", "goal": "open"},
                report={"preflight_only": True, "failure_mode": "not_grounded"},
            ),
        ]
    )
    check = Step(action="check_state", args={"target": "microwave", "goal": "open"})

    session._do_check_state(check)

    assert check.success is False
    assert check.report["motion_contradicts_visual"] is True


def test_closed_hinge_endpoint_semantics_are_deferred_to_agent() -> None:
    runtime = _StateCheckRuntime()
    runtime.inspect = lambda prompt, *, options, target: {"ok": True, "answer": options[0]}
    session = Session(runtime, object())
    session.steps.extend(
        [
            Step(
                action="articulate",
                args={"target": "microwave", "goal": "closed"},
                report={
                    "mechanism": "revolute",
                    "motion_goal_match": False,
                    "planned_hinge_rotation_deg": 38.0,
                    "measured_hinge_rotation_deg": 18.0,
                    "measured_displacement_cm": 4.0,
                    "motion_measurement_reliable": True,
                    "contact_mode": "retained_grasp",
                    "preflight_only": False,
                },
            ),
            Step(
                action="articulate",
                args={"target": "microwave", "goal": "closed"},
                report={
                    "mechanism": "revolute",
                    "preflight_only": True,
                    "grounding_failure": "handle",
                },
            ),
        ]
    )
    check = Step(action="check_state", args={"target": "microwave", "goal": "closed"})

    session._do_check_state(check)

    assert check.success is False
    assert "closed_handle_endpoint" not in check.report["aggregate_motion"]
    assert check.report["semantic_state_verification"] == "deferred_to_visual_agent"


def test_control_state_does_not_reuse_an_internal_visual_verdict() -> None:
    runtime = _StateCheckRuntime()
    runtime.inspect = lambda prompt, *, options, target: {"ok": True, "answer": options[0]}
    session = Session(runtime, object())
    session.steps.append(
        Step(
            action="actuate_control",
            args={"target": "stove", "goal": "on"},
            report={
                "motion_goal_match": False,
                "measured_control_rotation_deg": -13.2,
                "visual_goal_match": True,
                "preflight_only": False,
            },
        )
    )
    check = Step(action="check_state", args={"target": "stove", "goal": "on"})

    session._do_check_state(check)

    assert check.success is False
    assert check.report["aggregate_motion"] == {}
    assert check.report["semantic_state_verification"] == "deferred_to_visual_agent"


def test_control_state_rejects_motion_when_action_visual_saw_other_state() -> None:
    runtime = _StateCheckRuntime()
    runtime.inspect = lambda prompt, *, options, target: {"ok": True, "answer": options[0]}
    session = Session(runtime, object())
    session.steps.append(
        Step(
            action="actuate_control",
            args={"target": "stove", "goal": "on"},
            report={
                "motion_goal_match": False,
                "measured_control_rotation_deg": -58.8,
                "visual_goal_match": False,
                "preflight_only": False,
            },
        )
    )
    check = Step(action="check_state", args={"target": "stove", "goal": "on"})

    session._do_check_state(check)

    assert check.success is False


def test_state_check_reports_but_does_not_trust_accumulated_prismatic_progress() -> None:
    runtime = _StateCheckRuntime()
    runtime.inspect = lambda prompt, *, options, target: {"ok": True, "answer": options[1]}
    session = Session(runtime, object())
    for progress in (-7.5, -8.6):
        session.steps.append(
            Step(
                action="articulate",
                args={"target": "cabinet", "goal": "closed"},
                report={
                    "mechanism": "prismatic",
                    "motion_goal_match": False,
                    "planned_linear_stroke_cm": 20.0,
                    "directed_progress_cm": progress,
                    "measured_displacement_cm": abs(progress),
                    "contact_mode": "retained_grasp",
                    "preflight_only": False,
                    "motion_measurement_reliable": True,
                },
            )
        )
    check = Step(action="check_state", args={"target": "cabinet", "goal": "closed"})

    session._do_check_state(check)

    assert check.success is False
    assert check.report["aggregate_motion"] == {
        "mechanism": "prismatic",
        "cumulative_directed_progress_cm": 16.1,
        "expected_progress_cm": 16.0,
        "endpoint_stall": False,
        "compressive_endpoint_consensus": False,
        "action_count": 2,
        "reliable_action_count": 2,
    }


def test_state_check_motion_history_is_scoped_to_requested_part() -> None:
    session = Session(_StateCheckRuntime(), object())
    for part in ("top drawer", "bottom drawer"):
        session.steps.append(
            Step(
                action="articulate",
                args={
                    "target": "cabinet",
                    "part": part,
                    "goal": "closed",
                },
                report={
                    "mechanism": "prismatic",
                    "motion_goal_match": False,
                    "planned_linear_stroke_cm": 20.0,
                    "directed_progress_cm": -8.0,
                    "measured_displacement_cm": 8.0,
                    "contact_mode": "contact_push",
                    "preflight_only": False,
                    "motion_measurement_reliable": True,
                },
            )
        )
    check = Step(
        action="check_state",
        args={
            "target": "cabinet",
            "part": "bottom drawer",
            "goal": "closed",
        },
    )

    session._do_check_state(check)

    assert check.report["part"] == "bottom drawer"
    assert check.report["aggregate_motion"]["action_count"] == 1
    assert check.report["aggregate_motion"]["cumulative_directed_progress_cm"] == 8.0


def test_state_check_accepts_contacted_prismatic_endpoint_stall() -> None:
    runtime = _StateCheckRuntime()
    runtime.inspect = lambda prompt, *, options, target: {"ok": True, "answer": options[0]}
    session = Session(runtime, object())
    for progress in (-9.0, -0.4):
        session.steps.append(
            Step(
                action="articulate",
                args={"target": "cabinet", "goal": "closed"},
                report={
                    "mechanism": "prismatic",
                    "motion_goal_match": False,
                    "planned_linear_stroke_cm": 20.0,
                    "directed_progress_cm": progress,
                    "measured_displacement_cm": abs(progress),
                    "contact_mode": "contact_push",
                    "preflight_only": False,
                    "motion_measurement_reliable": True,
                },
            )
        )
    check = Step(action="check_state", args={"target": "cabinet", "goal": "closed"})

    session._do_check_state(check)

    assert check.success is True
    assert check.report["aggregate_motion"]["endpoint_stall"] is True


def test_state_check_preserves_compressive_endpoint_consensus() -> None:
    runtime = _StateCheckRuntime()
    runtime.inspect = lambda prompt, *, options, target: {"ok": True, "answer": options[0]}
    session = Session(runtime, object())
    session.steps.append(
        Step(
            action="articulate",
            args={"target": "cabinet", "goal": "closed"},
            report={
                "mechanism": "prismatic",
                "motion_goal_match": True,
                "compressive_endpoint_consensus": True,
                "planned_linear_stroke_cm": 21.0,
                "directed_progress_cm": -14.0,
                "measured_displacement_cm": 15.7,
                "contact_mode": "closed_finger_external_face_push",
                "preflight_only": False,
                "motion_measurement_reliable": True,
            },
        )
    )
    check = Step(action="check_state", args={"target": "cabinet", "goal": "closed"})

    session._do_check_state(check)

    assert check.success is True
    assert check.report["aggregate_motion"]["compressive_endpoint_consensus"] is True


def test_state_check_requires_more_than_partial_hinge_swing() -> None:
    runtime = _StateCheckRuntime()
    runtime.inspect = lambda prompt, *, options, target: {"ok": True, "answer": options[0]}
    session = Session(runtime, object())
    for measured in (-4.8, 2.5, -30.9):
        session.steps.append(
            Step(
                action="articulate",
                args={"target": "microwave", "goal": "open"},
                report={
                    "mechanism": "revolute",
                    "motion_goal_match": False,
                    "planned_hinge_rotation_deg": -82.0,
                    "measured_hinge_rotation_deg": measured,
                    "measured_displacement_cm": abs(measured) / 2,
                    "motion_measurement_reliable": True,
                },
            )
        )
    check = Step(action="check_state", args={"target": "microwave", "goal": "open"})

    session._do_check_state(check)

    assert check.success is False
    assert check.report["aggregate_motion"]["cumulative_directed_rotation_deg"] == 33.2
    assert check.report["motion_contradicts_visual"] is True


class _WrongLoadRuntime:
    def __init__(self) -> None:
        self.holding = True
        self.moves: list[list[float]] = []

    def verify_grasp(self, label: str) -> bool:
        return self.holding

    def move_to(self, position) -> None:
        self.moves.append([float(v) for v in position])

    def open_gripper(self) -> None:
        self.holding = False


class _ArticulationRegroundRuntime:
    def __init__(self, local_ok: bool) -> None:
        self.local_ok = local_ok
        self.detectors: list[str] = []

    def localize(self, label, **kwargs):
        detector = kwargs.get("detector", "vlm_bbox_sam3")
        self.detectors.append(detector)
        confidence = 1.0 if detector != "sam3" or self.local_ok else 0.0
        return GroundedTarget(
            label=label,
            pose=(0.4, 0.1, 0.07),
            confidence=confidence,
            metadata={"n_points": 80, "extent": [0.08, 0.02, 0.02]},
        )


def test_articulation_motion_reground_uses_local_mask_before_vlm() -> None:
    runtime = _ArticulationRegroundRuntime(local_ok=True)

    target, source = _reground_visible(runtime, "drawer handle", contact=True)

    assert target is not None
    assert source == "sam3"
    assert runtime.detectors == ["sam3"]


def test_articulation_motion_reground_falls_back_when_local_mask_is_absent() -> None:
    runtime = _ArticulationRegroundRuntime(local_ok=False)

    target, source = _reground_visible(runtime, "drawer handle", contact=True)

    assert target is not None
    assert source == "vlm_bbox_sam3_fallback"
    assert runtime.detectors == ["sam3", "vlm_bbox_sam3"]


def test_closing_can_ground_a_broad_moving_panel_without_a_handle() -> None:
    fixture = GroundedTarget(
        label="cabinet",
        pose=(0.6, -0.25, 0.12),
        confidence=0.9,
        metadata={"n_points": 2000, "extent": [0.3, 0.25, 0.3]},
    )

    class Runtime:
        def localize(self, label, **kwargs):
            del kwargs
            if label.startswith("external front panel"):
                return GroundedTarget(
                    label=label,
                    pose=(0.61, -0.12, 0.10),
                    confidence=0.9,
                    metadata={"n_points": 800, "extent": [0.18, 0.03, 0.10]},
                )
            return GroundedTarget(label=label, pose=(0.0, 0.0, 0.0), confidence=0.0)

    trace: list[dict] = []
    panel = _ground_moving_panel(Runtime(), fixture, "cabinet", "bottom drawer", trace)

    assert panel is not None
    assert panel.label.startswith("external front panel")
    assert any(row["event"] == "ground_panel" and row["usable"] for row in trace)


def test_generic_large_panel_cannot_override_requested_drawer_face() -> None:
    fixture = GroundedTarget(
        label="cabinet",
        pose=(0.6, -0.25, 0.12),
        confidence=0.9,
        metadata={"n_points": 2000, "extent": [0.3, 0.25, 0.3]},
    )

    class Runtime:
        def localize(self, label, **kwargs):
            del kwargs
            if "bottom drawer" in label:
                extent = [0.14, 0.03, 0.07]
            else:
                extent = [0.40, 0.04, 0.28]
            return GroundedTarget(
                label=label,
                pose=(0.61, -0.12, 0.10),
                confidence=0.9,
                metadata={"n_points": 800, "extent": extent},
            )

    panel = _ground_moving_panel(Runtime(), fixture, "cabinet", "bottom drawer", [])

    assert panel is not None
    assert "bottom drawer" in panel.label


def test_wrong_object_grasp_is_returned_before_failure() -> None:
    runtime = _WrongLoadRuntime()
    target = GroundedTarget(
        label="requested object",
        pose=(0.2, -0.1, 0.05),
        metadata={"top_z": 0.09},
    )
    trace: list[dict] = []
    assert _return_unintended_grasp(runtime, target, trace)
    assert not runtime.holding
    assert runtime.moves[0] == [0.2, -0.1, 0.15]
    assert runtime.moves[-1] == [0.2, -0.1, 0.23]
    assert trace[-1]["released"] is True


class _AmbiguousHeldObjectRuntime:
    def ee_pose(self):
        return {"position": [0.50, 0.20, 0.30]}

    def localize(self, label, **kwargs):
        if kwargs.get("close_view"):
            return GroundedTarget(
                label=label,
                pose=(0.54, 0.18, 0.22),
                metadata={"n_points": 500, "extent": [0.1, 0.1, 0.05]},
            )
        # Relational language changed after lifting and now selects a distant
        # identical object, so fresh visual offset must be rejected.
        return GroundedTarget(
            label=label,
            pose=(0.80, 0.20, 0.04),
            metadata={"n_points": 500, "extent": [0.1, 0.1, 0.05]},
        )


def test_carry_offset_uses_isolated_wrist_view_for_identical_objects() -> None:
    source = GroundedTarget(
        label="middle bowl",
        pose=(0.54, 0.18, 0.04),
        metadata={"n_points": 500, "extent": [0.1, 0.1, 0.05]},
    )
    trace: list[dict] = []

    offset = _carry_offset(_AmbiguousHeldObjectRuntime(), "middle bowl", trace, source=source)

    assert np.allclose(offset, [0.04, -0.02])
    assert trace[-1]["source"] == "wrist_rgbd"


def test_centring_rejects_distant_duplicate_and_uses_attached_wrist_object() -> None:
    class Runtime(_AmbiguousHeldObjectRuntime):
        def __init__(self):
            self.moves = []

        def move_to(self, position):
            self.moves.append([float(v) for v in position])

        def localize_many(self, labels):
            return {label: self.localize(label) for label in labels}

    runtime = Runtime()
    trace: list[dict] = []

    centred = _centre_over(
        runtime,
        "middle bowl",
        np.asarray([0.55, 0.20]),
        0.30,
        trace,
        rounds=1,
    )

    assert np.allclose(centred, [0.56, 0.22])
    assert np.allclose(runtime.moves[-1], [0.56, 0.22, 0.30])
    assert trace[-1]["attachment_source"] == "wrist_rgbd"


def test_full_plan_preserves_relational_labels_and_safe_schema() -> None:
    goal = _normalise_goal(
        {
            "skill": "stack",
            "pick": "the left black bowl",
            "support": "the right black bowl",
            "x": 0.42,
        }
    )
    assert goal == {
        "skill": "stack",
        "pick": "the left black bowl",
        "support": "the right black bowl",
    }


def test_full_plan_does_not_guess_missing_state_goal() -> None:
    assert _normalise_goal({"skill": "actuate_control", "target": "stove"}) is None
    assert _normalise_goal({"skill": "articulate", "target": "drawer"}) is None


def test_full_plan_semantics_rejects_visual_entity_renaming_and_dropped_top() -> None:
    assert _plan_semantic_errors(
        "put the ketchup on top of the tray",
        [{"skill": "pickplace", "pick": "ketchup box", "place": "tray"}],
    ) == [
        "goal 1 pick invents tokens ['box']",
        "instruction entity/qualifier tokens were dropped: ['top']",
    ]


def test_full_plan_semantics_preserves_selector_attachment_to_pick() -> None:
    errors = _plan_semantic_errors(
        "put the butter at the front in the top drawer of the cabinet",
        [
            {
                "skill": "pickplace",
                "pick": "butter",
                "place": "front in the top drawer of the cabinet",
            }
        ],
    )

    assert any("selector changed attachment" in error for error in errors)
    assert (
        _plan_semantic_errors(
            "put the black bowl at the front on the plate",
            [
                {
                    "skill": "stack",
                    "pick": "black bowl at the front",
                    "support": "plate",
                }
            ],
        )
        == []
    )
    # Here "front" belongs to the destination noun, not the picked book.
    assert (
        _plan_semantic_errors(
            "pick up the book and place it in the front compartment of the caddy",
            [
                {
                    "skill": "insert",
                    "pick": "book",
                    "place": "front compartment of the caddy",
                }
            ],
        )
        == []
    )


def test_full_plan_resolves_pronoun_to_previous_destination_phrase() -> None:
    goals = _resolve_plan_pronouns(
        [
            {"skill": "pickplace", "pick": "frying pan", "place": "stove"},
            {"skill": "actuate_control", "target": "it", "goal": "on"},
        ]
    )
    assert goals[1]["target"] == "stove"


def test_full_plan_resolves_surface_pronoun_before_current_pick() -> None:
    goals = _resolve_plan_pronouns(
        [
            {
                "skill": "articulate",
                "target": "cabinet",
                "part": "top drawer",
                "goal": "closed",
            },
            {
                "skill": "pickplace",
                "pick": "black bowl",
                "place": "top of it",
            },
        ]
    )
    assert goals[1]["place"] == "top of cabinet"


def test_full_plan_repairs_planner_that_bound_surface_pronoun_to_drawer() -> None:
    goals = _resolve_plan_pronouns(
        [
            {
                "skill": "articulate",
                "target": "cabinet",
                "part": "top drawer",
                "goal": "closed",
            },
            {
                "skill": "pickplace",
                "pick": "black bowl",
                "place": "top drawer of the cabinet",
            },
        ],
        "close the top drawer of the cabinet and put the black bowl on top of it",
    )

    assert goals[1]["place"] == "top of cabinet"


def test_full_controller_passes_planner_transport_family_to_child(
    monkeypatch,
) -> None:
    class PlannerSession:
        def __init__(self, runtime, skill) -> None:
            del runtime, skill

        def frame(self):
            return "initial-frame"

    captured = []

    monkeypatch.setattr(full_react, "Session", PlannerSession)
    monkeypatch.setattr(
        full_react,
        "plan_instruction",
        lambda instruction, image, *, model: (
            [{"skill": "insert", "pick": "book", "place": "shelf"}],
            "plan",
        ),
    )

    def fake_run(runtime, skill, episode, *, verbose, session):
        del runtime, skill, verbose, session
        captured.append(episode)
        return full_react.Outcome(
            success=True,
            turns=1,
            steps=[
                {
                    "turn": 0,
                    "action": "done",
                    "args": {},
                    "thought": "verified",
                    "observation": "done",
                    "report": {},
                }
            ],
        )

    monkeypatch.setattr(full_react, "run_episode", fake_run)
    outcome = full_react.run_full_episode(
        object(),
        object(),
        full_react.FullEpisode(instruction="put the book on the shelf"),
    )

    assert outcome.success is True
    assert len(captured) == 1
    assert captured[0].initial_skill == "insert"


def test_anytime_collection_skips_failed_sibling_and_replans(
    monkeypatch,
) -> None:
    class PlannerSession:
        def __init__(self, runtime, skill) -> None:
            del runtime, skill
            self.frames = 0

        def frame(self):
            self.frames += 1
            return f"frame-{self.frames}"

    planner_calls: list[str] = []

    def fake_plan(instruction, image, *, model):
        del image, model
        planner_calls.append(instruction)
        if len(planner_calls) == 1:
            return (
                [{"skill": "pickplace", "pick": "blue can", "place": "basket"}],
                "initial-plan",
            )
        return (
            [{"skill": "pickplace", "pick": "orange box", "place": "basket"}],
            "continuation-plan",
        )

    transported: list[str] = []

    def fake_run(runtime, skill, episode, *, verbose, session):
        del runtime, skill, verbose, session
        transported.append(episode.pick)
        return full_react.Outcome(
            success=episode.pick == "orange box",
            turns=1,
            steps=[],
            stopped="verified" if episode.pick == "orange box" else "empty grasp",
        )

    monkeypatch.setattr(full_react, "Session", PlannerSession)
    monkeypatch.setattr(full_react, "plan_instruction", fake_plan)
    monkeypatch.setattr(full_react, "run_episode", fake_run)

    outcome = full_react.run_full_episode(
        object(),
        object(),
        full_react.FullEpisode(
            instruction="put all tabletop objects in the basket",
            continue_after_subgoal_failure=True,
            max_replans=1,
        ),
    )

    assert transported == ["blue can", "orange box"]
    assert len(planner_calls) == 2
    assert "blue can" in planner_calls[1]
    assert outcome.success is False
    assert outcome.stopped == "subgoal verification results: [False, True]"
    assert any("ANYTIME REPLAN 1" in row for row in outcome.reflections)


def test_default_compositional_task_still_fails_closed(monkeypatch) -> None:
    class PlannerSession:
        def __init__(self, runtime, skill) -> None:
            del runtime, skill

        def frame(self):
            return "frame"

    monkeypatch.setattr(full_react, "Session", PlannerSession)
    monkeypatch.setattr(
        full_react,
        "plan_instruction",
        lambda instruction, image, *, model: (
            [
                {"skill": "pickplace", "pick": "first object", "place": "tray"},
                {"skill": "pickplace", "pick": "second object", "place": "tray"},
            ],
            "plan",
        ),
    )
    transported: list[str] = []

    def fake_run(runtime, skill, episode, *, verbose, session):
        del runtime, skill, verbose, session
        transported.append(episode.pick)
        return full_react.Outcome(False, 1, steps=[], stopped="empty grasp")

    monkeypatch.setattr(full_react, "run_episode", fake_run)
    outcome = full_react.run_full_episode(
        object(),
        object(),
        full_react.FullEpisode(instruction="put two objects in the tray"),
    )

    assert transported == ["first object"]
    assert outcome.stopped == "subgoal verification results: [False]"


def test_full_planner_uses_semantic_warnings_as_agent_advice(
    monkeypatch,
) -> None:
    replies = iter(
        [
            (
                "bad",
                {
                    "goals": [
                        {
                            "skill": "pickplace",
                            "pick": "ketchup box",
                            "place": "wooden tray",
                        }
                    ]
                },
            ),
            (
                "good",
                {
                    "goals": [
                        {
                            "skill": "pickplace",
                            "pick": "ketchup",
                            "place": "tray",
                        }
                    ]
                },
            ),
        ]
    )
    calls = []

    def fake_ask(system, prompt, *, images, model, max_tokens):
        del system, images, model, max_tokens
        calls.append(prompt)
        return next(replies)

    monkeypatch.setattr(full_react, "_ask_json_reply", fake_ask)
    goals, trace = full_react.plan_instruction(
        "pick up the ketchup and put it in the tray", "frame", model="test"
    )

    assert goals == [{"skill": "pickplace", "pick": "ketchup", "place": "tray"}]
    assert len(calls) == 2
    assert "SEMANTIC ADVISORY" in trace
    assert "AGENT PLAN REVIEW" in trace
    assert "smallest field-level correction" in calls[1]


def test_semantic_checker_cannot_reject_agent_final_plan(monkeypatch) -> None:
    replies = iter(
        [
            (
                "first",
                {
                    "goals": [
                        {
                            "skill": "pickplace",
                            "pick": "ketchup box",
                            "place": "tray",
                        }
                    ]
                },
            ),
            (
                "agent-final",
                {
                    "goals": [
                        {
                            "skill": "pickplace",
                            "pick": "ketchup box",
                            "place": "tray",
                        }
                    ]
                },
            ),
        ]
    )

    def fake_ask(system, prompt, *, images, model, max_tokens):
        del system, prompt, images, model, max_tokens
        return next(replies)

    monkeypatch.setattr(full_react, "_ask_json_reply", fake_ask)
    goals, trace = full_react.plan_instruction("put the ketchup in the tray", "frame", model="test")

    assert goals == [
        {
            "skill": "pickplace",
            "pick": "ketchup box",
            "place": "tray",
        }
    ]
    assert "AGENT PLAN REVIEW\nagent-final" in trace


def test_push_instruction_is_a_valid_transport_plan(monkeypatch) -> None:
    calls = []

    def fake_ask(system, prompt, *, images, model, max_tokens):
        del system, images, model, max_tokens
        calls.append(prompt)
        return (
            "agent-plan",
            {
                "goals": [
                    {
                        "skill": "pickplace",
                        "pick": "cream cheese",
                        "place": "front of the stove",
                    }
                ]
            },
        )

    monkeypatch.setattr(full_react, "_ask_json_reply", fake_ask)
    goals, trace = full_react.plan_instruction(
        "Push the cream cheese to the front of the stove", "frame", model="test"
    )

    assert goals == [
        {
            "skill": "pickplace",
            "pick": "cream cheese",
            "place": "front of the stove",
        }
    ]
    assert trace == "agent-plan"
    assert len(calls) == 1


class _AgentAuthorityStateSession:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.calls: list[tuple[str, dict]] = []
        self.checks = 0

    def frame(self):
        return "before"

    def run(self, action, args, thought=""):
        del thought
        self.actions.append(action)
        self.calls.append((action, dict(args)))
        if action == "articulate":
            return Step(
                action=action,
                args=args,
                image="after-motion",
                success=False,
                report={
                    "failure_mode": "verification_inconclusive",
                    "visual_state_before": "the mechanism is open",
                    "visual_state_after": "the mechanism is open",
                    "visual_goal_match": True,
                    "motion_goal_match": False,
                    "directed_progress_cm": 2.9,
                },
                observation="motion measurement did not confirm a full stroke",
            )
        self.checks += 1
        return Step(
            action=action,
            args=args,
            image=f"check-{self.checks}",
            success=False,
            report={
                "answer": "the target is open",
                "visual_goal_match": True,
                "motion_goal_match": False,
                "motion_contradicts_visual": self.checks > 1,
            },
            observation="auxiliary check withheld success",
        )


def test_visual_agent_overrides_motion_contradiction_and_stops_retrying(
    monkeypatch,
) -> None:
    replies = iter(
        [
            (
                "preflight",
                {
                    "goal_satisfied": False,
                    "what_you_see": "the state is uncertain before motion",
                    "reason": "inspect after one action",
                },
            ),
            (
                "authoritative",
                {
                    "goal_satisfied": True,
                    "retry": False,
                    "what_you_see": "the drawer is visibly open",
                    "args": {},
                    "reason": "the current image clearly satisfies open",
                },
            ),
        ]
    )

    def fake_ask(system, prompt, *, images, model, max_tokens):
        del system, prompt, images, model, max_tokens
        return next(replies)

    monkeypatch.setattr(full_react, "_ask_json_reply", fake_ask)
    session = _AgentAuthorityStateSession()
    outcome = full_react.Outcome(success=False, turns=0)
    success = full_react._run_state_goal(
        session,
        {
            "skill": "articulate",
            "target": "cabinet",
            "part": "top drawer",
            "goal": "open",
        },
        full_react.FullEpisode(instruction="open the top drawer", turns=12),
        outcome,
        0,
    )

    assert success is True
    assert session.actions == ["check_state", "articulate", "check_state"]
    assert sum(action == "articulate" for action in session.actions) == 1
    assert session.calls[0][1]["part"] == "top drawer"
    assert session.calls[2][1]["part"] == "top drawer"
    assert outcome.turns == 3
    assert any("STATE DECISION\nauthoritative" in value for value in outcome.reflections)


def test_visual_agent_can_confirm_state_before_any_motion(monkeypatch) -> None:
    def fake_ask(system, prompt, *, images, model, max_tokens):
        del system, prompt, images, model, max_tokens
        return "already-open", {
            "goal_satisfied": True,
            "what_you_see": "the top drawer is visibly pulled outward",
            "reason": "the requested open state is already satisfied",
        }

    monkeypatch.setattr(full_react, "_ask_json_reply", fake_ask)
    session = _AgentAuthorityStateSession()
    outcome = full_react.Outcome(success=False, turns=0)
    success = full_react._run_state_goal(
        session,
        {
            "skill": "articulate",
            "target": "cabinet",
            "part": "top drawer",
            "goal": "open",
        },
        full_react.FullEpisode(instruction="open the top drawer", turns=8),
        outcome,
        0,
    )

    assert success is True
    assert session.actions == ["check_state"]
    assert outcome.turns == 1
    assert outcome.reflections == ["STATE PREFLIGHT\nalready-open"]


def test_language_license_prevents_visual_planner_from_adding_transport() -> None:
    assert _language_allowed_skills("turn on the stove") == {"actuate_control"}
    assert _language_allowed_skills("turn on the stove and put the frying pan on it") == {
        "actuate_control",
        "pickplace",
        "insert",
    }
    assert "articulate" in _language_allowed_skills(
        "put the black bowl in the top drawer of the cabinet"
    )


def test_stack_is_licensed_only_by_an_explicit_stack_command() -> None:
    assert "stack" not in _language_allowed_skills("put the black bowl on the plate")
    assert "stack" not in _language_allowed_skills("place the mug on the movable bowl")
    assert "stack" in _language_allowed_skills("stack the front bowl on the middle bowl")


def test_full_plan_closes_drawer_after_insertion_but_before_unrelated_top_place() -> None:
    close = {
        "skill": "articulate",
        "target": "cabinet",
        "part": "top drawer",
        "goal": "closed",
    }
    insert = {"skill": "pickplace", "pick": "butter", "place": "top drawer of cabinet"}
    assert _order_goals([close, insert]) == [insert, close]
    top = {"skill": "pickplace", "pick": "bowl", "place": "top of cabinet"}
    assert _order_goals([top, close]) == [close, top]


def test_plan_rejects_pick_selector_duplicated_onto_destination() -> None:
    instruction = "put the butter at the back in the top drawer of the cabinet and close it"
    errors = _plan_semantic_errors(
        instruction,
        [
            {
                "skill": "pickplace",
                "pick": "butter at the back",
                "place": "back in the top drawer of the cabinet",
            }
        ],
    )

    assert any("also attached to the destination" in error for error in errors)


def test_full_plan_closes_target_drawer_after_transport_when_part_is_empty() -> None:
    transport = {
        "skill": "pickplace",
        "pick": "butter at the back",
        "place": "top drawer of the cabinet",
    }
    close = {
        "skill": "articulate",
        "target": "top drawer of the cabinet",
        "part": "",
        "goal": "closed",
    }
    assert _order_goals([transport, close]) == [transport, close]
    assert _order_goals([close, transport]) == [transport, close]


def test_full_plan_actuates_appliance_before_placing_cookware() -> None:
    pan = {
        "skill": "pickplace",
        "pick": "frying pan",
        "place": "stove",
    }
    control = {
        "skill": "actuate_control",
        "target": "stove",
        "goal": "on",
    }

    assert _order_goals([pan, control]) == [control, pan]
