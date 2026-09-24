"""Non-privileged, metered runtime surface passed to evolving controllers."""

from __future__ import annotations

from collections import Counter
import inspect
import math
from typing import Any


PUBLIC_RUNTIME_SPEC: dict[str, dict[str, str]] = {
    "observation": {
        "observe()": "RGB-D cameras plus proprioception; never native predicates",
        "_camera(observation, wrist=False)": (
            "compatibility camera decoder used by the retained visual ReAct agent; "
            "RGB-D/calibration only"
        ),
        "views()/move_camera()/reset_cameras()": "active multi-view perception",
        "crop_frame(bbox) / inspect(label, target=...)": "independent visual identity evidence",
    },
    "grounding": {
        "localize(label, **options) -> GroundedTarget": (
            "one semantic 3-D target with pose/confidence/metadata; pass this returned "
            "target to grasp/contact_grasp/place, or pass the same label and the public "
            "adapter will reuse the most recent grounding"
        ),
        "localize_many(labels)": "joint VLM grounding for genuinely distinct labels",
        "localize_across_views(label)": "recover an occluded or edge-on target",
        "localize_with_feedback(label, rejected)": "negative-feedback re-grounding",
        "object_points(label)": "world-frame points from the most recent mask",
        "probe_pixel(s)/probe_bbox/image_axes": "depth deprojection and calibrated image axes",
    },
    "state": {
        "ee_pose()": "EEF position, quaternion and gripper opening",
        "verify_grasp(label) / verify_place(source, destination)": "online advisory checks",
        "measure(label_a, label_b)": "perception-only relative displacement",
        "capability_manifest()": "machine-readable SDK availability and signatures",
    },
    "motion": {
        "move_to(position, quat=None, gripper=0, yaw=None)": "absolute Cartesian pose",
        "delta_move(dx=0, dy=0, dz=0)": "world-frame Cartesian correction",
        "execute_waypoints(waypoints)": "generic bounded Cartesian path",
        "linear_contact(start, end, ...)": "sampled straight contact path",
        "arc_contact(center, radius, start_angle, sweep, z, ...)": "sampled planar arc path",
        "grasp(target_or_label, *, strategy)": (
            "grounded transport grasp; target_or_label is a GroundedTarget returned by "
            "localize, or its string label"
        ),
        "contact_grasp(target_or_label, *, strategy, lift=0.0)": (
            "grounded contact grasp with an explicit bounded post-contact lift; accepts "
            "a GroundedTarget or its string label"
        ),
        "place(target_or_label, *, strategy)": (
            "grounded placement building block; accepts a GroundedTarget or its string label"
        ),
        "open_gripper/close_gripper/go_home/recover": "gripper and safe-retreat hooks",
        "ee_pose_arm(arm) / move_to_arm(arm, ...)": (
            "geometry-neutral per-arm Cartesian control when the embodiment has multiple arms"
        ),
        "move_bimanual(position0, position1, ...)": (
            "simultaneous two-arm Cartesian command; no task or object semantics"
        ),
        "open_gripper_arm(arm) / close_gripper_arm(arm)": (
            "indexed gripper control on a multi-arm embodiment"
        ),
    },
}


PERCEPTION_METHODS = frozenset(
    {
        "observe",
        "_camera",
        "views",
        "move_camera",
        "reset_cameras",
        "inspect",
        "crop_frame",
        "verify_destination_alias",
        "verify_destination_grounding",
        "verify_pick_grounding",
        "track_target",
        "track_near_hand",
        "localize",
        "localize_across_views",
        "list_objects",
        "localize_many",
        "localize_with_feedback",
        "object_points",
        "probe_pixel",
        "probe_pixels",
        "probe_bbox",
        "image_axes",
        "measure",
        "verify_grasp",
        "verify_place",
    }
)

MOTION_METHODS = frozenset(
    {
        "open_gripper",
        "close_gripper",
        "ee_pose",
        "move_to",
        "delta_move",
        "go_home",
        "grasp",
        "contact_grasp",
        "place",
        "recover",
        "execute_waypoints",
        "linear_contact",
        "arc_contact",
        "ee_pose_arm",
        "move_to_arm",
        "move_bimanual",
        "open_gripper_arm",
        "close_gripper_arm",
    }
)


class PublicRuntime:
    """Allow-list adapter that hides lifecycle, video and native oracle state."""

    __slots__ = ("__backend", "__calls", "__max_waypoints", "__targets")

    def __init__(self, backend: Any, *, max_waypoints: int = 128):
        object.__setattr__(self, "_PublicRuntime__backend", backend)
        object.__setattr__(self, "_PublicRuntime__calls", Counter())
        object.__setattr__(self, "_PublicRuntime__max_waypoints", max(1, int(max_waypoints)))
        object.__setattr__(self, "_PublicRuntime__targets", {})

    def _meter(self, name: str, category: str) -> None:
        calls = object.__getattribute__(self, "_PublicRuntime__calls")
        calls["public_calls"] += 1
        calls[f"{category}_calls"] += 1
        calls[name] += 1

    def _backend_call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        value = getattr(object.__getattribute__(self, "_PublicRuntime__backend"), name)
        return value(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name not in PERCEPTION_METHODS | MOTION_METHODS:
            raise AttributeError(f"{name!r} is not part of the public candidate runtime")
        value = getattr(object.__getattribute__(self, "_PublicRuntime__backend"), name)
        if not callable(value):
            return value

        def metered(*args: Any, **kwargs: Any) -> Any:
            self._meter(name, "motion" if name in MOTION_METHODS else "perception")
            return value(*args, **kwargs)

        return metered

    def capability_manifest(self) -> dict[str, Any]:
        """Describe the non-privileged SDK without exposing backend internals."""
        self._meter("capability_manifest", "perception")
        backend = object.__getattribute__(self, "_PublicRuntime__backend")
        primitive_names = sorted(PERCEPTION_METHODS | MOTION_METHODS)
        signatures: dict[str, str] = {}
        for name in primitive_names:
            value = None
            # Explicit PublicRuntime wrappers define the stable contract. For
            # transparent allow-listed methods, report the backend signature.
            if callable(getattr(type(self), name, None)):
                value = getattr(self, name)
            elif callable(getattr(backend, name, None)):
                value = getattr(backend, name)
            if value is not None:
                try:
                    signatures[name] = str(inspect.signature(value))
                except (TypeError, ValueError):
                    signatures[name] = "(*args, **kwargs)"
        return {
            "spec": PUBLIC_RUNTIME_SPEC,
            "available_primitives": [
                name for name in primitive_names if callable(getattr(backend, name, None))
            ],
            "composed_primitives": ["execute_waypoints", "linear_contact", "arc_contact"],
            "signatures": signatures,
            "coordinate_frame": "world_xyz_meters",
            "angles": "radians",
            "native_oracle_available": False,
        }

    @staticmethod
    def _target_key(label: Any) -> str:
        return " ".join(str(label or "").strip().lower().split())

    def localize(self, label: str, **options: Any) -> Any:
        """Ground and cache a public target for subsequent contact primitives."""
        self._meter("localize", "perception")
        target = self._backend_call("localize", label, **options)
        targets = object.__getattribute__(self, "_PublicRuntime__targets")
        targets[self._target_key(label)] = target
        target_label = getattr(target, "label", "")
        if target_label:
            targets[self._target_key(target_label)] = target
        return target

    def _coerce_target(self, target_or_label: Any) -> Any:
        """Resolve the documented GroundedTarget-or-label union.

        This is an embodiment-neutral contract adapter. It does not inspect
        simulator state: a string either reuses the latest visual grounding or
        invokes the same public visual localizer that candidate code can call.
        """
        if not isinstance(target_or_label, str):
            return target_or_label
        targets = object.__getattribute__(self, "_PublicRuntime__targets")
        key = self._target_key(target_or_label)
        if key in targets:
            return targets[key]
        return self.localize(target_or_label)

    def grasp(self, target_or_label: Any, *, strategy: str) -> Any:
        self._meter("grasp", "motion")
        return self._backend_call(
            "grasp", self._coerce_target(target_or_label), strategy=strategy
        )

    def contact_grasp(
        self,
        target_or_label: Any,
        *,
        strategy: str,
        lift: float = 0.0,
    ) -> Any:
        self._meter("contact_grasp", "motion")
        return self._backend_call(
            "contact_grasp",
            self._coerce_target(target_or_label),
            strategy=strategy,
            lift=lift,
        )

    def place(self, target_or_label: Any, *, strategy: str) -> Any:
        self._meter("place", "motion")
        return self._backend_call(
            "place", self._coerce_target(target_or_label), strategy=strategy
        )

    @staticmethod
    def _waypoint(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            if "position" not in value:
                raise ValueError("waypoint mapping requires position")
            waypoint = dict(value)
        else:
            waypoint = {"position": value}
        position = tuple(float(item) for item in waypoint["position"])
        if len(position) != 3 or not all(math.isfinite(item) for item in position):
            raise ValueError("waypoint position must contain three finite world coordinates")
        waypoint["position"] = position
        return waypoint

    def execute_waypoints(self, waypoints: Any) -> list[dict[str, Any]]:
        """Execute candidate-planned Cartesian waypoints and report reached poses.

        This is a geometry-neutral actuator. It contains no drawer, insertion,
        stacking or benchmark logic; evolved Policy APIs own that mechanism.
        """
        sequence = [self._waypoint(value) for value in waypoints]
        maximum = object.__getattribute__(self, "_PublicRuntime__max_waypoints")
        if not sequence:
            raise ValueError("execute_waypoints requires at least one waypoint")
        if len(sequence) > maximum:
            raise ValueError(f"path has {len(sequence)} waypoints; runtime safety bound is {maximum}")
        self._meter("execute_waypoints", "motion")
        reached: list[dict[str, Any]] = []
        for index, waypoint in enumerate(sequence):
            kwargs = {
                key: waypoint[key]
                for key in ("quat", "gripper", "yaw")
                if key in waypoint and waypoint[key] is not None
            }
            self._backend_call("move_to", waypoint["position"], **kwargs)
            pose = self._backend_call("ee_pose")
            reached.append({"index": index, "command": waypoint, "reached": pose})
        return reached

    def linear_contact(
        self,
        start: Any,
        end: Any,
        *,
        steps: int = 6,
        quat: Any = None,
        gripper: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Interpolate a straight Cartesian contact path."""
        count = int(steps)
        if count < 2:
            raise ValueError("linear_contact requires at least two samples")
        a = self._waypoint(start)["position"]
        b = self._waypoint(end)["position"]
        path = []
        for index in range(count):
            alpha = index / (count - 1)
            position = tuple((1.0 - alpha) * a[axis] + alpha * b[axis] for axis in range(3))
            path.append({"position": position, "quat": quat, "gripper": gripper})
        self._meter("linear_contact", "motion")
        return self.execute_waypoints(path)

    def arc_contact(
        self,
        center: Any,
        radius: float,
        start_angle: float,
        sweep: float,
        z: float,
        *,
        steps: int = 10,
        quat: Any = None,
        gripper: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Interpolate a planar circular contact path in world coordinates."""
        count = int(steps)
        if count < 2 or not math.isfinite(float(radius)) or float(radius) <= 0:
            raise ValueError("arc_contact requires positive radius and at least two samples")
        xy = tuple(float(value) for value in center)
        if len(xy) < 2:
            raise ValueError("arc center requires x and y")
        path = []
        for index in range(count):
            angle = float(start_angle) + float(sweep) * index / (count - 1)
            path.append(
                {
                    "position": (
                        xy[0] + float(radius) * math.cos(angle),
                        xy[1] + float(radius) * math.sin(angle),
                        float(z),
                    ),
                    "quat": quat,
                    "gripper": gripper,
                }
            )
        self._meter("arc_contact", "motion")
        return self.execute_waypoints(path)

    def call_stats(self) -> dict[str, int]:
        calls = object.__getattribute__(self, "_PublicRuntime__calls")
        return dict(calls)

    def record_evaluator_checkpoint(self, event: str) -> None:
        """Write native evaluator telemetry without returning native state.

        The visual controller calls this only after a public tool boundary.
        Forwarding the write-only callback lets anytime analysis recover
        completion at fixed wall-clock budgets while preserving the policy
        boundary: no predicate value, reward, pose, or success bit is returned
        to candidate code or included in its subsequent prompts.
        """
        label = str(event or "").strip()
        if not label.startswith("after_tool:"):
            raise ValueError("public evaluator checkpoints require an after_tool event")
        self._backend_call("record_evaluator_checkpoint", label)
