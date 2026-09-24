"""The surface a reasoning agent drives, and the feedback it gets back.

Two halves. ``TOOLS`` describes what can be called, in the form the model is
asked to answer in. ``Session`` executes those calls against a runtime and
turns each one into something the model can look at next turn: a sentence, a
few measured numbers, and a photograph.

The photograph is the point. A skill reports ``verification_inconclusive``
whether the bowl missed the plate by two centimetres, landed upside down, or
was never picked up at all, and those want different next attempts. The
picture separates them and the failure code does not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from racap.backends.vlm import PerceptionUnavailableError, encode_png
from racap.backends.llm import LLMError, LLMQuotaError
from racap.contracts import GroundedTarget
from racap.policy_api.pickplace_api import _appearance_label

TOOLS: dict[str, dict[str, Any]] = {
    "pickplace": {
        "what": "Move one object onto or into another. This is the whole "
        "manipulation surface; everything else only looks.",
        "args": {
            "pick": "label of the object to move, e.g. 'akita black bowl'",
            "place": "where it goes, worded as the instruction words it, e.g. "
            "'the bottom drawer of the cabinet'",
            "grasp": "optional strategy to try first: top_down, rim, pca_axis, "
            "affordance, graspnet (learned 6-DoF for irregular shapes), "
            "or a list of them, each optionally suffixed "
            "@top or @low for how high on the object to close. rim for "
            "open bowls, pca_axis across the narrow side of something "
            "long, @low for a mug too wide at the lip. the automatic "
            "ladder still follows as fallback.",
            "grasp_depth": "optional metres below the object's highest point to "
            "close the fingers. default 0.025. raise for a "
            "bottle with a narrow cap, lower for something flat.",
            "destination": "optional: auto, object, region, inside, top. "
            "'region' for a part such as a drawer or "
            "compartment, 'inside' to aim at the opening of a "
            "container whose outline was detected badly, 'top' "
            "for the upper surface of furniture.",
            "nudge": "optional [right, up] in metres, as seen in the picture, "
            "shifting where the object is released. use when the last "
            "attempt landed visibly to one side.",
            "place_margin": "optional metres of clearance under the object at "
            "release. default 0.03. smaller for a slot it has "
            "to stand in, larger if the hand is catching.",
            "release_on": "optional: rim (default) or floor. 'floor' releases "
            "lower, aiming at the inside bottom of a drawer "
            "rather than clearing its front edge.",
            "yaw_deg": "optional wrist angle in degrees at release, or null to "
            "keep the grasp angle. omit for automatic.",
            "centre": "optional. look at the carried object and correct the aim "
            "before opening the fingers. helps when the grasp was "
            "off-centre, hurts when the object is hard to see.",
            "compensate": "optional. measure how the held object hangs beside "
            "the hand and subtract that offset. useful for a "
            "precision relation or narrow-compartment placement.",
            "identity_check": "optional boolean. verify the target crop before "
            "grasping; use after evidence that a neighbour "
            "was grasped, not on every first attempt.",
        },
    },
    "insert": {
        "what": "Insert an object into a narrow opening using its full RGB-D "
        "footprint, candidate yaw and an eroded feasible region. It "
        "preflights before moving; no_feasible_insertion means a "
        "top-down presentation cannot fit and the arm did not move.",
        "args": {
            "pick": "label of the object to insert",
            "place": "the named compartment, drawer, slot or cubby",
            "grasp": "optional geometry-preserving preference: top_down or "
            "pca_axis, optionally suffixed @top/@low",
            "grasp_depth": "optional metres below the visible top to close",
            "uncertainty_m": "optional opening/footprint erosion margin in metres, default 0.004",
            "insertion_depth": "optional desired depth below the opening rim "
            "before release, default 0.05 m",
        },
    },
    "push": {
        "what": "Nudge an already released, visible object by at most 5 cm "
        "without re-grasping it. Use only after check measured a "
        "small lateral miss; another check is mandatory afterwards.",
        "args": {
            "pick": "label of the object to nudge",
            "place": "the intended destination phrase",
            "nudge": "[right, up] displacement in metres in the current "
            "picture, normally copied from check.suggested_nudge",
        },
    },
    "stack": {
        "what": "Place one object stably on another. It erodes the measured "
        "support footprint by the carried footprint; equal hollow "
        "vessels use concentric nesting instead.",
        "args": {
            "pick": "the object to put on top",
            "support": "the object that must remain underneath",
            "uncertainty_m": "optional RGB-D/contact margin, default 0.002 m",
            "grasp": "optional pickplace grasp strategy or ladder",
        },
    },
    "articulate": {
        "what": "Open or close a drawer or hinged door through its visually "
        "grounded handle or external moving panel and a non-lifting "
        "line/arc contact path.",
        "args": {
            "target": "fixture, e.g. 'cabinet' or 'microwave'",
            "goal": "open or closed",
            "part": "optional moving part, e.g. 'top drawer'",
            "mechanism": "optional auto, prismatic, or revolute",
            "amount": "optional metres for prismatic or degrees for revolute",
            "contact_strategy": (
                "optional side_bar, pca_axis, top_down, hook, graspnet_6d, or collision-ranked graspgen_6d side contact"
            ),
        },
    },
    "actuate_control": {
        "what": "Set an appliance on/off using a visually grounded rotary knob or push button.",
        "args": {
            "target": "the appliance or fixture",
            "goal": "on or off",
            "control_hint": "optional control phrase such as 'front knob'",
            "mechanism": "optional auto, rotary, or press",
            "amount": "optional rotation degrees or press depth metres",
            "contact_strategy": "optional auto, pca_axis, or top_down",
        },
    },
    "look": {
        "what": "Take a fresh picture and list what the detector can name. "
        "Costs nothing but a turn.",
        "args": {
            "verify_destination": "optional original destination phrase; "
            "closed-set check each inventory crop and "
            "report only visually verified aliases",
            "verify_source": "optional original source phrase; independently "
            "verify category and spatial relation for each "
            "inventory candidate",
            "exclude": "optional object label that cannot be the destination",
        },
    },
    "locate": {
        "what": "Ask where one label is, and get its position and how many "
        "points were found. Few points means the detection is not to "
        "be trusted.",
        "args": {"label": "the label to find"},
    },
    "check": {
        "what": "Measure where the moved object ended up relative to where it "
        "was supposed to go: how far off centre, whether it is resting "
        "at the destination's height, and how much room there was. Use "
        "this before calling done -- an object on the rim of a plate "
        "looks placed in a photograph and is not.",
        "args": {"pick": "the object that was moved", "place": "where it was supposed to go"},
    },
    "check_state": {
        "what": "Fresh visual semantic check for an articulation or appliance "
        "state. It never reads simulator predicates.",
        "args": {
            "target": "fixture or appliance to inspect",
            "part": "optional requested moving part, e.g. 'bottom drawer'",
            "goal": "open, closed, on, or off",
        },
    },
    "done": {
        "what": "Stop, when the instruction is carried out. Only after check agrees.",
        "args": {"reason": "what shows it"},
    },
}


def describe_tools(names: tuple[str, ...] | list[str] | None = None) -> str:
    """Render only the tools that a controller is actually allowed to call.

    The complete Session deliberately exposes every Policy API, but a
    transport sub-controller must not see mechanism or stacking actions.  The
    old all-tools prompt both wasted context and let the transport model repeat
    an already completed ``articulate`` / ``actuate_control`` prerequisite.
    ``None`` preserves the full description for callers that genuinely need
    it.
    """
    lines = []
    selected = tuple(TOOLS) if names is None else tuple(names)
    for name in selected:
        if name not in TOOLS:
            raise KeyError(f"unknown tool {name!r}")
        spec = TOOLS[name]
        lines.append(f"- {name}: {spec['what']}")
        for arg, text in spec["args"].items():
            lines.append(f"    {arg}: {text}")
    return "\n".join(lines)


@dataclass
class Step:
    action: str
    args: dict[str, Any]
    thought: str = ""
    observation: str = ""
    image: str = ""
    success: bool = False
    report: dict[str, Any] = field(default_factory=dict)

    def for_history(self) -> str:
        args = json.dumps(self.args, ensure_ascii=False)
        return f"{self.action}({args})\n  -> {self.observation}"


_PICKPLACE_ARGS = (
    "grasp",
    "grasp_depth",
    "destination",
    "nudge",
    "place_margin",
    "release_on",
    "yaw_deg",
    "centre",
    "compensate",
    "identity_check",
)
_INSERT_ARGS = (
    "grasp",
    "grasp_depth",
    "uncertainty_m",
    "insertion_depth",
    "frontload",
    "destination_description",
)

_RELATION = re.compile(r"\b(?:left|right|front|back|behind)\s+(?:side\s+)?of\b")
_CONTAINER = re.compile(
    r"\b(?:drawer|compartment|basket|caddy|tray|bin|box|slot|container|cubby|rack|shelf)\b"
)
_BROAD_CONTAINER = re.compile(r"\b(?:drawer|basket|tray|bin|box|container)\b")
_TOP_REGION = re.compile(r"\btop\s+(?:side\s+|surface\s+)?of\s+(?:the\s+)?(?:cabinet|shelf)\b")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+")


def _normalised_label(label: str) -> str:
    return _LEADING_ARTICLE.sub("", " ".join(label.lower().split()))


def _verify_destination_alias(
    runtime, previous: Step | None, canonical: str
) -> tuple[bool, dict[str, Any]]:
    """Check that a visual inventory alias denotes the requested destination.

    Grounding recovery deliberately allows an inventory name such as
    ``wooden stand`` to replace an instruction name such as ``wine rack``.
    Geometry alone cannot prove those phrases are equivalent: ``black box``
    can be near a cabinet shelf without being the shelf. Reuse the exact
    pre-action destination crop and ask a closed-set visual identity question.
    A failed or uncertain check is conservative: it prevents ``done`` but is
    treated as unreliable evidence, so it cannot trigger a destructive retry.
    """
    if previous is None:
        return True, {"used": False, "matches": True}
    alias = str(previous.args.get("place", ""))
    if not alias or _normalised_label(alias) == _normalised_label(canonical):
        return True, {"used": False, "alias": alias, "matches": True}

    report = previous.report or {}
    bbox = report.get("destination_bbox")
    pose = report.get("destination_pose")
    if not bbox or not pose:
        return False, {
            "used": True,
            "alias": alias,
            "matches": False,
            "reason": "no_pre_action_crop",
        }
    target = GroundedTarget(
        label=alias,
        pose=tuple(float(v) for v in pose[:3]),
        kind=str(report.get("destination_kind", "object")),
        confidence=1.0,
        metadata={"bbox": list(bbox)},
    )
    different = "a different destination"
    try:
        verifier = getattr(runtime, "verify_destination_alias", None)
        if callable(verifier):
            verdict = verifier(canonical, alias, target=target)
            positive_answer = "same destination"
        else:
            verdict = runtime.inspect(
                canonical,
                options=[canonical, different],
                target=target,
            )
            positive_answer = canonical
    except (PerceptionUnavailableError, LLMError):
        raise
    except Exception as exc:
        return False, {
            "used": True,
            "alias": alias,
            "matches": False,
            "reason": f"inspect_failed:{type(exc).__name__}",
        }
    matches = bool(
        verdict.get("ok")
        and str(verdict.get("answer", "")).strip().lower() == positive_answer.strip().lower()
    )
    return matches, {
        "used": True,
        "alias": alias,
        "matches": matches,
        "answer": verdict.get("answer", ""),
    }


def _destination_kind(place: str) -> str:
    """Classify the predicate geometry from the instruction phrase.

    This intentionally uses whole words.  The previous substring test found
    ``bin`` in ``cabinet`` and treated "top of the cabinet" as a container;
    it also treated "right of the caddy" as an insertion task merely because
    the reference object happened to be a caddy.
    """
    lowered = place.lower()
    if _RELATION.search(lowered):
        return "relation"
    if _TOP_REGION.search(lowered):
        return "top_region"
    if _CONTAINER.search(lowered):
        return "container"
    return "surface"


def _correction_nudge(right_cm: float, down_cm: float, distance_m: float) -> list[float]:
    """Convert measured image ``(right, down)`` error to ``(right, up)`` correction."""
    if distance_m <= 1e-4:
        return [0.0, 0.0]
    scale = min(0.04, 0.75 * distance_m)
    # Desired correction in measurement coordinates is (-right, -down).
    # The public second coordinate is +up == -down, so its numeric value is
    # the original down error (not its negation).
    return [
        round(-(right_cm / 100.0) * (scale / distance_m), 3),
        round((down_cm / 100.0) * (scale / distance_m), 3),
    ]


def _footprint_in_polygon(points: np.ndarray, polygon: np.ndarray) -> dict[str, Any]:
    """Measure full visible footprint containment in a persistent region."""
    import cv2

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if points.shape[0] < 20 or polygon.shape[0] < 3:
        return {"usable": False}
    contour = polygon.astype("float32").reshape(-1, 1, 2)
    xy = points[:, :2].astype("float32")
    inside = np.asarray(
        [cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0 for p in xy],
        dtype=bool,
    )
    hull = cv2.convexHull(xy).reshape(-1, 2)
    hull_inside = np.asarray(
        [cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0 for p in hull],
        dtype=bool,
    )
    centre = np.median(xy, axis=0)
    signed_clearance = float(
        cv2.pointPolygonTest(contour, (float(centre[0]), float(centre[1])), True)
    )
    point_ratio = float(inside.mean())
    hull_ratio = float(hull_inside.mean()) if hull_inside.size else 0.0
    # Pixel masks can bleed a thin fringe onto a wall; requiring every point
    # would reject real insertions.  A centre inside plus most of the convex
    # footprint inside rejects rim-leaning objects while tolerating that fringe.
    contained = bool(signed_clearance >= -0.003 and point_ratio >= 0.72 and hull_ratio >= 0.60)
    return {
        "usable": True,
        "contained": contained,
        "point_inside_ratio": point_ratio,
        "hull_inside_ratio": hull_ratio,
        "signed_clearance_m": signed_clearance,
        "footprint_points": int(points.shape[0]),
        "hull_vertices": int(hull.shape[0]),
    }


def _visibly_recoverable_miss(report: dict[str, Any]) -> bool:
    """Whether post-release vision proves the object is safe to re-grasp.

    A globally unreliable check is not necessarily ambiguous.  In particular,
    release-coordinate continuity becomes unreliable when an object falls all
    the way back to the source surface.  If independent identity evidence and
    a dense footprint both put the object wholly outside a confident target
    region, preserving the scene cannot help: the task is visibly incomplete
    and a fresh grounding/retry is safer than stopping.

    Broad relation/object targets remain conservative because their desired
    region is not a persistent measured polygon.  For support surfaces we also
    require a height mismatch; this avoids re-grasping an object that may be
    correctly supported near an under-segmented edge.
    """
    footprint = report.get("footprint_containment") or {}
    try:
        point_ratio = float(footprint.get("point_inside_ratio", 1.0))
        hull_ratio = float(footprint.get("hull_inside_ratio", 1.0))
        region_confidence = float(report.get("destination_region_confidence", 0.0) or 0.0)
        moved_points = int(report.get("moved_points") or 0)
    except (TypeError, ValueError):
        return False
    kind = str(report.get("destination_kind", ""))
    wholly_outside = bool(
        footprint.get("usable")
        and not footprint.get("contained")
        and point_ratio <= 0.05
        and hull_ratio <= 0.05
    )
    if not (
        report.get("identity_clear")
        and moved_points >= 500
        and region_confidence >= 0.70
        and wholly_outside
    ):
        return False
    if kind == "container":
        return True
    return bool(kind in {"surface", "top_region"} and not report.get("landed"))


class Session:
    """Runs agent actions against one episode and reports what happened."""

    def __init__(
        self,
        runtime,
        skill,
        push_skill=None,
        insert_skill=None,
        stack_skill=None,
        articulate_skill=None,
        actuate_skill=None,
    ) -> None:
        self.runtime = runtime
        self.skill = skill
        if push_skill is None:
            from racap.policy_api.push_api import push as push_skill
        if insert_skill is None:
            from racap.policy_api.insert_api import insert as insert_skill
        if stack_skill is None:
            from racap.policy_api.stack_api import stack as stack_skill
        if articulate_skill is None:
            from racap.policy_api.articulate_api import articulate as articulate_skill
        if actuate_skill is None:
            from racap.policy_api.actuate_control_api import (
                actuate_control as actuate_skill,
            )
        self.push_skill = push_skill
        self.insert_skill = insert_skill
        self.stack_skill = stack_skill
        self.articulate_skill = articulate_skill
        self.actuate_skill = actuate_skill
        self.steps: list[Step] = []
        # A narrow opening is easiest to ground before manipulation, while it
        # is unobstructed.  Preserve that RGB-D geometry across retries instead
        # of asking a semantic model to establish a new coordinate frame after
        # the moved object or gripper occludes it.
        self._destination_locks: dict[str, GroundedTarget] = {}
        # Spatial qualifiers identify an instance at task time, not a rule to
        # re-rank identical objects after every failed grasp.  Preserve the
        # first clean source grounding across no-release retries so "front
        # butter" cannot silently become the back butter after the arm has
        # disturbed the scene.
        self._pick_locks: dict[str, GroundedTarget] = {}
        # A hinge is fixed in the world while its door rotates.  Re-estimating
        # it from the combined fixture+door cloud after every partial action
        # lets the moving panel drag the PCA axis.  Cache the first RGB-D
        # estimate for bounded retries on the same semantic mechanism.
        self._articulation_kinematics: dict[str, dict[str, Any]] = {}

    def frame(self) -> str:
        rgb, _, _, _ = self.runtime._camera(self.runtime.observe(), wrist=False)
        return encode_png(np.asarray(rgb).astype("uint8"))

    def holding(self) -> str:
        """Whether the fingers are closed on something, in words.

        The agent kept reasoning as though an object were still in the gripper
        several turns after it had been dropped, and planned releases for a
        hand that was empty. The state is one number away and it never has to
        be guessed.
        """
        try:
            held = bool(self.runtime.verify_grasp(""))
        except Exception:
            return ""
        return "the gripper is closed on something" if held else "the gripper is empty"

    def run(self, action: str, args: dict[str, Any], thought: str = "") -> Step:
        step = Step(action=action, args=dict(args), thought=thought)
        capture_enabled = getattr(self.runtime, "video_capture_enabled", None)
        recording = bool(capture_enabled()) if callable(capture_enabled) else False
        frame_count = getattr(self.runtime, "video_frame_count", None)
        video_start = int(frame_count()) if recording and callable(frame_count) else 0
        try:
            handler = getattr(self, f"_do_{action}")
        except AttributeError:
            step.observation = f"no such action {action!r}; available: {', '.join(TOOLS)}"
            self.steps.append(step)
            return step
        try:
            handler(step)
        except PerceptionUnavailableError as exc:
            # Preserve the distinction between a valid visual "not found" and
            # a call that yielded no visual evidence at all.  The agent must
            # not reinterpret provider downtime as scene state.
            step.report = {
                **step.report,
                "failure_mode": "perception_unavailable",
                "infrastructure_failure": exc.report(),
            }
            step.observation = f"visual evidence unavailable: {exc}"
        except LLMError as exc:
            kind = "provider_quota" if isinstance(exc, LLMQuotaError) else "provider_unavailable"
            failure = {
                "kind": kind,
                "operation": action,
                "model": "runtime_model",
                "detail": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
            step.report = {
                **step.report,
                "failure_mode": "perception_unavailable",
                "infrastructure_failure": failure,
            }
            step.observation = f"visual evidence unavailable: {failure['detail']}"
        except Exception as exc:
            step.observation = f"the call raised {type(exc).__name__}: {exc}"
        if action in {
            "pickplace",
            "insert",
            "push",
            "stack",
            "articulate",
            "actuate_control",
        }:
            self._go_home_after_motion(step)
        step.image = self.frame()
        if recording:
            # Low-level movement is sampled inside the backend.  This extra
            # frame makes perception-only actions visible and guarantees that
            # every tool range ends on its exact post-action state.
            self.runtime.capture_video_frame()
            step.report["video_frame_range"] = [video_start, int(self.runtime.video_frame_count())]
        evaluator_checkpoint = getattr(self.runtime, "record_evaluator_checkpoint", None)
        if callable(evaluator_checkpoint):
            # Evaluator-only telemetry: no value from this callback is added
            # to the Step or exposed to the agent's subsequent prompt.
            try:
                evaluator_checkpoint(f"after_tool:{action}")
            except Exception:
                pass
        self.steps.append(step)
        return step

    def _go_home_after_motion(self, step: Step) -> None:
        """Clear the scene camera after a manipulation without moving a hold.

        The hand frequently stops directly over the destination, so the next
        RGB-D frame sees fingers instead of the placed object.  Returning home
        fixes that observation, but doing it while an object is still held can
        undo an otherwise recoverable placement attempt.  Treat the hook as a
        best-effort observation cleanup and report every outcome explicitly.
        """
        if bool((step.report or {}).get("preflight_only")):
            step.report["go_home_hook"] = "skipped_preflight"
            return
        go_home = getattr(self.runtime, "go_home", None)
        if not callable(go_home):
            step.report["go_home_hook"] = "unsupported"
            return
        released = bool(
            (step.report or {}).get("release_xy") is not None
            or (step.report or {}).get("force_empty_after_contact")
        )
        if released:
            # Every transport policy records release_xy only after its first
            # open command completed.  The opening-band grasp heuristic can
            # nevertheless remain true for a few simulation frames, which
            # previously left the hand parked over the destination and hid the
            # object from the next visual check.  Reassert the intended empty
            # state before going home; this is idempotent and cannot release a
            # different in-transit object because the report proves this call
            # already reached its release phase.
            try:
                self.runtime.open_gripper()
                step.report["release_cleanup"] = "reopened"
            except Exception as exc:
                step.report["release_cleanup"] = f"open_failed:{type(exc).__name__}"
        else:
            try:
                holding = bool(self.runtime.verify_grasp(""))
            except Exception:
                holding = False
            if holding:
                step.report["go_home_hook"] = "skipped_holding"
                return
        try:
            go_home()
            step.report["go_home_hook"] = "completed"
        except Exception as exc:
            step.report["go_home_hook"] = f"failed:{type(exc).__name__}"

    def _do_pickplace(self, step: Step) -> None:
        args = step.args
        pick = str(args.get("pick", ""))
        place = str(args.get("place", ""))
        options = {k: args[k] for k in _PICKPLACE_ARGS if k in args and args[k] is not None}
        if "nudge" in options:
            options["nudge"] = tuple(float(v) for v in options["nudge"])[:2]

        lock_key = _normalised_label(place)
        locked = self._destination_locks.get(lock_key)
        if locked is not None and options.get("destination", "auto") in {
            "auto",
            "region",
            "inside",
        }:
            options["_destination_target"] = locked

        pick_lock_key = _normalised_label(pick)
        locked_pick = self._pick_locks.get(pick_lock_key)
        if locked_pick is not None:
            options["_pick_target"] = locked_pick

        result = self.skill(self.runtime, pick, place, **options)
        step.success = bool(result.success)
        step.report = dict(result.report)
        step.report["failure_mode"] = result.failure_mode

        report = result.report
        self._remember_destination(lock_key, place, report)
        if report.get("release_xy") is not None:
            # The source pose is stale once transport reached release.  A
            # subsequent visual check may seed a new continuity lock at the
            # measured release location if correction is required.
            self._pick_locks.pop(pick_lock_key, None)
        else:
            self._remember_pick(pick_lock_key, pick, report)
            # An empty contact can roll or tip a uniquely named object.  Keeping
            # the old 3-D lock then makes the next materially different grasp
            # close in air (irregular cookware is the common geometry, but the
            # rule is identity-agnostic).  A relational selector must retain its
            # original instance lock because re-evaluating front/back after
            # contact could select an untouched duplicate instead.
            transient_selector = _normalised_label(_appearance_label(pick)) != _normalised_label(
                pick
            )
            if (
                result.failure_mode in {"empty_grasp", "wrong_object_grasp"}
                and not transient_selector
            ):
                self._pick_locks.pop(pick_lock_key, None)
                step.report["pick_lock_after_failure"] = "invalidated_for_reground"

        # What the skill believed, in the terms the next attempt can change.
        # Reporting only "it failed" makes every retry a repeat.
        facts = []
        if "pick_pose" in report:
            facts.append(f"grounded {pick!r} at {report['pick_pose']}")
        if "destination_pose" in report:
            facts.append(f"aimed at {report['destination_pose']}")
        if "grasp_strategy" in report:
            facts.append(f"held it with {report['grasp_strategy']}")
        elif "ladder" in report:
            facts.append(f"tried {', '.join(report['ladder'])} and held nothing")
        if "release_xy" in report:
            facts.append(f"opened the fingers at {report['release_xy']}, z={report['release_z']}")
        detail = "; ".join(facts)

        if result.success:
            if report.get("placement_verification") == "deferred_to_external_check":
                step.observation = f"released; placement awaits check. {detail}"
            else:
                step.observation = f"placed. {detail}"
        else:
            step.observation = f"{result.failure_mode}: {result.message} [{detail}]"

    def _remember_pick(self, lock_key: str, pick: str, report: dict[str, Any]) -> None:
        """Persist an undelivered task referent across physical retries."""
        if lock_key in self._pick_locks:
            return
        pose = report.get("pick_pose")
        extent = report.get("pick_extent")
        if (
            not lock_key
            or not isinstance(pose, (list, tuple))
            or len(pose) < 3
            or not isinstance(extent, (list, tuple))
            or len(extent) < 3
        ):
            return
        try:
            xyz = np.asarray(pose[:3], dtype=float)
            shape = np.asarray(extent[:3], dtype=float)
            confidence = float(report.get("pick_confidence") or 0.75)
            n_points = int(report.get("pick_n_points") or 60)
        except (TypeError, ValueError):
            return
        if (
            not np.isfinite(xyz).all()
            or not np.isfinite(shape).all()
            or n_points < 20
            or not 0.005 < float(max(np.abs(shape))) < 1.0
        ):
            return
        metadata = {
            "top_z": report.get("pick_top_z", float(xyz[2])),
            "n_points": n_points,
            "extent": shape.tolist(),
            "principal_axis": report.get("pick_principal_axis"),
            "bbox": report.get("pick_bbox"),
            "points_label": pick,
            "pregrasp_identity_evidence": report.get("pregrasp_identity_evidence"),
        }
        self._pick_locks[lock_key] = GroundedTarget(
            label=pick,
            pose=tuple(float(v) for v in xyz),
            kind="object",
            confidence=float(np.clip(confidence, 0.01, 1.0)),
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    def _remember_destination(self, lock_key: str, place: str, report: dict[str, Any]) -> None:
        """Persist the first clean static destination across a transport retry.

        Openings need their polygon, but shelves, plates and synthetic
        relations need stable coordinates just as much.  A failed later
        grounding must never replace a destination measured before the arm or
        released object occluded it.
        """
        pose = report.get("destination_pose")
        if not lock_key or not isinstance(pose, (list, tuple)) or len(pose) < 3:
            return
        try:
            xyz = np.asarray(pose[:3], dtype=float)
            extent = np.asarray(
                report.get("destination_extent") or [0.0, 0.0, 0.0],
                dtype=float,
            ).reshape(-1)
        except (TypeError, ValueError):
            return
        if (
            not np.isfinite(xyz).all()
            or not np.isfinite(extent).all()
            or float(max(np.abs(extent), default=0.0)) >= 2.0
        ):
            return

        region_confidence = float(report.get("destination_region_confidence") or 0.0)
        region_polygon = report.get("destination_region_polygon_xy")
        region_usable = bool(
            isinstance(region_polygon, (list, tuple))
            and len(region_polygon) >= 3
            and region_confidence >= 0.55
        )
        existing = self._destination_locks.get(lock_key)
        if existing is not None:
            existing_region = bool(existing.metadata.get("region_polygon_xy"))
            # Preserve the earliest unobstructed frame.  The sole admissible
            # upgrade is adding a confident geometric region to a pose-only
            # lock obtained earlier in the same subgoal.
            if existing_region or not region_usable:
                return

        metadata = {
            "top_z": report.get("destination_rim_z"),
            "floor_z": report.get("destination_floor_z"),
            "n_points": int(report.get("destination_region_area_pixels") or 60),
            "extent": extent.tolist(),
            "synthetic": bool(report.get("destination_synthetic")),
            "bbox": report.get("destination_bbox"),
            "region_source": report.get("destination_region_source"),
            "region_polygon_xy": region_polygon,
            "region_boundary_pixels": report.get("destination_region_boundary_pixels"),
            "region_safe_pixel": report.get("destination_region_safe_pixel"),
            "region_area_pixels": report.get("destination_region_area_pixels"),
            "region_clearance_m": report.get("destination_region_clearance_m"),
            "region_confidence": (region_confidence if region_usable else None),
        }
        confidence = float(
            report.get("destination_confidence") or (region_confidence if region_usable else 0.75)
        )
        self._destination_locks[lock_key] = GroundedTarget(
            label=place,
            pose=tuple(float(v) for v in xyz),
            kind=str(report.get("destination_kind", "region")),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    def _do_insert(self, step: Step) -> None:
        pick = str(step.args.get("pick", ""))
        place = str(step.args.get("place", ""))
        options = {
            key: step.args[key]
            for key in _INSERT_ARGS
            if key in step.args and step.args[key] is not None
        }
        lock_key = _normalised_label(place)
        locked = self._destination_locks.get(lock_key)
        if locked is not None:
            options["_destination_target"] = locked
        pick_lock_key = _normalised_label(pick)
        locked_pick = self._pick_locks.get(pick_lock_key)
        if locked_pick is not None:
            options["_pick_target"] = locked_pick
        result = self.insert_skill(self.runtime, pick, place, **options)
        step.success = bool(result.success)
        step.report = dict(result.report)
        step.report["failure_mode"] = result.failure_mode
        self._remember_destination(lock_key, place, result.report)

        report = result.report
        if report.get("release_xy") is not None:
            self._pick_locks.pop(pick_lock_key, None)
        else:
            self._remember_pick(pick_lock_key, pick, report)
            transient_selector = _normalised_label(_appearance_label(pick)) != _normalised_label(
                pick
            )
            if (
                result.failure_mode in {"empty_grasp", "wrong_object_grasp"}
                and not transient_selector
            ):
                self._pick_locks.pop(pick_lock_key, None)
                step.report["pick_lock_after_failure"] = "invalidated_for_reground"
        facts = []
        if report.get("feasible_centre_xy"):
            facts.append(
                f"feasible centre {report['feasible_centre_xy']} with "
                f"{report.get('feasible_clearance_cm', 0)} cm clearance"
            )
        if report.get("planned_rotation_deg") is not None:
            facts.append(f"rotate object {report['planned_rotation_deg']} degrees")
        if report.get("grasp_strategy"):
            facts.append(f"held it with {report['grasp_strategy']}")
        if report.get("release_xy"):
            facts.append(
                f"released at {report['release_xy']}, "
                f"depth={report.get('achieved_insertion_depth_cm', 0)} cm"
            )
        detail = "; ".join(facts)
        if result.success:
            step.observation = f"inserted. {detail}"
        elif report.get("preflight_only"):
            step.observation = (
                f"{result.failure_mode}: {result.message} (preflight only; scene unchanged)"
            )
        else:
            step.observation = f"{result.failure_mode}: {result.message} [{detail}]"

    def _do_push(self, step: Step) -> None:
        pick = str(step.args.get("pick", ""))
        place = str(step.args.get("place", ""))
        nudge = step.args.get("nudge", (0.0, 0.0))
        previous_check = next(
            (old for old in reversed(self.steps) if old.action == "check"),
            None,
        )
        expected_pose = (
            (previous_check.report or {}).get("moved_pose") if previous_check is not None else None
        )
        result = self.push_skill(
            self.runtime,
            pick,
            place,
            nudge=tuple(float(v) for v in nudge)[:2],
            expected_pose=expected_pose,
        )
        step.success = bool(result.success)
        step.report = dict(result.report)
        step.report["failure_mode"] = result.failure_mode
        if result.success:
            step.observation = f"push moved the object. {result.message}"
        else:
            step.observation = f"{result.failure_mode}: {result.message}"

    def _do_stack(self, step: Step) -> None:
        pick = str(step.args.get("pick", ""))
        support = str(step.args.get("support", step.args.get("place", "")))
        try:
            uncertainty = float(step.args.get("uncertainty_m", 0.002))
        except (TypeError, ValueError):
            uncertainty = 0.002
        result = self.stack_skill(
            self.runtime,
            pick,
            support,
            uncertainty_m=float(np.clip(uncertainty, 0.0, 0.02)),
            grasp=step.args.get("grasp"),
        )
        step.success = bool(result.success)
        step.report = dict(result.report)
        step.report["failure_mode"] = result.failure_mode
        if result.success:
            step.observation = f"stacked {pick!r} on {support!r}. {result.message}"
        else:
            step.observation = f"{result.failure_mode}: {result.message}"

    def _do_articulate(self, step: Step) -> None:
        target = str(step.args.get("target", ""))
        part = str(step.args.get("part", ""))
        hinge_key = _normalised_label(f"{target} {part}")
        remembered = self._articulation_kinematics.get(hinge_key, {})
        # Reuse the opening that was observed before the arm transported an
        # object into it.  This is both less occluded and gives a direct local
        # drawer-front normal; re-grounding the whole cabinet here makes an
        # off-centre handle look like a diagonal rail axis.
        opening_lock = None
        candidate_labels = [target]
        if part:
            candidate_labels.extend(
                (
                    f"{part} of {target}",
                    f"the {part} of the {target}",
                    f"{target} {part}",
                )
            )
        for label in candidate_labels:
            candidate = self._destination_locks.get(_normalised_label(label))
            if candidate is not None and candidate.metadata.get("region_polygon_xy"):
                opening_lock = candidate
                break
        result = self.articulate_skill(
            self.runtime,
            target,
            goal=str(step.args.get("goal", "open")),
            part=part,
            mechanism=str(step.args.get("mechanism", "auto")),
            amount=step.args.get("amount"),
            # Opening is tensile: begin with the learned side-contact selector,
            # which explicitly aligns the approach with the visually measured
            # handle normal.  Top-down PCA remains a recovery mode.  Closing
            # is handled inside articulate as an external face push.
            contact_strategy=str(step.args.get("contact_strategy", "auto")),
            hinge_hint=remembered.get("hinge_xy"),
            handle_z_hint=remembered.get("handle_z"),
            hinge_radius_hint=remembered.get("hinge_radius_cm", 0.0) / 100.0
            if remembered.get("hinge_radius_cm") is not None
            else None,
            panel_label_hint=remembered.get("panel_label"),
            axis_hint=remembered.get("axis"),
            opening_polygon_hint=(
                None if opening_lock is None else opening_lock.metadata.get("region_polygon_xy")
            ),
        )
        step.success = bool(result.success)
        step.report = dict(result.report)
        step.report["failure_mode"] = result.failure_mode
        # Preserve the initially selected handle height for every mechanism.
        # Re-grounding a stacked drawer bank otherwise tends to jump to the
        # neighbouring drawer after the first attempt.  Revolute mechanisms
        # additionally retain their hinge geometry.
        if step.report.get("handle_before"):
            kinematics = self._articulation_kinematics.setdefault(hinge_key, {})
            kinematics.setdefault("handle_z", float(step.report["handle_before"][2]))
            if step.report.get("moving_panel_contact") and step.report.get("handle_label"):
                kinematics["panel_label"] = str(step.report["handle_label"])
            if step.report.get("mechanism") == "revolute" and step.report.get("hinge_xy"):
                source_priority = {
                    "fixture_rgbd_footprint": 0,
                    "moving_panel_near_endpoint": 1,
                    "episode_memory": -1,
                }
                new_source = str(step.report.get("hinge_source", ""))
                old_source = str(kinematics.get("hinge_source", ""))
                replace = "hinge_xy" not in kinematics or source_priority.get(
                    new_source, 0
                ) > source_priority.get(old_source, 0)
                if replace:
                    kinematics["hinge_xy"] = [float(v) for v in step.report["hinge_xy"][:2]]
                    kinematics["hinge_radius_cm"] = float(step.report["hinge_radius_cm"])
                    kinematics["hinge_source"] = new_source
            if step.report.get("mechanism") == "prismatic" and step.report.get("motion_axis"):
                kinematics.setdefault("axis", [float(v) for v in step.report["motion_axis"][:2]])
        step.observation = (
            f"articulation reached the requested state. {result.message}"
            if result.success
            else f"{result.failure_mode}: {result.message}"
        )

    def _do_actuate_control(self, step: Step) -> None:
        result = self.actuate_skill(
            self.runtime,
            str(step.args.get("target", "")),
            goal=str(step.args.get("goal", "on")),
            control_hint=str(step.args.get("control_hint", "")),
            mechanism=str(step.args.get("mechanism", "auto")),
            amount=step.args.get("amount"),
            contact_strategy=str(step.args.get("contact_strategy", "auto")),
        )
        step.success = bool(result.success)
        step.report = dict(result.report)
        step.report["failure_mode"] = result.failure_mode
        step.observation = (
            f"control reached the requested state. {result.message}"
            if result.success
            else f"{result.failure_mode}: {result.message}"
        )

    def _do_look(self, step: Step) -> None:
        names = self.runtime.list_objects()
        canonical = str(step.args.get("verify_destination", "")).strip()
        source_canonical = str(step.args.get("verify_source", "")).strip()
        excluded = _normalised_label(str(step.args.get("exclude", "")))
        verified: list[str] = []
        verified_sources: list[str] = []
        evidence: list[dict[str, Any]] = []
        source_evidence: list[dict[str, Any]] = []
        targets: dict[str, GroundedTarget] = {}
        if canonical:
            different = "a different destination"
            for candidate in names:
                # The moved object can cover most of a container after a
                # placement.  It is nevertheless impossible for that object
                # itself to be the destination.  Apply this symbolic identity
                # constraint before asking the visual classifier.
                if excluded and _normalised_label(candidate) == excluded:
                    evidence.append(
                        {
                            "candidate": candidate,
                            "answer": "excluded_moved_object",
                            "matches": False,
                        }
                    )
                    continue
                try:
                    target = self.runtime.localize(candidate, detector="vlm_bbox_sam3")
                    targets[candidate] = target
                    verifier = getattr(self.runtime, "verify_destination_alias", None)
                    if verifier is not None:
                        verdict = verifier(canonical, candidate, target=target)
                        positive_answer = "same destination"
                    else:
                        # Lightweight runtimes used outside LIBERO retain the
                        # original closed-set inspection contract.
                        verdict = self.runtime.inspect(
                            canonical,
                            options=[canonical, different],
                            target=target,
                        )
                        positive_answer = canonical
                    answer = str(verdict.get("answer", ""))
                    matches = bool(
                        verdict.get("ok") and answer.strip().lower() == positive_answer.lower()
                    )
                except (PerceptionUnavailableError, LLMError):
                    raise
                except Exception as exc:
                    answer = f"inspect_failed:{type(exc).__name__}"
                    matches = False
                evidence.append(
                    {
                        "candidate": candidate,
                        "answer": answer,
                        "matches": matches,
                    }
                )
                if matches:
                    verified.append(candidate)
            # Do not force a closest visual option after strict verification
            # rejects every candidate.  That fallback can turn a nearby book or
            # box into a shelf and authorize a destructive placement.  The
            # grounder instead receives explicit negative visual feedback.
        if source_canonical:
            verifier = getattr(self.runtime, "verify_pick_grounding", None)
            for candidate in names:
                try:
                    target = targets.get(candidate)
                    if target is None:
                        target = self.runtime.localize(candidate, detector="vlm_bbox_sam3")
                        targets[candidate] = target
                    if verifier is None:
                        verdict = self.runtime.inspect(
                            source_canonical,
                            options=[source_canonical, "a different source"],
                            target=target,
                        )
                        matches = bool(
                            verdict.get("ok")
                            and str(verdict.get("answer", "")).strip().lower()
                            == source_canonical.lower()
                        )
                    else:
                        verdict = verifier(source_canonical, target=target)
                        matches = bool(verdict.get("ok") and verdict.get("matches") is True)
                    answer = str(verdict.get("answer", ""))
                except (PerceptionUnavailableError, LLMError):
                    raise
                except Exception as exc:
                    answer = f"inspect_failed:{type(exc).__name__}"
                    matches = False
                source_evidence.append(
                    {
                        "candidate": candidate,
                        "answer": answer,
                        "matches": matches,
                    }
                )
                if matches:
                    verified_sources.append(candidate)
        step.report = {
            "inventory": list(names),
            "verified_destination_aliases": verified,
            "alias_evidence": evidence,
            "verified_source_aliases": verified_sources,
            "source_alias_evidence": source_evidence,
        }
        inventory_text = (
            "the detector names: " + ", ".join(names) if names else "the detector named nothing"
        )
        if canonical:
            inventory_text += f". Visually verified aliases for {canonical!r}: " + (
                ", ".join(verified) if verified else "none"
            )
        if source_canonical:
            inventory_text += (
                f". Independently verified source aliases for "
                f"{source_canonical!r}: "
                + (", ".join(verified_sources) if verified_sources else "none")
            )
        step.observation = inventory_text

    def _do_locate(self, step: Step) -> None:
        label = str(step.args.get("label", ""))
        target = self.runtime.localize(label, detector="vlm_bbox_sam3")
        points = int(target.metadata.get("n_points", 0))
        step.observation = (
            f"{label!r} at {[round(float(v), 3) for v in target.pose]}, "
            f"{points} points, top at {round(float(target.metadata.get('top_z', 0)), 3)}"
            + (" -- too few points to trust" if points < 300 else "")
        )

    def _do_check(self, step: Step) -> None:
        """Measure the placement instead of judging it.

        A photograph of a bowl resting on the edge of a plate and one of a bowl
        sitting in the middle of it differ by a few dozen pixels, and the model
        called both of them done -- ending the episode on a placement the
        simulator scored as a miss. Centimetres are not ambiguous in the same
        way, and they say which direction to nudge.
        """
        pick = str(step.args.get("pick", ""))
        place = str(step.args.get("place", ""))
        kind = _destination_kind(place)

        # Keep action history and physical evidence separate.  A later retry
        # can fail during grounding without releasing anything; treating that
        # no-motion report as "the previous manipulation" used to erase the
        # only valid release coordinate and clean destination crop.
        release_step = next(
            (
                old
                for old in reversed(self.steps)
                if old.action in {"pickplace", "insert", "stack"}
                and (old.report or {}).get("release_xy") is not None
            ),
            None,
        )
        destination_step = next(
            (
                old
                for old in reversed(self.steps)
                if old.action in {"pickplace", "insert", "stack"}
                and (old.report or {}).get("destination_pose") is not None
            ),
            None,
        )
        previous = release_step or destination_step
        prior: dict[str, Any] = {}
        locked = self._destination_locks.get(_normalised_label(place))
        if locked is not None:
            prior.update(
                {
                    "destination_pose": list(locked.pose),
                    "destination_confidence": float(locked.confidence),
                    "destination_kind": locked.kind,
                    "destination_rim_z": locked.metadata.get("top_z", locked.pose[2]),
                    "destination_floor_z": locked.metadata.get("floor_z"),
                    "destination_extent": locked.metadata.get("extent"),
                    "destination_synthetic": locked.metadata.get("synthetic"),
                    "destination_bbox": locked.metadata.get("bbox"),
                    "destination_region_source": locked.metadata.get("region_source"),
                    "destination_region_polygon_xy": locked.metadata.get("region_polygon_xy"),
                    "destination_region_confidence": locked.metadata.get("region_confidence"),
                }
            )
        evidence_step = release_step or destination_step
        if evidence_step is not None:
            # A real release is allowed to supply its carry/release fields and
            # destination alias.  It must not be overwritten by a later failed
            # preflight or not-grounded retry.
            prior.update(evidence_step.report or {})
        alias_matches, alias_check = _verify_destination_alias(self.runtime, previous, place)
        goal = None
        if prior.get("destination_pose"):
            goal_pose = np.asarray(prior["destination_pose"], dtype=float)
        else:
            goal = self.runtime.localize(place, detector="vlm_bbox_sam3")
            goal_pose = np.asarray(goal.pose, dtype=float)

        release_xy = prior.get("release_xy")

        def candidate_facts(target: GroundedTarget | None) -> tuple[int, float, bool]:
            if target is None:
                return 0, float("inf"), False
            points = int(target.metadata.get("n_points", 0))
            extent = target.metadata.get("extent") or [0.0, 0.0, 0.0]
            plausible = bool(
                points >= 60
                and 0.005 < max((abs(float(v)) for v in extent), default=0.0) < 1.0
                and np.isfinite(np.asarray(target.pose, dtype=float)).all()
            )
            error = (
                float("inf")
                if release_xy is None
                else float(
                    np.linalg.norm(
                        np.asarray(target.pose[:2], dtype=float)
                        - np.asarray(release_xy, dtype=float)
                    )
                )
            )
            return points, error, plausible

        moved = self.runtime.localize_many([pick]).get(pick)
        if moved is None:
            moved = self.runtime.localize(pick, detector="vlm_bbox_sam3")
        moved_label = pick
        moved_source = "instruction_phrase"
        candidate_rows: list[dict[str, Any]] = []
        points0, error0, plausible0 = candidate_facts(moved)
        candidate_rows.append(
            {
                "source": moved_source,
                "label": pick,
                "points": points0,
                "release_error_cm": (None if not np.isfinite(error0) else round(error0 * 100, 1)),
                "plausible": plausible0,
            }
        )

        # Positional qualifiers such as "middle bowl" describe the object
        # before motion and may bind to a distractor afterwards.  When the
        # original phrase is sparse or discontinuous with the commanded
        # release, ask a second, explicitly post-action visual question and
        # select by physical continuity.  The two labels are grounded in
        # separate calls because localize_many intentionally forbids shared
        # boxes, while these phrases are aliases for the same object.
        needs_rebind = bool(
            release_xy is not None and (not plausible0 or points0 < 1000 or error0 > 0.10)
        )
        if needs_rebind:
            relational_label = f"{pick} now resting at {place}"
            try:
                rebound = self.runtime.localize(relational_label, detector="vlm_bbox_sam3")
            except Exception:
                rebound = None
            points1, error1, plausible1 = candidate_facts(rebound)
            candidate_rows.append(
                {
                    "source": "post_release_relation",
                    "label": relational_label,
                    "points": points1,
                    "release_error_cm": (
                        None if not np.isfinite(error1) else round(error1 * 100, 1)
                    ),
                    "plausible": plausible1,
                }
            )
            continuity_gate = 0.14 if kind == "container" else 0.10
            causal_transport = bool(prior.get("transport_identity_established", False))
            sparse_but_causal = bool(
                causal_transport
                and points1 >= 20
                and np.isfinite(error1)
                and error1 <= min(0.08, continuity_gate)
            )
            if (
                (plausible1 or sparse_but_causal)
                and error1 <= continuity_gate
                and (not plausible0 or error1 + 0.01 < error0 or points0 < 1000)
            ):
                moved = rebound
                moved_label = relational_label
                moved_source = "post_release_relation"

        offset = np.asarray(moved.pose[:2], dtype=float) - goal_pose[:2]
        distance = float(np.linalg.norm(offset))
        goal_metadata = goal.metadata if goal is not None else {}
        extent = prior.get("destination_extent") or goal_metadata.get("extent") or [0.0, 0.0, 0.0]
        radius = 0.5 * float(max(extent[0], extent[1], 0.06))
        top_z = float(prior.get("destination_rim_z", goal_metadata.get("top_z", goal_pose[2])))
        rise = float(moved.pose[2]) - top_z
        container = kind == "container"

        moved_points = int(moved.metadata.get("n_points", 0))
        moved_extent = moved.metadata.get("extent") or [0.0, 0.0, 0.0]
        sparse_causal_rebind = bool(
            moved_source == "post_release_relation"
            and prior.get("transport_identity_established", False)
            and moved_points >= 20
        )
        plausible_moved = (moved_points >= 60 or sparse_causal_rebind) and 0.005 < max(
            (float(v) for v in moved_extent), default=0.0
        ) < 1.0
        release_error = None
        if release_xy is not None:
            release_error = float(
                np.linalg.norm(
                    np.asarray(moved.pose[:2], dtype=float) - np.asarray(release_xy, dtype=float)
                )
            )
        # A newly released object cannot teleport tens of centimetres.  In a
        # container it commonly becomes occluded and the detector locks onto a
        # distractor.  Mark that measurement uncertain so it cannot trigger a
        # destructive re-grasp of a potentially successful placement.
        if kind == "relation" and "mug" in pick.lower():
            release_limit = 0.60
        elif kind == "container" and "compartment" in place.lower():
            # A narrow compartment usually occludes the released object.  A
            # new mask tens of centimetres from the commanded release is a
            # distractor, not evidence that the placed object teleported.
            # Acting on that false match caused repeated blind re-grasps.
            release_limit = 0.12
        elif kind == "container" and not "compartment" in place.lower():
            release_limit = 0.18
        else:
            release_limit = 0.30
        geometry_reliable = bool(
            plausible_moved and (release_error is None or release_error <= release_limit)
        )
        region_polygon = prior.get("destination_region_polygon_xy")
        region_confidence = float(prior.get("destination_region_confidence") or 0.0)
        footprint: dict[str, Any] = {"usable": False}
        if region_polygon and region_confidence >= 0.55:
            try:
                footprint = _footprint_in_polygon(
                    np.asarray(self.runtime.object_points(moved_label), dtype=float),
                    np.asarray(region_polygon, dtype=float),
                )
            except Exception as exc:
                footprint = {"usable": False, "error": f"{type(exc).__name__}: {exc}"}
            # A stored region is stronger than a new centre estimate only when
            # the moved object's footprint can also be measured.
            geometry_reliable = bool(geometry_reliable and footprint.get("usable"))

        guided_insert = bool(
            release_step is not None
            and release_step.action == "insert"
            and not (release_step.report or {}).get("preflight_only", False)
            and float((release_step.report or {}).get("achieved_insertion_depth_cm", 0.0) or 0.0)
            >= 1.0
        )

        identity_visual: dict[str, Any] = {"used": False, "matches": None}
        relation_rebound = moved_source == "post_release_relation"
        pregrasp_identity = prior.get("pregrasp_identity_evidence") or {}
        pregrasp_match = pregrasp_identity.get("matches")
        weak_identity = bool(
            release_xy is not None
            and pregrasp_match is not True
            and (
                relation_rebound
                or moved_points < 1000
                or (guided_insert and footprint.get("usable") and not footprint.get("contained"))
            )
        )
        if weak_identity:
            appearance = _appearance_label(pick)
            options = [
                f"the crop is {appearance}",
                "the crop is a different object",
            ]
            try:
                verdict = self.runtime.inspect(
                    "identity of the tightly cropped released object; judge "
                    "visible appearance only, independent of the commanded "
                    "action",
                    options=options,
                    target=moved,
                )
                answer = str(verdict.get("answer", "")).strip().lower()
                identity_visual = {
                    "used": True,
                    "appearance_label": appearance,
                    "matches": (
                        True
                        if verdict.get("ok") and answer == options[0].lower()
                        else False
                        if (verdict.get("ok") and answer == options[1].lower())
                        else None
                    ),
                    "answer": answer,
                }
            except (PerceptionUnavailableError, LLMError):
                raise
            except Exception as exc:
                identity_visual = {
                    "used": True,
                    "matches": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        right_cm = down_cm = 0.0
        try:
            axes = self.runtime.image_axes()
            right = np.asarray(axes["right"], dtype=float)[:2]
            down = np.asarray(axes["down"], dtype=float)[:2]
            right /= max(float(np.linalg.norm(right)), 1e-6)
            down /= max(float(np.linalg.norm(down)), 1e-6)
            right_cm = float(offset @ right) * 100
            down_cm = float(offset @ down) * 100
            where = (
                f"{abs(right_cm):.0f} cm to the "
                f"{'right' if right_cm > 0 else 'left'} and "
                f"{abs(down_cm):.0f} cm "
                f"{'down' if down_cm > 0 else 'up'} in the picture"
            )
        except Exception:
            where = f"{distance * 100:.0f} cm away"

        # Containers are judged by being inside the opening, not sitting on the
        # rim height. A bowl in a drawer is several centimetres below the drawer's
        # top_z and still a success; the old ±6 cm band called every drawer
        # placement a miss and the agent kept re-grasping winning episodes.
        if kind == "surface":
            # LIBERO's object-on-object predicate requires centres within 3 cm
            # as well as contact.  The old 4 cm floor is exactly why t9 was a
            # false done.
            tolerance = 0.025
            landed = -0.04 < rise < 0.075
        elif kind == "top_region":
            tolerance = min(max(0.45 * radius, 0.04), 0.10)
            landed = -0.04 < rise < 0.075
        elif kind == "relation":
            tolerance = 0.045 if "mug" in pick.lower() else 0.025
            landed = -0.06 < rise < 0.15
        else:
            broad = bool(_BROAD_CONTAINER.search(place.lower()))
            # The native containment predicate tests whether the moved object's
            # centre is inside the site box.  Therefore a radial threshold of
            # only 45% of the perceived half-width rejects valid placements
            # near an opening edge (t74 was native-successful at 4.2 cm in a
            # roughly 4 cm perceived half-width).  Keep a small detector-noise
            # allowance, while retaining a cap so a broad bad box cannot make
            # a distant table placement look contained.
            tolerance = min(max(radius + 0.02, 0.045), 0.08 if broad else 0.065)
            # Successful caddy placements in the calibration set were at most
            # 8 cm below the perceived rim; -15 to -23 cm were detections on
            # the table/floor outside the compartment.
            landed = -0.12 < rise < 0.015 if "compartment" in place.lower() else -0.12 < rise < 0.08
            # An inserted upright object can legitimately extend above the
            # rim.  For that motion family, a contained post-action footprint
            # plus measured insertion depth is stronger evidence than applying
            # a horizontal-surface centroid-height band to the visible upper
            # half of the object.
            if guided_insert and footprint.get("usable"):
                landed = bool(footprint.get("contained"))

        # A stack relation additionally requires the moved object's measured
        # centre to sit above the support plane.  A relation-conditioned mask
        # can otherwise rebound onto the plate itself: its XY looks perfect,
        # but its centroid is at the plate height.  Scale the lower bound by
        # the pre-action object height while retaining a small floor for thin
        # objects and noisy depth.
        required_support_rise = 0.0
        if kind == "surface" and release_step is not None and release_step.action == "stack":
            # Native object-on-object checks are stricter than a generic
            # tabletop placement.  Keep a conservative centre margin instead
            # of adding detector uncertainty to the acceptance boundary: that
            # uncertainty is a reason to retry, not evidence of success.
            tolerance = min(tolerance, 0.020)
            pick_extent = prior.get("pick_extent") or [0.0, 0.0, 0.0]
            try:
                object_height = abs(float(pick_extent[2]))
            except (IndexError, TypeError, ValueError):
                object_height = 0.0
            required_support_rise = max(0.003, 0.20 * object_height)
            landed = bool(landed and rise >= required_support_rise)

        # Container walls frequently make the mask inherit background depth,
        # so its Z centroid can be tens of centimetres too low even while the
        # deprojected footprint is securely inside the saved cavity.  A causal
        # transport, a near-release post-action rebind, and high whole-hull
        # containment are three independent visual/geometric constraints; use
        # their conjunction instead of retrying and disturbing the scene.
        causal_container_footprint = bool(
            container
            and prior.get("transport_identity_established", False)
            and moved_source == "post_release_relation"
            and release_error is not None
            and release_error <= 0.08
            and footprint.get("usable")
            and float(footprint.get("point_inside_ratio", 0.0)) >= 0.90
            and float(footprint.get("hull_inside_ratio", 0.0)) >= 0.80
        )
        if causal_container_footprint:
            landed = True
            geometry_reliable = True

        visual_identity_match = identity_visual.get("matches") is True
        transport_identity = bool(prior.get("transport_identity_established", False))
        pregrasp_veto_enabled = bool(pregrasp_identity.get("veto_enabled", True))
        explicit_wrong_identity = bool(
            (pregrasp_match is False and pregrasp_veto_enabled)
            or (identity_visual.get("matches") is False and not transport_identity)
        )
        identity_continuity = bool(pregrasp_match is True or transport_identity)
        identity_clear = bool(
            plausible_moved
            and not explicit_wrong_identity
            and (
                identity_continuity or visual_identity_match
                if relation_rebound
                else (
                    moved_points >= 1000
                    or identity_continuity
                    or visual_identity_match
                    or kind not in {"surface", "top_region"}
                )
            )
        )
        reliable = bool(geometry_reliable and alias_matches and identity_clear)
        measurement_uncertainty = 0.0
        footprint_margin_ok = (
            bool(
                footprint.get("contained")
                and float(footprint.get("signed_clearance_m", -float("inf"))) >= 0.003
            )
            if footprint.get("usable")
            else False
        )
        # Do not endorse a rounded measurement exactly on the predicate
        # boundary.  A small RGB-D/control margin turns those marginal cases
        # into one safe micro-correction instead of a false done.
        acceptance_margin = 0.003 if kind in {"surface", "relation"} else 0.0
        safe_tolerance = max(0.0, tolerance - acceptance_margin)
        centred = (
            footprint_margin_ok
            if footprint.get("usable")
            else distance <= safe_tolerance + measurement_uncertainty
        )
        step.success = bool(reliable and identity_clear and centred and landed)
        recoverable_miss_evidence = {
            "destination_kind": kind,
            "destination_region_confidence": region_confidence,
            "identity_clear": identity_clear,
            "moved_points": moved_points,
            "landed": landed,
            "footprint_containment": footprint,
        }
        visibly_recoverable_miss = _visibly_recoverable_miss(recoverable_miss_evidence)
        preserve_state = bool(
            release_xy is not None
            and not step.success
            and not explicit_wrong_identity
            and not visibly_recoverable_miss
            and (
                not reliable
                or (centred and landed and not identity_clear)
                or (guided_insert and not identity_clear)
            )
        )

        continuity_lock_updated = False
        if (
            release_step is not None
            and transport_identity
            and plausible_moved
            and not explicit_wrong_identity
            and release_error is not None
            and release_error <= release_limit
        ):
            metadata = dict(moved.metadata)
            metadata["points_label"] = moved_label
            metadata["pregrasp_identity_evidence"] = pregrasp_identity
            self._pick_locks[_normalised_label(pick)] = GroundedTarget(
                label=pick,
                pose=tuple(float(v) for v in moved.pose),
                kind="object",
                confidence=float(np.clip(float(moved.confidence), 0.01, 1.0)),
                metadata=metadata,
            )
            continuity_lock_updated = True

        # Suggested nudge in image axes (right, up), metres. Object too far
        # right → release further left next time.
        nudge = [0.0, 0.0]
        if not centred and distance > 1e-4:
            nudge = _correction_nudge(right_cm, down_cm, distance)

        step.report = {
            "moved_pose": [round(float(v), 4) for v in moved.pose],
            "offset_cm": round(distance * 100, 1),
            "radius_cm": round(radius * 100, 1),
            "rise_cm": round(rise * 100, 1),
            "required_support_rise_cm": round(required_support_rise * 100, 1),
            "centred": centred,
            "landed": landed,
            "container": container,
            "destination_kind": kind,
            "tolerance_cm": round(tolerance * 100, 1),
            "acceptance_margin_cm": round(acceptance_margin * 100, 1),
            "measurement_uncertainty_cm": round(measurement_uncertainty * 100, 1),
            "reliable": reliable,
            "geometry_reliable": geometry_reliable,
            "destination_alias": alias_check,
            "identity_clear": identity_clear,
            "transport_identity_established": transport_identity,
            "identity_evidence": identity_visual,
            "pregrasp_identity_evidence": pregrasp_identity,
            "failure_mode": ("wrong_object_grasp" if explicit_wrong_identity else ""),
            "localization_source": moved_source,
            "localization_candidates": candidate_rows,
            "moved_points": moved_points,
            "release_error_cm": (None if release_error is None else round(release_error * 100, 1)),
            "suggested_nudge": nudge,
            "right_cm": round(right_cm, 1),
            "down_cm": round(down_cm, 1),
            "destination_region_source": prior.get("destination_region_source"),
            "destination_region_confidence": (
                round(region_confidence, 2) if region_polygon else None
            ),
            "footprint_containment": footprint,
            "footprint_margin_ok": (footprint_margin_ok if footprint.get("usable") else None),
            "visibly_recoverable_miss": visibly_recoverable_miss,
            "guided_insert_evidence": guided_insert,
            "causal_container_footprint": causal_container_footprint,
            "pick_continuity_lock_updated": continuity_lock_updated,
            "preserve_state": preserve_state,
        }
        if step.success:
            advice = "This looks placed. Call done."
        elif alias_check.get("used") and not alias_matches:
            advice = (
                f"The visual crop for alias {alias_check.get('alias')!r} did "
                f"not verify the requested destination {place!r}. Do not "
                "declare done or re-grasp from this ambiguous evidence."
            )
        elif preserve_state:
            advice = (
                "Post-release evidence is ambiguous and is not safe to act "
                "on. Preserve the current scene; do not re-grasp a possibly "
                "placed or occluded object."
            )
        else:
            push_safe = kind == "surface"
            correction = (
                f"Use push with nudge={nudge} (image right/up metres), then check again."
                if (push_safe and landed and not centred and 0.004 < distance <= 0.055)
                else (
                    f"Retry pickplace with nudge={nudge} (image right/up metres)."
                    if not centred
                    else "Retry with release_on='floor' or a smaller place_margin."
                )
            )
            advice = (
                "This is not placed properly: "
                + ("the object identity is weakly observed. " if not identity_clear else "")
                + (
                    (
                        "its footprint is not contained by the measured opening. "
                        if footprint.get("usable")
                        else "it is off to one side. "
                    )
                    if not centred
                    else ""
                )
                + ("it is not resting at the destination's height. " if not landed else "")
                + correction
            )
        region_detail = ""
        if footprint.get("usable"):
            region_detail = (
                f" The persistent RGB-D opening contains "
                f"{100 * float(footprint['point_inside_ratio']):.0f}% of the "
                f"visible object footprint and "
                f"{100 * float(footprint['hull_inside_ratio']):.0f}% of its "
                "footprint hull."
            )
        step.observation = (
            f"{pick!r} sits {where} from the middle of {place!r} "
            f"(off by {distance * 100:.1f} cm, the destination is about "
            f"{radius * 100:.0f} cm across at the half-width; this task "
            f"allows {tolerance * 100:.1f} cm), and "
            f"{abs(rise) * 100:.1f} cm "
            f"{'above' if rise > 0 else 'below'} its top surface. " + region_detail + " " + advice
        )

    def _do_done(self, step: Step) -> None:
        step.success = True
        step.observation = f"stopped: {step.args.get('reason', '')}"

    def _do_check_state(self, step: Step) -> None:
        target_label = str(step.args.get("target", ""))
        part_label = str(step.args.get("part", ""))
        requested = str(step.args.get("goal", "")).strip().lower()
        requested = "closed" if requested in {"close", "closed"} else requested
        if requested not in {"open", "closed", "on", "off"}:
            step.observation = "invalid_state: goal must be open, closed, on, or off"
            return
        # This tool now produces motion history only.  The full agent receives
        # the fresh post-home image and owns the one semantic open/closed/on/off
        # decision.  Localizing and classifying the same fixture here first
        # spent two VLM calls and then gave an advisory helper a competing
        # state verdict.
        answer = ""
        visual_match = False
        matching_actions = [
            old
            for old in self.steps
            if old.action in {"articulate", "actuate_control"}
            and _normalised_label(str(old.args.get("target", "")))
            == _normalised_label(target_label)
            and (
                not _normalised_label(part_label)
                or _normalised_label(str(old.args.get("part", ""))) == _normalised_label(part_label)
            )
            and str(old.args.get("goal", "")).strip().lower()
            in {requested, "close" if requested == "closed" else requested}
        ]
        previous = matching_actions[-1] if matching_actions else None
        motion_match = bool(
            previous is not None and (previous.report or {}).get("motion_goal_match")
        )
        previous_report = previous.report if previous is not None else {}
        already_satisfied = bool(previous_report.get("already_satisfied"))
        aggregate_evidence: dict[str, Any] = {}
        articulation_reports = [
            old.report or {}
            for old in matching_actions
            if old.action == "articulate"
            and (old.report or {}).get("mechanism") in {"prismatic", "revolute"}
        ]
        if articulation_reports:
            mechanism = str(articulation_reports[-1].get("mechanism"))
            same_mechanism = [
                report for report in articulation_reports if report.get("mechanism") == mechanism
            ]
            if mechanism == "prismatic":
                direction = 1.0 if requested == "open" else -1.0
                signed_progress = [
                    direction * float(report["directed_progress_cm"])
                    for report in same_mechanism
                    if report.get("directed_progress_cm") is not None
                    and report.get("motion_measurement_reliable", False)
                ]
                cumulative_cm = max(0.0, float(sum(signed_progress)))
                requested_strokes = [
                    float(report["planned_linear_stroke_cm"])
                    for report in same_mechanism
                    if report.get("planned_linear_stroke_cm") is not None
                ]
                expected_cm = 0.80 * max(requested_strokes or [20.0])
                latest_cm = signed_progress[-1] if signed_progress else 0.0
                contacted = bool(
                    previous_report.get("contact_mode")
                    and not previous_report.get("preflight_only", False)
                    and previous_report.get("motion_measurement_reliable", False)
                )
                # A short mechanism may hit its physical stop before the
                # generic requested stroke.  Accept that endpoint only after
                # prior directed travel, a contacted near-zero final attempt,
                # and an independent fresh visual state match.
                endpoint_stall = bool(
                    contacted and cumulative_cm >= 4.0 and -0.3 <= latest_cm <= 0.8
                )
                # A single over-travel close command need not stall in the
                # visually tracked handle centroid: once the front reaches its
                # stop, the last waypoints move only the compliant controller.
                # The articulation policy records the stronger two-modal
                # evidence (open->closed appearance and >=60% reliable inward
                # travel) at the time of contact.  Preserve it across a fresh
                # check instead of forcing another destructive retry.
                compressive_endpoint = any(
                    bool(report.get("compressive_endpoint_consensus")) for report in same_mechanism
                )
                # Accumulated re-grounded centroids are useful diagnostics but
                # retain centimetre-scale bias.  Endpoint contact is the
                # state-defining evidence for a prismatic mechanism.
                motion_match = bool(motion_match or endpoint_stall or compressive_endpoint)
                aggregate_evidence = {
                    "mechanism": mechanism,
                    "cumulative_directed_progress_cm": round(cumulative_cm, 1),
                    "expected_progress_cm": round(expected_cm, 1),
                    "endpoint_stall": endpoint_stall,
                    "compressive_endpoint_consensus": compressive_endpoint,
                    "action_count": len(same_mechanism),
                    "reliable_action_count": sum(
                        bool(report.get("motion_measurement_reliable", False))
                        for report in same_mechanism
                    ),
                }
            else:
                directional_angles = []
                planned_angles = []
                for report in same_mechanism:
                    measured = report.get("measured_hinge_rotation_deg")
                    planned = report.get("planned_hinge_rotation_deg")
                    if (
                        measured is None
                        or planned is None
                        or float(planned) == 0.0
                        or not report.get("motion_measurement_reliable", False)
                    ):
                        continue
                    directional_angles.append(
                        float(measured) * (1.0 if float(planned) > 0 else -1.0)
                    )
                    planned_angles.append(abs(float(planned)))
                cumulative_deg = max(0.0, float(sum(directional_angles)))
                expected_deg = max(65.0, 0.80 * max(planned_angles or [82.0]))
                motion_match = bool(motion_match or cumulative_deg >= expected_deg)
                aggregate_evidence = {
                    "mechanism": mechanism,
                    "cumulative_directed_rotation_deg": round(cumulative_deg, 1),
                    "expected_rotation_deg": round(expected_deg, 1),
                    "action_count": len(same_mechanism),
                }
            latest_articulate = next(
                (old for old in reversed(matching_actions) if old.action == "articulate"),
                None,
            )
            latest_articulate_report = (
                latest_articulate.report or {} if latest_articulate is not None else {}
            )
            closed_handle_endpoint = False
            if closed_handle_endpoint:
                motion_match = True
                aggregate_evidence["closed_handle_endpoint"] = True
        control_reports = [
            old.report or {}
            for old in matching_actions
            if old.action == "actuate_control"
            and not (old.report or {}).get("preflight_only", False)
        ]
        visual_control_consensus = False
        if visual_control_consensus:
            motion_match = True
            aggregate_evidence = {
                "mechanism": "control",
                "visual_motion_consensus": True,
                "action_count": len(control_reports),
            }
        motion_measured = bool(
            previous_report.get("measured_displacement_cm") is not None
            or previous_report.get("measured_control_rotation_deg") is not None
            or (
                previous is not None
                and previous.action == "articulate"
                and not previous_report.get("preflight_only", False)
            )
        )
        motion_contradicts = bool(
            (
                motion_measured
                and "motion_goal_match" in previous_report
                and not previous_report.get("motion_goal_match")
            )
            # A later preflight/not-grounded action must not erase reliable or
            # unreliable insufficient-motion evidence from earlier physical
            # articulation attempts.  Once articulation has been attempted,
            # a binary state label is supporting evidence only.
            or (articulation_reports and not motion_match and not already_satisfied)
        )
        # A binary VLM state answer must not overrule a fresh metric showing
        # zero/wrong-direction motion.  It remains useful when motion was not
        # measurable (for example a push button or an already-satisfied goal).
        step.success = bool(motion_match or already_satisfied)
        step.report = {
            "target": target_label,
            "part": part_label,
            "goal": requested,
            "answer": answer,
            "visual_goal_match": visual_match,
            "motion_goal_match": motion_match,
            "motion_contradicts_visual": motion_contradicts,
            "aggregate_motion": aggregate_evidence,
            "semantic_state_verification": "deferred_to_visual_agent",
            "evidence": ("measured_motion" if motion_match else "none"),
        }
        step.observation = (
            f"motion history supports {requested!r}; semantic decision deferred."
            if step.success
            else f"no decisive motion-state evidence for {requested!r}; "
            "semantic decision deferred to the visual agent."
        )
