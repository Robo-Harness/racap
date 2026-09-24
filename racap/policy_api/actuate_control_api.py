"""General visual actuation for rotary and press controls.

The skill grounds a compact control on a named appliance, classifies the
mechanism from its crop, executes a bounded contact action, and re-observes the
semantic on/off state.  The same API covers knobs and push buttons; it does not
encode any LIBERO task id, asset name, joint range, or native predicate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from racap.backends.vlm import PerceptionUnavailableError
from racap.backends.llm import LLMError
from racap.contracts import GroundedTarget, SkillResult


DEFAULT_ROTATION_DEG = 145.0
MIN_SEMANTIC_ROTATION_DEG = 30.0
MIN_CONTROL_POINTS = 20


def _usable(target: GroundedTarget | None) -> bool:
    if target is None or target.confidence <= 0.0:
        return False
    pose = np.asarray(target.pose, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        return False
    points = int(target.metadata.get("n_points", 0))
    if points and points < 15:
        return False
    extent = target.metadata.get("extent") or [0.0, 0.0, 0.0]
    return max(float(v) for v in extent) < 0.45


def _ground(runtime, label: str) -> GroundedTarget | None:
    try:
        return runtime.localize(label, detector="vlm_bbox_sam3")
    except TypeError:
        return runtime.localize(label)
    except (PerceptionUnavailableError, LLMError):
        raise
    except Exception:
        return None


def _choice(runtime, crop: GroundedTarget, prompt: str, options: list[str]) -> str:
    try:
        verdict = runtime.inspect(prompt, options=options, target=crop)
    except (PerceptionUnavailableError, LLMError):
        raise
    except Exception:
        return ""
    return str(verdict.get("answer", "")).strip().lower() if verdict.get("ok") else ""


def _control_candidates(target: str, hint: str) -> list[str]:
    candidates = []
    if hint:
        candidates.extend([f"{hint} of {target}", f"{target} {hint}"])
    candidates.extend(
        [
            f"control knob of {target}",
            f"{target} control knob",
            f"power button of {target}",
            f"on off control of {target}",
        ]
    )
    return list(dict.fromkeys(candidates))


def _ground_control(
    runtime, fixture: GroundedTarget, target: str, hint: str, trace: list
) -> GroundedTarget | None:
    best = None
    best_score = -float("inf")
    for label in _control_candidates(target, hint):
        control = _ground(runtime, label)
        if not _usable(control):
            trace.append({"event": "ground_control", "label": label, "usable": False})
            continue
        distance = float(
            np.linalg.norm(np.asarray(control.pose[:2]) - np.asarray(fixture.pose[:2]))
        )
        extent = control.metadata.get("extent") or [0.0, 0.0, 0.0]
        compact = max(float(v) for v in extent)
        trace.append(
            {
                "event": "ground_control",
                "label": label,
                "usable": True,
                "distance_cm": round(distance * 100, 1),
                "extent_cm": round(compact * 100, 1),
            }
        )
        score = -distance - compact
        if distance <= 0.40 and compact <= 0.18 and score > best_score:
            best, best_score = control, score
            # The planner's explicit hint is language-grounded.  A compact
            # first match close to the appliance is already a stronger
            # referent than additional generic captions, and avoiding those
            # calls preserves simulator horizon for the contact action.
            if hint and label == f"{hint} of {target}" and distance <= 0.25:
                return control
    return best


def raised_control_contact(
    points: np.ndarray,
    expected_xy: np.ndarray,
) -> dict[str, Any] | None:
    """Separate a raised knob/lever from an appliance surface in RGB-D.

    SAM frequently merges a small dark control with the stove plate under it.
    Within the VLM's semantic control box, the actuated component is the
    compact upper depth band near the semantic centre.  Selecting that band
    prevents a closed gripper from pinching the static appliance body and
    falsely reporting retention.
    """
    cloud = np.asarray(points, dtype=float).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    expected = np.asarray(expected_xy, dtype=float).reshape(2)
    if cloud.shape[0] < MIN_CONTROL_POINTS:
        return None
    nearby = cloud[np.linalg.norm(cloud[:, :2] - expected, axis=1) <= 0.07]
    if nearby.shape[0] >= MIN_CONTROL_POINTS:
        cloud = nearby
    z_low, z_high = np.percentile(cloud[:, 2], [5, 98])
    height = float(z_high - z_low)
    if height < 0.012 or height > 0.18:
        return None
    threshold = float(
        max(
            z_high - np.clip(0.40 * height, 0.012, 0.030),
            np.percentile(cloud[:, 2], 85),
        )
    )
    raised = cloud[cloud[:, 2] >= threshold]
    if raised.shape[0] < MIN_CONTROL_POINTS:
        return None
    centre = np.median(raised[:, :2], axis=0)
    flat = raised[:, :2] - centre
    try:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        principal = vt[0]
    except np.linalg.LinAlgError:
        principal = np.asarray([1.0, 0.0])
    extent = np.percentile(raised, 95, axis=0) - np.percentile(raised, 5, axis=0)
    if max(float(extent[0]), float(extent[1])) > 0.075:
        return None
    return {
        "pose": np.asarray([centre[0], centre[1], np.median(raised[:, 2])]),
        "top_z": float(np.percentile(raised[:, 2], 95)),
        "principal_axis": np.asarray(principal, dtype=float),
        "extent": np.asarray(extent, dtype=float),
        "n_points": int(raised.shape[0]),
    }


def _refine_control_contact(
    runtime,
    control: GroundedTarget,
) -> tuple[GroundedTarget, dict[str, Any]]:
    bbox = control.metadata.get("bbox")
    if not bbox:
        return control, {"source": "semantic_mask"}
    try:
        probed = runtime.probe_bbox(list(bbox))
        fit = raised_control_contact(
            np.asarray(probed["points"], dtype=float),
            np.asarray(control.pose[:2], dtype=float),
        )
    except Exception as exc:
        return control, {
            "source": "semantic_mask",
            "refinement_error": f"{type(exc).__name__}:{exc}",
        }
    if fit is None:
        return control, {"source": "semantic_mask"}
    metadata = dict(control.metadata)
    metadata.update(
        {
            "top_z": float(fit["top_z"]),
            "principal_axis": [float(v) for v in fit["principal_axis"]],
            "extent": [float(v) for v in fit["extent"]],
            "n_points": int(fit["n_points"]),
        }
    )
    pose = tuple(float(v) for v in fit["pose"])
    refined = GroundedTarget(
        label=control.label,
        pose=pose,
        kind=control.kind,
        confidence=control.confidence,
        metadata=metadata,
    )
    return refined, {
        "source": "rgbd_raised_band",
        "raw_pose": [round(float(v), 4) for v in control.pose],
        "refined_pose": [round(float(v), 4) for v in pose],
        "top_z": round(float(fit["top_z"]), 4),
        "extent_cm": [round(float(v) * 100, 1) for v in fit["extent"]],
        "points": int(fit["n_points"]),
    }


def _topdown_yaw(quat: np.ndarray) -> float:
    """Recover yaw from the runtime's top-down ``[0, cos h, sin h, 0]`` quat."""
    value = np.asarray(quat, dtype=float).reshape(4)
    return float(2.0 * np.arctan2(value[2], value[1]))


def _axis_angle(target: GroundedTarget | None) -> float | None:
    if target is None:
        return None
    axis = target.metadata.get("principal_axis")
    if not axis or len(axis) < 2:
        return None
    return float(np.arctan2(float(axis[1]), float(axis[0])))


def _half_turn_delta(after: float, before: float) -> float:
    return float((after - before + 0.5 * np.pi) % np.pi - 0.5 * np.pi)


def _lowered_control(control: GroundedTarget, fraction: float) -> GroundedTarget:
    """Move a rotary grasp from the visible cap into its graspable body."""
    metadata = dict(control.metadata)
    extent = metadata.get("extent") or [0.0, 0.0, 0.0]
    height = float(np.clip(float(extent[2]), 0.02, 0.10))
    top = float(metadata.get("top_z", control.pose[2] + 0.5 * height))
    depth = float(np.clip(float(fraction) * height, 0.012, 0.045))
    metadata["top_z"] = top - depth
    return GroundedTarget(
        label=control.label,
        pose=control.pose,
        kind=control.kind,
        confidence=control.confidence,
        metadata=metadata,
    )


def actuate_control(
    runtime,
    target: str,
    *,
    goal: str,
    control_hint: str = "",
    mechanism: str = "auto",
    amount: float | None = None,
    contact_strategy: str = "auto",
) -> SkillResult:
    """Set an appliance control to ``on`` or ``off`` with visual feedback."""
    desired = "on" if str(goal).lower() in {"on", "turnon", "turn_on"} else "off"
    requested_mechanism = str(mechanism).lower()
    requested_contact = str(contact_strategy).lower()
    if requested_contact not in {"auto", "pca_axis", "top_down"}:
        requested_contact = "auto"
    trace: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "params": {
            "target": target,
            "goal": desired,
            "control_hint": control_hint,
            "mechanism": requested_mechanism,
            "amount": amount,
            "contact_strategy": requested_contact,
        },
        "preflight_only": True,
    }

    def finish(success: bool, mode: str = "", message: str = "") -> SkillResult:
        return SkillResult(
            success=success,
            status="success" if success else "retryable_failure",
            pick_label=target,
            place_label=desired,
            attempts=1,
            strategy="visual_control_actuation",
            failure_mode=mode,
            message=message,
            trace=trace,
            report=report,
        )

    fixture = _ground(runtime, target)
    if not _usable(fixture):
        report.update({"grounding_failure": "target", "failed_label": target})
        return finish(False, "not_grounded", f"no usable grounding for {target!r}")
    control = _ground_control(runtime, fixture, target, control_hint, trace)
    if control is None:
        report.update({"grounding_failure": "control", "failed_label": control_hint or target})
        return finish(False, "not_grounded", "no compact control grounded on the appliance")
    control, contact_fit = _refine_control_contact(runtime, control)

    other = "off" if desired == "on" else "on"
    state_options = [f"the appliance is {desired}", f"the appliance is {other}"]
    before_state = _choice(runtime, fixture, "current appliance state", state_options)
    report.update(
        {
            "control_label": control.label,
            "control_pose": [round(float(v), 4) for v in control.pose],
            "control_extent": control.metadata.get("extent"),
            "control_top_z": control.metadata.get("top_z"),
            "control_bbox": control.metadata.get("bbox"),
            "control_contact": contact_fit,
            "visual_state_before": before_state,
        }
    )
    report["visual_preflight_match"] = before_state == state_options[0]

    resolved = requested_mechanism
    if resolved not in {"rotary", "press"}:
        mechanism_options = ["a rotary knob", "a push button"]
        answer = _choice(runtime, control, "type of control", mechanism_options)
        resolved = "press" if answer == mechanism_options[1] else "rotary"
    report["mechanism"] = resolved
    # A push button is commonly a toggle, so repeating it after a positive
    # preflight can undo the requested state.  Rotary set-state actions have a
    # directional endpoint and remain safe to execute conservatively.
    if report["visual_preflight_match"] and resolved == "press":
        return finish(True, message="visual state already matches; toggle press skipped")
    before_angle = _axis_angle(control)

    try:
        report["preflight_only"] = False
        if resolved == "press":
            top = float(control.metadata.get("top_z", control.pose[2]))
            position = np.asarray(control.pose, dtype=float)
            press_depth = (
                0.012 if amount is None else float(np.clip(abs(float(amount)), 0.004, 0.025))
            )
            runtime.open_gripper()
            runtime.move_to([position[0], position[1], top + 0.06], yaw=0.0)
            runtime.close_gripper()
            runtime.move_to([position[0], position[1], top - press_depth], yaw=0.0)
            runtime.move_to([position[0], position[1], top + 0.07], yaw=0.0)
            runtime.open_gripper()
            report["commanded_press_depth_cm"] = round(press_depth * 100, 1)
        else:
            degrees = (
                DEFAULT_ROTATION_DEG
                if amount is None
                else float(np.clip(abs(float(amount)), 25.0, 150.0))
            )
            signed = np.radians(degrees) * (1.0 if desired == "on" else -1.0)
            held = False
            thin_contact_proxy = False
            contact_z = None
            contact_strategy = ""
            contact_ladder = (
                [(requested_contact, 0.35 if requested_contact == "pca_axis" else 0.60)]
                if requested_contact != "auto"
                else [("pca_axis", 0.35), ("top_down", 0.60)]
            )
            for strategy, fraction in contact_ladder:
                candidate = _lowered_control(control, fraction)
                runtime.open_gripper()
                runtime.contact_grasp(candidate, strategy=strategy, lift=0.0)
                extent = candidate.metadata.get("extent") or [1.0, 1.0, 1.0]
                thin_contact_proxy = bool(
                    contact_fit.get("source") == "rgbd_raised_band"
                    and min(float(v) for v in extent[:2]) <= 0.022
                )
                if runtime.verify_grasp("") or thin_contact_proxy:
                    held = bool(runtime.verify_grasp(""))
                    contact_strategy = strategy
                    contact_z = float(candidate.metadata["top_z"])
                    break
            report.update(
                {
                    "control_retained": held,
                    "thin_contact_proxy": thin_contact_proxy,
                    "contact_strategy": contact_strategy,
                    "contact_z": None if contact_z is None else round(contact_z, 4),
                }
            )
            if not held and not thin_contact_proxy:
                runtime.open_gripper()
                return finish(
                    False,
                    "contact_failed",
                    "the rotary control was not retained at either body depth",
                )
            ee = runtime.ee_pose()
            position = np.asarray(ee["position"], dtype=float)
            start_yaw = _topdown_yaw(np.asarray(ee["quat"], dtype=float))
            for yaw in np.linspace(start_yaw, start_yaw + signed, 6)[1:]:
                runtime.move_to(position, yaw=float(yaw))
            runtime.open_gripper()
            runtime.move_to(position + np.asarray([0.0, 0.0, 0.08]))
            report["commanded_rotation_deg"] = round(float(np.degrees(signed)), 1)
        report["preflight_only"] = False
    except Exception as exc:
        try:
            runtime.recover()
        except Exception:
            pass
        return finish(
            False, "actuation_failed", f"control contact raised {type(exc).__name__}: {exc}"
        )

    after_control = _ground(runtime, control.label)
    after_angle = _axis_angle(after_control)
    measured_rotation = None
    if before_angle is not None and after_angle is not None:
        measured_rotation = float(np.degrees(_half_turn_delta(after_angle, before_angle)))
    report["measured_control_rotation_deg"] = (
        None if measured_rotation is None else round(measured_rotation, 1)
    )
    refreshed = _ground(runtime, target) or fixture
    after_state = _choice(runtime, refreshed, "current appliance state", state_options)
    report["visual_state_after"] = after_state
    visual_ok = after_state == state_options[0]
    # A PCA axis is pi-periodic and has no arrow, so its signed delta cannot
    # distinguish +56 from -124 degrees.  It can honestly establish rotation
    # magnitude; the executed wrist trajectory establishes direction.  Never
    # unwrap the measurement toward the command, since that assumes success.
    # A press is admissible only when pre-action semantics said the opposite
    # state, since a button is commonly a toggle.
    motion_ok = (resolved == "press" and before_state == state_options[1]) or (
        resolved == "rotary"
        and measured_rotation is not None
        and abs(measured_rotation) >= MIN_SEMANTIC_ROTATION_DEG
    )
    # PCA supplies an unoriented axis, so rotation magnitude alone cannot say
    # that an on/off endpoint was reached.  Require state semantics and
    # physical motion to agree; Session.check_state obtains a second fresh
    # semantic observation before the agent may stop.
    consensus_ok = bool(visual_ok and motion_ok)
    report["visual_goal_match"] = bool(visual_ok)
    report["control_motion_observed"] = bool(motion_ok)
    report["motion_goal_match"] = consensus_ok
    if consensus_ok:
        return finish(
            True,
            message="fresh visual state and measured control motion agree",
        )
    return finish(False, "no_motion", "no measurable control motion or target-state evidence")
