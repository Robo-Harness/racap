"""Bind the Policy API contracts to a real LIBERO simulator through RATs.

Two rules shape this module.

First, the policy side may only ever see what a real robot could see. LIBERO's
underlying robosuite observation carries ``<object>_pos`` / ``<object>_quat``
ground truth for every object in the scene; letting that reach the policy or
the runtime agent would invalidate every generalization claim. RATs' own
``get_observation`` already returns a non-privileged view, and the raw handle is
reachable only through :class:`LiberoOracle`, which the evaluation harness uses
and the policy never receives.

Second, the geometric constants here are not invented. They come from RATs
skills that survived 50 play iterations: ``place_carried_object_at_target_center``
(verified tier, 10/12 successes) contributes the hover/release/retreat heights,
and ``verify_grasp_and_lift_via_robot_state`` contributes the gripper-opening
band that distinguishes a held object from a closed empty gripper.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from racap.contracts import GroundedTarget, SkillResult  # noqa: F401  (re-export)

# Heights above a target's centroid, in meters. From
# place_carried_object_at_target_center.
HOVER_HEIGHT = 0.22
RELEASE_HEIGHT = 0.14
RETREAT_HEIGHT = 0.16

# Normalized gripper opening, 0 closed to 1 open. Outside this band the gripper
# either closed on nothing or never closed at all. From
# verify_grasp_and_lift_via_robot_state.
GRIPPER_HOLDING_MIN = 0.02
GRIPPER_HOLDING_MAX = 0.9
# GraspGen's Franka checkpoint predicts the hand-frame pose. The fingertips
# contact the object roughly this far along local +Z; compare that contact
# point—not the wrist origin—to a visually refined handle envelope.
GRASPGEN_FRANKA_CONTACT_OFFSET_M = 0.103

# Straight-down orientation for the panda hand as a wxyz quaternion.
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

PREGRASP_APPROACH = 0.075
LIFT_HEIGHT = 0.15
MIN_MASK_PIXELS = 20

# Height above the object to pause at before re-grounding from the wrist. Close
# enough that the object fills much of the frame, far enough not to occlude it.
WRIST_HOVER = 0.18
# A wrist correction larger than this is more likely a different object than a
# better estimate of the same one.
WRIST_MAX_CORRECTION = 0.06


def _principal_axis(points: np.ndarray) -> list[float]:
    """Dominant horizontal direction of a point cloud.

    A parallel-jaw gripper should close across an object's short axis, so the
    long axis of its footprint tells a policy how to orient the hand. Computed
    in the xy plane because the z spread is dominated by the object's height.
    """
    flat = np.asarray(points)[:, :2]
    if flat.shape[0] < 3:
        return [1.0, 0.0]
    centered = flat - flat.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return [1.0, 0.0]
    axis = vt[0]
    norm = float(np.linalg.norm(axis))
    return [float(axis[0] / norm), float(axis[1] / norm)] if norm > 1e-9 else [1.0, 0.0]


def _yaw_quat(yaw: float) -> np.ndarray:
    """Top-down orientation rotated by ``yaw`` about the world z axis (wxyz).

    The Hamilton product of q_yaw = (cos h, 0, 0, sin h) with
    q_topdown = (0, 1, 0, 0) is (0, cos h, sin h, 0): still a half turn, about
    an axis rotated within the xy plane, so the hand keeps pointing down.

    An earlier version returned (-sin h, cos h, 0, 0), which agrees at yaw = 0
    and is a rotation about x everywhere else -- it pointed the gripper
    sideways. Nothing caught it because the only caller was ``pca_axis``, and a
    strategy that never grasps anything looks like a strategy that is merely
    bad at grasping.
    """
    half = 0.5 * yaw
    return np.array([0.0, np.cos(half), np.sin(half), 0.0], dtype=np.float64)


def _pca_grasp_yaw(long_axis: Any) -> float:
    """Yaw whose *finger closing axis* crosses the object's short axis.

    For the canonical top-down Panda pose, the two fingers translate along
    world ``-Y``.  After a world-Z yaw ``a`` that direction has angle
    ``a - pi/2``.  If the object's long footprint axis has angle ``theta``,
    its short axis is ``theta + pi/2``; therefore ``a = theta + pi``.

    The old code used ``theta + pi/2`` and consequently made the fingers close
    along the long axis.  That is especially damaging for thin packages and
    books: the jaws close on empty space at their ends or push the object flat
    across the table.
    """
    axis = np.asarray(long_axis, dtype=np.float64).reshape(2)
    return float(np.arctan2(axis[1], axis[0]) + np.pi)


def _tilted_pca_quat(
    long_axis: Any,
    *,
    angle_deg: float = 25.0,
) -> np.ndarray:
    """PCA-aligned pinch with one fingertip lowered to catch a flush edge."""
    axis = np.asarray(long_axis, dtype=np.float64).reshape(2)
    theta = float(np.arctan2(axis[1], axis[0]))
    # PCA axes are unoriented.  Select one deterministic representative while
    # preserving the physical grasp under an axis sign flip.
    yaw = float((theta + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
    canonical_axis = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    half = 0.5 * np.deg2rad(float(angle_deg))
    tilt = np.asarray(
        [
            np.cos(half),
            np.sin(half) * canonical_axis[0],
            np.sin(half) * canonical_axis[1],
            0.0,
        ]
    )
    base = _yaw_quat(yaw)
    tw, tx, ty, tz = tilt
    bw, bx, by, bz = base
    quat = np.asarray(
        [
            tw * bw - tx * bx - ty * by - tz * bz,
            tw * bx + tx * bw + ty * bz - tz * by,
            tw * by - tx * bz + ty * bw + tz * bx,
            tw * bz + tx * by - ty * bx + tz * bw,
        ]
    )
    return quat / max(float(np.linalg.norm(quat)), 1e-9)


def select_mechanism_grasp(
    grasps_camera: np.ndarray,
    scores: np.ndarray,
    camera_to_world: np.ndarray,
    preferred_axis_xy: np.ndarray,
    *,
    max_vertical_alignment: float = 0.30,
    min_axis_alignment: float = 0.50,
    min_vertical_closing_alignment: float = 0.55,
) -> tuple[np.ndarray | None, float]:
    """Select a learned side grasp suited to transmitting planar force.

    Transport grasp selection deliberately prefers a vertical approach.  A
    thin drawer bar needs the complementary geometry: the fingers should
    straddle it vertically while the wrist approaches along the measured
    handle/fixture normal.  This selector is asset-independent; it filters
    Contact-GraspNet candidates by their world-frame approach vector and a
    visual RGB-D axis supplied in ``GroundedTarget.metadata``.
    """
    grasps = np.asarray(grasps_camera, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    extrinsics = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
    axis = np.asarray(preferred_axis_xy, dtype=np.float64).reshape(2)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        return None, -float("inf")
    axis /= axis_norm

    best: np.ndarray | None = None
    best_score = -float("inf")
    for grasp, score in zip(grasps, values):
        if grasp.shape != (4, 4) or not np.isfinite(grasp).all():
            continue
        world = extrinsics @ grasp
        approach = np.asarray(world[:3, 2], dtype=np.float64)
        norm = float(np.linalg.norm(approach))
        if norm <= 1e-6:
            continue
        approach /= norm
        horizontal = float(np.linalg.norm(approach[:2]))
        if horizontal <= 1e-6 or abs(float(approach[2])) > max_vertical_alignment:
            continue
        closing_axis = np.asarray(world[:3, 0], dtype=np.float64)
        closing_norm = float(np.linalg.norm(closing_axis))
        if (
            closing_norm <= 1e-6
            or abs(float(closing_axis[2] / closing_norm)) < min_vertical_closing_alignment
        ):
            continue
        # ``approach`` points from wrist toward the predicted fingertip
        # contact. For a fixture handle the wrist must remain outside, so the
        # approach direction is inward (opposite the measured outward normal).
        # Absolute alignment admitted the mirrored grasp with its palm inside
        # the cabinet, which is geometrically parallel but physically invalid.
        alignment = -float((approach[:2] / horizontal) @ axis)
        if alignment < min_axis_alignment:
            continue
        # Alignment is a geometric prior, not a replacement for the learned
        # contact quality.  It only breaks near-score ties among admissible
        # side approaches.
        combined = float(score) + 0.10 * alignment
        if combined > best_score:
            best, best_score = world, combined
    return best, best_score


def canonical_side_bar_grasp(
    contact_xyz: np.ndarray,
    outward_axis_xy: np.ndarray,
    *,
    finger_depth_m: float = GRASPGEN_FRANKA_CONTACT_OFFSET_M,
) -> np.ndarray | None:
    """Construct a front-on Panda hand pose for a protruding horizontal bar.

    The RGB-D articulation geometry already supplies the drawer's outward
    normal.  Local +Z points from the hand base toward the fingertips, so it
    must face inward; local +X is the Panda jaw-closing axis and is made
    vertical so the fingers straddle the bar.  This is a deterministic,
    kinematically simple fallback for thin bars, not an asset-specific pose.
    """
    contact = np.asarray(contact_xyz, dtype=np.float64).reshape(3)
    outward = np.asarray(outward_axis_xy, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(outward))
    if norm <= 1e-6 or not np.isfinite(contact).all():
        return None
    outward /= norm
    approach = np.asarray([-outward[0], -outward[1], 0.0], dtype=np.float64)
    closing = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = np.cross(approach, closing)
    rotation = np.column_stack([closing, lateral, approach])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = contact - approach * float(finger_depth_m)
    return pose


def _rim_contact(
    points: np.ndarray,
    *,
    wedges: int = 8,
) -> tuple[np.ndarray, float, float] | None:
    """A grasp on the outer wall of a perceived cloud, not its centre.

    Bowls and mugs are empty in the middle: descending on the centroid puts the
    fingers into the cavity or against the far rim, and the hand stops short.
    The usable contact is the outermost high point of some angular wedge, with
    the hand yawed so the jaws close across the wall rather than along it.

    Returns ``(xy, yaw, local_top_z)``, or ``None`` when the cloud is too thin
    to have a rim distinguishable from its centre.

    The high band is a fraction of the cloud's own height, not a fixed 1.5 cm:
    a depth mask that includes the table or a cabinet wall can stretch the
    z-range to 20 cm, and a fixed band then holds only a handful of points.
    Falling back to the whole cloud was worse -- it picked the outer footprint
    of the base and closed 17 cm below the rim.
    """
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if cloud.shape[0] < MIN_MASK_PIXELS:
        return None

    top_z = float(cloud[:, 2].max())
    bottom_z = float(cloud[:, 2].min())
    band = max(0.02, 0.35 * max(top_z - bottom_z, 0.0))
    high = cloud[cloud[:, 2] >= top_z - band]
    if high.shape[0] < wedges:
        return None

    centre = high[:, :2].mean(axis=0)
    offsets = high[:, :2] - centre
    radii = np.linalg.norm(offsets, axis=1)
    span = float(radii.max()) if radii.size else 0.0
    # A solid box's high points sit near the centre; a bowl's sit on a ring.
    if span < 0.02:
        return None

    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    best: tuple[float, np.ndarray, float, float] | None = None
    for index in range(wedges):
        lo = -np.pi + 2 * np.pi * index / wedges
        hi = lo + 2 * np.pi / wedges
        in_wedge = (angles >= lo) & (angles < hi) if index < wedges - 1 else (angles >= lo)
        if in_wedge.sum() == 0:
            continue
        pick = high[in_wedge][radii[in_wedge].argmax()]
        xy = pick[:2]
        support = high[np.linalg.norm(high[:, :2] - xy, axis=1) <= 0.025]
        local_top = float(support[:, 2].max()) if support.size else float(pick[2])
        # A candidate whose local top is far below the cloud top is the base of
        # the object, not its rim.
        if local_top < top_z - 0.04:
            continue
        score = float(np.linalg.norm(xy - centre)) + 0.001 * support.shape[0]
        radial = xy - centre
        yaw = float(np.arctan2(radial[1], radial[0]))
        if best is None or score > best[0]:
            best = (score, xy.copy(), yaw, local_top)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _close_height(target: GroundedTarget, local_top: float, cloud_top: float) -> float:
    """Transfer a measured depth lowering onto a local contact height.

    Policies often write a lowered ``top_z`` into the target's metadata so the
    fingers close below the visible top. Rim and affordance contacts have their
    own local tops, so the same absolute value would aim below the table on a
    short rim. The depth the metadata asks for relative to the cloud top is
    what transfers.
    """
    requested = target.metadata.get("top_z")
    if requested is None:
        return local_top
    depth = max(0.0, cloud_top - float(requested))
    return local_top - depth


class LiberoOracle:
    """Privileged accessors. For verification and statistics only.

    Never hand this object to a policy or to the runtime agent. Doing so turns
    a perception problem into a lookup and makes the resulting success rates
    meaningless.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def task_completed(self) -> bool:
        return bool(self._env.task_completed())

    def reward(self) -> float:
        return float(self._env.compute_reward())

    def predicate_status(self) -> list[dict[str, Any]]:
        """Evaluator-only native predicate breakdown for failure diagnosis.

        This remains behind :class:`LiberoOracle`; policy code never receives
        it.  LIBERO wraps the task environment one or two times, so walk the
        bounded wrapper chain until the parsed goal and predicate evaluator
        live on the same object.
        """
        current = getattr(getattr(self._env, "handle", None), "env", None)
        visited: set[int] = set()
        for _ in range(8):
            if current is None or id(current) in visited:
                break
            visited.add(id(current))
            parsed = getattr(current, "parsed_problem", None)
            evaluate = getattr(current, "_eval_predicate", None)
            if parsed is not None and callable(evaluate):
                result = []
                for state in parsed.get("goal_state", []) or []:
                    try:
                        satisfied = bool(evaluate(state))
                    except Exception:
                        satisfied = False
                    result.append(
                        {
                            "predicate": [str(value) for value in state],
                            "satisfied": satisfied,
                        }
                    )
                return result
            current = getattr(current, "env", None) or getattr(current, "unwrapped", None)
        return []

    def predicate_diagnostics(self) -> list[dict[str, Any]]:
        """Evaluator-only metric details behind native containment checks."""
        current = getattr(getattr(self._env, "handle", None), "env", None)
        visited: set[int] = set()
        for _ in range(8):
            if current is None or id(current) in visited:
                break
            visited.add(id(current))
            parsed = getattr(current, "parsed_problem", None)
            sim = getattr(current, "sim", None)
            body_ids = getattr(current, "obj_body_id", {}) or {}
            sites = getattr(current, "object_sites_dict", {}) or {}
            if parsed is not None and sim is not None:
                rows: list[dict[str, Any]] = []
                for state in parsed.get("goal_state", []) or []:
                    values = [str(value) for value in state]
                    row: dict[str, Any] = {"predicate": values}
                    if (
                        len(values) >= 3
                        and values[0].lower() == "in"
                        and values[1] in body_ids
                        and values[2] in sites
                    ):
                        body_position = np.asarray(
                            sim.data.body_xpos[int(body_ids[values[1]])],
                            dtype=float,
                        )
                        site_position = np.asarray(sim.data.get_site_xpos(values[2]), dtype=float)
                        site_matrix = np.asarray(
                            sim.data.get_site_xmat(values[2]), dtype=float
                        ).reshape(3, 3)
                        half_size = np.abs(
                            site_matrix @ np.asarray(sites[values[2]].size, dtype=float)
                        )
                        lower = site_position - half_size
                        lower[2] -= 0.01  # native SiteObject.in_box allowance
                        upper = site_position + half_size
                        lower_margin = body_position - lower
                        upper_margin = upper - body_position
                        row.update(
                            {
                                "object_world_position": body_position.tolist(),
                                "site_world_position": site_position.tolist(),
                                "site_lower_bound": lower.tolist(),
                                "site_upper_bound": upper.tolist(),
                                "lower_margin": lower_margin.tolist(),
                                "upper_margin": upper_margin.tolist(),
                                "axis_inside": [
                                    bool(lower[i] < body_position[i] < upper[i]) for i in range(3)
                                ],
                            }
                        )
                    rows.append(row)
                return rows
            current = getattr(current, "env", None) or getattr(current, "unwrapped", None)
        return []

    def scene_object_poses(self) -> dict[str, list[float]]:
        """World-frame rigid-object positions for offline rollout audits."""
        current = getattr(getattr(self._env, "handle", None), "env", None)
        visited: set[int] = set()
        for _ in range(8):
            if current is None or id(current) in visited:
                break
            visited.add(id(current))
            sim = getattr(current, "sim", None)
            body_ids = getattr(current, "obj_body_id", {}) or {}
            if sim is not None and body_ids:
                rows: dict[str, list[float]] = {}
                for name, body_id in sorted(body_ids.items()):
                    try:
                        rows[str(name)] = [
                            round(float(v), 6) for v in sim.data.body_xpos[int(body_id)]
                        ]
                    except Exception:
                        continue
                return rows
            current = getattr(current, "env", None) or getattr(current, "unwrapped", None)
        return {}

    def object_names(self) -> list[str]:
        raw = self._env.handle.env.env._get_observations()
        return sorted(k[: -len("_pos")] for k in raw if k.endswith("_pos"))

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Truth for one entity, or ``None`` when it has no rigid pose.

        Not every ``*_pos`` key in the observation names an object:
        ``robot0_joint_pos`` is a joint vector with no matching ``_quat``, and
        pairing the two blindly raises on it.
        """
        # Read rigid bodies and sites from the same simulator/world frame.
        # Robosuite observations express object positions in a robot-relative
        # frame, while get_site_xpos is world-frame; mixing the two produced
        # meaningless 50 cm containment deltas in failure artifacts.
        current = getattr(getattr(self._env, "handle", None), "env", None)
        visited: set[int] = set()
        for _ in range(8):
            if current is None or id(current) in visited:
                break
            visited.add(id(current))
            sites = getattr(current, "object_sites_dict", {}) or {}
            sim = getattr(current, "sim", None)
            body_ids = getattr(current, "obj_body_id", {}) or {}
            if name in body_ids and sim is not None:
                try:
                    body_id = int(body_ids[name])
                    return (
                        np.asarray(sim.data.body_xpos[body_id], dtype=float),
                        np.asarray(sim.data.body_xquat[body_id], dtype=float),
                    )
                except Exception:
                    return None
            if name in sites and sim is not None:
                try:
                    from robosuite.utils import transform_utils

                    return (
                        np.asarray(sim.data.get_site_xpos(name), dtype=float),
                        np.asarray(
                            transform_utils.mat2quat(sim.data.get_site_xmat(name)), dtype=float
                        ),
                    )
                except Exception:
                    return None
            current = getattr(current, "env", None) or getattr(current, "unwrapped", None)

        # Non-object observation entities retain the older best-effort
        # fallback, still useful for evaluator diagnostics when no body/site
        # registry entry exists.
        raw = self._env.handle.env.env._get_observations()
        if f"{name}_pos" in raw and f"{name}_quat" in raw:
            return np.asarray(raw[f"{name}_pos"]), np.asarray(raw[f"{name}_quat"])
        return None


class LiberoPrimitiveRuntime:
    """A :class:`racap.contracts.PrimitiveRuntime` backed by LIBERO + RATs.

    Every method maps onto primitives from ``FrankaLiberoApiReducedSkillLibrary``
    and nothing else, so a policy written against this runtime stays portable to
    any other backend that can supply the same primitive surface.
    """

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        *,
        seed: int = 0,
        public_seed_indexed: bool = False,
        grasp_backend: str = "graspnet",
        max_steps: int = 8000,
        control_freq: int = 20,
    ) -> None:
        from rats.envs.simulators.libero import FrankaLiberoEnv
        from rats.integrations.franka.libero_reduced_skill_library import (
            FrankaLiberoApiReducedSkillLibrary,
        )

        self.suite_name = suite_name
        self.task_id = task_id
        self.seed = seed
        self.public_seed_indexed = bool(public_seed_indexed)
        simulator_seed = seed + 1 if self.public_seed_indexed else seed

        self._env = FrankaLiberoEnv(
            suite_name=suite_name,
            task_id=task_id,
            privileged=False,
            max_steps=max_steps,
            control_freq=control_freq,
            seed=simulator_seed,
        )
        self._env.reset(seed=simulator_seed)
        self._record_experiment_reset(
            public_seed=seed,
            simulator_seed=simulator_seed,
        )
        self._api = FrankaLiberoApiReducedSkillLibrary(self._env, grasp_backend=grasp_backend)
        self._fn = self._api.functions()
        self.oracle = LiberoOracle(self._env)
        self.instruction = getattr(self._env.handle, "task_language", "")
        self._last_grasp_quat = TOP_DOWN_QUAT.copy()
        self._points_cache: dict[str, np.ndarray] = {}
        self._camera_home: dict[str, np.ndarray] = {}
        self._view_offset: np.ndarray | None = None
        self._grasp_origin: dict[str, np.ndarray] = {}

    def _record_experiment_reset(
        self,
        *,
        public_seed: int,
        simulator_seed: int,
    ) -> None:
        """Append evaluator-only reset telemetry when a measured run requests it.

        This hook has no policy-visible return value and performs no simulator
        action. Recording every backend reset, rather than inferring one from a
        terminal summary, lets the comparison reject hidden retries and seed
        mismatches symmetrically across RACaP, RATS, and CaP-X.
        """

        target = os.environ.get("RACAP_SIM_EPISODE_TELEMETRY_PATH", "").strip()
        if not target:
            return
        record = {
            "event": "environment_reset",
            "time": time.time(),
            "pid": os.getpid(),
            "episode_key": os.environ.get("RACAP_EPISODE_KEY", ""),
            "task_prompt": str(getattr(self._env.handle, "task_language", "") or ""),
            "public_seed": int(public_seed),
            "seed": int(simulator_seed),
            "init_state_index": getattr(self._env, "_current_init_state_index", None),
            "suite": self.suite_name,
            "task_id": self.task_id,
            "environment_class": type(self._env).__name__,
        }
        path = Path(target).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_evaluator_checkpoint(self, event: str) -> None:
        """Persist native progress for offline timing curves only.

        The controller never receives this record.  It is intentionally
        enabled only when the evaluator supplies ``RACAP_NATIVE_TELEMETRY_PATH``
        and lets long-horizon analysis reconstruct predicate completion at a
        wall-clock budget without polling MuJoCo from another thread.
        """
        target = os.environ.get("RACAP_NATIVE_TELEMETRY_PATH", "").strip()
        if not target:
            return
        predicates = self.oracle.predicate_status()
        record = {
            "event": str(event),
            "time": time.time(),
            "pid": os.getpid(),
            "episode_key": os.environ.get("RACAP_EPISODE_KEY", ""),
            "native_success": bool(predicates and all(row["satisfied"] for row in predicates)),
            "native_predicates": predicates,
            "simulator_steps": int(getattr(self._env, "_sim_step_count", 0)),
            "init_state_index": getattr(self._env, "_current_init_state_index", None),
        }
        path = Path(target).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- lifecycle

    def reset(self, seed: int | None = None) -> None:
        # Cameras are restored first: a moved camera survives an env reset,
        # which would silently carry one episode's viewpoint into the next.
        self.reset_cameras()
        public_seed = self.seed if seed is None else seed
        simulator_seed = public_seed + 1 if self.public_seed_indexed else public_seed
        self._env.reset(seed=simulator_seed)
        self._record_experiment_reset(
            public_seed=public_seed,
            simulator_seed=simulator_seed,
        )
        self._last_grasp_quat = TOP_DOWN_QUAT.copy()
        self._points_cache.clear()
        self._grasp_origin.clear()

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    # ------------------------------------------------------------- perception

    def observe(self) -> dict[str, Any]:
        """Non-privileged observation: two RGB-D cameras plus proprioception."""
        return self._fn["get_observation"]()

    # --------------------------------------------------------- rollout capture

    def enable_video_capture(self, *, subsample_rate: int = 8, wrist_camera: bool = False) -> None:
        """Record simulator frames without changing the controller trajectory.

        RATs already samples frames immediately after low-level simulator
        steps.  Exposing that recorder here keeps evaluation instrumentation
        outside the policy and avoids reconstructing a "video" from the much
        sparser before/after VLM images.  ``subsample_rate`` is measured in
        simulator control steps (20 Hz in the default environment).
        """
        self._env._subsample_rate = max(1, int(subsample_rate))
        self._env.enable_video_capture(True, clear=True, wrist_camera=bool(wrist_camera))

    def disable_video_capture(self) -> None:
        self._env.enable_video_capture(False, clear=False)

    def capture_video_frame(self) -> None:
        """Add an exact current-state frame, e.g. at a ReAct tool boundary."""
        self._env._record_frame()

    def video_capture_enabled(self) -> bool:
        return bool(self._env._record_frames)

    def video_frame_count(self) -> int:
        return int(self._env.get_video_frame_count())

    def video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        return self._env.get_video_frames(clear=clear)

    def wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        return self._env.get_wrist_video_frames(clear=clear)

    def _camera(
        self, obs: dict[str, Any], wrist: bool, view: str | None = None
    ) -> tuple[np.ndarray, ...]:
        """RGB, depth, intrinsics and pose for one viewpoint.

        The two cameras the environment renders every step come straight from
        the observation. Anything else is rendered on demand, which costs a
        frame but opens up the rest of the scene's cameras.
        """
        name = view or ("robot0_eye_in_hand" if wrist else "agentview")
        cam = obs.get(name)
        if cam is not None:
            return (
                cam["images"]["rgb"],
                cam["images"]["depth"],
                cam["intrinsics"],
                cam["pose_mat"],
            )
        return self._render_view(name)

    # -------------------------------------------------------- active perception

    @property
    def _sim(self):
        return self._env.handle.env.sim

    def views(self) -> list[str]:
        """Every viewpoint this scene can be observed from.

        Only ``agentview`` and ``robot0_eye_in_hand`` are rendered each step;
        the others exist in the scene and can be rendered on request. They are
        worth reaching for when an object is hidden behind another from the
        default angle, which on libero_object is the usual reason a grounding
        lands on the wrong thing.
        """
        model = self._sim.model
        return [model.camera_id2name(i) for i in range(model.ncam)]

    def _robot_frame_offset(self) -> np.ndarray:
        """Translation from robosuite's world frame to the frame poses use here.

        The per-step observation reports camera poses in the robot's frame,
        while ``get_camera_extrinsic_matrix`` reports them in the simulator's
        world frame. On libero_object the two differ by 0.6 m along x, which is
        just large enough to look like a plausible perception error rather than
        a frame mismatch: every on-demand view came back roughly 0.6-1.0 m off
        while its image and intrinsics were byte-identical to the observation's.

        Measured against the one camera both paths describe, rather than
        hard-coded, so it stays correct if the scene moves the robot base.
        """
        if self._view_offset is None:
            from robosuite.utils.camera_utils import get_camera_extrinsic_matrix

            reference = self.observe().get("agentview")
            if reference is None:
                self._view_offset = np.zeros(3)
            else:
                world = get_camera_extrinsic_matrix(self._sim, "agentview")
                self._view_offset = np.asarray(reference["pose_mat"])[:3, 3] - world[:3, 3]
        return self._view_offset

    def _render_view(
        self, view: str, *, width: int = 256, height: int = 256
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Render one camera and return RGB, metric depth, intrinsics, pose."""
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_real_depth_map,
        )

        sim = self._sim
        rgb, depth = sim.render(camera_name=view, width=width, height=height, depth=True)
        # MuJoCo renders bottom-up and returns depth on a normalised buffer.
        rgb = np.ascontiguousarray(rgb[::-1])
        depth = get_real_depth_map(sim, depth[::-1][..., None])[..., 0]

        extrinsic = get_camera_extrinsic_matrix(sim, view).copy()
        extrinsic[:3, 3] += self._robot_frame_offset()
        return (
            rgb,
            np.ascontiguousarray(depth),
            get_camera_intrinsic_matrix(sim, view, height, width),
            extrinsic,
        )

    def move_camera(
        self,
        view: str = "agentview",
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
    ) -> dict[str, Any]:
        """Shift a camera in world coordinates and keep it there.

        MuJoCo will not pick the new pose up until the model is stepped
        forward, so a render taken immediately after writing ``cam_pos`` shows
        the old viewpoint. Forgetting that call yields images that look
        plausible and are simply stale.

        Cameras stay where they are put until :meth:`reset_cameras`, so a skill
        that moves one is responsible for restoring it.
        """
        sim = self._sim
        try:
            index = sim.model.camera_name2id(view)
        except Exception:
            return {"ok": False, "reason": f"unknown view {view!r}"}

        self._camera_home.setdefault(view, sim.model.cam_pos[index].copy())
        sim.model.cam_pos[index] = sim.model.cam_pos[index] + np.array([dx, dy, dz])
        sim.forward()
        return {"ok": True, "view": view, "pos": sim.model.cam_pos[index].tolist()}

    def reset_cameras(self) -> None:
        """Return every moved camera to where the scene put it."""
        sim = self._sim
        for view, home in self._camera_home.items():
            sim.model.cam_pos[sim.model.camera_name2id(view)] = home
        if self._camera_home:
            sim.forward()
        self._camera_home.clear()

    def inspect(
        self,
        label: str,
        *,
        view: str = "agentview",
        options: list[str] | None = None,
        grounder_model: str | None = None,
        target: GroundedTarget | None = None,
    ) -> dict[str, Any]:
        """Crop to a label's neighbourhood, enlarge it, and ask what is there.

        Grounding a small object in a 256-pixel frame asks the VLM to read a
        patch a few dozen pixels across. Cropping around the current estimate
        and upsampling recovers the label text on a can, which is what
        distinguishes lookalike products that no amount of geometry can
        separate.

        ``options`` narrows the answer to a closed set. Left open, the model
        will happily invent a plausible product name.

        Pass ``target`` to check a grounding you already have. That is the
        useful form: a wrong grounding is typically 0.2 m out, sitting squarely
        on a neighbouring object, and cropping to it asks the one question that
        distinguishes the two. Without ``target`` this grounds the label afresh
        and can only tell you whether that fresh attempt is self-consistent.
        """
        from racap.backends.vlm import identify

        if target is None:
            target = self.localize(label, detector="vlm_bbox_sam3", grounder_model=grounder_model)
        box = target.metadata.get("bbox")
        rgb, _, _, _ = self._camera(self.observe(), wrist=False, view=view)
        if not box:
            return {"ok": False, "reason": "no_bbox", "label": label}

        h, w = rgb.shape[:2]
        pad = 12
        x0 = max(0, int(box[0]) - pad)
        y0 = max(0, int(box[1]) - pad)
        x1 = min(w, int(box[2]) + pad)
        y1 = min(h, int(box[3]) + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return {"ok": False, "reason": "degenerate_crop", "bbox": box}

        crop = rgb[y0:y1, x0:x1]
        answer = identify(crop, options=options, model=grounder_model)
        return {
            "ok": True,
            "label": label,
            "answer": answer,
            "bbox": [x0, y0, x1, y1],
            "crop_shape": list(crop.shape[:2]),
            "matches": answer.strip().lower() == label.strip().lower(),
        }

    def crop_frame(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        *,
        view: str = "agentview",
        pad: int = 24,
    ) -> str:
        """Encode a current scene crop from an already grounded pixel box.

        This is deliberately perception-free: state controllers can zoom an
        unobstructed post-home fixture using the pre-action box without paying
        for, or risking drift from, another semantic grounding call.
        """
        from racap.backends.vlm import encode_png

        rgb, _, _, _ = self._camera(self.observe(), wrist=False, view=view)
        h, w = rgb.shape[:2]
        x0 = max(0, int(bbox[0]) - int(pad))
        y0 = max(0, int(bbox[1]) - int(pad))
        x1 = min(w, int(bbox[2]) + int(pad))
        y1 = min(h, int(bbox[3]) + int(pad))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return ""
        return encode_png(np.asarray(rgb[y0:y1, x0:x1]).astype("uint8"))

    def verify_destination_alias(
        self,
        canonical: str,
        candidate: str,
        *,
        target: GroundedTarget,
        view: str = "agentview",
        grounder_model: str | None = None,
    ) -> dict[str, Any]:
        """Judge whether two names denote the same cropped destination.

        ``identify(..., options=...)`` always chooses the closest option even
        when unsure. That is useful for classifying products, but too permissive
        for alias admission: a nearby drawer can be the closest visual match to
        "wine rack" without being the rack. This stricter question explicitly
        separates identity from proximity and permits a negative answer.
        """
        import cv2

        from racap.backends.llm import ask
        from racap.backends.vlm import DEFAULT_GROUNDER, encode_png

        box = target.metadata.get("bbox")
        rgb, _, _, _ = self._camera(self.observe(), wrist=False, view=view)
        if not box:
            return {"ok": False, "reason": "no_bbox"}
        h, w = rgb.shape[:2]
        pad = 12
        x0 = max(0, int(box[0]) - pad)
        y0 = max(0, int(box[1]) - pad)
        x1 = min(w, int(box[2]) + pad)
        y1 = min(h, int(box[3]) + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return {"ok": False, "reason": "degenerate_crop"}
        crop = np.asarray(rgb[y0:y1, x0:x1]).astype("uint8")
        marked = np.asarray(rgb).astype("uint8").copy()
        cv2.rectangle(marked, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 3)
        same = "same destination"
        different = "different destination"
        reply = ask(
            "You resolve whether a generic visual-inventory caption and a "
            "task phrase refer to the same physical destination. The first "
            "image is the complete scene with that candidate in a red box; "
            "the second is its crop.",
            f"The inventory called the cropped structure {candidate!r}. The "
            f"instruction calls its destination {canonical!r}. Do these names "
            "refer to the same physical structure shown in the crop? Inventory "
            "captions are generic and may omit a structure's purpose, so exact "
            "word synonymy is not required; judge its visible geometry and "
            "functional affordance. Still require the same referent: a nearby "
            "object, a surface that merely could hold the item, or a container "
            "next to the requested structure is different. If the defining "
            "structure or affordance is not visually supported, answer "
            f"{different!r}. "
            f"Answer exactly {same!r} or {different!r}.",
            images=[encode_png(marked), encode_png(crop)],
            model=grounder_model or DEFAULT_GROUNDER,
            max_tokens=40,
            temperature=0.0,
        )
        answer = str(reply or "").strip().strip('".').lower()
        return {
            "ok": answer == same,
            "answer": same if answer == same else different,
            "bbox": [x0, y0, x1, y1],
        }

    def verify_destination_grounding(
        self,
        canonical: str,
        *,
        target: GroundedTarget,
        view: str = "agentview",
        grounder_model: str | None = None,
    ) -> dict[str, Any]:
        """Independently veto a proposed placement-destination box.

        Grounding and verification deliberately use different visual prompts.
        The verifier sees both the full scene with the proposal marked and a
        tight crop.  This lets it reject a visually plausible background shelf,
        a nearby movable object, or a parent fixture when the instruction asks
        for a particular opening.  A rejected box can then be fed back to the
        grounder instead of silently selecting the closest inventory caption.
        """
        import cv2

        from racap.backends.llm import ask
        from racap.backends.vlm import DEFAULT_GROUNDER, bbox as vlm_bbox, encode_png

        box = target.metadata.get("bbox")
        rgb, _, _, _ = self._camera(self.observe(), wrist=False, view=view)
        if not box or len(box) < 4:
            return {"ok": False, "answer": "invalid", "reason": "no_bbox"}
        h, w = rgb.shape[:2]
        x0 = max(0, min(w - 1, int(box[0])))
        y0 = max(0, min(h - 1, int(box[1])))
        x1 = max(x0 + 1, min(w, int(box[2])))
        y1 = max(y0 + 1, min(h, int(box[3])))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return {"ok": False, "answer": "invalid", "reason": "degenerate_bbox"}

        marked = np.asarray(rgb).astype("uint8").copy()
        cv2.rectangle(marked, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 3)
        crop = np.asarray(rgb[y0:y1, x0:x1]).astype("uint8")
        lowered = " ".join(str(canonical).lower().split())
        upper_tier = bool(re.search(r"\b(?:topmost|upper)\b|above the horizontal divider", lowered))
        lower_tier = bool(
            re.search(r"\b(?:bottommost|lower)\b|below the horizontal divider", lowered)
        )
        tier = (
            "upper"
            if upper_tier and not lower_tier
            else ("lower" if lower_tier and not upper_tier else "")
        )
        if tier and any(word in lowered for word in ("shelf", "rack")):
            # Verify the requested sub-part by its relation to an independently
            # grounded parent fixture.  This catches a confident VLM proposal
            # for the lower cubby when the language asks for the upper cubby
            # (and vice versa) without relying on an asset name or world pose.
            parent_box = vlm_bbox(
                rgb,
                "the entire foreground shelf or rack fixture containing the "
                "requested open compartment; include all of its vertical tiers",
                model=grounder_model or DEFAULT_GROUNDER,
            )
            if parent_box is not None:
                parent_mid = 0.5 * (float(parent_box.y0) + float(parent_box.y1))
                candidate_mid = 0.5 * (float(y0) + float(y1))
                parent_height = max(float(parent_box.y1 - parent_box.y0), 1.0)
                candidate_height = float(y1 - y0)
                # A whole-fixture proposal is allowed through; dense RGB-D
                # component selection below will apply the requested tier.
                is_local_part = candidate_height < 0.72 * parent_height
                wrong_half = (
                    tier == "upper" and candidate_mid > parent_mid + 0.05 * parent_height
                ) or (tier == "lower" and candidate_mid < parent_mid - 0.05 * parent_height)
                parent_bbox = [
                    int(parent_box.x0),
                    int(parent_box.y0),
                    int(parent_box.x1),
                    int(parent_box.y1),
                ]
                if is_local_part and wrong_half:
                    return {
                        "ok": False,
                        "answer": "incorrect destination",
                        "reason": "wrong_vertical_tier",
                        "tier": tier,
                        "bbox": [x0, y0, x1, y1],
                        "parent_bbox": parent_bbox,
                    }
                target.metadata["vertical_tier"] = tier
                target.metadata["vertical_tier_scope"] = (
                    "local_part" if is_local_part else "parent_crop"
                )
                target.metadata["parent_bbox"] = parent_bbox
        correct = "correct destination"
        incorrect = "incorrect destination"
        reply = ask(
            "You independently verify visual grounding for a robot placement "
            "destination. The first image is the full robot scene and the red "
            "box is the proposed grounding; the second is its crop.",
            f"The requested destination is {canonical!r}. Decide whether the "
            "red box identifies the correct physical destination or its correct "
            "parent fixture. Use the full "
            "scene, visible geometry, functional affordance, and robot workspace "
            "context. Reject background furniture, a nearby movable object, the "
            "wrong fixture, and a wrong tier/part when it is visibly distinct. "
            "A box around the correct parent shelf/rack/cabinet is acceptable: "
            "a separate RGB-D geometry stage will refine its empty opening. An "
            "outer wall or nearby object without the requested structure is not. "
            "Do not choose the "
            "closest plausible thing when the requested destination is absent. "
            f"Answer exactly {correct!r} or {incorrect!r}.",
            images=[encode_png(marked), encode_png(crop)],
            model=grounder_model or DEFAULT_GROUNDER,
            max_tokens=40,
            temperature=0.0,
        )
        answer = str(reply or "").strip().strip('".').lower()
        return {
            "ok": answer == correct,
            "answer": correct if answer == correct else incorrect,
            "bbox": [x0, y0, x1, y1],
            **({"tier": tier} if tier else {}),
            **(
                {"parent_bbox": target.metadata["parent_bbox"]}
                if "parent_bbox" in target.metadata
                else {}
            ),
        }

    def verify_pick_grounding(
        self,
        canonical: str,
        *,
        target: GroundedTarget,
        view: str = "agentview",
        grounder_model: str | None = None,
    ) -> dict[str, Any]:
        """Verify a proposed source referent without discarding scene relations.

        A tight crop can establish appearance, but it cannot establish that one
        of several identical objects is the front/middle/back/left/right
        instance.  Mark the candidate in the unobstructed full scene and ask an
        independent closed-set question.  Rejected candidates are fed back to
        ``localize_with_feedback`` by the policy, so this is a general
        perception-revision loop rather than a camera-axis heuristic.
        """
        import cv2

        from racap.backends.llm import ask
        from racap.backends.vlm import DEFAULT_GROUNDER, encode_png

        box = target.metadata.get("bbox")
        rgb, _, _, _ = self._camera(self.observe(), wrist=False, view=view)
        if not box or len(box) < 4:
            return {"ok": False, "matches": None, "reason": "no_bbox"}
        h, w = rgb.shape[:2]
        x0 = max(0, min(w - 1, int(box[0])))
        y0 = max(0, min(h - 1, int(box[1])))
        x1 = max(x0 + 1, min(w, int(box[2])))
        y1 = max(y0 + 1, min(h, int(box[3])))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return {"ok": False, "matches": None, "reason": "degenerate_bbox"}

        marked = np.asarray(rgb).astype("uint8").copy()
        cv2.rectangle(marked, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 3)
        crop = np.asarray(rgb[y0:y1, x0:x1]).astype("uint8")
        # Manipulation targets are often only 30--70 pixels wide in the full
        # camera.  Sending that tiny crop unchanged encourages a verifier to
        # answer from the requested noun or scene context rather than visible
        # package / shape evidence.  Upsampling adds no information and no
        # category prior, but preserves the available pixels through VLM image
        # preprocessing so category-level differences remain inspectable.
        crop_scale = min(8.0, max(1.0, 256.0 / max(1, min(crop.shape[:2]))))
        if crop_scale > 1.01:
            crop = cv2.resize(
                crop,
                None,
                fx=crop_scale,
                fy=crop_scale,
                interpolation=cv2.INTER_CUBIC,
            )
        # Do not show the requested noun to the visual category inspector.
        # The old single prompt did so and produced a correlated self-check:
        # the same VLM that proposed a dark pudding box as ``butter`` then
        # repeated ``correct referent`` because the task supplied that word.
        # A blind, independently-routed caption makes the visible evidence the
        # premise of the comparison rather than its conclusion.
        verifier_model = (
            os.environ.get("RACAP_VERIFIER_MODEL", "gpt-5.5").strip()
            or "gpt-5.5"
        )
        try:
            blind_description = ask(
                "You are an independent visual object inspector. You have no "
                "access to the robot task or its requested object name.",
                "Describe the single object in this tight crop from visible "
                "evidence only. Name its likely category and mention shape, "
                "material, package colours, readable words, or recognizable "
                "graphics. If the category is uncertain, say so. Be concise.",
                images=[encode_png(crop)],
                model=verifier_model,
                max_tokens=160,
                temperature=0.0,
            )
            category_reply = ask(
                "You compare visual category evidence to a requested robot "
                "referent. Spatial words do not affect object category.",
                f"Requested referent: {canonical!r}\n"
                f"Blind crop description: {blind_description!r}\n"
                "Does the description positively support the requested base "
                "object category? Compare at the manipulation ontology level, "
                "not at the captioner's preferred synonym: a common subtype "
                "or visually synonymous form (for example binder/notebook as "
                "a book, or cup as a mug) supports the requested category. A "
                "functionally different named product/category is a mismatch. "
                "A merely generic or uncertain description is unknown, not a "
                "match. Answer exactly 'category match', "
                "'category mismatch', or 'category unknown'.",
                images=[],
                model=grounder_model or DEFAULT_GROUNDER,
                max_tokens=30,
                temperature=0.0,
            )
        except Exception as exc:
            return {
                "ok": False,
                "matches": None,
                "reason": f"blind_category_inspect_failed:{type(exc).__name__}",
                "bbox": [x0, y0, x1, y1],
                "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
            }
        category_answer = str(category_reply or "").strip().strip('".').lower()
        if category_answer == "category mismatch":
            return {
                "ok": True,
                "matches": False,
                "answer": "incorrect referent",
                "category_answer": category_answer,
                "blind_description": str(blind_description).strip(),
                "verifier_model": verifier_model,
                "bbox": [x0, y0, x1, y1],
                "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
                "view": "blind_crop",
            }
        if category_answer != "category match":
            return {
                "ok": False,
                "matches": None,
                "answer": "category unknown",
                "category_answer": category_answer,
                "blind_description": str(blind_description).strip(),
                "verifier_model": verifier_model,
                "bbox": [x0, y0, x1, y1],
                "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
                "view": "blind_crop",
            }

        correct = "correct relation"
        incorrect = "incorrect relation"
        reply = ask(
            "You verify only which physical instance a marked object is in a "
            "robot scene. Its base visual category was independently verified.",
            f"The requested source phrase is {canonical!r}. The red box marks "
            "the candidate. Compare it with all other instances and check only "
            "the spatial qualifier (front/back/middle/left/right or an explicit "
            "relation). Use the scene camera viewpoint. For front/middle/back, "
            "rank all same-category instances by apparent table depth: front "
            "is nearest the camera, back is farthest, and middle is the one "
            "between them in depth. 'Middle' does not mean the geometric centre "
            "of the image, especially when three objects form a triangle. For "
            "left/right, compare lateral image position. Do not reconsider the "
            f"category. Answer exactly {correct!r} or {incorrect!r}.",
            images=[encode_png(marked)],
            model=grounder_model or DEFAULT_GROUNDER,
            max_tokens=30,
            temperature=0.0,
        )
        answer = str(reply or "").strip().strip('".').lower()
        valid = answer in {correct, incorrect}
        return {
            "ok": valid,
            "matches": (answer == correct) if valid else None,
            "answer": (
                "correct referent"
                if answer == correct
                else "incorrect referent"
                if answer == incorrect
                else answer
            ),
            "category_answer": category_answer,
            "blind_description": str(blind_description).strip(),
            "verifier_model": verifier_model,
            "bbox": [x0, y0, x1, y1],
            "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
            "view": "marked_full_scene",
        }

    def track_target(
        self,
        target: GroundedTarget,
        *,
        appearance_label: str,
        padding_px: int = 24,
        view: str = "agentview",
    ) -> GroundedTarget:
        """Reacquire an established source locally without another VLM vote.

        A failed grasp may slide or tip an object, so its metric pose must be
        refreshed. Re-running global language grounding is both slow and
        unsafe for relational labels: ``front`` can bind to another duplicate
        after contact. This expands the last verified box and uses local SAM +
        RGB-D, preserving causal instance continuity while still measuring the
        object's new pose from pixels.
        """
        box = target.metadata.get("bbox")
        if not box or len(box) < 4:
            return GroundedTarget(
                label=target.label,
                pose=(0.0, 0.0, 0.0),
                confidence=0.0,
                metadata={"reason": "local_track_has_no_bbox"},
            )
        rgb, depth, intr, extr = self._camera(self.observe(), wrist=False, view=view)
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = (int(value) for value in box[:4])
        width, height = max(1, x1 - x0), max(1, y1 - y0)
        pad = max(int(padding_px), int(round(0.35 * max(width, height))))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        crop = np.asarray(rgb)[y0:y1, x0:x1]
        mask_crop, score, info = self._mask_sam3_text(crop, str(appearance_label))
        if mask_crop is None:
            return GroundedTarget(
                label=target.label,
                pose=(0.0, 0.0, 0.0),
                confidence=0.0,
                metadata={
                    "reason": "local_track_no_mask",
                    "bbox": [x0, y0, x1, y1],
                    "detector": "local_sam3_rgbd",
                },
            )
        mask = np.zeros(np.asarray(rgb).shape[:2], dtype=bool)
        mask[y0:y1, x0:x1] = np.asarray(mask_crop, dtype=bool)
        ys, xs = np.where(mask)
        tracked_box = (
            [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            if xs.size
            else [x0, y0, x1, y1]
        )
        tracked = self._target_from_mask(
            target.label,
            mask,
            depth,
            intr,
            extr,
            score,
            {
                **info,
                "detector": "local_sam3_rgbd",
                "bbox": tracked_box,
                "track_search_bbox": [x0, y0, x1, y1],
                "points_label": target.label,
                "from_wrist": False,
            },
        )
        tracked.metadata["pregrasp_identity_evidence"] = target.metadata.get(
            "pregrasp_identity_evidence"
        )
        return tracked

    def track_near_hand(
        self,
        appearance_label: str,
        *,
        reference_size: tuple[float, float] | list[float] | None = None,
        radius_m: float = 0.18,
        view: str = "agentview",
    ) -> GroundedTarget:
        """Reacquire a carried object locally around the visible end effector.

        Text grounding over the full scene is unsafe once the hand is above a
        receptacle: a prompt such as ``book`` can merge the gripper, rim and
        receptacle into one large mask.  The robot's *measured* EEF pose is a
        causal continuity cue, not privileged object state.  Project it into
        the camera, segment only a metric-sized neighbourhood, then retain a
        mask only when its RGB-D body is close to the hand and agrees with the
        clean pre-grasp footprint.  No simulator object pose or predicate is
        consulted.
        """
        rgb, depth, intr, extr = self._camera(self.observe(), wrist=False, view=view)
        rgb = np.asarray(rgb)
        height, width = rgb.shape[:2]
        hand = np.asarray(self.ee_pose()["position"], dtype=float).reshape(3)
        camera_from_world = np.linalg.inv(np.asarray(extr, dtype=float))
        camera = camera_from_world @ np.r_[hand, 1.0]
        if not np.isfinite(camera).all() or float(camera[2]) <= 1e-4:
            return GroundedTarget(
                label=appearance_label,
                pose=(0.0, 0.0, 0.0),
                confidence=0.0,
                metadata={"reason": "hand_not_visible", "detector": "near_hand_sam3_rgbd"},
            )
        u = float(intr[0, 0]) * float(camera[0] / camera[2]) + float(intr[0, 2])
        v = float(intr[1, 1]) * float(camera[1] / camera[2]) + float(intr[1, 2])
        pixel_radius = int(np.clip(float(radius_m) * float(intr[0, 0]) / float(camera[2]), 36, 150))
        x0 = max(0, int(round(u)) - pixel_radius)
        y0 = max(0, int(round(v)) - pixel_radius)
        x1 = min(width, int(round(u)) + pixel_radius)
        y1 = min(height, int(round(v)) + pixel_radius)
        if x1 <= x0 or y1 <= y0:
            return GroundedTarget(
                label=appearance_label,
                pose=(0.0, 0.0, 0.0),
                confidence=0.0,
                metadata={"reason": "empty_hand_crop", "detector": "near_hand_sam3_rgbd"},
            )

        results = self._fn["segment_sam3_text_prompt"](
            np.ascontiguousarray(rgb[y0:y1, x0:x1]), str(appearance_label)
        )
        reference = None
        if reference_size is not None:
            values = np.asarray(reference_size, dtype=float).reshape(-1)
            if values.size >= 2 and np.isfinite(values[:2]).all():
                reference = np.sort(np.abs(values[:2]))[::-1]
        candidates: list[tuple[float, GroundedTarget, np.ndarray]] = []
        for index, item in enumerate(results):
            mask_crop = item.get("mask")
            if mask_crop is None or int(np.count_nonzero(mask_crop)) < MIN_MASK_PIXELS:
                continue
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = np.asarray(mask_crop, dtype=bool)
            points = np.asarray(
                self._fn["mask_to_world_points"](mask, depth, intr, extr), dtype=float
            ).reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
            if points.shape[0] < MIN_MASK_PIXELS:
                continue
            obb = self._fn["get_oriented_bounding_box_from_3d_points"](points)
            centre = np.asarray(obb.get("center", np.median(points, axis=0)), dtype=float).reshape(
                3
            )
            hand_distance = float(np.linalg.norm(centre - hand))
            if hand_distance > float(radius_m):
                continue

            shape_error = 0.0
            robust_size = None
            xy = points[:, :2] - np.median(points[:, :2], axis=0)
            try:
                _, _, vt = np.linalg.svd(xy, full_matrices=False)
                projected = xy @ vt[:2].T
                robust_size = np.sort(
                    np.percentile(projected, 96, axis=0) - np.percentile(projected, 4, axis=0)
                )[::-1]
            except np.linalg.LinAlgError:
                pass
            if reference is not None:
                if robust_size is None or np.any(robust_size <= 1e-4):
                    continue
                ratios = robust_size / np.maximum(reference, 1e-4)
                # The fingers can inflate the short axis, but a caddy/table
                # mask is much longer than the clean carried object.
                if not (0.55 <= ratios[0] <= 1.65 and 0.30 <= ratios[1] <= 3.5):
                    continue
                shape_error = float(np.sum(np.abs(np.log(ratios))))
            score = float(item.get("score", 0.0))
            rank = score - 2.0 * hand_distance - 0.20 * shape_error
            ys, xs = np.where(mask)
            target = GroundedTarget(
                label=appearance_label,
                pose=(float(centre[0]), float(centre[1]), float(centre[2])),
                confidence=max(score, 1e-6),
                metadata={
                    "detector": "near_hand_sam3_rgbd",
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    "track_search_bbox": [x0, y0, x1, y1],
                    "points_label": appearance_label,
                    "top_z": float(points[:, 2].max()),
                    "n_points": int(points.shape[0]),
                    "extent": [
                        float(value) for value in np.asarray(obb.get("extent", [0.0, 0.0, 0.0]))
                    ],
                    "principal_axis": _principal_axis(points),
                    "near_hand_distance_m": hand_distance,
                    "near_hand_shape_size": (
                        None if robust_size is None else [float(value) for value in robust_size]
                    ),
                    "candidate_index": index,
                },
            )
            candidates.append((rank, target, points))
        if not candidates:
            return GroundedTarget(
                label=appearance_label,
                pose=(0.0, 0.0, 0.0),
                confidence=0.0,
                metadata={
                    "reason": "no_shape_consistent_near_hand_mask",
                    "detector": "near_hand_sam3_rgbd",
                    "track_search_bbox": [x0, y0, x1, y1],
                    "n_results": len(results),
                },
            )
        _, target, points = max(candidates, key=lambda row: row[0])
        self._points_cache[appearance_label] = points
        return target

    def _mask_sam3_text(self, rgb: np.ndarray, label: str) -> tuple[np.ndarray | None, float, dict]:
        """Plain SAM3 text prompt: fast, but blind to spatial disambiguation."""
        results = self._fn["segment_sam3_text_prompt"](rgb, label)
        best = None
        for item in results:
            mask = item.get("mask")
            if mask is None or int(np.count_nonzero(mask)) < MIN_MASK_PIXELS:
                continue
            if best is None or item.get("score", 0.0) > best.get("score", 0.0):
                best = item
        if best is None:
            return None, 0.0, {"reason": "no_mask", "n_results": len(results)}
        return (
            np.asarray(best["mask"]).astype(bool),
            float(best.get("score", 0.0)),
            {"n_results": len(results)},
        )

    def _mask_vlm_bbox_sam3(
        self, rgb: np.ndarray, label: str, model: str, *, exhaustive: bool = False
    ) -> tuple[np.ndarray | None, float, dict]:
        """VLM proposes a box, SAM3 segments inside it.

        This is what makes relational labels like "the bowl between the plate and
        the ramekin" tractable: SAM3 alone would return both bowls, but it only
        ever sees the crop the VLM chose.
        """
        from racap.backends import vlm

        box = vlm.bbox(rgb, label, model=model, exhaustive=exhaustive)
        if box is None:
            return None, 0.0, {"reason": "vlm_no_bbox"}

        crop = np.asarray(rgb)[box.y0 : box.y1, box.x0 : box.x1]
        if crop.size == 0:
            return None, 0.0, {"reason": "vlm_empty_bbox"}

        sub_mask, score, info = self._mask_sam3_text(crop, label)
        full = np.zeros(np.asarray(rgb).shape[:2], dtype=bool)
        if sub_mask is None:
            # The VLM localized the object even though SAM3 found no boundary
            # inside the crop; the box itself is still a usable, coarser mask.
            full[box.y0 : box.y1, box.x0 : box.x1] = True
            return full, box.confidence * 0.6, {**info, "fallback": "bbox_as_mask"}
        full[box.y0 : box.y1, box.x0 : box.x1] = sub_mask
        return (
            full,
            float(min(score, box.confidence) if box.confidence else score),
            {
                **info,
                "bbox": [box.x0, box.y0, box.x1, box.y1],
                "grounder": model,
            },
        )

    def _mask_vlm_point_sam3(
        self, rgb: np.ndarray, label: str, model: str
    ) -> tuple[np.ndarray | None, float, dict]:
        """VLM points at a graspable pixel, SAM3 grows the mask from that point."""
        from racap.backends import vlm

        point = vlm.grasp_point(rgb, label, model=model)
        if point is None:
            return None, 0.0, {"reason": "vlm_no_point"}
        u, v, conf = point
        results = self._fn["segment_sam3_point_prompt"](rgb, (u, v))
        best = max(
            (r for r in results if r.get("mask") is not None),
            key=lambda r: r.get("score", 0.0),
            default=None,
        )
        if best is None:
            return None, 0.0, {"reason": "sam3_point_no_mask", "point": [u, v]}
        mask = np.asarray(best["mask"]).astype(bool)
        rows, cols = np.nonzero(mask)
        # Point prompting previously returned no pixel box.  That made the
        # downstream independent crop verifier unable to inspect this fallback
        # candidate, precisely when an alternate grounding prompt was most
        # valuable.  Derive a tight mask box from pixels; this remains ordinary
        # image evidence and exposes no simulator identity or pose.
        info: dict[str, Any] = {"point": [u, v], "grounder": model}
        if rows.size and cols.size:
            info["bbox"] = [
                int(cols.min()),
                int(rows.min()),
                int(cols.max()) + 1,
                int(rows.max()) + 1,
            ]
        return mask, float(min(best.get("score", 0.0), conf)), info

    def localize(
        self,
        label: str,
        *,
        close_view: bool = False,
        detector: str = "sam3",
        grounder_model: str | None = None,
        view: str | None = None,
    ) -> GroundedTarget:
        """Ground a language label to a world-frame centroid.

        ``view`` overrides the camera. Pass a name from :meth:`views` to look
        from somewhere else when the default angle is obstructed; ``close_view``
        is the shorthand for the wrist camera.

        ``detector`` selects the grounding pipeline and is one of the main knobs
        evolution can turn:

        - ``sam3``: SAM3 text prompt only. Cheapest, and adequate when the label
          names a visually unique object.
        - ``vlm_bbox_sam3``: a VLM boxes the object first, SAM3 segments inside
          the box. Slower, but the only option that can resolve labels which
          depend on a spatial relation between lookalike objects.
        - ``vlm_point_sam3``: a VLM points at a graspable pixel and SAM3 grows a
          mask from it. Useful when the object is partly occluded.

        The returned ``confidence`` is the detector's own score. Measured
        against ground truth on libero_object, it does **not** predict whether
        the right object was found: correct groundings came back with scores as
        low as 0.02, and a grounding that was 0.38 m off came back with 0.82.
        Thresholding on it discards good results and keeps bad ones. Judge a
        grounding by geometry instead -- ``n_points``, ``extent``, whether the
        centroid is somewhere a robot could plausibly reach -- and by
        cross-checking with a second view or with :meth:`localize_many`.
        """
        from racap.backends.vlm import DEFAULT_GROUNDER

        obs = self.observe()
        rgb, depth, intr, extr = self._camera(obs, wrist=close_view, view=view)
        model = grounder_model or DEFAULT_GROUNDER

        if detector == "sam3":
            mask, score, info = self._mask_sam3_text(rgb, label)
        elif detector == "vlm_bbox_sam3":
            mask, score, info = self._mask_vlm_bbox_sam3(rgb, label, model)
        elif detector == "vlm_search_bbox_sam3":
            mask, score, info = self._mask_vlm_bbox_sam3(rgb, label, model, exhaustive=True)
        elif detector == "vlm_point_sam3":
            mask, score, info = self._mask_vlm_point_sam3(rgb, label, model)
        else:
            raise ValueError(f"unknown detector {detector!r}")

        if mask is None:
            return GroundedTarget(
                label=label,
                pose=(0.0, 0.0, 0.0),
                confidence=0.0,
                metadata={**info, "detector": detector},
            )
        return self._target_from_mask(
            label,
            mask,
            depth,
            intr,
            extr,
            score,
            {
                **info,
                "detector": detector,
                "from_wrist": close_view,
                "view": view or ("robot0_eye_in_hand" if close_view else "agentview"),
            },
        )

    def localize_across_views(
        self,
        label: str,
        *,
        detector: str = "vlm_search_bbox_sam3",
        views: list[str] | None = None,
        grounder_model: str | None = None,
    ) -> GroundedTarget:
        """Find a missed object from a genuinely different fixed viewpoint.

        A thin object can be nearly edge-on in ``agentview``. Rephrasing the
        same pixels then has a hard recall ceiling, while the scene's fixed
        side/front cameras expose its face without moving the robot or using
        simulator identity. Return the first geometrically usable proposal;
        policy code still performs its independent category check afterward.
        """
        candidates = views or ["sideview", "frontview", "birdview"]
        available = set(self.views())
        for view in candidates:
            if view not in available:
                continue
            target = self.localize(
                label,
                detector=detector,
                grounder_model=grounder_model,
                view=view,
            )
            if (
                target.confidence > 0.0
                and int(target.metadata.get("n_points", 0)) >= MIN_MASK_PIXELS
            ):
                return target
        return GroundedTarget(
            label=label,
            pose=(0.0, 0.0, 0.0),
            confidence=0.0,
            metadata={
                "reason": "not_found_across_views",
                "detector": detector,
                "views_tried": list(candidates),
            },
        )

    def _target_from_mask(
        self,
        label: str,
        mask: np.ndarray,
        depth: np.ndarray,
        intr: np.ndarray,
        extr: np.ndarray,
        score: float,
        info: dict[str, Any],
    ) -> GroundedTarget:
        """Lift a 2D mask to a world-frame target using the depth image."""
        points = self._fn["mask_to_world_points"](mask, depth, intr, extr)
        points = np.asarray(points).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < MIN_MASK_PIXELS:
            return GroundedTarget(
                label=label,
                pose=(0.0, 0.0, 0.0),
                confidence=0.0,
                metadata={**info, "reason": "no_depth_support", "n_points": int(points.shape[0])},
            )

        obb = self._fn["get_oriented_bounding_box_from_3d_points"](points)
        centroid = np.asarray(obb.get("center", points.mean(axis=0)), dtype=float)
        self._points_cache[label] = points

        return GroundedTarget(
            label=label,
            pose=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            confidence=score,
            metadata={
                **info,
                "top_z": float(points[:, 2].max()),
                "n_points": int(points.shape[0]),
                "extent": [float(v) for v in np.asarray(obb.get("extent", [0, 0, 0]))],
                "principal_axis": _principal_axis(points),
            },
        )

    def list_objects(
        self, *, close_view: bool = False, grounder_model: str | None = None
    ) -> list[str]:
        """Names of the manipulable objects currently visible, from a VLM.

        Useful for knowing what is on the table. Do **not** feed the result to
        :meth:`localize_many` as extra labels: the names come back as generic
        descriptions such as "blue can", which alias the object you are after,
        and the no-shared-box constraint then pushes your target onto something
        else. Measured, that drops grounding from 5 correct in 6 to 2 in 5.
        """
        from racap.backends.vlm import DEFAULT_GROUNDER, inventory

        obs = self.observe()
        rgb, _, _, _ = self._camera(obs, wrist=close_view)
        return inventory(rgb, model=grounder_model or DEFAULT_GROUNDER)

    def localize_many(
        self,
        labels: list[str],
        *,
        close_view: bool = False,
        grounder_model: str | None = None,
        cache: bool = True,
    ) -> dict[str, GroundedTarget]:
        """Ground several labels in one call, under a no-shared-box constraint.

        One VLM call boxes every name at once, then SAM3 refines each box into a
        mask. Cheaper than one call per label.

        Be careful what you put in ``labels``. Mutual exclusion helps only when
        the names describe genuinely different objects. Measured on
        libero_object, grounding one label alone was correct 5 times in 6 with a
        6 mm median error; adding the destination dropped that to 3 in 6, and
        adding the output of :meth:`list_objects` to 2 in 5. The reason is that
        those generic descriptions -- "blue can", "red can" -- alias the target,
        and forbidding two names from sharing a box then pushes the target onto
        the wrong object.         Passing a single label is the accurate default.

        ``cache=False`` bypasses the response cache. Policies should leave it
        alone -- a replayed episode should see identical perception -- but
        measuring how repeatable a grounding is needs it, since otherwise every
        repeat returns the first draw.
        """
        from racap.backends.vlm import DEFAULT_GROUNDER, detect_many

        obs = self.observe()
        rgb, depth, intr, extr = self._camera(obs, wrist=close_view)
        model = grounder_model or DEFAULT_GROUNDER
        boxes = detect_many(rgb, list(labels), model=model, cache=cache)

        results: dict[str, GroundedTarget] = {}
        for label in labels:
            box = boxes.get(label)
            if box is None:
                results[label] = GroundedTarget(
                    label=label,
                    pose=(0.0, 0.0, 0.0),
                    confidence=0.0,
                    metadata={
                        "reason": "not_in_joint_detection",
                        "detector": "vlm_joint_sam3",
                        "grounder": model,
                    },
                )
                continue
            crop = np.asarray(rgb)[box.y0 : box.y1, box.x0 : box.x1]
            sub_mask, score, info = self._mask_sam3_text(crop, label)
            mask = np.zeros(np.asarray(rgb).shape[:2], dtype=bool)
            if sub_mask is None:
                mask[box.y0 : box.y1, box.x0 : box.x1] = True
                score, info = box.confidence * 0.6, {"fallback": "bbox_as_mask"}
            else:
                mask[box.y0 : box.y1, box.x0 : box.x1] = sub_mask
            results[label] = self._target_from_mask(
                label,
                mask,
                depth,
                intr,
                extr,
                score,
                {
                    **info,
                    "detector": "vlm_joint_sam3",
                    "grounder": model,
                    "bbox": [box.x0, box.y0, box.x1, box.y1],
                    "from_wrist": close_view,
                },
            )
        return results

    def localize_with_feedback(
        self,
        label: str,
        rejected: list[GroundedTarget],
        *,
        grounder_model: str | None = None,
    ) -> GroundedTarget:
        """Re-ground a referent while visually excluding verified mistakes.

        Exclusions come only from independent crop-level identity checks. The
        full-scene VLM still interprets the original phrase; SAM3 then refines
        its revised box into metric geometry.
        """
        from racap.backends.vlm import BBox, DEFAULT_GROUNDER, bbox_with_feedback

        obs = self.observe()
        rgb, depth, intr, extr = self._camera(obs, wrist=False)
        old_boxes: list[BBox] = []
        for target in rejected:
            values = target.metadata.get("bbox")
            if not values or len(values) < 4:
                continue
            old_boxes.append(
                BBox(
                    *(int(value) for value in values[:4]),
                    float(target.confidence),
                )
            )
        if not old_boxes:
            return self.localize(
                label,
                detector="vlm_bbox_sam3",
                grounder_model=grounder_model,
            )

        model = grounder_model or DEFAULT_GROUNDER
        box = bbox_with_feedback(rgb, label, old_boxes, model=model)
        rejected_boxes = [[b.x0, b.y0, b.x1, b.y1] for b in old_boxes]
        if box is None:
            return GroundedTarget(
                label=label,
                pose=(0.0, 0.0, 0.0),
                confidence=0.0,
                metadata={
                    "reason": "vlm_no_revised_bbox",
                    "detector": "vlm_feedback_sam3",
                    "grounder": model,
                    "rejected_bboxes": rejected_boxes,
                },
            )

        crop = np.asarray(rgb)[box.y0 : box.y1, box.x0 : box.x1]
        sub_mask, score, info = self._mask_sam3_text(crop, label)
        mask = np.zeros(np.asarray(rgb).shape[:2], dtype=bool)
        if sub_mask is None:
            mask[box.y0 : box.y1, box.x0 : box.x1] = True
            score, info = box.confidence * 0.6, {"fallback": "bbox_as_mask"}
        else:
            mask[box.y0 : box.y1, box.x0 : box.x1] = sub_mask
        return self._target_from_mask(
            label,
            mask,
            depth,
            intr,
            extr,
            score,
            {
                **info,
                "detector": "vlm_feedback_sam3",
                "grounder": model,
                "bbox": [box.x0, box.y0, box.x1, box.y1],
                "rejected_bboxes": rejected_boxes,
                "from_wrist": False,
            },
        )

    def object_points(self, label: str) -> np.ndarray:
        """World-frame point cloud from the most recent localize of ``label``."""
        cached = self._points_cache.get(label)
        if cached is None:
            self.localize(label)
            cached = self._points_cache.get(label)
        return np.zeros((0, 3)) if cached is None else np.asarray(cached)

    # ------------------------------------------------------------- measurement

    def probe_pixel(self, u: int, v: int, *, wrist: bool = False) -> tuple[float, float, float]:
        """Deproject one pixel to a world point.

        Lets a runtime agent ask "what is actually at this spot" and correct a
        placement from the depth image rather than from a label it may have
        grounded wrongly.
        """
        rgb, depth, intr, extr = self._camera(self.observe(), wrist=wrist)
        height, width = np.asarray(depth).shape[:2]
        u = int(np.clip(u, 0, width - 1))
        v = int(np.clip(v, 0, height - 1))
        z = float(np.asarray(depth)[v, u])
        point = self._fn["pixel_to_world_point"](u, v, z, intr, extr)
        return tuple(float(c) for c in np.asarray(point).reshape(3))

    def probe_pixels(
        self, pixels: list[tuple[int, int]], *, wrist: bool = False
    ) -> list[tuple[float, float, float]]:
        """Deproject many pixels from a single frame.

        :meth:`probe_pixel` re-renders for every point, so asking about the ring
        of table around an object -- the cheapest way to find out how high the
        surface it stands on is -- cost sixteen renders. Here it costs one.
        """
        rgb, depth, intr, extr = self._camera(self.observe(), wrist=wrist)
        depth = np.asarray(depth)
        height, width = depth.shape[:2]
        out: list[tuple[float, float, float]] = []
        for u, v in pixels:
            u = int(np.clip(int(u), 0, width - 1))
            v = int(np.clip(int(v), 0, height - 1))
            point = self._fn["pixel_to_world_point"](u, v, float(depth[v, u]), intr, extr)
            out.append(tuple(float(c) for c in np.asarray(point).reshape(3)))
        return out

    def probe_bbox(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        *,
        wrist: bool = False,
    ) -> dict[str, Any]:
        """Dense world points for a rectangular RGB-D crop.

        Container openings are areas, not isolated probe points.  Returning a
        point at every crop pixel preserves the 2-D adjacency needed to find a
        cavity boundary and costs one render / one vectorised deprojection.
        Invalid depth stays as NaN so ``points[v, u]`` always corresponds to
        the same pixel in ``pixels[v, u]``.
        """
        _, depth, intr, extr = self._camera(self.observe(), wrist=wrist)
        depth = np.asarray(depth, dtype=float)
        height, width = depth.shape[:2]
        x0, y0, x1, y1 = (int(v) for v in bbox)
        x0, x1 = sorted((int(np.clip(x0, 0, width)), int(np.clip(x1, 0, width))))
        y0, y1 = sorted((int(np.clip(y0, 0, height)), int(np.clip(y1, 0, height))))
        if x1 <= x0 or y1 <= y0:
            return {
                "bbox": [x0, y0, x1, y1],
                "points": np.zeros((0, 0, 3), dtype=float),
                "pixels": np.zeros((0, 0, 2), dtype=int),
            }

        ys, xs = np.mgrid[y0:y1, x0:x1]
        z = depth[y0:y1, x0:x1]
        valid = np.isfinite(z) & (z > 0.0)
        fx, fy = float(intr[0, 0]), float(intr[1, 1])
        cx, cy = float(intr[0, 2]), float(intr[1, 2])
        camera = np.stack(
            [
                (xs - cx) * z / fx,
                (ys - cy) * z / fy,
                z,
                np.ones_like(z),
            ],
            axis=-1,
        )
        world = camera @ np.asarray(extr, dtype=float).T
        points = np.asarray(world[..., :3], dtype=float)
        points[~valid] = np.nan
        return {
            "bbox": [x0, y0, x1, y1],
            "points": points,
            "pixels": np.stack([xs, ys], axis=-1),
        }

    def image_axes(self) -> dict[str, list[float]]:
        """World directions of "right" and "down" in the scene image.

        A destination named by a relation -- *to the right of the plate* -- is
        relative to whoever is looking, and the only viewpoint the instruction
        can mean is the one the scene is shown from. Converting that into a
        world offset needs the camera's axes, which a policy cannot read
        directly and should not have to hard-code per suite. Measured by
        deprojecting three pixels of the same frame, so it stays correct if the
        camera moves.
        """
        rgb, depth, intr, extr = self._camera(self.observe(), wrist=False)
        depth = np.asarray(depth, dtype=float)
        h, w = depth.shape[:2]
        span = max(8, w // 8)

        # These points describe camera directions, not scene geometry.  Using
        # each pixel's observed depth mixes the desired image-plane direction
        # with jumps between foreground objects and the table: in one clean
        # relation scene that rotated image-right by 29 degrees and displaced
        # the target six centimetres.  Back-project at one representative
        # depth so only the camera calibration changes between pixels.
        finite = depth[np.isfinite(depth) & (depth > 0.0)]
        shared_depth = float(np.median(finite)) if finite.size else 1.0
        pixels = (
            (w // 2, h // 2),
            (w // 2 + span, h // 2),
            (w // 2, h // 2 + span),
        )
        origin, right, down = (
            np.asarray(
                self._fn["pixel_to_world_point"](u, v, shared_depth, intr, extr), dtype=float
            ).reshape(3)
            for u, v in pixels
        )

        def unit(a, b):
            delta = np.asarray(b, dtype=float)[:2] - np.asarray(a, dtype=float)[:2]
            norm = float(np.linalg.norm(delta))
            return (delta / norm).tolist() if norm > 1e-6 else [0.0, 0.0]

        return {"right": unit(origin, right), "down": unit(origin, down)}

    def measure(self, label_a: str, label_b: str, **kwargs) -> dict[str, Any]:
        """Distance between two grounded labels, for runtime self-correction."""
        a = self.localize(label_a, **kwargs)
        b = self.localize(label_b, **kwargs)
        delta = np.asarray(b.pose) - np.asarray(a.pose)
        return {
            "from": label_a,
            "to": label_b,
            "delta": [float(d) for d in delta],
            "distance": float(np.linalg.norm(delta)),
            "horizontal_distance": float(np.linalg.norm(delta[:2])),
            "height_difference": float(delta[2]),
            "confidence": float(min(a.confidence, b.confidence)),
        }

    # ------------------------------------------------------------------ motion

    def open_gripper(self) -> None:
        self._fn["open_gripper"]()

    def close_gripper(self) -> None:
        self._fn["close_gripper"]()

    def ee_pose(self) -> dict[str, Any]:
        """Current end-effector position, orientation and gripper opening."""
        state = np.asarray(self.observe()["robot_cartesian_pos"], dtype=float)
        return {
            "position": [float(v) for v in state[:3]],
            "quat": [float(v) for v in state[3:7]],
            "gripper_opening": float(state[7]),
        }

    def move_to(
        self,
        position: Any,
        quat: Any = None,
        *,
        gripper: float = 0.0,
        yaw: float | None = None,
    ) -> None:
        """Move the end effector to a world-frame pose.

        ``gripper`` is the commanded finger action, 0.0 to hold the current
        state. Orientation defaults to the last grasp orientation so a carried
        object is not rotated mid-transfer.

        ``yaw`` asks for a straight-down hand turned by that angle about the
        world z axis, which is the only orientation change a top-down pick and
        place needs and the only one a policy can express without building a
        quaternion by hand. It matters when the destination is a slot: a book
        held across its faces goes into a caddy compartment only if the jaws
        are square to the slot.
        """
        if yaw is not None:
            quat = _yaw_quat(float(yaw))
        target_quat = self._last_grasp_quat if quat is None else np.asarray(quat, dtype=np.float64)
        if yaw is not None:
            self._last_grasp_quat = target_quat
        self._fn["goto_pose"](np.asarray(position, dtype=np.float64), target_quat, gripper)

    def delta_move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """Move by a world-frame offset in meters, relative to where we are now.

        The semantic move a ReAct agent reaches for when it can see it is a few
        centimeters off: ``delta_move(dy=0.04)`` rather than a recomputed
        absolute pose.
        """
        current = np.asarray(self.ee_pose()["position"], dtype=np.float64)
        self.move_to(current + np.array([dx, dy, dz], dtype=np.float64))

    def go_home(self) -> None:
        """Return the arm to LIBERO's neutral joint pose without resetting state."""
        self._fn["goto_home_joint_position"]()
        self._last_grasp_quat = TOP_DOWN_QUAT.copy()

    def grasp(self, target: GroundedTarget, *, strategy: str) -> None:
        """Close on ``target`` using one of several contact strategies.

        - ``top_down``: fixed straight-down hand at the grounded centroid.
          Robust default for upright bottles and boxes.
        - ``rim``: grasp the outer wall of the perceived cloud. The right
          contact for bowls, mugs, and other open containers whose centroid is
          empty air; also a recovery for short objects the arm cannot reach at
          their centre but can reach a few centimetres nearer the base.
        - ``affordance``: a VLM points at a graspable pixel on the object; that
          pixel is deprojected and grasped top-down. Useful when the geometry
          is irregular and neither the centroid nor a rim heuristic fits, and
          a fallback when Contact-GraspNet returns nothing.
        - ``graspnet``: Contact-GraspNet proposes a 6-DoF pose from the depth
          image and mask. Can fail silently on thin or poorly observed objects.
        - ``pca_axis``: fit the object's footprint and close across its short
          axis. Cheap, no network call, usually right for elongated objects.
        - ``wrist_closeloop``: hover, re-ground from the wrist camera, correct,
          then close. Costs one extra grounding call.
        """
        if strategy == "wrist_closeloop":
            return self._grasp_wrist_closeloop(target)

        self.contact_grasp(target, strategy=strategy, lift=LIFT_HEIGHT)

    def contact_grasp(
        self, target: GroundedTarget, *, strategy: str, lift: float = 0.0
    ) -> dict[str, Any] | None:
        """Close on a target while keeping contact with its mechanism.

        Transport grasps deliberately lift by 15 cm.  That is exactly the
        wrong primitive for a drawer handle, hinged-door handle, or rotary
        control: lifting either loses the contact or applies an uncontrolled
        force to the joint.  This variant shares the same visually grounded
        contact planner but makes post-contact lift an explicit, bounded
        parameter.  It exposes no simulator joint state.
        """
        if strategy == "wrist_closeloop":
            # Wrist closed-loop grasp includes its own transport lift and is
            # therefore not an admissible mechanism-contact strategy.
            strategy = "top_down"

        origin = np.asarray(target.pose, dtype=np.float64)
        self._grasp_origin[target.label] = origin.copy()

        if strategy in {"side_bar", "graspgen_6d", "graspnet_6d"}:
            if strategy == "side_bar":
                axis = np.asarray(
                    target.metadata.get("contact_axis") or [1.0, 0.0],
                    dtype=np.float64,
                )
                side_pose = canonical_side_bar_grasp(target.pose, axis)
                learned = None
                if side_pose is not None:
                    position, quat = self._fn["decompose_transform"](side_pose)
                    learned = (
                        np.asarray(position, dtype=np.float64).reshape(3),
                        np.asarray(quat, dtype=np.float64).reshape(4),
                        np.asarray(side_pose[:3, 2], dtype=np.float64),
                    )
                learned_source = "side_bar_6d"
            else:
                learned = self._graspgen_pose(target) if strategy == "graspgen_6d" else None
                learned_source = "graspgen_6d"
            if learned is None:
                learned = self._graspnet_pose(target, side_contact=True)
                learned_source = "contact_graspnet_6d"
            if learned is not None:
                position, quat, approach = learned
                self._last_grasp_quat = quat
                # Contact-GraspNet's local +Z is the approach direction.  The
                # pre-contact retreat must follow that local axis, not world Z.
                precontact = position - approach * PREGRASP_APPROACH
                self._fn["goto_pose"](precontact, quat, 0.0)
                self._fn["goto_pose"](position, quat, 0.0)
                reached = self.ee_pose()
                reached_position = np.asarray(
                    reached.get("position", position),
                    dtype=np.float64,
                ).reshape(3)
                pose_error = float(np.linalg.norm(reached_position - position))
                self._fn["close_gripper"]()
                bounded_lift = float(np.clip(float(lift), 0.0, LIFT_HEIGHT))
                if bounded_lift > 1e-4:
                    self._fn["goto_pose"](
                        position + np.asarray([0.0, 0.0, bounded_lift]), quat, 0.0
                    )
                return {
                    "source": learned_source,
                    "position": [round(float(v), 4) for v in position],
                    "approach": [round(float(v), 4) for v in approach],
                    "reached_position": [round(float(v), 4) for v in reached_position],
                    "pose_error_cm": round(pose_error * 100.0, 1),
                    **(
                        {"graspgen_fallback": dict(getattr(self, "_last_graspgen_diagnostic", {}))}
                        if strategy == "graspgen_6d" and learned_source != "graspgen_6d"
                        else {}
                    ),
                }
            # A learned proposal is optional.  Keep the recovery safe and
            # deterministic when the thin mask yields no admissible candidate.
            strategy = "pca_axis"

        if strategy == "scoop_edge":
            # A flush package cannot be pinched by descending and closing in
            # place: the low finger hits the support while the high finger
            # remains above the object.  Approach from outside its short edge
            # with open jaws and sweep the low tilted finger underneath before
            # closing.  Distances are footprint-relative and object-agnostic.
            axis = np.asarray(
                target.metadata.get("principal_axis") or [1.0, 0.0],
                dtype=np.float64,
            )
            theta = float((np.arctan2(axis[1], axis[0]) + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
            short = np.asarray([np.sin(theta), -np.cos(theta)], dtype=np.float64)
            centre = np.asarray(target.pose[:2], dtype=np.float64)
            extent = np.asarray(
                target.metadata.get("extent") or [0.06, 0.03, 0.015],
                dtype=np.float64,
            )
            start_clearance = float(
                np.clip(
                    0.5 * min(abs(float(extent[0])), abs(float(extent[1]))) + 0.020,
                    0.032,
                    0.050,
                )
            )
            start_xy = centre + short * start_clearance
            top_z = float(target.metadata.get("top_z", target.pose[2]))
            contact_z = top_z + 0.012
            quat = _tilted_pca_quat(axis, angle_deg=-35.0)
            self._last_grasp_quat = quat
            self._fn["goto_pose"](
                np.asarray([start_xy[0], start_xy[1], contact_z + PREGRASP_APPROACH]),
                quat,
                0.0,
            )
            self._fn["goto_pose"](np.asarray([start_xy[0], start_xy[1], contact_z]), quat, 0.0)
            # Lift slightly during the inward sweep so the fingertip levers
            # rather than merely translating the package across the support.
            self._fn["goto_pose"](np.asarray([centre[0], centre[1], contact_z + 0.010]), quat, 0.0)
            self._fn["close_gripper"]()
            bounded_lift = float(np.clip(float(lift), 0.0, LIFT_HEIGHT))
            if bounded_lift > 1e-4:
                self._fn["goto_pose"](
                    np.asarray(
                        [
                            centre[0],
                            centre[1],
                            contact_z + 0.010 + bounded_lift,
                        ]
                    ),
                    quat,
                    0.0,
                )
            return {
                "source": "open_jaw_edge_scoop",
                "position": [round(float(v), 4) for v in (centre[0], centre[1], contact_z)],
                "start_clearance_m": round(start_clearance, 4),
            }

        xy, quat, grasp_z = self._grasp_contact(target, strategy)
        self._last_grasp_quat = quat

        approach = np.array([xy[0], xy[1], grasp_z + PREGRASP_APPROACH])
        contact = np.array([xy[0], xy[1], grasp_z])
        self._fn["goto_pose"](approach, quat, 0.0)
        self._fn["goto_pose"](contact, quat, 0.0)
        reached = self.ee_pose()
        reached_position = np.asarray(reached.get("position", contact), dtype=np.float64).reshape(3)
        pose_error = float(np.linalg.norm(reached_position - contact))
        self._fn["close_gripper"]()
        bounded_lift = float(np.clip(float(lift), 0.0, LIFT_HEIGHT))
        if bounded_lift > 1e-4:
            self._fn["goto_pose"](np.array([xy[0], xy[1], grasp_z + bounded_lift]), quat, 0.0)
        return {
            "source": strategy,
            "position": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(grasp_z), 4)],
            "approach": [0.0, 0.0, -1.0],
            "reached_position": [round(float(value), 4) for value in reached_position],
            "pose_error_cm": round(pose_error * 100.0, 1),
        }

    def _grasp_contact(
        self,
        target: GroundedTarget,
        strategy: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Resolve ``(xy, quat, grasp_z)`` for the named strategy.

        Every strategy that needs the cloud reads it from the most recent
        localize of this label. Strategies that cannot produce a contact fall
        back to the grounded centroid with a top-down hand, so a policy that
        asks for ``rim`` on a solid box still closes somewhere rather than
        raising.
        """
        position = np.asarray(target.pose, dtype=np.float64)
        points = self.object_points(target.label)
        cloud = np.asarray(points).reshape(-1, 3) if points is not None else np.zeros((0, 3))
        cloud = cloud[np.isfinite(cloud).all(axis=1)] if cloud.size else cloud
        # SAM masks occasionally contain a thin background/depth spur.  The
        # grounded OBB centre and extent remain stable while max-Z and a rim
        # selector jump to that spur.  Restrict contact planning to a generous
        # OBB neighbourhood; retain the raw cloud if the gate would leave too
        # little support.  This changes neither semantic grounding nor any
        # privileged state boundary.
        if cloud.shape[0] >= MIN_MASK_PIXELS:
            extent = np.asarray(
                target.metadata.get("extent") or [0.0, 0.0, 0.0],
                dtype=np.float64,
            )
            limits = np.maximum(0.035, 0.75 * np.abs(extent) + 0.012)
            nearby = cloud[np.all(np.abs(cloud - position[None, :]) <= limits[None, :], axis=1)]
            if nearby.shape[0] >= MIN_MASK_PIXELS:
                cloud = nearby
        cloud_top = (
            float(cloud[:, 2].max())
            if cloud.shape[0]
            else float(target.metadata.get("top_z", position[2]))
        )

        if strategy == "rim":
            rim = _rim_contact(cloud) if cloud.shape[0] else None
            if rim is not None:
                xy, yaw, local_top = rim
                return (
                    np.asarray(xy, dtype=np.float64),
                    _yaw_quat(yaw),
                    _close_height(target, local_top, cloud_top),
                )

        if strategy == "affordance":
            hit = self._affordance_contact(target)
            if hit is not None:
                xy, local_top = hit
                return (
                    np.asarray(xy, dtype=np.float64),
                    TOP_DOWN_QUAT.copy(),
                    _close_height(target, local_top, cloud_top),
                )

        if strategy == "graspnet":
            learned = self._graspnet_pose(target, side_contact=False)
            if learned is not None:
                learned_position, quat, _ = learned
                return learned_position[:2].copy(), quat, float(learned_position[2])
            quat = TOP_DOWN_QUAT.copy()
        elif strategy in {"tilted_edge", "tilted_edge_reverse"}:
            axis = target.metadata.get("principal_axis") or [1.0, 0.0]
            quat = _tilted_pca_quat(
                axis,
                # 25 and 35 degrees both caught the flush edge; 35 retained a
                # measurably wider, less marginal finger opening during lift.
                angle_deg=(35.0 if strategy == "tilted_edge" else -35.0),
            )
        elif strategy == "pca_axis":
            axis = target.metadata.get("principal_axis") or [1.0, 0.0]
            quat = _yaw_quat(_pca_grasp_yaw(axis))
        else:
            quat = TOP_DOWN_QUAT.copy()

        grasp_z = float(target.metadata.get("top_z", cloud_top))
        return position[:2].copy(), quat, grasp_z

    def _affordance_contact(
        self,
        target: GroundedTarget,
    ) -> tuple[np.ndarray, float] | None:
        """VLM grasp pixel → world xy and a local top height.

        The point itself is often on the visible surface; the closing height is
        taken from nearby points of the object's cloud so a single bad depth
        sample at that pixel does not send the hand into the table.
        """
        from racap.backends.vlm import DEFAULT_GROUNDER, grasp_point

        try:
            obs = self.observe()
            rgb, depth, intr, extr = self._camera(obs, wrist=False)
            hit = grasp_point(np.asarray(rgb), target.label, model=DEFAULT_GROUNDER)
            if hit is None:
                return None
            u, v, _ = hit
            world = np.asarray(
                self._fn["pixel_to_world_point"](u, v, float(np.asarray(depth)[v, u]), intr, extr),
                dtype=np.float64,
            ).reshape(3)
            if not np.isfinite(world).all():
                return None

            points = np.asarray(self.object_points(target.label)).reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)] if points.size else points
            if points.shape[0]:
                near = points[np.linalg.norm(points[:, :2] - world[:2], axis=1) <= 0.03]
                local_top = float(near[:, 2].max()) if near.size else float(points[:, 2].max())
            else:
                local_top = float(world[2])
            return world[:2].copy(), local_top
        except Exception:
            return None

    def _grasp_wrist_closeloop(self, target: GroundedTarget) -> None:
        """Approach from above, re-ground up close, correct, then close.

        The scene camera sees a 25 mm can across a handful of pixels, so a
        centimetre of depth noise becomes a centimetre of grasp error. Hovering
        first and looking again from the wrist puts the object across a large
        part of the frame, where the same relative error is worth far less.

        The correction is capped: a large disagreement between the two views
        usually means the wrist camera grounded a different object, and moving
        to it would turn a near miss into a grasp of the wrong thing.
        """
        position = np.asarray(target.pose, dtype=np.float64)
        quat = TOP_DOWN_QUAT.copy()
        self._last_grasp_quat = quat
        self._grasp_origin[target.label] = position.copy()
        top_z = float(target.metadata.get("top_z", position[2]))

        hover = np.array([position[0], position[1], top_z + WRIST_HOVER])
        self._fn["goto_pose"](hover, quat, 0.0)

        refined = position
        try:
            close = self.localize(target.label, close_view=True, detector="vlm_bbox_sam3")
            if close.confidence > 0:
                candidate = np.asarray(close.pose, dtype=np.float64)
                shift = float(np.linalg.norm(candidate[:2] - position[:2]))
                if shift <= WRIST_MAX_CORRECTION:
                    refined = candidate
                    top_z = float(close.metadata.get("top_z", top_z))
        except Exception:
            pass

        contact = np.array([refined[0], refined[1], top_z])
        self._fn["goto_pose"](
            np.array([refined[0], refined[1], top_z + PREGRASP_APPROACH]), quat, 0.0
        )
        self._fn["goto_pose"](contact, quat, 0.0)
        self._fn["close_gripper"]()
        self._fn["goto_pose"](np.array([refined[0], refined[1], top_z + LIFT_HEIGHT]), quat, 0.0)

    def _graspnet_pose(
        self,
        target: GroundedTarget,
        *,
        side_contact: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return a world-frame learned contact pose and approach vector.

        The old implementation accidentally omitted camera extrinsics when it
        called RATs' selector.  Its exception handler therefore reduced every
        nominal ``graspnet`` action to a top-down quaternion.  Keeping position,
        orientation and approach together both fixes transport graspnet and
        enables a genuine 6-DoF mechanism contact recovery.
        """
        try:
            obs = self.observe()
            rgb, depth, intr, extr = self._camera(obs, wrist=False)
            results = self._fn["segment_sam3_text_prompt"](rgb, target.label)
            if not results:
                return None
            chosen_mask = max(results, key=lambda value: float(value.get("score", 0.0)))
            mask = np.asarray(chosen_mask["mask"]).astype(bool)
            poses, scores = self._fn["plan_grasp"](depth, intr, mask)
            if poses is None or len(poses) == 0:
                return None
            if side_contact:
                axis = target.metadata.get("contact_axis") or [1.0, 0.0]
                chosen, _ = select_mechanism_grasp(poses, scores, extr, axis)
            else:
                chosen, _ = self._fn["select_top_down_grasp"](poses, scores, extr)
            if chosen is None:
                return None
            position, quat = self._fn["decompose_transform"](chosen)
            position = np.asarray(position, dtype=np.float64).reshape(3)
            quat = np.asarray(quat, dtype=np.float64).reshape(4)
            approach = np.asarray(chosen[:3, 2], dtype=np.float64).reshape(3)
            # Reject a segmentation/candidate mismatch before moving the arm.
            # A mechanism mask often contains much of the drawer face; its
            # learned high-score grasp may be several centimetres away from
            # the compact protruding handle. Transport masks retain the broad
            # legacy tolerance, while side contact must stay within the
            # observed handle-sized envelope.
            if side_contact:
                extent = np.asarray(
                    target.metadata.get("extent") or [0.04, 0.04, 0.04],
                    dtype=float,
                )
                max_distance = float(
                    np.clip(
                        0.5 * max(abs(float(v)) for v in extent[:2]) + 0.015,
                        0.035,
                        0.08,
                    )
                )
            else:
                max_distance = 0.20
            if float(np.linalg.norm(position - np.asarray(target.pose))) > max_distance:
                return None
            return position, quat, approach
        except Exception:
            return None

    def _graspgen_pose(
        self,
        target: GroundedTarget,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Generate and task-filter a GraspGen mechanism contact pose.

        GraspGen supplies grasp quality and collision-aware candidates.  The
        articulation controller supplies the missing task constraint: the
        approach must be horizontal and aligned with the visually measured
        mechanism normal so that the grasp can transmit the requested pull.
        """
        diagnostic: dict[str, Any] = {"enabled": False}
        self._last_graspgen_diagnostic = diagnostic
        try:
            from racap.backends.graspgen import infer_grasps

            refined_points = target.metadata.get("contact_points")
            points = np.asarray(
                refined_points if refined_points is not None else self.object_points(target.label),
                dtype=float,
            ).reshape(-1, 3)
            diagnostic.update({"enabled": True, "point_count": int(len(points))})
            predicted = infer_grasps(points)
            if predicted is None:
                diagnostic["reason"] = "no_server_result"
                return None
            poses, scores = predicted
            diagnostic["candidate_count"] = int(len(poses))
            axis = np.asarray(
                target.metadata.get("contact_axis") or [1.0, 0.0],
                dtype=np.float64,
            )
            chosen, selected_score = select_mechanism_grasp(
                poses,
                scores,
                np.eye(4, dtype=float),
                axis,
            )
            if chosen is None:
                diagnostic["reason"] = "no_axis_compatible_candidate"
                return None
            diagnostic["selected_score"] = round(float(selected_score), 4)
            position, quat = self._fn["decompose_transform"](chosen)
            position = np.asarray(position, dtype=np.float64).reshape(3)
            quat = np.asarray(quat, dtype=np.float64).reshape(4)
            approach = np.asarray(chosen[:3, 2], dtype=np.float64).reshape(3)
            # The semantic mask can include a broad drawer face.  GraspGen is
            # allowed to choose anywhere on the handle cloud, but not a pose
            # that has left its observed physical envelope altogether.
            extent = np.asarray(
                target.metadata.get("extent") or [0.04, 0.04, 0.04],
                dtype=float,
            )
            limit = float(np.clip(0.75 * np.linalg.norm(extent) + 0.015, 0.035, 0.12))
            contact_point = position + approach * GRASPGEN_FRANKA_CONTACT_OFFSET_M
            wrist_distance = float(np.linalg.norm(position - np.asarray(target.pose)))
            distance = float(np.linalg.norm(contact_point - np.asarray(target.pose)))
            diagnostic.update(
                {
                    "wrist_distance_cm": round(wrist_distance * 100, 1),
                    "contact_distance_cm": round(distance * 100, 1),
                    "max_distance_cm": round(limit * 100, 1),
                    "predicted_contact": [round(float(v), 4) for v in contact_point],
                }
            )
            if distance > limit:
                diagnostic["reason"] = "candidate_outside_handle_envelope"
                return None
            diagnostic["reason"] = "accepted"
            return position, quat, approach
        except Exception as exc:
            diagnostic["reason"] = f"exception:{type(exc).__name__}"
            return None

    def _graspnet_quat(self, target: GroundedTarget) -> np.ndarray:
        """Compatibility helper for callers that only need top-down yaw."""
        learned = self._graspnet_pose(target, side_contact=False)
        return learned[1] if learned is not None else TOP_DOWN_QUAT.copy()

    def verify_grasp(self, label: str) -> bool:
        """A held object keeps the fingers apart; an empty close bottoms out."""
        ee = self.observe()["robot_cartesian_pos"]
        opening = float(ee[7])
        return GRIPPER_HOLDING_MIN < opening < GRIPPER_HOLDING_MAX

    def place(self, target: GroundedTarget, *, strategy: str) -> None:
        center = np.asarray(target.pose, dtype=np.float64)
        # Release relative to the destination's top surface, not its centroid,
        # so that a tall container and a flat plate both get a sane drop height.
        base_z = float(target.metadata.get("top_z", center[2]))
        quat = self._last_grasp_quat

        for height in (HOVER_HEIGHT, RELEASE_HEIGHT):
            self._fn["goto_pose"](np.array([center[0], center[1], base_z + height]), quat, 0.0)
        self._fn["open_gripper"]()
        self._fn["goto_pose"](np.array([center[0], center[1], base_z + RETREAT_HEIGHT]), quat, 0.0)

    def verify_place(self, pick_label: str, place_label: str) -> bool:
        """Perception-only check that the object now sits over the destination.

        This is deliberately not the native predicate. Native success is
        recorded separately through :class:`LiberoOracle` so that a policy can
        never be tuned against the benchmark's own answer key.

        Looking for the object where it landed does not work, and no choice of
        detector fixes it. An object dropped into a basket is behind the
        basket's walls: on a placement the benchmark scored as a success, the
        object's true position was 4 mm from the basket's, while grounding it
        returned a spurious detection half a metre away at confidence 0.08. The
        earlier version reported failure on four of five real successes, which
        would teach a downstream agent to undo work already done.

        So this verifies what remains visible. The destination grounds reliably
        even when full. If the object can be seen over it, that settles it.
        Otherwise the object having left the place it was picked from, with an
        empty gripper, means it was released -- and :meth:`place` releases
        nowhere but over the destination.
        """
        placed = self.localize_many([place_label]).get(place_label)
        if placed is None or placed.confidence <= 0.0:
            return False
        extent = placed.metadata.get("extent") or [0.1, 0.1, 0.1]
        radius = max(0.5 * max(float(extent[0]), float(extent[1])), 0.05)

        picked = self.localize_many([pick_label]).get(pick_label)
        visible = (
            picked is not None
            and picked.confidence > 0.0
            and int(picked.metadata.get("n_points", 0)) >= MIN_MASK_PIXELS
        )
        if visible:
            horizontal = float(
                np.linalg.norm(np.asarray(picked.pose[:2]) - np.asarray(placed.pose[:2]))
            )
            if horizontal <= radius:
                return True

        origin = self._grasp_origin.get(pick_label)
        if origin is None or self.verify_grasp(pick_label):
            return False
        if not visible:
            # Nowhere to be seen, and the hand is empty: it is inside something.
            return True
        moved = float(np.linalg.norm(np.asarray(picked.pose[:2]) - origin[:2]))
        return moved > radius

    def recover(self) -> None:
        try:
            self._fn["open_gripper"]()
            self._fn["goto_home_joint_position"]()
        except Exception:
            pass


def make_runtime_from_env() -> LiberoPrimitiveRuntime:
    """Build a runtime from RACAP_SUITE / RACAP_TASK / RACAP_SEED."""
    return LiberoPrimitiveRuntime(
        suite_name=os.environ.get("RACAP_SUITE", "libero_object"),
        task_id=int(os.environ.get("RACAP_TASK", "0")),
        seed=int(os.environ.get("RACAP_SEED", "0")),
    )
