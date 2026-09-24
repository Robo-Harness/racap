"""Real MolmoSpaces bridge wrapping the molmo_spaces MuJoCo simulator.

Provides the same call surface as MockMolmoSpacesBridge but backed by the real
molmo_spaces CPUMujocoEnv, FrankaRobot, and task infrastructure. The RATS
code-execution layer calls into this bridge for physics stepping, rendering,
reward, and success evaluation.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

logger = logging.getLogger("rats.molmospaces_bridge")

_THIRD_PARTY_ROOT = Path(__file__).resolve().parent.parent.parent / "third_party" / "molmospaces"
if _THIRD_PARTY_ROOT.is_dir() and str(_THIRD_PARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY_ROOT))

_MOLMO_SPACES_IMPORT_ERROR: ModuleNotFoundError | None = None


class _MolmoSpacesUnavailable:
    """Placeholder used so module-level helpers can be imported without the submodule."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass


try:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.camera_configs import (
        CameraSystemConfig,
        EvalExocentricCameraConfig,
        FixedExocentricCameraConfig,
        FrankaEvalCameraSystem,
        FrankaRandomizedD405D455CameraSystem,
        MjcfCameraConfig,
        RobotMountedCameraConfig,
    )
    from molmo_spaces.configs.policy_configs import (
        BasePolicyConfig,
        OpenClosePlannerPolicyConfig,
        PickPlannerPolicyConfig,
    )
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.configs.task_configs import (
        BaseMujocoTaskConfig,
        OpeningTaskConfig,
        PickAndPlaceTaskConfig,
        PickTaskConfig,
    )
    from molmo_spaces.configs.task_sampler_configs import (
        BaseMujocoTaskSamplerConfig,
        OpenTaskSamplerConfig,
        PickAndPlaceTaskSamplerConfig,
        PickTaskSamplerConfig,
    )
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec, load_all_episodes
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
    from molmo_spaces.tasks.opening_task_samplers import OpenTaskSampler
    from molmo_spaces.tasks.opening_tasks import OpeningTask
    from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
    from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
    from molmo_spaces.tasks.pick_task import PickTask
    from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
    from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
except ModuleNotFoundError as exc:
    _MOLMO_SPACES_IMPORT_ERROR = exc
    MlSpacesExpConfig = _MolmoSpacesUnavailable
    CameraSystemConfig = _MolmoSpacesUnavailable
    EvalExocentricCameraConfig = _MolmoSpacesUnavailable
    FixedExocentricCameraConfig = _MolmoSpacesUnavailable

    # Distinct stub class so multiple-inheritance bases don't collide
    # when the submodule is missing (else: ``class X(A, B)`` with
    # ``A is B is _MolmoSpacesUnavailable`` raises TypeError at class
    # construction time even though X is never instantiated).
    class _FrankaEvalCameraSystemUnavailable(_MolmoSpacesUnavailable):
        pass

    FrankaEvalCameraSystem = _FrankaEvalCameraSystemUnavailable
    FrankaRandomizedD405D455CameraSystem = _MolmoSpacesUnavailable
    MjcfCameraConfig = _MolmoSpacesUnavailable
    RobotMountedCameraConfig = _MolmoSpacesUnavailable
    BasePolicyConfig = _MolmoSpacesUnavailable
    OpenClosePlannerPolicyConfig = _MolmoSpacesUnavailable
    PickPlannerPolicyConfig = _MolmoSpacesUnavailable
    FrankaRobotConfig = _MolmoSpacesUnavailable
    BaseMujocoTaskConfig = _MolmoSpacesUnavailable
    OpeningTaskConfig = _MolmoSpacesUnavailable
    PickAndPlaceTaskConfig = _MolmoSpacesUnavailable
    PickTaskConfig = _MolmoSpacesUnavailable
    BaseMujocoTaskSamplerConfig = _MolmoSpacesUnavailable
    OpenTaskSamplerConfig = _MolmoSpacesUnavailable
    PickAndPlaceTaskSamplerConfig = _MolmoSpacesUnavailable
    PickTaskSamplerConfig = _MolmoSpacesUnavailable
    EpisodeSpec = Any
    load_all_episodes = None
    JsonEvalTaskSampler = _MolmoSpacesUnavailable
    OpenTaskSampler = _MolmoSpacesUnavailable
    OpeningTask = _MolmoSpacesUnavailable
    PickAndPlaceTask = _MolmoSpacesUnavailable
    PickAndPlaceTaskSampler = _MolmoSpacesUnavailable
    PickTask = _MolmoSpacesUnavailable
    PickTaskSampler = _MolmoSpacesUnavailable
    BaseMujocoTaskSampler = _MolmoSpacesUnavailable

if TYPE_CHECKING:
    from molmo_spaces.env.env import CPUMujocoEnv
    from molmo_spaces.robots.franka import FrankaRobot
    from molmo_spaces.tasks.task import BaseMujocoTask


_TASK_TYPE_MAP: dict[str, dict[str, Any]] = {
    "pick": {
        "task_cls": PickTask,
        "sampler_cls": PickTaskSampler,
        "config_cls": PickTaskConfig,
        "sampler_config_cls": PickTaskSamplerConfig,
        "policy_config_cls": PickPlannerPolicyConfig,
    },
    "pick_and_place": {
        "task_cls": PickAndPlaceTask,
        "sampler_cls": PickAndPlaceTaskSampler,
        "config_cls": PickAndPlaceTaskConfig,
        "sampler_config_cls": PickAndPlaceTaskSamplerConfig,
        "policy_config_cls": PickPlannerPolicyConfig,
    },
    # Articulated-object opening. OpenTaskSampler defaults pickup_types to
    # EXTENDED_ARTICULATION_TYPES_THOR when None, which covers every asset
    # category with grasp files on disk (cabinet, drawer, oven, dishwasher,
    # showerdoor, Fridge, Microwave, Toilet, Doorway, Doorway_Double, Safe,
    # Dresser, Desk, Shelving_Unit, Side_Table, Coffee_Table, Laptop,
    # Laundry_Hamper). The bridge's ``pickup_types`` ctor arg overrides this.
    "open": {
        "task_cls": OpeningTask,
        "sampler_cls": OpenTaskSampler,
        "front_facing_sampler_cls": None,  # filled in after _FrontFacingOpenTaskSampler is defined
        "config_cls": OpeningTaskConfig,
        "sampler_config_cls": OpenTaskSamplerConfig,
        "policy_config_cls": OpenClosePlannerPolicyConfig,
        "task_config_overrides": {
            "task_success_threshold": 0.20,
            "any_inst_of_category": True,
        },
        "sampler_config_overrides": {
            "target_initial_state_open_percentage": 0.0,
        },
    },
    "close": {
        "task_cls": OpeningTask,
        "sampler_cls": OpenTaskSampler,
        "front_facing_sampler_cls": None,
        "config_cls": OpeningTaskConfig,
        "sampler_config_cls": OpenTaskSamplerConfig,
        "policy_config_cls": OpenClosePlannerPolicyConfig,
        "task_config_overrides": {
            "task_success_threshold": 0.85,
            "any_inst_of_category": False,
        },
        "sampler_config_overrides": {
            "target_initial_state_open_percentage": 0.5,
        },
    },
}


_ARTICULATED_TASK_TYPES = frozenset({"open", "close"})

# Cap for sequential house-index probing when ``scene_dataset`` is a
# procedural set (procthor-10k has 10k houses, holodeck-objaverse has 100k).
# We stop early to bound auto-discovery latency; pin ``house_index``
# explicitly if you need a specific far-out house.
_PROCGEN_AUTO_DISCOVER_LIMIT = 40


def _root_object_internal_name(internal_name: str) -> str:
    """Best-effort map a MolmoSpaces link/joint name to its root object name.

    Scene XML joint/body names commonly look like
    ``category_asset_instance_link_variant_extra`` while inventory roots use
    ``category_asset_instance_0_0``.  This lets benchmark/XML-derived
    articulation joints be merged back into the live scene inventory even when
    the MolmoSpaces object-manager articulation enumerator misses them.
    """
    parts = str(internal_name or "").split("_")
    for i in range(1, len(parts) - 2):
        if (
            parts[i].isdigit()
            and parts[i + 1].isdigit()
            and parts[i + 2].isdigit()
        ):
            return "_".join(parts[: i + 1] + ["0", "0"])
    return str(internal_name or "")


def _task_cls_to_family(task_cls: str) -> str:
    cls_name = task_cls.rsplit(".", 1)[-1]
    if cls_name.endswith("Task"):
        cls_name = cls_name[: -len("Task")]
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", cls_name).lower()


def _episode_task_family(task: dict[str, Any], task_cls: str) -> str:
    """Return the benchmark catalog family for one JSON episode.

    ``OpeningTask`` backs both open and close tasks.  Prefer the optional
    JSON ``task_type`` field when it is present so mixed benchmarks keep
    open and close episodes addressable as distinct canonical task families.
    """
    task_type = task.get("task_type")
    if isinstance(task_type, str) and task_type:
        return task_type
    return _task_cls_to_family(task_cls) if task_cls else "unknown"


def _extract_episode_objects(task: dict[str, Any]) -> list[str]:
    objects: list[str] = []
    for key in ("pickup_obj_name", "place_receptacle_name"):
        val = task.get(key)
        if isinstance(val, str) and val:
            objects.append(val)
    return objects


def _parse_canonical_task_id(canonical_task_id: str | None) -> dict[str, str] | None:
    if not canonical_task_id:
        return None
    parts = canonical_task_id.split(":")
    if len(parts) != 5 or parts[0] != "molmospaces":
        return None
    return {
        "benchmark": parts[1],
        "scene_family": parts[2],
        "task_family": parts[3],
        "variant": parts[4],
    }


def _sample_task_compat(sampler: Any, **kwargs: Any) -> Any:
    """Call MolmoSpaces sampler with kwargs supported by the installed version."""
    params = inspect.signature(sampler.sample_task).parameters
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    return sampler.sample_task(**kwargs)


def _ithor_scene_split_for_house_index(house_index: int) -> str:
    """Return the split used by upstream iTHOR scene indexing."""
    index_in_scene_type = int(house_index) % 100
    if index_in_scene_type <= 12:
        return "train"
    if index_in_scene_type <= 24:
        return "val"
    if index_in_scene_type <= 30:
        return "test"
    raise ValueError(
        f"Unknown iTHOR scene split for house_index={house_index} "
        f"(index_in_scene_type={index_in_scene_type})"
    )


def _normalize_benchmark_episode_scene_split(episode_spec: EpisodeSpec) -> EpisodeSpec:
    """Adapt benchmark metadata to upstream MolmoSpaces scene-index splits.

    Some RATS benchmark subsets label all held-out episodes as ``test`` for
    evaluation provenance, but upstream MolmoSpaces indexes iTHOR XML assets by
    FloorPlan number: 0-12 train, 13-24 val, 25-30 test within each room type.
    JsonEvalTaskSampler uses ``episode_spec.data_split`` directly for scene
    lookup, so normalize only the runtime scene lookup field here.
    """
    if getattr(episode_spec, "scene_dataset", None) != "ithor":
        return episode_spec

    scene_split = _ithor_scene_split_for_house_index(int(episode_spec.house_index))
    if getattr(episode_spec, "data_split", None) == scene_split:
        return episode_spec

    try:
        return episode_spec.model_copy(update={"data_split": scene_split})
    except AttributeError:
        import copy

        cloned = copy.copy(episode_spec)
        cloned.data_split = scene_split
        return cloned


@dataclass
class _RATSEvalRuntimeParams:
    """Lightweight runtime params placeholder.

    The upstream MlSpacesExpConfig lazily imports evaluation entrypoint code in
    model_post_init() to populate eval_runtime_params. That import path pulls in
    heavy evaluation / data-generation modules and can stall remote bridge
    startup. For the RATS bridge we only need the small subset of runtime
    fields accessed by the simulator stack.
    """

    episode_idx: int | None = None
    max_episodes: int | None = None
    add_custom_object: bool = False
    custom_object_path: str | Path | None = None
    custom_object_name: str | None = None
    robot_override_fn: Any | None = None


class _RATSFrankaExpConfig(MlSpacesExpConfig):
    """Minimal experiment config for RATS bridge usage."""

    num_envs: int = 1
    num_workers: int = 1
    use_passive_viewer: bool = False
    viewer_cam_dict: dict = {
        "distance": 5.0,
        "azimuth": 45.0,
        "elevation": -30.0,
        "lookat": [0.0, 0.0, 0.5],
    }
    policy_dt_ms: float = 66.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0
    task_horizon: int | None = 500
    terminate_upon_success: bool = True

    scene_dataset: str = "procthor-10k"
    data_split: str = "train"
    task_type: str = "pick"
    use_filament: bool = False

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    task_sampler_config: BaseMujocoTaskSamplerConfig = BaseMujocoTaskSamplerConfig(
        task_sampler_class=BaseMujocoTaskSampler,
        house_inds=[0],
        samples_per_house=1,
        task_batch_size=1,
        max_tasks=10000,
        load_robot_from_file=True,
    )
    task_config: BaseMujocoTaskConfig = BaseMujocoTaskConfig(task_cls=None)
    # Allow PickPlannerPolicyConfig OR OpenClosePlannerPolicyConfig (both subclass BasePolicyConfig).
    policy_config: BasePolicyConfig = PickPlannerPolicyConfig()
    output_dir: Path = Path("outputs/molmospaces_bridge")

    def model_post_init(self, _context) -> None:
        assert (self.policy_dt_ms / self.ctrl_dt_ms).is_integer(), (
            "policy_dt_ms must be a multiple of ctrl_dt_ms"
        )
        assert (self.ctrl_dt_ms / self.sim_dt_ms).is_integer(), (
            "ctrl_dt_ms must be a multiple of sim_dt"
        )
        if self.eval_runtime_params is None:
            self.eval_runtime_params = _RATSEvalRuntimeParams()

    @property
    def tag(self) -> str:
        return "rats_molmospaces_bridge"


class _RATSJsonBenchmarkConfig(MlSpacesExpConfig):
    """Minimal config used to replay exact benchmark episodes for RATS."""

    num_envs: int = 1
    num_workers: int = 1
    use_passive_viewer: bool = False
    viewer_cam_dict: dict = {
        "distance": 5.0,
        "azimuth": 45.0,
        "elevation": -30.0,
        "lookat": [0.0, 0.0, 0.5],
    }
    policy_dt_ms: float = 66.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0
    task_horizon: int = 500
    use_filament: bool = False

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    task_sampler_config: BaseMujocoTaskSamplerConfig = BaseMujocoTaskSamplerConfig(
        task_sampler_class=BaseMujocoTaskSampler,
        house_inds=[0],
        samples_per_house=1,
        task_batch_size=1,
        max_tasks=10000,
        load_robot_from_file=True,
    )
    task_config: BaseMujocoTaskConfig = BaseMujocoTaskConfig(task_cls=None)
    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig()
    output_dir: Path = Path("outputs/molmospaces_bridge")

    def model_post_init(self, _context) -> None:
        assert (self.policy_dt_ms / self.ctrl_dt_ms).is_integer(), (
            "policy_dt_ms must be a multiple of ctrl_dt_ms"
        )
        assert (self.ctrl_dt_ms / self.sim_dt_ms).is_integer(), (
            "ctrl_dt_ms must be a multiple of sim_dt"
        )
        if self.eval_runtime_params is None:
            self.eval_runtime_params = _RATSEvalRuntimeParams()

    @property
    def tag(self) -> str:
        return "rats_json_benchmark_bridge"


class _RATSCameraSystem(FrankaEvalCameraSystem):
    """Default (randomized) camera system: wrist cam + workspace-relative exo cam.

    For a deterministic variant (same pose every episode) use
    :func:`_build_rats_camera_system` with ``randomize_agentview=False``.

    The exocentric azimuth covers the full circle around the workspace
    center, but the visibility thresholds are intentionally strict: the
    camera must actually see the task objects (drawer / handle) and the
    gripper before the placement is accepted. Combined with the bridge's
    front-facing robot placement, that pushes accepted poses to the
    approach hemisphere even though the sampler itself doesn't know the
    robot's orientation. ``allow_relaxed_constraints`` is left enabled so
    we degrade to the best attempt rather than crashing the episode.

    NOTE: marked as a ``FrankaEvalCameraSystem`` subclass so the
    ``JsonEvalTaskSampler`` benchmark mode (see
    rats/third_party/molmospaces/.../json_eval_task_sampler.py:235)
    keeps it instead of replacing it with the episode-JSON's recorded
    cameras. Without this, procthor-objaverse Pick-v2 / PnP-v2 episodes
    silently swapped in the droid robot-mounted shoulder camera, which
    rendered all-black frames (verified: 50/50 procthor-objaverse
    videos were completely black, 0/96 ithor videos were).
    """

    img_resolution: tuple[int, int] = (800, 512)
    cameras: list = [
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_cam",
            robot_namespace="robot_0/",
            fov=58.0,
            record_depth=True,
        ),
        EvalExocentricCameraConfig(
            name="exo_camera_1",
            fov=65.0,
            distance_range=(-0.2, 0.4),
            height_range=(-0.1, 0.25),
            azimuth_range=(-np.pi, np.pi),
            workspace_center_weight=1.0,
            lookat_noise_range=(-0.02, 0.02),
            fov_range=(55, 75),
            record_depth=True,
            visibility_constraints={
                "__task_objects__": 0.0001,
                "__gripper__": 0.0001,
            },
            max_placement_attempts=80,
        ),
    ]


class _RATSFixedCameraSystem(FrankaEvalCameraSystem):
    """Deterministic variant: same workspace-relative exo pose every reset.

    Keeps the exocentric camera anchored to ``get_workspace_center()`` so it
    still tracks across scene changes, but collapses the sampling ranges to a
    single point so successive resets produce the same viewpoint.
    """

    img_resolution: tuple[int, int] = (800, 512)
    cameras: list = [
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_cam",
            robot_namespace="robot_0/",
            fov=58.0,
            record_depth=True,
        ),
        EvalExocentricCameraConfig(
            name="exo_camera_1",
            fov=65.0,
            distance_range=(0.0, 0.0),
            height_range=(0.0, 0.0),
            azimuth_range=(0.0, 0.0),
            workspace_center_weight=1.0,
            lookat_noise_range=(0.0, 0.0),
            fov_range=(65.0, 65.0),
            record_depth=True,
            visibility_constraints={
                "__task_objects__": 0.001,
                "__gripper__": 0.001,
            },
            max_placement_attempts=50,
        ),
    ]


def _build_rats_camera_system(
    *, randomize_agentview: bool, use_recorded_cameras: bool = False
):
    if use_recorded_cameras:
        return CameraSystemConfig()
    return _RATSCameraSystem() if randomize_agentview else _RATSFixedCameraSystem()


class _FrontFacingOpenTaskSampler(OpenTaskSampler):
    """OpenTaskSampler that retries robot placement until it lands on the
    *front* side of the articulated joint.

    The upstream sampler calls ``env.place_robot_near`` with uniform sampling
    around the joint's leaf body. For prismatic joints (drawers, sliding
    doors) and hinge joints (cabinet doors, oven doors), only one side of
    the joint is actually approachable — the side the joint opens toward.
    Random placement frequently puts the robot perpendicular or behind the
    target, which leaves the handle out of reach for IK and out of the
    front-hemisphere camera frustum.

    Strategy: call the parent placement up to ``front_side_max_attempts``
    times, accept the first one whose robot-base position lies on the
    opening side of the joint's leaf body within
    ``front_side_dot_threshold``. If all constrained super() attempts miss,
    fall back to a geometric placement that computes a point at
    ``leaf_pos + offset * world_axis_xy`` (prismatic) or
    ``leaf_pos + offset * door_normal_xy`` (hinge) and drives
    ``env.place_robot_near`` with ``face_target=True`` and
    ``check_camera_visibility=True``. This is especially important for
    articulated assets without precomputed grasp files (microwave, fridge,
    doorways) where the super() sampler's uniform radius around the leaf
    body regularly puts the robot on the wrong half-plane.
    """

    front_side_max_attempts: int = 16
    front_side_dot_threshold: float = -0.3
    geometric_fallback_offsets: tuple[float, ...] = (0.85, 0.70, 1.00, 0.55)

    def _sample_and_place_robot(self, env) -> None:
        # Clear the per-object exclusion list at the start of every reset.
        # The parent sampler appends each successful placement to
        # ``used_robot_positions[name]`` with a 0.15 m exclusion radius, and
        # that list is only cleared by ``sampler.reset()`` (which the bridge
        # never calls). Across iterations the cache accumulates until every
        # free point near the target is within 0.15 m of a prior position,
        # and ``env.place_robot_near`` fails with "Position excluded" on all
        # 10 internal tries. Clearing here also prevents our own retry loop
        # from poisoning itself: each wrong-side super() call would otherwise
        # carve out a new 0.15 m hole from the already-shrinking candidate
        # set.
        task_cfg = self.config.task_config
        pickup_obj_name = getattr(task_cfg, "pickup_obj_name", None)
        if pickup_obj_name is not None:
            self.used_robot_positions[pickup_obj_name] = []

        last_error: Exception | None = None
        had_placement_on_wrong_side = False
        for attempt in range(self.front_side_max_attempts):
            try:
                super()._sample_and_place_robot(env)
            except ValueError as e:
                last_error = e
                logger.info(
                    "Front-facing placement attempt %d/%d raised %s; retrying",
                    attempt + 1,
                    self.front_side_max_attempts,
                    e,
                )
                continue
            if self._robot_is_on_front_side(env):
                if attempt > 0:
                    logger.info(
                        "Front-facing placement converged on attempt %d/%d",
                        attempt + 1,
                        self.front_side_max_attempts,
                    )
                return
            had_placement_on_wrong_side = True
            logger.info(
                "Front-facing placement attempt %d/%d landed on the wrong side; retrying",
                attempt + 1,
                self.front_side_max_attempts,
            )
            # Drop the wrong-side position we just appended so it doesn't
            # carve out another 0.15 m hole from subsequent retries.
            if pickup_obj_name is not None and self.used_robot_positions[pickup_obj_name]:
                self.used_robot_positions[pickup_obj_name].pop()

        if self._geometric_front_side_fallback(env):
            logger.warning(
                "Front-facing super() placement failed after %d attempts; "
                "accepted geometric fallback placement on the opening side.",
                self.front_side_max_attempts,
            )
            return

        if had_placement_on_wrong_side:
            # At least one attempt produced a valid (but suboptimal)
            # placement and geometric fallback didn't help. The env already
            # reflects the last successful call, so we accept it rather
            # than raise.
            logger.warning(
                "Could not place robot on the front side after %d attempts "
                "(geometric fallback also failed); keeping last placement.",
                self.front_side_max_attempts,
            )
            return
        # Every super() attempt raised and geometric fallback failed too.
        # One final unconstrained super() call so reset() doesn't crash.
        try:
            super()._sample_and_place_robot(env)
        except ValueError:
            if last_error is not None:
                raise last_error
            raise
        logger.warning(
            "Front-facing placement raised on every attempt (last error: %s); "
            "accepting unconstrained fallback placement.",
            last_error,
        )

    def _geometric_front_side_fallback(self, env) -> bool:
        """Compute a target point on the joint's opening side and re-place.

        Returns True if a placement succeeded and lands on the front side.
        Used when super()'s uniform sampling can't find a front-side pose —
        common for no-grasp-file articulated objects where the occupancy
        map is sparse around the joint.
        """
        from molmo_spaces.env.data_views import MlSpacesArticulationObject

        try:
            task_cfg = self.config.task_config
            om = env.object_managers[env.current_batch_index]
            pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
            if not isinstance(pickup_obj, MlSpacesArticulationObject):
                return False

            joint_index = task_cfg.joint_index
            leaf_pos = np.asarray(
                pickup_obj.get_joint_leaf_body_position(joint_index),
                dtype=np.float64,
            )
            front_axis_xy = self._compute_front_axis_xy(pickup_obj, joint_index)
            if front_axis_xy is None:
                return False

            robot_view = env.current_robot.robot_view
            sampler_cfg = self.config.task_sampler_config

            min_z_offset = getattr(sampler_cfg, "robot_object_z_offset_random_min", 0.0)
            max_z_offset = getattr(sampler_cfg, "robot_object_z_offset_random_max", 0.0)
            z_jitter = np.random.uniform(min_z_offset, max_z_offset)
            initial_robot_z = (
                leaf_pos[2]
                + getattr(sampler_cfg, "robot_object_z_offset", 0.0)
                + z_jitter
            )
            safety_radius = getattr(sampler_cfg, "robot_safety_radius", 0.35)

            # Intentionally pass no exclusions: the fallback targets a
            # geometrically-computed point on the opening side and we don't
            # want stale wrong-side attempts to carve out 0.15 m holes
            # around the exact spot we just chose.
            for offset in self.geometric_fallback_offsets:
                target_point = leaf_pos.copy()
                target_point[0] += float(offset) * float(front_axis_xy[0])
                target_point[1] += float(offset) * float(front_axis_xy[1])
                placed = env.place_robot_near(
                    robot_view=robot_view,
                    target=target_point,
                    max_tries=10,
                    sampling_radius_range=(0.0, 0.3),
                    robot_safety_radius=safety_radius,
                    preserve_z=initial_robot_z,
                    face_target=True,
                    check_camera_visibility=True,
                    visibility_resolver=self.get_visibility_resolver(env),
                    excluded_positions=None,
                )
                if placed and self._robot_is_on_front_side(env):
                    self.used_robot_positions[pickup_obj.name].append(
                        robot_view.base.pose[:3, 3]
                    )
                    from molmo_spaces.utils.pose import pose_mat_to_7d

                    task_cfg.robot_base_pose = pose_mat_to_7d(
                        robot_view.base.pose
                    ).tolist()
                    return True
            return False
        except Exception:
            logger.warning(
                "Geometric front-side fallback crashed; skipping", exc_info=True
            )
            return False

    def _compute_front_axis_xy(self, pickup_obj, joint_index: int) -> np.ndarray | None:
        """Unit XY vector pointing away from the joint on its opening side.

        Prismatic: joint axis projected onto XY (drawer pulls along its axis).
        Hinge: local +X of the joint body (the door's outward normal).
        Returns None if the axis is effectively vertical and can't be
        projected onto the floor plane.
        """
        try:
            joint_type = pickup_obj.get_joint_type(joint_index)
            body_rot = pickup_obj.get_joint_body_orientation(joint_index)

            if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                local_axis = pickup_obj.get_joint_axis(joint_index)
                world_axis = body_rot @ np.asarray(local_axis, dtype=np.float64)
                axis_xy = world_axis[:2]
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                door_normal = body_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
                axis_xy = door_normal[:2]
            else:
                return None

            norm = float(np.linalg.norm(axis_xy))
            if norm < 1e-6:
                return None
            return axis_xy / norm
        except Exception:
            return None

    def _robot_is_on_front_side(self, env) -> bool:
        from molmo_spaces.env.data_views import MlSpacesArticulationObject

        try:
            task_cfg = self.config.task_config
            om = env.object_managers[env.current_batch_index]
            pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
            if not isinstance(pickup_obj, MlSpacesArticulationObject):
                return True

            joint_index = task_cfg.joint_index
            local_axis = pickup_obj.get_joint_axis(joint_index)
            body_rot = pickup_obj.get_joint_body_orientation(joint_index)
            world_axis = body_rot @ np.asarray(local_axis, dtype=np.float64)
            world_axis_xy = world_axis[:2]
            axis_norm = np.linalg.norm(world_axis_xy)
            if axis_norm < 1e-6:
                return True
            world_axis_xy = world_axis_xy / axis_norm

            leaf_pos = pickup_obj.get_joint_leaf_body_position(joint_index)
            robot_view = env.current_robot.robot_view
            robot_pos = np.asarray(robot_view.base.pose[:3, 3], dtype=np.float64)

            joint_type = pickup_obj.get_joint_type(joint_index)
            if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                # Drawer slides along +axis when opening; the robot must be
                # standing on that same side to be able to pull it open.
                offset = (robot_pos[:2] - leaf_pos[:2])
                offset_norm = np.linalg.norm(offset)
                if offset_norm < 1e-6:
                    return True
                offset = offset / offset_norm
                return float(np.dot(offset, world_axis_xy)) > self.front_side_dot_threshold
            if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                # Hinge doors open by sweeping; "front" is the half-plane on
                # the swing side. Approximate with the robot sitting on the
                # side where the leaf body's local +X (door normal) points.
                door_normal = body_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
                door_normal_xy = door_normal[:2]
                normal_norm = np.linalg.norm(door_normal_xy)
                if normal_norm < 1e-6:
                    return True
                door_normal_xy = door_normal_xy / normal_norm
                offset = robot_pos[:2] - leaf_pos[:2]
                offset_norm = np.linalg.norm(offset)
                if offset_norm < 1e-6:
                    return True
                offset = offset / offset_norm
                return float(np.dot(offset, door_normal_xy)) > self.front_side_dot_threshold
            return True
        except Exception:
            logger.warning(
                "Front-side check failed; accepting placement", exc_info=True
            )
            return True

    def get_workspace_center(self, env) -> np.ndarray:
        """Bias the exo-camera workspace center toward the robot side of
        the articulated face.

        The base ``OpenTaskSampler`` inherits ``get_workspace_center`` from
        ``task_sampler.TaskSampler``, which returns the gripper position.
        the exocentric camera config then samples a camera around that
        point with full 2π azimuth, keeping only poses that satisfy
        ``visibility_constraints``. For cavity
        articulations (oven interior, fridge interior, shower booth) half
        that sphere lies *inside* the cavity or *behind* the articulated
        face: the picker often lands there with a view of the handle but
        rotated 180° from the robot — the camera ends up inside the oven
        looking back through the door, or outside the shower booth looking
        at a wall.

        Override: put the center on the line between the handle (joint
        leaf body) and the robot base, biased 60/40 toward the robot, and
        raised 0.2 m above the taller of the two. The sampling sphere now
        sits on the robot's side of the articulated face, so azimuths that
        would land behind the face get culled by visibility constraints
        (the handle is occluded from behind) while useful front-facing
        views survive.

        Falls back to the base implementation if the task doesn't expose
        a pickup articulation (shouldn't happen during open/close tasks
        but protects us from subclass reuse elsewhere).
        """
        from molmo_spaces.env.data_views import MlSpacesArticulationObject

        try:
            task_cfg = self.config.task_config
            pickup_obj_name = getattr(task_cfg, "pickup_obj_name", None)
            if pickup_obj_name is None:
                return super().get_workspace_center(env)
            om = env.object_managers[env.current_batch_index]
            pickup_obj = om.get_object_by_name(pickup_obj_name)
            if not isinstance(pickup_obj, MlSpacesArticulationObject):
                return super().get_workspace_center(env)

            joint_index = getattr(task_cfg, "joint_index", 0)
            leaf_pos = np.asarray(
                pickup_obj.get_joint_leaf_body_position(joint_index),
                dtype=np.float64,
            )
            robot_view = env.current_robot.robot_view
            robot_pos = np.asarray(
                robot_view.base.pose[:3, 3], dtype=np.float64
            )

            center = 0.4 * leaf_pos + 0.6 * robot_pos
            center[2] = max(float(leaf_pos[2]), float(robot_pos[2])) + 0.2
            return center
        except Exception:
            logger.warning(
                "Articulated workspace-center override failed; "
                "falling back to base implementation",
                exc_info=True,
            )
            return super().get_workspace_center(env)


_TASK_TYPE_MAP["open"]["front_facing_sampler_cls"] = _FrontFacingOpenTaskSampler
_TASK_TYPE_MAP["close"]["front_facing_sampler_cls"] = _FrontFacingOpenTaskSampler


_GRASP_FILE_BYPASS_APPLIED = False


def _apply_grasp_file_bypass() -> None:
    """Make all grasp-file gating in molmo_spaces samplers a no-op.

    The opening / pick samplers filter both candidate objects (must have a
    grasp folder + at least one transform) and joints (must have a per-joint
    grasp .npz) before placing the robot. RATS drives grasp generation at
    runtime via SAM3 + GraspNet, so we don't need molmo_spaces' grasp catalog
    for sample-time eligibility — but the filter still gates which objects
    we're allowed to manipulate. This patch widens that gate to ``True`` so
    every articulated joint becomes eligible.

    Idempotent and process-global: the bridge owns these modules in its
    server process, so monkey-patching is fine. Once applied it stays
    applied for the process lifetime.
    """
    global _GRASP_FILE_BYPASS_APPLIED
    if _GRASP_FILE_BYPASS_APPLIED:
        return

    from molmo_spaces.tasks import opening_task_samplers as _ots
    from molmo_spaces.tasks import pick_task_sampler as _pts
    from molmo_spaces.tasks import json_eval_task_sampler as _jets
    from molmo_spaces.tasks import eval_task_sampler as _ets
    from molmo_spaces.env.data_views import MlSpacesArticulationObject as _MlSpacesArticulationObject

    def _always_true(*_args, **_kwargs):  # pragma: no cover - trivial
        return True

    def _set_joint_values_without_grasp_gate(self, env: Any) -> None:
        if "pickup_obj_name" not in self.episode_spec.task:
            return

        object_manager = env.object_managers[env.current_batch_index]
        pickup_obj = object_manager.get_object_by_name(
            self.episode_spec.task["pickup_obj_name"],
        )
        if not isinstance(pickup_obj, _MlSpacesArticulationObject):
            return

        try:
            target_joint_name = self.episode_spec.task["joint_name"]
            joint_start_position = self.episode_spec.task["joint_start_position"][0]
        except (AttributeError, KeyError) as exc:
            logger.warning("Not setting articulated benchmark joint.", exc_info=True)
            raise exc

        target_joint_index = list(pickup_obj.joint_names).index(target_joint_name)
        pickup_obj.set_joint_position(target_joint_index, joint_start_position)

    def _wrap_set_joint_values(original):
        def _wrapped(self, env: Any) -> None:
            try:
                return original(self, env)
            except ValueError as exc:
                if "No joints with grasp file found" not in str(exc):
                    raise
                logger.warning(
                    "Bypassing JSON benchmark joint grasp-file gate for %s; "
                    "using recorded joint_start_position instead.",
                    self.episode_spec.task.get("pickup_obj_name"),
                )
                return _set_joint_values_without_grasp_gate(self, env)

        return _wrapped

    _ots.has_joint_grasp_file = _always_true
    _pts.has_grasp_folder = _always_true
    _pts.has_valid_grasp_file = _always_true
    _jets.JsonEvalTaskSampler.set_joint_values = _wrap_set_joint_values(
        _jets.JsonEvalTaskSampler.set_joint_values,
    )
    _ets.EvalTaskSampler.set_joint_values = _wrap_set_joint_values(
        _ets.EvalTaskSampler.set_joint_values,
    )

    logger.warning(
        "Grasp-file bypass enabled: every articulated joint is now eligible "
        "for sample-time selection. RATS GraspNet/SAM3 will own runtime "
        "grasp synthesis. Sampler may pick joints with no precomputed grasps."
    )
    _GRASP_FILE_BYPASS_APPLIED = True


class MolmoSpacesBridge:
    """Real bridge wrapping the molmo_spaces MuJoCo simulator.

    Provides session management, physics stepping, rendering, and reward/success
    evaluation backed by the actual simulator.
    """

    def __init__(
        self,
        *,
        task_type: str = "pick",
        scene_dataset: str = "procthor-10k",
        data_split: str = "train",
        house_index: int | None = None,
        benchmark_dir: str | Path | None = None,
        canonical_task_id: str | None = None,
        episode_index: int | None = None,
        max_steps: int = 4000,
        render_width: int = 800,
        render_height: int = 512,
        seed: int | None = None,
        reset_physical_state: bool = True,
        randomize_agentview: bool = True,
        use_recorded_cameras: bool = False,
        pickup_types: list[str] | None = None,
        require_grasp_files: bool = True,
        pin_pickup_obj_name: bool = True,
        front_facing_robot_placement: bool = True,
        candidate_house_indices: list[int] | None = None,
    ) -> None:
        if _MOLMO_SPACES_IMPORT_ERROR is not None:
            raise ImportError(
                "MolmoSpacesBridge requires the molmo_spaces package. "
                "Initialize rats/third_party/molmospaces or install molmo_spaces "
                "before constructing the real bridge."
            ) from _MOLMO_SPACES_IMPORT_ERROR

        self._task_type = task_type
        self._scene_dataset = scene_dataset
        self._data_split = data_split
        self._house_index = house_index
        self._benchmark_dir = benchmark_dir
        self._requested_canonical_task_id = canonical_task_id
        self._requested_episode_index = episode_index
        self._max_steps = max_steps
        self._render_width = render_width
        self._render_height = render_height
        self._seed = seed
        self._reset_physical_state = reset_physical_state
        self._randomize_agentview = randomize_agentview
        self._use_recorded_cameras = bool(use_recorded_cameras)
        if self._use_recorded_cameras and benchmark_dir is None:
            raise ValueError("use_recorded_cameras=True requires benchmark_dir")
        self._pickup_types = list(pickup_types) if pickup_types else None
        self._require_grasp_files = bool(require_grasp_files)
        self._pin_pickup_obj_name = bool(pin_pickup_obj_name)
        self._front_facing_robot_placement = bool(front_facing_robot_placement)
        self._candidate_house_indices = [
            int(h) for h in (candidate_house_indices or [])
        ] or None
        self._pinned_pickup_obj_name: str | None = None
        self._pinned_place_receptacle_name: str | None = None
        self._pinned_joint_index: int | None = None
        self._pinned_joint_name: str | None = None
        # True for tasks explicitly materialized by the RATS open proposer via
        # set_task_from_spec(). For those tasks retry recovery must not silently
        # drop the target and switch to a different object: the planner and
        # verifier still refer to the proposer-selected object.
        self._strict_pinned_task_identity = False
        # Snapshot of referral_expressions from the first successful sample.
        # PickTask.get_task_description() reads
        # task_config.referral_expressions["pickup_obj_name"]; when a later
        # sample_task replaces task_config and then raises (HouseInvalidForTask,
        # robot placement failures, etc.) the task's config points at a fresh
        # blank dict and the next get_task_descriptor call explodes with
        # KeyError. Keeping a copy here lets _apply_pinned_task_identity
        # reinject the original sampled phrase on every reset so the
        # description is stable even across failed intermediate draws.
        self._pinned_referral_expressions: dict[str, str] | None = None

        if not self._require_grasp_files:
            _apply_grasp_file_bypass()

        self._config: _RATSFrankaExpConfig | _RATSJsonBenchmarkConfig | None = None
        self._sampler: BaseMujocoTaskSampler | None = None
        self._task: BaseMujocoTask | None = None
        self._session_counter = 0
        self._benchmark_episodes: list[EpisodeSpec] = []
        self._benchmark_catalog: list[dict[str, Any]] = []
        self._benchmark_index_by_canonical_id: dict[str, int] = {}
        self._current_episode_index: int | None = None
        self._current_canonical_task_id: str | None = None
        self._logged_camera_aliases: set[tuple[str, str]] = set()

        if self._benchmark_dir is not None:
            self._load_benchmark_episodes()
        else:
            self._build_config()
            self._build_sampler()
        self._sample_initial_task()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_config(self) -> None:
        type_info = _TASK_TYPE_MAP.get(self._task_type, _TASK_TYPE_MAP["pick"])

        task_config_overrides = type_info.get("task_config_overrides", {})
        task_config = type_info["config_cls"](
            task_cls=type_info["task_cls"], **task_config_overrides
        )

        sampler_config_overrides = dict(type_info.get("sampler_config_overrides", {}))
        # pickup_types flows into both OpenTaskSamplerConfig (articulated
        # categories like "drawer", "oven") and PickTaskSamplerConfig
        # (semantic object categories like "cup", "bowl"). No articulated-
        # only gate — let pick tasks filter scene objects by category too.
        if self._pickup_types is not None:
            sampler_config_overrides["pickup_types"] = self._pickup_types
        if self._candidate_house_indices:
            sampler_config_overrides["house_inds"] = self._candidate_house_indices
        sampler_config = type_info["sampler_config_cls"](
            task_sampler_class=type_info["sampler_cls"],
            **sampler_config_overrides,
        )

        policy_config_cls = type_info.get("policy_config_cls", PickPlannerPolicyConfig)

        self._config = _RATSFrankaExpConfig(
            task_type=self._task_type,
            scene_dataset=self._scene_dataset,
            data_split=self._data_split,
            task_horizon=self._max_steps,
            seed=self._seed,
            task_config=task_config,
            task_sampler_config=sampler_config,
            camera_config=_build_rats_camera_system(
                randomize_agentview=self._randomize_agentview,
                use_recorded_cameras=self._use_recorded_cameras,
            ),
            policy_config=policy_config_cls(),
        )

    def _build_sampler(self) -> None:
        type_info = _TASK_TYPE_MAP.get(self._task_type, _TASK_TYPE_MAP["pick"])
        sampler_cls = type_info["sampler_cls"]
        if (
            self._front_facing_robot_placement
            and self._task_type in _ARTICULATED_TASK_TYPES
            and type_info.get("front_facing_sampler_cls") is not None
        ):
            sampler_cls = type_info["front_facing_sampler_cls"]
        self._sampler = sampler_cls(self._config)

    def _sample_initial_task(self) -> None:
        if self._benchmark_episodes:
            self.resample_task(
                house_index=self._house_index,
                canonical_task_id=self._requested_canonical_task_id,
                episode_index=self._requested_episode_index,
            )
            return

        # Auto-discover a house for any task type when house_index is unset
        # (and we're not already bound to a benchmark). Pick tasks benefit
        # from this as much as open/close: the configured pickup_types
        # constrain which houses contain valid candidates, so a random
        # house pick from procthor-10k (10k houses) usually misses.
        if self._house_index is None:
            self._task = self._auto_discover_house_for_pickup_types()
        else:
            self._task = _sample_task_compat(
                self._sampler,
                house_index=self._house_index,
                variant="ceiling",
            )
        if self._task is None:
            raise RuntimeError("Failed to sample initial task from MolmoSpaces")
        if self._requested_canonical_task_id:
            self._current_canonical_task_id = self._requested_canonical_task_id
        self._capture_pinned_task_identity()

    def _auto_discover_house_for_pickup_types(self):
        """Sweep candidate iThor FloorPlans until one yields a valid task.

        Used when ``house_index`` is left unset on an articulated-opens
        config. We rank houses by which scene category most likely contains
        the requested ``pickup_types`` (kitchens have fridges, bathrooms
        have toilets, etc.) and fall through to the full iThor range as a
        last resort. The first house whose ``sample_task`` doesn't raise
        ``HouseInvalidForTask`` or ``ValueError`` is pinned as
        ``self._house_index`` so subsequent resets re-use it.
        """
        candidates = self._rank_candidate_houses()
        logger.info(
            "Auto-discovering house for pickup_types=%s across %d candidates",
            self._pickup_types,
            len(candidates),
        )
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                task = _sample_task_compat(
                    self._sampler,
                    house_index=candidate,
                    variant="ceiling",
                )
            except Exception as e:
                last_error = e
                logger.debug(
                    "Auto-discovery skipped house %d: %s", candidate, e
                )
                continue
            if task is not None:
                logger.info(
                    "Auto-discovered house_index=%d for pickup_types=%s",
                    candidate,
                    self._pickup_types,
                )
                self._house_index = candidate
                return task
        raise RuntimeError(
            f"Auto-discovery failed for pickup_types={self._pickup_types} "
            f"across {len(candidates)} candidate houses. Last error: {last_error}"
        )

    def _rank_candidate_houses(self) -> list[int]:
        """Return house indices ordered by expected asset fit.

        iThor scene ranges (each scene is ``FloorPlan{idx}``):
          - 1..30   kitchen
          - 201..230 living room
          - 301..330 bedroom
          - 401..430 bathroom

        For procthor-10k / holodeck-objaverse / procthor-objaverse we
        don't get per-room-type scene ranges (each house has mixed rooms),
        so we sweep sequential indices 0..N and let the sampler reject
        houses whose inventory doesn't contain the requested pickup_types.
        The sweep width is capped at ``_PROCGEN_AUTO_DISCOVER_LIMIT`` so
        we don't spend minutes probing dead-end houses — if you need a
        specific scene, pin ``house_index`` explicitly.

        The ranking is a heuristic speedup — the sweep still falls through
        to the full range if the preferred group doesn't hit.
        """
        if self._candidate_house_indices:
            candidates = list(dict.fromkeys(self._candidate_house_indices))
            # Seeded random order: smoke configs get a deterministic but not
            # always-first house from their curated list. Each task-family
            # reinit uses the same seed/list, so all smoke task types start
            # from the same diverse house when it is compatible.
            import random
            rng = random.Random(self._seed)
            rng.shuffle(candidates)
            return candidates

        kitchen = list(range(1, 31))
        living = list(range(201, 231))
        bedroom = list(range(301, 331))
        bathroom = list(range(401, 431))

        if self._scene_dataset != "ithor":
            # Procedural scene datasets aren't partitioned by room type;
            # sequential indices sweep with the first valid house winning.
            return list(range(_PROCGEN_AUTO_DISCOVER_LIMIT))

        # Articulated categories + semantic pickup categories both resolve
        # to a scene group. Unknown categories default to "all".
        scene_for_pickup = {
            # Articulated categories (open/close tasks).
            "cabinet": "all",
            "drawer": "all",
            "oven": "kitchen",
            "dishwasher": "kitchen",
            "Fridge": "kitchen",
            "Microwave": "kitchen",
            "showerdoor": "bathroom",
            "Toilet": "bathroom",
            "Doorways": "all",
            "Doorway_Double": "all",
            "Safe": "all",
            "Dresser": "bedroom",
            "Desk": "bedroom",
            "Shelving_Unit": "living",
            "Side_Table": "living",
            "Coffee_Table": "living",
            "Laptop": "all",
            "Laundry_Hamper": "bedroom",
            # Semantic pickup categories (pick tasks). Derived from iThor
            # object-type-to-room associations.
            "alarm_clock": "bedroom",
            "apple": "kitchen",
            "bottle": "kitchen",
            "bread": "kitchen",
            "butterknife": "kitchen",
            "candle": "bathroom",
            "cd": "bedroom",
            "cellphone": "all",
            "cloth": "bathroom",
            "cup": "kitchen",
            "dish_sponge": "kitchen",
            "egg": "kitchen",
            "fork": "kitchen",
            "hand_towel": "bathroom",
            "keychain": "all",
            "knife": "kitchen",
            "ladle": "kitchen",
            "mug": "kitchen",
            "newspaper": "living",
            "pan": "kitchen",
            "pen": "bedroom",
            "pencil": "bedroom",
            "pepper_shaker": "kitchen",
            "plate": "kitchen",
            "pot": "kitchen",
            "potato": "kitchen",
            "remote": "living",
            "salt_shaker": "kitchen",
            "scrub_brush": "bathroom",
            "soap_bar": "bathroom",
            "soap_bottle": "bathroom",
            "spatula": "kitchen",
            "spoon": "kitchen",
            "spray_bottle": "bathroom",
            "tissue_box": "bathroom",
            "toilet_paper": "bathroom",
            "tomato": "kitchen",
            "watch": "bedroom",
            "wine_bottle": "kitchen",
        }
        types = self._pickup_types or []

        def group_order(group: str) -> list[int]:
            return {
                "kitchen": kitchen + living + bedroom + bathroom,
                "living": living + bedroom + kitchen + bathroom,
                "bedroom": bedroom + living + kitchen + bathroom,
                "bathroom": bathroom + kitchen + living + bedroom,
                "all": kitchen + living + bedroom + bathroom,
            }[group]

        groups = {scene_for_pickup.get(t, "all") for t in types} or {"all"}
        if "bathroom" in groups and len(groups) == 1:
            return group_order("bathroom")
        if "kitchen" in groups and len(groups) == 1:
            return group_order("kitchen")
        if "bedroom" in groups and len(groups) == 1:
            return group_order("bedroom")
        if "living" in groups and len(groups) == 1:
            return group_order("living")
        return group_order("all")

    def _capture_pinned_task_identity(self) -> None:
        """Snapshot which object/joint the sampler picked on the first sample.

        ``OpenTaskSampler._sample_task`` cycles through ``candidate_objects``
        via an internal ``_task_counter`` whenever ``pickup_obj_name is None``.
        Without intervention every reset() would advance to a different object
        (drawer -> oven -> dishwasher -> ...), so the live task language no
        longer matches what the policy sees.
        """
        if not self._pin_pickup_obj_name or self._sampler is None:
            return
        task_cfg = getattr(self._sampler.config, "task_config", None)
        if task_cfg is None:
            return
        pickup_obj_name = getattr(task_cfg, "pickup_obj_name", None)
        if pickup_obj_name is None:
            return
        self._pinned_pickup_obj_name = pickup_obj_name
        place_receptacle_name = getattr(task_cfg, "place_receptacle_name", None)
        if place_receptacle_name is not None:
            self._pinned_place_receptacle_name = place_receptacle_name
        self._pinned_joint_index = getattr(task_cfg, "joint_index", None)
        self._pinned_joint_name = getattr(task_cfg, "joint_name", None)
        referral = getattr(task_cfg, "referral_expressions", None)
        if isinstance(referral, dict) and referral:
            self._pinned_referral_expressions = dict(referral)
        logger.info(
            "Pinned task identity: pickup_obj_name=%s place_receptacle_name=%s joint_name=%s",
            self._pinned_pickup_obj_name,
            self._pinned_place_receptacle_name,
            self._pinned_joint_name,
        )

    def _set_sampler_place_receptacle_name(self, name: str | None) -> None:
        """Set the pick-and-place sampler's current place target if present."""
        if self._sampler is None or name is None:
            return
        if hasattr(self._sampler, "place_receptacle_name"):
            self._sampler.place_receptacle_name = name
        for cfg_attr in ("task_config", "task_config_preset_scn"):
            cfg = getattr(self._sampler.config, cfg_attr, None)
            if cfg is None:
                continue
            if hasattr(cfg, "place_receptacle_name"):
                cfg.place_receptacle_name = name
            if hasattr(cfg, "place_target_name"):
                cfg.place_target_name = name

    def _default_sampler_place_receptacle_name(self) -> str | None:
        """Best-effort default place target when clearing an explicit pin."""
        if self._sampler is None:
            return None
        try:
            active = getattr(self._sampler, "active_receptacle_names", None)
            if active:
                return str(list(active)[0])
        except Exception:
            pass
        names = getattr(self._sampler, "_receptacle_names", None)
        if names:
            try:
                return str(list(names)[0])
            except Exception:
                return None
        current = getattr(self._sampler, "place_receptacle_name", None)
        return str(current) if current else None

    def _apply_pinned_task_identity(self) -> None:
        """Re-inject the captured object name into both task_config caches.

        ``BaseMujocoTaskSampler.sample_task`` resets ``task_config`` from
        ``task_config_preset_scn`` when the scene is re-used (the common case
        across attempts), so we have to mutate the cached preset too — setting
        only ``task_config.pickup_obj_name`` would be overwritten before
        ``_sample_task`` runs.
        """
        if (
            not self._pin_pickup_obj_name
            or self._sampler is None
            or self._pinned_pickup_obj_name is None
        ):
            return
        self._set_sampler_place_receptacle_name(self._pinned_place_receptacle_name)
        for cfg_attr in ("task_config", "task_config_preset_scn"):
            cfg = getattr(self._sampler.config, cfg_attr, None)
            if cfg is None:
                continue
            cfg.pickup_obj_name = self._pinned_pickup_obj_name
            if (
                self._pinned_place_receptacle_name is not None
                and hasattr(cfg, "place_receptacle_name")
            ):
                cfg.place_receptacle_name = self._pinned_place_receptacle_name
            if (
                self._pinned_place_receptacle_name is not None
                and hasattr(cfg, "place_target_name")
            ):
                cfg.place_target_name = self._pinned_place_receptacle_name
            if self._pinned_joint_index is not None and hasattr(cfg, "joint_index"):
                cfg.joint_index = self._pinned_joint_index
            if self._pinned_joint_name is not None and hasattr(cfg, "joint_name"):
                cfg.joint_name = self._pinned_joint_name
            # Restore the sampled referral phrase so PickTask.get_task_description
            # has a value to read even if the current sample_task replaced
            # task_config and raised before upstream repopulated it.
            if self._pinned_referral_expressions and hasattr(cfg, "referral_expressions"):
                existing = getattr(cfg, "referral_expressions", None)
                if not isinstance(existing, dict) or not existing:
                    cfg.referral_expressions = dict(self._pinned_referral_expressions)
                else:
                    for key, value in self._pinned_referral_expressions.items():
                        existing.setdefault(key, value)

    def _resample_on_reset(self) -> None:
        """Re-sample the task during ``reset()`` while preserving good state.

        Two hazards this function guards against:

        1. ``BaseMujocoTaskSampler.sample_task`` replaces
           ``self.config.task_config`` with a fresh copy from
           ``task_config_preset_scn`` at its very top, so fields that
           ``_sample_task`` populates (``pickup_obj_goal_pose``,
           ``pickup_obj_start_pose``, ``referral_expressions``, etc.) are
           wiped *before* sampling runs. If ``_sample_task`` then raises
           (``HouseInvalidForTask``, ``RobotPlacementError``, …) the shared
           ``task_config`` object stays in that half-initialized state. The
           still-live ``self._task`` holds the same sampler config by
           reference, so any subsequent ``judge_success`` / ``get_info``
           call trips ``TypeError: 'NoneType' object is not subscriptable``
           on ``pickup_obj_goal_pose[:3]``. We snapshot the live config
           before sampling and restore on failure.

        2. A pinned object that placed successfully on the first sample may
           fail to place on retry because the pre-sample RNG state and
           scene state are not truly identical across resets. For ordinary
           sampled tasks, drop the pin once and attempt a fresh sample on the
           same house so the run continues. For explicit open-proposer specs,
           keep the pin strict: switching objects would make the task prompt,
           planner code, and scene state disagree.

        3. Per-object ``used_robot_positions`` carve-outs accumulate across
           resets. Each successful placement adds a 0.15 m exclusion around
           the chosen robot pose, and after enough resets every nearby
           candidate point is excluded — placement starts failing on
           previously-fine objects. We clear this cache on retry so the
           sampler gets the full free-space back.
        """
        # Re-seed the sampler's RNG from the configured seed before each
        # resample. Upstream seeds only in __init__ / sampler.reset(), so
        # without this the shared RNG advances across attempts and robot
        # placement, camera pose, and joint randomization drift.
        #
        # Also clear the sampler's per-object placement exclusion cache before
        # the first draw. PickTaskSampler records every successful robot base
        # pose in used_robot_positions and avoids those poses on later samples.
        # That is useful for dataset diversity but harmful for RATS retries:
        # attempt 2 should see the same pinned task from the same camera/base
        # pose as attempt 1. If we keep the cache, every retry walks the robot
        # around the object until the target leaves the camera view or the
        # placement sampler exhausts nearby points.
        self._clear_used_robot_positions()
        if self._seed is not None:
            self._sampler.seed_task_sampling(self._seed)
        self._apply_pinned_task_identity()

        strict_pin = (
            self._strict_pinned_task_identity
            and self._pinned_pickup_obj_name is not None
        )

        # Attempt 0: pinned, original seed. Attempts 1..N: progressively
        # more aggressive recovery. Each attempt is independent — failure
        # restores the snapshot so the next attempt starts clean.
        last_exc: Exception | None = None
        for attempt in range(self._RESAMPLE_MAX_ATTEMPTS):
            if attempt >= 1:
                if strict_pin:
                    if attempt == 1:
                        logger.warning(
                            "Strict MolmoSpaces task target %r failed to "
                            "resample once; keeping the pinned identity instead "
                            "of switching objects",
                            self._pinned_pickup_obj_name,
                        )
                else:
                    # Drop the pin and let the sampler pick fresh.
                    self._clear_pinned_task_identity_on_sampler()
            if attempt >= 2 or (strict_pin and attempt >= 1):
                # Free up exclusion zones the sampler accumulated across
                # past resets. Without this every candidate point near the
                # target falls inside a 0.15 m carve-out and placement
                # fails on objects that worked fine earlier.
                self._clear_used_robot_positions()
            if attempt >= 3 and self._seed is not None:
                # Bump the seed to break out of a deterministic bad state
                # (e.g. RNG keeps picking the same un-placeable joint).
                self._sampler.seed_task_sampling(self._seed + attempt)

            self._refresh_pick_candidate_objects_if_empty()
            saved_sampler_state = self._snapshot_sampler_state()
            try:
                task = _sample_task_compat(
                    self._sampler,
                    house_index=self._house_index,
                    variant="ceiling",
                )
            except Exception as exc:
                last_exc = exc
                self._restore_sampler_state(saved_sampler_state)
                if not self._is_retryable_sample_failure(exc):
                    # The exception type isn't one we know how to recover
                    # from (e.g. asset-loading bug). Re-raise immediately
                    # so the caller sees the real error.
                    raise
                if attempt < self._RESAMPLE_MAX_ATTEMPTS - 1:
                    logger.warning(
                        "sample_task attempt %d/%d failed (%s: %s); "
                        "retrying with broader recovery on house %s",
                        attempt + 1,
                        self._RESAMPLE_MAX_ATTEMPTS,
                        type(exc).__name__,
                        exc,
                        self._house_index,
                    )
                continue

            if task is None:
                self._restore_sampler_state(saved_sampler_state)
                last_exc = RuntimeError("sample_task returned None")
                continue

            self._task = task
            if attempt > 0 or self._pinned_referral_expressions is None:
                # Capture the new identity so future resets pin it instead
                # of cycling. For strict open-proposer specs this records the
                # freshly generated referral text without changing the target.
                self._capture_pinned_task_identity()
            return

        # All attempts exhausted.
        raise RuntimeError(
            f"Failed to resample MolmoSpaces task after "
            f"{self._RESAMPLE_MAX_ATTEMPTS} attempts on house "
            f"{self._house_index}; last error: "
            f"{type(last_exc).__name__ if last_exc else 'None'}: {last_exc}"
        ) from last_exc

    # Per-call attempt budget for ``_resample_on_reset``. Empirically a
    # single retry is too few — a pinned object that fell out of the
    # candidate set can need both an unpinned reroll AND an exclusion
    # cache reset before placement converges.
    _RESAMPLE_MAX_ATTEMPTS = 5

    def _clear_used_robot_positions(self) -> None:
        """Drop the sampler's per-object placement exclusion cache.

        Each successful ``env.place_robot_near`` appends the chosen robot
        base position to ``used_robot_positions[obj_name]``, which is then
        passed back as ``excluded_positions`` to bias future placements
        away from already-tried spots. The exclusion radius is 0.15 m, so
        after ~4-6 resets on the same object the candidate set is fully
        carved out and placement fails. The base sampler only clears it
        in ``sampler.reset()`` (which the bridge never calls), so we
        manage it ourselves.
        """
        if self._sampler is None:
            return
        cache = getattr(self._sampler, "used_robot_positions", None)
        if cache is None:
            return
        try:
            cache.clear()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to clear used_robot_positions", exc_info=True)

    def _snapshot_sampler_state(self) -> dict[str, Any]:
        """Snapshot mutable sampler state so failed retries don't poison later attempts.

        ``PickTaskSampler._select_pickup_object`` destructively removes
        candidates when placement / grasp checks fail. In RATS reset retries
        those removals are recovery-local: the next retry may deliberately
        clear placement exclusions or drop the pin, so it needs to see the
        original candidate list again. Snapshot after each attempt's recovery
        mutations so restore preserves the intended pin/unpin/cache-clearing
        state for that attempt while rolling back upstream's destructive
        candidate pruning.
        """
        if self._sampler is None:
            return {}
        state: dict[str, Any] = {}
        cfg = getattr(self._sampler, "config", None)
        if cfg is not None:
            for attr in ("task_config", "task_config_preset_scn"):
                value = getattr(cfg, attr, None)
                if value is not None and hasattr(value, "model_copy"):
                    state[attr] = value.model_copy(deep=True)
        candidates = getattr(self._sampler, "candidate_objects", None)
        if candidates is not None:
            state["candidate_objects"] = list(candidates)
        for attr in ("_task_counter", "_grasp_failure_counts"):
            if hasattr(self._sampler, attr):
                value = getattr(self._sampler, attr)
                if isinstance(value, dict):
                    value = dict(value)
                state[attr] = value
        cache = getattr(self._sampler, "used_robot_positions", None)
        if cache is not None:
            try:
                state["used_robot_positions"] = {
                    key: list(value) for key, value in cache.items()
                }
            except Exception:
                state["used_robot_positions"] = dict(cache)
        return state

    def _restore_sampler_state(self, snapshot: dict[str, Any]) -> None:
        if not snapshot or self._sampler is None:
            return
        cfg = getattr(self._sampler, "config", None)
        if cfg is not None:
            for attr in ("task_config", "task_config_preset_scn"):
                if attr in snapshot:
                    setattr(cfg, attr, snapshot[attr])
        if "candidate_objects" in snapshot:
            self._sampler.candidate_objects = list(snapshot["candidate_objects"])
        for attr in ("_task_counter", "_grasp_failure_counts"):
            if attr in snapshot:
                value = snapshot[attr]
                if isinstance(value, dict):
                    value = dict(value)
                setattr(self._sampler, attr, value)
        if "used_robot_positions" in snapshot:
            cache = getattr(self._sampler, "used_robot_positions", None)
            if cache is not None:
                try:
                    cache.clear()
                    for key, value in snapshot["used_robot_positions"].items():
                        cache[key] = list(value)
                except Exception:  # pragma: no cover - defensive
                    setattr(self._sampler, "used_robot_positions", snapshot["used_robot_positions"])

    # Exception class names raised by upstream samplers that mean
    # "this draw failed but a different draw on the same house could
    # still succeed" — i.e. retrying unpinned with cleared exclusions
    # has a real chance. Anything outside this set propagates as-is.
    _RETRYABLE_SAMPLE_ERRORS: frozenset[str] = frozenset({
        "HouseInvalidForTask",
        "RobotPlacementError",
        "ObjectPlacementError",
        "ValueError",
        # Front-side placement helpers raise these when the joint /
        # leaf body lookup transiently fails on a stale config.
        "KeyError",
        "RuntimeError",
    })

    @classmethod
    def _should_retry_unpinned(cls, exc: BaseException) -> bool:
        """Return True for sampling failures that are worth retrying unpinned.

        ``HouseInvalidForTask`` / ``RobotPlacementError`` /
        ``ObjectPlacementError`` mean the pinned object can't be placed
        right now — a different object in the same house may still work.
        ``ValueError`` and ``KeyError`` are used by upstream for per-object
        grasp / asset / referral-cache issues. Generic ``RuntimeError`` is
        included because a few code paths wrap the placement loop and
        re-raise as RuntimeError after exhausting their own retries.
        """
        return type(exc).__name__ in cls._RETRYABLE_SAMPLE_ERRORS

    def _sampler_candidates_exhausted(self) -> bool:
        if self._sampler is None or not hasattr(self._sampler, "candidate_objects"):
            return False
        candidates = getattr(self._sampler, "candidate_objects", None)
        return candidates is None or len(candidates) == 0

    def _is_retryable_sample_failure(self, exc: BaseException) -> bool:
        if self._should_retry_unpinned(exc):
            return True
        # PickTaskSampler asserts when its candidate list has been exhausted.
        # That can happen immediately after a recoverable placement failure
        # because the failed object was removed from candidate_objects before
        # the bridge got control back. Treat only this exhausted-candidate
        # assertion as retryable; unrelated assertions should still surface.
        return isinstance(exc, AssertionError) and self._sampler_candidates_exhausted()

    def _refresh_pick_candidate_objects_if_empty(self) -> None:
        """Rebuild pick candidates when a previous failed draw exhausted them.

        Upstream initializes ``PickTaskSampler.candidate_objects`` only from
        ``init_scene`` when a house is loaded. On same-house resets it reuses
        the scene and does not rebuild the list, but failed draws can remove
        every object. If the list is empty at retry start, mirror the relevant
        part of ``PickTaskSampler.init_scene`` without reloading the house.
        """
        if self._sampler is None or not self._sampler_candidates_exhausted():
            return
        if not all(
            hasattr(self._sampler, name)
            for name in ("_get_scene_objects", "balance_sample_names")
        ):
            return
        try:
            candidate_objects = self._sampler._get_scene_objects(self.env)
            candidate_objects = self._sampler.balance_sample_names(candidate_objects)
            np.random.shuffle(candidate_objects)
            self._sampler.candidate_objects = candidate_objects
            if hasattr(self._sampler, "_task_counter"):
                self._sampler._task_counter = 0
            logger.info(
                "Rebuilt exhausted pick candidate list with %d objects on house %s",
                len(candidate_objects),
                self._house_index,
            )
        except Exception:
            logger.debug("Failed to rebuild exhausted pick candidates", exc_info=True)

    def _clear_pinned_task_identity_on_sampler(self) -> None:
        """Drop the pin so the sampler is free to pick a different candidate."""
        self._strict_pinned_task_identity = False
        self._pinned_pickup_obj_name = None
        old_place_receptacle_name = self._pinned_place_receptacle_name
        self._pinned_place_receptacle_name = None
        self._pinned_joint_index = None
        self._pinned_joint_name = None
        self._pinned_referral_expressions = None
        if old_place_receptacle_name is not None:
            self._set_sampler_place_receptacle_name(
                self._default_sampler_place_receptacle_name()
            )
        if self._sampler is None:
            return
        for cfg_attr in ("task_config", "task_config_preset_scn"):
            cfg = getattr(self._sampler.config, cfg_attr, None)
            if cfg is None:
                continue
            if hasattr(cfg, "pickup_obj_name"):
                cfg.pickup_obj_name = None
            if hasattr(cfg, "referral_expressions"):
                cfg.referral_expressions = {}

    def _load_benchmark_episodes(self) -> None:
        benchmark_path = Path(self._benchmark_dir)
        episodes = load_all_episodes(benchmark_path)
        if not episodes:
            raise RuntimeError(f"No benchmark episodes found in {benchmark_path}")
        normalized_episodes = [
            _normalize_benchmark_episode_scene_split(episode)
            for episode in episodes
        ]
        normalized_count = sum(
            1
            for original, normalized in zip(episodes, normalized_episodes, strict=True)
            if getattr(original, "data_split", None) != getattr(normalized, "data_split", None)
        )
        if normalized_count:
            logger.info(
                "Normalized runtime scene split for %d benchmark episode(s) "
                "to match upstream MolmoSpaces scene indexing.",
                normalized_count,
            )
        self._benchmark_episodes = normalized_episodes
        self._benchmark_catalog = [
            self._episode_to_descriptor(idx, episode)
            for idx, episode in enumerate(normalized_episodes)
        ]
        self._benchmark_index_by_canonical_id = {
            descriptor["canonical_id"]: idx
            for idx, descriptor in enumerate(self._benchmark_catalog)
        }

    def _episode_to_descriptor(self, idx: int, episode_spec: EpisodeSpec) -> dict[str, Any]:
        benchmark = episode_spec.scene_dataset
        scene_family = f"house_{episode_spec.house_index}"
        task_family = _episode_task_family(episode_spec.task, episode_spec.get_task_cls())
        variant = f"ep{idx}"
        return {
            "canonical_id": f"molmospaces:{benchmark}:{scene_family}:{task_family}:{variant}",
            "benchmark": benchmark,
            "scene_family": scene_family,
            "task_family": task_family,
            "variant": variant,
            "language": episode_spec.language.task_description,
            "objects": _extract_episode_objects(episode_spec.task),
            "privileged_requirements": [],
            "metadata": {
                "catalog_source": "real_bridge_benchmark",
                "episode_index": idx,
                "house_index": episode_spec.house_index,
                "scene_dataset": episode_spec.scene_dataset,
                "data_split": episode_spec.data_split,
                "task_cls": episode_spec.get_task_cls(),
                "robot_name": episode_spec.robot.robot_name,
                "referral_expressions": episode_spec.language.referral_expressions,
            },
        }

    def _resolve_benchmark_episode_index(
        self,
        *,
        canonical_task_id: str | None = None,
        episode_index: int | None = None,
        house_index: int | None = None,
    ) -> int:
        if episode_index is not None:
            if episode_index < 0 or episode_index >= len(self._benchmark_episodes):
                raise IndexError(
                    f"Episode index {episode_index} out of range for benchmark with "
                    f"{len(self._benchmark_episodes)} episodes"
                )
            return episode_index

        if canonical_task_id is not None:
            if canonical_task_id not in self._benchmark_index_by_canonical_id:
                raise KeyError(f"Unknown benchmark canonical task id: {canonical_task_id}")
            return self._benchmark_index_by_canonical_id[canonical_task_id]

        if house_index is not None:
            for idx, episode in enumerate(self._benchmark_episodes):
                if episode.house_index == house_index:
                    return idx
            raise KeyError(f"No benchmark episode found for house index {house_index}")

        return 0

    def _build_benchmark_config(self, episode_spec: EpisodeSpec) -> _RATSJsonBenchmarkConfig:
        return _RATSJsonBenchmarkConfig(
            task_type=episode_spec.get_task_type() or self._task_type,
            scene_dataset=episode_spec.scene_dataset,
            data_split=episode_spec.data_split,
            task_horizon=self._max_steps,
            seed=self._seed if self._seed is not None else episode_spec.seed,
            camera_config=_build_rats_camera_system(
                randomize_agentview=self._randomize_agentview,
                use_recorded_cameras=self._use_recorded_cameras,
            ),
            output_dir=Path(os.environ.get("RATS_MOLMOSPACES_BRIDGE_OUTPUT_DIR", "outputs/molmospaces_bridge")),
        )

    def _build_benchmark_task(self, episode_spec: EpisodeSpec, episode_index: int) -> None:
        if self._sampler is not None:
            try:
                self._sampler.close()
            except Exception:
                logger.warning("Failed to close previous benchmark sampler cleanly", exc_info=True)

        self._config = self._build_benchmark_config(episode_spec)
        sampler = JsonEvalTaskSampler(self._config, episode_spec)
        task = _sample_task_compat(
            sampler,
            house_index=episode_spec.house_index,
            variant="ceiling",
        )
        if task is None:
            raise RuntimeError(
                f"Failed to load benchmark episode {episode_index} "
                f"from {self._benchmark_dir}"
            )

        self._sampler = sampler
        self._task = task
        self._scene_dataset = episode_spec.scene_dataset
        self._data_split = episode_spec.data_split
        self._house_index = episode_spec.house_index
        self._current_episode_index = episode_index
        self._current_canonical_task_id = self._benchmark_catalog[episode_index]["canonical_id"]

    def _build_active_nonbenchmark_descriptor(
        self,
        canonical_task_id: str | None = None,
    ) -> dict[str, Any]:
        active_id = canonical_task_id or self._current_canonical_task_id or self._requested_canonical_task_id
        parsed = _parse_canonical_task_id(active_id)
        # PickTask.get_task_description reads referral_expressions["pickup_obj_name"].
        # If a prior sample_task replaced task_config and then raised (e.g. the
        # pinned object no longer places successfully), the entry is gone.
        # Fall back to the pinned obj name rather than propagating KeyError —
        # the descriptor still identifies the intended task.
        try:
            language = self.get_task_description()
        except KeyError:
            fallback = self._pinned_pickup_obj_name or "object"
            if self._task_type in _ARTICULATED_TASK_TYPES:
                language = f"{self._task_type.capitalize()} the {fallback}"
            else:
                language = f"Pick up the {fallback}"
            logger.warning(
                "get_task_description raised KeyError; falling back to pinned name: %s",
                language,
            )
        # Upstream OpeningTask.get_task_description hardcodes "Open the {obj}"
        # even when task_type=close (the same OpeningTask class is reused with
        # an inverted reward / different threshold). Without this rewrite,
        # every close task is labeled "Open the …" which mis-prompts the
        # planner and shows the wrong verb in iteration_*.json,
        # task_proposals/, and generated_molmospaces_specs/.  judge_success()
        # already branches on task_type so behavior is correct — only the
        # human-readable label needs fixing here.
        if (
            self._task_type == "close"
            and isinstance(language, str)
            and language.startswith("Open the ")
        ):
            language = "Close the " + language[len("Open the "):]
        descriptor: dict[str, Any] = {
            "canonical_id": active_id or "",
            "language": language,
            "metadata": {
                "catalog_source": "real_bridge_active_task",
                "task_type": self._task_type,
                "scene_dataset": self._scene_dataset,
                "data_split": self._data_split,
                "house_index": self._house_index,
            },
        }
        if parsed is not None:
            descriptor.update(parsed)
        else:
            descriptor.update(
                {
                    "benchmark": self._scene_dataset,
                    "scene_family": f"house_{self._house_index}" if self._house_index is not None else "",
                    "task_family": self._task_type,
                    "variant": "active",
                }
            )
        return descriptor

    @staticmethod
    def _camera_alias_candidates(canonical_name: str) -> list[str]:
        if canonical_name == "agentview":
            return [
                "agentview",
                "exo_camera_1",
                "exo_camera_2",
                "external_camera",
                "randomized_zed2_analogue_1",
                "randomized_zed2_analogue_2",
                "randomized_gopro_analogue_1",
                "droid_shoulder_light_randomization",
            ]
        if canonical_name == "robot0_eye_in_hand":
            return [
                "robot0_eye_in_hand",
                "wrist_camera",
                "wrist_camera_zed_mini",
                "robot_wrist_camera",
            ]
        return [canonical_name]

    def _resolve_camera_name(self, canonical_name: str) -> str | None:
        registry = getattr(self.env.camera_manager, "registry", {})
        if canonical_name in registry:
            return canonical_name

        for candidate in self._camera_alias_candidates(canonical_name):
            if candidate in registry:
                alias_key = (canonical_name, candidate)
                if alias_key not in self._logged_camera_aliases:
                    logger.info(
                        "Camera alias resolved: %s -> %s",
                        canonical_name,
                        candidate,
                    )
                    self._logged_camera_aliases.add(alias_key)
                return candidate

        registry_names = list(registry.keys())
        if canonical_name == "agentview":
            for name in registry_names:
                lowered = name.lower()
                if (
                    lowered.startswith("exo_camera")
                    or "exo" in lowered
                    or "agent" in lowered
                    or "zed2" in lowered
                    or "gopro" in lowered
                    or "shoulder" in lowered
                ):
                    alias_key = (canonical_name, name)
                    if alias_key not in self._logged_camera_aliases:
                        logger.info("Camera alias resolved: %s -> %s", canonical_name, name)
                        self._logged_camera_aliases.add(alias_key)
                    return name
        elif canonical_name == "robot0_eye_in_hand":
            for name in registry_names:
                lowered = name.lower()
                if "wrist" in lowered or "eye_in_hand" in lowered or "robot_mounted" in lowered:
                    alias_key = (canonical_name, name)
                    if alias_key not in self._logged_camera_aliases:
                        logger.info("Camera alias resolved: %s -> %s", canonical_name, name)
                        self._logged_camera_aliases.add(alias_key)
                    return name

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def env(self) -> CPUMujocoEnv:
        return self._sampler.env

    @property
    def task(self) -> BaseMujocoTask:
        assert self._task is not None
        return self._task

    @property
    def robot(self) -> FrankaRobot:
        return self.env.robots[0]

    @property
    def robot_view(self):
        return self.robot.robot_view

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the current task episode.

        When ``reset_physical_state`` is enabled (default), re-samples the
        task on the current house so the underlying MuJoCo physics state
        (drawer joint angles, object placements, robot pose) is restored.
        `task.reset()` alone only clears Python-side bookkeeping and leaves
        success-threshold-crossing state intact, which makes the next attempt
        terminate immediately. When disabled, behaviour matches the previous
        bookkeeping-only reset.
        """
        if self._reset_physical_state:
            if self._benchmark_episodes and self._current_episode_index is not None:
                self._build_benchmark_task(
                    self._benchmark_episodes[self._current_episode_index],
                    self._current_episode_index,
                )
            elif self._sampler is not None:
                self._resample_on_reset()
        result = self.task.reset()
        if result is None:
            return {}, {}
        obs_list, info_list = result
        obs = obs_list[0] if isinstance(obs_list, list) else (obs_list or {})
        info = info_list[0] if isinstance(info_list, list) else (info_list or {})
        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step the simulator with a robot action dict.

        Args:
            action: Dict with keys like ``"arm"`` (7-dim joint targets) and
                ``"gripper"`` (1-dim gripper target).

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
        """
        obs_list, reward_arr, terminated_arr, truncated_arr, info_list = self.task.step(action)
        obs = obs_list[0] if isinstance(obs_list, list) else obs_list
        reward = float(reward_arr[0]) if hasattr(reward_arr, "__getitem__") else float(reward_arr)
        terminated = bool(terminated_arr[0]) if hasattr(terminated_arr, "__getitem__") else bool(terminated_arr)
        truncated = bool(truncated_arr[0]) if hasattr(truncated_arr, "__getitem__") else bool(truncated_arr)
        info = info_list[0] if isinstance(info_list, list) else info_list
        return obs, reward, terminated, truncated, info

    def build_observation(self) -> dict[str, Any]:
        """Build a RATS-compatible observation dict from the current simulator state.

        Matches the observation format from FrankaLiberoEnv.get_observation():
        - agentview: images.rgb, images.depth, intrinsics, pose_mat
        - robot0_eye_in_hand: same structure
        - robot_cartesian_pos: (8,) xyz + wxyz + gripper
        - robot_joint_pos: (8,) 7 joints + gripper
        - robot_base_pose: (7,) xyz + wxyz
        """
        obs: dict[str, Any] = {}
        env = self.env
        camera_names = ["agentview", "robot0_eye_in_hand"]

        for cam_name in camera_names:
            cam_obs: dict[str, Any] = {"images": {}}
            resolved_name = self._resolve_camera_name(cam_name)

            if resolved_name is not None:
                camera = env.camera_manager.registry[resolved_name]

                rgb = env.render_rgb_frame(resolved_name)
                cam_obs["images"]["rgb"] = rgb

                try:
                    depth = env.render_depth_frame(resolved_name)
                    cam_obs["images"]["depth"] = depth
                except Exception:
                    cam_obs["images"]["depth"] = np.zeros(rgb.shape[:2], dtype=np.float32)

                cam2world = camera.get_pose()
                cam_obs["pose_mat"] = cam2world.astype(np.float64)

                height, width = rgb.shape[:2]
                fovy = camera.fov
                f = (height / 2.0) / np.tan(np.radians(fovy / 2.0))
                K = np.array(
                    [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]],
                    dtype=np.float64,
                )
                cam_obs["intrinsics"] = K
            else:
                cam_obs["images"]["rgb"] = np.zeros(
                    (self._render_height, self._render_width, 3), dtype=np.uint8
                )
                cam_obs["images"]["depth"] = np.zeros(
                    (self._render_height, self._render_width), dtype=np.float32
                )
                cam_obs["intrinsics"] = np.eye(3, dtype=np.float64)
                cam_obs["pose_mat"] = np.eye(4, dtype=np.float64)

            obs[cam_name] = cam_obs

        arm_mg = self.robot_view.get_move_group("arm")
        gripper_mg = self.robot_view.get_move_group("gripper")

        joint_pos_7 = np.array(arm_mg.joint_pos, dtype=np.float64)
        gripper_dist = gripper_mg.inter_finger_dist
        gripper_max = gripper_mg.inter_finger_dist_range[1]
        gripper_frac = gripper_dist / gripper_max if gripper_max > 0 else 0.0
        obs["robot_joint_pos"] = np.concatenate([joint_pos_7, [gripper_frac]])

        ee_pose_mat = arm_mg.leaf_frame_to_world
        ee_pos = ee_pose_mat[:3, 3]
        ee_rot = SciRotation.from_matrix(ee_pose_mat[:3, :3])
        ee_quat_xyzw = ee_rot.as_quat()
        ee_quat_wxyz = np.array(
            [ee_quat_xyzw[3], ee_quat_xyzw[0], ee_quat_xyzw[1], ee_quat_xyzw[2]],
            dtype=np.float64,
        )
        obs["robot_cartesian_pos"] = np.concatenate([ee_pos, ee_quat_wxyz, [gripper_frac]])

        base_pose_mat = np.asarray(self.robot_view.base.pose, dtype=np.float64)
        base_rot = base_pose_mat[:3, :3]
        base_pos = base_pose_mat[:3, 3].copy()
        config = getattr(self, "_config", None)
        robot_config = getattr(config, "robot_config", None) if config is not None else None
        base_size = getattr(robot_config, "base_size", None)
        if base_size is not None and len(base_size) >= 3:
            mount_offset_local = np.array([0.0, 0.0, float(base_size[2])], dtype=np.float64)
            base_pos = base_pos + base_rot @ mount_offset_local
        base_quat_xyzw = SciRotation.from_matrix(base_rot).as_quat()
        base_quat_wxyz = np.array(
            [base_quat_xyzw[3], base_quat_xyzw[0], base_quat_xyzw[1], base_quat_xyzw[2]],
            dtype=np.float64,
        )
        obs["robot_base_pose"] = np.concatenate([base_pos, base_quat_wxyz])

        return obs

    def judge_success(self) -> bool:
        return self.task.judge_success()

    def get_reward(self) -> float:
        reward = self.task.get_reward()
        return float(reward[0]) if hasattr(reward, "__getitem__") else float(reward)

    def get_info(self) -> dict[str, Any]:
        """Per-subgoal metrics for the current task (first batch element).

        Shape varies by task family:
          - pick:               {position_error, rotation_error, success, episode_step}
          - pick_and_place:     {position_error, success, supported_by_receptacle,
                                 supported_by_carry_forward, robot_contact,
                                 receptacle_pos_displacement, receptacle_rot_displacement,
                                 receptacle_tilt_displacement, episode_step, ...}
          - pick_and_place_next_to: similar to pick_and_place
          - opening / closing:  {joint_position, success, episode_step}
          - nav:                {..., success, episode_step}

        Returns an empty dict if the underlying task does not implement
        get_info() or raises. All numpy scalars are coerced to Python
        primitives so the dict round-trips cleanly through msgpack.
        """
        try:
            infos = self.task.get_info()
        except Exception:
            return {}
        if not isinstance(infos, (list, tuple)) or not infos:
            return {}
        info = infos[0]
        if not isinstance(info, dict):
            return {}

        coerced: dict[str, Any] = {}
        for key, val in info.items():
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    coerced[key] = float(val.reshape(-1)[0])
                else:
                    coerced[key] = val.astype(np.float64, copy=False)
            elif isinstance(val, (np.floating, np.integer)):
                coerced[key] = float(val)
            elif isinstance(val, np.bool_):
                coerced[key] = bool(val)
            else:
                coerced[key] = val
        return coerced

    def describe_object_relation(self, pickup_obj_name: str, receptacle_name: str) -> dict[str, Any]:
        """Return privileged MuJoCo support/contact relation for an object pair.

        This mirrors the core of MolmoSpaces PickAndPlaceTask.get_info() but is
        parameterized by arbitrary object names, so playtime/freeform place/stack
        proposals can be verified even when the currently pinned benchmark task
        is not itself a pick-and-place task for that exact pair.
        """
        try:
            from molmo_spaces.env.data_views import create_mlspaces_body
            from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
            from molmo_spaces.utils.mujoco_scene_utils import is_object_supported_by_body
        except Exception as exc:
            return {"available": False, "error": f"import_failed: {type(exc).__name__}: {exc}"}

        env = getattr(self, "env", None)
        data = getattr(env, "current_data", None)
        if env is None or data is None:
            return {"available": False, "error": "missing_mujoco_env_or_data"}
        pickup_name = str(pickup_obj_name or "").strip()
        receptacle = str(receptacle_name or "").strip()
        if not pickup_name or not receptacle:
            return {"available": False, "error": "missing_object_name"}

        try:
            pickup_obj = create_mlspaces_body(data, pickup_name)
            place_receptacle = create_mlspaces_body(data, receptacle)
        except Exception as exc:
            return {
                "available": False,
                "pickup_obj_name": pickup_name,
                "place_receptacle_name": receptacle,
                "error": f"object_lookup_failed: {type(exc).__name__}: {exc}",
            }

        try:
            pickup_aabb_center, pickup_aabb_size = body_aabb(data.model, data, pickup_obj.body_id)
            pickup_aabb_min = pickup_aabb_center - pickup_aabb_size / 2
            pickup_aabb_max = pickup_aabb_center + pickup_aabb_size / 2
            receptacle_aabb_center, receptacle_aabb_size = body_aabb(
                data.model, data, place_receptacle.body_id
            )
            pos_err = float(np.linalg.norm(
                np.maximum(0, pickup_aabb_min - receptacle_aabb_center)
                + np.maximum(0, receptacle_aabb_center - pickup_aabb_max)
            ))
        except Exception:
            pickup_aabb_center = np.asarray(getattr(pickup_obj, "position", [0.0, 0.0, 0.0]), dtype=np.float64)
            pickup_aabb_size = np.zeros(3, dtype=np.float64)
            receptacle_aabb_center = np.asarray(getattr(place_receptacle, "position", [0.0, 0.0, 0.0]), dtype=np.float64)
            receptacle_aabb_size = np.zeros(3, dtype=np.float64)
            pos_err = float(np.linalg.norm(pickup_aabb_center - receptacle_aabb_center))

        supported_by_receptacle = False
        support_source = "contact_force"
        try:
            supported_by_receptacle = bool(is_object_supported_by_body(
                data,
                pickup_obj.body_id,
                place_receptacle.body_id,
                frac_weight_threshold=0.5,
            ))
        except Exception:
            supported_by_receptacle = False
            support_source = "object_manager_receptacle_heuristic"

        if not supported_by_receptacle:
            try:
                om = env.object_managers[env.current_batch_index]
                objects_on_receptacle = om.objects_on_receptacle(
                    [om.get_object_by_name(pickup_name)],
                    om.get_object_by_name(receptacle).geom_ids,
                )
                names_on_receptacle = {obj.name for obj in objects_on_receptacle}
                supported_by_receptacle = pickup_name in names_on_receptacle
                if supported_by_receptacle:
                    support_source = "object_manager_receptacle_heuristic"
            except Exception:
                pass

        robot_contact = False
        try:
            robot_root_body_id = env.current_robot.robot_view.base.root_body_id
            for c in data.contact:
                root_body1 = data.model.body_rootid[data.model.geom_bodyid[c.geom1]]
                root_body2 = data.model.body_rootid[data.model.geom_bodyid[c.geom2]]
                if (root_body1 == pickup_obj.body_id) ^ (root_body2 == pickup_obj.body_id):
                    other_root_body = root_body1 if root_body1 != pickup_obj.body_id else root_body2
                    if other_root_body == robot_root_body_id:
                        robot_contact = True
                        break
        except Exception:
            robot_contact = False

        return {
            "available": True,
            "source": "mujoco_object_relation",
            "pickup_obj_name": pickup_name,
            "place_receptacle_name": receptacle,
            "supported_by_receptacle": bool(supported_by_receptacle),
            "support_source": support_source,
            "robot_contact": bool(robot_contact),
            "position_error": pos_err,
            "pickup_aabb_center": np.asarray(pickup_aabb_center, dtype=np.float64),
            "pickup_aabb_size": np.asarray(pickup_aabb_size, dtype=np.float64),
            "receptacle_aabb_center": np.asarray(receptacle_aabb_center, dtype=np.float64),
            "receptacle_aabb_size": np.asarray(receptacle_aabb_size, dtype=np.float64),
        }

    def describe_contact_pairs(self, *, max_pairs: int = 64) -> dict[str, Any]:
        """Return compact MuJoCo contact pairs involving the robot end effector.

        The privileged playtime checker uses this as exact simulator contact
        evidence instead of inferring "near grasp" from root-object distances.
        Contacts are exact for the MuJoCo collision geoms currently active in
        the scene.  We keep the payload compact and JSON/msgpack friendly so it
        can be sampled every dense state-trace step, including through the
        remote bridge.
        """
        env = getattr(self, "env", None)
        model = getattr(env, "current_model", None)
        data = getattr(env, "current_data", None)
        if model is None or data is None:
            return {
                "available": False,
                "not_available_reason": "missing_mujoco_model_or_data",
                "robot_eef_contact_pairs": [],
                "robot_eef_contact_pair_count": 0,
                "ncon": 0,
            }

        finger_geom_ids = self._gripper_finger_geom_ids()
        pairs: list[dict[str, Any]] = []
        ncon = int(getattr(data, "ncon", 0) or 0)
        for idx in range(ncon):
            try:
                contact = data.contact[idx]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                side1_eef = self._is_robot_eef_geom(model, geom1, finger_geom_ids)
                side2_eef = self._is_robot_eef_geom(model, geom2, finger_geom_ids)
                if not (side1_eef or side2_eef):
                    continue
                pair = self._contact_pair_payload(model, contact, idx, side1_eef, side2_eef)
                pairs.append(pair)
                if len(pairs) >= int(max_pairs):
                    break
            except Exception:
                continue
        return {
            "available": True,
            "source": "mujoco.current_data.contact",
            "ncon": ncon,
            "robot_eef_contact_pair_count": len(pairs),
            "robot_eef_contact_pairs": pairs,
        }

    def _gripper_finger_geom_ids(self) -> set[int]:
        try:
            gripper_group = self.robot_view.get_move_group("gripper")
        except Exception:
            return set()
        ids: set[int] = set()
        for attr in ("_finger_1_geom_id", "_finger_2_geom_id"):
            if hasattr(gripper_group, attr):
                try:
                    ids.add(int(getattr(gripper_group, attr)))
                except Exception:
                    pass
        return ids

    @staticmethod
    def _mj_name(model: Any, obj_type: Any, obj_id: int) -> str:
        try:
            name = mujoco.mj_id2name(model, obj_type, int(obj_id))
        except Exception:
            name = None
        return str(name or "")

    @classmethod
    def _geom_body_name(cls, model: Any, geom_id: int) -> tuple[int | None, str]:
        try:
            body_id = int(model.geom_bodyid[int(geom_id)])
        except Exception:
            return None, ""
        return body_id, cls._mj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

    @classmethod
    def _is_robot_eef_geom(cls, model: Any, geom_id: int, finger_geom_ids: set[int]) -> bool:
        if int(geom_id) in finger_geom_ids:
            return True
        geom_name = cls._mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)).lower()
        _body_id, body_name = cls._geom_body_name(model, int(geom_id))
        text = f"{geom_name} {body_name.lower()}"
        # Prefer actual finger IDs when available.  These name heuristics cover
        # remote/procedural variants where the move-group does not expose IDs.
        return bool(
            "robot_0/" in text
            and any(
                token in text
                for token in (
                    "finger",
                    "gripper",
                    "hand",
                    "wrist",
                    "eef",
                    "ee_link",
                    "tool",
                    "palm",
                    "robotiq",
                )
            )
        )

    @classmethod
    def _contact_pair_payload(
        cls,
        model: Any,
        contact: Any,
        contact_index: int,
        geom1_is_eef: bool,
        geom2_is_eef: bool,
    ) -> dict[str, Any]:
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1_id, body1_name = cls._geom_body_name(model, geom1)
        body2_id, body2_name = cls._geom_body_name(model, geom2)
        pos = getattr(contact, "pos", None)
        try:
            pos_list = np.asarray(pos, dtype=np.float64).reshape(-1)[:3].tolist()
        except Exception:
            pos_list = []
        if geom1_is_eef and geom2_is_eef:
            robot_side = "both"
        elif geom1_is_eef:
            robot_side = "geom1"
        elif geom2_is_eef:
            robot_side = "geom2"
        else:
            robot_side = "none"
        return {
            "contact_index": int(contact_index),
            "geom1": geom1,
            "geom2": geom2,
            "geom1_name": cls._mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "geom2_name": cls._mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "body1": body1_id,
            "body2": body2_id,
            "body1_name": body1_name,
            "body2_name": body2_name,
            "dist": float(getattr(contact, "dist", 0.0)),
            "pos": pos_list,
            "robot_side": robot_side,
        }

    def render(self, camera_name: str = "agentview") -> np.ndarray:
        """Return RGB frame from the named camera."""
        resolved_name = self._resolve_camera_name(camera_name)
        if resolved_name is not None:
            return self.env.render_rgb_frame(resolved_name)
        return np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)

    def render_wrist(self) -> np.ndarray:
        return self.render("robot0_eye_in_hand")

    def get_task_description(self) -> str:
        return self.task.get_task_description()

    def describe_scene_obstacles(
        self,
        *,
        exclude_internal_names: list[str] | None = None,
        max_distance_m: float = 3.0,
    ) -> list[dict[str, Any]]:
        """Return AABB-box obstacles around the robot for pyroki collision IK.

        Used by RATS-side ``goto_pose`` to feed pyroki's collision-aware
        trajopt solver. The default basic IK has no obstacle info and
        the straight-line joint-interp from current to target pose can
        plow through chairs / table legs / other objects in the scene.
        Pyroki's ``solve_trajopt`` accepts a list of
        ``CollGeom`` obstacles and routes around them — but the server
        needs to know what they ARE first. That's this method.

        Each obstacle is a ``{"type": "box", "position": [x,y,z],
        "extent": [dx, dy, dz], "internal_name": str}`` dict matching the
        pyroki server's ObstacleEntry schema (see launch_pyroki_server.py
        :_build_world_coll).

        Filters:
          - ``exclude_internal_names``: drop anchored target / receptacle
            etc. from the obstacle list (we WANT the robot to touch
            them; treating them as obstacles makes IK fail).
          - ``max_distance_m``: drop anything outside this radius from
            the robot base. procthor houses have 100+ objects; sending
            all of them inflates the payload and slows the solver.
        """
        try:
            from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
            from molmo_spaces.utils.constants.object_constants import (
                PICK_AND_PLACE_OBJECTS,
                RECEPTACLE_TYPES_THOR,
                ITHOR_ARTICULATED_OBJECTS,
            )
        except Exception:
            return []
        if self.env is None:
            return []
        excluded = set(exclude_internal_names or [])
        try:
            # MolmoSpaces env exposes the live mujoco data handle as
            # ``current_data`` (not ``data`` — that pattern is used elsewhere
            # in this file at line 2048 and was the bug in this method's
            # first cut, which made every call return [] silently and the
            # RATS-side ``_try_goto_pose_via_trajopt`` early-return False
            # without firing /plan).
            data = getattr(self.env, "current_data", None)
            model = data.model if data is not None else None
            if data is None or model is None:
                return []
            om = self.env.object_managers[self.env.current_batch_index]
        except Exception:
            return []
        # Robot base position for distance filter.
        robot_base = np.zeros(3, dtype=np.float64)
        try:
            base = getattr(self.env, "robot_base_pose", None)
            if base is not None and len(base) >= 3:
                robot_base = np.asarray(base[:3], dtype=np.float64)
        except Exception:
            pass
        obstacles: list[dict[str, Any]] = []
        # ObjectManager doesn't expose a single "all objects" iterator;
        # enumerate by category union (pickables + receptacles +
        # articulated objects). Mirrors the bridge's own object-typed
        # access pattern used elsewhere in this file (e.g. the inventory
        # describe call uses ``om.get_objects_of_type(PICK_AND_PLACE_
        # OBJECTS)``). The earlier draft used ``getattr(om, "objects",
        # [])`` which silently returned [] because ObjectManager has no
        # such attribute.
        candidate_types: list[str] = []
        for type_list in (
            PICK_AND_PLACE_OBJECTS,
            RECEPTACLE_TYPES_THOR,
            ITHOR_ARTICULATED_OBJECTS,
        ):
            try:
                candidate_types.extend(list(type_list or []))
            except Exception:
                pass
        # De-dupe while preserving order — get_objects_of_type accepts
        # a Collection so dupes only waste cycles.
        seen_types: set[str] = set()
        unique_types = [t for t in candidate_types if not (t in seen_types or seen_types.add(t))]
        try:
            candidate_objs = om.get_objects_of_type(unique_types)
        except Exception:
            candidate_objs = []
        seen_names: set[str] = set()
        for obj in candidate_objs:
            try:
                internal_name = getattr(obj, "name", "") or ""
                if internal_name in excluded or internal_name in seen_names:
                    continue
                seen_names.add(internal_name)
                body_id = getattr(obj, "body_id", None)
                if body_id is None:
                    continue
                center, size = body_aabb(model, data, body_id)
                center = np.asarray(center, dtype=np.float64)
                size = np.asarray(size, dtype=np.float64)
                if (size <= 0).all():
                    continue
                if float(np.linalg.norm(center - robot_base)) > max_distance_m:
                    continue
                obstacles.append(
                    {
                        "type": "box",
                        "internal_name": internal_name,
                        "position": center.tolist(),
                        "extent": size.tolist(),
                    }
                )
            except Exception:
                continue
        # Floor as a halfspace at z=0 (normal +Z) so the planner never
        # routes through the floor.
        obstacles.append(
            {
                "type": "halfspace",
                "internal_name": "_floor",
                "point": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
            }
        )
        return obstacles

    def get_anchored_task_target(self) -> dict[str, Any]:
        """Return the sampler's currently anchored task target.

        ``self._sampler.sample_task(...)`` (called inside init and on every
        ``request_new_house``) picks a concrete pickable / receptacle /
        articulation AND places the robot to be able to reach it. RATS-side
        playtime mode was throwing this anchor info away: the LLM proposer
        was given the full house inventory (122+ items) and the per-item
        reach filter then rejected ~90% of it as "out of reach (>1.5m)",
        leaving 0 valid taskable targets even though the bridge had a
        perfectly reachable anchor selected.

        Reads the LIVE sampler ``task_config`` so the values reflect the
        most recent ``sample_task`` call. Falls back to the cached
        ``_pinned_*`` snapshot (which only refreshes on ``request_new_house``
        when ``pin_pickup_obj_name`` is on) so callers still get something
        useful even when the sampler config isn't introspectable.

        Returns:
            ``{"pickup_obj_name", "place_receptacle_name", "joint_name",
              "joint_index", "task_type", "house_index"}`` — any value may
            be ``None`` if the current sampler doesn't have a corresponding
            anchored identity.
        """
        live: dict[str, Any] = {}
        if self._sampler is not None:
            cfg = getattr(self._sampler.config, "task_config", None)
            if cfg is not None:
                live["pickup_obj_name"] = getattr(cfg, "pickup_obj_name", None)
                live["place_receptacle_name"] = getattr(cfg, "place_receptacle_name", None)
                live["joint_name"] = getattr(cfg, "joint_name", None)
                live["joint_index"] = getattr(cfg, "joint_index", None)
        return {
            # Prefer live task_config values; fall back to pinned cache.
            "pickup_obj_name": live.get("pickup_obj_name") or self._pinned_pickup_obj_name,
            "place_receptacle_name": live.get("place_receptacle_name") or self._pinned_place_receptacle_name,
            "joint_name": live.get("joint_name") or self._pinned_joint_name,
            "joint_index": live.get("joint_index") if live.get("joint_index") is not None else self._pinned_joint_index,
            "task_type": self._task_type,
            "house_index": self._house_index,
        }

    def anchor_to_pickup(self, target_internal_name: str) -> dict[str, Any]:
        """Re-anchor the currently loaded house to a different pickable target.

        Decouples target selection from the sampler's round-robin
        ``_select_pickup_object`` so RATS-side curiosity can drive which
        object the robot is placed near. Stays in the same house — this
        is NOT a request_new_house — only switches:
          1. ``task_config.pickup_obj_name`` (so verifier / grasp paths
             reference the chosen object)
          2. The robot base pose (via ``env.place_robot_near`` for the
             chosen target's world position)
          3. The pinned-target cache (so ``get_anchored_task_target``
             returns the new identity)

        Camera setup is intentionally NOT redone — exo cameras are
        workspace-anchored (see ``setup_eval_cameras``) so they track
        the new workspace center automatically; wrist cam is mounted
        on the robot and doesn't need re-init.

        Raises ``ValueError`` if ``target_internal_name`` is not a
        pickable object in the current scene. Returns the updated
        anchored-task descriptor so the caller can immediately read
        ``pickup_obj_name`` / 3D pose / etc.
        """
        if self._sampler is None:
            raise RuntimeError("anchor_to_pickup called before sampler init")
        if not target_internal_name:
            raise ValueError("anchor_to_pickup requires a non-empty target name")
        env = getattr(self, "env", None)
        if env is None:
            raise RuntimeError("anchor_to_pickup called before env loaded")

        # Resolve the object in the current scene's ObjectManager.
        om = env.object_managers[env.current_batch_index]
        try:
            obj_handle = om.get_object_by_name(target_internal_name)
        except (KeyError, AttributeError) as exc:
            raise ValueError(
                f"anchor_to_pickup: '{target_internal_name}' not found in current "
                f"scene (house={self._house_index}): {exc}"
            )
        # Pull the target's world position. The 3D pose is what
        # place_robot_near needs; we don't need orientation.
        try:
            target_pose = obj_handle.pose  # 4x4 or (xyz+quat) depending on env
            target_xyz = (
                target_pose[:3, 3]
                if hasattr(target_pose, "shape") and target_pose.shape == (4, 4)
                else target_pose[:3]
            )
        except Exception as exc:
            raise RuntimeError(
                f"anchor_to_pickup: could not read pose for {target_internal_name}: {exc}"
            )

        # 1. Update task_config so the rest of the bridge sees the new identity.
        cfg = getattr(self._sampler.config, "task_config", None)
        if cfg is not None and hasattr(cfg, "pickup_obj_name"):
            cfg.pickup_obj_name = target_internal_name

        # 2. Re-place the robot near the new target.
        robot_view = env.current_robot.robot_view
        env.place_robot_near(
            robot_view=robot_view,
            target=target_xyz,
            face_target=True,
        )

        # 3. Refresh pinned cache so get_anchored_task_target returns the
        # new identity.
        self._pinned_pickup_obj_name = target_internal_name

        logger.info(
            "[anchor_to_pickup] re-anchored house %s to pickup=%s at xyz=(%.3f, %.3f, %.3f)",
            self._house_index, target_internal_name,
            float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]),
        )
        return self.get_anchored_task_target()

    def list_task_descriptors(self) -> list[dict[str, Any]]:
        if self._benchmark_catalog:
            return [dict(descriptor) for descriptor in self._benchmark_catalog]
        return [self._build_active_nonbenchmark_descriptor()]

    def get_task_metadata(self) -> dict[str, Any]:
        if self._current_episode_index is not None and self._benchmark_catalog:
            descriptor = self._benchmark_catalog[self._current_episode_index]
            return dict(descriptor)
        return self._build_active_nonbenchmark_descriptor()

    def get_task_descriptor(self, canonical_task_id: str | None = None) -> dict[str, Any]:
        if canonical_task_id is not None and self._benchmark_catalog:
            if canonical_task_id not in self._benchmark_index_by_canonical_id:
                raise KeyError(f"Unknown benchmark canonical task id: {canonical_task_id}")
            return dict(
                self._benchmark_catalog[
                    self._benchmark_index_by_canonical_id[canonical_task_id]
                ]
            )
        return self._build_active_nonbenchmark_descriptor(canonical_task_id=canonical_task_id)

    def resample_task(
        self,
        *,
        house_index: int | None = None,
        canonical_task_id: str | None = None,
        episode_index: int | None = None,
    ) -> None:
        """Sample a new task (possibly in a new scene)."""
        if self._benchmark_episodes:
            resolved_index = self._resolve_benchmark_episode_index(
                canonical_task_id=canonical_task_id,
                episode_index=episode_index,
                house_index=house_index,
            )
            self._build_benchmark_task(
                self._benchmark_episodes[resolved_index],
                resolved_index,
            )
            return
        # An explicit resample_task is the user asking for a fresh draw, so
        # forget any previously pinned identity before re-sampling.
        self._strict_pinned_task_identity = False
        self._pinned_pickup_obj_name = None
        old_place_receptacle_name = self._pinned_place_receptacle_name
        self._pinned_place_receptacle_name = None
        self._pinned_joint_index = None
        self._pinned_joint_name = None
        self._pinned_referral_expressions = None
        if old_place_receptacle_name is not None:
            self._set_sampler_place_receptacle_name(
                self._default_sampler_place_receptacle_name()
            )
        if self._sampler is not None:
            for cfg_attr in ("task_config", "task_config_preset_scn"):
                cfg = getattr(self._sampler.config, cfg_attr, None)
                if cfg is not None and hasattr(cfg, "pickup_obj_name"):
                    cfg.pickup_obj_name = None

        self._task = _sample_task_compat(
            self._sampler,
            house_index=house_index,
            variant="ceiling",
        )
        if self._task is None:
            raise RuntimeError("Failed to sample new task from MolmoSpaces")
        if canonical_task_id is not None:
            self._current_canonical_task_id = canonical_task_id
        elif self._requested_canonical_task_id is not None:
            self._current_canonical_task_id = self._requested_canonical_task_id
        self._capture_pinned_task_identity()

    def close(self) -> None:
        if self._sampler is not None and hasattr(self._sampler, "env") and self._sampler.env is not None:
            try:
                self._sampler.env.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Inventory introspection (drives the open-mode task proposer)
    # ------------------------------------------------------------------

    def _joint_inventory_entry_from_model(
        self,
        model: Any,
        joint_name_or_id: str | int,
        *,
        index_hint: int | None = None,
    ) -> dict[str, Any] | None:
        """Return a JSON-friendly hinge/slide joint entry from MuJoCo state.

        This intentionally bypasses MolmoSpaces' object-manager articulation
        wrapper. Some fixed iTHOR benchmark scenes expose an openable object as
        a normal receptacle in the object manager even though the underlying
        MJCF contains a hinge/slide joint and the benchmark episode targets it.
        """
        try:
            if isinstance(joint_name_or_id, str):
                joint_id = int(
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        joint_name_or_id,
                    )
                )
            else:
                joint_id = int(joint_name_or_id)
        except Exception:
            return None
        if joint_id < 0:
            return None

        try:
            jtype_int = int(model.jnt_type[joint_id])
        except Exception:
            return None
        if jtype_int == int(mujoco.mjtJoint.mjJNT_HINGE):
            jtype_str = "hinge"
        elif jtype_int == int(mujoco.mjtJoint.mjJNT_SLIDE):
            jtype_str = "slide"
        else:
            return None

        try:
            jname = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        except Exception:
            jname = str(joint_name_or_id)
        entry: dict[str, Any] = {
            "index": int(index_hint if index_hint is not None else joint_id),
            "model_joint_id": joint_id,
            "name": jname,
            "type": jtype_str,
            "source": "mujoco_model",
        }
        try:
            entry["range"] = [float(x) for x in model.jnt_range[joint_id].tolist()]
        except Exception:
            pass
        try:
            entry["axis"] = [float(x) for x in model.jnt_axis[joint_id].tolist()]
        except Exception:
            pass
        try:
            qposadr = int(model.jnt_qposadr[joint_id])
            entry["current_position"] = float(self.env.current_data.qpos[qposadr])
        except Exception:
            pass
        return entry

    def _benchmark_articulation_hints(self) -> dict[str, dict[str, Any]]:
        """Articulation roots known from benchmark metadata / allowlists."""
        hints: dict[str, dict[str, Any]] = {}

        def ensure(root: str, *, source: str) -> dict[str, Any]:
            root = str(root or "").strip()
            if not root:
                return {}
            item = hints.setdefault(root, {"internal_name": root, "sources": []})
            if source not in item["sources"]:
                item["sources"].append(source)
            return item

        if self._current_episode_index is not None and self._benchmark_episodes:
            try:
                episode = self._benchmark_episodes[self._current_episode_index]
                task = episode.task or {}
            except Exception:
                task = {}
            task_type = str(task.get("task_type") or "").strip().lower()
            task_cls = str(task.get("task_cls") or "")
            if task_type in _ARTICULATED_TASK_TYPES or "OpeningTask" in task_cls:
                item = ensure(task.get("pickup_obj_name"), source="benchmark_episode")
                if item:
                    joint_name = str(task.get("joint_name") or "").strip()
                    if joint_name:
                        item.setdefault("joint_names", []).append(joint_name)
                    if task.get("joint_index") is not None:
                        item["joint_index"] = task.get("joint_index")

        if self._benchmark_dir is not None:
            allowlist_path = Path(self._benchmark_dir) / "mutation-allowlist.json"
            try:
                allowlist = json.loads(allowlist_path.read_text())
            except Exception:
                allowlist = {}
            for root in allowlist.get("articulations") or []:
                ensure(root, source="mutation_allowlist")

        return hints

    def _merge_fallback_articulations(
        self,
        *,
        articulations: list[dict[str, Any]],
        pickables: list[dict[str, Any]],
        receptacles: list[dict[str, Any]],
        model: Any,
        scene_metadata: dict[str, Any],
        name_to_room: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Merge benchmark/model-discovered articulations into inventory.

        The normal object-manager path is still preferred.  This fallback fills
        holes where a scene object is categorized as a receptacle/static object
        while the underlying MuJoCo model and benchmark metadata expose an
        openable hinge/slide joint.
        """
        by_name: dict[str, dict[str, Any]] = {
            str(entry.get("internal_name")): dict(entry)
            for entry in articulations
            if isinstance(entry, dict) and entry.get("internal_name")
        }
        metadata_objects = scene_metadata.get("objects", {})
        if not isinstance(metadata_objects, dict):
            metadata_objects = {}
        inventory_meta: dict[str, dict[str, Any]] = {}
        for entry in list(receptacles) + list(pickables) + list(articulations):
            if isinstance(entry, dict) and entry.get("internal_name"):
                inventory_meta[str(entry["internal_name"])] = entry

        hints = self._benchmark_articulation_hints()
        known_scene_roots = set(inventory_meta) | {
            str(name) for name in metadata_objects.keys()
        }
        hinted_roots = set(hints)
        joint_names_by_root: dict[str, set[str]] = {
            root: set(str(j) for j in (hint.get("joint_names") or []) if str(j))
            for root, hint in hints.items()
        }

        # Direct MuJoCo scan: recover hinge/slide joints whose root object is a
        # real scene object, including benchmark/allowlist hints. This avoids
        # depending on object_manager.get_objects_of_type(...articulation...).
        try:
            njnt = int(model.njnt)
        except Exception:
            njnt = 0
        for joint_id in range(njnt):
            entry = self._joint_inventory_entry_from_model(model, joint_id)
            if not entry:
                continue
            joint_name = str(entry.get("name") or "")
            root = _root_object_internal_name(joint_name)
            if root not in known_scene_roots and root not in hinted_roots:
                continue
            joint_names_by_root.setdefault(root, set()).add(joint_name)

        for root, joint_names in joint_names_by_root.items():
            if not root:
                continue
            base = by_name.get(root, {"internal_name": root})
            meta = inventory_meta.get(root)
            scene_obj_meta = metadata_objects.get(root, {})
            if isinstance(meta, dict):
                base.setdefault("category", meta.get("category") or "")
                base.setdefault("position", meta.get("position") or [0.0, 0.0, 0.0])
                base.setdefault("room", meta.get("room") or name_to_room.get(root, "unknown"))
            if isinstance(scene_obj_meta, dict):
                base.setdefault("category", str(scene_obj_meta.get("category") or "").lower())
                pose = scene_obj_meta.get("pose") or scene_obj_meta.get("position")
                if isinstance(pose, (list, tuple)) and len(pose) >= 3:
                    base.setdefault("position", [float(x) for x in pose[:3]])
            base.setdefault("category", root.split("_", 1)[0].lower())
            base.setdefault("position", [0.0, 0.0, 0.0])
            base.setdefault("room", name_to_room.get(root, "unknown"))

            joints: list[dict[str, Any]] = []
            seen_joint_names: set[str] = set()
            for existing in base.get("joints") or []:
                if not isinstance(existing, dict):
                    continue
                name = str(existing.get("name") or "")
                if name:
                    seen_joint_names.add(name)
                joints.append(existing)
            for joint_name in sorted(joint_names):
                if joint_name in seen_joint_names:
                    continue
                joint_entry = self._joint_inventory_entry_from_model(model, joint_name)
                if joint_entry:
                    joint_entry["source"] = "articulations_fallback"
                    joints.append(joint_entry)
            hint = hints.get(root) or {}
            sources = list(base.get("sources") or [])
            for source in hint.get("sources") or []:
                if source not in sources:
                    sources.append(source)
            if joints and "mujoco_model" not in sources:
                sources.append("mujoco_model")
            base["joints"] = joints
            base["sources"] = sources or ["articulations_fallback"]
            by_name[root] = base

        return list(by_name.values())

    def describe_scene_inventory(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the active house's contents.

        Used by the rats novel-task proposer (`_propose_novel_molmospaces_open`)
        so the LLM sees what's actually in the loaded scene rather than a
        static catalog. Three lists:

          - ``pickables``: free-bodied small objects with a grasp file.
          - ``receptacles``: objects exposing a receptacle site
            (tables, counters, plates, shelves).
          - ``articulations``: objects with at least one hinge / slide
            joint (drawers, cabinets, microwave doors, …) plus the joint
            indices the bridge can target via ``set_task_from_spec``.

        ``has_grasp_file`` on pickables is the same gate
        ``PickTaskSampler._get_scene_objects`` applies — entries with
        ``False`` will not survive the sampler's own filter, so the
        proposer should avoid them.
        """
        from molmo_spaces.env.data_views import MlSpacesArticulationObject
        from molmo_spaces.utils.asset_names import get_thor_name
        from molmo_spaces.utils.constants.object_constants import (
            EXTENDED_ARTICULATION_TYPES_THOR,
            PICK_AND_PLACE_OBJECTS,
        )
        from molmo_spaces.utils.grasps import (
            has_pickup_grasp_path,
            has_valid_pickup_grasps,
        )

        env = self.env
        om = env.object_managers[env.current_batch_index]
        model = env.current_model
        scene_metadata = env.current_scene_metadata or {}
        room_to_objects = self._extract_room_layout(scene_metadata)

        # Reverse-lookup: object name -> room name. Procthor scenes
        # expose this; ithor / holodeck may not, in which case room
        # entries come back as "unknown".
        name_to_room: dict[str, str] = {}
        for room, names in room_to_objects.items():
            for n in names:
                name_to_room[n] = room

        pickables: list[dict[str, Any]] = []
        receptacles: list[dict[str, Any]] = []
        articulations: list[dict[str, Any]] = []

        try:
            pickable_objs = om.get_objects_of_type(PICK_AND_PLACE_OBJECTS)
        except Exception:
            pickable_objs = []
        for obj in pickable_objs:
            asset_uid = (
                scene_metadata.get("objects", {}).get(obj.name, {}).get("asset_id")
            )
            if asset_uid is None:
                try:
                    asset_uid = get_thor_name(model, obj)
                except Exception:
                    asset_uid = None
            try:
                has_grasp = bool(
                    asset_uid
                    and has_pickup_grasp_path(asset_uid)
                    and has_valid_pickup_grasps(asset_uid)
                )
            except Exception:
                has_grasp = False
            try:
                position = [float(x) for x in obj.position[:3]]
            except Exception:
                position = [0.0, 0.0, 0.0]
            try:
                category = om.get_annotation_category(obj.name).lower()
            except Exception:
                category = ""
            pickables.append({
                "internal_name": obj.name,
                "category": category,
                "asset_uid": asset_uid,
                "position": position,
                "room": name_to_room.get(obj.name, "unknown"),
                "has_grasp_file": has_grasp,
            })

        try:
            receptacle_objs = om.get_receptacles()
        except Exception:
            receptacle_objs = []
        for obj in receptacle_objs:
            try:
                category = om.get_annotation_category(obj.name).lower()
            except Exception:
                category = ""
            try:
                position = [float(x) for x in obj.position[:3]]
            except Exception:
                position = [0.0, 0.0, 0.0]
            receptacles.append({
                "internal_name": obj.name,
                "category": category,
                "position": position,
                "room": name_to_room.get(obj.name, "unknown"),
            })

        # Articulations: enumerate via the same path OpenTaskSampler uses
        # (om.get_objects_of_type(EXTENDED_ARTICULATION_TYPES_THOR) plus
        # an isinstance(MlSpacesArticulationObject) filter). The previous
        # top_level_objects walk missed every articulated asset on
        # procthor-10k scenes because cabinets / drawers / fridges live
        # several levels deep in the room hierarchy and never surface as
        # top-level bodies; OpenTaskSampler-style category lookup is the
        # source of truth and matches what set_task_from_spec can
        # actually instantiate.
        try:
            articulated_candidates = om.get_objects_of_type(
                EXTENDED_ARTICULATION_TYPES_THOR
            )
        except Exception:
            articulated_candidates = []
        seen_articulation_names: set[str] = set()
        for obj in articulated_candidates:
            if obj.name in seen_articulation_names:
                continue
            seen_articulation_names.add(obj.name)
            if isinstance(obj, MlSpacesArticulationObject):
                art = obj
            else:
                # OpenTaskSampler discards non-articulation hits at
                # _sample_task time; mirror that filter here so the LLM
                # never sees an entry it could pick but the bridge could
                # not materialize.
                try:
                    art = MlSpacesArticulationObject(
                        data=env.current_data, object_name=obj.name
                    )
                except Exception:
                    continue
            if art.njoints == 0:
                continue
            joints: list[dict[str, Any]] = []
            for j in range(art.njoints):
                try:
                    jtype_int = int(art.get_joint_type(j))
                except Exception:
                    jtype_int = -1
                # mujoco.mjtJoint enum: HINGE=2, SLIDE=3 (others not interesting here)
                if jtype_int == 2:
                    jtype_str = "hinge"
                elif jtype_int == 3:
                    jtype_str = "slide"
                else:
                    continue
                try:
                    jname = art.joint_id2name.get(art.joint_ids[j], f"joint_{j}")
                except Exception:
                    jname = f"joint_{j}"
                joint_entry: dict[str, Any] = {
                    "index": j,
                    "name": jname,
                    "type": jtype_str,
                }
                # Expose current articulation state when the upstream
                # MlSpacesArticulationObject API makes it available. Method
                # names differ across MolmoSpaces versions, so this is
                # deliberately best-effort and omitted if unavailable.
                for method_name in (
                    "get_joint_position",
                    "get_joint_value",
                    "get_joint_qpos",
                    "get_joint_state",
                ):
                    method = getattr(art, method_name, None)
                    if not callable(method):
                        continue
                    try:
                        val = method(j)
                        if isinstance(val, dict):
                            for key in ("position", "qpos", "value"):
                                if key in val:
                                    val = val[key]
                                    break
                        if isinstance(val, (list, tuple)):
                            val = val[0] if val else None
                        if val is not None:
                            joint_entry["current_position"] = float(val)
                            break
                    except Exception:
                        continue
                joints.append(joint_entry)
            if not joints:
                continue
            try:
                category = om.get_annotation_category(obj.name).lower()
            except Exception:
                category = ""
            try:
                position = [float(x) for x in obj.position[:3]]
            except Exception:
                position = [0.0, 0.0, 0.0]
            articulations.append({
                "internal_name": obj.name,
                "category": category,
                "position": position,
                "room": name_to_room.get(obj.name, "unknown"),
                "joints": joints,
            })

        articulations = self._merge_fallback_articulations(
            articulations=articulations,
            pickables=pickables,
            receptacles=receptacles,
            model=model,
            scene_metadata=scene_metadata,
            name_to_room=name_to_room,
        )

        rooms_payload = [
            {"name": room, "object_names": list(names)}
            for room, names in room_to_objects.items()
        ]
        return {
            "house_index": self._house_index,
            "scene_dataset": self._scene_dataset,
            "rooms": rooms_payload,
            "pickables": pickables,
            "receptacles": receptacles,
            "articulations": articulations,
        }

    # Marker-category heuristics for inferring a human-friendly room type
    # from a procthor room's object list. The first hit wins; keep this list
    # tight (one or two strong markers per room type) since per-object
    # category names overlap across rooms (e.g. Chair appears everywhere).
    _ROOM_TYPE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("bathroom", ("Toilet", "Sink", "Faucet", "ShowerDoor", "ShowerHead")),
        ("kitchen", ("Fridge", "Microwave", "StoveBurner", "CounterTop")),
        ("bedroom", ("Bed",)),
        ("livingroom", ("Sofa", "ArmChair", "Television", "TVStand")),
        ("office", ("Desk", "Laptop")),
    )

    @classmethod
    def _extract_room_layout(cls, scene_metadata: dict[str, Any]) -> dict[str, list[str]]:
        """Pull `{room_label: [object_names]}` from scene metadata.

        Two paths:

        1. **Per-object `room_id`** (procthor / ithor / holodeck — what every
           current dataset actually ships). Walks ``metadata['objects'][name]
           ['room_id']`` and groups by id, then labels each group with the
           dominant marker category (kitchen / bathroom / bedroom / …) so the
           LLM sees a useful room name instead of just an integer.
        2. **Top-level `rooms[]` array** with explicit ``roomType`` /
           ``objects[]``. Reserved for future scene formats that pre-bake
           room metadata; preserved for backwards-compat.

        Returns ``{}`` only when neither path finds anything.
        """
        if not isinstance(scene_metadata, dict):
            return {}

        # Path 2 first — when it exists it's already canonical.
        rooms_field = scene_metadata.get("rooms")
        if isinstance(rooms_field, list) and rooms_field:
            out: dict[str, list[str]] = {}
            for room in rooms_field:
                if not isinstance(room, dict):
                    continue
                label = (
                    room.get("roomType")
                    or room.get("name")
                    or room.get("id")
                    or "room"
                )
                objs = room.get("objects") or []
                names: list[str] = []
                for o in objs:
                    if isinstance(o, dict) and o.get("id"):
                        names.append(str(o["id"]))
                    elif isinstance(o, str):
                        names.append(o)
                if names:
                    out.setdefault(str(label), []).extend(names)
            if out:
                return out

        # Path 1 — derive from per-object room_id. This is the path that
        # actually fires for procthor-10k / ithor / holodeck scenes; the
        # previous implementation only handled path 2 and so reported every
        # entry as "unknown" room, which made the proposer blind to
        # cross-room (mug-vs-bed) pairings.
        objects = scene_metadata.get("objects")
        if not isinstance(objects, dict) or not objects:
            return {}

        room_id_to_names: dict[int, list[str]] = {}
        room_id_to_categories: dict[int, list[str]] = {}
        for obj_name, obj_meta in objects.items():
            if not isinstance(obj_meta, dict):
                continue
            rid = obj_meta.get("room_id")
            if rid is None:
                continue
            try:
                rid_int = int(rid)
            except (TypeError, ValueError):
                continue
            room_id_to_names.setdefault(rid_int, []).append(str(obj_name))
            cat = obj_meta.get("category")
            if isinstance(cat, str) and cat:
                room_id_to_categories.setdefault(rid_int, []).append(cat)

        if not room_id_to_names:
            return {}

        out: dict[str, list[str]] = {}
        for rid, names in sorted(room_id_to_names.items()):
            cats = set(room_id_to_categories.get(rid, []))
            label_kind = "room"
            for kind, markers in cls._ROOM_TYPE_MARKERS:
                if any(m in cats for m in markers):
                    label_kind = kind
                    break
            label = f"room_{rid}_{label_kind}"
            out[label] = list(names)
        return out

    # ------------------------------------------------------------------
    # Open-mode task instantiation (drives the open-mode task proposer)
    # ------------------------------------------------------------------

    def set_task_from_spec(
        self,
        *,
        task_type: str,
        target_internal_name: str | None = None,
        place_receptacle_internal_name: str | None = None,
        joint_internal_name: str | None = None,
        joint_index: int | None = None,
    ) -> dict[str, Any]:
        """Instantiate a task from a free-form spec instead of the catalog.

        Used by the rats novel-task proposer to materialize an LLM-chosen
        ``(task_type, target_object, …)`` triple. Pre-sets the relevant
        ``task_config`` fields so the next ``_resample_on_reset`` /
        ``sample_task`` call instantiates exactly the requested task,
        then captures the new identity via ``_capture_pinned_task_identity``
        so the choice persists across subsequent resets.

        Returns a dict shaped like ``get_task_descriptor`` so the caller
        can build a canonical id without an extra RPC.

        ``task_type`` must already match the active sampler family —
        switching from ``pick`` to ``open`` requires a server-side
        bridge restart (issue an ``init`` RPC with the new
        ``task_type`` first). We validate here to fail loudly if not.
        """
        if self._sampler is None:
            raise RuntimeError("set_task_from_spec called before sampler init")
        if task_type != self._task_type:
            raise ValueError(
                f"set_task_from_spec(task_type={task_type!r}) does not match "
                f"the active sampler ({self._task_type!r}); restart the bridge "
                f"with the desired task_type before calling this RPC"
            )

        # PickAndPlaceTaskSampler keeps its own ``place_receptacle_name`` and
        # writes that value back into task_config during _configure_pick_and_place().
        # If we only mutate task_config below, the sampler silently restores its
        # default added receptacle (e.g. a floral dish) and the realized task
        # diverges from the open-proposer spec (e.g. apple -> countertop).
        self._set_sampler_place_receptacle_name(place_receptacle_internal_name)

        for cfg_attr in ("task_config", "task_config_preset_scn"):
            cfg = getattr(self._sampler.config, cfg_attr, None)
            if cfg is None:
                continue
            pickup_obj_name = target_internal_name or joint_internal_name
            if hasattr(cfg, "pickup_obj_name") and pickup_obj_name is not None:
                # Pick / pick_and_place use target_internal_name. Open / close
                # use the articulated object as the pickup_obj_name in upstream
                # OpeningTaskSampler, while joint_index selects which joint.
                cfg.pickup_obj_name = pickup_obj_name
            if (
                hasattr(cfg, "place_receptacle_name")
                and place_receptacle_internal_name is not None
            ):
                cfg.place_receptacle_name = place_receptacle_internal_name
            if (
                hasattr(cfg, "place_target_name")
                and place_receptacle_internal_name is not None
            ):
                cfg.place_target_name = place_receptacle_internal_name
            if hasattr(cfg, "joint_index") and joint_index is not None:
                cfg.joint_index = int(joint_index)
            # Drop any cached referral text — it might reference the
            # previous object and would propagate into ``get_task_description``.
            if hasattr(cfg, "referral_expressions"):
                cfg.referral_expressions = {}

        # Mirror to the bridge-side pinned identity so subsequent resets
        # re-inject these fields rather than letting the sampler cycle.
        self._strict_pinned_task_identity = True
        pickup_obj_name = target_internal_name or joint_internal_name
        if pickup_obj_name is not None:
            self._pinned_pickup_obj_name = pickup_obj_name
        self._pinned_place_receptacle_name = place_receptacle_internal_name
        if joint_index is not None:
            self._pinned_joint_index = int(joint_index)
        # Force the next sampler draw to use the new identity.
        self._pinned_referral_expressions = None

        return {
            "task_type": task_type,
            "target_internal_name": target_internal_name,
            "place_receptacle_internal_name": place_receptacle_internal_name,
            "joint_internal_name": joint_internal_name,
            "joint_index": joint_index,
        }

    def request_new_house(
        self,
        house_index: int | None = None,
        *,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """Advance the sampler to a different house.

        ``house_index=None`` lets ``sample_task`` pick the next index
        from the configured ``house_inds`` list (or the auto-discovery
        path if unset). ``house_index=int`` pins to that exact house.

        Clears pinned task identity + placement-exclusion cache before
        sampling so the first draw on the new house starts clean.
        Returns a small dict with the new ``house_index`` for the
        caller's bookkeeping.
        """
        if task_type is not None and str(task_type) != self._task_type:
            raise ValueError(
                f"request_new_house(task_type={task_type!r}) does not match "
                f"the active sampler ({self._task_type!r}); reinitialize the "
                "bridge with the desired task_type before calling this RPC"
            )
        if self._sampler is None:
            raise RuntimeError("request_new_house called before sampler init")
        self._strict_pinned_task_identity = False
        self._pinned_pickup_obj_name = None
        old_place_receptacle_name = self._pinned_place_receptacle_name
        self._pinned_place_receptacle_name = None
        self._pinned_joint_index = None
        self._pinned_joint_name = None
        self._pinned_referral_expressions = None
        if old_place_receptacle_name is not None:
            self._set_sampler_place_receptacle_name(
                self._default_sampler_place_receptacle_name()
            )
        self._clear_used_robot_positions()

        # Benchmark-mode advance. ``self._sampler.sample_task(force_advance_
        # scene=True)`` advances the upstream sampler's auto-discovery
        # counter, but in benchmark mode (when ``_benchmark_episodes`` is
        # populated) the sampler isn't the one tracking which episode comes
        # next — the bridge's ``_current_episode_index`` is. Without this
        # branch, ``request_new_house(house_index=None)`` would re-sample
        # on the same benchmark episode forever (observed on the v3 30-iter
        # run: every iter stayed on house 0 with the same anchored target,
        # because the upstream sampler's house_inds list was effectively
        # one-element under benchmark replay). Advance through the catalog
        # explicitly via ``resample_task`` to get true per-iter rotation
        # across the 100 mixed-benchmark episodes / 23 distinct houses.
        if self._benchmark_episodes and house_index is None:
            n_episodes = len(self._benchmark_episodes)
            current = self._current_episode_index
            next_index = 0 if current is None else (int(current) + 1) % n_episodes
            self.resample_task(episode_index=next_index)
            return {"house_index": self._house_index, "task_type": self._task_type}

        for cfg_attr in ("task_config", "task_config_preset_scn"):
            cfg = getattr(self._sampler.config, cfg_attr, None)
            if cfg is None:
                continue
            if hasattr(cfg, "pickup_obj_name"):
                cfg.pickup_obj_name = None
            if hasattr(cfg, "referral_expressions"):
                cfg.referral_expressions = {}

        # Retry on physics-invalid / placement-failed houses. Some
        # procthor houses fail validation at sampling time
        # (HouseInvalidForTask, RobotPlacementError, ObjectPlacementError);
        # the bridge already has ``_is_retryable_sample_failure`` to
        # classify these but ``request_new_house``'s original
        # ``except Exception: raise`` ignored that. The RATS lifelong loop
        # then sees a failed rebind and falls back to the bridge's
        # bootstrap task (a non-playtime "Pick up X" that the LLM never
        # asked for). Instead, keep advancing to the next house with
        # ``force_advance_scene=True`` until either (a) a valid house +
        # task triple is found or (b) the retry budget is exhausted.
        #
        # We only auto-advance when the caller passed ``house_index=None``
        # (i.e. "give me any next house"). When a specific house_index is
        # pinned, propagate the exception — the caller asked for THAT
        # exact house and a retry-with-advance would silently bypass it.
        task = None
        max_advance_retries = 20 if house_index is None else 0
        last_exc: BaseException | None = None
        for attempt_idx in range(max_advance_retries + 1):
            try:
                task = _sample_task_compat(
                    self._sampler,
                    force_advance_scene=(house_index is None),
                    house_index=house_index,
                    variant="ceiling",
                )
                break  # success
            except Exception as exc:
                last_exc = exc
                if (
                    attempt_idx >= max_advance_retries
                    or not self._is_retryable_sample_failure(exc)
                ):
                    raise
                logger.warning(
                    "  request_new_house: %s on advance attempt %d/%d "
                    "(%s); advancing to the next house and retrying",
                    type(exc).__name__,
                    attempt_idx + 1,
                    max_advance_retries + 1,
                    str(exc)[:200],
                )
                # Make sure any partial state from the failed draw doesn't
                # poison the next try — clear pinned identity again + reset
                # candidate exclusions, then loop.
                self._strict_pinned_task_identity = False
                self._pinned_pickup_obj_name = None
                self._pinned_place_receptacle_name = None
                self._pinned_joint_index = None
                self._pinned_joint_name = None
                self._pinned_referral_expressions = None
                self._clear_used_robot_positions()
                self._refresh_pick_candidate_objects_if_empty()
        if task is None:
            if last_exc is not None:
                raise RuntimeError(
                    "Failed to sample task on any of the next "
                    f"{max_advance_retries + 1} houses; last error: "
                    f"{type(last_exc).__name__}: {last_exc}"
                )
            raise RuntimeError("Failed to sample task on requested new house")
        self._task = task
        if house_index is not None:
            self._house_index = int(house_index)
        else:
            self._house_index = int(getattr(self._sampler, "current_house_index", self._house_index or 0))
        self._capture_pinned_task_identity()
        return {"house_index": self._house_index, "task_type": self._task_type}


__all__ = ["MolmoSpacesBridge"]
