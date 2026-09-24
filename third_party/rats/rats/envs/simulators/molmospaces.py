from __future__ import annotations

import json
import logging
import re
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rats.envs.base import BaseEnv

logger = logging.getLogger("rats.molmospaces")


class MolmoSpacesResetError(RuntimeError):
    """Raised when ``FrankaMolmoSpacesEnv.reset`` exhausts its retry budget.

    Distinguishes init-time failures (sampler can't find a valid task,
    robot won't place, etc.) from execution-time failures (policy crashed
    mid-trajectory). Callers in capx + RATS catch this specifically so a
    single failed init doesn't tear down the whole batch / iteration loop.
    """


class MolmoSpacesMotionAbort(TimeoutError):
    """Raised when a blocking arm move stalls or exhausts its step budget.

    The exception carries a compact, JSON-like ``diagnostics`` payload so the
    API layer can persist target pose, IK solution, and the robot pose trace
    that led to the abort.  It subclasses ``TimeoutError`` because callers
    should treat it like a failed/timeout motion attempt and retry rather than
    continuing as if the arm reached the target.
    """

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}
        self.motion_debug = self.diagnostics


@dataclass(frozen=True)
class MolmoSpacesTaskDescriptor:
    benchmark: str
    scene_family: str
    task_family: str
    variant: str = "default"
    language: str = ""
    objects: list[str] = field(default_factory=list)
    privileged_requirements: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_id(self) -> str:
        return f"molmospaces:{self.benchmark}:{self.scene_family}:{self.task_family}:{self.variant}"

    def to_runtime_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["canonical_id"] = self.canonical_id
        return data


_DEFAULT_CATALOG: tuple[MolmoSpacesTaskDescriptor, ...] = (
    MolmoSpacesTaskDescriptor(
        benchmark="phase1",
        scene_family="kitchen",
        task_family="put_bowl_on_plate",
        variant="default",
        language="Pick up the bowl and place it on the plate.",
        objects=["bowl", "plate"],
        metadata={"fixtures": ["counter"], "catalog_source": "builtin"},
    ),
    MolmoSpacesTaskDescriptor(
        benchmark="phase1",
        scene_family="kitchen",
        task_family="open_drawer_and_store_utensil",
        variant="default",
        language="Open the drawer and place the utensil inside it.",
        objects=["drawer", "utensil"],
        privileged_requirements=["drawer_state"],
        metadata={"fixtures": ["drawer_bank"], "catalog_source": "builtin"},
    ),
    MolmoSpacesTaskDescriptor(
        benchmark="phase1",
        scene_family="living_room",
        task_family="place_remote_on_table",
        variant="default",
        language="Move the remote onto the coffee table.",
        objects=["remote", "coffee_table"],
        metadata={"fixtures": ["sofa"], "catalog_source": "builtin"},
    ),
    # ----- Articulated-object manipulation (drawer / cabinet) -----
    # task_family is matched by _infer_task_type to map to bridge task_type
    # ("open" / "close"). The MolmoSpaces OpenTaskSampler picks a concrete
    # articulated instance from the iThor scene at sample time.
    MolmoSpacesTaskDescriptor(
        benchmark="ithor",
        scene_family="kitchen",
        task_family="open_drawer",
        variant="default",
        language="Open the drawer.",
        objects=["drawer"],
        metadata={
            "catalog_source": "builtin",
            "task_type": "open",
            "scene_dataset": "ithor",
            "pickup_types": ["drawer"],
        },
    ),
    MolmoSpacesTaskDescriptor(
        benchmark="ithor",
        scene_family="kitchen",
        task_family="close_drawer",
        variant="default",
        language="Close the drawer.",
        objects=["drawer"],
        metadata={
            "catalog_source": "builtin",
            "task_type": "close",
            "scene_dataset": "ithor",
            "pickup_types": ["drawer"],
        },
    ),
    MolmoSpacesTaskDescriptor(
        benchmark="ithor",
        scene_family="kitchen",
        task_family="open_cabinet",
        variant="default",
        language="Open the cabinet.",
        objects=["cabinet"],
        metadata={
            "catalog_source": "builtin",
            "task_type": "open",
            "scene_dataset": "ithor",
            "pickup_types": ["cabinet"],
        },
    ),
)


def default_molmospaces_catalog(
    benchmark_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the MolmoSpaces task catalog.

    If *benchmark_dir* points to a valid MolmoSpaces benchmark directory the
    catalog is loaded from its episode JSON files.  Otherwise the built-in
    phase-1 placeholder catalog is returned.
    """
    if benchmark_dir is not None:
        loaded = load_benchmark_catalog(benchmark_dir)
        if loaded:
            return loaded
    return [descriptor.to_runtime_dict() for descriptor in _DEFAULT_CATALOG]


# ------------------------------------------------------------------
# Benchmark catalog loading (dependency-free JSON parsing)
# ------------------------------------------------------------------

def _task_cls_to_family(task_cls: str) -> str:
    """Extract a short task family name from a fully-qualified task class.

    >>> _task_cls_to_family("molmo_spaces.tasks.pick_task.PickTask")
    'pick'
    >>> _task_cls_to_family("molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask")
    'pick_and_place'
    """
    # Take the class name (last dotted component)
    cls_name = task_cls.rsplit(".", 1)[-1]
    # Strip trailing "Task"
    if cls_name.endswith("Task"):
        cls_name = cls_name[: -len("Task")]
    # CamelCase -> snake_case
    family = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", cls_name).lower()
    return family


def _episode_task_family(task: dict[str, Any], task_cls: str) -> str:
    """Return the benchmark catalog family for one JSON episode.

    ``OpeningTask`` is reused for both open and close episodes, so the class
    name alone collapses two benchmark families into ``opening``.  JSON
    benchmark episodes may carry the more specific ``task_type`` field; prefer
    it when present so mixed suites expose separate ``open`` and ``close``
    catalog entries.
    """
    task_type = task.get("task_type")
    if isinstance(task_type, str) and task_type:
        return task_type
    return _task_cls_to_family(task_cls) if task_cls else "unknown"


def _extract_episode_objects(task: dict[str, Any]) -> list[str]:
    """Pull object names from an episode task dict."""
    objects: list[str] = []
    for key in ("pickup_obj_name", "place_receptacle_name"):
        val = task.get(key)
        if val and isinstance(val, str):
            objects.append(val)
    return objects


def load_benchmark_catalog(benchmark_dir: str | Path) -> list[dict[str, Any]]:
    """Load a MolmoSpaces benchmark directory into a catalog list.

    Supports two on-disk layouts:
      1. ``benchmark.json`` — a single JSON array of episode dicts (preferred).
      2. ``house_*/episode_*.json`` — legacy per-house directory structure.

    Returns a list of ``MolmoSpacesTaskDescriptor.to_runtime_dict()``-shaped
    dicts.  Returns an empty list when *benchmark_dir* does not exist or
    contains no episodes.
    """
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        logger.debug("Benchmark dir does not exist: %s", benchmark_path)
        return []

    # --- Load raw episode dicts ---
    episodes: list[dict[str, Any]] = []

    single_file = benchmark_path / "benchmark.json"
    if single_file.exists():
        data = json.loads(single_file.read_text())
        if isinstance(data, list):
            episodes = data
    else:
        # Legacy directory structure
        for house_dir in sorted(benchmark_path.glob("house_*")):
            if not house_dir.is_dir():
                continue
            for ep_file in sorted(house_dir.glob("episode_*.json")):
                try:
                    episodes.append(json.loads(ep_file.read_text()))
                except Exception:
                    continue

    if not episodes:
        logger.debug("No episodes found in benchmark dir: %s", benchmark_path)
        return []

    # --- Convert episodes to catalog entries ---
    catalog: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, ep in enumerate(episodes):
        task = ep.get("task", {})
        task_cls = task.get("task_cls", "")
        language_spec = ep.get("language", {})
        robot_spec = ep.get("robot", {})

        task_family = _episode_task_family(task, task_cls)
        house_index = ep.get("house_index", 0)
        scene_dataset = ep.get("scene_dataset", "unknown")
        variant = f"ep{idx}"

        descriptor = MolmoSpacesTaskDescriptor(
            benchmark=scene_dataset,
            scene_family=f"house_{house_index}",
            task_family=task_family,
            variant=variant,
            language=language_spec.get("task_description", ""),
            objects=_extract_episode_objects(task),
            metadata={
                "catalog_source": "benchmark",
                "house_index": house_index,
                "scene_dataset": scene_dataset,
                "task_cls": task_cls,
                "episode_index": idx,
                "robot_name": robot_spec.get("robot_name", ""),
                "referral_expressions": language_spec.get("referral_expressions", {}),
            },
        )
        cid = descriptor.canonical_id
        if cid in seen_ids:
            logger.warning("Duplicate canonical_id in benchmark: %s", cid)
        seen_ids.add(cid)
        catalog.append(descriptor.to_runtime_dict())

    logger.info(
        "Loaded %d episodes from benchmark dir: %s", len(catalog), benchmark_path,
    )
    return catalog


def parse_molmospaces_task_id(canonical_id: str) -> dict[str, str]:
    parts = canonical_id.split(":")
    if len(parts) != 5 or parts[0] != "molmospaces":
        raise ValueError(f"Invalid MolmoSpaces task id: {canonical_id}")
    return {
        "env_type": parts[0],
        "benchmark": parts[1],
        "scene_family": parts[2],
        "task_family": parts[3],
        "variant": parts[4],
    }


class MockMolmoSpacesBridge:
    """Local phase-1 bridge stub for MolmoSpaces session semantics.

    The main process talks to a stable task/session contract without importing any
    external MolmoSpaces package. A future mlsp2 subprocess transport can replace
    this class behind the same call surface.
    """

    def __init__(
        self,
        catalog: list[dict[str, Any]] | None = None,
        benchmark_dir: str | Path | None = None,
    ) -> None:
        task_catalog = catalog or default_molmospaces_catalog(benchmark_dir=benchmark_dir)
        self._catalog = {task["canonical_id"]: dict(task) for task in task_catalog}
        self._session_counter = 0

    def list_task_descriptors(self) -> list[dict[str, Any]]:
        return [dict(task) for task in self._catalog.values()]

    def get_task_descriptor(self, canonical_id: str) -> dict[str, Any]:
        if canonical_id not in self._catalog:
            raise KeyError(f"Unknown MolmoSpaces task: {canonical_id}")
        return dict(self._catalog[canonical_id])

    def create_session(self, canonical_id: str) -> dict[str, Any]:
        descriptor = self.get_task_descriptor(canonical_id)
        self._session_counter += 1
        return {
            "session_id": f"mlsp2-session-{self._session_counter}",
            "descriptor": descriptor,
        }

    def build_observation(self, descriptor: dict[str, Any], step_count: int) -> dict[str, Any]:
        rgb = self._render_task_rgb(descriptor)
        wrist_rgb = np.flip(rgb, axis=1).copy()
        depth = np.full(rgb.shape[:2], fill_value=0.25 + (step_count % 5) * 0.01, dtype=np.float32)
        return {
            "agentview": {
                "images": {"rgb": rgb, "depth": depth[..., None]},
                "intrinsics": np.eye(3, dtype=np.float64),
                "pose_mat": np.eye(4, dtype=np.float64),
            },
            "robot0_eye_in_hand": {
                "images": {"rgb": wrist_rgb, "depth": depth[..., None]},
                "intrinsics": np.eye(3, dtype=np.float64),
                "pose_mat": np.eye(4, dtype=np.float64),
            },
            "robot_cartesian_pos": np.zeros(8, dtype=np.float64),
            "robot_joint_pos": np.zeros(8, dtype=np.float64),
            "robot_base_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "task_descriptor": descriptor,
        }

    def _render_task_rgb(self, descriptor: dict[str, Any]) -> np.ndarray:
        seed = abs(hash(descriptor["canonical_id"])) % 255
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        rgb[..., 0] = (seed + 17) % 255
        rgb[..., 1] = (seed + 83) % 255
        rgb[..., 2] = (seed + 149) % 255
        rgb[16:112, 16:112] = np.array([(seed + 40) % 255, (seed + 120) % 255, (seed + 200) % 255], dtype=np.uint8)
        return rgb


def _try_import_real_bridge():
    """Attempt to import MolmoSpacesBridge; return None on failure."""
    try:
        from rats.envs.simulators.molmospaces_bridge import MolmoSpacesBridge
        return MolmoSpacesBridge
    except Exception:
        return None


class FrankaMolmoSpacesEnv(BaseEnv):
    """MolmoSpaces low-level environment for RATS.

    Supports three modes:
      - **Remote bridge** (``remote_bridge_url`` set): connects to an
        ``mlspaces_server.py`` process over TCP so the simulator can live in a
        separate conda/venv with incompatible dependencies.
      - **Real bridge** (``use_real_bridge=True``): wraps the molmo_spaces MuJoCo
        simulator via ``MolmoSpacesBridge`` for physics, rendering, and reward.
      - **Mock bridge** (default): uses ``MockMolmoSpacesBridge`` with synthetic
        observations for pipeline testing without the simulator.

    When ``use_real_bridge=True`` but ``molmo_spaces`` is not importable, falls
    back to the mock bridge with a warning.
    """

    def __init__(
        self,
        benchmark: str = "phase1",
        scene_family: str = "kitchen",
        task_family: str = "put_bowl_on_plate",
        variant: str = "default",
        canonical_task_id: str | None = None,
        privileged: bool = False,
        max_steps: int = 200,
        enable_render: bool = True,
        catalog_path: str | None = None,
        benchmark_dir: str | None = None,
        use_real_bridge: bool = False,
        remote_bridge_url: str | None = None,
        task_type: str | None = None,
        scene_dataset: str = "procthor-10k",
        data_split: str = "train",
        house_index: int | None = None,
        seed: int | None = None,
        max_joint_step_rad: float = 0.04,
        move_max_steps: int = 200,
        reset_physical_state: bool = True,
        randomize_agentview: bool = True,
        use_recorded_cameras: bool = False,
        pickup_types: list[str] | None = None,
        require_grasp_files: bool = True,
        pin_pickup_obj_name: bool = True,
        front_facing_robot_placement: bool = True,
        candidate_house_indices: list[int] | None = None,
        capx_only: bool = False,
    ) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.scene_family = scene_family
        self.task_family = task_family
        self.variant = variant
        self.privileged = privileged
        self.max_steps = max_steps
        self.enable_render = enable_render
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.move_max_steps = int(move_max_steps)
        self._default_max_joint_step_rad = float(max_joint_step_rad)
        self._default_move_max_steps = int(move_max_steps)
        self._policy_motion_speed_label = "normal"
        # Fail fast when the remote bridge accepts actions but the actual arm
        # is not making measurable progress toward the joint target.  This
        # converts silent max-step exhaustion into a retryable policy failure
        # with motion diagnostics.
        self.motion_progress_abort_enabled = True
        self.motion_progress_check_window = 20
        self.motion_progress_min_error_reduction = 2e-3
        self._active_motion_debug_context: dict[str, Any] = {}
        self.catalog_path = catalog_path
        self.benchmark_dir = benchmark_dir
        self.use_real_bridge = use_real_bridge
        self.remote_bridge_url = remote_bridge_url
        self.task_type = task_type
        self.scene_dataset = scene_dataset
        self.data_split = data_split
        self.house_index = house_index
        self.seed = seed
        self.reset_physical_state = bool(reset_physical_state)
        self.randomize_agentview = bool(randomize_agentview)
        self.use_recorded_cameras = bool(use_recorded_cameras)
        self.pickup_types = list(pickup_types) if pickup_types else None
        self.require_grasp_files = bool(require_grasp_files)
        self.pin_pickup_obj_name = bool(pin_pickup_obj_name)
        self.front_facing_robot_placement = bool(front_facing_robot_placement)
        self.candidate_house_indices = (
            [int(h) for h in (candidate_house_indices or [])] or None
        )
        self.capx_only = bool(capx_only)

        self._record_frames = False
        self._wrist_recording = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self._step_count = 0
        self._sim_step_count = 0
        self._task_completed = False
        # Set once the underlying task reports terminated/truncated. Further
        # bridge.step() calls emit upstream "all environments already done"
        # spam and hang the attempt, so we short-circuit every control path
        # (move_to_joints_blocking, _step_once, gym step) once this flips.
        self._task_done = False
        self._runtime_notes: list[str] = []
        self._last_observation_warnings: list[str] = []
        self._last_reset_seed: int | None = None
        self._session: dict[str, Any] | None = None
        self._current_obs: dict[str, Any] | None = None
        self._gripper_fraction = 1.0
        self._subsample_rate = 1
        self._viser_publisher = None
        self._state_trace_samples: list[dict[str, Any]] = []
        self._state_trace_enabled = False
        self._state_trace_max_samples = 2500

        self._real_bridge = None
        self._active_real_task_type: str | None = None
        self.home_joint_position: np.ndarray | None = None

        requested_canonical_task_id = canonical_task_id or self._build_canonical_id(
            benchmark=benchmark,
            scene_family=scene_family,
            task_family=task_family,
            variant=variant,
        )
        if canonical_task_id is None and benchmark_dir:
            benchmark_catalog = load_benchmark_catalog(benchmark_dir)
            benchmark_ids = [item["canonical_id"] for item in benchmark_catalog]
            if benchmark_ids and requested_canonical_task_id not in benchmark_ids:
                requested_canonical_task_id = benchmark_ids[0]

        if remote_bridge_url:
            from rats.envs.simulators.molmospaces_remote import RemoteMolmoSpacesBridge

            host, _, port_str = remote_bridge_url.rpartition(":")
            host = host or "localhost"
            port = int(port_str) if port_str else 9100
            self._real_bridge = RemoteMolmoSpacesBridge(
                host=host,
                port=port,
                init_kwargs=self._build_remote_init_kwargs(
                    canonical_task_id=requested_canonical_task_id,
                ),
            )
            self._active_real_task_type = task_type or self._infer_task_type(task_family)
            logger.info("Using remote MolmoSpaces bridge at %s", remote_bridge_url)
        elif use_real_bridge:
            BridgeCls = _try_import_real_bridge()
            if BridgeCls is not None:
                resolved_task_type = task_type or self._infer_task_type(task_family)
                self._real_bridge = BridgeCls(
                    task_type=resolved_task_type,
                    scene_dataset=scene_dataset,
                    data_split=data_split,
                    house_index=house_index,
                    benchmark_dir=benchmark_dir,
                    max_steps=max_steps,
                    seed=seed,
                    reset_physical_state=self.reset_physical_state,
                    randomize_agentview=self.randomize_agentview,
                    use_recorded_cameras=self.use_recorded_cameras,
                    pickup_types=self.pickup_types,
                    require_grasp_files=self.require_grasp_files,
                    pin_pickup_obj_name=self.pin_pickup_obj_name,
                    front_facing_robot_placement=self.front_facing_robot_placement,
                    candidate_house_indices=self.candidate_house_indices,
                )
                self._active_real_task_type = resolved_task_type
                logger.info("Using real MolmoSpaces bridge")
            else:
                logger.warning(
                    "use_real_bridge=True but molmo_spaces is not importable; "
                    "falling back to MockMolmoSpacesBridge"
                )

        if self._real_bridge is None:
            self.bridge = MockMolmoSpacesBridge(
                catalog=self._load_catalog(catalog_path),
                benchmark_dir=benchmark_dir,
            )
        else:
            self.bridge = None

        if canonical_task_id:
            self.canonical_task_id = canonical_task_id
        else:
            candidate = requested_canonical_task_id
            if self._real_bridge is None:
                catalog_ids = {t["canonical_id"] for t in self.bridge.list_task_descriptors()}
                if candidate not in catalog_ids and catalog_ids:
                    self.canonical_task_id = next(iter(catalog_ids))
                else:
                    self.canonical_task_id = candidate
            else:
                self.canonical_task_id = candidate

        if self._real_bridge is None:
            self._set_task(self.canonical_task_id)
        else:
            self._task_completed = False
            self._step_count = 0
            self._sim_step_count = 0
            self._init_from_real_bridge()

    @staticmethod
    def _infer_task_type(task_family: str) -> str:
        family = (task_family or "").lower()
        # Multi-step task families ("open_drawer_and_store_utensil") should
        # resolve to pick_and_place, so check place/store FIRST.
        if "place" in family or "store" in family:
            return "pick_and_place"
        # Prefix or suffix handles "open_drawer", "drawer_open", "opening_*", etc.
        if family.startswith("open") or family.endswith("_open") or "opening" in family:
            return "open"
        if family.startswith("close") or family.endswith("_close") or "closing" in family:
            return "close"
        return "pick"

    @staticmethod
    def _build_canonical_id(*, benchmark: str, scene_family: str, task_family: str, variant: str) -> str:
        return f"molmospaces:{benchmark}:{scene_family}:{task_family}:{variant}"

    def _build_remote_init_kwargs(
        self,
        *,
        canonical_task_id: str | None = None,
    ) -> dict[str, Any]:
        parsed_task = parse_molmospaces_task_id(canonical_task_id) if canonical_task_id else None
        task_family = parsed_task["task_family"] if parsed_task is not None else self.task_family
        task_type = self.task_type or self._infer_task_type(task_family)
        kwargs: dict[str, Any] = {
            "task_type": task_type,
            "scene_dataset": self.scene_dataset,
            "data_split": self.data_split,
            "max_steps": self.max_steps,
            "benchmark_dir": self.benchmark_dir,
            "house_index": self.house_index,
            "seed": self.seed,
            "canonical_task_id": canonical_task_id,
            "reset_physical_state": self.reset_physical_state,
            "randomize_agentview": self.randomize_agentview,
            "use_recorded_cameras": self.use_recorded_cameras,
            "pickup_types": self.pickup_types,
            "require_grasp_files": self.require_grasp_files,
            "pin_pickup_obj_name": self.pin_pickup_obj_name,
            "front_facing_robot_placement": self.front_facing_robot_placement,
            "candidate_house_indices": self.candidate_house_indices,
        }
        return {key: value for key, value in kwargs.items() if value is not None}

    def _live_house_index(self) -> int | None:
        """Best-effort active house index for same-house task-type switches."""
        if self._real_bridge is None or not hasattr(self._real_bridge, "describe_scene_inventory"):
            return self.house_index
        try:
            inv = self._real_bridge.describe_scene_inventory()
            house_index = inv.get("house_index") if isinstance(inv, dict) else None
            return int(house_index) if house_index is not None else self.house_index
        except Exception:
            return self.house_index

    def _ensure_real_bridge_task_type(
        self,
        task_type: str,
        *,
        preserve_house: bool = True,
    ) -> None:
        """Recreate/re-init the real bridge when an open proposer changes type.

        MolmoSpaces samplers are task-family-specific: a bridge initialized as
        ``pick`` cannot materialize an ``open`` or ``pick_and_place`` spec via
        ``set_task_from_spec``. For concrete specs we preserve the current
        house. For a house-switch request we intentionally do *not* preserve an
        invalid current house; leaving ``house_index`` unset lets the new
        sampler auto-discover a compatible house for the requested task family.
        """
        task_type = str(task_type or "pick").lower()
        if self._real_bridge is None or self._active_real_task_type == task_type:
            return

        live_house_index = self._live_house_index() if preserve_house else None
        self.house_index = live_house_index
        self.task_type = task_type
        self.task_family = task_type
        self.canonical_task_id = self._build_canonical_id(
            benchmark=self.benchmark,
            scene_family=self.scene_family,
            task_family=task_type,
            variant=self.variant,
        )

        if self.remote_bridge_url and hasattr(self._real_bridge, "init"):
            init_kwargs = self._build_remote_init_kwargs(
                canonical_task_id=self.canonical_task_id,
            )
            init_kwargs["task_type"] = task_type
            logger.info(
                "Reinitializing remote MolmoSpaces bridge for task_type=%s "
                "on house_index=%s",
                task_type,
                self.house_index,
            )
            self._real_bridge.init(**init_kwargs)
            self._active_real_task_type = task_type
            # If house_index was omitted for auto-discovery, capture the house
            # selected by the newly-created sampler for subsequent resets.
            discovered_house = self._live_house_index()
            if discovered_house is not None:
                self.house_index = discovered_house
            self._task_completed = False
            self._task_done = False
            self._step_count = 0
            self._sim_step_count = 0
            return

        if self.use_real_bridge:
            BridgeCls = _try_import_real_bridge()
            if BridgeCls is None:
                return
            if hasattr(self._real_bridge, "close"):
                try:
                    self._real_bridge.close()
                except Exception:
                    logger.debug("Failed closing MolmoSpaces bridge during task-type switch", exc_info=True)
            self._real_bridge = BridgeCls(
                task_type=task_type,
                scene_dataset=self.scene_dataset,
                data_split=self.data_split,
                house_index=self.house_index,
                benchmark_dir=self.benchmark_dir,
                max_steps=self.max_steps,
                seed=self.seed,
                reset_physical_state=self.reset_physical_state,
                randomize_agentview=self.randomize_agentview,
                use_recorded_cameras=self.use_recorded_cameras,
                pickup_types=self.pickup_types,
                require_grasp_files=self.require_grasp_files,
                pin_pickup_obj_name=self.pin_pickup_obj_name,
                front_facing_robot_placement=self.front_facing_robot_placement,
                candidate_house_indices=self.candidate_house_indices,
            )
            self._active_real_task_type = task_type
            discovered_house = self._live_house_index()
            if discovered_house is not None:
                self.house_index = discovered_house
            self._task_completed = False
            self._task_done = False
            self._step_count = 0
            self._sim_step_count = 0

    def _load_catalog(self, catalog_path: str | None) -> list[dict[str, Any]] | None:
        if not catalog_path:
            return None
        path = Path(catalog_path)
        if not path.exists():
            raise FileNotFoundError(f"MolmoSpaces catalog path not found: {catalog_path}")
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError("MolmoSpaces catalog JSON must be a list of task descriptors")
        return payload

    def _init_from_real_bridge(self) -> None:
        """Initialize observation and home position from the real bridge after first task sample."""
        raw_obs, _info = self._real_bridge.reset()
        self._current_obs = self._real_bridge.build_observation()
        arm_mg = self._real_bridge.robot_view.get_move_group("arm")
        self.home_joint_position = np.array(arm_mg.joint_pos, dtype=np.float64)
        self._validate_observation(self._current_obs)

    def _refresh_current_obs_from_real_bridge(self) -> None:
        self._current_obs = self._real_bridge.build_observation()

    def _validate_observation(self, obs: dict[str, Any] | None) -> list[str]:
        """Validate observation structure and flag suspicious real-bridge frames."""
        warnings: list[str] = []
        if not isinstance(obs, dict):
            warnings.append("observation is not a dict")
            self._last_observation_warnings = warnings
            return warnings

        for cam_name in ("agentview", "robot0_eye_in_hand"):
            cam = obs.get(cam_name)
            if not isinstance(cam, dict):
                warnings.append(f"{cam_name} missing from observation")
                continue

            images = cam.get("images")
            if not isinstance(images, dict):
                warnings.append(f"{cam_name}.images missing")
                continue

            rgb = images.get("rgb")
            if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[-1] != 3:
                warnings.append(f"{cam_name}.images.rgb invalid shape")
            elif rgb.size == 0:
                warnings.append(f"{cam_name}.images.rgb empty")
            elif self._real_bridge is not None and not np.any(rgb):
                warnings.append(f"{cam_name}.images.rgb all zeros")

            depth = images.get("depth")
            if not isinstance(depth, np.ndarray) or depth.ndim not in (2, 3):
                warnings.append(f"{cam_name}.images.depth invalid shape")
            elif depth.size == 0:
                warnings.append(f"{cam_name}.images.depth empty")

            intrinsics = cam.get("intrinsics")
            if not isinstance(intrinsics, np.ndarray) or intrinsics.shape != (3, 3):
                warnings.append(f"{cam_name}.intrinsics invalid shape")

            pose_mat = cam.get("pose_mat")
            if not isinstance(pose_mat, np.ndarray) or pose_mat.shape != (4, 4):
                warnings.append(f"{cam_name}.pose_mat invalid shape")

        for key in ("robot_joint_pos", "robot_cartesian_pos"):
            if not isinstance(obs.get(key), np.ndarray):
                warnings.append(f"{key} missing")

        robot_base_pose = obs.get("robot_base_pose")
        if not isinstance(robot_base_pose, np.ndarray) or robot_base_pose.shape != (7,):
            warnings.append("robot_base_pose invalid shape")
        elif not np.all(np.isfinite(robot_base_pose)):
            warnings.append("robot_base_pose contains non-finite values")

        self._last_observation_warnings = warnings
        return warnings

    def _stabilize_after_reset(self, settle_steps: int = 3) -> None:
        """Let the simulator settle briefly and refresh the observation."""
        if self._real_bridge is None:
            return

        initial_warnings = self._validate_observation(self._current_obs)
        if initial_warnings:
            logger.warning(
                "MolmoSpaces reset observation had warnings before settling: %s",
                "; ".join(initial_warnings),
            )

        for _ in range(max(settle_steps, 0)):
            self._step_once()

        self._refresh_current_obs_from_real_bridge()
        settled_warnings = self._validate_observation(self._current_obs)
        if settled_warnings:
            logger.warning(
                "MolmoSpaces observation still suspicious after reset settle: %s",
                "; ".join(settled_warnings),
            )
        elif initial_warnings:
            logger.info("MolmoSpaces reset observation stabilized after %d settle steps", settle_steps)

    # ------------------------------------------------------------------
    # Task catalog (mock bridge only)
    # ------------------------------------------------------------------

    def list_task_descriptors(self) -> list[dict[str, Any]]:
        if self.bridge is not None:
            return self.bridge.list_task_descriptors()
        if self._real_bridge is not None and hasattr(self._real_bridge, "list_task_descriptors"):
            return self._real_bridge.list_task_descriptors()
        return []

    def get_task_descriptor(self, canonical_id: str | None = None) -> dict[str, Any]:
        if canonical_id is None:
            playtime = getattr(self, "_playtime_task_descriptor", None)
            if isinstance(playtime, dict) and playtime.get("canonical_id"):
                return dict(playtime)
        if self.bridge is not None:
            task_id = canonical_id or self.canonical_task_id
            return self.bridge.get_task_descriptor(task_id)
        if self._real_bridge is not None and hasattr(self._real_bridge, "get_task_descriptor"):
            descriptor = self._real_bridge.get_task_descriptor(canonical_id)
            if canonical_id is None and descriptor.get("canonical_id"):
                self.canonical_task_id = descriptor["canonical_id"]
            return descriptor
        desc = {
            "canonical_id": self.canonical_task_id,
            "language": self._real_bridge.get_task_description() if self._real_bridge else "",
        }
        return desc

    def set_playtime_task_descriptor(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        """Install a prompt-only descriptor for VLM-verified playtime tasks."""
        self._playtime_task_descriptor = dict(descriptor or {})
        if self._playtime_task_descriptor.get("canonical_id"):
            self.canonical_task_id = str(self._playtime_task_descriptor["canonical_id"])
        return dict(self._playtime_task_descriptor)

    def set_task(self, canonical_id: str) -> dict[str, Any]:
        if hasattr(self, "_playtime_task_descriptor"):
            self._playtime_task_descriptor = None
        self._set_task(canonical_id)
        return self.get_task_descriptor()

    # ------------------------------------------------------------------
    # Open-mode task proposer plumbing (mirrors RemoteMolmoSpacesBridge)
    # ------------------------------------------------------------------

    def get_anchored_task_target(self) -> dict[str, Any]:
        """Forward anchored-target query to the bridge.

        Returns the sampler's pinned identity (pickup_obj_name /
        place_receptacle_name / joint_name) so the RATS playtime proposer
        can prefer the bridge-anchored object — which is guaranteed
        reachable from the current robot pose — instead of asking the
        LLM to choose from the full out-of-reach scene inventory.
        """
        if self._real_bridge is not None and hasattr(self._real_bridge, "get_anchored_task_target"):
            return self._real_bridge.get_anchored_task_target() or {}
        return {}

    def anchor_to_pickup(self, target_internal_name: str) -> dict[str, Any]:
        """Forward anchor-to-pickup to the bridge.

        Switches the sampler's anchored target to ``target_internal_name``
        and re-runs ``env.place_robot_near`` so the robot stands within
        reach of the new target. Returns the updated anchored-target
        descriptor. Lets RATS-side curiosity drive which object the
        robot interacts with, instead of being stuck with the bridge's
        round-robin first pick.
        """
        if self._real_bridge is not None and hasattr(self._real_bridge, "anchor_to_pickup"):
            return self._real_bridge.anchor_to_pickup(target_internal_name) or {}
        return {}

    def describe_scene_obstacles(
        self,
        *,
        exclude_internal_names: list[str] | None = None,
        max_distance_m: float = 3.0,
    ) -> list[dict[str, Any]]:
        """Forward scene-obstacle query to the bridge (for pyroki collision IK)."""
        if self._real_bridge is not None and hasattr(self._real_bridge, "describe_scene_obstacles"):
            return self._real_bridge.describe_scene_obstacles(
                exclude_internal_names=exclude_internal_names,
                max_distance_m=max_distance_m,
            ) or []
        return []

    def describe_scene_inventory(self) -> dict[str, Any]:
        """Snapshot of pickables / receptacles / articulations.

        Real bridge or remote bridge only — the mock bridge has no live
        scene to introspect, so we synthesise a tiny payload from the
        active task descriptor for pipeline tests.
        """
        if self._real_bridge is not None and hasattr(self._real_bridge, "describe_scene_inventory"):
            return self._real_bridge.describe_scene_inventory()
        descriptor = self.get_task_descriptor() or {}
        objects = descriptor.get("objects", []) or []
        return {
            "house_index": getattr(self, "house_index", None),
            "scene_dataset": getattr(self, "scene_dataset", "mock"),
            "rooms": [],
            "pickables": [
                {
                    "internal_name": obj,
                    "category": obj,
                    "asset_uid": None,
                    "position": [0.0, 0.0, 0.0],
                    "room": "unknown",
                    "has_grasp_file": True,
                }
                for obj in objects
            ],
            "receptacles": [],
            "articulations": [],
        }

    def set_task_from_spec(
        self,
        *,
        task_type: str,
        target_internal_name: str | None = None,
        place_receptacle_internal_name: str | None = None,
        joint_internal_name: str | None = None,
        joint_index: int | None = None,
    ) -> dict[str, Any]:
        """Forward an open-mode task spec to the real bridge.

        Mock-bridge path is a no-op (mock has no live sampler) but
        returns a dict shaped like the real path so callers can stay
        bridge-agnostic.
        """
        if hasattr(self, "_playtime_task_descriptor"):
            self._playtime_task_descriptor = None
        if self._real_bridge is not None and hasattr(self._real_bridge, "set_task_from_spec"):
            self._ensure_real_bridge_task_type(task_type)
            return self._real_bridge.set_task_from_spec(
                task_type=task_type,
                target_internal_name=target_internal_name,
                place_receptacle_internal_name=place_receptacle_internal_name,
                joint_internal_name=joint_internal_name,
                joint_index=joint_index,
            )
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
        """Forward a house-switch request to the real bridge.

        When ``task_type`` is supplied, reinitialize/recreate the real bridge
        to that sampler family before sampling the new house. This keeps smoke
        tests like forced ``open`` from accidentally reusing the previous
        iteration's sampler family.
        """
        requested_task_type = str(task_type).lower() if task_type else None
        if self._real_bridge is not None and hasattr(self._real_bridge, "request_new_house"):
            if requested_task_type and requested_task_type != self._active_real_task_type:
                # This is the important smoke/open-proposer path: the current
                # house was rejected for the requested family, so reinitialize
                # that family with house_index omitted and let the new sampler
                # auto-discover a compatible house. The init call already
                # samples the first task, so do not immediately call
                # request_new_house() again and skip past it.
                if house_index is not None:
                    self.house_index = int(house_index)
                    preserve_house = True
                else:
                    preserve_house = False
                self._ensure_real_bridge_task_type(
                    requested_task_type,
                    preserve_house=preserve_house,
                )
                return {
                    "house_index": self.house_index,
                    "task_type": self._active_real_task_type,
                }

            result = self._real_bridge.request_new_house(house_index=house_index)
            try:
                if isinstance(result, dict) and result.get("house_index") is not None:
                    self.house_index = int(result["house_index"])
            except Exception:
                pass
            return result
        if requested_task_type:
            self.task_type = requested_task_type
            self.task_family = requested_task_type
        return {"house_index": house_index, "task_type": requested_task_type}

    def _set_task(self, canonical_id: str) -> None:
        self.canonical_task_id = canonical_id
        if self.bridge is not None:
            self._session = self.bridge.create_session(canonical_id)
        elif self._real_bridge is not None:
            self._real_bridge.resample_task(canonical_task_id=canonical_id)
            self._init_from_real_bridge()
        self._task_completed = False
        self._task_done = False
        self._step_count = 0
        self._sim_step_count = 0
        if self._real_bridge is None and self.bridge is not None:
            self._current_obs = self.bridge.build_observation(self.get_task_descriptor(), self._step_count)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.reset_policy_motion_speed()
        if options and options.get("canonical_task_id"):
            self._set_task(str(options["canonical_task_id"]))
        self._task_completed = False
        self._task_done = False
        self._step_count = 0
        self._sim_step_count = 0
        self._last_reset_seed = seed
        self._gripper_fraction = 1.0
        self._last_observation_warnings = []

        if self._real_bridge is not None:
            # Retry the full bridge reset twice on transient resample
            # failures (e.g. an articulated joint that placed cleanly on
            # the previous reset can transiently fail on the next due to
            # accumulated exclusion zones, RNG drift, or a stale pinned
            # name that the bridge's own retry loop couldn't resolve).
            # Without this wrapper any failure propagates up and either
            # kills the capx trial batch (single-worker mode has no
            # per-trial except) or wedges a RATS iteration's env so all
            # subsequent iterations also fail. Each attempt uses a
            # different seed to break out of deterministic bad states.
            base_seed = seed if seed is not None else self._last_reset_seed
            last_exc: Exception | None = None
            raw_obs = raw_info = None
            for attempt in range(self._RESET_RETRY_BUDGET):
                attempt_seed = (
                    base_seed + attempt
                    if base_seed is not None and attempt > 0
                    else base_seed
                )
                try:
                    raw_obs, raw_info = self._real_bridge.reset(seed=attempt_seed)
                    if attempt > 0:
                        logger.warning(
                            "MolmoSpaces real-bridge reset succeeded on retry "
                            "%d/%d after %s",
                            attempt + 1,
                            self._RESET_RETRY_BUDGET,
                            type(last_exc).__name__ if last_exc else "?",
                        )
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "MolmoSpaces real-bridge reset attempt %d/%d failed: %s: %s",
                        attempt + 1,
                        self._RESET_RETRY_BUDGET,
                        type(exc).__name__,
                        exc,
                    )
            else:
                # Loop completed without break -> every attempt failed.
                # Surface a typed error so callers can distinguish init
                # failures from execution failures and decide whether to
                # mark the iteration as a soft skip vs. a hard crash.
                raise MolmoSpacesResetError(
                    f"Failed to reset MolmoSpaces real bridge after "
                    f"{self._RESET_RETRY_BUDGET} attempts; "
                    f"last error: {type(last_exc).__name__}: {last_exc}"
                ) from last_exc

            self._refresh_current_obs_from_real_bridge()
            arm_mg = self._real_bridge.robot_view.get_move_group("arm")
            self.home_joint_position = np.array(arm_mg.joint_pos, dtype=np.float64)
            self._stabilize_after_reset()
            task_descriptor = self.get_task_descriptor()
            task_prompt = task_descriptor.get("language") or self._real_bridge.get_task_description()
        else:
            self._current_obs = self.bridge.build_observation(self.get_task_descriptor(), self._step_count)
            self._validate_observation(self._current_obs)
            task_prompt = self.get_task_descriptor().get("language", "")

        if self._record_frames:
            self._record_frame()
        self._publish_viser_update("reset")
        return self.get_observation(), {"task_prompt": task_prompt}

    # Number of reset attempts before giving up and raising
    # ``MolmoSpacesResetError``. The bridge already retries inside a
    # single sample_task call (see ``_resample_on_reset``); this wraps
    # that with one outer retry so transient *cross-attempt* state
    # (e.g. a stale remote-bridge socket on the first retry, or a
    # placement-cache that only clears between full reset() calls) gets
    # a second chance.
    _RESET_RETRY_BUDGET = 2

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        # Once the task has already terminated, further bridge.step calls
        # trigger the upstream "all environments already done" warning and
        # waste an RPC per call. Short-circuit with the last observation.
        if self._task_done:
            reward = self.compute_reward()
            return self.get_observation(), reward, True, False, {
                "task_completed": self._task_completed,
                "action": action,
                "short_circuited": True,
            }

        if self._real_bridge is not None:
            if isinstance(action, dict):
                _obs, _rew, term, trunc, _info = self._real_bridge.step(action)
                if term or trunc:
                    self._task_done = True
            self._current_obs = self._real_bridge.build_observation()
        else:
            self._current_obs = self.bridge.build_observation(self.get_task_descriptor(), self._step_count)

        if self._record_frames:
            self._record_frame()
        self._publish_viser_update("step")

        reward = self.compute_reward()
        terminated = self._task_done or reward >= 1.0
        truncated = self._step_count >= self.max_steps
        if terminated:
            self._task_done = True
        return self.get_observation(), reward, terminated, truncated, {
            "task_completed": self._task_completed,
            "action": action,
        }

    def close(self) -> None:
        if self._real_bridge is not None:
            close_fn = getattr(self._real_bridge, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.warning("Failed to close MolmoSpaces bridge cleanly", exc_info=True)
            self._real_bridge = None
        super().close()

    def get_observation(self) -> dict[str, Any]:
        if self._current_obs is None:
            if self._real_bridge is not None:
                self._refresh_current_obs_from_real_bridge()
            else:
                self._current_obs = self.bridge.build_observation(self.get_task_descriptor(), self._step_count)
            self._validate_observation(self._current_obs)
        obs = dict(self._current_obs)
        obs["task_descriptor"] = self.get_task_descriptor()
        return obs

    def compute_reward(self) -> float:
        if self._real_bridge is not None:
            return self._real_bridge.get_reward()
        return 1.0 if self._task_completed else 0.0

    def task_completed(self) -> bool:
        if self._real_bridge is not None:
            return self._real_bridge.judge_success()
        return self._task_completed

    def get_task_info(self) -> dict[str, Any]:
        """Per-subgoal task metrics from the underlying MuJoCo task.

        Returns an empty dict when no real bridge is attached (mock mode) or
        when the bridge does not expose ``get_info``. See
        ``MolmoSpacesBridge.get_info`` for the per-family key layout. Used by
        the verifier to emit per-predicate state without needing a second VLM
        call.
        """
        if self._real_bridge is not None and hasattr(self._real_bridge, "get_info"):
            try:
                return dict(self._real_bridge.get_info() or {})
            except Exception as exc:
                logger.debug("MolmoSpaces get_task_info failed: %s", exc)
                return {}
        return {}

    def describe_object_relation(self, pickup_obj_name: str, receptacle_name: str) -> dict[str, Any]:
        """Privileged MuJoCo support/contact relation for an arbitrary object pair."""
        if self._real_bridge is not None and hasattr(self._real_bridge, "describe_object_relation"):
            try:
                return dict(self._real_bridge.describe_object_relation(pickup_obj_name, receptacle_name) or {})
            except Exception as exc:
                logger.debug("MolmoSpaces describe_object_relation failed: %s", exc)
                return {"available": False, "error": str(exc)}
        return {"available": False, "error": "real_bridge_unavailable"}

    def start_state_trace(self, reason: str = "execution_start") -> None:
        """Start a compact dense numeric trace for one policy execution."""
        self._state_trace_samples = []
        self._state_trace_enabled = True
        self._append_state_trace_sample(reason)

    def clear_state_trace(self) -> None:
        self._state_trace_samples = []
        self._state_trace_enabled = False

    def get_state_trace(self, *, clear: bool = False) -> dict[str, Any]:
        trace = {
            "schema_version": "molmospaces_numeric_state_trace_v1",
            "sample_count": len(self._state_trace_samples),
            "sample_stride": 1,
            "source": "FrankaMolmoSpacesEnv",
            "samples": list(self._state_trace_samples),
        }
        if clear:
            self.clear_state_trace()
        return trace

    def _append_state_trace_sample(
        self,
        reason: str,
        *,
        action: dict[str, Any] | None = None,
        reward: Any | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
    ) -> None:
        if not self._state_trace_enabled:
            return
        if len(self._state_trace_samples) >= self._state_trace_max_samples:
            return
        sample: dict[str, Any] = {
            "step": len(self._state_trace_samples),
            "sim_step_count": int(self._sim_step_count),
            "reason": str(reason),
        }
        robot = self._numeric_robot_state_snapshot()
        if robot:
            sample["robot"] = robot
        task_info = self.get_task_info()
        if task_info:
            sample["task_info"] = self._trace_jsonable(task_info)
        contacts = self._numeric_contact_state_snapshot()
        if contacts:
            sample["contacts"] = contacts
        if action is not None:
            sample["action"] = self._trace_jsonable(action)
        if reward is not None:
            sample["reward"] = self._trace_jsonable(reward)
        if terminated is not None:
            sample["terminated"] = bool(terminated)
        if truncated is not None:
            sample["truncated"] = bool(truncated)
        self._state_trace_samples.append(sample)

    def _numeric_robot_state_snapshot(self) -> dict[str, Any]:
        if self._real_bridge is None:
            obs = self.get_observation()
            return {
                key: self._trace_jsonable(obs.get(key))
                for key in ("robot_cartesian_pos", "robot_joint_pos", "robot_base_pose")
                if key in obs
            }
        out: dict[str, Any] = {}
        try:
            arm_mg = self._real_bridge.robot_view.get_move_group("arm")
            joint_pos_7 = np.asarray(arm_mg.joint_pos, dtype=np.float64)
            out["robot_joint_pos_arm"] = self._trace_jsonable(joint_pos_7)
            ee_pose_mat = np.asarray(arm_mg.leaf_frame_to_world, dtype=np.float64)
            if ee_pose_mat.shape == (4, 4):
                ee_pos = ee_pose_mat[:3, 3]
                ee_quat_wxyz = self._rotation_matrix_to_wxyz(ee_pose_mat[:3, :3])
                out["robot_cartesian_pos"] = self._trace_jsonable(
                    np.concatenate([ee_pos, ee_quat_wxyz, [self._gripper_fraction]])
                )
        except Exception as exc:
            out["arm_error"] = str(exc)
        try:
            gripper_mg = self._real_bridge.robot_view.get_move_group("gripper")
            gripper_dist = float(gripper_mg.inter_finger_dist)
            gripper_range = gripper_mg.inter_finger_dist_range
            gripper_max = float(gripper_range[1]) if gripper_range else 0.0
            gripper_frac = gripper_dist / gripper_max if gripper_max > 0 else self._gripper_fraction
            out["gripper_fraction"] = float(gripper_frac)
            if "robot_cartesian_pos" in out and len(out["robot_cartesian_pos"]) >= 8:
                out["robot_cartesian_pos"][7] = float(gripper_frac)
        except Exception:
            out["gripper_fraction"] = float(self._gripper_fraction)
        return out

    def _numeric_contact_state_snapshot(self) -> dict[str, Any]:
        """Return compact robot end-effector contact-pair telemetry.

        This is sampled into the dense execution trace so the post-hoc
        privileged checker can verify exact robot↔target surface contact
        without relying on root-object proximity.  In remote mode the actual
        MuJoCo inspection is performed server-side by ``describe_contact_pairs``.
        """
        if self._real_bridge is None or not hasattr(self._real_bridge, "describe_contact_pairs"):
            return {}
        try:
            snapshot = self._real_bridge.describe_contact_pairs(max_pairs=64)
        except Exception as exc:
            return {
                "available": False,
                "not_available_reason": f"{type(exc).__name__}: {exc}",
                "robot_eef_contact_pairs": [],
                "robot_eef_contact_pair_count": 0,
            }
        if not isinstance(snapshot, dict):
            return {}
        return self._trace_jsonable(snapshot)

    @staticmethod
    def _rotation_matrix_to_wxyz(rot: np.ndarray) -> np.ndarray:
        """Convert a 3x3 rotation matrix to quaternion wxyz without SciPy."""
        m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        tr = float(np.trace(m))
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            return np.array([
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ], dtype=np.float64)
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            return np.array([
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ], dtype=np.float64)
        if i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            return np.array([
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ], dtype=np.float64)
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        return np.array([
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ], dtype=np.float64)

    @classmethod
    def _trace_jsonable(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): cls._trace_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._trace_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return cls._trace_jsonable(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    # ------------------------------------------------------------------
    # Robot control (real bridge only)
    # ------------------------------------------------------------------

    def reset_policy_motion_speed(self) -> dict[str, Any]:
        """Restore the configured default arm speed for a fresh episode.

        RATS policy code may temporarily slow down movement during an attempt
        via ``set_policy_motion_speed``. Resetting here prevents a cautious
        setting from silently leaking into the next task/attempt. CaP-X does
        not expose the policy helper, so its behavior remains the YAML default.
        """
        self.max_joint_step_rad = self._default_max_joint_step_rad
        self.move_max_steps = self._default_move_max_steps
        self._policy_motion_speed_label = "normal"
        return self.get_policy_motion_speed()

    def get_policy_motion_speed(self) -> dict[str, Any]:
        """Return the current arm-speed settings used by blocking moves."""
        return {
            "speed": self._policy_motion_speed_label,
            "max_joint_step_rad": float(self.max_joint_step_rad),
            "move_max_steps": int(self.move_max_steps),
            "default_max_joint_step_rad": float(self._default_max_joint_step_rad),
            "default_move_max_steps": int(self._default_move_max_steps),
        }

    def set_policy_motion_speed(
        self,
        speed: str | float = "normal",
        *,
        max_joint_step_rad: float | None = None,
        move_max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Set the arm speed used by future ``goto_pose``/joint moves.

        Args:
            speed: Preset name or relative scale. Presets are
                ``"very_slow"`` (25%), ``"slow"`` (50%), ``"normal"`` (100%),
                and ``"fast"`` (125%) of the YAML ``max_joint_step_rad``.
                Numeric values are interpreted as a relative scale where
                smaller is slower.
            max_joint_step_rad: Optional direct per-step joint delta cap.
                Smaller values move more cautiously.
            move_max_steps: Deprecated/ignored. Policies are not allowed to
                change the per-move step budget; the YAML
                ``move_max_steps`` remains authoritative.

        Returns:
            Dict with the active speed label and resolved movement settings.
        """
        presets = {
            "very_slow": 0.25,
            "cautious": 0.35,
            "slow": 0.5,
            "normal": 1.0,
            "default": 1.0,
            "fast": 1.25,
        }

        label: str
        if max_joint_step_rad is not None:
            step_cap = float(max_joint_step_rad)
            label = f"custom:{step_cap:g}"
            scale_for_steps = (
                step_cap / self._default_max_joint_step_rad
                if self._default_max_joint_step_rad > 0
                else 1.0
            )
        else:
            if isinstance(speed, str):
                key = speed.strip().lower().replace("-", "_")
                if key not in presets:
                    raise ValueError(
                        "Unknown arm speed preset "
                        f"{speed!r}; use one of {sorted(presets)} or a numeric scale."
                    )
                scale_for_steps = presets[key]
                label = key
            else:
                scale_for_steps = float(speed)
                label = f"{scale_for_steps:g}x"

            # Keep the control loop bounded. Speeds above 1.25x are rarely
            # useful in MolmoSpaces and increase table-object collisions.
            scale_for_steps = float(np.clip(scale_for_steps, 0.1, 1.25))
            step_cap = self._default_max_joint_step_rad * scale_for_steps

        if step_cap <= 0 or not np.isfinite(step_cap):
            raise ValueError(f"Invalid max_joint_step_rad: {step_cap}")

        ignored_move_max_steps = int(move_max_steps) if move_max_steps is not None else None
        # The policy-visible speed helper may adjust only the per-step joint
        # delta.  Keep the move step budget pinned to the configured YAML
        # value so generated policy code cannot stretch attempts into long
        # hidden retries via set_arm_speed(..., move_max_steps=...).
        resolved_steps = max(1, int(self._default_move_max_steps))

        self.max_joint_step_rad = float(step_cap)
        self.move_max_steps = resolved_steps
        self._policy_motion_speed_label = label
        info = self.get_policy_motion_speed()
        if ignored_move_max_steps is not None:
            info["ignored_move_max_steps_override"] = ignored_move_max_steps
        return info

    def _motion_progress_sample(
        self,
        *,
        step_i: int,
        error: float,
        current_joints: np.ndarray,
        waypoint: np.ndarray | None = None,
    ) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "step_i": int(step_i),
            "sim_step_count": int(self._sim_step_count),
            "joint_error_l2": float(error),
            "current_joints": self._trace_jsonable(current_joints),
        }
        if waypoint is not None:
            sample["commanded_waypoint"] = self._trace_jsonable(waypoint)
        robot = self._numeric_robot_state_snapshot()
        if robot:
            sample["robot"] = robot
        return sample

    def _motion_abort_diagnostics(
        self,
        *,
        reason: str,
        target_joints: np.ndarray,
        step_i: int,
        current_joints: np.ndarray,
        current_error: float,
        progress_trace: list[dict[str, Any]],
        loop_steps: int,
        tolerance: float,
        step_cap: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = getattr(self, "_active_motion_debug_context", {}) or {}
        diag: dict[str, Any] = {
            "reason": reason,
            "step_i": int(step_i),
            "loop_steps": int(loop_steps),
            "tolerance": float(tolerance),
            "max_joint_step_rad": float(step_cap),
            "target_joints": self._trace_jsonable(target_joints),
            "ik_solution": self._trace_jsonable(context.get("ik_solution", target_joints)),
            "target_pose": self._trace_jsonable(context.get("target_pose")),
            "target_quaternion_wxyz": self._trace_jsonable(context.get("target_quaternion_wxyz")),
            "current_joints": self._trace_jsonable(current_joints),
            "current_error_l2": float(current_error),
            "robot_pose_trace": progress_trace,
            "final_robot": self._numeric_robot_state_snapshot(),
            "motion_settings": self.get_policy_motion_speed(),
        }
        if extra:
            diag.update(extra)
        return diag

    def _raise_motion_abort(self, message: str, *, diagnostics: dict[str, Any]) -> None:
        raise MolmoSpacesMotionAbort(message, diagnostics=diagnostics)

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float = 0.01,
        max_steps: int | None = None,
        max_joint_step_rad: float | None = None,
        refresh_joints_every: int = 1,
    ) -> None:
        """Move to target joint positions by stepping the simulator in a loop.

        Each iteration sends an interpolated waypoint capped at
        ``max_joint_step_rad`` per joint, so the arm moves at a bounded speed
        instead of jumping straight to ``joints``. The per-move step budget is
        pinned to the YAML ``move_max_steps``; policy code cannot extend it via
        ``set_arm_speed(..., move_max_steps=...)`` or direct ``max_steps``.

        A progress guard aborts when the actual refreshed joint state does not
        reduce target error over a short window. Abort/timeout exceptions carry
        target pose, IK solution, and robot pose trace diagnostics for retry
        debugging.
        """
        if self._real_bridge is None:
            return
        # A previous goto_pose / gripper step may already have pushed the
        # task past the success threshold. Upstream responds to further
        # step() calls with "all environments already done" and the wire
        # otherwise deadlocks waiting on reward. Exit here.
        if self._task_done:
            return

        target = np.asarray(joints, dtype=np.float64).reshape(7)
        gripper_target = self._compute_gripper_target()
        step_cap = float(max_joint_step_rad) if max_joint_step_rad is not None else self.max_joint_step_rad
        # Lock the policy-facing move budget to the configured default.  A
        # non-None max_steps is accepted for API compatibility but ignored so
        # policy code cannot lengthen attempts by calling env directly.
        _ignored_max_steps = int(max_steps) if max_steps is not None else None
        loop_steps = max(1, int(self._default_move_max_steps))
        # Keep actual-state refresh pinned to every step for accurate motion
        # logging and progress detection; accept the argument only for API
        # compatibility with older call sites.
        _ignored_refresh_joints_every = int(refresh_joints_every)
        refresh_every = 1
        progress_window = max(1, int(getattr(self, "motion_progress_check_window", 20)))
        progress_min_reduction = max(
            0.0,
            float(getattr(self, "motion_progress_min_error_reduction", 2e-3)),
        )
        progress_enabled = bool(getattr(self, "motion_progress_abort_enabled", True))

        arm_mg = self._real_bridge.robot_view.get_move_group("arm")
        current = np.array(arm_mg.joint_pos, dtype=np.float64)
        progress_trace: list[dict[str, Any]] = []
        anchor_step = 0
        anchor_error = float(np.linalg.norm(current - target))
        converged = False
        last_error = anchor_error

        for step_i in range(loop_steps):
            if step_i > 0 and step_i % refresh_every == 0:
                arm_mg = self._real_bridge.robot_view.get_move_group("arm")
                current = np.array(arm_mg.joint_pos, dtype=np.float64)

            error = float(np.linalg.norm(current - target))
            last_error = error
            progress_trace.append(
                self._motion_progress_sample(
                    step_i=step_i,
                    error=error,
                    current_joints=current,
                )
            )
            if error < tolerance and step_i > 0:
                converged = True
                break

            if (
                progress_enabled
                and step_i - anchor_step >= progress_window
                and error >= tolerance
            ):
                reduction = anchor_error - error
                if reduction < progress_min_reduction:
                    diag = self._motion_abort_diagnostics(
                        reason="progress_stalled",
                        target_joints=target,
                        step_i=step_i,
                        current_joints=current,
                        current_error=error,
                        progress_trace=progress_trace,
                        loop_steps=loop_steps,
                        tolerance=tolerance,
                        step_cap=step_cap,
                        extra={
                            "anchor_step": int(anchor_step),
                            "anchor_error_l2": float(anchor_error),
                            "error_reduction_l2": float(reduction),
                            "required_error_reduction_l2": float(progress_min_reduction),
                            "ignored_max_steps_override": _ignored_max_steps,
                            "ignored_refresh_joints_every": _ignored_refresh_joints_every,
                        },
                    )
                    self._raise_motion_abort(
                        "move_to_joints aborted: no measurable progress toward target",
                        diagnostics=diag,
                    )
                anchor_step = step_i
                anchor_error = error

            delta = target - current
            max_abs = float(np.abs(delta).max())
            if step_cap > 0 and max_abs > step_cap:
                waypoint = current + delta * (step_cap / max_abs)
            else:
                waypoint = target
            if progress_trace:
                progress_trace[-1]["commanded_waypoint"] = self._trace_jsonable(waypoint)

            action = {"arm": waypoint, "gripper": gripper_target}
            try:
                _obs, _rew, term, trunc, _info = self._real_bridge.step(action)
            except Exception as exc:
                diag = self._motion_abort_diagnostics(
                    reason="bridge_step_exception",
                    target_joints=target,
                    step_i=step_i,
                    current_joints=current,
                    current_error=error,
                    progress_trace=progress_trace,
                    loop_steps=loop_steps,
                    tolerance=tolerance,
                    step_cap=step_cap,
                    extra={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "ignored_max_steps_override": _ignored_max_steps,
                        "ignored_refresh_joints_every": _ignored_refresh_joints_every,
                    },
                )
                self._raise_motion_abort(
                    f"move_to_joints bridge step failed at step {step_i}: {exc}",
                    diagnostics=diag,
                )
            self._sim_step_count += 1
            self._append_state_trace_sample(
                "move_to_joints_step",
                action=action,
                reward=_rew,
                terminated=term,
                truncated=trunc,
            )
            # Propagate the commanded waypoint as the next iteration's "current"
            # so non-default refresh intervals can still interpolate smoothly.
            # The default refresh interval is 1, so convergence/progress checks
            # use actual bridge state every simulator step.
            current = waypoint

            if (
                (self._record_frames or self._viser_publisher is not None)
                and self._sim_step_count % self._subsample_rate == 0
            ):
                self._refresh_current_obs_from_real_bridge()
                if self._record_frames:
                    self._record_frame()
                self._publish_viser_update("move_to_joints_step")

            if term or trunc:
                self._task_done = True
                break

        self._refresh_current_obs_from_real_bridge()
        self._validate_observation(self._current_obs)
        if not converged and not self._task_done:
            try:
                arm_mg = self._real_bridge.robot_view.get_move_group("arm")
                current = np.array(arm_mg.joint_pos, dtype=np.float64)
                final_error = float(np.linalg.norm(current - target))
            except Exception:
                final_error = float(last_error)
            if final_error >= tolerance:
                diag = self._motion_abort_diagnostics(
                    reason="move_max_steps_exhausted",
                    target_joints=target,
                    step_i=loop_steps,
                    current_joints=current,
                    current_error=final_error,
                    progress_trace=progress_trace,
                    loop_steps=loop_steps,
                    tolerance=tolerance,
                    step_cap=step_cap,
                    extra={
                        "ignored_max_steps_override": _ignored_max_steps,
                        "ignored_refresh_joints_every": _ignored_refresh_joints_every,
                    },
                )
                self._raise_motion_abort(
                    f"move_to_joints timed out after {loop_steps} steps; final error {final_error:.4f}",
                    diagnostics=diag,
                )

    def _set_gripper(self, fraction: float) -> None:
        """Set gripper opening fraction: 0.0 (closed) to 1.0 (open)."""
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _compute_gripper_target(self) -> np.ndarray:
        """Map abstract open/close state to the FrankaDroid gripper command."""
        # FrankaDroid / Robotiq-backed MolmoSpaces uses 0=open, 255=closed.
        return np.array([0.0 if self._gripper_fraction > 0.5 else 255.0], dtype=np.float64)

    def _step_once(self) -> None:
        """Execute one simulation step with current control state."""
        if self._real_bridge is None:
            return
        if self._task_done:
            return

        arm_mg = self._real_bridge.robot_view.get_move_group("arm")
        current_joints = np.array(arm_mg.joint_pos, dtype=np.float64)
        gripper_target = self._compute_gripper_target()

        action = {"arm": current_joints, "gripper": gripper_target}
        _obs, _rew, term, trunc, _info = self._real_bridge.step(action)
        self._sim_step_count += 1
        self._append_state_trace_sample(
            "_step_once",
            action=action,
            reward=_rew,
            terminated=term,
            truncated=trunc,
        )
        if term or trunc:
            self._task_done = True

        if (
            (self._record_frames or self._viser_publisher is not None)
            and self._sim_step_count % self._subsample_rate == 0
        ):
            self._refresh_current_obs_from_real_bridge()
            if self._record_frames:
                self._record_frame()
            self._publish_viser_update("_step_once")
        else:
            self._refresh_current_obs_from_real_bridge()

    # ------------------------------------------------------------------
    # Rendering and video capture
    # ------------------------------------------------------------------

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        _ = mode
        if self._real_bridge is not None:
            return self._real_bridge.render("agentview")
        return self.get_observation()["agentview"]["images"]["rgb"].copy()

    def render_wrist(self) -> np.ndarray:
        if self._real_bridge is not None:
            return self._real_bridge.render_wrist()
        return self.get_observation()["robot0_eye_in_hand"]["images"]["rgb"].copy()

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True, wrist_camera: bool = False) -> None:
        self._record_frames = enabled
        self._wrist_recording = wrist_camera
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._wrist_frame_buffer[start:end]]

    def _record_frame(self) -> None:
        self._frame_buffer.append(self.render())
        if self._wrist_recording:
            self._wrist_frame_buffer.append(self.render_wrist())

    def _publish_viser_update(self, reason: str) -> None:
        """Best-effort per-step MolmoSpaces Viser update when WebUI attached one."""
        publisher = getattr(self, "_viser_publisher", None)
        if publisher is None:
            return
        with suppress(Exception):
            publisher.publish_env(self, reason=reason)

    # ------------------------------------------------------------------
    # Privileged / bookkeeping helpers
    # ------------------------------------------------------------------

    def mark_task_complete(self, note: str | None = None) -> None:
        self._task_completed = True
        if note:
            self._runtime_notes.append(note)

    def append_runtime_note(self, note: str) -> None:
        self._runtime_notes.append(note)

    def runtime_summary(self) -> dict[str, Any]:
        return {
            "session_id": self._session["session_id"] if self._session else None,
            "canonical_task_id": self.canonical_task_id,
            "task_completed": self._task_completed or (self._real_bridge.judge_success() if self._real_bridge else False),
            "notes": list(self._runtime_notes),
            "last_reset_seed": self._last_reset_seed,
            "step_count": self._step_count,
        }


__all__ = [
    "FrankaMolmoSpacesEnv",
    "MockMolmoSpacesBridge",
    "MolmoSpacesResetError",
    "MolmoSpacesTaskDescriptor",
    "default_molmospaces_catalog",
    "load_benchmark_catalog",
    "parse_molmospaces_task_id",
]
