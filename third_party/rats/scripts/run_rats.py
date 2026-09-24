#!/usr/bin/env python3
"""Run the RATS lifelong learning loop.

Usage:
    # With real BEHAVIOR-1K environment:
    cd ~/rats
    source rats/third_party/b1k/.venv/bin/activate
    export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=7
    export HF_HUB_OFFLINE=1 OMNIGIBSON_APPDATA_PATH=$PWD/og_appdata
    export OPENAI_API_KEY='sk-proj-...'
    python scripts/run_rats.py --config env_configs/r1pro/r1pro_pick_up_radio.yaml --iterations 10

    # With real LIBERO environment:
    export OPENAI_API_KEY='sk-proj-...'
    python scripts/run_rats.py --config env_configs/libero/rats_libero_play.yaml --iterations 10
    # Or specify suite + task directly:
    python scripts/run_rats.py --env-type libero --libero-suite libero_spatial --libero-task 0 --iterations 10

    # With mock environment (for pipeline testing):
    python scripts/run_rats.py --mock --iterations 3
    python scripts/run_rats.py --mock --env-type libero --iterations 3
"""

from __future__ import annotations

# Shim: robosuite's log_utils.py hardcodes /tmp/robosuite.log. On a shared
# machine where another user already created that file we can't append to
# it; redirect the hardcoded path to a per-uid file BEFORE robosuite is
# imported transitively (robosuite gets pulled in by create_real_environment).
# We subclass FileHandler so downstream code that uses it as a base class
# (e.g. logging.handlers.BaseRotatingHandler) still works.
import logging as _logging
import os as _os
_ORIG_FILE_HANDLER = _logging.FileHandler
class _RedirectedFileHandler(_ORIG_FILE_HANDLER):
    def __init__(self, filename, *args, **kwargs):
        if str(filename) == "/tmp/robosuite.log":
            filename = f"/tmp/robosuite_{_os.getuid()}.log"
        super().__init__(filename, *args, **kwargs)
_logging.FileHandler = _RedirectedFileHandler

import argparse
import json
import logging
import os
import sys
from typing import Any
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rats.utils.json_safety import json_safe
from rats.integrations.libero import trial_seed_from_public_seed

DEFAULT_MOLMOSPACES_SEEDED_SKILL_LIBRARY = (
    "skill_library/molmospaces_seeded_skills.json"
)

# Point MolmoSpaces at the repo-local asset cache by default. Set
# MLSPACES_CACHE_DIR in your env to point at a shared cache if needed.
os.environ.setdefault("MLSPACES_CACHE_DIR", str(PROJECT_ROOT / "rats-cache" / "molmospaces"))
os.environ.setdefault(
    "MLSPACES_ASSETS_DIR",
    str(PROJECT_ROOT / "rats" / "third_party" / "molmospaces" / "assets"),
)


def _load_llm_keys_from_dotenv() -> None:
    """Load LLM API keys from project root ``.env`` when env vars are unset.

    Cursor/IDE subprocesses often do not inherit a shell ``export``; a one-line
    ``.env`` file lets runs pick up OpenAI, Gemini, or OpenRouter credentials.
    """
    wanted = {
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    }
    missing = {key for key in wanted if not os.environ.get(key)}
    if not missing:
        return
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, raw_val = line.partition("=")
            if not sep or key not in missing:
                continue
            val = raw_val.strip().strip('"').strip("'")
            if val:
                os.environ[key] = val
    except OSError:
        pass


def _canonicalize_llm_model(model: str) -> str:
    """Normalize common CLI shorthand without importing agents.base_agent early."""
    import re as _re

    model = str(model or "").strip()
    if not model:
        return model
    provider = ""
    name = model
    if "/" in model:
        provider, name = model.split("/", 1)
    name = _re.sub(r"^gpt(?=\d)", "gpt-", name, flags=_re.IGNORECASE)
    if provider:
        return f"{provider}/{name}"
    if name.lower().startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai/{name}"
    return name


_RUN_MODEL_ENV_KEYS = (
    "RATS_CURATOR_MODEL",
    "RATS_POLICY_WRITER_MODEL",
    "RATS_VERIFIER_MODEL",
    "RATS_DIAGNOSER_MODEL",
    "RATS_PER_STEP_VERIFIER_MODEL",
    "RATS_PLANNER_VERIFIER_MODEL",
    "RATS_FEEDBACK_GENERATOR_MODEL",
)


def _set_unified_run_model(model: str) -> None:
    """Make --model mean every RATS LLM agent, including per-agent defaults."""
    os.environ["RATS_LLM_MODEL"] = model
    for key in _RUN_MODEL_ENV_KEYS:
        os.environ[key] = model


def _load_run_config_payload(config_path: str | None) -> dict[str, Any]:
    """Load a YAML run config once for top-level runtime defaults.

    Environment YAMLs historically configured only the environment and
    MolmoSpaces proposer. A small top-level ``rats:`` block now lets a config
    also declare run-level carryover artifacts such as learned skills, failure
    memory, and playtime memory, while CLI flags still take precedence.
    """
    if not config_path:
        return {}
    try:
        import yaml

        with open(config_path) as fh:
            payload = yaml.safe_load(fh) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _csv_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(items) if items else None
    return str(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _apply_rats_runtime_defaults(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Apply top-level YAML run defaults without overriding explicit CLI args."""
    runtime = cfg.get("rats") or {}
    if not isinstance(runtime, dict):
        runtime = {}

    def get_runtime(*keys: str) -> Any:
        for key in keys:
            if key in runtime:
                return runtime[key]
        return None

    if args.skill_library is None:
        val = get_runtime("skill_library", "skill_library_path")
        if val:
            args.skill_library = str(val)
    if args.skill_library_merge is None:
        val = get_runtime("skill_library_merge", "skill_library_merge_paths")
        csv = _csv_or_none(val)
        if csv:
            args.skill_library_merge = csv
    if args.skill_library_min_tier is None:
        val = get_runtime("skill_library_min_tier")
        if val:
            args.skill_library_min_tier = str(val)
    if args.playtime_memory_seed is None:
        val = get_runtime("playtime_memory_seed", "playtime_memory_seed_path")
        if val:
            args.playtime_memory_seed = str(val)
    if args.failure_memory_path is None:
        val = get_runtime("failure_memory_path", "failure_memory_seed")
        if val:
            args.failure_memory_path = str(val)
    if args.molmospaces_seeded_skill_library is None:
        val = get_runtime(
            "molmospaces_seeded_skill_library",
            "molmospaces_seeded_skill_library_path",
        )
        if val:
            args.molmospaces_seeded_skill_library = str(val)
    if args.output_dir is None:
        val = get_runtime("output_dir") or cfg.get("output_dir")
        if val:
            args.output_dir = str(val)

    # Boolean CLI flags default to False; a YAML true enables them, but YAML
    # false never disables an explicit CLI true.
    for attr in ("no_skill_reuse", "no_failure_memory", "random_order", "fixed_task"):
        val = get_runtime(attr)
        if bool(val) and not bool(getattr(args, attr, False)):
            setattr(args, attr, True)

    val = get_runtime("include_molmospaces_seeded_skills")
    if _truthy(val) and not bool(
        getattr(args, "include_molmospaces_seeded_skills", False)
    ):
        args.include_molmospaces_seeded_skills = True


def discover_available_tasks(
    env_type: str = "behavior",
    scene_model: str | None = None,
    *,
    benchmark_dir: str | None = None,
) -> list[dict]:
    """Discover available tasks from env_configs and scene-compatible task lists.

    Args:
        env_type: ``"behavior"`` for BEHAVIOR-1K / R1Pro, ``"libero"`` for LIBERO.
        scene_model: If provided, also include all tasks compatible with this
            scene from OmniGibson's ``task_custom_lists.json``.
        benchmark_dir: MolmoSpaces benchmark directory (overrides hardcoded catalog).
    """
    if env_type == "libero":
        from rats.loop.libero_utils import discover_libero_tasks
        return discover_libero_tasks()
    if env_type == "molmospaces":
        from rats.loop.molmospaces_utils import discover_molmospaces_tasks
        return discover_molmospaces_tasks(benchmark_dir=benchmark_dir)

    tasks = []
    seen_activities: set[str] = set()
    try:
        from rats.rats.catalog import discover_r1pro_task_catalog
        catalog = discover_r1pro_task_catalog()
        for entry in catalog:
            tasks.append({
                "activity_name": entry.activity_name,
                "scene_model": entry.scene_model,
                "activity_definition_id": entry.activity_definition_id,
                "env_config_path": entry.env_config_path,
            })
            seen_activities.add(entry.activity_name)
    except Exception:
        # Fallback: scan env_configs directory
        config_dir = PROJECT_ROOT / "env_configs" / "r1pro"
        if config_dir.exists():
            import yaml
            for path in sorted(config_dir.glob("b1k_*.yaml")):
                try:
                    with path.open() as f:
                        cfg = yaml.safe_load(f)
                    low_level = cfg.get("env", {}).get("cfg", {}).get("low_level", {})
                    activity = low_level.get("activity_name", path.stem.replace("b1k_", ""))
                    tasks.append({
                        "activity_name": activity,
                        "scene_model": "",
                        "activity_definition_id": 0,
                        "env_config_path": str(path),
                    })
                    seen_activities.add(activity)
                except Exception:
                    continue

    # Discover scene-compatible tasks from task_custom_lists.json
    if scene_model:
        tcl_path = (
            PROJECT_ROOT / "rats" / "third_party" / "b1k" / "OmniGibson"
            / "omnigibson" / "sampling" / "task_custom_lists.json"
        )
        if tcl_path.exists():
            import json
            try:
                with tcl_path.open() as f:
                    tcl_data = json.load(f)
                for task_name, task_info in tcl_data.items():
                    if isinstance(task_info, dict) and scene_model in task_info:
                        if task_name not in seen_activities:
                            tasks.append({
                                "activity_name": task_name,
                                "scene_model": scene_model,
                                "activity_definition_id": 0,
                            })
                            seen_activities.add(task_name)
            except Exception:
                pass

    return tasks


def create_real_environment(
    config_path: str,
    *,
    libero_suite: str | None = None,
    libero_task: int | None = None,
    viser_debug: bool = False,
):
    """Create a real CaP-Gym BEHAVIOR-1K or LIBERO environment.

    If ``libero_suite`` / ``libero_task`` are given, they override the
    ``low_level.suite_name`` / ``low_level.task_id`` fields from the YAML.
    Useful for parallel LIBERO runs that share one config but target
    different tasks per worker.
    """
    import yaml
    from rats.envs.configs.instantiate import instantiate

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env_factory = cfg.get("env", {})
    if libero_suite is not None or libero_task is not None:
        low = env_factory.setdefault("cfg", {}).setdefault("low_level", {})
        if libero_suite is not None:
            low["suite_name"] = libero_suite
        if libero_task is not None:
            low["task_id"] = int(libero_task)
    if viser_debug:
        cfg = env_factory.setdefault("cfg", {})
        cfg["viser_debug"] = True
        low = cfg.setdefault("low_level", {})
        if isinstance(low, dict):
            low["viser_debug"] = True
    # Strip CLI args to avoid OmniGibson crashes
    original_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        env = instantiate(env_factory)
    finally:
        sys.argv = original_argv

    return env


class MockEnvironment:
    """Mock environment for testing the pipeline without OmniGibson."""

    def __init__(self, scene_model: str = "house_double_floor_lower"):
        self._task_prompt = "Pick up the red radio from the table."
        self._step_count = 0
        self.low_level_env = self
        self.task_name = "pick_up_radio"
        # Mock OmniGibson env/task/scene structure for validation
        self.env = type("MockOGEnv", (), {
            "task": type("MockTask", (), {
                "scene_name": scene_model,
                "activity_name": "turning_on_radio",
                "activity_definition_id": 0,
                "object_scope": {},
                "low_dim_obs_keys": [],
                "update_activity": lambda self, **kw: None,
                "initialize_activity": lambda self, **kw: (True, None),
            })(),
            "scene": type("MockScene", (), {
                "write_task_metadata": lambda self, **kw: None,
                "update_initial_file": lambda self: None,
                "reset": lambda self: None,
            })(),
        })()
        # Simulate available functions from R1Pro Control API
        self._mock_functions = [
            "get_env_observation", "get_object_pose", "get_sam3_mask",
            "segment_sam3_text_prompt", "point_prompt_molmo",
            "navigate_to_pose", "get_navigation_pose",
            "move_hand", "move_to_joint_positions", "solve_ik",
            "get_current_eef_pose", "get_current_joint_positions",
            "get_robot_position", "lift_arm", "reset_torso",
            "sample_grasp_pose", "grasp_object",
            "open_gripper", "close_gripper", "check_object_in_hand",
            "save_current_observation", "write_video",
        ]
        # Build mock API object
        self._apis = {"R1ProControlApi": type("MockApi", (), {
            "functions": lambda self: {fn: None for fn in self._fns},
            "combined_doc": lambda self: "get_env_observation() -> (rgb, depth)\\n"
                "get_object_pose(object_name: str) -> (position, quaternion)\\n"
                "sample_grasp_pose(object_name: str) -> (pregrasp_poses, grasp_poses)\\n"
                "grasp_object(pregrasp, grasp, object_name, arm=0) -> None\\n"
                "navigate_to_pose(pose_2d) -> None\\n"
                "open_gripper(arm=0) -> None\\nclose_gripper(arm=0) -> None\\n"
                "check_object_in_hand(arm=0) -> bool\\n"
                "solve_ik(position, quaternion_wxyz, arm=0) -> joint_angles\\n"
                "move_to_joint_positions(joints, max_steps=20) -> None\\n"
                "move_hand(target_pose, arm=0) -> None\\n"
                "get_robot_position() -> (position, quaternion, yaw)\\n"
                "lift_arm(arm=0) -> None\\nreset_torso() -> None",
            "_fns": self._mock_functions,
        })()}

    def configure_behavior_task(self, activity_name: str, activity_definition_id: int = 0):
        self.task_name = activity_name
        self.env.task.activity_name = activity_name
        return True, None

    def load_task_instance(self, instance_id: int):
        pass

    def reset(self, **kwargs):
        self._step_count = 0
        obs = {"full_prompt": [{"role": "user", "content": [{"type": "text", "text": self._task_prompt}]}]}
        return obs, {}

    def step(self, code: str):
        self._step_count += 1
        # Simulate execution
        stdout = ""
        stderr = ""
        reward = 0.0
        task_completed = False

        try:
            compile(code, "<policy>", "exec")
            stdout = f"Mock execution of {len(code)} chars code. Step {self._step_count}."
            # Simulate some success probability
            import random
            if random.random() < 0.3:
                reward = 1.0
                task_completed = True
                stdout += " Task completed!"
        except SyntaxError as e:
            stderr = f"SyntaxError: {e.msg} at line {e.lineno}"

        obs = {}
        info = {
            "sandbox_rc": 0 if not stderr else 1,
            "stdout": stdout,
            "stderr": stderr,
            "task_completed": task_completed,
        }
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        import numpy as np
        return np.zeros((64, 64, 3), dtype=np.uint8)


def create_libero_environment(
    config_path: str | None = None,
    *,
    suite_name: str = "libero_spatial",
    task_id: int = 0,
    viser_debug: bool = False,
):
    """Create a real LIBERO environment.

    Either from a YAML config (same code path as BEHAVIOR) or directly from
    suite_name + task_id.
    """
    if config_path:
        return create_real_environment(config_path, viser_debug=viser_debug)

    # Direct construction without YAML
    import yaml
    from rats.envs.configs.instantiate import instantiate

    cfg = {
        "_target_": "rats.envs.tasks.franka.franka_libero_env.FrankaLiberoCodeEnv",
        "cfg": {
            "_target_": "rats.envs.tasks.base.CodeExecEnvConfig",
            "low_level": {
                "_target_": "rats.envs.simulators.libero.FrankaLiberoEnv",
                "suite_name": suite_name,
                "task_id": task_id,
                "viser_debug": viser_debug,
            },
            "privileged": False,
            "viser_debug": viser_debug,
            "apis": ["FrankaLiberoApi"],
            "prompt": (
                "You are controlling a Franka Emika robot with API described below.\n"
                f"Goal: {{libero_environment_goal}}\n"
                "You may write python code comments for reasoning but ONLY write "
                "the executable Python code and do not write it in code fences.\n"
                "The functions (APIs) below are already imported to the environment. "
                "If you want to use numpy, you need to import it explicitly.\n"
            ),
        },
    }
    original_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        env = instantiate(cfg)
    finally:
        sys.argv = original_argv
    return env


def create_molmospaces_environment(
    config_path: str | None = None,
    *,
    benchmark: str = "phase1",
    scene_family: str = "kitchen",
    task_family: str = "put_bowl_on_plate",
    variant: str = "default",
    benchmark_dir: str | None = None,
    use_real_bridge: bool = False,
    remote_bridge_url: str | None = None,
    task_type: str | None = None,
    scene_dataset: str = "procthor-10k",
    data_split: str = "train",
    house_index: int | None = None,
    max_steps: int = 4000,
    seed: int | None = None,
):
    """Create a MolmoSpaces environment (mock, real bridge, or remote bridge)."""
    if config_path:
        return create_real_environment(config_path)

    from rats.envs.configs.instantiate import instantiate

    use_control_api = use_real_bridge or remote_bridge_url

    low_level_cfg: dict[str, Any] = {
        "_target_": "rats.envs.simulators.molmospaces.FrankaMolmoSpacesEnv",
        "benchmark": benchmark,
        "scene_family": scene_family,
        "task_family": task_family,
        "variant": variant,
        "use_real_bridge": use_real_bridge,
    }
    if remote_bridge_url:
        low_level_cfg["remote_bridge_url"] = remote_bridge_url
    if benchmark_dir:
        low_level_cfg["benchmark_dir"] = benchmark_dir
    if use_control_api:
        low_level_cfg["max_steps"] = max_steps
        if task_type:
            low_level_cfg["task_type"] = task_type
        low_level_cfg["scene_dataset"] = scene_dataset
        low_level_cfg["data_split"] = data_split
        if house_index is not None:
            low_level_cfg["house_index"] = house_index
        if seed is not None:
            low_level_cfg["seed"] = seed

    if use_control_api:
        apis = ["FrankaMolmoSpacesControlApi"]
        privileged = False
        prompt = (
            "You are controlling a Franka robot arm in a MolmoSpaces MuJoCo environment.\n"
            f"Goal: {task_family.replace('_', ' ')}\n"
            "You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.\n"
            "The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.\n"
        )
    else:
        apis = ["FrankaMolmoSpacesPrivilegedApi"]
        privileged = True
        prompt = (
            "You are controlling a Franka robot through the MolmoSpaces phase-1 bridge runtime.\n"
            f"Goal: {task_family.replace('_', ' ')}\n"
            "You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.\n"
            "The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.\n"
        )

    cfg = {
        "_target_": "rats.envs.tasks.franka.franka_molmospaces_env.FrankaMolmoSpacesCodeEnv",
        "cfg": {
            "_target_": "rats.envs.tasks.base.CodeExecEnvConfig",
            "low_level": low_level_cfg,
            "privileged": privileged,
            "apis": apis,
            "prompt": prompt,
        },
    }
    original_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        env = instantiate(cfg)
    finally:
        sys.argv = original_argv
    return env


class MockLiberoEnvironment:
    """Mock LIBERO environment for testing the pipeline without robosuite/MuJoCo."""

    def __init__(self, suite_name: str = "libero_spatial", task_id: int = 0):
        self._suite_name = suite_name
        self._task_id = task_id
        self._task_language = f"pick up the black bowl and place it on the plate (mock {suite_name} task {task_id})"
        self._task_prompt = (
            f"You are controlling a Franka Emika robot.\n"
            f"Goal: {self._task_language}\n"
        )
        self._step_count = 0
        self.low_level_env = self
        # Simulate the LIBERO handle for env-type detection
        self.handle = type("MockHandle", (), {
            "suite_name": suite_name,
            "task_id": task_id,
            "task_language": self._task_language,
            "env": None,
            "init_states": None,
        })()
        # Franka LIBERO API functions
        self._mock_functions = [
            "get_observation", "get_object_pose",
            "sample_grasp_pose", "goto_pose",
            "open_gripper", "close_gripper",
            "get_oriented_bounding_box_from_3d_points",
            "get_object_3d_points_and_masks_from_language",
            "goto_home_joint_position",
            "segment_sam3_text_prompt", "segment_sam3_point_prompt",
            "point_prompt_molmo",
        ]
        self._apis = {"FrankaLiberoApi": type("MockApi", (), {
            "functions": lambda self: {fn: None for fn in self._fns},
            "combined_doc": lambda self: (
                "get_observation() -> dict: Get RGB, depth, and robot state.\n"
                "get_object_pose(object_name: str) -> (position, quaternion_wxyz): Get object pose.\n"
                "sample_grasp_pose(object_name: str) -> (positions, quaternions, scores): Plan grasps.\n"
                "goto_pose(position, quaternion_wxyz, z_approach=0.0) -> None: Move end-effector.\n"
                "open_gripper() -> None: Open gripper fully.\n"
                "close_gripper() -> None: Close gripper fully.\n"
                "goto_home_joint_position() -> None: Move to home position.\n"
                "get_object_3d_points_and_masks_from_language(object_name) -> dict: "
                "Get 3D points and masks for an object.\n"
                "get_oriented_bounding_box_from_3d_points(points) -> dict: Get OBB from points.\n"
            ),
            "_fns": self._mock_functions,
        })()}
        # Mock observation with some objects
        self._current_obs = {
            "black_bowl_1_pos": [0.1, 0.2, 0.8],
            "black_bowl_1_quat": [0, 0, 0, 1],
            "plate_1_pos": [0.3, 0.0, 0.8],
            "plate_1_quat": [0, 0, 0, 1],
        }

    def reset(self, **kwargs):
        self._step_count = 0
        obs = {"full_prompt": [{"role": "user", "content": [
            {"type": "text", "text": self._task_prompt}
        ]}]}
        return obs, {"task_prompt": self._task_prompt}

    def step(self, code: str):
        self._step_count += 1
        stdout = ""
        stderr = ""
        reward = 0.0
        task_completed = False

        try:
            compile(code, "<policy>", "exec")
            stdout = f"Mock LIBERO execution of {len(code)} chars. Step {self._step_count}."
            import random
            if random.random() < 0.3:
                reward = 1.0
                task_completed = True
                stdout += " Task completed!"
        except SyntaxError as e:
            stderr = f"SyntaxError: {e.msg} at line {e.lineno}"

        obs = {}
        info = {
            "sandbox_rc": 0 if not stderr else 1,
            "stdout": stdout,
            "stderr": stderr,
            "task_completed": task_completed,
        }
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        import numpy as np
        return np.zeros((512, 800, 3), dtype=np.uint8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RATS lifelong learning loop")
    parser.add_argument("--config", type=str, default=None, help="Environment config YAML path")
    parser.add_argument(
        "--model",
        "--llm-model",
        dest="model",
        type=str,
        default=None,
        help=(
            "LLM model for all RATS LLM agents. Overrides rats/config/default.yaml "
            "and RATS_LLM_MODEL for this run. Examples: gpt5.5, gpt-5.5, "
            "openai/gpt-5.5."
        ),
    )
    parser.add_argument("--mock", action="store_true", help="Use mock environment")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per task")
    parser.add_argument("--timeout", type=int, default=None, help="Execution timeout seconds (default: from rats/config/default.yaml)")
    parser.add_argument(
        "--policy-self-check-repairs",
        type=int,
        default=None,
        help=(
            "Maximum policy-writer repair passes after runtime self-check "
            "crashes before official execution. Default comes from "
            "rats/config/default.yaml::policy_self_check_max_repairs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to outputs/rats_lifelong, or to the "
            "--resume directory when resuming in place."
        ),
    )
    parser.add_argument("--skill-library", type=str, default=None, help="Skill library path (auto-selected per env type if unset)")
    parser.add_argument(
        "--skill-library-merge",
        type=str,
        default=None,
        help=(
            "Comma-separated extra skill library JSON paths to merge into "
            "the seed library. Names colliding with the seed have their "
            "usage_count / success_count summed across all sources, so a "
            "skill that worked 5x in playtime + 4x in curriculum starts "
            "this run with empirical reliability of 9/9 instead of 0/0. "
            "Skills only present in the extras are appended verbatim. "
            "Useful for carrying playtime/curriculum-discovered skills "
            "into a benchmark run."
        ),
    )
    parser.add_argument(
        "--skill-library-min-tier",
        type=str,
        default=None,
        choices=["experimental", "verified"],
        help=(
            "Drop seed/extra learned skills below this tier before the "
            "run starts. 'verified' is the strictest setting and only "
            "carries skills that an upstream run promoted via successful "
            "usage. Primitives are always retained regardless of this "
            "filter. Default (unset) keeps experimental + verified, drops "
            "deprecated."
        ),
    )
    parser.add_argument(
        "--include-molmospaces-seeded-skills",
        action="store_true",
        help=(
            "Opt in to MolmoSpaces helper skills seeded outside the primitive "
            "API surface. Defaults off so skill_library/molmospaces_nonpriv_skills.json "
            "stays primitive-only. Equivalent YAML: "
            "rats.include_molmospaces_seeded_skills: true."
        ),
    )
    parser.add_argument(
        "--molmospaces-seeded-skill-library",
        type=str,
        default=None,
        help=(
            "Path to the MolmoSpaces seeded helper skill JSON merged when "
            "--include-molmospaces-seeded-skills is set. Defaults to "
            f"{DEFAULT_MOLMOSPACES_SEEDED_SKILL_LIBRARY}."
        ),
    )
    parser.add_argument(
        "--playtime-memory-seed",
        type=str,
        default=None,
        help=(
            "Path to a prior run's playtime_memory.jsonl. Loaded "
            "READ-ONLY (the source file is never modified). The seed's "
            "category-level affordance cards (e.g. 'spoons slide easily, "
            "approach vertically') are surfaced to the policy writer "
            "alongside this run's own playtime memory. Per-object cards "
            "are intentionally NOT used since their ProcTHOR internal "
            "names are house-specific. Typical use: feed an earlier "
            "playtime run's memory into a benchmark run."
        ),
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--log-agent-io",
        action="store_true",
        help=(
            "Record every query_llm call to <output_dir>/agent_io/ as a "
            "self-contained JSON record (image/video bytes extracted to sibling "
            "files). Use scripts/agent_io_viewer.py to bundle the directory "
            "into a portable HTML viewer. Honoured unless RATS_AGENT_IO_DIR "
            "is already set in the environment."
        ),
    )
    parser.add_argument(
        "--viser-debug",
        action="store_true",
        help="Enable Viser browser-based 3D debugging for LIBERO envs, "
             "including explore-mode generated BDDL envs.",
    )
    parser.add_argument(
        "--web-ui",
        "--rats-web-ui",
        action="store_true",
        help=(
            "Start the CaP-X React/FastAPI Web UI as a live RATS debugger. "
            "This implies --viser-debug where the environment supports it; "
            "MolmoSpaces attempt videos and execution steps are time-aligned "
            "when video capture is available."
        ),
    )
    parser.add_argument(
        "--web-ui-port",
        type=int,
        default=8200,
        help="Port for --web-ui live debugger (default: 8200).",
    )
    parser.add_argument("--fixed-task", action="store_true", help="Run the same task each iteration (no rebinding)")
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Curiosity mode: Task Proposer + rebind env each iteration. "
        "Requires --config to bootstrap the real BEHAVIOR env (or use --mock for tests).",
    )
    # LIBERO-specific options
    parser.add_argument(
        "--env-type",
        type=str,
        default=None,
        choices=["behavior", "libero", "molmospaces"],
        help="Environment type. Auto-detected from --config if unset.",
    )
    parser.add_argument("--scene-model", type=str, default=None,
                        help="Scene model name for task discovery (e.g. house_double_floor_lower)")
    parser.add_argument("--libero-suite", type=str, default=None,
                        help="LIBERO suite name. Overrides YAML when --config is set; "
                             "falls back to 'libero_spatial' when no --config.")
    parser.add_argument("--libero-task", type=int, default=None,
                        help="LIBERO task index. Overrides YAML when --config is set; "
                             "falls back to 0 when no --config.")
    parser.add_argument("--molmospaces-benchmark", type=str, default="phase1",
                        help="MolmoSpaces benchmark/catalog name (used with --env-type molmospaces when no --config)")
    parser.add_argument("--molmospaces-scene-family", type=str, default="kitchen",
                        help="MolmoSpaces scene family (used with --env-type molmospaces when no --config)")
    parser.add_argument("--molmospaces-task-family", type=str, default="put_bowl_on_plate",
                        help="MolmoSpaces task family (used with --env-type molmospaces when no --config)")
    parser.add_argument("--molmospaces-variant", type=str, default="default",
                        help="MolmoSpaces task variant (used with --env-type molmospaces when no --config)")
    parser.add_argument("--molmospaces-benchmark-dir", type=str, default=None,
                        help="Path to MolmoSpaces benchmark dir (overrides hardcoded catalog)")
    parser.add_argument("--use-real-bridge", action="store_true", default=False,
                        help="Use real MolmoSpaces MuJoCo bridge instead of mock")
    parser.add_argument("--remote-bridge-url", type=str, default=None,
                        help="Connect to a remote mlspaces_server.py (e.g. localhost:9100) "
                        "for two-process MolmoSpaces operation")
    parser.add_argument("--molmospaces-task-type", type=str, default=None,
                        choices=["pick", "pick_and_place", "open", "close"],
                        help="MolmoSpaces task type (pick, pick_and_place, open, close)")
    parser.add_argument("--molmospaces-scene-dataset", type=str, default="procthor-10k",
                        help="MolmoSpaces scene dataset name")
    parser.add_argument("--molmospaces-house-index", type=int, default=None,
                        help="MolmoSpaces specific house index to use")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a previous run's output dir (loads skill library + iteration count)")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Resume from the output dir while skipping existing completed "
            "iterations. MolmoSpaces playtime no-safe-target fallback "
            "iterations are not skipped, so they can be rerun after fixes."
        ),
    )
    parser.add_argument("--failure-memory-path", type=str, default=None,
                        help="Path to existing failure memory dir for cross-run persistence")
    parser.add_argument("--catalog", action="store_true",
                        help="Use catalog (predefined) tasks instead of novel generation in LIBERO explore mode")
    parser.add_argument(
        "--proposer-include-eval-task-context",
        action="store_true",
        help=(
            "Prepend downstream evaluation-task context to the play task "
            "proposer prompt. MolmoSpaces uses the Core40 benchmark JSON "
            "unless the config overrides "
            "molmospaces.playtime.eval_task_context_benchmark_dir."
        ),
    )
    parser.add_argument("--curriculum", action="store_true",
                        help="Gate novel task proposer by success count: start with single-step "
                             "pick+place (stage 1, <3 successes), unlock novel single-step "
                             "variants (stage 2, 3-9), then compound tasks (stage 3, 10+). "
                             "Fixes explore-mode over-scoping where proposer picks libero_10 "
                             "compound tasks the seed library can't solve.")
    parser.add_argument(
        "--curiosity",
        action="store_true",
        help="DEPRECATED / no-op. LIBERO explore runs always use the "
             "novelty×frontier candidate scorer now; this flag is retained "
             "only so older launch scripts that pass it don't error.",
    )
    parser.add_argument(
        "--proposer-no-context",
        action="store_true",
        help="Strip skill library + task history + curriculum hint from the "
             "task-proposer prompt (keeps static catalog / env limitations / "
             "pick reliability blocks). Used for ablations that measure how "
             "much the proposer depends on run-state context vs prior facts "
             "about the environment.",
    )
    parser.add_argument(
        "--proposer-temperature",
        type=float,
        default=None,
        help="Override the LLM sampling temperature ONLY for the task "
             "proposer's candidate-pool call (propose_novel_candidates). "
             "Other agents (writer/planner/verifier) keep the global default "
             "from rats/config/default.yaml (0.2). Bump to 0.7-0.9 when running "
             "--proposer-no-context to break the deterministic mode-collapse "
             "that a static prompt + low temperature otherwise produces.",
    )
    parser.add_argument("--num-fresh-candidates", type=int, default=3,
                        help="K — fresh candidates proposed per iteration.")
    parser.add_argument("--num-retry-candidates", type=int, default=2,
                        help="K_retry — retry-derived candidates per iteration (0 disables).")
    parser.add_argument("--curiosity-warmup-iters", type=int, default=0,
                        help="Skip the candidate-based selection for the first N iterations. "
                             "During warmup the legacy single-propose path runs instead "
                             "(simple atomic tasks). After N iters, candidate mode activates "
                             "with whatever skill reliability has accumulated. Fixes the "
                             "Goldilocks-cold-start bias toward compound tasks with many "
                             "unknown skill names. Only meaningful for "
                             "LIBERO explore runs.")
    parser.add_argument("--snapshot-interval", type=int, default=0,
                        help="Save a snapshot of skills.json + failure_memory every N iterations "
                             "into output_dir/snapshots/iterNNN/. 0 = disabled.")
    parser.add_argument("--retry-bank-size", type=int, default=8,
                        help="Max items the retry bank retains.")
    parser.add_argument("--retry-bank-ttl", type=int, default=3,
                        help="Initial TTL for new retry-bank items; decremented each iteration.")
    parser.add_argument("--retry-bonus-weight", type=float, default=0.15,
                        help="Weight of retry_bonus in final_score. Keep low to avoid "
                             "the loop collapsing onto past failures.")
    parser.add_argument("--failure-penalty-weight", type=float, default=0.10,
                        help="Weight of recent-failure penalty in final_score.")
    parser.add_argument("--score-composition",
                        choices=["product", "weighted_sum"], default="product",
                        help="How to combine novelty + frontier: 'product' "
                             "(N*F, Goldilocks-shaped) or 'weighted_sum' "
                             "(0.5N + 0.5F).")
    parser.add_argument("--retry-min-pred-success", type=float, default=0.5,
                        help="Predicted-success threshold below which a "
                             "failure is NOT added to the retry bank.")
    # Ablation flags
    parser.add_argument("--no-skill-reuse", action="store_true",
                        help="Ablation: disable skill library storage and retrieval (planner sees only primitives)")
    parser.add_argument("--random-order", action="store_true",
                        help="Ablation: replace curiosity-driven proposer with uniform random task selection")
    parser.add_argument("--no-failure-memory", action="store_true",
                        help="Ablation: disable failure memory (no recording, no retrieval)")
    parser.add_argument("--ensemble", type=int, default=0,
                        help="Number of ensemble candidates (0=disabled, 3-9 recommended). "
                             "Generates N code candidates at different temperatures, picks best.")
    # Play mode is the DEFAULT explore proposer style — a 3-4 year old
    # curious-child persona used for every iteration. Disable with
    # --no-play-mode to fall back to the adult-style "novel" proposer.
    # Legacy flags (--play-prompt-mode, --play-iterations) are kept as
    # silent no-ops for back-compat with old launch scripts; they no
    # longer change behavior since the warm-up phase has been retired
    # (play mode runs for the entire run by default).
    parser.add_argument(
        "--play-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Play mode on by default — task proposer uses the unified "
             "3-4 year-old curious-child prompt for the whole run, no "
             "warm-up flip. Use --no-play-mode to disable.",
    )
    parser.add_argument("--play-prompt-mode", action="store_true",
                        help=argparse.SUPPRESS)  # deprecated alias for --play-mode
    parser.add_argument("--play-iterations", type=int, default=0,
                        help=argparse.SUPPRESS)  # deprecated; warm-up retired
    parser.add_argument(
        "--turns-per-attempt",
        type=int,
        default=None,
        help=(
            "Number of policy_writer turns inside one attempt. When >1, the "
            "env is NOT reset between turns (state persists, so turn 2 can "
            "build on turn 1's grasp). Mirrors capx's multi_turn_limit. If "
            "unset, falls back to the YAML config's 'turns_per_attempt' "
            "(or legacy 'turns_per_iteration') key, then to 1."
        ),
    )
    parser.add_argument(
        "--attempts-per-iteration",
        type=int,
        default=None,
        help=(
            "Number of attempts per iteration. Each attempt starts with a "
            "fresh env reset and runs up to --turns-per-attempt turns. "
            "Default = max_retries + 1 (legacy single-shot retry budget). "
            "Set to 5 with --turns-per-attempt 5 to match capx's "
            "5 trials × 5 turns = 25 code-gens per task."
        ),
    )
    parser.add_argument(
        "--multi-turn-decision",
        action="store_true",
        default=None,
        help=(
            "Enable the CaP-X-style intra-attempt FINISH/REGENERATE decider. "
            "After each non-terminal turn (only effective when "
            "--turns-per-attempt > 1), a single LLM call inspects the "
            "after-frame + console output and either returns FINISH (fall "
            "through to verifier+diagnoser) or REGENERATE+code (skip the "
            "heavy pipeline and feed the new code straight into the next "
            "turn). Off by default; can also be enabled via YAML key "
            "'multi_turn_decision: true'."
        ),
    )
    parser.add_argument(
        "--multiturn-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "LIBERO-only opt-in mode (independent of --multi-turn-decision). "
            "Run each plan step in isolation: write step code, execute with "
            "prior committed steps replaying from a deterministic reset, ask "
            "per_step_verifier whether THIS step succeeded, commit on pass / "
            "retry-this-step-only on fail. On step-level stagnation (default "
            "10 retries) escalate to a new attempt's plan rewrite. Default off."
        ),
    )
    parser.add_argument(
        "--multiturn-reset-max-step-retries",
        type=int,
        default=10,
        help="Max retries per plan step before escalating to plan rewrite (default 10).",
    )
    parser.add_argument(
        "--collision-aware-motion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "MolmoSpaces: enable collision-aware IK + trajectory planning. "
            "Uses scene obstacles for pyroki solve_ik and trajopt. "
            "Default off (causes trajectory drift issues)."
        ),
    )
    return parser


def main():
    _load_llm_keys_from_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    # Set the unified LLM model before importing loop/agent modules. Several
    # agent modules import agents.base_agent at module import time; keeping this
    # early makes --model affect planner, writer, diagnoser, verifier, task
    # proposer, environment creator, skill proposer, and curation calls.
    if args.model:
        args.model = _canonicalize_llm_model(args.model)
        # A run-level --model should mean every RATS LLM call uses the same
        # model. Some newer agents intentionally have separate defaults
        # (policy_writer / verifier / diagnoser / planner_verifier); pin them
        # here so `--model openai/gpt-5.5` cannot silently route through
        # Gemini/OpenRouter.
        _set_unified_run_model(args.model)
    elif os.environ.get("RATS_LLM_MODEL"):
        os.environ["RATS_LLM_MODEL"] = _canonicalize_llm_model(os.environ["RATS_LLM_MODEL"])

    # Resolve env_type: explicit flag > auto-detect from config > default
    if args.env_type is None and args.config:
        from rats.loop.libero_utils import detect_env_type_from_config
        args.env_type = detect_env_type_from_config(args.config)
    if args.env_type is None:
        args.env_type = "behavior"

    run_cfg = _load_run_config_payload(args.config)
    _apply_rats_runtime_defaults(args, run_cfg)

    # Pick the right skill library default per env type
    if args.skill_library is None:
        if args.env_type == "libero":
            # Auto-detect privileged vs non-privileged from config
            _is_nonpriv = False
            if args.config:
                try:
                    import yaml as _yl
                    with open(args.config) as _cf:
                        _yc = _yl.safe_load(_cf) or {}
                    _apis = _yc.get("env", {}).get("cfg", {}).get("apis", [])
                    _is_nonpriv = any("Reduced" in a for a in _apis)
                except Exception:
                    pass
            if _is_nonpriv:
                args.skill_library = "skill_library/libero_nonpriv_skills.json"
            else:
                args.skill_library = "skill_library/libero_skills.json"
        elif args.env_type == "molmospaces":
            args.skill_library = "skill_library/molmospaces_nonpriv_skills.json"
        else:
            args.skill_library = "skill_library/skills.json"

    # Load defaults from config for any unset args
    _cfg_path = PROJECT_ROOT / "rats" / "config" / "default.yaml"
    if _cfg_path.exists():
        import yaml
        with _cfg_path.open() as _f:
            _cfg = yaml.safe_load(_f) or {}
        if args.timeout is None:
            args.timeout = int(_cfg.get("execution_timeout_seconds", 900))
        if args.policy_self_check_repairs is None:
            args.policy_self_check_repairs = int(
                _cfg.get("policy_self_check_max_repairs", 2)
            )
    else:
        if args.timeout is None:
            args.timeout = 900
        if args.policy_self_check_repairs is None:
            args.policy_self_check_repairs = 2

    if args.output_dir is None:
        if args.resume:
            resume_path = Path(args.resume).expanduser()
            args.output_dir = str(
                resume_path.parent
                if resume_path.is_file() or resume_path.suffix == ".json"
                else resume_path
            )
        else:
            args.output_dir = "outputs/rats_lifelong"
    if args.skip_completed and not args.resume:
        args.resume = args.output_dir

    # Create output directory first (needed for log file)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Opt-in per-agent LLM IO logging. When --log-agent-io is passed (or
    # RATS_AGENT_IO_DIR is already set in the environment) every query_llm
    # call writes a self-contained JSON record (with image bytes extracted to
    # sibling files) under <output_dir>/agent_io/ so the user can audit
    # exactly what context each agent saw. Bundle with
    # scripts/agent_io_viewer.py.
    if args.log_agent_io and os.environ.get("RATS_AGENT_IO_DIR") is None:
        agent_io_dir = Path(args.output_dir) / "agent_io"
        agent_io_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RATS_AGENT_IO_DIR"] = str(agent_io_dir)

    resume_same_dir = False
    if args.resume:
        try:
            resume_path = Path(args.resume).expanduser()
            resume_dir = (
                resume_path.parent
                if resume_path.is_file() or resume_path.suffix == ".json"
                else resume_path
            )
            resume_same_dir = (
                resume_dir.resolve()
                == Path(args.output_dir).expanduser().resolve()
            )
        except Exception:
            resume_same_dir = False
    log_mode = "a" if resume_same_dir else "w"

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.output_dir) / "rats.log", mode=log_mode),
        ],
    )
    logger = logging.getLogger("rats")
    if resume_same_dir:
        logger.info("%s", "-" * 60)
        logger.info("Resuming in-place; appending to existing rats.log")
    logger.info(f"Environment type: {args.env_type}")
    if args.model:
        logger.info("LLM model override: %s", args.model)
    if getattr(args, "web_ui", False):
        args.viser_debug = True
        logger.info(f"RATS Web UI requested on port {args.web_ui_port}; enabling Viser debug")
    if getattr(args, "collision_aware_motion", False):
        os.environ["MOLMOSPACES_COLLISION_AWARE_MOTION"] = "1"
        logger.info("Collision-aware motion enabled (IK + trajopt with scene obstacles)")

    # Candidate-based curiosity selection. LIBERO explore runs always score
    # K fresh + up-to-K_retry candidates with the novelty×frontier formula
    # and pick the argmax; every other run uses the single-proposal path.
    # (This was previously gated behind --curiosity-candidate-mode; formula
    # is now the only candidate scorer and is enabled automatically.)
    candidate_mode = (
        "formula"
        if (args.env_type == "libero" and getattr(args, "explore", False))
        else "none"
    )
    if candidate_mode == "formula":
        logger.info("Task proposer: candidate selection ON (formula scorer)")

    # The legacy persistent task_queue (--curiosity) is superseded by the
    # formula scorer above and can no longer win for LIBERO explore, so it
    # stays disabled. The deprecated flag is a harmless no-op.
    curiosity_enabled = False
    if getattr(args, "curiosity", False):
        logger.info(
            "--curiosity is deprecated and has no effect; LIBERO explore uses "
            "the formula candidate scorer."
        )

    # Create environment
    if args.mock:
        if args.env_type == "libero":
            logger.info("Using mock LIBERO environment")
            env = MockLiberoEnvironment(
                suite_name=args.libero_suite or "libero_spatial",
                task_id=args.libero_task or 0,
            )
        elif args.env_type == "molmospaces":
            logger.info("Using mock MolmoSpaces environment")
            env = create_molmospaces_environment(
                benchmark=args.molmospaces_benchmark,
                scene_family=args.molmospaces_scene_family,
                task_family=args.molmospaces_task_family,
                variant=args.molmospaces_variant,
                benchmark_dir=args.molmospaces_benchmark_dir,
                use_real_bridge=args.use_real_bridge,
                remote_bridge_url=args.remote_bridge_url,
                task_type=args.molmospaces_task_type,
                scene_dataset=args.molmospaces_scene_dataset,
                house_index=args.molmospaces_house_index,
            )
        else:
            logger.info("Using mock BEHAVIOR environment")
            env = MockEnvironment()
    elif args.config:
        logger.info(f"Loading environment from {args.config}")
        if args.env_type == "libero":
            # Forward explicit --libero-suite / --libero-task so parallel
            # workers on one config can target different tasks.
            env = create_real_environment(
                args.config,
                libero_suite=args.libero_suite,
                libero_task=args.libero_task,
                viser_debug=args.viser_debug,
            )
        elif args.env_type == "molmospaces":
            env = create_molmospaces_environment(config_path=args.config)
        else:
            env = create_real_environment(args.config, viser_debug=args.viser_debug)
    elif args.env_type == "libero":
        suite = args.libero_suite or "libero_spatial"
        task = args.libero_task if args.libero_task is not None else 0
        logger.info(f"Creating LIBERO env: {suite} task {task}")
        env = create_libero_environment(
            suite_name=suite,
            task_id=task,
            viser_debug=args.viser_debug,
        )
    elif args.env_type == "molmospaces":
        logger.info(
            "Creating MolmoSpaces env: %s/%s/%s (real_bridge=%s)",
            args.molmospaces_benchmark,
            args.molmospaces_scene_family,
            args.molmospaces_task_family,
            args.use_real_bridge,
        )
        env = create_molmospaces_environment(
            benchmark=args.molmospaces_benchmark,
            scene_family=args.molmospaces_scene_family,
            task_family=args.molmospaces_task_family,
            variant=args.molmospaces_variant,
            benchmark_dir=args.molmospaces_benchmark_dir,
            use_real_bridge=args.use_real_bridge,
            remote_bridge_url=args.remote_bridge_url,
            task_type=args.molmospaces_task_type,
            scene_dataset=args.molmospaces_scene_dataset,
            house_index=args.molmospaces_house_index,
        )
    else:
        logger.error("Must specify --config, --mock, or --env-type libero/molmospaces")
        sys.exit(1)

    # Discover available tasks — for BEHAVIOR, include scene-compatible tasks
    scene_model = getattr(args, "scene_model", None)
    if not scene_model and args.env_type == "behavior":
        low_level = getattr(env, "low_level_env", env)
        og_env = getattr(low_level, "env", None)
        if og_env and hasattr(og_env, "task"):
            scene_model = getattr(og_env.task, "scene_name", None)
        if not scene_model:
            # Try extracting from config
            if args.config:
                import yaml as _yaml
                try:
                    with open(args.config) as _f:
                        _cfg = _yaml.safe_load(_f)
                    ctrl_cfg_name = _cfg.get("env", {}).get("cfg", {}).get("low_level", {}).get("controller_cfg", "")
                    from rats.rats.catalog import load_r1pro_controller_metadata
                    _meta = load_r1pro_controller_metadata(ctrl_cfg_name)
                    scene_model = _meta.get("scene_model")
                except Exception:
                    pass
    # Resolve benchmark_dir: CLI arg takes priority, then extract from YAML config.
    benchmark_dir = getattr(args, "molmospaces_benchmark_dir", None)
    molmospaces_loop_config = None
    if not benchmark_dir and args.env_type == "molmospaces" and args.config:
        try:
            import yaml as _yaml
            with open(args.config) as _f:
                _cfg = _yaml.safe_load(_f) or {}
            benchmark_dir = (
                _cfg.get("env", {}).get("cfg", {}).get("low_level", {}).get("benchmark_dir")
            )
        except Exception:
            pass
    if args.env_type == "molmospaces" and args.config:
        try:
            import yaml as _yaml
            with open(args.config) as _f:
                _cfg = _yaml.safe_load(_f) or {}
            # Optional per-run overrides for rats/config/default.yaml::molmospaces.
            # This lets env_configs/molmospaces/*.yaml choose catalog/open
            # proposer mode, house-switching, and VLM grounding without
            # editing the repository-global default.
            loaded = _cfg.get("molmospaces", {}) or {}
            if isinstance(loaded, dict) and loaded:
                molmospaces_loop_config = loaded
        except Exception:
            pass

    # Resolve turns_per_attempt and attempts_per_iteration. Precedence is
    # CLI > top-level YAML key > default. The legacy YAML key
    # `turns_per_iteration` is still accepted (treated as turns_per_attempt)
    # so older configs keep working — log a deprecation note when matched.
    if args.config:
        try:
            import yaml as _yaml
            with open(args.config) as _f:
                _cfg = _yaml.safe_load(_f) or {}
            if args.turns_per_attempt is None:
                yaml_tpa = _cfg.get("turns_per_attempt")
                if yaml_tpa is None:
                    legacy = _cfg.get("turns_per_iteration")
                    if legacy is not None:
                        yaml_tpa = legacy
                        # Use logging here is fine; logger isn't set up
                        # yet so just print so the user notices.
                        print(
                            "[deprecation] yaml key 'turns_per_iteration' "
                            "is now 'turns_per_attempt'; please rename."
                        )
                if yaml_tpa is not None:
                    args.turns_per_attempt = int(yaml_tpa)
            if args.attempts_per_iteration is None:
                yaml_api = _cfg.get("attempts_per_iteration")
                if yaml_api is not None:
                    args.attempts_per_iteration = int(yaml_api)
            if getattr(args, "multi_turn_decision", None) is None:
                yaml_mtd = _cfg.get("multi_turn_decision")
                if yaml_mtd is not None:
                    args.multi_turn_decision = bool(yaml_mtd)
        except Exception:
            pass
    if args.turns_per_attempt is None:
        args.turns_per_attempt = 1
    if getattr(args, "multi_turn_decision", None) is None:
        args.multi_turn_decision = False
    available_tasks = discover_available_tasks(
        env_type=args.env_type,
        scene_model=scene_model,
        benchmark_dir=benchmark_dir,
    )
    logger.info(f"Discovered {len(available_tasks)} available tasks (scene={scene_model})")

    # Reset the registered initial state before the first fixed-task
    # iteration.  LifelongLoop applies ``1 + RATS_SEED_OFFSET`` when it
    # resets between iterations, but a one-iteration evaluation never reaches
    # that path before executing its policy.  Calling ``reset()`` without a
    # seed here therefore made every public seed use LIBERO init state 0.
    # FrankaLiberoEnv intentionally accepts the upstream 1-based trial seed,
    # so public seed k maps to internal seed k + 1 and init-state index k.
    if args.env_type == "libero":
        seed_offset = int(os.getenv("RATS_SEED_OFFSET", "0") or 0)
        env.reset(seed=trial_seed_from_public_seed(seed_offset))
    else:
        env.reset()

    # Start the RATS live debugger after the initial env exists. The debugger
    # reuses the existing CaP-X web-ui frontend/protocol and mirrors the RATS
    # loop into it; it does not launch a separate CaP-X trial.
    web_debugger = None
    if args.web_ui:
        from rats.loop.rats_web_debug import RatsWebDebugger

        web_debugger = RatsWebDebugger(
            output_dir=args.output_dir or "outputs/rats_lifelong",
            config_path=args.config,
            model=os.environ.get("RATS_LLM_MODEL") or args.model,
            env_type=args.env_type,
            port=args.web_ui_port,
        )
        web_debugger.start(env=env)
        logger.info(f"RATS Web UI: http://localhost:{args.web_ui_port}")

    # Run lifelong loop
    from rats.loop.lifelong_loop import LifelongLoop

    # Fixed-task vs exploration (Task Proposer + rebind)
    if args.explore:
        if not args.mock and not args.config and args.env_type != "libero":
            logger.error("--explore with real BEHAVIOR requires --config for initial environment bootstrap")
            sys.exit(1)
        use_fixed_task = False
        logger.info("Running in exploration mode (Task Proposer + task rebinding)")
    elif args.env_type in {"libero", "molmospaces"} and not args.explore:
        # LIBERO and MolmoSpaces default to fixed-task unless explore is requested.
        use_fixed_task = True
        logger.info("Running in fixed-task mode (%s default)", args.env_type)
    elif args.fixed_task or (args.config is not None and not args.mock):
        use_fixed_task = True
        logger.info("Running in fixed-task mode (same task each iteration)")
    else:
        use_fixed_task = False
        logger.info("Running in exploration mode (mock default: Task Proposer + task rebinding)")

    skill_merge_paths: list[str] = []
    if args.skill_library_merge:
        skill_merge_paths = [
            p.strip() for p in args.skill_library_merge.split(",") if p.strip()
        ]
    if args.env_type == "molmospaces" and args.include_molmospaces_seeded_skills:
        seeded_path = (
            args.molmospaces_seeded_skill_library
            or DEFAULT_MOLMOSPACES_SEEDED_SKILL_LIBRARY
        )
        if seeded_path not in skill_merge_paths:
            skill_merge_paths.append(seeded_path)
        logger.info("MolmoSpaces seeded helper skills enabled: %s", seeded_path)

    loop = LifelongLoop(
        env,
        skill_library_path=args.skill_library,
        skill_library_merge_paths=skill_merge_paths,
        skill_library_min_tier=args.skill_library_min_tier,
        playtime_memory_seed_path=args.playtime_memory_seed,
        max_retries_per_task=args.max_retries,
        execution_timeout=args.timeout,
        output_dir=args.output_dir,
        available_tasks=available_tasks,
        fixed_task=use_fixed_task,
        env_type=args.env_type,
        failure_memory_path=args.failure_memory_path,
        use_catalog=getattr(args, "catalog", False),
        curriculum=getattr(args, "curriculum", False),
        curiosity=curiosity_enabled,
        no_skill_reuse=getattr(args, "no_skill_reuse", False),
        random_order=getattr(args, "random_order", False),
        no_failure_memory=getattr(args, "no_failure_memory", False),
        ensemble_n=getattr(args, "ensemble", 0),
        # Play mode is default-on (3-4 year-old curious-child proposer
        # prompt). The legacy --play-prompt-mode flag still flips play
        # mode on for old launch scripts; both collapse onto the same
        # ``play_mode`` keyword now that the warm-up flip has been
        # retired and ``play_iterations`` is a no-op.
        play_mode=(
            bool(getattr(args, "play_mode", True))
            or bool(getattr(args, "play_prompt_mode", False))
        ),
        turns_per_attempt=int(getattr(args, "turns_per_attempt", 1) or 1),
        attempts_per_iteration=getattr(args, "attempts_per_iteration", None),
        multi_turn_decision=bool(getattr(args, "multi_turn_decision", False)),
        molmospaces_config=molmospaces_loop_config,
        proposer_include_eval_task_context=bool(
            getattr(args, "proposer_include_eval_task_context", False)
        ),
        policy_self_check_max_repairs=getattr(
            args, "policy_self_check_repairs", 2
        ),
        multiturn_reset_mode=bool(getattr(args, "multiturn_reset", False)),
        multiturn_reset_max_step_retries=int(
            getattr(args, "multiturn_reset_max_step_retries", 10) or 10
        ),
        curiosity_candidate_mode=candidate_mode,
        proposer_no_context=bool(getattr(args, "proposer_no_context", False)),
        proposer_temperature=getattr(args, "proposer_temperature", None),
        curiosity_warmup_iters=int(getattr(args, "curiosity_warmup_iters", 0) or 0),
        snapshot_interval=int(getattr(args, "snapshot_interval", 0) or 0),
        num_fresh_candidates=int(getattr(args, "num_fresh_candidates", 3) or 3),
        num_retry_candidates=int(getattr(args, "num_retry_candidates", 2) or 2),
        retry_bank_size=int(getattr(args, "retry_bank_size", 8) or 8),
        retry_bank_ttl=int(getattr(args, "retry_bank_ttl", 3) or 3),
        retry_bonus_weight=float(getattr(args, "retry_bonus_weight", 0.15) or 0.15),
        failure_penalty_weight=float(
            getattr(args, "failure_penalty_weight", 0.10) or 0.10
        ),
        score_composition=str(
            getattr(args, "score_composition", "product") or "product"
        ),
        retry_min_pred_success=float(
            getattr(args, "retry_min_pred_success", 0.5) or 0.5
        ),
        web_debugger=web_debugger,
    )

    # Resume from previous run if specified
    if args.resume:
        resumed = loop.resume_from(
            args.resume,
            skip_completed=bool(getattr(args, "skip_completed", False)),
        )
        if args.skip_completed:
            logger.info(
                "Loaded skip-completed resume state from %s (%d iteration files)",
                args.resume,
                resumed,
            )
        else:
            logger.info(f"Resumed {resumed} iterations from {args.resume}")

    try:
        summary = loop.run(num_iterations=args.iterations)
    except KeyboardInterrupt as e:
        if web_debugger:
            web_debugger.error(str(e) or "Interrupted")
        raise
    except Exception as e:
        if web_debugger:
            web_debugger.error(f"{type(e).__name__}: {e}")
        raise

    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total iterations: {summary['total_iterations']}")
    logger.info(f"Successful: {summary['successful_iterations']}")
    logger.info(f"Failed: {summary['failed_iterations']}")
    logger.info(f"Success rate: {summary.get('success_rate', 0):.1%}")
    logger.info(f"Final skill library: {summary['final_skill_library_size']} total, {summary['learned_skills']} learned")

    metrics = summary.get("metrics", {})
    if metrics.get("total"):
        logger.info(f"Metrics success rate: {metrics.get('single_task_success_rate', 0):.1%}")
    logger.info(f"Avg retries per success: {metrics.get('average_retries_per_success', 'N/A')}")

    # Generate plots
    try:
        from rats.evaluation.plotting import plot_metrics
        plot_metrics(summary, args.output_dir)
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    # Save summary
    summary_path = Path(args.output_dir) / "final_summary.json"
    with summary_path.open("w") as f:
        json.dump(json_safe(summary), f, indent=2, allow_nan=False)
    logger.info(f"Summary saved to {summary_path}")

    # Persist exact per-attempt policy code and the skill/API usage graph by
    # default for fixed-scene benchmark/playtime runs. The exporter only reads
    # JSON/timeline artifacts that already exist, so failures should never fail
    # the benchmark itself.
    try:
        from scripts.export_attempt_artifacts import export_attempt_artifacts

        artifact_summary = export_attempt_artifacts(Path(args.output_dir))
        logger.info(
            "Attempt code + skill/API usage artifacts written: %s",
            artifact_summary.get("skill_api_usage", {}).get("summary"),
        )
    except Exception as e:
        logger.warning(f"Could not export attempt artifacts: {e}")

    # Render a self-contained HTML report alongside the summary so the run is
    # browsable without scripting (per-iter task, plan, code, video, diagnosis).
    try:
        from scripts.render_run_html import render as _render_html
        report_path = Path(args.output_dir) / "report.html"
        report_path.write_text(_render_html(Path(args.output_dir)))
        logger.info(f"HTML report written to {report_path}")
    except Exception as e:
        logger.warning(f"Could not render HTML report: {e}")

    # Render a markdown variant (TOC + image thumbnails) for VSCode preview,
    # which doesn't play <video> tags from the HTML report.
    try:
        from scripts.render_run_md import render as _render_md
        md_path = Path(args.output_dir) / "report.md"
        md_path.write_text(_render_md(Path(args.output_dir)))
        logger.info(f"Markdown report written to {md_path}")
    except Exception as e:
        logger.warning(f"Could not render Markdown report: {e}")


if __name__ == "__main__":
    main()
