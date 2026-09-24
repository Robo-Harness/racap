"""MolmoSpaces-specific utilities for the RATS lifelong loop."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from rats.envs.configs.instantiate import instantiate
from rats.envs.simulators.molmospaces import (
    default_molmospaces_catalog,
    load_benchmark_catalog,
    parse_molmospaces_task_id,
)

logger = logging.getLogger("rats.molmospaces_utils")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MOLMOSPACES_EVAL_BENCHMARK_DIR = (
    _PROJECT_ROOT / "rats" / "benchmarks" / "molmospaces" / "capx_rats_eval_core_40"
)
_MOLMOSPACES_EVAL_TASK_CACHE: dict[str, list[dict[str, Any]]] = {}


def detect_molmospaces_env(env: Any) -> bool:
    low_level = getattr(env, "low_level_env", env)
    cls_name = type(low_level).__name__.lower()
    return "molmospaces" in cls_name or hasattr(low_level, "list_task_descriptors")


def detect_molmospaces_env_type_from_config(config_path: str) -> bool:
    return "molmospaces" in config_path.lower()


def discover_molmospaces_tasks(
    *,
    from_catalog: bool = True,
    root: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    # If a benchmark directory is provided, load real episodes from it.
    if benchmark_dir is not None:
        catalog = load_benchmark_catalog(benchmark_dir)
        for descriptor in catalog:
            tasks.append(
                {
                    "activity_name": descriptor["canonical_id"],
                    "canonical_task_id": descriptor["canonical_id"],
                    "scene_model": descriptor["scene_family"],
                    "scene_family": descriptor["scene_family"],
                    "benchmark": descriptor["benchmark"],
                    "task_family": descriptor["task_family"],
                    "variant": descriptor["variant"],
                    "language": descriptor.get("language", ""),
                    "objects": list(descriptor.get("objects") or []),
                    "metadata": dict(descriptor.get("metadata") or {}),
                    "env_config_path": "",
                }
            )
        if tasks:
            return tasks

    root_path = Path(root) if root is not None else None
    config_dir = root_path / "env_configs" / "molmospaces" if root_path is not None else None
    if config_dir is not None and config_dir.exists():
        try:
            import yaml
        except ImportError:
            yaml = None  # type: ignore[assignment]
        if yaml is not None:
            for path in sorted(config_dir.glob("*.yaml")):
                try:
                    with path.open() as f:
                        cfg = yaml.safe_load(f) or {}
                    low_level = cfg.get("env", {}).get("cfg", {}).get("low_level", {})
                    prompt = cfg.get("env", {}).get("cfg", {}).get("prompt", "")
                    benchmark = str(low_level.get("benchmark", "phase1"))
                    scene_family = str(low_level.get("scene_family", "unknown_scene"))
                    task_family = str(low_level.get("task_family", path.stem))
                    variant = str(low_level.get("variant", "default"))
                    language = (
                        low_level.get("language")
                        or prompt
                        or task_family.replace("_", " ")
                    )
                    canonical_id = (
                        f"molmospaces:{benchmark}:{scene_family}:{task_family}:{variant}"
                    )
                    tasks.append(
                        {
                            "activity_name": canonical_id,
                            "canonical_task_id": canonical_id,
                            "scene_model": scene_family,
                            "scene_family": scene_family,
                            "benchmark": benchmark,
                            "task_family": task_family,
                            "variant": variant,
                            "language": str(language),
                            "env_config_path": str(path),
                        }
                    )
                except Exception:
                    continue
            if tasks:
                return tasks

    if from_catalog:
        for descriptor in default_molmospaces_catalog():
            tasks.append(
                {
                    "activity_name": descriptor["canonical_id"],
                    "canonical_task_id": descriptor["canonical_id"],
                    "scene_model": descriptor["scene_family"],
                    "scene_family": descriptor["scene_family"],
                    "benchmark": descriptor["benchmark"],
                    "task_family": descriptor["task_family"],
                    "variant": descriptor["variant"],
                    "language": descriptor.get("language", ""),
                    "env_config_path": "",
                }
            )
    return tasks


def _resolve_molmospaces_benchmark_dir(
    benchmark_dir: str | Path | None = None,
) -> Path:
    if benchmark_dir is None:
        return DEFAULT_MOLMOSPACES_EVAL_BENCHMARK_DIR
    path = Path(benchmark_dir).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _molmospaces_referral_name(
    language: Any,
    *keys: str,
) -> str:
    if not isinstance(language, dict):
        return ""
    referrals = language.get("referral_expressions")
    if not isinstance(referrals, dict):
        return ""
    for key in keys:
        value = referrals.get(key)
        if value:
            return str(value)
    return ""


def _molmospaces_source_label(source: dict[str, Any]) -> str:
    for key in (
        "playtime_core100_source_benchmark",
        "all_combined_core200_selection",
        "playtime_core100_selection",
    ):
        value = source.get(key)
        if value:
            return str(value)
    return "benchmark_json"


def _summarize_molmospaces_eval_episode(
    episode: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    task = dict(episode.get("task") or {})
    source = dict(episode.get("source") or {})
    language = episode.get("language") or {}
    task_type = str(task.get("task_type") or "unknown")
    task_description = ""
    if isinstance(language, dict):
        task_description = str(language.get("task_description") or "")
    if not task_description:
        task_description = task_type.replace("_", " ")

    pickup_display = _molmospaces_referral_name(
        language,
        "pickup_obj_name",
        "pickup_name",
    )
    place_display = _molmospaces_referral_name(language, "place_name")
    pickup_internal = str(task.get("pickup_obj_name") or "")
    place_internal = str(task.get("place_receptacle_name") or "")
    joint_name = str(task.get("joint_name") or "")

    target_lines: list[str] = []
    if pickup_display:
        target_lines.append(f"target={pickup_display}")
    if place_display:
        target_lines.append(f"place={place_display}")
    internal_parts = []
    if pickup_internal:
        internal_parts.append(f"pickup_id={pickup_internal}")
    if place_internal:
        internal_parts.append(f"place_id={place_internal}")
    if joint_name:
        internal_parts.append(f"joint={joint_name}")

    if task_type in {"open", "close"}:
        threshold = task.get("task_success_threshold")
        goal_summary = (
            f"{task_type} the articulated target"
            + (f" toward threshold {threshold}" if threshold is not None else "")
        )
    elif task_type == "pick_and_place":
        goal_summary = "pick the target object and place it in/on the target receptacle"
    elif task_type == "pick":
        goal_summary = "pick up and lift the target object"
    else:
        goal_summary = task_description

    canonical_id = (
        f"molmospaces:{episode.get('scene_dataset', 'unknown')}:"
        f"house_{episode.get('house_index', 'unknown')}:{task_type}:ep{index}"
    )
    return {
        "index": index,
        "canonical_id": canonical_id,
        "task_type": task_type,
        "language": task_description,
        "scene_dataset": episode.get("scene_dataset"),
        "house_index": episode.get("house_index"),
        "data_split": episode.get("data_split"),
        "source": _molmospaces_source_label(source),
        "targets": "; ".join(target_lines),
        "internals": "; ".join(internal_parts),
        "goal": goal_summary,
    }


def get_molmospaces_eval_tasks(
    benchmark_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load MolmoSpaces benchmark JSON tasks for task-aware play prompts."""
    resolved_dir = _resolve_molmospaces_benchmark_dir(benchmark_dir)
    cache_key = str(resolved_dir)
    cached = _MOLMOSPACES_EVAL_TASK_CACHE.get(cache_key)
    if cached is not None:
        return [dict(task) for task in cached]

    benchmark_path = resolved_dir / "benchmark.json"
    with benchmark_path.open() as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        episodes = raw.get("episodes") or raw.get("tasks") or []
    else:
        episodes = raw
    if not isinstance(episodes, list):
        raise ValueError(f"Unexpected benchmark.json shape in {benchmark_path}")

    tasks = [
        _summarize_molmospaces_eval_episode(ep, idx)
        for idx, ep in enumerate(episodes)
        if isinstance(ep, dict)
    ]
    _MOLMOSPACES_EVAL_TASK_CACHE[cache_key] = tasks
    return [dict(task) for task in tasks]


def format_molmospaces_eval_tasks_block(
    benchmark_dir: str | Path | None = None,
) -> str:
    """Render downstream MolmoSpaces eval tasks as proposer prompt context."""
    tasks = get_molmospaces_eval_tasks(benchmark_dir)
    if not tasks:
        return ""

    by_type: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_type.setdefault(str(task.get("task_type") or "unknown"), []).append(task)

    benchmark_path = _resolve_molmospaces_benchmark_dir(benchmark_dir)
    lines = [
        "# EVALUATION TASKS - MolmoSpaces Core40 context",
        (
            "These are the downstream MolmoSpaces benchmark tasks this play run "
            "is meant to help with. Propose safe exploratory play that practices "
            "the same objects, affordances, grasps, placements, and articulated "
            "interactions when similar targets are visible."
        ),
        f"Benchmark JSON: {benchmark_path / 'benchmark.json'}",
        (
            "This is guidance, not a command to copy a task verbatim; keep obeying "
            "the live inventory, visibility, reachability, no-push, and safety rules."
        ),
    ]
    for task_type in sorted(by_type):
        group = by_type[task_type]
        lines.append("")
        lines.append(f"## {task_type} ({len(group)} tasks)")
        for task in group:
            scene = (
                f"{task.get('scene_dataset')} house_{task.get('house_index')} "
                f"split={task.get('data_split')} source={task.get('source')}"
            )
            lines.append(
                f"- ep{task.get('index'):02d} {task.get('canonical_id')}: "
                f"{task.get('language')}"
            )
            if task.get("targets"):
                lines.append(f"  targets: {task.get('targets')}")
            lines.append(f"  scene: {scene}")
            lines.append(f"  goal: {task.get('goal')}")
            if task.get("internals"):
                lines.append(f"  benchmark ids: {task.get('internals')}")
    return "\n".join(lines)


def extract_molmospaces_scene_context(env: Any) -> dict[str, Any]:
    low_level = getattr(env, "low_level_env", env)
    descriptor = low_level.get_task_descriptor()
    apis = getattr(env, "_apis", {})
    api_docs_parts: list[str] = []
    available_functions: list[str] = []
    for api in apis.values():
        if hasattr(api, "combined_doc") and callable(api.combined_doc):
            api_docs_parts.append(api.combined_doc())
        if hasattr(api, "functions") and callable(api.functions):
            available_functions.extend(list(api.functions().keys()))

    object_scope = {name: name for name in descriptor.get("objects", [])}
    task_prompt = descriptor.get("language", "") or getattr(env, "_task_prompt", "")
    task_family = str(descriptor.get("task_family") or "")
    metadata = descriptor.get("metadata") or {}
    task_kind = _infer_molmospaces_task_kind(task_family, metadata, task_prompt)
    return {
        "env_type": "molmospaces",
        "scene_model": descriptor.get("scene_family", "molmospaces_scene"),
        "activity_name": descriptor["canonical_id"],
        "object_scope": object_scope,
        "goal_conditions_nl": descriptor.get("language", ""),
        "task_prompt": task_prompt,
        "available_functions": sorted(set(available_functions)),
        "api_docs": "\n\n".join(api_docs_parts),
        "benchmark": descriptor.get("benchmark"),
        "task_family": task_family,
        "variant": descriptor.get("variant"),
        "task_descriptor": descriptor,
        "molmospaces_task_kind": task_kind,
    }


def _infer_molmospaces_task_kind(
    task_family: str, metadata: dict[str, Any], task_prompt: str,
) -> str:
    """Classify a MolmoSpaces task into a recipe bucket.

    Returns one of: ``pick``, ``pick_and_place``, ``open``, ``close``,
    ``nav``, ``other``. Drives policy_writer's recipe dispatch.
    """
    # Prefer the explicit task_type metadata (set on non-benchmark configs
    # by FrankaMolmoSpacesEnv._build_active_nonbenchmark_descriptor).
    task_type = str(metadata.get("task_type") or "").lower()
    family = task_family.lower()
    prompt = task_prompt.lower()

    if task_type in {"open", "close"}:
        return task_type
    if task_type == "playtime" or family.startswith("playtime"):
        return "playtime"
    if family.startswith("open") or "open" in prompt.split():
        return "open"
    if family.startswith("close") or "close" in prompt.split():
        return "close"
    if family == "pick":
        return "pick"
    if family.startswith("pick_and_place") or task_type == "pick_and_place":
        return "pick_and_place"
    if family == "nav":
        return "nav"
    return "other"


def recreate_molmospaces_env(old_env: Any, canonical_task_id: str) -> Any:
    cfg_obj = getattr(old_env, "cfg", None)
    api_names = list(getattr(old_env, "_apis", {}).keys()) or ["FrankaMolmoSpacesApi"]
    privileged = getattr(cfg_obj, "privileged", False) if cfg_obj else False
    prompt = getattr(old_env, "_task_prompt_template", None)
    if prompt is None:
        prompt = getattr(old_env, "_task_prompt", None)

    old_ll_env = getattr(old_env, "low_level_env", old_env)
    use_real_bridge = getattr(old_ll_env, "use_real_bridge", False)

    low_level_cfg: dict[str, Any] = {
        "_target_": "rats.envs.simulators.molmospaces.FrankaMolmoSpacesEnv",
        "canonical_task_id": canonical_task_id,
        "use_real_bridge": use_real_bridge,
    }
    old_ll_cfg = getattr(cfg_obj, "low_level", None) if cfg_obj else None
    if old_ll_cfg is not None:
        for attr in (
            "benchmark", "scene_family", "task_family", "variant",
            "max_steps", "enable_render", "catalog_path", "benchmark_dir",
            "task_type", "scene_dataset", "data_split", "house_index", "seed",
            "remote_bridge_url", "candidate_house_indices", "capx_only",
        ):
            val = getattr(old_ll_cfg, attr, None)
            if val is not None:
                low_level_cfg[attr] = val
    if "benchmark_dir" not in low_level_cfg:
        bd = getattr(old_ll_env, "benchmark_dir", None)
        if bd is not None:
            low_level_cfg["benchmark_dir"] = bd
    if "remote_bridge_url" not in low_level_cfg:
        remote_bridge_url = getattr(old_ll_env, "remote_bridge_url", None)
        if remote_bridge_url:
            low_level_cfg["remote_bridge_url"] = remote_bridge_url

    env_cfg = {
        "_target_": "rats.envs.tasks.franka.franka_molmospaces_env.FrankaMolmoSpacesCodeEnv",
        "cfg": {
            "_target_": "rats.envs.tasks.base.CodeExecEnvConfig",
            "low_level": low_level_cfg,
            "privileged": privileged,
            "apis": api_names,
        },
    }
    if prompt is not None:
        env_cfg["cfg"]["prompt"] = prompt

    original_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        new_env = instantiate(env_cfg)
    finally:
        sys.argv = original_argv

    if getattr(old_ll_env, "remote_bridge_url", None):
        new_ll_env = getattr(new_env, "low_level_env", new_env)
        if getattr(new_ll_env, "_real_bridge", None) is None:
            raise RuntimeError("Recreated MolmoSpaces env lost remote bridge mode")

    new_env.reset()
    logger.info("Recreated MolmoSpaces env: %s", canonical_task_id)
    return new_env


def extract_molmospaces_scene_inventory(env: Any) -> dict[str, Any]:
    """Snapshot of the active house's pickables / receptacles / articulations.

    Thin wrapper over ``FrankaMolmoSpacesEnv.describe_scene_inventory`` /
    ``RemoteMolmoSpacesBridge.describe_scene_inventory`` so the rats
    novel-task proposer (`_propose_novel_molmospaces_open`) doesn't have
    to know which bridge backs the env.

    Returns the same dict the bridge produces:
      {house_index, scene_dataset, rooms, pickables, receptacles, articulations}

    Returns an empty payload (no exception) if the env doesn't expose
    the inventory call — keeps fallback paths simple.
    """
    low_level = getattr(env, "low_level_env", env)
    fn = getattr(low_level, "describe_scene_inventory", None)
    if not callable(fn):
        return {
            "house_index": None,
            "scene_dataset": "unknown",
            "rooms": [],
            "pickables": [],
            "receptacles": [],
            "articulations": [],
        }
    try:
        inventory = dict(fn() or {})
    except Exception as exc:
        logger.warning("describe_scene_inventory failed: %s", exc)
        return {
            "house_index": None,
            "scene_dataset": "unknown",
            "rooms": [],
            "pickables": [],
            "receptacles": [],
            "articulations": [],
        }

    # Pull the bridge's currently-pinned task identity and attach it as
    # ``inventory["anchored_target"]`` so the playtime proposer can prefer
    # the anchored (reach-guaranteed) item over the full house inventory.
    # See get_anchored_task_target docstring for the rationale. Best-effort:
    # if the bridge doesn't expose it (older capx or mock bridge), leave
    # the key absent — downstream callers should treat its absence as
    # "no anchor known".
    anchor_fn = getattr(low_level, "get_anchored_task_target", None)
    if callable(anchor_fn):
        try:
            anchored = anchor_fn() or {}
        except Exception as exc:
            logger.debug("get_anchored_task_target failed: %s", exc)
            anchored = {}
        if anchored:
            inventory["anchored_target"] = anchored
    return inventory


def apply_molmospaces_task_spec(
    env: Any,
    *,
    task_type: str,
    target_internal_name: str | None = None,
    place_receptacle_internal_name: str | None = None,
    joint_internal_name: str | None = None,
    joint_index: int | None = None,
) -> dict[str, Any]:
    """Forward an open-mode task spec to the bridge.

    Wraps ``FrankaMolmoSpacesEnv.set_task_from_spec`` so the lifelong
    loop's rebind path stays bridge-agnostic. Re-raises any bridge
    error — callers handle them in ``_rebind_molmospaces_env``.
    """
    low_level = getattr(env, "low_level_env", env)
    fn = getattr(low_level, "set_task_from_spec", None)
    if not callable(fn):
        raise RuntimeError(
            "Active env does not expose set_task_from_spec; "
            "open-mode proposer requires a real or remote MolmoSpaces bridge."
        )
    return fn(
        task_type=task_type,
        target_internal_name=target_internal_name,
        place_receptacle_internal_name=place_receptacle_internal_name,
        joint_internal_name=joint_internal_name,
        joint_index=joint_index,
    )


def request_molmospaces_new_house(
    env: Any,
    house_index: int | None = None,
    *,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Ask the bridge to switch to a different house. See bridge docstring.

    ``task_type`` is used by smoke/open-proposer fallback: if the proposer
    requested an ``open``/``close`` house switch from a bridge currently
    initialized for another sampler, the env should switch sampler family
    before sampling the new house.
    """
    low_level = getattr(env, "low_level_env", env)
    fn = getattr(low_level, "request_new_house", None)
    if not callable(fn):
        raise RuntimeError(
            "Active env does not expose request_new_house; "
            "house-switching requires a real or remote MolmoSpaces bridge."
        )
    kwargs = {"house_index": house_index}
    if task_type is not None:
        kwargs["task_type"] = task_type
    return fn(**kwargs)


__all__ = [
    "detect_molmospaces_env",
    "detect_molmospaces_env_type_from_config",
    "discover_molmospaces_tasks",
    "extract_molmospaces_scene_context",
    "extract_molmospaces_scene_inventory",
    "apply_molmospaces_task_spec",
    "request_molmospaces_new_house",
    "recreate_molmospaces_env",
    "parse_molmospaces_task_id",
]
