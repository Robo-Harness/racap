from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class R1ProTaskCatalogEntry:
    scene_model: str
    activity_name: str
    activity_definition_id: int
    controller_cfg: str
    env_config_path: str
    candidate_instance_ids: list[int]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    return data or {}


def resolve_omnigibson_controller_cfg(controller_cfg: str, project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    candidate = root / "rats" / "third_party" / "b1k" / "OmniGibson" / "omnigibson" / "configs" / controller_cfg
    return candidate


def load_r1pro_controller_metadata(controller_cfg: str, project_root: str | Path | None = None) -> dict[str, Any]:
    path = resolve_omnigibson_controller_cfg(controller_cfg, project_root=project_root)
    cfg = _load_yaml(path)
    return {
        "scene_model": cfg.get("scene", {}).get("scene_model"),
        "activity_name": cfg.get("task", {}).get("activity_name"),
        "activity_definition_id": cfg.get("task", {}).get("activity_definition_id", 0),
        "controller_cfg": controller_cfg,
        "controller_cfg_path": str(path),
    }


def discover_r1pro_task_catalog(
    scene_model: str | None = None,
    *,
    env_config_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> list[R1ProTaskCatalogEntry]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    config_dir = Path(env_config_dir) if env_config_dir is not None else root / "env_configs" / "r1pro"
    entries: dict[tuple[str, str, int], R1ProTaskCatalogEntry] = {}

    for path in sorted(config_dir.glob("*.yaml")):
        cfg = _load_yaml(path)
        low_level = cfg.get("env", {}).get("cfg", {}).get("low_level", {})
        if not isinstance(low_level, dict):
            continue
        controller_cfg = low_level.get("controller_cfg")
        if low_level.get("_target_") != "rats.envs.simulators.r1pro_b1k.R1ProBehaviourLowLevel" or controller_cfg is None:
            continue
        # Tolerate missing OmniGibson submodule: the controller-cfg file lives
        # under rats/third_party/b1k, which may not be initialized in dev
        # environments that only run RATS on LIBERO/MolmoSpaces. Skip entries
        # whose controller cfg can't be loaded instead of failing the whole
        # catalog scan (which other discovery paths depend on).
        if not resolve_omnigibson_controller_cfg(controller_cfg, project_root=root).exists():
            continue
        metadata = load_r1pro_controller_metadata(controller_cfg, project_root=root)
        candidate_scene = metadata.get("scene_model")
        if scene_model is not None and candidate_scene != scene_model:
            continue
        key = (candidate_scene or "", metadata.get("activity_name") or "", int(metadata.get("activity_definition_id", 0)))
        if key not in entries:
            entries[key] = R1ProTaskCatalogEntry(
                scene_model=candidate_scene or "unknown_scene",
                activity_name=metadata.get("activity_name") or path.stem,
                activity_definition_id=int(metadata.get("activity_definition_id", 0)),
                controller_cfg=controller_cfg,
                env_config_path=str(path),
                candidate_instance_ids=[0],
            )
    return list(entries.values())
