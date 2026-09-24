#!/usr/bin/env bash
set -euo pipefail

# Run inside the `mlspaces` conda environment.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$REPO_ROOT/rats-cache/molmospaces}"
case "$MLSPACES_CACHE_DIR" in
  /*) ;;
  *) MLSPACES_CACHE_DIR="$REPO_ROOT/$MLSPACES_CACHE_DIR" ;;
esac
export MLSPACES_CACHE_DIR
export MLSPACES_ASSETS_DIR="$REPO_ROOT/rats/third_party/molmospaces/assets"
export RATS_MOLMOSPACES_BENCHMARK_DIR="${RATS_MOLMOSPACES_BENCHMARK_DIR:-$REPO_ROOT/rats/benchmarks/molmospaces/capx_rats_play_disjoint_from_eval40}"

if ! python -c "import molmo_spaces" >/dev/null 2>&1; then
  echo "ERROR: 'molmo_spaces' is not importable with the current python." >&2
  echo "Run this script inside the 'mlspaces' conda env, e.g.:" >&2
  echo "  conda activate mlspaces && bash scripts/bootstrap_molmospaces_assets.sh" >&2
  exit 1
fi

python - <<'PY'
import logging
import json
import os
from pathlib import Path

# resource_manager_log_level is defined inside molmo_spaces_constants
# itself, NOT in the molmospaces_resources package — importing it from
# the latter raises ImportError. The function just bumps the resource-
# manager logger; if it ever moves, drop the call (nice-to-have only).
from molmo_spaces.molmo_spaces_constants import (
    get_resource_manager,
    get_scenes,
    resource_manager_log_level,
)
from molmo_spaces.utils.lazy_loading_utils import (
    install_scene_with_objects_and_grasps_from_path,
)


def _load_benchmark_episodes(benchmark_dir: Path) -> list[dict]:
    benchmark_file = benchmark_dir / "benchmark.json"
    with benchmark_file.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("episodes", [])


def _benchmark_scene_paths(benchmark_dir: Path) -> list[str]:
    paths = []
    seen = set()
    for episode in _load_benchmark_episodes(benchmark_dir):
        dataset = episode["scene_dataset"]
        split = episode["data_split"]
        house_index = int(episode["house_index"])
        split_map = get_scenes(dataset, split)[split]
        scene_entry = split_map.get(house_index)
        if isinstance(scene_entry, dict):
            scene_path = scene_entry.get("ceiling") or scene_entry.get("base")
        else:
            scene_path = scene_entry
        if scene_path is None:
            raise RuntimeError(
                f"No scene path for {dataset}/{split} house_index={house_index}"
            )
        scene_path = str(scene_path)
        if scene_path not in seen:
            seen.add(scene_path)
            paths.append(scene_path)
    return paths


resource_manager_log_level(logging.INFO)

mgr = get_resource_manager()  # builds the manager without forcing post_setup

# Small + essential. Each call returns when its source is fully installed.
print("== robots ==")
mgr.install_all_for_source("robots", "franka_droid")
mgr.install_all_for_source("robots", "franka_cap")

print("== grasps ==")
mgr.install_all_for_source("grasps", "droid")

print("== objects.objathor_metadata ==")
mgr.install_all_for_source("objects", "objathor_metadata")
mgr.install_all_for_source('objects', 'thor')

print("== scenes ==")
mgr.install_all_for_source("scenes", "refs")          # tiny, required
mgr.install_all_for_source("scenes", "ithor")         # ~minutes

benchmark_dir = Path(os.environ["RATS_MOLMOSPACES_BENCHMARK_DIR"])
if benchmark_dir.exists():
    scene_paths = _benchmark_scene_paths(benchmark_dir)
    print(f"== benchmark scenes ({len(scene_paths)} unique) ==")
    for idx, scene_path in enumerate(scene_paths, start=1):
        print(f"[{idx}/{len(scene_paths)}] {scene_path}")
        install_scene_with_objects_and_grasps_from_path(scene_path)
else:
    print(f"WARNING: benchmark dir not found; skipping scene prefetch: {benchmark_dir}")

print("DONE")
PY
