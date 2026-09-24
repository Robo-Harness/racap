#!/usr/bin/env python3
"""Materialize the frozen LIBERO cohorts without running any controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _digest(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("task_manifest.json"),
    )
    args = parser.parse_args()

    from racap.benchmark.tasks import build_task_episodes
    from racap.envs.register_suites import register_missing_suites

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    register_missing_suites()
    rows: list[dict[str, object]] = []
    for cohort in protocol["evaluation"]["cohorts"]:
        # Robosuite uses a different evaluator and is materialized by its
        # adapter.  This manifest covers every native LIBERO episode exactly.
        if cohort["id"] == "robosuite_transfer":
            continue
        selected_task_ids = cohort.get("task_ids", "all")
        selected = (
            None
            if selected_task_ids == "all"
            else {int(value) for value in selected_task_ids}
        )
        for suite in cohort["suites"]:
            episodes = build_task_episodes(suite, seeds=tuple(cohort["seeds"]))
            for episode in episodes:
                if selected is not None and int(episode.task_id) not in selected:
                    continue
                row = episode.to_dict()
                row["cohort"] = cohort["id"]
                rows.append(row)

    payload = {
        "schema_version": 2,
        "protocol_id": protocol["protocol_id"],
        "episodes": len(rows),
        "sha256": _digest(rows),
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("episodes", "sha256")}))


if __name__ == "__main__":
    main()
