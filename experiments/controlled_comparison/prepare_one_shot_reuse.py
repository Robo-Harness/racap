#!/usr/bin/env python3
"""Materialize the pre-registered one-shot calibration reuse index.

The one labeled interaction per family is exactly the task-0/seed-0 episode
already present in the zero-shot grid.  This utility selects those rows from
normalized, audited episode tables and records source/index hashes.  It does
not execute a policy or alter any measured result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

METHODS = ["capx", "rats_base", "rats_90", "racap_phase1", "racap_phase2"]
SUITES = ["libero_spatial_swap", "libero_goal_swap", "libero_object_swap"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes-csv",
        type=Path,
        action="append",
        required=True,
        help="audited normalized episode table; repeat for disjoint method tables",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    missing_sources = [str(path) for path in args.episodes_csv if not path.is_file()]
    if missing_sources:
        raise SystemExit("missing episode tables: " + ", ".join(missing_sources))

    tables = [pd.read_csv(path) for path in args.episodes_csv]
    all_rows = pd.concat(tables, ignore_index=True)
    selected = all_rows[
        all_rows["method"].isin(METHODS)
        & (all_rows["cohort"] == "libero_pro_zero_shot")
        & all_rows["suite"].isin(SUITES)
        & (all_rows["task_id"] == 0)
        & (all_rows["seed"] == 0)
    ].copy()
    key = ["method", "suite", "task_id", "seed"]
    duplicated = selected.duplicated(key, keep=False)
    expected = {(method, suite, 0, 0) for method in METHODS for suite in SUITES}
    actual = {
        (str(row.method), str(row.suite), int(row.task_id), int(row.seed))
        for row in selected.itertuples()
    }
    if duplicated.any() or len(selected) != len(expected) or actual != expected:
        raise SystemExit(
            "invalid calibration reuse selection: "
            f"rows={len(selected)}, pairs={len(actual)}, "
            f"duplicates={int(duplicated.sum())}, expected={len(expected)}"
        )
    if not selected["evidence_path"].map(lambda value: Path(str(value)).is_file()).all():
        raise SystemExit("at least one selected calibration row lacks EVIDENCE.json")

    selected = selected.sort_values(key).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    manifest = {
        "schema_version": 1,
        "mode": "reuse_exact_registered_zero_shot_episode",
        "selection_was_registered_before_outcomes": True,
        "source_cohort": "libero_pro_zero_shot",
        "selector": {"suites": SUITES, "task_id": 0, "seed": 0},
        "resource_accounting": (
            "count original policy wall time, model calls/tokens/cost, and simulator steps"
        ),
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in args.episodes_csv
        ],
        "rows": len(selected),
        "methods": METHODS,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
