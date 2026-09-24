#!/usr/bin/env python3
"""Consolidate audited main results without inventing missing coverage.

Two tables are intentionally produced:

* main_four_method: CaP-X, RATS-base, RATS-90, and RACaP-Phase 1 on the
  complete 350-episode registered grid.  The RACaP-Phase 1 row deliberately
  reuses its audited archived LIBERO-90 episodes rather than rerunning them.
* pro_five_method: all five frozen methods, including RACaP-Phase 2, on
  the current matched 180-episode LIBERO-PRO grid.

The champion has no current base-diagnostic or official-LIBERO-Long rollout.
Keeping these tables separate prevents a report generator from silently
imputing or relabeling those missing evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

try:
    from .analyze_results import (
        _aggregate,
        _format_excel_sheet,
        _paired_statistics,
        _plots,
    )
except ImportError:
    from analyze_results import (  # type: ignore
        _aggregate,
        _format_excel_sheet,
        _paired_statistics,
        _plots,
    )


FOUR_METHODS = ["capx", "rats_base", "rats_90", "racap_phase1"]
FIVE_METHODS = [*FOUR_METHODS, "racap_phase2"]
PRO_COHORT = "libero_pro_zero_shot"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boolean_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin({"true", "false", "1", "0"})
    if bool(unknown.any()):
        examples = sorted(set(normalized[unknown]))[:5]
        raise SystemExit(f"unparseable boolean evidence values: {examples}")
    return normalized.isin({"true", "1"})


def _read_pair(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes_path = directory / "episodes.csv"
    completeness_path = directory / "completeness.csv"
    missing = [
        str(path)
        for path in (episodes_path, completeness_path)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit("missing audited source tables:\n" + "\n".join(missing))
    return pd.read_csv(episodes_path), pd.read_csv(completeness_path)


def _validate(
    episodes: pd.DataFrame,
    completeness: pd.DataFrame,
    *,
    methods: list[str],
    episodes_per_method: int,
    name: str,
) -> None:
    expected = {method: episodes_per_method for method in methods}
    counts = episodes.groupby("method")["episode_key"].nunique().to_dict()
    if counts != expected:
        raise SystemExit(f"{name} episode coverage mismatch: {counts} != {expected}")
    duplicate = episodes.duplicated(["method", "episode_key"], keep=False)
    if bool(duplicate.any()):
        rows = episodes.loc[duplicate, ["method", "episode_key"]].to_dict("records")
        raise SystemExit(f"{name} duplicate episode evidence: {rows[:10]}")
    complete_counts = (
        completeness.assign(
            completed_bool=_boolean_series(completeness["completed"])
        )
        .groupby("method")
        .agg(rows=("episode_key", "size"), complete=("completed_bool", "sum"))
    )
    for method in methods:
        if method not in complete_counts.index:
            raise SystemExit(f"{name} completeness is missing {method}")
        row = complete_counts.loc[method]
        if (
            int(row["rows"]) != episodes_per_method
            or int(row["complete"]) != episodes_per_method
        ):
            raise SystemExit(
                f"{name}/{method} incomplete: "
                f"{int(row['complete'])}/{int(row['rows'])}, "
                f"expected {episodes_per_method}"
            )
    if set(episodes["method"]) != set(methods):
        raise SystemExit(f"{name} contains an unregistered method")
    if not bool(_boolean_series(episodes["instruction_match"]).all()):
        raise SystemExit(f"{name} contains a delivered-instruction mismatch")
    if not bool((episodes["simulator_episode_resets"] == 1).all()):
        raise SystemExit(f"{name} contains a reset-count violation")
    actual_models = {
        model.strip()
        for value in episodes["actual_models"].dropna().astype(str)
        for model in value.split(",")
        if model.strip()
    }
    if actual_models != {"gpt-5.5"}:
        raise SystemExit(f"{name} model identity mismatch: {actual_models}")


def _statistics(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = pd.concat(
        [_aggregate(episodes), _aggregate(episodes.assign(cohort="ALL"))],
        ignore_index=True,
    )
    pairwise = _paired_statistics(episodes)
    return aggregate, pairwise


def _coverage_frame() -> pd.DataFrame:
    rows = []
    for method in FIVE_METHODS:
        rows.extend(
            [
                {
                    "method": method,
                    "table": "main_four_method",
                    "episodes_expected": 350 if method in FOUR_METHODS else 0,
                    "coverage": (
                        "not run; never imputed"
                        if method == "racap_phase2"
                        else (
                            "complete audited mixed-source grid; archived "
                            "LIBERO-90 reused, current transfer/diagnostics"
                            if method == "racap_phase1"
                            else "complete current controlled grid"
                        )
                    ),
                },
                {
                    "method": method,
                    "table": "pro_five_method",
                    "episodes_expected": 180,
                    "coverage": "complete current matched PRO grid",
                },
            ]
        )
    return pd.DataFrame(rows)


def _write_result(
    directory: Path,
    *,
    episodes: pd.DataFrame,
    aggregate: pd.DataFrame,
    pairwise: pd.DataFrame,
    completeness: pd.DataFrame,
    coverage: pd.DataFrame,
    source_index: pd.DataFrame,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "Episodes": episodes,
        "Aggregate": aggregate,
        "Pairwise": pairwise,
        "Completeness": completeness,
        "Coverage Contract": coverage,
        "Source Index": source_index,
    }
    for name, frame in tables.items():
        frame.to_csv(directory / f"{name.lower().replace(' ', '_')}.csv", index=False)
    workbook = directory / "controlled_comparison.xlsx"
    with pd.ExcelWriter(workbook, engine="xlsxwriter") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            _format_excel_sheet(writer, name, frame)
    _plots(aggregate, episodes, directory / "figures")
    summary = {
        "schema_version": 1,
        "generated_at_unix": time.time(),
        "status": "pass",
        "episodes": len(episodes),
        "methods": sorted(episodes["method"].unique()),
        "cohorts": sorted(episodes["cohort"].unique()),
        "native_successes": int(episodes["native_success"].sum()),
        "workbook": str(workbook.resolve()),
        "workbook_sha256": _sha256(workbook),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-main-dir", type=Path, required=True)
    parser.add_argument("--phase1-main-dir", type=Path, required=True)
    parser.add_argument("--racap-pro-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    generated_episodes, generated_complete = _read_pair(args.generated_main_dir)
    phase1_episodes, phase1_complete = _read_pair(args.phase1_main_dir)
    racap_pro_episodes, racap_pro_complete = _read_pair(args.racap_pro_dir)

    main_episodes = pd.concat(
        [
            generated_episodes,
            phase1_episodes[phase1_episodes["method"] == "racap_phase1"],
        ],
        ignore_index=True,
    )
    main_complete = pd.concat(
        [
            generated_complete,
            phase1_complete[phase1_complete["method"] == "racap_phase1"],
        ],
        ignore_index=True,
    )
    _validate(
        main_episodes,
        main_complete,
        methods=FOUR_METHODS,
        episodes_per_method=350,
        name="main_four_method",
    )

    pro_episodes = pd.concat(
        [
            generated_episodes[generated_episodes["cohort"] == PRO_COHORT],
            racap_pro_episodes[racap_pro_episodes["cohort"] == PRO_COHORT],
        ],
        ignore_index=True,
    )
    pro_complete = pd.concat(
        [
            generated_complete[generated_complete["cohort"] == PRO_COHORT],
            racap_pro_complete[racap_pro_complete["cohort"] == PRO_COHORT],
        ],
        ignore_index=True,
    )
    _validate(
        pro_episodes,
        pro_complete,
        methods=FIVE_METHODS,
        episodes_per_method=180,
        name="pro_five_method",
    )

    sources = [
        args.generated_main_dir / "episodes.csv",
        args.generated_main_dir / "completeness.csv",
        args.phase1_main_dir / "episodes.csv",
        args.phase1_main_dir / "completeness.csv",
        args.racap_pro_dir / "episodes.csv",
        args.racap_pro_dir / "completeness.csv",
    ]
    source_index = pd.DataFrame(
        [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sources
        ]
    )
    coverage = _coverage_frame()
    main_aggregate, main_pairwise = _statistics(main_episodes)
    pro_aggregate, pro_pairwise = _statistics(pro_episodes)
    _write_result(
        args.output_root / "main_four_method",
        episodes=main_episodes,
        aggregate=main_aggregate,
        pairwise=main_pairwise,
        completeness=main_complete,
        coverage=coverage[coverage["table"] == "main_four_method"],
        source_index=source_index,
    )
    _write_result(
        args.output_root / "pro_five_method",
        episodes=pro_episodes,
        aggregate=pro_aggregate,
        pairwise=pro_pairwise,
        completeness=pro_complete,
        coverage=coverage[coverage["table"] == "pro_five_method"],
        source_index=source_index,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "main_four_method": len(main_episodes),
                "pro_five_method": len(pro_episodes),
                "output_root": str(args.output_root.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
