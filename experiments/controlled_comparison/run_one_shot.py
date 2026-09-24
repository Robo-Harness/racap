#!/usr/bin/env python3
"""Run the registered one-trial adaptation protocol one method at a time."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .run_libero import DEFAULT_PYTHON, _run_logged
    from .build_one_shot_cards import (
        FAMILY_SUITE,
        _calibration_rows,
        _calibration_rows_from_index,
        _episode_root,
        _select_video,
        _sha256,
    )
except ImportError:  # direct script execution
    from run_libero import DEFAULT_PYTHON, _run_logged  # type: ignore[no-redef]
    from build_one_shot_cards import (  # type: ignore[no-redef]
        FAMILY_SUITE,
        _calibration_rows,
        _calibration_rows_from_index,
        _episode_root,
        _select_video,
        _sha256,
    )

HERE = Path(__file__).resolve().parent
METHODS = ["capx", "rats_base", "rats_90", "racap_phase1", "racap_phase2"]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), existing) if value
    )
    return env


def _run(argv: list[str], log_path: Path) -> int:
    result = _run_logged(
        argv,
        cwd=ROOT,
        env=_child_env(),
        log_path=log_path,
    )
    return int(result["returncode"])


def _write_calibration_reuse_ledger(
    source_root: Path,
    destination: Path,
    source_index: Path | None = None,
) -> None:
    """Bind one-shot adaptation to the already measured registered interaction.

    Reusing the exact task-0/seed-0 zero-shot episode avoids changing the
    information available to the method.  The original trajectory resources
    remain part of the adaptation budget; this ledger only prevents executing
    that identical simulator interaction twice.
    """

    rows = (
        _calibration_rows_from_index(source_index, "libero_pro_zero_shot")
        if source_index is not None
        else _calibration_rows(source_root, "libero_pro_zero_shot")
    )
    expected = {
        (method, suite)
        for method in METHODS
        for suite in FAMILY_SUITE.values()
    }
    actual = {(str(row["method"]), str(row["suite"])) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise SystemExit(
            f"zero-shot calibration reuse grid mismatch: rows={len(rows)}, "
            f"pairs={len(actual)}, expected={len(expected)}"
        )
    selected: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (str(item["method"]), str(item["suite"]))):
        evidence = Path(str(row["evidence_path"]))
        if not evidence.is_file():
            raise SystemExit(f"missing reusable calibration evidence: {evidence}")
        episode_root = _episode_root(row)
        video = _select_video(episode_root)
        if video is None:
            raise SystemExit(f"missing reusable calibration video: {episode_root}")
        selected.append(
            {
                "method": str(row["method"]),
                "suite": str(row["suite"]),
                "task_id": int(row["task_id"]),
                "seed": int(row["seed"]),
                "episode_key": str(row["episode_key"]),
                "native_success_bit": bool(row["native_success"]),
                "evidence_path": str(evidence.resolve()),
                "evidence_sha256": _sha256(evidence),
                "episode_artifact": str(episode_root.resolve()),
                "video": str(video.resolve()),
                "video_sha256": _sha256(video),
                "original_policy_wall_seconds": float(
                    row.get("policy_wall_seconds") or 0.0
                ),
                "original_model_calls": int(row.get("model_calls") or 0),
                "original_estimated_api_cost_usd": float(
                    row.get("estimated_api_cost_usd") or 0.0
                ),
                "original_simulator_steps": int(row.get("simulator_steps") or 0),
            }
        )
    _write_json(
        destination,
        {
            "schema_version": 1,
            "mode": "reuse_exact_registered_zero_shot_episode",
            "source_root": str(source_root.resolve()),
            "source_index": (
                str(source_index.resolve()) if source_index is not None else None
            ),
            "source_index_sha256": (
                _sha256(source_index) if source_index is not None else None
            ),
            "source_cohort": "libero_pro_zero_shot",
            "selector": {
                "suites": sorted(FAMILY_SUITE.values()),
                "task_id": 0,
                "seed": 0,
            },
            "selection_was_registered_before_outcomes": True,
            "original_resources_count_toward_adaptation_budget": True,
            "simulator_episodes_saved": len(selected),
            "selected": selected,
        },
    )


def _method_command(
    *,
    python: Path,
    method: str,
    cohort: str,
    output_root: Path,
    frozen_root: Path,
    cards: Path | None,
    workers: int,
) -> list[str]:
    argv = [
        str(python),
        str(HERE / "run_libero.py"),
        "--method",
        method,
        "--cohorts",
        cohort,
        "--output-root",
        str(output_root),
        "--workers",
        str(workers),
    ]
    if method == "rats_90":
        argv.extend(
            [
                "--rats-library",
                str(frozen_root / "skills.json"),
                "--rats-memory",
                str(frozen_root / "failure_memory"),
            ]
        )
    if cards is not None:
        argv.extend(["--experience-card-dir", str(cards)])
    return argv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["calibration", "cards", "test", "all"],
        default="all",
    )
    parser.add_argument(
        "--calibration-source",
        choices=["reuse-zero-shot", "fresh"],
        default="reuse-zero-shot",
        help=(
            "reuse the exact pre-registered zero-shot task-0/seed-0 interaction "
            "or execute a duplicate fresh interaction"
        ),
    )
    parser.add_argument(
        "--zero-shot-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "measured",
    )
    parser.add_argument(
        "--calibration-index",
        type=Path,
        help="normalized 15-row zero-shot reuse index with measured resource fields",
    )
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=METHODS)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "one_shot",
    )
    parser.add_argument(
        "--rats-frozen",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "artifacts" / "rats90_frozen",
    )
    args = parser.parse_args()
    if args.workers != 10:
        raise SystemExit("the amended protocol requires exactly ten workers")
    if not os.environ.get("RACAP_VAPI_KEY") or not os.environ.get("RACAP_VAPI_BASE"):
        raise SystemExit("RACAP_VAPI_KEY and RACAP_VAPI_BASE must be configured")
    if "rats_90" in args.methods and not (args.rats_frozen / "skills.json").is_file():
        raise SystemExit(f"RATS frozen artifact is not ready: {args.rats_frozen}")

    started = time.time()
    results: list[dict[str, object]] = []
    phases = (
        ["calibration", "cards", "test"] if args.phase == "all" else [args.phase]
    )
    for phase in phases:
        if phase == "calibration" and args.calibration_source == "reuse-zero-shot":
            _write_calibration_reuse_ledger(
                args.zero_shot_root,
                args.root / "calibration_reuse.json",
                args.calibration_index,
            )
            results.append(
                {
                    "phase": phase,
                    "mode": "reuse-zero-shot",
                    "source_root": str(args.zero_shot_root.resolve()),
                    "returncode": 0,
                }
            )
            continue
        if phase == "cards":
            card_input_root = (
                args.zero_shot_root
                if args.calibration_source == "reuse-zero-shot"
                else args.root / "calibration"
            )
            argv = [
                str(args.python),
                str(HERE / "build_one_shot_cards.py"),
                "--input-root",
                str(card_input_root),
                "--input-cohort",
                (
                    "libero_pro_zero_shot"
                    if args.calibration_source == "reuse-zero-shot"
                    else "one_shot_calibration"
                ),
                "--output-dir",
                str(args.root / "cards"),
            ]
            if args.calibration_index is not None:
                argv.extend(["--input-index", str(args.calibration_index)])
            rc = _run(argv, args.root / "cards.log")
            results.append({"phase": phase, "returncode": rc})
            if rc != 0:
                break
            continue
        cohort = "one_shot_calibration" if phase == "calibration" else "libero_pro_one_shot"
        output_root = args.root / ("calibration" if phase == "calibration" else "measured")
        cards = None if phase == "calibration" else args.root / "cards"
        for method in args.methods:
            argv = _method_command(
                python=args.python,
                method=method,
                cohort=cohort,
                output_root=output_root,
                frozen_root=args.rats_frozen,
                cards=cards,
                workers=args.workers,
            )
            rc = _run(argv, args.root / f"{phase}_{method}.log")
            results.append({"phase": phase, "method": method, "returncode": rc})
            if rc != 0:
                break
        if results and int(results[-1]["returncode"]) != 0:
            break
    orchestration = {
        "schema_version": 1,
        "phase": args.phase,
        "methods": args.methods,
        "one_method_at_a_time": True,
        "calibration_source": args.calibration_source,
        "zero_shot_root": str(args.zero_shot_root.resolve()),
        "calibration_index": (
            str(args.calibration_index.resolve())
            if args.calibration_index is not None
            else None
        ),
        "wall_clock_seconds": round(time.time() - started, 3),
        "results": results,
    }
    _write_json(args.root / "orchestration.json", orchestration)
    method_slug = "-".join(args.methods) if args.methods else "none"
    _write_json(
        args.root / f"orchestration_{args.phase}_{method_slug}.json",
        orchestration,
    )
    if any(int(row["returncode"]) == 75 for row in results):
        return 75
    return 0 if results and all(int(row["returncode"]) == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
