#!/usr/bin/env python3
"""Evolve and freeze a RATS skill library on the Robosuite dev pool.

The physical curriculum, workers, model route, and sealed test boundary match
RACaP-RS.  The development budget is compute-matched rather than
trajectory-matched: a registered pilot found a RATS sweep substantially slower
than a RACaP sweep, so RATS receives three complete candidate sweeps while
RACaP receives fifteen.  Actual wall time, trajectories, calls, tokens, and
cost are all retained for disclosure.  Official seeds and held-out task types
are never read by this program.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .run_libero import (
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _public_model_route,
        _require_frozen_rats_seal,
        _run_logged,
        _sha256,
        _shared_api_server_bundle,
        _tree_sha256,
        _write_json,
    )
    from .supervise_after_development import _seal_frozen_artifact
except ImportError:
    from run_libero import (  # type: ignore[no-redef]
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _public_model_route,
        _require_frozen_rats_seal,
        _run_logged,
        _sha256,
        _shared_api_server_bundle,
        _tree_sha256,
        _write_json,
    )
    from supervise_after_development import _seal_frozen_artifact  # type: ignore[no-redef]

try:
    from .run_robosuite import _rats_first_party_source_sha256
except ImportError:
    from run_robosuite import _rats_first_party_source_sha256  # type: ignore[no-redef]

from evolution.harness.robosuite import DEVELOPMENT_SEEDS, DEVELOPMENT_TASKS

QUOTA_MARKERS = (
    "insufficient_user_quota",
    "insufficient quota",
    "insufficient balance",
    "balance is insufficient",
    "need pre-deduct",
    "no available channel",
    "no available route",
)

REGISTERED_CANDIDATE_SWEEPS = 3


def _last_native_rows(sweep: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for task in DEVELOPMENT_TASKS:
        path = sweep / task / "native_states.jsonl"
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "native_state_after_code":
                continue
            key = str(row.get("episode_key") or "")
            if key:
                rows[key] = row
    return rows


def _sweep_metrics(sweep: Path) -> dict[str, Any]:
    expected = [
        f"robosuite/{task}/seed{seed}"
        for task in DEVELOPMENT_TASKS
        for seed in DEVELOPMENT_SEEDS
    ]
    native = _last_native_rows(sweep)
    episodes = []
    for key in expected:
        row = native.get(key)
        if row is None:
            raise RuntimeError(f"complete sweep lacks final native record: {key}")
        episodes.append(
            {
                "episode_key": key,
                "native_success": bool(row.get("native_success")),
                "reward": float(row.get("reward") or 0.0),
                "simulator_steps": int(row.get("simulator_steps") or 0),
            }
        )
    success = sum(int(row["native_success"]) for row in episodes)
    return {
        "expected": len(expected),
        "native_success": success,
        "native_rate": success / len(expected),
        "mean_simulator_steps": sum(row["simulator_steps"] for row in episodes)
        / len(episodes),
        "successes": [row["episode_key"] for row in episodes if row["native_success"]],
        "failures": [row["episode_key"] for row in episodes if not row["native_success"]],
        "episodes": episodes,
    }


def _make_writable(root: Path) -> None:
    for item in [root, *root.rglob("*")]:
        mode = item.stat().st_mode
        item.chmod(mode | (0o700 if item.is_dir() else 0o600))


def _archive_incomplete_candidate(root: Path, candidate: Path, iteration: int) -> Path | None:
    """Retain, but never reuse, an interrupted skill-extraction transaction."""

    if not candidate.exists() or (candidate / "seal_manifest.json").is_file():
        return None
    archive_root = root / "invalidated_extractions"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    destination = archive_root / f"candidate_{iteration:03d}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = archive_root / f"candidate_{iteration:03d}_{stamp}_{suffix:02d}"
        suffix += 1
    shutil.move(str(candidate), str(destination))
    _write_json(
        destination / "NOT_SCORED.json",
        {
            "schema_version": 1,
            "reason": "incomplete skill-extraction transaction",
            "iteration": iteration,
            "source_experiment": str(root.resolve()),
        },
    )
    return destination


def _run_sweep(
    args: argparse.Namespace,
    *,
    label: str,
    library: Path,
    sweeps_root: Path,
) -> int:
    library_sha256_before = _sha256(library)
    argv = [
        str(args.python),
        str(ROOT / "experiments" / "controlled_comparison" / "run_robosuite.py"),
        "--method",
        "rats_rs_evolved",
        "--method-label",
        label,
        "--tasks",
        *DEVELOPMENT_TASKS,
        "--seeds",
        *(str(seed) for seed in DEVELOPMENT_SEEDS),
        "--model",
        args.model,
        "--python",
        str(args.python),
        "--workers",
        str(args.workers),
        "--rats-root",
        str(args.rats_root),
        "--robosuite-root",
        str(args.robosuite_root),
        "--rats-library",
        str(library),
        "--output-root",
        str(sweeps_root),
        "--services-already-running",
    ]
    result = _run_logged(
        argv,
        cwd=ROOT,
        env=os.environ.copy(),
        log_path=sweeps_root / label / "parent.log",
    )
    library_sha256_after = _sha256(library)
    _write_json(
        sweeps_root / label / "frozen_library_audit.json",
        {
            "before_sha256": library_sha256_before,
            "after_sha256": library_sha256_after,
            "unchanged": library_sha256_before == library_sha256_after,
            "library": str(library.resolve()),
        },
    )
    if library_sha256_before != library_sha256_after:
        raise RuntimeError(f"RATS-RS sweep mutated its frozen library: {label}")
    if result["quota_failure"] or result["returncode"] == 75:
        return 75
    return int(result["returncode"])


def _extract_candidate(
    args: argparse.Namespace,
    *,
    iteration: int,
    champion_library: Path,
    source_sweep: Path,
    candidate: Path,
    log_path: Path,
) -> int:
    argv = [
        str(args.python),
        str(ROOT / "scripts" / "extract_robosuite_rats_skills.py"),
        "--rats-root",
        str(args.rats_root),
        "--source-library",
        str(champion_library),
        "--source-sweep",
        str(source_sweep),
        "--output-dir",
        str(candidate),
        "--tasks",
        *DEVELOPMENT_TASKS,
        "--seeds",
        *(str(seed) for seed in DEVELOPMENT_SEEDS),
        "--model",
        args.model,
        "--iteration",
        str(iteration),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(args.rats_root.resolve()), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = _run_logged(argv, cwd=args.rats_root, env=env, log_path=log_path)
    # Extraction is part of the development compute budget.  Persist its
    # process-level clock even though it consumes no simulator trajectory.
    _write_json(candidate / "extraction_process.json", result)
    if result["quota_failure"] or result["returncode"] == 75:
        return 75
    return int(result["returncode"])


def _freeze_final(root: Path, champion_artifact: Path, payload: dict[str, Any]) -> Path:
    frozen = root / "frozen_champion"
    if frozen.exists():
        return frozen
    shutil.copytree(champion_artifact, frozen, symlinks=False)
    _make_writable(frozen)
    # A copied seal describes the source path and must be regenerated.
    (frozen / "seal_manifest.json").unlink(missing_ok=True)
    _write_json(frozen / "champion_manifest.json", payload)
    _seal_frozen_artifact(frozen)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations", type=int, default=REGISTERED_CANDIDATE_SWEEPS
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Matched Robosuite development concurrency for both methods.",
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--rats-root", type=Path, default=DEFAULT_RATS_ROOT)
    parser.add_argument(
        "--robosuite-root",
        type=Path,
        default=DEFAULT_RATS_ROOT / "rats" / "third_party" / "robosuite",
    )
    parser.add_argument(
        "--start-library",
        type=Path,
        default=ROOT
        / "outputs"
        / "controlled_comparison"
        / "artifacts"
        / "rats90_frozen"
        / "skills.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "evolution" / "robosuite_rats_rs_time3",
    )
    args = parser.parse_args()
    if args.iterations != REGISTERED_CANDIDATE_SWEEPS:
        raise SystemExit(
            "the registered compute-matched comparison requires exactly "
            f"{REGISTERED_CANDIDATE_SWEEPS} RATS candidate iterations"
        )
    if not 1 <= args.workers <= 5:
        raise SystemExit("--workers must be between 1 and 5")
    if not os.environ.get("RACAP_VAPI_KEY") or not os.environ.get("RACAP_VAPI_BASE"):
        raise SystemExit("RACAP_VAPI_KEY and RACAP_VAPI_BASE must be configured")
    _require_frozen_rats_seal(args.start_library)

    root = args.output_dir.resolve()
    sweeps_root = root / "sweeps"
    candidates_root = root / "candidate_libraries"
    root.mkdir(parents=True, exist_ok=True)
    sweeps_root.mkdir(parents=True, exist_ok=True)
    candidates_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "method": "RATS-RS-evolved",
        "algorithm": "upstream RATS success-code skill distillation and library reuse",
        "start": str(args.start_library.resolve()),
        "start_sha256": _sha256(args.start_library),
        "iterations": args.iterations,
        "development_tasks": list(DEVELOPMENT_TASKS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "episodes_per_sweep": len(DEVELOPMENT_TASKS) * len(DEVELOPMENT_SEEDS),
        "evaluation_tasks_visible": False,
        "evaluation_seeds_visible": False,
        "model": args.model,
        "workers": int(args.workers),
        "hosted_model_route": _public_model_route(os.environ["RACAP_VAPI_BASE"]),
        "selection": "strictly greater paired native-success count; ties do not promote",
        "continuation": (
            "each candidate is distilled onto the current champion; source code comes "
            "from the preceding development sweep, and upstream validation rejects "
            "dependencies absent from the champion"
        ),
        "physical_budget": {
            "baseline_sweeps": 1,
            "candidate_sweeps": args.iterations,
            "total_development_episodes": (args.iterations + 1)
            * len(DEVELOPMENT_TASKS)
            * len(DEVELOPMENT_SEEDS),
        },
        "budget_basis": "approximately_equal_effective_development_wall_time",
        "budget_rationale": (
            "registered pre-run timing: one RATS 15-episode sweep is about "
            "37.6 minutes versus about 6 minutes for RACaP; three RATS "
            "candidate sweeps plus baseline target the RACaP development "
            "wall-time envelope without changing RATS runtime semantics"
        ),
        "resource_disclosure": [
            "effective_wall_seconds",
            "development_episodes",
            "model_calls",
            "prompt_tokens",
            "completion_tokens",
            "estimated_api_cost_usd",
            "simulator_steps",
        ],
        # Robosuite is independently attested by its clean Git tree in every
        # sweep. Hash only mutable RATS first-party Python here; recursively
        # rehashing ``rats/third_party/robosuite`` adds minutes without adding
        # provenance coverage.
        "rats_first_party_python_source_sha256": _rats_first_party_source_sha256(
            args.rats_root / "rats"
        ),
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit("refusing incompatible RATS-RS evolution resume")
    if not manifest_path.exists():
        _write_json(manifest_path, manifest)
    if (root / "COMPLETE.json").is_file():
        return 0

    state_path = root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else None
    service_args = Namespace(python=args.python, rats_root=args.rats_root)
    with _shared_api_server_bundle(
        service_args,
        root,
        owner="robosuite_rats_evolution_parent",
    ):
        baseline_label = "sweep_000_baseline"
        baseline = sweeps_root / baseline_label
        if not (baseline / "COMPLETE.json").is_file():
            rc = _run_sweep(
                args,
                label=baseline_label,
                library=args.start_library,
                sweeps_root=sweeps_root,
            )
            if rc == 75:
                _write_json(root / "ABORTED.json", {"phase": "baseline", "quota": True})
                return 75
            if rc:
                raise RuntimeError(f"RATS-RS baseline sweep failed with return code {rc}")
        baseline_metrics = _sweep_metrics(baseline)
        _write_json(baseline / "sweep_metrics.json", baseline_metrics)

        if state is None:
            state = {
                "schema_version": 1,
                "next_iteration": 1,
                "champion_library": str(args.start_library.resolve()),
                "champion_artifact": str(args.start_library.resolve().parent),
                "champion_metrics": baseline_metrics,
                "source_sweep": str(baseline.resolve()),
                "history": [],
                "updated_at_unix": time.time(),
            }
            _write_json(state_path, state)

        for iteration in range(int(state["next_iteration"]), args.iterations + 1):
            candidate = candidates_root / f"candidate_{iteration:03d}"
            extraction_log = root / "logs" / f"extract_{iteration:03d}.log"
            if not (candidate / "seal_manifest.json").is_file():
                _archive_incomplete_candidate(root, candidate, iteration)
                rc = _extract_candidate(
                    args,
                    iteration=iteration,
                    champion_library=Path(state["champion_library"]),
                    source_sweep=Path(state["source_sweep"]),
                    candidate=candidate,
                    log_path=extraction_log,
                )
                if rc == 75:
                    _write_json(
                        root / "ABORTED.json",
                        {"phase": "skill_extraction", "iteration": iteration, "quota": True},
                    )
                    return 75
                if rc:
                    raise RuntimeError(
                        f"RATS skill extraction failed at iteration {iteration}: rc={rc}"
                    )
                _seal_frozen_artifact(candidate)
            _require_frozen_rats_seal(candidate / "skills.json")

            label = f"sweep_{iteration:03d}_candidate"
            sweep = sweeps_root / label
            if not (sweep / "COMPLETE.json").is_file():
                rc = _run_sweep(
                    args,
                    label=label,
                    library=candidate / "skills.json",
                    sweeps_root=sweeps_root,
                )
                if rc == 75:
                    _write_json(
                        root / "ABORTED.json",
                        {"phase": "candidate_sweep", "iteration": iteration, "quota": True},
                    )
                    return 75
                if rc:
                    raise RuntimeError(
                        f"RATS candidate sweep failed at iteration {iteration}: rc={rc}"
                    )
            metrics = _sweep_metrics(sweep)
            _write_json(sweep / "sweep_metrics.json", metrics)
            extraction = json.loads((candidate / "extraction_log.json").read_text())
            promoted = metrics["native_success"] > state["champion_metrics"]["native_success"]
            decision = {
                "iteration": iteration,
                "candidate_artifact": str(candidate.resolve()),
                "candidate_library_sha256": _sha256(candidate / "skills.json"),
                "accepted_skills": extraction.get("accepted_skills", []),
                "candidate_metrics": metrics,
                "parent_champion_metrics": state["champion_metrics"],
                "promoted": promoted,
                "reason": (
                    f"native success improved {state['champion_metrics']['native_success']}"
                    f"->{metrics['native_success']}"
                    if promoted
                    else "no strict native-success improvement"
                ),
                "decided_at_unix": time.time(),
            }
            _write_json(root / "decisions" / f"iteration_{iteration:03d}.json", decision)
            state["history"].append(decision)
            if promoted:
                state["champion_library"] = str((candidate / "skills.json").resolve())
                state["champion_artifact"] = str(candidate.resolve())
                state["champion_metrics"] = metrics
            # The next skill proposal sees the newest physical evidence, but is
            # always applied to the retained champion. Unknown dependencies on
            # a rejected library fail the upstream AST gate.
            state["source_sweep"] = str(sweep.resolve())
            state["next_iteration"] = iteration + 1
            state["updated_at_unix"] = time.time()
            _write_json(state_path, state)

    final_payload = {
        "schema_version": 1,
        "method": "RATS-RS-evolved",
        "source_artifact": state["champion_artifact"],
        "source_library_sha256": _sha256(Path(state["champion_library"])),
        "development_metrics": state["champion_metrics"],
        "iterations": args.iterations,
        "development_tasks": list(DEVELOPMENT_TASKS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "sealed_evaluation_used_during_development": False,
    }
    frozen = _freeze_final(root, Path(state["champion_artifact"]), final_payload)
    audit = _require_frozen_rats_seal(frozen / "skills.json")
    complete = {
        **final_payload,
        "frozen_champion": str(frozen),
        "frozen_library_sha256": _sha256(frozen / "skills.json"),
        "seal_audit": audit,
        "completed_at_unix": time.time(),
    }
    _write_json(root / "COMPLETE.json", complete)
    (root / "ABORTED.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
