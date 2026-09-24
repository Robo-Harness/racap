#!/usr/bin/env python3
"""Resume-safe orchestrator for the frozen controlled comparison.

This driver never runs two methods concurrently. Generated-code LIBERO methods
use the amended ten-worker cap; audited RACaP main results are reused and are
never silently rerun. One-shot calibration, custom long-horizon, and Robosuite
phases begin only after the previous method has exited successfully. A
provider/quota pause (exit 75) stops the schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Support the documented ``python experiments/.../run_all.py`` entry point
# before importing the repo-local ``experiments`` package below.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from experiments.controlled_comparison.supervise_after_development import (
    _verify_frozen_artifact_seal,
)

ROOT = _BOOTSTRAP_ROOT
HERE = Path(__file__).resolve().parent
METHODS = ["capx", "rats_base", "rats_90", "racap_phase1", "racap_phase2"]
# RACaP's in-domain rows are immutable, audited reuse artifacts in the amended
# protocol.  Only generated-code baselines execute the 350-row main runner.
# Missing RACaP reuse data must fail during consolidation/reporting rather than
# silently triggering an expensive LIBERO-90 rerun.
MAIN_EXECUTION_METHODS = ["capx", "rats_base", "rats_90"]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _resume_fingerprint() -> dict[str, str]:
    """Bind a resumable phase to the exact source and registered protocol."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "protocol_sha256": _file_sha256(HERE / "protocol.yaml"),
        "task_manifest_sha256": _file_sha256(HERE / "task_manifest.json"),
    }


def _orchestration_env() -> dict[str, str]:
    """Return an environment in which repo-local phase scripts are importable.

    The documented entry point invokes this file by path.  In that mode Python
    places ``experiments/controlled_comparison`` rather than the repository
    root on ``sys.path``.  Every child phase imports the ``experiments``
    package, so make the root explicit while preserving any caller-supplied
    paths.  Method runners still replace policy-child PYTHONPATH with their
    attested import roots before a simulator reset.
    """

    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    entries = [str(ROOT)]
    if current:
        entries.append(current)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _run(
    name: str,
    argv: list[str],
    *,
    log_dir: Path,
    history: list[dict[str, Any]],
) -> int:
    started = time.time()
    log = log_dir / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n# launch: " + " ".join(argv) + "\n")
        handle.flush()
        returncode = subprocess.run(
            argv,
            cwd=ROOT,
            env=_orchestration_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    history.append(
        {
            "phase": name,
            "argv": argv,
            "returncode": returncode,
            "wall_seconds": round(time.time() - started, 3),
            "log": str(log.resolve()),
            "resume_fingerprint": _resume_fingerprint(),
        }
    )
    return returncode


def _phase_completed(
    history: list[dict[str, Any]],
    *,
    name: str,
    argv: list[str],
) -> bool:
    """Return true only for an identical phase command that last exited zero.

    A quota/interruption record must never cause a skip: re-entering that phase
    lets its episode-level runner resume from the evidence already on disk.
    Comparing argv prevents stale orchestration state from silently skipping a
    phase after its frozen artifact, interpreter, or command has changed.
    """

    # Preflight attests the current host, imports, model assets, and free
    # comparison ports. It is cheap and must be recaptured on every launch.
    if name == "preflight":
        return False
    matches = [row for row in history if str(row.get("phase")) == name]
    if not matches:
        return False
    latest = matches[-1]
    return (
        int(latest.get("returncode", 1)) == 0
        and latest.get("argv") == argv
        and latest.get("resume_fingerprint") == _resume_fingerprint()
    )


def _immutable_method_result(
    method: str,
    *,
    output_root: Path | None = None,
) -> dict[str, Any] | None:
    """Return provenance for a fully measured and audited method result.

    Raw rollout artifacts remain bound to the source tree recorded in their
    own run manifest. A later infrastructure fix for another method changes
    the repository fingerprint, but must not force a completed method back
    through its rollout runner (which correctly rejects the mixed-source
    resume). Reuse is allowed only when all registered episode COMPLETE files,
    the method-level COMPLETE marker, and a passing cardinality audit agree.
    Current analysis and audit phases still rerun after this rollout skip.
    """

    if method not in METHODS:
        return None
    base = output_root or (ROOT / "outputs" / "controlled_comparison")
    method_root = base / "measured" / method
    manifest_path = method_root / "run_manifest.json"
    complete_path = method_root / "COMPLETE.json"
    audit_path = base / "analysis" / "method_audits" / method / "method_audit.json"
    if not all(path.is_file() for path in (manifest_path, complete_path, audit_path)):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(complete_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    jobs = manifest.get("jobs")
    if manifest.get("method") != method or not isinstance(jobs, list) or not jobs:
        return None
    expected = len(jobs)
    episode_complete = [
        path
        for path in method_root.rglob("COMPLETE.json")
        if path.resolve() != complete_path.resolve()
    ]
    if (
        completion.get("method") != method
        or int(completion.get("jobs", -1)) != expected
        or len(episode_complete) != expected
        or audit.get("method") != method
        or audit.get("status") != "pass"
        or int(audit.get("episodes_parsed", -1)) != expected
        or int(audit.get("episodes_expected", -1)) != expected
    ):
        return None
    return {
        "method": method,
        "episodes": expected,
        "run_manifest": str(manifest_path.resolve()),
        "run_manifest_sha256": _file_sha256(manifest_path),
        "method_complete": str(complete_path.resolve()),
        "method_complete_sha256": _file_sha256(complete_path),
        "prior_audit": str(audit_path.resolve()),
        "prior_audit_sha256": _file_sha256(audit_path),
    }


def _rats_frozen_args(root: Path, *, include_memory: bool) -> list[str]:
    """Build the explicitly registered frozen RATS artifact arguments."""

    argv = ["--rats-library", str(root / "skills.json")]
    if include_memory:
        argv.extend(["--rats-memory", str(root / "failure_memory")])
    return argv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-python",
        type=Path,
        default=Path(os.environ.get("RACAP_EXPERIMENT_PYTHON", sys.executable)),
        help="Python environment containing the vendored RATS/LIBERO extras.",
    )
    parser.add_argument(
        "--rats-frozen",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "artifacts" / "rats90_frozen",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "orchestration.json",
    )
    args = parser.parse_args()
    if not (args.rats_frozen / "skills.json").is_file():
        raise SystemExit(f"frozen RATS artifact is not ready: {args.rats_frozen}")
    seal = _verify_frozen_artifact_seal(args.rats_frozen)
    if seal["status"] != "pass":
        raise SystemExit(f"frozen RATS artifact seal failed: {seal['error']}")
    if not os.environ.get("RACAP_VAPI_KEY") or not os.environ.get("RACAP_VAPI_BASE"):
        raise SystemExit("RACAP_VAPI_KEY and RACAP_VAPI_BASE must be configured")

    history: list[dict[str, Any]] = []
    if args.state.is_file():
        try:
            history = json.loads(args.state.read_text(encoding="utf-8")).get("history", [])
        except (OSError, json.JSONDecodeError):
            history = []
    log_dir = args.state.parent / "orchestration_logs"

    commands: list[tuple[str, list[str]]] = []
    commands.append(
        (
            "development_analysis",
            [sys.executable, str(HERE / "analyze_development.py")],
        )
    )
    commands.append(
        (
            "preflight",
            [
                sys.executable,
                str(HERE / "audit_environment.py"),
                "--output",
                str(args.state.parent / "provenance" / "preflight.json"),
                "--rats-library",
                str(args.rats_frozen),
            ],
        )
    )
    for method in MAIN_EXECUTION_METHODS:
        argv = [
            sys.executable,
            str(HERE / "run_libero.py"),
            "--method",
            method,
            "--python",
            str(args.experiment_python),
            "--workers",
            "10",
        ]
        if method == "rats_90":
            argv.extend(_rats_frozen_args(args.rats_frozen, include_memory=True))
        commands.append((f"main_{method}", argv))
        method_analysis = args.state.parent / "analysis" / "method_audits" / method
        commands.append(
            (
                f"main_analysis_{method}",
                [
                    sys.executable,
                    str(HERE / "analyze_results.py"),
                    "--methods",
                    method,
                    "--output-dir",
                    str(method_analysis),
                ],
            )
        )
        commands.append(
            (
                f"main_audit_{method}",
                [
                    sys.executable,
                    str(HERE / "audit_method_results.py"),
                    "--method",
                    method,
                    "--analysis-dir",
                    str(method_analysis),
                ],
            )
        )
    generated_main = args.state.parent / "analysis" / "generated_main_current"
    racap_pro = args.state.parent / "analysis" / "racap_pro_current"
    consolidated_main = (
        args.state.parent / "analysis" / "consolidated_main"
    )
    commands.append(
        (
            "main_generated_analysis",
            [
                sys.executable,
                str(HERE / "analyze_results.py"),
                "--methods",
                *MAIN_EXECUTION_METHODS,
                "--output-dir",
                str(generated_main),
            ],
        )
    )
    commands.append(
        (
            "main_racap_pro_reuse_analysis",
            [
                sys.executable,
                str(HERE / "analyze_results.py"),
                "--methods",
                "racap_phase1",
                "racap_phase2",
                "--cohorts",
                "libero_pro_zero_shot",
                "--output-dir",
                str(racap_pro),
            ],
        )
    )
    commands.append(
        (
            "main_consolidation",
            [
                sys.executable,
                str(HERE / "consolidate_main_results.py"),
                "--generated-main-dir",
                str(generated_main),
                "--phase1-main-dir",
                str(args.state.parent / "analysis" / "method_audits" / "racap_phase1"),
                "--racap-pro-dir",
                str(racap_pro),
                "--output-root",
                str(consolidated_main),
            ],
        )
    )
    commands.append(
        (
            "one_trial",
            [
                sys.executable,
                str(HERE / "run_one_shot.py"),
                "--phase",
                "all",
                "--python",
                str(args.experiment_python),
                "--rats-frozen",
                str(args.rats_frozen),
                "--workers",
                "10",
            ],
        )
    )
    commands.append(
        (
            "one_trial_analysis",
            [sys.executable, str(HERE / "analyze_one_shot.py")],
        )
    )
    for method in METHODS:
        argv = [
            sys.executable,
            str(HERE / "run_long_horizon.py"),
            "--method",
            method,
            "--python",
            str(args.experiment_python),
            "--workers",
            "10",
        ]
        if method == "rats_90":
            argv.extend(_rats_frozen_args(args.rats_frozen, include_memory=True))
        commands.append((f"long_{method}", argv))
    commands.append(
        (
            "long_analysis",
            [sys.executable, str(HERE / "analyze_long_horizon.py")],
        )
    )
    for method in ("capx", "rats_90"):
        argv = [
            sys.executable,
            str(HERE / "run_robosuite.py"),
            "--method",
            method,
            "--python",
            str(args.experiment_python),
            "--workers",
            "10",
        ]
        if method == "rats_90":
            argv.extend(_rats_frozen_args(args.rats_frozen, include_memory=False))
        commands.append((f"robosuite_{method}", argv))
    commands.append(
        (
            "robosuite_analysis",
            [sys.executable, str(HERE / "analyze_robosuite.py")],
        )
    )
    commands.append(
        (
            "final_report",
            [sys.executable, str(HERE / "assemble_report.py")],
        )
    )

    for name, argv in commands:
        method = name.removeprefix("main_") if name.startswith("main_") else ""
        immutable_result = _immutable_method_result(method)
        if immutable_result is not None:
            history.append(
                {
                    "phase": name,
                    "argv": argv,
                    "returncode": 0,
                    "wall_seconds": 0.0,
                    "log": None,
                    "resume_fingerprint": _resume_fingerprint(),
                    "action": "reused_immutable_measured_method",
                    "immutable_result": immutable_result,
                }
            )
            _write_json(
                args.state,
                {
                    "schema_version": 1,
                    "one_method_at_a_time": True,
                    "workers_per_method": 10,
                    "history": history,
                    "last_phase": name,
                    "last_returncode": 0,
                    "last_action": "reused_immutable_measured_method",
                    "updated_at_unix": time.time(),
                },
            )
            continue
        if _phase_completed(history, name=name, argv=argv):
            _write_json(
                args.state,
                {
                    "schema_version": 1,
                    "one_method_at_a_time": True,
                    "workers_per_method": 10,
                    "history": history,
                    "last_phase": name,
                    "last_returncode": 0,
                    "last_action": "skipped_identical_completed_phase",
                    "updated_at_unix": time.time(),
                },
            )
            continue
        returncode = _run(name, argv, log_dir=log_dir, history=history)
        _write_json(
            args.state,
            {
                "schema_version": 1,
                "one_method_at_a_time": True,
                "workers_per_method": 10,
                "history": history,
                "last_phase": name,
                "last_returncode": returncode,
                "last_action": "executed_phase",
                "updated_at_unix": time.time(),
            },
        )
        if returncode != 0:
            return 75 if returncode == 75 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
