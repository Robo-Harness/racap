#!/usr/bin/env python3
"""Resume the registered Robosuite campaign in one fail-closed sequence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .run_libero import DEFAULT_PYTHON, DEFAULT_RATS_ROOT, _run_logged, _write_json
except ImportError:
    from run_libero import (  # type: ignore[no-redef]
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _run_logged,
        _write_json,
    )

try:
    from .run_robosuite_racap import ROBOSUITE_CARTESIAN_CONTRACT_ID
except ImportError:
    from run_robosuite_racap import ROBOSUITE_CARTESIAN_CONTRACT_ID  # type: ignore[no-redef]


def _json_status(path: Path, expected: str = "pass") -> bool:
    if not path.is_file():
        return False
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status")) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _complete_file(path: Path) -> bool:
    return path.is_file()


def _corrected_racap_complete(root: Path) -> bool:
    """Accept only evidence produced under the audited live-EEF contract."""

    complete = root / "COMPLETE.json"
    manifest_path = root / "run_manifest.json"
    if not complete.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = json.loads(complete.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    contract = manifest.get("robosuite_cartesian_contract") or {}
    return (
        contract.get("id") == ROBOSUITE_CARTESIAN_CONTRACT_ID
        and int(completed.get("episodes") or 0) == 35
    )


def _archive_invalid_racap_zero_shot(transfer_root: Path) -> Path | None:
    """Move pre-audit zero-shot evidence outside the scored method tree."""

    source = transfer_root / "racap_phase2_zero_shot"
    if not source.exists() or not any(source.iterdir()) or _corrected_racap_complete(source):
        return None
    invalid_root = transfer_root / "invalid_attempts"
    invalid_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    destination = invalid_root / f"racap_zero_shot_pre_tcp_fix_{stamp}"
    suffix = 1
    while destination.exists():
        destination = invalid_root / f"racap_zero_shot_pre_tcp_fix_{stamp}_{suffix:02d}"
        suffix += 1
    shutil.move(str(source), str(destination))
    _write_json(
        destination / "INVALID_ATTEMPT.json",
        {
            "schema_version": 1,
            "status": "invalid_non_scoring",
            "reason": (
                "pre-audit Robosuite Cartesian contract duplicated the local TCP "
                "translation and exposed a stale cached EEF pose"
            ),
            "required_contract": ROBOSUITE_CARTESIAN_CONTRACT_ID,
            "calibration_pre_contact_pose_error_cm": 29.7,
            "calibration_post_contact_pose_error_cm": 0.6,
            "archived_from": str(source.resolve()),
        },
    )
    return destination


def _current_final_report(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "pass"
        and int(payload.get("robosuite_transfer_registered_episodes") or 0) == 175
        and int(payload.get("robosuite_development_episodes") or 0) == 300
    )


def _phase_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    python = str(args.python)
    common_executor = [
        "--python",
        str(args.executor_python),
        "--rats-root",
        str(args.rats_root),
        "--robosuite-root",
        str(args.robosuite_root),
    ]
    racap_final_root = args.transfer_root / "racap_rs_evolved"
    racap_zero_root = args.transfer_root / "racap_phase2_zero_shot"
    racap_zero_command = [
        python,
        str(ROOT / "experiments" / "controlled_comparison" / "run_robosuite_racap.py"),
        "--solution-root",
        str(ROOT / "policies" / "phase2"),
        "--output-root",
        str(args.transfer_root),
        "--method-label",
        "racap_phase2_zero_shot",
        "--protocol-note",
        "frozen LIBERO-90 champion; corrected live-EEF Cartesian contract; no Robosuite rollout feedback",
        "--workers",
        "5",
        *common_executor,
    ]
    if racap_zero_root.exists() and any(racap_zero_root.iterdir()):
        racap_zero_command.append("--resume")
    racap_final_command = [
        python,
        str(ROOT / "experiments" / "controlled_comparison" / "run_robosuite_racap.py"),
        "--solution-root",
        str(args.racap_evolution_root / "frozen_champion"),
        "--output-root",
        str(args.transfer_root),
        "--method-label",
        "racap_rs_evolved",
        "--protocol-note",
        "frozen after 15 candidate sweeps under the compute-matched Robosuite development protocol",
        "--workers",
        "5",
        *common_executor,
    ]
    if racap_final_root.exists() and any(racap_final_root.iterdir()):
        racap_final_command.append("--resume")

    return [
        {
            "name": "racap_rs_development",
            "command": [
                python,
                str(
                    ROOT
                    / "experiments"
                    / "controlled_comparison"
                    / "run_robosuite_evolution.py"
                ),
                "--experiment-root",
                str(args.racap_evolution_root),
                "--workers",
                "3",
                *common_executor,
            ],
            "complete": lambda: _complete_file(
                args.racap_evolution_root / "COMPLETE.json"
            ),
        },
        {
            "name": "rats_rs_development",
            "command": [
                python,
                str(
                    ROOT
                    / "experiments"
                    / "controlled_comparison"
                    / "run_robosuite_rats_evolution.py"
                ),
                "--output-dir",
                str(args.rats_evolution_root),
                "--workers",
                "3",
                "--iterations",
                "3",
                *common_executor,
            ],
            "complete": lambda: _complete_file(
                args.rats_evolution_root / "COMPLETE.json"
            ),
        },
        {
            "name": "development_analysis",
            "command": [
                python,
                str(
                    ROOT
                    / "experiments"
                    / "controlled_comparison"
                    / "analyze_robosuite_development.py"
                ),
                "--racap-root",
                str(args.racap_evolution_root),
                "--rats-root",
                str(args.rats_evolution_root),
                "--output-dir",
                str(args.development_analysis_root),
            ],
            "complete": lambda: _json_status(
                args.development_analysis_root / "audit.json"
            ),
        },
        {
            "name": "racap_zero_shot_corrected_evaluation",
            "command": racap_zero_command,
            "complete": lambda: _corrected_racap_complete(racap_zero_root),
        },
        {
            "name": "racap_rs_sealed_evaluation",
            "command": racap_final_command,
            "complete": lambda: _complete_file(racap_final_root / "COMPLETE.json"),
        },
        {
            "name": "rats_rs_sealed_evaluation",
            "command": [
                python,
                str(ROOT / "experiments" / "controlled_comparison" / "run_robosuite.py"),
                "--method",
                "rats_rs_evolved",
                "--method-label",
                "rats_rs_evolved",
                "--rats-library",
                str(args.rats_evolution_root / "frozen_champion" / "skills.json"),
                "--output-root",
                str(args.transfer_root),
                "--workers",
                "5",
                *common_executor,
            ],
            "complete": lambda: _complete_file(
                args.transfer_root / "rats_rs_evolved" / "COMPLETE.json"
            ),
        },
        {
            "name": "transfer_analysis",
            "command": [
                python,
                str(
                    ROOT
                    / "experiments"
                    / "controlled_comparison"
                    / "analyze_robosuite_transfer.py"
                ),
                "--input-root",
                str(args.transfer_root),
                "--output-dir",
                str(args.transfer_analysis_root),
            ],
            "complete": lambda: _json_status(
                args.transfer_analysis_root / "audit.json"
            ),
        },
        {
            "name": "final_workbook",
            "command": [
                python,
                str(ROOT / "experiments" / "controlled_comparison" / "assemble_report.py"),
                "--output-dir",
                str(args.final_report_root),
            ],
            "complete": lambda: _current_final_report(
                args.final_report_root / "final_report_manifest.json"
            ),
        },
    ]


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "status": "running", "phases": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _record_phase(state: dict[str, Any], payload: dict[str, Any]) -> None:
    phases = list(state.get("phases") or [])
    phases.append(payload)
    state["phases"] = phases
    state["updated_at_unix"] = time.time()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--executor-python",
        type=Path,
        default=None,
        help=(
            "Python environment containing RATS/Robosuite dependencies. On "
            "resume, the launcher recorded in shared_api_services/lifecycle.json "
            "is reused automatically."
        ),
    )
    parser.add_argument(
        "--rats-root",
        type=Path,
        default=None,
        help=(
            "Pinned RATS checkout. If omitted, require and reuse the identical "
            "source recorded by the completed CaP-X and RATS-90 baselines."
        ),
    )
    parser.add_argument(
        "--robosuite-root",
        type=Path,
        default=None,
        help=(
            "Pinned Robosuite checkout.  On resume, the campaign reuses the "
            "source recorded by the RACaP development manifest; for a fresh "
            "campaign it prefers third_party/robosuite_pinned."
        ),
    )
    parser.add_argument(
        "--racap-evolution-root",
        type=Path,
        default=ROOT / "outputs" / "evolution" / "robosuite_racap_rs_15",
    )
    parser.add_argument(
        "--rats-evolution-root",
        type=Path,
        default=ROOT / "outputs" / "evolution" / "robosuite_rats_rs_time3",
    )
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "robosuite",
    )
    parser.add_argument(
        "--development-analysis-root",
        type=Path,
        default=ROOT
        / "outputs"
        / "controlled_comparison"
        / "analysis"
        / "robosuite_development",
    )
    parser.add_argument(
        "--transfer-analysis-root",
        type=Path,
        default=ROOT
        / "outputs"
        / "controlled_comparison"
        / "analysis"
        / "robosuite_transfer",
    )
    parser.add_argument(
        "--final-report-root",
        type=Path,
        default=ROOT
        / "outputs"
        / "controlled_comparison"
        / "analysis"
        / "final",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=ROOT
        / "outputs"
        / "controlled_comparison"
        / "robosuite_campaign",
    )
    args = parser.parse_args()

    # Reuse the baseline executor checkout by provenance, not a convenient
    # repository-local default.  This prevents a resumed campaign from mixing
    # two RATS source trees after the first methods have already been scored.
    if args.rats_root is None:
        recorded_rats_sources: set[str] = set()
        recorded_robosuite_sources: set[str] = set()
        for method in ("capx", "rats_90"):
            manifest_path = args.transfer_root / method / "run_manifest.json"
            if manifest_path.is_file():
                try:
                    baseline_manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    baseline_manifest = {}
                value = str(baseline_manifest.get("rats_source") or "")
                if value:
                    recorded_rats_sources.add(value)
                robosuite_value = str(baseline_manifest.get("robosuite_source") or "")
                if robosuite_value:
                    recorded_robosuite_sources.add(robosuite_value)
        if not recorded_rats_sources and len(recorded_robosuite_sources) == 1:
            # Historical generated-method manifests predate ``rats_source``.
            # Their pinned Robosuite checkout is stored at
            # <RATS>/rats/third_party/robosuite, which identifies the same
            # executor repository without guessing from the current checkout.
            source = Path(recorded_robosuite_sources.pop()).resolve()
            if len(source.parents) >= 3:
                inferred = source.parents[2]
                if (inferred / "rats").is_dir():
                    recorded_rats_sources.add(str(inferred))
        if len(recorded_rats_sources) != 1:
            raise SystemExit(
                "completed Robosuite baselines do not identify one shared RATS source: "
                f"{sorted(recorded_rats_sources)!r}"
            )
        args.rats_root = Path(recorded_rats_sources.pop())

    # A resumed development run must use the exact source path registered in
    # its immutable protocol manifest.  Falling back blindly to the vendored
    # RATS submodule is unsafe because source archives may contain an empty,
    # uninitialized gitlink.  Fresh runs use the repository-local pinned clone.
    if args.robosuite_root is None:
        protocol_manifest = args.racap_evolution_root / "protocol_manifest.json"
        recorded_source: str | None = None
        if protocol_manifest.is_file():
            try:
                recorded_source = str(
                    json.loads(protocol_manifest.read_text(encoding="utf-8")).get(
                        "robosuite_source"
                    )
                    or ""
                )
            except (OSError, json.JSONDecodeError):
                recorded_source = None
        args.robosuite_root = (
            Path(recorded_source)
            if recorded_source
            else ROOT / "third_party" / "robosuite_pinned"
        )
    if args.executor_python is None:
        lifecycle = (
            args.racap_evolution_root / "shared_api_services" / "lifecycle.json"
        )
        recorded_python: str | None = None
        if lifecycle.is_file():
            try:
                argv = json.loads(lifecycle.read_text(encoding="utf-8")).get("argv") or []
                if argv:
                    recorded_python = str(argv[0])
            except (OSError, json.JSONDecodeError, TypeError):
                recorded_python = None
        args.executor_python = (
            Path(recorded_python) if recorded_python else DEFAULT_PYTHON
        )
    for name in (
        "rats_root",
        "robosuite_root",
        "racap_evolution_root",
        "rats_evolution_root",
        "transfer_root",
        "development_analysis_root",
        "transfer_analysis_root",
        "final_report_root",
        "campaign_root",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve())
    args.python = Path(os.path.abspath(args.python))
    # Keep the venv launcher symlink intact so child site-packages remain active.
    args.executor_python = Path(os.path.abspath(args.executor_python))
    if not os.environ.get("RACAP_VAPI_KEY") or not os.environ.get("RACAP_VAPI_BASE"):
        raise SystemExit("RACAP_VAPI_KEY and RACAP_VAPI_BASE must be configured")
    for method in ("capx", "rats_90"):
        required = args.transfer_root / method / "COMPLETE.json"
        if not required.is_file():
            raise SystemExit(f"missing frozen Robosuite baseline: {required}")
    _archive_invalid_racap_zero_shot(args.transfer_root)

    args.campaign_root.mkdir(parents=True, exist_ok=True)
    state_path = args.campaign_root / "state.json"
    state = _load_state(state_path)
    state.update(
        {
            "status": "running",
            "current_phase": None,
            "execution_order": [phase["name"] for phase in _phase_specs(args)],
            "methods_run_sequentially": True,
        }
    )
    for stale_key in ("failed_phase", "paused_phase", "failure_reason"):
        state.pop(stale_key, None)
    _write_json(state_path, state)
    for phase in _phase_specs(args):
        complete: Callable[[], bool] = phase["complete"]
        if complete():
            _record_phase(
                state,
                {
                    "name": phase["name"],
                    "status": "already_complete",
                    "time_unix": time.time(),
                },
            )
            _write_json(state_path, state)
            continue
        state["current_phase"] = phase["name"]
        state["phase_started_at_unix"] = time.time()
        _write_json(state_path, state)
        log = args.campaign_root / "logs" / f"{phase['name']}.log"
        result = _run_logged(
            phase["command"],
            cwd=ROOT,
            env=os.environ.copy(),
            log_path=log,
        )
        record = {
            "name": phase["name"],
            "status": "complete" if result["returncode"] == 0 else "failed",
            "returncode": result["returncode"],
            "quota_failure": bool(result.get("quota_failure")),
            "command": phase["command"],
            "log": str(log.resolve()),
            "time_unix": time.time(),
        }
        _record_phase(state, record)
        state["current_phase"] = None
        if result["returncode"] == 75 or result.get("quota_failure"):
            state["status"] = "paused_quota"
            state["paused_phase"] = phase["name"]
            _write_json(state_path, state)
            _write_json(args.campaign_root / "PAUSED_QUOTA.json", record)
            return 75
        if result["returncode"] != 0:
            state["status"] = "failed"
            state["failed_phase"] = phase["name"]
            _write_json(state_path, state)
            return int(result["returncode"])
        if not complete():
            state["status"] = "failed"
            state["failed_phase"] = phase["name"]
            state["failure_reason"] = "phase returned zero without completion evidence"
            _write_json(state_path, state)
            return 2
        _write_json(state_path, state)

    state["status"] = "complete"
    state["current_phase"] = None
    state["completed_at_unix"] = time.time()
    _write_json(state_path, state)
    _write_json(args.campaign_root / "COMPLETE.json", state)
    (args.campaign_root / "PAUSED_QUOTA.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
