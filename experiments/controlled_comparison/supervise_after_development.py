#!/usr/bin/env python3
"""Start the frozen comparison only after registered RATS development exits.

All paths are anchored to the repository rather than the launcher's current
directory.  This small supervisor is intentionally credential-blind: provider
configuration is sourced only inside the child shell and is never printed or
written to its state file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_rows(root: Path, seal_path: Path) -> list[dict[str, object]]:
    return [
        {
            "path": item.relative_to(root).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": _sha256(item),
        }
        for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        if item != seal_path
    ]


def _artifact_content_sha256(rows: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _verify_frozen_artifact_seal(root: Path) -> dict[str, object]:
    """Recompute the seal and permissions without modifying the artifact."""

    seal_path = root / "seal_manifest.json"
    if not seal_path.is_file():
        return {"status": "error", "error": "missing seal_manifest.json"}
    try:
        expected = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "error": f"invalid seal_manifest.json: {type(exc).__name__}: {exc}",
        }
    rows = _artifact_rows(root, seal_path)
    content_sha256 = _artifact_content_sha256(rows)
    writable = [
        item.relative_to(root).as_posix() or "."
        for item in [root, *sorted(root.rglob("*"))]
        if item.stat().st_mode & 0o222
    ]
    matches = (
        expected.get("files") == rows
        and expected.get("content_sha256") == content_sha256
        and not writable
    )
    return {
        "status": "pass" if matches else "error",
        "manifest": str(seal_path.resolve()),
        "manifest_sha256": _sha256(seal_path),
        "content_sha256": content_sha256,
        "files": len(rows),
        "writable_paths": writable,
        "error": None if matches else "frozen artifact differs from its seal",
    }


def _seal_frozen_artifact(root: Path) -> dict[str, object]:
    """Write a content manifest and remove write bits from the frozen artifact."""

    seal_path = root / "seal_manifest.json"
    rows = _artifact_rows(root, seal_path)
    content_sha256 = _artifact_content_sha256(rows)
    payload: dict[str, object] = {
        "schema_version": 1,
        "root": str(root.resolve()),
        "content_sha256": content_sha256,
        "files": rows,
        "read_only_permissions": {"files": "0444", "directories": "0555"},
    }
    if seal_path.is_file():
        existing = json.loads(seal_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("frozen artifact content differs from its seal manifest")
    else:
        _write_state(seal_path, payload)

    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        item.chmod(0o444)
    for directory in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    root.chmod(0o555)
    verified = _verify_frozen_artifact_seal(root)
    if verified["status"] != "pass":
        raise RuntimeError(str(verified["error"]))
    return verified


def _bind_development_provenance(artifact: Path) -> dict[str, object]:
    """Copy the append-only launch ledger into the frozen artifact.

    This runs only after the development writer exits, so the source ledger is
    stable.  Missing provenance blocks evaluation: a skill library without a
    record of the source that produced it is not a complete frozen artifact.
    """

    source = (
        ROOT
        / "outputs"
        / "controlled_comparison"
        / "development"
        / "rats90_selfplay"
        / "launch_source_provenance.jsonl"
    )
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError(f"missing non-empty development source ledger: {source}")
    destination = artifact.parent / source.name
    shutil.copy2(source, destination)
    digest = _sha256(destination)
    manifest_path = artifact.parent / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing frozen artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "development_launch_provenance": str(destination.resolve()),
            "development_launch_provenance_sha256": digest,
        }
    )
    _write_state(manifest_path, manifest)
    return {"path": str(destination.resolve()), "sha256": digest}


def _finalize_budget_boundary(artifact: Path) -> dict[str, object]:
    """Freeze only the last committed round when the reset cap fires mid-round.

    The development child may mutate live skills and failure memory before it
    commits ``iteration_NNN.json``.  All of that work remains in the raw audit
    trail and counts against the reset/API budget, but an uncommitted state is
    not an admissible evolved artifact.  The wrapper normally restores the
    latest snapshot itself; this supervisor-side check is an independent
    safeguard for an already-running wrapper that loaded older source.
    """

    development = (
        ROOT
        / "outputs"
        / "controlled_comparison"
        / "development"
        / "rats90_selfplay"
    )
    checkpoint_path = development / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"missing development checkpoint: {checkpoint_path}")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("termination") != "simulator_reset_budget":
        return {
            "status": "not_applicable",
            "termination": checkpoint.get("termination"),
        }

    try:
        from experiments.controlled_comparison.run_rats_development import (
            _restore_latest_completed_snapshot,
            _tree_sha256,
        )
    except ImportError:  # direct ``python path/to/script.py`` execution
        from run_rats_development import (  # type: ignore[no-redef]
            _restore_latest_completed_snapshot,
            _tree_sha256,
        )

    restoration = _restore_latest_completed_snapshot(development)
    if restoration is None:
        raise RuntimeError(
            "simulator reset budget ended without a completed round snapshot"
        )
    source_skills = development / "skills.json"
    source_memory = development / "failure_memory"
    if not source_skills.is_file():
        raise RuntimeError("restored development state has no skills.json")

    artifact_root = artifact.parent
    manifest_path = artifact_root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing frozen artifact manifest: {manifest_path}")
    before = {
        "skills_sha256": _sha256(artifact) if artifact.is_file() else "",
        "failure_memory_tree_sha256": _tree_sha256(
            artifact_root / "failure_memory"
        ),
    }
    shutil.copy2(source_skills, artifact)
    frozen_memory = artifact_root / "failure_memory"
    if frozen_memory.exists():
        shutil.rmtree(frozen_memory)
    if source_memory.is_dir():
        shutil.copytree(source_memory, frozen_memory)
    else:
        frozen_memory.mkdir(parents=True, exist_ok=True)

    after = {
        "skills_sha256": _sha256(artifact),
        "failure_memory_tree_sha256": _tree_sha256(frozen_memory),
    }
    transaction: dict[str, object] = {
        "status": "pass",
        "termination": "simulator_reset_budget",
        "restoration": restoration,
        "frozen_before": before,
        "frozen_after": after,
        "artifact_matches_last_completed_snapshot": True,
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "frozen_library_sha256": after["skills_sha256"],
            "frozen_failure_memory_tree_sha256": after[
                "failure_memory_tree_sha256"
            ],
            "budget_boundary_transaction": transaction,
        }
    )
    _write_state(manifest_path, manifest)
    return transaction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--experiment-python",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT
        / "outputs"
        / "controlled_comparison"
        / "post_development_supervisor.json",
    )
    args = parser.parse_args()
    artifact = (
        ROOT
        / "outputs"
        / "controlled_comparison"
        / "artifacts"
        / "rats90_frozen"
        / "skills.json"
    )
    started = time.time()
    state: dict[str, object] = {
        "schema_version": 1,
        "repository": str(ROOT),
        "development_pid": args.development_pid,
        "experiment_python": str(args.experiment_python),
        "started_at_unix": started,
        "status": "waiting_for_development",
        "provider_credentials_recorded": False,
    }
    _write_state(args.state, state)
    while _pid_alive(args.development_pid):
        time.sleep(max(1.0, args.poll_seconds))
    state["development_exited_at_unix"] = time.time()
    if not artifact.is_file():
        state.update(
            {
                "status": "development_stopped_without_frozen_artifact",
                "returncode": 75,
            }
        )
        _write_state(args.state, state)
        return 75

    try:
        budget_boundary = _finalize_budget_boundary(artifact)
        provenance = _bind_development_provenance(artifact)
        seal = _seal_frozen_artifact(artifact.parent)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        state.update(
            {
                "status": "development_artifact_provenance_failed",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_state(args.state, state)
        return 1

    env_file = ROOT / "configs" / "env.sh"
    run_all = ROOT / "experiments" / "controlled_comparison" / "run_all.py"
    shell = """
set -a
source "$1"
set +a
export RACAP_EXPERIMENT_PYTHON="$2"
export RACAP_LLM_CACHE=0
export PYTHONPATH="$3:$3/third_party/rats${PYTHONPATH:+:$PYTHONPATH}"
exec python "$4"
"""
    state.update(
        {
            "status": "running_frozen_evaluation",
            "evaluation_started_at_unix": time.time(),
            "artifact": str(artifact),
            "budget_boundary_transaction": budget_boundary,
            "development_launch_provenance": provenance,
            "frozen_artifact_seal": seal,
        }
    )
    _write_state(args.state, state)
    returncode = subprocess.run(
        [
            "bash",
            "-lc",
            shell,
            "supervisor-child",
            str(env_file),
            str(args.experiment_python),
            str(ROOT),
            str(run_all),
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        check=False,
    ).returncode
    state.update(
        {
            "status": "evaluation_complete" if returncode == 0 else "evaluation_stopped",
            "evaluation_finished_at_unix": time.time(),
            "returncode": returncode,
        }
    )
    _write_state(args.state, state)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
