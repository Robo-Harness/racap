#!/usr/bin/env python3
"""Run and freeze the registered RATS LIBERO-90 development artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .analyze_results import _request_list_price_usd
except ImportError:  # direct ``python path/to/script.py`` execution
    from analyze_results import _request_list_price_usd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_RATS_ROOT = ROOT / "third_party" / "rats"
DEFAULT_PYTHON = Path(os.environ.get("RACAP_EXPERIMENT_PYTHON", sys.executable))
QUOTA_MARKERS = (
    "insufficient_user_quota",
    "insufficient quota",
    "insufficient balance",
    "balance is insufficient",
    "need pre-deduct",
    "no available channel",
    "no available route",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    """Append one credential-free provenance record atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _launch_source_provenance(
    *,
    argv: list[str],
    rats_root: Path,
    output_dir: Path,
    resumed: bool,
) -> dict[str, object]:
    """Capture the exact source snapshot a newly spawned child will import.

    The development process imports Python modules once at launch.  Recording
    hashes here, immediately before ``subprocess.run``, avoids later worktree
    edits being mistaken for code that influenced already-running rollouts.
    Provider credentials and environment values are deliberately excluded.
    """

    tracked_diff = _git_text("diff", "--binary", "--", "third_party/rats", "experiments/controlled_comparison")
    key_sources = [
        rats_root / "rats" / "loop" / "lifelong_loop.py",
        rats_root / "rats" / "envs" / "tasks" / "base.py",
        rats_root / "rats" / "executor" / "sandbox.py",
        rats_root / "rats" / "control_flow.py",
        rats_root / "rats" / "agents" / "base_agent.py",
        rats_root / "rats" / "agents" / "policy_writer.py",
        rats_root / "rats" / "agents" / "verifier.py",
        rats_root / "rats" / "agents" / "multi_turn_decider.py",
        HERE / "protocol.yaml",
        HERE / "configs" / "rats_libero.yaml",
    ]
    return {
        "schema_version": 1,
        "captured_at_unix": time.time(),
        "parent_pid": os.getpid(),
        "resumed": resumed,
        "argv": argv,
        "repository_commit": _git_text("rev-parse", "HEAD"),
        "repository_status_porcelain": _git_text("status", "--porcelain=v1"),
        "comparison_diff_sha256": hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest(),
        "rats_source_tree_sha256": _tree_sha256(rats_root / "rats"),
        "key_source_sha256": {
            str(path.resolve()): _sha256(path) if path.is_file() else ""
            for path in key_sources
        },
        "output_dir": str(output_dir.resolve()),
    }


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line)


def _agent_usage(path: Path) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "model_latency_seconds": 0.0,
        "estimated_api_cost_usd": 0.0,
    }
    for item in path.glob("*.json"):
        try:
            row: dict[str, Any] = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        response = row.get("response") or {}
        usage = response.get("usage") or row.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        totals["calls"] += 1
        totals["prompt_tokens"] += int(
            usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        )
        totals["completion_tokens"] += int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        totals["cached_tokens"] += int(
            details.get("cached_tokens") or usage.get("cached_tokens") or 0
        )
        totals["estimated_api_cost_usd"] += _request_list_price_usd(usage)
        totals["model_latency_seconds"] += float(
            row.get("elapsed_s") or row.get("elapsed_seconds") or 0.0
        )
    totals["model_latency_seconds"] = round(float(totals["model_latency_seconds"]), 3)
    totals["estimated_api_cost_usd"] = round(
        float(totals["estimated_api_cost_usd"]), 8
    )
    return totals


def _iteration_error(path: Path) -> str:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return str(row.get("error") or "")


def _read_log_since(path: Path, offset: int) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, int(offset)))
        return handle.read().decode("utf-8", errors="replace")


def _archive_previous_abort(output_dir: Path) -> Path | None:
    """Preserve a prior quota sentinel without poisoning a resumed launch."""

    source = output_dir / "ABORTED.json"
    if not source.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    archive_dir = output_dir / "interrupted_launches" / stamp
    suffix = 1
    while archive_dir.exists():
        archive_dir = output_dir / "interrupted_launches" / f"{stamp}_{suffix}"
        suffix += 1
    archive_dir.mkdir(parents=True, exist_ok=False)
    destination = archive_dir / source.name
    shutil.move(str(source), str(destination))
    return destination


def _restore_latest_completed_snapshot(output_dir: Path) -> dict[str, object] | None:
    """Restore transactional RATS state before resuming an interrupted round.

    RATS records failures during a round, while ``iteration_NNN.json`` is only
    committed after that round finishes. Without a transaction boundary, a
    killed or timed-out round leaks partial failure memory into its retry. A
    per-round snapshot is the authoritative state after the latest completed
    iteration. Preserve the newer partial state for audit, then restore the
    snapshot before ``--resume`` reconstructs proposer history.
    """

    completed: list[int] = []
    for path in sorted(output_dir.glob("iteration_*.json")):
        if _iteration_error(path):
            continue
        try:
            completed.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not completed:
        return None
    latest = max(completed)
    snapshot = output_dir / "snapshots" / f"iter{latest:03d}"
    if not (snapshot / "skills.json").is_file():
        return None

    live_skills = output_dir / "skills.json"
    live_memory = output_dir / "failure_memory"
    snap_skills = snapshot / "skills.json"
    snap_memory = snapshot / "failure_memory"
    live_skills_hash = _sha256(live_skills) if live_skills.is_file() else ""
    live_memory_hash = _tree_sha256(live_memory)
    snapshot_skills_hash = _sha256(snap_skills)
    snapshot_memory_hash = _tree_sha256(snap_memory)
    if (
        live_skills_hash == snapshot_skills_hash
        and live_memory_hash == snapshot_memory_hash
    ):
        return {
            "latest_completed_iteration": latest,
            "snapshot": str(snapshot.resolve()),
            "restored": False,
            "reason": "live_state_already_matches_snapshot",
        }

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    archive = output_dir / "interrupted_launches" / stamp / "partial_state_before_resume"
    suffix = 1
    while archive.exists():
        archive = (
            output_dir
            / "interrupted_launches"
            / f"{stamp}_{suffix}"
            / "partial_state_before_resume"
        )
        suffix += 1
    archive.mkdir(parents=True, exist_ok=False)
    if live_skills.is_file():
        shutil.copy2(live_skills, archive / "skills.json")
    if live_memory.is_dir():
        shutil.copytree(live_memory, archive / "failure_memory")

    shutil.copy2(snap_skills, live_skills)
    if live_memory.exists():
        shutil.rmtree(live_memory)
    if snap_memory.is_dir():
        shutil.copytree(snap_memory, live_memory)
    else:
        live_memory.mkdir(parents=True, exist_ok=True)

    record: dict[str, object] = {
        "time": time.time(),
        "latest_completed_iteration": latest,
        "snapshot": str(snapshot.resolve()),
        "restored": True,
        "archived_partial_state": str(archive.resolve()),
        "live_before": {
            "skills_sha256": live_skills_hash,
            "failure_memory_tree_sha256": live_memory_hash,
        },
        "restored_state": {
            "skills_sha256": snapshot_skills_hash,
            "failure_memory_tree_sha256": snapshot_memory_hash,
        },
    }
    _append_jsonl(output_dir / "resume_restorations.jsonl", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--episode-budget", type=int, default=596)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--rats-root", type=Path, default=DEFAULT_RATS_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "development" / "rats90_selfplay",
    )
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "artifacts" / "rats90_frozen",
    )
    args = parser.parse_args()
    if args.rounds <= 0 or args.episode_budget <= 0:
        raise SystemExit("rounds and episode budget must be positive")
    if not os.environ.get("RACAP_VAPI_KEY") or not os.environ.get("RACAP_VAPI_BASE"):
        raise SystemExit("RACAP_VAPI_KEY and RACAP_VAPI_BASE must be configured")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_library = args.rats_root / "skill_library" / "libero_nonpriv_skills.json"
    verifier_source = args.rats_root / "rats" / "agents" / "verifier.py"
    manifest = {
        "schema_version": 1,
        "development_pool": "libero_90",
        "rounds": args.rounds,
        "simulator_episode_cap": args.episode_budget,
        "attempts_per_round": 5,
        "turns_per_attempt": 5,
        "model": args.model,
        "runtime_vlm_model": args.model,
        "temperature": 0.0,
        "initial_library": str(initial_library.resolve()),
        "initial_library_sha256": _sha256(initial_library),
        "verifier_source": str(verifier_source.resolve()),
        "verifier_source_sha256": _sha256(verifier_source),
        "native_control_feedback": "binary_success_label_only",
        "exact_native_predicates_private_audit_only": True,
        "internal_model_policy": "all RATS reasoning and VLM verification calls use the bound model route",
        "selection": "RATS catalog curiosity proposer over the LIBERO-90 task IDs with curriculum",
        "pro_feedback_forbidden": True,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit("refusing incompatible development resume")
    else:
        _write_json(manifest_path, manifest)

    env = os.environ.copy()
    env.update(
        {
            "RATS_VAPI_URL": env.get("RACAP_VAPI_BASE", "").rstrip("/")
            + "/chat/completions",
            "RATS_VAPI_KEY": env.get("RACAP_VAPI_KEY", ""),
            "RATS_LLM_MODEL": args.model,
            "RATS_RUNTIME_VLM_MODEL": args.model,
            "RATS_LLM_FALLBACK": "0",
            "RATS_VERIFY_STEP_MODE": "strict",
            "RATS_VERIFIER_STRICT_BENCHMARK": "1",
            "RATS_AGENT_IO_DIR": str(args.output_dir / "agent_io"),
            "RATS_SIM_EPISODE_TELEMETRY_PATH": str(args.output_dir / "sim_episodes.jsonl"),
            "RATS_NATIVE_TELEMETRY_PATH": str(args.output_dir / "native_states.jsonl"),
            "RATS_SIM_EPISODE_MAX": str(args.episode_budget),
            "RATS_ABORT_SENTINEL": str(args.output_dir / "ABORTED.json"),
            "RATS_EPISODE_KEY": "development/libero_90",
            "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    argv = [
        str(args.python),
        "scripts/run_rats.py",
        "--config",
        str(HERE / "configs" / "rats_libero.yaml"),
        "--model",
        args.model,
        "--env-type",
        "libero",
        "--libero-suite",
        "libero_90",
        "--iterations",
        str(args.rounds),
        "--explore",
        "--catalog",
        "--curriculum",
        "--no-play-mode",
        "--skill-library",
        str(initial_library),
        "--turns-per-attempt",
        "5",
        "--attempts-per-iteration",
        "5",
        "--multi-turn-decision",
        "--snapshot-interval",
        "1",
        "--output-dir",
        str(args.output_dir),
        "--log-agent-io",
    ]
    invalid_before_launch = []
    for iteration_path in sorted(args.output_dir.glob("iteration_*.json")):
        error = _iteration_error(iteration_path)
        if not error:
            continue
        invalid_path = iteration_path.with_suffix(".invalid.json")
        suffix = 1
        while invalid_path.exists():
            invalid_path = iteration_path.with_suffix(f".invalid{suffix}.json")
            suffix += 1
        iteration_path.rename(invalid_path)
        invalid_before_launch.append(
            {"source": str(iteration_path), "quarantined": str(invalid_path), "error": error}
        )
    if invalid_before_launch:
        _write_json(args.output_dir / "invalid_iterations.json", invalid_before_launch)
    archived_abort = _archive_previous_abort(args.output_dir)
    if archived_abort is not None:
        print(f"archived previous abort sentinel: {archived_abort}", flush=True)
    resumed = bool(list(args.output_dir.glob("iteration_*.json")))
    if resumed:
        restoration = _restore_latest_completed_snapshot(args.output_dir)
        if restoration is not None:
            print(
                "resume state audit: " + json.dumps(restoration, ensure_ascii=False),
                flush=True,
            )
        argv.extend(["--resume", str(args.output_dir), "--skip-completed"])
    started = time.time()
    run_log = args.output_dir / "run.log"
    launch_log_offset = run_log.stat().st_size if run_log.exists() else 0
    _append_jsonl(
        args.output_dir / "launch_source_provenance.jsonl",
        _launch_source_provenance(
            argv=argv,
            rats_root=args.rats_root,
            output_dir=args.output_dir,
            resumed=resumed,
        ),
    )
    with run_log.open("a", encoding="utf-8") as log:
        log.write("\n# launch: " + " ".join(argv) + "\n")
        log.flush()
        result = subprocess.run(
            argv,
            cwd=args.rats_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    iterations = sorted(args.output_dir.glob("iteration_*.json"))
    valid_iterations = [path for path in iterations if not _iteration_error(path)]
    errored_iterations = [
        {"path": str(path), "error": _iteration_error(path)}
        for path in iterations
        if _iteration_error(path)
    ]
    launch_log_text = _read_log_since(run_log, launch_log_offset)
    budget_exhausted = "RATS simulator-episode budget reached:" in launch_log_text
    completed_by_rounds = result.returncode == 0 and len(valid_iterations) >= args.rounds
    budget_boundary_transaction = None
    if budget_exhausted:
        # The reset cap can fire in the middle of a round, after that round has
        # already appended failure memory or provisional skills.  Those calls
        # and rollouts count toward development cost, but only a completed
        # iteration is an admissible state transition.  Restore the latest
        # per-round snapshot before copying the frozen artifact.
        budget_boundary_transaction = _restore_latest_completed_snapshot(
            args.output_dir
        )
        if budget_boundary_transaction is None:
            _write_json(
                args.output_dir / "checkpoint.json",
                {
                    "status": "paused",
                    "termination": "simulator_reset_budget_without_completed_snapshot",
                    "returncode": result.returncode,
                    "completed_rounds": len(valid_iterations),
                    "simulator_episodes": _line_count(
                        args.output_dir / "sim_episodes.jsonl"
                    ),
                },
            )
            return 1
    summary = {
        "status": "complete" if completed_by_rounds or budget_exhausted else "paused",
        "termination": (
            "round_budget" if completed_by_rounds else
            "simulator_reset_budget" if budget_exhausted else
            "interrupted_or_error"
        ),
        "returncode": result.returncode,
        "wall_clock_seconds_this_launch": round(time.time() - started, 3),
        "completed_rounds": len(valid_iterations),
        "errored_rounds": errored_iterations,
        "simulator_episodes": _line_count(args.output_dir / "sim_episodes.jsonl"),
        "llm": _agent_usage(args.output_dir / "agent_io"),
        "budget_boundary_transaction": budget_boundary_transaction,
    }
    _write_json(args.output_dir / "checkpoint.json", summary)
    if (args.output_dir / "ABORTED.json").exists() or any(
        marker in launch_log_text.lower() for marker in QUOTA_MARKERS
    ):
        return 75
    if summary["status"] != "complete":
        return 1

    skills = args.output_dir / "skills.json"
    if not skills.is_file():
        raise SystemExit("development completed without skills.json")
    args.frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen = args.frozen_dir / "skills.json"
    shutil.copy2(skills, frozen)
    memory_source = args.output_dir / "failure_memory"
    memory_frozen = args.frozen_dir / "failure_memory"
    if memory_source.exists():
        source_hash = _tree_sha256(memory_source)
        if memory_frozen.exists() and _tree_sha256(memory_frozen) != source_hash:
            raise SystemExit("refusing to overwrite a different frozen failure memory")
        if not memory_frozen.exists():
            shutil.copytree(memory_source, memory_frozen)
    _write_json(
        args.frozen_dir / "artifact_manifest.json",
        {
            **manifest,
            **summary,
            "source": str(skills.resolve()),
            "frozen_library": str(frozen.resolve()),
            "frozen_library_sha256": _sha256(frozen),
            "frozen_failure_memory": (
                str(memory_frozen.resolve()) if memory_frozen.exists() else None
            ),
            "frozen_failure_memory_tree_sha256": _tree_sha256(memory_frozen),
            "frozen_at_unix": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
