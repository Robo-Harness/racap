#!/usr/bin/env python3
"""Evaluate a frozen RACaP candidate on the registered Robosuite transfer set."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class EpisodeWallTimeExceeded(BaseException):
    pass


@contextmanager
def _deadline(seconds: float):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def expire(_signum, _frame):
        raise EpisodeWallTimeExceeded(f"episode exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _normalise_outcome(raw: Any) -> SimpleNamespace:
    if not isinstance(raw, dict):
        raw = vars(raw)
    return SimpleNamespace(
        success=bool(raw.get("success", False)),
        turns=int(raw.get("turns", 0)),
        steps=list(raw.get("steps") or []),
        reflections=list(raw.get("reflections") or []),
        stopped=str(raw.get("stopped", "")),
    )


def _quota_or_provider(error: BaseException) -> str:
    try:
        from racap.backends.llm import LLMProviderUnavailableError, LLMQuotaError

        if isinstance(error, LLMQuotaError):
            return "quota_exhausted"
        if isinstance(error, LLMProviderUnavailableError):
            return "provider_unavailable"
    except Exception:
        pass
    text = f"{type(error).__name__}: {error}".lower()
    if any(
        marker in text
        for marker in (
            "insufficient_user_quota",
            "insufficient quota",
            "insufficient balance",
            "balance is insufficient",
            "need pre-deduct",
        )
    ):
        return "quota_exhausted"
    if "no available channel" in text or "no available route" in text:
        return "provider_unavailable"
    if "par_vapi_key is unset" in text or "vapi key is unset" in text:
        return "provider_unavailable"
    return ""


def _outcome_abort_reason(outcome: SimpleNamespace | None, llm_stats: dict[str, Any]) -> str:
    """Surface provider failures even when a controller catches the exception.

    ReAct intentionally converts tool/model exceptions into observations so it
    can recover. A terminal quota error, however, makes the episode incomplete
    evidence and must propagate to the evaluator checkpoint rather than being
    scored as a policy failure.
    """
    if int(llm_stats.get("quota_failures", 0) or 0) > 0:
        return "quota_exhausted"
    if outcome is not None:
        reason = _quota_or_provider(RuntimeError(str(outcome.stopped or "")))
        if reason:
            return reason
    return ""


def _terminal_model_failure(llm_stats: dict[str, Any]) -> bool:
    """True when one logical hosted-model call exhausted all exact retries."""

    return int(llm_stats.get("terminal_failures", 0) or 0) > 0


def _episode_worker(payload: dict[str, Any]) -> dict[str, Any]:
    task = str(payload["task"])
    seed = int(payload["seed"])
    task_id = int(payload["task_id"])
    key = f"robosuite/{task}/seed{seed}"
    episode_dir = Path(payload["output_dir"]) / "raw" / task / f"seed{seed}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RACAP_EPISODE_KEY"] = key
    os.environ["RACAP_LLM_TELEMETRY_PATH"] = str(episode_dir / "llm_calls.jsonl")
    os.environ["RACAP_NATIVE_TELEMETRY_PATH"] = str(episode_dir / "native_states.jsonl")
    os.environ["RACAP_SIM_EPISODE_TELEMETRY_PATH"] = str(episode_dir / "sim_episodes.jsonl")
    os.environ.setdefault("RACAP_MODEL", str(payload["model"]))
    started = time.time()
    runtime = None
    public_runtime = None
    outcome = None
    frames = []
    native = False
    reward = 0.0
    error = ""
    abort_reason = ""
    video_error = ""
    llm_stats: dict[str, Any] = {}
    runtime_stats: dict[str, Any] = {}
    simulator_steps = 0
    phase = "candidate_import"
    infrastructure_error = False
    try:
        solution_root = Path(payload["solution_root"]).resolve()
        if str(solution_root) not in sys.path:
            sys.path.insert(0, str(solution_root))
        from solution.controller import run_episode
        from evolution.harness.public_runtime import PublicRuntime
        from racap.backends.llm import call_stats, reset_call_stats
        from racap.backends.robosuite import RobosuitePrimitiveRuntime, TASK_SPECS

        phase = "runtime_initialization"
        runtime = RobosuitePrimitiveRuntime(
            task,
            seed=seed,
            max_steps=int(payload["max_steps"]),
        )
        if bool(payload["record_rollouts"]):
            runtime.enable_video_capture(subsample_rate=int(payload["video_subsample"]))
        runtime.record_evaluator_checkpoint("episode_start")
        public_runtime = PublicRuntime(runtime)
        reset_call_stats()
        phase = "policy_execution"
        with _deadline(float(payload["episode_timeout_seconds"])):
            outcome = _normalise_outcome(
                run_episode(
                    public_runtime,
                    {
                        "instruction": TASK_SPECS[task].instruction,
                        "model": str(payload["model"]),
                        "budgets": dict(payload["budgets"]),
                    },
                )
            )
        llm_stats = call_stats()
        abort_reason = _outcome_abort_reason(outcome, llm_stats)
        runtime_stats = public_runtime.call_stats()
        simulator_steps = int(getattr(runtime._env, "_sim_step_count", 0))
        native = runtime.oracle.task_completed()
        reward = runtime.oracle.reward()
        runtime.record_evaluator_checkpoint("episode_end")
    except (EpisodeWallTimeExceeded, Exception) as exc:
        error = f"{type(exc).__name__}: {exc}"
        abort_reason = _quota_or_provider(exc)
        # Registration, reset, and API wiring belong to the evaluator. A
        # failure before a runtime reaches the candidate is not a policy
        # failure and must not enter the accuracy denominator.
        infrastructure_error = phase == "runtime_initialization"
        if runtime is not None:
            try:
                simulator_steps = int(getattr(runtime._env, "_sim_step_count", 0))
                native = runtime.oracle.task_completed()
                reward = runtime.oracle.reward()
                runtime.record_evaluator_checkpoint(
                    "episode_wall_timeout"
                    if isinstance(exc, EpisodeWallTimeExceeded)
                    else "episode_error"
                )
            except Exception as audit_error:
                error += f"; terminal_audit={type(audit_error).__name__}: {audit_error}"
        if public_runtime is not None:
            runtime_stats = public_runtime.call_stats()
        try:
            from racap.backends.llm import call_stats

            llm_stats = call_stats()
            abort_reason = abort_reason or _outcome_abort_reason(outcome, llm_stats)
        except Exception:
            pass
    finally:
        if runtime is not None:
            if bool(payload["record_rollouts"]):
                try:
                    runtime.capture_video_frame()
                    frames = runtime.video_frames(clear=True)
                    runtime.disable_video_capture()
                except Exception as exc:
                    video_error = f"{type(exc).__name__}: {exc}"
            runtime.close()

    if not abort_reason and _terminal_model_failure(llm_stats):
        infrastructure_error = True
        phase = "hosted_model_terminal_failure"
        terminal = (
            "hosted model request exhausted its registered exact-request retries"
        )
        error = f"{error}; {terminal}" if error else terminal

    row = {
        "schema_version": 1,
        "full_task": True,
        "key": key,
        "suite": "robosuite_transfer",
        "task": task,
        "task_id": task_id,
        "seed": seed,
        "instruction": str(payload["instruction"]),
        "native_success": bool(native),
        "reward": float(reward),
        "agent_success": bool(outcome.success) if outcome else False,
        "turns": int(outcome.turns) if outcome else 0,
        "steps": outcome.steps if outcome else [],
        "reflections": outcome.reflections if outcome else [],
        "stopped": outcome.stopped if outcome else error,
        "error": error,
        "error_phase": phase if error else "",
        "infrastructure_error": bool(infrastructure_error),
        "scorable": not bool(infrastructure_error or abort_reason),
        "abort_reason": abort_reason,
        "seconds": round(time.time() - started, 3),
        "model": str(payload["model"]),
        "turn_budget": int(payload["budgets"]["turns"]),
        "pickplace_budget": int(payload["budgets"]["pickplace"]),
        "push_budget": int(payload["budgets"]["push"]),
        "insert_budget": int(payload["budgets"]["insert"]),
        "state_budget": int(payload["budgets"]["state"]),
        "stack_budget": int(payload["budgets"]["stack"]),
        "simulator_horizon": int(payload["max_steps"]),
        "episode_timeout_seconds": float(payload["episode_timeout_seconds"]),
        "controller": str(payload["method_label"]),
        "evaluator_runtime_calls": runtime_stats,
        "evaluator_llm_calls": llm_stats,
        "simulator_steps": simulator_steps,
    }
    # The policy has finished and no longer needs raw RGB/depth arrays embedded
    # in tool reports. Persist a compact, analysis-complete record; rollout.mp4
    # remains the authoritative visual stream. This also bounds records.jsonl
    # and multiprocessing result sizes for long-horizon candidates.
    from racap.agent.artifacts import compact_trajectory_row

    row = compact_trajectory_row(row)
    if bool(payload["record_rollouts"]):
        try:
            from racap.agent.artifacts import write_episode_artifacts

            row["artifacts"] = write_episode_artifacts(
                Path(payload["output_dir"]), row, frames, fps=int(payload["video_fps"])
            )
        except Exception as exc:
            video_error = f"{type(exc).__name__}: {exc}"
    if video_error:
        row["video_error"] = video_error
    (episode_dir / "record.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return row


def _completed(output_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in output_dir.glob("raw/*/seed*/record.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        inferred_abort = str(row.get("abort_reason") or "") or _outcome_abort_reason(
            SimpleNamespace(stopped=str(row.get("stopped") or "")),
            dict(row.get("evaluator_llm_calls") or {}),
        )
        if (
            row.get("key")
            and not inferred_abort
            and not _terminal_model_failure(
                dict(row.get("evaluator_llm_calls") or {})
            )
            and not row.get("infrastructure_error")
            and row.get("scorable", True)
        ):
            rows[str(row["key"])] = row
    return rows


def _archive_invalid_episode_attempts(
    output_dir: Path, complete: dict[str, dict[str, Any]]
) -> list[Path]:
    """Move non-scorable raw episodes aside before a clean resume reset."""

    archived: list[Path] = []
    archive_root = output_dir / "invalid_attempts"
    for episode_root in sorted(
        path for path in output_dir.glob("raw/*/seed*") if path.is_dir()
    ):
        record_path = episode_root / "record.json"
        try:
            row = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            row = {}
        inferred_task = episode_root.parent.name
        inferred_seed = episode_root.name.removeprefix("seed")
        key = str(
            row.get("key") or f"robosuite/{inferred_task}/seed{inferred_seed}"
        )
        if key and key in complete:
            continue
        relative = episode_root.relative_to(output_dir / "raw")
        stem = "_".join(relative.parts)
        destination = archive_root / f"{stem}_{time.time_ns()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        episode_root.replace(destination)
        (destination / "NOT_SCORED.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "invalid_non_scoring",
                    "reason": (
                        "episode contained infrastructure or exhausted hosted-model "
                        "transport failure; archived before clean resume"
                    ),
                    "episode_key": key,
                    "formal_score_use": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        archived.append(destination)
    return archived


def main() -> None:
    from racap.backends.robosuite import TASK_SPECS

    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", choices=tuple(TASK_SPECS), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--model", default=os.environ.get("RACAP_MODEL", "gpt-5.5"))
    parser.add_argument("--method-label", default="racap_phase2_zero_shot")
    parser.add_argument(
        "--protocol-note",
        default="frozen LIBERO-90 champion; no Robosuite rollout feedback",
    )
    parser.add_argument("--turns", type=int, default=45)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--episode-timeout-seconds", type=float, default=1000.0)
    parser.add_argument("--record-rollouts", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-subsample", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tasks = list(args.tasks or TASK_SPECS)
    seeds = list(dict.fromkeys(int(value) for value in args.seeds))
    solution_root = args.solution_root.resolve()
    output_dir = args.output_dir.resolve()
    if not (solution_root / "solution" / "controller.py").is_file():
        raise SystemExit(f"invalid frozen candidate: {solution_root}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"refusing non-empty output without --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = {
        "turns": int(args.turns),
        "pickplace": 8,
        "push": 3,
        "insert": 2,
        "state": 5,
        "stack": 2,
    }
    manifest = {
        "schema_version": 1,
        "method": str(args.method_label),
        "tasks": tasks,
        "seeds": seeds,
        "episodes": len(tasks) * len(seeds),
        "workers": min(int(args.workers), 5),
        "model": str(args.model),
        "solution_root": str(solution_root),
        "solution_tree_sha256": _tree_sha256(solution_root),
        "backend_path": str((ROOT / "racap/backends/robosuite.py").resolve()),
        "backend_sha256": _sha256(ROOT / "racap/backends/robosuite.py"),
        "public_runtime_sha256": _sha256(ROOT / "evolution/harness/public_runtime.py"),
        "protocol_note": str(args.protocol_note),
        "native_oracle_visible_to_policy": False,
        "privileged": False,
        "budgets": budgets,
        "max_steps": int(args.max_steps),
        "episode_timeout_seconds": float(args.episode_timeout_seconds),
        "record_rollouts": bool(args.record_rollouts),
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior != manifest:
            raise SystemExit("resume manifest mismatch")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    complete = _completed(output_dir)
    if args.resume:
        _archive_invalid_episode_attempts(output_dir, complete)
    payloads = []
    for task_id, task in enumerate(TASK_SPECS):
        if task not in tasks:
            continue
        for seed in seeds:
            key = f"robosuite/{task}/seed{seed}"
            if key in complete:
                continue
            payloads.append(
                {
                    "task": task,
                    "task_id": task_id,
                    "seed": seed,
                    "instruction": TASK_SPECS[task].instruction,
                    "solution_root": str(solution_root),
                    "output_dir": str(output_dir),
                    "model": str(args.model),
                    "budgets": budgets,
                    "max_steps": int(args.max_steps),
                    "episode_timeout_seconds": float(args.episode_timeout_seconds),
                    "record_rollouts": bool(args.record_rollouts),
                    "video_fps": int(args.video_fps),
                    "video_subsample": int(args.video_subsample),
                    "method_label": str(args.method_label),
                }
            )
    # A historical evaluator bug could write COMPLETE even though caught quota
    # failures were present inside ReAct steps. If resume validation finds any
    # pending keys, preserve those stale aggregate files for audit and remove
    # their misleading active markers before launching retries.
    if payloads and (output_dir / "COMPLETE.json").exists():
        archive = output_dir / f"invalidated_completion_{time.time_ns()}"
        archive.mkdir(parents=True, exist_ok=False)
        for name in (
            "COMPLETE.json",
            "summary.json",
            "records.jsonl",
            "README.md",
            "index.html",
            "ABORTED.json",
            "checkpoint.json",
        ):
            path = output_dir / name
            if path.exists():
                path.replace(archive / name)
    print(
        f"RACaP Robosuite: {len(complete)} complete, {len(payloads)} pending, "
        f"{min(int(args.workers), 5)} workers",
        flush=True,
    )
    context = mp.get_context("spawn")
    abort = None
    infrastructure_failure = None
    if payloads:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(int(args.workers), 5),
            mp_context=context,
        ) as executor:
            futures = {executor.submit(_episode_worker, payload): payload for payload in payloads}
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                print(
                    f"{'PASS' if row['native_success'] else 'fail'} {row['key']} "
                    f"{row['turns']} turns {row['seconds']:.1f}s",
                    flush=True,
                )
                if row.get("abort_reason"):
                    abort = {
                        "reason": row["abort_reason"],
                        "episode_key": row["key"],
                        "error": row["error"],
                    }
                    for pending in futures:
                        pending.cancel()
                    break
                if row.get("infrastructure_error"):
                    infrastructure_failure = {
                        "reason": "infrastructure_error",
                        "episode_key": row["key"],
                        "error_phase": row.get("error_phase", ""),
                        "error": row.get("error", ""),
                    }
                    for pending in futures:
                        pending.cancel()
                    break
    rows = _completed(output_dir)
    if abort:
        (output_dir / "ABORTED.json").write_text(
            json.dumps(abort, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        missing = sorted(
            f"robosuite/{task}/seed{seed}"
            for task in tasks
            for seed in seeds
            if f"robosuite/{task}/seed{seed}" not in rows
        )
        (output_dir / "checkpoint.json").write_text(
            json.dumps(
                {"status": "paused", "completed": len(rows), "missing_keys": missing},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise SystemExit(75)
    if infrastructure_failure:
        (output_dir / "FAILED.json").write_text(
            json.dumps(infrastructure_failure, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        missing = sorted(
            f"robosuite/{task}/seed{seed}"
            for task in tasks
            for seed in seeds
            if f"robosuite/{task}/seed{seed}" not in rows
        )
        (output_dir / "checkpoint.json").write_text(
            json.dumps(
                {
                    "status": "infrastructure_failed",
                    "completed_scorable": len(rows),
                    "missing_keys": missing,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise SystemExit(2)
    expected = len(tasks) * len(seeds)
    if len(rows) != expected:
        raise SystemExit(f"incomplete run: {len(rows)}/{expected}")
    ordered = sorted(rows.values(), key=lambda row: (row["task_id"], row["seed"]))
    (output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in ordered),
        encoding="utf-8",
    )
    from racap.agent.artifacts import write_rollout_index

    if args.record_rollouts:
        write_rollout_index(output_dir, ordered)
    successes = sum(bool(row["native_success"]) for row in ordered)
    summary = {
        "method": str(args.method_label),
        "episodes": len(ordered),
        "successes": successes,
        "success_rate": successes / len(ordered),
        "records": ordered,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "COMPLETE.json").write_text(
        json.dumps({"episodes": len(ordered), "successes": successes}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
