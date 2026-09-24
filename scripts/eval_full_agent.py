#!/usr/bin/env python3
"""Run and record the compositional visual agent on complete LIBERO tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from types import SimpleNamespace


class InfrastructureInvariantError(RuntimeError):
    """A preregistered episode differs from the simulator state actually loaded."""


class EpisodeWallTimeExceeded(BaseException):
    """The policy exceeded the comparison-wide per-episode wall-time budget.

    This is deliberately outside ``Exception``.  Perception and hosted-model
    clients commonly retry ``Exception`` subclasses; allowing them to catch a
    process-level episode deadline silently extends the registered budget.
    The evaluator catches this control-flow signal explicitly so it can score
    the actual terminal simulator state and flush the episode artifacts.
    """


@contextmanager
def _episode_wall_timeout(seconds: float):
    """Interrupt one episode without killing its worker or losing artifacts.

    CaP-X and RATS both enforce a 1000-second SIGALRM around each trial.  RACaP
    evaluates several isolated episodes in one worker process, so the same cap
    must be scoped to one queue item and cancelled before the next item starts.
    """

    seconds = float(seconds)
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame) -> None:
        raise EpisodeWallTimeExceeded(
            f"episode exceeded {seconds:g} seconds of wall time"
        )

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _validate_applied_init_state(runtime, payload: dict) -> int:
    """Fail before the first model call when public seed pairing is broken."""

    expected = int(payload["seed"])
    applied = getattr(runtime._env, "_current_init_state_index", None)
    try:
        applied_index = int(applied)
    except (TypeError, ValueError) as exc:
        raise InfrastructureInvariantError(
            f"missing applied init-state index for public seed {expected}"
        ) from exc
    if applied_index != expected:
        raise InfrastructureInvariantError(
            "applied init-state index does not match public seed: "
            f"{applied_index} != {expected}"
        )
    return applied_index


def _abort_exit_code(abort: dict) -> int:
    """Quota/provider pauses are resumable; structural violations are errors."""

    return 75 if str(abort.get("reason")) in {
        "quota_exhausted",
        "provider_unavailable",
    } else 1


def _resolved_episode_grid(payloads: list[dict]) -> tuple[list[int], list[int]]:
    """Return the task IDs and seeds that will actually be executed."""

    task_ids = list(dict.fromkeys(int(payload["task_id"]) for payload in payloads))
    seeds = list(dict.fromkeys(int(payload["seed"]) for payload in payloads))
    return task_ids, seeds


def _full_episode_controls(
    task_id: int,
    episode_context: dict[str, dict] | None,
) -> dict[str, int | bool]:
    """Resolve public task-level control semantics for the mature controller.

    Candidate controllers receive the same mapping in their public episode
    payload.  The mature path must not silently drop it, otherwise an anytime
    collection task becomes fail-closed only for RACaP Phase 1.
    """

    context = (episode_context or {}).get(str(int(task_id)), {})
    raw_mode = context.get("objective_mode", "") if isinstance(context, dict) else ""
    mode = " ".join(str(raw_mode).lower().replace("_", " ").replace("-", " ").split())
    anytime = mode == "independent anytime"
    return {
        "continue_after_subgoal_failure": anytime,
        "max_replans": 3 if anytime else 0,
    }


def _outcome_infrastructure_failure(outcome) -> dict | None:
    """Return a swallowed Policy-API infrastructure failure, if any.

    ReAct intentionally turns a failed tool call into a structured observation
    so it can stop cleanly and preserve artifacts.  The evaluator must still
    promote quota/provider failures to a campaign-wide abort; otherwise every
    remaining task consumes simulator startup time only to reproduce the same
    external outage.
    """

    for step in list(getattr(outcome, "steps", []) or []):
        report = step.get("report") if isinstance(step, dict) else getattr(step, "report", None)
        if not isinstance(report, dict):
            continue
        failure = report.get("infrastructure_failure")
        if isinstance(failure, dict) and failure:
            return dict(failure)
    return None


def _abort_reason_for_infrastructure_failure(failure: dict) -> str:
    return (
        "quota_exhausted"
        if str(failure.get("kind", "")).lower() == "provider_quota"
        else "provider_unavailable"
    )


def _worker(
    payloads: list[dict],
    jsonl: Path,
    out_dir: Path,
    *,
    turns: int,
    max_pickplace_calls: int,
    max_push_calls: int,
    max_insert_calls: int,
    max_state_calls: int,
    max_stack_calls: int,
    model: str,
    grasp_backend: str,
    verbose: bool,
    worker_index: int,
    record_rollouts: bool,
    video_fps: int,
    video_subsample: int,
    max_steps: int,
    episode_timeout_seconds: float,
    solution_root: str = "",
    episode_context: dict[str, dict] | None = None,
) -> None:
    os.environ.setdefault("RACAP_WORKER", str(worker_index))
    abort_sentinel = out_dir / "ABORTED.json"
    os.environ["RACAP_ABORT_SENTINEL"] = str(abort_sentinel.resolve())
    os.environ["RACAP_LLM_TELEMETRY_PATH"] = str(
        (out_dir / f"llm_calls.worker{worker_index}.jsonl").resolve()
    )
    os.environ["RACAP_NATIVE_TELEMETRY_PATH"] = str(
        (out_dir / f"native_states.worker{worker_index}.jsonl").resolve()
    )
    os.environ["RACAP_SIM_EPISODE_TELEMETRY_PATH"] = str(
        (out_dir / f"sim_episodes.worker{worker_index}.jsonl").resolve()
    )

    from racap.backends.libero import LiberoPrimitiveRuntime
    from racap.envs.register_suites import register_missing_suites

    candidate_run = None
    if solution_root:
        candidate_source = str(Path(solution_root).resolve())
        if candidate_source not in sys.path:
            sys.path.insert(0, candidate_source)
        from solution.controller import run_episode as candidate_run
    else:
        from racap.agent.full_react import FullEpisode, run_full_episode
        from racap.policy_api.pickplace_api import pickplace

    register_missing_suites()
    for payload in payloads:
        if abort_sentinel.exists():
            return
        started = time.time()
        runtime = None
        outcome = None
        native = False
        native_predicates = []
        native_predicate_diagnostics = []
        native_entity_poses = {}
        native_scene_poses_initial = {}
        native_scene_poses_final = {}
        error = ""
        video_error = ""
        frames = []
        public_runtime = None
        runtime_call_stats = {}
        llm_call_stats = {}
        simulator_steps = 0
        global_abort = False
        infrastructure_failure = None
        key = f"{payload['suite']}/{payload['task_id']}/seed{payload['seed']}"
        os.environ["RACAP_EPISODE_KEY"] = key
        try:
            runtime = LiberoPrimitiveRuntime(
                suite_name=payload["suite"],
                task_id=int(payload["task_id"]),
                seed=int(payload["seed"]),
                public_seed_indexed=True,
                grasp_backend=grasp_backend,
                max_steps=max_steps,
            )
            _validate_applied_init_state(runtime, payload)
            if record_rollouts:
                runtime.enable_video_capture(subsample_rate=video_subsample)
            runtime.record_evaluator_checkpoint("episode_start")
            native_scene_poses_initial = runtime.oracle.scene_object_poses()
            from racap.backends.llm import call_stats, reset_call_stats

            reset_call_stats()
            # Only the natural-language instruction and images cross the
            # policy boundary. ``predicates`` stays in this evaluator.
            with _episode_wall_timeout(episode_timeout_seconds):
                if candidate_run is not None:
                    from evolution.harness.public_runtime import PublicRuntime

                    public_runtime = PublicRuntime(runtime)
                    # This payload intentionally excludes task ID, predicates and
                    # oracle state.  Stage-0 may provide curated semantic arguments
                    # parsed from the natural-language instruction; they are not
                    # evaluator answers or coordinates.
                    public_episode = {
                        "instruction": payload["instruction"],
                        "model": model,
                        "budgets": {
                            "turns": turns,
                            "pickplace": max_pickplace_calls,
                            "push": max_push_calls,
                            "insert": max_insert_calls,
                            "state": max_state_calls,
                            "stack": max_stack_calls,
                        },
                        **(episode_context or {}).get(str(payload["task_id"]), {}),
                    }
                    raw_outcome = candidate_run(public_runtime, public_episode)
                    if not isinstance(raw_outcome, dict):
                        raw_outcome = vars(raw_outcome)
                    outcome = SimpleNamespace(
                        success=bool(raw_outcome.get("success", False)),
                        turns=int(raw_outcome.get("turns", 0)),
                        steps=list(raw_outcome.get("steps") or []),
                        reflections=list(raw_outcome.get("reflections") or []),
                        stopped=str(raw_outcome.get("stopped", "")),
                    )
                else:
                    full_episode_controls = _full_episode_controls(
                        int(payload["task_id"]), episode_context
                    )
                    outcome = run_full_episode(
                        runtime,
                        pickplace,
                        FullEpisode(
                            instruction=payload["instruction"],
                            turns=turns,
                            max_pickplace_calls=max_pickplace_calls,
                            max_push_calls=max_push_calls,
                            max_insert_calls=max_insert_calls,
                            max_state_calls=max_state_calls,
                            max_stack_calls=max_stack_calls,
                            **full_episode_controls,
                            model=model,
                        ),
                        verbose=verbose,
                    )
            infrastructure_failure = _outcome_infrastructure_failure(outcome)
            if infrastructure_failure and str(
                infrastructure_failure.get("kind", "")
            ).lower() in {"provider_quota", "provider_unavailable"}:
                abort_sentinel.write_text(
                    json.dumps(
                        {
                            "time": time.time(),
                            "reason": _abort_reason_for_infrastructure_failure(
                                infrastructure_failure
                            ),
                            "episode_key": key,
                            "failure": infrastructure_failure,
                            "preserved_episode": True,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            llm_call_stats = call_stats()
            runtime_call_stats = public_runtime.call_stats() if public_runtime else {}
            simulator_steps = int(getattr(runtime._env, "_sim_step_count", 0))
            native = bool(runtime.oracle.task_completed())
            native_predicates = runtime.oracle.predicate_status()
            native_predicate_diagnostics = runtime.oracle.predicate_diagnostics()
            native_scene_poses_final = runtime.oracle.scene_object_poses()
            runtime.record_evaluator_checkpoint("episode_end")
            # Evaluator-only geometry for diagnosing a failed native
            # predicate.  This never crosses the policy boundary: the agent
            # has already stopped, and only the offline artifact receives it.
            entities = {
                str(value)
                for predicate in (payload.get("predicates") or [])
                for value in predicate[1:]
            }
            for entity in sorted(entities):
                pose = runtime.oracle.object_pose(entity)
                if pose is not None:
                    native_entity_poses[entity] = {
                        "position": [round(float(v), 6) for v in pose[0]],
                        "quaternion": [round(float(v), 6) for v in pose[1]],
                    }
        except (EpisodeWallTimeExceeded, Exception) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, EpisodeWallTimeExceeded) and runtime is not None:
                # Score the actual terminal simulator state at the common
                # deadline, just as the generated-code runners retain their
                # last native state on timeout.
                try:
                    simulator_steps = int(getattr(runtime._env, "_sim_step_count", 0))
                    native = bool(runtime.oracle.task_completed())
                    native_predicates = runtime.oracle.predicate_status()
                    native_predicate_diagnostics = runtime.oracle.predicate_diagnostics()
                    native_scene_poses_final = runtime.oracle.scene_object_poses()
                    runtime.record_evaluator_checkpoint("episode_wall_timeout")
                    entities = {
                        str(value)
                        for predicate in (payload.get("predicates") or [])
                        for value in predicate[1:]
                    }
                    for entity in sorted(entities):
                        pose = runtime.oracle.object_pose(entity)
                        if pose is not None:
                            native_entity_poses[entity] = {
                                "position": [round(float(v), 6) for v in pose[0]],
                                "quaternion": [round(float(v), 6) for v in pose[1]],
                            }
                except Exception as audit_exc:
                    error += (
                        "; timeout_terminal_audit_error="
                        f"{type(audit_exc).__name__}: {audit_exc}"
                    )
                try:
                    from racap.backends.llm import call_stats

                    llm_call_stats = call_stats()
                except Exception:
                    pass
                runtime_call_stats = public_runtime.call_stats() if public_runtime else {}
            try:
                from racap.backends.llm import LLMProviderUnavailableError, LLMQuotaError

                should_abort = isinstance(exc, (LLMQuotaError, LLMProviderUnavailableError))
            except Exception:
                should_abort = False
            infrastructure_abort = isinstance(exc, InfrastructureInvariantError)
            should_abort = should_abort or infrastructure_abort
            normalized_error = error.lower()
            should_abort = should_abort or any(
                marker in normalized_error
                for marker in (
                    "insufficient_user_quota",
                    "insufficient quota",
                    "insufficient balance",
                    "balance is insufficient",
                    "need pre-deduct",
                    "no available channel",
                    "no available route",
                )
            )
            if should_abort:
                global_abort = True
                if infrastructure_abort:
                    reason = "infrastructure_invariant"
                elif (
                    "quota" in normalized_error
                    or "balance" in normalized_error
                    or "pre-deduct" in normalized_error
                ):
                    reason = "quota_exhausted"
                else:
                    reason = "provider_unavailable"
                try:
                    abort_sentinel.write_text(
                        json.dumps(
                            {
                                "time": time.time(),
                                "reason": reason,
                                "episode_key": key,
                                "error": error,
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
        finally:
            if runtime is not None:
                if record_rollouts:
                    try:
                        runtime.capture_video_frame()
                        frames = runtime.video_frames(clear=True)
                        runtime.disable_video_capture()
                    except Exception as exc:
                        video_error = f"{type(exc).__name__}: {exc}"
                runtime.close()

        if global_abort:
            print(f"  [w{worker_index}] ABORTED {key}: {error}", flush=True)
            return

        predicates = payload.get("predicates") or []
        row = {
            "full_task": True,
            "key": key,
            "suite": payload["suite"],
            "task_id": payload["task_id"],
            "seed": payload["seed"],
            "instruction": payload["instruction"],
            "predicates": predicates,
            "predicate_types": [value[0] for value in predicates if value],
            "native_success": native,
            "native_predicates": native_predicates,
            "native_predicate_diagnostics": native_predicate_diagnostics,
            "native_entity_poses": native_entity_poses,
            "native_scene_poses_initial": native_scene_poses_initial,
            "native_scene_poses_final": native_scene_poses_final,
            "infrastructure_error": bool(infrastructure_failure),
            "infrastructure_failure": infrastructure_failure,
            "scorable": not bool(infrastructure_failure),
            "agent_success": bool(outcome.success) if outcome else False,
            "turns": outcome.turns if outcome else 0,
            "steps": outcome.steps if outcome else [],
            "reflections": outcome.reflections if outcome else [],
            "stopped": outcome.stopped if outcome else error,
            "error": error,
            "seconds": round(time.time() - started, 1),
            "model": model,
            "turn_budget": turns,
            "pickplace_budget": max_pickplace_calls,
            "push_budget": max_push_calls,
            "insert_budget": max_insert_calls,
            "state_budget": max_state_calls,
            "stack_budget": max_stack_calls,
            "simulator_horizon": max_steps,
            "episode_timeout_seconds": episode_timeout_seconds,
            "controller": "candidate" if solution_root else "phase1_core",
            "evaluator_runtime_calls": runtime_call_stats,
            "evaluator_llm_calls": llm_call_stats,
            "simulator_steps": simulator_steps,
            "init_state_index": (
                getattr(runtime._env, "_current_init_state_index", None)
                if runtime is not None
                else None
            ),
        }
        if record_rollouts:
            try:
                from racap.agent.artifacts import write_episode_artifacts

                row["artifacts"] = write_episode_artifacts(out_dir, row, frames, fps=video_fps)
            except Exception as exc:
                video_error = f"{type(exc).__name__}: {exc}"
        if video_error:
            row["video_error"] = video_error
        with jsonl.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(
            f"  [w{worker_index}] {'PASS' if native else 'fail'} {key} "
            f"{row['turns']} turns ({row['seconds']}s)" + (f" ERROR {error}" if error else ""),
            flush=True,
        )
        if abort_sentinel.exists():
            return


def _queue_worker(payload_queue, jsonl: Path, out_dir: Path, **kwargs) -> None:
    """Run episodes from a shared queue until its sentinel is reached.

    Episode durations vary by more than an order of magnitude.  A shared queue
    keeps simulator processes busy without sharing simulator state: every call
    to ``_worker`` still constructs and tears down one independent episode.
    """
    while True:
        if (out_dir / "ABORTED.json").exists():
            return
        payload = payload_queue.get()
        if payload is None:
            return
        _worker([payload], jsonl, out_dir, **kwargs)


def _historical_task_costs(output_root: Path, suite: str) -> dict[tuple[int, int], float]:
    """Estimate episode wall time from completed sibling rollouts.

    This is scheduling evidence only: it never crosses the policy boundary or
    changes an episode.  Using measured duration prevents two known long-tail
    episodes from being stranded serially on one worker while the other
    simulator workers sit idle.
    """
    samples: dict[tuple[int, int], list[float]] = defaultdict(list)
    for summary_path in output_root.glob("*/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if summary.get("suite") != suite:
            continue
        for row in summary.get("records") or []:
            try:
                key = (int(row["task_id"]), int(row["seed"]))
                seconds = float(row["seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            if seconds > 0:
                samples[key].append(seconds)
    return {key: float(median(values)) for key, values in samples.items()}


def _deal(
    values: list[dict], workers: int, costs: dict[tuple[int, int], float] | None = None
) -> list[list[dict]]:
    lanes: list[list[dict]] = [[] for _ in range(max(1, workers))]
    if not costs:
        for index, value in enumerate(values):
            lanes[index % len(lanes)].append(value)
        return [lane for lane in lanes if lane]

    known = tuple(costs.values())
    unseen_cost = float(median(known)) if known else 1.0
    weighted = [
        (
            costs.get((int(value["task_id"]), int(value["seed"])), unseen_cost),
            index,
            value,
        )
        for index, value in enumerate(values)
    ]
    totals = [0.0] * len(lanes)
    # Longest-processing-time scheduling is deterministic and gives a useful
    # approximation to work stealing without sharing MuJoCo environments
    # across processes.
    for cost, _, value in sorted(weighted, key=lambda item: (-item[0], item[1])):
        lane_index = min(range(len(lanes)), key=lambda index: (totals[index], index))
        lanes[lane_index].append(value)
        totals[lane_index] += cost
    return [lane for lane in lanes if lane]


def _payload_key(payload: dict) -> str:
    return f"{payload['suite']}/{payload['task_id']}/seed{payload['seed']}"


def _recorded_keys(jsonl: Path) -> set[str]:
    """Return complete episode keys already written by one worker lane.

    A simulator worker can be terminated below Python (for example by a
    MuJoCo/CUDA native crash), so its JSONL may end after several valid
    episodes.  Those evaluator-owned records are safe to retain; only the
    missing payloads need a fresh process.
    """
    if not jsonl.is_file():
        return set()
    keys: set[str] = set()
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A hard kill can leave one torn final line.  A retry will replace
            # that missing episode with a complete append-only record.
            continue
        key = str(row.get("key") or "")
        # Provider outages are persisted for auditability, but they are not a
        # completed experimental outcome.  A resumed run must replay them.
        if key and bool(row.get("scorable", True)):
            keys.add(key)
    return keys


def _all_recorded_keys(consolidated: Path, worker_jsonls: list[Path]) -> set[str]:
    """Return records from both a completed run and interrupted worker lanes.

    A successful run consolidates ``records.worker*.jsonl`` into
    ``records.jsonl`` and removes the lane files.  Resume must therefore read
    both forms; consulting only lane files would replay every episode in an
    already completed group and violate the one-reset contract.
    """

    paths = [consolidated, *worker_jsonls]
    return set().union(*(_recorded_keys(path) for path in paths))


def _load_unique_records(consolidated: Path, worker_jsonls: list[Path]) -> list[dict]:
    """Load resume records once, rejecting conflicting duplicate keys."""

    by_key: dict[str, dict] = {}
    for path in [consolidated, *worker_jsonls]:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("key") or "")
            if not key:
                continue
            previous = by_key.get(key)
            if previous is not None and previous != row:
                previous_scorable = bool(previous.get("scorable", True))
                row_scorable = bool(row.get("scorable", True))
                if not previous_scorable and row_scorable:
                    by_key[key] = row
                    continue
                if previous_scorable and not row_scorable:
                    continue
                if not previous_scorable and not row_scorable:
                    # Multiple diagnostic attempts may fail differently before
                    # a later resume succeeds.  Keep the newest diagnostic;
                    # neither is an experimental outcome.
                    by_key[key] = row
                    continue
                raise InfrastructureInvariantError(
                    f"conflicting resume records for episode {key}"
                )
            by_key[key] = row
    return list(by_key.values())


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument(
        "--episode-manifest",
        type=Path,
        default=None,
        help=(
            "JSON object with a rows list of public instructions and evaluator-only "
            "predicates. This supports preregistered custom tasks without adding them "
            "to the installed LIBERO package."
        ),
    )
    parser.add_argument("--task-ids", nargs="*", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--worker-retries",
        type=int,
        default=2,
        help=(
            "fresh-process retries for only the missing episodes of a lane "
            "after a native worker crash"
        ),
    )
    parser.add_argument("--turns", type=int, default=45)
    parser.add_argument("--max-pickplace-calls", type=int, default=8)
    parser.add_argument("--max-push-calls", type=int, default=3)
    parser.add_argument("--max-insert-calls", type=int, default=2)
    parser.add_argument("--max-state-calls", type=int, default=5)
    parser.add_argument("--max-stack-calls", type=int, default=2)
    parser.add_argument("--model", default=os.environ.get("RACAP_MODEL", "gpt-5.5"))
    parser.add_argument("--grasp-backend", default="graspnet")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--record-rollouts", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only missing episodes from a compatible interrupted output directory",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-subsample", type=int, default=8)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8000,
        help="shared simulator horizon for the complete long-chain episode",
    )
    parser.add_argument(
        "--episode-timeout-seconds",
        type=float,
        default=1000.0,
        help="comparison-wide wall-time cap for each isolated episode",
    )
    parser.add_argument("--tag", default="full_agent")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full_agent_eval"))
    parser.add_argument(
        "--solution-root",
        type=Path,
        default=None,
        help="isolated candidate repository exposing solution.controller.run_episode",
    )
    parser.add_argument(
        "--episode-context",
        type=Path,
        default=None,
        help="JSON task-id to public semantic context mapping used by skill-isolation stages",
    )
    args = parser.parse_args()

    from racap.envs.register_suites import register_missing_suites
    from racap.benchmark.tasks import build_task_episodes

    register_missing_suites()
    task_ids = tuple(args.task_ids) if args.task_ids else None
    if args.episode_manifest:
        manifest_payload = json.loads(args.episode_manifest.read_text(encoding="utf-8"))
        raw_rows = manifest_payload.get("rows") if isinstance(manifest_payload, dict) else None
        if not isinstance(raw_rows, list):
            raise SystemExit("--episode-manifest must be a JSON object containing rows")
        payloads = []
        for index, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                raise SystemExit(f"episode manifest row {index} is not an object")
            required = {"suite", "task_id", "seed", "instruction", "predicates"}
            missing = sorted(required - set(row))
            if missing:
                raise SystemExit(f"episode manifest row {index} missing {missing}")
            payloads.append(
                {
                    "suite": str(row["suite"]),
                    "task_id": int(row["task_id"]),
                    "seed": int(row["seed"]),
                    "instruction": str(row["instruction"]),
                    "predicates": [list(value) for value in row["predicates"]],
                }
            )
    else:
        episodes = build_task_episodes(
            args.suite, task_ids=task_ids, seeds=tuple(dict.fromkeys(args.seeds))
        )
        payloads = [episode.to_dict() for episode in episodes]
    if not payloads:
        raise SystemExit("no tasks selected")
    out_dir = args.output_dir / args.tag
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"refusing to reuse non-empty output directory: {out_dir}; "
            "choose a new --tag so stale episodes cannot contaminate a run"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    abort_sentinel = out_dir / "ABORTED.json"
    if args.resume and abort_sentinel.exists():
        # An explicit resume is the operator's acknowledgement that the
        # external condition (usually account balance or model routing) was
        # repaired.  Preserve the old reason for audit while allowing the
        # first new network request to test that assumption.
        suffix = int(time.time())
        abort_sentinel.replace(out_dir / f"ABORTED.previous.{suffix}.json")
    episode_context = {}
    if args.episode_context:
        episode_context = json.loads(args.episode_context.read_text(encoding="utf-8"))
        if not isinstance(episode_context, dict):
            raise SystemExit("--episode-context must contain a JSON object")
    resolved_task_ids, resolved_seeds = _resolved_episode_grid(payloads)
    run_manifest = {
        "schema_version": 2,
        "suite": args.suite,
        "episode_manifest": (
            str(args.episode_manifest.resolve()) if args.episode_manifest else None
        ),
        "episode_manifest_sha256": _file_sha256(args.episode_manifest),
        "task_ids": resolved_task_ids,
        "seeds": resolved_seeds,
        "cli_task_ids": list(args.task_ids) if args.task_ids else None,
        "cli_seeds": list(dict.fromkeys(args.seeds)),
        "workers": args.workers,
        "turns": args.turns,
        "max_pickplace_calls": args.max_pickplace_calls,
        "max_push_calls": args.max_push_calls,
        "max_insert_calls": args.max_insert_calls,
        "max_state_calls": args.max_state_calls,
        "max_stack_calls": args.max_stack_calls,
        "model": args.model,
        "grasp_backend": args.grasp_backend,
        "record_rollouts": bool(args.record_rollouts),
        "video_fps": args.video_fps,
        "video_subsample": args.video_subsample,
        "max_steps": args.max_steps,
        "episode_timeout_seconds": args.episode_timeout_seconds,
        "solution_root": str(args.solution_root.resolve()) if args.solution_root else None,
        "solution_controller_sha256": _file_sha256(
            args.solution_root / "solution" / "controller.py" if args.solution_root else None
        ),
        "episode_context_sha256": _file_sha256(args.episode_context),
        "public_episode_context": episode_context,
        "full_episode_controls_by_task_id": {
            str(task_id): _full_episode_controls(task_id, episode_context)
            for task_id in resolved_task_ids
        },
        "racap_git_commit": _git_commit(),
    }
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_manifest != run_manifest:
            raise SystemExit(
                "refusing incompatible --resume: run_manifest.json differs from current arguments"
            )
    else:
        manifest_path.write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    jsonl = out_dir / "records.jsonl"
    print(
        f"running full agent on {len(payloads)} tasks, {args.workers} workers, "
        f"{args.turns} turns -> {out_dir}",
        flush=True,
    )
    started = time.time()
    historical_costs = _historical_task_costs(args.output_dir, args.suite)
    known_costs = tuple(historical_costs.values())
    unseen_cost = float(median(known_costs)) if known_costs else 1.0
    # Queue expensive episodes first.  Unlike static lanes, workers that finish
    # an unexpectedly cheap episode can immediately help with the remaining
    # work, while each episode still owns an isolated simulator lifecycle.
    payloads.sort(
        key=lambda payload: (
            -historical_costs.get(
                (int(payload["task_id"]), int(payload["seed"])), unseen_cost
            ),
            int(payload["task_id"]),
            int(payload["seed"]),
        )
    )
    context = mp.get_context("spawn")
    worker_count = min(max(1, args.workers), len(payloads))
    worker_jsonls = [out_dir / f"records.worker{index}.jsonl" for index in range(worker_count)]
    already_recorded = _all_recorded_keys(jsonl, worker_jsonls)
    pending_payloads = [
        payload for payload in payloads if _payload_key(payload) not in already_recorded
    ]
    if already_recorded:
        print(
            f"resume state: {len(already_recorded)}/{len(payloads)} episodes already recorded; "
            f"{len(pending_payloads)} pending",
            flush=True,
        )
    worker_kwargs = [
        {
            "turns": args.turns,
            "max_pickplace_calls": args.max_pickplace_calls,
            "max_push_calls": args.max_push_calls,
            "max_insert_calls": args.max_insert_calls,
            "max_state_calls": args.max_state_calls,
            "max_stack_calls": args.max_stack_calls,
            "model": args.model,
            "grasp_backend": args.grasp_backend,
            "verbose": args.verbose,
            "worker_index": index,
            "record_rollouts": args.record_rollouts,
            "video_fps": args.video_fps,
            "video_subsample": args.video_subsample,
            "max_steps": args.max_steps,
            "episode_timeout_seconds": args.episode_timeout_seconds,
            "solution_root": str(args.solution_root.resolve()) if args.solution_root else "",
            "episode_context": episode_context,
        }
        for index in range(worker_count)
    ]

    def make_process(index: int, payload_queue, *, retry: int = 0):
        suffix = f"-retry{retry}" if retry else ""
        return context.Process(
            target=_queue_worker,
            args=(payload_queue, worker_jsonls[index], out_dir),
            kwargs=worker_kwargs[index],
            name=f"eval-worker-{index}{suffix}",
        )

    def run_batch(batch: list[dict], *, retry: int = 0) -> list[mp.Process]:
        payload_queue = context.Queue()
        for payload in batch:
            payload_queue.put(payload)
        count = min(worker_count, len(batch))
        for _ in range(count):
            payload_queue.put(None)
        processes = [make_process(index, payload_queue, retry=retry) for index in range(count)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        payload_queue.close()
        return processes

    processes = run_batch(pending_payloads) if pending_payloads else []
    if abort_sentinel.exists():
        completed = _all_recorded_keys(jsonl, worker_jsonls)
        abort = json.loads(abort_sentinel.read_text(encoding="utf-8"))
        exit_code = _abort_exit_code(abort)
        checkpoint = {
            "schema_version": 1,
            "status": "paused" if exit_code == 75 else "infrastructure_failed",
            "completed": len(completed),
            "expected": len(payloads),
            "missing_keys": [
                _payload_key(payload)
                for payload in payloads
                if _payload_key(payload) not in completed
            ],
            "abort": abort,
        }
        (out_dir / "checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise SystemExit(exit_code)
    missing: list[dict] = []
    retry = 0
    while True:
        recorded = _all_recorded_keys(jsonl, worker_jsonls)
        missing = [payload for payload in payloads if _payload_key(payload) not in recorded]
        if not missing or retry >= args.worker_retries:
            break
        retry += 1
        exits = ", ".join(f"{p.name}={p.exitcode}" for p in processes)
        print(
            f"  [supervisor] worker exits [{exits}]; retrying {len(missing)} "
            f"missing episode(s) in fresh processes ({retry}/{args.worker_retries})",
            flush=True,
        )
        processes = run_batch(missing, retry=retry)
        if abort_sentinel.exists():
            completed = _all_recorded_keys(jsonl, worker_jsonls)
            abort = json.loads(abort_sentinel.read_text(encoding="utf-8"))
            exit_code = _abort_exit_code(abort)
            checkpoint = {
                "schema_version": 1,
                "status": "paused" if exit_code == 75 else "infrastructure_failed",
                "completed": len(completed),
                "expected": len(payloads),
                "missing_keys": [
                    _payload_key(payload)
                    for payload in payloads
                    if _payload_key(payload) not in completed
                ],
                "abort": abort,
            }
            (out_dir / "checkpoint.json").write_text(
                json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            raise SystemExit(exit_code)
    if missing:
        exits = ", ".join(f"{p.name}={p.exitcode}" for p in processes)
        raise SystemExit(
            f"evaluation workers failed after retries: exits=[{exits}] missing="
            + ",".join(_payload_key(payload) for payload in missing)
        )

    rows = _load_unique_records(jsonl, worker_jsonls)
    expected = len(payloads)
    if len(rows) != expected:
        raise SystemExit(f"incomplete evaluation: received {len(rows)}/{expected} episode records")
    rows.sort(key=lambda row: (int(row["task_id"]), int(row["seed"])))
    jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows)
    )
    for worker_jsonl in worker_jsonls:
        worker_jsonl.unlink(missing_ok=True)
    if args.record_rollouts:
        from racap.agent.artifacts import write_rollout_index

        write_rollout_index(out_dir, rows)
    native = sum(bool(row["native_success"]) for row in rows)
    claimed = sum(bool(row["agent_success"]) for row in rows)
    by_predicate: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "success": 0})
    for row in rows:
        for predicate in set(row.get("predicate_types") or []):
            by_predicate[predicate]["n"] += 1
            by_predicate[predicate]["success"] += int(bool(row["native_success"]))
    summary = {
        "tag": args.tag,
        "suite": args.suite,
        "n_episodes": len(rows),
        "native_success": native,
        "native_success_rate": round(native / len(rows), 4) if rows else 0.0,
        "agent_claimed_rate": round(claimed / len(rows), 4) if rows else 0.0,
        "mean_turns": round(sum(row["turns"] for row in rows) / len(rows), 2) if rows else 0.0,
        "minutes": round((time.time() - started) / 60, 1),
        "by_predicate": dict(by_predicate),
        "record_rollouts": bool(args.record_rollouts),
        "simulator_horizon": args.max_steps,
        "controller": "candidate" if args.solution_root else "phase1_core",
        "solution_root": str(args.solution_root.resolve()) if args.solution_root else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps({**summary, "records": rows}, indent=2, ensure_ascii=False)
    )
    print(
        f"\n{args.tag}: native {native}/{len(rows)} = "
        f"{summary['native_success_rate']:.1%}; claimed "
        f"{summary['agent_claimed_rate']:.1%} ({summary['minutes']} min)\n"
        f"-> {out_dir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
