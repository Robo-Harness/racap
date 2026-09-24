#!/usr/bin/env python3
"""Run the preregistered seven-object anytime manipulation task.

Each method receives one uninterrupted trajectory per initial state. Native
predicate checkpoints are evaluator-only and let analysis read completion at
5 and 10 minutes from the same trajectory; the final result runs until the
agent stops or the common 8000-step simulator horizon is reached.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

# Keep the documented direct invocation
# ``python experiments/controlled_comparison/run_long_horizon.py`` portable.
# The fallback import below loads ``run_libero`` as a local module, while that
# module intentionally uses absolute ``experiments.*`` imports.  A clean shell
# therefore needs the repository root on ``sys.path`` before either import.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .run_libero import (
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _controlled_libero_env,
        _controlled_pythonpath,
        _controlled_rats_libero_env,
        _experiment_import_roots,
        _generated_episode_evidence,
        _jsonl_rows,
        _nonempty_files,
        _public_model_route,
        _racap_group_evidence,
        _rats_eval_budget_args,
        _run_logged,
        _require_frozen_rats_seal,
        _service_client_env,
        _sha256,
        _shared_api_server_bundle,
        _tree_sha256,
        _write_json,
    )
except ImportError:  # direct ``python path/to/script.py`` execution
    from run_libero import (
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _controlled_libero_env,
        _controlled_pythonpath,
        _controlled_rats_libero_env,
        _experiment_import_roots,
        _generated_episode_evidence,
        _jsonl_rows,
        _nonempty_files,
        _public_model_route,
        _racap_group_evidence,
        _rats_eval_budget_args,
        _run_logged,
        _require_frozen_rats_seal,
        _service_client_env,
        _sha256,
        _shared_api_server_bundle,
        _tree_sha256,
        _write_json,
    )

HERE = Path(__file__).resolve().parent
ASSET = HERE / "assets" / "libero_long_all_to_basket"
BDDL = next((ASSET / "bddl_files" / "libero_long_all_to_basket").glob("*.bddl"))
INIT = Path(os.environ.get(
    "RACAP_LONG_HORIZON_INIT",
    str(ASSET / "init_files" / "libero_long_all_to_basket"
        / "LIVING_ROOM_SCENE2_put_all_tabletop_objects_in_the_basket.pruned_init"),
))
EPISODES = ASSET / "episodes.json"
PUBLIC_EPISODE_CONTEXT = ASSET / "public_episode_context.json"


def _materialize_isolated_rats_memory(
    source: Path,
    task_root: Path,
) -> tuple[Path, str]:
    """Give one evaluation episode an immutable-by-audit memory input.

    Upstream RATS currently merges an external failure memory into a run-local
    store before recording new failures.  The controlled harness should not
    rely on that implementation detail: concurrent seeds must never share a
    writable input.  Each seed therefore reads an identical private copy and
    the caller verifies its hash again after the child exits.
    """

    source = source.resolve()
    expected = _tree_sha256(source)
    if not expected:
        raise FileNotFoundError(f"empty or missing frozen RATS memory: {source}")
    destination = task_root / "frozen_failure_memory_input"
    if destination.exists():
        observed = _tree_sha256(destination)
        if observed != expected:
            raise RuntimeError(
                "refusing incompatible per-seed RATS memory resume: "
                f"{observed} != {expected}"
            )
    else:
        shutil.copytree(source, destination)
    return destination, expected


def _common_env(seed: int, task_root: Path, method: str, model: str) -> dict[str, str]:
    env = os.environ.copy()
    prefix = "CAPX" if method == "capx" else "RATS"
    env.update(
        {
            "PYTHONPATH": _controlled_pythonpath(),
            **_controlled_libero_env(),
            **_controlled_rats_libero_env(),
            "CONTROLLED_LIBERO_BDDL_FILE": str(BDDL.resolve()),
            "CONTROLLED_LIBERO_INIT_FILE": str(INIT.resolve()),
            f"{prefix}_EPISODE_KEY": f"libero_long_all_to_basket/seed{seed}",
            f"{prefix}_SIM_EPISODE_TELEMETRY_PATH": str(task_root / "sim_episodes.jsonl"),
            f"{prefix}_NATIVE_TELEMETRY_PATH": str(task_root / "native_states.jsonl"),
            f"{prefix}_SEED_OFFSET": str(seed),
            # This protocol defines the final checkpoint by agent stop or the
            # 8000-step simulator horizon.  Disable the generated runners'
            # ordinary 1000-second single-episode cap for this experiment only.
            f"{prefix}_TRIAL_TIMEOUT_SECONDS": "0",
            **_service_client_env(),
            "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    if method == "capx":
        env.update(
            {
                "CAPX_REGISTERED_EPISODE_KEY": (
                    f"libero_long_all_to_basket/seed{seed}"
                ),
                "OPENAI_API_KEY": env.get("RACAP_VAPI_KEY", ""),
                "CAPX_LLM_TELEMETRY_PATH": str(task_root / "llm_calls.jsonl"),
                "CAPX_RUNTIME_VLM_URL": env.get("RACAP_VAPI_BASE", "").rstrip("/")
                + "/chat/completions",
                "CAPX_RUNTIME_VLM_MODEL": model,
                "CAPX_RUNTIME_VLM_KEY": env.get("RACAP_VAPI_KEY", ""),
            }
        )
    else:
        env["RATS_REGISTERED_EPISODE_KEY"] = f"libero_long_all_to_basket/seed{seed}"
        env.update(
            {
                "RATS_VAPI_URL": env.get("RACAP_VAPI_BASE", "").rstrip("/")
                + "/chat/completions",
                "RATS_VAPI_KEY": env.get("RACAP_VAPI_KEY", ""),
                "RATS_LLM_MODEL": model,
                "RATS_RUNTIME_VLM_MODEL": model,
                "RATS_LLM_FALLBACK": "0",
                "RATS_VERIFY_STEP_MODE": "strict",
                "RATS_VERIFIER_STRICT_BENCHMARK": "1",
                "RATS_AGENT_IO_DIR": str(task_root / "agent_io"),
            }
        )
    return env


def _generated_job(
    method: str,
    seed: int,
    args: argparse.Namespace,
    method_root: Path,
    stop_event: threading.Event,
) -> dict[str, Any]:
    task_root = method_root / f"seed_{seed:02d}"
    complete = task_root / "COMPLETE.json"
    if complete.exists():
        return {"seed": seed, "returncode": 0, "skipped": True}
    task_root.mkdir(parents=True, exist_ok=True)
    env = _common_env(seed, task_root, method, args.model)
    env["RACAP_VLM_MAX_CONCURRENCY"] = str(args.workers)
    base = env.get("RACAP_VAPI_BASE", "").rstrip("/")
    isolated_memory: Path | None = None
    isolated_memory_before = ""
    if method == "capx":
        config = copy.deepcopy(
            yaml.safe_load((HERE / "configs" / "capx_libero.yaml").read_text())
        )
        config["env"]["cfg"]["low_level"]["suite_name"] = "libero_10"
        config["env"]["cfg"]["low_level"]["task_id"] = 5
        config["output_dir"] = str(task_root / "artifacts")
        config["trials"] = 1
        config["multi_turn_limit"] = 25
        config_path = task_root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        argv = [
            str(args.python),
            "-m",
            "capx.envs.launch",
            "--config-path",
            str(config_path),
            "--server-url",
            f"{base}/chat/completions",
            "--model",
            args.model,
            "--temperature",
            "0",
            "--max-tokens",
            "8192",
            "--reasoning-effort",
            "medium",
            "--visual-differencing-model",
            args.model,
            "--visual-differencing-model-server-url",
            f"{base}/chat/completions",
            "--total-trials",
            "1",
            "--num-workers",
            "1",
            "--record-video",
            "True",
            "--output-dir",
            str(task_root / "artifacts"),
        ]
        cwd = args.rats_root / "capx-baseline"
    else:
        library = (
            args.rats_root / "skill_library" / "libero_nonpriv_skills.json"
            if method == "rats_base"
            else args.rats_library
        )
        if library is None or not library.is_file():
            raise FileNotFoundError(f"missing RATS library: {library}")
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
            "libero_10",
            "--libero-task",
            "5",
            "--iterations",
            "1",
            "--fixed-task",
            "--skill-library",
            str(library),
            *_rats_eval_budget_args(turns=25),
            "--output-dir",
            str(task_root / "artifacts"),
            "--log-agent-io",
        ]
        if method == "rats_base":
            argv.extend(["--no-skill-reuse", "--no-failure-memory"])
        elif args.rats_memory:
            isolated_memory, isolated_memory_before = _materialize_isolated_rats_memory(
                args.rats_memory,
                task_root,
            )
            argv.extend(["--failure-memory-path", str(isolated_memory)])
        cwd = args.rats_root
    result = _run_logged(
        argv,
        cwd=cwd,
        env=env,
        log_path=task_root / "run.log",
        stop_event=stop_event,
    )
    if isolated_memory is not None:
        isolated_memory_after = _tree_sha256(isolated_memory)
        memory_audit = {
            "schema_version": 1,
            "source": str(args.rats_memory.resolve()),
            "isolated_input": str(isolated_memory.resolve()),
            "before": isolated_memory_before,
            "after": isolated_memory_after,
            "unchanged": isolated_memory_before == isolated_memory_after,
        }
        _write_json(task_root / "frozen_input_audit.json", memory_audit)
        if not memory_audit["unchanged"]:
            result["returncode"] = 3
            result["frozen_input_mutation"] = memory_audit
    if result["returncode"] == 0:
        evidence = _generated_episode_evidence(
            task_root,
            method,
            expected_episode_keys={f"libero_long_all_to_basket/seed{seed}"},
            maximum_registered_resets=1,
            expected_init_state_indices={
                f"libero_long_all_to_basket/seed{seed}": seed
            },
        )
        run_log = (task_root / "run.log").read_text(
            encoding="utf-8", errors="replace"
        )
        wall_timeout_detected = (
            "exceeded 1000 seconds" in run_log
            or "timed out after 1000 seconds" in run_log
        )
        evidence["episode_wall_timeout_detected"] = wall_timeout_detected
        if wall_timeout_detected:
            evidence["complete"] = False
            evidence.setdefault("missing", []).append(
                "protocol_forbids_episode_wall_timeout"
            )
        _write_json(task_root / "EVIDENCE.json", evidence)
        if evidence["complete"]:
            _write_json(complete, {"seed": seed, "evidence": evidence, **result})
        else:
            result["returncode"] = 2
            result["missing_evidence"] = evidence["missing"]
    return {"seed": seed, **result}


def _run_generated(method: str, args: argparse.Namespace, method_root: Path) -> int:
    stop_event = threading.Event()
    results: list[dict[str, Any]] = []
    frozen_before = None
    if method == "rats_90":
        if args.rats_library is None or not args.rats_library.is_file():
            raise SystemExit("rats_90 requires the frozen skill library")
        if args.rats_memory is None or not args.rats_memory.is_dir():
            raise SystemExit("rats_90 requires the frozen failure-memory directory")
        frozen_before = {
            "library_sha256": _sha256(args.rats_library),
            "memory_tree_sha256": _tree_sha256(args.rats_memory),
        }
    with _shared_api_server_bundle(args, method_root):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, 5)
        ) as pool:
            futures = {
                pool.submit(_generated_job, method, seed, args, method_root, stop_event): seed
                for seed in range(5)
            }
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                results.append(row)
                print(
                    f"[{method}] long seed={row['seed']} rc={row['returncode']}",
                    flush=True,
                )
    _write_json(method_root / "job_results.json", results)
    if frozen_before is not None:
        frozen_after = {
            "library_sha256": _sha256(args.rats_library),
            "memory_tree_sha256": _tree_sha256(args.rats_memory),
        }
        frozen_audit = {
            "schema_version": 1,
            "before": frozen_before,
            "after": frozen_after,
            "unchanged": frozen_before == frozen_after,
        }
        _write_json(method_root / "frozen_artifact_audit.json", frozen_audit)
        if not frozen_audit["unchanged"]:
            _write_json(method_root / "FAILURES.json", frozen_audit)
            return 1
    if any(row.get("quota_failure") for row in results):
        _write_json(method_root / "ABORTED.json", {"reason": "quota", "jobs": results})
        return 75
    if any(row["returncode"] != 0 for row in results):
        _write_json(method_root / "FAILURES.json", results)
        return 1
    _write_json(method_root / "COMPLETE.json", {"method": method, "seeds": 5})
    return 0


def _run_racap(method: str, args: argparse.Namespace, method_root: Path) -> int:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": _controlled_pythonpath(),
            **_controlled_libero_env(),
            "CONTROLLED_LIBERO_BDDL_FILE": str(BDDL.resolve()),
            "CONTROLLED_LIBERO_INIT_FILE": str(INIT.resolve()),
            "RACAP_LLM_CACHE": "0",
            "RACAP_MODEL": args.model,
            "RACAP_VLM_MAX_CONCURRENCY": str(args.workers),
            "RACAP_SAM3_URL": _service_client_env()["SAM3_SERVICE_URL"],
            "RACAP_GRASPNET_URL": _service_client_env()["GRASPNET_SERVICE_URL"],
            "RACAP_PYROKI_URL": _service_client_env()["PYROKI_SERVICE_URL"],
            "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    tag = "libero_long_all_to_basket"
    out = method_root / "artifacts"
    argv = [
        str(args.python),
        str(ROOT / "scripts" / "eval_full_agent.py"),
        "--suite",
        "libero_10",
        "--episode-manifest",
        str(EPISODES),
        "--episode-context",
        str(PUBLIC_EPISODE_CONTEXT),
        "--workers",
        str(min(args.workers, 5)),
        "--turns",
        "45",
        "--max-pickplace-calls",
        "8",
        "--max-push-calls",
        "3",
        "--max-insert-calls",
        "2",
        "--max-state-calls",
        "5",
        "--max-stack-calls",
        "2",
        "--model",
        args.model,
        "--max-steps",
        "8000",
        "--episode-timeout-seconds",
        "0",
        "--record-rollouts",
        "--tag",
        tag,
        "--output-dir",
        str(out),
    ]
    phase = "phase2" if method == "racap_phase2" else "phase1"
    argv.extend(["--solution-root", str(ROOT / "policies" / phase)])
    if (out / tag / "run_manifest.json").exists():
        argv.append("--resume")
    result = _run_logged(
        argv,
        cwd=ROOT,
        env=env,
        log_path=method_root / "run.log",
    )
    _write_json(method_root / "job_results.json", [result])
    if result["quota_failure"] or result["returncode"] == 75:
        _write_json(method_root / "ABORTED.json", result)
        return 75
    if result["returncode"] != 0:
        _write_json(method_root / "FAILURES.json", result)
        return 1
    evidence = _racap_group_evidence(
        out / tag,
        suite="libero_10",
        task_ids={5},
        seeds=set(range(5)),
    )
    # The anytime curves require native state after every public tool boundary,
    # not merely start/end labels.  Candidate code receives no value from this
    # write-only callback.  Fail closed when a runtime adapter accidentally
    # suppresses the telemetry, as that would make 5/10-minute completion
    # impossible to reconstruct without privileged post-hoc guessing.
    native_rows = _jsonl_rows(_nonempty_files(out / tag, "native_states*.jsonl"))
    record_rows = _jsonl_rows([out / tag / "records.jsonl"])
    expected_checkpoint_counts = {
        str(row.get("key") or ""): len(row.get("steps") or [])
        for row in record_rows
    }
    checkpoint_counts: dict[str, int] = {}
    for seed in range(5):
        key = f"libero_10/5/seed{seed}"
        checkpoint_counts[key] = sum(
            str(row.get("episode_key") or "") == key
            and str(row.get("event") or "").startswith("after_tool:")
            for row in native_rows
        )
    missing_checkpoints = sorted(
        key
        for key, count in checkpoint_counts.items()
        if count <= 0 or count != expected_checkpoint_counts.get(key)
    )
    evidence["anytime_checkpoint_counts"] = checkpoint_counts
    evidence["expected_anytime_checkpoint_counts"] = expected_checkpoint_counts
    evidence["missing_anytime_checkpoints"] = missing_checkpoints
    if missing_checkpoints:
        evidence["complete"] = False
        evidence.setdefault("artifact_errors", {})["anytime_native_telemetry"] = [
            "missing_after_tool_checkpoints:" + ",".join(missing_checkpoints)
        ]
    _write_json(out / tag / "EVIDENCE.json", evidence)
    result["evidence"] = evidence
    _write_json(method_root / "job_results.json", [result])
    if not evidence["complete"]:
        result["returncode"] = 2
        _write_json(method_root / "FAILURES.json", result)
        _write_json(method_root / "job_results.json", [result])
        return 1
    _write_json(method_root / "COMPLETE.json", {"method": method, "seeds": 5})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["capx", "rats_base", "rats_90", "racap_phase1", "racap_phase2"],
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--rats-root", type=Path, default=DEFAULT_RATS_ROOT)
    parser.add_argument("--rats-library", type=Path, default=None)
    parser.add_argument("--rats-memory", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "long_horizon",
    )
    args = parser.parse_args()
    if not INIT.is_file():
        raise SystemExit(
            "Missing external initial-state asset. Set RACAP_LONG_HORIZON_INIT or "
            "run scripts/prepare_assets.py --source /path/to/authorized-assets."
        )
    if args.workers != 10:
        raise SystemExit("the evaluation protocol requires exactly ten workers")
    if not os.environ.get("RACAP_VAPI_KEY") or not os.environ.get("RACAP_VAPI_BASE"):
        raise SystemExit("RACAP_VAPI_KEY and RACAP_VAPI_BASE must be configured")
    if args.method == "rats_90":
        if args.rats_library is None or not args.rats_library.is_file():
            raise SystemExit("rats_90 requires the frozen skill library")
        if args.rats_memory is None or not args.rats_memory.is_dir():
            raise SystemExit("rats_90 requires the frozen failure-memory directory")
        _require_frozen_rats_seal(args.rats_library, args.rats_memory)
    method_root = args.output_root / args.method
    method_root.mkdir(parents=True, exist_ok=True)
    import_roots = _experiment_import_roots(
        args.python,
        verify_rats_registry=args.method in {"rats_base", "rats_90"},
    )
    manifest = {
        "schema_version": 1,
        "method": args.method,
        "model": args.model,
        "hosted_model_route": _public_model_route(os.environ["RACAP_VAPI_BASE"]),
        "python_import_roots": import_roots,
        "simulator_horizon": 8000,
        "registered_initial_states_per_episode": 1,
        "post_action_environment_resets": 0,
        "generated_continuous_turns": 25,
        "rats_policy_self_check_repairs": 0,
        "shared_service_urls": _service_client_env(),
        "workers": args.workers,
        "effective_episode_workers": min(args.workers, 5),
        "seeds": list(range(5)),
        "bddl_sha256": _sha256(BDDL),
        "init_sha256": _sha256(INIT),
        "episodes_sha256": _sha256(EPISODES),
        "public_episode_context_sha256": _sha256(PUBLIC_EPISODE_CONTEXT),
        "objective_mode": "independent_anytime",
        "anytime_minutes": [5, 10],
        "final_stop_condition": "agent_stop_or_simulator_horizon_8000",
        "episode_wall_timeout_seconds": 0,
        "rats_library": str(args.rats_library.resolve()) if args.rats_library else None,
        "rats_library_sha256": _sha256(args.rats_library) if args.rats_library else None,
        "rats_memory": str(args.rats_memory.resolve()) if args.rats_memory else None,
        "rats_memory_tree_sha256": (
            _tree_sha256(args.rats_memory) if args.rats_memory else None
        ),
    }
    path = method_root / "run_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise SystemExit("refusing incompatible long-horizon resume")
    if not path.exists():
        _write_json(path, manifest)
    if args.method in {"capx", "rats_base", "rats_90"}:
        return _run_generated(args.method, args, method_root)
    with _shared_api_server_bundle(args, method_root):
        return _run_racap(args.method, args, method_root)


if __name__ == "__main__":
    raise SystemExit(main())
