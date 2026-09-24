#!/usr/bin/env python3
"""Evolve RACaP on a leakage-safe Robosuite development pool."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .run_libero import (
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _service_client_env,
        _shared_api_server_bundle,
        _tree_sha256,
        _write_json,
    )
    from .run_robosuite import _validate_robosuite_source
except ImportError:
    from run_libero import (  # type: ignore[no-redef]
        DEFAULT_PYTHON,
        DEFAULT_RATS_ROOT,
        _service_client_env,
        _shared_api_server_bundle,
        _tree_sha256,
        _write_json,
    )
    from run_robosuite import _validate_robosuite_source  # type: ignore[no-redef]

from evolution.harness.coder import CodingAgent
from evolution.harness.critic import VisualCritic
from evolution.harness.orchestrator import EvolutionOrchestrator
from evolution.harness.robosuite import (
    DEVELOPMENT_SEEDS,
    DEVELOPMENT_TASKS,
    EVALUATION_SEEDS,
    HELDOUT_TASKS,
    RobosuiteEvaluationRunner,
    development_curriculum,
)
from evolution.harness.scheduler import CurriculumScheduler, SchedulerConfig
from racap.backends.llm import LLMQuotaError


def _export_commit(repository: Path, commit: str, destination: Path) -> None:
    """Materialize the winning candidate as a standalone frozen artifact."""
    if destination.exists():
        prior = destination / "FROZEN_COMMIT"
        if prior.is_file() and prior.read_text(encoding="utf-8").strip() == commit:
            return
        raise RuntimeError(f"refusing to overwrite another frozen artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="racap-rs-export-") as raw_tmp:
        temporary = Path(raw_tmp)
        archive = temporary / "champion.tar"
        with archive.open("wb") as handle:
            result = subprocess.run(
                ["git", "-C", str(repository), "archive", "--format=tar", commit],
                stdout=handle,
                stderr=subprocess.PIPE,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        unpacked = temporary / "tree"
        unpacked.mkdir()
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                    raise RuntimeError(f"unsafe path in candidate archive: {member.name}")
            bundle.extractall(unpacked)
        shutil.copytree(unpacked, destination)
    (destination / "FROZEN_COMMIT").write_text(commit + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=ROOT / "outputs" / "evolution" / "robosuite_racap_rs_15",
    )
    parser.add_argument(
        "--seed-root", type=Path, default=ROOT / "policies" / "phase2"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help=(
            "scheduler patience parameter; the registered physical budget is "
            "controlled by --runtime-candidates and failed code proposals do not "
            "consume a simulator-tested candidate"
        ),
    )
    parser.add_argument("--runtime-candidates", type=int, default=15)
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Matched Robosuite development concurrency for RACaP and RATS.",
    )
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--coder-model", default="gpt-5.5")
    parser.add_argument("--critic-model", default="gpt-5.5")
    parser.add_argument("--runtime-model", default="gpt-5.5")
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help=(
            "Administrative recovery mode: evaluate exactly the already checkpointed "
            "pending candidate, then exit without generating a new proposal. This does "
            "not change the registered 15-candidate physical budget."
        ),
    )
    parser.add_argument("--critic-limit", type=int, default=8)
    parser.add_argument("--implementation-repairs", type=int, default=2)
    parser.add_argument(
        "--smoke-episodes",
        type=int,
        default=0,
        help="Registered comparison uses zero extra smoke episodes; the 15-episode paired sweep is runtime validation.",
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--rats-root", type=Path, default=DEFAULT_RATS_ROOT)
    parser.add_argument("--robosuite-root", type=Path, default=None)
    args = parser.parse_args()
    if args.iterations != 15 or args.runtime_candidates != 15:
        raise SystemExit("the registered Robosuite comparison requires 15 candidate iterations")
    if args.smoke_episodes != 0:
        raise SystemExit("matched physical budget requires --smoke-episodes 0")
    if not os.environ.get("RACAP_VAPI_KEY", "").strip():
        raise SystemExit(
            "RACAP_VAPI_KEY is unset; source configs/env.sh before launching so "
            "credential failures cannot be recorded as policy rollouts"
        )
    if not os.environ.get("RACAP_VAPI_BASE", "").strip():
        raise SystemExit(
            "RACAP_VAPI_BASE is unset; source configs/env.sh before launching"
        )

    experiment_root = args.experiment_root.resolve()
    seed_root = args.seed_root.resolve()
    rats_root = args.rats_root.resolve()
    capx_root = rats_root / "capx-baseline"
    robosuite_root = (
        args.robosuite_root.resolve()
        if args.robosuite_root
        else rats_root / "rats" / "third_party" / "robosuite"
    )
    # Keep the virtual-environment launcher path intact. ``Path.resolve()``
    # follows ``bin/python`` to the system interpreter and loses site-packages.
    args.python = Path(os.path.abspath(args.python))
    args.rats_root = rats_root
    imports = _validate_robosuite_source(args.python, robosuite_root, rats_root=rats_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    # Parent-process proposal, implementation, and critic calls are separate
    # from episode runtime telemetry.  Persist them so the final workbook can
    # report the algorithm-native coding budget instead of inferring it from
    # the number of candidates.
    os.environ["RACAP_LLM_TELEMETRY_PATH"] = str(
        experiment_root / "evolution_llm_calls.jsonl"
    )
    os.environ["RACAP_EPISODE_KEY"] = "robosuite_evolution/coding_and_critique"
    manifest = {
        "schema_version": 1,
        "method": "racap_rs_evolved",
        "start": "frozen_libero90_champion",
        "seed_root": str(seed_root),
        "seed_tree_sha256": _tree_sha256(seed_root),
        "development_tasks": list(DEVELOPMENT_TASKS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "development_episodes_per_full_sweep": len(DEVELOPMENT_TASKS)
        * len(DEVELOPMENT_SEEDS),
        "heldout_task_families": list(HELDOUT_TASKS),
        "sealed_evaluation_seeds": list(EVALUATION_SEEDS),
        "runtime_candidate_budget": int(args.runtime_candidates),
        "proposal_attempt_cap": None,
        "scheduler_patience": int(args.iterations),
        "physical_budget": {
            "baseline_sweeps": 1,
            "candidate_sweeps": 15,
            "episodes_per_sweep": len(DEVELOPMENT_TASKS) * len(DEVELOPMENT_SEEDS),
            "extra_smoke_episodes": 0,
            "total_development_episodes": 16
            * len(DEVELOPMENT_TASKS)
            * len(DEVELOPMENT_SEEDS),
        },
        "workers": min(max(1, int(args.workers)), 5),
        "coder_model": str(args.coder_model),
        "critic_model": str(args.critic_model),
        "runtime_model": str(args.runtime_model),
        "robosuite_source": str(robosuite_root),
        "robosuite_imports": imports,
        "promotion": "strictly higher paired development native success; ties only log efficiency",
        "oracle_visible_to_candidate": False,
        "official_evaluation_visible_during_development": False,
    }
    manifest_path = experiment_root / "protocol_manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior != manifest:
            raise SystemExit("refusing to resume with a changed Robosuite evolution protocol")
    else:
        _write_json(manifest_path, manifest)

    os.environ.update(_service_client_env())
    curriculum = development_curriculum()
    runner = RobosuiteEvaluationRunner(
        ROOT,
        experiment_root,
        python=args.python,
        robosuite_root=robosuite_root,
        rats_root=rats_root,
        capx_root=capx_root,
        workers=args.workers,
        model=args.runtime_model,
        max_steps=args.max_steps,
        record_rollouts=True,
    )
    orchestrator = EvolutionOrchestrator(
        ROOT,
        experiment_root,
        curriculum,
        runner,
        CodingAgent(args.coder_model),
        VisualCritic(args.critic_model),
        critic_limit=args.critic_limit,
        implementation_repairs=args.implementation_repairs,
        smoke_episodes=args.smoke_episodes,
        seed_root=seed_root,
        scheduler=CurriculumScheduler(
            curriculum,
            SchedulerConfig(min_visits=0, patience=max(1, args.iterations), exploration=0.0),
        ),
    )
    try:
        runtime_candidate_target = int(args.runtime_candidates)
        if args.pending_only:
            if not orchestrator.events.state_path.is_file():
                raise SystemExit("--pending-only requires an existing evolution state")
            recovery_state = orchestrator.events.load_state()
            if not recovery_state.pending_candidate:
                raise SystemExit("--pending-only requires a checkpointed pending candidate")
            runtime_candidate_target = int(recovery_state.runtime_candidates) + 1
        with _shared_api_server_bundle(
            args, experiment_root, owner="racap_robosuite_evolution_parent"
        ):
            state = orchestrator.run(
                "rs_cross_embodiment",
                args.iterations,
                runtime_candidates=runtime_candidate_target,
            )
    except LLMQuotaError:
        state = orchestrator.events.load_state()
        pending = dict(state.pending_candidate or {})
        rollout_tag = str(pending.get("rollout_tag") or "")
        rollout_checkpoint = experiment_root / "rollouts" / rollout_tag / "checkpoint.json"
        checkpoint = {}
        if rollout_checkpoint.is_file():
            try:
                checkpoint = json.loads(rollout_checkpoint.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                checkpoint = {}
        _write_json(
            experiment_root / "PAUSED_QUOTA.json",
            {
                "schema_version": 1,
                "status": "paused_quota_checkpoint",
                "paused_at_unix": time.time(),
                "champion_commit": state.champion_commit,
                "runtime_candidates_completed": state.runtime_candidates,
                "runtime_candidate_target": int(args.runtime_candidates),
                "pending_candidate_commit": pending.get("candidate_commit"),
                "pending_rollout_tag": rollout_tag or None,
                "completed_scorable_episodes": checkpoint.get("completed"),
                "retry_after_recharge": checkpoint.get("missing_keys") or [],
                "resume_semantics": (
                    "Resume the exact pending commit and rollout tag; completed "
                    "scorable episodes are reused and quota-invalid episodes retried."
                ),
            },
        )
        return 75
    if args.pending_only:
        _write_json(
            experiment_root / "PENDING_REVALIDATION_COMPLETE.json",
            {
                "schema_version": 1,
                "status": "pending_candidate_revalidated",
                "runtime_candidates_completed": state.runtime_candidates,
                "registered_runtime_candidate_target": int(args.runtime_candidates),
                "champion_commit": state.champion_commit,
                "completed_at_unix": time.time(),
            },
        )
        return 0
    frozen = experiment_root / "frozen_champion"
    _export_commit(
        experiment_root / "solution_git" / "repository",
        state.champion_commit,
        frozen,
    )
    _write_json(
        experiment_root / "frozen_champion_manifest.json",
        {
            "commit": state.champion_commit,
            "tree_sha256": _tree_sha256(frozen),
            "source_experiment": str(experiment_root),
            "development_tasks": list(DEVELOPMENT_TASKS),
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "heldout_task_families": list(HELDOUT_TASKS),
            "sealed_evaluation_seeds": list(EVALUATION_SEEDS),
        },
    )
    _write_json(experiment_root / "FINAL_STATE.json", state.to_dict())
    _write_json(
        experiment_root / "COMPLETE.json",
        {
            "schema_version": 1,
            "method": "RACaP-RS-evolved",
            "runtime_candidates": state.runtime_candidates,
            "development_episodes": (
                (state.runtime_candidates + 1)
                * len(DEVELOPMENT_TASKS)
                * len(DEVELOPMENT_SEEDS)
            ),
            "champion_commit": state.champion_commit,
            "frozen_champion": str(frozen.resolve()),
            "frozen_tree_sha256": _tree_sha256(frozen),
            "completed_at_unix": time.time(),
        },
    )
    (experiment_root / "PAUSED_QUOTA.json").unlink(missing_ok=True)
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
