#!/usr/bin/env python3
"""Parallel batch driver: RATS on LIBERO-PRO, no skill carry-over across tasks.

For each (suite, task_id) in the LIBERO-PRO eval grid:
  - Spawns `scripts/run_rats.py` with:
      --config env_configs/libero/rats_libero_pro_nonpriv.yaml
      --libero-suite SUITE --libero-task TASK_ID
      --iterations N            (one iteration per "trial"; each iter uses env.reset(seed=iter))
      --fixed-task              (same task each iter; only init_state changes)
      --no-skill-reuse          (frozen library: planner reads seeded skills,
                                  but new skills are NOT extracted/stored across iters)
      --skill-library SEED_PATH (optional: in-context seed of iter050 skills)
      --output-dir outputs/<run>/<SUITE>/<TASK_ID>/

Per-task isolation is by design — each task gets its own output_dir and its own
SkillLibrary instance (loaded from SEED_PATH on every spawn). Even without
--no-skill-reuse, skills learned within task A don't leak to task B.

Parallelism: assigns one (suite, task_id) per worker; rotates through the GPU
list. Default 6 workers (one per GPU). Each worker runs RATS to
completion before picking up the next task.

Usage:
    # bare baseline (no seed library)
    python scripts/run_rats_libero_pro_batch.py \
        --output-dir outputs/rats_libero_pro_noseed \
        --gpus 0,1,2,3,4,5 --workers 6

    # with iter050 frozen skill library as context
    python scripts/run_rats_libero_pro_batch.py \
        --output-dir outputs/rats_libero_pro_iter050seed \
        --seed-skill-library skill_library/libero_formula_warmup5_iter050.json \
        --gpus 0,1,2,3,4,5 --workers 6

    # smaller smoke run
    python scripts/run_rats_libero_pro_batch.py \
        --suites libero_object_swap --task-ids 0,1 --trials 2 \
        --output-dir outputs/rats_libero_pro_smoke \
        --gpus 0 --workers 1
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Optional


_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = (
    "env_configs/libero/rats_libero_pro_nonpriv.yaml"
)
_DEFAULT_SUITES = [
    "libero_object_swap",
    "libero_object_task",
    "libero_goal_swap",
    "libero_goal_task",
    "libero_spatial_swap",
    "libero_spatial_task",
]


@dataclasses.dataclass
class Job:
    suite: str
    task_id: int
    output_dir: Path
    log_path: Path


def build_argv(
    *,
    config: str,
    suite: str,
    task_id: int,
    iterations: int,
    output_dir: Path,
    seed_skill_library: Optional[str],
    extra_flags: list[str],
) -> list[str]:
    """Build the run_rats.py argv for a single (suite, task_id) job."""
    argv: list[str] = [
        os.environ.get("RATS_PYTHON", sys.executable),
        str(_REPO / "scripts/run_rats.py"),
        "--config", config,
        "--env-type", "libero",
        "--libero-suite", suite,
        "--libero-task", str(task_id),
        "--iterations", str(iterations),
        "--fixed-task",
        "--no-skill-reuse",       # frozen library: planner reads seed, no extraction
        "--no-failure-memory",    # disable failure memory carryover across iters
        "--output-dir", str(output_dir),
    ]
    if seed_skill_library:
        argv.extend(["--skill-library", str(seed_skill_library)])
    argv.extend(extra_flags)
    return argv


def run_job(job: Job, argv: list[str], gpu: int, env_overrides: dict[str, str]) -> int:
    """Execute one RATS job, streaming stdout/stderr to log_path. Returns rc."""
    job.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env.update(env_overrides)
    t0 = time.time()
    with open(job.log_path, "w") as logf:
        logf.write(f"# Job: {job.suite}/{job.task_id} on GPU {gpu}\n")
        logf.write(f"# argv: {' '.join(argv)}\n")
        logf.write(f"# env overrides: {env_overrides}\n")
        logf.flush()
        proc = subprocess.Popen(
            argv,
            cwd=str(_REPO),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        rc = proc.wait()
    elapsed = time.time() - t0
    print(
        f"[batch] {job.suite}/{job.task_id} done in {elapsed:.0f}s rc={rc} "
        f"log={job.log_path}",
        flush=True,
    )
    return rc


def worker_loop(
    worker_id: int,
    gpu: int,
    queue: "Queue[Job]",
    argv_factory,
    env_overrides: dict[str, str],
    failures: list,
    lock: threading.Lock,
) -> None:
    while True:
        try:
            job = queue.get_nowait()
        except Exception:
            return
        try:
            argv = argv_factory(job)
            rc = run_job(job, argv, gpu, env_overrides)
            if rc != 0:
                with lock:
                    failures.append((job.suite, job.task_id, rc))
        finally:
            queue.task_done()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=_DEFAULT_CONFIG,
                   help="run_rats.py --config value (default: rats_libero_pro_nonpriv.yaml)")
    p.add_argument("--suites", default=",".join(_DEFAULT_SUITES),
                   help="Comma-separated LIBERO-PRO suite names (default: all 6).")
    p.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9",
                   help="Comma-separated task IDs to run per suite (default: 0..9).")
    p.add_argument("--trials", "--iterations", dest="iterations", type=int, default=5,
                   help="--iterations passed to run_rats.py — each iter is one trial with a "
                        "fresh env.reset(seed=iter). Default 5 to match capx-baseline.")
    p.add_argument("--seed-skill-library", default=None,
                   help="Optional path to a fixed JSON skill library. When set, planner sees "
                        "these skills + primitives as context throughout the run. Combined "
                        "with --no-skill-reuse (always on here), the library does NOT evolve.")
    p.add_argument("--output-dir", required=True,
                   help="Top-level output dir. Per-task results land under "
                        "<output-dir>/<suite>/<task_id>/.")
    p.add_argument("--gpus", default="0,1,2,3,4,5",
                   help="Comma-separated GPU indices to assign workers to (round-robin).")
    p.add_argument("--workers", type=int, default=6,
                   help="Parallel worker count (= max concurrent RATS processes).")
    p.add_argument("--extra-rats-flags", default="",
                   help="Whitespace-separated extra flags forwarded verbatim to run_rats.py.")
    p.add_argument("--env",
                   action="append",
                   default=[],
                   help="Extra env var KEY=VAL to set in worker processes (repeatable). "
                        "Useful for CAPX_DISABLE_VERTEX=1 / OPENROUTER_API_KEY=... etc.")
    p.add_argument("--skip-completed",
                   action="store_true",
                   help="Skip (suite, task_id) where final_summary.json already exists.")
    args = p.parse_args()

    suites = [s for s in args.suites.split(",") if s.strip()]
    task_ids = [int(t) for t in args.task_ids.split(",") if t.strip()]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        print("ERROR: no GPUs in --gpus", file=sys.stderr)
        return 2
    if args.workers > len(gpus):
        print(
            f"WARN: --workers {args.workers} > #GPUs {len(gpus)}; "
            f"workers will share GPUs (round-robin).",
            file=sys.stderr,
        )

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    env_overrides = {}
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env_overrides[k.strip()] = v.strip()

    extra_flags = args.extra_rats_flags.split() if args.extra_rats_flags else []

    def argv_factory(job: Job) -> list[str]:
        return build_argv(
            config=args.config,
            suite=job.suite,
            task_id=job.task_id,
            iterations=args.iterations,
            output_dir=job.output_dir,
            seed_skill_library=args.seed_skill_library,
            extra_flags=extra_flags,
        )

    # Build the job queue: 6 suites × 10 tasks = 60 jobs (defaults).
    queue: "Queue[Job]" = Queue()
    queued = 0
    skipped = 0
    for suite in suites:
        for tid in task_ids:
            out = output_root / suite / f"task_{tid:02d}"
            final = out / "final_summary.json"
            if args.skip_completed and final.exists():
                skipped += 1
                continue
            log = log_root / f"{suite}_task{tid:02d}.log"
            queue.put(Job(suite=suite, task_id=tid, output_dir=out, log_path=log))
            queued += 1

    print(
        f"[batch] suites={suites} task_ids={task_ids} iterations={args.iterations}",
        flush=True,
    )
    print(
        f"[batch] queued {queued} jobs ({skipped} skipped as completed). "
        f"workers={args.workers}, gpus={gpus}",
        flush=True,
    )
    if args.seed_skill_library:
        print(f"[batch] seed library: {args.seed_skill_library}", flush=True)
    if env_overrides:
        print(f"[batch] env overrides: {sorted(env_overrides)}", flush=True)
    print(f"[batch] output_root: {output_root}", flush=True)

    failures: list = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []
    for wid in range(args.workers):
        gpu = gpus[wid % len(gpus)]
        t = threading.Thread(
            target=worker_loop,
            args=(wid, gpu, queue, argv_factory, env_overrides, failures, lock),
            daemon=True,
            name=f"batch-worker-{wid}",
        )
        t.start()
        threads.append(t)

    queue.join()
    for t in threads:
        t.join(timeout=5.0)

    print(
        f"\n[batch] DONE. {queued - len(failures)}/{queued} succeeded. "
        f"failures={len(failures)}",
        flush=True,
    )
    for s, tid, rc in failures:
        print(f"  FAIL {s}/{tid}: rc={rc}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
