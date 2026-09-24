#!/usr/bin/env python3
"""M0 acceptance: call pickplace() against a real LIBERO scene.

The bar for M0 is not task success. It is that the Policy API executes against
real perception, real IK and real contact physics, and that whatever happens
comes back as a structured SkillResult whose failure_mode a runtime agent could
actually act on. A run that fails with ``empty_grasp`` passes M0; a run that
throws an unhandled exception, or that reports success without the native
predicate agreeing, does not.

Usage:
    source configs/env.sh
    python scripts/run_m0_pickplace.py --suite libero_object --task 0 \
        --pick "alphabet soup can" --place "basket"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from racap.backends.libero import LiberoPrimitiveRuntime
from racap.policy_api import pickplace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pick", required=True)
    parser.add_argument("--place", required=True)
    parser.add_argument("--grasp-backend", default="graspnet")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m0_pickplace"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    runtime = LiberoPrimitiveRuntime(
        suite_name=args.suite,
        task_id=args.task,
        seed=args.seed,
        grasp_backend=args.grasp_backend,
    )
    print(f"scene instruction: {runtime.instruction}")
    print(f"oracle objects:    {runtime.oracle.object_names()}")

    try:
        result = pickplace(runtime, args.pick, args.place)
        # The oracle is read only after the policy has finished, so it cannot
        # influence any decision the policy made.
        native_success = runtime.oracle.task_completed()
    finally:
        runtime.close()

    record = {
        "schema_version": "policy_rollout_v1",
        "backend": "racap_libero",
        "suite": args.suite,
        "task_id": args.task,
        "seed": args.seed,
        "instruction": runtime.instruction,
        "pick_label": args.pick,
        "place_label": args.place,
        "policy_success": result.success,
        "native_success": native_success,
        "failure_mode": result.failure_mode,
        "attempts": result.attempts,
        "strategy": result.strategy,
        "elapsed_seconds": round(time.time() - started, 2),
        "trace": result.trace,
    }

    out = args.output_dir / f"{args.suite}_task{args.task}_seed{args.seed}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))

    print(
        json.dumps(
            {k: v for k, v in record.items() if k != "trace"},
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    print(f"\nwrote {out}")

    # A policy claiming success while the benchmark disagrees is the single
    # most damaging failure in this pipeline, because evolution would optimize
    # toward it. Surface it loudly rather than burying it in the JSON.
    if result.success and not native_success:
        print("\nWARNING: policy reported success but the native predicate did not.")


if __name__ == "__main__":
    main()
