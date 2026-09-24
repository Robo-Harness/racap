#!/usr/bin/env python3
"""Summarize LIBERO batch success rates from CaP-X output directories."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize success rates from CaP-X LIBERO batch outputs."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="outputs/capx_eval_gpt55_object_swap_learned_skills",
        help=(
            "Output root. This can be the batch root containing suite directories, "
            "or one specific suite directory such as outputs/.../libero_object_swap."
        ),
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help=(
            "Suite name to summarize. Can be repeated. If omitted, all "
            "libero_* suite directories under root are summarized."
        ),
    )
    return parser.parse_args()


def is_trial_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("trial_") and "_taskcompleted_" in path.name


def trial_success(path: Path) -> bool:
    return "_taskcompleted_1" in path.name


def find_suite_dirs(root: Path, suite_filters: list[str] | None) -> list[Path]:
    if root.name.startswith("libero_"):
        if suite_filters and root.name not in suite_filters:
            return []
        return [root]

    if suite_filters:
        return [root / suite for suite in suite_filters if (root / suite).is_dir()]

    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("libero_"))


def summarize_task(task_dir: Path) -> tuple[int, int]:
    # Standard run_libero_batch layout:
    # root/suite/task/model/run/trial_*_taskcompleted_*
    trial_dirs = [p for p in task_dir.glob("*/run/trial_*") if is_trial_dir(p)]

    # Fallback for simpler layouts:
    # root/suite/task/trial_*_taskcompleted_*
    if not trial_dirs:
        trial_dirs = [p for p in task_dir.glob("trial_*") if is_trial_dir(p)]

    successes = sum(1 for p in trial_dirs if trial_success(p))
    return successes, len(trial_dirs)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Output root does not exist: {root}")

    suites = find_suite_dirs(root, args.suite)
    if not suites:
        raise SystemExit(f"No LIBERO suite directories found under: {root}")

    grand_successes = 0
    grand_trials = 0

    for suite_dir in suites:
        print(f"\n{suite_dir.name}")
        print("-" * len(suite_dir.name))

        suite_successes = 0
        suite_trials = 0
        task_dirs = sorted(p for p in suite_dir.iterdir() if p.is_dir())

        for task_dir in task_dirs:
            successes, trials = summarize_task(task_dir)
            if trials == 0:
                continue
            rate = successes / trials
            print(f"{task_dir.name}: {successes}/{trials} = {rate:.1%}")
            suite_successes += successes
            suite_trials += trials

        if suite_trials == 0:
            print("No trial directories found.")
            continue

        suite_rate = suite_successes / suite_trials
        print(f"{suite_dir.name} TOTAL: {suite_successes}/{suite_trials} = {suite_rate:.1%}")

        grand_successes += suite_successes
        grand_trials += suite_trials

    if len(suites) > 1 and grand_trials:
        grand_rate = grand_successes / grand_trials
        print(f"\nOVERALL: {grand_successes}/{grand_trials} = {grand_rate:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
