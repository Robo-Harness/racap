from __future__ import annotations

import copy
import os
import re
import shutil
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rats.envs.simulators.molmospaces import default_molmospaces_catalog


@dataclass
class MolmoSpacesBatchLaunchArgs:
    """Command-line arguments for MolmoSpaces batch execution."""

    base_config_path: str = "env_configs/molmospaces/franka_molmospaces_put_bowl_on_plate.yaml"
    config_path: str | None = None
    """Alias for base_config_path, matching rats.envs.launch."""
    benchmarks: list[str] = field(default_factory=lambda: ["phase1"])
    benchmark_dir: str | None = None
    models: list[str] = field(default_factory=lambda: ["openai/gpt-5.4-mini"])
    model: str | None = None
    """Single-model alias for models, matching rats.envs.launch."""
    server_url: str = "http://127.0.0.1:8110/chat/completions"
    output_dir: str | None = None
    """Output directory. Defaults to resume dir when --resume is used."""
    resume: str | None = None
    """Resume an interrupted batch from this output dir, skipping completed runs."""
    temperature: float = 1.0
    max_tokens: int = 2048 * 10
    reasoning_effort: str = "medium"
    api_key: str | None = None
    use_visual_feedback: bool | None = None
    use_img_differencing: bool | None = None
    total_trials: int | None = None
    num_workers: int | None = None
    record_video: bool | None = None
    debug: bool = False
    use_oracle_code: bool | None = None
    launch: bool = True
    skip_completed: bool = False
    """Skip task/model runs that already have an aaa_done_flag."""
    shard: int = 0
    """Shard index for parallel runs. Pair with --num-shards to split the
    100-task benchmark across N processes. ``shard=k, num_shards=N`` runs
    tasks whose index ``i`` satisfies ``i % N == k``. Default 0/1 means
    run all tasks (no sharding)."""
    num_shards: int = 1
    """Total number of shards. Combined with --shard to filter tasks.
    Use the same num_shards in every paralleled invocation, vary --shard
    from 0..num_shards-1."""


def build_molmospaces_batch_tasks(
    *,
    benchmarks: list[str] | None = None,
    benchmark_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Return bridge-catalog task descriptors for the requested benchmarks."""
    # Preserve the historical CLI default (phase1) for the built-in catalog,
    # but when a JSON benchmark_dir is supplied, the common intent is to sweep
    # every episode in that directory unless the caller explicitly filters.
    selected = set(benchmarks or [])
    if benchmark_dir and selected == {"phase1"}:
        selected = set()
    tasks = []
    for descriptor in default_molmospaces_catalog(benchmark_dir=benchmark_dir):
        if selected and descriptor["benchmark"] not in selected:
            continue
        tasks.append(descriptor)
    return tasks


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _descriptor_output_parts(descriptor: dict[str, Any]) -> list[str]:
    return [
        _safe_path_component(descriptor["benchmark"]),
        _safe_path_component(descriptor["scene_family"]),
        _safe_path_component(descriptor["task_family"]),
        _safe_path_component(descriptor["variant"]),
    ]


def _done_flag_path(output_root: str, descriptor: dict[str, Any], model: str) -> Path:
    return _run_output_dir(output_root, descriptor, model) / "aaa_done_flag" / "aaa_done_flag.txt"


def _run_output_dir(output_root: str, descriptor: dict[str, Any], model: str) -> Path:
    return (
        Path(output_root)
        .joinpath(*_descriptor_output_parts(descriptor))
        .joinpath(str(model).replace("/", "_"))
        .joinpath("run")
    )


def _is_completed_run(
    output_root: str,
    descriptor: dict[str, Any],
    model: str,
    *,
    expected_trials: int | None = None,
) -> bool:
    """Return True when an interrupted batch can safely skip this run.

    The normal completion marker is ``aaa_done_flag``.  On Ctrl-C, however,
    a run can have already written ``summaries.txt`` but not the done flag, so
    resume also treats a parseable summary with the expected trial count as
    complete.  Partial trial artifacts without a summary are intentionally
    re-run.
    """
    if _done_flag_path(output_root, descriptor, model).exists():
        return True
    summary = _parse_launch_summary(
        _run_output_dir(output_root, descriptor, model) / "summaries.txt"
    )
    if not summary:
        return False
    trials = int(summary.get("trials", 0) or 0)
    if expected_trials is None:
        return trials > 0
    return trials >= int(expected_trials)


def _restore_resume_skill_artifacts(
    *,
    resume_dir: str | None,
    output_root: str,
    base_config: dict[str, Any],
) -> list[str]:
    """Copy stateful CAPx skill-library artifacts when resuming elsewhere.

    MolmoSpaces CAPx benchmark configs mostly use a static API skill library,
    but CAPx also has an optional evolving ``rats.skills.SkillLibrary``.  This
    preserves that state for configs that use it, while doing nothing for the
    static reduced-skill API case.
    """
    if not resume_dir:
        return []
    src_root = Path(resume_dir).expanduser()
    dst_root = Path(output_root).expanduser()
    if src_root.resolve() == dst_root.resolve():
        return []

    copied: list[str] = []
    candidate_names = {
        ".capx_skills.json",
        "capx_skills.json",
        "skills.json",
        "skill_library.json",
    }
    configured = base_config.get("skill_library_path")
    if configured:
        candidate_names.add(Path(str(configured)).name)

    for name in sorted(candidate_names):
        src = src_root / name
        if not src.exists() or not src.is_file():
            continue
        dst = dst_root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))

    return copied


def _config_benchmark_dir(config: dict[str, Any]) -> str | None:
    low_level = (
        config.get("env", {})
        .get("cfg", {})
        .get("low_level", {})
    )
    if isinstance(low_level, dict):
        benchmark_dir = low_level.get("benchmark_dir")
        if benchmark_dir:
            return str(benchmark_dir)
    return None


def write_molmospaces_batch_config(
    *,
    base_config: dict[str, Any],
    descriptor: dict[str, Any],
    output_root: str,
    benchmark_dir: str | None = None,
) -> str:
    """Write a task-specific MolmoSpaces config and return its path."""
    config = copy.deepcopy(base_config)
    env_cfg = config.setdefault("env", {}).setdefault("cfg", {})
    low_level = env_cfg.setdefault("low_level", {})
    low_level.update(
        {
            "_target_": "rats.envs.simulators.molmospaces.FrankaMolmoSpacesEnv",
            "benchmark": descriptor["benchmark"],
            "scene_family": descriptor["scene_family"],
            "task_family": descriptor["task_family"],
            "variant": descriptor["variant"],
            "canonical_task_id": descriptor["canonical_id"],
        }
    )
    if benchmark_dir is not None:
        low_level["benchmark_dir"] = benchmark_dir
    config["task_name"] = descriptor["canonical_id"]
    if config.get("evolve_skill_library") and not config.get("skill_library_path"):
        config["skill_library_path"] = os.path.join(
            os.path.abspath(output_root),
            ".capx_skills.json",
        )
    rel_parts = _descriptor_output_parts(descriptor)
    config["output_dir"] = os.path.join(
        os.path.abspath(output_root),
        *rel_parts,
        "run",
    )
    config_dir = Path(output_root).joinpath(*rel_parts)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(config_path)


def _parse_launch_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    parsed: dict[str, Any] = {"summary_path": str(path)}
    for idx, line in enumerate(lines):
        if line.startswith("Total number of trials:"):
            try:
                parsed["trials"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                parsed["trials"] = 0
        elif line.startswith("Code generation success rate / Average reward / Task completed:"):
            if idx + 1 < len(lines):
                parts = lines[idx + 1].strip().split("/")
                if len(parts) >= 3:
                    try:
                        parsed["success_rate"] = float(parts[0])
                        parsed["average_reward"] = float(parts[1])
                        parsed["task_completed"] = int(float(parts[2]))
                    except ValueError:
                        pass
        elif line.startswith("Average code blocks:"):
            parsed["average_code_blocks"] = _parse_float_after_colon(line)
        elif line.startswith("Average regenerations:"):
            parsed["average_regenerations"] = _parse_float_after_colon(line)
        elif line.startswith("Average finishes:"):
            parsed["average_finishes"] = _parse_float_after_colon(line)
        elif line.startswith("Elapsed time:"):
            elapsed = line.split(":", 1)[1].strip().removesuffix(" seconds")
            try:
                parsed["elapsed_seconds"] = float(elapsed)
            except ValueError:
                pass
    parsed.setdefault("trials", 0)
    parsed.setdefault("success_rate", 0.0)
    parsed.setdefault("average_reward", 0.0)
    parsed.setdefault("task_completed", 0)
    return parsed


def _parse_float_after_colon(line: str) -> float:
    try:
        return float(line.split(":", 1)[1].strip())
    except (IndexError, ValueError):
        return 0.0


def _collect_batch_rows(
    *,
    output_root: str,
    descriptors: list[dict[str, Any]],
    models: list[str],
    failed_runs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    failed = set(failed_runs)
    rows: list[dict[str, Any]] = []
    for model in models:
        for descriptor in descriptors:
            canonical_id = descriptor["canonical_id"]
            run_dir = _run_output_dir(output_root, descriptor, model)
            summary = _parse_launch_summary(run_dir / "summaries.txt")
            row = {
                "model": model,
                "canonical_id": canonical_id,
                "task_family": descriptor.get("task_family", "unknown"),
                "benchmark": descriptor.get("benchmark", "unknown"),
                "scene_family": descriptor.get("scene_family", "unknown"),
                "variant": descriptor.get("variant", "unknown"),
                "run_dir": str(run_dir),
                "status": "missing",
                "trials": 0,
                "success_count": 0.0,
                "reward_sum": 0.0,
                "task_completed": 0,
                "code_blocks_sum": 0.0,
                "regenerations_sum": 0.0,
                "finishes_sum": 0.0,
                "elapsed_seconds": 0.0,
            }
            if (model, canonical_id) in failed:
                row["status"] = "failed"
            if summary:
                trials = int(summary.get("trials", 0) or 0)
                row.update(
                    {
                        "status": "complete",
                        "trials": trials,
                        "success_count": float(summary.get("success_rate", 0.0)) * trials,
                        "reward_sum": float(summary.get("average_reward", 0.0)) * trials,
                        "task_completed": int(summary.get("task_completed", 0) or 0),
                        "code_blocks_sum": float(summary.get("average_code_blocks", 0.0)) * trials,
                        "regenerations_sum": float(summary.get("average_regenerations", 0.0)) * trials,
                        "finishes_sum": float(summary.get("average_finishes", 0.0)) * trials,
                        "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0) or 0.0),
                    }
                )
            elif _done_flag_path(output_root, descriptor, model).exists():
                row["status"] = "done_missing_summary"
            rows.append(row)
    return rows


def write_batch_summary(
    *,
    output_root: str,
    descriptors: list[dict[str, Any]],
    models: list[str],
    failed_runs: list[tuple[str, str]] | None = None,
) -> Path:
    """Write an aggregate markdown summary for a MolmoSpaces batch run."""
    failed_runs = failed_runs or []
    rows = _collect_batch_rows(
        output_root=output_root,
        descriptors=descriptors,
        models=models,
        failed_runs=failed_runs,
    )
    completed_rows = [r for r in rows if r["status"] == "complete"]
    total_runs = len(rows)
    completed_runs = len(completed_rows)
    missing_runs = sum(1 for r in rows if r["status"] == "missing")
    failed_count = sum(1 for r in rows if r["status"] == "failed")
    done_missing = sum(1 for r in rows if r["status"] == "done_missing_summary")
    total_trials = sum(int(r["trials"]) for r in completed_rows)
    task_completed = sum(int(r["task_completed"]) for r in completed_rows)
    success_count = sum(float(r["success_count"]) for r in completed_rows)
    reward_sum = sum(float(r["reward_sum"]) for r in completed_rows)
    elapsed_sum = sum(float(r["elapsed_seconds"]) for r in completed_rows)
    code_blocks_sum = sum(float(r["code_blocks_sum"]) for r in completed_rows)
    regenerations_sum = sum(float(r["regenerations_sum"]) for r in completed_rows)
    finishes_sum = sum(float(r["finishes_sum"]) for r in completed_rows)

    def rate(num: float, den: int) -> float:
        return float(num) / den if den else 0.0

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["task_family"])].append(row)

    lines = [
        "# MolmoSpaces Batch Summary",
        "",
        f"- Output root: `{output_root}`",
        f"- Models: {', '.join(f'`{m}`' for m in models)}",
        f"- Planned task/model runs: {total_runs}",
        f"- Completed runs with summaries: {completed_runs}",
        f"- Missing/not-yet-run: {missing_runs}",
        f"- Failed launches: {failed_count}",
        f"- Done flag but missing summary: {done_missing}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total trials summarized | {total_trials} |",
        f"| Sandbox/code success rate | {rate(success_count, total_trials):.3f} ({success_count:.0f}/{total_trials}) |",
        f"| Task completion rate | {rate(task_completed, total_trials):.3f} ({task_completed}/{total_trials}) |",
        f"| Average reward | {rate(reward_sum, total_trials):.3f} |",
        f"| Average code blocks | {rate(code_blocks_sum, total_trials):.3f} |",
        f"| Average regenerations | {rate(regenerations_sum, total_trials):.3f} |",
        f"| Average finishes | {rate(finishes_sum, total_trials):.3f} |",
        f"| Total elapsed in task runners | {elapsed_sum:.2f}s |",
        "",
        "## By task family",
        "",
        "| Task family | Runs complete/planned | Trials | Task completion | Sandbox success | Avg reward | Avg code blocks | Avg regenerations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in sorted(by_family):
        family_rows = by_family[family]
        family_complete = [r for r in family_rows if r["status"] == "complete"]
        fam_trials = sum(int(r["trials"]) for r in family_complete)
        fam_completed = sum(int(r["task_completed"]) for r in family_complete)
        fam_success = sum(float(r["success_count"]) for r in family_complete)
        fam_reward = sum(float(r["reward_sum"]) for r in family_complete)
        fam_blocks = sum(float(r["code_blocks_sum"]) for r in family_complete)
        fam_regens = sum(float(r["regenerations_sum"]) for r in family_complete)
        lines.append(
            f"| `{family}` | {len(family_complete)}/{len(family_rows)} | {fam_trials} | "
            f"{rate(fam_completed, fam_trials):.3f} ({fam_completed}/{fam_trials}) | "
            f"{rate(fam_success, fam_trials):.3f} | {rate(fam_reward, fam_trials):.3f} | "
            f"{rate(fam_blocks, fam_trials):.3f} | {rate(fam_regens, fam_trials):.3f} |"
        )

    if failed_runs:
        lines.extend(["", "## Failed launches", ""])
        for model, canonical_id in failed_runs:
            lines.append(f"- `{model}` / `{canonical_id}`")

    incomplete = [r for r in rows if r["status"] != "complete"]
    if incomplete:
        lines.extend(["", "## Incomplete runs", ""])
        for row in incomplete[:200]:
            lines.append(f"- `{row['status']}` `{row['model']}` / `{row['canonical_id']}`")
        if len(incomplete) > 200:
            lines.append(f"- ... {len(incomplete) - 200} more")

    path = Path(output_root) / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main(args: MolmoSpacesBatchLaunchArgs) -> None:
    from rats.envs.launch import LaunchArgs
    from rats.envs.launch import main as launch_main

    if args.output_dir is None:
        args.output_dir = args.resume or "./outputs/molmospaces_batch_run"
    output_root = str(args.output_dir)
    resume_enabled = args.resume is not None
    skip_completed = bool(args.skip_completed or resume_enabled)

    base_config_path = args.config_path or args.base_config_path
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    benchmark_dir = args.benchmark_dir or _config_benchmark_dir(base_config)
    models = [args.model] if args.model else args.models

    copied_skill_artifacts = _restore_resume_skill_artifacts(
        resume_dir=args.resume,
        output_root=output_root,
        base_config=base_config,
    )
    if resume_enabled:
        print(
            f"Resuming MolmoSpaces batch from {args.resume}; "
            f"output_dir={output_root}; completed runs will be skipped."
        )
        if copied_skill_artifacts:
            print(
                "Restored CAPx skill artifact(s): "
                + ", ".join(copied_skill_artifacts)
            )

    tasks_to_run = build_molmospaces_batch_tasks(
        benchmarks=args.benchmarks,
        benchmark_dir=benchmark_dir,
    )
    # Shard filter for parallel runs. With --shard k --num-shards N, this
    # process only takes tasks whose original index i satisfies i % N == k.
    # The benchmark.json ordering is preserved within the shard, and each
    # task keeps its output path (so --skip-completed still works across
    # shards if you re-launch with different sharding).
    if args.num_shards > 1:
        sharded = [
            t for i, t in enumerate(tasks_to_run)
            if i % args.num_shards == args.shard
        ]
        print(
            f"Sharding: shard {args.shard}/{args.num_shards} → "
            f"{len(sharded)}/{len(tasks_to_run)} tasks"
        )
        tasks_to_run = sharded
    total_runs = len(models) * len(tasks_to_run)
    failed_runs: list[tuple[str, str]] = []
    experiment_idx = 1

    for model in models:
        for descriptor in tasks_to_run:
            if skip_completed and _is_completed_run(
                output_root,
                descriptor,
                model,
                expected_trials=args.total_trials,
            ):
                print(f"[{experiment_idx}/{total_runs}] skipping completed {descriptor['canonical_id']} for model={model}")
                experiment_idx += 1
                continue
            config_path = write_molmospaces_batch_config(
                base_config=base_config,
                descriptor=descriptor,
                output_root=output_root,
                benchmark_dir=benchmark_dir,
            )
            if args.launch:
                launch_args = LaunchArgs(
                    config_path=config_path,
                    server_url=args.server_url,
                    model=model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    api_key=args.api_key,
                    use_visual_feedback=args.use_visual_feedback,
                    use_img_differencing=args.use_img_differencing,
                    total_trials=args.total_trials,
                    num_workers=args.num_workers,
                    record_video=args.record_video,
                    output_dir=None,
                    debug=args.debug,
                    use_oracle_code=args.use_oracle_code,
                )
                try:
                    launch_main(launch_args)
                except Exception:
                    traceback.print_exc()
                    failed_runs.append((model, descriptor["canonical_id"]))
            print(f"[{experiment_idx}/{total_runs}] prepared {descriptor['canonical_id']} for model={model}")
            summary_path = write_batch_summary(
                output_root=output_root,
                descriptors=tasks_to_run,
                models=models,
                failed_runs=failed_runs,
            )
            experiment_idx += 1

    summary_path = write_batch_summary(
        output_root=output_root,
        descriptors=tasks_to_run,
        models=models,
        failed_runs=failed_runs,
    )
    print(f"Batch summary written to {summary_path}")

    if failed_runs:
        raise RuntimeError(f"MolmoSpaces batch execution failed for {failed_runs}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(MolmoSpacesBatchLaunchArgs))
