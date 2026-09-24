#!/usr/bin/env python3
"""Analyze the executor-matched CaP-X versus RATS Robosuite diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.controlled_comparison.plot_style import (
    METHOD_COLORS,
    clean_axis,
    save_figure,
)

try:
    from .analyze_results import (
        _jsonl,
        _holm,
        _telemetry_by_key,
        _total_simulator_steps,
        _usage,
        _verified_evidence,
        _wilson,
    )
    from .run_robosuite import TASKS, _robosuite_git_identity
except ImportError:  # direct ``python path/to/script.py`` execution
    from analyze_results import (  # type: ignore[no-redef]
        _jsonl,
        _holm,
        _telemetry_by_key,
        _total_simulator_steps,
        _usage,
        _verified_evidence,
        _wilson,
    )
    from run_robosuite import TASKS, _robosuite_git_identity  # type: ignore[no-redef]

METHODS = ["capx", "rats_90"]
QUOTA_PATTERN = re.compile(
    r"quota|insufficient[^\n]*(?:balance|credit)|(?:balance|credit)[^\n]*insufficient|payment required",
    re.IGNORECASE,
)
RATS_CONFIG_DELTA = {
    "external_skill_library_mode": "planner",
    "external_skill_library_max_skills": 0,
    "external_skill_library_include_code": True,
    "external_skill_library_include_primitives": False,
    "external_skill_planner_model": "gpt-5.5",
    "external_skill_planner_max_selected": 6,
    "external_skill_planner_include_code": False,
    "external_skill_policy_include_code": True,
}


def _seed_from_key(key: str) -> int | None:
    match = re.search(r"/seed(-?\d+)$", key)
    return int(match.group(1)) if match else None


def _model_and_quota_audit(
    rows: list[dict[str, Any]],
) -> tuple[list[str], bool, int]:
    success_rows = [row for row in rows if row.get("event") == "network_success"]
    actual_models = sorted(
        {
            str(row.get("actual_model") or "").strip()
            for row in success_rows
            if str(row.get("actual_model") or "").strip()
        }
    )
    identity_complete = bool(success_rows) and all(
        str(row.get("actual_model") or "").strip() for row in success_rows
    )
    quota_markers = sum(
        bool(
            QUOTA_PATTERN.search(
                " ".join(
                    str(row.get(field) or "")
                    for field in ("error", "message", "detail", "response_text")
                )
            )
        )
        for row in rows
    )
    return actual_models, identity_complete, quota_markers


def _resolved_config_audit(
    input_root: Path,
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    """Prove that task YAMLs differ only by the registered RATS retrieval delta."""

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for task in TASKS:
        paths = {
            method: input_root / method / task / "config.yaml" for method in METHODS
        }
        missing = [f"{method}:{path}" for method, path in paths.items() if not path.is_file()]
        if missing:
            errors.append(f"missing_resolved_config:{task}:{','.join(missing)}")
            continue
        configs = {
            method: yaml.safe_load(path.read_text(encoding="utf-8"))
            for method, path in paths.items()
        }
        capx = configs["capx"]
        rats = configs["rats_90"]
        capx_env = capx.get("env", {}).get("cfg", {})
        rats_env = rats.get("env", {}).get("cfg", {})
        leaked = sorted(
            key
            for key in ("external_skill_library_path", *RATS_CONFIG_DELTA)
            if key in capx_env
        )
        if leaked:
            errors.append(f"capx_contains_rats_delta:{task}:{','.join(leaked)}")
        library_path = str(rats_env.get("external_skill_library_path") or "")
        observed_delta = {key: rats_env.get(key) for key in RATS_CONFIG_DELTA}
        delta_valid = bool(library_path) and observed_delta == RATS_CONFIG_DELTA
        if not delta_valid:
            errors.append(f"invalid_rats_retrieval_delta:{task}")
        if capx.get("save_skill_planner_prompts") not in (None, False):
            errors.append(f"capx_skill_planner_prompt_flag_enabled:{task}")
        if rats.get("save_skill_planner_prompts") is not True:
            errors.append(f"rats_skill_planner_prompt_flag_missing:{task}")

        normalized: dict[str, dict[str, Any]] = {}
        for method, original in configs.items():
            config = json.loads(json.dumps(original))
            config.pop("output_dir", None)
            config.pop("save_skill_planner_prompts", None)
            env_cfg = config.get("env", {}).get("cfg", {})
            env_cfg.pop("external_skill_library_path", None)
            for key in RATS_CONFIG_DELTA:
                env_cfg.pop(key, None)
            normalized[method] = config
        matched = normalized["capx"] == normalized["rats_90"]
        if not matched:
            errors.append(f"unregistered_resolved_config_delta:{task}")
        rows.append(
            {
                "task": task,
                "capx_config": str(paths["capx"].resolve()),
                "rats_90_config": str(paths["rats_90"].resolve()),
                "normalized_configs_match": matched,
                "rats_delta_valid": delta_valid,
                "rats_library_path": library_path,
                "capx_rats_delta_fields": ",".join(leaked),
            }
        )
    return not errors and len(rows) == len(TASKS), rows, errors


def _is_physical_api_boundary(row: dict[str, Any]) -> bool:
    if row.get("event") != "native_state_after_api":
        return False
    name = str(row.get("api_name") or "").lower()
    return name.startswith(("move_to_joints", "goto_pose")) or name.startswith(
        ("open_gripper", "close_gripper")
    )


def _load_method(method_root: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        root = method_root / task
        evidence = _verified_evidence(root, require_complete_marker=True)
        if evidence is None:
            continue
        states = _telemetry_by_key([root / "native_states.jsonl"])
        resets = _telemetry_by_key([root / "sim_episodes.jsonl"])
        calls = _telemetry_by_key([root / "llm_calls.jsonl"])
        for key, unordered in states.items():
            seed = _seed_from_key(key)
            if seed is None:
                continue
            ordered = sorted(unordered, key=lambda row: float(row.get("time") or 0.0))
            final = ordered[-1]
            reset_rows = resets.get(key, [])
            call_rows = calls.get(key, [])
            actual_models, model_identity_complete, quota_markers = (
                _model_and_quota_audit(call_rows)
            )
            timestamps = [
                float(row.get("time") or row.get("timestamp") or 0.0)
                for row in ordered + reset_rows + call_rows
                if row.get("time") or row.get("timestamp")
            ]
            reset_times = [
                float(row.get("time") or row.get("timestamp") or 0.0)
                for row in reset_rows
                if row.get("time") or row.get("timestamp")
            ]
            rows.append(
                {
                    "method": method,
                    "task": task,
                    "seed": seed,
                    "episode_key": f"{task}/seed{seed}",
                    "native_success": bool(final.get("native_success")),
                    "reward": float(final.get("reward") or 0.0),
                    "wall_seconds": (
                        max(timestamps) - min(timestamps) if len(timestamps) >= 2 else math.nan
                    ),
                    "reset_time_unix": min(reset_times) if reset_times else math.nan,
                    **_usage(call_rows),
                    "simulator_steps": _total_simulator_steps(ordered, reset_rows),
                    "simulator_resets": len(reset_rows),
                    "actual_models": ",".join(actual_models),
                    "model_identity_complete": model_identity_complete,
                    "quota_marker_count": quota_markers,
                    "public_api_boundaries": sum(
                        str(row.get("event", "")).startswith("native_state_after_api")
                        for row in ordered
                    ),
                    "physical_api_boundaries": sum(
                        _is_physical_api_boundary(row) for row in ordered
                    ),
                    "artifact_path": str(root.resolve()),
                    "evidence_path": str(evidence.resolve()),
                }
            )
    return rows


def _aggregate(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, task), group in episodes.groupby(["method", "task"]):
        successes = int(group.native_success.sum())
        low, high = _wilson(successes, len(group))
        rows.append(
            {
                "method": method,
                "task": task,
                "episodes": len(group),
                "successes": successes,
                "success_rate": successes / len(group),
                "wilson_low": low,
                "wilson_high": high,
                "median_wall_seconds": group.wall_seconds.median(),
                "mean_model_calls": group.model_calls.mean(),
                "mean_estimated_api_cost_usd": group.estimated_api_cost_usd.mean(),
                "mean_simulator_steps": group.simulator_steps.mean(),
                "mean_public_api_boundaries": group.public_api_boundaries.mean(),
                "mean_physical_api_boundaries": group.physical_api_boundaries.mean(),
            }
        )
    for method, group in episodes.groupby("method"):
        successes = int(group.native_success.sum())
        low, high = _wilson(successes, len(group))
        rows.append(
            {
                "method": method,
                "task": "ALL",
                "episodes": len(group),
                "successes": successes,
                "success_rate": successes / len(group),
                "wilson_low": low,
                "wilson_high": high,
                "median_wall_seconds": group.wall_seconds.median(),
                "mean_model_calls": group.model_calls.mean(),
                "mean_estimated_api_cost_usd": group.estimated_api_cost_usd.mean(),
                "mean_simulator_steps": group.simulator_steps.mean(),
                "mean_public_api_boundaries": group.public_api_boundaries.mean(),
                "mean_physical_api_boundaries": group.physical_api_boundaries.mean(),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    values: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan
    samples = rng.choice(values, size=(resamples, len(values)), replace=True).mean(1)
    return (
        float(values.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _paired(episodes: pd.DataFrame, resamples: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(20260822)
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [
        (task, episodes[episodes.task == task]) for task in TASKS
    ] + [("ALL", episodes)]
    for scope, frame in scopes:
        left = frame[frame.method == "capx"].set_index("episode_key")
        right = frame[frame.method == "rats_90"].set_index("episode_key")
        keys = left.index.intersection(right.index)
        if not len(keys):
            continue
        capx = left.loc[keys, "native_success"].astype(bool)
        rats = right.loc[keys, "native_success"].astype(bool)
        capx_only = int((capx & ~rats).sum())
        rats_only = int((~capx & rats).sum())
        discord = capx_only + rats_only

        def delta(metric: str) -> tuple[float, float, float]:
            values = (
                pd.to_numeric(right.loc[keys, metric], errors="coerce").astype(float)
                - pd.to_numeric(left.loc[keys, metric], errors="coerce").astype(float)
            )
            if scope == "ALL":
                tasks = left.loc[keys, "task"]
                values = values.groupby(tasks).mean()
            return _bootstrap(values.to_numpy(float), rng, resamples)

        success, success_low, success_high = delta("native_success")
        wall, wall_low, wall_high = delta("wall_seconds")
        calls, calls_low, calls_high = delta("model_calls")
        cost, cost_low, cost_high = delta("estimated_api_cost_usd")
        rows.append(
            {
                "scope": scope,
                "paired_episodes": len(keys),
                "capx_success_rate": float(capx.mean()),
                "rats_90_success_rate": float(rats.mean()),
                "rats_minus_capx": success,
                "success_difference_bootstrap_low": success_low,
                "success_difference_bootstrap_high": success_high,
                "bootstrap_unit": "task" if scope == "ALL" else "paired_seed",
                "capx_only_success": capx_only,
                "rats_only_success": rats_only,
                "mcnemar_exact_p": (
                    binomtest(min(capx_only, rats_only), discord, 0.5).pvalue
                    if discord
                    else 1.0
                ),
                "mean_wall_seconds_rats_minus_capx": wall,
                "wall_delta_bootstrap_low": wall_low,
                "wall_delta_bootstrap_high": wall_high,
                "mean_model_calls_rats_minus_capx": calls,
                "model_calls_delta_bootstrap_low": calls_low,
                "model_calls_delta_bootstrap_high": calls_high,
                "mean_cost_rats_minus_capx": cost,
                "cost_delta_bootstrap_low": cost_low,
                "cost_delta_bootstrap_high": cost_high,
            }
        )
    if rows:
        adjusted = _holm([float(row["mcnemar_exact_p"]) for row in rows])
        for row, value in zip(rows, adjusted):
            row["mcnemar_holm_p"] = value
    return pd.DataFrame(rows)


def _plot(aggregate: pd.DataFrame, output: Path) -> None:
    part = aggregate[aggregate.task != "ALL"].pivot(
        index="task", columns="method", values="success_rate"
    )
    if part.empty:
        return
    part = part.reindex(index=list(TASKS), columns=METHODS)
    axis = part.plot(
        kind="bar",
        figsize=(7.15, 3.75),
        color=[METHOD_COLORS.get(str(column), "#707070") for column in part.columns],
        edgecolor="white",
        linewidth=0.5,
    )
    axis.set_ylabel("Native success rate")
    axis.set_ylim(0, 1.12)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.set_xlabel("")
    axis.tick_params(axis="x", labelrotation=25)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    clean_axis(axis, grid_axis="y")
    axis.legend(["CaP-X", "RATS-90"], ncol=2, loc="upper center")
    axis.set_title(
        "Executor-matched Robosuite transfer (5 seeds per task)",
        loc="left",
        fontweight="bold",
    )
    for container in axis.containers:
        labels = [f"{int(round(bar.get_height() * 5))}/5" for bar in container]
        axis.bar_label(container, labels=labels, fontsize=7, padding=2)
    axis.figure.tight_layout()
    save_figure(axis.figure, output)
    plt.close(axis.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "robosuite",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "analysis" / "robosuite",
    )
    parser.add_argument("--required-model", default="gpt-5.5")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows.extend(_load_method(args.input_root / method, method))
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        raise SystemExit(f"no Robosuite telemetry under {args.input_root}")
    duplicate = episodes.duplicated(["method", "episode_key"], keep=False)
    if duplicate.any():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        conflict_path = args.output_dir / "duplicate_episode_keys.csv"
        episodes.loc[duplicate].sort_values(
            ["method", "episode_key", "artifact_path"]
        ).to_csv(conflict_path, index=False)
        raise SystemExit(
            "refusing duplicate Robosuite episode keys; "
            f"conflicts written to {conflict_path}"
        )
    aggregate = _aggregate(episodes)
    paired = _paired(episodes)
    expected = pd.DataFrame(
        [
            {
                "method": method,
                "task": task,
                "seed": seed,
                "episode_key": f"{task}/seed{seed}",
            }
            for method in METHODS
            for task in TASKS
            for seed in range(5)
        ]
    )
    observed = set(zip(episodes.method, episodes.episode_key))
    expected["completed"] = [
        (method, key) in observed
        for method, key in zip(expected.method, expected.episode_key)
    ]
    evidence_by_episode = {
        (str(row.method), str(row.episode_key)): str(row.evidence_path)
        for row in episodes.itertuples()
    }
    expected["evidence_path"] = [
        evidence_by_episode.get((method, key), "")
        for method, key in zip(expected.method, expected.episode_key)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.output_dir / "episodes.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
    paired.to_csv(args.output_dir / "paired.csv", index=False)
    expected.to_csv(args.output_dir / "completeness.csv", index=False)
    errors: list[str] = []
    warnings: list[str] = []
    configs_matched, config_rows, config_errors = _resolved_config_audit(
        args.input_root
    )
    config_audit = pd.DataFrame(config_rows)
    config_audit.to_csv(args.output_dir / "resolved_config_audit.csv", index=False)
    errors.extend(config_errors)
    if not bool(expected["completed"].all()):
        errors.append(
            "incomplete_grid:"
            f"{int(expected.completed.sum())}/{len(expected)}"
        )
    if not bool(episodes["model_identity_complete"].all()):
        errors.append("missing_actual_model_identity")
    observed_models = sorted(
        {
            model
            for value in episodes["actual_models"].astype(str)
            for model in value.split(",")
            if model
        }
    )
    if observed_models != [args.required_model]:
        errors.append(
            f"actual_model_mismatch:{observed_models}!={[args.required_model]}"
        )
    quota_markers = int(episodes["quota_marker_count"].sum())
    if quota_markers:
        errors.append(f"quota_markers:{quota_markers}")
    if not bool((episodes["simulator_resets"] == 1).all()):
        errors.append("registered_reset_count_not_one")
    method_complete = {
        method: (args.input_root / method / "COMPLETE.json").is_file()
        for method in METHODS
    }
    if not all(method_complete.values()):
        errors.append(f"missing_method_complete_marker:{method_complete}")
    manifests: dict[str, dict[str, Any]] = {}
    lifecycles: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        manifest_path = args.input_root / method / "run_manifest.json"
        lifecycle_path = (
            args.input_root / method / "shared_api_services" / "lifecycle.json"
        )
        if not manifest_path.is_file() or not lifecycle_path.is_file():
            errors.append(f"missing_manifest_or_service_lifecycle:{method}")
            continue
        manifests[method] = json.loads(manifest_path.read_text(encoding="utf-8"))
        lifecycles[method] = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        ready = float(lifecycles[method].get("ready_at_unix") or math.nan)
        first_reset = float(
            episodes.loc[episodes.method == method, "reset_time_unix"].min()
        )
        if not math.isfinite(ready) or not math.isfinite(first_reset) or ready > first_reset:
            errors.append(
                f"service_not_ready_before_first_reset:{method}:{ready}:{first_reset}"
            )
        if lifecycles[method].get("stopped_at_unix") is None:
            errors.append(f"service_lifecycle_not_closed:{method}")
    shared_manifest_fields = (
        "tasks",
        "seeds",
        "workers",
        "effective_trial_workers",
        "model",
        "hosted_model_route",
        "registered_initial_states_per_episode",
        "post_action_environment_resets",
        "generated_method_service_urls",
        "common_config_template",
    )
    stable_source_identity: dict[str, Any] = {}
    if len(manifests) == len(METHODS):
        registered_library = str(manifests["rats_90"].get("rats_library") or "")
        observed_libraries = (
            set(config_audit["rats_library_path"].astype(str))
            if "rats_library_path" in config_audit
            else set()
        )
        if observed_libraries != {registered_library}:
            errors.append(
                "resolved_config_library_mismatch:"
                f"{sorted(observed_libraries)}!={registered_library}"
            )
        for field in shared_manifest_fields:
            values = [manifests[method].get(field) for method in METHODS]
            if values[0] != values[1]:
                errors.append(f"cross_method_manifest_mismatch:{field}")
        source_paths = [
            str(manifests[method].get("robosuite_source") or "")
            for method in METHODS
        ]
        import_paths = [manifests[method].get("robosuite_imports") for method in METHODS]
        legacy_tree_hashes = [
            str(manifests[method].get("robosuite_source_tree_sha256") or "")
            for method in METHODS
        ]
        if len(set(source_paths)) != 1 or not source_paths[0]:
            errors.append("cross_method_manifest_mismatch:robosuite_source")
        elif import_paths[0] != import_paths[1]:
            errors.append("cross_method_manifest_mismatch:robosuite_imports")
        else:
            try:
                stable_source_identity = _robosuite_git_identity(Path(source_paths[0]))
            except Exception as exc:
                errors.append(f"stable_source_identity_failed:{type(exc).__name__}:{exc}")
            else:
                if not stable_source_identity.get("tracked_clean"):
                    errors.append("registered_robosuite_source_has_tracked_modifications")
                if legacy_tree_hashes[0] != legacy_tree_hashes[1]:
                    warnings.append(
                        "legacy_recursive_source_hash_mismatch_tolerated: the old field "
                        "included transient untracked/import artifacts; both methods name "
                        "the same imported files and the registered git tree is tracked-clean"
                    )
        assets = [lifecycles[method].get("model_assets") for method in METHODS]
        if assets[0] != assets[1]:
            errors.append("cross_method_service_asset_mismatch")
    audit = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "episodes": len(episodes),
        "expected_episodes": len(expected),
        "completed_episodes": int(expected.completed.sum()),
        "required_model": args.required_model,
        "actual_models": observed_models,
        "episodes_with_complete_model_identity": int(
            episodes["model_identity_complete"].sum()
        ),
        "quota_markers": quota_markers,
        "single_reset_episodes": int((episodes["simulator_resets"] == 1).sum()),
        "method_complete_markers": method_complete,
        "service_lifecycle_owners": {
            method: lifecycle.get("owner") for method, lifecycle in lifecycles.items()
        },
        "service_ready_before_first_reset": not any(
            error.startswith("service_not_ready_before_first_reset")
            for error in errors
        ),
        "shared_manifest_fields_matched": not any(
            error.startswith("cross_method_manifest_mismatch")
            for error in errors
        ),
        "shared_service_assets_matched": (
            "cross_method_service_asset_mismatch" not in errors
        ),
        "resolved_task_configs_matched_except_registered_delta": configs_matched,
        "legacy_recursive_source_hashes": (
            {
                method: manifests.get(method, {}).get("robosuite_source_tree_sha256")
                for method in METHODS
            }
        ),
        "stable_robosuite_git_identity": stable_source_identity,
        "warnings": warnings,
        "errors": errors,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise SystemExit(
            "Robosuite audit failed: " + "; ".join(errors)
        )
    with pd.ExcelWriter(args.output_dir / "robosuite_diagnostic.xlsx", engine="xlsxwriter") as writer:
        episodes.to_excel(writer, sheet_name="Episodes", index=False)
        aggregate.to_excel(writer, sheet_name="Aggregate", index=False)
        paired.to_excel(writer, sheet_name="Paired", index=False)
        expected.to_excel(writer, sheet_name="Completeness", index=False)
        config_audit.to_excel(writer, sheet_name="Resolved Config Audit", index=False)
        pd.DataFrame(
            [
                {
                    "check": key,
                    "value": (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                    ),
                }
                for key, value in audit.items()
            ]
        ).to_excel(writer, sheet_name="Audit", index=False)
    _plot(aggregate, args.output_dir / "figures" / "success_by_task")
    summary = {
        "episodes": len(episodes),
        "complete_expected": int(expected.completed.sum()),
        "total_expected": len(expected),
        "audit_status": audit["status"],
        "required_model": args.required_model,
        "actual_models": observed_models,
        "quota_markers": quota_markers,
        "scope": "executor-matched secondary diagnostic; no RACaP row",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
