#!/usr/bin/env python3
"""Consolidate frozen and domain-evolved Robosuite transfer experiments.

The historical CaP-X / RATS rows use the upstream generated-code artifact
layout, while RACaP uses one sealed candidate with per-episode records.  This
analyzer normalizes both layouts without treating agent claims as success:
only evaluator-owned Robosuite native predicates enter the score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.controlled_comparison.analyze_results import _holm, _usage, _wilson
from experiments.controlled_comparison.analyze_robosuite import (
    _load_method as _load_generated_method,
    _model_and_quota_audit,
)
from experiments.controlled_comparison.plot_style import clean_axis, save_figure
from experiments.controlled_comparison.run_robosuite import TASKS
from experiments.controlled_comparison.run_robosuite_racap import (
    ROBOSUITE_CARTESIAN_CONTRACT_ID,
)
from evolution.harness.robosuite import DEVELOPMENT_TASKS, HELDOUT_TASKS


METHOD_ORDER = (
    "capx",
    "rats_90",
    "racap_phase2_zero_shot",
    "rats_rs_evolved",
    "racap_rs_evolved",
)
DISPLAY = {
    "capx": "CaP-X",
    "rats_90": "RATS-90",
    "racap_phase2_zero_shot": "RACaP (LIBERO frozen)",
    "rats_rs_evolved": "RATS-RS evolved",
    "racap_rs_evolved": "RACaP-RS evolved",
}
COLORS = {
    "capx": "#8E8E8E",
    "rats_90": "#D18F00",
    "racap_phase2_zero_shot": "#4C78A8",
    "rats_rs_evolved": "#E45756",
    "racap_rs_evolved": "#2A9D8F",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load_racap_method(method_root: Path, method: str) -> list[dict[str, Any]]:
    evaluation = method_root / "evaluation"
    records = _read_jsonl(evaluation / "records.jsonl")
    if not records:
        for path in evaluation.glob("raw/*/seed*/record.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    rows: list[dict[str, Any]] = []
    for record in records:
        task = str(record.get("task") or "")
        seed = int(record.get("seed"))
        raw_root = evaluation / "raw" / task / f"seed{seed}"
        calls = _read_jsonl(raw_root / "llm_calls.jsonl")
        resets = _read_jsonl(raw_root / "sim_episodes.jsonl")
        actual_models, identity_complete, quota_markers = _model_and_quota_audit(calls)
        rows.append(
            {
                "method": method,
                "task": task,
                "seed": seed,
                "episode_key": f"{task}/seed{seed}",
                "native_success": bool(record.get("native_success")),
                "reward": float(record.get("reward") or 0.0),
                "wall_seconds": float(record.get("seconds") or math.nan),
                "turns": int(record.get("turns") or 0),
                **_usage(calls),
                "simulator_steps": int(record.get("simulator_steps") or 0),
                "simulator_resets": len(
                    [row for row in resets if row.get("event") == "environment_reset"]
                ),
                "actual_models": ",".join(actual_models),
                "model_identity_complete": identity_complete,
                "quota_marker_count": quota_markers,
                "infrastructure_error": bool(record.get("infrastructure_error")),
                "scorable": bool(record.get("scorable", True)),
                "llm_terminal_failures": int(
                    (record.get("evaluator_llm_calls") or {}).get(
                        "terminal_failures", 0
                    )
                    or 0
                ),
                "llm_quota_failures": int(
                    (record.get("evaluator_llm_calls") or {}).get(
                        "quota_failures", 0
                    )
                    or 0
                ),
                "error": str(record.get("error") or ""),
                "artifact_path": str(
                    Path(record.get("artifacts", {}).get("episode_dir") or raw_root).resolve()
                ),
                "evidence_path": str((raw_root / "record.json").resolve()),
            }
        )
    return rows


def _scope(task: str) -> str:
    if task in HELDOUT_TASKS:
        return "heldout_task_type"
    if task in DEVELOPMENT_TASKS:
        return "seen_task_type_new_seed"
    raise KeyError(task)


def _normalise_episode_attestations(episodes: pd.DataFrame) -> pd.DataFrame:
    """Apply schema defaults without erasing explicit negative attestations."""
    if "infrastructure_error" not in episodes:
        episodes["infrastructure_error"] = False
    else:
        episodes["infrastructure_error"] = episodes["infrastructure_error"].map(
            lambda value: False if pd.isna(value) else bool(value)
        )
    if "scorable" not in episodes:
        episodes["scorable"] = True
    else:
        episodes["scorable"] = episodes["scorable"].map(
            lambda value: True if pd.isna(value) else bool(value)
        )
    return episodes


def _generated_protocol_audit(
    method_root: Path,
    *,
    tasks: tuple[str, ...] = TASKS,
    seeds: tuple[int, ...] = tuple(range(5)),
) -> tuple[dict[str, Any], list[str]]:
    """Validate the registered generated-code transport/evidence contract.

    A method-level ``COMPLETE.json`` alone is intentionally insufficient.  In
    particular, an upstream provider timeout can otherwise look exactly like a
    policy failure.  Every task block must therefore attest paired model,
    native-state, reset, trace, and video evidence, and any exhausted
    mechanism-neutral transport retry invalidates the scored grid.
    """

    errors: list[str] = []
    source_path = method_root / "executor_source_audit.json"
    source: dict[str, Any] = {}
    if not source_path.is_file():
        errors.append("missing_executor_source_audit")
    else:
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("invalid_executor_source_audit")
        for field in (
            "rats_root",
            "git_commit",
            "git_status_sha256",
            "capx_python_source_sha256",
            "rats_first_party_python_source_sha256",
        ):
            if not str(source.get(field) or ""):
                errors.append(f"executor_source_missing_{field}")

    protocol_path = method_root / "transport_protocol_audit.json"
    protocol: dict[str, Any] = {}
    if not protocol_path.is_file():
        errors.append("missing_transport_protocol_audit")
    else:
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("invalid_transport_protocol_audit")
        expected_protocol = {
            "hook_installed": True,
            "maximum_attempts": 3,
            "request_invariant": True,
            "environment_resets": 0,
        }
        for field, expected in expected_protocol.items():
            if protocol.get(field) != expected:
                errors.append(
                    f"transport_protocol_{field}:{protocol.get(field)!r}!={expected!r}"
                )

    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        task_root = method_root / task
        evidence_path = task_root / "EVIDENCE.json"
        task_errors: list[str] = []
        evidence: dict[str, Any] = {}
        if not (task_root / "COMPLETE.json").is_file():
            task_errors.append("missing_complete_marker")
        if not evidence_path.is_file():
            task_errors.append("missing_evidence")
        else:
            try:
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                task_errors.append("invalid_evidence")
            if evidence.get("complete") is not True:
                task_errors.append("evidence_not_complete")
            if evidence.get("missing"):
                task_errors.append(f"evidence_missing:{evidence.get('missing')!r}")
            resets = evidence.get("registered_reset_counts") or {}
            if len(resets) != len(seeds) or any(
                int(value) != 1 for value in resets.values()
            ):
                task_errors.append(f"reset_contract:{resets!r}")
            retry = evidence.get("transport_retry") or {}
            if int(retry.get("exhaustions") or 0) != 0:
                task_errors.append("transport_retry_exhausted")
            if int(retry.get("environment_resets") or 0) != 0:
                task_errors.append("transport_retry_reset_environment")
            cardinality = evidence.get("artifact_cardinality") or {}
            for field in ("human_traces", "trial_summaries", "videos"):
                if int(cardinality.get(field) or 0) < len(seeds):
                    task_errors.append(
                        f"{field}_cardinality:"
                        f"{int(cardinality.get(field) or 0)}/{len(seeds)}"
                    )
        task_rows.append(
            {
                "task": task,
                "complete": not task_errors,
                "errors": task_errors,
                "transport_retry": evidence.get("transport_retry") or {},
                "artifact_cardinality": evidence.get("artifact_cardinality") or {},
            }
        )
        errors.extend(f"{task}:{error}" for error in task_errors)
    return {
        "executor_source": source,
        "transport_protocol": protocol,
        "tasks": task_rows,
        "complete_tasks": sum(bool(row["complete"]) for row in task_rows),
    }, errors


def _aggregate(episodes: pd.DataFrame) -> pd.DataFrame:
    frames: list[tuple[str, pd.DataFrame]] = [("ALL", episodes)]
    frames.extend((task, episodes[episodes.task == task]) for task in TASKS)
    frames.extend(
        (scope, episodes[episodes.transfer_scope == scope])
        for scope in ("seen_task_type_new_seed", "heldout_task_type")
    )
    rows: list[dict[str, Any]] = []
    for scope, frame in frames:
        for method in METHOD_ORDER:
            group = frame[frame.method == method]
            if group.empty:
                continue
            success = int(group.native_success.sum())
            low, high = _wilson(success, len(group))
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "episodes": len(group),
                    "successes": success,
                    "success_rate": success / len(group),
                    "wilson_low": low,
                    "wilson_high": high,
                    "median_wall_seconds": group.wall_seconds.median(),
                    "p90_wall_seconds": group.wall_seconds.quantile(0.9),
                    "mean_turns": group.turns.mean(),
                    "mean_model_calls": group.model_calls.mean(),
                    "mean_prompt_tokens": group.prompt_tokens.mean(),
                    "mean_completion_tokens": group.completion_tokens.mean(),
                    "mean_estimated_api_cost_usd": group.estimated_api_cost_usd.mean(),
                    "mean_simulator_steps": group.simulator_steps.mean(),
                }
            )
    return pd.DataFrame(rows)


def _paired(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260827)
    scopes = ("ALL", "seen_task_type_new_seed", "heldout_task_type")
    for left_method, right_method in combinations(METHOD_ORDER, 2):
        for scope in scopes:
            frame = episodes if scope == "ALL" else episodes[episodes.transfer_scope == scope]
            left = frame[frame.method == left_method].set_index("episode_key")
            right = frame[frame.method == right_method].set_index("episode_key")
            keys = left.index.intersection(right.index)
            if not len(keys):
                continue
            a = left.loc[keys, "native_success"].astype(bool)
            b = right.loc[keys, "native_success"].astype(bool)
            left_only = int((a & ~b).sum())
            right_only = int((~a & b).sum())
            discordant = left_only + right_only
            task_clusters = left.loc[keys, "task"].astype(str)

            def task_bootstrap(delta: pd.Series) -> tuple[float, float, float, int]:
                values = pd.to_numeric(delta, errors="coerce")
                valid = values.notna()
                task_means = (
                    pd.DataFrame(
                        {
                            "task": task_clusters.loc[valid].to_numpy(),
                            "delta": values.loc[valid].to_numpy(float),
                        }
                    )
                    .groupby("task", sort=True)["delta"]
                    .mean()
                    .to_numpy()
                )
                if not len(task_means):
                    return math.nan, math.nan, math.nan, 0
                samples = rng.choice(
                    task_means,
                    size=(10_000, len(task_means)),
                    replace=True,
                ).mean(axis=1)
                return (
                    float(task_means.mean()),
                    float(np.quantile(samples, 0.025)),
                    float(np.quantile(samples, 0.975)),
                    len(task_means),
                )

            success_mean, success_low, success_high, task_count = task_bootstrap(
                b.astype(float) - a.astype(float)
            )
            record: dict[str, Any] = {
                    "scope": scope,
                    "left_method": left_method,
                    "right_method": right_method,
                    "paired_episodes": len(keys),
                    "left_success_rate": float(a.mean()),
                    "right_success_rate": float(b.mean()),
                    "right_minus_left": float(b.mean() - a.mean()),
                    "left_only_success": left_only,
                    "right_only_success": right_only,
                    "paired_task_clusters": task_count,
                    "bootstrap_unit": "robosuite_task",
                    "right_minus_left_task_mean": success_mean,
                    "right_minus_left_task_bootstrap_low": success_low,
                    "right_minus_left_task_bootstrap_high": success_high,
                    "mcnemar_exact_p": (
                        float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
                        if discordant
                        else 1.0
                    ),
            }
            for metric in (
                "wall_seconds",
                "turns",
                "model_calls",
                "prompt_tokens",
                "completion_tokens",
                "estimated_api_cost_usd",
                "simulator_steps",
            ):
                delta = pd.to_numeric(right.loc[keys, metric], errors="coerce") - pd.to_numeric(
                    left.loc[keys, metric], errors="coerce"
                )
                mean, low, high, clusters = task_bootstrap(delta)
                record[f"{metric}_right_minus_left_task_mean"] = mean
                record[f"{metric}_right_minus_left_bootstrap_low"] = low
                record[f"{metric}_right_minus_left_bootstrap_high"] = high
                record[f"{metric}_task_clusters"] = clusters
            rows.append(record)
    if rows:
        adjusted = _holm([row["mcnemar_exact_p"] for row in rows])
        for row, value in zip(rows, adjusted):
            row["mcnemar_holm_all_reported_p"] = value
    return pd.DataFrame(rows)


def _plot(aggregate: pd.DataFrame, output: Path) -> None:
    part = aggregate[aggregate.scope == "ALL"].set_index("method").reindex(METHOD_ORDER)
    if part.empty:
        return
    fig, ax = plt.subplots(figsize=(7.15, 3.7))
    x = np.arange(len(part))
    rates = part.success_rate.to_numpy(float)
    low = rates - part.wilson_low.to_numpy(float)
    high = part.wilson_high.to_numpy(float) - rates
    bars = ax.bar(x, rates, color=[COLORS[m] for m in part.index], width=0.72)
    ax.errorbar(x, rates, yerr=np.vstack([low, high]), fmt="none", color="#222", capsize=3)
    ax.set_xticks(x, [DISPLAY[m] for m in part.index], rotation=20, ha="right")
    ax.set_ylabel("Native success rate")
    ax.set_ylim(0, max(0.45, float(np.nanmax(part.wilson_high)) + 0.12))
    clean_axis(ax, grid_axis="y")
    for bar, success, total in zip(bars, part.successes, part.episodes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                f"{int(success)}/{int(total)}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Robosuite transfer and domain evolution", loc="left", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def _plot_task_heatmap(aggregate: pd.DataFrame, output: Path) -> None:
    """Show where each frozen artifact succeeds, not only its aggregate score."""
    task_labels = {
        "cube_lifting": "Lift",
        "cube_restack": "Restack",
        "cube_stack": "Stack",
        "nut_assembly": "Nut",
        "spill_wipe": "Wipe",
        "two_arm_handover": "Handover",
        "two_arm_lift": "2-arm lift",
    }
    task_rows = aggregate[aggregate.scope.isin(TASKS)].copy()
    if task_rows.empty:
        return
    successes = (
        task_rows.pivot(index="method", columns="scope", values="successes")
        .reindex(index=METHOD_ORDER, columns=TASKS)
        .astype(float)
    )
    episodes = (
        task_rows.pivot(index="method", columns="scope", values="episodes")
        .reindex(index=METHOD_ORDER, columns=TASKS)
        .astype(float)
    )
    rates = successes / episodes

    fig, ax = plt.subplots(figsize=(7.15, 3.35))
    image = ax.imshow(rates.to_numpy(), cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(
        np.arange(len(TASKS)),
        [task_labels[task] for task in TASKS],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(METHOD_ORDER)),
        [DISPLAY[method] for method in METHOD_ORDER],
    )
    ax.tick_params(length=0)
    for row in range(len(METHOD_ORDER)):
        for column in range(len(TASKS)):
            success = successes.iat[row, column]
            total = episodes.iat[row, column]
            if not np.isfinite(success) or not np.isfinite(total):
                label = "--"
                color = "#555555"
            else:
                label = f"{int(success)}/{int(total)}"
                color = "white" if rates.iat[row, column] >= 0.58 else "#222222"
            ax.text(column, row, label, ha="center", va="center", color=color, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    colorbar.set_label("Native success rate", rotation=90)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    ax.set_title("Cross-embodiment capability profile", loc="left", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


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
        default=ROOT / "outputs" / "controlled_comparison" / "analysis" / "robosuite_transfer",
    )
    parser.add_argument("--required-model", default="gpt-5.5")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for method in ("capx", "rats_90", "rats_rs_evolved"):
        rows.extend(_load_generated_method(args.input_root / method, method))
    for method in ("racap_phase2_zero_shot", "racap_rs_evolved"):
        rows.extend(_load_racap_method(args.input_root / method, method))
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        raise SystemExit("no Robosuite transfer evidence")
    episodes["turns"] = pd.to_numeric(episodes.get("turns", 0), errors="coerce").fillna(0)
    # RACaP's current record schema writes these fields per episode.  The
    # generated-code schema predates them and establishes scorability through
    # the method/task EVIDENCE audit below.  Concatenating the two schemas
    # creates NaN in generated rows, which must retain the documented defaults
    # rather than being interpreted as an explicit negative attestation.
    episodes = _normalise_episode_attestations(episodes)
    episodes["transfer_scope"] = episodes.task.map(_scope)
    duplicate = episodes.duplicated(["method", "episode_key"], keep=False)
    if duplicate.any():
        raise SystemExit("duplicate method/episode keys in Robosuite transfer evidence")

    expected = pd.DataFrame(
        [
            {"method": method, "task": task, "seed": seed, "episode_key": f"{task}/seed{seed}"}
            for method in METHOD_ORDER
            for task in TASKS
            for seed in range(5)
        ]
    )
    observed = set(zip(episodes.method, episodes.episode_key))
    expected_pairs = set(zip(expected.method, expected.episode_key))
    expected["completed"] = [
        (method, key) in observed for method, key in zip(expected.method, expected.episode_key)
    ]
    errors: list[str] = []
    if len(episodes) != len(expected):
        errors.append(f"episode_cardinality:{len(episodes)}/{len(expected)}")
    if not expected.completed.all():
        errors.append(f"incomplete_grid:{int(expected.completed.sum())}/{len(expected)}")
    unexpected = sorted(observed - expected_pairs)
    if unexpected:
        errors.append(f"unexpected_episode_keys:{unexpected[:10]}")
    if bool(episodes.infrastructure_error.fillna(False).any()):
        errors.append("infrastructure_error_in_scored_rows")
    if not bool(episodes.scorable.fillna(False).all()):
        errors.append("unscorable_row_in_grid")
    if int(pd.to_numeric(episodes.llm_terminal_failures, errors="coerce").fillna(0).sum()):
        errors.append("terminal_hosted_model_failure_in_scored_rows")
    if int(pd.to_numeric(episodes.llm_quota_failures, errors="coerce").fillna(0).sum()):
        errors.append("quota_failure_in_scored_rows")
    if not bool((episodes.simulator_resets == 1).all()):
        errors.append("reset_count_not_one")
    if int(episodes.quota_marker_count.sum()):
        errors.append("quota_marker_present")
    missing_model_identity = (
        (pd.to_numeric(episodes.model_calls, errors="coerce").fillna(0) > 0)
        & ~episodes.model_identity_complete.fillna(False).astype(bool)
    )
    if bool(missing_model_identity.any()):
        errors.append(f"missing_actual_model_identity:{int(missing_model_identity.sum())}")
    actual_models = sorted(
        {model for value in episodes.actual_models.astype(str) for model in value.split(",") if model}
    )
    if actual_models != [args.required_model]:
        errors.append(f"actual_model_mismatch:{actual_models}")
    complete_markers = {
        method: (args.input_root / method / "COMPLETE.json").is_file()
        for method in METHOD_ORDER
    }
    if not all(complete_markers.values()):
        errors.append(f"missing_method_complete_marker:{complete_markers}")
    generated_protocol_audits: dict[str, Any] = {}
    for method in ("capx", "rats_90", "rats_rs_evolved"):
        generated_audit, generated_errors = _generated_protocol_audit(
            args.input_root / method
        )
        generated_protocol_audits[method] = generated_audit
        errors.extend(f"{method}:{error}" for error in generated_errors)
    for field in (
        "rats_root",
        "git_commit",
        "git_status_sha256",
        "capx_python_source_sha256",
        "rats_first_party_python_source_sha256",
    ):
        values = {
            str((audit.get("executor_source") or {}).get(field) or "")
            for audit in generated_protocol_audits.values()
        }
        if len(values) != 1 or "" in values:
            errors.append(f"generated_executor_source_mismatch:{field}:{sorted(values)!r}")

    frozen_audits: dict[str, Any] = {}
    for method in ("racap_phase2_zero_shot", "racap_rs_evolved"):
        manifest_path = args.input_root / method / "run_manifest.json"
        if not manifest_path.is_file():
            errors.append(f"missing_run_manifest:{method}")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            contract = manifest.get("robosuite_cartesian_contract") or {}
            if contract.get("id") != ROBOSUITE_CARTESIAN_CONTRACT_ID:
                errors.append(f"stale_cartesian_contract:{method}")
        path = args.input_root / method / "frozen_artifact_audit.json"
        if not path.is_file():
            errors.append(f"missing_frozen_artifact_audit:{method}")
            continue
        frozen_audits[method] = json.loads(path.read_text(encoding="utf-8"))
        if not frozen_audits[method].get("unchanged"):
            errors.append(f"frozen_artifact_changed:{method}")
    for method in ("rats_90", "rats_rs_evolved"):
        manifest_path = args.input_root / method / "run_manifest.json"
        if not manifest_path.is_file():
            errors.append(f"missing_run_manifest:{method}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        library_text = str(manifest.get("rats_library") or "")
        expected_sha256 = str(manifest.get("rats_library_sha256") or "")
        library = Path(library_text) if library_text else Path("/__missing__")
        current_sha256 = _sha256(library) if library.is_file() else ""
        unchanged = bool(expected_sha256) and current_sha256 == expected_sha256
        frozen_audits[method] = {
            "artifact": library_text,
            "before_sha256": expected_sha256,
            "after_sha256": current_sha256,
            "unchanged": unchanged,
        }
        if not unchanged:
            errors.append(f"frozen_artifact_changed:{method}")

    aggregate = _aggregate(episodes)
    paired = _paired(episodes)
    audit = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "episodes": len(episodes),
        "expected_episodes": len(expected),
        "actual_models": actual_models,
        "single_reset_episodes": int((episodes.simulator_resets == 1).sum()),
        "method_complete_markers": complete_markers,
        "generated_protocol_audits": generated_protocol_audits,
        "racap_frozen_artifact_audits": frozen_audits,
        "frozen_artifact_audits": frozen_audits,
        "errors": errors,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.output_dir / "episodes.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
    paired.to_csv(args.output_dir / "paired.csv", index=False)
    expected.to_csv(args.output_dir / "completeness.csv", index=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with pd.ExcelWriter(args.output_dir / "robosuite_transfer.xlsx", engine="xlsxwriter") as writer:
        episodes.to_excel(writer, sheet_name="Episodes", index=False)
        aggregate.to_excel(writer, sheet_name="Aggregate", index=False)
        paired.to_excel(writer, sheet_name="Paired", index=False)
        expected.to_excel(writer, sheet_name="Completeness", index=False)
        pd.DataFrame([{"check": key, "value": json.dumps(value) if isinstance(value, (list, dict)) else value}
                      for key, value in audit.items()]).to_excel(writer, sheet_name="Audit", index=False)
    _plot(aggregate, args.output_dir / "figures" / "robosuite_transfer_success")
    _plot_task_heatmap(
        aggregate,
        args.output_dir / "figures" / "robosuite_task_success_heatmap",
    )
    if errors:
        raise SystemExit("Robosuite transfer audit failed: " + "; ".join(errors))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
