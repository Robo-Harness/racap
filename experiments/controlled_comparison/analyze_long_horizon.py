#!/usr/bin/env python3
"""Normalize anytime completion, time, and model usage for the custom task."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .audit_method_results import _frozen_artifact_audit
    from .plot_style import clean_axis, save_figure
except ImportError:  # direct ``python path/to/script.py`` execution
    from audit_method_results import _frozen_artifact_audit
    from plot_style import clean_axis, save_figure

try:
    from .analyze_results import (
        COLORS,
        LABELS,
        _holm,
        _request_list_price_usd,
        _total_simulator_steps,
        _verified_evidence,
        _wilson,
    )
except ImportError:  # direct ``python path/to/script.py`` execution
    from analyze_results import (  # type: ignore[no-redef]
        COLORS,
        LABELS,
        _holm,
        _request_list_price_usd,
        _total_simulator_steps,
        _verified_evidence,
        _wilson,
    )

METHODS = ["capx", "rats_base", "rats_90", "racap_phase1", "racap_phase2"]
BUDGET_SECONDS = [300.0, 600.0]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _satisfied(event: dict[str, Any]) -> int:
    return sum(bool(row.get("satisfied")) for row in event.get("native_predicates", []))


def _usage(record: dict[str, Any]) -> tuple[int, int, int, float]:
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    if not usage:
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    cached = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached_tokens = int(cached.get("cached_tokens") or 0) if isinstance(cached, dict) else 0
    return prompt, completion, cached_tokens, _request_list_price_usd(usage)


def _timestamp(record: dict[str, Any]) -> float | None:
    for key in ("timestamp", "time"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _rats_execution_start(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Step 5: Execution$"
    )
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
    return None


def _generated_calls(task_root: Path, method: str) -> list[dict[str, Any]]:
    if method == "capx":
        return _jsonl(task_root / "llm_calls.jsonl")
    records: list[dict[str, Any]] = []
    for path in sorted((task_root / "agent_io").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and _timestamp(value) is not None:
            records.append(value)
    return records


def _metrics_from_trace(
    *,
    method: str,
    seed: int,
    events: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    resets: list[dict[str, Any]] | None = None,
    execution_start: float | None = None,
) -> list[dict[str, Any]]:
    events = sorted((row for row in events if _timestamp(row) is not None), key=_timestamp)
    calls = sorted((row for row in calls if _timestamp(row) is not None), key=_timestamp)
    resets = sorted(
        (row for row in (resets or []) if _timestamp(row) is not None),
        key=_timestamp,
    )
    if not events:
        return []
    start = float(_timestamp(events[0]))
    state_events = events
    if execution_start is not None:
        state_events = [row for row in events if float(_timestamp(row)) >= execution_start]

    first_full = next((row for row in state_events if _satisfied(row) == 7), None)
    final = state_events[-1] if state_events else events[-1]
    first_horizon = next(
        (
            row
            for row in state_events
            if int(row.get("simulator_steps") or 0) >= 8000
        ),
        None,
    )
    # A generated controller can finish its last physical action and then use
    # additional hosted-model calls for visual verification, diagnosis, or an
    # explicit stop decision.  Native state telemetry correctly remains at
    # the final physical boundary, but those later calls are still part of the
    # trajectory's runtime and cost.  Use the end of *all* public telemetry as
    # the final resource cutoff; otherwise an early final state can make the
    # reported final usage smaller than the 5/10-minute prefixes.
    trace_timestamps = [
        float(value)
        for row in (*events, *calls, *resets)
        if (value := _timestamp(row)) is not None
    ]
    trace_end = max(trace_timestamps, default=float(_timestamp(final)))
    physical_end = float(_timestamp(first_horizon or final))
    physical_elapsed = max(0.0, physical_end - start)
    post_physical_overhead = max(0.0, trace_end - physical_end)
    rows: list[dict[str, Any]] = []
    checkpoints: list[tuple[str, float]] = [("5min", 300.0), ("10min", 600.0)]
    checkpoints.append(("final", max(0.0, trace_end - start)))
    for label, elapsed in checkpoints:
        cutoff = start + elapsed
        eligible = [row for row in state_events if float(_timestamp(row)) <= cutoff]
        resource_events = [row for row in events if float(_timestamp(row)) <= cutoff]
        resource_resets = [row for row in resets if float(_timestamp(row)) <= cutoff]
        state = eligible[-1] if eligible else None
        call_prefix = [row for row in calls if float(_timestamp(row)) <= cutoff]
        prompt = completion = cached = 0
        estimated_cost = 0.0
        for call in call_prefix:
            values = _usage(call)
            prompt += values[0]
            completion += values[1]
            cached += values[2]
            estimated_cost += values[3]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "checkpoint": label,
                "budget_seconds": elapsed if label != "final" else None,
                "elapsed_seconds": elapsed,
                "completed_objects": _satisfied(state) if state else 0,
                "completion_fraction": (_satisfied(state) / 7.0) if state else 0.0,
                "full_success": bool(state and _satisfied(state) == 7),
                "time_to_full_success_seconds": (
                    float(_timestamp(first_full)) - start if first_full else None
                ),
                "simulator_horizon_reached": (
                    bool(first_horizon) if label == "final" else None
                ),
                "time_to_simulator_horizon_seconds": (
                    physical_elapsed
                    if label == "final" and first_horizon is not None
                    else None
                ),
                "physical_trajectory_end_seconds": (
                    physical_elapsed if label == "final" else None
                ),
                "post_physical_overhead_seconds": (
                    post_physical_overhead if label == "final" else None
                ),
                "model_calls": len(call_prefix),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cached_tokens": cached,
                "estimated_api_cost_usd": round(estimated_cost, 8),
                "physical_api_calls": sum(
                    str(row.get("event", "")).startswith("native_state_after_api")
                    or str(row.get("event", "")).startswith("after_tool:")
                    for row in resource_events
                ),
                "simulator_steps": _total_simulator_steps(
                    resource_events, resource_resets
                ),
            }
        )
    return rows


def _load_generated(method_root: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in range(5):
        task_root = method_root / f"seed_{seed:02d}"
        evidence = _verified_evidence(task_root, require_complete_marker=True)
        if evidence is None:
            continue
        execution_start = _rats_execution_start(task_root / "run.log") if method != "capx" else None
        metrics = _metrics_from_trace(
            method=method,
            seed=seed,
            events=_jsonl(task_root / "native_states.jsonl"),
            calls=_generated_calls(task_root, method),
            resets=_jsonl(task_root / "sim_episodes.jsonl"),
            execution_start=execution_start,
        )
        for row in metrics:
            row["evidence_path"] = str(evidence.resolve())
        rows.extend(metrics)
    return rows


def _load_racap(method_root: Path, method: str) -> list[dict[str, Any]]:
    run = method_root / "artifacts" / "libero_long_all_to_basket"
    evidence = _verified_evidence(run, require_complete_marker=False)
    if evidence is None:
        return []
    events_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calls_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    resets_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in run.glob("native_states.worker*.jsonl"):
        for row in _jsonl(path):
            events_by_key[str(row.get("episode_key", ""))].append(row)
    for path in run.glob("llm_calls.worker*.jsonl"):
        for row in _jsonl(path):
            calls_by_key[str(row.get("episode_key", ""))].append(row)
    for path in run.glob("sim_episodes.worker*.jsonl"):
        for row in _jsonl(path):
            resets_by_key[str(row.get("episode_key", ""))].append(row)
    rows: list[dict[str, Any]] = []
    for seed in range(5):
        key = f"libero_10/5/seed{seed}"
        metrics = _metrics_from_trace(
            method=method,
            seed=seed,
            events=events_by_key[key],
            calls=calls_by_key[key],
            resets=resets_by_key[key],
        )
        for row in metrics:
            row["evidence_path"] = str(evidence.resolve())
        rows.extend(metrics)
    return rows


def _bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    resamples: int = 10_000,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    samples = rng.choice(values, size=(resamples, len(values)), replace=True).mean(1)
    return (
        float(values.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _aggregate(df: pd.DataFrame, resamples: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(20260822)
    rows: list[dict[str, Any]] = []
    for (method, checkpoint), group in df.groupby(["method", "checkpoint"]):
        completed = pd.to_numeric(group["completed_objects"], errors="coerce").to_numpy()
        fraction = pd.to_numeric(group["completion_fraction"], errors="coerce").to_numpy()
        completed_mean, completed_low, completed_high = _bootstrap_mean(
            completed, rng, resamples
        )
        fraction_mean, fraction_low, fraction_high = _bootstrap_mean(
            fraction, rng, resamples
        )
        successes = int(group["full_success"].astype(bool).sum())
        wilson_low, wilson_high = _wilson(successes, len(group))
        # Permit direct unit-level use with minimal paired frames. The CLI
        # path below still requires these fields for every real report.
        physical_end = pd.to_numeric(
            group.get(
                "physical_trajectory_end_seconds",
                pd.Series(float("nan"), index=group.index),
            ),
            errors="coerce",
        )
        post_physical = pd.to_numeric(
            group.get(
                "post_physical_overhead_seconds",
                pd.Series(float("nan"), index=group.index),
            ),
            errors="coerce",
        )
        horizon_reached = group.get(
            "simulator_horizon_reached",
            pd.Series(False, index=group.index),
        ).map(lambda value: value is True)
        rows.append(
            {
                "method": method,
                "method_label": LABELS.get(method, method),
                "checkpoint": checkpoint,
                "n": len(group),
                "mean_completed_objects": completed_mean,
                "completed_objects_bootstrap_low": completed_low,
                "completed_objects_bootstrap_high": completed_high,
                "mean_completion_fraction": fraction_mean,
                "completion_fraction_bootstrap_low": fraction_low,
                "completion_fraction_bootstrap_high": fraction_high,
                "full_successes": successes,
                "full_success_rate": successes / len(group),
                "full_success_wilson_low": wilson_low,
                "full_success_wilson_high": wilson_high,
                "simulator_horizon_reached_count": int(horizon_reached.sum()),
                "mean_physical_trajectory_end_seconds": physical_end.mean(),
                "mean_post_physical_overhead_seconds": post_physical.mean(),
                "mean_model_calls": group["model_calls"].mean(),
                "mean_prompt_tokens": group["prompt_tokens"].mean(),
                "mean_completion_tokens": group["completion_tokens"].mean(),
                "mean_estimated_api_cost_usd": group[
                    "estimated_api_cost_usd"
                ].mean(),
                "mean_physical_api_calls": group["physical_api_calls"].mean(),
                "mean_simulator_steps": group["simulator_steps"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _pairwise(df: pd.DataFrame, resamples: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(20260822)
    rows: list[dict[str, Any]] = []
    for checkpoint, group in df.groupby("checkpoint"):
        present = [method for method in METHODS if method in set(group["method"])]
        for method_a, method_b in itertools.combinations(present, 2):
            left = group[group.method == method_a]
            right = group[group.method == method_b]
            merged = left.merge(
                right,
                on="seed",
                suffixes=("_a", "_b"),
                validate="one_to_one",
            )
            if merged.empty:
                continue
            delta = (
                merged["completed_objects_a"].astype(float)
                - merged["completed_objects_b"].astype(float)
            ).to_numpy()
            mean, low, high = _bootstrap_mean(delta, rng, resamples)
            success_a = merged["full_success_a"].astype(bool)
            success_b = merged["full_success_b"].astype(bool)
            a_only = int((success_a & ~success_b).sum())
            b_only = int((~success_a & success_b).sum())
            discordant = a_only + b_only
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "method_a": method_a,
                    "method_b": method_b,
                    "paired_seeds": len(merged),
                    "completed_objects_difference_a_minus_b": mean,
                    "completed_objects_bootstrap_low": low,
                    "completed_objects_bootstrap_high": high,
                    "bootstrap_unit": "paired_seed",
                    "a_only_full_success": a_only,
                    "b_only_full_success": b_only,
                    "mcnemar_exact_p": (
                        binomtest(min(a_only, b_only), discordant, 0.5).pvalue
                        if discordant
                        else 1.0
                    ),
                }
            )
    if rows:
        adjusted = _holm([float(row["mcnemar_exact_p"]) for row in rows])
        for row, value in zip(rows, adjusted):
            row["mcnemar_holm_p"] = value
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, output: Path) -> None:
    order = ["5min", "10min", "final"]
    fixed = summary[summary["checkpoint"].isin(order)].copy()
    if fixed.empty:
        return
    fig, axis = plt.subplots(figsize=(6.25, 3.8))
    fixed["checkpoint_index"] = fixed["checkpoint"].map(
        {checkpoint: index for index, checkpoint in enumerate(order)}
    )
    methods = [method for method in METHODS if method in set(fixed["method"])]
    offsets = np.linspace(-0.12, 0.12, max(1, len(methods)))
    for offset, method in zip(offsets, methods):
        values = fixed[fixed["method"] == method].sort_values("checkpoint_index")
        centre = values["mean_completed_objects"].to_numpy(float)
        low = values["completed_objects_bootstrap_low"].to_numpy(float)
        high = values["completed_objects_bootstrap_high"].to_numpy(float)
        positions = values["checkpoint_index"].to_numpy(float) + float(offset)
        axis.errorbar(
            positions,
            centre,
            yerr=[centre - low, high - centre],
            marker="o",
            markersize=5.5,
            linewidth=1.7,
            capsize=3,
            color=COLORS.get(method),
            label=LABELS.get(method, method),
        )
    axis.axhline(7.0, color="#7F8C8D", linewidth=1.0, linestyle="--", zorder=0)
    axis.text(
        1.98,
        6.88,
        "complete (7/7)",
        color="#566573",
        fontsize=8,
        ha="right",
        va="top",
    )
    axis.set(
        xlabel="Checkpoint",
        ylabel="Mean objects in basket",
        xlim=(-0.35, 2.35),
        ylim=(-0.08, 7.15),
    )
    axis.set_xticks([0.0, 1.0, 2.0], ["5 min", "10 min", "Terminal\n(≤8000 steps)"])
    clean_axis(axis, grid_axis="y")
    axis.legend(ncol=3, loc="upper left", frameon=False, columnspacing=1.0)
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "long_horizon",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "analysis" / "long_horizon",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Method subset to audit; the default retains the strict five-method grid.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_audit = (
        _frozen_artifact_audit(
            args.input_root / "rats_90" / "frozen_artifact_audit.json"
        )
        if "rats_90" in args.methods
        else {
            "status": "not_applicable",
            "reason": "rats_90 is not included in this analysis subset",
        }
    )
    (args.output_dir / "frozen_artifact_audit.json").write_text(
        json.dumps(frozen_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if "rats_90" in args.methods and frozen_audit["status"] != "pass":
        raise SystemExit(f"long-horizon frozen RATS artifact audit failed: {frozen_audit['error']}")
    rows: list[dict[str, Any]] = []
    for method in args.methods:
        method_root = args.input_root / method
        rows.extend(
            _load_generated(method_root, method)
            if method in {"capx", "rats_base", "rats_90"}
            else _load_racap(method_root, method)
        )
    df = pd.DataFrame(rows)
    required_runtime_fields = {
        "physical_trajectory_end_seconds",
        "post_physical_overhead_seconds",
        "simulator_horizon_reached",
    }
    missing_runtime_fields = sorted(required_runtime_fields - set(df.columns))
    if rows and missing_runtime_fields:
        raise SystemExit(
            "long-horizon telemetry schema is incomplete; missing: "
            + ", ".join(missing_runtime_fields)
        )
    if not df.empty:
        duplicate = df.duplicated(["method", "seed", "checkpoint"], keep=False)
        if duplicate.any():
            conflict_path = args.output_dir / "duplicate_checkpoint_rows.csv"
            df.loc[duplicate].sort_values(["method", "seed", "checkpoint"]).to_csv(
                conflict_path, index=False
            )
            raise SystemExit(
                "refusing duplicate long-horizon checkpoint rows; "
                f"conflicts written to {conflict_path}"
            )
    df.to_csv(args.output_dir / "episodes.csv", index=False)
    summary = _aggregate(df) if not df.empty else pd.DataFrame()
    pairwise = _pairwise(df) if not df.empty else pd.DataFrame()
    summary.to_csv(args.output_dir / "aggregate.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise.csv", index=False)
    expected = pd.MultiIndex.from_product(
        [args.methods, range(5), ["5min", "10min", "final"]],
        names=["method", "seed", "checkpoint"],
    ).to_frame(index=False)
    observed = set(zip(df.method, df.seed, df.checkpoint)) if not df.empty else set()
    expected["completed"] = [
        (method, seed, checkpoint) in observed
        for method, seed, checkpoint in zip(
            expected.method, expected.seed, expected.checkpoint
        )
    ]
    evidence_by_row = {
        (str(row.method), int(row.seed), str(row.checkpoint)): str(row.evidence_path)
        for row in df.itertuples()
    } if not df.empty else {}
    expected["evidence_path"] = [
        evidence_by_row.get((method, seed, checkpoint), "")
        for method, seed, checkpoint in zip(
            expected.method, expected.seed, expected.checkpoint
        )
    ]
    expected.to_csv(args.output_dir / "completeness.csv", index=False)
    if not bool(expected["completed"].all()):
        raise SystemExit(
            "incomplete long-horizon grid: "
            f"{int(expected.completed.sum())}/{len(expected)}; see completeness.csv"
        )
    with pd.ExcelWriter(args.output_dir / "long_horizon.xlsx", engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Episodes", index=False)
        summary.to_excel(writer, sheet_name="Aggregate", index=False)
        pairwise.to_excel(writer, sheet_name="Pairwise", index=False)
        expected.to_excel(writer, sheet_name="Completeness", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes(1, 0)
    _plot(summary, args.output_dir / "completion_vs_time")
    print(
        json.dumps(
            {
                "rows": len(df),
                "complete_expected": int(expected.completed.sum()),
                "total_expected": len(expected),
                "output": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
