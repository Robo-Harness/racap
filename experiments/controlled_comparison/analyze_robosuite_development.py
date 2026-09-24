#!/usr/bin/env python3
"""Audit and summarize compute-matched RACaP-RS / RATS-RS development.

The analysis is intentionally independent of the sealed evaluation grid.  It
accepts only the five registered development tasks at seeds 100--102, requires
the preregistered complete-sweep budgets (15 RACaP and 3 RATS candidates), and
treats only evaluator-owned native success as a promotion signal.  Unequal
trajectory counts and every measured resource are reported explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolution.harness.robosuite import DEVELOPMENT_SEEDS, DEVELOPMENT_TASKS
from experiments.controlled_comparison.analyze_results import _usage
from experiments.controlled_comparison.plot_style import clean_axis, save_figure
from experiments.controlled_comparison.analyze_robosuite_transfer import (
    _generated_protocol_audit,
)


METHODS = ("racap_rs_evolved", "rats_rs_evolved")
DISPLAY = {
    "racap_rs_evolved": "RACaP-RS",
    "rats_rs_evolved": "RATS-RS",
}
COLORS = {
    "racap_rs_evolved": "#2A9D8F",
    "rats_rs_evolved": "#E45756",
}
VALID_RACAP_STATUSES = {
    "capability_promoted",
    "efficiency_recorded",
    "retained_not_promoted",
}
EXPECTED_KEYS = {
    f"robosuite/{task}/seed{seed}"
    for task in DEVELOPMENT_TASKS
    for seed in DEVELOPMENT_SEEDS
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _telemetry_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.extend(_read_jsonl(path))
    return rows


def _agent_io_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/agent_io/*.json")):
        try:
            row = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("actual_model") or row.get("response"):
            rows.append(row)
    return rows


def _usage_for_root(
    root: Path,
    *,
    include_agent_io: bool = False,
    extra_telemetry: Iterable[Path] = (),
) -> dict[str, Any]:
    rows = _telemetry_rows([*root.glob("**/llm_calls.jsonl"), *extra_telemetry])
    if include_agent_io:
        known = {str(row.get("request_id") or "") for row in rows}
        rows.extend(
            row
            for row in _agent_io_rows(root)
            if not row.get("request_id") or str(row.get("request_id")) not in known
        )
    return _usage(rows)


def _json_seconds(paths: Iterable[Path]) -> float:
    """Sum process clocks from immutable launcher result records."""

    total = 0.0
    for path in paths:
        try:
            value = float(_read_json(path).get("seconds") or 0.0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if np.isfinite(value) and value >= 0.0:
            total += value
    return float(total)


def _resource_summary(
    episodes: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    racap_root: Path,
    rats_root: Path,
) -> pd.DataFrame:
    """Disclose physical, process, and hosted-model development resources.

    Episode wall times are additive service demand and may overlap across
    workers. Evaluation process wall time is the elapsed time of each parallel
    sweep. Model latency may also overlap for concurrent critic/runtime calls,
    so it is intentionally not added to process wall time.
    """

    racap_state_path = racap_root / "FINAL_STATE.json"
    if not racap_state_path.is_file():
        racap_state_path = racap_root / "lineage" / "state.json"
    racap_state = _read_json(racap_state_path)
    stage_states = racap_state.get("stage_states") or {}
    racap_evaluation_wall = float(
        (stage_states.get("rs_cross_embodiment") or {}).get("evaluation_seconds")
        or 0.0
    )

    # RATS evaluates task families and sweeps sequentially; each task launcher
    # runs its three registered seeds concurrently. Summing immutable task
    # COMPLETE clocks is therefore the effective evaluation wall time.
    rats_evaluation_wall = _json_seconds(
        rats_root.glob("sweeps/sweep_*/*/COMPLETE.json")
    )
    rats_extraction_wall = _json_seconds(
        [
            *rats_root.glob("candidate_libraries/**/extraction_process.json"),
            *rats_root.glob("invalidated_extractions/**/extraction_process.json"),
        ]
    )

    usages = {
        "racap_rs_evolved": _usage_for_root(
            racap_root,
            include_agent_io=True,
            extra_telemetry=(racap_root / "evolution_llm_calls.jsonl",),
        ),
        "rats_rs_evolved": _usage_for_root(rats_root, include_agent_io=True),
    }
    rows: list[dict[str, Any]] = []
    for method, root in (
        ("racap_rs_evolved", racap_root),
        ("rats_rs_evolved", rats_root),
    ):
        part = episodes[episodes.method == method]
        cohort_part = candidates[candidates.method == method]
        evaluation_wall = (
            racap_evaluation_wall
            if method == "racap_rs_evolved"
            else rats_evaluation_wall
        )
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY[method],
                "development_sweeps": int(len(cohort_part)),
                "runtime_candidate_sweeps": int(max(0, len(cohort_part) - 1)),
                "development_episodes": int(len(part)),
                "native_success_observations": int(part.native_success.sum()),
                "total_simulator_steps": int(part.simulator_steps.sum()),
                "total_episode_wall_seconds": float(part.wall_seconds.sum()),
                "evaluation_process_wall_seconds": evaluation_wall,
                "skill_extraction_process_wall_seconds": (
                    rats_extraction_wall if method == "rats_rs_evolved" else 0.0
                ),
                "wall_clock_definition": (
                    "sum of sequential parallel-sweep launcher clocks; "
                    "skill extraction reported separately"
                ),
                **usages[method],
                "experiment_root": str(root.resolve()),
            }
        )
    return pd.DataFrame(rows)


def _key_parts(key: str) -> tuple[str, int]:
    prefix = "robosuite/"
    if not key.startswith(prefix) or "/seed" not in key:
        raise ValueError(f"invalid development episode key: {key}")
    body = key[len(prefix) :]
    task, seed_text = body.rsplit("/seed", 1)
    return task, int(seed_text)


def _normalize_episode(
    row: dict[str, Any],
    *,
    method: str,
    cohort_index: int,
    cohort_label: str,
    proposal_iteration: int,
) -> dict[str, Any]:
    key = str(row.get("key") or row.get("episode_key") or "")
    task, seed = _key_parts(key)
    return {
        "method": method,
        "cohort_index": cohort_index,
        "cohort_label": cohort_label,
        "proposal_iteration": proposal_iteration,
        "episode_key": key,
        "task": task,
        "seed": seed,
        "native_success": bool(row.get("native_success")),
        "reward": float(row.get("reward") or 0.0),
        "simulator_steps": int(row.get("simulator_steps") or 0),
        "wall_seconds": float(row.get("seconds") or row.get("wall_seconds") or 0.0),
        "agent_success": bool(row.get("agent_success") or row.get("agent_claimed")),
        "scorable": bool(row.get("scorable", True)),
        "infrastructure_error": bool(row.get("infrastructure_error", False)),
        "abort_reason": str(row.get("abort_reason") or ""),
        "simulator_resets": int(row.get("_evidence_simulator_resets") or 0),
        "artifact_bundle_complete": bool(
            row.get("_evidence_artifact_bundle_complete", False)
        ),
        "model_telemetry_complete": bool(
            row.get("_evidence_model_telemetry_complete", False)
        ),
        "llm_terminal_failures": int(
            (row.get("evaluator_llm_calls") or {}).get("terminal_failures") or 0
        ),
        "llm_quota_failures": int(
            (row.get("evaluator_llm_calls") or {}).get("quota_failures") or 0
        ),
    }


def _racap_rollout_rows(output: Path) -> list[dict[str, Any]]:
    records = _read_jsonl(output / "records.jsonl")
    if not records:
        raise RuntimeError(f"missing RACaP records: {output}")
    for record in records:
        task = str(record.get("task") or "")
        seed = int(record.get("seed"))
        raw = output / "raw" / task / f"seed{seed}"
        reset_rows = _read_jsonl(raw / "sim_episodes.jsonl")
        record["_evidence_simulator_resets"] = sum(
            row.get("event") == "environment_reset" for row in reset_rows
        )
        artifacts = record.get("artifacts") or {}
        retained = [
            Path(str(artifacts.get(field) or ""))
            for field in ("video", "trace", "trajectory")
        ]
        record["_evidence_artifact_bundle_complete"] = all(
            path.is_file() and path.stat().st_size > 0 for path in retained
        )
        logical_calls = int(
            (record.get("evaluator_llm_calls") or {}).get("logical_calls") or 0
        )
        telemetry = raw / "llm_calls.jsonl"
        record["_evidence_model_telemetry_complete"] = (
            logical_calls == 0 or (telemetry.is_file() and telemetry.stat().st_size > 0)
        )
    return records


def _rats_sweep_rows(sweep: Path) -> list[dict[str, Any]]:
    last: dict[str, dict[str, Any]] = {}
    for task in DEVELOPMENT_TASKS:
        for row in _read_jsonl(sweep / task / "native_states.jsonl"):
            if row.get("event") == "native_state_after_code":
                key = str(row.get("episode_key") or "")
                if key:
                    last[key] = row
    if not last:
        raise RuntimeError(f"missing RATS native records: {sweep}")
    return list(last.values())


def _cohort_summary(
    episodes: list[dict[str, Any]],
    *,
    method: str,
    cohort_index: int,
    cohort_label: str,
    proposal_iteration: int,
    status: str,
    promoted: bool,
    parent_native_success: int,
    champion_native_success_after: int,
    output: Path,
    include_agent_io: bool = False,
) -> dict[str, Any]:
    usage = _usage_for_root(output, include_agent_io=include_agent_io)
    native = sum(int(row["native_success"]) for row in episodes)
    return {
        "method": method,
        "cohort_index": cohort_index,
        "cohort_label": cohort_label,
        "proposal_iteration": proposal_iteration,
        "status": status,
        "promoted": promoted,
        "native_success": native,
        "native_rate": native / len(episodes),
        "parent_native_success": parent_native_success,
        "native_delta": native - parent_native_success,
        "champion_native_success_after": champion_native_success_after,
        "episodes": len(episodes),
        "mean_simulator_steps": float(
            np.mean([row["simulator_steps"] for row in episodes])
        ),
        "total_simulator_steps": int(
            sum(row["simulator_steps"] for row in episodes)
        ),
        "total_episode_wall_seconds": float(
            sum(row["wall_seconds"] for row in episodes)
        ),
        "output_dir": str(output.resolve()),
        **usage,
    }


def _load_racap(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (root / "COMPLETE.json").is_file():
        raise RuntimeError(f"RACaP-RS evolution is incomplete: {root}")
    state_path = root / "FINAL_STATE.json"
    if not state_path.is_file():
        state_path = root / "lineage" / "state.json"
    state = _read_json(state_path)
    if int(state.get("runtime_candidates") or 0) != 15:
        raise RuntimeError("RACaP-RS does not contain 15 runtime candidates")
    baselines = [
        path
        for path in (root / "rollouts").glob("i000_*")
        if (path / "COMPLETE.json").is_file() and (path / "records.jsonl").is_file()
    ]
    if len(baselines) != 1:
        raise RuntimeError(f"RACaP-RS baseline cardinality is {len(baselines)}, expected 1")
    completed = [
        row
        for row in state.get("history", [])
        if row.get("status") in VALID_RACAP_STATUSES and row.get("candidate_metrics")
    ]
    if len(completed) != 15:
        raise RuntimeError(
            f"RACaP-RS complete candidate cardinality is {len(completed)}, expected 15"
        )

    episode_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    baseline_raw = _racap_rollout_rows(baselines[0])
    baseline = [
        _normalize_episode(
            row,
            method="racap_rs_evolved",
            cohort_index=0,
            cohort_label="baseline",
            proposal_iteration=0,
        )
        for row in baseline_raw
    ]
    episode_rows.extend(baseline)
    baseline_native = sum(int(row["native_success"]) for row in baseline)
    candidate_rows.append(
        _cohort_summary(
            baseline,
            method="racap_rs_evolved",
            cohort_index=0,
            cohort_label="baseline",
            proposal_iteration=0,
            status="baseline",
            promoted=True,
            parent_native_success=baseline_native,
            champion_native_success_after=baseline_native,
            output=baselines[0],
        )
    )
    champion = baseline_native
    for cohort_index, history in enumerate(completed, start=1):
        output = Path(str(history["output_dir"]))
        raw = _racap_rollout_rows(output)
        proposal_iteration = int(history.get("iteration") or cohort_index)
        normalized = [
            _normalize_episode(
                row,
                method="racap_rs_evolved",
                cohort_index=cohort_index,
                cohort_label=f"candidate_{cohort_index:03d}",
                proposal_iteration=proposal_iteration,
            )
            for row in raw
        ]
        episode_rows.extend(normalized)
        native = sum(int(row["native_success"]) for row in normalized)
        promoted = history.get("status") == "capability_promoted"
        parent = int((history.get("champion_metrics") or {}).get("native_success", champion))
        if promoted:
            champion = native
        candidate_rows.append(
            _cohort_summary(
                normalized,
                method="racap_rs_evolved",
                cohort_index=cohort_index,
                cohort_label=f"candidate_{cohort_index:03d}",
                proposal_iteration=proposal_iteration,
                status=str(history.get("status")),
                promoted=promoted,
                parent_native_success=parent,
                champion_native_success_after=champion,
                output=output,
            )
        )
    return episode_rows, candidate_rows


def _load_rats(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (root / "COMPLETE.json").is_file():
        raise RuntimeError(f"RATS-RS evolution is incomplete: {root}")
    state = _read_json(root / "state.json")
    manifest = _read_json(root / "run_manifest.json")
    iterations = int(manifest.get("iterations") or 0)
    history = list(state.get("history") or [])
    if iterations != 3:
        raise RuntimeError(f"RATS-RS manifest has {iterations} candidates, expected 3")
    if len(history) != iterations or int(state.get("next_iteration") or 0) != iterations + 1:
        raise RuntimeError(
            f"RATS-RS does not contain exactly {iterations} completed candidates"
        )
    episode_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    champion = 0
    registered_executor_identity: tuple[str, ...] | None = None
    for cohort_index in range(iterations + 1):
        label = "sweep_000_baseline" if cohort_index == 0 else f"sweep_{cohort_index:03d}_candidate"
        sweep = root / "sweeps" / label
        if not (sweep / "COMPLETE.json").is_file():
            raise RuntimeError(f"incomplete RATS-RS sweep: {sweep}")
        protocol_audit, protocol_errors = _generated_protocol_audit(
            sweep,
            tasks=tuple(DEVELOPMENT_TASKS),
            seeds=tuple(DEVELOPMENT_SEEDS),
        )
        if protocol_errors:
            raise RuntimeError(
                f"RATS-RS sweep evidence failed for {sweep}: "
                + "; ".join(protocol_errors)
            )
        source = protocol_audit.get("executor_source") or {}
        executor_identity = tuple(
            str(source.get(field) or "")
            for field in (
                "rats_root",
                "git_commit",
                "git_status_sha256",
                "capx_python_source_sha256",
                "rats_first_party_python_source_sha256",
            )
        )
        if registered_executor_identity is None:
            registered_executor_identity = executor_identity
        elif executor_identity != registered_executor_identity:
            raise RuntimeError(
                "RATS-RS executor source changed between development sweeps"
            )
        proposal_iteration = cohort_index
        raw = _rats_sweep_rows(sweep)
        normalized = [
            _normalize_episode(
                row,
                method="rats_rs_evolved",
                cohort_index=cohort_index,
                cohort_label="baseline" if cohort_index == 0 else f"candidate_{cohort_index:03d}",
                proposal_iteration=proposal_iteration,
            )
            for row in raw
        ]
        episode_rows.extend(normalized)
        native = sum(int(row["native_success"]) for row in normalized)
        if cohort_index == 0:
            status = "baseline"
            promoted = True
            parent = native
            champion = native
        else:
            decision = history[cohort_index - 1]
            promoted = bool(decision.get("promoted"))
            status = "capability_promoted" if promoted else "retained_not_promoted"
            parent = int(
                (decision.get("parent_champion_metrics") or {}).get(
                    "native_success", champion
                )
            )
            if promoted:
                champion = native
        candidate_rows.append(
            _cohort_summary(
                normalized,
                method="rats_rs_evolved",
                cohort_index=cohort_index,
                cohort_label="baseline" if cohort_index == 0 else f"candidate_{cohort_index:03d}",
                proposal_iteration=proposal_iteration,
                status=status,
                promoted=promoted,
                parent_native_success=parent,
                champion_native_success_after=champion,
                output=sweep,
                include_agent_io=True,
            )
        )
    return episode_rows, candidate_rows


def _invalidated_attempts(method: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for marker in sorted(root.glob("**/NOT_SCORED.json")):
        attempt = marker.parent
        payload = _read_json(marker)
        usage = _usage_for_root(attempt, include_agent_io=True)
        artifact_bytes = sum(
            path.stat().st_size
            for path in attempt.rglob("*")
            if path.is_file()
        )
        rows.append(
            {
                "method": method,
                "attempt_path": str(attempt.resolve()),
                "reason": str(payload.get("reason") or "explicitly invalidated"),
                "formal_score_use": False,
                "artifact_bytes": artifact_bytes,
                "videos": len(list(attempt.glob("**/*.mp4"))),
                "trajectories": len(list(attempt.glob("**/trajectory.json"))),
                **usage,
            }
        )
    return rows


def _audit(
    episodes: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    racap_root: Path,
    rats_root: Path,
    invalidated: pd.DataFrame | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_sweeps = {"racap_rs_evolved": 16, "rats_rs_evolved": 4}
    expected_episodes_by_method = {
        method: sweeps * len(EXPECTED_KEYS) for method, sweeps in expected_sweeps.items()
    }
    expected_episodes = sum(expected_episodes_by_method.values())
    expected_cohorts = sum(expected_sweeps.values())
    if len(episodes) != expected_episodes:
        errors.append(
            f"development_episode_cardinality:{len(episodes)}/{expected_episodes}"
        )
    if len(candidates) != expected_cohorts:
        errors.append(
            f"development_cohort_cardinality:{len(candidates)}/{expected_cohorts}"
        )
    duplicates = episodes.duplicated(["method", "cohort_index", "episode_key"], keep=False)
    if bool(duplicates.any()):
        errors.append(f"duplicate_episode_keys:{int(duplicates.sum())}")
    for method in METHODS:
        part = episodes[episodes.method == method]
        method_expected = expected_episodes_by_method[method]
        if len(part) != method_expected:
            errors.append(f"{method}_episodes:{len(part)}/{method_expected}")
        cohort_counts = part.groupby("cohort_index").size().to_dict()
        expected_counts = {
            index: len(EXPECTED_KEYS) for index in range(expected_sweeps[method])
        }
        if cohort_counts != expected_counts:
            errors.append(f"{method}_cohort_counts:{cohort_counts}")
        for index, block in part.groupby("cohort_index"):
            if set(block.episode_key.astype(str)) != EXPECTED_KEYS:
                errors.append(f"{method}_wrong_keys_cohort_{index}")
    if not bool(episodes.scorable.astype(bool).all()):
        errors.append("unscorable_development_episode")
    if bool(episodes.infrastructure_error.astype(bool).any()):
        errors.append("infrastructure_error_in_development")
    if episodes.abort_reason.astype(str).str.contains("quota", case=False).any():
        errors.append("quota_marker_in_scored_development")
    racap_episodes = episodes[episodes.method == "racap_rs_evolved"]
    if not bool((racap_episodes.simulator_resets == 1).all()):
        errors.append("racap_rs_evolved_reset_count_not_one")
    if not bool(racap_episodes.artifact_bundle_complete.astype(bool).all()):
        errors.append("racap_rs_evolved_incomplete_artifact_bundle")
    if not bool(racap_episodes.model_telemetry_complete.astype(bool).all()):
        errors.append("racap_rs_evolved_incomplete_model_telemetry")
    if int(racap_episodes.llm_terminal_failures.sum()) != 0:
        errors.append("racap_rs_evolved_terminal_model_failure")
    if int(racap_episodes.llm_quota_failures.sum()) != 0:
        errors.append("racap_rs_evolved_quota_failure")
    for row in candidates.itertuples(index=False):
        if row.cohort_index == 0:
            continue
        strict_gain = int(row.native_success) > int(row.parent_native_success)
        if bool(row.promoted) != strict_gain:
            errors.append(
                f"promotion_rule_violation:{row.method}:candidate_{row.cohort_index:03d}"
            )
    actual_models = sorted(
        {
            model
            for value in candidates.actual_models.fillna("").astype(str)
            for model in value.split(",")
            if model
        }
    )
    if actual_models != ["gpt-5.5"]:
        errors.append(f"runtime_model_identity:{actual_models}")
    for method, root in (
        ("racap_rs_evolved", racap_root),
        ("rats_rs_evolved", rats_root),
    ):
        manifest_name = "protocol_manifest.json" if method.startswith("racap") else "run_manifest.json"
        manifest = _read_json(root / manifest_name)
        if tuple(manifest.get("development_tasks") or ()) != tuple(DEVELOPMENT_TASKS):
            errors.append(f"{method}_development_task_manifest")
        if tuple(manifest.get("development_seeds") or ()) != tuple(DEVELOPMENT_SEEDS):
            errors.append(f"{method}_development_seed_manifest")
        if int(manifest.get("workers") or 0) != 3:
            errors.append(f"{method}_worker_manifest")
        physical = manifest.get("physical_budget") or {}
        total = int(physical.get("total_development_episodes") or 0)
        if total != expected_episodes_by_method[method]:
            errors.append(
                f"{method}_physical_budget:{total}/{expected_episodes_by_method[method]}"
            )
        if method == "rats_rs_evolved" and manifest.get("budget_basis") != (
            "approximately_equal_effective_development_wall_time"
        ):
            errors.append("rats_rs_evolved_budget_basis")
        if not (root / "frozen_champion").is_dir():
            errors.append(f"{method}_missing_frozen_champion")
    return {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "methods": list(METHODS),
        "episodes": len(episodes),
        "expected_episodes": expected_episodes,
        "cohorts": len(candidates),
        "expected_cohorts": expected_cohorts,
        "physical_episodes_by_method": expected_episodes_by_method,
        "candidate_sweeps_by_method": {
            method: count - 1 for method, count in expected_sweeps.items()
        },
        "budget_basis": "approximately_equal_effective_development_wall_time",
        "episodes_per_sweep": 15,
        "actual_runtime_models": actual_models,
        "invalidated_attempts_retained": (
            0 if invalidated is None else len(invalidated)
        ),
        "invalidated_hosted_calls": (
            0
            if invalidated is None or invalidated.empty
            else int(invalidated.model_calls.sum())
        ),
        "errors": errors,
    }


def _plot(candidates: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.75))
    for method in METHODS:
        part = candidates[candidates.method == method].sort_values("cohort_index")
        x = part.cohort_index.to_numpy(int)
        candidate = part.native_success.to_numpy(float) / 15.0
        champion = part.champion_native_success_after.to_numpy(float) / 15.0
        color = COLORS[method]
        ax.plot(
            x,
            candidate,
            color=color,
            alpha=0.42,
            linewidth=1.2,
            marker="o",
            markersize=3,
            label=f"{DISPLAY[method]} candidate",
        )
        ax.step(
            x,
            champion,
            where="post",
            color=color,
            linewidth=2.4,
            label=f"{DISPLAY[method]} champion",
        )
    maximum = int(candidates.cohort_index.max())
    ax.set_xlim(0, maximum)
    ax.set_xticks(range(0, maximum + 1, 3))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Simulator-tested candidate sweep")
    ax.set_ylabel("Development native success")
    ax.set_title("Compute-matched Robosuite domain evolution", loc="left", fontweight="bold")
    clean_axis(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--racap-root",
        type=Path,
        default=ROOT / "outputs" / "evolution" / "robosuite_racap_rs_15",
    )
    parser.add_argument(
        "--rats-root",
        type=Path,
        default=ROOT / "outputs" / "evolution" / "robosuite_rats_rs_time3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "controlled_comparison"
        / "analysis"
        / "robosuite_development",
    )
    args = parser.parse_args()
    racap_episodes, racap_candidates = _load_racap(args.racap_root)
    rats_episodes, rats_candidates = _load_rats(args.rats_root)
    episodes = pd.DataFrame([*racap_episodes, *rats_episodes])
    candidates = pd.DataFrame([*racap_candidates, *rats_candidates])
    resources = _resource_summary(
        episodes,
        candidates,
        racap_root=args.racap_root,
        rats_root=args.rats_root,
    )
    invalidated = pd.DataFrame(
        [
            *_invalidated_attempts("racap_rs_evolved", args.racap_root),
            *_invalidated_attempts("rats_rs_evolved", args.rats_root),
        ]
    )
    audit = _audit(
        episodes,
        candidates,
        racap_root=args.racap_root,
        rats_root=args.rats_root,
        invalidated=invalidated,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.output_dir / "episodes.csv", index=False)
    candidates.to_csv(args.output_dir / "candidates.csv", index=False)
    resources.to_csv(args.output_dir / "resources.csv", index=False)
    invalidated.to_csv(args.output_dir / "invalidated_attempts.csv", index=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    audit_rows = [
        {
            "check": key,
            "value": json.dumps(value, ensure_ascii=False)
            if isinstance(value, (list, dict))
            else value,
        }
        for key, value in audit.items()
    ]
    with pd.ExcelWriter(
        args.output_dir / "robosuite_development.xlsx", engine="xlsxwriter"
    ) as writer:
        episodes.to_excel(writer, sheet_name="Episodes", index=False)
        candidates.to_excel(writer, sheet_name="Candidates", index=False)
        resources.to_excel(writer, sheet_name="Resources", index=False)
        invalidated.to_excel(writer, sheet_name="Invalidated Attempts", index=False)
        pd.DataFrame(audit_rows).to_excel(writer, sheet_name="Audit", index=False)
    _plot(candidates, args.output_dir / "figures" / "robosuite_development_curve")
    if audit["errors"]:
        raise SystemExit("Robosuite development audit failed: " + "; ".join(audit["errors"]))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
