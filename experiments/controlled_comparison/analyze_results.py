#!/usr/bin/env python3
"""Normalize raw method artifacts, compute paired statistics, and export paper data."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

try:
    from .plot_style import METHOD_COLORS, clean_axis, save_figure
except ImportError:  # direct ``python path/to/script.py`` execution
    from plot_style import METHOD_COLORS, clean_axis, save_figure

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ["capx", "rats_base", "rats_90", "racap_phase1", "racap_phase2"]
MAIN_COHORTS = [
    "libero90_id_replay",
    "libero_pro_zero_shot",
    "libero_base_diagnostic",
    "libero_long",
]
LABELS = {
    "capx": "CaP-X",
    "rats_base": "RATS-base",
    "rats_90": "RATS-90",
    "racap_phase1": "RACaP-Phase 1",
    "racap_phase2": "RACaP-Phase 2",
}
COLORS = METHOD_COLORS
PRICE_SNAPSHOT = yaml.safe_load((HERE / "pricing_snapshot.yaml").read_text(encoding="utf-8"))


def _request_list_price_usd(usage: dict[str, Any]) -> float:
    """Estimate one request at the frozen OpenAI list-price snapshot.

    The relay does not return an actual billed amount, so this deliberately
    remains separate from provider billing.  Computing request by request is
    important because the long-context multiplier is session-local.
    """

    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    details = usage.get("prompt_tokens_details") or {}
    cached = min(prompt, int(details.get("cached_tokens") or usage.get("cached_tokens") or 0))
    uncached = prompt - cached
    rates = PRICE_SNAPSHOT["short_context"]
    multipliers = PRICE_SNAPSHOT["long_context_multiplier"]
    is_long = prompt > int(PRICE_SNAPSHOT["long_context_threshold_input_tokens"])
    input_multiplier = float(multipliers["input"]) if is_long else 1.0
    cached_multiplier = float(multipliers["cached_input"]) if is_long else 1.0
    output_multiplier = float(multipliers["output"]) if is_long else 1.0
    return (
        uncached * float(rates["input"]) * input_multiplier
        + cached * float(rates["cached_input"]) * cached_multiplier
        + completion * float(rates["output"]) * output_multiplier
    ) / 1_000_000.0


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _normalize_instruction(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _verified_evidence(root: Path, *, require_complete_marker: bool) -> Path | None:
    """Return the evidence ledger only for a scoreable artifact root."""

    if require_complete_marker and not (root / "COMPLETE.json").is_file():
        return None
    evidence = root / "EVIDENCE.json"
    if not evidence.is_file() or evidence.stat().st_size <= 0:
        return None
    try:
        payload = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return evidence if payload.get("complete") is True else None


def _artifact_wall_seconds(task_root: Path) -> float:
    """Return the generated-method launch-to-artifact wall time.

    ``run_libero.py`` writes this duration into the per-episode COMPLETE
    transaction only after the child process has exited and its evidence gate
    has passed.  It therefore complements policy telemetry: the latter stops
    at the terminal simulator/model event, whereas this clock also includes
    environment startup and the child's video/trace serialization.  RACaP's
    batched runner cannot currently separate this overhead per episode, so its
    rows deliberately remain NaN rather than receiving an imputed value.
    """

    marker = task_root / "COMPLETE.json"
    if not marker.is_file():
        return math.nan
    try:
        raw = json.loads(marker.read_text(encoding="utf-8"))
        seconds = float(raw.get("seconds"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return math.nan
    return seconds if math.isfinite(seconds) and seconds >= 0.0 else math.nan


def _episode_parts(key: str) -> tuple[str, int, int] | None:
    match = re.fullmatch(r"(.+)/(\d+)/seed(-?\d+)", key or "")
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def _predicate_fraction(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(bool(row.get("satisfied")) for row in rows) / len(rows)


def _total_simulator_steps(
    states: list[dict[str, Any]], resets: list[dict[str, Any]]
) -> int:
    """Sum the final step counter from every simulator reset in an episode."""
    if not states:
        return 0
    boundaries = sorted(float(row.get("time") or 0.0) for row in resets)
    maxima: dict[int, int] = {}
    for state in states:
        timestamp = float(state.get("time") or 0.0)
        interval = sum(boundary <= timestamp for boundary in boundaries) - 1
        maxima[interval] = max(
            maxima.get(interval, 0), int(state.get("simulator_steps") or 0)
        )
    return sum(maxima.values())


def _event_count(states: list[dict[str, Any]], event: str) -> int:
    """Count one canonical telemetry boundary, not raw audit rows."""

    return sum(str(row.get("event") or "") == event for row in states)


def _usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model_calls": 0,
        "model_latency_seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "estimated_api_cost_usd": 0.0,
        "actual_models": set(),
        "model_call_latency_samples": [],
    }
    for row in rows:
        usage = row.get("usage") or (row.get("response") or {}).get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        out["model_calls"] += 1
        elapsed = float(row.get("elapsed_s") or row.get("elapsed_seconds") or 0.0)
        out["model_latency_seconds"] += elapsed
        if math.isfinite(elapsed) and elapsed >= 0.0:
            out["model_call_latency_samples"].append(elapsed)
        out["prompt_tokens"] += int(
            usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        )
        out["completion_tokens"] += int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        out["cached_tokens"] += int(
            details.get("cached_tokens") or usage.get("cached_tokens") or 0
        )
        out["estimated_api_cost_usd"] += _request_list_price_usd(usage)
        actual = row.get("actual_model") or row.get("model")
        if actual:
            out["actual_models"].add(str(actual))
    out["model_latency_seconds"] = round(out["model_latency_seconds"], 6)
    out["estimated_api_cost_usd"] = round(
        float(out["estimated_api_cost_usd"]), 8
    )
    out["actual_models"] = ",".join(sorted(out["actual_models"]))
    samples = np.asarray(out.pop("model_call_latency_samples"), dtype=float)
    out["model_call_latency_samples_json"] = json.dumps(samples.tolist())
    out["median_model_call_latency_seconds"] = (
        float(np.median(samples)) if samples.size else math.nan
    )
    out["p90_model_call_latency_seconds"] = (
        float(np.quantile(samples, 0.90)) if samples.size else math.nan
    )
    out["mean_model_call_latency_seconds"] = (
        float(np.mean(samples)) if samples.size else math.nan
    )
    return out


def _pooled_model_call_latencies(group: pd.DataFrame) -> np.ndarray:
    """Recover normalized per-request latency samples for one report group."""

    samples: list[float] = []
    for raw in group["model_call_latency_samples_json"].fillna("[]"):
        try:
            values = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(values, list):
            continue
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number >= 0.0:
                samples.append(number)
    return np.asarray(samples, dtype=float)


def _response_text(row: dict[str, Any]) -> str:
    response = row.get("response") or {}
    content = response.get("content") if isinstance(response, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return ""


def _rats_source_events(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Count deploy-time source drafts and repair drafts in RATS calls."""

    generated = 0
    repaired = 0
    for row in rows:
        caller = str(row.get("caller") or "").lower()
        response = _response_text(row).strip()
        if not response:
            continue
        if "policy_writer" in caller:
            generated += 1
            # PolicyWriter calls after an explicit retry/self-repair carry a
            # retry artifact or caller context, but the stable telemetry does
            # not expose that context reliably. Count them as generation; the
            # decider-derived repair count below is exact.
        elif "multi_turn_decider" in caller and re.search(
            r"\bREGENERATE\b", response, re.I
        ):
            generated += 1
            repaired += 1
    return generated, repaired


def _capx_source_events(task_root: Path) -> tuple[int, int]:
    """Count code-bearing CaP-X decisions from the longest final trace."""

    best: list[dict[str, Any]] = []
    for path in task_root.rglob("all_responses.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(raw, list) and len(raw) > len(best):
            best = [row for row in raw if isinstance(row, dict)]
    generated = 0
    repaired = 0
    for row in best:
        blocks = [str(value).strip() for value in row.get("code_blocks") or []]
        if not any(blocks):
            continue
        generated += 1
        if str(row.get("decision") or "").lower() != "initial":
            repaired += 1
    return generated, repaired


_WALL_TIMEOUT_PATTERN = re.compile(
    r"(?:exceeded|timed out after)\s+1000(?:\.0+)?\s+seconds", re.I
)


def _generated_wall_timeout_reached(task_root: Path) -> bool:
    """Read the generated-code runner's explicit 1000-second timeout marker."""

    # The comparison harness has a registered 1150-second outer watchdog so a
    # wedged upstream epilogue cannot run forever.  When it fires, run_libero
    # materializes a terminal summary with this explicit marker.  It is a
    # per-episode resource outcome, equivalent to the upstream 1000-second
    # marker for reporting purposes.
    for path in (
        task_root / "artifacts" / "final_summary.json",
        task_root / "artifacts" / "iteration_001.json",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        termination = payload.get("termination") if isinstance(payload, dict) else None
        if (
            isinstance(termination, dict)
            and termination.get("kind") == "registered_hard_wall_timeout"
        ):
            return True

    candidates = [task_root / "run.log", *task_root.rglob("summary.txt")]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _WALL_TIMEOUT_PATTERN.search(text):
            return True
    return False


ManifestKey = tuple[str, str, int, int]


def _manifest_index() -> tuple[dict[ManifestKey, dict[str, Any]], pd.DataFrame]:
    payload = json.loads((HERE / "task_manifest.json").read_text(encoding="utf-8"))
    indexed_rows = []
    for registered_job_index, raw in enumerate(payload["rows"]):
        row = dict(raw)
        row["registered_job_index"] = registered_job_index
        indexed_rows.append(row)
    index = {
        (
            str(row["cohort"]),
            str(row["suite"]),
            int(row["task_id"]),
            int(row["seed"]),
        ): row
        for row in indexed_rows
    }
    if len(index) != len(payload["rows"]):
        raise ValueError("task manifest contains duplicate cohort/suite/task/seed rows")
    return index, pd.DataFrame(indexed_rows)


def _base_row(
    method: str,
    cohort: str,
    suite: str,
    task_id: int,
    seed: int,
    manifest: dict[ManifestKey, dict[str, Any]],
) -> dict[str, Any]:
    key = (cohort, suite, task_id, seed)
    meta = manifest.get(key)
    if meta is None:
        raise ValueError(f"measured episode is not registered in task manifest: {key}")
    registered_job_index = int(meta.get("registered_job_index", -1))
    pre_amendment = method == "capx" and 0 <= registered_job_index < 141
    return {
        "method": method,
        "method_label": LABELS[method],
        "cohort": meta.get("cohort", "unregistered"),
        "suite": suite,
        "task_id": task_id,
        "seed": seed,
        "episode_key": f"{suite}/{task_id}/seed{seed}",
        "registered_job_index": registered_job_index,
        "episode_worker_cap": 2 if pre_amendment else 10,
        "parallelism_regime": (
            "pre_amendment_2_workers" if pre_amendment else "amended_10_workers"
        ),
        "instruction": meta.get("instruction", ""),
        "instruction_source": meta.get(
            "instruction_source", "benchmark_task_language"
        ),
        "delivered_instruction": "",
        "native_success": False,
        "predicate_completion": np.nan,
        "policy_wall_seconds": np.nan,
        "artifact_wall_seconds": np.nan,
        "model_latency_seconds": 0.0,
        "model_call_latency_samples_json": "[]",
        "median_model_call_latency_seconds": np.nan,
        "p90_model_call_latency_seconds": np.nan,
        "mean_model_call_latency_seconds": np.nan,
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "estimated_api_cost_usd": 0.0,
        "policy_or_skill_calls": 0,
        "simulator_steps": 0,
        "simulator_episode_resets": 0,
        "init_state_index": np.nan,
        "method_native_turns": 0,
        "source_generation_events": 0,
        "source_repair_events": 0,
        "actual_models": "",
        "stop_reason": "",
        "error": "",
        "wall_time_limit_reached": False,
        "artifact_path": "",
        "evidence_path": "",
    }


def _parse_racap(
    method_root: Path,
    method: str,
    manifest: dict[ManifestKey, dict[str, Any]],
) -> list[dict[str, Any]]:
    calls: dict[str, list[dict[str, Any]]] = {}
    for path in method_root.rglob("llm_calls.worker*.jsonl"):
        for call in _jsonl(path):
            calls.setdefault(str(call.get("episode_key", "")), []).append(call)
    resets: dict[str, list[dict[str, Any]]] = {}
    for path in method_root.rglob("sim_episodes.worker*.jsonl"):
        for reset in _jsonl(path):
            resets.setdefault(str(reset.get("episode_key", "")), []).append(reset)
    out = []
    for path in method_root.rglob("records.jsonl"):
        evidence = _verified_evidence(path.parent, require_complete_marker=False)
        if evidence is None:
            continue
        cohort = path.relative_to(method_root).parts[0]
        for raw in _jsonl(path):
            suite, task_id, seed = raw["suite"], int(raw["task_id"]), int(raw["seed"])
            row = _base_row(method, cohort, suite, task_id, seed, manifest)
            stats = _usage(calls.get(row["episode_key"], []))
            reset_rows = resets.get(row["episode_key"], [])
            runtime_calls = raw.get("evaluator_runtime_calls") or {}
            row.update(
                {
                    "native_success": bool(raw.get("native_success")),
                    "delivered_instruction": str(raw.get("instruction") or ""),
                    "predicate_completion": _predicate_fraction(raw.get("native_predicates") or []),
                    "policy_wall_seconds": float(raw.get("seconds") or 0.0),
                    **stats,
                    "policy_or_skill_calls": int(runtime_calls.get("public_calls") or 0),
                    "simulator_steps": int(raw.get("simulator_steps") or 0),
                    "simulator_episode_resets": len(reset_rows),
                    "init_state_index": raw.get("init_state_index"),
                    "method_native_turns": int(raw.get("turns") or 0),
                    "source_generation_events": 0,
                    "source_repair_events": 0,
                    "public_runtime_calls": int(runtime_calls.get("public_calls") or 0),
                    "stop_reason": str(raw.get("stopped") or ""),
                    "error": str(raw.get("error") or ""),
                    "wall_time_limit_reached": "EpisodeWallTimeExceeded" in str(
                        raw.get("error") or ""
                    ),
                    "artifact_path": str(path.parent.resolve()),
                    "evidence_path": str(evidence.resolve()),
                }
            )
            out.append(row)
    return out


def _telemetry_by_key(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = [payload] if isinstance(payload, dict) else []
            except (OSError, json.JSONDecodeError):
                rows = []
        else:
            rows = _jsonl(path)
        for row in rows:
            grouped.setdefault(str(row.get("episode_key", "")), []).append(row)
    return grouped


_CAPX_GOAL_PATTERN = re.compile(r"Goal:\s*(.*?)(?:\\\\n|\n)", re.DOTALL)


def _delivered_instruction(
    task_root: Path, reset_rows: list[dict[str, Any]], *, capx: bool
) -> str:
    """Recover the public instruction that was actually delivered to a policy."""

    for reset in reversed(reset_rows):
        prompt = str(reset.get("task_prompt") or "").strip()
        if prompt:
            return prompt
    if not capx:
        return ""
    # Older CaP-X rows predate reset-level prompt telemetry.  Its retained
    # initial prompt is still an exact record of the user message, so use that
    # as a compatibility fallback instead of dropping instruction provenance.
    goals: list[str] = []
    for path in task_root.rglob("initial_prompt.txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = _CAPX_GOAL_PATTERN.search(text)
        if match:
            goal = match.group(1).strip()
            if goal and goal not in goals:
                goals.append(goal)
    return goals[0] if len(goals) == 1 else ""


def _parse_rats(
    method_root: Path,
    method: str,
    manifest: dict[ManifestKey, dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for task_root in method_root.glob("*/*/task_*/seed_*"):
        evidence = _verified_evidence(task_root, require_complete_marker=True)
        if evidence is None:
            continue
        native = _telemetry_by_key([task_root / "native_states.jsonl"])
        resets = _telemetry_by_key([task_root / "sim_episodes.jsonl"])
        call_paths = list((task_root / "agent_io").glob("*.json"))
        calls = _telemetry_by_key(call_paths)
        artifacts = task_root / "artifacts"
        cohort = task_root.relative_to(method_root).parts[0]
        for iteration_path in sorted(artifacts.glob("iteration_*.json")):
            raw = json.loads(iteration_path.read_text(encoding="utf-8"))
            proposal = raw.get("task_proposal") or {}
            suite = str(proposal.get("scene_model") or task_root.parent.parent.name)
            task_id = int(task_root.parent.name.split("_")[-1])
            seed = int(task_root.name.split("_")[-1])
            row = _base_row(method, cohort, suite, task_id, seed, manifest)
            states = sorted(native.get(row["episode_key"], []), key=lambda x: x.get("time", 0))
            reset_rows = resets.get(row["episode_key"], [])
            final = states[-1] if states else {}
            stats = _usage(calls.get(row["episode_key"], []))
            source_events, repair_events = _rats_source_events(
                calls.get(row["episode_key"], [])
            )
            row.update(
                {
                    "native_success": bool(final.get("native_success", raw.get("success", False))),
                    "predicate_completion": _predicate_fraction(final.get("native_predicates") or []),
                    "policy_wall_seconds": float(raw.get("elapsed_seconds") or 0.0),
                    "artifact_wall_seconds": _artifact_wall_seconds(task_root),
                    **stats,
                    "policy_or_skill_calls": _event_count(states, "native_state_after_api"),
                    "simulator_steps": _total_simulator_steps(states, reset_rows),
                    "method_native_turns": _event_count(states, "native_state_after_code"),
                    "source_generation_events": source_events,
                    "source_repair_events": repair_events,
                    "simulator_episode_resets": len(reset_rows),
                    "delivered_instruction": _delivered_instruction(
                        task_root, reset_rows, capx=False
                    ),
                    "init_state_index": (
                        reset_rows[-1].get("init_state_index") if reset_rows else None
                    ),
                    "stop_reason": str(raw.get("feedback_action") or ""),
                    "wall_time_limit_reached": _generated_wall_timeout_reached(
                        task_root
                    ),
                    "artifact_path": str(iteration_path.resolve()),
                    "evidence_path": str(evidence.resolve()),
                }
            )
            out.append(row)
    return out


def _parse_capx(
    method_root: Path,
    manifest: dict[ManifestKey, dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for task_root in method_root.glob("*/*/task_*/seed_*"):
        evidence = _verified_evidence(task_root, require_complete_marker=True)
        if evidence is None:
            continue
        native = _telemetry_by_key([task_root / "native_states.jsonl"])
        resets = _telemetry_by_key([task_root / "sim_episodes.jsonl"])
        calls = _telemetry_by_key([task_root / "llm_calls.jsonl"])
        cohort = task_root.relative_to(method_root).parts[0]
        for key, states_unsorted in native.items():
            parts = _episode_parts(key)
            if parts is None:
                continue
            suite, task_id, seed = parts
            states = sorted(states_unsorted, key=lambda x: x.get("time", 0))
            final = states[-1]
            reset_rows = resets.get(key, [])
            call_rows = calls.get(key, [])
            timeline = [float(x.get("time", 0)) for x in reset_rows + states + call_rows if x.get("time")]
            source_events, repair_events = _capx_source_events(task_root)
            row = _base_row("capx", cohort, suite, task_id, seed, manifest)
            row.update(
                {
                    "native_success": bool(final.get("native_success")),
                    "predicate_completion": _predicate_fraction(final.get("native_predicates") or []),
                    "policy_wall_seconds": max(timeline) - min(timeline) if len(timeline) >= 2 else np.nan,
                    "artifact_wall_seconds": _artifact_wall_seconds(task_root),
                    **_usage(call_rows),
                    "policy_or_skill_calls": _event_count(states, "native_state_after_api"),
                    "simulator_steps": _total_simulator_steps(states, reset_rows),
                    "method_native_turns": _event_count(states, "native_state_after_code"),
                    "source_generation_events": source_events,
                    "source_repair_events": repair_events,
                    "simulator_episode_resets": len(reset_rows),
                    "delivered_instruction": _delivered_instruction(
                        task_root, reset_rows, capx=True
                    ),
                    "init_state_index": (
                        reset_rows[-1].get("init_state_index") if reset_rows else None
                    ),
                    "wall_time_limit_reached": _generated_wall_timeout_reached(
                        task_root
                    ),
                    "artifact_path": str(task_root.resolve()),
                    "evidence_path": str(evidence.resolve()),
                }
            )
            out.append(row)
    return out


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - margin, centre + margin


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, cohort), group in df.groupby(["method", "cohort"], dropna=False):
        successes = int(group["native_success"].sum())
        low, high = _wilson(successes, len(group))
        call_latencies = _pooled_model_call_latencies(group)
        rows.append(
            {
                "method": method,
                "method_label": LABELS.get(method, method),
                "cohort": cohort,
                "episodes": len(group),
                "successes": successes,
                "success_rate": successes / len(group),
                "wilson_low": low,
                "wilson_high": high,
                "mean_predicate_completion": group["predicate_completion"].mean(),
                "median_wall_seconds": group["policy_wall_seconds"].median(),
                "p90_wall_seconds": group["policy_wall_seconds"].quantile(0.90),
                "mean_wall_seconds": group["policy_wall_seconds"].mean(),
                "median_artifact_wall_seconds": group[
                    "artifact_wall_seconds"
                ].median(),
                "p90_artifact_wall_seconds": group[
                    "artifact_wall_seconds"
                ].quantile(0.90),
                "mean_artifact_wall_seconds": group[
                    "artifact_wall_seconds"
                ].mean(),
                "median_model_latency_seconds": group[
                    "model_latency_seconds"
                ].median(),
                "p90_model_latency_seconds": group[
                    "model_latency_seconds"
                ].quantile(0.90),
                "mean_model_latency_seconds": group[
                    "model_latency_seconds"
                ].mean(),
                "pooled_model_calls_with_latency": int(call_latencies.size),
                "pooled_median_model_call_latency_seconds": (
                    float(np.median(call_latencies))
                    if call_latencies.size
                    else math.nan
                ),
                "pooled_p90_model_call_latency_seconds": (
                    float(np.quantile(call_latencies, 0.90))
                    if call_latencies.size
                    else math.nan
                ),
                "pooled_mean_model_call_latency_seconds": (
                    float(np.mean(call_latencies))
                    if call_latencies.size
                    else math.nan
                ),
                "median_model_calls": group["model_calls"].median(),
                "p90_model_calls": group["model_calls"].quantile(0.90),
                "mean_model_calls": group["model_calls"].mean(),
                "mean_estimated_api_cost_usd": group[
                    "estimated_api_cost_usd"
                ].mean(),
                "median_estimated_api_cost_usd": group[
                    "estimated_api_cost_usd"
                ].median(),
                "p90_estimated_api_cost_usd": group[
                    "estimated_api_cost_usd"
                ].quantile(0.90),
                "total_estimated_api_cost_usd": group[
                    "estimated_api_cost_usd"
                ].sum(),
                "mean_prompt_tokens": group["prompt_tokens"].mean(),
                "mean_completion_tokens": group["completion_tokens"].mean(),
                "mean_simulator_steps": group["simulator_steps"].mean(),
                "mean_simulator_episode_resets": group[
                    "simulator_episode_resets"
                ].mean(),
                "mean_source_generation_events": group[
                    "source_generation_events"
                ].mean(),
                "mean_source_repair_events": group["source_repair_events"].mean(),
                "wall_time_limit_count": int(group["wall_time_limit_reached"].sum()),
                "wall_time_limit_rate": group["wall_time_limit_reached"].mean(),
                "simulator_horizon_count": int(
                    group["simulator_horizon_reached"].sum()
                ),
                "simulator_horizon_rate": group[
                    "simulator_horizon_reached"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _paired_statistics(df: pd.DataFrame, bootstrap: int = 10_000) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260822)
    for cohort, cohort_df in df.groupby("cohort"):
        present = [method for method in METHODS if method in set(cohort_df["method"])]
        for a, b in itertools.combinations(present, 2):
            left = cohort_df[cohort_df.method == a].set_index("episode_key")
            right = cohort_df[cohort_df.method == b].set_index("episode_key")
            keys = left.index.intersection(right.index)
            if len(keys) == 0:
                continue
            sa = left.loc[keys, "native_success"].astype(bool).to_numpy()
            sb = right.loc[keys, "native_success"].astype(bool).to_numpy()
            discord_a = int(np.sum(sa & ~sb))
            discord_b = int(np.sum(~sa & sb))
            discord = discord_a + discord_b
            p = binomtest(min(discord_a, discord_b), discord, 0.5).pvalue if discord else 1.0
            record = {
                "cohort": cohort,
                "method_a": a,
                "method_b": b,
                "paired_episodes": len(keys),
                "success_rate_a": float(sa.mean()),
                "success_rate_b": float(sb.mean()),
                "success_difference_a_minus_b": float(sa.mean() - sb.mean()),
                "a_only_success": discord_a,
                "b_only_success": discord_b,
                "mcnemar_exact_p": p,
            }
            task_clusters = pd.Series(
                [f"{suite}/{int(task_id)}" for suite, task_id in zip(
                    left.loc[keys, "suite"], left.loc[keys, "task_id"]
                )],
                index=keys,
            )
            success_differences = pd.Series(
                sa.astype(float) - sb.astype(float), index=keys
            )
            success_by_task = success_differences.groupby(task_clusters).mean().to_numpy()
            record["paired_task_clusters"] = len(success_by_task)
            record["bootstrap_unit"] = "suite_task"
            if len(success_by_task):
                samples = rng.choice(
                    success_by_task,
                    size=(bootstrap, len(success_by_task)),
                    replace=True,
                ).mean(1)
                record["success_difference_task_bootstrap_low"] = float(
                    np.quantile(samples, 0.025)
                )
                record["success_difference_task_bootstrap_high"] = float(
                    np.quantile(samples, 0.975)
                )
            for metric in (
                "predicate_completion",
                "policy_wall_seconds",
                "artifact_wall_seconds",
                "model_latency_seconds",
                "median_model_call_latency_seconds",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "model_calls",
                "estimated_api_cost_usd",
                "policy_or_skill_calls",
                "simulator_steps",
                "method_native_turns",
                "source_generation_events",
                "source_repair_events",
            ):
                if metric not in left.columns or metric not in right.columns:
                    continue
                va = pd.to_numeric(left.loc[keys, metric], errors="coerce").to_numpy(float)
                vb = pd.to_numeric(right.loc[keys, metric], errors="coerce").to_numpy(float)
                valid = np.isfinite(va) & np.isfinite(vb)
                differences = va[valid] - vb[valid]
                if len(differences):
                    clusters = task_clusters.to_numpy()[valid]
                    task_means = (
                        pd.DataFrame({"cluster": clusters, "difference": differences})
                        .groupby("cluster", sort=True)["difference"]
                        .mean()
                        .to_numpy()
                    )
                    samples = rng.choice(
                        task_means,
                        size=(bootstrap, len(task_means)),
                        replace=True,
                    ).mean(1)
                    record[f"{metric}_mean_difference"] = float(task_means.mean())
                    record[f"{metric}_task_clusters"] = len(task_means)
                    record[f"{metric}_bootstrap_low"] = float(np.quantile(samples, 0.025))
                    record[f"{metric}_bootstrap_high"] = float(np.quantile(samples, 0.975))
            rows.append(record)
    if rows:
        # ``protocol.yaml`` pre-registers Holm correction but does not state
        # whether the family is one benchmark cohort or the complete main
        # grid.  Record both rather than letting the set of cohorts passed to
        # this script silently change the inferential conclusion.  The legacy
        # column remains the more conservative all-grid correction.
        by_cohort: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            by_cohort.setdefault(str(row["cohort"]), []).append(index)
        for indices in by_cohort.values():
            adjusted = _holm(
                [float(rows[index]["mcnemar_exact_p"]) for index in indices]
            )
            for index, value in zip(indices, adjusted):
                rows[index]["mcnemar_holm_within_cohort_p"] = value

        adjusted = _holm([float(row["mcnemar_exact_p"]) for row in rows])
        for row, value in zip(rows, adjusted):
            row["mcnemar_holm_global_p"] = value
            row["mcnemar_holm_p"] = value
    return pd.DataFrame(rows)


def _plots(aggregate: pd.DataFrame, episodes: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cohorts = [
        c
        for c in aggregate.cohort.unique()
        if c not in {"unregistered", "ALL"}
    ]
    maximum_interval = float(
        pd.to_numeric(aggregate["wilson_high"], errors="coerce").max()
    )
    # Most manipulation success rates occupy only the left part of [0, 1].
    # Preserve a common axis across panels while using the available figure
    # area, and leave room for exact-count annotations beyond each interval.
    x_max = min(
        1.0,
        max(0.25, math.ceil(maximum_interval * 1.25 * 20.0) / 20.0),
    )
    columns = 2 if len(cohorts) > 1 else 1
    rows = max(1, math.ceil(len(cohorts) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.1 if columns == 2 else 3.5, 2.5 * rows),
        squeeze=False,
        sharex=True,
    )
    cohort_labels = {
        "libero90_id_replay": "LIBERO-90 (in-domain)",
        "libero_pro_zero_shot": "LIBERO-PRO (zero-shot)",
        "libero_base_diagnostic": "LIBERO base diagnostic",
        "libero_long": "LIBERO-Long",
    }
    for axis, cohort in zip(axes.flat, cohorts):
        part = aggregate[aggregate.cohort == cohort].set_index("method")
        methods = [m for m in METHODS if m in part.index]
        y = np.arange(len(methods))[::-1]
        values = np.asarray([part.loc[m, "success_rate"] for m in methods], dtype=float)
        lower = np.asarray(
            [values[i] - part.loc[m, "wilson_low"] for i, m in enumerate(methods)]
        )
        upper = np.asarray(
            [part.loc[m, "wilson_high"] - values[i] for i, m in enumerate(methods)]
        )
        for index, method in enumerate(methods):
            axis.errorbar(
                values[index],
                y[index],
                xerr=[[lower[index]], [upper[index]]],
                fmt="o",
                color=COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=6.5,
            )
            axis.text(
                min(float(part.loc[method, "wilson_high"]) + 0.012 * x_max, 0.985 * x_max),
                y[index],
                f"{int(part.loc[method, 'successes'])}/{int(part.loc[method, 'episodes'])}",
                va="center",
                ha="left",
                fontsize=7.4,
                color="#3A3A3A",
            )
        axis.set_title(cohort_labels.get(cohort, cohort.replace("_", " ")))
        axis.set_xlim(0, x_max)
        axis.set_yticks(y, [LABELS[m] for m in methods])
        clean_axis(axis, grid_axis="x")
    for axis in axes.flat[len(cohorts) :]:
        axis.set_visible(False)
    fig.supxlabel("Native success rate (Wilson 95% CI)", y=0.01)
    fig.tight_layout(rect=(0, 0.025, 1, 1), h_pad=1.0, w_pad=1.1)
    save_figure(fig, output / "success_by_cohort")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.8, 3.35))
    for method in METHODS:
        part = episodes[episodes.method == method]
        if part.empty:
            continue
        axis.scatter(
            part["model_calls"].mean(),
            part["policy_wall_seconds"].median(),
            s=58,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.65,
            label=LABELS[method],
        )
    axis.set_xlabel("Mean hosted-model calls / episode")
    axis.set_ylabel("Median policy wall time (s)")
    clean_axis(axis, grid_axis="both")
    axis.legend(loc="best", ncol=1)
    fig.tight_layout()
    save_figure(fig, output / "runtime_calls_tradeoff")
    plt.close(fig)


def _format_excel_sheet(writer: pd.ExcelWriter, name: str, frame: pd.DataFrame) -> None:
    """Apply compact, readable, analysis-friendly workbook formatting."""

    worksheet = writer.sheets[name]
    workbook = writer.book
    header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#1F4E78",
            "border": 0,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        }
    )
    percent = workbook.add_format({"num_format": "0.0%"})
    seconds = workbook.add_format({"num_format": "0.00"})
    currency = workbook.add_format({"num_format": "$0.0000"})
    p_value = workbook.add_format({"num_format": "0.0000"})
    integer = workbook.add_format({"num_format": "0"})
    decimal = workbook.add_format({"num_format": "0.00"})
    wrapped = workbook.add_format({"text_wrap": True, "valign": "top"})
    worksheet.freeze_panes(1, 0)
    worksheet.set_row(0, 30)
    if len(frame.columns):
        worksheet.autofilter(0, 0, max(1, len(frame)), len(frame.columns) - 1)
    for column_index, column in enumerate(frame.columns):
        worksheet.write(0, column_index, column, header)
        values = frame[column].astype(str) if column in frame else pd.Series(dtype=str)
        max_value = int(values.str.len().max()) if len(values) else 0
        width = min(55, max(10, len(str(column)) + 2, max_value + 2))
        lower = str(column).lower()
        fmt = None
        if any(token in lower for token in ("instruction", "artifact", "evidence", "definition", "json", "yaml")):
            width = min(80, max(width, 28))
            fmt = wrapped
        elif "usd" in lower:
            fmt = currency
        elif lower.endswith("_p") or "p_value" in lower or "mcnemar" in lower:
            fmt = p_value
        elif any(token in lower for token in ("rate", "fraction", "wilson", "completion")):
            fmt = percent
        elif "seconds" in lower or "latency" in lower:
            fmt = seconds
        elif any(
            token in lower
            for token in (
                "episodes",
                "successes",
                "calls",
                "tokens",
                "steps",
                "turns",
                "events",
                "task_id",
                "seed",
            )
        ):
            fmt = decimal if any(token in lower for token in ("mean", "median", "p90")) else integer
        worksheet.set_column(column_index, column_index, width, fmt)
        if lower in {"native_success", "completed"} and len(frame):
            worksheet.conditional_format(
                1,
                column_index,
                len(frame),
                column_index,
                {
                    "type": "cell",
                    "criteria": "==",
                    "value": True,
                    "format": workbook.add_format(
                        {"bg_color": "#C6EFCE", "font_color": "#006100"}
                    ),
                },
            )
            worksheet.conditional_format(
                1,
                column_index,
                len(frame),
                column_index,
                {
                    "type": "cell",
                    "criteria": "==",
                    "value": False,
                    "format": workbook.add_format(
                        {"bg_color": "#FFC7CE", "font_color": "#9C0006"}
                    ),
                },
            )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _provenance_tables(
    input_root: Path,
    methods: list[str],
    *,
    preflight_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact, non-secret provenance suitable for an Excel workbook.

    The full preflight file is larger than Excel's per-cell text limit and is
    captured before method-local perception services start.  Embedding it as a
    single cell therefore both truncates evidence and makes its expected
    ``reachable=false`` probes look like evaluation failures.  The workbook
    instead records a hash-addressed preflight summary and one row per method's
    run manifest and service lifecycle.  The full JSON files remain alongside
    the measured artifacts at the recorded paths.
    """

    if preflight_path is None:
        artifact_sibling = input_root.parent / "provenance" / "preflight.json"
        repository_default = (
            ROOT / "outputs" / "controlled_comparison" / "provenance" / "preflight.json"
        )
        preflight_path = (
            artifact_sibling if artifact_sibling.is_file() else repository_default
        )
    preflight = _read_json_object(preflight_path)
    repository = (preflight.get("repositories") or {}).get("racap") or {}
    model = preflight.get("model_configuration") or {}
    service_probe = ", ".join(
        f"{row.get('host')}:{row.get('port')}={'reachable' if row.get('reachable') else 'not-yet-started'}"
        for row in preflight.get("services") or []
        if isinstance(row, dict)
    )
    preflight_summary = pd.DataFrame(
        [
            ("capture_phase", "before method-local perception-service launch"),
            ("path", str(preflight_path)),
            ("sha256", _file_sha256(preflight_path)),
            ("captured_at_unix", preflight.get("captured_at_unix", "")),
            ("hostname", (preflight.get("host") or {}).get("hostname", "")),
            ("python_version", (preflight.get("host") or {}).get("python_version", "")),
            ("racap_commit", repository.get("commit", "")),
            ("racap_status_porcelain", repository.get("status_porcelain", "")),
            ("requested_model", model.get("requested_model", "")),
            ("model_base_url", model.get("base_url", "")),
            ("maximum_concurrency", model.get("maximum_concurrency", "")),
            ("credential_value_recorded", model.get("credential_value_recorded", "")),
            ("preflight_service_probe", service_probe),
            (
                "service_probe_interpretation",
                "Preflight probes precede method-local launch; evaluation readiness is recorded in Run Provenance.",
            ),
        ],
        columns=["item", "value"],
    )

    run_rows: list[dict[str, Any]] = []
    for method in methods:
        method_root = input_root / method
        manifest_path = method_root / "run_manifest.json"
        lifecycle_path = method_root / "shared_api_services" / "lifecycle.json"
        manifest = _read_json_object(manifest_path)
        lifecycle = _read_json_object(lifecycle_path)
        assets = lifecycle.get("model_assets") or {}
        run_rows.append(
            {
                "method": method,
                "manifest_present": manifest_path.is_file(),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _file_sha256(manifest_path),
                "protocol_sha256": manifest.get("protocol_sha256", ""),
                "task_manifest_sha256": manifest.get("task_manifest_sha256", ""),
                "model": manifest.get("model", ""),
                "workers": manifest.get("workers", ""),
                "registered_jobs": len(manifest.get("jobs") or []),
                "cohorts": ", ".join(manifest.get("cohorts") or []),
                "simulator_horizon": manifest.get("simulator_horizon", ""),
                "initial_resets_per_episode": manifest.get(
                    "registered_initial_states_per_episode", ""
                ),
                "post_action_resets": manifest.get("post_action_environment_resets", ""),
                "generated_continuous_turns": manifest.get(
                    "generated_continuous_turns", ""
                ),
                "service_urls": json.dumps(
                    manifest.get("shared_service_urls") or {}, sort_keys=True
                ),
                "lifecycle_present": lifecycle_path.is_file(),
                "lifecycle_path": str(lifecycle_path),
                "lifecycle_sha256": _file_sha256(lifecycle_path),
                "service_owner": lifecycle.get("owner", ""),
                "service_pid": lifecycle.get("pid", ""),
                "service_ports": ", ".join(map(str, lifecycle.get("ports") or [])),
                "service_started_at_unix": lifecycle.get("started_at_unix", ""),
                "service_ready_at_unix": lifecycle.get("ready_at_unix", ""),
                "service_stopped_at_unix": lifecycle.get("stopped_at_unix", ""),
                "service_returncode": lifecycle.get("returncode", ""),
                "sam3_sha256": (assets.get("sam3") or {}).get("sha256", ""),
                "contact_graspnet_sha256": (assets.get("contact_graspnet") or {}).get(
                    "sha256", ""
                ),
            }
        )
    return preflight_summary, pd.DataFrame(run_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=ROOT / "outputs" / "controlled_comparison" / "measured"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "controlled_comparison" / "analysis"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Methods included in both parsing and the completeness audit.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=MAIN_COHORTS,
        help=(
            "Registered cohorts included in this report. One-shot and custom "
            "long-horizon experiments have dedicated analyzers and are excluded "
            "from the main report by default."
        ),
    )
    args = parser.parse_args()
    manifest, manifest_df = _manifest_index()
    unknown_cohorts = sorted(set(args.cohorts) - set(manifest_df["cohort"]))
    if unknown_cohorts:
        raise SystemExit(f"unknown manifest cohorts: {unknown_cohorts}")
    manifest_df = manifest_df[manifest_df["cohort"].isin(args.cohorts)].copy()
    rows: list[dict[str, Any]] = []
    if "capx" in args.methods:
        rows.extend(_parse_capx(args.input_root / "capx", manifest))
    if "rats_base" in args.methods:
        rows.extend(_parse_rats(args.input_root / "rats_base", "rats_base", manifest))
    if "rats_90" in args.methods:
        rows.extend(_parse_rats(args.input_root / "rats_90", "rats_90", manifest))
    if "racap_phase1" in args.methods:
        rows.extend(_parse_racap(args.input_root / "racap_phase1", "racap_phase1", manifest))
    if "racap_phase2" in args.methods:
        rows.extend(_parse_racap(args.input_root / "racap_phase2", "racap_phase2", manifest))
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        raise SystemExit(f"no completed measured episodes under {args.input_root}")
    episodes = episodes[episodes["cohort"].isin(args.cohorts)].copy()
    episodes["instruction_match"] = [
        _normalize_instruction(registered) == _normalize_instruction(delivered)
        and bool(_normalize_instruction(registered))
        for registered, delivered in zip(
            episodes["instruction"], episodes["delivered_instruction"]
        )
    ]
    episodes["simulator_horizon_reached"] = episodes["simulator_steps"] >= 8000
    if episodes.empty:
        raise SystemExit(
            f"no completed episodes for cohorts {args.cohorts} under {args.input_root}"
        )
    duplicate_mask = episodes.duplicated(["method", "episode_key"], keep=False)
    if duplicate_mask.any():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        conflict_path = args.output_dir / "duplicate_episode_keys.csv"
        episodes.loc[duplicate_mask].sort_values(
            ["method", "episode_key", "artifact_path"]
        ).to_csv(conflict_path, index=False)
        raise SystemExit(
            "refusing to silently choose among duplicate measured episodes; "
            f"conflicts written to {conflict_path}"
        )
    aggregate = pd.concat(
        [
            _aggregate(episodes),
            _aggregate(episodes.assign(cohort="ALL")),
        ],
        ignore_index=True,
    )
    paired = _paired_statistics(episodes)
    parallelism = (
        episodes.groupby(
            ["method", "parallelism_regime", "episode_worker_cap"], dropna=False
        )
        .agg(
            episodes=("episode_key", "count"),
            native_successes=("native_success", "sum"),
            native_success_rate=("native_success", "mean"),
            median_policy_wall_seconds=("policy_wall_seconds", "median"),
            p90_policy_wall_seconds=(
                "policy_wall_seconds", lambda values: values.quantile(0.90)
            ),
            mean_model_calls=("model_calls", "mean"),
            estimated_api_cost_usd=("estimated_api_cost_usd", "sum"),
        )
        .reset_index()
    )

    expected = pd.MultiIndex.from_product(
        [args.methods, manifest_df.index], names=["method", "manifest_row"]
    ).to_frame(index=False)
    expected = expected.merge(manifest_df.reset_index(names="manifest_row"), on="manifest_row")
    observed = set(zip(episodes.method, episodes.episode_key))
    expected["episode_key"] = expected.apply(
        lambda row: f"{row.suite}/{int(row.task_id)}/seed{int(row.seed)}", axis=1
    )
    expected["completed"] = [
        (method, key) in observed for method, key in zip(expected.method, expected.episode_key)
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
    paired.to_csv(args.output_dir / "pairwise.csv", index=False)
    parallelism.to_csv(
        args.output_dir / "parallelism_diagnostics.csv", index=False
    )
    expected.to_csv(args.output_dir / "completeness.csv", index=False)
    workbook = args.output_dir / "controlled_comparison.xlsx"
    definitions = pd.DataFrame(
        [
            ("native_success", "LIBERO native goal predicate conjunction after the controller stops"),
            ("predicate_completion", "Fraction of native goal predicates satisfied; evaluator-only"),
            ("policy_wall_seconds", "Policy runtime excluding offline workbook/video serialization when separable"),
            (
                "artifact_wall_seconds",
                "Generated-method worker wall time from child launch through child-side video/trace serialization; "
                "left missing when per-episode overhead is not separable",
            ),
            (
                "model_call_latency_samples_json",
                "Per-request hosted-model latency samples retained as JSON; includes reasoning, verification, and VLM calls",
            ),
            (
                "pooled_*_model_call_latency_seconds",
                "Median/P90/mean over every retained hosted-model request in a method/cohort; not assumed to equal one ReAct decision",
            ),
            (
                "wall_time_limit_reached",
                "Whether the episode hit the common 1000-second policy wall-time cap",
            ),
            (
                "simulator_horizon_reached",
                "Whether the episode consumed the common 8000 simulator-step horizon",
            ),
            ("model_calls", "Network hosted-model responses; no exact response-cache hits"),
            (
                "parallelism_regime",
                "Episode-level concurrency provenance: the first 141 registered CaP-X jobs used the original two-worker cap; all subsequent jobs use the amended ten-worker cap",
            ),
            (
                "policy_or_skill_calls",
                "Public wrapped API calls, counted once at each native_state_after_api boundary; "
                "RACaP uses evaluator_runtime_calls.public_calls",
            ),
            (
                "method_native_turns",
                "Method-native outer control iterations: executed code blocks for CaP-X/RATS "
                "and ReAct turns for RACaP; reported but not assumed semantically identical",
            ),
            (
                "source_generation_events",
                "Deploy-time code-bearing drafts: PolicyWriter/REGENERATE drafts for RATS, "
                "code-bearing decisions in the final CaP-X all_responses trace, and zero "
                "for frozen-source RACaP",
            ),
            (
                "source_repair_events",
                "Code-bearing drafts after the initial CaP-X decision or code returned by "
                "a RATS multi-turn REGENERATE decision; RACaP remains zero",
            ),
            (
                "estimated_api_cost_usd",
                "OpenAI standard-list-price estimate using the frozen pricing_snapshot.yaml; "
                "raw token counts are authoritative and VAPI relay markup is unavailable",
            ),
            (
                "mcnemar_holm_within_cohort_p",
                "Exact paired McNemar p-value Holm-adjusted across the method pairs in the same evaluation cohort",
            ),
            (
                "mcnemar_holm_global_p",
                "Exact paired McNemar p-value Holm-adjusted across every method pair and cohort in this workbook",
            ),
            (
                "mcnemar_holm_p",
                "Backward-compatible alias of mcnemar_holm_global_p",
            ),
        ],
        columns=["field", "definition"],
    )
    provenance, run_provenance = _provenance_tables(args.input_root, args.methods)
    pricing = pd.DataFrame(
        [{"yaml": (HERE / "pricing_snapshot.yaml").read_text(encoding="utf-8")}]
    )
    overview = pd.DataFrame(
        [
            ("protocol", "racap_controlled"),
            ("success authority", "Native simulator predicate conjunction, evaluator-only"),
            ("episodes parsed", len(episodes)),
            ("registered episodes expected", len(expected)),
            ("registered episodes complete", int(expected.completed.sum())),
            ("methods", ", ".join(args.methods)),
            ("cohorts", ", ".join(args.cohorts)),
            (
                "parallelism amendment",
                "CaP-X registered jobs 0--140 used 2 workers; all subsequent jobs use the amended 10-worker cap. See Parallelism.",
            ),
            (
                "inference note",
                "Wilson intervals are descriptive; paired McNemar and suite-task clustered bootstrap provide paired inference. Pairwise sheets report both within-cohort and all-grid Holm families.",
            ),
            (
                "cost note",
                "USD is a frozen list-price estimate from request tokens, not an actual VAPI invoice.",
            ),
            (
                "provenance note",
                "Preflight is captured before service launch; method-time readiness and artifact hashes are in Run Provenance.",
            ),
        ],
        columns=["item", "value"],
    )
    with pd.ExcelWriter(workbook, engine="xlsxwriter") as writer:
        overview.to_excel(writer, sheet_name="Overview", index=False)
        episodes.to_excel(writer, sheet_name="Episodes", index=False)
        aggregate.to_excel(writer, sheet_name="Aggregate", index=False)
        paired.to_excel(writer, sheet_name="Pairwise", index=False)
        parallelism.to_excel(writer, sheet_name="Parallelism", index=False)
        expected.to_excel(writer, sheet_name="Completeness", index=False)
        definitions.to_excel(writer, sheet_name="Definitions", index=False)
        pricing.to_excel(writer, sheet_name="Pricing", index=False)
        provenance.to_excel(writer, sheet_name="Provenance", index=False)
        run_provenance.to_excel(writer, sheet_name="Run Provenance", index=False)
        for sheet_name, frame in {
            "Overview": overview,
            "Episodes": episodes,
            "Aggregate": aggregate,
            "Pairwise": paired,
            "Parallelism": parallelism,
            "Completeness": expected,
            "Definitions": definitions,
            "Pricing": pricing,
            "Provenance": provenance,
            "Run Provenance": run_provenance,
        }.items():
            _format_excel_sheet(writer, sheet_name, frame)
    _plots(aggregate, episodes, args.output_dir / "figures")
    summary = {
        "episodes": len(episodes),
        "methods": sorted(episodes.method.unique()),
        "registered_methods": args.methods,
        "registered_cohorts": args.cohorts,
        "complete_expected": int(expected.completed.sum()),
        "total_expected": len(expected),
        "workbook": str(workbook),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
