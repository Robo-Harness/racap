#!/usr/bin/env python3
"""Paired zero-shot versus one-trial adaptation analysis."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.controlled_comparison.plot_style import clean_axis, save_figure

FAMILY_SUITE = {
    "spatial": "libero_spatial_swap",
    "goal": "libero_goal_swap",
    "object": "libero_object_swap",
}

try:
    from .analyze_results import (
        LABELS,
        METHODS,
        _manifest_index,
        _holm,
        _parse_capx,
        _parse_racap,
        _parse_rats,
        _paired_statistics,
        _jsonl,
        _usage,
        _wilson,
    )
except ImportError:  # direct script execution
    from analyze_results import (  # type: ignore[no-redef]
        LABELS,
        METHODS,
        _manifest_index,
        _holm,
        _parse_capx,
        _parse_racap,
        _parse_rats,
        _paired_statistics,
        _jsonl,
        _usage,
        _wilson,
    )


def _load(input_root: Path) -> pd.DataFrame:
    manifest, _ = _manifest_index()
    rows: list[dict[str, Any]] = []
    rows.extend(_parse_capx(input_root / "capx", manifest))
    rows.extend(_parse_rats(input_root / "rats_base", "rats_base", manifest))
    rows.extend(_parse_rats(input_root / "rats_90", "rats_90", manifest))
    rows.extend(_parse_racap(input_root / "racap_phase1", "racap_phase1", manifest))
    rows.extend(_parse_racap(input_root / "racap_phase2", "racap_phase2", manifest))
    return pd.DataFrame(rows)


def _family(suite: str) -> str:
    for value in ("spatial", "goal", "object"):
        if f"_{value}_" in suite:
            return value
    return "unknown"


def _select_calibration(rows: pd.DataFrame, cohort: str) -> pd.DataFrame:
    selected = rows[rows["cohort"] == cohort].copy()
    if cohort == "libero_pro_zero_shot":
        selected = selected[
            selected["suite"].isin(FAMILY_SUITE.values())
            & (selected["task_id"] == 0)
            & (selected["seed"] == 0)
        ]
    expected = {
        (method, suite)
        for method in METHODS
        for suite in FAMILY_SUITE.values()
    }
    actual = {
        (str(row.method), str(row.suite)) for row in selected.itertuples()
    }
    if len(selected) != len(expected) or actual != expected:
        raise SystemExit(
            f"calibration grid mismatch for {cohort}: rows={len(selected)}, "
            f"pairs={len(actual)}, expected={len(expected)}"
        )
    return selected.sort_values(["method", "suite"]).reset_index(drop=True)


def _card_generation_calls(cards_dir: Path) -> pd.DataFrame:
    """Normalize the visual critic calls used to create one-trial cards."""

    rows: list[dict[str, Any]] = []
    for call in _jsonl(cards_dir / "critic_calls.jsonl"):
        key = str(call.get("episode_key") or "")
        parts = key.split("/")
        if len(parts) != 3 or parts[0] != "one_shot_card":
            continue
        usage = _usage([call])
        rows.append(
            {
                "method": parts[1],
                "family": parts[2],
                "episode_key": key,
                "event": str(call.get("event") or ""),
                "status_code": call.get("status_code"),
                **usage,
            }
        )
    return pd.DataFrame(rows)


def _adaptation_artifact_audit(
    calibration: pd.DataFrame,
    cards: pd.DataFrame,
    card_calls: pd.DataFrame,
    *,
    required_model: str = "gpt-5.5",
) -> dict[str, Any]:
    expected = {(method, family) for method in METHODS for family in FAMILY_SUITE}
    calibration_pairs = {
        (str(row.method), _family(str(row.suite))) for row in calibration.itertuples()
    }
    card_pairs = {
        (str(row.method), str(row.family)) for row in cards.itertuples()
    } if not cards.empty else set()
    call_pairs = {
        (str(row.method), str(row.family)) for row in card_calls.itertuples()
    } if not card_calls.empty else set()
    errors: list[str] = []
    if len(calibration) != len(expected) or calibration_pairs != expected:
        errors.append(
            f"calibration grid mismatch: rows={len(calibration)}, "
            f"pairs={len(calibration_pairs)}, expected={len(expected)}"
        )
    if len(cards) != len(expected) or card_pairs != expected:
        errors.append(
            f"experience-card grid mismatch: rows={len(cards)}, "
            f"pairs={len(card_pairs)}, expected={len(expected)}"
        )
    if call_pairs != expected:
        errors.append(
            f"card-generation call coverage mismatch: {len(call_pairs)}/{len(expected)}"
        )
    missing_card_files = [
        str(row.card)
        for row in cards.itertuples()
        if not Path(str(row.card)).is_file()
    ] if not cards.empty and "card" in cards else []
    if missing_card_files:
        errors.append(f"missing {len(missing_card_files)} experience-card files")
    actual_models = set()
    if "actual_models" in card_calls:
        for value in card_calls["actual_models"].fillna("").astype(str):
            actual_models.update(part for part in value.split(",") if part)
    unexpected = sorted(actual_models - {required_model})
    if unexpected:
        errors.append(f"unexpected card-generation models: {unexpected}")
    if card_calls.size and required_model not in actual_models:
        errors.append(f"no successful card-generation call recorded for {required_model}")
    return {
        "schema_version": 1,
        "status": "pass" if not errors else "error",
        "expected_method_family_pairs": len(expected),
        "calibration_rows": len(calibration),
        "card_rows": len(cards),
        "card_call_rows": len(card_calls),
        "actual_models": sorted(actual_models),
        "errors": errors,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_card_delivery_audit(
    one_shot_root: Path, cards: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bind every adapted method run to the registered three card files.

    The runner injects a family-specific card through a process environment
    variable, so it does not appear in the method's YAML task config.  The
    immutable method run manifest records the exact path and content hash used
    to construct every child environment.  This audit checks that binding
    against both the critic manifest and the files retained on disk.
    """

    expected = {(method, family) for method in METHODS for family in FAMILY_SUITE}
    card_index = {
        (str(row.method), str(row.family)): row
        for row in cards.itertuples()
    } if not cards.empty else {}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for method in METHODS:
        manifest_path = one_shot_root / method / "run_manifest.json"
        if not manifest_path.is_file():
            errors.append(f"missing method run manifest: {manifest_path}")
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("method") != method:
            errors.append(f"method mismatch in {manifest_path}")
        if payload.get("cohorts") != ["libero_pro_one_shot"]:
            errors.append(f"unexpected one-trial cohort in {manifest_path}")
        delivered = payload.get("experience_cards") or {}
        for family in FAMILY_SUITE:
            critic = card_index.get((method, family))
            actual = delivered.get(family) or {}
            path = Path(str(actual.get("path") or ""))
            retained_sha = _sha256(path) if path.is_file() else None
            critic_path = (
                str(Path(str(critic.card)).resolve()) if critic is not None else None
            )
            critic_sha = str(critic.card_sha256) if critic is not None else None
            row = {
                "method": method,
                "method_label": LABELS[method],
                "family": family,
                "run_manifest": str(manifest_path.resolve()),
                "delivered_card_path": str(path.resolve()) if path.is_file() else str(path),
                "delivered_card_sha256": actual.get("sha256"),
                "critic_manifest_card_path": critic_path,
                "critic_manifest_card_sha256": critic_sha,
                "retained_file_sha256": retained_sha,
                "path_match": path.is_file() and str(path.resolve()) == critic_path,
                "hash_match": (
                    retained_sha is not None
                    and retained_sha == critic_sha == actual.get("sha256")
                ),
            }
            rows.append(row)
            if not row["path_match"] or not row["hash_match"]:
                errors.append(f"card delivery mismatch for {method}/{family}")
    actual_pairs = {(row["method"], row["family"]) for row in rows}
    if actual_pairs != expected or len(rows) != len(expected):
        errors.append(
            f"card delivery coverage mismatch: rows={len(rows)}, "
            f"pairs={len(actual_pairs)}, expected={len(expected)}"
        )
    frame = pd.DataFrame(rows)
    audit = {
        "schema_version": 1,
        "status": "pass" if not errors else "error",
        "expected_method_family_pairs": len(expected),
        "observed_rows": len(rows),
        "all_paths_match": bool(len(frame)) and bool(frame["path_match"].all()),
        "all_hashes_match": bool(len(frame)) and bool(frame["hash_match"].all()),
        "errors": errors,
    }
    return frame, audit


def _flatten_prompt_text(value: Any) -> str:
    """Decode text from OpenAI messages and retained Python/JSON payloads."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_flatten_prompt_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_flatten_prompt_text(item) for item in value)
    return ""


def _generated_episode_root(artifact_path: Path) -> Path:
    """Resolve a generated-code artifact back to its scored episode root.

    Normalized RATS rows normally retain ``artifacts/iteration_*.json`` while
    older fixtures and timeout rows may retain the episode directory itself.
    Prompt telemetry lives at episode scope, so support both representations
    without depending on a fixed number of parent directories.
    """

    candidate = artifact_path if artifact_path.is_dir() else artifact_path.parent
    for root in (candidate, *candidate.parents):
        if (root / "agent_io").is_dir() or (root / "EVIDENCE.json").is_file():
            return root
    return candidate


def _generated_prompt_has_card(task_root: Path, method: str, card: str) -> tuple[bool, int]:
    """Inspect the requests retained by one generated-code episode."""

    inspected = 0
    if method == "capx":
        paths = sorted(task_root.glob("*/artifacts/initial_prompt.txt"))
        for path in paths:
            inspected += 1
            raw = path.read_text(encoding="utf-8", errors="replace")
            try:
                decoded: Any = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                decoded = raw
            if card in _flatten_prompt_text(decoded):
                return True, inspected
        return False, inspected

    for path in sorted((task_root / "agent_io").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        inspected += 1
        if card in _flatten_prompt_text(payload.get("request") or {}):
            return True, inspected
    return False, inspected


def _racap_prompt_card_index(group_root: Path, card: str) -> tuple[set[str], int]:
    """Index RACaP episode keys whose retained requests contain ``card``."""

    matched: set[str] = set()
    inspected = 0
    expected_hash = hashlib.sha256(card[:4_000].encode()).hexdigest()
    for path in sorted(group_root.glob("llm_calls*.jsonl")):
        for payload in _jsonl(path):
            inspected += 1
            episode_key = str(payload.get("episode_key") or "")
            hash_attested = bool(payload.get("one_shot_card_in_request")) and str(
                payload.get("one_shot_card_sha256") or ""
            ) == expected_hash
            text_attested = card in _flatten_prompt_text(payload.get("request") or {})
            if episode_key and (hash_attested or text_attested):
                matched.add(episode_key)
    return matched, inspected


def _first_prompt_token_index(group_root: Path) -> dict[str, int]:
    """Return the first hosted-model prompt size for every RACaP episode."""

    first: dict[str, tuple[float, int]] = {}
    for path in sorted(group_root.glob("llm_calls*.jsonl")):
        for payload in _jsonl(path):
            episode_key = str(payload.get("episode_key") or "")
            usage = payload.get("usage") or {}
            raw_tokens = payload.get("prompt_tokens", usage.get("prompt_tokens"))
            if not episode_key or raw_tokens is None:
                continue
            timestamp = float(payload.get("time") or payload.get("timestamp") or 0.0)
            candidate = (timestamp, int(raw_tokens))
            if episode_key not in first or candidate[0] < first[episode_key][0]:
                first[episode_key] = candidate
    return {key: value[1] for key, value in first.items()}


def _racap_loader_audit() -> dict[str, Any]:
    """Verify the source-level card path used by legacy RACaP telemetry.

    RACaP Phase 1 was measured before request-hash attestation was added.  We do
    not relabel those rows as request-level proof.  Instead this audit records
    the weaker launch-chain evidence: the run manifest binds a retained card,
    and the runtime loader reads the registered environment variable into
    agent memory.  First-call token deltas are retained separately as runtime
    corroboration rather than treated as exact proof.
    """

    source = ROOT / "racap" / "agent" / "experience.py"
    text = source.read_text(encoding="utf-8") if source.is_file() else ""
    required = (
        'ONE_SHOT_CARD_ENV = "CONTROLLED_ONE_SHOT_EXPERIENCE_CARD"',
        "def _read_optional_card()",
        "card = _read_optional_card()",
        "ONE-TRIAL TARGET-DOMAIN EXPERIENCE CARD",
    )
    return {
        "supported": bool(text) and all(marker in text for marker in required),
        "source": str(source.resolve()),
        "source_sha256": _sha256(source) if source.is_file() else None,
        "required_markers_present": {
            marker: marker in text for marker in required
        },
    }


def _test_card_prompt_audit(
    episodes: pd.DataFrame,
    cards: pd.DataFrame,
    *,
    zero_episodes: pd.DataFrame | None = None,
    delivery_rows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit exact and legacy evidence that cards reached scored policies.

    A manifest proves which file the launcher intended to inject.  This audit
    prefers the stronger execution-level claim by decoding retained requests
    or checking a text-free request hash.  Legacy RACaP Phase 1 telemetry did
    not retain either field; those rows remain explicitly labeled as weaker
    launch-chain corroboration and are never counted as request-level proof.
    """

    card_index = {
        (str(row.method), str(row.family)): Path(str(row.card)).read_text(
            encoding="utf-8"
        ).strip()
        for row in cards.itertuples()
    }
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    racap_cache: dict[tuple[str, str], tuple[set[str], int]] = {}
    token_cache: dict[str, dict[str, int]] = {}
    zero_index = {
        (
            str(row.method),
            str(row.suite),
            int(row.task_id),
            int(row.seed),
        ): row
        for row in (zero_episodes.itertuples() if zero_episodes is not None else [])
    }
    delivered = {
        (str(row.method), str(row.family)): bool(row.path_match and row.hash_match)
        for row in (
            delivery_rows.itertuples()
            if delivery_rows is not None and not delivery_rows.empty
            else []
        )
    }
    loader_audit = _racap_loader_audit()
    for episode in episodes.itertuples():
        method = str(episode.method)
        family = _family(str(episode.suite))
        card = card_index.get((method, family), "")
        root = Path(str(episode.artifact_path))
        if method in {"capx", "rats_base", "rats_90"}:
            root = _generated_episode_root(root)
        found = False
        inspected = 0
        first_prompt_token_delta: int | None = None
        if not card:
            errors.append(f"missing card content for {method}/{family}")
        elif method in {"capx", "rats_base", "rats_90"}:
            found, inspected = _generated_prompt_has_card(root, method, card)
        else:
            cache_key = (str(root.resolve()), hashlib.sha256(card.encode()).hexdigest())
            if cache_key not in racap_cache:
                racap_cache[cache_key] = _racap_prompt_card_index(root, card)
            matched, inspected = racap_cache[cache_key]
            found = str(episode.episode_key) in matched
        legacy_launch_chain = bool(
            method == "racap_phase1"
            and delivered.get((method, family), False)
            and loader_audit["supported"]
        )
        zero_row = zero_index.get(
            (method, str(episode.suite), int(episode.task_id), int(episode.seed))
        )
        if method == "racap_phase1" and zero_row is not None:
            zero_root = Path(str(zero_row.artifact_path))
            for token_root in (root, zero_root):
                token_key = str(token_root.resolve())
                if token_key not in token_cache:
                    token_cache[token_key] = _first_prompt_token_index(token_root)
            one_tokens = token_cache[str(root.resolve())].get(str(episode.episode_key))
            zero_tokens = token_cache[str(zero_root.resolve())].get(
                str(episode.episode_key)
            )
            if one_tokens is not None and zero_tokens is not None:
                first_prompt_token_delta = int(one_tokens - zero_tokens)
        delivery_supported = bool(found or legacy_launch_chain)
        rows.append(
            {
                "method": method,
                "method_label": LABELS[method],
                "family": family,
                "suite": str(episode.suite),
                "task_id": int(episode.task_id),
                "seed": int(episode.seed),
                "episode_key": str(episode.episode_key),
                "artifact_path": str(root.resolve()),
                "prompt_card_found": bool(found),
                "exact_request_attestation": bool(found),
                "legacy_launch_chain_corroboration": legacy_launch_chain,
                "delivery_supported": delivery_supported,
                "verification_basis": (
                    "retained_request_or_text_free_request_hash"
                    if found
                    else (
                        "launch_manifest_runtime_loader"
                        if legacy_launch_chain
                        else "none"
                    )
                ),
                "first_prompt_token_delta_vs_zero": first_prompt_token_delta,
                "request_records_inspected": int(inspected),
            }
        )
        if not delivery_supported:
            errors.append(
                f"no execution or launch-chain card evidence: {episode.episode_key}"
            )
    frame = pd.DataFrame(rows)
    frame["suite_modal_first_prompt_token_delta"] = np.nan
    frame["first_prompt_token_signature_match"] = False
    if not frame.empty:
        oracle = frame[
            (frame["method"] == "racap_phase1")
            & frame["first_prompt_token_delta_vs_zero"].notna()
        ]
        # Prompt scaffolds differ between *_swap and *_task suites even when
        # they share the same family card, so the appropriate runtime signature
        # is the mode within one suite rather than across the entire family.
        for suite, group in oracle.groupby("suite"):
            values = [int(value) for value in group["first_prompt_token_delta_vs_zero"]]
            modal = Counter(values).most_common(1)[0][0]
            mask = (frame["method"] == "racap_phase1") & (frame["suite"] == suite)
            frame.loc[mask, "suite_modal_first_prompt_token_delta"] = modal
            frame.loc[mask, "first_prompt_token_signature_match"] = (
                pd.to_numeric(
                    frame.loc[mask, "first_prompt_token_delta_vs_zero"],
                    errors="coerce",
                )
                == modal
            )
    expected = 60 * len(METHODS)
    if len(frame) != expected:
        errors.append(f"prompt-delivery coverage mismatch: {len(frame)}/{expected}")
    audit = {
        "schema_version": 1,
        "status": "pass" if not errors else "error",
        "expected_episodes": expected,
        "observed_episodes": len(frame),
        "episodes_with_card_in_retained_request": int(
            frame["prompt_card_found"].sum() if not frame.empty else 0
        ),
        "episodes_with_exact_request_attestation": int(
            frame["exact_request_attestation"].sum() if not frame.empty else 0
        ),
        "episodes_with_legacy_launch_chain_corroboration": int(
            frame["legacy_launch_chain_corroboration"].sum()
            if not frame.empty
            else 0
        ),
        "episodes_with_delivery_support": int(
            frame["delivery_supported"].sum() if not frame.empty else 0
        ),
        "legacy_first_prompt_token_signature_matches": int(
            frame["first_prompt_token_signature_match"].sum()
            if not frame.empty
            else 0
        ),
        "legacy_first_prompt_token_signature_rows": int(
            frame["first_prompt_token_delta_vs_zero"].notna().sum()
            if not frame.empty
            else 0
        ),
        "legacy_racap_loader_audit": loader_audit,
        "audit_basis": (
            "exact retained-request/hash attestation where instrumented; "
            "explicitly labeled launch-manifest/runtime-loader corroboration "
            "for legacy RACaP Phase 1"
        ),
        "errors": errors,
    }
    return frame, audit


def _adaptation_budget(
    calibration: pd.DataFrame, card_calls: pd.DataFrame
) -> pd.DataFrame:
    """Account for both the labeled rollout and card-construction overhead."""

    metrics = (
        "policy_wall_seconds",
        "model_latency_seconds",
        "model_calls",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "estimated_api_cost_usd",
        "simulator_steps",
    )
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        cal = calibration[calibration["method"] == method]
        calls = card_calls[card_calls["method"] == method] if len(card_calls) else card_calls
        row: dict[str, Any] = {
            "method": method,
            "method_label": LABELS[method],
            "calibration_episodes": int(len(cal)),
            "card_generation_calls": int(len(calls)),
        }
        for metric in metrics:
            cal_value = (
                float(pd.to_numeric(cal[metric], errors="coerce").fillna(0).sum())
                if metric in cal
                else 0.0
            )
            # Card construction has no simulator trajectory. Its wall overhead
            # is represented by hosted-model latency; frame extraction remains
            # in the global orchestration wall clock.
            card_metric = "model_latency_seconds" if metric == "policy_wall_seconds" else metric
            card_value = (
                float(
                    pd.to_numeric(calls[card_metric], errors="coerce")
                    .fillna(0)
                    .sum()
                )
                if card_metric in calls
                else 0.0
            )
            row[f"calibration_{metric}"] = cal_value
            row[f"card_generation_{metric}"] = card_value
            row[f"total_adaptation_{metric}"] = cal_value + card_value
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(zero: pd.DataFrame, adapted: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "suite", "task_id", "seed"]
    metrics = [
        "native_success",
        "predicate_completion",
        "policy_wall_seconds",
        "model_calls",
        "prompt_tokens",
        "completion_tokens",
        "estimated_api_cost_usd",
        "simulator_steps",
    ]
    merged = zero[keys + metrics].merge(
        adapted[keys + metrics],
        on=keys,
        how="inner",
        suffixes=("_zero", "_one_trial"),
        validate="one_to_one",
    )
    merged["family"] = merged["suite"].map(_family)
    return merged


def _aggregate(paired: pd.DataFrame, bootstrap: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(20260822)

    def record(method: str, family: str, group: pd.DataFrame) -> dict[str, Any]:
        zero = group["native_success_zero"].astype(bool)
        one = group["native_success_one_trial"].astype(bool)
        a_only = int((zero & ~one).sum())
        b_only = int((~zero & one).sum())
        discordant = a_only + b_only
        low_zero, high_zero = _wilson(int(zero.sum()), len(group))
        low_one, high_one = _wilson(int(one.sum()), len(group))
        task_key = pd.Series(
            [f"{suite}/{int(task_id)}" for suite, task_id in zip(
                group["suite"], group["task_id"]
            )],
            index=group.index,
        )

        def task_bootstrap(metric_one: str, metric_zero: str) -> tuple[float, float, float]:
            delta = pd.to_numeric(group[metric_one], errors="coerce").astype(
                float
            ) - pd.to_numeric(group[metric_zero], errors="coerce").astype(float)
            valid = delta.notna()
            task_means = (
                pd.DataFrame(
                    {"task": task_key[valid].to_numpy(), "delta": delta[valid].to_numpy()}
                )
                .groupby("task", sort=True)["delta"]
                .mean()
                .to_numpy()
            )
            if not len(task_means):
                return float("nan"), float("nan"), float("nan")
            samples = rng.choice(
                task_means, size=(bootstrap, len(task_means)), replace=True
            ).mean(1)
            return (
                float(task_means.mean()),
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            )

        gain, gain_low, gain_high = task_bootstrap(
            "native_success_one_trial", "native_success_zero"
        )
        wall, wall_low, wall_high = task_bootstrap(
            "policy_wall_seconds_one_trial", "policy_wall_seconds_zero"
        )
        calls, calls_low, calls_high = task_bootstrap(
            "model_calls_one_trial", "model_calls_zero"
        )
        cost, cost_low, cost_high = task_bootstrap(
            "estimated_api_cost_usd_one_trial", "estimated_api_cost_usd_zero"
        )
        return {
            "method": method,
            "method_label": LABELS[method],
            "family": family,
            "episodes": len(group),
            "zero_shot_successes": int(zero.sum()),
            "one_trial_successes": int(one.sum()),
            "zero_shot_rate": float(zero.mean()),
            "one_trial_rate": float(one.mean()),
            "absolute_gain": gain,
            "gain_task_bootstrap_low": gain_low,
            "gain_task_bootstrap_high": gain_high,
            "paired_task_clusters": int(task_key.nunique()),
            "bootstrap_unit": "suite_task",
            "zero_wilson_low": low_zero,
            "zero_wilson_high": high_zero,
            "one_wilson_low": low_one,
            "one_wilson_high": high_one,
            "zero_only_success": a_only,
            "one_trial_only_success": b_only,
            "mcnemar_exact_p": (
                binomtest(min(a_only, b_only), discordant, 0.5).pvalue
                if discordant
                else 1.0
            ),
            "mean_wall_delta_seconds": wall,
            "wall_delta_bootstrap_low": wall_low,
            "wall_delta_bootstrap_high": wall_high,
            "mean_model_call_delta": calls,
            "model_call_delta_bootstrap_low": calls_low,
            "model_call_delta_bootstrap_high": calls_high,
            "mean_cost_delta_usd": cost,
            "cost_delta_bootstrap_low": cost_low,
            "cost_delta_bootstrap_high": cost_high,
        }

    rows: list[dict[str, Any]] = [
        record(method, family, group)
        for (method, family), group in paired.groupby(["method", "family"])
    ]
    # Overall row per method, in addition to the three family rows.
    for method, group in paired.groupby("method"):
        rows.append(record(method, "all", group))
    adjusted = _holm([float(row["mcnemar_exact_p"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["mcnemar_holm_p"] = value
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, output: Path) -> None:
    part = summary[summary["family"] == "all"].set_index("method")
    methods = [method for method in METHODS if method in part.index]
    if not methods:
        return
    x = list(range(len(methods)))
    fig, axis = plt.subplots(figsize=(7.0, 3.65))
    zero = np.asarray([part.loc[method, "zero_shot_rate"] for method in methods])
    one = np.asarray([part.loc[method, "one_trial_rate"] for method in methods])
    zero_low = zero - np.asarray(
        [part.loc[method, "zero_wilson_low"] for method in methods]
    )
    zero_high = np.asarray(
        [part.loc[method, "zero_wilson_high"] for method in methods]
    ) - zero
    one_low = one - np.asarray(
        [part.loc[method, "one_wilson_low"] for method in methods]
    )
    one_high = np.asarray(
        [part.loc[method, "one_wilson_high"] for method in methods]
    ) - one
    axis.bar(
        [value - 0.19 for value in x],
        zero,
        width=0.38,
        yerr=[zero_low, zero_high],
        capsize=2.5,
        label="Zero-shot (same seed-1 slice)",
        color="#A6A6A6",
        edgecolor="white",
        linewidth=0.5,
    )
    axis.bar(
        [value + 0.19 for value in x],
        one,
        width=0.38,
        yerr=[one_low, one_high],
        capsize=2.5,
        label="After one labeled interaction",
        color="#D55E00",
        edgecolor="white",
        linewidth=0.5,
    )
    axis.set_xticks(x, [LABELS[method] for method in methods], rotation=25, ha="right")
    axis.set_ylabel("Native success rate")
    axis.set_ylim(0, 1)
    clean_axis(axis, grid_axis="y")
    axis.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zero-shot-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "measured",
    )
    parser.add_argument(
        "--zero-shot-index",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "controlled_comparison"
            / "analysis"
            / "consolidated_main"
            / "pro_five_method"
            / "episodes.csv"
        ),
        help=(
            "Audited normalized zero-shot table. Reading this avoids rescanning "
            "large retained generated-code request payloads; pass an empty/nonexistent "
            "path only when rebuilding directly from raw artifacts."
        ),
    )
    parser.add_argument(
        "--calibration-cohort",
        choices=["one_shot_calibration", "libero_pro_zero_shot"],
        default="libero_pro_zero_shot",
    )
    parser.add_argument(
        "--calibration-index",
        type=Path,
        help="normalized 15-row reuse index; avoids rescanning all measured runs",
    )
    parser.add_argument(
        "--one-shot-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "one_shot" / "measured",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "measured",
    )
    parser.add_argument(
        "--cards-dir",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "one_shot" / "cards",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "analysis" / "one_shot",
    )
    args = parser.parse_args()
    zero_all = (
        pd.read_csv(args.zero_shot_index)
        if args.zero_shot_index is not None and args.zero_shot_index.is_file()
        else _load(args.zero_shot_root)
    )
    one_all = _load(args.one_shot_root)
    if zero_all.empty or one_all.empty:
        raise SystemExit("zero-shot or one-trial measured results are missing")
    zero = zero_all[
        (zero_all["cohort"] == "libero_pro_zero_shot") & (zero_all["seed"] == 1)
    ]
    one = one_all[one_all["cohort"] == "libero_pro_one_shot"]
    paired = _paired(zero, one)
    if len(paired) != 60 * len(METHODS):
        raise SystemExit(
            f"incomplete paired one-trial grid: {len(paired)}/{60 * len(METHODS)}"
        )
    summary = _aggregate(paired)
    adapted_method_comparison = _paired_statistics(one)
    calibration_source = (
        pd.read_csv(args.calibration_index)
        if args.calibration_index is not None
        else _load(args.calibration_root)
    )
    calibration = _select_calibration(
        calibration_source, args.calibration_cohort
    )
    card_calls = _card_generation_calls(args.cards_dir)
    adaptation_budget = _adaptation_budget(calibration, card_calls)
    cards_manifest = args.cards_dir / "manifest.json"
    cards = pd.DataFrame(
        (json.loads(cards_manifest.read_text(encoding="utf-8")).get("cards") or [])
        if cards_manifest.is_file()
        else []
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adaptation_audit = _adaptation_artifact_audit(calibration, cards, card_calls)
    adaptation_audit["calibration_root"] = str(args.calibration_root.resolve())
    adaptation_audit["calibration_index"] = (
        str(args.calibration_index.resolve())
        if args.calibration_index is not None
        else None
    )
    adaptation_audit["calibration_cohort"] = args.calibration_cohort
    adaptation_audit["zero_shot_index"] = (
        str(args.zero_shot_index.resolve())
        if args.zero_shot_index is not None and args.zero_shot_index.is_file()
        else None
    )
    adaptation_audit["zero_shot_index_sha256"] = (
        _sha256(args.zero_shot_index)
        if args.zero_shot_index is not None and args.zero_shot_index.is_file()
        else None
    )
    adaptation_audit["calibration_selector"] = {
        "suites": sorted(FAMILY_SUITE.values()),
        "task_id": 0,
        "seed": 0,
    }
    (args.output_dir / "adaptation_audit.json").write_text(
        json.dumps(adaptation_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if adaptation_audit["status"] != "pass":
        raise SystemExit(
            "incomplete or inconsistent one-trial adaptation artifacts; "
            "see adaptation_audit.json"
        )
    delivery_rows, delivery_audit = _test_card_delivery_audit(
        args.one_shot_root, cards
    )
    (args.output_dir / "test_card_delivery_audit.json").write_text(
        json.dumps(delivery_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if delivery_audit["status"] != "pass":
        raise SystemExit(
            "one-trial experience-card delivery is incomplete or inconsistent; "
            "see test_card_delivery_audit.json"
        )
    prompt_rows, prompt_audit = _test_card_prompt_audit(
        one,
        cards,
        zero_episodes=zero,
        delivery_rows=delivery_rows,
    )
    (args.output_dir / "test_card_prompt_audit.json").write_text(
        json.dumps(prompt_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if prompt_audit["status"] != "pass":
        raise SystemExit(
            "one-trial experience card lacks execution or launch-chain evidence; "
            "see test_card_prompt_audit.json"
        )
    paired.to_csv(args.output_dir / "paired_episodes.csv", index=False)
    summary.to_csv(args.output_dir / "aggregate.csv", index=False)
    adapted_method_comparison.to_csv(
        args.output_dir / "adapted_method_pairwise.csv", index=False
    )
    card_calls.to_csv(args.output_dir / "card_generation_calls.csv", index=False)
    adaptation_budget.to_csv(args.output_dir / "adaptation_budget.csv", index=False)
    delivery_rows.to_csv(args.output_dir / "test_card_delivery.csv", index=False)
    prompt_rows.to_csv(args.output_dir / "test_card_prompt_delivery.csv", index=False)
    with pd.ExcelWriter(args.output_dir / "one_trial_adaptation.xlsx", engine="xlsxwriter") as writer:
        paired.to_excel(writer, sheet_name="Paired episodes", index=False)
        summary.to_excel(writer, sheet_name="Aggregate", index=False)
        adapted_method_comparison.to_excel(
            writer, sheet_name="Adapted method pairwise", index=False
        )
        calibration.to_excel(writer, sheet_name="Calibration", index=False)
        cards.to_excel(writer, sheet_name="Experience cards", index=False)
        card_calls.to_excel(writer, sheet_name="Card generation calls", index=False)
        adaptation_budget.to_excel(writer, sheet_name="Adaptation budget", index=False)
        delivery_rows.to_excel(writer, sheet_name="Test card delivery", index=False)
        prompt_rows.to_excel(writer, sheet_name="Prompt card delivery", index=False)
        pd.DataFrame([adaptation_audit]).to_excel(
            writer, sheet_name="Adaptation audit", index=False
        )
        pd.DataFrame([delivery_audit]).to_excel(
            writer, sheet_name="Delivery audit", index=False
        )
        pd.DataFrame([prompt_audit]).to_excel(
            writer, sheet_name="Prompt delivery audit", index=False
        )
        for sheet in writer.sheets.values():
            sheet.freeze_panes(1, 0)
    _plot(summary, args.output_dir / "zero_vs_one_trial")
    print(json.dumps({"paired_episodes": len(paired), "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
