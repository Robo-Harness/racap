#!/usr/bin/env python3
"""Fail closed on incomplete or structurally implausible method results.

This audit deliberately does not encode the expected method ranking. Native
success remains an empirical outcome. The gate only stops orchestration when
the registered evidence is incomplete, telemetry is inconsistent with the
declared method, the hosted-model route differs from the frozen protocol, or a
large actuator sanity cohort indicates that a known implementation may not be
connected correctly. The two frozen RACaP rows additionally have a
preregistered, deliberately loose in-domain reproduction floor far below their
archived seed-0 scores; crossing it triggers review but never changes a score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.controlled_comparison.analyze_results import MAIN_COHORTS, METHODS

GENERATED_METHODS = {"capx", "rats_base", "rats_90"}
FROZEN_SOURCE_METHODS = {"racap_phase1", "racap_phase2"}
# Archived native seed-0 scores were approximately 47/90 and 48/90. A 30/90
# review floor is intentionally loose: it catches broken routing, loading, or
# actuation while allowing substantial stochastic regression. It is not used
# for ranking, censoring, retries, or success computation.
HISTORICAL_ID_REVIEW_FLOORS = {"racap_phase1": 30, "racap_phase2": 30}


def _frozen_artifact_audit(path: Path) -> dict[str, Any]:
    """Validate the before/after content-hash ledger for a frozen RATS phase."""

    if not path.is_file() or path.stat().st_size <= 0:
        return {
            "status": "error",
            "path": str(path),
            "error": "missing frozen artifact audit",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "path": str(path),
            "error": f"invalid frozen artifact audit: {type(exc).__name__}: {exc}",
        }
    before = payload.get("before")
    after = payload.get("after")
    unchanged = payload.get("unchanged") is True and before == after
    return {
        "status": "pass" if unchanged else "error",
        "path": str(path.resolve()),
        "unchanged": unchanged,
        "before": before,
        "after": after,
        "error": None if unchanged else "frozen RATS artifact changed during evaluation",
    }


def _split_models(values: pd.Series) -> set[str]:
    models: set[str] = set()
    for value in values.fillna("").astype(str):
        models.update(part.strip() for part in value.split(",") if part.strip())
    return models


def _numeric_total(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _normalize_instruction(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def audit_method_frame(
    episodes: pd.DataFrame,
    completeness: pd.DataFrame,
    *,
    method: str,
    required_model: str = "gpt-5.5",
) -> dict[str, Any]:
    """Return a machine-readable, non-ranking audit of one completed method."""

    errors: list[str] = []
    review_reasons: list[str] = []
    warnings: list[str] = []
    rows = episodes[episodes["method"] == method].copy()
    expected = completeness[completeness["method"] == method].copy()

    duplicate_keys = sorted(
        rows.loc[rows.duplicated("episode_key", keep=False), "episode_key"]
        .astype(str)
        .unique()
        .tolist()
    )
    if duplicate_keys:
        errors.append(f"duplicate episode keys: {duplicate_keys[:10]}")

    missing = expected.loc[
        ~expected["completed"].astype(bool), "episode_key"
    ].astype(str).tolist()
    if missing:
        errors.append(f"missing {len(missing)} registered episodes")
    if len(rows) != len(expected):
        errors.append(f"parsed {len(rows)} episodes but expected {len(expected)}")

    absent_evidence: list[str] = []
    for row in rows.itertuples():
        path = Path(str(row.evidence_path))
        if not path.is_file() or path.stat().st_size <= 0:
            absent_evidence.append(str(row.episode_key))
    if absent_evidence:
        errors.append(f"missing/nonempty evidence for {len(absent_evidence)} episodes")

    total_calls = _numeric_total(rows, "model_calls")
    total_steps = _numeric_total(rows, "simulator_steps")
    total_generation = _numeric_total(rows, "source_generation_events")
    total_repairs = _numeric_total(rows, "source_repair_events")
    if total_calls <= 0:
        errors.append("no hosted-model calls were recorded")
    if total_steps <= 0:
        errors.append("no simulator steps were recorded")

    model_values = rows.get("actual_models", pd.Series(dtype=str))
    models = _split_models(model_values)
    if total_calls > 0 and not models:
        errors.append("model calls exist but no actual response model was recorded")
    unexpected_models = sorted(models - {required_model})
    if unexpected_models:
        errors.append(
            f"actual model route differs from {required_model}: {unexpected_models}"
        )

    if method in GENERATED_METHODS and total_generation <= 0:
        errors.append("generated-policy method recorded no deploy-time source generation")
    if method in FROZEN_SOURCE_METHODS and (total_generation or total_repairs):
        errors.append("frozen RACaP method unexpectedly generated or repaired source at runtime")
    if method in GENERATED_METHODS:
        if "delivered_instruction" not in rows or "instruction" not in rows:
            errors.append("generated-policy rows do not record delivered instructions")
        else:
            instruction_mismatches = []
            for row in rows.itertuples():
                registered = _normalize_instruction(row.instruction)
                delivered = _normalize_instruction(row.delivered_instruction)
                if not registered or registered != delivered:
                    instruction_mismatches.append(
                        {
                            "episode_key": str(row.episode_key),
                            "registered": str(row.instruction),
                            "delivered": str(row.delivered_instruction),
                        }
                    )
            if instruction_mismatches:
                errors.append(
                    "delivered task instructions do not exactly match the registered "
                    f"public instructions: {instruction_mismatches[:10]}"
                )
        if "simulator_episode_resets" not in rows:
            errors.append("generated-policy rows do not record simulator reset counts")
        else:
            reset_counts = pd.to_numeric(
                rows["simulator_episode_resets"], errors="coerce"
            )
            invalid_resets = rows.loc[
                reset_counts.isna() | (reset_counts != 1),
                ["episode_key", "simulator_episode_resets"],
            ].to_dict("records")
            if invalid_resets:
                errors.append(
                    "frozen evaluation must contain exactly one registered "
                    f"environment reset per episode: {invalid_resets[:10]}"
                )
    if "init_state_index" not in rows:
        errors.append("episode rows do not record the applied init-state index")
    else:
        applied = pd.to_numeric(rows["init_state_index"], errors="coerce")
        registered = pd.to_numeric(rows["seed"], errors="coerce")
        wrong_states = rows.loc[
            applied.isna() | registered.isna() | (applied != registered),
            ["episode_key", "seed", "init_state_index"],
        ].to_dict("records")
        if wrong_states:
            errors.append(
                "applied init-state indices do not match registered public seeds: "
                f"{wrong_states[:10]}"
            )

    cohort_rows: list[dict[str, Any]] = []
    for cohort in MAIN_COHORTS:
        part = rows[rows["cohort"] == cohort]
        successes = int(part["native_success"].astype(bool).sum()) if len(part) else 0
        cohort_rows.append(
            {
                "cohort": cohort,
                "episodes": int(len(part)),
                "successes": successes,
                "success_rate": successes / len(part) if len(part) else None,
            }
        )
        if len(part) and successes == 0:
            warnings.append(f"zero native successes in cohort {cohort}")

    successes = int(rows["native_success"].astype(bool).sum()) if len(rows) else 0
    id_rows = rows[rows["cohort"] == "libero90_id_replay"]
    id_successes = int(id_rows["native_success"].astype(bool).sum()) if len(id_rows) else 0
    # These are investigation gates, not score thresholds. With 350 registered
    # episodes (including 90 development-domain tasks), an all-zero controller
    # gives no evidence that the actuation/evaluation path is connected.
    if len(rows) and successes == 0:
        review_reasons.append("all registered episodes have zero native success")
    if len(id_rows) and id_successes == 0:
        review_reasons.append("all LIBERO-90 in-domain episodes have zero native success")
    historical_floor = HISTORICAL_ID_REVIEW_FLOORS.get(method)
    if (
        historical_floor is not None
        and len(id_rows) == 90
        and id_successes < historical_floor
    ):
        review_reasons.append(
            "frozen RACaP in-domain reproduction is below its preregistered "
            f"implementation sanity floor: {id_successes}/90 < {historical_floor}/90"
        )

    status = "error" if errors else "review_required" if review_reasons else "pass"
    return {
        "schema_version": 1,
        "method": method,
        "status": status,
        "ranking_assumption_enforced": False,
        "historical_id_review_floor": historical_floor,
        "required_model": required_model,
        "actual_models": sorted(models),
        "episodes_parsed": int(len(rows)),
        "episodes_expected": int(len(expected)),
        "native_successes": successes,
        "native_success_rate": successes / len(rows) if len(rows) else None,
        "total_model_calls": total_calls,
        "total_simulator_steps": total_steps,
        "total_source_generation_events": total_generation,
        "total_source_repair_events": total_repairs,
        "cohorts": cohort_rows,
        "errors": errors,
        "review_reasons": review_reasons,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--required-model", default="gpt-5.5")
    parser.add_argument(
        "--method-root",
        type=Path,
        default=None,
        help="Raw measured-method root containing frozen_artifact_audit.json.",
    )
    args = parser.parse_args()

    episodes = pd.read_csv(args.analysis_dir / "episodes.csv")
    completeness = pd.read_csv(args.analysis_dir / "completeness.csv")
    audit = audit_method_frame(
        episodes, completeness, method=args.method, required_model=args.required_model
    )
    if args.method == "rats_90":
        method_root = args.method_root or (
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "controlled_comparison"
            / "measured"
            / args.method
        )
        frozen = _frozen_artifact_audit(method_root / "frozen_artifact_audit.json")
        audit["frozen_artifact_audit"] = frozen
        if frozen["status"] != "pass":
            audit["errors"].append(str(frozen["error"]))
            audit["status"] = "error"
    output = args.analysis_dir / "method_audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
