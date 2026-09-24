#!/usr/bin/env python3
"""Assemble every controlled-comparison result into one audited workbook."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .analyze_results import _format_excel_sheet, _provenance_tables
except ImportError:  # direct script execution
    from analyze_results import _format_excel_sheet, _provenance_tables

from experiments.controlled_comparison.supervise_after_development import (
    _verify_frozen_artifact_seal,
)


ANALYSIS = ROOT / "outputs" / "controlled_comparison" / "analysis"
MEASURED = ROOT / "outputs" / "controlled_comparison" / "measured"
HERE = Path(__file__).resolve().parent
METHODS = ("capx", "rats_base", "rats_90", "racap_phase1", "racap_phase2")
FROZEN_RATS = ROOT / "outputs" / "controlled_comparison" / "artifacts" / "rats90_frozen"
CONSOLIDATED_MAIN = ANALYSIS / "consolidated_main"
MAIN_FOUR = CONSOLIDATED_MAIN / "main_four_method"
PRO_FIVE = CONSOLIDATED_MAIN / "pro_five_method"
LONG_FINAL = ANALYSIS / "long_horizon"
PUBLISHED_RESULTS = ROOT / "experiments" / "paper_results.csv"
PUBLISHED_PROVENANCE = ROOT / "experiments" / "PROVENANCE.md"
PHASE2_MANIFEST = ROOT / "policies" / "phase2" / "PHASE2_MANIFEST.md"
PHASE2_FULL90 = (
    ROOT
    / "outputs"
    / "full_agent_eval"
    / "phase2_full90"
)
METHOD_AUDIT_DIRS = {
    "capx": "capx",
    "rats_base": "rats_base",
    "rats_90": "rats_90",
    "racap_phase1": "racap_phase1",
    # The champion's current registered coverage is PRO-180, not the complete
    # 350-episode grid.  Naming this explicitly prevents accidental imputation.
    "racap_phase2": "racap_phase2_pro180",
}


TABLES: tuple[tuple[str, Path], ...] = (
    ("Main4 Episodes", MAIN_FOUR / "episodes.csv"),
    ("Main4 Aggregate", MAIN_FOUR / "aggregate.csv"),
    ("Main4 Pairwise", MAIN_FOUR / "pairwise.csv"),
    ("Main4 Completeness", MAIN_FOUR / "completeness.csv"),
    ("Main4 Coverage", MAIN_FOUR / "coverage_contract.csv"),
    ("PRO5 Episodes", PRO_FIVE / "episodes.csv"),
    ("PRO5 Aggregate", PRO_FIVE / "aggregate.csv"),
    ("PRO5 Pairwise", PRO_FIVE / "pairwise.csv"),
    ("PRO5 Completeness", PRO_FIVE / "completeness.csv"),
    ("PRO5 Coverage", PRO_FIVE / "coverage_contract.csv"),
    ("OneTrial Paired", ANALYSIS / "one_shot" / "paired_episodes.csv"),
    ("OneTrial Aggregate", ANALYSIS / "one_shot" / "aggregate.csv"),
    ("OneTrial Methods", ANALYSIS / "one_shot" / "adapted_method_pairwise.csv"),
    ("Adaptation Budget", ANALYSIS / "one_shot" / "adaptation_budget.csv"),
    ("Card Calls", ANALYSIS / "one_shot" / "card_generation_calls.csv"),
    ("Card Delivery", ANALYSIS / "one_shot" / "test_card_delivery.csv"),
    ("Prompt Delivery", ANALYSIS / "one_shot" / "test_card_prompt_delivery.csv"),
    ("Long Episodes", LONG_FINAL / "episodes.csv"),
    ("Long Aggregate", LONG_FINAL / "aggregate.csv"),
    ("Long Pairwise", LONG_FINAL / "pairwise.csv"),
    ("Long Completeness", LONG_FINAL / "completeness.csv"),
    ("RS Transfer Episodes", ANALYSIS / "robosuite_transfer" / "episodes.csv"),
    ("RS Transfer Aggregate", ANALYSIS / "robosuite_transfer" / "aggregate.csv"),
    ("RS Transfer Paired", ANALYSIS / "robosuite_transfer" / "paired.csv"),
    ("RS Transfer Complete", ANALYSIS / "robosuite_transfer" / "completeness.csv"),
    ("RS Dev Episodes", ANALYSIS / "robosuite_development" / "episodes.csv"),
    ("RS Dev Candidates", ANALYSIS / "robosuite_development" / "candidates.csv"),
    ("RS Dev Invalidated", ANALYSIS / "robosuite_development" / "invalidated_attempts.csv"),
    ("Development RATS", ANALYSIS / "development" / "rats_development.csv"),
    ("Development RACaP", ANALYSIS / "development" / "racap_development.csv"),
    ("Development Usage", ANALYSIS / "development" / "rats_observed_usage.csv"),
    ("Development Calls", ANALYSIS / "development" / "rats_call_categories.csv"),
    ("Development Attempts", ANALYSIS / "development" / "rats_attempt_diagnostics.csv"),
    ("Development SelfCheck", ANALYSIS / "development" / "rats_runtime_self_checks.csv"),
    ("Development Resumes", ANALYSIS / "development" / "rats_resume_transactions.csv"),
    ("Development Admin", ANALYSIS / "development" / "rats_administrative_events.csv"),
    ("Development Identity", ANALYSIS / "development" / "rats_identity_search.csv"),
    ("Development NativeEvents", ANALYSIS / "development" / "rats_native_event_audit.csv"),
    ("Development PromptAudit", ANALYSIS / "development" / "rats_prompt_privilege_audit.csv"),
    ("Development Critic", ANALYSIS / "development" / "rats_critic_evidence_audits.csv"),
    ("Development VisualAudit", ANALYSIS / "development" / "rats_human_visual_spot_checks.csv"),
    ("Development SkillAdmission", ANALYSIS / "development" / "rats_skill_admission_audit.csv"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_complete(frame: pd.DataFrame, *, name: str, expected: int) -> None:
    if len(frame) != expected:
        raise SystemExit(f"{name} cardinality mismatch: {len(frame)}/{expected}")
    if "completed" not in frame:
        raise SystemExit(f"{name} has no completed column")
    completed = frame["completed"]
    if completed.dtype != bool:
        completed = completed.astype(str).str.lower().isin({"true", "1"})
    if not bool(completed.all()):
        raise SystemExit(f"{name} is incomplete: {int(completed.sum())}/{len(frame)}")


def _method_audit_frame() -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for method in METHODS:
        path = (
            ANALYSIS
            / "method_audits"
            / METHOD_AUDIT_DIRS[method]
            / "method_audit.json"
        )
        if not path.is_file():
            raise SystemExit(f"missing method audit: {path}")
        paths.append(path)
        payload = _read_json(path)
        rows.append(
            {
                "method": method,
                "status": payload.get("status"),
                "ranking_assumption_enforced": payload.get(
                    "ranking_assumption_enforced"
                ),
                "episodes_parsed": payload.get("episodes_parsed"),
                "episodes_expected": payload.get("episodes_expected"),
                "native_successes": payload.get("native_successes"),
                "native_success_rate": payload.get("native_success_rate"),
                "required_model": payload.get("required_model"),
                "actual_models": ", ".join(payload.get("actual_models") or []),
                "errors": json.dumps(payload.get("errors") or [], ensure_ascii=False),
                "review_reasons": json.dumps(
                    payload.get("review_reasons") or [], ensure_ascii=False
                ),
                "warnings": json.dumps(
                    payload.get("warnings") or [], ensure_ascii=False
                ),
                "frozen_artifact_status": (
                    (payload.get("frozen_artifact_audit") or {}).get("status")
                ),
                "frozen_artifact_unchanged": (
                    (payload.get("frozen_artifact_audit") or {}).get("unchanged")
                ),
            }
        )
    return pd.DataFrame(rows), paths


def _audit(
    tables: dict[str, pd.DataFrame], method_audits: pd.DataFrame
) -> dict[str, Any]:
    _require_complete(
        tables["Main4 Completeness"], name="four-method main grid", expected=1400
    )
    _require_complete(
        tables["PRO5 Completeness"], name="five-method PRO grid", expected=900
    )
    _require_complete(tables["Long Completeness"], name="long grid", expected=75)
    _require_complete(
        tables["RS Transfer Complete"],
        name="five-method Robosuite transfer grid",
        expected=175,
    )
    if len(tables["OneTrial Paired"]) != 300:
        raise SystemExit(
            f"one-trial paired grid mismatch: {len(tables['OneTrial Paired'])}/300"
        )
    if set(method_audits["method"].astype(str)) != set(METHODS):
        raise SystemExit("method audit grid does not contain all five frozen methods")
    nonpassing = method_audits.loc[
        method_audits["status"].astype(str) != "pass", ["method", "status"]
    ].to_dict("records")
    if nonpassing:
        raise SystemExit(f"method audits did not all pass: {nonpassing}")
    expected_rows = {
        "Main4 Episodes": 1400,
        "PRO5 Episodes": 900,
        "Long Episodes": 75,
        "RS Transfer Episodes": 175,
        "RS Dev Episodes": 300,
        "RS Dev Candidates": 20,
    }
    for name, expected in expected_rows.items():
        if len(tables[name]) != expected:
            raise SystemExit(f"{name} cardinality mismatch: {len(tables[name])}/{expected}")

    adaptation_path = ANALYSIS / "one_shot" / "adaptation_audit.json"
    delivery_path = ANALYSIS / "one_shot" / "test_card_delivery_audit.json"
    prompt_delivery_path = ANALYSIS / "one_shot" / "test_card_prompt_audit.json"
    development_path = ANALYSIS / "development" / "summary.json"
    robosuite_transfer_audit_path = ANALYSIS / "robosuite_transfer" / "audit.json"
    robosuite_development_audit_path = (
        ANALYSIS / "robosuite_development" / "audit.json"
    )
    if not all(
        path.is_file()
        for path in (
            adaptation_path,
            delivery_path,
            prompt_delivery_path,
            development_path,
            robosuite_transfer_audit_path,
            robosuite_development_audit_path,
        )
    ):
        raise SystemExit("missing adaptation, card-delivery, or development audit summary")
    adaptation = _read_json(adaptation_path)
    delivery = _read_json(delivery_path)
    prompt_delivery = _read_json(prompt_delivery_path)
    development = _read_json(development_path)
    robosuite_transfer_audit = _read_json(robosuite_transfer_audit_path)
    robosuite_development_audit = _read_json(robosuite_development_audit_path)
    if adaptation.get("status") != "pass":
        raise SystemExit("one-trial adaptation audit did not pass")
    if delivery.get("status") != "pass":
        raise SystemExit("one-trial card file/hash delivery audit did not pass")
    if prompt_delivery.get("status") != "pass":
        raise SystemExit("one-trial request-level prompt delivery audit did not pass")
    if int(prompt_delivery.get("episodes_with_delivery_support") or 0) != 300:
        raise SystemExit("one-trial card-delivery evidence does not cover all 300 episodes")
    # RACaP Phase 1's 60 trials predate text-free request attestation.  They are
    # intentionally retained as the weaker, separately labeled launch-chain
    # tier rather than being misrepresented as request-level proof.
    if int(prompt_delivery.get("episodes_with_exact_request_attestation") or 0) != 240:
        raise SystemExit("one-trial exact request attestation does not cover 240 episodes")
    if int(
        prompt_delivery.get("episodes_with_legacy_launch_chain_corroboration") or 0
    ) != 60:
        raise SystemExit("legacy RACaP Phase 1 launch-chain tier does not cover 60 episodes")
    if robosuite_transfer_audit.get("status") != "pass":
        raise SystemExit("five-method Robosuite transfer audit did not pass")
    if int(robosuite_transfer_audit.get("episodes") or 0) != 175:
        raise SystemExit("five-method Robosuite transfer grid is not 175 episodes")
    if robosuite_transfer_audit.get("actual_models") != ["gpt-5.5"]:
        raise SystemExit("Robosuite transfer responses were not exclusively GPT-5.5")
    if robosuite_development_audit.get("status") != "pass":
        raise SystemExit("matched Robosuite development audit did not pass")
    if int(robosuite_development_audit.get("episodes") or 0) != 300:
        raise SystemExit("compute-matched Robosuite development is not 300 episodes")
    if int(development.get("rats_forbidden_prompt_payload_hits") or 0) != 0:
        raise SystemExit("RATS development contains forbidden private prompt payloads")
    if int(development.get("rats_native_event_unexplained_key_mismatches") or 0) != 0:
        raise SystemExit("RATS development contains unexplained task-key mismatches")
    if int(development.get("rats_resume_transaction_failures") or 0) != 0:
        raise SystemExit("RATS development contains a failed resume transaction audit")
    if int(
        development.get(
            "rats_critic_evidence_conflicts_changed_algorithm_mid_chain"
        )
        or 0
    ) != 0:
        raise SystemExit("RATS development algorithm changed after a critic audit")
    if int(development.get("rats_skill_admission_failures") or 0) != 0:
        raise SystemExit(
            "RATS development contains an unaudited skill-library admission"
        )
    if int(development.get("rats_human_visual_spot_checks_changed_control") or 0) != 0:
        raise SystemExit("a human visual spot check changed the running RATS controller")
    if int(development.get("rats_human_visual_spot_checks") or 0) != len(
        tables["Development VisualAudit"]
    ):
        raise SystemExit("human visual spot-check ledger cardinality mismatch")
    if str(development.get("rats_actual_models_observed") or "") != "gpt-5.5":
        raise SystemExit("RATS development was not exclusively routed to gpt-5.5")
    if int(development.get("rats_iterations_complete") or 0) <= 0:
        raise SystemExit("RATS development has no completed update round")
    frozen_seal = _verify_frozen_artifact_seal(FROZEN_RATS)
    if frozen_seal["status"] != "pass":
        raise SystemExit(f"frozen RATS artifact seal failed: {frozen_seal['error']}")
    frozen_manifest = _read_json(FROZEN_RATS / "artifact_manifest.json")
    budget_boundary = frozen_manifest.get("budget_boundary_transaction")
    if frozen_manifest.get("termination") == "simulator_reset_budget":
        if not isinstance(budget_boundary, dict) or (
            budget_boundary.get("status") != "pass"
            or not budget_boundary.get("artifact_matches_last_completed_snapshot")
        ):
            raise SystemExit(
                "reset-budget artifact does not prove a committed snapshot boundary"
            )
    return {
        "schema_version": 1,
        "status": "pass",
        "main_four_method_registered_episodes": len(
            tables["Main4 Completeness"]
        ),
        "pro_five_method_registered_episodes": len(
            tables["PRO5 Completeness"]
        ),
        "one_trial_paired_episodes": len(tables["OneTrial Paired"]),
        "long_checkpoint_rows": len(tables["Long Completeness"]),
        "robosuite_transfer_registered_episodes": len(
            tables["RS Transfer Complete"]
        ),
        "robosuite_development_episodes": len(tables["RS Dev Episodes"]),
        "robosuite_development_cohorts": len(tables["RS Dev Candidates"]),
        "rats_development_rounds": int(development["rats_iterations_complete"]),
        "rats_development_resets": int(
            development["rats_simulator_resets_observed"]
        ),
        "adaptation_audit_status": adaptation["status"],
        "card_delivery_audit_status": delivery["status"],
        "prompt_delivery_audit_status": prompt_delivery["status"],
        "robosuite_transfer_audit_status": robosuite_transfer_audit["status"],
        "robosuite_development_audit_status": robosuite_development_audit[
            "status"
        ],
        "episodes_with_card_in_retained_request": int(
            prompt_delivery["episodes_with_card_in_retained_request"]
        ),
        "episodes_with_exact_request_attestation": int(
            prompt_delivery["episodes_with_exact_request_attestation"]
        ),
        "episodes_with_legacy_launch_chain_corroboration": int(
            prompt_delivery["episodes_with_legacy_launch_chain_corroboration"]
        ),
        "episodes_with_card_delivery_support": int(
            prompt_delivery["episodes_with_delivery_support"]
        ),
        "forbidden_private_prompt_hits": 0,
        "unexplained_task_key_mismatches": 0,
        "resume_transaction_audit_failures": 0,
        "critic_evidence_conflicts_audited": int(
            development.get("rats_critic_evidence_conflicts_audited") or 0
        ),
        "critic_conflicts_changed_algorithm_mid_chain": 0,
        "skill_admission_audit_failures": 0,
        "human_visual_spot_checks": int(
            development.get("rats_human_visual_spot_checks") or 0
        ),
        "human_visual_spot_checks_changed_control": 0,
        "actual_development_model": development["rats_actual_models_observed"],
        "method_audits_passed": int((method_audits.status == "pass").sum()),
        "frozen_rats_content_sha256": frozen_seal["content_sha256"],
        "frozen_rats_files": frozen_seal["files"],
        "rats_budget_boundary_status": (
            budget_boundary.get("status")
            if isinstance(budget_boundary, dict)
            else "not_applicable"
        ),
    }


def _artifact_index(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def _invalid_attempt_frame() -> tuple[pd.DataFrame, list[Path]]:
    """Index excluded attempts without allowing them into scored tables."""

    roots = (
        ROOT / "outputs" / "controlled_comparison" / "robosuite" / "invalid_attempts",
        ROOT / "outputs" / "evolution" / "invalid_attempts",
        ROOT / "outputs" / "evolution" / "robosuite_rats_rs_time3" / "invalidated_extractions",
    )
    markers = sorted(
        {
            path.resolve()
            for root in roots
            if root.is_dir()
            for name in ("INVALID_ATTEMPT.json", "NOT_SCORED.json")
            for path in root.rglob(name)
        }
    )
    rows: list[dict[str, Any]] = []
    for marker in markers:
        payload = _read_json(marker)
        rows.append(
            {
                "marker": str(marker.relative_to(ROOT)),
                "attempt_root": str(marker.parent.relative_to(ROOT)),
                "status": payload.get("status", "invalid_non_scoring"),
                "method": payload.get("method", ""),
                "reason": payload.get("reason", ""),
                "affected_episodes": json.dumps(
                    payload.get("affected_episodes") or [], ensure_ascii=False
                ),
                "registered_trials_started": payload.get("registered_trials_started"),
                "registered_trials_scored": payload.get("registered_trials_scored"),
                "candidate_promotions": payload.get(
                    "candidate_promotions_from_this_root"
                ),
                "formal_score_use": False,
                "marker_sha256": _sha256(marker),
            }
        )
    return pd.DataFrame(rows), markers


def _protocol_frame() -> pd.DataFrame:
    path = HERE / "protocol.yaml"
    return pd.DataFrame(
        {
            "line": list(range(1, len(path.read_text(encoding="utf-8").splitlines()) + 1)),
            "protocol_yaml": path.read_text(encoding="utf-8").splitlines(),
        }
    )


def _published_racap_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep published aggregates distinct from optional local episode evidence."""
    published = pd.read_csv(PUBLISHED_RESULTS)
    published = published.loc[
        (published["benchmark"] == "LIBERO-90")
        & published["method"].str.startswith("RACaP ")
    ].copy()
    published["evidence_tier"] = "published aggregate; raw episodes not bundled"
    published["primary_controlled_comparison"] = False
    columns = ["key", "native_success", "agent_success", "suite", "task_id", "seed"]
    episodes = pd.DataFrame(columns=columns)
    summary_path = PHASE2_FULL90 / "summary.json"
    records_path = PHASE2_FULL90 / "records.jsonl"
    if summary_path.is_file() != records_path.is_file():
        raise SystemExit("Phase 2 local evidence requires both summary.json and records.jsonl")
    if records_path.is_file():
        summary = _read_json(summary_path)
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        if len(records) != len({str(row.get("key")) for row in records}):
            raise SystemExit("Phase 2 local episode keys are duplicated")
        if int(summary.get("n_episodes", -1)) != len(records) or int(
            summary.get("native_success", -1)
        ) != sum(bool(row.get("native_success")) for row in records):
            raise SystemExit("Phase 2 local summary and episode records disagree")
        episodes = pd.DataFrame(records)
    claims = published.rename(columns={"native_success": "success"}).copy()
    claims["scope"] = "LIBERO-90, seed 0"
    claims["rate"] = claims["success"] / claims["episodes"]
    claims["source"] = str(PUBLISHED_RESULTS.relative_to(ROOT))
    return published, episodes, claims


def _audit_run_provenance(frame: pd.DataFrame) -> None:
    """Fail closed if a final method row lacks its registered runtime evidence."""

    if set(frame.get("method", pd.Series(dtype=str)).astype(str)) != set(METHODS):
        raise SystemExit("run provenance does not contain all five methods")
    for boolean_field in ("manifest_present", "lifecycle_present"):
        values = frame[boolean_field]
        if values.dtype != bool:
            values = values.astype(str).str.lower().isin({"true", "1"})
        if not bool(values.all()):
            missing = frame.loc[~values, "method"].astype(str).tolist()
            raise SystemExit(f"missing {boolean_field} for methods: {missing}")
    expected_jobs = {
        "capx": 350,
        "rats_base": 350,
        "rats_90": 350,
        "racap_phase1": 350,
        "racap_phase2": 180,
    }
    registered_jobs = pd.to_numeric(frame["registered_jobs"], errors="coerce")
    expected_registered_jobs = frame["method"].map(expected_jobs)
    checks = {
        "model": (frame["model"].astype(str) == "gpt-5.5"),
        "workers": (pd.to_numeric(frame["workers"], errors="coerce") == 10),
        "registered_jobs": (registered_jobs == expected_registered_jobs),
        "initial_resets_per_episode": (
            pd.to_numeric(frame["initial_resets_per_episode"], errors="coerce") == 1
        ),
        "post_action_resets": (
            pd.to_numeric(frame["post_action_resets"], errors="coerce") == 0
        ),
        "service_ready_at_unix": (
            pd.to_numeric(frame["service_ready_at_unix"], errors="coerce").notna()
        ),
    }
    for field, valid in checks.items():
        if not bool(valid.all()):
            methods = frame.loc[~valid, "method"].astype(str).tolist()
            raise SystemExit(f"invalid {field} in run provenance for methods: {methods}")


def _runtime_provenance_paths() -> list[Path]:
    paths = [
        ROOT / "outputs" / "controlled_comparison" / "provenance" / "preflight.json"
    ]
    for method in METHODS:
        method_root = MEASURED / method
        paths.extend(
            [
                method_root / "run_manifest.json",
                method_root / "shared_api_services" / "lifecycle.json",
            ]
        )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS / "final",
    )
    args = parser.parse_args()
    missing = [str(path) for _, path in TABLES if not path.is_file()]
    if missing:
        raise SystemExit("missing required analysis tables:\n" + "\n".join(missing))

    tables = {name: pd.read_csv(path) for name, path in TABLES}
    published_racap, phase2_full90_episodes, claim_evidence = (
        _published_racap_frames()
    )
    invalid_attempts, invalid_attempt_paths = _invalid_attempt_frame()
    current_phase1_l90 = tables["Main4 Episodes"].loc[
        (tables["Main4 Episodes"]["method"].astype(str) == "racap_phase1")
        & (tables["Main4 Episodes"]["suite"].astype(str) == "libero_90")
    ]
    if len(current_phase1_l90) != 90:
        raise SystemExit(
            "Phase 1 LIBERO-90 evidence does not cover all 90 episodes"
        )
    method_audits, method_audit_paths = _method_audit_frame()
    audit = _audit(tables, method_audits)
    protocol = _protocol_frame()
    preflight_provenance, run_provenance = _provenance_tables(
        MEASURED, list(METHODS)
    )
    _audit_run_provenance(run_provenance)
    source_paths = [path for _, path in TABLES] + method_audit_paths + invalid_attempt_paths + [
        HERE / "protocol.yaml",
        HERE / "task_manifest.json",
        MAIN_FOUR / "summary.json",
        PRO_FIVE / "summary.json",
        ANALYSIS / "one_shot" / "adaptation_audit.json",
        ANALYSIS / "one_shot" / "test_card_delivery_audit.json",
        ANALYSIS / "one_shot" / "test_card_prompt_audit.json",
        ANALYSIS / "development" / "summary.json",
        ANALYSIS / "robosuite_transfer" / "audit.json",
        ANALYSIS / "robosuite_development" / "audit.json",
        LONG_FINAL / "frozen_artifact_audit.json",
        FROZEN_RATS / "artifact_manifest.json",
        FROZEN_RATS / "seal_manifest.json",
        PUBLISHED_RESULTS,
        PUBLISHED_PROVENANCE,
        PHASE2_MANIFEST,
        ROOT
        / "outputs"
        / "controlled_comparison"
        / "measured"
        / "rats_90"
        / "frozen_artifact_audit.json",
    ] + [path for path in (PHASE2_FULL90 / "summary.json", PHASE2_FULL90 / "records.jsonl") if path.is_file()] + _runtime_provenance_paths()
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise SystemExit("missing report provenance artifacts:\n" + "\n".join(missing_sources))
    artifacts = _artifact_index(source_paths)
    frozen_files = pd.DataFrame(
        _read_json(FROZEN_RATS / "seal_manifest.json").get("files") or []
    )
    overview = pd.DataFrame(
        [
            ("report", "RACaP controlled comparison"),
            ("protocol", "racap_controlled"),
            ("audit status", audit["status"]),
            ("success authority", "native simulator predicates; evaluator-only"),
            ("development pool", "LIBERO-90"),
            ("zero-shot target", "six LIBERO-PRO perturbation suites"),
            ("runtime model", "GPT-5.5 via VAPI, temperature 0"),
            (
                "workers",
                "ten after the registered parallelism amendment; methods executed sequentially",
            ),
            (
                "four-method complete main episodes",
                audit["main_four_method_registered_episodes"],
            ),
            (
                "five-method matched PRO episodes",
                audit["pro_five_method_registered_episodes"],
            ),
            ("one-trial paired episodes", audit["one_trial_paired_episodes"]),
            ("long-horizon checkpoint rows", audit["long_checkpoint_rows"]),
            (
                "Robosuite five-method transfer episodes",
                audit["robosuite_transfer_registered_episodes"],
            ),
            (
                "Robosuite matched development episodes",
                audit["robosuite_development_episodes"],
            ),
            (
                "RACaP result tiers",
                "current Phase 1 42/90; retained Phase 1 48/90; archived Phase 2 49/90",
            ),
            (
                "statistics",
                "Wilson 95%; exact McNemar + Holm; paired suite-task bootstrap (10,000)",
            ),
            (
                "cost",
                "frozen list-price estimate from request token telemetry; not a VAPI invoice",
            ),
        ],
        columns=["item", "value"],
    )
    audit_frame = pd.DataFrame(
        [{"check": key, "value": value} for key, value in audit.items()]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workbook = args.output_dir / "final_report.xlsx"
    with pd.ExcelWriter(workbook, engine="xlsxwriter") as writer:
        overview.to_excel(writer, sheet_name="Overview", index=False)
        audit_frame.to_excel(writer, sheet_name="Final Audit", index=False)
        method_audits.to_excel(writer, sheet_name="Method Audits", index=False)
        artifacts.to_excel(writer, sheet_name="Artifact Index", index=False)
        protocol.to_excel(writer, sheet_name="Protocol", index=False)
        frozen_files.to_excel(writer, sheet_name="Frozen RATS Files", index=False)
        preflight_provenance.to_excel(
            writer, sheet_name="Preflight Provenance", index=False
        )
        run_provenance.to_excel(writer, sheet_name="Run Provenance", index=False)
        published_racap.to_excel(writer, sheet_name="RACaP Results", index=False)
        phase2_full90_episodes.to_excel(
            writer, sheet_name="Champion L90 Episodes", index=False
        )
        claim_evidence.to_excel(
            writer, sheet_name="RACaP Evidence Tiers", index=False
        )
        invalid_attempts.to_excel(
            writer, sheet_name="Invalid Attempt Index", index=False
        )
        _format_excel_sheet(writer, "Overview", overview)
        _format_excel_sheet(writer, "Final Audit", audit_frame)
        _format_excel_sheet(writer, "Method Audits", method_audits)
        _format_excel_sheet(writer, "Artifact Index", artifacts)
        _format_excel_sheet(writer, "Protocol", protocol)
        _format_excel_sheet(writer, "Frozen RATS Files", frozen_files)
        _format_excel_sheet(writer, "Preflight Provenance", preflight_provenance)
        _format_excel_sheet(writer, "Run Provenance", run_provenance)
        _format_excel_sheet(writer, "RACaP Results", published_racap)
        _format_excel_sheet(
            writer, "Champion L90 Episodes", phase2_full90_episodes
        )
        _format_excel_sheet(writer, "RACaP Evidence Tiers", claim_evidence)
        _format_excel_sheet(writer, "Invalid Attempt Index", invalid_attempts)
        for name, frame in tables.items():
            if len(name) > 31:
                raise SystemExit(f"Excel sheet name is too long: {name}")
            frame.to_excel(writer, sheet_name=name, index=False)
            _format_excel_sheet(writer, name, frame)

    result = {
        **audit,
        "generated_at_unix": time.time(),
        "workbook": str(workbook.resolve()),
        "workbook_bytes": workbook.stat().st_size,
        "workbook_sha256": _sha256(workbook),
        "source_artifacts": len(artifacts),
        "invalid_attempts_retained": len(invalid_attempts),
        "sheets": 12 + len(tables),
    }
    (args.output_dir / "final_report_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
