from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import experiments.controlled_comparison.supervise_after_development as development_supervisor
import experiments.controlled_comparison.build_one_shot_cards as one_shot_cards
import experiments.controlled_comparison.run_one_shot as one_shot_runner
from experiments.controlled_comparison.audit_provider_latency import (
    audit_latencies,
    latencies_from_run_dir,
)
from experiments.controlled_comparison.adjudicate_provider_probes import adjudicate
from experiments.controlled_comparison.probe_provider_latency import _probe_payload
import experiments.controlled_comparison.run_libero as run_libero
from experiments.controlled_comparison.run_all import (
    MAIN_EXECUTION_METHODS,
    ROOT as ORCHESTRATOR_ROOT,
    _immutable_method_result,
    _orchestration_env,
    _phase_completed,
    _rats_frozen_args,
    _resume_fingerprint,
)

from experiments.controlled_comparison.analyze_one_shot import (
    _adaptation_artifact_audit,
    _adaptation_budget,
    _aggregate as _one_shot_aggregate,
    _racap_prompt_card_index,
    _select_calibration,
    _test_card_delivery_audit,
    _test_card_prompt_audit,
)
from experiments.controlled_comparison.run_libero import (
    _acquire_method_run_lock,
    _apply_registered_manifest_migration,
    _archive_incomplete_generated_resume_state,
    _capx_seed_binding,
    _controlled_pythonpath,
    _controlled_rats_libero_env,
    _episode_key_set_sha256,
    _experiment_import_roots,
    _frozen_rats_state,
    _generated_episode_evidence,
    _json_payload_sha256,
    _jobs,
    _materialize_rats_hard_wall_timeout_artifacts,
    _port_ready,
    _public_model_route,
    _racap_group_evidence,
    _remap_api_server_ports,
    _require_frozen_rats_seal,
    _rats_eval_budget_args,
    _rats_seed_binding,
    _run_generated_method,
    _run_logged,
    _service_client_env,
    _validate_import_roots,
)
from experiments.controlled_comparison.run_long_horizon import (
    _common_env as _long_common_env,
    _materialize_isolated_rats_memory,
)
from experiments.controlled_comparison.run_rats_development import (
    _archive_previous_abort,
    _read_log_since,
    _restore_latest_completed_snapshot,
)
from experiments.controlled_comparison.build_one_shot_cards import (
    _bounded_card,
    _episode_root,
    _public_trace,
)
from experiments.controlled_comparison.analyze_results import (
    _artifact_wall_seconds,
    _capx_source_events,
    _event_count,
    _format_excel_sheet,
    _generated_wall_timeout_reached,
    _manifest_index,
    _paired_statistics,
    _parse_capx,
    _pooled_model_call_latencies,
    _provenance_tables,
    _request_list_price_usd,
    _rats_source_events,
    _telemetry_by_key,
    _total_simulator_steps,
    _usage,
    _verified_evidence,
)
from experiments.controlled_comparison.assemble_report import (
    _audit_run_provenance,
    _published_racap_frames,
)
from racap.benchmark.tasks import build_task_episodes
from racap.envs.register_suites import register_missing_suites


def test_task_perturbation_uses_bddl_language_but_swap_keeps_metadata() -> None:
    pytest.importorskip("libero.benchmark", reason="requires the external LIBERO simulator and assets")
    register_missing_suites()

    task_episode = build_task_episodes(
        "libero_spatial_task", task_ids=(2,), seeds=(0,)
    )[0]
    assert task_episode.instruction == (
        "Pick the akita black bowl next to the plate and place it on the plate"
    )
    assert task_episode.instruction_source == "bddl_task_perturbation_language"

    swap_episode = build_task_episodes(
        "libero_spatial_swap", task_ids=(2,), seeds=(0,)
    )[0]
    assert swap_episode.instruction == (
        "pick up the black bowl from table center and place it on the plate"
    )
    assert swap_episode.instruction_source == "benchmark_task_language"


@pytest.mark.parametrize(
    "script",
    ["run_one_shot.py", "run_robosuite.py"],
)
def test_transfer_runners_support_direct_script_invocation(script: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "experiments" / "controlled_comparison" / script),
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_one_shot_orchestrator_exposes_repo_to_child_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/import/root")
    roots = one_shot_runner._child_env()["PYTHONPATH"].split(os.pathsep)
    assert roots[0] == str(one_shot_runner.ROOT)
    assert "/existing/import/root" in roots


def test_generated_jobs_retain_instruction_authority() -> None:
    manifest = {
        "rows": [
            {
                "cohort": "pro",
                "suite": "libero_goal_task",
                "task_id": 0,
                "seed": 2,
                "instruction": "open the bottom drawer of the cabinet",
                "instruction_source": "bddl_task_perturbation_language",
            }
        ]
    }
    assert _jobs(manifest, {"pro"}) == [
        {
            "cohort": "pro",
            "suite": "libero_goal_task",
            "task_id": 0,
            "seed": 2,
            "instruction": "open the bottom drawer of the cabinet",
            "instruction_source": "bddl_task_perturbation_language",
        }
    ]


def test_analysis_manifest_index_preserves_cross_cohort_episode_keys() -> None:
    manifest, rows = _manifest_index()

    assert len(manifest) == len(rows)
    zero_shot = manifest[
        ("libero_pro_zero_shot", "libero_spatial_swap", 0, 1)
    ]
    one_trial = manifest[
        ("libero_pro_one_shot", "libero_spatial_swap", 0, 1)
    ]
    assert zero_shot["cohort"] == "libero_pro_zero_shot"
    assert one_trial["cohort"] == "libero_pro_one_shot"
    assert zero_shot is not one_trial


def test_capx_parser_retains_outer_artifact_wall_time(tmp_path: Path) -> None:
    task_root = (
        tmp_path
        / "libero90_id_replay"
        / "libero_90"
        / "task_00"
        / "seed_00"
    )
    task_root.mkdir(parents=True)
    (task_root / "COMPLETE.json").write_text(
        json.dumps({"seconds": 12.75}), encoding="utf-8"
    )
    (task_root / "EVIDENCE.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    (task_root / "sim_episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_key": "libero_90/0/seed0",
                "time": 10.0,
                "task_prompt": "close the top drawer of the cabinet",
                "init_state_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (task_root / "native_states.jsonl").write_text(
        json.dumps(
            {
                "episode_key": "libero_90/0/seed0",
                "time": 13.0,
                "event": "native_state_after_code",
                "native_success": False,
                "simulator_steps": 20,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (task_root / "llm_calls.jsonl").write_text("", encoding="utf-8")

    manifest, _ = _manifest_index()
    rows = _parse_capx(tmp_path, manifest)

    assert _artifact_wall_seconds(task_root) == 12.75
    assert len(rows) == 1
    assert rows[0]["policy_wall_seconds"] == 3.0
    assert rows[0]["artifact_wall_seconds"] == 12.75


def test_usage_retains_pooled_hosted_call_latency_samples() -> None:
    first = _usage(
        [
            {"elapsed_s": 2.0, "usage": {}},
            {"elapsed_seconds": 6.0, "usage": {}},
        ]
    )
    second = _usage([{"elapsed_s": 10.0, "usage": {}}])
    group = pd.DataFrame([first, second])

    pooled = _pooled_model_call_latencies(group)

    assert pooled.tolist() == [2.0, 6.0, 10.0]
    assert first["model_latency_seconds"] == 8.0
    assert first["median_model_call_latency_seconds"] == 4.0
    assert first["p90_model_call_latency_seconds"] == pytest.approx(5.6)


def test_excel_formatting_does_not_depend_on_plot_globals(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [{"native_success": True, "model_calls": 2, "policy_wall_seconds": 1.5}]
    )
    with pd.ExcelWriter(tmp_path / "audit.xlsx", engine="xlsxwriter") as writer:
        frame.to_excel(writer, sheet_name="Episodes", index=False)
        _format_excel_sheet(writer, "Episodes", frame)


def test_orchestrator_prepends_repo_root_to_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    entries = _orchestration_env()["PYTHONPATH"].split(os.pathsep)
    assert entries == [str(ORCHESTRATOR_ROOT), "/existing/path"]


def test_workbook_provenance_separates_preflight_probe_from_runtime_services(
    tmp_path: Path,
) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "captured_at_unix": 10.0,
                "host": {"hostname": "worker", "python_version": "3.11"},
                "repositories": {"racap": {"commit": "abc", "status_porcelain": ""}},
                "model_configuration": {
                    "requested_model": "gpt-5.5",
                    "base_url": "http://127.0.0.1:8110",
                    "maximum_concurrency": 2,
                    "credential_value_recorded": False,
                    "credential": "must-not-enter-workbook",
                },
                "services": [
                    {"host": "127.0.0.1", "port": 8214, "reachable": False}
                ],
            }
        ),
        encoding="utf-8",
    )
    method_root = tmp_path / "measured" / "capx"
    lifecycle_root = method_root / "shared_api_services"
    lifecycle_root.mkdir(parents=True)
    (method_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": "protocol",
                "task_manifest_sha256": "tasks",
                "model": "gpt-5.5",
                "workers": 10,
                "jobs": [{"task_id": 0}],
                "cohorts": ["libero90_id_replay"],
                "simulator_horizon": 8000,
                "registered_initial_states_per_episode": 1,
                "post_action_environment_resets": 0,
                "generated_continuous_turns": 10,
                "shared_service_urls": {"SAM3_SERVICE_URL": "http://127.0.0.1:8214"},
            }
        ),
        encoding="utf-8",
    )
    (lifecycle_root / "lifecycle.json").write_text(
        json.dumps(
            {
                "owner": "run_libero_method_parent",
                "pid": 123,
                "ports": [8214, 8215, 8216],
                "started_at_unix": 20.0,
                "ready_at_unix": 21.0,
                "stopped_at_unix": None,
                "returncode": None,
                "model_assets": {"sam3": {"sha256": "sam-hash"}},
            }
        ),
        encoding="utf-8",
    )

    summary, runs = _provenance_tables(
        tmp_path / "measured", ["capx", "rats_base"], preflight_path=preflight
    )

    summary_values = dict(zip(summary.item, summary.value))
    assert summary_values["capture_phase"] == "before method-local perception-service launch"
    assert "not-yet-started" in summary_values["preflight_service_probe"]
    assert "must-not-enter-workbook" not in summary.to_json()
    capx = runs[runs.method == "capx"].iloc[0]
    assert bool(capx.manifest_present)
    assert bool(capx.lifecycle_present)
    assert capx.service_ready_at_unix == 21.0
    assert capx.service_ports == "8214, 8215, 8216"
    assert capx.sam3_sha256 == "sam-hash"
    missing = runs[runs.method == "rats_base"].iloc[0]
    assert not bool(missing.manifest_present)
    assert not bool(missing.lifecycle_present)


def test_final_report_fails_closed_on_runtime_provenance() -> None:
    rows = []
    for method in ("capx", "rats_base", "rats_90", "racap_phase1", "racap_phase2"):
        rows.append(
            {
                "method": method,
                "manifest_present": True,
                "lifecycle_present": True,
                "model": "gpt-5.5",
                "workers": 10,
                "registered_jobs": 180 if method == "racap_phase2" else 350,
                "initial_resets_per_episode": 1,
                "post_action_resets": 0,
                "service_ready_at_unix": 21.0,
            }
        )
    valid = pd.DataFrame(rows)
    _audit_run_provenance(valid)

    wrong_model = valid.copy()
    wrong_model.loc[wrong_model.method == "rats_90", "model"] = "other"
    with pytest.raises(SystemExit, match="invalid model.*rats_90"):
        _audit_run_provenance(wrong_model)

    missing_lifecycle = valid.copy()
    missing_lifecycle.loc[
        missing_lifecycle.method == "racap_phase2", "lifecycle_present"
    ] = False
    with pytest.raises(SystemExit, match="missing lifecycle_present.*racap_phase2"):
        _audit_run_provenance(missing_lifecycle)

    imputed_champion = valid.copy()
    imputed_champion.loc[
        imputed_champion.method == "racap_phase2", "registered_jobs"
    ] = 350
    with pytest.raises(SystemExit, match="invalid registered_jobs.*racap_phase2"):
        _audit_run_provenance(imputed_champion)


def test_public_model_route_is_stable_and_credential_free() -> None:
    assert _public_model_route("http://127.0.0.1:8110/") == "http://127.0.0.1:8110"
    assert (
        _public_model_route(
            "HTTPS://alice:secret@example.com:443/v1/?token=private#fragment"
        )
        == "https://example.com:443/v1"
    )


@pytest.mark.parametrize("endpoint", ["", "localhost:8110", "ftp://example.com/v1"])
def test_public_model_route_rejects_ambiguous_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError):
        _public_model_route(endpoint)


def test_controlled_pythonpath_is_absolute_and_checkout_stable() -> None:
    entries = _controlled_pythonpath().split(os.pathsep)

    assert entries
    assert all(Path(entry).is_absolute() for entry in entries)
    assert "." not in entries
    assert str(Path(__file__).resolve().parents[1]) in entries


def test_rats_evaluation_restricts_optional_registry_to_libero() -> None:
    assert _controlled_rats_libero_env() == {"CAPX_ENV_STACK": "libero"}


def test_import_root_audit_rejects_mixed_libero_and_robosuite_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    common = tmp_path / "registered" / "third_party"
    libero = common / "LIBERO-PRO" / "libero" / "libero" / "__init__.py"
    robosuite = (
        common
        / "libero_dependencies"
        / "robosuite"
        / "robosuite"
        / "__init__.py"
    )
    for path in (libero, robosuite):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    for resource in ("assets", "bddl_files", "init_files"):
        (libero.parent / resource).mkdir()
    monkeypatch.setattr(
        run_libero, "CONTROLLED_LIBERO_CONFIG_DIR", tmp_path / "libero_config"
    )
    module_files = {
        "capx": str(
            repository
            / "third_party"
            / "rats"
            / "capx-baseline"
            / "capx"
            / "__init__.py"
        ),
        "rats": str(repository / "third_party" / "rats" / "rats" / "__init__.py"),
        "libero": str(libero),
        "robosuite": str(robosuite),
    }

    audit = _validate_import_roots(module_files)

    assert audit["simulator_third_party_root"] == str(common.resolve())
    assert audit["module_files"]["capx"].startswith(str(repository))
    assert all(len(value) == 64 for value in audit["module_sha256"].values())
    assert all(len(value) == 64 for value in audit["source_tree_sha256"].values())
    assert audit["libero_runtime"]["config_dir"] == str(
        (tmp_path / "libero_config").resolve()
    )
    assert set(audit["libero_runtime"]["resource_tree_sha256"]) == {
        "assets",
        "bddl_files",
        "init_states",
    }

    other = tmp_path / "unrelated" / "libero_dependencies" / "robosuite" / "robosuite" / "__init__.py"
    other.parent.mkdir(parents=True)
    other.write_text("# mixed fixture\n", encoding="utf-8")
    module_files["robosuite"] = str(other)
    with pytest.raises(RuntimeError, match="different third_party roots"):
        _validate_import_roots(module_files)
from experiments.controlled_comparison.audit_method_results import (
    _frozen_artifact_audit,
    audit_method_frame,
)
from experiments.controlled_comparison.audit_environment import (
    _protocol_exact_response_cache,
)
from experiments.controlled_comparison.analyze_development import (
    _call_category,
    _committed_iteration_windows,
    _critic_evidence_audits,
    _development_progress,
    _human_visual_spot_checks,
    _identity_search_runs,
    _in_progress_attempt_diagnostics,
    _iteration_windows,
    _native_event_audit,
    _native_authority_mismatches,
    _prompt_privilege_audit,
    _resume_transaction_audit,
    _skill_admission_audit,
    _rats_attempt_diagnostics,
    _rats_invalidated_usage,
    _rats_resource_accounting,
    _rats_runtime_self_checks,
    _rats_observed_usage,
    _successful_code_metadata,
)
from experiments.controlled_comparison.analyze_long_horizon import (
    _aggregate as _long_aggregate,
    _metrics_from_trace,
    _pairwise as _long_pairwise,
)
from experiments.controlled_comparison.analyze_robosuite import (
    _is_physical_api_boundary,
    _model_and_quota_audit,
    _paired as _robosuite_paired,
    _resolved_config_audit,
)
from experiments.controlled_comparison.analyze_robosuite_development import (
    _audit as _robosuite_development_audit,
)
from experiments.controlled_comparison.analyze_robosuite_transfer import (
    _aggregate as _robosuite_transfer_aggregate,
    _generated_protocol_audit as _robosuite_generated_protocol_audit,
    _normalise_episode_attestations as _robosuite_normalise_episode_attestations,
    _paired as _robosuite_transfer_paired,
)
from experiments.controlled_comparison.run_robosuite import (
    SINGLE_RESET_SHIM_ROOT,
    TRANSPORT_RETRY_SHIM_ROOT,
    TASKS as ROBOSUITE_TASKS,
    _archive_incomplete_task,
    _configure_robosuite_eval,
    _robosuite_process_env,
    _robosuite_task_evidence,
    _validate_registered_transport_retry,
    _validate_single_reset_runner,
    _validate_robosuite_source,
    _validate_worker_protocol,
)
from experiments.controlled_comparison.run_robosuite_rats_evolution import (
    _archive_incomplete_candidate,
)
from experiments.controlled_comparison.run_robosuite_campaign import (
    _archive_invalid_racap_zero_shot,
    _corrected_racap_complete,
    _phase_specs,
)
from scripts.eval_robosuite_agent import (
    _archive_invalid_episode_attempts as _archive_invalid_racap_rs_episodes,
    _completed as _completed_racap_rs_episodes,
    _terminal_model_failure,
)
from scripts.extract_robosuite_rats_skills import _install_registered_transport_retry
from experiments.controlled_comparison.supervise_after_development import _pid_alive
from experiments.controlled_comparison.supervise_after_development import (
    _seal_frozen_artifact,
    _verify_frozen_artifact_seal,
)
from experiments.controlled_comparison.assemble_report import _require_complete


def test_environment_audit_uses_registered_exact_cache_policy_when_env_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "runtime:\n  exact_response_cache: false\n", encoding="utf-8"
    )
    monkeypatch.delenv("RACAP_LLM_CACHE", raising=False)
    assert _protocol_exact_response_cache(protocol) is False


def test_development_progress_separates_committed_from_active_round(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"rounds": 32}), encoding="utf-8"
    )
    for iteration in range(1, 6):
        (tmp_path / f"iteration_{iteration:03d}.json").write_text(
            "{}", encoding="utf-8"
        )
    (tmp_path / "iteration_006").mkdir()

    assert _development_progress(tmp_path) == {
        "committed_rounds": 5,
        "latest_committed_round": 5,
        "active_round": 6,
        "target_rounds": 32,
        "status": "running",
    }


def test_development_progress_detects_planning_round_before_directory(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"rounds": 32}), encoding="utf-8"
    )
    for iteration in range(1, 7):
        (tmp_path / f"iteration_{iteration:03d}.json").write_text(
            "{}", encoding="utf-8"
        )
    (tmp_path / "rats.log").write_text(
        "\n".join(
            [
                "ITERATION 1/32",
                "Skipping completed iteration 001 from resume cache",
                "ITERATION 6/32",
                "Skipping completed iteration 006 from resume cache",
                "ITERATION 7/32",
                "Step 2: Planning",
            ]
        ),
        encoding="utf-8",
    )

    assert _development_progress(tmp_path) == {
        "committed_rounds": 6,
        "latest_committed_round": 6,
        "active_round": 7,
        "target_rounds": 32,
        "status": "running",
    }


def test_human_visual_audit_marks_invalidated_transaction_without_deleting_history(
    tmp_path: Path,
) -> None:
    reused = tmp_path / "iter012_attempt0_failed" / "combined.mp4"
    reused.parent.mkdir(parents=True)
    reused.write_bytes(b"new transaction video")
    os.utime(reused, (250.0, 250.0))
    old = {
        "schema_version": 1,
        "time": "1970-01-01T00:01:40+00:00",
        "iteration": 12,
        "attempt": 0,
        "artifact_paths": [str(reused)],
        "observation": "old transaction annotation",
        "control_effect": False,
    }
    current = {
        "schema_version": 1,
        "time": "1970-01-01T00:05:00+00:00",
        "iteration": 12,
        "attempt": 0,
        "artifact_paths": [str(reused)],
        "observation": "rerun annotation",
        "control_effect": False,
    }
    (tmp_path / "human_visual_spot_checks.jsonl").write_text(
        json.dumps(old) + "\n" + json.dumps(current) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "administrative_events.jsonl").write_text(
        json.dumps(
            {
                "time": 200.0,
                "event": "native_predicate_bug_invalidated_round12",
                "invalidated_iteration": 12,
                "partial_next_iteration": 13,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audits = _human_visual_spot_checks(tmp_path)

    assert len(audits) == 2
    assert audits.iloc[0].lineage_status == "invalidated_transaction"
    assert bool(audits.iloc[0].artifact_path_reused_after_invalidation)
    assert audits.iloc[1].lineage_status == "active"
    assert not bool(audits.iloc[1].artifact_path_reused_after_invalidation)


def test_environment_audit_allows_explicit_exact_cache_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "runtime:\n  exact_response_cache: false\n", encoding="utf-8"
    )
    monkeypatch.setenv("RACAP_LLM_CACHE", "1")
    assert _protocol_exact_response_cache(protocol) is True


def test_final_report_requires_exact_complete_registered_grid() -> None:
    complete = pd.DataFrame({"completed": [True, True]})
    _require_complete(complete, name="test grid", expected=2)

    with pytest.raises(SystemExit, match="cardinality mismatch"):
        _require_complete(complete, name="test grid", expected=3)
    with pytest.raises(SystemExit, match="incomplete"):
        _require_complete(
            pd.DataFrame({"completed": [True, False]}),
            name="test grid",
            expected=2,
        )


def test_final_report_indexes_all_development_audit_ledgers() -> None:
    from experiments.controlled_comparison.assemble_report import TABLES

    names = {name for name, _ in TABLES}
    assert {
        "Development Identity",
        "Development NativeEvents",
        "Development PromptAudit",
        "Development Critic",
        "Development VisualAudit",
        "Development SkillAdmission",
    }.issubset(names)


def test_final_report_forbids_human_visual_feedback_into_control() -> None:
    from experiments.controlled_comparison import assemble_report

    source = Path(assemble_report.__file__).read_text(encoding="utf-8")
    assert "rats_human_visual_spot_checks_changed_control" in source
    assert "human visual spot check changed the running RATS controller" in source


def test_final_report_records_frozen_artifact_audits() -> None:
    from experiments.controlled_comparison import assemble_report

    source = Path(assemble_report.__file__).read_text(encoding="utf-8")
    assert '"Frozen RATS Files"' in source
    assert '"frozen_artifact_audit.json"' in source


def test_final_report_requires_every_method_audit_to_pass(tmp_path: Path) -> None:
    from experiments.controlled_comparison import assemble_report

    original_analysis = assemble_report.ANALYSIS
    assemble_report.ANALYSIS = tmp_path
    try:
        for method in assemble_report.METHODS:
            path = (
                tmp_path
                / "method_audits"
                / assemble_report.METHOD_AUDIT_DIRS[method]
                / "method_audit.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "method": method,
                        "status": "pass",
                        "actual_models": ["gpt-5.5"],
                    }
                ),
                encoding="utf-8",
            )
        audits, paths = assemble_report._method_audit_frame()
        assert len(audits) == 5
        assert len(paths) == 5
        assert set(audits.status) == {"pass"}
    finally:
        assemble_report.ANALYSIS = original_analysis


def test_critic_evidence_audit_preserves_cross_view_adjudication(tmp_path: Path) -> None:
    record = {
        "schema_version": "critic_evidence_audit_v1",
        "iteration": 3,
        "attempt": 0,
        "task_id": "libero_90_task71",
        "conflict_type": "cross_view_perspective_false_displacement_inference",
        "human_visual_adjudication": "Temporal agent view shows no displacement.",
        "algorithm_changed_mid_chain": False,
    }
    (tmp_path / "critic_evidence_audits.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    audit = _critic_evidence_audits(tmp_path)

    assert len(audit) == 1
    assert audit.iloc[0].conflict_type == record["conflict_type"]
    assert audit.iloc[0].effective_status == "active_conflict"
    assert not bool(audit.iloc[0].algorithm_changed_mid_chain)
    assert _critic_evidence_audits(tmp_path / "missing").empty


def test_critic_evidence_correction_preserves_and_retracts_human_error(
    tmp_path: Path,
) -> None:
    records = [
        {
            "schema_version": "critic_evidence_audit_v1",
            "time": 10.0,
            "conflict_type": "initial_human_claim",
            "algorithm_changed_mid_chain": False,
        },
        {
            "schema_version": "critic_evidence_audit_v2",
            "record_type": "correction",
            "time": 20.0,
            "supersedes_time": 10.0,
            "conflict_type": "audit_correction_no_confirmed_critic_conflict",
            "algorithm_changed_mid_chain": False,
        },
    ]
    (tmp_path / "critic_evidence_audits.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    audit = _critic_evidence_audits(tmp_path)

    assert list(audit.effective_status) == ["retracted", "correction"]
    assert list(audit.record_type) == ["conflict", "correction"]


def test_skill_admission_audit_requires_native_success(tmp_path: Path) -> None:
    (tmp_path / "skills.json").write_text(
        json.dumps([{"name": "learned_pick"}]), encoding="utf-8"
    )
    (tmp_path / "iteration_001.json").write_text(
        json.dumps(
            {
                "iteration": 1,
                "success": True,
                "task_proposal": {"activity_name": "task_ok"},
                "verification_attempt_2": {
                    "details": {"native_predicate_success": True}
                },
                "skills_learned": ["learned_pick"],
                "skills_added": ["learned_pick"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "iteration_002.json").write_text(
        json.dumps(
            {
                "iteration": 2,
                "success": True,
                "task_proposal": {"activity_name": "task_bad"},
                "verification_attempt_0": {
                    "details": {"native_predicate_success": False}
                },
                "skills_learned": ["learned_pick"],
                "skills_added": ["learned_pick"],
            }
        ),
        encoding="utf-8",
    )

    audit = _skill_admission_audit(tmp_path)

    assert list(audit.audit_pass) == [True, False]
    assert audit.iloc[0].native_success_attempts == "2"
    assert not bool(audit.iloc[1].native_success_required_for_admission)


def test_skill_admission_audit_allows_labeled_failure_proposal(
    tmp_path: Path,
) -> None:
    skill = {
        "name": "safe_tcp_reader",
        "source_task": "proposed_from_failures",
        "proposed": True,
        "tier": "experimental",
        "learned_iteration": 1,
    }
    (tmp_path / "skills.json").write_text(
        json.dumps([skill]), encoding="utf-8"
    )
    snapshot = tmp_path / "snapshots" / "iter001"
    snapshot.mkdir(parents=True)
    (snapshot / "skills.json").write_text(
        json.dumps([skill]), encoding="utf-8"
    )
    (tmp_path / "iteration_001.json").write_text(
        json.dumps(
            {
                "iteration": 1,
                "success": False,
                "task_proposal": {"activity_name": "task_failed"},
                "verification_attempt_0": {
                    "details": {"native_predicate_success": False}
                },
                "skills_learned": [],
                "proposed_skills": [{"name": "safe_tcp_reader"}],
                "skills_added": ["safe_tcp_reader"],
            }
        ),
        encoding="utf-8",
    )

    row = _skill_admission_audit(tmp_path).iloc[0]

    assert bool(row.audit_pass)
    assert row.admission_rule == "failure_proposed_experimental"
    assert row.failure_proposed_experimental_skills == "safe_tcp_reader"
    assert bool(row.failure_proposal_rule_satisfied)


def test_skill_admission_audit_rejects_unlabeled_failure_write(
    tmp_path: Path,
) -> None:
    skill = {
        "name": "unsafe_helper",
        "tier": "experimental",
        "learned_iteration": 1,
    }
    (tmp_path / "skills.json").write_text(
        json.dumps([skill]), encoding="utf-8"
    )
    (tmp_path / "iteration_001.json").write_text(
        json.dumps(
            {
                "iteration": 1,
                "success": False,
                "verification_attempt_0": {
                    "details": {"native_predicate_success": False}
                },
                "skills_added": ["unsafe_helper"],
            }
        ),
        encoding="utf-8",
    )

    row = _skill_admission_audit(tmp_path).iloc[0]

    assert not bool(row.audit_pass)
    assert row.admission_rule == "unclassified"
    assert row.unclassified_skills == "unsafe_helper"


def test_orchestrator_skips_only_identical_successful_phase() -> None:
    command = ["python", "run_libero.py", "--method", "capx"]
    fingerprint = _resume_fingerprint()
    history = [
        {
            "phase": "main_capx", "argv": command, "returncode": 75,
            "resume_fingerprint": fingerprint,
        },
        {
            "phase": "main_capx", "argv": command, "returncode": 0,
            "resume_fingerprint": fingerprint,
        },
    ]

    assert _phase_completed(history, name="main_capx", argv=command)
    assert not _phase_completed(
        history,
        name="main_capx",
        argv=command + ["--changed"],
    )
    assert not _phase_completed(history, name="main_rats_base", argv=command)
    assert not _phase_completed(
        history + [{
            "phase": "main_capx", "argv": command, "returncode": 1,
            "resume_fingerprint": fingerprint,
        }],
        name="main_capx",
        argv=command,
    )
    assert not _phase_completed(history, name="preflight", argv=command)
    stale = [
        {
            "phase": "main_capx", "argv": command, "returncode": 0,
            "resume_fingerprint": {**fingerprint, "git_commit": "stale"},
        }
    ]
    assert not _phase_completed(stale, name="main_capx", argv=command)


def test_orchestrator_never_schedules_racap_libero90_main_rerun() -> None:
    assert MAIN_EXECUTION_METHODS == ["capx", "rats_base", "rats_90"]
    assert "racap_phase1" not in MAIN_EXECUTION_METHODS
    assert "racap_phase2" not in MAIN_EXECUTION_METHODS


def test_orchestrator_reuses_only_complete_audited_method_outputs(
    tmp_path: Path,
) -> None:
    method_root = tmp_path / "measured" / "capx"
    audit_root = tmp_path / "analysis" / "method_audits" / "capx"
    method_root.mkdir(parents=True)
    audit_root.mkdir(parents=True)
    jobs = [{"suite": "suite", "task_id": task_id, "seed": 0} for task_id in range(2)]
    (method_root / "run_manifest.json").write_text(
        json.dumps({"method": "capx", "jobs": jobs}), encoding="utf-8"
    )
    (method_root / "COMPLETE.json").write_text(
        json.dumps({"method": "capx", "jobs": 2}), encoding="utf-8"
    )
    for task_id in range(2):
        episode = method_root / f"task_{task_id:02d}" / "seed_00"
        episode.mkdir(parents=True)
        (episode / "COMPLETE.json").write_text("{}", encoding="utf-8")
    (audit_root / "method_audit.json").write_text(
        json.dumps(
            {
                "method": "capx",
                "status": "pass",
                "episodes_parsed": 2,
                "episodes_expected": 2,
            }
        ),
        encoding="utf-8",
    )

    reused = _immutable_method_result("capx", output_root=tmp_path)

    assert reused is not None
    assert reused["episodes"] == 2
    (method_root / "task_01" / "seed_00" / "COMPLETE.json").unlink()
    assert _immutable_method_result("capx", output_root=tmp_path) is None


def test_frozen_artifact_seal_records_hashes_and_removes_write_bits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frozen"
    memory = root / "failure_memory"
    memory.mkdir(parents=True)
    (root / "skills.json").write_text("[]", encoding="utf-8")
    (memory / "episodes.jsonl").write_text("{}\n", encoding="utf-8")

    seal = _seal_frozen_artifact(root)

    assert seal["files"] == 2
    assert (root / "seal_manifest.json").is_file()
    assert root.stat().st_mode & 0o222 == 0
    assert (root / "skills.json").stat().st_mode & 0o222 == 0
    assert memory.stat().st_mode & 0o222 == 0
    assert _seal_frozen_artifact(root)["content_sha256"] == seal["content_sha256"]
    assert _verify_frozen_artifact_seal(root)["status"] == "pass"

    # Root can still mutate mode bits in this test environment. The verifier
    # must detect either content or permission drift without trusting chmod.
    (root / "skills.json").chmod(0o644)
    (root / "skills.json").write_text("[1]", encoding="utf-8")
    assert _verify_frozen_artifact_seal(root)["status"] == "error"


def test_frozen_rats_state_is_content_sensitive_and_timestamp_independent(
    tmp_path: Path,
) -> None:
    library = tmp_path / "skills.json"
    memory = tmp_path / "failure_memory"
    memory.mkdir()
    episode = memory / "episodes.jsonl"
    library.write_text("[]", encoding="utf-8")
    episode.write_text("{}\n", encoding="utf-8")

    before = _frozen_rats_state(library, memory)
    os.utime(episode, None)
    assert _frozen_rats_state(library, memory) == before

    episode.write_text('{"failure": true}\n', encoding="utf-8")
    after = _frozen_rats_state(library, memory)
    assert after["library_sha256"] == before["library_sha256"]
    assert after["memory_tree_sha256"] != before["memory_tree_sha256"]


def test_registered_rats_inputs_must_share_one_valid_seal(tmp_path: Path) -> None:
    root = tmp_path / "frozen"
    memory = root / "failure_memory"
    memory.mkdir(parents=True)
    library = root / "skills.json"
    library.write_text("[]", encoding="utf-8")
    (memory / "episodes.json").write_text("[]", encoding="utf-8")
    _seal_frozen_artifact(root)

    assert _require_frozen_rats_seal(library, memory)["status"] == "pass"
    with pytest.raises(SystemExit, match="same frozen artifact"):
        _require_frozen_rats_seal(library, tmp_path / "different_memory")


def test_registered_rats_artifact_arguments_match_each_protocol(tmp_path: Path) -> None:
    full = _rats_frozen_args(tmp_path, include_memory=True)
    skills_only = _rats_frozen_args(tmp_path, include_memory=False)

    assert full == [
        "--rats-library",
        str(tmp_path / "skills.json"),
        "--rats-memory",
        str(tmp_path / "failure_memory"),
    ]
    assert skills_only == ["--rats-library", str(tmp_path / "skills.json")]


def test_method_frozen_artifact_audit_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "frozen_artifact_audit.json"
    assert _frozen_artifact_audit(path)["status"] == "error"

    state = {"library_sha256": "abc", "memory_tree_sha256": "def"}
    path.write_text(
        json.dumps({"before": state, "after": state, "unchanged": True}),
        encoding="utf-8",
    )
    assert _frozen_artifact_audit(path)["status"] == "pass"

    path.write_text(
        json.dumps({"before": state, "after": {**state, "library_sha256": "bad"}, "unchanged": True}),
        encoding="utf-8",
    )
    assert _frozen_artifact_audit(path)["status"] == "error"


def test_manifest_rows_become_independent_episode_jobs() -> None:
    manifest = {
        "rows": [
            {
                "cohort": "zero_shot", "suite": "suite", "task_id": 2,
                "seed": 0, "instruction": "do task two",
            },
            {
                "cohort": "zero_shot", "suite": "suite", "task_id": 2,
                "seed": 1, "instruction": "do task two",
            },
            {
                "cohort": "other", "suite": "suite", "task_id": 3,
                "seed": 0, "instruction": "do task three",
            },
        ]
    }
    assert _jobs(manifest, {"zero_shot"}) == [
        {
            "cohort": "zero_shot", "suite": "suite", "task_id": 2, "seed": 0,
            "instruction": "do task two",
            "instruction_source": "benchmark_task_language",
        },
        {
            "cohort": "zero_shot", "suite": "suite", "task_id": 2, "seed": 1,
            "instruction": "do task two",
            "instruction_source": "benchmark_task_language",
        },
    ]


def _method_audit_frames(tmp_path: Path, *, success: bool = True):
    episode_rows = []
    complete_rows = []
    for index, cohort in enumerate(
        (
            "libero90_id_replay",
            "libero_pro_zero_shot",
            "libero_base_diagnostic",
            "libero_long",
        )
    ):
        key = f"suite/{index}/seed0"
        evidence = tmp_path / f"evidence_{index}.json"
        evidence.write_text('{"complete": true}', encoding="utf-8")
        episode_rows.append(
            {
                "method": "rats_90",
                "cohort": cohort,
                "episode_key": key,
                "instruction": f"complete task {index}",
                "delivered_instruction": f"complete task {index}",
                "seed": 0,
                "init_state_index": 0,
                "native_success": success,
                "evidence_path": str(evidence),
                "model_calls": 2,
                "simulator_steps": 100,
                "simulator_episode_resets": 1,
                "source_generation_events": 1,
                "source_repair_events": 0,
                "actual_models": "gpt-5.5",
            }
        )
        complete_rows.append(
            {"method": "rats_90", "episode_key": key, "completed": True}
        )
    return pd.DataFrame(episode_rows), pd.DataFrame(complete_rows)


def test_method_audit_passes_complete_nonranking_evidence(tmp_path: Path) -> None:
    episodes, completeness = _method_audit_frames(tmp_path)

    audit = audit_method_frame(
        episodes, completeness, method="rats_90", required_model="gpt-5.5"
    )

    assert audit["status"] == "pass"
    assert audit["ranking_assumption_enforced"] is False
    assert audit["actual_models"] == ["gpt-5.5"]


def test_method_audit_pauses_all_zero_in_domain_result(tmp_path: Path) -> None:
    episodes, completeness = _method_audit_frames(tmp_path, success=False)

    audit = audit_method_frame(episodes, completeness, method="rats_90")

    assert audit["status"] == "review_required"
    assert any("LIBERO-90" in reason for reason in audit["review_reasons"])


def test_method_audit_uses_loose_historical_floor_only_for_frozen_racap(
    tmp_path: Path,
) -> None:
    rows = []
    complete = []
    for index in range(90):
        evidence = tmp_path / f"evidence_{index}.json"
        evidence.write_text('{"complete": true}', encoding="utf-8")
        key = f"libero_90/{index}/seed0"
        rows.append(
            {
                "method": "racap_phase2",
                "cohort": "libero90_id_replay",
                "episode_key": key,
                "seed": 0,
                "init_state_index": 0,
                "native_success": index < 29,
                "evidence_path": str(evidence),
                "model_calls": 1,
                "simulator_steps": 100,
                "source_generation_events": 0,
                "source_repair_events": 0,
                "actual_models": "gpt-5.5",
            }
        )
        complete.append(
            {"method": "racap_phase2", "episode_key": key, "completed": True}
        )

    audit = audit_method_frame(
        pd.DataFrame(rows), pd.DataFrame(complete), method="racap_phase2"
    )

    assert audit["status"] == "review_required"
    assert audit["historical_id_review_floor"] == 30
    assert audit["ranking_assumption_enforced"] is False
    assert any("29/90 < 30/90" in reason for reason in audit["review_reasons"])


def test_method_audit_rejects_wrong_model_and_missing_episode(tmp_path: Path) -> None:
    episodes, completeness = _method_audit_frames(tmp_path)
    episodes.loc[0, "actual_models"] = "different-model"
    completeness.loc[0, "completed"] = False

    audit = audit_method_frame(episodes, completeness, method="rats_90")

    assert audit["status"] == "error"
    assert any("missing 1" in error for error in audit["errors"])
    assert any("different-model" in error for error in audit["errors"])


def test_method_audit_rejects_hidden_evaluation_resets(tmp_path: Path) -> None:
    episodes, completeness = _method_audit_frames(tmp_path)
    episodes.loc[0, "simulator_episode_resets"] = 2

    audit = audit_method_frame(episodes, completeness, method="rats_90")

    assert audit["status"] == "error"
    assert any("exactly one" in error for error in audit["errors"])


def test_method_audit_rejects_mismatched_applied_init_state(tmp_path: Path) -> None:
    episodes, completeness = _method_audit_frames(tmp_path)
    episodes.loc[0, "init_state_index"] = 49

    audit = audit_method_frame(episodes, completeness, method="rats_90")

    assert audit["status"] == "error"
    assert any("init-state indices" in error for error in audit["errors"])


def test_method_audit_rejects_wrong_delivered_instruction(tmp_path: Path) -> None:
    episodes, completeness = _method_audit_frames(tmp_path)
    episodes.loc[0, "delivered_instruction"] = "complete a different task"

    audit = audit_method_frame(episodes, completeness, method="rats_90")

    assert audit["status"] == "error"
    assert any("delivered task instructions" in error for error in audit["errors"])


def test_one_trial_budget_includes_calibration_and_visual_card_cost() -> None:
    calibration = pd.DataFrame(
        [
            {
                "method": "capx",
                "policy_wall_seconds": 20.0,
                "model_latency_seconds": 10.0,
                "model_calls": 2,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_tokens": 0,
                "estimated_api_cost_usd": 0.1,
                "simulator_steps": 200,
            }
        ]
    )
    cards = pd.DataFrame(
        [
            {
                "method": "capx",
                "model_latency_seconds": 5.0,
                "model_calls": 1,
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "cached_tokens": 0,
                "estimated_api_cost_usd": 0.05,
            }
        ]
    )

    budget = _adaptation_budget(calibration, cards).set_index("method").loc["capx"]

    assert budget["total_adaptation_policy_wall_seconds"] == 25.0
    assert budget["total_adaptation_model_calls"] == 3.0
    assert budget["total_adaptation_prompt_tokens"] == 150.0
    assert abs(budget["total_adaptation_estimated_api_cost_usd"] - 0.15) < 1e-12
    assert budget["total_adaptation_simulator_steps"] == 200.0


def test_one_trial_cards_checkpoint_and_resume_on_quota(
    tmp_path: Path, monkeypatch,
) -> None:
    suites = {
        "spatial": "libero_spatial_swap",
        "goal": "libero_goal_swap",
        "object": "libero_object_swap",
    }
    rows = []
    for method in one_shot_cards.METHODS:
        for family, suite in suites.items():
            artifact = tmp_path / "calibration" / method / family
            artifact.mkdir(parents=True)
            evidence = artifact / "EVIDENCE.json"
            evidence.write_text('{"complete": true}', encoding="utf-8")
            (artifact / "rollout.mp4").write_bytes(b"test-public-video")
            rows.append(
                {
                    "method": method,
                    "suite": suite,
                    "task_id": 0,
                    "seed": 0,
                    "episode_key": f"{suite}/0/seed0",
                    "instruction": f"calibrate {family}",
                    "native_success": False,
                    "artifact_path": str(artifact),
                    "evidence_path": str(evidence),
                }
            )
    monkeypatch.setattr(one_shot_cards, "_calibration_rows", lambda _: rows)
    monkeypatch.setenv("RACAP_VAPI_KEY", "test")
    monkeypatch.setenv("RACAP_VAPI_BASE", "https://example.invalid/v1")
    output = tmp_path / "cards"
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_one_shot_cards.py", "--input-root", str(tmp_path / "calibration"), "--output-dir", str(output)],
    )
    calls = {"count": 0}

    def quota_after_first(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise one_shot_cards.LLMQuotaError("insufficient balance")
        return "## Observed mechanism\nok\n## Transferable guidance\nok\n## Avoid\nnone\n## Uncertainty\nlow"

    monkeypatch.setattr(one_shot_cards, "ask", quota_after_first)
    assert one_shot_cards.main() == 75
    partial = json.loads((output / "manifest.json").read_text())
    assert partial["complete"] is False
    assert len(partial["cards"]) == 1

    resumed_calls = {"count": 0}

    def resumed(*args, **kwargs):
        resumed_calls["count"] += 1
        return "## Observed mechanism\nok\n## Transferable guidance\nok\n## Avoid\nnone\n## Uncertainty\nlow"

    monkeypatch.setattr(one_shot_cards, "ask", resumed)
    assert one_shot_cards.main() == 0
    final = json.loads((output / "manifest.json").read_text())
    assert final["complete"] is True
    assert len(final["cards"]) == 15
    assert resumed_calls["count"] == 14


def test_one_shot_zero_shot_calibration_selects_only_registered_identity() -> None:
    rows = []
    for method in one_shot_cards.METHODS:
        for suite in one_shot_cards.FAMILY_SUITE.values():
            for seed in (0, 1):
                rows.append(
                    {
                        "method": method,
                        "cohort": "libero_pro_zero_shot",
                        "suite": suite,
                        "task_id": 0,
                        "seed": seed,
                    }
                )
    selected = _select_calibration(
        pd.DataFrame(rows), "libero_pro_zero_shot"
    )
    assert len(selected) == 15
    assert set(selected["seed"]) == {0}
    assert set(selected["task_id"]) == {0}


def test_api_service_port_probe_distinguishes_live_and_free_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = int(server.getsockname()[1])
        assert _port_ready(port)
    assert not _port_ready(port)


def test_api_service_bundle_stops_the_complete_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    monkeypatch.setattr(run_libero, "API_SERVER_PORTS", (port,))
    parent_code = """
import subprocess
import sys
import time

subprocess.Popen(
    [sys.executable, "-m", "http.server", sys.argv[1], "--bind", "127.0.0.1"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(60)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(port)],
        start_new_session=True,
    )
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not _port_ready(port):
            time.sleep(0.05)
        assert _port_ready(port)

        run_libero._stop_service_bundle(process)

        assert not _port_ready(port)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait(timeout=5)


def test_generated_methods_use_comparison_owned_service_ports() -> None:
    assert _service_client_env() == {
        "SAM3_SERVICE_URL": "http://127.0.0.1:8214",
        "GRASPNET_SERVICE_URL": "http://127.0.0.1:8215",
        "PYROKI_SERVICE_URL": "http://127.0.0.1:8216",
    }
    config = {
        "api_servers": [
            {"_target_": "x.launch_sam3_server.main", "port": 8114},
            {"_target_": "x.launch_contact_graspnet_server.main", "port": 8115},
            {"_target_": "x.launch_pyroki_server.main", "port": 8116},
        ]
    }

    remapped = _remap_api_server_ports(config)

    assert [row["port"] for row in remapped["api_servers"]] == [8214, 8215, 8216]


def test_published_racap_results_do_not_fabricate_episode_evidence(monkeypatch, tmp_path) -> None:
    from experiments.controlled_comparison import assemble_report

    monkeypatch.setattr(assemble_report, "PHASE2_FULL90", tmp_path)
    published, episodes, claims = _published_racap_frames()
    assert set(published["method"]) == {
        "RACaP Phase 1 instrumented", "RACaP Phase 1 retained", "RACaP Phase 2"
    }
    assert episodes.empty
    assert not claims["primary_controlled_comparison"].any()
    (tmp_path / "summary.json").write_text('{"n_episodes": 1, "native_success": 1}')
    with pytest.raises(SystemExit, match="requires both"):
        _published_racap_frames()
    (tmp_path / "records.jsonl").write_text('{"key": "synthetic/0", "native_success": false}\n')
    with pytest.raises(SystemExit, match="disagree"):
        _published_racap_frames()


def test_rats_public_seed_is_separate_from_one_based_internal_trial() -> None:
    for public_seed in (0, 1, 2):
        binding = _rats_seed_binding(
            {"suite": "libero_spatial_swap", "task_id": 4, "seed": public_seed}
        )
        assert binding["RATS_SEED_OFFSET"] == str(public_seed)
        assert binding["RATS_REGISTERED_EPISODE_KEY"] == (
            f"libero_spatial_swap/4/seed{public_seed}"
        )
        # One evaluation iteration means internal seed = 1 + offset; the
        # LIBERO adapter then loads init-state index internal_seed - 1.
        internal_seed = 1 + int(binding["RATS_SEED_OFFSET"])
        assert internal_seed - 1 == public_seed


def test_capx_public_seed_is_separate_from_one_based_internal_trial() -> None:
    for public_seed in (0, 1, 2):
        binding = _capx_seed_binding(
            {"suite": "libero_spatial_swap", "task_id": 4, "seed": public_seed}
        )
        assert binding["CAPX_SEED_OFFSET"] == str(public_seed)
        assert binding["CAPX_REGISTERED_EPISODE_KEY"] == (
            f"libero_spatial_swap/4/seed{public_seed}"
        )
        internal_seed = 1 + int(binding["CAPX_SEED_OFFSET"])
        assert internal_seed - 1 == public_seed


def test_frozen_rats_evaluation_is_one_continuous_episode() -> None:
    args = _rats_eval_budget_args()

    def value(flag: str) -> str:
        return args[args.index(flag) + 1]

    assert value("--turns-per-attempt") == "10"
    assert value("--attempts-per-iteration") == "1"
    assert value("--policy-self-check-repairs") == "0"
    assert args.count("--multi-turn-decision") == 1
    with pytest.raises(ValueError, match="positive"):
        _rats_eval_budget_args(turns=0)


def _write_evidence(path: Path, text: str = "evidence") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_frozen_rats_artifact_binds_development_source_ledger(
    tmp_path: Path, monkeypatch,
) -> None:
    source = (
        tmp_path
        / "outputs"
        / "controlled_comparison"
        / "development"
        / "rats90_selfplay"
        / "launch_source_provenance.jsonl"
    )
    _write_evidence(source, '{"repository_commit":"abc"}\n')
    artifact = (
        tmp_path
        / "outputs"
        / "controlled_comparison"
        / "artifacts"
        / "rats90_frozen"
        / "skills.json"
    )
    _write_evidence(artifact, "[]")
    _write_evidence(artifact.parent / "artifact_manifest.json", '{"status":"complete"}')
    monkeypatch.setattr(development_supervisor, "ROOT", tmp_path)

    bound = development_supervisor._bind_development_provenance(artifact)

    copied = artifact.parent / "launch_source_provenance.jsonl"
    assert copied.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    manifest = json.loads((artifact.parent / "artifact_manifest.json").read_text())
    assert manifest["development_launch_provenance"] == str(copied.resolve())
    assert manifest["development_launch_provenance_sha256"] == bound["sha256"]


def test_budget_boundary_freezes_last_completed_round_snapshot(
    tmp_path: Path, monkeypatch,
) -> None:
    development = (
        tmp_path
        / "outputs"
        / "controlled_comparison"
        / "development"
        / "rats90_selfplay"
    )
    snapshot = development / "snapshots" / "iter001"
    (snapshot / "failure_memory").mkdir(parents=True)
    _write_evidence(development / "iteration_001.json", '{"iteration": 1}')
    _write_evidence(snapshot / "skills.json", '["committed"]')
    _write_evidence(snapshot / "failure_memory" / "episodes.json", '["committed"]')
    _write_evidence(development / "skills.json", '["partial-round"]')
    _write_evidence(
        development / "failure_memory" / "episodes.json", '["partial-round"]'
    )
    _write_evidence(
        development / "checkpoint.json",
        '{"termination":"simulator_reset_budget"}',
    )

    artifact = (
        tmp_path
        / "outputs"
        / "controlled_comparison"
        / "artifacts"
        / "rats90_frozen"
        / "skills.json"
    )
    _write_evidence(artifact, '["partial-round"]')
    _write_evidence(
        artifact.parent / "failure_memory" / "episodes.json", '["partial-round"]'
    )
    _write_evidence(artifact.parent / "artifact_manifest.json", '{}')
    monkeypatch.setattr(development_supervisor, "ROOT", tmp_path)

    transaction = development_supervisor._finalize_budget_boundary(artifact)

    assert transaction["status"] == "pass"
    assert transaction["restoration"]["restored"] is True
    assert artifact.read_text(encoding="utf-8") == '["committed"]'
    assert (
        artifact.parent / "failure_memory" / "episodes.json"
    ).read_text(encoding="utf-8") == '["committed"]'
    manifest = json.loads((artifact.parent / "artifact_manifest.json").read_text())
    assert manifest["budget_boundary_transaction"][
        "artifact_matches_last_completed_snapshot"
    ] is True
    assert (development / "resume_restorations.jsonl").is_file()


def test_prompt_privilege_audit_distinguishes_public_docs_from_native_payload() -> None:
    public = {
        "timestamp": 1.0,
        "caller": "task_proposer",
        "episode_key": "libero_90/0/seed1",
        "request": {
            "messages": [
                {
                    "content": "Use In (for open containers like baskets and trays)."
                }
            ]
        },
        "_path": "public.json",
    }
    leaked = {
        "timestamp": 2.0,
        "caller": "planner",
        "episode_key": "libero_90/0/seed1",
        "request": {
            "messages": [
                {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"native_predicates": [{"satisfied": false}]}',
                        }
                    ]
                }
            ]
        },
        "_path": "leaked.json",
    }

    audit = _prompt_privilege_audit([public, leaked])

    assert set(audit.forbidden_payload_pattern) == {
        "native_predicates_json",
        "predicate_satisfaction_json",
    }
    assert set(audit.artifact) == {"leaked.json"}


def test_capx_network_call_records_cost_telemetry_without_prompt_or_key(
    tmp_path: Path, monkeypatch,
) -> None:
    capx_root = Path(__file__).parents[1] / "third_party" / "rats" / "capx-baseline"
    monkeypatch.syspath_prepend(str(capx_root))
    from capx.llm import client as capx_client

    class FakeResponse:
        ok = True
        status_code = 200
        headers = {"x-request-id": "header-request"}

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {
                "id": "body-request",
                "model": "gpt-5.5",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
                "choices": [{"message": {"content": "FINISH", "reasoning": None}}],
            }

    telemetry = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("CAPX_LLM_TELEMETRY_PATH", str(telemetry))
    monkeypatch.setenv("CAPX_EPISODE_KEY", "libero_90/0/seed0")
    monkeypatch.setattr(capx_client.requests, "post", lambda *args, **kwargs: FakeResponse())
    query_args = capx_client.ModelQueryArgs(
        model="gpt-5.5",
        server_url="https://example.invalid/chat/completions",
        api_key="must-not-be-logged",
        temperature=0.0,
    )

    response = capx_client.query_model(
        query_args,
        [{"role": "user", "content": "private prompt must not be logged"}],
    )

    assert response["content"] == "FINISH"
    row = json.loads(telemetry.read_text(encoding="utf-8"))
    assert row["episode_key"] == "libero_90/0/seed0"
    assert row["actual_model"] == "gpt-5.5"
    assert row["usage"]["total_tokens"] == 13
    serialized = json.dumps(row)
    assert "private prompt" not in serialized
    assert "must-not-be-logged" not in serialized


def test_capx_point_grounding_uses_registered_runtime_vlm_and_telemetry(
    tmp_path: Path, monkeypatch,
) -> None:
    capx_root = Path(__file__).parents[1] / "third_party" / "rats" / "capx-baseline"
    monkeypatch.syspath_prepend(str(capx_root))
    import importlib.util
    from PIL import Image
    from capx.llm import client as capx_client  # ensure runtime telemetry import resolves

    spec = importlib.util.spec_from_file_location(
        "controlled_capx_molmo",
        capx_root / "capx" / "integrations" / "vision" / "molmo.py",
    )
    assert spec is not None and spec.loader is not None
    capx_molmo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capx_molmo)

    posted: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {
                "id": "point-request",
                "model": "gpt-5.5",
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
                "choices": [
                    {"message": {"content": '<point x="25" y="75">'}}
                ],
            }

    class FakeSession:
        @staticmethod
        def post(url, *, json, headers, timeout):
            posted.update(url=url, payload=json, headers=headers, timeout=timeout)
            return FakeResponse()

    telemetry = tmp_path / "capx_point_calls.jsonl"
    monkeypatch.setenv("CAPX_RUNTIME_VLM_URL", "https://vapi.invalid/chat/completions")
    monkeypatch.setenv("CAPX_RUNTIME_VLM_MODEL", "gpt-5.5")
    monkeypatch.setenv("CAPX_RUNTIME_VLM_KEY", "private-key")
    monkeypatch.setenv("CAPX_LLM_TELEMETRY_PATH", str(telemetry))
    monkeypatch.setenv("CAPX_EPISODE_KEY", "libero_90/1/seed0")
    monkeypatch.setattr(capx_molmo.requests, "Session", FakeSession)

    detector = capx_molmo.init_molmo()
    point = detector(Image.new("RGB", (200, 100)), ["black bowl"])["black bowl"]

    assert point == (50, 75)
    assert posted["url"] == "https://vapi.invalid/chat/completions"
    assert posted["payload"]["model"] == "gpt-5.5"
    assert posted["headers"]["Authorization"] == "Bearer private-key"
    row = json.loads(telemetry.read_text(encoding="utf-8"))
    assert row["call_site"] == "point_prompt_molmo"
    assert row["actual_model"] == "gpt-5.5"


def test_rats_point_grounding_uses_bound_agent_vlm_route(monkeypatch) -> None:
    rats_root = Path(__file__).parents[1] / "third_party" / "rats"
    monkeypatch.syspath_prepend(str(rats_root))
    import importlib.util
    from PIL import Image
    from rats.agents import base_agent

    spec = importlib.util.spec_from_file_location(
        "controlled_rats_molmo",
        rats_root / "rats" / "integrations" / "vision" / "molmo.py",
    )
    assert spec is not None and spec.loader is not None
    rats_molmo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rats_molmo)

    captured: dict[str, object] = {}

    def fake_query(system, prompt, **kwargs):
        captured.update(system=system, prompt=prompt, **kwargs)
        return '<point x="40" y="20">'

    monkeypatch.setenv("RATS_RUNTIME_VLM_MODEL", "gpt-5.5")
    monkeypatch.setattr(base_agent, "query_llm_text", fake_query)
    detector = rats_molmo.init_molmo(base_url="https://must-not-be-used.invalid")

    point = detector(Image.new("RGB", (100, 200)), ["plate"])["plate"]

    assert point == (40, 40)
    assert captured["model"] == "gpt-5.5"
    assert captured["temperature"] == 0.0
    assert len(captured["images"]) == 1


def test_source_generation_counts_use_final_capx_trace_and_rats_call_sites(
    tmp_path: Path,
) -> None:
    _write_evidence(
        tmp_path / "partial" / "all_responses.json",
        json.dumps(
            [{"decision": "initial", "code_blocks": ["x = 1"]}]
        ),
    )
    _write_evidence(
        tmp_path / "final" / "all_responses.json",
        json.dumps(
            [
                {"decision": "initial", "code_blocks": ["x = 1"]},
                {"decision": "regenerate", "code_blocks": ["x = 2"]},
                {"decision": "finish", "code_blocks": []},
            ]
        ),
    )
    assert _capx_source_events(tmp_path) == (2, 1)

    rats_calls = [
        {
            "caller": "policy_writer._generate_once",
            "response": {"content": "x = 1"},
        },
        {
            "caller": "multi_turn_decider.decide",
            "response": {"content": "REGENERATE\n```python\nx = 2\n```"},
        },
        {
            "caller": "multi_turn_decider.decide",
            "response": {"content": "FINISH"},
        },
        {
            "caller": "policy_writer._generate_once",
            "response": {"content": ""},
        },
    ]
    assert _rats_source_events(rats_calls) == (2, 1)


def test_capx_completion_requires_every_registered_evidence_category(
    tmp_path: Path,
) -> None:
    _write_evidence(tmp_path / "native_states.jsonl")
    _write_evidence(tmp_path / "sim_episodes.jsonl")
    _write_evidence(tmp_path / "llm_calls.jsonl")
    _write_evidence(tmp_path / "gpt-5.5" / "artifacts" / "summaries.txt")
    _write_evidence(
        tmp_path / "gpt-5.5" / "artifacts" / "trial_01" / "all_responses.json"
    )
    _write_evidence(
        tmp_path / "gpt-5.5" / "artifacts" / "trial_01" / "video_combined.mp4"
    )
    _write_evidence(
        tmp_path / "gpt-5.5" / "artifacts" / "aaa_done_flag" / "aaa_done_flag.txt"
    )

    evidence = _generated_episode_evidence(tmp_path, "capx")

    assert evidence["complete"] is True


def test_generated_evidence_rejects_more_than_one_registered_reset(
    tmp_path: Path,
) -> None:
    key = "libero_90/0/seed0"
    _write_evidence(tmp_path / "native_states.jsonl")
    _write_evidence(
        tmp_path / "sim_episodes.jsonl",
        "\n".join(
            json.dumps({"episode_key": key, "event": "environment_reset"})
            for _ in range(2)
        ),
    )
    _write_evidence(tmp_path / "agent_io" / "call.json", "{}")
    _write_evidence(tmp_path / "artifacts" / "final_summary.json", "{}")
    _write_evidence(tmp_path / "artifacts" / "iteration_001.json", "{}")
    _write_evidence(tmp_path / "artifacts" / "video.mp4")

    evidence = _generated_episode_evidence(
        tmp_path,
        "rats_90",
        expected_episode_keys={key},
        maximum_registered_resets=1,
    )

    assert evidence["complete"] is False
    assert evidence["registered_reset_counts"] == {key: 2}
    assert any("excess_registered_resets" in item for item in evidence["missing"])


@pytest.mark.parametrize(
    ("relative_path", "environment_variable"),
    [
        (
            "third_party/rats/capx-baseline/capx/envs/runner.py",
            "CAPX_MAX_TRIAL_RETRIES",
        ),
        ("third_party/rats/rats/envs/runner.py", "RATS_MAX_TRIAL_RETRIES"),
    ],
)
def test_generated_runner_timeout_retry_is_opt_in(
    relative_path: str,
    environment_variable: str,
) -> None:
    """A timed-out frozen episode must not receive another physical reset."""

    source = (Path(__file__).resolve().parents[1] / relative_path).read_text(
        encoding="utf-8"
    )
    assert f'os.getenv("{environment_variable}", "1")' in source


def test_generated_evidence_rejects_wrong_applied_init_state(tmp_path: Path) -> None:
    key = "libero_90/0/seed0"
    _write_evidence(tmp_path / "native_states.jsonl")
    _write_evidence(
        tmp_path / "sim_episodes.jsonl",
        json.dumps(
            {"episode_key": key, "event": "environment_reset", "init_state_index": 49}
        ),
    )
    _write_evidence(tmp_path / "agent_io" / "call.json", "{}")
    _write_evidence(tmp_path / "artifacts" / "final_summary.json", "{}")
    _write_evidence(tmp_path / "artifacts" / "iteration_001.json", "{}")
    _write_evidence(tmp_path / "artifacts" / "video.mp4")

    evidence = _generated_episode_evidence(
        tmp_path,
        "rats_90",
        expected_episode_keys={key},
        maximum_registered_resets=1,
        expected_init_state_indices={key: 0},
    )

    assert evidence["complete"] is False
    assert evidence["registered_init_state_indices"] == {key: [49]}
    assert any("wrong_init_state_index" in item for item in evidence["missing"])


def test_rats_completion_rejects_missing_recording(tmp_path: Path) -> None:
    _write_evidence(tmp_path / "native_states.jsonl")
    _write_evidence(tmp_path / "sim_episodes.jsonl")
    _write_evidence(tmp_path / "agent_io" / "0001.json")
    _write_evidence(tmp_path / "artifacts" / "final_summary.json")
    _write_evidence(tmp_path / "artifacts" / "iteration_001.json")

    evidence = _generated_episode_evidence(tmp_path, "rats_90")

    assert evidence["complete"] is False
    assert evidence["missing"] == ["videos"]


def test_racap_completion_is_keyed_by_every_registered_episode(tmp_path: Path) -> None:
    records = []
    for seed in (0, 1):
        key = f"libero_goal_task/2/seed{seed}"
        episode = tmp_path / "episodes" / f"t002_seed{seed}"
        video = episode / "rollout.mp4"
        trace = episode / "trace.md"
        trajectory = episode / "trajectory.json"
        for path in (video, trace, trajectory):
            _write_evidence(path)
        records.append(
            {
                "key": key,
                "native_success": seed == 0,
                "simulator_steps": 100 + seed,
                "init_state_index": seed,
                "artifacts": {
                    "video": str(video),
                    "trace": str(trace),
                    "trajectory": str(trajectory),
                    "video_frames": 10,
                },
            }
        )
        for stem in ("llm_calls.worker0.jsonl", "native_states.worker0.jsonl"):
            path = tmp_path / stem
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f'{{"episode_key":"{key}"}}\n')
        with (tmp_path / "sim_episodes.worker0.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "environment_reset",
                        "episode_key": key,
                        "public_seed": seed,
                        "seed": seed + 1,
                        "init_state_index": seed,
                    }
                )
                + "\n"
            )
    _write_evidence(tmp_path / "summary.json", json.dumps({"records": records}))

    evidence = _racap_group_evidence(
        tmp_path,
        suite="libero_goal_task",
        task_ids={2},
        seeds={0, 1},
    )

    assert evidence["complete"] is True
    assert evidence["expected_episode_count"] == 2


def test_racap_completion_rejects_unkeyed_model_telemetry(tmp_path: Path) -> None:
    video = tmp_path / "rollout.mp4"
    trace = tmp_path / "trace.md"
    trajectory = tmp_path / "trajectory.json"
    for path in (video, trace, trajectory):
        _write_evidence(path)
    key = "libero_object_swap/0/seed0"
    record = {
        "key": key,
        "native_success": False,
        "simulator_steps": 42,
        "artifacts": {
            "video": str(video),
            "trace": str(trace),
            "trajectory": str(trajectory),
            "video_frames": 5,
        },
    }
    _write_evidence(tmp_path / "summary.json", json.dumps({"records": [record]}))
    _write_evidence(tmp_path / "llm_calls.worker0.jsonl", '{"episode_key":"wrong"}\n')
    _write_evidence(tmp_path / "native_states.worker0.jsonl", f'{{"episode_key":"{key}"}}\n')
    _write_evidence(
        tmp_path / "sim_episodes.worker0.jsonl",
        json.dumps(
            {
                "event": "environment_reset",
                "episode_key": key,
                "public_seed": 0,
                "seed": 1,
                "init_state_index": 0,
            }
        )
        + "\n",
    )

    evidence = _racap_group_evidence(
        tmp_path,
        suite="libero_object_swap",
        task_ids={0},
        seeds={0},
    )

    assert evidence["complete"] is False
    assert evidence["missing_llm_telemetry"] == [key]


def test_racap_completion_rejects_hidden_retry_reset(tmp_path: Path) -> None:
    key = "libero_goal_task/2/seed0"
    episode = tmp_path / "episodes" / "t002_seed0"
    artifacts = {}
    for name, filename in (
        ("video", "rollout.mp4"),
        ("trace", "trace.md"),
        ("trajectory", "trajectory.json"),
    ):
        path = episode / filename
        _write_evidence(path)
        artifacts[name] = str(path)
    artifacts["video_frames"] = 10
    _write_evidence(
        tmp_path / "summary.json",
        json.dumps(
            {
                "records": [
                    {
                        "key": key,
                        "native_success": False,
                        "simulator_steps": 10,
                        "init_state_index": 0,
                        "artifacts": artifacts,
                    }
                ]
            }
        ),
    )
    for stem in ("llm_calls.worker0.jsonl", "native_states.worker0.jsonl"):
        _write_evidence(tmp_path / stem, json.dumps({"episode_key": key}) + "\n")
    reset = {
        "event": "environment_reset",
        "episode_key": key,
        "public_seed": 0,
        "seed": 1,
        "init_state_index": 0,
    }
    _write_evidence(
        tmp_path / "sim_episodes.worker0.jsonl",
        json.dumps(reset) + "\n" + json.dumps(reset) + "\n",
    )

    evidence = _racap_group_evidence(
        tmp_path,
        suite="libero_goal_task",
        task_ids={2},
        seeds={0},
    )

    assert evidence["complete"] is False
    assert "reset_count:2!=1" in evidence["reset_errors"][key]


def test_long_horizon_uses_method_specific_seed_conventions(tmp_path: Path) -> None:
    capx = _long_common_env(2, tmp_path / "capx", "capx", "gpt-5.5")
    rats = _long_common_env(2, tmp_path / "rats", "rats_90", "gpt-5.5")

    assert capx["CAPX_SEED_OFFSET"] == "2"
    assert capx["CAPX_REGISTERED_EPISODE_KEY"] == (
        "libero_long_all_to_basket/seed2"
    )
    assert rats["RATS_SEED_OFFSET"] == "2"
    assert rats["RATS_REGISTERED_EPISODE_KEY"] == (
        "libero_long_all_to_basket/seed2"
    )
    assert capx["CAPX_ENV_STACK"] == "libero"
    assert rats["CAPX_ENV_STACK"] == "libero"
    assert capx["CAPX_TRIAL_TIMEOUT_SECONDS"] == "0"
    assert rats["RATS_TRIAL_TIMEOUT_SECONDS"] == "0"
    assert Path(capx["PYTHONPATH"].split(os.pathsep)[0]).is_absolute()
    assert capx["PYTHONPATH"] == rats["PYTHONPATH"]


def test_long_horizon_isolates_frozen_rats_memory_per_seed(tmp_path: Path) -> None:
    source = tmp_path / "frozen" / "failure_memory"
    source.mkdir(parents=True)
    (source / "episodes.json").write_text('[{"episode_id":"seed"}]', encoding="utf-8")
    (source / "lessons.json").write_text("[]", encoding="utf-8")

    isolated, expected = _materialize_isolated_rats_memory(
        source,
        tmp_path / "seed_00",
    )

    assert isolated != source
    assert (isolated / "episodes.json").read_text(encoding="utf-8") == (
        source / "episodes.json"
    ).read_text(encoding="utf-8")
    assert expected
    (isolated / "episodes.json").write_text("[]", encoding="utf-8")
    assert (source / "episodes.json").read_text(encoding="utf-8") != "[]"


def test_resumed_development_ignores_old_quota_log_text(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("old launch: insufficient balance\n", encoding="utf-8")
    offset = log.stat().st_size
    with log.open("a", encoding="utf-8") as handle:
        handle.write("new launch completed normally\n")

    assert "insufficient balance" not in _read_log_since(log, offset)


def test_resumed_development_archives_old_abort_sentinel(tmp_path: Path) -> None:
    sentinel = tmp_path / "ABORTED.json"
    sentinel.write_text('{"reason": "quota"}', encoding="utf-8")

    archived = _archive_previous_abort(tmp_path)

    assert archived is not None and archived.is_file()
    assert not sentinel.exists()
    assert archived.read_text(encoding="utf-8") == '{"reason": "quota"}'


def test_resumed_development_restores_latest_completed_transaction(
    tmp_path: Path,
) -> None:
    (tmp_path / "iteration_001.json").write_text(
        '{"iteration": 1, "success": true}', encoding="utf-8"
    )
    snapshot = tmp_path / "snapshots" / "iter001"
    (snapshot / "failure_memory").mkdir(parents=True)
    (snapshot / "skills.json").write_text('[{"name":"committed"}]', encoding="utf-8")
    (snapshot / "failure_memory" / "episodes.json").write_text(
        '[{"iteration":1}]', encoding="utf-8"
    )
    (tmp_path / "failure_memory").mkdir()
    (tmp_path / "skills.json").write_text('[{"name":"partial"}]', encoding="utf-8")
    (tmp_path / "failure_memory" / "episodes.json").write_text(
        '[{"iteration":1},{"iteration":2}]', encoding="utf-8"
    )

    restored = _restore_latest_completed_snapshot(tmp_path)

    assert restored is not None and restored["restored"] is True
    assert json.loads((tmp_path / "skills.json").read_text()) == [
        {"name": "committed"}
    ]
    assert json.loads(
        (tmp_path / "failure_memory" / "episodes.json").read_text()
    ) == [{"iteration": 1}]
    archive = Path(str(restored["archived_partial_state"]))
    assert json.loads((archive / "skills.json").read_text()) == [
        {"name": "partial"}
    ]
    assert (tmp_path / "resume_restorations.jsonl").is_file()

    no_op = _restore_latest_completed_snapshot(tmp_path)
    assert no_op is not None and no_op["restored"] is False

    audit = _resume_transaction_audit(tmp_path)
    assert len(audit) == 1
    assert bool(audit.iloc[0].audit_pass)
    assert bool(audit.iloc[0].archive_exists)
    assert audit.iloc[0].latest_completed_iteration == 1


def test_quota_failure_stops_peer_process(tmp_path: Path) -> None:
    stop = threading.Event()
    quota = [
        sys.executable,
        "-c",
        "import time; print('insufficient balance', flush=True); time.sleep(30)",
    ]
    peer = [
        sys.executable,
        "-c",
        "import time; print('peer started', flush=True); time.sleep(30)",
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        quota_result = pool.submit(
            _run_logged,
            quota,
            cwd=tmp_path,
            env=os.environ.copy(),
            log_path=tmp_path / "quota.log",
            stop_event=stop,
        )
        peer_result = pool.submit(
            _run_logged,
            peer,
            cwd=tmp_path,
            env=os.environ.copy(),
            log_path=tmp_path / "peer.log",
            stop_event=stop,
        )
        quota_row = quota_result.result()
        peer_row = peer_result.result()
    assert quota_row["returncode"] == 75
    assert quota_row["quota_failure"] is True
    assert peer_row["stopped_by_peer_quota"] is True
    assert max(quota_row["seconds"], peer_row["seconds"]) < 10


def test_generated_resume_archives_partial_payloads_before_clean_rerun(
    tmp_path: Path,
) -> None:
    method_root = tmp_path / "measured" / "rats_base"
    complete_job = {
        "cohort": "libero90_id_replay",
        "suite": "libero_90",
        "task_id": 0,
        "seed": 0,
    }
    partial_job = {
        "cohort": "libero90_id_replay",
        "suite": "libero_90",
        "task_id": 1,
        "seed": 0,
    }
    not_started_job = {
        "cohort": "libero90_id_replay",
        "suite": "libero_90",
        "task_id": 2,
        "seed": 0,
    }
    complete_root = (
        method_root
        / "libero90_id_replay/libero_90/task_00/seed_00"
    )
    partial_root = (
        method_root
        / "libero90_id_replay/libero_90/task_01/seed_00"
    )
    complete_root.mkdir(parents=True)
    partial_root.mkdir(parents=True)
    (complete_root / "COMPLETE.json").write_text("{}", encoding="utf-8")
    (complete_root / "native_states.jsonl").write_text(
        '{"native_success":true}\n', encoding="utf-8"
    )
    (partial_root / "sim_episodes.jsonl").write_text(
        '{"event":"environment_reset"}\n', encoding="utf-8"
    )
    (partial_root / "native_states.jsonl").write_text(
        '{"simulator_steps":17}\n', encoding="utf-8"
    )
    (partial_root / "artifacts").mkdir()
    (partial_root / "artifacts" / "partial.mp4").write_bytes(b"partial-video")
    (method_root / "shared_api_services").mkdir(parents=True)
    (method_root / "shared_api_services" / "launcher.log").write_text(
        "stopped after quota\n", encoding="utf-8"
    )
    (method_root / "ABORTED.json").write_text(
        '{"reason":"quota"}', encoding="utf-8"
    )
    (method_root / "job_results.json").write_text("[]", encoding="utf-8")
    (method_root / "run_manifest.json").write_text(
        '{"immutable":true}', encoding="utf-8"
    )

    archived = _archive_incomplete_generated_resume_state(
        "rats_base",
        [complete_job, partial_job, not_started_job],
        method_root,
    )

    assert archived is not None
    assert archived["already_complete_jobs"] == 1
    assert archived["incomplete_jobs"] == 2
    assert archived["archived_partial_episode_directories"] == 1
    assert complete_root.is_dir()
    assert (complete_root / "COMPLETE.json").is_file()
    assert not partial_root.exists()
    assert not (method_root / "ABORTED.json").exists()
    assert not (method_root / "job_results.json").exists()
    assert not (method_root / "shared_api_services").exists()
    assert (method_root / "run_manifest.json").is_file()

    manifest_path = Path(str(archived["manifest"]))
    assert manifest_path.is_file()
    assert tmp_path / "interrupted_attempts" in manifest_path.parents
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_payload = next(
        row for row in manifest["payloads"]
        if row["category"] == "incomplete_episode"
    )
    archived_partial = Path(episode_payload["archive"])
    assert archived_partial.is_dir()
    assert (archived_partial / "sim_episodes.jsonl").is_file()
    assert episode_payload["payload"]["files"] == 3
    assert episode_payload["payload"]["bytes"] > 0

    # The second preparation is idempotent: there is still work to run, but
    # no stale payload remains to contaminate the new registered episode.
    assert _archive_incomplete_generated_resume_state(
        "rats_base",
        [complete_job, partial_job, not_started_job],
        method_root,
    ) is None


def test_method_run_lock_rejects_concurrent_resume_mutation(tmp_path: Path) -> None:
    method_root = tmp_path / "measured" / "rats_base"
    method_root.mkdir(parents=True)

    first = _acquire_method_run_lock(method_root)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            _acquire_method_run_lock(method_root)
    finally:
        first.close()

    # Process exit closes the same descriptor; a later clean resume can then
    # acquire the persistent metadata path without deleting it.
    resumed = _acquire_method_run_lock(method_root)
    resumed.close()


def test_registered_resume_migration_is_exact_and_auditable(tmp_path: Path) -> None:
    method_root = tmp_path / "measured" / "rats_base"
    manifest_path = method_root / "run_manifest.json"
    registry = tmp_path / "registry"
    method_root.mkdir(parents=True)
    registry.mkdir()
    jobs = [
        {
            "cohort": "libero90_id_replay",
            "suite": "libero_90",
            "task_id": 0,
            "seed": 0,
        }
    ]
    episode = method_root / "libero90_id_replay/libero_90/task_00/seed_00"
    episode.mkdir(parents=True)
    (episode / "COMPLETE.json").write_text("{}", encoding="utf-8")
    before = {"method": "rats_base", "source": {"rats": "old"}}
    after = {"method": "rats_base", "source": {"rats": "new"}}
    manifest_path.write_text(
        json.dumps(before, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    keys = ["libero_90/0/seed0"]
    registration = {
        "migration_id": "seed_fix",
        "method": "rats_base",
        "from_manifest_sha256": _json_payload_sha256(before),
        "to_manifest_sha256": _json_payload_sha256(after),
        "allowed_manifest_diff_paths": ["source.rats"],
        "expected_completed_episode_count": 1,
        "expected_completed_episode_keys_sha256": _episode_key_set_sha256(keys),
        "compatible_completed_public_seeds": [0],
    }
    (registry / "seed_fix.json").write_text(
        json.dumps(registration), encoding="utf-8"
    )

    applied = _apply_registered_manifest_migration(
        method="rats_base",
        method_root=method_root,
        manifest_path=manifest_path,
        before=before,
        after=after,
        jobs=jobs,
        registry_dir=registry,
    )

    assert applied is not None
    assert applied["completed_public_seeds"] == [0]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == after
    archive = method_root / "resume_migrations/seed_fix"
    assert json.loads(
        (archive / "run_manifest_before.json").read_text(encoding="utf-8")
    ) == before
    assert json.loads(
        (archive / "run_manifest_after.json").read_text(encoding="utf-8")
    ) == after
    assert (archive / "APPLIED.json").is_file()


def test_registered_resume_migration_rejects_uncovered_completed_seed(
    tmp_path: Path,
) -> None:
    method_root = tmp_path / "measured" / "rats_base"
    manifest_path = method_root / "run_manifest.json"
    registry = tmp_path / "registry"
    episode = method_root / "cohort/libero_spatial/task_00/seed_01"
    episode.mkdir(parents=True)
    registry.mkdir()
    (episode / "COMPLETE.json").write_text("{}", encoding="utf-8")
    before = {"source": "old"}
    after = {"source": "new"}
    manifest_path.write_text(json.dumps(before, indent=2), encoding="utf-8")
    keys = ["libero_spatial/0/seed1"]
    (registry / "seed_fix.json").write_text(
        json.dumps(
            {
                "migration_id": "seed_fix",
                "method": "rats_base",
                "from_manifest_sha256": _json_payload_sha256(before),
                "to_manifest_sha256": _json_payload_sha256(after),
                "allowed_manifest_diff_paths": ["source"],
                "expected_completed_episode_count": 1,
                "expected_completed_episode_keys_sha256": _episode_key_set_sha256(keys),
                "compatible_completed_public_seeds": [0],
            }
        ),
        encoding="utf-8",
    )
    jobs = [
        {
            "cohort": "cohort",
            "suite": "libero_spatial",
            "task_id": 0,
            "seed": 1,
        }
    ]

    with pytest.raises(RuntimeError, match="does not cover completed public seeds"):
        _apply_registered_manifest_migration(
            method="rats_base",
            method_root=method_root,
            manifest_path=manifest_path,
            before=before,
            after=after,
            jobs=jobs,
            registry_dir=registry,
        )


def test_generated_resume_archive_rolls_back_if_manifest_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_root = tmp_path / "measured" / "rats_base"
    job = {
        "cohort": "libero90_id_replay",
        "suite": "libero_90",
        "task_id": 7,
        "seed": 0,
    }
    partial_root = (
        method_root
        / "libero90_id_replay/libero_90/task_07/seed_00"
    )
    partial_root.mkdir(parents=True)
    (partial_root / "sim_episodes.jsonl").write_text(
        '{"event":"environment_reset"}\n', encoding="utf-8"
    )
    (method_root / "ABORTED.json").write_text(
        '{"reason":"quota"}', encoding="utf-8"
    )

    def fail_manifest(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic manifest failure")

    monkeypatch.setattr(run_libero, "_write_json", fail_manifest)
    with pytest.raises(RuntimeError, match="synthetic manifest failure"):
        _archive_incomplete_generated_resume_state(
            "rats_base", [job], method_root
        )

    assert (partial_root / "sim_episodes.jsonl").is_file()
    assert (method_root / "ABORTED.json").is_file()


def test_generated_method_interrupt_stops_peer_before_thread_pool_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer_started = threading.Event()
    peer_stopped = threading.Event()

    @contextlib.contextmanager
    def no_services(*_args: object, **_kwargs: object):
        yield

    def fake_capx_job(
        job: dict[str, object],
        _args: object,
        _root: Path,
        stop_event: threading.Event,
    ) -> dict[str, object]:
        if int(job["task_id"]) == 0:
            assert peer_started.wait(timeout=2)
            raise KeyboardInterrupt
        peer_started.set()
        assert stop_event.wait(timeout=2)
        peer_stopped.set()
        return {"job": job, "returncode": 75, "quota_failure": False}

    monkeypatch.setattr(run_libero, "_shared_api_server_bundle", no_services)
    monkeypatch.setattr(run_libero, "_capx_job", fake_capx_job)
    args = SimpleNamespace(workers=2)
    jobs = [
        {"cohort": "fixture", "suite": "fixture", "task_id": 0, "seed": 0},
        {"cohort": "fixture", "suite": "fixture", "task_id": 1, "seed": 0},
    ]

    with pytest.raises(KeyboardInterrupt):
        _run_generated_method("capx", jobs, args, tmp_path)

    assert peer_stopped.is_set()


def test_simulator_steps_sum_across_runtime_self_check_and_execution() -> None:
    resets = [{"time": 10.0}, {"time": 20.0}, {"time": 30.0}]
    states = [
        {"time": 21.0, "simulator_steps": 64},
        {"time": 31.0, "simulator_steps": 400},
        {"time": 32.0, "simulator_steps": 799},
    ]
    assert _total_simulator_steps(states, resets) == 64 + 799


def test_public_api_call_count_excludes_code_audit_rows() -> None:
    states = [
        {"event": "native_state_after_code"},
        {"event": "native_state_after_api", "api_name": "get_observation"},
        {"event": "native_state_after_api", "api_name": "goto_pose"},
        {"event": "native_state_after_code"},
    ]

    assert _event_count(states, "native_state_after_api") == 2
    assert _event_count(states, "native_state_after_code") == 2


def test_analysis_accepts_only_completed_verified_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "EVIDENCE.json"
    _write_evidence(evidence, '{"complete": true}')
    assert _verified_evidence(tmp_path, require_complete_marker=True) is None

    _write_evidence(tmp_path / "COMPLETE.json", "{}")
    assert _verified_evidence(tmp_path, require_complete_marker=True) == evidence

    _write_evidence(evidence, '{"complete": false}')
    assert _verified_evidence(tmp_path, require_complete_marker=True) is None


def test_paired_bootstrap_clusters_repeated_seeds_by_suite_task() -> None:
    rows = []
    for method, successes, walls in (
        ("capx", [0, 0, 0, 1, 1, 1], [1, 1, 1, 9, 9, 9]),
        ("rats_90", [1, 1, 1, 1, 1, 1], [0, 0, 0, 6, 6, 6]),
    ):
        for index, (success, wall) in enumerate(zip(successes, walls)):
            task_id, seed = divmod(index, 3)
            rows.append(
                {
                    "method": method,
                    "cohort": "zero_shot",
                    "suite": "suite",
                    "task_id": task_id,
                    "seed": seed,
                    "episode_key": f"suite/{task_id}/seed{seed}",
                    "native_success": bool(success),
                    "predicate_completion": success,
                    "policy_wall_seconds": wall,
                    "artifact_wall_seconds": wall + 1,
                    "model_latency_seconds": wall + 2,
                    "median_model_call_latency_seconds": wall + 3,
                    "prompt_tokens": wall,
                    "completion_tokens": wall + 4,
                    "cached_tokens": wall + 5,
                    "model_calls": wall,
                    "estimated_api_cost_usd": wall,
                    "policy_or_skill_calls": wall + 6,
                    "simulator_steps": wall + 7,
                    "method_native_turns": wall + 8,
                    "source_generation_events": wall + 9,
                    "source_repair_events": wall + 10,
                }
            )

    paired = _paired_statistics(pd.DataFrame(rows), bootstrap=1000)
    row = paired.iloc[0]

    assert row.paired_episodes == 6
    assert row.paired_task_clusters == 2
    assert row.bootstrap_unit == "suite_task"
    assert row.mcnemar_holm_within_cohort_p == row.mcnemar_holm_global_p
    assert row.mcnemar_holm_p == row.mcnemar_holm_global_p
    assert row.policy_wall_seconds_task_clusters == 2
    for metric in (
        "predicate_completion",
        "artifact_wall_seconds",
        "model_latency_seconds",
        "median_model_call_latency_seconds",
        "completion_tokens",
        "cached_tokens",
        "policy_or_skill_calls",
        "simulator_steps",
        "method_native_turns",
        "source_generation_events",
        "source_repair_events",
    ):
        assert row[f"{metric}_task_clusters"] == 2
        assert pd.notna(row[f"{metric}_bootstrap_low"])
        assert pd.notna(row[f"{metric}_bootstrap_high"])


def test_paired_statistics_records_within_cohort_and_global_holm_families() -> None:
    rows = []
    for cohort in ("cohort_a", "cohort_b"):
        for method, success in (("capx", True), ("rats_base", False)):
            for task_id in range(4):
                rows.append(
                    {
                        "method": method,
                        "cohort": cohort,
                        "suite": cohort,
                        "task_id": task_id,
                        "seed": 0,
                        "episode_key": f"{cohort}/{task_id}/seed0",
                        "native_success": success,
                    }
                )

    paired = _paired_statistics(pd.DataFrame(rows), bootstrap=100)

    assert len(paired) == 2
    assert set(paired.mcnemar_exact_p) == {0.125}
    assert set(paired.mcnemar_holm_within_cohort_p) == {0.125}
    assert set(paired.mcnemar_holm_global_p) == {0.25}
    assert paired.mcnemar_holm_p.equals(paired.mcnemar_holm_global_p)


def test_rats_failure_audit_pairs_diagnosis_with_same_execution_attempt(
    tmp_path: Path,
) -> None:
    from experiments.controlled_comparison.analyze_rats_failure_modes import analyze

    artifact = tmp_path / "summary.json"
    artifact.write_text(
        json.dumps(
            {
                "diagnosis_attempt_1": {
                    "failure_mode": "code_bug",
                    "failed_step": "step-2",
                },
                "execution_attempt_1": {
                    "success": False,
                    "stderr_snippet": "unexpected keyword argument",
                },
                "execution_attempt_2": {
                    "success": False,
                    "stderr_snippet": "agentview camera dictionary",
                },
            }
        )
    )
    episodes = pd.DataFrame(
        [
            {
                "method": "rats_90",
                "cohort": "libero90_id_replay",
                "episode_key": "suite/0/seed0",
                "native_success": False,
                "artifact_path": str(artifact),
            }
        ]
    )

    failures, summary = analyze(episodes, "rats_90")

    assert summary["structured_diagnoses"] == 1
    assert failures.iloc[0].diagnosis_attempt == 1
    assert failures.iloc[0].execution_attempt == 1
    assert failures.iloc[0].diagnostic_signatures == (
        "failure_mode:code_bug;unsupported_argument_contract"
    )


def test_multiline_agent_io_json_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "call.json"
    path.write_text(
        '{\n  "episode_key": "suite/0/seed0",\n  "actual_model": "gpt-5.5"\n}',
        encoding="utf-8",
    )
    grouped = _telemetry_by_key([path])
    assert grouped["suite/0/seed0"][0]["actual_model"] == "gpt-5.5"


def test_price_estimate_separates_cached_and_uncached_input() -> None:
    usage = {
        "prompt_tokens": 1_000,
        "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 400},
    }
    assert _request_list_price_usd(usage) == 0.0062


def test_price_estimate_applies_long_context_multiplier_per_request() -> None:
    usage = {
        "prompt_tokens": 300_000,
        "completion_tokens": 10_000,
        "prompt_tokens_details": {"cached_tokens": 100_000},
    }
    assert _request_list_price_usd(usage) == 2.55


def test_one_shot_critic_trace_excludes_evaluator_only_lines(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "agent called pickplace\n"
        "Predicates: [inside object receptacle]=false\n"
        "unsatisfied_conditions: [on secret_object secret_target]\n"
        "native_success=false\n"
        "agent observed empty grasp\n",
        encoding="utf-8",
    )

    trace = _public_trace(tmp_path)

    assert "agent called pickplace" in trace
    assert "agent observed empty grasp" in trace
    assert "Predicates" not in trace
    assert "native_success" not in trace
    assert "secret_object" not in trace


def test_one_shot_card_bound_preserves_all_registered_sections() -> None:
    card = "\n".join(
        [
            "## Observed mechanism",
            " ".join(["observation"] * 250),
            "## Transferable guidance",
            " ".join(["guidance"] * 150),
            "## Avoid",
            " ".join(["avoid"] * 80),
        ]
    )

    bounded = _bounded_card(card)

    assert len(bounded.split()) <= 400
    for heading in one_shot_cards.CARD_HEADINGS:
        assert heading in bounded
    assert "current visual evidence must override it" in bounded


def test_one_shot_rats_iteration_artifact_resolves_to_task_root(tmp_path: Path) -> None:
    artifact = tmp_path / "task_00" / "seed_00" / "artifacts" / "iteration_001.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    root = _episode_root(
        {"artifact_path": str(artifact), "task_id": 0, "seed": 0}
    )

    assert root == tmp_path / "task_00" / "seed_00"


def test_long_horizon_resources_include_runtime_self_check() -> None:
    events = [
        {"time": 0.0, "event": "native_state_after_api", "simulator_steps": 40},
        {"time": 1.0, "event": "native_state_after_code", "simulator_steps": 60},
        {
            "time": 10.0,
            "event": "native_state_after_api",
            "simulator_steps": 100,
            "native_predicates": [{"satisfied": True}],
        },
    ]
    resets = [{"time": 0.0}, {"time": 9.0}]

    rows = _metrics_from_trace(
        method="rats_90",
        seed=0,
        events=events,
        calls=[],
        resets=resets,
        execution_start=9.0,
    )
    final = next(row for row in rows if row["checkpoint"] == "final")

    assert final["completed_objects"] == 1
    assert final["physical_api_calls"] == 2
    assert final["simulator_steps"] == 60 + 100


def test_long_horizon_final_resources_include_calls_after_last_physical_state() -> None:
    events = [
        {"time": 0.0, "event": "native_state_after_code", "simulator_steps": 10},
        {
            "time": 100.0,
            "event": "native_state_after_api",
            "simulator_steps": 50,
            "native_predicates": [{"satisfied": True}],
        },
    ]
    calls = [
        {"time": 90.0, "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        {"time": 350.0, "usage": {"prompt_tokens": 20, "completion_tokens": 3}},
    ]

    rows = _metrics_from_trace(
        method="rats_base",
        seed=0,
        events=events,
        calls=calls,
    )
    at_five = next(row for row in rows if row["checkpoint"] == "5min")
    final = next(row for row in rows if row["checkpoint"] == "final")

    assert at_five["model_calls"] == 1
    assert final["model_calls"] == 2
    assert final["prompt_tokens"] == 30
    assert final["elapsed_seconds"] == 350.0
    assert final["completed_objects"] == 1
    assert final["simulator_horizon_reached"] is False
    assert final["physical_trajectory_end_seconds"] == 100.0
    assert final["post_physical_overhead_seconds"] == 250.0


def test_long_horizon_records_first_horizon_and_post_horizon_overhead() -> None:
    events = [
        {"time": 0.0, "event": "native_state_after_api", "simulator_steps": 0},
        {
            "time": 100.0,
            "event": "native_state_after_api",
            "simulator_steps": 8000,
            "native_predicates": [{"satisfied": True}],
        },
        {
            "time": 180.0,
            "event": "native_state_after_api",
            "simulator_steps": 8000,
            "native_predicates": [{"satisfied": True}],
        },
    ]
    calls = [{"time": 220.0, "usage": {"prompt_tokens": 5}}]

    final = next(
        row
        for row in _metrics_from_trace(
            method="capx", seed=0, events=events, calls=calls
        )
        if row["checkpoint"] == "final"
    )

    assert final["simulator_horizon_reached"] is True
    assert final["time_to_simulator_horizon_seconds"] == 100.0
    assert final["physical_trajectory_end_seconds"] == 100.0
    assert final["post_physical_overhead_seconds"] == 120.0


def test_long_horizon_statistics_are_paired_by_seed() -> None:
    rows = []
    for method, values in (("capx", [0, 1]), ("rats_90", [1, 2])):
        for seed, completed in enumerate(values):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "checkpoint": "5min",
                    "completed_objects": completed,
                    "completion_fraction": completed / 7,
                    "full_success": False,
                    "model_calls": 1,
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "estimated_api_cost_usd": 0.1,
                    "physical_api_calls": 4,
                    "simulator_steps": 5,
                }
            )
    frame = pd.DataFrame(rows)

    aggregate = _long_aggregate(frame, resamples=1000)
    paired = _long_pairwise(frame, resamples=1000)

    assert len(aggregate) == 2
    assert len(paired) == 1
    assert paired.iloc[0].paired_seeds == 2
    assert paired.iloc[0].completed_objects_difference_a_minus_b == -1
    assert paired.iloc[0].bootstrap_unit == "paired_seed"


def test_robosuite_pairing_reports_task_and_overall_scopes() -> None:
    rows = []
    for task in ("cube_lifting", "cube_restack"):
        for seed in range(2):
            for method, success in (("capx", False), ("rats_90", seed == 0)):
                rows.append(
                    {
                        "method": method,
                        "task": task,
                        "seed": seed,
                        "episode_key": f"{task}/seed{seed}",
                        "native_success": success,
                        "wall_seconds": 1.0,
                        "model_calls": 1,
                        "estimated_api_cost_usd": 0.1,
                    }
                )

    paired = _robosuite_paired(pd.DataFrame(rows), resamples=1000)

    assert set(paired.scope) == {"cube_lifting", "cube_restack", "ALL"}
    overall = paired[paired.scope == "ALL"].iloc[0]
    assert overall.paired_episodes == 4
    assert overall.bootstrap_unit == "task"
    assert 0 <= overall.mcnemar_holm_p <= 1


def test_robosuite_worker_protocol_separates_development_and_evaluation() -> None:
    assert (
        _validate_worker_protocol(
            method="rats_rs_evolved",
            workers=3,
            seeds=(100, 101, 102),
            services_already_running=True,
        )
        == "matched_domain_development"
    )
    assert (
        _validate_worker_protocol(
            method="rats_rs_evolved",
            workers=5,
            seeds=(0, 1, 2, 3, 4),
            services_already_running=False,
        )
        == "sealed_evaluation"
    )
    with pytest.raises(SystemExit, match="three workers"):
        _validate_worker_protocol(
            method="rats_rs_evolved",
            workers=5,
            seeds=(100, 101, 102),
            services_already_running=True,
        )
    with pytest.raises(SystemExit, match="parent-owned services"):
        _validate_worker_protocol(
            method="capx",
            workers=3,
            seeds=(100, 101, 102),
            services_already_running=True,
        )


def test_robosuite_campaign_is_sequential_and_resumes_partial_racap_eval(
    tmp_path: Path,
) -> None:
    transfer = tmp_path / "transfer"
    partial = transfer / "racap_rs_evolved"
    partial.mkdir(parents=True)
    (partial / "PAUSED.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        python=Path(sys.executable),
        executor_python=Path("/registered/venv/bin/python"),
        rats_root=tmp_path / "rats_source",
        robosuite_root=tmp_path / "robosuite_source",
        racap_evolution_root=tmp_path / "racap_evolution",
        rats_evolution_root=tmp_path / "rats_evolution",
        transfer_root=transfer,
        development_analysis_root=tmp_path / "development_analysis",
        transfer_analysis_root=tmp_path / "transfer_analysis",
        final_report_root=tmp_path / "final_report",
    )

    phases = _phase_specs(args)

    assert [phase["name"] for phase in phases] == [
        "racap_rs_development",
        "rats_rs_development",
        "development_analysis",
        "racap_zero_shot_corrected_evaluation",
        "racap_rs_sealed_evaluation",
        "rats_rs_sealed_evaluation",
        "transfer_analysis",
        "final_workbook",
    ]
    racap_eval = next(
        phase for phase in phases if phase["name"] == "racap_rs_sealed_evaluation"
    )
    rats_eval = next(
        phase for phase in phases if phase["name"] == "rats_rs_sealed_evaluation"
    )
    assert "--resume" in racap_eval["command"]
    assert racap_eval["command"][racap_eval["command"].index("--workers") + 1] == "5"
    assert rats_eval["command"][rats_eval["command"].index("--workers") + 1] == "5"
    racap_dev = next(
        phase for phase in phases if phase["name"] == "racap_rs_development"
    )
    rats_dev = next(
        phase for phase in phases if phase["name"] == "rats_rs_development"
    )
    assert "--iterations" not in racap_dev["command"]
    assert rats_dev["command"][rats_dev["command"].index("--iterations") + 1] == "3"


def test_robosuite_campaign_rejects_and_archives_pre_tcp_fix_racap_result(
    tmp_path: Path,
) -> None:
    transfer = tmp_path / "transfer"
    stale = transfer / "racap_phase2_zero_shot"
    stale.mkdir(parents=True)
    (stale / "COMPLETE.json").write_text(
        json.dumps({"episodes": 35, "successes": 5}), encoding="utf-8"
    )
    (stale / "run_manifest.json").write_text(
        json.dumps({"method": "racap_phase2_zero_shot"}), encoding="utf-8"
    )
    assert not _corrected_racap_complete(stale)

    archived = _archive_invalid_racap_zero_shot(transfer)
    assert archived is not None
    assert not stale.exists()
    marker = json.loads((archived / "INVALID_ATTEMPT.json").read_text(encoding="utf-8"))
    assert marker["status"] == "invalid_non_scoring"
    assert marker["required_contract"] == "live_eef_link_robot0_base"


def test_robosuite_resume_archives_incomplete_task_outside_scored_tree(
    tmp_path: Path,
) -> None:
    method_root = tmp_path / "robosuite" / "sweep_001_candidate"
    task_root = method_root / "cube_lifting"
    task_root.mkdir(parents=True)
    (task_root / "llm_calls.jsonl").write_text(
        '{"quota": true}\n', encoding="utf-8"
    )

    archived = _archive_incomplete_task(method_root, task_root, "cube_lifting")

    assert archived is not None
    assert (
        archived.parent
        == method_root.parent / "sweep_001_candidate_invalidated_attempts"
    )
    assert (archived / "llm_calls.jsonl").is_file()
    assert (archived / "NOT_SCORED.json").is_file()
    assert not task_root.exists()

    task_root.mkdir(parents=True)
    (task_root / "COMPLETE.json").write_text("{}", encoding="utf-8")
    assert _archive_incomplete_task(method_root, task_root, "cube_lifting") is None
    assert task_root.exists()


def test_rats_robosuite_resume_archives_partial_skill_extraction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "robosuite_rats_rs_15"
    candidate = root / "candidate_libraries" / "candidate_004"
    candidate.mkdir(parents=True)
    (candidate / "skills.json").write_text("[]", encoding="utf-8")
    (candidate / "ABORTED.json").write_text(
        '{"reason": "quota"}', encoding="utf-8"
    )

    archived = _archive_incomplete_candidate(root, candidate, 4)

    assert archived is not None
    assert archived.parent == root / "invalidated_extractions"
    assert (archived / "skills.json").is_file()
    assert (archived / "NOT_SCORED.json").is_file()
    assert not candidate.exists()

    candidate.mkdir(parents=True)
    (candidate / "seal_manifest.json").write_text("{}", encoding="utf-8")
    assert _archive_incomplete_candidate(root, candidate, 4) is None
    assert candidate.exists()


def test_robosuite_development_audit_requires_compute_matched_products(
    tmp_path: Path,
) -> None:
    methods = ("racap_rs_evolved", "rats_rs_evolved")
    episodes = []
    candidates = []
    cohorts_by_method = {"racap_rs_evolved": 16, "rats_rs_evolved": 4}
    for method in methods:
        champion = 3
        for cohort in range(cohorts_by_method[method]):
            native = 3 if cohort != 1 else 4
            parent = 3 if cohort <= 1 else 4
            promoted = cohort in {0, 1}
            if promoted:
                champion = native
            candidates.append(
                {
                    "method": method,
                    "cohort_index": cohort,
                    "native_success": native,
                    "parent_native_success": parent,
                    "promoted": promoted,
                    "actual_models": "gpt-5.5",
                    "champion_native_success_after": champion,
                }
            )
            success_index = 0
            for task in (
                "cube_lifting",
                "cube_stack",
                "nut_assembly",
                "spill_wipe",
                "two_arm_lift",
            ):
                for seed in (100, 101, 102):
                    episodes.append(
                        {
                            "method": method,
                            "cohort_index": cohort,
                            "episode_key": f"robosuite/{task}/seed{seed}",
                            "native_success": success_index < native,
                            "scorable": True,
                            "infrastructure_error": False,
                            "abort_reason": "",
                            "simulator_resets": 1,
                            "artifact_bundle_complete": True,
                            "model_telemetry_complete": True,
                            "llm_terminal_failures": 0,
                            "llm_quota_failures": 0,
                        }
                    )
                    success_index += 1

    racap_root = tmp_path / "racap"
    rats_root = tmp_path / "rats"
    for root, manifest_name, total in (
        (racap_root, "protocol_manifest.json", 240),
        (rats_root, "run_manifest.json", 60),
    ):
        (root / "frozen_champion").mkdir(parents=True)
        (root / manifest_name).write_text(
            json.dumps(
                {
                    "development_tasks": [
                        "cube_lifting",
                        "cube_stack",
                        "nut_assembly",
                        "spill_wipe",
                        "two_arm_lift",
                    ],
                    "development_seeds": [100, 101, 102],
                    "workers": 3,
                    "physical_budget": {"total_development_episodes": total},
                    "budget_basis": (
                        "approximately_equal_effective_development_wall_time"
                        if manifest_name == "run_manifest.json"
                        else None
                    ),
                }
            ),
            encoding="utf-8",
        )

    audit = _robosuite_development_audit(
        pd.DataFrame(episodes),
        pd.DataFrame(candidates),
        racap_root=racap_root,
        rats_root=rats_root,
    )
    assert audit["status"] == "pass"
    assert audit["episodes"] == 300
    assert audit["cohorts"] == 20
    assert audit["physical_episodes_by_method"] == {
        "racap_rs_evolved": 240,
        "rats_rs_evolved": 60,
    }

    broken = pd.DataFrame(candidates)
    broken.loc[
        (broken.method == "racap_rs_evolved") & (broken.cohort_index == 1),
        "promoted",
    ] = False
    failed = _robosuite_development_audit(
        pd.DataFrame(episodes),
        broken,
        racap_root=racap_root,
        rats_root=rats_root,
    )
    assert failed["status"] == "fail"
    assert any("promotion_rule_violation" in error for error in failed["errors"])


def test_robosuite_transfer_statistics_cover_five_methods_and_heldout_scope() -> None:
    methods = (
        "capx",
        "rats_90",
        "racap_phase2_zero_shot",
        "rats_rs_evolved",
        "racap_rs_evolved",
    )
    tasks = (
        "cube_lifting",
        "cube_restack",
        "cube_stack",
        "nut_assembly",
        "spill_wipe",
        "two_arm_handover",
        "two_arm_lift",
    )
    heldout = {"cube_restack", "two_arm_handover"}
    rows = []
    for method_index, method in enumerate(methods):
        for task in tasks:
            for seed in range(5):
                rows.append(
                    {
                        "method": method,
                        "task": task,
                        "seed": seed,
                        "episode_key": f"{task}/seed{seed}",
                        "native_success": seed < method_index,
                        "wall_seconds": 10.0,
                        "turns": 2,
                        "model_calls": 3,
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "estimated_api_cost_usd": 0.1,
                        "simulator_steps": 400,
                        "transfer_scope": (
                            "heldout_task_type"
                            if task in heldout
                            else "seen_task_type_new_seed"
                        ),
                    }
                )
    episodes = pd.DataFrame(rows)

    aggregate = _robosuite_transfer_aggregate(episodes)
    paired = _robosuite_transfer_paired(episodes)

    assert len(episodes) == 175
    assert len(aggregate) == 50
    assert set(aggregate.scope) == {
        "ALL",
        *tasks,
        "seen_task_type_new_seed",
        "heldout_task_type",
    }
    assert len(paired) == 30
    assert set(paired.scope) == {
        "ALL",
        "seen_task_type_new_seed",
        "heldout_task_type",
    }
    assert "mean_prompt_tokens" in aggregate
    assert "mean_completion_tokens" in aggregate
    assert "right_minus_left_task_bootstrap_low" in paired
    assert "wall_seconds_right_minus_left_bootstrap_high" in paired
    assert set(paired.bootstrap_unit) == {"robosuite_task"}


def test_robosuite_transfer_schema_defaults_do_not_make_generated_rows_unscorable() -> None:
    episodes = pd.DataFrame(
        [
            {
                "method": "capx",
                "scorable": None,
                "infrastructure_error": None,
            },
            {
                "method": "racap_rs_evolved",
                "scorable": False,
                "infrastructure_error": True,
            },
        ]
    )

    normalised = _robosuite_normalise_episode_attestations(episodes)

    assert normalised.scorable.tolist() == [True, False]
    assert normalised.infrastructure_error.tolist() == [False, True]


def test_robosuite_transfer_rejects_exhausted_generated_transport_retry(
    tmp_path: Path,
) -> None:
    method = tmp_path / "capx"
    method.mkdir()
    (method / "executor_source_audit.json").write_text(
        json.dumps(
            {
                "rats_root": "/registered/rats",
                "git_commit": "abc123",
                "git_status_sha256": "status-sha",
                "capx_python_source_sha256": "capx-sha",
                "rats_first_party_python_source_sha256": "rats-sha",
            }
        ),
        encoding="utf-8",
    )
    (method / "transport_protocol_audit.json").write_text(
        json.dumps(
            {
                "hook_installed": True,
                "maximum_attempts": 3,
                "request_invariant": True,
                "environment_resets": 0,
            }
        ),
        encoding="utf-8",
    )
    for task in ROBOSUITE_TASKS:
        task_root = method / task
        task_root.mkdir()
        (task_root / "COMPLETE.json").write_text("{}", encoding="utf-8")
        (task_root / "EVIDENCE.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "missing": [],
                    "registered_reset_counts": {
                        f"robosuite/{task}/seed{seed}": 1 for seed in range(5)
                    },
                    "transport_retry": {
                        "exhaustions": int(task == "nut_assembly"),
                        "environment_resets": 0,
                    },
                    "artifact_cardinality": {
                        "human_traces": 5,
                        "trial_summaries": 5,
                        "videos": 10,
                    },
                }
            ),
            encoding="utf-8",
        )

    audit, errors = _robosuite_generated_protocol_audit(method)

    assert audit["complete_tasks"] == 6
    assert errors == ["nut_assembly:transport_retry_exhausted"]


def test_racap_robosuite_terminal_model_failure_is_archived_before_resume(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    failed = output / "raw" / "cube_lifting" / "seed0"
    failed.mkdir(parents=True)
    (failed / "llm_calls.jsonl").write_text(
        '{"event":"network_exception"}\n', encoding="utf-8"
    )
    (failed / "record.json").write_text(
        json.dumps(
            {
                "key": "robosuite/cube_lifting/seed0",
                "scorable": True,
                "infrastructure_error": False,
                "evaluator_llm_calls": {
                    "terminal_failures": 1,
                    "quota_failures": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _terminal_model_failure({"terminal_failures": 1})
    complete = _completed_racap_rs_episodes(output)
    assert complete == {}

    archived = _archive_invalid_racap_rs_episodes(output, complete)

    assert len(archived) == 1
    assert not failed.exists()
    marker = json.loads(
        (archived[0] / "NOT_SCORED.json").read_text(encoding="utf-8")
    )
    assert marker["formal_score_use"] is False
    assert marker["episode_key"] == "robosuite/cube_lifting/seed0"


def test_robosuite_source_preflight_rejects_uninitialized_checkout(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="source is not initialized"):
        _validate_robosuite_source(Path(sys.executable), tmp_path)


def test_robosuite_executor_path_overrides_shared_service_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.controlled_comparison.run_robosuite as module

    robosuite = tmp_path / "registered_robosuite"
    capx = tmp_path / "capx"
    rats = tmp_path / "rats"
    monkeypatch.setenv("PYTHONPATH", "/inherited/public/path")
    monkeypatch.setattr(
        module,
        "_service_process_env",
        lambda: {"PYTHONPATH": "/wrong/libero/robosuite", "SERVICE_MARKER": "kept"},
    )

    env = _robosuite_process_env(
        robosuite_root=robosuite, capx_root=capx, rats_root=rats
    )
    paths = env["PYTHONPATH"].split(os.pathsep)
    assert paths[:5] == [
        str(SINGLE_RESET_SHIM_ROOT.resolve()),
        str(TRANSPORT_RETRY_SHIM_ROOT.resolve()),
        str(robosuite.resolve()),
        str(capx.resolve()),
        str(rats.resolve()),
    ]
    assert paths[-1] == "/inherited/public/path"
    assert "/wrong/libero/robosuite" not in paths
    assert env["SERVICE_MARKER"] == "kept"
    assert env["RACAP_SINGLE_RESET_PROTOCOL"] == "1"
    assert env["CAPX_MAX_TRIAL_RETRIES"] == "1"
    assert env["RATS_MAX_TRIAL_RETRIES"] == "1"
    assert env["RACAP_REGISTERED_TRANSPORT_RETRY"] == "1"
    assert env["RACAP_TRANSPORT_MAX_ATTEMPTS"] == "3"


def test_single_reset_runner_audit_fails_closed_when_hook_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.controlled_comparison.run_robosuite as module

    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                "sitecustomize": str(
                    SINGLE_RESET_SHIM_ROOT / "sitecustomize.py"
                ),
                "hook_installed": False,
                "target_modules": ["capx.envs.runner", "rats.envs.runner"],
                "protocol_environment": "1",
            }
        )
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="not installed"):
        _validate_single_reset_runner(Path(sys.executable), {})


def test_single_reset_sitecustomize_lazily_patches_upstream_runner(
    tmp_path: Path,
) -> None:
    package = tmp_path / "capx" / "envs"
    package.mkdir(parents=True)
    (tmp_path / "capx" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runner.py").write_text(
        "MAX_TRIAL_RETRIES = 3\nORIGIN_UNCHANGED = True\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["RACAP_SINGLE_RESET_PROTOCOL"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SINGLE_RESET_SHIM_ROOT.resolve()), str(tmp_path.resolve())]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, capx.envs.runner as r; "
                "print(json.dumps({'retries': r.MAX_TRIAL_RETRIES, "
                "'origin': r.__file__, 'unchanged': r.ORIGIN_UNCHANGED}))"
            ),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["retries"] == 1
    assert Path(payload["origin"]).resolve() == (package / "runner.py").resolve()
    assert payload["unchanged"] is True


def test_registered_transport_usercustomize_retries_exact_request_without_reset(
    tmp_path: Path,
) -> None:
    fake_requests = tmp_path / "requests.py"
    fake_requests.write_text(
        """
CALLS = []
class Timeout(Exception):
    pass
class ConnectionError(Exception):
    pass
def post(*args, **kwargs):
    CALLS.append((args, kwargs))
    if len(CALLS) == 1:
        raise Timeout('temporary read timeout')
    return {'ok': True}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "transport_attempts.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "RACAP_SINGLE_RESET_PROTOCOL": "1",
            "RACAP_REGISTERED_TRANSPORT_RETRY": "1",
            "RACAP_REGISTERED_VAPI_BASE": "https://registered.vapi/v1",
            "RACAP_TRANSPORT_MAX_ATTEMPTS": "3",
            "RACAP_TRANSPORT_RETRY_LOG": str(log_path),
            "CAPX_EPISODE_KEY": "robosuite/cube_lifting/seed100",
            "PYTHONPATH": os.pathsep.join(
                [
                    str(SINGLE_RESET_SHIM_ROOT.resolve()),
                    str(TRANSPORT_RETRY_SHIM_ROOT.resolve()),
                    str(tmp_path.resolve()),
                ]
            ),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, requests, usercustomize; "
                "payload={'model':'gpt-5.5','messages':[{'role':'user','content':'same'}]}; "
                "response=requests.post('https://registered.vapi/v1/chat/completions', "
                "json=payload, timeout=200); "
                "print(json.dumps({'response':response,'calls':len(requests.CALLS),"
                "'same_object':requests.CALLS[0][1]['json'] is requests.CALLS[1][1]['json'],"
                "'installed':usercustomize.RACAP_REGISTERED_TRANSPORT_RETRY_INSTALLED}))"
            ),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload == {
        "response": {"ok": True},
        "calls": 2,
        "same_object": True,
        "installed": True,
    }
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["outcome"] for record in records] == [
        "transport_error",
        "recovered",
    ]
    assert records[0]["request_body_sha256"] == records[1]["request_body_sha256"]
    assert all(record["environment_reset"] is False for record in records)
    assert all(record["prompt_or_policy_changed"] is False for record in records)


def test_registered_transport_retry_audit_fails_closed_when_hook_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.controlled_comparison.run_robosuite as module

    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                "usercustomize": str(
                    TRANSPORT_RETRY_SHIM_ROOT / "usercustomize.py"
                ),
                "hook_installed": False,
                "maximum_attempts": 3,
            }
        )
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="not installed"):
        _validate_registered_transport_retry(Path(sys.executable), {})


def test_rats_extraction_transport_retry_preserves_request_and_recovers(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    class FakeTimeout(Exception):
        pass

    class FakeConnectionError(Exception):
        pass

    class FakeProviderAbort(BaseException):
        pass

    request_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    response = object()

    def post(*args: object, **kwargs: object) -> object:
        request_calls.append((args, kwargs))
        if len(request_calls) == 1:
            raise FakeTimeout("temporary read timeout")
        return response

    requests_module = SimpleNamespace(
        post=post,
        Timeout=FakeTimeout,
        ConnectionError=FakeConnectionError,
    )
    base_agent = SimpleNamespace(
        requests=requests_module,
        RATSProviderAbort=FakeProviderAbort,
    )
    log_path = tmp_path / "transport_attempts.jsonl"
    abort_path = tmp_path / "TRANSPORT_ABORTED.json"
    request_json = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "same prompt"}],
        "temperature": 0.0,
    }

    _install_registered_transport_retry(
        base_agent,
        log_path=log_path,
        abort_path=abort_path,
        maximum_attempts=3,
    )
    actual = base_agent.requests.post(
        "https://registered.vapi/chat/completions",
        json=request_json,
        timeout=200,
    )

    assert actual is response
    assert len(request_calls) == 2
    assert request_calls[0] == request_calls[1]
    assert request_calls[0][1]["json"] is request_json
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["outcome"] for record in records] == [
        "transport_error",
        "recovered",
    ]
    assert all(record["prompt_or_policy_changed"] is False for record in records)
    assert not abort_path.exists()


def test_rats_extraction_transport_retry_exhaustion_is_not_scored(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    class FakeTimeout(Exception):
        pass

    class FakeConnectionError(Exception):
        pass

    class FakeProviderAbort(BaseException):
        pass

    calls = 0

    def post(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise FakeConnectionError("provider connection unavailable")

    base_agent = SimpleNamespace(
        requests=SimpleNamespace(
            post=post,
            Timeout=FakeTimeout,
            ConnectionError=FakeConnectionError,
        ),
        RATSProviderAbort=FakeProviderAbort,
    )
    log_path = tmp_path / "transport_attempts.jsonl"
    abort_path = tmp_path / "TRANSPORT_ABORTED.json"

    _install_registered_transport_retry(
        base_agent,
        log_path=log_path,
        abort_path=abort_path,
        maximum_attempts=3,
    )
    with pytest.raises(FakeProviderAbort):
        base_agent.requests.post(
            "https://registered.vapi/chat/completions",
            json={"model": "gpt-5.5", "messages": []},
            timeout=200,
        )

    assert calls == 3
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 3
    assert all(record["outcome"] == "transport_error" for record in records)
    abort = json.loads(abort_path.read_text())
    assert abort["status"] == "infrastructure_invalid"
    assert abort["strategy_outputs_scored"] is False


def test_robosuite_rats_delta_starts_from_common_clean_template(
    tmp_path: Path,
) -> None:
    library = tmp_path / "skills.json"
    library.write_text("[]", encoding="utf-8")
    template = {
        "trials": 10,
        "num_workers": 9,
        "output_dir": "old",
        "env": {"cfg": {"task": "same"}},
        "api_servers": [{"_target_": "example.should_not_start_per_task"}],
    }

    capx = _configure_robosuite_eval(
        template,
        method="capx",
        output_dir=tmp_path / "capx",
        model="gpt-5.5",
        rats_library=None,
        workers=10,
    )
    rats = _configure_robosuite_eval(
        template,
        method="rats_90",
        output_dir=tmp_path / "rats",
        model="gpt-5.5",
        rats_library=library,
        workers=10,
    )

    assert capx["env"]["cfg"] == {"task": "same"}
    assert rats["env"]["cfg"]["task"] == "same"
    assert rats["env"]["cfg"]["external_skill_library_path"] == str(
        library.resolve()
    )
    assert rats["env"]["cfg"]["external_skill_planner_model"] == "gpt-5.5"
    assert rats["env"]["cfg"]["external_skill_library_mode"] == "planner"
    assert rats["env"]["cfg"]["external_skill_planner_max_selected"] == 6
    assert capx["trials"] == rats["trials"] == 5
    assert capx["num_workers"] == rats["num_workers"] == 5
    assert capx["api_servers"] == rats["api_servers"] == []
    assert template["trials"] == 10
    assert template["api_servers"]


def test_robosuite_evolved_rats_forces_nonprivileged_runtime(
    tmp_path: Path,
) -> None:
    library = tmp_path / "skills.json"
    library.write_text("[]", encoding="utf-8")
    template = {
        "env": {"cfg": {"task": "nut", "privileged": True}},
        "api_servers": [],
    }
    resolved = _configure_robosuite_eval(
        template,
        method="rats_rs_evolved",
        output_dir=tmp_path / "candidate",
        model="gpt-5.5",
        rats_library=library,
        workers=10,
        trials=3,
    )

    assert resolved["env"]["cfg"]["privileged"] is False
    assert resolved["trials"] == resolved["num_workers"] == 3
    assert template["env"]["cfg"]["privileged"] is True


def test_robosuite_resolved_config_audit_allows_only_registered_rats_delta(
    tmp_path: Path,
) -> None:
    library = tmp_path / "skills.json"
    library.write_text("[]", encoding="utf-8")
    template = {
        "env": {"cfg": {"task": "same"}},
        "api_servers": [{"_target_": "common.service"}],
    }
    for task in ROBOSUITE_TASKS:
        capx = _configure_robosuite_eval(
            template,
            method="capx",
            output_dir=tmp_path / "capx" / task / "artifacts",
            model="gpt-5.5",
            rats_library=None,
            workers=10,
        )
        rats = _configure_robosuite_eval(
            template,
            method="rats_90",
            output_dir=tmp_path / "rats_90" / task / "artifacts",
            model="gpt-5.5",
            rats_library=library,
            workers=10,
        )
        for method, config in (("capx", capx), ("rats_90", rats)):
            path = tmp_path / method / task / "config.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config), encoding="utf-8")

    matched, rows, errors = _resolved_config_audit(tmp_path)
    assert matched
    assert len(rows) == len(ROBOSUITE_TASKS)
    assert errors == []

    changed = tmp_path / "rats_90" / ROBOSUITE_TASKS[0] / "config.yaml"
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload["unregistered_behavior_change"] = True
    changed.write_text(json.dumps(payload), encoding="utf-8")
    matched, _, errors = _resolved_config_audit(tmp_path)
    assert not matched
    assert any(error.startswith("unregistered_resolved_config_delta") for error in errors)


def test_robosuite_model_and_quota_audit_is_fail_closed() -> None:
    models, complete, quota = _model_and_quota_audit(
        [
            {
                "event": "network_success",
                "actual_model": "gpt-5.5",
                "message": "ok",
            },
            {
                "event": "network_failure",
                "error": "insufficient credit balance",
            },
        ]
    )
    assert models == ["gpt-5.5"]
    assert complete is True
    assert quota == 1


def test_robosuite_physical_api_count_excludes_perception_and_ik() -> None:
    assert _is_physical_api_boundary(
        {"event": "native_state_after_api", "api_name": "move_to_joints_arm0"}
    )
    assert _is_physical_api_boundary(
        {"event": "native_state_after_api", "api_name": "close_gripper"}
    )
    assert not _is_physical_api_boundary(
        {"event": "native_state_after_api", "api_name": "solve_ik"}
    )
    assert not _is_physical_api_boundary(
        {"event": "native_state_after_api", "api_name": "get_observation"}
    )

    models, complete, quota = _model_and_quota_audit(
        [{"event": "network_success", "actual_model": ""}]
    )
    assert models == []
    assert complete is False
    assert quota == 0


def test_robosuite_completion_requires_five_keyed_trials(tmp_path: Path) -> None:
    _write_evidence(tmp_path / "llm_calls.jsonl", "")
    _write_evidence(tmp_path / "native_states.jsonl", "")
    _write_evidence(tmp_path / "sim_episodes.jsonl", "")
    for seed in range(5):
        key = f"robosuite/cube_lifting/seed{seed}"
        for stem in ("llm_calls.jsonl", "native_states.jsonl"):
            with (tmp_path / stem).open("a", encoding="utf-8") as handle:
                handle.write(f'{{"episode_key":"{key}"}}\n')
        with (tmp_path / "sim_episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "environment_reset",
                        "episode_key": key,
                        "seed": seed,
                    }
                )
                + "\n"
            )
        trial = tmp_path / "artifacts" / f"trial_{seed + 1:02d}"
        _write_evidence(trial / "all_responses.json")
        _write_evidence(trial / "summary.txt")
        _write_evidence(trial / "video_combined.mp4")
    _write_evidence(tmp_path / "artifacts" / "summaries.txt")
    _write_evidence(tmp_path / "artifacts" / "aaa_done_flag" / "aaa_done_flag.txt")

    evidence = _robosuite_task_evidence(tmp_path, "cube_lifting")

    assert evidence["complete"] is True
    assert evidence["artifact_cardinality"] == {
        "human_traces": 5,
        "trial_summaries": 5,
        "videos": 5,
    }


def test_robosuite_completion_accepts_registered_development_seeds(
    tmp_path: Path,
) -> None:
    seeds = (100, 101, 102)
    for stem in ("llm_calls.jsonl", "native_states.jsonl", "sim_episodes.jsonl"):
        _write_evidence(tmp_path / stem, "")
    for trial_index, seed in enumerate(seeds, start=1):
        key = f"robosuite/cube_stack/seed{seed}"
        for stem in ("llm_calls.jsonl", "native_states.jsonl"):
            with (tmp_path / stem).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"episode_key": key}) + "\n")
        with (tmp_path / "sim_episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": "environment_reset", "episode_key": key, "seed": seed}
                )
                + "\n"
            )
        trial = tmp_path / "artifacts" / f"trial_{trial_index:02d}"
        for artifact in ("all_responses.json", "summary.txt", "video_combined.mp4"):
            _write_evidence(trial / artifact)
    _write_evidence(tmp_path / "artifacts" / "summaries.txt")
    _write_evidence(tmp_path / "artifacts" / "aaa_done_flag" / "aaa_done_flag.txt")

    evidence = _robosuite_task_evidence(
        tmp_path,
        "cube_stack",
        seeds=seeds,
    )

    assert evidence["complete"] is True
    assert evidence["expected_episode_keys"] == [
        "robosuite/cube_stack/seed100",
        "robosuite/cube_stack/seed101",
        "robosuite/cube_stack/seed102",
    ]


def test_robosuite_completion_rejects_duplicate_or_mispaired_reset(
    tmp_path: Path,
) -> None:
    _write_evidence(tmp_path / "llm_calls.jsonl", "")
    _write_evidence(tmp_path / "native_states.jsonl", "")
    _write_evidence(tmp_path / "sim_episodes.jsonl", "")
    for seed in range(5):
        key = f"robosuite/cube_lifting/seed{seed}"
        for stem in ("llm_calls.jsonl", "native_states.jsonl"):
            with (tmp_path / stem).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"episode_key": key}) + "\n")
        with (tmp_path / "sim_episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "environment_reset",
                        "episode_key": key,
                        "seed": 99 if seed == 2 else seed,
                    }
                )
                + "\n"
            )
            if seed == 3:
                handle.write(
                    json.dumps(
                        {
                            "event": "environment_reset",
                            "episode_key": key,
                            "seed": seed,
                        }
                    )
                    + "\n"
                )
        trial = tmp_path / "artifacts" / f"trial_{seed + 1:02d}"
        _write_evidence(trial / "all_responses.json")
        _write_evidence(trial / "summary.txt")
        _write_evidence(trial / "video_combined.mp4")
    _write_evidence(tmp_path / "artifacts" / "summaries.txt")
    _write_evidence(tmp_path / "artifacts" / "aaa_done_flag" / "aaa_done_flag.txt")

    evidence = _robosuite_task_evidence(tmp_path, "cube_lifting")

    assert evidence["complete"] is False
    assert "wrong_or_duplicate_reset_seed" in evidence["missing"]
    assert any(
        item.startswith("excess_registered_resets:robosuite/cube_lifting/seed3")
        for item in evidence["missing"]
    )
    assert evidence["wrong_reset_seeds"]["robosuite/cube_lifting/seed2"] == [99]
    assert evidence["wrong_reset_seeds"]["robosuite/cube_lifting/seed3"] == [3, 3]


def test_robosuite_completion_rejects_exhausted_registered_transport_retry(
    tmp_path: Path,
) -> None:
    seed = 100
    key = f"robosuite/cube_lifting/seed{seed}"
    for stem in ("llm_calls.jsonl", "native_states.jsonl"):
        _write_evidence(tmp_path / stem, json.dumps({"episode_key": key}) + "\n")
    _write_evidence(
        tmp_path / "sim_episodes.jsonl",
        json.dumps({"event": "environment_reset", "episode_key": key, "seed": seed})
        + "\n",
    )
    _write_evidence(
        tmp_path / "transport_attempts.jsonl",
        json.dumps(
            {
                "event": "registered_vapi_transport_attempt",
                "episode_key": key,
                "attempt": 3,
                "maximum_attempts": 3,
                "outcome": "exhausted",
                "environment_reset": False,
            }
        )
        + "\n",
    )
    trial = tmp_path / "artifacts" / "trial_01"
    for artifact in ("all_responses.json", "summary.txt", "video_combined.mp4"):
        _write_evidence(trial / artifact)
    _write_evidence(tmp_path / "artifacts" / "summaries.txt")
    _write_evidence(tmp_path / "artifacts" / "aaa_done_flag" / "aaa_done_flag.txt")

    evidence = _robosuite_task_evidence(
        tmp_path,
        "cube_lifting",
        seeds=(seed,),
    )

    assert evidence["complete"] is False
    assert "registered_vapi_transport_retry_exhausted" in evidence["missing"]
    assert evidence["transport_retry"] == {
        "records": 1,
        "transient_failures": 0,
        "recoveries": 0,
        "exhaustions": 1,
        "environment_resets": 0,
    }


def test_post_development_supervisor_pid_probe() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert _pid_alive(process.pid)
    finally:
        process.terminate()
        process.wait(timeout=5)
    assert not _pid_alive(process.pid)


def test_one_shot_aggregate_has_family_and_overall_rows() -> None:
    paired = pd.DataFrame(
        [
            {
                "method": "capx",
                "family": "spatial",
                "suite": "libero_spatial_swap",
                "task_id": 0,
                "native_success_zero": False,
                "native_success_one_trial": True,
                "policy_wall_seconds_zero": 10.0,
                "policy_wall_seconds_one_trial": 9.0,
                "model_calls_zero": 2,
                "model_calls_one_trial": 2,
                "estimated_api_cost_usd_zero": 0.1,
                "estimated_api_cost_usd_one_trial": 0.08,
            },
            {
                "method": "capx",
                "family": "goal",
                "suite": "libero_goal_swap",
                "task_id": 0,
                "native_success_zero": True,
                "native_success_one_trial": True,
                "policy_wall_seconds_zero": 12.0,
                "policy_wall_seconds_one_trial": 11.0,
                "model_calls_zero": 3,
                "model_calls_one_trial": 2,
                "estimated_api_cost_usd_zero": 0.2,
                "estimated_api_cost_usd_one_trial": 0.1,
            },
        ]
    )

    summary = _one_shot_aggregate(paired)
    overall = summary[summary.family == "all"].iloc[0]

    assert set(summary.family) == {"spatial", "goal", "all"}
    assert overall.zero_shot_successes == 1
    assert overall.one_trial_successes == 2
    assert overall.absolute_gain == 0.5
    assert overall.paired_task_clusters == 2
    assert 0 <= overall.mcnemar_holm_p <= 1


def test_one_shot_delivery_audit_binds_all_method_family_cards(
    tmp_path: Path,
) -> None:
    rows = []
    for method in one_shot_cards.METHODS:
        delivered = {}
        for family in one_shot_cards.FAMILY_SUITE:
            card = tmp_path / "cards" / f"{method}_{family}.md"
            card.parent.mkdir(parents=True, exist_ok=True)
            card.write_text(f"{method}/{family}\n", encoding="utf-8")
            digest = one_shot_cards._sha256(card)
            delivered[family] = {"path": str(card.resolve()), "sha256": digest}
            rows.append(
                {
                    "method": method,
                    "family": family,
                    "card": str(card.resolve()),
                    "card_sha256": digest,
                }
            )
        root = tmp_path / "measured" / method
        root.mkdir(parents=True)
        (root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "cohorts": ["libero_pro_one_shot"],
                    "experience_cards": delivered,
                }
            ),
            encoding="utf-8",
        )

    frame, audit = _test_card_delivery_audit(
        tmp_path / "measured", pd.DataFrame(rows)
    )

    assert audit["status"] == "pass"
    assert len(frame) == 15
    assert frame.path_match.all()
    assert frame.hash_match.all()


def test_one_shot_prompt_audit_decodes_actual_retained_requests(
    tmp_path: Path,
) -> None:
    card_rows = []
    card_text: dict[tuple[str, str], str] = {}
    episode_rows = []
    for method in one_shot_cards.METHODS:
        method_root = tmp_path / method
        method_root.mkdir(parents=True)
        for family in one_shot_cards.FAMILY_SUITE:
            card = tmp_path / "cards" / f"{method}_{family}.md"
            card.parent.mkdir(parents=True, exist_ok=True)
            text = f"## Observed mechanism\n{method}/{family} exact card"
            card.write_text(text, encoding="utf-8")
            card_text[(method, family)] = text
            card_rows.append({"method": method, "family": family, "card": str(card)})

        if method == "capx":
            prompt = method_root / "gpt-5.5" / "artifacts" / "initial_prompt.txt"
            prompt.parent.mkdir(parents=True)
            prompt.write_text(
                repr(
                    [
                        {
                            "role": "system",
                            "content": "\n".join(
                                card_text[(method, family)]
                                for family in one_shot_cards.FAMILY_SUITE
                            ),
                        }
                    ]
                ),
                encoding="utf-8",
            )
        elif method in {"rats_base", "rats_90"}:
            prompt = method_root / "agent_io" / "0001.json"
            prompt.parent.mkdir(parents=True)
            prompt.write_text(
                json.dumps(
                    {
                        "request": {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "\n".join(
                                        card_text[(method, family)]
                                        for family in one_shot_cards.FAMILY_SUITE
                                    ),
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

        racap_calls = []
        for family in one_shot_cards.FAMILY_SUITE:
            for task_id in range(20):
                suite = f"libero_{family}_{'swap' if task_id < 10 else 'task'}"
                episode_key = f"{suite}/{task_id % 10}/seed1"
                episode_rows.append(
                    {
                        "method": method,
                        "suite": suite,
                        "task_id": task_id % 10,
                        "seed": 1,
                        "episode_key": episode_key,
                        "artifact_path": str(method_root),
                    }
                )
                if method.startswith("racap_"):
                    racap_calls.append(
                        {
                            "episode_key": episode_key,
                            "request": {
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": card_text[(method, family)],
                                    }
                                ]
                            },
                        }
                    )
        if racap_calls:
            (method_root / "llm_calls.worker0.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in racap_calls),
                encoding="utf-8",
            )

    frame, audit = _test_card_prompt_audit(
        pd.DataFrame(episode_rows), pd.DataFrame(card_rows)
    )

    assert audit["status"] == "pass"
    assert len(frame) == 300
    assert frame.prompt_card_found.all()


def test_one_shot_prompt_audit_resolves_rats_iteration_artifact(
    tmp_path: Path,
) -> None:
    card = tmp_path / "rats_90_spatial.md"
    card.write_text("exact one-trial experience card", encoding="utf-8")
    episode = tmp_path / "episode"
    artifact = episode / "artifacts" / "iteration_001.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    prompt = episode / "agent_io" / "0001.json"
    prompt.parent.mkdir(parents=True)
    prompt.write_text(
        json.dumps(
            {
                "request": {
                    "messages": [
                        {"role": "user", "content": card.read_text(encoding="utf-8")}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    frame, audit = _test_card_prompt_audit(
        pd.DataFrame(
            [
                {
                    "method": "rats_90",
                    "suite": "libero_spatial_swap",
                    "task_id": 0,
                    "seed": 1,
                    "episode_key": "libero_spatial_swap/0/seed1",
                    "artifact_path": str(artifact),
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "method": "rats_90",
                    "family": "spatial",
                    "card": str(card),
                }
            ]
        ),
    )

    # The aggregate audit is intentionally incomplete for this one-row fixture;
    # the row-level request verification must nevertheless succeed.
    assert audit["status"] == "error"
    assert len(frame) == 1
    assert bool(frame.iloc[0].prompt_card_found)
    assert Path(frame.iloc[0].artifact_path) == episode


def test_racap_prompt_audit_accepts_text_free_card_hash_attestation(
    tmp_path: Path,
) -> None:
    card = "transferable one-trial lesson"
    group = tmp_path / "group"
    group.mkdir()
    (group / "llm_calls.worker0.jsonl").write_text(
        json.dumps(
            {
                "episode_key": "libero_spatial_swap/0/seed1",
                "one_shot_card_in_request": True,
                "one_shot_card_sha256": hashlib.sha256(card.encode()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    matched, inspected = _racap_prompt_card_index(group, card)

    assert inspected == 1
    assert matched == {"libero_spatial_swap/0/seed1"}


def test_one_shot_adaptation_artifact_audit_requires_complete_grid(
    tmp_path: Path,
) -> None:
    calibration_rows = []
    card_rows = []
    call_rows = []
    for method in one_shot_cards.METHODS:
        for family, suite in one_shot_cards.FAMILY_SUITE.items():
            card = tmp_path / f"{method}_{family}.md"
            card.write_text(f"{method}/{family}\n", encoding="utf-8")
            calibration_rows.append({"method": method, "suite": suite})
            card_rows.append(
                {"method": method, "family": family, "card": str(card)}
            )
            call_rows.append(
                {
                    "method": method,
                    "family": family,
                    "actual_models": "gpt-5.5",
                }
            )

    audit = _adaptation_artifact_audit(
        pd.DataFrame(calibration_rows),
        pd.DataFrame(card_rows),
        pd.DataFrame(call_rows),
    )
    assert audit["status"] == "pass"
    assert audit["calibration_rows"] == 15
    assert audit["card_rows"] == 15
    assert audit["card_call_rows"] == 15

    incomplete = _adaptation_artifact_audit(
        pd.DataFrame(calibration_rows[:-1]),
        pd.DataFrame(card_rows),
        pd.DataFrame(call_rows),
    )
    assert incomplete["status"] == "error"
    assert any("calibration grid mismatch" in item for item in incomplete["errors"])


def test_development_call_windows_start_at_each_task_proposal() -> None:
    calls = [
        {"timestamp": 1.0, "caller": "task_proposer._propose_catalog"},
        {"timestamp": 2.0, "caller": "planner.plan"},
        {"timestamp": 8.0, "caller": "task_proposer._propose_catalog"},
        {"timestamp": 9.0, "caller": "policy_writer._generate_once"},
    ]

    assert _iteration_windows(calls) == [(1.0, 8.0), (8.0, float("inf"))]


def test_committed_windows_exclude_rolled_back_proposal(tmp_path: Path) -> None:
    def proposer(timestamp: float, task: str) -> dict:
        return {
            "timestamp": timestamp,
            "caller": "task_proposer._propose_catalog",
            "response": {"content": json.dumps({"selected_task": task})},
        }

    calls = [
        proposer(10.0, "libero_90_task9"),
        proposer(20.0, "libero_90_task46"),
        proposer(30.0, "libero_90_task47"),
        proposer(60.0, "libero_90_task71"),
        proposer(70.0, "libero_90_task2"),
    ]
    for iteration, task in enumerate((9, 46, 71), start=1):
        (tmp_path / f"iteration_{iteration:03d}.json").write_text(
            json.dumps(
                {
                    "iteration": iteration,
                    "task_proposal": {
                        "activity_name": f"libero_90_task{task}"
                    },
                }
            ),
            encoding="utf-8",
        )
    (tmp_path / "resume_restorations.jsonl").write_text(
        json.dumps(
            {
                "time": 50.0,
                "latest_completed_iteration": 2,
                "restored": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _committed_iteration_windows(tmp_path, calls) == {
        1: (10.0, 20.0),
        2: (20.0, 30.0),
        3: (60.0, 70.0),
    }


def test_development_observed_usage_includes_unfinished_iteration(tmp_path: Path) -> None:
    agent_io = tmp_path / "agent_io"
    agent_io.mkdir()
    (agent_io / "0001.json").write_text(
        json.dumps(
            {
                "timestamp": 1.0,
                "caller": "task_proposer._propose_catalog",
                "model": "gpt-5.5",
                "actual_model": "gpt-5.5",
                "elapsed_s": 2.5,
                "response": {
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 3},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "sim_episodes.jsonl").write_text(
        '{"event":"environment_reset","time":1.5}\n', encoding="utf-8"
    )

    observed = _rats_observed_usage(tmp_path).iloc[0]

    assert observed.completed_iterations == 0
    assert observed.iterations_started == 1
    assert observed.simulator_resets_observed == 1
    assert observed.model_calls == 1
    assert observed.prompt_tokens == 20
    assert observed.completion_tokens == 5
    assert observed.cached_tokens == 3
    assert observed.model_latency_seconds == 2.5
    assert observed.actual_models == "gpt-5.5"


def test_invalidated_rats_lineages_remain_in_engineering_resource_total(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid"
    lineage = invalid / "rats90_selfplay_bad_route"
    agent_io = lineage / "agent_io"
    agent_io.mkdir(parents=True)
    (lineage / "INVALIDATED.json").write_text(
        '{"reason":"mixed_internal_model_route"}', encoding="utf-8"
    )
    (lineage / "iteration_001.json").write_text(
        '{"iteration":1}', encoding="utf-8"
    )
    (lineage / "sim_episodes.jsonl").write_text(
        '{"event":"environment_reset"}\n{"event":"environment_reset"}\n',
        encoding="utf-8",
    )
    (agent_io / "0001.json").write_text(
        json.dumps(
            {
                "timestamp": 1.0,
                "actual_model": "gpt-5.5",
                "elapsed_s": 2.0,
                "response": {
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 3},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    # Frozen copies must not be counted a second time.
    (invalid / "rats90_frozen_bad_route").mkdir()

    discarded = _rats_invalidated_usage(invalid)

    assert len(discarded) == 1
    row = discarded.iloc[0]
    assert row.lineage == "rats90_selfplay_bad_route"
    assert row.invalidation_reason == "mixed_internal_model_route"
    assert row.completed_iterations == 1
    assert row.simulator_resets_observed == 2
    assert row.model_calls == 1
    assert row.prompt_tokens == 20
    assert row.completion_tokens == 5
    assert row.cached_tokens == 3
    assert bool(row.counted_in_engineering_total)
    assert not bool(row.eligible_for_success_or_skill_metrics)

    committed = pd.DataFrame(
        [
            {
                "simulator_resets": 4,
                "model_calls": 10,
                "model_latency_seconds": 8.0,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_tokens": 10,
                "estimated_api_cost_usd": 1.0,
                "actual_models": "gpt-5.5",
            }
        ]
    )
    active = pd.DataFrame(
        [
            {
                "simulator_resets_observed": 6,
                "model_calls": 12,
                "model_latency_seconds": 9.0,
                "prompt_tokens": 120,
                "completion_tokens": 25,
                "cached_tokens": 11,
                "estimated_api_cost_usd": 1.2,
                "actual_models": "gpt-5.5",
            }
        ]
    )
    accounting = _rats_resource_accounting(committed, active, discarded)
    total = accounting[
        accounting.scope == "all_recorded_engineering_consumption"
    ].iloc[0]
    assert total.simulator_resets == 8
    assert total.model_calls == 13
    assert total.prompt_tokens == 140
    assert total.completion_tokens == 30
    assert total.cached_tokens == 14
    assert total.model_latency_seconds == 11.0


def test_development_separates_task_code_from_reusable_skills() -> None:
    raw = {
        "code_attempt_0": "print('failed')",
        "verification_attempt_0": {
            "success": False,
            "details": {"native_predicate_success": False},
        },
        "code_attempt_1": "pixel_seed = (252, 419)\nRESULT = {'success': True}",
        "verification_attempt_1": {
            "success": True,
            "details": {"native_predicate_success": True},
        },
        "skills_added": ["generic_grasp"],
    }

    metadata = _successful_code_metadata(raw)

    assert metadata["task_specific_success_code_cached"] is True
    assert metadata["task_specific_success_code_index"] == 1
    assert metadata["task_specific_success_code_chars"] == len(raw["code_attempt_1"])
    assert len(metadata["task_specific_success_code_sha256"]) == 64


def test_development_native_audit_corrects_stale_rebind_key(tmp_path: Path) -> None:
    agent_io = tmp_path / "agent_io"
    agent_io.mkdir()
    (agent_io / "0001_task_proposer.json").write_text(
        '{"timestamp": 10.0, "caller": "task_proposer._propose_catalog"}',
        encoding="utf-8",
    )
    (tmp_path / "iteration_001.json").write_text(
        '{"iteration": 1, "task_proposal": '
        '{"activity_name": "libero_90_task12"}}',
        encoding="utf-8",
    )
    (tmp_path / "native_states.jsonl").write_text(
        '{"time": 12.0, "event": "native_state_after_api", '
        '"episode_key": "libero_90/9/seed1", "simulator_steps": 40, '
        '"native_success": false}\n',
        encoding="utf-8",
    )

    audit = _native_event_audit(tmp_path)

    assert len(audit) == 1
    assert audit.iloc[0].registered_task_id == 12
    assert audit.iloc[0].effective_task_id == 12
    assert not bool(audit.iloc[0].raw_key_task_matches)
    assert audit.iloc[0].identity_source == "proposal_time_window"
    assert not bool(audit.iloc[0].expected_rebind_label_lag)


def test_development_native_audit_marks_first_simulator_bound_rebind_lag(
    tmp_path: Path,
) -> None:
    agent_io = tmp_path / "agent_io"
    agent_io.mkdir()
    (agent_io / "0001_task_proposer.json").write_text(
        '{"timestamp": 10.0, "caller": "task_proposer._propose_catalog"}',
        encoding="utf-8",
    )
    (tmp_path / "iteration_001.json").write_text(
        '{"iteration": 1, "task_proposal": '
        '{"activity_name": "libero_90_task12"}}',
        encoding="utf-8",
    )
    (tmp_path / "native_states.jsonl").write_text(
        '{"time": 12.0, "event": "native_state_after_code", '
        '"episode_key": "libero_90/9/seed1", "simulator_task_id": 12, '
        '"simulator_steps": 10, "native_success": false}\n'
        '{"time": 13.0, "event": "native_state_after_api", '
        '"episode_key": "libero_90/12/seed1", "simulator_task_id": 12, '
        '"simulator_steps": 20, "native_success": false}\n',
        encoding="utf-8",
    )

    audit = _native_event_audit(tmp_path)

    assert bool(audit.iloc[0].first_native_event_in_iteration)
    assert bool(audit.iloc[0].expected_rebind_label_lag)
    assert not bool(audit.iloc[1].first_native_event_in_iteration)
    assert bool(audit.iloc[1].raw_key_task_matches)


def _identity_call(label: str, verified: bool) -> dict:
    return {
        "caller": "libero_reduced_skill_library.verify_object_identity",
        "request": {
            "messages": [
                {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Robot intends to grasp: '{label}'.",
                        }
                    ]
                }
            ]
        },
        "response": {"content": '{"verified": ' + str(verified).lower() + "}"},
    }


def test_identity_search_counts_calls_after_first_accepted_candidate() -> None:
    calls = [
        _identity_call("black bowl", False),
        _identity_call("black bowl", True),
        _identity_call("black bowl", False),
        _identity_call("black bowl", True),
        {"caller": "multi_turn_decider.decide"},
        _identity_call("plate", True),
        _identity_call("plate", False),
    ]

    runs = _identity_search_runs(calls)

    assert list(runs.expected_object) == ["black bowl", "plate"]
    assert list(runs.calls) == [4, 2]
    assert list(runs.first_true_call) == [2, 1]
    assert list(runs.calls_after_first_true) == [2, 1]


def test_in_progress_rats_artifact_recovers_error_without_private_predicates(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "verifier_artifacts"
    artifacts.mkdir()
    stem = "iter002_attempt02_turn04_step14"
    (artifacts / f"{stem}.json").write_text(
        """{
          "final_success": false,
          "task": {"activity_name": "libero_90_task12"},
          "evidence": {
            "native_predicate_success": false,
            "execution_success": false,
            "reward": 0.0,
            "private_native_audit": {
              "predicate_status": [{"predicate": "[on secret bowl]"}]
            }
          }
        }""",
        encoding="utf-8",
    )
    (tmp_path / "run.log").write_text(
        "Step 5: Execution\n"
        "Traceback (most recent call last):\n"
        "TimeoutError: Execution timed out after 180s\n"
        f"Verifier artifacts saved: /tmp/{stem}.json\n",
        encoding="utf-8",
    )

    rows = _in_progress_attempt_diagnostics(tmp_path, set())

    assert len(rows) == 1
    assert rows[0]["record_status"] == "in_progress_artifact"
    assert rows[0]["execution_timeout"] is True
    assert rows[0]["runtime_error_type"] == "execution_timeout"
    assert "secret" not in str(rows[0])


def test_completed_iteration_suppresses_provisional_attempt_rows(tmp_path: Path) -> None:
    artifacts = tmp_path / "verifier_artifacts"
    artifacts.mkdir()
    (artifacts / "iter001_attempt00_turn00_step00.json").write_text(
        '{"task": {"activity_name": "libero_90_task9"}, "evidence": {}}',
        encoding="utf-8",
    )

    assert _in_progress_attempt_diagnostics(tmp_path, {1}) == []


def test_native_authority_audit_blocks_outer_failure_after_native_success() -> None:
    rows = pd.DataFrame(
        [
            {"native_success": True, "outer_verification_success": False},
            {"native_success": True, "outer_verification_success": True},
            {"native_success": False, "outer_verification_success": False},
            {"native_success": None, "outer_verification_success": False},
        ]
    )

    mismatches = _native_authority_mismatches(rows)

    assert len(mismatches) == 1
    assert bool(mismatches.iloc[0].native_success)
    assert not bool(mismatches.iloc[0].outer_verification_success)


def test_interrupted_attempt_artifacts_are_not_labeled_current(tmp_path: Path) -> None:
    artifacts = tmp_path / "verifier_artifacts"
    artifacts.mkdir()
    path = artifacts / "iter003_attempt01_turn04_step09.json"
    path.write_text(
        '{"task": {"activity_name": "libero_90_task47"}, "evidence": {}}',
        encoding="utf-8",
    )
    os.utime(path, (100.0, 100.0))
    (tmp_path / "resume_restorations.jsonl").write_text(
        json.dumps(
            {
                "time": 200.0,
                "latest_completed_iteration": 2,
                "restored": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # A later retry may successfully commit the same iteration number. The
    # pre-resume artifact must remain visible as rolled back, not disappear.
    rows = _in_progress_attempt_diagnostics(tmp_path, {3})

    assert len(rows) == 1
    assert rows[0]["record_status"] == "rolled_back_interrupted"
    assert rows[0]["rollback_time"] == 200.0
    assert rows[0]["rollback_committed_through"] == 2


def test_development_attempt_diagnostics_redacts_predicates_and_labels_runtime_errors(
    tmp_path: Path,
) -> None:
    (tmp_path / "iteration_002.json").write_text(
        """{
          "iteration": 2,
          "task_proposal": {"activity_name": "libero_90_task12"},
          "execution_attempt_3": {
            "success": false,
            "stderr_snippet": "Traceback...\\nValueError: executing action in terminated episode",
            "user_result": {"success": false}
          },
          "diagnosis_attempt_3": {
            "failure_mode": "code_bug",
            "failed_step": "step-4",
            "confidence": 0.9
          },
          "verification_attempt_3": {
            "success": false,
            "details": {
              "native_predicate_success": false,
              "diagnostic_disagreement": true,
              "visual_custom_verifier": {"success": true, "confidence": 0.8},
              "private_native_audit": {
                "predicate_status": [{"predicate": "[on secret object]", "satisfied": false}]
              }
            }
          }
        }""",
        encoding="utf-8",
    )

    audit = _rats_attempt_diagnostics(tmp_path)

    assert len(audit) == 1
    assert audit.iloc[0].runtime_error_type == "terminated_episode"
    assert bool(audit.iloc[0].terminated_episode_retry)
    assert bool(audit.iloc[0].diagnostic_disagreement)
    assert "predicate" not in " ".join(audit.columns).lower()
    assert "secret" not in audit.to_csv(index=False)


def test_completed_iteration_recovers_timeout_hidden_by_stderr_truncation(
    tmp_path: Path,
) -> None:
    stem = "iter002_attempt00_turn02_step02"
    (tmp_path / "iteration_002.json").write_text(
        f"""{{
          "iteration": 2,
          "task_proposal": {{"activity_name": "libero_90_task46"}},
          "execution_attempt_2": {{
            "success": false,
            "stderr_snippet": "Traceback...\\nFile /deep/stack/was/clipped",
            "user_result": {{"success": false}}
          }},
          "diagnosis_attempt_2": {{"failure_mode": "grasp_failure"}},
          "verification_attempt_2": {{
            "success": false,
            "details": {{
              "native_predicate_success": false,
              "artifact_paths": {{"json": "/tmp/{stem}.json"}}
            }}
          }}
        }}""",
        encoding="utf-8",
    )
    (tmp_path / "run.log").write_text(
        "Step 5: Execution\n"
        "Traceback (most recent call last):\n"
        "TimeoutError: Execution timed out after 180s\n"
        f"Verifier artifacts saved: /tmp/{stem}.json\n",
        encoding="utf-8",
    )

    audit = _rats_attempt_diagnostics(tmp_path)

    assert len(audit) == 1
    assert audit.iloc[0].runtime_error_type == "execution_timeout"
    assert bool(audit.iloc[0].execution_timeout)
    assert audit.iloc[0].runtime_error_message == (
        "TimeoutError: Execution timed out after 180s"
    )


def test_runtime_self_check_is_never_counted_as_native_task_success(
    tmp_path: Path,
) -> None:
    video = tmp_path / "iter003_policy_self_check_attempt5_passed.mp4"
    skills = tmp_path / "iter003_policy_self_check_attempt5_passed_skills.mp4"
    video.write_bytes(b"plain-video")
    skills.write_bytes(b"skills-video")
    (tmp_path / "run.log").write_text(
        "Saved policy self-check video: "
        "iter003_policy_self_check_attempt5_passed.mp4 (206 frames) + "
        "iter003_policy_self_check_attempt5_passed_skills.mp4\n",
        encoding="utf-8",
    )

    audit = _rats_runtime_self_checks(tmp_path)

    assert len(audit) == 1
    row = audit.iloc[0]
    assert row.iteration == 3
    assert row.runtime_check_index == 5
    assert bool(row.runtime_execution_check_passed)
    assert not bool(row.native_task_success_evaluated)
    assert not bool(row.task_success_claimed)
    assert row.frame_count == 206
    assert row.video_bytes == len(b"plain-video")


def test_development_call_categories_are_stable() -> None:
    assert _call_category("planner_verifier.verify_plan") == "planning"
    assert _call_category("policy_writer._generate_once") == "code_generation"
    assert (
        _call_category("libero_reduced_skill_library.verify_object_identity")
        == "perception_verification"
    )
    assert _call_category("failure_diagnoser._query_llm_text") == "reflection"


def test_generated_wall_timeout_requires_explicit_runner_marker(tmp_path: Path) -> None:
    (tmp_path / "run.log").write_text(
        "ordinary request timeout 120 seconds\n", encoding="utf-8"
    )
    assert not _generated_wall_timeout_reached(tmp_path)

    summary = tmp_path / "artifacts" / "trial" / "summary.txt"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "TimeoutError: Trial 1 exceeded 1000 seconds\n", encoding="utf-8"
    )
    assert _generated_wall_timeout_reached(tmp_path)

    summary.unlink()
    registered = tmp_path / "artifacts" / "final_summary.json"
    registered.parent.mkdir(parents=True, exist_ok=True)
    registered.write_text(
        json.dumps(
            {"termination": {"kind": "registered_hard_wall_timeout"}}
        ),
        encoding="utf-8",
    )
    assert _generated_wall_timeout_reached(tmp_path)


def test_capx_wall_timeout_flushes_recorded_video_before_summary() -> None:
    """The SIGALRM path must not bypass the normal video epilogue."""

    source = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "rats"
        / "capx-baseline"
        / "capx"
        / "envs"
        / "runner.py"
    ).read_text(encoding="utf-8")
    timeout_branch = source.split(
        'print(f"Trial {trial} timed out after {timeout_seconds} seconds")', 1
    )[1].split("def _build_timeout_summary", 1)[0]

    assert "_save_trial_video(" in timeout_branch
    assert 'suffix_extra="timeout"' in timeout_branch
    assert timeout_branch.index("_save_trial_video(") < timeout_branch.index(
        "return _build_timeout_summary("
    )


def test_capx_wall_timeout_uses_periodic_timer_until_wrapper_catches_it() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "rats"
        / "capx-baseline"
        / "capx"
        / "envs"
        / "runner.py"
    ).read_text(encoding="utf-8")

    assert "arm_periodic_deadline(" in source
    assert "disarm_deadline()" in source
    assert "signal.alarm(" not in source


def test_logged_subprocess_hard_watchdog_kills_process_group(tmp_path: Path) -> None:
    stop = threading.Event()
    result = _run_logged(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=tmp_path / "watchdog.log",
        stop_event=stop,
        hard_wall_time_seconds=0.2,
    )

    assert result["returncode"] == 124
    assert result["hard_wall_timeout"] is True
    assert result["seconds"] < 5
    assert not stop.is_set()


def test_rats_hard_watchdog_materializes_auditable_terminal_trace(
    tmp_path: Path,
) -> None:
    from PIL import Image

    episode_key = "libero_10/7/seed4"
    (tmp_path / "native_states.jsonl").write_text(
        json.dumps(
            {
                "episode_key": episode_key,
                "time": 2.0,
                "native_success": False,
                "native_predicates": [{"success": False}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    job = {
        "suite": "libero_10",
        "task_id": 7,
        "seed": 4,
        "instruction": "put both objects in the basket",
    }
    result = {
        "returncode": 124,
        "seconds": 1151.2,
        "hard_wall_time_seconds": 1150,
    }
    agent_io = tmp_path / "agent_io"
    agent_io.mkdir()
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(
        agent_io / "0001_planner_img_0.png"
    )
    Image.new("RGB", (16, 16), color=(0, 255, 0)).save(
        agent_io / "0002_verifier_img_0.png"
    )

    _materialize_rats_hard_wall_timeout_artifacts(tmp_path, job, result)

    iteration = json.loads(
        (tmp_path / "artifacts" / "iteration_001.json").read_text()
    )
    summary = json.loads(
        (tmp_path / "artifacts" / "final_summary.json").read_text()
    )
    assert iteration["task_proposal"]["scene_model"] == "libero_10"
    assert iteration["feedback_action"] == "registered_hard_wall_timeout"
    assert iteration["success"] is False
    assert summary["termination"]["launcher_returncode"] == 124
    assert summary["termination"]["harness_generated"] is True
    timeout_video = (
        tmp_path / "artifacts" / "registered_timeout_public_trace.mp4"
    )
    assert timeout_video.is_file() and timeout_video.stat().st_size > 0
    provenance = json.loads(
        (
            tmp_path
            / "artifacts"
            / "registered_timeout_visual_trace.json"
        ).read_text()
    )
    assert provenance["private_simulator_state_used"] is False
    assert len(provenance["frames"]) == 2
    assert provenance["output_sha256"]


def test_rats_hard_watchdog_preserves_existing_upstream_epilogue(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    iteration_path = artifacts / "iteration_001.json"
    summary_path = artifacts / "final_summary.json"
    upstream_iteration = {"source": "upstream", "success": False}
    upstream_summary = {"source": "upstream", "success_rate": 0.0}
    iteration_path.write_text(json.dumps(upstream_iteration), encoding="utf-8")
    summary_path.write_text(json.dumps(upstream_summary), encoding="utf-8")

    _materialize_rats_hard_wall_timeout_artifacts(
        tmp_path,
        {
            "suite": "libero_10",
            "task_id": 7,
            "seed": 4,
            "instruction": "put both objects in the basket",
        },
        {
            "returncode": 124,
            "seconds": 1151.2,
            "hard_wall_time_seconds": 1150,
        },
    )

    assert json.loads(iteration_path.read_text()) == upstream_iteration
    assert json.loads(summary_path.read_text()) == upstream_summary


def test_provider_latency_gate_is_outcome_independent_and_conservative() -> None:
    healthy = audit_latencies([10.0] * 24 + [40.0, 50.0, 60.0, 70.0, 80.0, 100.0])
    assert healthy["status"] == "pass"
    assert healthy["outcome_independent"] is True
    assert healthy["p90_seconds"] == 60.0

    outage = audit_latencies([20.0] * 24 + [120.0] * 6)
    assert outage["status"] == "infrastructure_invalid"
    assert outage["p90_seconds"] == 120.0


def test_provider_latency_gate_requires_enough_requests() -> None:
    result = audit_latencies([180.0] * 29)
    assert result["status"] == "insufficient_data"
    assert result["p90_seconds"] is None


def test_provider_latency_run_dir_recurses_into_generated_seed_layout(
    tmp_path: Path,
) -> None:
    flat = tmp_path / "llm_calls.worker0.jsonl"
    nested = tmp_path / "seed_00" / "llm_calls.jsonl"
    nested.parent.mkdir()
    flat.write_text(
        json.dumps({"event": "network_success", "elapsed_s": 2.5}) + "\n",
        encoding="utf-8",
    )
    nested.write_text(
        "\n".join(
            [
                json.dumps({"event": "network_success", "elapsed_s": 4.5}),
                json.dumps({"event": "network_error", "elapsed_s": 99.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    agent_io = tmp_path / "seed_01" / "agent_io" / "0001_planner.json"
    agent_io.parent.mkdir(parents=True)
    agent_io.write_text(
        json.dumps(
            {
                "request_id": "rats-request-1",
                "actual_model": "gpt-5.5",
                "elapsed_s": 6.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert latencies_from_run_dir(tmp_path) == [2.5, 4.5, 6.5]


def test_provider_latency_gate_uses_nearest_rank_for_small_heavy_sample() -> None:
    result = audit_latencies(
        [5.0, 10.0, 55.0], minimum_requests=3, maximum_p90_seconds=60.0
    )
    assert result["status"] == "pass"
    assert result["p90_seconds"] == 55.0


def test_representative_provider_probe_reuses_production_state_decision_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "racap.agent.experience.load_runtime_memory",
        lambda scope: "fixed runtime experience",
    )
    payload = _probe_payload(
        3,
        image_url="data:image/png;base64,AA==",
        model="gpt-5.5",
        profile="representative_state_decision",
    )
    assert payload["model"] == "gpt-5.5"
    assert payload["max_tokens"] == 1400
    assert "fixed runtime experience" in payload["messages"][1]["content"][0]["text"]
    assert len(payload["messages"][1]["content"]) == 5
    assert payload["messages"][1]["content"][1]["type"] == "image_url"


def test_provider_probe_adjudication_combines_heavy_requests_across_window(
    tmp_path: Path,
) -> None:
    paths = []
    for probe_index, heavy_count in enumerate((2, 1)):
        rows = []
        for index in range(5):
            heavy = index < heavy_count
            rows.append(
                {
                    "success": True,
                    "elapsed_s": 12.0 if heavy else 5.0,
                    "actual_model": "gpt-5.5",
                    "usage": {"prompt_tokens": 7000 if heavy else 3000},
                }
            )
        path = tmp_path / f"probe_{probe_index}.json"
        path.write_text(
            json.dumps(
                {
                    "model": "gpt-5.5",
                    "profile": "representative_state_decision",
                    "rows": rows,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = adjudicate(paths, minimum_heavy_requests=3)
    assert result["status"] == "pass"
    assert result["heavy_successful_requests"] == 3
    assert result["heavy_request_latency_audit"]["p90_seconds"] == 12.0
