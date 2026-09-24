#!/usr/bin/env python3
"""Audit the registered LIBERO-90 development histories.

RATS and RACaP optimize different persistent artifacts, so their training-time
success curves are not a shared benchmark score.  This report deliberately
keeps them in separate panels: RATS reports native success on its selected play
tasks, while RACaP reports paired candidate/champion measurements on each
curriculum stage's registered cohort.  Frozen evaluation is the only direct
system comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.controlled_comparison.plot_style import clean_axis, save_figure

try:
    from .analyze_results import _jsonl, _usage
except ImportError:  # direct ``python path/to/script.py`` execution
    from analyze_results import _jsonl, _usage

DEFAULT_RATS = ROOT / "outputs" / "controlled_comparison" / "development" / "rats90_selfplay"
DEFAULT_RATS_INVALID = ROOT / "outputs" / "controlled_comparison" / "invalid"
DEFAULT_RACAP = ROOT / "outputs" / "evolution" / "racap_phase2"

_FORBIDDEN_PROMPT_PAYLOAD_PATTERNS = {
    "native_predicates_json": re.compile(r'''["']native_predicates["']\s*:''', re.I),
    "goal_state_json": re.compile(r'''["']goal_state["']\s*:''', re.I),
    "predicate_satisfaction_json": re.compile(r'''["']satisfied["']\s*:''', re.I),
    "parsed_problem_json": re.compile(r'''["']parsed_problem["']\s*:''', re.I),
    "bddl_goal_block": re.compile(r"\(\s*:goal\b", re.I),
    "simulator_pose_json": re.compile(
        # LIBERO privileged observation keys use instance-indexed names such
        # as ``akita_black_bowl_1_pos``.  Do not flag agent-computed/public
        # proprioceptive fields such as grasp_pos, eef_pos or joint_pos.
        r'''["'][A-Za-z][A-Za-z0-9_]*_[0-9]+_(?:pos|quat)["']\s*:''', re.I
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _development_progress(root: Path) -> dict[str, Any]:
    """Report committed and transient rounds without trusting restart log counters."""

    manifest_path = root / "run_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    target = int(manifest.get("rounds") or 0)
    committed = sorted(
        int(match.group(1))
        for path in root.glob("iteration_*.json")
        if (match := re.fullmatch(r"iteration_(\d+)\.json", path.name))
    )
    transient = sorted(
        int(match.group(1))
        for path in root.glob("iteration_*")
        if path.is_dir()
        and (match := re.fullmatch(r"iteration_(\d+)", path.name))
        and int(match.group(1)) not in committed
    )
    active = transient[-1] if transient else None
    # A round directory is created only after task proposal / planning.  During
    # that potentially long interval, the append-only log is the sole evidence
    # that the next round has started.  Use only the *last* loop marker and only
    # when it is newer than the latest atomic iteration file.  This avoids
    # mistaking the ``ITERATION 1/32`` markers printed during resume scans for a
    # regression to round one.
    if active is None:
        log_path = root / "rats.log"
        if log_path.is_file():
            markers = re.findall(
                r"\bITERATION\s+(\d+)/(\d+)\b",
                log_path.read_text(encoding="utf-8", errors="replace"),
            )
            if markers:
                last_started, logged_target = (int(value) for value in markers[-1])
                latest_committed = committed[-1] if committed else 0
                if (
                    last_started > latest_committed
                    and last_started <= (target or logged_target)
                ):
                    active = last_started
    if target and len(committed) >= target:
        status = "complete"
    elif active is not None:
        status = "running"
    else:
        status = "idle_or_paused"
    return {
        "committed_rounds": len(committed),
        "latest_committed_round": committed[-1] if committed else 0,
        "active_round": active,
        "target_rounds": target,
        "status": status,
    }


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resume_transaction_audit(root: Path) -> pd.DataFrame:
    """Verify that interrupted development resumed from committed state only."""

    columns = [
        "time",
        "latest_completed_iteration",
        "restored",
        "snapshot",
        "snapshot_exists",
        "archived_partial_state",
        "archive_exists",
        "live_before_skills_sha256",
        "archived_skills_sha256",
        "live_before_failure_memory_sha256",
        "archived_failure_memory_sha256",
        "restored_skills_sha256",
        "snapshot_skills_sha256",
        "restored_failure_memory_sha256",
        "snapshot_failure_memory_sha256",
        "recovery_reason",
        "kept_failure_episodes",
        "dropped_partial_episodes",
        "simulator_and_agent_io_telemetry_retained",
        "audit_pass",
    ]
    rows: list[dict[str, Any]] = []
    for record in _jsonl(root / "resume_restorations.jsonl"):
        snapshot = Path(str(record.get("snapshot") or ""))
        archive_text = str(record.get("archived_partial_state") or "")
        archive = Path(archive_text) if archive_text else None
        recovery_path = snapshot / "recovery_manifest.json"
        recovery = _read_json(recovery_path) if recovery_path.is_file() else {}
        live_before = record.get("live_before") or {}
        restored_state = record.get("restored_state") or {}
        snapshot_skills = snapshot / "skills.json"
        archived_skills = archive / "skills.json" if archive else None
        archived_memory = archive / "failure_memory" if archive else None
        snapshot_skills_hash = (
            hashlib.sha256(snapshot_skills.read_bytes()).hexdigest()
            if snapshot_skills.is_file()
            else ""
        )
        archived_skills_hash = (
            hashlib.sha256(archived_skills.read_bytes()).hexdigest()
            if archived_skills is not None and archived_skills.is_file()
            else ""
        )
        archived_memory_hash = (
            _tree_sha256(archived_memory)
            if archived_memory is not None and archived_memory.is_dir()
            else ""
        )
        snapshot_memory_hash = _tree_sha256(snapshot / "failure_memory")
        restored = bool(record.get("restored"))
        archive_ok = (not restored) or (
            archive is not None
            and archive.is_dir()
            and archived_skills_hash == str(live_before.get("skills_sha256") or "")
            and archived_memory_hash
            == str(live_before.get("failure_memory_tree_sha256") or "")
        )
        snapshot_ok = (
            snapshot.is_dir()
            and snapshot_skills_hash
            == str(restored_state.get("skills_sha256") or snapshot_skills_hash)
            and snapshot_memory_hash
            == str(
                restored_state.get("failure_memory_tree_sha256")
                or snapshot_memory_hash
            )
        )
        rows.append(
            {
                "time": record.get("time"),
                "latest_completed_iteration": record.get(
                    "latest_completed_iteration"
                ),
                "restored": restored,
                "snapshot": str(snapshot),
                "snapshot_exists": snapshot.is_dir(),
                "archived_partial_state": archive_text,
                "archive_exists": bool(archive and archive.is_dir()),
                "live_before_skills_sha256": live_before.get("skills_sha256"),
                "archived_skills_sha256": archived_skills_hash,
                "live_before_failure_memory_sha256": live_before.get(
                    "failure_memory_tree_sha256"
                ),
                "archived_failure_memory_sha256": archived_memory_hash,
                "restored_skills_sha256": restored_state.get("skills_sha256"),
                "snapshot_skills_sha256": snapshot_skills_hash,
                "restored_failure_memory_sha256": restored_state.get(
                    "failure_memory_tree_sha256"
                ),
                "snapshot_failure_memory_sha256": snapshot_memory_hash,
                "recovery_reason": recovery.get("reason", ""),
                "kept_failure_episodes": len(
                    recovery.get("kept_failure_episode_ids") or []
                ),
                "dropped_partial_episodes": len(
                    recovery.get("dropped_partial_episode_ids") or []
                ),
                "simulator_and_agent_io_telemetry_retained": recovery.get(
                    "simulator_and_agent_io_telemetry_retained"
                ),
                "audit_pass": bool(snapshot_ok and archive_ok),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _agent_calls(root: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for path in sorted((root / "agent_io").glob("*.json")):
        try:
            row = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        row["_path"] = str(path)
        calls.append(row)
    return sorted(calls, key=lambda row: float(row.get("timestamp") or 0.0))


_CRITIC_EVIDENCE_COLUMNS = [
    "schema_version",
    "record_type",
    "time",
    "supersedes_time",
    "iteration",
    "attempt",
    "turn",
    "task_id",
    "caller",
    "request_artifact",
    "agent_view_evidence",
    "wrist_view_evidence",
    "per_step_verifier_claim",
    "terminal_verifier_claim",
    "critic_claim",
    "conflict_type",
    "human_visual_adjudication",
    "development_chain_action",
    "algorithm_changed_mid_chain",
    "correction_reason",
    "effective_status",
]


def _critic_evidence_audits(root: Path) -> pd.DataFrame:
    """Load adjudications without hiding later corrections.

    Human video review is itself fallible.  A correction therefore appends a
    new row naming the timestamp it supersedes rather than deleting history.
    Both rows remain in the workbook, while only unsuperseded conflict records
    count as active evidence.
    """

    rows = _jsonl(root / "critic_evidence_audits.jsonl")
    superseded = {
        float(row["supersedes_time"])
        for row in rows
        if row.get("record_type") == "correction"
        and row.get("supersedes_time") is not None
    }
    for row in rows:
        row_type = str(row.get("record_type") or "conflict")
        timestamp = row.get("time")
        if row_type == "correction":
            row["effective_status"] = "correction"
        elif timestamp is not None and float(timestamp) in superseded:
            row["effective_status"] = "retracted"
        else:
            row["effective_status"] = "active_conflict"
        row["record_type"] = row_type
    return pd.DataFrame(rows, columns=_CRITIC_EVIDENCE_COLUMNS)


def _skill_admission_audit(root: Path) -> pd.DataFrame:
    """Audit both native-success extraction and RATS failure proposals.

    RATS has two intentionally different library-write paths.  A solved task
    may extract reusable functions from the successful program, which requires
    native success.  Independently, the published failure-driven
    ``SkillProposer`` may add an *experimental* helper after a failed round.
    Those helpers are not success claims: they must be explicitly marked as
    proposed from failures and are subsequently validated (or deprecated) by
    later use.  Conflating the second path with success extraction produces a
    false audit failure and silently changes the reproduced baseline.
    """

    columns = [
        "iteration",
        "task",
        "top_level_success",
        "native_success",
        "native_success_attempts",
        "skills_added",
        "skills_added_count",
        "success_extracted_skills",
        "failure_proposed_experimental_skills",
        "unclassified_skills",
        "skills_present_in_current_library",
        "skills_present_in_iteration_snapshot",
        "top_level_matches_native",
        "native_success_required_for_admission",
        "failure_proposal_rule_satisfied",
        "admission_rule",
        "audit_pass",
        "artifact",
    ]

    def library_records(path: Path) -> dict[str, dict[str, Any]]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, list):
            return {}
        return {
            str(row.get("name")): row
            for row in payload
            if isinstance(row, dict) and row.get("name")
        }

    skills_path = root / "skills.json"
    current_library = library_records(skills_path)
    library_names = set(current_library)

    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("iteration_*.json")):
        if not re.fullmatch(r"iteration_\d+\.json", path.name):
            continue
        try:
            raw = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if raw.get("error"):
            continue
        native_success_attempts: list[int] = []
        native_observations: list[bool] = []
        for key, value in raw.items():
            match = re.fullmatch(r"verification_attempt_(\d+)", str(key))
            if not match or not isinstance(value, dict):
                continue
            details = value.get("details") or {}
            native = details.get("native_predicate_success")
            if native is None:
                native = (details.get("native_verifier") or {}).get("success")
            if native is None:
                continue
            native_bool = bool(native)
            native_observations.append(native_bool)
            if native_bool:
                native_success_attempts.append(int(match.group(1)))
        native_success = any(native_observations)
        top_level_success = bool(raw.get("success"))
        added = [str(value) for value in raw.get("skills_added") or []]
        try:
            iteration = int(raw.get("iteration") or path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        extracted = {
            str(value) for value in raw.get("skills_learned") or [] if value
        }
        proposed = {
            str(row.get("name"))
            for row in raw.get("proposed_skills") or []
            if isinstance(row, dict) and row.get("name")
        }
        added_set = set(added)
        failure_proposed = added_set & proposed
        success_extracted = added_set & extracted
        unclassified = added_set - failure_proposed - success_extracted

        snapshot_path = (
            root / "snapshots" / f"iter{iteration:03d}" / "skills.json"
        )
        snapshot_library = library_records(snapshot_path)
        # Unit fixtures and legacy runs may predate per-round snapshots.  The
        # current library remains sufficient for success-extraction checks,
        # while controlled runs always exercise the snapshot branch.
        admission_library = snapshot_library or current_library
        present_current = all(name in library_names for name in added)
        present_snapshot = all(name in admission_library for name in added)
        top_matches = top_level_success == native_success
        native_required = not success_extracted or native_success

        failure_rule = True
        for name in failure_proposed:
            record = admission_library.get(name) or {}
            failure_rule = bool(
                failure_rule
                and not native_success
                and record.get("source_task") == "proposed_from_failures"
                and record.get("proposed") is True
                and record.get("tier") == "experimental"
                and int(record.get("learned_iteration") or -1) == iteration
                and name not in extracted
            )
        if success_extracted and failure_proposed:
            admission_rule = "mixed_invalid"
        elif failure_proposed:
            admission_rule = "failure_proposed_experimental"
        elif success_extracted:
            admission_rule = "native_success_extracted"
        elif added:
            admission_rule = "unclassified"
        else:
            admission_rule = "no_library_write"
        audit_pass = bool(
            present_snapshot
            and top_matches
            and native_required
            and failure_rule
            and not unclassified
            and not (success_extracted and failure_proposed)
        )
        proposal = raw.get("task_proposal") or {}
        rows.append(
            {
                "iteration": iteration,
                "task": proposal.get("activity_name"),
                "top_level_success": top_level_success,
                "native_success": native_success,
                "native_success_attempts": ",".join(
                    str(value) for value in native_success_attempts
                ),
                "skills_added": ",".join(added),
                "skills_added_count": len(added),
                "success_extracted_skills": ",".join(sorted(success_extracted)),
                "failure_proposed_experimental_skills": ",".join(
                    sorted(failure_proposed)
                ),
                "unclassified_skills": ",".join(sorted(unclassified)),
                "skills_present_in_current_library": present_current,
                "skills_present_in_iteration_snapshot": present_snapshot,
                "top_level_matches_native": top_matches,
                "native_success_required_for_admission": native_required,
                "failure_proposal_rule_satisfied": failure_rule,
                "admission_rule": admission_rule,
                "audit_pass": audit_pass,
                "artifact": str(path.resolve()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _request_text(row: dict[str, Any]) -> str:
    """Return only text sent to the model, excluding private audit metadata."""

    chunks: list[str] = []
    for message in (row.get("request") or {}).get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") not in {"text", "input_text"}:
                continue
            chunks.append(str(item.get("text") or ""))
    return "\n".join(chunks)


def _prompt_privilege_audit(calls: list[dict[str, Any]]) -> pd.DataFrame:
    """Flag structured evaluator/simulator payloads accidentally sent to an LLM.

    Generic words such as ``predicate`` or ``In (...)`` are intentionally not
    flagged: they occur in public API documentation.  These patterns target
    serialized native-state, BDDL-goal, or simulator-pose payloads.  Prompt
    content stays in the raw agent-I/O audit; the public table stores only a
    hash and call identity.
    """

    columns = [
        "call_index",
        "timestamp",
        "caller",
        "episode_key",
        "forbidden_payload_pattern",
        "prompt_sha256",
        "artifact",
    ]
    rows: list[dict[str, Any]] = []
    for index, call in enumerate(calls, start=1):
        prompt = _request_text(call)
        if not prompt:
            continue
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for name, pattern in _FORBIDDEN_PROMPT_PAYLOAD_PATTERNS.items():
            if not pattern.search(prompt):
                continue
            rows.append(
                {
                    "call_index": index,
                    "timestamp": call.get("timestamp"),
                    "caller": call.get("caller"),
                    "episode_key": call.get("episode_key"),
                    "forbidden_payload_pattern": name,
                    "prompt_sha256": digest,
                    "artifact": call.get("_path"),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _iteration_windows(calls: list[dict[str, Any]]) -> list[tuple[float, float]]:
    starts = [
        float(row.get("timestamp") or 0.0)
        for row in calls
        if "task_proposer" in str(row.get("caller") or "")
    ]
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else float("inf"))
        for index, start in enumerate(starts)
    ]


def _proposal_selected_task(call: dict[str, Any]) -> str:
    """Parse the public catalog task selected by one proposer response."""

    content = (call.get("response") or {}).get("content")
    if isinstance(content, dict):
        return str(content.get("selected_task") or "")
    if not isinstance(content, str):
        return ""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'"selected_task"\s*:\s*"([^"]+)"', text)
        return match.group(1) if match else ""
    return str(parsed.get("selected_task") or "") if isinstance(parsed, dict) else ""


def _resume_boundaries(root: Path) -> list[tuple[float, int]]:
    return sorted(
        (
            float(record.get("time") or 0.0),
            int(record.get("latest_completed_iteration") or 0),
        )
        for record in _jsonl(root / "resume_restorations.jsonl")
        if bool(record.get("restored")) and float(record.get("time") or 0.0) > 0
    )


def _committed_iteration_windows(
    root: Path, calls: list[dict[str, Any]]
) -> dict[int, tuple[float, float]]:
    """Match committed iteration files to their actual, non-rolled-back proposal.

    A resumed round may have more than one proposer call: the earlier call and
    all of its executions remain in the append-only audit, but its mutable
    state is rolled back. Positional matching would then assign that abandoned
    proposal to the later committed ``iteration_NNN.json``. Match by the
    selected public task and exclude proposal windows invalidated by a resume
    boundary instead.
    """

    proposer_calls = sorted(
        (
            row
            for row in calls
            if "task_proposer" in str(row.get("caller") or "")
        ),
        key=lambda row: float(row.get("timestamp") or 0.0),
    )
    proposals = [
        {
            "start": float(call.get("timestamp") or 0.0),
            "end": (
                float(proposer_calls[index + 1].get("timestamp") or 0.0)
                if index + 1 < len(proposer_calls)
                else float("inf")
            ),
            "task": _proposal_selected_task(call),
        }
        for index, call in enumerate(proposer_calls)
    ]
    boundaries = _resume_boundaries(root)
    windows: dict[int, tuple[float, float]] = {}
    cursor = -float("inf")
    for path in sorted(root.glob("iteration_*.json")):
        raw = _read_json(path)
        iteration = int(raw.get("iteration") or len(windows) + 1)
        proposal = raw.get("task_proposal") or {}
        target = str(
            proposal.get("activity_name")
            or (raw.get("scene_context") or {}).get("activity_name")
            or ""
        )

        def rolled_back(start: float) -> bool:
            return any(
                iteration > committed_through and start < boundary_time
                for boundary_time, committed_through in boundaries
            )

        eligible = [
            row
            for row in proposals
            if float(row["start"]) > cursor
            and not rolled_back(float(row["start"]))
            and (not target or row["task"] == target)
        ]
        if not eligible:
            # Old RATS logs may lack a structured selected_task. Only use an
            # unlabelled window as a compatibility fallback; never silently
            # bind a differently labelled public task.
            eligible = [
                row
                for row in proposals
                if float(row["start"]) > cursor
                and not rolled_back(float(row["start"]))
                and not row["task"]
            ]
        if not eligible:
            continue
        selected = eligible[0]
        start = float(selected["start"])
        windows[iteration] = (start, float(selected["end"]))
        cursor = start
    return windows


def _call_category(caller: str) -> str:
    """Map raw RATS call sites to stable paper-facing cost categories."""

    name = caller.lower()
    if "task_proposer" in name:
        return "proposal"
    if "planner" in name:
        return "planning"
    if "policy_writer" in name:
        return "code_generation"
    if "verify_object_identity" in name or "point_prompt" in name:
        return "perception_verification"
    if "multi_turn_decider" in name:
        return "runtime_decision"
    if "per_step_verifier" in name or name.startswith("verifier."):
        return "outcome_diagnostics"
    if "failure_diagnoser" in name or "feedback_generator" in name:
        return "reflection"
    if "skill_proposer" in name or "memory" in name:
        return "skill_memory"
    return "other"


def _identity_label(row: dict[str, Any]) -> str:
    request = row.get("request") or {}
    for message in request.get("messages") or []:
        for item in message.get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            match = re.search(r"Robot intends to grasp: '([^']+)'", str(item.get("text") or ""))
            if match:
                return match.group(1)
    return "unknown"


def _identity_verified(row: dict[str, Any]) -> bool | None:
    content = (row.get("response") or {}).get("content")
    if isinstance(content, dict):
        return bool(content.get("verified"))
    if not isinstance(content, str):
        return None
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return bool(parsed.get("verified"))


def _identity_search_runs(calls: list[dict[str, Any]]) -> pd.DataFrame:
    """Measure exhaustive candidate checking without assuming a task label.

    A run is a consecutive block of ``verify_object_identity`` calls for the
    same expected object.  Calls after the first accepted candidate in that
    block are observable redundant search, regardless of whether the final
    manipulation succeeds.
    """

    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in calls:
        caller = str(row.get("caller") or "")
        if "verify_object_identity" not in caller:
            current = None
            continue
        label = _identity_label(row)
        if current is None or current["expected_object"] != label:
            current = {
                "search_run": len(runs) + 1,
                "expected_object": label,
                "calls": 0,
                "verified_true": 0,
                "first_true_call": None,
                "calls_after_first_true": 0,
            }
            runs.append(current)
        current["calls"] += 1
        verdict = _identity_verified(row)
        if verdict:
            current["verified_true"] += 1
            if current["first_true_call"] is None:
                current["first_true_call"] = current["calls"]
        if current["first_true_call"] is not None and current["calls"] > current["first_true_call"]:
            current["calls_after_first_true"] += 1
    for run in runs:
        run["redundant_fraction_after_first_true"] = (
            run["calls_after_first_true"] / run["calls"] if run["calls"] else 0.0
        )
    return pd.DataFrame(runs)


def _rats_cost_breakdowns(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    calls = _agent_calls(root)
    windows = _committed_iteration_windows(root, calls)
    categories: list[dict[str, Any]] = []
    identity: list[pd.DataFrame] = []
    for path in sorted(root.glob("iteration_*.json")):
        raw = _read_json(path)
        iteration = int(raw.get("iteration") or len(identity) + 1)
        if iteration not in windows:
            continue
        start, end = windows[iteration]
        selected = [row for row in calls if start <= float(row.get("timestamp") or 0.0) < end]
        for category in sorted({_call_category(str(row.get("caller") or "")) for row in selected}):
            part = [row for row in selected if _call_category(str(row.get("caller") or "")) == category]
            categories.append({"iteration": iteration, "category": category, **_usage(part)})
        runs = _identity_search_runs(selected)
        if not runs.empty:
            runs.insert(0, "iteration", iteration)
            identity.append(runs)
    return pd.DataFrame(categories), (pd.concat(identity, ignore_index=True) if identity else pd.DataFrame())


def _rats_observed_usage(root: Path) -> pd.DataFrame:
    """Summarize every append-only development event observed so far.

    Completed ``iteration_*.json`` files remain the sole authority for the
    development success curve. Usage and reset accounting, however, must not
    disappear merely because a long iteration is still running or an API
    quota checkpoint arrives before finalization. This table therefore
    reports resource consumption over the complete append-only ledgers and
    labels it explicitly as observed rather than completed-iteration usage.
    """

    calls = _agent_calls(root)
    usage = _usage(calls)
    resets = _jsonl(root / "sim_episodes.jsonl")
    completed = len(list(root.glob("iteration_*.json")))
    proposed = sum(
        "task_proposer" in str(row.get("caller") or "") for row in calls
    )
    return pd.DataFrame(
        [
            {
                "record_status": "append_only_observed",
                "completed_iterations": completed,
                "iterations_started": proposed,
                "simulator_resets_observed": len(resets),
                **usage,
            }
        ]
    )


def _rats_invalidated_usage(root: Path) -> pd.DataFrame:
    """Account for every discarded RATS development lineage exactly once.

    These runs cannot contribute skills, success rates, or the frozen
    artifact.  They still consumed simulator resets and hosted-model calls,
    however, so omitting them would understate engineering-time resources and
    could silently violate the registered 596-reset ceiling.  Frozen-artifact
    copies are deliberately excluded because they duplicate their associated
    ``rats90_selfplay_*`` source tree.
    """

    columns = [
        "lineage",
        "invalidation_file",
        "invalidation_reason",
        "completed_iterations",
        "simulator_resets_observed",
        "model_calls",
        "model_latency_seconds",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "estimated_api_cost_usd",
        "actual_models",
        "counted_in_engineering_total",
        "eligible_for_success_or_skill_metrics",
    ]
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return pd.DataFrame(columns=columns)
    for lineage in sorted(root.glob("rats90_selfplay_*")):
        if not lineage.is_dir():
            continue
        marker = next(
            (
                path
                for name in ("INVALIDATED.json", "INVALID_RUN.json")
                if (path := lineage / name).is_file()
            ),
            None,
        )
        metadata: dict[str, Any] = {}
        if marker is not None:
            try:
                metadata = _read_json(marker)
            except (OSError, json.JSONDecodeError):
                metadata = {}
        reason = str(
            metadata.get("reason")
            or metadata.get("details")
            or lineage.name.removeprefix("rats90_selfplay_")
        )
        resets: list[dict[str, Any]] = []
        for path in sorted(lineage.rglob("sim_episodes.jsonl")):
            resets.extend(_jsonl(path))
        usage = _usage(_agent_calls(lineage))
        rows.append(
            {
                "lineage": lineage.name,
                "invalidation_file": str(marker.resolve()) if marker else "",
                "invalidation_reason": reason,
                "completed_iterations": len(list(lineage.glob("iteration_*.json"))),
                "simulator_resets_observed": len(resets),
                **usage,
                "counted_in_engineering_total": True,
                "eligible_for_success_or_skill_metrics": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _rats_resource_accounting(
    committed: pd.DataFrame,
    active_observed: pd.DataFrame,
    invalidated: pd.DataFrame,
) -> pd.DataFrame:
    """Separate scientific lineage metrics from all engineering consumption."""

    numeric = [
        "simulator_resets",
        "model_calls",
        "model_latency_seconds",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "estimated_api_cost_usd",
    ]

    def row_from_frame(scope: str, frame: pd.DataFrame, reset_column: str) -> dict[str, Any]:
        result: dict[str, Any] = {"scope": scope}
        for key in numeric:
            source = reset_column if key == "simulator_resets" else key
            result[key] = float(frame[source].sum()) if source in frame else 0.0
        models: set[str] = set()
        if "actual_models" in frame:
            for value in frame.actual_models.dropna().astype(str):
                models.update(part for part in value.split(",") if part)
        result["actual_models"] = ",".join(sorted(models))
        return result

    committed_row = row_from_frame(
        "committed_15_round_lineage", committed, "simulator_resets"
    )
    active_row = row_from_frame(
        "valid_lineage_append_only", active_observed, "simulator_resets_observed"
    )
    invalid_row = row_from_frame(
        "invalidated_preprotocol_lineages", invalidated, "simulator_resets_observed"
    )
    total_row = {"scope": "all_recorded_engineering_consumption"}
    for key in numeric:
        total_row[key] = float(active_row[key]) + float(invalid_row[key])
    total_row["actual_models"] = ",".join(
        sorted(
            {
                model
                for value in (active_row["actual_models"], invalid_row["actual_models"])
                for model in str(value).split(",")
                if model
            }
        )
    )
    frame = pd.DataFrame([committed_row, active_row, invalid_row, total_row])
    for key in ("simulator_resets", "model_calls", "prompt_tokens", "completion_tokens", "cached_tokens"):
        frame[key] = frame[key].astype(int)
    return frame


def _runtime_error(stderr: str) -> tuple[str, str]:
    """Return a compact, non-privileged runtime-error label and final line."""

    lines = [line.strip() for line in str(stderr or "").splitlines() if line.strip()]
    final = lines[-1] if lines else ""
    lowered = final.lower()
    if "executing action in terminated episode" in lowered:
        return "terminated_episode", final
    if "timed out after" in lowered or "timeouterror" in lowered:
        return "execution_timeout", final
    if final:
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:", final)
        return (match.group(1) if match else "runtime_error"), final
    return "none", ""


def _runtime_errors_from_log(root: Path) -> dict[str, tuple[str, str]]:
    """Recover full runtime errors for verifier artifacts from the append-only log.

    Verifier JSON deliberately stores only a short stderr preview.  That is
    sufficient for the controller but often truncates the final exception
    line, so the offline audit joins each saved artifact to the most recent
    exception emitted by its execution block.  This parser is evaluator-only
    and never exposes the result back to RATS.
    """

    log = root / "run.log"
    if not log.is_file():
        return {}
    errors: dict[str, tuple[str, str]] = {}
    current_error = ("none", "")
    exception = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\s*:\s*(.*)$"
    )
    artifact = re.compile(r"Verifier artifacts saved:\s+.*?/([^/]+)\.json\s*$")
    for raw_line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if "Step 5: Execution" in line:
            current_error = ("none", "")
            continue
        match = exception.match(line)
        if match:
            current_error = _runtime_error(line)
            continue
        saved = artifact.search(line)
        if saved:
            errors[saved.group(1)] = current_error
            current_error = ("none", "")
    return errors


def _in_progress_attempt_diagnostics(
    root: Path, completed_iterations: set[int]
) -> list[dict[str, Any]]:
    """Read append-only verifier artifacts before an iteration is finalized.

    A long RATS iteration may run for hours.  Keeping these records in the
    interim workbook makes timeouts and failed attempts auditable immediately,
    including when a quota checkpoint stops the run before ``iteration_*.json``
    is written.  Completed iteration JSON remains authoritative and replaces
    these provisional rows on the next analysis pass.
    """

    errors = _runtime_errors_from_log(root)
    resume_boundaries = _resume_boundaries(root)
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"iter(?P<iteration>\d+)_attempt(?P<attempt>\d+)_"
        r"turn(?P<turn>\d+)_step(?P<step>\d+)"
    )
    for path in sorted((root / "verifier_artifacts").glob("iter*.json")):
        match = pattern.fullmatch(path.stem)
        if not match:
            continue
        iteration = int(match.group("iteration"))
        try:
            raw = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        artifact_mtime = path.stat().st_mtime
        rollback = next(
            (
                (boundary_time, committed_through)
                for boundary_time, committed_through in reversed(resume_boundaries)
                if iteration > committed_through and artifact_mtime < boundary_time
            ),
            None,
        )
        if iteration in completed_iterations and rollback is None:
            continue
        evidence = raw.get("evidence") or {}
        task = raw.get("task") or {}
        visual = evidence.get("visual_custom_verifier") or {}
        error_type, error_message = errors.get(path.stem, ("none", ""))
        native_success = evidence.get("native_predicate_success")
        if native_success is None:
            native_success = raw.get("final_success")
        rows.append(
            {
                "iteration": iteration,
                "record_status": (
                    "rolled_back_interrupted" if rollback else "in_progress_artifact"
                ),
                "attempt": int(match.group("attempt")),
                "turn": int(match.group("turn")),
                "flat_step": int(match.group("step")),
                "task": task.get("activity_name"),
                "execution_index": None,
                "execution_success": evidence.get("execution_success"),
                "policy_reported_success": None,
                "execution_reward": float(evidence.get("reward") or 0.0),
                "execution_task_completed": bool(evidence.get("task_completed")),
                "verification_present": True,
                "outer_verification_success": raw.get("final_success"),
                "native_success": native_success,
                "visual_success": visual.get("success"),
                "visual_confidence": visual.get("confidence"),
                "diagnostic_disagreement": evidence.get("diagnostic_disagreement"),
                "failure_mode": (
                    error_type if error_type != "none" else "in_progress_native_failure"
                ),
                "failed_step": None,
                "diagnosis_confidence": None,
                "verifier_challenge_requery": False,
                "runtime_error_type": error_type,
                "runtime_error_message": error_message,
                "terminated_episode_retry": error_type == "terminated_episode",
                "execution_timeout": error_type == "execution_timeout",
                "artifact_mtime": artifact_mtime,
                "rollback_time": rollback[0] if rollback else None,
                "rollback_committed_through": rollback[1] if rollback else None,
                "artifact": str(path.resolve()),
            }
        )
    return rows


def _rats_attempt_diagnostics(root: Path) -> pd.DataFrame:
    """Flatten completed RATS execution/diagnosis records for failure audit.

    The table intentionally excludes private predicate strings and simulator
    object state.  It preserves the policy-visible execution result, advisory
    visual diagnosis, and binary native outcome so disagreements and wasted
    retries can be audited without widening the controller's information
    channel.
    """

    rows: list[dict[str, Any]] = []
    log_errors = _runtime_errors_from_log(root)
    completed_paths = sorted(root.glob("iteration_*.json"))
    completed_iterations: set[int] = set()
    for path in completed_paths:
        raw = _read_json(path)
        iteration = int(raw.get("iteration") or 0)
        completed_iterations.add(iteration)
        proposal = raw.get("task_proposal") or {}
        task = proposal.get("activity_name") or (raw.get("scene_context") or {}).get(
            "activity_name"
        )
        indexes = sorted(
            int(key.removeprefix("execution_attempt_"))
            for key in raw
            if re.fullmatch(r"execution_attempt_\d+", key)
        )
        for index in indexes:
            execution = raw.get(f"execution_attempt_{index}") or {}
            diagnosis = raw.get(f"diagnosis_attempt_{index}") or {}
            verification = raw.get(f"verification_attempt_{index}") or {}
            details = verification.get("details") or {}
            visual = details.get("visual_custom_verifier") or {}
            error_type, error_message = _runtime_error(
                execution.get("stderr_snippet") or ""
            )
            # Completed iteration JSON stores only a bounded stderr preview.
            # A deep traceback can therefore be cut before its final
            # ``TimeoutError`` / terminated-episode line.  Join the verifier's
            # append-only artifact identity back to run.log so completion does
            # not erase a runtime failure that was visible while the iteration
            # was in progress.  This is analysis-only and never enters the
            # controller feedback channel.
            artifact_json = (details.get("artifact_paths") or {}).get("json")
            artifact_stem = Path(str(artifact_json)).stem if artifact_json else ""
            log_error_type, log_error_message = log_errors.get(
                artifact_stem, ("none", "")
            )
            if error_type in {"none", "runtime_error"} and log_error_type != "none":
                error_type, error_message = log_error_type, log_error_message
            failure_mode = diagnosis.get("failure_mode")
            if not failure_mode:
                if bool(verification.get("success")):
                    failure_mode = "none"
                elif error_type != "none":
                    failure_mode = error_type
                elif verification:
                    failure_mode = "unclassified_failure"
                else:
                    failure_mode = "intermediate_execution"
            native_success = details.get("native_predicate_success")
            if native_success is None and verification:
                native_success = verification.get("success")
            rows.append(
                {
                    "iteration": iteration,
                    "record_status": "completed_iteration",
                    "attempt": None,
                    "turn": None,
                    "flat_step": None,
                    "task": task,
                    "execution_index": index,
                    "execution_success": bool(execution.get("success")),
                    "policy_reported_success": bool(
                        (execution.get("user_result") or {}).get("success")
                    ),
                    "execution_reward": float(execution.get("reward") or 0.0),
                    "execution_task_completed": bool(execution.get("task_completed")),
                    "verification_present": bool(verification),
                    "outer_verification_success": (
                        verification.get("success") if verification else None
                    ),
                    "native_success": native_success,
                    "visual_success": visual.get("success"),
                    "visual_confidence": visual.get("confidence"),
                    "diagnostic_disagreement": details.get("diagnostic_disagreement"),
                    "failure_mode": failure_mode,
                    "failed_step": diagnosis.get("failed_step"),
                    "diagnosis_confidence": diagnosis.get("confidence"),
                    "verifier_challenge_requery": bool(
                        diagnosis.get("verifier_challenge_requery")
                    ),
                    "runtime_error_type": error_type,
                    "runtime_error_message": error_message,
                    "terminated_episode_retry": error_type == "terminated_episode",
                    "execution_timeout": error_type == "execution_timeout",
                    "artifact": str(path.resolve()),
                }
            )
    rows.extend(_in_progress_attempt_diagnostics(root, completed_iterations))
    return pd.DataFrame(rows)


def _native_authority_mismatches(attempts: pd.DataFrame) -> pd.DataFrame:
    """Return rows where the outer verifier overruled native task success.

    Native LIBERO predicates are the pre-registered scoring authority.  This
    invariant is deliberately checked from persisted artifacts, after policy
    execution, so it cannot expose private state to the controller.  Any row
    returned here means the development/evaluation harness is not admissible
    for comparison until the scorer is fixed or the affected round is rerun.
    """

    required = {"native_success", "outer_verification_success"}
    if attempts.empty or not required.issubset(attempts.columns):
        return attempts.iloc[0:0].copy()
    return attempts[
        attempts["native_success"].eq(True)  # noqa: E712
        & attempts["outer_verification_success"].eq(False)  # noqa: E712
    ].copy()


def _rats_runtime_self_checks(root: Path) -> pd.DataFrame:
    """Index execution-only policy checks without treating them as task wins.

    RATS names these videos ``*_passed`` when generated code runs without an
    exception.  The check does *not* consult the registered native predicate,
    so a physically unsuccessful grasp may still receive that filename.  Keep
    this distinction machine-readable to prevent self-check health from being
    mistaken for benchmark success in later analysis.
    """

    columns = [
        "iteration",
        "runtime_check_index",
        "runtime_execution_check_passed",
        "native_task_success_evaluated",
        "task_success_claimed",
        "frame_count",
        "video_bytes",
        "skills_overlay_bytes",
        "video_artifact",
        "skills_overlay_artifact",
    ]
    log = root / "run.log"
    if not log.is_file():
        return pd.DataFrame(columns=columns)
    saved = re.compile(
        r"Saved policy self-check video:\s+(?P<video>\S+)\s+"
        r"\((?P<frames>\d+) frames\)\s+\+\s+(?P<skills>\S+)"
    )
    identity = re.compile(
        r"iter(?P<iteration>\d+)_policy_self_check_"
        r"attempt(?P<index>\d+)_(?P<status>[A-Za-z0-9_-]+)\.mp4$"
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = saved.search(line)
        if not match:
            continue
        video_name = match.group("video")
        skills_name = match.group("skills")
        key = (video_name, skills_name)
        if key in seen:
            continue
        seen.add(key)
        parsed = identity.search(Path(video_name).name)
        if not parsed:
            continue
        video = root / video_name
        skills = root / skills_name
        rows.append(
            {
                "iteration": int(parsed.group("iteration")),
                "runtime_check_index": int(parsed.group("index")),
                "runtime_execution_check_passed": parsed.group("status") == "passed",
                "native_task_success_evaluated": False,
                "task_success_claimed": False,
                "frame_count": int(match.group("frames")),
                "video_bytes": video.stat().st_size if video.is_file() else None,
                "skills_overlay_bytes": skills.stat().st_size if skills.is_file() else None,
                "video_artifact": str(video.resolve()),
                "skills_overlay_artifact": str(skills.resolve()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _native_event_audit(root: Path) -> pd.DataFrame:
    """Index private native events by the completed development iteration.

    Dynamic catalog rebinding historically updated ``RATS_EPISODE_KEY`` before
    the proposer selected the next task, so the raw key can lag one task behind
    even though the simulator and controller use the newly rebound task.  Raw
    events remain append-only; this evaluator-only table reconstructs their
    task identity from proposal time windows and records whether the old label
    agrees.  Exact predicate content is intentionally excluded.
    """

    calls = _agent_calls(root)
    windows = _committed_iteration_windows(root, calls)
    events = _jsonl(root / "native_states.jsonl")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("iteration_*.json")):
        raw = _read_json(path)
        iteration = int(raw.get("iteration") or len(rows) + 1)
        if iteration not in windows:
            continue
        start, end = windows[iteration]
        proposal = raw.get("task_proposal") or {}
        task = str(
            proposal.get("activity_name")
            or (raw.get("scene_context") or {}).get("activity_name")
            or ""
        )
        match = re.fullmatch(r"(.+)_task(\d+)", task)
        suite = match.group(1) if match else ""
        task_id = int(match.group(2)) if match else None
        corrected_key = (
            f"{suite}/{task_id}/development_round{iteration}"
            if suite and task_id is not None
            else f"development/round{iteration}"
        )
        selected_events = sorted(
            (
                event
                for event in events
                if start <= float(event.get("time") or 0.0) < end
            ),
            key=lambda event: float(event.get("time") or 0.0),
        )
        for event_index, event in enumerate(selected_events):
            timestamp = float(event.get("time") or 0.0)
            raw_key = str(event.get("episode_key") or "")
            simulator_task_id = event.get("simulator_task_id")
            observed_task_id = (
                int(simulator_task_id) if simulator_task_id is not None else task_id
            )
            raw_match = re.search(r"/(\d+)/seed-?\d+$", raw_key)
            raw_task_id = int(raw_match.group(1)) if raw_match else None
            raw_key_matches = (
                raw_task_id == observed_task_id
                if raw_task_id is not None and observed_task_id is not None
                else None
            )
            expected_rebind_label_lag = bool(
                event_index == 0
                and raw_key_matches is False
                and simulator_task_id is not None
                and task_id is not None
                and int(simulator_task_id) == task_id
            )
            rows.append(
                {
                    "iteration": iteration,
                    "registered_task": task,
                    "registered_suite": suite,
                    "registered_task_id": task_id,
                    "internal_trial_seed": iteration,
                    "libero_init_state_index": iteration - 1,
                    "event_time": timestamp,
                    "event": event.get("event"),
                    "api_name": event.get("api_name"),
                    "simulator_steps": event.get("simulator_steps"),
                    "native_success": bool(event.get("native_success")),
                    "raw_episode_key": raw_key,
                    "corrected_episode_key": corrected_key,
                    "simulator_reported_task_id": simulator_task_id,
                    "effective_task_id": observed_task_id,
                    "raw_key_task_matches": raw_key_matches,
                    "first_native_event_in_iteration": event_index == 0,
                    "expected_rebind_label_lag": expected_rebind_label_lag,
                    "identity_source": (
                        "simulator_runtime"
                        if simulator_task_id is not None
                        else "proposal_time_window"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _successful_code_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Identify task-specific success code without conflating it with skills.

    RATS caches the complete policy that solved a development task and may
    separately distill reusable functions into the skill library.  The former
    can contain current-frame coordinates and other episode-local choices; it
    must therefore be reported independently from reusable learned skills.
    """

    accepted: list[int] = []
    for key, value in raw.items():
        match = re.fullmatch(r"verification_attempt_(\d+)", str(key))
        if not match or not isinstance(value, dict):
            continue
        details = value.get("details") or {}
        native = details.get("native_predicate_success")
        if native is None:
            native = value.get("task_completed")
        if native is True:
            accepted.append(int(match.group(1)))
    index = max(accepted) if accepted else None
    code = raw.get(f"code_attempt_{index}") if index is not None else None
    if not isinstance(code, str):
        code = ""
    return {
        "task_specific_success_code_cached": bool(code),
        "task_specific_success_code_index": index,
        "task_specific_success_code_chars": len(code),
        "task_specific_success_code_sha256": (
            hashlib.sha256(code.encode("utf-8")).hexdigest() if code else ""
        ),
    }


def _human_visual_spot_checks(root: Path) -> pd.DataFrame:
    """Annotate audit-only visual records with active-lineage validity.

    Administrative rollback keeps every consumed rollout and human audit in
    append-only ledgers.  A rerun may reuse names such as
    ``iter012_attempt3_failed/combined.mp4``; consequently, the existence of an
    artifact path is not enough to prove that an old annotation still refers
    to the active transaction.  Mark records from explicitly invalidated
    iteration ranges instead of deleting them, so Excel retains both the full
    history and an unambiguous active-lineage view.
    """

    base_columns = [
        "schema_version",
        "time",
        "iteration",
        "attempt",
        "turn",
        "artifact_type",
        "audit_category",
        "artifact_paths",
        "observation",
        "native_task_success_evaluated",
        "native_success",
        "control_effect",
        "scope",
    ]
    rows = _jsonl(root / "human_visual_spot_checks.jsonl")
    audits = pd.DataFrame(rows)
    for column in base_columns:
        if column not in audits:
            audits[column] = None
    audits = audits[base_columns].copy()
    audits["lineage_status"] = "active"
    audits["lineage_invalidation_event"] = ""
    audits["lineage_invalidation_time_unix"] = None
    audits["artifact_path_reused_after_invalidation"] = False
    if audits.empty:
        return audits

    audit_time = pd.to_datetime(audits["time"], utc=True, errors="coerce")
    audit_unix = audit_time.map(lambda value: value.timestamp() if pd.notna(value) else None)
    audit_iteration = pd.to_numeric(audits["iteration"], errors="coerce")
    events = _jsonl(root / "administrative_events.jsonl")
    for event in events:
        event_name = str(event.get("event") or "")
        if "invalidated" not in event_name:
            continue
        start = event.get("invalidated_iteration")
        end = event.get("partial_next_iteration", start)
        event_time = event.get("time")
        try:
            start_value = int(start)
            end_value = int(end)
            event_time_value = float(event_time)
        except (TypeError, ValueError):
            continue
        mask = (
            audit_iteration.between(start_value, end_value, inclusive="both")
            & pd.Series(audit_unix, index=audits.index).le(event_time_value)
        )
        audits.loc[mask, "lineage_status"] = "invalidated_transaction"
        audits.loc[mask, "lineage_invalidation_event"] = event_name
        audits.loc[mask, "lineage_invalidation_time_unix"] = event_time_value
        for index in audits.index[mask]:
            paths = audits.at[index, "artifact_paths"]
            if not isinstance(paths, list):
                continue
            reused = False
            for raw_path in paths:
                path = Path(str(raw_path))
                try:
                    reused = reused or (
                        path.is_file() and path.stat().st_mtime > event_time_value
                    )
                except OSError:
                    continue
            audits.at[index, "artifact_path_reused_after_invalidation"] = reused
    return audits


def parse_rats(root: Path) -> pd.DataFrame:
    calls = _agent_calls(root)
    windows = _committed_iteration_windows(root, calls)
    resets = _jsonl(root / "sim_episodes.jsonl")
    rows: list[dict[str, Any]] = []
    cumulative_success = 0
    cumulative_resets = 0
    cumulative_calls = 0
    cumulative_cost = 0.0
    cumulative_wall = 0.0
    for path in sorted(root.glob("iteration_*.json")):
        raw = _read_json(path)
        iteration = int(raw.get("iteration") or len(rows) + 1)
        if iteration in windows:
            start, end = windows[iteration]
            selected_calls = [
                row for row in calls if start <= float(row.get("timestamp") or 0.0) < end
            ]
            reset_count = sum(
                start <= float(row.get("time") or 0.0) < end for row in resets
            )
        else:
            selected_calls = []
            reset_count = 0
        usage = _usage(selected_calls)
        success = bool(raw.get("success"))
        success_code = _successful_code_metadata(raw)
        cumulative_success += int(success)
        cumulative_resets += reset_count
        cumulative_calls += int(usage["model_calls"])
        cumulative_cost += float(usage["estimated_api_cost_usd"])
        elapsed = float(raw.get("elapsed_seconds") or 0.0)
        cumulative_wall += elapsed
        proposal = raw.get("task_proposal") or {}
        success_extracted = [
            str(value) for value in raw.get("skills_learned") or [] if value
        ]
        failure_proposed = [
            str(record.get("name"))
            for record in raw.get("proposed_skills") or []
            if isinstance(record, dict) and record.get("name")
        ]
        rows.append(
            {
                "iteration": iteration,
                "task": proposal.get("activity_name") or (raw.get("scene_context") or {}).get("activity_name"),
                "instruction": proposal.get("goal_conditions", ""),
                "native_success": success,
                "cumulative_play_success_rate": cumulative_success / iteration,
                "attempts": int(raw.get("total_attempts") or 0),
                "skills_learned_this_iteration": len(success_extracted),
                "native_success_extracted_skills": ",".join(success_extracted),
                "failure_proposed_experimental_skills": ",".join(failure_proposed),
                "failure_proposed_experimental_count": len(failure_proposed),
                "reusable_skills_added": ",".join(raw.get("skills_added") or []),
                "reusable_skills_reused": ",".join(raw.get("skills_reused") or []),
                "reusable_skills_failed": ",".join(raw.get("skills_failed") or []),
                "skill_usage_source": raw.get("skill_usage_source"),
                "learned_skill_count": int(raw.get("learned_skill_count") or 0),
                "active_learned_skill_count": int(raw.get("active_learned_skill_count") or 0),
                **success_code,
                "elapsed_seconds": elapsed,
                "simulator_resets": reset_count,
                "model_calls": int(usage["model_calls"]),
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "cached_tokens": int(usage["cached_tokens"]),
                "estimated_api_cost_usd": float(usage["estimated_api_cost_usd"]),
                "cumulative_resets": cumulative_resets,
                "cumulative_model_calls": cumulative_calls,
                "cumulative_estimated_api_cost_usd": cumulative_cost,
                "cumulative_elapsed_seconds": cumulative_wall,
                "artifact": str(path.resolve()),
            }
        )
    return pd.DataFrame(rows)


def parse_racap(root: Path) -> pd.DataFrame:
    state_path = root / "lineage" / "state.json"
    state = _read_json(state_path)
    rows: list[dict[str, Any]] = []
    # Lineage also records proposal/implementation failures that never reached
    # a simulator-backed paired comparison.  The experiment ledger contains
    # exactly the runtime-tested candidates (23 in the archived run).
    runtime_candidates = {
        str(row.get("candidate"))
        for row in _jsonl(root / "experience" / "EXPERIMENT_LEDGER.jsonl")
    }
    for entry in state.get("history") or []:
        if str(entry.get("candidate")) not in runtime_candidates:
            continue
        champion = entry.get("champion_metrics") or {}
        candidate = entry.get("candidate_metrics") or {}
        paired = entry.get("paired_delta") or {}
        rows.append(
            {
                "iteration": int(entry.get("iteration") or 0),
                "stage": entry.get("stage", ""),
                "status": entry.get("status", ""),
                "promoted": entry.get("status") == "capability_promoted",
                "proposal": entry.get("proposal", ""),
                "parent": entry.get("parent", ""),
                "candidate": entry.get("candidate", ""),
                "cohort_episodes": int(candidate.get("expected") or 0),
                "champion_native_success": int(champion.get("native_success") or 0),
                "candidate_native_success": int(candidate.get("native_success") or 0),
                "champion_native_rate": champion.get("native_rate"),
                "candidate_native_rate": candidate.get("native_rate"),
                "native_delta": int(paired.get("native_delta") or 0),
                "new_wins": len(paired.get("new_wins") or []),
                "regressions": len(paired.get("regressions") or []),
                "candidate_mean_turns": candidate.get("mean_turns"),
                "candidate_mean_seconds": candidate.get("mean_seconds"),
                "candidate_mean_simulator_steps": candidate.get("mean_simulator_steps"),
                "candidate_total_vlm_calls": candidate.get("total_vlm_calls"),
                "artifact": str(state_path.resolve()),
            }
        )
    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)


def _plot(rats: pd.DataFrame, racap: pd.DataFrame, output: Path) -> None:
    """Render descriptive development traces without implying matched training.

    RATS and RACaP use different update units and development procedures.  The
    panels therefore expose each lineage separately; only frozen evaluation is
    a valid head-to-head comparison.
    """

    blue = "#0072B2"
    light_blue = "#9ECAE1"
    green = "#009E73"
    orange = "#E69F00"
    red = "#D55E00"
    gray = "#7A7A7A"
    light_gray = "#D9D9D9"
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25))
    if not rats.empty:
        x = rats["iteration"].astype(int)
        success = rats["native_success"].astype(bool)
        rate = rats["cumulative_play_success_rate"].astype(float)
        axes[0, 0].step(x, rate, where="mid", color=blue, linewidth=1.8)
        axes[0, 0].scatter(
            x[success], rate[success], marker="o", color=blue, s=24,
            label="Native success",
        )
        axes[0, 0].scatter(
            x[~success], rate[~success], marker="x", color=red, s=30,
            linewidth=1.5, label="Native failure",
        )
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].set_ylabel("Cumulative success rate")
        axes[0, 0].set_title("(a) RATS selected-play outcomes")
        axes[0, 0].legend(loc="lower left", ncol=2, fontsize=6.8)

        validated = rats["skills_learned_this_iteration"].astype(int).cumsum()
        provisional = rats["failure_proposed_experimental_count"].astype(int).cumsum()
        axes[0, 1].fill_between(
            x, 0, validated, step="mid", color=light_blue, alpha=0.9,
            label="Native-success extracted",
        )
        axes[0, 1].fill_between(
            x, validated, validated + provisional, step="mid", color=orange,
            alpha=0.8, label="Failure-proposed (experimental)",
        )
        axes[0, 1].plot(
            x, rats["learned_skill_count"].astype(int), color=gray,
            marker="o", markersize=3.5, label="Recorded library total",
        )
        axes[0, 1].set_ylabel("Cumulative functions")
        axes[0, 1].set_title("(b) RATS skill-library growth")
        axes[0, 1].legend(loc="upper left", fontsize=6.4)
    if not racap.empty:
        racap = racap.reset_index(drop=True).copy()
        trial = pd.Series(range(1, len(racap) + 1), index=racap.index)
        promoted = racap["promoted"].astype(bool)
        champion_rate = racap["champion_native_rate"].astype(float)
        candidate_rate = racap["candidate_native_rate"].astype(float)
        for index in racap.index:
            axes[1, 0].plot(
                [trial[index], trial[index]],
                [champion_rate[index], candidate_rate[index]],
                color=light_gray, linewidth=1.0, zorder=1,
            )
        axes[1, 0].scatter(
            trial, champion_rate, facecolor="white", edgecolor=gray, marker="o",
            s=20, linewidth=0.9, zorder=2,
        )
        axes[1, 0].scatter(
            trial[promoted], candidate_rate[promoted], color=green, marker="^",
            s=28, zorder=3,
        )
        axes[1, 0].scatter(
            trial[~promoted], candidate_rate[~promoted], color=gray, marker="x",
            s=24, linewidth=1.2, zorder=3,
        )
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].set_ylabel("Native success rate")
        axes[1, 0].set_title("(c) RACaP paired proposal tests")
        axes[1, 0].legend(
            handles=[
                Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=gray,
                       linestyle="none", label="Parent champion"),
                Line2D([], [], marker="^", color=green, linestyle="none",
                       label="Promoted candidate"),
                Line2D([], [], marker="x", color=gray, linestyle="none",
                       label="Retained parent"),
            ],
            loc="lower left", ncol=3, fontsize=6.2, columnspacing=0.8,
            handletextpad=0.35,
        )

        delta = racap["native_delta"].astype(int)
        colors = [green if keep else red if value < 0 else gray
                  for keep, value in zip(promoted, delta)]
        axes[1, 1].bar(trial, delta, color=colors, width=0.72)
        axes[1, 1].axhline(0, color="black", linewidth=0.8)
        axes[1, 1].set_ylabel(r"Candidate $-$ parent successes")
        axes[1, 1].set_title("(d) RACaP promotion evidence")
        axes[1, 1].legend(
            handles=[
                Patch(facecolor=green, label="Promoted"),
                Patch(facecolor=red, label="Regression"),
                Patch(facecolor=gray, label="No gain"),
            ],
            loc="lower left", ncol=3, fontsize=6.4, columnspacing=0.8,
            handlelength=1.1,
        )

        # Delineate curriculum cohorts without a ten-color legend.  Short labels
        # identify contiguous blocks; the spreadsheet preserves full names.
        stage_short = {
            "s0_direct_transport": "S0",
            "s1_transport_react": "S1",
            "s2_bounded_placement": "S2",
            "s3a_insert": "S3a",
            "s3b_stack": "S3b",
            "s3c_articulation": "S3c",
            "s3d_control": "S3d",
            "s4_tool_routing": "S4",
            "s5_causal_composition": "S5",
            "s6_efficiency": "S6",
        }
        starts: list[tuple[int, int, str]] = []
        start = 1
        stages = racap["stage"].astype(str).tolist()
        for position in range(2, len(stages) + 2):
            boundary = position == len(stages) + 1 or stages[position - 1] != stages[position - 2]
            if boundary:
                end = position - 1
                starts.append((start, end, stages[start - 1]))
                start = position
        for axis in (axes[1, 0], axes[1, 1]):
            for start, end, stage in starts:
                if start > 1:
                    axis.axvline(start - 0.5, color="#BDBDBD", linestyle=":", linewidth=0.7)
                axis.text(
                    (start + end) / 2, 0.985, stage_short.get(stage, stage),
                    transform=axis.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6.1, color="#4D4D4D", clip_on=True,
                )
            tick_positions = list(range(1, len(racap) + 1, 2))
            axis.set_xticks(tick_positions)
            axis.set_xticklabels(
                [str(int(racap.loc[value - 1, "iteration"])) for value in tick_positions]
            )
            axis.set_xlabel("Tested proposal (archived iteration ID)")

    axes[0, 0].set_xlabel("RATS update round")
    axes[0, 1].set_xlabel("RATS update round")
    for axis in axes.flat:
        clean_axis(axis, grid_axis="y")
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.095, top=0.94, wspace=0.28, hspace=0.42)
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rats-root", type=Path, default=DEFAULT_RATS)
    parser.add_argument(
        "--rats-invalid-root", type=Path, default=DEFAULT_RATS_INVALID
    )
    parser.add_argument("--racap-root", type=Path, default=DEFAULT_RACAP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "controlled_comparison" / "analysis" / "development",
    )
    args = parser.parse_args()
    progress = _development_progress(args.rats_root)
    rats = parse_rats(args.rats_root)
    racap = parse_racap(args.racap_root)
    rats_calls = _agent_calls(args.rats_root)
    rats_categories, rats_identity = _rats_cost_breakdowns(args.rats_root)
    rats_observed = _rats_observed_usage(args.rats_root)
    rats_invalidated = _rats_invalidated_usage(args.rats_invalid_root)
    rats_resource_accounting = _rats_resource_accounting(
        rats, rats_observed, rats_invalidated
    )
    rats_attempts = _rats_attempt_diagnostics(args.rats_root)
    native_authority_mismatches = _native_authority_mismatches(rats_attempts)
    rats_self_checks = _rats_runtime_self_checks(args.rats_root)
    rats_native_events = _native_event_audit(args.rats_root)
    rats_prompt_privilege = _prompt_privilege_audit(rats_calls)
    rats_resume_transactions = _resume_transaction_audit(args.rats_root)
    administrative_events = pd.DataFrame(
        _jsonl(args.rats_root / "administrative_events.jsonl"),
        columns=[
            "schema_version",
            "time",
            "event",
            "parent_pid",
            "source_provenance_captured_at_unix",
            "reason",
            "resolution",
            "simulator_resets_attributed",
            "model_calls_attributed",
            "strategy_state_restored_or_committed",
            "original_development_process_retained",
            "original_development_parent_pid",
            "excluded_from_development_round_count",
            "completed_iteration_boundary_before_invalidation",
            "restored_iteration_boundary",
            "invalidated_iteration",
            "partial_next_iteration",
            "archived_transaction",
        ],
    )
    human_visual_spot_checks = _human_visual_spot_checks(args.rats_root)
    active_human_visual_spot_checks = human_visual_spot_checks[
        human_visual_spot_checks["lineage_status"] == "active"
    ]
    critic_evidence_audits = _critic_evidence_audits(args.rats_root)
    active_critic_conflicts = critic_evidence_audits[
        critic_evidence_audits.get(
            "effective_status", pd.Series(index=critic_evidence_audits.index, dtype=str)
        )
        == "active_conflict"
    ]
    skill_admission_audit = _skill_admission_audit(args.rats_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rats.to_csv(args.output_dir / "rats_development.csv", index=False)
    racap.to_csv(args.output_dir / "racap_development.csv", index=False)
    rats_categories.to_csv(args.output_dir / "rats_call_categories.csv", index=False)
    rats_observed.to_csv(args.output_dir / "rats_observed_usage.csv", index=False)
    rats_invalidated.to_csv(
        args.output_dir / "rats_invalidated_lineages.csv", index=False
    )
    rats_resource_accounting.to_csv(
        args.output_dir / "rats_resource_accounting.csv", index=False
    )
    rats_identity.to_csv(args.output_dir / "rats_identity_search.csv", index=False)
    rats_attempts.to_csv(args.output_dir / "rats_attempt_diagnostics.csv", index=False)
    native_authority_mismatches.to_csv(
        args.output_dir / "rats_native_authority_mismatches.csv", index=False
    )
    rats_self_checks.to_csv(args.output_dir / "rats_runtime_self_checks.csv", index=False)
    rats_native_events.to_csv(args.output_dir / "rats_native_event_audit.csv", index=False)
    rats_prompt_privilege.to_csv(
        args.output_dir / "rats_prompt_privilege_audit.csv", index=False
    )
    rats_resume_transactions.to_csv(
        args.output_dir / "rats_resume_transactions.csv", index=False
    )
    administrative_events.to_csv(
        args.output_dir / "rats_administrative_events.csv", index=False
    )
    human_visual_spot_checks.to_csv(
        args.output_dir / "rats_human_visual_spot_checks.csv", index=False
    )
    critic_evidence_audits.to_csv(
        args.output_dir / "rats_critic_evidence_audits.csv", index=False
    )
    skill_admission_audit.to_csv(
        args.output_dir / "rats_skill_admission_audit.csv", index=False
    )
    with pd.ExcelWriter(args.output_dir / "development_audit.xlsx", engine="xlsxwriter") as writer:
        rats.to_excel(writer, sheet_name="RATS", index=False)
        racap.to_excel(writer, sheet_name="RACaP", index=False)
        rats_categories.to_excel(writer, sheet_name="RATS call categories", index=False)
        rats_observed.to_excel(writer, sheet_name="RATS observed usage", index=False)
        rats_invalidated.to_excel(
            writer, sheet_name="RATS invalidated lineages", index=False
        )
        rats_resource_accounting.to_excel(
            writer, sheet_name="RATS resource accounting", index=False
        )
        rats_identity.to_excel(writer, sheet_name="RATS identity search", index=False)
        rats_attempts.to_excel(writer, sheet_name="RATS attempt diagnostics", index=False)
        native_authority_mismatches.to_excel(
            writer, sheet_name="RATS native authority", index=False
        )
        rats_self_checks.to_excel(writer, sheet_name="RATS runtime self-checks", index=False)
        rats_native_events.to_excel(writer, sheet_name="RATS native events", index=False)
        rats_prompt_privilege.to_excel(
            writer, sheet_name="RATS prompt privilege", index=False
        )
        rats_resume_transactions.to_excel(
            writer, sheet_name="RATS resume transactions", index=False
        )
        administrative_events.to_excel(
            writer, sheet_name="RATS administrative events", index=False
        )
        human_visual_spot_checks.to_excel(
            writer, sheet_name="RATS human visual audits", index=False
        )
        critic_evidence_audits.to_excel(
            writer, sheet_name="RATS critic evidence", index=False
        )
        skill_admission_audit.to_excel(
            writer, sheet_name="RATS skill admission", index=False
        )
        pd.DataFrame(
            [
                {
                    "warning": (
                        "Training-time rates are not a direct comparison: RATS measures "
                        "selected play tasks, while RACaP measures changing registered "
                        "curriculum cohorts. Compare systems only on frozen evaluation."
                        " RATS runtime self-check 'passed' means code execution completed "
                        "without an exception; it is never a native task-success claim."
                        " A raw-key mismatch marked expected_rebind_label_lag is the first"
                        " pre-rebind telemetry label only; simulator_runtime task identity"
                        " is authoritative and must match the registered task. Functions"
                        " extracted from successful code and experimental helpers proposed"
                        " from failures are reported in separate columns. Human visual"
                        " audits are append-only; lineage_status marks annotations from"
                        " invalidated transactions, including stale paths later reused by"
                        " a rerun, and only active records enter control-effect counts."
                        " Invalidated pre-protocol RATS lineages never enter success or"
                        " skill metrics, but their resets, calls, tokens, latency, and"
                        " estimated cost are retained in the engineering-total resource"
                        " row so discarded implementation work cannot disappear."
                    )
                }
            ]
        ).to_excel(writer, sheet_name="ReadMe", index=False)
    _plot(rats, racap, args.output_dir / "figures" / "development_curves")
    summary = {
        "rats_iterations_complete": len(rats),
        "rats_latest_committed_iteration": progress["latest_committed_round"],
        "rats_active_iteration": progress["active_round"],
        "rats_target_iterations": progress["target_rounds"],
        "rats_development_status": progress["status"],
        "rats_native_successes": int(rats.native_success.sum()) if not rats.empty else 0,
        "rats_native_success_extracted_skills": (
            int(rats.skills_learned_this_iteration.sum()) if not rats.empty else 0
        ),
        "rats_failure_proposed_experimental_skills": (
            int(rats.failure_proposed_experimental_count.sum())
            if not rats.empty
            else 0
        ),
        "rats_simulator_resets": int(rats.simulator_resets.sum()) if not rats.empty else 0,
        "rats_model_calls": int(rats.model_calls.sum()) if not rats.empty else 0,
        "rats_estimated_api_cost_usd": float(rats.estimated_api_cost_usd.sum()) if not rats.empty else 0.0,
        "rats_identity_verification_calls": int(rats_identity.calls.sum()) if not rats_identity.empty else 0,
        "rats_identity_calls_after_first_true": (
            int(rats_identity.calls_after_first_true.sum()) if not rats_identity.empty else 0
        ),
        "rats_executions_audited": len(rats_attempts),
        "rats_native_authority_mismatches": len(native_authority_mismatches),
        "rats_native_authority_invariant_pass": native_authority_mismatches.empty,
        "rats_runtime_self_checks": len(rats_self_checks),
        "rats_execution_timeouts": (
            int(rats_attempts.execution_timeout.sum()) if not rats_attempts.empty else 0
        ),
        "rats_terminated_episode_retries": (
            int(rats_attempts.terminated_episode_retry.sum()) if not rats_attempts.empty else 0
        ),
        "rats_native_visual_disagreements": (
            int((rats_attempts.diagnostic_disagreement == True).sum())  # noqa: E712
            if not rats_attempts.empty
            else 0
        ),
        "rats_native_events_indexed": len(rats_native_events),
        "rats_native_event_raw_key_mismatches": (
            int((rats_native_events.raw_key_task_matches == False).sum())  # noqa: E712
            if not rats_native_events.empty
            else 0
        ),
        "rats_native_event_expected_rebind_label_lags": (
            int(rats_native_events.expected_rebind_label_lag.sum())
            if not rats_native_events.empty
            else 0
        ),
        "rats_native_event_unexplained_key_mismatches": (
            int(
                (
                    (rats_native_events.raw_key_task_matches == False)  # noqa: E712
                    & ~rats_native_events.expected_rebind_label_lag.astype(bool)
                ).sum()
            )
            if not rats_native_events.empty
            else 0
        ),
        "rats_model_requests_privilege_scanned": len(rats_calls),
        "rats_simulator_resets_observed": int(
            rats_observed.iloc[0].simulator_resets_observed
        ),
        "rats_model_calls_observed": int(rats_observed.iloc[0].model_calls),
        "rats_prompt_tokens_observed": int(rats_observed.iloc[0].prompt_tokens),
        "rats_completion_tokens_observed": int(
            rats_observed.iloc[0].completion_tokens
        ),
        "rats_cached_tokens_observed": int(rats_observed.iloc[0].cached_tokens),
        "rats_model_latency_seconds_observed": float(
            rats_observed.iloc[0].model_latency_seconds
        ),
        "rats_estimated_api_cost_usd_observed": float(
            rats_observed.iloc[0].estimated_api_cost_usd
        ),
        "rats_actual_models_observed": str(rats_observed.iloc[0].actual_models),
        "rats_invalidated_lineages": len(rats_invalidated),
        "rats_invalidated_simulator_resets": int(
            rats_invalidated.simulator_resets_observed.sum()
        ),
        "rats_invalidated_model_calls": int(rats_invalidated.model_calls.sum()),
        "rats_invalidated_prompt_tokens": int(
            rats_invalidated.prompt_tokens.sum()
        ),
        "rats_invalidated_completion_tokens": int(
            rats_invalidated.completion_tokens.sum()
        ),
        "rats_invalidated_cached_tokens": int(
            rats_invalidated.cached_tokens.sum()
        ),
        "rats_invalidated_estimated_api_cost_usd": float(
            rats_invalidated.estimated_api_cost_usd.sum()
        ),
        "rats_engineering_total_simulator_resets": int(
            rats_resource_accounting.loc[
                rats_resource_accounting.scope
                == "all_recorded_engineering_consumption",
                "simulator_resets",
            ].iloc[0]
        ),
        "rats_engineering_total_model_calls": int(
            rats_resource_accounting.loc[
                rats_resource_accounting.scope
                == "all_recorded_engineering_consumption",
                "model_calls",
            ].iloc[0]
        ),
        "rats_engineering_total_estimated_api_cost_usd": float(
            rats_resource_accounting.loc[
                rats_resource_accounting.scope
                == "all_recorded_engineering_consumption",
                "estimated_api_cost_usd",
            ].iloc[0]
        ),
        "rats_forbidden_prompt_payload_hits": len(rats_prompt_privilege),
        "rats_resume_transactions": len(rats_resume_transactions),
        "rats_resume_transaction_failures": (
            int((rats_resume_transactions.audit_pass == False).sum())  # noqa: E712
            if not rats_resume_transactions.empty
            else 0
        ),
        "rats_human_visual_spot_checks": len(human_visual_spot_checks),
        "rats_human_visual_spot_checks_active_lineage": len(
            active_human_visual_spot_checks
        ),
        "rats_human_visual_spot_checks_invalidated": int(
            (human_visual_spot_checks.lineage_status == "invalidated_transaction").sum()
        ),
        "rats_human_visual_spot_checks_reused_artifact_paths": int(
            human_visual_spot_checks.artifact_path_reused_after_invalidation.sum()
        ),
        "rats_human_visual_spot_checks_changed_control": (
            int(active_human_visual_spot_checks.control_effect.fillna(False).sum())
            if not active_human_visual_spot_checks.empty
            else 0
        ),
        "rats_critic_evidence_records_audited": len(critic_evidence_audits),
        "rats_critic_evidence_corrections": int(
            (critic_evidence_audits.effective_status == "correction").sum()
        ),
        "rats_critic_evidence_retractions": int(
            (critic_evidence_audits.effective_status == "retracted").sum()
        ),
        "rats_critic_evidence_conflicts_audited": len(active_critic_conflicts),
        "rats_critic_evidence_conflicts_changed_algorithm_mid_chain": (
            int(active_critic_conflicts.algorithm_changed_mid_chain.fillna(False).sum())
            if not active_critic_conflicts.empty
            else 0
        ),
        "rats_skill_admission_rounds_audited": len(skill_admission_audit),
        "rats_skill_admission_failures": (
            int((skill_admission_audit.audit_pass == False).sum())  # noqa: E712
            if not skill_admission_audit.empty
            else 0
        ),
        "racap_lineage_attempts": len(
            (_read_json(args.racap_root / "lineage" / "state.json").get("history") or [])
        ),
        "racap_runtime_tested_candidates": len(racap),
        "racap_promotions": int(racap.promoted.sum()) if not racap.empty else 0,
        "direct_comparison_allowed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not native_authority_mismatches.empty:
        raise SystemExit(
            "native predicate authority invariant failed: "
            f"{len(native_authority_mismatches)} persisted verifier record(s) "
            "mark native success as outer failure"
        )


if __name__ == "__main__":
    main()
