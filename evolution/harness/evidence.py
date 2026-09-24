"""Load evaluator-owned evidence and reject incomplete or stale rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .schema import EpisodeRef, Metrics


class EvidenceError(RuntimeError):
    pass


_EXTERNAL_FAILURE_MARKERS = (
    "insufficient_user_quota",
    "need pre-deduct",
    "insufficient balance",
    "quota exceeded",
    "billing quota",
    "par_vapi_key is unset",
    "routes to the relay but",
)


def _external_failure_marker(row: dict[str, Any]) -> str | None:
    """Return an external-service marker embedded anywhere in a record.

    Runtime model failures are sometimes serialized into an otherwise valid
    episode record instead of making the evaluator process fail.  Such an
    episode is missing experimental evidence; counting it as a native failure
    silently rewards candidates that merely avoid model calls.
    """
    serialized = json.dumps(row, ensure_ascii=False, sort_keys=True).lower()
    return next((marker for marker in _EXTERNAL_FAILURE_MARKERS if marker in serialized), None)


def _infrastructure_failure(row: dict[str, Any]) -> str | None:
    llm = row.get("evaluator_llm_calls") or {}
    if int(llm.get("terminal_failures", 0)) > 0:
        return f"llm_terminal_failures={int(llm['terminal_failures'])}"
    for step in row.get("steps") or []:
        failure = (step.get("report") or {}).get("infrastructure_failure")
        if failure:
            return json.dumps(failure, ensure_ascii=False, sort_keys=True)
    return None


def _systemic_grounding_outage(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Recognise legacy swallowed-provider failures from their exact footprint.

    New records carry typed failures.  This detector also protects old or
    candidate-owned controllers: repeated no-motion grounding failures plus a
    retry-amplified network ratio are not evidence about manipulation quality.
    """
    suspicious: dict[str, str] = {}
    for row in rows:
        transports = [
            step
            for step in (row.get("steps") or [])
            if step.get("action") in {"pickplace", "insert", "stack"}
        ]
        if len(transports) < 3:
            continue
        if any(
            str((step.get("report") or {}).get("failure_mode", "")) != "not_grounded"
            for step in transports
        ):
            continue
        stats = row.get("evaluator_llm_calls") or {}
        logical = int(stats.get("logical_calls", 0))
        network = int(stats.get("network_attempts", 0))
        simulator_steps = int(row.get("simulator_steps", 0))
        if logical >= 6 and network >= 2 * logical and simulator_steps <= 100:
            suspicious[str(row.get("key", ""))] = (
                f"{len(transports)} repeated not_grounded calls, "
                f"llm={logical}/{network}, simulator_steps={simulator_steps}"
            )
    return suspicious


def _count_calls(row: dict[str, Any]) -> tuple[int, int]:
    runtime_calls = row.get("evaluator_runtime_calls") or {}
    llm_calls = row.get("evaluator_llm_calls") or {}
    if runtime_calls or llm_calls:
        return int(runtime_calls.get("motion_calls", 0)), int(llm_calls.get("logical_calls", 0))
    steps = row.get("steps") or []
    tool_calls = len([step for step in steps if step.get("action") not in (None, "done")])
    vlm = len(row.get("reflections") or [])
    for step in steps:
        report = step.get("report") or {}
        for key in ("vlm_calls", "grounding_calls", "semantic_calls"):
            value = report.get(key)
            if isinstance(value, int):
                vlm += value
    return tool_calls, vlm


def load_metrics(output_dir: Path, expected: Iterable[EpisodeRef]) -> Metrics:
    records_path = output_dir / "records.jsonl"
    if not records_path.is_file():
        raise EvidenceError(f"missing evaluator records: {records_path}")
    raw = records_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    expected_by_key = {episode.key: episode for episode in expected}
    actual_by_key: dict[str, dict] = {}
    for row in rows:
        key = str(row.get("key", ""))
        if key in actual_by_key:
            raise EvidenceError(f"duplicate evaluator record: {key}")
        actual_by_key[key] = row
    if set(actual_by_key) != set(expected_by_key):
        missing = sorted(set(expected_by_key) - set(actual_by_key))
        extra = sorted(set(actual_by_key) - set(expected_by_key))
        raise EvidenceError(f"episode mismatch; missing={missing}, extra={extra}")
    errors = {key: row.get("error") for key, row in actual_by_key.items() if row.get("error")}
    if errors:
        raise EvidenceError(f"episodes contain runtime errors: {errors}")
    external_failures = {
        key: marker
        for key, row in actual_by_key.items()
        if (marker := _external_failure_marker(row)) is not None
    }
    if external_failures:
        raise EvidenceError(
            "episodes contain external service failures and are not valid evaluation "
            f"evidence: {external_failures}"
        )

    infrastructure_failures = {
        key: failure
        for key, row in actual_by_key.items()
        if (failure := _infrastructure_failure(row)) is not None
    }
    if infrastructure_failures:
        raise EvidenceError(
            "episodes contain perception/model infrastructure failures and are not "
            f"valid evaluation evidence: {infrastructure_failures}"
        )
    outage = _systemic_grounding_outage(list(actual_by_key.values()))
    if outage:
        raise EvidenceError(
            "evaluation health gate detected retry-amplified grounding outage; "
            f"results are inconclusive: {outage}"
        )

    ordered = [actual_by_key[key] for key in sorted(expected_by_key)]
    successes = tuple(row["key"] for row in ordered if bool(row.get("native_success")))
    failures = tuple(row["key"] for row in ordered if not bool(row.get("native_success")))
    calls = [_count_calls(row) for row in ordered]
    return Metrics(
        expected=len(ordered),
        native_success=len(successes),
        agent_claimed=sum(bool(row.get("agent_success")) for row in ordered),
        mean_turns=sum(float(row.get("turns", 0)) for row in ordered) / len(ordered),
        mean_seconds=sum(float(row.get("seconds", 0)) for row in ordered) / len(ordered),
        mean_simulator_steps=sum(float(row.get("simulator_steps", 0)) for row in ordered)
        / len(ordered),
        total_tool_calls=sum(value[0] for value in calls),
        total_vlm_calls=sum(value[1] for value in calls),
        successes=successes,
        failures=failures,
        records_path=str(records_path.resolve()),
        digest=hashlib.sha256(raw).hexdigest(),
    )
