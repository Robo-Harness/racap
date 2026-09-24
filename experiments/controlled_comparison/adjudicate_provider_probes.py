#!/usr/bin/env python3
"""Combine consecutive outcome-free probes into one recovery decision."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments.controlled_comparison.audit_provider_latency import audit_latencies


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjudicate(
    probe_paths: list[Path],
    *,
    heavy_request_prompt_tokens: float = 6000.0,
    minimum_heavy_requests: int = 3,
    maximum_p90_seconds: float = 60.0,
    maximum_window_seconds: float = 900.0,
) -> dict[str, object]:
    if not probe_paths:
        raise ValueError("at least one probe is required")
    probes = [json.loads(path.read_text(encoding="utf-8")) for path in probe_paths]
    models = {str(probe.get("model", "")) for probe in probes}
    profiles = {str(probe.get("profile", "")) for probe in probes}
    mtimes = [path.stat().st_mtime for path in probe_paths]
    rows = [row for probe in probes for row in probe.get("rows", [])]
    successful = [row for row in rows if row.get("success")]
    actual_models = {str(row.get("actual_model", "")) for row in successful}
    usage_complete = all(
        isinstance(row.get("usage"), dict)
        and row["usage"].get("prompt_tokens") is not None
        for row in successful
    )
    heavy = [
        row
        for row in successful
        if isinstance(row.get("usage"), dict)
        and float(row["usage"].get("prompt_tokens", -1)) >= heavy_request_prompt_tokens
    ]
    all_audit = audit_latencies(
        [float(row["elapsed_s"]) for row in successful],
        minimum_requests=len(rows),
        maximum_p90_seconds=maximum_p90_seconds,
    )
    heavy_audit = audit_latencies(
        [float(row["elapsed_s"]) for row in heavy],
        minimum_requests=minimum_heavy_requests,
        maximum_p90_seconds=maximum_p90_seconds,
    )
    window_seconds = max(mtimes) - min(mtimes)
    expected_model = next(iter(models)) if len(models) == 1 else ""
    pass_gate = (
        len(models) == 1
        and len(profiles) == 1
        and profiles == {"representative_state_decision"}
        and len(successful) == len(rows)
        and usage_complete
        and actual_models == {expected_model}
        and window_seconds <= maximum_window_seconds
        and all_audit["status"] == "pass"
        and heavy_audit["status"] == "pass"
    )
    return {
        "schema_version": 1,
        "kind": "outcome_free_provider_recovery_adjudication",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if pass_gate else "insufficient_or_unhealthy",
        "outcome_independent": True,
        "probe_count": len(probes),
        "total_requests": len(rows),
        "successful_requests": len(successful),
        "model": expected_model,
        "actual_models": sorted(actual_models),
        "profiles": sorted(profiles),
        "window_seconds": window_seconds,
        "maximum_window_seconds": maximum_window_seconds,
        "heavy_request_prompt_tokens": heavy_request_prompt_tokens,
        "heavy_successful_requests": len(heavy),
        "minimum_heavy_requests": minimum_heavy_requests,
        "latency_audit": all_audit,
        "heavy_request_latency_audit": heavy_audit,
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in probe_paths
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, action="append", required=True)
    parser.add_argument("--heavy-request-prompt-tokens", type=float, default=6000.0)
    parser.add_argument("--minimum-heavy-requests", type=int, default=3)
    parser.add_argument("--maximum-p90-seconds", type=float, default=60.0)
    parser.add_argument("--maximum-window-seconds", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = adjudicate(
        args.probe,
        heavy_request_prompt_tokens=args.heavy_request_prompt_tokens,
        minimum_heavy_requests=args.minimum_heavy_requests,
        maximum_p90_seconds=args.maximum_p90_seconds,
        maximum_window_seconds=args.maximum_window_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
