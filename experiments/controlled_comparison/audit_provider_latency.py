#!/usr/bin/env python3
"""Audit hosted-model latency without using task outcomes.

The controlled protocol gives each episode 1000 seconds of policy wall time.
If the suite-level P90 of successful hosted-model requests exceeds 100 seconds,
ten otherwise successful requests can consume the complete episode budget.  We
therefore treat that condition as an infrastructure outage, independently of
native success, and archive/re-run the whole affected suite.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def _nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("latency sample is empty")
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def latencies_from_run_dir(run_dir: Path) -> list[float]:
    values: list[float] = []
    seen_request_ids: set[str] = set()
    # RACaP writes worker telemetry directly in the suite directory, whereas
    # generated-policy baselines isolate each registered episode under a
    # ``seed_*`` directory.  The audit is method-agnostic, so it must cover
    # both layouts.  Filenames are still constrained to the frozen telemetry
    # contract to avoid accidentally ingesting copied analysis artifacts.
    for path in sorted(run_dir.rglob("llm_calls*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "network_success" or row.get("elapsed_s") is None:
                continue
            request_id = str(row.get("request_id") or "").strip()
            if request_id and request_id in seen_request_ids:
                continue
            if request_id:
                seen_request_ids.add(request_id)
            values.append(float(row["elapsed_s"]))
    # RATS records one structured JSON file per hosted-model request.  These
    # files carry the same request id, actual model, usage, and elapsed time as
    # flat network telemetry but live under ``seed_*/agent_io``.  Include them
    # so the infrastructure gate covers every method family, while request-id
    # de-duplication protects layouts that retain both formats.
    for path in sorted(run_dir.rglob("agent_io/*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict) or row.get("elapsed_s") is None:
            continue
        request_id = str(row.get("request_id") or "").strip()
        if request_id and request_id in seen_request_ids:
            continue
        if request_id:
            seen_request_ids.add(request_id)
        values.append(float(row["elapsed_s"]))
    return values


def latencies_from_episode_frame(frame: pd.DataFrame) -> list[float]:
    values: list[float] = []
    for raw in frame.get("model_call_latency_samples_json", pd.Series(dtype=str)).fillna("[]"):
        try:
            samples = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        if isinstance(samples, list):
            values.extend(float(value) for value in samples)
    return values


def audit_latencies(
    values: Iterable[float],
    *,
    minimum_requests: int = 30,
    maximum_p90_seconds: float = 100.0,
) -> dict[str, object]:
    samples = sorted(float(value) for value in values)
    if len(samples) < minimum_requests:
        status = "insufficient_data"
        p90 = None
    else:
        p90 = _nearest_rank(samples, 0.90)
        status = "pass" if p90 <= maximum_p90_seconds else "infrastructure_invalid"
    return {
        "schema_version": 1,
        "status": status,
        "outcome_independent": True,
        "successful_requests": len(samples),
        "minimum_requests": minimum_requests,
        "maximum_p90_seconds": maximum_p90_seconds,
        "median_seconds": _nearest_rank(samples, 0.50) if samples else None,
        "p90_seconds": p90,
        "p99_seconds": _nearest_rank(samples, 0.99) if samples else None,
        "max_seconds": max(samples) if samples else None,
        "requests_over_60_seconds": sum(value > 60 for value in samples),
        "rationale": (
            "At P90 > 100 seconds, ten successful model calls can consume the "
            "registered 1000-second episode policy budget."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--episodes-csv", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--suite")
    parser.add_argument("--minimum-requests", type=int, default=30)
    parser.add_argument("--maximum-p90-seconds", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.run_dir is not None:
        values = latencies_from_run_dir(args.run_dir)
        source_label = str(args.run_dir.resolve())
    else:
        frame = pd.read_csv(args.episodes_csv)
        if args.method:
            frame = frame[frame["method"] == args.method]
        if args.suite:
            frame = frame[frame["suite"] == args.suite]
        values = latencies_from_episode_frame(frame)
        source_label = str(args.episodes_csv.resolve())
    result = audit_latencies(
        values,
        minimum_requests=args.minimum_requests,
        maximum_p90_seconds=args.maximum_p90_seconds,
    )
    result["source"] = source_label
    result["method"] = args.method
    result["suite"] = args.suite
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 2 if result["status"] == "infrastructure_invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
