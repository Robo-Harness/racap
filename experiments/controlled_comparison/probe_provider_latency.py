#!/usr/bin/env python3
"""Run an outcome-free concurrent multimodal latency health probe.

The default profile mirrors the expensive request class used by the runtime
mechanism-state judge: the production prompt and memory, four visual views,
and the same output budget.  A one-line/24-token probe can pass while real
multi-view decisions take minutes, so it is available only as an explicitly
lightweight diagnostic and is not the default recovery gate.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import mimetypes
import os
from pathlib import Path
import time

import requests

from experiments.controlled_comparison.audit_provider_latency import audit_latencies


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _probe_payload(index: int, *, image_url: str, model: str, profile: str) -> dict[str, object]:
    if profile == "representative_state_decision":
        # Import production text instead of keeping a probe-only copy that can
        # drift. This loads configuration only; it does not create a simulator
        # or inspect a benchmark state.
        from racap.agent.experience import load_runtime_memory, with_runtime_memory
        from racap.agent.full_react import STATE_DECIDE

        system = "You decide robot task state from vision. JSON only."
        synthetic_goals = (
            {"skill": "articulate", "target": "storage fixture", "goal": "open"},
            {"skill": "articulate", "target": "storage fixture", "goal": "closed"},
            {"skill": "actuate_control", "target": "appliance", "goal": "on"},
            {"skill": "actuate_control", "target": "appliance", "goal": "off"},
        )
        goal = synthetic_goals[index % len(synthetic_goals)]
        prompt = with_runtime_memory(
            STATE_DECIDE.format(
                goal=json.dumps(goal),
                action="synthetic provider-health motion",
                report=json.dumps({"status": "probe", "motion": "unknown"}),
                check_report=json.dumps({"status": "advisory probe"}),
                attempted_contacts=json.dumps(["auto"]),
            ),
            load_runtime_memory(scope="full"),
        )
        max_tokens = 1400
        image_count = 4
    elif profile == "lightweight":
        system = "This is a provider latency health probe. Answer briefly."
        prompt = f"Probe {index}: name one visible object."
        max_tokens = 24
        image_count = 1
    else:  # pragma: no cover - argparse constrains this at the CLI boundary.
        raise ValueError(f"unknown probe profile: {profile}")

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[
                        {"type": "image_url", "image_url": {"url": image_url}}
                        for _ in range(image_count)
                    ],
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def _request(
    index: int,
    *,
    image_url: str,
    model: str,
    timeout: float,
    profile: str,
) -> dict[str, object]:
    base = os.environ["RACAP_VAPI_BASE"].rstrip("/")
    key = os.environ["RACAP_VAPI_KEY"]
    body = _probe_payload(index, image_url=image_url, model=model, profile=profile)
    started = time.monotonic()
    try:
        response = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
        elapsed = time.monotonic() - started
        payload = response.json() if response.content else {}
        return {
            "probe_index": index,
            "status_code": response.status_code,
            "elapsed_s": elapsed,
            "actual_model": payload.get("model", "") if isinstance(payload, dict) else "",
            "request_id": payload.get("id", "") if isinstance(payload, dict) else "",
            "usage": payload.get("usage", {}) if isinstance(payload, dict) else {},
            "success": response.status_code == 200,
        }
    except requests.RequestException as exc:
        return {
            "probe_index": index,
            "status_code": None,
            "elapsed_s": time.monotonic() - started,
            "actual_model": "",
            "request_id": "",
            "usage": {},
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--healthy-p90-seconds", type=float, default=60.0)
    parser.add_argument(
        "--profile",
        choices=("representative_state_decision", "lightweight"),
        default="representative_state_decision",
    )
    parser.add_argument(
        "--heavy-request-prompt-tokens",
        type=float,
        default=6000.0,
        help="Prompt-token boundary for the production request class that exposed the outage.",
    )
    parser.add_argument(
        "--minimum-heavy-requests",
        type=int,
        default=3,
        help="How many heavy successful requests must be observed before declaring recovery.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("RACAP_VAPI_BASE") or not os.environ.get("RACAP_VAPI_KEY"):
        raise SystemExit("RACAP_VAPI_BASE and RACAP_VAPI_KEY must be configured")
    image_url = _data_url(args.image)
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _request,
                index,
                image_url=image_url,
                model=args.model,
                timeout=args.timeout,
                profile=args.profile,
            )
            for index in range(args.requests)
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row["probe_index"]))
    successful = [float(row["elapsed_s"]) for row in rows if row["success"]]
    audit = audit_latencies(
        successful,
        minimum_requests=args.requests,
        maximum_p90_seconds=args.healthy_p90_seconds,
    )
    actual_models = sorted({str(row["actual_model"]) for row in rows if row["actual_model"]})
    usage_rows = [
        row
        for row in rows
        if row["success"]
        and isinstance(row.get("usage"), dict)
        and row["usage"].get("prompt_tokens") is not None
    ]
    prompt_tokens = sorted(float(row["usage"]["prompt_tokens"]) for row in usage_rows)
    median_prompt_tokens = (
        prompt_tokens[(len(prompt_tokens) - 1) // 2] if prompt_tokens else None
    )
    heavy_rows = [
        row
        for row in usage_rows
        if float(row["usage"]["prompt_tokens"]) >= args.heavy_request_prompt_tokens
    ]
    required_heavy_requests = (
        args.minimum_heavy_requests if args.profile == "representative_state_decision" else 0
    )
    heavy_audit = audit_latencies(
        [float(row["elapsed_s"]) for row in heavy_rows],
        minimum_requests=max(1, required_heavy_requests),
        maximum_p90_seconds=args.healthy_p90_seconds,
    )
    payload_integrity = len(usage_rows) == args.requests and (
        len(heavy_rows) >= required_heavy_requests
    )
    healthy = (
        audit["status"] == "pass"
        and len(successful) == args.requests
        and actual_models == [args.model]
        and payload_integrity
        and (required_heavy_requests == 0 or heavy_audit["status"] == "pass")
    )
    payload = {
        "schema_version": 2,
        "kind": "outcome_free_multimodal_provider_health_probe",
        "model": args.model,
        "profile": args.profile,
        "workers": args.workers,
        "requests": args.requests,
        "actual_models": actual_models,
        "payload_integrity": {
            "pass": payload_integrity,
            "successful_usage_records": len(usage_rows),
            "median_prompt_tokens": median_prompt_tokens,
            "heavy_request_prompt_tokens": args.heavy_request_prompt_tokens,
            "heavy_successful_requests": len(heavy_rows),
            "minimum_heavy_requests": required_heavy_requests,
        },
        "healthy": healthy,
        "latency_audit": audit,
        "heavy_request_latency_audit": heavy_audit,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if healthy else 2


if __name__ == "__main__":
    raise SystemExit(main())
