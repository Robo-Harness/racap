#!/usr/bin/env python3
"""Summarize structured RATS evaluation failures without scanning policy source.

The analyzer deliberately reads only the latest structured diagnosis and its
matching execution record.  It never treats an arbitrary substring elsewhere
in a generated program, prompt, or trace as evidence that an error occurred.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ATTEMPT_PATTERN = re.compile(r"^(diagnosis|execution)_attempt_(\d+)$")
SIGNATURES = {
    "camera_observation_contract": (
        "agentview",
        "camera dictionary",
        "camera dict",
    ),
    "pose_contract": ("pose_mat",),
    "numpy_truth_contract": (
        "truth value of an array",
        "ambiguous truth",
    ),
    "unsupported_argument_contract": (
        "unexpected keyword",
        "unsupported keyword",
        "unexpected argument",
    ),
}


def _latest_attempt(payload: dict[str, Any], kind: str) -> tuple[int | None, dict[str, Any]]:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for key, value in payload.items():
        match = ATTEMPT_PATTERN.match(key)
        if match is None or match.group(1) != kind or not isinstance(value, dict):
            continue
        candidates.append((int(match.group(2)), value))
    return max(candidates, default=(None, {}), key=lambda item: -1 if item[0] is None else item[0])


def _matching_execution(
    payload: dict[str, Any], diagnosis_index: int | None
) -> tuple[int | None, dict[str, Any]]:
    """Return the execution paired with a diagnosis, never a different attempt."""

    if diagnosis_index is None:
        return _latest_attempt(payload, "execution")
    value = payload.get(f"execution_attempt_{diagnosis_index}")
    if isinstance(value, dict):
        return diagnosis_index, value
    return None, {}


def _diagnostic_signatures(
    diagnosis: dict[str, Any], execution: dict[str, Any]
) -> list[str]:
    structured_text = "\n".join(
        str(value or "")
        for value in (
            diagnosis.get("policy_feedback"),
            diagnosis.get("failure_mode"),
            execution.get("stderr_snippet"),
            execution.get("api_diagnostics_summary"),
            execution.get("api_diagnostics"),
        )
    ).lower()
    labels = [
        label
        for label, markers in SIGNATURES.items()
        if any(marker in structured_text for marker in markers)
    ]
    failure_mode = str(diagnosis.get("failure_mode") or "").strip().lower()
    if failure_mode:
        labels.append(f"failure_mode:{failure_mode}")
    return sorted(set(labels)) or ["no_structured_signature"]


def analyze(episodes: pd.DataFrame, method: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = episodes[episodes["method"] == method].copy()
    failures = selected[~selected["native_success"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for episode in failures.itertuples(index=False):
        path = Path(str(episode.artifact_path))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parse_errors.append(f"{episode.episode_key}: {exc}")
            rows.append(
                {
                    "method": method,
                    "cohort": episode.cohort,
                    "episode_key": episode.episode_key,
                    "failure_mode": "artifact_parse_error",
                    "failed_step": "",
                    "diagnosis_attempt": None,
                    "execution_attempt": None,
                    "execution_success": False,
                    "diagnostic_signatures": "artifact_parse_error",
                    "artifact_path": str(path),
                }
            )
            continue

        diagnosis_index, diagnosis = _latest_attempt(payload, "diagnosis")
        execution_index, execution = _matching_execution(payload, diagnosis_index)
        signatures = _diagnostic_signatures(diagnosis, execution)
        rows.append(
            {
                "method": method,
                "cohort": episode.cohort,
                "episode_key": episode.episode_key,
                "failure_mode": str(
                    diagnosis.get("failure_mode") or "missing_structured_diagnosis"
                ),
                "failed_step": str(diagnosis.get("failed_step") or "unspecified"),
                "diagnosis_attempt": diagnosis_index,
                "execution_attempt": execution_index,
                "execution_success": bool(execution.get("success", False)),
                "diagnostic_signatures": ";".join(signatures),
                "policy_feedback": str(diagnosis.get("policy_feedback") or ""),
                "stderr_snippet": str(execution.get("stderr_snippet") or ""),
                "artifact_path": str(path),
            }
        )

    frame = pd.DataFrame(rows)
    diagnosed = int((frame["failure_mode"] != "missing_structured_diagnosis").sum())
    summary = {
        "schema_version": 1,
        "method": method,
        "episodes": int(len(selected)),
        "native_successes": int(selected["native_success"].astype(bool).sum()),
        "native_failures": int(len(failures)),
        "structured_diagnoses": diagnosed,
        "structured_diagnosis_coverage": diagnosed / len(failures) if len(failures) else 1.0,
        "artifact_parse_errors": len(parse_errors),
        "parse_error_examples": parse_errors[:10],
        "interpretation": (
            "Failure modes come from the latest structured diagnosis record. "
            "Diagnostic signatures are non-causal indicators extracted only from "
            "structured diagnosis/execution fields and may overlap."
        ),
    }
    return frame, summary


def _summary_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[column, "episodes", "fraction_of_failures"])
    counts = Counter(str(value) for value in frame[column])
    return pd.DataFrame(
        [
            {column: value, "episodes": count, "fraction_of_failures": count / len(frame)}
            for value, count in counts.most_common()
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="rats_90")
    args = parser.parse_args()

    episodes = pd.read_csv(args.episodes)
    frame, summary = analyze(episodes, args.method)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "failure_episodes.csv", index=False)
    _summary_table(frame, "failure_mode").to_csv(
        args.output_dir / "failure_modes.csv", index=False
    )
    _summary_table(frame, "failed_step").to_csv(
        args.output_dir / "failed_steps.csv", index=False
    )

    signature_rows = []
    signature_counts: Counter[str] = Counter()
    for labels in frame["diagnostic_signatures"]:
        signature_counts.update(str(labels).split(";"))
    for label, count in signature_counts.most_common():
        signature_rows.append(
            {
                "diagnostic_signature": label,
                "episodes": count,
                "fraction_of_failures": count / len(frame),
            }
        )
    pd.DataFrame(signature_rows).to_csv(
        args.output_dir / "diagnostic_signatures.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# Final structured RATS-90 failure audit\n\n"
        f"This audit covers {summary['native_failures']} native failures among "
        f"{summary['episodes']} completed episodes. Structured diagnosis coverage is "
        f"{summary['structured_diagnosis_coverage']:.1%}.\n\n"
        "`failure_modes.csv` and `failed_steps.csv` are mutually exclusive summaries "
        "of the latest structured diagnosis per episode. `diagnostic_signatures.csv` "
        "contains overlapping, non-causal indicators derived only from structured "
        "diagnosis and execution fields. Generated source and arbitrary trace text are "
        "not searched.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
