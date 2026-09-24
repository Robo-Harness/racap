#!/usr/bin/env python3
"""Distill RATS skills from native-success Robosuite development rollouts.

This is deliberately a thin adapter around the upstream RATS
``FeedbackGenerator`` and ``SkillLibrary``.  It does not invent a second skill
learner: native success selects eligible executions, the upstream LLM extractor
proposes at most two reusable functions per execution, and the upstream AST and
semantic-dedup gates decide whether each proposal enters the candidate library.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any


TASK_GOALS = {
    "cube_lifting": "Pick up the red cube and lift it.",
    "cube_restack": "Restack the cubes into the requested configuration.",
    "cube_stack": "Stack the red cube on the green cube.",
    "nut_assembly": "Pick up the square nut and place it onto the square peg.",
    "spill_wipe": "Use the wiping tool to clean the spill from the table.",
    "two_arm_handover": "Hand the hammer from one robot arm to the other.",
    "two_arm_lift": "Use both robot arms to lift the pot by its handles.",
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


_TRANSPORT_LOG_LOCK = threading.Lock()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TRANSPORT_LOG_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _install_registered_transport_retry(
    base_agent_module: Any,
    *,
    log_path: Path,
    abort_path: Path,
    maximum_attempts: int = 3,
) -> None:
    """Retry transport failures without changing a RATS reasoning request.

    The pinned upstream client retries HTTP 5xx responses, but a read timeout
    escapes ``requests.post`` and is then silently converted by
    ``FeedbackGenerator`` into an empty skill proposal.  That makes a provider
    outage look like a valid learning decision.  This process-local wrapper
    preserves the exact prompt/model/temperature while logging and retrying
    only connection/read timeouts.  Exhaustion raises the upstream
    ``BaseException``-derived infrastructure abort so it cannot be swallowed
    as ``new_skills=[]``.
    """

    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must be positive")
    requests_module = base_agent_module.requests
    original_post = requests_module.post
    transient = (requests_module.Timeout, requests_module.ConnectionError)

    def registered_post(*args: Any, **kwargs: Any) -> Any:
        request_json = kwargs.get("json")
        model = request_json.get("model") if isinstance(request_json, dict) else None
        url = str(args[0] if args else kwargs.get("url", ""))
        for attempt in range(1, maximum_attempts + 1):
            started = time.time()
            try:
                response = original_post(*args, **kwargs)
            except transient as exc:
                record = {
                    "schema_version": 1,
                    "time_unix": time.time(),
                    "attempt": attempt,
                    "maximum_attempts": maximum_attempts,
                    "outcome": "transport_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.time() - started,
                    "model": model,
                    "url": url,
                    "prompt_or_policy_changed": False,
                }
                _append_jsonl(log_path, record)
                if attempt < maximum_attempts:
                    continue
                _write_json(
                    abort_path,
                    {
                        "schema_version": 1,
                        "status": "infrastructure_invalid",
                        "reason": "RATS skill-extraction transport retries exhausted",
                        "last_transport_record": record,
                        "strategy_outputs_scored": False,
                    },
                )
                raise base_agent_module.RATSProviderAbort(
                    "registered RATS extraction transport retries exhausted"
                ) from exc
            if attempt > 1:
                _append_jsonl(
                    log_path,
                    {
                        "schema_version": 1,
                        "time_unix": time.time(),
                        "attempt": attempt,
                        "maximum_attempts": maximum_attempts,
                        "outcome": "recovered",
                        "elapsed_seconds": time.time() - started,
                        "model": model,
                        "url": url,
                        "prompt_or_policy_changed": False,
                    },
                )
            return response
        raise AssertionError("unreachable transport retry state")

    base_agent_module.requests.post = registered_post


def _native_successes(task_root: Path) -> dict[str, dict[str, Any]]:
    """Return the last post-code native record for every episode key."""

    records: dict[str, dict[str, Any]] = {}
    path = task_root / "native_states.jsonl"
    if not path.is_file():
        return records
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "native_state_after_code":
            continue
        key = str(row.get("episode_key") or "")
        if key:
            records[key] = row
    return records


def _successful_code(task_root: Path, trial: int) -> Path | None:
    candidates = sorted(
        task_root.glob(
            f"**/trial_{trial:02d}_sandboxrc_*_taskcompleted_1/code.py"
        ),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rats-root", type=Path, required=True)
    parser.add_argument("--source-library", type=Path, required=True)
    parser.add_argument("--source-sweep", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_GOALS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    if not args.source_library.is_file():
        raise SystemExit(f"missing source library: {args.source_library}")
    seeds = tuple(args.seeds)
    if not seeds or tuple(sorted(set(seeds))) != seeds:
        raise SystemExit("--seeds must be sorted and duplicate-free")
    if any(right != left + 1 for left, right in zip(seeds, seeds[1:])):
        raise SystemExit("--seeds must be contiguous")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite candidate directory: {args.output_dir}")

    # Bind every upstream RATS reasoning call to the registered VAPI route
    # before importing agent modules (their defaults are resolved at import).
    base = os.environ.get("RACAP_VAPI_BASE", "").rstrip("/")
    key = os.environ.get("RACAP_VAPI_KEY", "")
    if not base or not key:
        raise SystemExit("RACAP_VAPI_BASE and RACAP_VAPI_KEY must be configured")
    os.environ.update(
        {
            "OPENAI_API_KEY": key,
            "RATS_VAPI_URL": f"{base}/chat/completions",
            "RATS_VAPI_KEY": key,
            "RATS_LLM_MODEL": args.model,
            "RATS_FEEDBACK_GENERATOR_MODEL": args.model,
            "RATS_LLM_FALLBACK": "0",
            "RATS_AGENT_IO_DIR": str(args.output_dir / "agent_io"),
            "RATS_ABORT_SENTINEL": str(args.output_dir / "ABORTED.json"),
        }
    )
    sys.path.insert(0, str(args.rats_root.resolve()))

    from rats.agents import base_agent
    from rats.agents.feedback_generator import FeedbackGenerator
    from skill_library.library import SkillLibrary

    _install_registered_transport_retry(
        base_agent,
        log_path=args.output_dir / "transport_attempts.jsonl",
        abort_path=args.output_dir / "TRANSPORT_ABORTED.json",
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_library = args.output_dir / "skills.json"
    shutil.copy2(args.source_library, candidate_library)
    library = SkillLibrary(storage_path=str(candidate_library))
    feedback = FeedbackGenerator(model=args.model)

    log: dict[str, Any] = {
        "schema_version": 1,
        "iteration": args.iteration,
        "created_at_unix": time.time(),
        "source_library": str(args.source_library.resolve()),
        "source_sweep": str(args.source_sweep.resolve()),
        "tasks": list(args.tasks),
        "seeds": list(seeds),
        "model": args.model,
        "extractor": "upstream_rats.FeedbackGenerator",
        "validator": "upstream_rats.SkillLibrary.add_skill",
        "records": [],
        "accepted_skills": [],
    }
    # The live primitive names are the validator's explicit runtime allowlist.
    available = {
        str(skill.get("name"))
        for skill in library._skills
        if skill.get("name") and skill.get("is_primitive")
    }

    for task in args.tasks:
        task_root = args.source_sweep / task
        final_native = _native_successes(task_root)
        for seed in seeds:
            episode_key = f"robosuite/{task}/seed{seed}"
            native = final_native.get(episode_key) or {}
            record: dict[str, Any] = {
                "episode_key": episode_key,
                "native_success": bool(native.get("native_success")),
                "reward": float(native.get("reward") or 0.0),
                "code": None,
                "proposed": [],
                "accepted": [],
            }
            if not record["native_success"]:
                log["records"].append(record)
                continue
            trial = seed - seeds[0] + 1
            code_path = _successful_code(task_root, trial)
            if code_path is None:
                record["error"] = "native success had no taskcompleted_1 code artifact"
                log["records"].append(record)
                continue
            record["code"] = str(code_path.resolve())
            code = code_path.read_text(encoding="utf-8", errors="replace")
            result = feedback.generate(
                execution_result={
                    "reward": record["reward"],
                    "success": True,
                    "task_completed": True,
                },
                verification={"success": True},
                diagnosis={},
                attempt=1,
                plan={"task_id": task, "steps": []},
                code=code,
                task_language=TASK_GOALS[task],
                existing_skills=list(library._skills),
            )
            proposals = result.get("new_skills") or []
            record["proposed"] = [
                {key: value for key, value in proposal.items() if key != "code"}
                for proposal in proposals
                if isinstance(proposal, dict)
            ]
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                candidate = dict(proposal)
                candidate.setdefault("source_task", TASK_GOALS[task])
                candidate.setdefault("learned_iteration", args.iteration)
                original_name = str(candidate.get("name") or "unnamed")
                accepted = library.add_skill(
                    candidate,
                    available_functions=available,
                )
                stored_name = str(candidate.get("name") or original_name)
                if accepted:
                    library.record_usage(
                        [stored_name],
                        success=True,
                        iteration=args.iteration,
                        source="robosuite_native_success_extraction",
                    )
                    available.add(stored_name)
                    record["accepted"].append(stored_name)
                    log["accepted_skills"].append(stored_name)
            log["records"].append(record)
            _write_json(args.output_dir / "extraction_log.json", log)

    log["finished_at_unix"] = time.time()
    log["accepted_skill_count"] = len(log["accepted_skills"])
    log["library_size"] = len(library._skills)
    _write_json(args.output_dir / "extraction_log.json", log)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException:
        sentinel = Path(os.environ.get("RATS_ABORT_SENTINEL", ""))
        if sentinel.is_file():
            # Preserve the upstream sentinel and normalize the process code so
            # the outer evolution harness checkpoints instead of treating a
            # balance/provider outage as an implementation failure.
            print(sentinel.read_text(encoding="utf-8", errors="replace"), flush=True)
            raise SystemExit(75)
        raise
