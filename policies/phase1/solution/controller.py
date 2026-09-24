"""Frozen hierarchical visual-ReAct controller after RACaP Phase 1.

The capability curriculum builds this controller before autonomous
self-evolution. Phase 2 can modify planning, routing, verification, memory, or
one physical skill while preserving the rest of the working system.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from racap.agent import full_react, react
from racap.agent.tools import Session

from .policy_api import actuate_control, articulate, insert, pickplace, push, stack


def _normalise_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _appearance_memory(source: str) -> str:
    """Retrieve only the source-relevant visual prior from candidate memory.

    Appending an entire memory file perturbs every ReAct prompt and invalidates
    otherwise reusable model-cache entries. Exact retrieval keeps unrelated
    tasks bit-for-bit on the retained prompt while still giving the visual
    agent an explicit identity prior for known ambiguous categories.
    """
    key = _normalise_name(source)
    if not key:
        return ""
    path = Path(__file__).resolve().parents[1] / "memory" / "object_appearances.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    for entry in payload.get("objects", []):
        if not isinstance(entry, dict):
            continue
        names = {_normalise_name(value) for value in entry.get("names", [])}
        appearance = str(entry.get("appearance") or "").strip()
        if key in names and appearance:
            return (
                f"Advisory visual identity memory for {source!r}: {appearance} "
                "Keep the canonical task noun as the detector query. Compare "
                "visible candidates with this description; current RGB evidence "
                "and tool reports have priority, and uncertainty is preferable "
                "to forcing the closest-looking object."
            )
    return ""


def _session(runtime: Any, transport_skill=pickplace) -> Session:
    """Bind every ReAct tool to the candidate-owned editable API surface."""
    return Session(
        runtime,
        transport_skill,
        push_skill=push,
        insert_skill=insert,
        stack_skill=stack,
        articulate_skill=articulate,
        actuate_skill=actuate_control,
    )


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _budgets(episode: dict[str, Any]) -> dict[str, int]:
    raw = episode.get("budgets")
    raw = raw if isinstance(raw, dict) else {}
    return {
        "turns": _positive_int(raw.get("turns"), 45),
        "pickplace": _positive_int(raw.get("pickplace"), 8),
        "push": _positive_int(raw.get("push"), 3),
        "insert": _positive_int(raw.get("insert"), 2),
        "state": _positive_int(raw.get("state"), 5),
        "stack": _positive_int(raw.get("stack"), 2),
    }


def _outcome_dict(outcome: Any) -> dict[str, Any]:
    if is_dataclass(outcome):
        result = asdict(outcome)
    elif isinstance(outcome, dict):
        result = dict(outcome)
    else:
        result = dict(vars(outcome))
    result.setdefault("success", False)
    result.setdefault("turns", 0)
    result.setdefault("steps", [])
    result.setdefault("reflections", [])
    result.setdefault("stopped", "")
    return result


def run_episode(runtime: Any, episode: dict[str, Any]) -> dict[str, Any]:
    """Run either a bounded transport subtask or a complete instruction."""
    instruction = str(episode.get("instruction") or "").strip()
    model = str(episode.get("model") or "gpt-5.5").strip()
    source = str(episode.get("source") or "").strip()
    destination = str(episode.get("destination") or "").strip()
    budget = _budgets(episode)

    if source and destination:
        call_options: dict[str, Any] = {"session": _session(runtime)}
        appearance_memory = _appearance_memory(source)
        if appearance_memory:
            call_options["system_prompt"] = react.SYSTEM + "\n\n" + appearance_memory
        outcome = react.run_episode(
            runtime,
            pickplace,
            react.Episode(
                instruction=instruction,
                pick=source,
                place=destination,
                turns=min(budget["turns"], 21),
                max_pickplace_calls=budget["pickplace"],
                max_push_calls=budget["push"],
                max_insert_calls=budget["insert"],
                initial_skill="pickplace",
                model=model,
            ),
            **call_options,
        )
        return _outcome_dict(outcome)

    # Full ReAct constructs its Session internally. Bind that constructor to
    # this candidate's six editable API wrappers for the duration of the call.
    previous_session = full_react.Session
    full_react.Session = _session
    try:
        outcome = full_react.run_full_episode(
            runtime,
            pickplace,
            full_react.FullEpisode(
                instruction=instruction,
                turns=budget["turns"],
                max_pickplace_calls=budget["pickplace"],
                max_push_calls=budget["push"],
                max_insert_calls=budget["insert"],
                max_state_calls=budget["state"],
                max_stack_calls=budget["stack"],
                model=model,
            ),
        )
    finally:
        full_react.Session = previous_session
    return _outcome_dict(outcome)


__all__ = ["run_episode"]
