"""Frozen hierarchical visual-ReAct controller produced by RACaP Phase 2.

This snapshot retains the Phase 1 controller and adds promoted experience
retrieval, routing priors, and long-horizon recovery behavior.
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
    """Retrieve only the source-relevant visual prior from candidate memory."""
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


def _strategy_memory(source: str) -> str:
    """Retrieve source-relevant advisory physical strategy memory."""
    key = _normalise_name(source)
    if not key:
        return ""
    source_words = set(key.split())
    path = Path(__file__).resolve().parents[1] / "memory" / "strategy_priors.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""

    matched: list[str] = []
    for entry in payload.get("priors", []):
        if not isinstance(entry, dict):
            continue
        names = {_normalise_name(value) for value in entry.get("names", [])}
        class_terms = {_normalise_name(value) for value in entry.get("class_terms", [])}
        class_terms.discard("")
        exact_or_phrase = key in names or any(
            name and (name in key or key in name) for name in names
        )
        term_match = bool(source_words & class_terms)
        if not (exact_or_phrase or term_match):
            continue
        visual_condition = str(entry.get("visual_condition") or "").strip()
        strategy = str(entry.get("strategy") or "").strip()
        uncertainty = str(entry.get("uncertainty") or "").strip()
        if not (visual_condition and strategy):
            continue
        matched.append(
            "\n".join(
                part
                for part in (
                    f"Visual condition to check: {visual_condition}",
                    f"Suggested explicit strategy controls: {strategy}",
                    f"Uncertainty / override rule: {uncertainty}" if uncertainty else "",
                )
                if part
            )
        )

    if not matched:
        return ""
    return (
        f"Advisory strategy memory for source {source!r}. This is not a hidden "
        "object-name dispatcher: inspect current RGB-D evidence, accept or "
        "reject the visual condition, and choose any pickplace grasp/destination "
        "arguments explicitly. Treat post-close attachment/lift evidence as the "
        "gate before transit or repeating an equivalent grasp.\n"
        + "\n\n".join(matched)
    )


def _trigger_group_matches(instruction: str, values: Any) -> bool:
    """Return whether one declarative memory trigger group matches language."""
    if not isinstance(values, list) or not values:
        return True
    text = _normalise_name(instruction)
    return any(
        (term := _normalise_name(value)) and term in text
        for value in values
    )


def _instruction_strategy_memory(instruction: str) -> str:
    """Retrieve advisory routing memory for a complete instruction.

    Trigger vocabulary lives in experience memory, not in a Policy API. Every
    trigger group must match, which keeps unrelated full-ReAct prompts unchanged.
    The retrieved content still requires the visual agent to confirm geometry
    and explicitly select or reject the proposed route.
    """
    if not _normalise_name(instruction):
        return ""
    path = Path(__file__).resolve().parents[1] / "memory" / "strategy_priors.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""

    matched: list[str] = []
    for entry in payload.get("routing_priors", []):
        if not isinstance(entry, dict):
            continue
        trigger = entry.get("instruction_trigger")
        if not isinstance(trigger, dict) or not trigger:
            continue
        if not all(
            _trigger_group_matches(instruction, values)
            for values in trigger.values()
        ):
            continue
        visual_condition = str(entry.get("visual_condition") or "").strip()
        strategy = str(entry.get("strategy") or "").strip()
        uncertainty = str(entry.get("uncertainty") or "").strip()
        if not (visual_condition and strategy):
            continue
        matched.append(
            "\n".join(
                part
                for part in (
                    f"Routing condition to verify visually: {visual_condition}",
                    f"Capability comparison and explicit route: {strategy}",
                    f"Uncertainty / override rule: {uncertainty}" if uncertainty else "",
                )
                if part
            )
        )

    if not matched:
        return ""
    return (
        "Advisory tool-routing memory for this complete instruction. It is not "
        "a forced object-name route. Inspect current RGB-D geometry and public "
        "skill reports, then explicitly accept, reject, or revise the route.\n"
        + "\n\n".join(matched)
    )


def _install_full_prompt_memory(memory: str) -> list[tuple[Any, str, str]]:
    """Temporarily augment public agent prompt constants and return restorations.

    RACaP versions may build full ReAct from its own SYSTEM constant or reuse
    the transport ReAct constant. Updating only existing string constants keeps
    this compatibility mechanism bounded and fully reversible.
    """
    if not memory:
        return []
    changed: list[tuple[Any, str, str]] = []
    seen: set[tuple[int, str]] = set()
    for module in (full_react, react):
        for name in ("SYSTEM", "SYSTEM_PROMPT"):
            key = (id(module), name)
            if key in seen:
                continue
            seen.add(key)
            original = getattr(module, name, None)
            if not isinstance(original, str):
                continue
            setattr(module, name, original + "\n\n" + memory)
            changed.append((module, name, original))
    return changed


def _restore_prompt_memory(changed: list[tuple[Any, str, str]]) -> None:
    for module, name, original in reversed(changed):
        setattr(module, name, original)


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
    objective_mode = _normalise_name(episode.get("objective_mode"))
    independent_anytime = objective_mode == "independent anytime"

    if source and destination:
        call_options: dict[str, Any] = {"session": _session(runtime)}
        prompt_memories = [
            memory
            for memory in (_appearance_memory(source), _strategy_memory(source))
            if memory
        ]
        if prompt_memories:
            call_options["system_prompt"] = react.SYSTEM + "\n\n" + "\n\n".join(prompt_memories)
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

    routing_memory = _instruction_strategy_memory(instruction)
    changed_prompts = _install_full_prompt_memory(routing_memory)
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
                continue_after_subgoal_failure=independent_anytime,
                max_replans=3 if independent_anytime else 0,
                model=model,
            ),
        )
    finally:
        full_react.Session = previous_session
        _restore_prompt_memory(changed_prompts)
    return _outcome_dict(outcome)


__all__ = ["run_episode"]
