"""Composable visual ReAct controller for complete LIBERO-90 instructions.

The single-relation agent remains intentionally specialised and stable.  This
module adds a thin task layer: a VLM decomposes natural language into semantic
subgoals, generic dependency rules order them, and each Policy API retains its
own geometric controller and verification loop.  BDDL goals and native
predicates are not supplied to the planner or policy.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from racap.agent.experience import load_runtime_memory, with_runtime_memory
from racap.agent.react import Episode as TransportEpisode
from racap.agent.react import Outcome, _ask_json_reply, run_episode
from racap.agent.tools import Session, Step, describe_tools


PLAN_SYSTEM = """You decompose one robot instruction into a short ordered plan.

Available manipulation skills:
- pickplace: ordinary move onto/into/beside a destination, including a
  commanded planar push; its transport controller chooses grasping or pushing
  from current visual feedback.
- insert: narrow compartment/slot insertion only; otherwise use pickplace.
- stack: build an object-on-object stack only when the instruction explicitly
  says "stack". Ordinary "put/place X on Y" is pickplace, including when Y is
  a bowl, plate, tray, pan, or another movable object.
- articulate: open/close a drawer or hinged door.
- actuate_control: set an appliance control on/off.

Output JSON only:
{"goals": [{"skill": "...", ...}], "reason": "one sentence"}

Schemas:
- {"skill":"pickplace|insert", "pick":"visual object phrase", "place":"visual destination phrase"}
- {"skill":"stack", "pick":"top object phrase", "support":"bottom object phrase"}
- {"skill":"articulate", "target":"fixture noun", "part":"moving part or empty", "goal":"open|closed"}
- {"skill":"actuate_control", "target":"appliance noun", "control_hint":"optional", "goal":"on|off"}

Rules:
1. Entity fields must copy noun phrases from the instruction. Preserve every
   identity and spatial qualifier: colour/material, front/middle/back,
   left/right, top/bottom/under. Vision may resolve a copied phrase later but
   the planner must not rename it (for example ketchup -> ketchup box, tray ->
   wooden tray, or top of cabinet -> cabinet are invalid).
2. Open a receptacle before putting something inside; close it afterwards if
   requested. Never close it before insertion.
3. If a stacked pair must also go in a tray/container, place the bottom support
   there first and then stack the top object on it. Do not transport a loose
   two-object stack.
4. Set an appliance control before placing cookware on it, so the placed item
   cannot occlude the control or block the hand's approach.
5. Use at most four goals and do not invent coordinates or task identifiers.
6. Each transport child receives exactly one source-to-destination subgoal.
   Do not put a previously completed open/close/control/stack prerequisite into
   its pick/place phrase and do not expect the child to repeat that action.
7. For a quantified collection instruction such as "put all tabletop objects
   in the basket", name up to four currently visible loose members using short
   visual phrases. Do not include objects already visibly inside the
   destination. A later planning round may cover additional visible members.
"""


PLAN_REVIEW = """You are the final decision-maker for a robot task plan.

Instruction: {instruction}
Your first plan: {plan}

A deterministic language checker produced these advisory warnings:
{warnings}

Review the warnings against the instruction and image. They are suggestions,
not commands. Return your final plan as JSON using the original schema. Preserve
every valid goal and its order. Make the smallest field-level correction you
judge appropriate. Do not add, remove, or reorder an action merely because the
checker warned about an entity phrase. In particular, do not add an `open`
action when the image shows that the receptacle is already open.
"""


STATE_PREFLIGHT = """You are the final authority on the current visual state of
a robot task. Look at the current image and decide whether the requested state
is already satisfied before any new motion.

Goal: {goal}
An auxiliary visual/geometry checker reported: {check_report}

The auxiliary report is advisory and may be wrong. Base the final decision
primarily on what you see. Reply JSON only:
{{"goal_satisfied": true|false,
  "what_you_see": "one concrete sentence",
  "reason": "why the visible state does or does not satisfy the goal"}}
"""


STATE_DECIDE = """You are the final decision-maker for a robot mechanism state.
The images show the scene before the action, immediately after it, and the
current home-cleared view. When a fourth image is supplied, it is a zoomed
current view of the same fixture from its already-grounded pre-action box.

Goal: {goal}
Action: {action}
Tool report (advisory): {report}
Independent visual/geometry check (advisory): {check_report}
Contact strategies already attempted: {attempted_contacts}

Motion measurements, semantic checks, and failure codes are supporting evidence
only. They can be noisy and cannot overrule your visual state judgment. If you
can see that the requested state is satisfied, set `goal_satisfied=true` even
when a motion measurement says the incremental travel was too small. If it is
not satisfied, decide whether another materially different attempt is useful.
For `open`, require both a persistent visible cavity/gap and substantial panel
displacement compared with the first image. A seam, a few-pixel gap, or a
perspective/occlusion change is not an open mechanism. Use the zoom to inspect
the requested part, but keep its identity from the full scene.

Reply JSON only:
{{"goal_satisfied": true|false,
  "retry": true|false, "what_you_see": "one sentence",
  "args": {{"mechanism": "auto|prismatic|revolute|rotary|press",
             "contact_strategy": "auto|side_bar|pca_axis|top_down|graspnet_6d|graspgen_6d|hook",
             "amount": number or null}},
  "reason": "one sentence"}}

When `goal_satisfied=true`, set `retry=false`. Retry only if the requested state
is visibly not satisfied and another path or contact mode could correct it.
Do not change target, part, control identity, or requested state. A retry must
choose a contact family that is both valid for the measured mechanism and not
already listed as attempted. For a prismatic drawer use side_bar,
graspgen_6d, hook, graspnet_6d, pca_axis, or top_down. For a revolute door use
graspgen_6d, graspnet_6d, pca_axis, or top_down. For appliance controls use
pca_axis or top_down. If no materially different valid contact remains, set
retry=false instead of renaming the same contact.
"""


@dataclass
class FullEpisode:
    instruction: str
    turns: int = 45
    max_pickplace_calls: int = 8
    max_push_calls: int = 3
    max_insert_calls: int = 2
    max_state_calls: int = 5
    max_stack_calls: int = 2
    # Ordinary compositional tasks remain dependency ordered and fail closed.
    # An explicitly registered anytime collection task may instead skip one
    # failed independent object and re-observe for additional loose objects.
    continue_after_subgoal_failure: bool = False
    max_replans: int = 0
    model: str = field(default_factory=lambda: os.environ.get("RACAP_MODEL", "gpt-5.5"))


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


_ENTITY_FIELDS = ("pick", "place", "support", "target", "part")
_PLAN_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "then",
    "please",
    "to",
    "from",
    "of",
    "in",
    "into",
    "inside",
    "on",
    "onto",
    "at",
    "with",
    "by",
    "for",
    "put",
    "place",
    "move",
    "pick",
    "push",
    "take",
    "grab",
    "insert",
    "stack",
    "open",
    "close",
    "closed",
    "shut",
    "turn",
    "switch",
    "set",
    "is",
    "be",
    "it",
    "them",
    "this",
    "that",
    "up",
    "off",
}


def _semantic_token(token: str) -> str:
    """Small morphology normalisation, not a synonym/alias generator."""
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content_tokens(text: str) -> set[str]:
    return {
        _semantic_token(token)
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token not in _PLAN_STOP_WORDS
    }


_INSTANCE_SELECTOR = re.compile(
    r"\b(?:at|in)\s+the\s+(?:front|back|middle|center|centre)\b"
    r"|\bon\s+the\s+(?:left|right)\b",
    flags=re.IGNORECASE,
)


def _instruction_pick_selectors(instruction: str) -> list[str]:
    """Spatial selectors grammatically attached to the moved object phrase."""
    text = " ".join(str(instruction).split())
    pick_up = re.search(
        r"\bpick\s+up\s+(.+?)\s+and\s+(?:put|place|move)\b",
        text,
        flags=re.IGNORECASE,
    )
    source = pick_up.group(1) if pick_up else ""
    if not source:
        move = re.search(
            r"\b(?:put|place|move|stack)\s+(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if not move:
            return []
        remainder = move.group(1)
        # In a compound command such as "stack A on B and place them in C",
        # the second transport clause is not part of A's noun phrase. Without
        # this boundary, "on the right bowl" is truncated to the apparent
        # selector "on the right" and a valid causal plan is rejected.
        remainder = re.split(
            r"\s+and\s+(?:put|place|move)\b",
            remainder,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        # The final support/container preposition separates the source noun
        # phrase from its destination. Earlier "at the front" / "on the left"
        # constructions remain part of the source selector.
        boundaries = list(
            re.finditer(
                r"\s+(?:in|into|inside|on|onto)\s+(?=(?:the\s+)?[a-z])",
                remainder,
                flags=re.IGNORECASE,
            )
        )
        source = remainder[: boundaries[-1].start()] if boundaries else remainder
    return [match.group(0).lower() for match in _INSTANCE_SELECTOR.finditer(source)]


def _resolve_plan_pronouns(
    goals: list[dict[str, Any]],
    instruction: str = "",
) -> list[dict[str, Any]]:
    """Resolve anaphora to the prior goal's principal physical referent.

    Updating the antecedent while walking fields made ``top of it`` refer to
    the object being picked in the *same* goal.  The antecedent of a new goal
    is instead the previous goal's destination/fixture/support.  For an
    articulation that is the fixture target, not its handle or moving part.
    """
    resolved: list[dict[str, Any]] = []
    recent_referent = ""
    pronouns = {"it", "them", "this", "that", "the object", "the appliance"}
    for source in goals:
        goal = dict(source)
        antecedent = recent_referent
        for field in _ENTITY_FIELDS:
            value = _text(goal.get(field))
            lowered = value.lower()
            if antecedent and lowered in pronouns:
                goal[field] = antecedent
            elif antecedent and re.search(r"\b(?:it|them|this|that)\b", lowered):
                goal[field] = re.sub(
                    r"\b(?:it|them|this|that)\b",
                    antecedent,
                    value,
                    flags=re.IGNORECASE,
                )
        resolved.append(goal)
        skill = goal.get("skill")
        recent_referent = (
            _text(
                goal.get("place")
                if skill in {"pickplace", "insert"}
                else goal.get("support")
                if skill == "stack"
                else goal.get("target")
                if skill in {"articulate", "actuate_control"}
                else ""
            )
            or recent_referent
        )
    # A planner may prematurely replace an instruction pronoun with the
    # articulated *part*, leaving no pronoun for the field-wise pass above to
    # repair.  Preserve the grammatical surface relation from the instruction:
    # in "close the drawer of the cabinet and put X on top of it", the stable
    # fixture referent is the cabinet, not the drawer opening.  Restrict this
    # correction to the explicit "on top of it" construction and an actual
    # articulated fixture so ordinary destination phrases are untouched.
    if re.search(
        r"\bon\s+top\s+of\s+(?:it|this|that)\b",
        instruction.lower(),
    ):
        fixture = next(
            (
                _text(goal.get("target"))
                for goal in resolved
                if goal.get("skill") == "articulate" and _text(goal.get("target"))
            ),
            "",
        )
        if fixture:
            for goal in resolved:
                if goal.get("skill") in {"pickplace", "insert"}:
                    goal["place"] = f"top of {fixture}"
    return resolved


def _plan_semantic_errors(instruction: str, goals: list[dict[str, Any]]) -> list[str]:
    """Reject entity invention and dropped identity/spatial information."""
    instruction_tokens = _content_tokens(instruction)
    present: set[str] = set()
    errors: list[str] = []
    for index, goal in enumerate(goals):
        for field in _ENTITY_FIELDS:
            value = _text(goal.get(field))
            if not value:
                continue
            tokens = _content_tokens(value)
            present.update(tokens)
            invented = sorted(tokens - instruction_tokens)
            if invented:
                errors.append(f"goal {index + 1} {field} invents tokens {invented}")
    missing = sorted(instruction_tokens - present)
    if missing:
        errors.append(f"instruction entity/qualifier tokens were dropped: {missing}")
    selectors = _instruction_pick_selectors(instruction)
    if selectors:
        picks = " ".join(
            _text(goal.get("pick")).lower()
            for goal in goals
            if goal.get("skill") in {"pickplace", "insert", "stack"}
        )
        destinations = " ".join(
            _text(goal.get("place") or goal.get("support")).lower()
            for goal in goals
            if goal.get("skill") in {"pickplace", "insert", "stack"}
        )
        misplaced = [selector for selector in selectors if selector not in picks]
        if misplaced:
            errors.append("moved-object spatial selector changed attachment: " + repr(misplaced))
        duplicated = []
        for selector in selectors:
            selector_tokens = _content_tokens(selector)
            if any(
                len(re.findall(rf"\b{re.escape(token)}\b", picks + " " + destinations))
                > len(re.findall(rf"\b{re.escape(token)}\b", instruction.lower()))
                for token in selector_tokens
            ):
                duplicated.append(selector)
        if duplicated:
            errors.append(
                "moved-object spatial selector was also attached to the "
                "destination: " + repr(duplicated)
            )
    return errors


def _normalise_goal(raw: dict[str, Any]) -> dict[str, Any] | None:
    skill = _text(raw.get("skill")).lower()
    aliases = {
        "move": "pickplace",
        "place": "pickplace",
        "open": "articulate",
        "close": "articulate",
        "turn_on": "actuate_control",
        "turn_off": "actuate_control",
        "control": "actuate_control",
    }
    skill = aliases.get(skill, skill)
    if skill in {"pickplace", "insert"}:
        pick, place = _text(raw.get("pick")), _text(raw.get("place"))
        return {"skill": skill, "pick": pick, "place": place} if pick and place else None
    if skill == "stack":
        pick = _text(raw.get("pick"))
        support = _text(raw.get("support") or raw.get("place"))
        return {"skill": skill, "pick": pick, "support": support} if pick and support else None
    if skill == "articulate":
        target = _text(raw.get("target"))
        part = _text(raw.get("part"))
        value = _text(raw.get("goal")).lower()
        if not target or value not in {"open", "close", "closed"}:
            return None
        goal = "closed" if value in {"close", "closed"} else "open"
        return {"skill": skill, "target": target, "part": part, "goal": goal}
    if skill == "actuate_control":
        target = _text(raw.get("target"))
        value = _text(raw.get("goal")).lower()
        if not target or value not in {"on", "turnon", "turn_on", "off", "turnoff", "turn_off"}:
            return None
        goal = "on" if value in {"on", "turnon", "turn_on"} else "off"
        return {
            "skill": skill,
            "target": target,
            "control_hint": _text(raw.get("control_hint")),
            "goal": goal,
        }
    return None


def _language_allowed_skills(instruction: str) -> set[str]:
    """Return skill families explicitly licensed by instruction verbs.

    The scene image is useful for resolving noun phrases, but it also contains
    tempting distractors.  Manipulation intent comes from language: vision may
    fill a referent, never invent another verb.  These are semantic verb
    classes rather than benchmark templates or task identifiers.
    """
    text = " " + " ".join(str(instruction).lower().split()) + " "
    allowed: set[str] = set()
    if re.search(r"\b(?:open|close|shut)\b", text):
        allowed.add("articulate")
    if re.search(r"\b(?:turn|switch)\s+(?:on|off)\b", text):
        allowed.add("actuate_control")
    # ``put/place`` describes ordinary transport even when its support is a
    # movable object. ``stack`` is a distinct commanded operation: using its
    # full-footprint stability preflight for a bowl or mug on a plate rejects
    # valid placements before the arm moves. Keep the skill boundary semantic
    # and category-independent rather than asking scene vision to infer an
    # unspoken verb.
    if re.search(r"\b(?:put|place|move|insert)\b", text):
        allowed.update({"pickplace", "insert"})
        # Opening an articulated receptacle is a physical prerequisite of an
        # explicitly requested inside placement, even when the instruction
        # omits the word "open".  This licenses only opening; the planner still
        # cannot invent closing or an unrelated articulation.
        if re.search(
            r"\b(?:in|into|inside)\s+(?:the\s+)?(?:top\s+|bottom\s+)?"
            r"(?:drawer|microwave|cabinet|box|bin|container|compartment)\b",
            text,
        ):
            allowed.add("articulate")
    # A commanded push is still a source-to-destination transport goal.  The
    # transport ReAct loop owns the runtime choice between its push primitive
    # and a grasping move; the language gate must not erase the agent's valid
    # transport plan merely because ``push`` is not itself a top-level schema.
    if re.search(r"\bpush(?:es|ed|ing)?\b", text):
        allowed.add("pickplace")
    if re.search(r"\bstack\b", text):
        allowed.add("stack")
    return allowed


_CONTAINMENT_PARTS = {
    "drawer",
    "microwave",
    "box",
    "bin",
    "container",
    "compartment",
    "cubby",
    "cabinet",
}


def _close_depends_on_transport(close: dict[str, Any], transport: dict[str, Any]) -> bool:
    """Whether closing would block the requested transport destination."""
    if (
        close.get("skill") != "articulate"
        or close.get("goal") != "closed"
        or transport.get("skill") not in {"pickplace", "insert"}
    ):
        return False
    place = _text(transport.get("place")).lower()
    if re.search(r"\b(?:on\s+)?top\s+of\b", place):
        return False
    place_tokens = _content_tokens(place)
    if not (place_tokens & _CONTAINMENT_PARTS):
        return False
    part = _text(close.get("part")).lower()
    reference = part or _text(close.get("target")).lower()
    reference_tokens = _content_tokens(reference)
    if not reference_tokens:
        return False
    # Prefer an explicit moving-part match.  With an empty part, a shared
    # fixture noun plus a containment destination is enough: "top drawer of
    # cabinet" is both the target and the place in many valid planner plans.
    if part:
        return bool(
            reference in place
            or place in reference
            or (reference_tokens & place_tokens & _CONTAINMENT_PARTS)
        )
    return bool(reference in place or place in reference or reference_tokens & place_tokens)


def _order_goals(goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generic partial order with receptacle-aware close dependencies."""

    def close_after_transport(goal: dict[str, Any]) -> bool:
        return any(_close_depends_on_transport(goal, candidate) for candidate in goals)

    def rank(goal: dict[str, Any]) -> int:
        if goal["skill"] == "articulate" and goal.get("goal") == "open":
            return 0
        # Controls are small and commonly sit behind the appliance's support
        # region. Actuate them while access is clear; a later pan or pot may
        # occlude both the camera and the gripper path.
        if goal["skill"] == "actuate_control":
            return 1
        if (
            goal["skill"] == "articulate"
            and goal.get("goal") == "closed"
            and not close_after_transport(goal)
        ):
            return 2
        if goal["skill"] in {"pickplace", "insert"}:
            return 3
        if goal["skill"] == "stack":
            return 4
        if goal["skill"] == "articulate":
            return 5
        return 6

    # Stable sorting retains linguistic order among independent operations.
    return sorted(goals, key=rank)


def plan_instruction(
    instruction: str, image: str, *, model: str
) -> tuple[list[dict[str, Any]], str]:
    memory = load_runtime_memory(scope="full")
    plan_system = with_runtime_memory(PLAN_SYSTEM, memory)
    reply, parsed = _ask_json_reply(
        plan_system,
        f"Instruction: {instruction}\nThe image is the initial scene. Plan now.",
        images=[image],
        model=model,
        max_tokens=2200,
    )
    allowed = _language_allowed_skills(instruction)

    def parse(candidate: dict[str, Any]) -> list[dict[str, Any]]:
        raw_goals = candidate.get("goals") or []
        normalised = [
            goal
            for item in raw_goals[:4]
            if isinstance(item, dict)
            and (goal := _normalise_goal(item)) is not None
            and goal["skill"] in allowed
        ]
        return _resolve_plan_pronouns(normalised, instruction)

    goals = parse(parsed)
    errors = _plan_semantic_errors(instruction, goals)
    if errors:
        review_prompt = PLAN_REVIEW.format(
            instruction=instruction,
            plan=json.dumps({"goals": goals}, ensure_ascii=False),
            warnings="\n".join(f"- {error}" for error in errors),
        )
        second_reply, second_parsed = _ask_json_reply(
            plan_system,
            with_runtime_memory(review_prompt, memory),
            images=[image],
            model=model,
            max_tokens=2200,
        )
        second_goals = parse(second_parsed)
        reply = (
            reply
            + "\nSEMANTIC ADVISORY\n"
            + json.dumps(errors, ensure_ascii=False)
            + "\nAGENT PLAN REVIEW\n"
            + second_reply
        )
        # The checker can ask the agent to reconsider, but it cannot reject the
        # agent's final semantic decision.  Keep the original usable plan only
        # when the review failed to produce any schema-valid goal.
        if second_goals:
            goals = second_goals
    # Goal order is also the agent's decision.  Dependency suggestions already
    # live in PLAN_SYSTEM; silently sorting a confirmed plan would give a
    # deterministic helper higher authority than the visual planner.
    return goals, reply


def _append_step(
    outcome: Outcome,
    step,
    goal_index: int,
    turn: int,
) -> None:
    outcome.steps.append(
        {
            "turn": turn,
            "goal_index": goal_index,
            "action": step.action,
            "args": step.args,
            "thought": step.thought,
            "observation": step.observation,
            "report": step.report,
        }
    )


def _stop_on_infrastructure_failure(outcome: Outcome, step: Step) -> bool:
    failure = (step.report or {}).get("infrastructure_failure")
    if not failure:
        return False
    outcome.stopped = (
        "perception infrastructure unavailable; evaluation is inconclusive: "
        + json.dumps(failure, ensure_ascii=False)
    )
    return True


def _run_state_goal(
    session: Session,
    goal: dict[str, Any],
    episode: FullEpisode,
    outcome: Outcome,
    goal_index: int,
) -> bool:
    action = goal["skill"]
    runtime_memory = load_runtime_memory(scope="full")
    args = {key: value for key, value in goal.items() if key != "skill"}
    state_args = {key: goal[key] for key in ("target", "part", "goal") if goal.get(key)}
    attempted_contacts: list[str] = []

    # Ask the visual agent whether motion is necessary at all.  check_state is
    # retained as a measurement producer, but its boolean cannot decide the
    # control flow.  This prevents an already-open drawer from being pulled
    # repeatedly merely because no *new* full stroke was measured.
    if outcome.turns < episode.turns:
        precheck = session.run(
            "check_state",
            state_args,
            thought="advisory state measurement before physical motion",
        )
        _append_step(outcome, precheck, goal_index, outcome.turns)
        outcome.turns += 1
        if _stop_on_infrastructure_failure(outcome, precheck):
            return False
        try:
            preflight_reply, preflight = _ask_json_reply(
                "You decide robot task state from vision. JSON only.",
                with_runtime_memory(
                    STATE_PREFLIGHT.format(
                        goal=json.dumps(goal, ensure_ascii=False),
                        check_report=json.dumps(precheck.report, ensure_ascii=False, default=str)[
                            :1800
                        ],
                    ),
                    runtime_memory,
                ),
                images=[precheck.image],
                model=episode.model,
                max_tokens=1000,
            )
            outcome.reflections.append("STATE PREFLIGHT\n" + preflight_reply)
            if bool(preflight.get("goal_satisfied")):
                return True
        except Exception as exc:
            outcome.reflections.append(
                f"state preflight decision failed: {type(exc).__name__}: {exc}"
            )
            # Only when the authoritative model is unavailable do we fall back
            # to the auxiliary check rather than issuing an unnecessary motion.
            if precheck.success:
                return True

    for attempt in range(episode.max_state_calls):
        if outcome.turns >= episode.turns:
            return False
        before = session.frame()
        step = session.run(action, args, thought=f"execute semantic subgoal {goal_index + 1}")
        attempted_contact = _text(
            ((step.report or {}).get("params") or {}).get("contact_strategy")
            if action == "articulate"
            else (step.report or {}).get("contact_strategy")
        ).lower()
        if attempted_contact:
            attempted_contacts.append(attempted_contact)
        _append_step(outcome, step, goal_index, outcome.turns)
        outcome.turns += 1
        if _stop_on_infrastructure_failure(outcome, step):
            return False
        if outcome.turns >= episode.turns:
            return bool(step.success)
        check = session.run(
            "check_state",
            state_args,
            thought="fresh visual state verification after contact motion",
        )
        _append_step(outcome, check, goal_index, outcome.turns)
        outcome.turns += 1
        if _stop_on_infrastructure_failure(outcome, check):
            return False
        try:
            decision_images = [before, step.image, check.image]
            fixture_bbox = (step.report or {}).get("fixture_bbox")
            crop_frame = getattr(getattr(session, "runtime", None), "crop_frame", None)
            if fixture_bbox and callable(crop_frame):
                try:
                    focused = crop_frame(fixture_bbox, pad=28)
                    if focused:
                        decision_images.append(focused)
                except Exception:
                    pass
            reflection, diagnosis = _ask_json_reply(
                "You decide robot task state from vision. JSON only.",
                with_runtime_memory(
                    STATE_DECIDE.format(
                        goal=json.dumps(goal, ensure_ascii=False),
                        action=step.for_history().split("\n", 1)[0],
                        report=json.dumps(step.report, ensure_ascii=False, default=str)[:1600],
                        check_report=json.dumps(check.report, ensure_ascii=False, default=str)[
                            :1800
                        ],
                        attempted_contacts=json.dumps(attempted_contacts, ensure_ascii=False),
                    ),
                    runtime_memory,
                ),
                images=decision_images,
                model=episode.model,
                max_tokens=1400,
            )
            outcome.reflections.append("STATE DECISION\n" + reflection)
        except Exception as exc:
            outcome.reflections.append(f"state decision failed: {type(exc).__name__}: {exc}")
            diagnosis = {
                "goal_satisfied": bool(check.success),
                "retry": not bool(check.success),
            }
        # The visual agent owns the state decision.  Measurements and failure
        # modes remain in its prompt and in the trajectory, but neither can
        # overturn this verdict.
        if bool(diagnosis.get("goal_satisfied")):
            return True
        if not bool(diagnosis.get("retry")):
            return False
        if attempt + 1 >= episode.max_state_calls:
            return False
        retry = diagnosis.get("args") if isinstance(diagnosis.get("args"), dict) else {}
        mechanism = _text(retry.get("mechanism")).lower()
        allowed = (
            {"auto", "prismatic", "revolute"}
            if action == "articulate"
            else {"auto", "rotary", "press"}
        )
        if mechanism in allowed:
            args["mechanism"] = mechanism
        contact = _text(retry.get("contact_strategy")).lower()
        if action == "articulate":
            # A mechanism-opening action must transmit tension along the
            # measured front normal. Try the side-contact proposal first;
            # top-down contacts are materially different fallbacks.
            ladder = (
                ("side_bar", "graspgen_6d", "hook", "graspnet_6d", "pca_axis", "top_down")
                if str((step.report or {}).get("mechanism", "")) == "prismatic"
                else ("graspgen_6d", "graspnet_6d", "pca_axis", "top_down")
            )
            if contact in ladder and contact not in attempted_contacts:
                args["contact_strategy"] = contact
            else:
                remaining = [value for value in ladder if value not in attempted_contacts]
                if not remaining:
                    outcome.reflections.append(
                        "state recovery stopped: every distinct contact "
                        "strategy was tried without visual success"
                    )
                    return False
                args["contact_strategy"] = remaining[0]
        elif action == "actuate_control":
            ladder = ("pca_axis", "top_down")
            if contact in ladder and contact not in attempted_contacts:
                args["contact_strategy"] = contact
            else:
                args["contact_strategy"] = next(
                    (value for value in ladder if value not in attempted_contacts),
                    ladder[-1],
                )
        amount = retry.get("amount")
        if isinstance(amount, (int, float)):
            args["amount"] = float(amount)
    return False


def _run_stack_goal(
    session: Session,
    goal: dict[str, Any],
    episode: FullEpisode,
    outcome: Outcome,
    goal_index: int,
) -> bool:
    args = {"pick": goal["pick"], "support": goal["support"]}
    same_vessel_category = any(
        noun in goal["pick"].lower() and noun in goal["support"].lower() for noun in ("bowl", "cup")
    )
    for attempt in range(episode.max_stack_calls):
        if outcome.turns >= episode.turns:
            return False
        if attempt:
            args["grasp"] = ["rim@top", "pca_axis@low", "top_down@low"]
        step = session.run("stack", args, thought="footprint-aware stable support placement")
        _append_step(outcome, step, goal_index, outcome.turns)
        outcome.turns += 1
        if _stop_on_infrastructure_failure(outcome, step):
            return False
        if bool((step.report or {}).get("preflight_only")):
            return False
        if outcome.turns >= episode.turns:
            return bool(step.success)
        check = session.run(
            "check",
            {"pick": goal["pick"], "place": goal["support"]},
            thought="measure concentricity and support height",
        )
        _append_step(outcome, check, goal_index, outcome.turns)
        outcome.turns += 1
        if _stop_on_infrastructure_failure(outcome, check):
            return False
        if check.success:
            return True
        if not bool((check.report or {}).get("reliable", False)):
            return False
        # Once two visually identical vessels touch or overlap, their original
        # front/middle/back relation no longer identifies the moved instance.
        # A second language-grounded grasp can pick the support (observed on
        # t17) and destroy a near stack.  Other-category stacks retain their
        # bounded retry.
        if same_vessel_category:
            # Re-grasping overlapping identical vessels can select the lower
            # support, while a planar push transmits through both vessels and
            # violates the verifier's stationary-reference assumption.  Stop
            # without disturbing an almost-correct nest; improvement belongs
            # in pre-release carry-offset control, not post-contact repair.
            return False
    return False


def run_full_episode(
    runtime,
    skill,
    episode: FullEpisode,
    *,
    verbose: bool = False,
) -> Outcome:
    """Plan and execute every semantic subgoal with visual feedback."""
    outcome = Outcome(success=False, turns=0)
    planner_session = Session(runtime, skill)
    initial = planner_session.frame()
    try:
        goals, planner_reply = plan_instruction(episode.instruction, initial, model=episode.model)
    except Exception as exc:
        outcome.stopped = f"task planner did not answer usably: {exc}"
        return outcome
    outcome.reflections.append("TASK PLAN\n" + planner_reply)
    if not goals:
        outcome.stopped = "task planner produced no valid semantic goals"
        return outcome

    completed: list[bool] = []
    attempted_signatures: set[tuple[str, ...]] = set()
    attempted_sources: list[str] = []
    planning_round = 0
    goal_index = 0
    pending_goals = goals
    stop_execution = False

    def goal_signature(goal: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            _text(goal.get(key)).lower()
            for key in ("skill", "pick", "place", "support", "target", "part", "goal")
        )

    while pending_goals and not stop_execution:
        for local_index, goal in enumerate(pending_goals):
            signature = goal_signature(goal)
            if signature in attempted_signatures:
                continue
            attempted_signatures.add(signature)
            if goal.get("pick"):
                attempted_sources.append(_text(goal["pick"]))
            if verbose:
                print(
                    f"  goal {goal_index + 1} "
                    f"(round {planning_round + 1}, {local_index + 1}/{len(pending_goals)}): "
                    f"{goal}",
                    flush=True,
                )
            remaining = episode.turns - outcome.turns
            if remaining <= 0:
                completed.append(False)
                stop_execution = True
                break
            if goal["skill"] in {"pickplace", "insert"}:
                # The mature transport ReAct loop keeps its own measurement-first
                # state machine. ``insert`` is selected automatically from the
                # bounded destination phrase, so both planner labels share it.
                sub = run_episode(
                    runtime,
                    skill,
                    TransportEpisode(
                        # Keep the original relation (``on`` versus ``in``).  A
                        # synthetic "move X to Y" erased precisely the semantic
                        # evidence needed to distinguish a support rack from a
                        # bounded shelf aperture.
                        instruction=episode.instruction,
                        pick=goal["pick"],
                        place=goal["place"],
                        turns=min(remaining, 21),
                        max_pickplace_calls=episode.max_pickplace_calls,
                        max_push_calls=episode.max_push_calls,
                        max_insert_calls=episode.max_insert_calls,
                        initial_skill=goal["skill"],
                        model=episode.model,
                    ),
                    verbose=verbose,
                    # Keep pre-transport cavity geometry and kinematic memory
                    # alive for dependent state goals.  A fresh Session here used
                    # to discard the clean open-drawer observation immediately
                    # before `articulate(close)` needed its front normal.
                    session=planner_session,
                )
                base = outcome.turns
                for row in sub.steps:
                    outcome.steps.append(
                        {
                            **row,
                            "turn": base + int(row.get("turn", 0)),
                            "goal_index": goal_index,
                        }
                    )
                outcome.turns += sub.turns
                outcome.reflections.extend(sub.reflections)
                completed.append(bool(sub.success))
            elif goal["skill"] == "stack":
                completed.append(
                    _run_stack_goal(planner_session, goal, episode, outcome, goal_index)
                )
            else:
                completed.append(
                    _run_state_goal(planner_session, goal, episode, outcome, goal_index)
                )
            goal_index += 1
            # Dependency-ordered tasks retain fail-closed semantics.  Only an
            # explicitly registered anytime collection may skip a failed
            # independent object and preserve progress on its siblings.
            if completed and not completed[-1] and not episode.continue_after_subgoal_failure:
                stop_execution = True
                break

        if stop_execution or not episode.continue_after_subgoal_failure:
            break
        if outcome.turns >= episode.turns or planning_round >= episode.max_replans:
            break

        planning_round += 1
        attempted = ", ".join(dict.fromkeys(attempted_sources)) or "none"
        continuation_instruction = (
            episode.instruction
            + "\nRuntime continuation context: Re-observe the current scene and plan only "
            "additional loose source objects still visibly outside the requested "
            "destination. Do not repeat these already attempted source phrases: "
            + attempted
            + ". Preserve the original destination and do not disturb objects already "
            "inside it."
        )
        try:
            replanned, replan_reply = plan_instruction(
                continuation_instruction,
                planner_session.frame(),
                model=episode.model,
            )
        except Exception as exc:
            outcome.reflections.append(
                f"ANYTIME REPLAN {planning_round} FAILED\n{type(exc).__name__}: {exc}"
            )
            break
        outcome.reflections.append(
            f"ANYTIME REPLAN {planning_round}\n" + replan_reply
        )
        pending_goals = [
            goal
            for goal in replanned
            if goal_signature(goal) not in attempted_signatures
        ]

    outcome.success = bool(completed and all(completed))
    outcome.stopped = (
        "all visually planned subgoals were verified"
        if outcome.success
        else f"subgoal verification results: {completed}"
    )
    return outcome


__all__ = [
    "FullEpisode",
    "PLAN_SYSTEM",
    "plan_instruction",
    "run_full_episode",
    "describe_tools",
]
