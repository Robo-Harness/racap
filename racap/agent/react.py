"""A VLM agent that retries a fixed skill differently after looking at it.

The division of labour matters more than either half. ``pickplace`` holds what
is true of every scene -- how to close fingers below a rim, how far a carried
object hangs, which detector answers which kind of phrase -- and exposes as
arguments everything that varies. This agent holds nothing general at all. It
looks at the picture, reads what the skill measured, and decides which
argument was wrong.

That split is why the loop can help where more tuning could not. Every
heuristic added to the skill has to be right on average across forty scenes,
and the ones worth adding ran out: the last four were each right on the
episode that motivated them and wrong on two others. A choice made in front of
the failed attempt does not have to be right on average.

The reflection is a separate turn from the retry, and it is a question about
an image rather than about a failure code. Asked "why did this fail" with the
trace alone, the model restates the failure mode. Asked the same with the
photograph of the object lying next to the caddy instead of in it, it says the
release was too high and the book toppled -- which is an argument, and an
argument can be changed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from racap.agent.tools import (
    Session,
    _normalised_label,
    _visibly_recoverable_miss,
    describe_tools,
)
from racap.agent.experience import load_runtime_memory, with_runtime_memory
from racap.backends.llm import ask

DEFAULT_MAX_PICKPLACE_CALLS = 8
DEFAULT_MAX_PUSH_CALLS = 3
DEFAULT_MAX_INSERT_CALLS = 2
TRANSPORT_TOOL_NAMES = (
    "pickplace",
    "insert",
    "push",
    "look",
    "check",
    "done",
)
TRANSPORT_ACTIONS = frozenset(TRANSPORT_TOOL_NAMES)
# A pickplace and a push each require a fresh check, then one final done turn.
# Keep the motion budgets independent: using a safe micro-correction must not
# silently consume one of the eight pickplace/check pairs.
DEFAULT_TURNS = (
    2 * DEFAULT_MAX_PICKPLACE_CALLS + 2 * DEFAULT_MAX_PUSH_CALLS + 2 * DEFAULT_MAX_INSERT_CALLS + 1
)


def _advisory_check_observation(value: str) -> str:
    """Remove state-decision imperatives from a measurement report.

    ``check`` contributes geometry and an independent visual alias vote.  It
    does not own semantic task completion.  Older reports embedded commands
    such as "do not declare done", which gave that auxiliary vote accidental
    prompt priority over the ReAct agent.  Keep the evidence and uncertainty,
    but express the conclusion as advisory risk rather than an action veto.
    """
    return (
        str(value)
        .replace(
            "This looks placed. Call done.",
            "The geometric check classifies this relation as placed.",
        )
        .replace(
            "Do not declare done or re-grasp from this ambiguous evidence.",
            "This alias evidence is inconclusive and requires independent visual judgment.",
        )
        .replace(
            "Preserve the current scene; do not re-grasp a possibly placed or occluded object.",
            "Re-grasping may disturb a possibly placed or occluded object.",
        )
    )


def _reflection_holding_evidence(motion_step: Any, measured_holding: str) -> str:
    """Resolve stale finger-width evidence against an explicit release event."""
    report = getattr(motion_step, "report", None) or {}
    observation = str(getattr(motion_step, "observation", "") or "").lower()
    released = bool(
        report.get("release_xy") is not None
        and (
            report.get("release_cleanup") in {"reopened", "already_open"}
            or observation.startswith("released")
            or "opened the fingers" in observation
        )
    )
    if released and "closed on something" in str(measured_holding).lower():
        return (
            "the skill explicitly opened the fingers at release; the later "
            "finger-width heuristic does not confirm that an object is held"
        )
    return str(measured_holding)

SYSTEM = """You control one source-to-destination transport subgoal from camera images.

The parent task controller has already handled open/close, appliance controls,
stacking, and causal ordering. Never repeat those actions even when they appear
in the full instruction. `pickplace` is
the broad transport skill. `insert` is a separate narrow-opening skill that
proves the full object footprint fits before moving. `push` is only a small
post-release surface correction after `check` measured a miss. Your job is to
look at what happened and select the smallest evidence-backed correction.

Actions:
{tools}

Reply with JSON and nothing else:
{{"thought": "<one or two sentences about what the image shows>",
  "action": "<action name>",
  "args": {{...}}}}

State machine (follow it; freestyle retries destroy already-good placements):
1. The controller deterministically dispatches the first transport; reason
   only about evidence-backed recovery after it returns.
2. After every `pickplace` or moved `insert`, call `check` next. Do not
   immediately retry. If insert reports a preflight-only failure, the scene is
   unchanged; fall back to pickplace rather than repeating insert.
3. If `check` says it looks placed, call `done`. Stop.
4. If `check` says the object landed and is only 0.5–5 cm off laterally, use
   `push` with exactly its suggested nudge, then `check` again. If it did not
   land or is farther away, retry `pickplace` with that measured nudge.
5. If the skill reported `empty_grasp` on the first attempt, retry once with
   a different `grasp` (try a list so fallbacks remain, e.g.
   ["top_down","rim","pca_axis"], or use `graspnet` when the analytic
   families already failed). Your valid grasp proposal is executed as given;
   do not repeat a family that just returned empty. Then `check` if anything moved.
   If destination grounding failed, call `look` first and use a concrete
   visually matching destination name from its inventory; removing an article
   or repeating the same ungrounded phrase is not a correction.
6. Never call a transport skill again after a placement that already looks
   right in the image or that `check` endorsed. Re-grasping an object that is
   already at the destination is the most common way to turn a success into a
   miss.
7. Never invent coordinates for `place`. Only labels and the documented
   optional arguments. `nudge` is how you shift the release.
8. Do not repeat the exact same arguments. Change one thing or call `check`
   / `done`.
9. Skill and check reports are measured evidence, not hidden truth. Resolve a
   disagreement from the before/after photographs and current gripper state.
10. For a caddy compartment retry, use destination="inside",
    release_on="floor", a small place_margin, and the measured nudge. For a
    mug empty-grasp retry, prefer a low-body ladder such as
    ["pca_axis@low", "top_down@low", "rim@top"].
11. `place_margin` and `grasp_depth` are scalar metres, never xyz vectors.
    The only vector argument is the two-number image-space `nudge`.
"""

REFLECT = """Instruction: {instruction}

The last attempt was: {call}
The skill reported: {observation}
Gripper now: {holding}

The first image is the scene before that attempt, the second is after it.
Look at what moved and where it ended up.

Reply with JSON and nothing else:
{{"what_happened": "<one sentence: where the object is now>",
  "looks_placed": true|false,
  "next": "check"|"retry"|"done"|"stop",
  "retry_args": {{... optional pickplace args to change on a retry ...}},
  "reason": "<which single argument would change the outcome, with direction>"}}

`retry_args` may use only these public controls:
- grasp: top_down, rim, pca_axis, affordance, graspnet; optional @top/@low.
- nudge: exactly two numeric metres `[right, up]`.
- release_on: `rim` or `floor`; destination: auto/object/region/inside/top.
- numeric grasp_depth/place_margin/yaw_deg; boolean centre/compensate.
Never invent handle/side/edge grasp names, prose nudges, string margins, pose
coordinates, or a new source/destination phrase.

Rules for `next`:
- `done` only if the object is clearly resting where the instruction asked.
- For an "on" placement, visible overlap in one camera view is not enough:
  require the moved object to be physically supported above the destination.
  If it lies on the table in front of/behind the destination, or only occludes
  it in projection, return `retry` even when their image masks overlap.
- On a small support such as a plate, tray, or burner, mere physical contact
  is not enough when the subsequent reliable check explicitly says the object
  centre is outside tolerance and supplies a small corrective nudge. Compare
  that metric evidence with the photographs. Return `retry` unless the check
  is visibly bound to the wrong object/support; a sideways object touching an
  edge is not by itself evidence that the requested `on` relation is complete.
- If that check is reliable and identity-clear, its explicit outside-tolerance
  result constrains the visual decision: overlap at the support edge cannot
  justify `done`. Return `retry` so the controller can execute the measured
  micro-push. Reserve visual override for a demonstrably wrong referent.
- For an "in" placement, a thin object leaning on a rim or protruding outside
  the named compartment is not complete; return `retry` so insertion can use
  a different presentation or contact finish.
- A relation such as left/right/front/back of a reference denotes its nearby
  designated placement region, not the entire half-plane anywhere on that
  side. If a fresh reliable check reports a small 0.5--5 cm offset and gives
  a concrete correction, normally return `retry` with that nudge unless the
  photographs clearly contradict the measured object or destination.
- When a geometric check is included in the report, treat it as advisory
  evidence rather than ground truth. You are the final visual state judge:
  return `done` when the photographs clearly show the requested placement,
  even if that check disagrees; return `retry` when the photographs clearly
  show a miss and name the one useful correction.
- `check` only when no post-motion geometric check has run yet and the images
  remain genuinely ambiguous.
- Treat the stated gripper status as proprioceptive evidence. If it says empty,
  do not claim the object is still held merely because the hand overlaps it in
  projection. After a broad-container release, rim occlusion or a rebound mask
  can make height unreliable; retry only when the images clearly bind the
  requested object outside the destination.
- If an attempted re-grasp made no motion and the unchanged current image now
  clearly shows the previously released object at the requested destination,
  return `done`; another grasp would only disturb it.
- `retry` only if the object is still on the table, or clearly beside / on the
  rim of the destination. Put concrete changes in `retry_args` (nudge,
  grasp as a list, release_on, destination, place_margin). Do not invent pose
  coordinates.
- `stop` if the arm is empty and further tries would just thrash.
"""


@dataclass
class Episode:
    instruction: str
    pick: str
    place: str
    turns: int = DEFAULT_TURNS
    max_pickplace_calls: int = DEFAULT_MAX_PICKPLACE_CALLS
    max_push_calls: int = DEFAULT_MAX_PUSH_CALLS
    max_insert_calls: int = DEFAULT_MAX_INSERT_CALLS
    # Full-task execution passes the planner-selected transport family and can
    # skip a redundant first VLM decision. ``model`` preserves the standalone
    # transport evaluator's inspect-and-select behaviour.
    initial_skill: str = "model"
    model: str = field(default_factory=lambda: os.environ.get("RACAP_MODEL", "gpt-5.5"))


@dataclass
class Outcome:
    success: bool
    turns: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    stopped: str = ""


def _json_from(reply: str) -> dict[str, Any]:
    """Pull the object out of a reply that may be fenced or prefaced."""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", reply, re.S)
    body = fenced.group(1) if fenced else reply
    start = body.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {reply[:200]}")
    try:
        value, _ = json.JSONDecoder().raw_decode(body[start:])
    except json.JSONDecodeError as exc:
        kind = "unterminated JSON" if exc.pos >= len(body[start:]) - 2 else "invalid JSON"
        raise ValueError(f"{kind} in reply: {reply[:200]}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON reply is not an object: {reply[:200]}")
    return value


def _ask_json_reply(
    system: str,
    prompt: str,
    *,
    images: list[str],
    model: str,
    max_tokens: int = 1800,
) -> tuple[str, dict[str, Any]]:
    """Ask for one JSON object, retrying only a malformed model reply.

    Network retries belong to :func:`ask`; this retry is different. Hosted
    vision models occasionally return a code fence cut off after the first
    sentence even though the request itself succeeded. Treating that transport
    artefact as visual evidence made reflection silently disappear in 23 of 87
    calls in one clean run. A fresh compact request is cheaper and safer than
    either acting on partial JSON or falling back to a task-specific rule.
    """
    last: Exception | None = None
    reply = ""
    for attempt in range(2):
        retry_note = (
            "\n\nYour previous reply was incomplete or not valid JSON. "
            "Try again and output one complete compact JSON object only; "
            "do not use Markdown."
            if attempt
            else ""
        )
        reply = ask(
            system,
            prompt + retry_note,
            images=images,
            model=model,
            max_tokens=max_tokens,
            cache=False,
        )
        try:
            return reply, _json_from(reply)
        except (TypeError, ValueError) as exc:
            last = exc
    raise ValueError(f"no usable JSON after two replies: {last}")


def _default_pickplace_args(episode: Episode) -> dict[str, Any]:
    return {"pick": episode.pick, "place": episode.place}


def _merge_retry(
    base: dict[str, Any], retry_args: dict[str, Any] | None, episode: Episode
) -> dict[str, Any]:
    """Build a pickplace call that keeps labels and applies one change set."""
    args = _default_pickplace_args(episode)
    if isinstance(retry_args, dict):
        for key, value in retry_args.items():
            if key in ("pick", "place") and value:
                args[key] = value
            elif key not in ("pick", "place") and value is not None:
                args[key] = value
    # Never drop the labels a retry forgot.
    args.setdefault("pick", episode.pick)
    args.setdefault("place", episode.place)
    # A lone grasp string used to burn the only attempt; prefer a short ladder.
    grasp = args.get("grasp")
    if isinstance(grasp, str):
        args["grasp"] = [grasp, "top_down", "rim", "pca_axis", "affordance"]
    return args


_GRASP_STRATEGIES = {
    "top_down",
    "rim",
    "pca_axis",
    "affordance",
    "graspnet",
    "wrist_closeloop",
}
_DESTINATIONS = {"auto", "object", "region", "inside", "top", "relation"}


def _spatially_qualified(label: str) -> bool:
    """Whether dropping the qualifier could select an identical distractor."""
    return bool(
        re.search(
            r"\b(?:at\s+the\s+|on\s+the\s+|in\s+the\s+)?"
            r"(?:front|back|left|right|middle|center|centre)\b|"
            r"\b(?:nearest|closest|farthest)\s+to\b",
            str(label).lower(),
        )
    )


def _qualified_visual_alias(alias: str, canonical: str) -> str:
    """Transfer a source selector to a verified appearance alias."""
    alias = " ".join(str(alias).split()).strip()
    canonical = " ".join(str(canonical).split()).strip()
    suffix = re.search(
        r"\b(?:at|on|in)\s+the\s+"
        r"(?:front|back|left|right|middle|center|centre)\b.*$",
        canonical,
        flags=re.IGNORECASE,
    )
    if suffix:
        return f"{alias} {suffix.group(0)}"
    leading = re.match(
        r"^(?:the\s+)?(front|back|left|right|middle|center|centre)\b",
        canonical,
        flags=re.IGNORECASE,
    )
    if leading:
        return f"{leading.group(1)} {alias}"
    return alias


def _valid_grasp(value: Any) -> str | list[str] | None:
    values = [value] if isinstance(value, str) else (value if isinstance(value, list) else [])
    clean: list[str] = []
    for item in values:
        name = str(item).strip()
        strategy, _, height = name.partition("@")
        if strategy in _GRASP_STRATEGIES and (not height or height in {"top", "low"}):
            clean.append(name)
    if not clean:
        return None
    return clean[0] if isinstance(value, str) else clean


def _uses_learned_grasp(value: Any) -> bool:
    """Whether a grasp request actually includes the learned 6-DoF family."""
    values = [value] if isinstance(value, str) else (value if isinstance(value, list) else [])
    return any(str(item).partition("@")[0].strip() == "graspnet" for item in values)


def _normalise_pickplace_args(
    raw: dict[str, Any],
    episode: Episode,
    *,
    allow_place_alias: bool = False,
    allow_pick_alias: bool = False,
) -> dict[str, Any]:
    """Make a model-produced call safe and keep semantic labels stable."""
    args = _default_pickplace_args(episode)
    raw_pick = str(raw.get("pick", ""))
    raw_place = str(raw.get("place", ""))
    if raw_pick and (
        allow_pick_alias or _normalised_label(raw_pick) == _normalised_label(episode.pick)
    ):
        args["pick"] = str(raw["pick"])
    if raw_place and (
        allow_place_alias or _normalised_label(raw_place) == _normalised_label(episode.place)
    ):
        args["place"] = str(raw["place"])

    grasp = _valid_grasp(raw.get("grasp"))
    if grasp is not None:
        args["grasp"] = grasp

    destination = raw.get("destination")
    if destination in _DESTINATIONS:
        args["destination"] = destination
    release_on = raw.get("release_on")
    if release_on in {"rim", "floor"}:
        args["release_on"] = release_on

    for key, lo, hi in (("grasp_depth", 0.005, 0.08), ("place_margin", 0.0, 0.10)):
        if key in raw:
            try:
                args[key] = max(lo, min(hi, float(raw[key])))
            except (TypeError, ValueError):
                pass

    nudge = raw.get("nudge")
    if isinstance(nudge, (list, tuple)) and len(nudge) == 2:
        try:
            args["nudge"] = [round(max(-0.08, min(0.08, float(v))), 3) for v in nudge]
        except (TypeError, ValueError):
            pass

    yaw = raw.get("yaw_deg")
    if yaw in (None, "auto") and "yaw_deg" in raw:
        args["yaw_deg"] = yaw
    elif "yaw_deg" in raw:
        try:
            args["yaw_deg"] = float(yaw)
        except (TypeError, ValueError):
            pass
    for key in ("centre", "compensate", "identity_check"):
        if isinstance(raw.get(key), bool):
            args[key] = raw[key]
    return args


def _normalise_push_args(
    raw: dict[str, Any], episode: Episode, measured: dict[str, Any]
) -> dict[str, Any]:
    """Push only the named object and only by the last measured correction."""
    nudge = measured.get("suggested_nudge") or [0.0, 0.0]
    try:
        nudge = [round(max(-0.05, min(0.05, float(v))), 3) for v in nudge[:2]]
    except (TypeError, ValueError):
        nudge = [0.0, 0.0]
    return {"pick": episode.pick, "place": episode.place, "nudge": nudge}


def _supports_insert(place: str) -> bool:
    """Whether the phrase denotes a bounded opening rather than an exterior.

    Shelves and racks are front-loaded cavities when language names the shelf
    itself (``on the cabinet shelf`` / ``under the cabinet shelf``), but their
    exposed top is an ordinary support surface.  Route by that semantic
    distinction and leave the insert Policy API to accept or reject the measured
    opening geometry.  This avoids treating every furniture noun as a narrow
    insertion while covering horizontal shelf apertures.
    """
    lowered = " ".join(str(place).lower().split())
    exterior_relation = bool(
        re.search(
            r"\b(?:on\s+)?top\s+of\b|\b(?:left|right|front|back)\s+of\b",
            lowered,
        )
    )
    if exterior_relation:
        return False
    # These nouns denote front-loaded bounded free space even when the planner
    # copied only the destination noun phrase (for example ``bottom drawer of
    # the cabinet`` and omitted the instruction's preceding ``in``).  The
    # insert primitive still performs a motion-free footprint/opening
    # preflight, so a visually exposed/non-narrow region can safely decline.
    if re.search(r"\b(?:drawer|compartment|slot|cubby|rack|shelf)\b", lowered):
        return True
    # Broad open receptacles need insertion only when language explicitly asks
    # for their interior.  ``on the tray`` remains ordinary support placement.
    return bool(
        re.search(
            r"\b(?:in|inside|into)\b.*\b(?:basket|caddy|tray|bin|box|container)\b",
            lowered,
        )
    )


def _requires_insert(place: str, instruction: str = "") -> bool:
    """Whether broad transport would contradict an unambiguous goal.

    ``_supports_insert`` is deliberately permissive: a shelf or rack *may*
    expose a front-loaded aperture.  It must not therefore be used as a hard
    action override.  Phrases such as ``on the wine rack`` are also ordinary
    support relations, and the visual agent can correctly choose pickplace.
    Reserve hard routing for explicit containment and nouns whose referent is
    intrinsically an opening.  A cabinet shelf is front-loaded even when the
    English relation says ``on``; a generic rack remains ambiguous and is
    eligible for insert only when the agent selects it from the image.
    """
    place_text = " ".join(str(place).lower().split())
    task_text = " ".join(str(instruction).lower().split())
    if re.search(
        r"\b(?:on\s+)?top\s+of\b|\b(?:left|right|front|back)\s+of\b",
        place_text,
    ):
        return False
    # A drawer is a bounded receptacle, but not intrinsically a *tight* one.
    # Bowls, cartons and bottles often have ample clearance and are safer with
    # the broad high-transit pickplace primitive.  Keep insert available to the
    # visual agent, but do not override its pickplace decision merely because
    # the destination noun is "drawer".  Slots, caddy compartments and shelf
    # apertures do impose presentation/footprint constraints regardless of the
    # transported object, so they remain hard insertion routes.
    if re.search(
        r"\b(?:compartment|slot|cubby|shelf|bookshelf)\b",
        place_text,
    ):
        return True
    relation_text = f"{task_text} {place_text}"
    return bool(
        re.search(
            r"\b(?:in|inside|into)\b[^.;,]{0,80}"
            r"\b(?:compartment|slot|cubby|rack|shelf|bookshelf|"
            r"basket|caddy|tray|bin|box|container)\b",
            relation_text,
        )
    )


def _normalise_insert_args(
    raw: dict[str, Any],
    episode: Episode,
    *,
    allow_place_alias: bool = False,
    allow_pick_alias: bool = False,
) -> dict[str, Any]:
    """Keep insert labels stable and expose only its geometric controls."""
    base = _normalise_pickplace_args(
        raw,
        episode,
        allow_place_alias=allow_place_alias,
        allow_pick_alias=allow_pick_alias,
    )
    args: dict[str, Any] = {"pick": base["pick"], "place": base["place"]}
    grasp = _valid_grasp(raw.get("grasp"))
    if grasp is not None:
        allowed = [grasp] if isinstance(grasp, str) else list(grasp)
        allowed = [
            value
            for value in allowed
            if value.split("@", 1)[0]
            in {
                "top_down",
                "pca_axis",
                "graspnet",
            }
        ]
        if allowed:
            args["grasp"] = allowed[0] if isinstance(grasp, str) else allowed
    for key, lo, hi in (
        ("grasp_depth", 0.005, 0.08),
        ("uncertainty_m", 0.0, 0.025),
        ("insertion_depth", 0.005, 0.06),
    ):
        if key in raw:
            try:
                args[key] = max(lo, min(hi, float(raw[key])))
            except (TypeError, ValueError):
                pass
    # Motion semantics come from the original instruction, not from a visual
    # alias such as "black box" that may be needed only for grounding.
    frontload = bool(
        re.search(r"\b(?:shelf|rack)\b", episode.place.lower())
        and not re.search(r"\b(?:on\s+)?top\s+of\b", episode.place.lower())
    )
    if frontload:
        args["frontload"] = True
        # A verified inventory alias may locate the right parent fixture while
        # omitting the task's tier/opening semantics (e.g. "black box" for a
        # cabinet shelf). Keep both signals instead of forcing either one to do
        # the other's job.
        args["destination_description"] = episode.place
    return args


def _check_supports_push(report: dict[str, Any]) -> bool:
    """Whether a small, supported lateral correction is safer than re-pick."""
    try:
        offset = float(report.get("offset_cm")) / 100.0
        nudge = [float(v) for v in (report.get("suggested_nudge") or [])]
    except (TypeError, ValueError):
        return False
    return bool(
        report.get("reliable")
        # Exposed surfaces and planar spatial relations share a support plane
        # and leave room for a short closed-finger sweep. Container openings do
        # not: a lateral push there can catch a rim or eject the object.
        and report.get("destination_kind") in {"surface", "relation"}
        and report.get("identity_clear", True)
        and report.get("landed")
        and not report.get("centred")
        and 0.004 < offset <= 0.055
        and len(nudge) == 2
        and any(abs(v) >= 0.003 for v in nudge)
    )


def _semantic_first_call(args: dict[str, Any], episode: Episode) -> dict[str, Any]:
    """Apply only high-confidence instruction priors to the first attempt."""
    # Make the first manipulation repeatable. Free-form model knobs created
    # large run-to-run swings on identical initial states; these few priors are
    # the ones supported by the clean-set trajectories.
    args = _default_pickplace_args(episode)
    place = episode.place.lower()
    relation = re.search(r"\b(?:left|right|front|back|behind)\s+(?:side\s+)?of\b", place)
    if "bowl" in episode.pick.lower():
        args["grasp"] = "rim"
    # A compact exposed support needs the carried body, rather than the wrist,
    # centred over it.  The controller now accepts a held-object correction
    # only when the observed instance is physically attached near the wrist,
    # so this closed-loop first placement is safe even with duplicate bowls.
    # Relation targets and bounded containers keep their separate geometry.
    if (
        not relation
        and re.search(r"\b(?:plate|dish|saucer)\b", place)
        and not re.search(r"\b(?:drawer|basket|tray|compartment)\b", place)
    ):
        # Compact supports have little lateral tolerance and tall vessels tip
        # when dropped from the generic three-centimetre transport margin.
        # Centre the *carried body* in closed loop, then lower its base to a
        # few millimetres above the measured support floor before releasing.
        # This is contact-conditioned support placement, not an object-class
        # waypoint: it applies equally to mugs, bowls, cans, and blocks on any
        # shallow exposed dish.
        args.update(
            {
                "centre": True,
                "release_on": "floor",
                "place_margin": 0.003,
            }
        )
    if relation and "mug" not in episode.pick.lower() and "bowl" not in episode.pick.lower():
        args["compensate"] = True
    if not relation and "compartment" in place:
        args.update({"destination": "region", "release_on": "floor"})
    elif not relation and "drawer" in place:
        args["destination"] = "region"
        # A tall rigid object is safer dropped over the rim; lowering it toward
        # the drawer floor catches its side on the front edge. Bowls benefit
        # from the lower release and have already cleared that edge reliably.
        args["release_on"] = "floor" if "bowl" in episode.pick.lower() else "rim"
    return args


def _retry_after_check(
    episode: Episode,
    report: dict[str, Any],
    failure_mode: str,
    reflected: dict[str, Any] | None,
) -> dict[str, Any]:
    """Translate measured failure evidence into one coherent retry."""
    raw = dict(reflected or {})
    place = episode.place.lower()
    if failure_mode in {"empty_grasp", "wrong_object_grasp"}:
        if failure_mode == "wrong_object_grasp":
            raw["identity_check"] = True
        if episode.pick.lower().startswith("new "):
            # LIBERO occasionally prefixes the answer-key label with ``new``
            # although the visual model only recognises the package name.
            # Changing the label is the retry; retain the strong default grasp.
            raw.setdefault("pick", episode.pick[4:])
            raw.pop("grasp", None)
        # A valid visual-agent proposal is the runtime policy decision.  The
        # previous controller silently replaced every proposal with a fixed
        # object-class ladder, so neither SFT nor RL could learn to skip an
        # already-failed family.  Keep a deterministic analytic fallback only
        # when the model supplied no admissible grasp control.
        explicit_grasp = _valid_grasp(raw.get("grasp"))
        if explicit_grasp is not None:
            raw["grasp"] = explicit_grasp
        elif "mug" in episode.pick.lower():
            raw.update(
                {
                    "grasp": ["pca_axis@low", "top_down@low", "rim@top"],
                    "grasp_depth": 0.04,
                }
            )
        else:
            raw["grasp"] = ["top_down@low", "pca_axis@low", "rim@top"]
        raw.pop("nudge", None)
    else:
        nudge = list(report.get("suggested_nudge") or [])
        if nudge and any(abs(float(v)) > 1e-4 for v in nudge):
            raw["nudge"] = nudge
        if re.search(r"\b(?:left|right|front|back|behind)\s+(?:side\s+)?of\b", place):
            if "mug" in episode.pick.lower():
                # Mugs commonly topple beside a destination; the observed
                # image correction and a lower release reproduced the winning
                # recovery, whereas carry-centre compensation did not.
                raw["release_on"] = "floor"
                raw.pop("compensate", None)
            else:
                # The first miss may simply be the object hanging beside the
                # hand; measure that directly instead of trusting a large
                # post-drop box.
                raw.pop("nudge", None)
                raw["compensate"] = True
        elif re.search(r"\bcompartment\b", place):
            raw.update(
                {
                    "destination": "inside",
                    "release_on": "floor",
                    "place_margin": 0.01,
                    "compensate": True,
                }
            )
        elif re.search(r"\b(?:drawer|basket|tray)\b", place):
            if "drawer" in place:
                # Reflections describe the desired relation in prose and may
                # return values such as release_on="inside".  The controller
                # API needs a physical reference surface; a contained drawer
                # retry should therefore deterministically descend relative
                # to its measured floor.
                raw.update({"destination": "inside", "release_on": "floor", "place_margin": 0.01})
            else:
                raw.setdefault("release_on", "floor")
        elif re.search(r"\bplate\b", place) and " of " not in place:
            # A rim grasp makes the carried centre differ from the hand centre;
            # close the final few centimetres visually on a plate retry. Keep
            # the release contact-conditioned as on the first attempt: raising
            # it again made tall vessels fall sideways after a good correction.
            raw.update({"centre": True, "place_margin": 0.003, "release_on": "floor"})
    return _normalise_pickplace_args(
        raw,
        episode,
        allow_place_alias=(failure_mode == "not_grounded"),
        allow_pick_alias=(failure_mode in {"empty_grasp", "wrong_object_grasp", "not_grounded"}),
    )


def run_episode(
    runtime,
    skill,
    episode: Episode,
    *,
    verbose: bool = False,
    system_prompt: str | None = None,
    reflect_prompt: str | None = None,
    session: Session | None = None,
) -> Outcome:
    """Drive one episode. Prompts are overridable so they can be co-evolved."""
    session = Session(runtime, skill) if session is None else session
    outcome = Outcome(success=False, turns=0)
    runtime_memory = load_runtime_memory(scope="transport")
    system = with_runtime_memory(
        (system_prompt or SYSTEM).format(tools=describe_tools(list(TRANSPORT_TOOL_NAMES))),
        runtime_memory,
    )
    reflect_template = reflect_prompt or REFLECT
    history: list[str] = []
    before = session.frame()
    requested_initial = str(episode.initial_skill or "model").strip().lower()
    initial_action = (
        "insert"
        if (
            requested_initial == "insert"
            or (
                requested_initial != "pickplace"
                and _requires_insert(episode.place, episode.instruction)
            )
        )
        else "pickplace"
    )
    pending_action: dict[str, Any] | None = (
        {
            "thought": (
                "parent-selected bounded transport; preflight insertion geometry"
                if initial_action == "insert"
                else "dispatch the parent-selected transport with general defaults"
            ),
            "action": initial_action,
            "args": {"pick": episode.pick, "place": episode.place},
        }
        if requested_initial in {"pickplace", "insert"}
        else None
    )
    pickplace_attempts = 0
    pickplace_calls = 0
    push_calls = 0
    insert_calls = 0
    last_check_ok = False
    last_check_reliable = True
    must_check = False
    last_failure_mode = ""
    last_grounding_failure = ""
    reflected_retry: dict[str, Any] | None = None
    allow_unreliable_retry = False
    unreliable_check_streak = 0
    verified_place_aliases: set[str] = set()
    verified_pick_aliases: set[str] = set()
    active_place_alias = ""
    active_pick_alias = ""
    allow_surface_fallback = False
    semantic_place_variant_tried = False
    has_successful_place = False
    has_released = False
    consecutive_empty_grasps = 0
    learned_grasp_attempted = False
    last_transport_action = ""
    last_check_report: dict[str, Any] = {}
    push_blocked_until_pickplace = False
    last_passive_look_signature = ""
    repeated_passive_looks = 0
    # Keep the images from the latest manipulation until its mandatory
    # geometric check resolves.  Successful releases need no separate visual
    # diagnosis; on a miss, the same before/after pair can still tell the
    # agent which argument to change.
    last_motion_context: tuple[Any, Any] | None = None

    def compact_history() -> str:
        """Keep only recent actionable evidence in the free-decision prompt."""
        recent = history[-6:]
        return "\n".join(value[:700] for value in recent)

    def reflect_on_motion(context, check_step=None) -> dict[str, Any]:
        motion_before, motion_step = context
        observation = motion_step.observation[:1800]
        if check_step is not None:
            observation += (
                "\nThe subsequent geometric check reported: "
                + _advisory_check_observation(check_step.observation[:1800])
            )
        if check_step is not None and not bool(
            (check_step.report or {}).get("transport_identity_established", False)
        ):
            observation += (
                "\nIdentity warning: the geometric check did not establish "
                "causal transport identity. Any object name in that check "
                "is a grounding hypothesis, not an observed fact. Inspect "
                "the before/after photographs independently: first verify "
                "the requested object category, then whether the originally "
                "qualified instance actually left its source."
            )
        try:
            holding_evidence = _reflection_holding_evidence(
                motion_step, session.holding()
            )
            reflection, diagnosis = _ask_json_reply(
                "You are diagnosing a robot attempt from before/after "
                "photographs. Reply with JSON only.",
                with_runtime_memory(
                    reflect_template.format(
                        instruction=episode.instruction,
                        call=motion_step.for_history().split("\n")[0],
                        observation=observation,
                        holding=holding_evidence,
                    ),
                    runtime_memory,
                ),
                images=[motion_before, motion_step.image],
                model=episode.model,
                max_tokens=900,
            )
        except Exception as exc:
            diagnosis = {
                "what_happened": f"(could not be diagnosed: {exc})",
                "looks_placed": False,
                "next": "check",
                "retry_args": {},
                "reason": "fallback to measurement",
            }
            reflection = json.dumps(diagnosis)

        outcome.reflections.append(
            reflection if isinstance(reflection, str) else json.dumps(diagnosis, ensure_ascii=False)
        )
        history.append(
            "  looking at the result: "
            f"{diagnosis.get('what_happened', '')} "
            f"(next={diagnosis.get('next')}; "
            f"{diagnosis.get('reason', '')})"
        )
        if verbose:
            print("    reflection: " + json.dumps(diagnosis, ensure_ascii=False)[:320])
        return diagnosis

    for turn in range(episode.turns):
        if pending_action is not None:
            decision = pending_action
            pending_action = None
        else:
            prompt = (
                f"Instruction: {episode.instruction}\n"
                f"The object to move is {episode.pick!r}; destination phrase "
                f"is {episode.place!r}.\n\n"
                + (
                    "What has happened so far:\n" + compact_history() + "\n\n"
                    if history
                    else "Nothing has been attempted yet.\n\n"
                )
                + f"Right now, {session.holding()}.\n"
                + (
                    "`check` already said this looks placed -- prefer `done`.\n"
                    if last_check_ok
                    else ""
                )
                + "The image is the scene right now. Decide the next action."
            )
            try:
                _, decision = _ask_json_reply(
                    system,
                    prompt,
                    images=[before],
                    model=episode.model,
                    max_tokens=900,
                )
            except Exception as exc:
                # Recoverable: if we already have a good check, stop as done;
                # if nothing tried yet, try defaults once.
                if last_check_ok:
                    decision = {
                        "thought": "check already passed",
                        "action": "done",
                        "args": {"reason": "check endorsed placement"},
                    }
                elif pickplace_attempts == 0:
                    decision = {
                        "thought": "model unusable; try defaults",
                        "action": "pickplace",
                        "args": _default_pickplace_args(episode),
                    }
                else:
                    outcome.stopped = f"the model did not answer usably: {exc}"
                    break

        action = str(decision.get("action", "")).strip()
        args = decision.get("args") or {}
        thought = str(decision.get("thought", ""))

        # This controller is deliberately incapable of repeating a completed
        # task-level prerequisite.  In the 90-task baseline, leaked all-tools
        # context caused transport subgoals to call articulate/control again
        # on t001/t008/t021/t045.  Those calls spent a turn and could undo the
        # state that Full ReAct had just verified.
        if action not in TRANSPORT_ACTIONS:
            if must_check or has_released:
                action = "check"
                args = {"pick": episode.pick, "place": episode.place}
                thought = "override: transport controller cannot repeat task-level skills"
            elif pickplace_attempts == 0 and insert_calls == 0:
                action = initial_action
                args = {"pick": episode.pick, "place": episode.place}
                thought = "override: dispatch only the assigned transport subgoal"
            else:
                outcome.stopped = (
                    f"transport model proposed out-of-scope action {action!r}; "
                    "stopped without changing unrelated task state"
                )
                break

        # Narrow openings get a motion-free configuration-space preflight on
        # the first turn.  This is a semantic class rule, not a task/object
        # patch: insert itself decides from measured geometry whether it can
        # act, and leaves the scene untouched when it cannot.
        if turn == 0 and _requires_insert(episode.place, episode.instruction):
            action = "insert"
            args = {"pick": episode.pick, "place": episode.place}
            thought = "override: preflight the bounded opening before transport"

        # Every manipulation must be checked before a further physical action.
        if must_check and action != "check":
            action = "check"
            args = {"pick": episode.pick, "place": episode.place}
            thought = "override: every manipulation must be measured before another action"

        if action == "done" and not last_check_ok:
            action = "check"
            args = {"pick": episode.pick, "place": episode.place}
            thought = "override: done requires an endorsed check"

        # A bounded opening remains an insertion problem on every retry, not
        # only on turn zero.  In particular, a failed edge grasp must not let a
        # later model response silently replace front-loading with pickplace.
        if (
            action == "pickplace"
            and _requires_insert(episode.place, episode.instruction)
            and not allow_surface_fallback
        ):
            if insert_calls < episode.max_insert_calls:
                action = "insert"
                # Preserve the agent's admissible grasp preference when
                # changing only the motion family.  Dropping this dictionary
                # silently discarded a requested learned 6-DoF recovery and
                # made every retry repeat the same analytic grasp ladder.
                args = dict(args)
                args.setdefault("pick", episode.pick)
                args.setdefault("place", episode.place)
                thought = "override: bounded openings require insertion geometry"
            elif has_released:
                if last_check_report:
                    history.append(
                        "  insertion budget exhausted after the release was "
                        "already checked; another identical check cannot "
                        "change the scene"
                    )
                    outcome.stopped = (
                        "insertion budget exhausted after a checked release; "
                        "stopped instead of repeating checks"
                    )
                    break
                action = "check"
                args = {"pick": episode.pick, "place": episode.place}
                thought = "override: insertion budget exhausted; verify the release"
            else:
                history.append(
                    "  insertion budget exhausted before any release; "
                    "passive inspection cannot create a new grasp"
                )
                outcome.stopped = (
                    "insertion budget exhausted without a release; "
                    "stopped instead of repeating identical looks"
                )
                break

        # A model may use a successful visual alias to manipulate, but the
        # verifier must always answer the original instruction. Otherwise a
        # failed alias check can be bypassed by asking the easier question
        # "is the binder on the black box?" instead of "is the book on the
        # cabinet shelf?".
        if action == "check":
            args = {"pick": episode.pick, "place": episode.place}

        if action == "insert":
            if last_check_ok and (pickplace_calls + insert_calls) >= 1:
                action = "done"
                args = {"reason": "check already said placed; refusing another insert"}
                thought = "override: do not re-grasp a placed object"
            elif not last_check_reliable and has_released and not allow_unreliable_retry:
                action = "look"
                args = {}
                thought = "override: localization was unreliable; do not re-grasp"
            elif not _supports_insert(episode.place):
                action = "pickplace"
                thought = "override: insert is only for bounded narrow openings"
            elif insert_calls >= episode.max_insert_calls:
                if has_released:
                    if last_check_report:
                        history.append(
                            "  insertion budget exhausted after the release "
                            "was already checked; stop the no-information loop"
                        )
                        outcome.stopped = (
                            "insertion budget exhausted after a checked release; "
                            "stopped instead of repeating checks"
                        )
                        break
                    action = "check"
                    args = {"pick": episode.pick, "place": episode.place}
                    thought = "override: insertion budget exhausted; verify the release"
                else:
                    # A passive look cannot change a scene after every
                    # admissible insertion/grasp proposal has already failed.
                    # Previously this branch generated identical inventory
                    # looks until the outer turn budget was exhausted.
                    history.append(
                        "  insertion budget exhausted before any release; "
                        "passive inspection cannot create a new grasp"
                    )
                    outcome.stopped = (
                        "insertion budget exhausted without a release; "
                        "stopped instead of repeating identical looks"
                    )
                    break
            else:
                if active_place_alias:
                    args = dict(args)
                    args["place"] = active_place_alias
                if active_pick_alias:
                    args = dict(args)
                    args["pick"] = active_pick_alias
                requested_place = str(args.get("place", ""))
                verified_alias = _normalised_label(requested_place) in verified_place_aliases
                requested_pick = str(args.get("pick", ""))
                args = _normalise_insert_args(
                    args,
                    episode,
                    allow_place_alias=verified_alias,
                    allow_pick_alias=(_normalised_label(requested_pick) in verified_pick_aliases)
                    or (
                        not _spatially_qualified(episode.pick)
                        and last_failure_mode
                        in {"empty_grasp", "wrong_object_grasp", "not_grounded"}
                    ),
                )

        if action == "push":
            if not _check_supports_push(last_check_report) or push_calls >= episode.max_push_calls:
                action = "check"
                args = {"pick": episode.pick, "place": episode.place}
                thought = "override: push requires a fresh reliable small-offset check"
            else:
                args = _normalise_push_args(args, episode, last_check_report)

        if action == "pickplace":
            if last_check_ok and pickplace_calls >= 1:
                action = "done"
                args = {"reason": "check already said placed; refusing another pick"}
                thought = "override: do not re-grasp a placed object"
            elif not last_check_reliable and pickplace_calls >= 1 and not allow_unreliable_retry:
                action = "look"
                args = {}
                thought = "override: localization was unreliable; do not re-grasp"
            elif pickplace_calls >= episode.max_pickplace_calls:
                action = "check"
                args = {"pick": episode.pick, "place": episode.place}
                thought = "override: manipulation budget exhausted"
            else:
                # Once a destination alias has been verified from a clean
                # pre-action crop, keep that identity stable for the episode.
                # Re-running inventory after the object covers the target can
                # make the detector rename the same structure or even confuse
                # the moved object with its destination.
                if active_place_alias:
                    args = dict(args)
                    args["place"] = active_place_alias
                if active_pick_alias:
                    args = dict(args)
                    args["pick"] = active_pick_alias
                requested_place = str(args.get("place", ""))
                verified_alias = _normalised_label(requested_place) in verified_place_aliases
                requested_pick = str(args.get("pick", ""))
                args = _normalise_pickplace_args(
                    args,
                    episode,
                    # Only aliases admitted by a closed-set crop check enter
                    # this set.  Keeping such an alias admissible after the
                    # first manipulation is what lets measured corrections
                    # return to the same physical destination.
                    allow_place_alias=verified_alias,
                    allow_pick_alias=(_normalised_label(requested_pick) in verified_pick_aliases)
                    or (
                        not _spatially_qualified(episode.pick)
                        and (
                            last_failure_mode in {"empty_grasp", "wrong_object_grasp"}
                            or (
                                last_failure_mode == "not_grounded"
                                and last_grounding_failure == "pick"
                            )
                        )
                    ),
                )
                if pickplace_attempts == 0 and not allow_surface_fallback:
                    args = _semantic_first_call(args, episode)

        if verbose:
            print(
                f"  turn {turn}: {action}({json.dumps(args, ensure_ascii=False)})"
                f"\n    thought: {thought}"
            )

        if action in {"pickplace", "insert"}:
            learned_grasp_attempted = bool(
                learned_grasp_attempted or _uses_learned_grasp(args.get("grasp"))
            )
        step = session.run(action, args, thought)
        if action in {"pickplace", "insert"}:
            # Remember the agent-approved manipulation family even when a
            # motion-free grounding/preflight fails.  Alias recovery and
            # retries should preserve that choice for ambiguous rack/shelf
            # goals instead of reclassifying them from the noun alone.
            last_transport_action = action
        outcome.turns = turn + 1
        outcome.steps.append(
            {
                "turn": turn,
                "action": action,
                "args": args,
                "thought": thought,
                "observation": step.observation,
                "report": step.report,
            }
        )
        if verbose:
            print(f"    -> {step.observation}")

        infrastructure_failure = (step.report or {}).get("infrastructure_failure")
        if infrastructure_failure:
            # No agent decision can be grounded in a response that never
            # arrived or could not be parsed.  Stop this episode as
            # inconclusive; evaluator evidence validation will invalidate the
            # batch instead of scoring it as a policy failure.
            outcome.stopped = (
                "perception infrastructure unavailable; evaluation is inconclusive: "
                + json.dumps(infrastructure_failure, ensure_ascii=False)
            )
            history.append(step.for_history())
            break

        if action == "done":
            outcome.success = True
            outcome.stopped = "the agent judged it complete"
            break

        if action == "look":
            look_report = step.report or {}
            supplied_alias = bool(
                look_report.get("verified_source_aliases")
                or look_report.get("verified_destination_aliases")
            )
            signature = " ".join(str(step.observation).lower().split())
            if supplied_alias:
                # A verified alias is new actionable information even when
                # the prose inventory happens to be unchanged.
                last_passive_look_signature = ""
                repeated_passive_looks = 0
            elif signature and signature == last_passive_look_signature:
                repeated_passive_looks += 1
            else:
                last_passive_look_signature = signature
                repeated_passive_looks = 1
            if repeated_passive_looks >= 2:
                history.append(step.for_history())
                outcome.stopped = (
                    "two passive looks returned the same inventory without "
                    "new aliases or motion; stopped the no-information loop"
                )
                break
        else:
            last_passive_look_signature = ""
            repeated_passive_looks = 0

        if action == "insert":
            insert_calls += 1
            insert_report = step.report or {}
            last_failure_mode = str(insert_report.get("failure_mode", ""))
            last_grounding_failure = str(insert_report.get("grounding_failure", ""))
            preflight_only = bool(insert_report.get("preflight_only"))
            released_this_call = bool(insert_report.get("release_xy"))
            has_released = bool(has_released or released_this_call)
            if step.success:
                has_successful_place = True
            if preflight_only:
                # No object state changed.  A bounded opening must remain an
                # insertion problem: silently downgrading to pickplace turns a
                # front-loaded shelf/rack into a top-down drop and is almost
                # guaranteed to collide with its roof or miss its cavity.
                if last_failure_mode == "broad_container_fit":
                    # The insert Policy API measured both the object footprint and
                    # opening and found no tight-fit presentation problem.  It
                    # is therefore safe to obey that geometric recommendation
                    # and return control to broad high-transit transport.
                    allow_surface_fallback = True
                    pending_action = {
                        "thought": (
                            "measured footprint has ample container clearance; "
                            "use broad high-transit placement"
                        ),
                        "action": "pickplace",
                        "args": _default_pickplace_args(episode),
                    }
                elif last_failure_mode in {"no_feasible_insertion", "no_feasible_region"}:
                    history.append(step.for_history())
                    outcome.stopped = (
                        "measured opening configuration space is empty; "
                        "stopped without an unsafe top-down fallback"
                    )
                    break
                if last_failure_mode == "not_grounded" and last_grounding_failure == "destination":
                    pending_action = {
                        "thought": "opening was not grounded; inspect the visual inventory",
                        "action": "look",
                        "args": {
                            "verify_destination": episode.place,
                            "exclude": episode.pick,
                        },
                    }
                elif last_failure_mode == "not_grounded" and last_grounding_failure == "pick":
                    pending_action = {
                        "thought": (
                            "source was not grounded; inspect independently verified source aliases"
                        ),
                        "action": "look",
                        "args": {"verify_source": episode.pick},
                    }
                elif last_failure_mode in {"empty_grasp", "wrong_object_grasp"}:
                    consecutive_empty_grasps += 1
                    pending_action = {
                        "thought": (
                            "the opening is feasible but the transport grasp "
                            "failed; retry the insertion primitive"
                        ),
                        "action": "insert",
                        "args": _normalise_insert_args({}, episode),
                    }
                elif _supports_insert(episode.place):
                    pending_action = {
                        "thought": (
                            "bounded-opening geometry was inconclusive; inspect "
                            "the destination instead of top-down placement"
                        ),
                        "action": "look",
                        "args": {
                            "verify_destination": episode.place,
                            "exclude": episode.pick,
                        },
                    }
                else:
                    pending_action = {
                        "thought": "insert preflight declined; use the broad transport fallback",
                        "action": "pickplace",
                        "args": _default_pickplace_args(episode),
                    }
                must_check = False
                last_check_ok = False
                last_check_reliable = True
                allow_unreliable_retry = False
            else:
                if last_failure_mode in {"empty_grasp", "wrong_object_grasp"}:
                    consecutive_empty_grasps += 1
                else:
                    consecutive_empty_grasps = 0
                must_check = bool(
                    released_this_call
                    or last_failure_mode
                    not in {"empty_grasp", "wrong_object_grasp", "not_grounded"}
                )
                last_check_ok = False
                last_check_reliable = True
                allow_unreliable_retry = False
                unreliable_check_streak = 0
                push_blocked_until_pickplace = False

        if action == "pickplace":
            pickplace_attempts += 1
            last_failure_mode = str((step.report or {}).get("failure_mode", ""))
            last_grounding_failure = str((step.report or {}).get("grounding_failure", ""))
            # A grounding failure never moved the arm. It consumes a ReAct
            # tool turn, but not one of the eight manipulation slots. Keeping
            # those counters separate leaves room to act after a visual
            # inventory supplies a usable alias.
            if last_failure_mode != "not_grounded":
                pickplace_calls += 1
            released_before_call = has_released
            released_this_call = bool((step.report or {}).get("release_xy"))
            has_released = bool(has_released or released_this_call)
            if step.success:
                has_successful_place = True
            if last_failure_mode in {"empty_grasp", "wrong_object_grasp"}:
                consecutive_empty_grasps += 1
            else:
                consecutive_empty_grasps = 0
            if (
                has_released
                and last_failure_mode == "not_grounded"
                and last_grounding_failure == "pick"
            ):
                # After a release, disappearance of the moved object commonly
                # means it is occluded inside/on the destination. No corrected
                # pick pose exists, so further blind retries cannot improve the
                # state and may disturb an already successful placement.
                history.append(step.for_history())
                outcome.stopped = (
                    "the object was released earlier and is no longer "
                    "groundable; stopped without another blind grasp"
                )
                break
            # A failed grasp/place lookup normally changes no task state.  But
            # after *any* earlier release, an empty grasp is positive evidence
            # that the object may already be at the destination.  Checking is
            # then mandatory; retrying the grasp ladder only thrashes a placed
            # object and was the source of the repeated t009 calls.
            must_check = bool(
                released_this_call
                or (
                    released_before_call
                    and last_failure_mode in {"empty_grasp", "wrong_object_grasp", "not_grounded"}
                )
                or last_failure_mode not in {"empty_grasp", "wrong_object_grasp", "not_grounded"}
            )
            last_check_ok = False
            last_check_reliable = True
            allow_unreliable_retry = False
            unreliable_check_streak = 0
            push_blocked_until_pickplace = False

        if action == "push":
            push_calls += 1
            last_push_failure = str((step.report or {}).get("failure_mode", ""))
            if last_push_failure in {
                "no_motion",
                "wrong_direction",
                "not_grounded",
                "unsafe_push",
                "push_failed",
                "verification_inconclusive",
            }:
                push_blocked_until_pickplace = True
            must_check = True
            last_check_ok = False
            last_check_reliable = True
            allow_unreliable_retry = False

        if action == "check":
            must_check = False
            last_check_report = dict(step.report or {})
            check_failure_mode = str((step.report or {}).get("failure_mode", ""))
            if check_failure_mode:
                last_failure_mode = check_failure_mode
            last_check_ok = bool(step.success)
            last_check_reliable = bool((step.report or {}).get("reliable", True))
            unreliable_check_streak = unreliable_check_streak + 1 if not last_check_reliable else 0
            visibly_recoverable_miss = bool(
                (step.report or {}).get("visibly_recoverable_miss")
                or _visibly_recoverable_miss(step.report or {})
            )
            # Measurement is a cheap first filter.  On disagreement the
            # visual agent is the final state authority: its explicit
            # done/retry/stop decision must not be silently overwritten by
            # recovery heuristics below.
            preserve_without_action = bool(
                (step.report or {}).get("preserve_state") and not visibly_recoverable_miss
            )
            visual_next = ""
            # A positive geometric check proves where *an* object landed, but
            # not which member of a relational set was transported.  When the
            # skill could not establish causal identity in-hand, give the
            # authoritative visual agent the before/after scene even if the
            # destination geometry itself passed.  This catches a wrong
            # look-alike placed perfectly inside the requested receptacle
            # without making an auxiliary identity flag a hard veto.
            needs_identity_review = bool(
                last_check_ok
                and _spatially_qualified(episode.pick)
                and not bool((step.report or {}).get("transport_identity_established", False))
                and last_motion_context is not None
            )
            if needs_identity_review:
                diagnosis = reflect_on_motion(last_motion_context, step)
                visual_next = str(diagnosis.get("next", "check")).strip().lower()
                if visual_next == "done":
                    history.append(
                        "  visual state authority confirmed relational source "
                        "identity after the geometric check"
                    )
                elif visual_next == "retry":
                    last_check_ok = False
                    candidate = diagnosis.get("retry_args")
                    reflected_retry = candidate if isinstance(candidate, dict) else None
                elif visual_next == "stop":
                    history.append(step.for_history())
                    outcome.stopped = (
                        "visual state authority stopped recovery after the "
                        "relational identity review"
                    )
                    last_motion_context = None
                    break
                else:
                    # ``check`` means the agent did not endorse completion.
                    # Keep the geometric result as advisory and preserve the
                    # scene for a passive follow-up rather than auto-done.
                    last_check_ok = False
                last_motion_context = None
            if last_check_ok:
                last_motion_context = None
            elif last_motion_context is not None:
                diagnosis = reflect_on_motion(last_motion_context, step)
                visual_next = str(diagnosis.get("next", "check")).strip().lower()
                if visual_next == "done":
                    last_check_ok = True
                    history.append(
                        "  visual state authority accepted the placement "
                        "despite the advisory geometric check"
                    )
                elif visual_next == "retry":
                    candidate = diagnosis.get("retry_args")
                    reflected_retry = candidate if isinstance(candidate, dict) else None
                elif visual_next == "stop":
                    history.append(step.for_history())
                    outcome.stopped = (
                        "visual state authority stopped recovery after the post-motion inspection"
                    )
                    last_motion_context = None
                    break
                last_motion_context = None
            if (
                not last_check_ok
                and visual_next != "retry"
                and bool((step.report or {}).get("preserve_state"))
                and not visibly_recoverable_miss
            ):
                history.append(step.for_history())
                outcome.stopped = (
                    "post-release evidence was ambiguous; preserved the "
                    "current scene instead of issuing a destructive re-grasp"
                )
                break
            if not last_check_ok and unreliable_check_streak >= 2:
                history.append(step.for_history())
                outcome.stopped = (
                    "two checks after the same manipulation remained "
                    "unreliable; stopped without changing the scene"
                )
                break
            if last_check_ok:
                pending_action = {
                    "thought": (
                        "visual state authority endorsed the placement"
                        if visual_next == "done"
                        else "check endorsed the placement"
                    ),
                    "action": "done",
                    "args": {"reason": step.observation[:160]},
                }
            elif has_released and consecutive_empty_grasps >= 2:
                history.append(step.for_history())
                outcome.stopped = (
                    "two consecutive empty grasps after an earlier release; "
                    "stopped instead of disturbing the scene again"
                )
                break
            elif visual_next == "retry":
                # The visual agent is the state authority.  In particular, an
                # auxiliary check can be globally unreliable because its old
                # release coordinate no longer agrees with a freshly grounded
                # object, while the before/after images still make a failed
                # transport obvious.  Do not let the generic ``look first``
                # recovery below overwrite that explicit decision.
                # ``retry`` is a state decision, not a mandate to re-grasp.
                # When the independent geometry has already established a
                # supported, identity-clear, small planar miss, the least
                # destructive implementation of that decision is the
                # measured micro-push.  Previously this branch always emitted
                # pickplace before the later push selector could run, causing
                # correctly landed bowls and mugs to be picked up repeatedly.
                selected_retry_action = str(
                    (reflected_retry or {}).get("_transport_action") or ""
                ).strip()
                if reflected_retry:
                    reflected_retry.pop("_transport_action", None)
                if selected_retry_action == "look":
                    retry_action = "look"
                    retry_args = {
                        "verify_destination": episode.place,
                        "exclude": episode.pick,
                    }
                elif (
                    selected_retry_action == "insert"
                    or last_transport_action == "insert"
                    or _requires_insert(episode.place, episode.instruction)
                ):
                    retry_action = "insert"
                    retry_args = _normalise_insert_args(reflected_retry or {}, episode)
                elif (
                    _check_supports_push(step.report or {})
                    and not push_blocked_until_pickplace
                    and push_calls < episode.max_push_calls
                    and turn + 1 < episode.turns
                ):
                    # ``retry`` is the visual agent's authoritative state
                    # decision.  The tool choice should still minimize scene
                    # disturbance: when independent geometry says the object
                    # is already supported, identity-clear and only a small
                    # planar offset away, carry out that retry as a measured
                    # push instead of re-grasping it.  This branch must precede
                    # the generic pickplace mapping; otherwise the later push
                    # selector is unreachable whenever reflection explicitly
                    # returned retry.
                    retry_action = "push"
                    retry_args = _normalise_push_args({}, episode, step.report or {})
                else:
                    retry_action = "pickplace"
                    retry_args = _retry_after_check(
                        episode,
                        step.report or {},
                        last_failure_mode,
                        reflected_retry,
                    )
                    # Obey the agent's decision to retry and its semantic
                    # grasp choice, but do not turn an explicitly unreliable
                    # centimetre estimate into an open-loop displacement.
                    if not last_check_reliable:
                        retry_args.pop("nudge", None)
                pending_action = {
                    "thought": (
                        "visual state authority identified a failed transport; "
                        + (
                            "apply the measured non-destructive correction"
                            if retry_action == "push"
                            else "retry with a materially different proposal"
                        )
                    ),
                    "action": retry_action,
                    "args": retry_args,
                }
                # Carry the authoritative decision through the next-turn
                # safety normaliser.  Without this flag that normaliser sees
                # only the auxiliary ``reliable=False`` bit and silently
                # rewrites the queued retry back into another passive look.
                allow_unreliable_retry = True
                reflected_retry = None
            elif not last_check_reliable:
                moved_points = int((step.report or {}).get("moved_points") or 0)
                footprint = (step.report or {}).get("footprint_containment") or {}
                is_sparse_container_miss = (
                    (step.report or {}).get("destination_kind") == "container"
                    and 0 < moved_points < 500
                    and pickplace_calls < episode.max_pickplace_calls
                    and turn + 1 < episode.turns
                )
                is_dense_confirmed_container_miss = (
                    (step.report or {}).get("destination_kind") == "container"
                    and moved_points >= 500
                    and bool((step.report or {}).get("identity_clear", False))
                    and float((step.report or {}).get("destination_region_confidence", 0.0) or 0.0)
                    >= 0.70
                    and bool(footprint.get("usable"))
                    and not bool(footprint.get("contained"))
                    and pickplace_calls < episode.max_pickplace_calls
                    and turn + 1 < episode.turns
                )
                if visibly_recoverable_miss and (step.report or {}).get("destination_kind") in {
                    "surface",
                    "top_region",
                }:
                    # Identity and a dense footprint prove that the released
                    # object is wholly outside the target and at the wrong
                    # support height.  The stale release-coordinate mismatch
                    # must not turn this clear miss into a terminal ambiguity.
                    # Re-ground the simplified scene and retain only VLM
                    # controls; an unreliable geometric nudge is discarded.
                    allow_unreliable_retry = True
                    retry_args = _merge_retry(
                        _default_pickplace_args(episode),
                        reflected_retry,
                        episode,
                    )
                    retry_args.pop("nudge", None)
                    pending_action = {
                        "thought": "identified object is wholly outside and below the target; re-ground and retry",
                        "action": "pickplace",
                        "args": retry_args,
                    }
                    reflected_retry = None
                elif is_dense_confirmed_container_miss:
                    # The global reliability flag can fail because the old
                    # release coordinate and the newly segmented object do not
                    # agree.  A dense, identity-consistent footprint visibly
                    # outside a confident container is nevertheless direct
                    # evidence of a miss.  Re-ground on the simplified scene;
                    # a passive look cannot change or sharpen this geometry.
                    allow_unreliable_retry = True
                    pending_action = {
                        "thought": "dense identified footprint is outside; retry with containment controls",
                        "action": "pickplace",
                        "args": _retry_after_check(
                            episode,
                            step.report or {},
                            last_failure_mode,
                            reflected_retry,
                        ),
                    }
                    reflected_retry = None
                elif is_sparse_container_miss:
                    # A tiny moved mask far from a broad container means the
                    # first drop is visibly still outside (butter/basket is the
                    # clean-set example). One lower retry is safer than four
                    # no-op looks and is still followed by a check.
                    allow_unreliable_retry = True
                    pending_action = {
                        "thought": "object is still outside; re-ground on the simplified scene",
                        "action": "pickplace",
                        # The measurement is explicitly unreliable, so none
                        # of its nudge/height suggestions are admissible.  A
                        # clean re-ground on the simplified scene is the only
                        # evidence-backed change.
                        "args": _default_pickplace_args(episode),
                    }
                    reflected_retry = None
                else:
                    pending_action = {
                        "thought": "measurement is inconsistent; inspect without moving",
                        "action": "look",
                        "args": {},
                    }
            elif (
                _check_supports_push(step.report or {})
                and not push_blocked_until_pickplace
                and push_calls < episode.max_push_calls
                and turn + 1 < episode.turns
            ):
                pending_action = {
                    "thought": "the object is supported and only slightly off; nudge without re-grasping",
                    "action": "push",
                    "args": _normalise_push_args({}, episode, step.report or {}),
                }
                reflected_retry = None
            elif (
                last_failure_mode != "not_grounded"
                and pickplace_calls < episode.max_pickplace_calls
                and turn + 1 < episode.turns
            ):
                pending_action = {
                    "thought": "retry from measured and reflected evidence",
                    "action": "pickplace",
                    "args": _retry_after_check(
                        episode,
                        step.report or {},
                        last_failure_mode,
                        reflected_retry,
                    ),
                }
                reflected_retry = None
            elif pickplace_calls >= episode.max_pickplace_calls:
                history.append(step.for_history())
                outcome.stopped = (
                    "manipulation budget exhausted after a measured miss; "
                    "stopped instead of repeating checks"
                )
                break

        if action == "look" and (last_failure_mode in {"not_grounded", "no_opening_geometry"}):
            source_aliases = list((step.report or {}).get("verified_source_aliases") or [])
            if last_failure_mode == "not_grounded" and last_grounding_failure == "pick":
                qualified = [
                    _qualified_visual_alias(alias, episode.pick) for alias in source_aliases
                ]
                verified_pick_aliases.update(_normalised_label(alias) for alias in qualified)
                if qualified and not active_pick_alias:
                    active_pick_alias = qualified[0]
                    pending_action = {
                        "thought": (
                            "retry once with independently verified source "
                            "appearance while preserving its spatial selector"
                        ),
                        "action": (
                            "insert"
                            if (
                                last_transport_action == "insert"
                                or _requires_insert(episode.place, episode.instruction)
                            )
                            else "pickplace"
                        ),
                        "args": {
                            "pick": active_pick_alias,
                            "place": active_place_alias or episode.place,
                        },
                    }
                else:
                    history.append(step.for_history())
                    outcome.stopped = (
                        "source was not grounded and independent inventory "
                        "verification found no new admissible referent"
                    )
                    break
                history.append(step.for_history())
                before = step.image
                continue
            aliases = list((step.report or {}).get("verified_destination_aliases") or [])
            verified_place_aliases.update(_normalised_label(alias) for alias in aliases)
            destination_recovery = bool(
                last_grounding_failure == "destination"
                or last_failure_mode == "no_opening_geometry"
            )
            if (
                destination_recovery
                and aliases
                and pickplace_calls < episode.max_pickplace_calls
                and turn + 1 < episode.turns
            ):
                first_verified_alias = not active_place_alias
                if first_verified_alias:
                    active_place_alias = aliases[0]
                if last_failure_mode == "no_opening_geometry" and not first_verified_alias:
                    # The original task phrase and an independently verified
                    # visual alias both produced no bounded opening polygon.
                    # That is positive geometry evidence for an exposed
                    # support, not a reason to repeat passive looks forever.
                    # Permit one ordinary placement; a true shelf/slot keeps
                    # its opening and never enters this fallback.
                    allow_surface_fallback = True
                    pending_action = {
                        "thought": (
                            "two semantic views found a support but no opening; "
                            "use one measured surface placement"
                        ),
                        "action": "pickplace",
                        "args": {
                            "pick": episode.pick,
                            "place": active_place_alias,
                            "destination": "region",
                        },
                    }
                else:
                    pending_action = {
                        "thought": "retry with the closed-set visually verified destination alias",
                        # Preserve insert for the first alias-aware geometry
                        # preflight. Surface fallback is admitted only if this
                        # second semantic view also finds no opening.
                        "action": (
                            "insert"
                            if (
                                last_transport_action == "insert"
                                or _requires_insert(episode.place, episode.instruction)
                            )
                            else "pickplace"
                        ),
                        "args": {"pick": episode.pick, "place": active_place_alias},
                    }
            elif destination_recovery and not aliases:
                # No action changed the scene between the failed grounding and
                # this exhaustive inventory check. Repeating the same lookup
                # cannot expose a new destination and only burns turns/API.
                history.append(step.for_history())
                outcome.stopped = (
                    "destination was not grounded and no inventory crop visually matched it"
                )
                break

        # A passive look does not change geometry.  When it was requested to
        # inspect an unreliable post-release check, immediately measure once
        # more instead of letting the VLM spend the remaining episode asking
        # for identical looks.  The second check is already covered by the
        # unreliable_check_streak stop above.  Preserve a pending alias retry:
        # that is the one case where the look supplied genuinely new action
        # information.
        if action == "look" and has_released and not last_check_reliable and pending_action is None:
            pending_action = {
                "thought": "re-measure once after the passive inspection",
                "action": "check",
                "args": {"pick": episode.pick, "place": episode.place},
            }

        history.append(step.for_history())

        # Retain the image pair for the mandatory check.  If no check will run
        # (for example an empty grasp), diagnose immediately so the next
        # physical attempt can still change its grasp family.
        if (
            action in {"pickplace", "push", "insert"}
            and not bool((step.report or {}).get("preflight_only"))
            and turn + 1 < episode.turns
        ):
            last_motion_context = (before, step)
            visual_done_after_release = False
            if not must_check:
                diagnosis = reflect_on_motion(last_motion_context)
                last_motion_context = None
                visual_next = str(diagnosis.get("next", "check")).strip().lower()
                if (
                    visual_next == "done"
                    and has_released
                    and bool(last_check_report)
                    and last_failure_mode in {"empty_grasp", "wrong_object_grasp", "not_grounded"}
                ):
                    # A previous manipulation released the object and was
                    # measured.  If a later no-motion re-grasp leaves the scene
                    # unchanged, the visual Agent may recognize that the object
                    # was already at the destination.  Honor that state verdict
                    # instead of letting the action-specific retry ladder below
                    # overwrite it (the t083 false-negative pattern).
                    last_check_ok = True
                    pending_action = {
                        "thought": (
                            "visual state authority confirmed that the earlier "
                            "release already satisfied this transport"
                        ),
                        "action": "done",
                        "args": {"reason": diagnosis.get("reason", "visual confirmation")},
                    }
                    reflected_retry = None
                    visual_done_after_release = True
                    history.append(
                        "  no-motion re-grasp left the earlier released object "
                        "unchanged at the requested destination"
                    )
                elif visual_next == "retry":
                    candidate = diagnosis.get("retry_args")
                    reflected_retry = candidate if isinstance(candidate, dict) else None
                elif visual_next == "stop":
                    outcome.stopped = (
                        "visual state authority stopped after a no-motion "
                        "transport attempt"
                    )
                    before = step.image
                    break
            if visual_done_after_release:
                before = step.image
                continue
            selected_retry_action = str(
                (reflected_retry or {}).get("_transport_action") or ""
            ).strip()
            if reflected_retry:
                reflected_retry.pop("_transport_action", None)
            if selected_retry_action == "look":
                pending_action = {
                    "thought": "the separate transport selector requested a fresh visual inventory",
                    "action": "look",
                    "args": {
                        "verify_destination": episode.place,
                        "exclude": episode.pick,
                    },
                }
                reflected_retry = None
                before = step.image
                continue
            if action == "push":
                # The before/after visual diagnosis is retained in the human
                # trace, but geometry decides whether another correction is
                # safe.  A push can never chain directly into another motion.
                pending_action = {
                    "thought": "measure the result of the visual micro-correction",
                    "action": "check",
                    "args": {"pick": episode.pick, "place": episode.place},
                }
                reflected_retry = None
            elif action == "insert":
                if last_failure_mode in {"empty_grasp", "wrong_object_grasp"} and not bool(
                    (step.report or {}).get("release_xy")
                ):
                    # No placement occurred, so a geometric destination check
                    # cannot add information.  Preserve the visual agent's
                    # requested grasp family inside the insert primitive.  The
                    # old branch discarded it and interleaved useless checks,
                    # causing four repetitions of the analytic ladder.
                    retry_args = _normalise_insert_args(reflected_retry or {}, episode)
                    agent_selected_grasp = "grasp" in retry_args
                    if (
                        not agent_selected_grasp
                        and consecutive_empty_grasps >= 2
                        and not learned_grasp_attempted
                    ):
                        retry_args["grasp"] = ["graspnet"]
                    if (
                        learned_grasp_attempted
                        and consecutive_empty_grasps >= 2
                        and not agent_selected_grasp
                    ):
                        outcome.stopped = (
                            "analytic insertion grasps and one learned 6-DoF "
                            "grasp remained empty; stopped without repetition"
                        )
                        break
                    pending_action = {
                        "thought": (
                            "the insertion grasp was empty; obey the visual "
                            "agent's materially different grasp proposal"
                        ),
                        "action": "insert",
                        "args": retry_args,
                    }
                else:
                    # A moved insert follows the same measure-before-motion
                    # rule as pickplace. Its configuration-space plan is
                    # deterministic; visual prose cannot safely alter its
                    # centre or yaw.
                    pending_action = {
                        "thought": "measure the guided insertion before any recovery",
                        "action": "check",
                        "args": {"pick": episode.pick, "place": episode.place},
                    }
                reflected_retry = None
            elif (
                last_failure_mode in {"empty_grasp", "wrong_object_grasp", "not_grounded"}
                and has_released
            ):
                pending_action = {
                    "thought": "the object was released earlier; measure before any re-grasp",
                    "action": "check",
                    "args": {"pick": episode.pick, "place": episode.place},
                }
                reflected_retry = None
            elif last_failure_mode == "not_grounded" and last_grounding_failure == "destination":
                equivalent = _normalised_label(episode.place)
                reflected_place = str((reflected_retry or {}).get("place", "")).strip()
                if reflected_place and _normalised_label(reflected_place) == equivalent:
                    equivalent = reflected_place
                if not semantic_place_variant_tried and equivalent != episode.place:
                    semantic_place_variant_tried = True
                    pending_action = {
                        "thought": "retry the exact destination semantics without a leading article",
                        "action": "pickplace",
                        "args": {"pick": episode.pick, "place": equivalent},
                    }
                else:
                    pending_action = {
                        "thought": "destination was not grounded; inspect the visual inventory before renaming it",
                        "action": "look",
                        "args": {
                            "verify_destination": episode.place,
                            "exclude": episode.pick,
                        },
                    }
                reflected_retry = None
            elif (
                last_failure_mode in {"empty_grasp", "wrong_object_grasp", "not_grounded"}
                and pickplace_calls < episode.max_pickplace_calls
            ):
                if (
                    last_failure_mode in {"empty_grasp", "wrong_object_grasp"}
                    and consecutive_empty_grasps >= 2
                ):
                    agent_retry = _retry_after_check(
                        episode,
                        step.report or {},
                        last_failure_mode,
                        reflected_retry,
                    )
                    agent_selected_grasp = _valid_grasp(
                        (reflected_retry or {}).get("grasp")
                    ) is not None
                    if agent_selected_grasp:
                        pending_action = {
                            "thought": (
                                "empty grasps persisted; obey the visual "
                                "agent's materially different proposal"
                            ),
                            "action": "pickplace",
                            "args": agent_retry,
                        }
                    elif learned_grasp_attempted:
                        outcome.stopped = (
                            "analytic ladders and one learned 6-DoF grasp "
                            "remained empty; stopped without blind repetition"
                        )
                        break
                    # Two different analytic ladders have provided contact
                    # evidence and local target tracking. Escalate exactly
                    # once to a learned 6-DoF proposal instead of either
                    # exhausting eight equivalent retries or stopping before
                    # the configured learned backend can be used.
                    else:
                        pending_action = {
                            "thought": (
                                "no valid recovery was proposed; try one learned "
                                "6-DoF proposal on the locally tracked object"
                            ),
                            "action": "pickplace",
                            "args": {
                                "pick": episode.pick,
                                "place": episode.place,
                                "grasp": ["graspnet"],
                            },
                        }
                else:
                    pending_action = {
                        "thought": "the manipulation did not occur; retry with corrected grounding or grasp",
                        "action": "pickplace",
                        "args": _retry_after_check(
                            episode,
                            step.report or {},
                            last_failure_mode,
                            reflected_retry,
                        ),
                    }
                reflected_retry = None
            else:
                pending_action = {
                    "thought": "measure before deciding",
                    "action": "check",
                    "args": {"pick": episode.pick, "place": episode.place},
                }

        before = step.image

    return outcome
