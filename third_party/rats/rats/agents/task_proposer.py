"""Task Proposer: selects next task via curiosity-driven in-context reasoning.

Reads the skill library, scene context, and task history to propose tasks
that would discover new, useful skills.

Supports two modes:
- **catalog**: select from predefined tasks (BEHAVIOR / LIBERO catalog).
- **novel**: propose a novel task spec (language + objects + goal) that the
  Environment Creator will turn into a BDDL file.  Used for LIBERO.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import logging

from rats.agents.base_agent import query_llm_json

logger = logging.getLogger("rats.task_proposer")


class TaskProposer:
    def __init__(
        self,
        available_tasks: list[dict[str, Any]] | None = None,
        mode: str = "catalog",
        random_order: bool = False,
        curriculum: bool = False,
        no_context: bool = False,
        proposer_temperature: float | None = None,
        include_eval_task_context: bool = False,
        molmospaces_proposer_mode: str = "catalog",
        molmospaces_allow_house_switching: bool = True,
        molmospaces_house_switch_max_per_run: int = 5,
        molmospaces_vlm_grounding: bool = True,
        molmospaces_geometric_visibility_gate: bool = False,
        molmospaces_allowed_task_types: list[str] | None = None,
        molmospaces_forced_task_type_sequence: list[str] | None = None,
        molmospaces_curiosity: bool = False,
        molmospaces_playtime_config: dict[str, Any] | None = None,
    ) -> None:
        self._task_history: list[dict[str, Any]] = []
        self._available_tasks = available_tasks or []
        self.mode = mode  # "catalog" or "novel"
        self._random_order = random_order
        self._curriculum = curriculum
        # When True, the novel-task prompt is stripped of all run-dependent
        # context (skill library, task history, curriculum hint). The catalog
        # / env limitations / pick reliability blocks (static facts about the
        # environment) are kept so proposals remain valid. Use for the
        # "formula_no_context" ablation: same scoring + selection pipeline,
        # but the proposer can't condition on what's been learned or tried.
        self._no_context = bool(no_context)
        # Per-proposer LLM sampling temperature. None defers to the
        # global llm_temperature (rats/config/default.yaml, currently 0.2 — near
        # deterministic). The no-context ablation revealed that a static
        # prompt + low temperature collapses the proposer into proposing the
        # same task every iter, so callers running no-context-style
        # experiments should bump this to 0.7-0.9 to recover diversity.
        # Only the candidate-pool path (propose_novel_candidates) reads this;
        # other proposer entry points fall through to the global default.
        self._proposer_temperature = proposer_temperature
        # When True, prepend an "EVALUATION TASKS" block listing the 60
        # LIBERO-PRO eval-suite task descriptions (deduped to 30 unique
        # strings × 3 families) so the proposer can target its play at the
        # kinds of downstream tasks the agent will be evaluated on. Default
        # False so existing baselines (curiosity, formula, random) stay
        # blind to the eval set — toggle on for "task-aware play" variants.
        self._include_eval_task_context = bool(include_eval_task_context)
        # Lazy-cache the rendered block; first-call resolution import LIBERO.
        self._eval_task_context_block: str | None = None
        # MolmoSpaces-specific knobs (read by `_propose_novel` dispatch).
        # "catalog" picks from the legacy benchmark catalog; "open" runs
        # the inventory-driven novel proposer that talks to the bridge's
        # set_task_from_spec RPC. See `rats/config/default.yaml::molmospaces:`.
        self._molmospaces_proposer_mode = molmospaces_proposer_mode
        self._molmospaces_curiosity = bool(molmospaces_curiosity)
        self._molmospaces_playtime_config = dict(molmospaces_playtime_config or {})
        self._include_eval_task_context = bool(include_eval_task_context)
        self._eval_task_context_block: str | None = None
        self._molmospaces_allow_house_switching = bool(molmospaces_allow_house_switching)
        self._molmospaces_house_switch_max_per_run = int(molmospaces_house_switch_max_per_run)
        self._molmospaces_house_switches_used = 0
        self._molmospaces_vlm_grounding = bool(molmospaces_vlm_grounding)
        self._molmospaces_geometric_visibility_gate = bool(
            molmospaces_geometric_visibility_gate
        )
        self._molmospaces_allowed_task_types = [
            str(t).strip().lower()
            for t in (molmospaces_allowed_task_types or ["pick", "pick_and_place", "open", "close"])
            if str(t).strip()
        ]
        self._molmospaces_forced_task_type_sequence = [
            str(t).strip().lower()
            for t in (molmospaces_forced_task_type_sequence or [])
            if str(t).strip()
        ]
        # Per-house grounding cache, keyed by (house_index, scene_dataset).
        # Cleared whenever the proposer requests a new house so the next
        # house gets a fresh VLM call.
        self._molmospaces_grounding_cache: dict[tuple[Any, Any], dict[str, Any]] = {}
        # Best-effort structured trace for the latest proposal. The lifelong
        # loop persists this under the run output directory so proposal-time
        # prompts, LLM JSON, validation decisions, and resolved bridge specs
        # are inspectable after long MolmoSpaces runs.
        self.last_proposal_trace: dict[str, Any] | None = None

    def _curriculum_stage(self) -> int:
        """Stage 1 (<3 successes): single-step pick+place only.
        Stage 2 (3–9 successes): novel single-step variants.
        Stage 3 (≥10): compound tasks unlocked.
        Gated on actually-completed successes, not attempts — the proposer
        only gets harder after the robot demonstrably solved easier ones.
        """
        if not self._curriculum:
            return 3
        successes = sum(1 for h in self._task_history if h.get("success"))
        if successes < 3:
            return 1
        if successes < 10:
            return 2
        return 3

    def _curriculum_hint(self) -> str:
        stage = self._curriculum_stage()
        if stage == 1:
            return (
                "CURRICULUM STAGE 1 (hard constraint, 0-2 successes so far):\n"
                "- Propose ONLY a single-step pick-and-place task: exactly ONE object "
                "moved, ONE source → ONE destination.\n"
                "- Forbidden: doors (open/close), drawers, microwaves, cabinets, "
                "stove knobs, wine-rack slots, multi-step 'and then', 'and close', "
                "'and turn on'.\n"
                "- Allowed predicates: On, In (for open containers like baskets/plates/trays).\n"
                "- Pick object+container pairs the seed library's grasp+place "
                "primitives can realistically handle.\n"
            )
        if stage == 2:
            return (
                "CURRICULUM STAGE 2 (3-9 successes so far):\n"
                "- Propose single-step pick-and-place, but you may combine novel "
                "object/container pairs not yet attempted.\n"
                "- Still forbidden: compound multi-step tasks (open + put + close), "
                "knob rotations, slot alignment.\n"
                "- Allowed predicates: On, In.\n"
            )
        return ""  # Stage 3: no extra constraint

    # Articulated / actuation fixtures + verbs that fail the seed library
    # without learned door/knob/drawer skills. Used by the stage 1/2
    # post-hoc validator to reject over-scoped proposals from the LLM.
    _STAGE12_FORBIDDEN_FIXTURES = (
        "microwave", "cabinet", "drawer", "wine_rack",
        "wine rack", "stove",
    )
    _STAGE12_FORBIDDEN_PREDICATES = (
        "open", "close", "turnon", "turn_on", "stack", "open_door",
    )
    _STAGE12_FORBIDDEN_PHRASES = (
        " and then ", " and close", " and open", " and turn",
        " and put", "open the ", "close the ", "turn on", "turn off",
    )

    # Goal predicates that are known to crash LIBERO-PRO's predicate
    # evaluator when their second argument is a bare movable object instance.
    # In/Stack call check_contain() on arg2; for ObjectState this reaches
    # base_object_states.py:68 and unconditionally calls object.in_box(...).
    # Regular movable objects such as WhiteBowl, Plate, WoodenTray, Basket,
    # and Ramekin do not implement in_box. Site/region targets such as
    # basket_1_contain_region or wooden_cabinet_1_top_region are valid.
    _IN_BOX_PREDICATES = {"in", "stack"}

    @classmethod
    def _env_limitations_block(cls) -> str:
        """Render `_BROKEN_PREDICATE_CONTAINERS` as prompt text.

        The proposer reads this and is asked to inspect its candidate
        against it BEFORE outputting (see `prompts/task_proposer_novel.txt`
        rule 8). The post-hoc `_validate_stage12` still vets the result
        but is meant as a safety net; the goal is for the proposer to
        avoid the bad combos by reasoning.
        """
        return (
            "- `In/Stack(*, movable_object_N)` is unsupported in this "
            "LIBERO-PRO build. The built-in reward/task-completion evaluator "
            "expects the second argument to be a site/region with `in_box()`, "
            "not a bare object like `white_bowl_1`, `wooden_tray_1`, "
            "`plate_1`, or `basket_1`. Use `On` for open bowl/tray/plate "
            "destinations, or a real site/region such as "
            "`basket_1_contain_region` or `wooden_cabinet_1_top_region`."
        )

    @classmethod
    def _validate_libero_predicate_runtime_compatibility(
        cls, proposal: dict[str, Any],
    ) -> str:
        """Reject goal predicates that will crash LIBERO's symbolic evaluator."""
        try:
            from rats.agents.libero_catalog import OBJECTS
        except Exception:
            OBJECTS = {}

        def _bare_instance_base(name: str) -> str | None:
            lowered = name.lower()
            parts = lowered.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
            return None

        for pred in proposal.get("goal", []) or []:
            if not isinstance(pred, list) or len(pred) < 3:
                continue
            head = str(pred[0]).lower()
            if head not in cls._IN_BOX_PREDICATES:
                continue
            target = str(pred[2])
            target_base = _bare_instance_base(target)
            if target_base and target_base in OBJECTS:
                return (
                    f"predicate '{pred[0]}({pred[1]}, {pred[2]})' is "
                    "unsupported in this LIBERO-PRO build: In/Stack targets "
                    "must be site/region names with in_box(), not bare movable "
                    f"object instance '{pred[2]}'. Use On for bowl/tray/plate "
                    "destinations or a real region such as "
                    "basket_1_contain_region / wooden_cabinet_1_top_region."
                )
        return ""

    def _validate_stage12(self, proposal: dict[str, Any]) -> str:
        """Return empty string if the proposal fits stage 1/2 constraints,
        else a one-line rejection reason that can be fed back into the LLM
        for a re-roll. Hard rules — the prompt hint alone was being ignored
        (4/4 of the iter0 explore-mode tasks bypassed it).
        """
        compatibility_error = self._validate_libero_predicate_runtime_compatibility(proposal)
        if compatibility_error:
            return compatibility_error

        stage = self._curriculum_stage()
        if stage >= 3:
            return ""

        goal = proposal.get("goal", []) or []
        if not isinstance(goal, list):
            return f"goal must be a list of predicate-lists, got {type(goal).__name__}"
        if len(goal) > 1:
            return (
                f"goal has {len(goal)} predicates; stage {stage} allows exactly "
                "one (single source → single destination)"
            )
        for pred in goal:
            if isinstance(pred, list) and pred:
                head = str(pred[0]).lower()
                if head in self._STAGE12_FORBIDDEN_PREDICATES:
                    return (
                        f"predicate '{pred[0]}' is forbidden in stage {stage} "
                        f"(allowed: On, In on open containers)"
                    )

        fixtures = [str(f).lower() for f in (proposal.get("fixtures") or [])]
        for fx in fixtures:
            for bad in self._STAGE12_FORBIDDEN_FIXTURES:
                if bad in fx:
                    return (
                        f"fixture '{fx}' is articulated/actuated; "
                        f"stage {stage} requires open containers (basket, plate, "
                        "tray, bowl, table surfaces) only"
                    )

        language = str(proposal.get("language") or "").lower()
        # Pad both ends with a space so phrase patterns like " and then "
        # match at sentence boundaries too.
        padded = f" {language} "
        for phrase in self._STAGE12_FORBIDDEN_PHRASES:
            if phrase in padded:
                return (
                    f"language contains forbidden phrase '{phrase.strip()}'; "
                    f"stage {stage} disallows compound / actuation steps"
                )
        return ""

    @staticmethod
    def _validate_pick_reliability(proposal: dict[str, Any]) -> str:
        """Reject proposals whose pick target is in UNRELIABLE_PICK_OBJECTS.

        The benchmark (docs/pick-primitive-benchmark.md) showed the
        non-priv pick pipeline is 0% SR on butter / chocolate_pudding /
        cream_cheese / wine_bottle. Proposing them wastes a 5-min
        iteration and teaches the LLM nothing about a fixable
        bottleneck — the primitive itself needs to change first.

        Pick target = the first argument of an `On`/`In` goal predicate
        (the object being moved), or anything in `objects` if no clear
        goal. Containers (2nd arg) are fine regardless.
        """
        from rats.agents.libero_catalog import UNRELIABLE_PICK_OBJECTS

        bad = set(UNRELIABLE_PICK_OBJECTS)

        # Derive pick targets from goal's 2-arity predicates first (On, In),
        # where arg1 is the moved object. Strip any trailing `_N` instance
        # index so `butter_1` collapses to `butter`.
        pick_targets: set[str] = set()
        for pred in proposal.get("goal", []) or []:
            if isinstance(pred, list) and len(pred) >= 2:
                head = str(pred[0]).lower()
                if head in ("on", "in"):
                    arg1 = str(pred[1]).lower()
                    # drop instance suffix: butter_1 -> butter
                    base = arg1.rsplit("_", 1)[0] if arg1[-1:].isdigit() else arg1
                    pick_targets.add(base)
                    pick_targets.add(arg1)

        for target in pick_targets:
            # Match exact object name OR a "<obj>_N"-style instance.
            for unreliable in bad:
                if target == unreliable or target.startswith(unreliable + "_"):
                    return (
                        f"pick target '{target}' is on the "
                        f"UNRELIABLE_PICK_OBJECTS list (benchmark-verified "
                        f"0% SR for the non-privileged pick primitive). "
                        f"Pick a different target object from the reliable "
                        f"list."
                    )

        # Fallback: if goal didn't resolve cleanly, scan the declared
        # objects for any unreliable one. Only reject if the unreliable
        # item is the SOLE manipulable object (else it might just be
        # scene clutter).
        objects = [str(o).lower() for o in (proposal.get("objects") or [])]
        if objects:
            bases = {o.rsplit("_", 1)[0] if o[-1:].isdigit() else o for o in objects}
            if len(bases) == 1 and bases & bad:
                return (
                    f"only object in proposal is on the "
                    f"UNRELIABLE_PICK_OBJECTS list; primitive cannot pick "
                    f"it (benchmark-verified). Pick a different target."
                )
        return ""

    @staticmethod
    def _validate_libero_catalog_membership(proposal: dict[str, Any]) -> str:
        """Reject proposals that reference objects/fixtures missing from
        the LIBERO-PRO asset catalog.

        LLMs frequently propose `corn` / `cherries` / `mayo` / `rack` /
        `red_box` / `red_sticker` / `blue_red_sticker` / `libero_mug_green`
        from prior LIBERO-suite knowledge, but the underlying mesh/XML files
        are not shipped in this LIBERO-PRO build. The Environment Creator
        writes the BDDL successfully and then crashes on instantiate, the
        loop falls back to the previous (already-novel) env, and the
        report tags the iteration with the LiberoHandle's hardcoded
        `novel_task0` activity name. Catching the bad name here lets the
        proposer re-roll instead of burning the iteration.
        """
        from rats.agents.libero_catalog import FIXTURES, OBJECTS

        valid_objects = {k.lower() for k in OBJECTS}
        valid_fixtures = {k.lower() for k in FIXTURES}

        def _base(name: str) -> str:
            """Strip the `_<digit>` instance suffix from a catalog name.

            Handles three shapes:
              `wooden_cabinet`             -> `wooden_cabinet`
              `wooden_cabinet_1`           -> `wooden_cabinet`
              `wooden_cabinet_1_top_region`-> `wooden_cabinet`  (fixture sub-region)

            The previous version only handled the second shape (last char must
            be a digit), so every articulated-fixture proposal that used a
            sub-region argument got rejected on the first roll — even though
            the prompt explicitly documents sub-region names like
            `wooden_cabinet_1_top_region` as valid In() arguments.
            """
            n = name.lower()
            # Find `_<digit>+` and treat everything before it as the base.
            # If no digit run is found, return the original name.
            import re as _re_base
            m = _re_base.match(r"^(.+?)_\d+(?:_.*)?$", n)
            if m:
                return m.group(1)
            return n

        for obj in proposal.get("objects") or []:
            base = _base(str(obj))
            if base not in valid_objects:
                return (
                    f"object '{obj}' is not in the LIBERO-PRO asset catalog "
                    f"(OBJECTS in agents/libero_catalog.py). Its mesh/XML is "
                    f"not shipped in this build, so Environment Creator will "
                    f"crash on instantiate. Pick a catalog-listed object."
                )
        for fx in proposal.get("fixtures") or []:
            base = _base(str(fx))
            if base not in valid_fixtures:
                return (
                    f"fixture '{fx}' is not in the LIBERO-PRO asset catalog "
                    f"(FIXTURES in agents/libero_catalog.py). Pick a "
                    f"catalog-listed fixture."
                )
        for pred in proposal.get("goal") or []:
            if not isinstance(pred, list):
                continue
            for arg in pred[1:]:
                base = _base(str(arg))
                if base in valid_objects or base in valid_fixtures:
                    continue
                # Allow workspace surface names referenced in BDDL goals.
                if base in {"main_table", "kitchen_table",
                            "living_room_table", "study_table"}:
                    continue
                return (
                    f"goal predicate references '{arg}' whose base name "
                    f"'{base}' is not in the OBJECTS/FIXTURES catalog. "
                    f"Use catalog-listed names with `_N` instance suffixes."
                )
        return ""

    @staticmethod
    def _draft_proposal(result: dict[str, Any]) -> dict[str, Any]:
        """Lightweight view of an LLM result for the curriculum validator.
        Mirrors the fields _validate_stage12 reads (language, fixtures,
        goal) without running the full proposal-assembly path. Keeps the
        validator side-effect-free so we can call it before the heavier
        affordance_hints / curiosity-scoring work.
        """
        return {
            "language": result.get("language", ""),
            "fixtures": result.get("fixtures", []) or [],
            "goal": result.get("goal", []) or [],
        }

    def set_available_tasks(self, tasks: list[dict[str, Any]]) -> None:
        self._available_tasks = tasks

    def set_mode(self, mode: str) -> None:
        """Switch proposer mode at runtime. Used by lifelong_loop to flip
        from "play" to "novel"/"catalog" when the play phase ends."""
        self.mode = mode

    @staticmethod
    def _is_molmospaces_scene(scene_context: dict[str, Any]) -> bool:
        """Single branch predicate for keeping MolmoSpaces behavior separate."""
        return scene_context.get("env_type") == "molmospaces"

    def propose(
        self,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Propose the next task using curiosity-driven reasoning."""
        self.last_proposal_trace = None
        if self._random_order:
            return self._propose_random(scene_context)

        # Keep MolmoSpaces behavior isolated from LIBERO's play mode.
        # MolmoSpaces play is selected by `molmospaces_proposer_mode`
        # inside `_propose_novel`, where proposals are validated against the
        # live scene inventory instead of the LIBERO asset catalog.
        # ``play_prompt`` is kept as a back-compat alias for ``play``;
        # both load the unified prompts/task_proposer_play.txt template.
        if self._is_molmospaces_scene(scene_context):
            if self.mode in {"novel", "play", "play_prompt"}:
                return self._propose_novel(scene_context, skill_context)
            return self._propose_catalog(scene_context, skill_context)

        if self.mode in {"play", "play_prompt"}:
            # LIBERO play mode: identical novel-task pipeline, child-flavoured
            # proposer prompt. The standalone ``_propose_play`` helper is
            # retired — both legacy mode names route through ``_propose_novel``
            # with the unified play prompt.
            return self._propose_novel(
                scene_context,
                skill_context,
                prompt_path=Path("rats/prompts/task_proposer_play.txt"),
            )
        if self.mode == "novel":
            return self._propose_novel(scene_context, skill_context)
        return self._propose_catalog(scene_context, skill_context)

    def _propose_random(self, scene_context: dict[str, Any]) -> dict[str, Any]:
        """Uniform random task selection (ablation: no curiosity)."""
        import random as rng
        if not self._available_tasks:
            return self._propose_catalog(scene_context, {})
        task = rng.choice(self._available_tasks)
        return {
            "mode": "catalog",
            "activity_name": task["activity_name"],
            "scene_model": task.get("scene_model", scene_context.get("scene_model", "unknown")),
            "activity_definition_id": task.get("activity_definition_id", 0),
            "goal_conditions": task.get("goal_conditions", task["activity_name"].replace("_", " ")),
            "reasoning": "Random selection (ablation)",
            "expected_new_skills": [],
            "novelty_score": 0.5,
            "difficulty_estimate": "medium",
        }

    # ------------------------------------------------------------------
    # Novel task proposal (generates spec for Environment Creator)
    # ------------------------------------------------------------------

    def _build_novel_prompt(
        self,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
        *,
        prompt_path: Path | None = None,
    ) -> tuple[str, str]:
        """Build the runtime novel-task prompt.

        `prompt_path` is LIBERO-only. MolmoSpaces exits through the explicit
        `env_type == "molmospaces"` branch below and uses its own prompt stack.
        """
        # Dispatch to MolmoSpaces-specific proposer when appropriate.
        if self._is_molmospaces_scene(scene_context):
            if self._molmospaces_proposer_mode == "open":
                return self._propose_novel_molmospaces_open(
                    scene_context, skill_context,
                )
            if self._molmospaces_proposer_mode == "playtime":
                return self._propose_novel_molmospaces_playtime(
                    scene_context, skill_context,
                )
            if self._molmospaces_proposer_mode in {
                "benchmark_order",
                "catalog_sequential",
                "sequential",
            }:
                return self._propose_catalog_molmospaces_sequential(scene_context)
            return self._propose_catalog_molmospaces(
                scene_context, skill_context,
            )
        from rats.agents.libero_catalog import (
            build_catalog_text,
            build_pick_reliability_text,
        )

        prompt_template = (
            prompt_path or Path("rats/prompts/task_proposer_novel.txt")
        ).read_text()
        # When --no-context is set, hide run-state context (skill library +
        # task history + curriculum hint) but keep static facts about the
        # environment (catalog, env limitations, pick reliability) so the
        # LLM can still produce valid proposals.
        if self._no_context:
            skill_block = "(hidden — no-context ablation)"
            history_str = "(hidden — no-context ablation)"
            curriculum_text = "(no curriculum hint — no-context ablation)"
        else:
            # Private ``_...`` entries carry runtime-only scorer objects such
            # as the callable skill lookup.  They are consumed after proposal
            # generation and must never be serialized into the LLM prompt.
            # MolmoSpaces already applied this filter; LIBERO did not, causing
            # every catalog self-play iteration to fail before its first call.
            prompt_skill_context = {
                key: value
                for key, value in skill_context.items()
                if not str(key).startswith("_")
            }
            skill_block = json.dumps(prompt_skill_context, indent=2)
            history_str = (
                "None yet." if not self._task_history
                else json.dumps(self._task_history[-10:], indent=2)
            )
            curriculum_text = self._curriculum_hint()
        catalog_text = build_catalog_text()
        user_prompt = prompt_template.replace(
            "{skill_context}", skill_block
        ).replace(
            "{task_history}", history_str
        ).replace(
            "{catalog}", catalog_text
        ).replace(
            "{curriculum_hint}", curriculum_text
        ).replace(
            "{env_limitations}", self._env_limitations_block()
        ).replace(
            "{pick_reliability}", build_pick_reliability_text()
        )
        # Optional task-aware-play block: append the 60 LIBERO-PRO eval-suite
        # task descriptions (deduped + grouped by family) so the proposer
        # can bias play toward skills that will be evaluated. Lazy-rendered
        # once per process; off by default (controlled by
        # ``include_eval_task_context`` ctor arg + CLI flag).
        if self._include_eval_task_context:
            if self._eval_task_context_block is None:
                try:
                    from rats.loop.libero_utils import format_libero_pro_eval_tasks_block
                    self._eval_task_context_block = (
                        format_libero_pro_eval_tasks_block() or ""
                    )
                except Exception as exc:
                    logger.warning(
                        "include_eval_task_context: failed to load LIBERO-PRO "
                        "eval task list (%s); proceeding without it", exc,
                    )
                    self._eval_task_context_block = ""
            if self._eval_task_context_block:
                user_prompt = (
                    self._eval_task_context_block
                    + "\n\n"
                    + user_prompt
                )
        system_prompt = (
            "You are a curiosity-driven task proposer for a robot learning system. "
            "Propose a NOVEL manipulation task. Respond only in valid JSON."
        )
        return user_prompt, system_prompt

    @staticmethod
    def _coerce_affordance_hints(raw_hints: Any) -> dict[str, str]:
        affordance_hints: dict[str, str] = {}
        if isinstance(raw_hints, dict):
            for k, v in raw_hints.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    continue
                v = v.strip()
                if v:
                    affordance_hints[k] = v[:200]
        return affordance_hints

    def _build_novel_proposal(self, result: dict[str, Any]) -> dict[str, Any]:
        language = result.get("language", "pick up an object and place it")
        return {
            "mode": "novel",
            "language": language,
            "scene_type": result.get("scene_type", "LIBERO_Kitchen_Tabletop_Manipulation"),
            "objects": result.get("objects", []),
            "fixtures": result.get("fixtures", []),
            "goal": result.get("goal", []),
            "affordance_hints": self._coerce_affordance_hints(
                result.get("affordance_hints", {}) or {}
            ),
            "activity_name": language.replace(" ", "_")[:60],
            "scene_model": result.get("scene_type", "libero"),
            "activity_definition_id": 0,
            "goal_conditions": language,
            "reasoning": result.get("reasoning", ""),
            "expected_new_skills": result.get("expected_new_skills", []),
            "novelty_score": result.get("novelty_score", 0.5),
            "difficulty_estimate": result.get("difficulty_estimate", "medium"),
        }

    def propose_novel_candidates(
        self,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
        *,
        num_candidates: int = 3,
        request_required_skills: bool = False,
    ) -> list[dict[str, Any]]:
        """Propose K fresh candidate tasks.

        ``request_required_skills`` adds a ``required_skills`` list per
        candidate — short skill-name hints that the formula scorer reads
        for object-skill novelty and Wilson-LB-based frontier.

        Each returned dict is the standard novel-proposal schema PLUS:
          - ``candidate_type``: ``"fresh"``
          - ``required_skills``: ``list[str]`` (empty when not requested)
        """
        if scene_context.get("env_type") == "molmospaces":
            return [
                self._propose_novel_molmospaces(
                    scene_context, skill_context,
                )
            ]

        user_prompt, system_prompt = self._build_novel_prompt(
            scene_context, skill_context,
        )
        user_prompt += (
            "\n\nADDITIONAL RUNTIME INSTRUCTION:\n"
            f"- Propose EXACTLY {num_candidates} DIFFERENT candidate tasks instead of one.\n"
            "- Return valid JSON with keys: `reasoning` and `candidates`.\n"
            "- `candidates` must be a list of task objects using the SAME schema as the original single-task response.\n"
            "- Make the candidates meaningfully different in object choice, fixture choice, or goal interaction.\n"
            "- ORDER the candidates from SIMPLEST to MOST COMPLEX:\n"
            "    * Candidate 1 MUST be a single-action atomic task (one verb, ≤2 objects, no compound goals).\n"
            "      Examples: 'put the X on the Y', 'open the Z', 'close the W'.\n"
            "    * Subsequent candidates may layer additional steps (e.g. compound 'open AND put inside',\n"
            "      perturbations 'put X to the left of Y', or stove/microwave-state changes).\n"
            "    * The complexity ordering matters: when scoring is uninformative (cold-start, no skill\n"
            "      reliability data yet) the selector falls back to the candidate ordering, so the FIRST\n"
            "      candidate is what gets tried in early iterations.\n"
            "- Each candidate's `language` field MUST start with 'I want to ' to match the schema the\n"
            "  downstream planner / BDDL generator expects (e.g. 'I want to put the mug on the plate.').\n"
        )
        if request_required_skills:
            user_prompt += (
                "- Each candidate ADDITIONALLY has a `required_skills` field:\n"
                "  a list of 2-5 short snake_case skill-name hints (e.g.\n"
                "  [\"grasp\", \"place_on_surface\", \"open_drawer\"]) describing\n"
                "  the subskills completing the task will exercise. Reuse names\n"
                "  from the skill library above when applicable; otherwise\n"
                "  invent short descriptive identifiers.\n"
            )
        kwargs: dict[str, Any] = {}
        if self._proposer_temperature is not None:
            kwargs["temperature"] = float(self._proposer_temperature)
        result = query_llm_json(system_prompt, user_prompt, **kwargs)
        raw_candidates = result.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raw_candidates = []
        if not raw_candidates and isinstance(result, dict) and result.get("language"):
            raw_candidates = [result]

        valid: list[dict[str, Any]] = []
        for cand in raw_candidates:
            if not isinstance(cand, dict):
                continue
            veto = self._validate_stage12(self._draft_proposal(cand))
            if veto:
                logger.warning(f"  candidate rejected by curriculum gate: {veto}")
                continue
            pick_veto = self._validate_pick_reliability({
                "goal": cand.get("goal", []) or [],
                "objects": cand.get("objects", []) or [],
            })
            if pick_veto:
                logger.warning(f"  candidate rejected by pick-reliability gate: {pick_veto}")
                continue
            catalog_veto = self._validate_libero_catalog_membership({
                "objects": cand.get("objects", []) or [],
                "fixtures": cand.get("fixtures", []) or [],
                "goal": cand.get("goal", []) or [],
            })
            if catalog_veto:
                logger.warning(f"  candidate rejected by catalog-membership gate: {catalog_veto}")
                continue
            built = self._build_novel_proposal(cand)
            built["candidate_type"] = "fresh"
            built["required_skills"] = self._coerce_skill_list(
                cand.get("required_skills")
            )
            valid.append(built)

        if not valid:
            fallback = self._propose_novel(scene_context, skill_context)
            fallback["candidate_type"] = "fresh"
            fallback.setdefault("required_skills", [])
            return [fallback]
        # The legacy token-overlap N*L scorer (`compute_curiosity_scores`)
        # is only useful for back-compat callers that read `novelty` /
        # `learnability` / `curiosity_score`. When the caller has opted
        # into the new candidate-based scorer (asking for required_skills
        # or LLM scoring), skip it — otherwise iteration logs end up with
        # two competing novelty values per candidate and the new fields
        # silently lose to the old ones in any dict-merge consumer.
        if request_required_skills:
            return valid
        return self.compute_curiosity_scores(valid, skill_context)

    @staticmethod
    def _coerce_skill_list(raw: Any) -> list[str]:
        """Sanitize an LLM-supplied skill name list to <=8 short strings."""
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for s in raw:
            if not isinstance(s, str):
                continue
            name = s.strip()[:60]
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            out.append(name)
            if len(out) >= 8:
                break
        return out

    def propose_retry_candidates(
        self,
        retry_items: list[dict[str, Any]],
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """For each retry-bank item, ask the LLM for a simplified variant.

        The variant is meant to *isolate* the missing skill / affordance
        — not literally re-run the failed task. Each returned candidate
        is a standard novel-proposal dict tagged ``candidate_type="retry_derived"``
        with ``source_retry_id``, ``source_failure_reason``, and the
        retry bank's ``surprise_score`` / ``ttl`` / ``max_ttl`` copied
        through for downstream retry-bonus scoring.

        Validation gates (curriculum / pick-reliability / catalog) are
        reused so the retry candidate never bypasses the same safety
        rails fresh candidates go through.
        """
        if not retry_items:
            return []
        if scene_context.get("env_type") == "molmospaces":
            return []  # LIBERO-only in this rev.

        from rats.agents.libero_catalog import (
            build_catalog_text,
            build_pick_reliability_text,
        )

        out: list[dict[str, Any]] = []
        catalog_text = build_catalog_text()
        pick_text = build_pick_reliability_text()
        for item in retry_items:
            failure_reason = str(item.get("failure_reason", "") or "")
            diagnosis = str(item.get("diagnosis_summary", "") or "")
            original_lang = str(item.get("language", "") or "")
            original_objects = list(item.get("objects") or [])
            original_fixtures = list(item.get("fixtures") or [])
            original_goal = list(item.get("goal") or [])

            extra = (
                "ADDITIONAL RUNTIME INSTRUCTION — RETRY-DERIVED CANDIDATE:\n"
                "Below is a previous DIAGNOSABLE failure. Propose ONE\n"
                "SIMPLER play task that isolates the missing skill or\n"
                "affordance the failure exposed. Do NOT just re-run the\n"
                "original task verbatim. Prefer: same object, simpler goal\n"
                "(e.g. just pick instead of pick-and-place); or same\n"
                "fixture, simpler interaction.\n"
                "\nORIGINAL FAILED TASK:\n"
                f"  language: {original_lang}\n"
                f"  objects: {original_objects}\n"
                f"  fixtures: {original_fixtures}\n"
                f"  goal: {original_goal}\n"
                f"  failure_reason: {failure_reason}\n"
                f"  diagnosis: {diagnosis}\n"
                "\nReturn JSON with the standard single-task schema (the\n"
                "same keys as a fresh novel proposal: language, scene_type,\n"
                "objects, fixtures, goal, expected_new_skills,\n"
                "novelty_score, difficulty_estimate, affordance_hints) PLUS\n"
                "a `required_skills` list (2-5 snake_case hints).\n"
            )
            user_prompt, system_prompt = self._build_novel_prompt(
                scene_context, skill_context,
            )
            user_prompt += "\n\n" + extra
            try:
                result = query_llm_json(system_prompt, user_prompt)
            except Exception as e:
                logger.warning(f"  retry candidate generation failed: {e}")
                continue
            if not isinstance(result, dict) or not result.get("language"):
                continue

            veto = self._validate_stage12(self._draft_proposal(result))
            if veto:
                logger.warning(f"  retry candidate rejected by curriculum: {veto}")
                continue
            pick_veto = self._validate_pick_reliability({
                "goal": result.get("goal", []) or [],
                "objects": result.get("objects", []) or [],
            })
            if pick_veto:
                logger.warning(f"  retry candidate rejected by pick-reliability: {pick_veto}")
                continue
            catalog_veto = self._validate_libero_catalog_membership({
                "objects": result.get("objects", []) or [],
                "fixtures": result.get("fixtures", []) or [],
                "goal": result.get("goal", []) or [],
            })
            if catalog_veto:
                logger.warning(f"  retry candidate rejected by catalog: {catalog_veto}")
                continue
            built = self._build_novel_proposal(result)
            built["candidate_type"] = "retry_derived"
            built["required_skills"] = self._coerce_skill_list(
                result.get("required_skills")
            )
            built["source_retry_id"] = item.get("retry_id")
            built["source_failure_reason"] = failure_reason
            built["source_failure_category"] = item.get("failure_category")
            # Threaded through so the curiosity scorer can compute
            # retry_bonus = surprise * ttl_decay without re-querying the bank.
            built["retry_surprise_score"] = float(item.get("surprise_score", 0.0) or 0.0)
            built["retry_ttl"] = int(item.get("ttl", 0) or 0)
            built["retry_max_ttl"] = int(item.get("max_ttl", 0) or 0)
            built["retry_diagnosable"] = bool(item.get("diagnosable", False))
            # Catalog-side computed scores stay None so iteration logs
            # show clearly which side the values came from.
            out.append(built)
        return out

    def _propose_novel(
        self,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
        *,
        prompt_path: Path | None = None,
    ) -> dict[str, Any]:
        """Propose a novel task with language + objects + goal predicates."""
        if scene_context.get("env_type") == "molmospaces":
            return self._propose_novel_molmospaces(
                scene_context, skill_context,
            )

        user_prompt, system_prompt = self._build_novel_prompt(
            scene_context, skill_context,
            prompt_path=prompt_path,
        )
        result = query_llm_json(system_prompt, user_prompt)
        veto = self._validate_stage12(self._draft_proposal(result))
        if veto:
            logger.warning(
                f"  proposer safety-net rejected (prompt should have "
                f"caught this): {veto} -> re-rolling"
            )
            retry_prompt = (
                user_prompt
                + "\n\nPREVIOUS PROPOSAL REJECTED by curriculum gate:\n"
                + f"  reason: {veto}\n"
                + "Re-propose a SINGLE-PREDICATE pick-and-place task with one "
                + "object, one open container/surface, no doors/drawers/knobs, "
                + "no compound steps. Respond again as JSON.\n"
            )
            result = query_llm_json(system_prompt, retry_prompt)

        pick_veto = self._validate_pick_reliability({
            "goal": result.get("goal", []) or [],
            "objects": result.get("objects", []) or [],
        })
        if pick_veto:
            logger.warning(
                f"  proposer pick-reliability rejected: {pick_veto} "
                f"-> re-rolling"
            )
            retry_prompt = (
                user_prompt
                + "\n\nPREVIOUS PROPOSAL REJECTED by pick-reliability gate:\n"
                + f"  reason: {pick_veto}\n"
                + "Pick a target object from the RELIABLE list in the "
                + "Pick-Primitive Reliability block above. Respond again "
                + "as JSON.\n"
            )
            result = query_llm_json(system_prompt, retry_prompt)

        catalog_veto = self._validate_libero_catalog_membership({
            "objects": result.get("objects", []) or [],
            "fixtures": result.get("fixtures", []) or [],
            "goal": result.get("goal", []) or [],
        })
        if catalog_veto:
            logger.warning(
                f"  proposer catalog-membership rejected: {catalog_veto} "
                f"-> re-rolling"
            )
            retry_prompt = (
                user_prompt
                + "\n\nPREVIOUS PROPOSAL REJECTED by catalog gate:\n"
                + f"  reason: {catalog_veto}\n"
                + "Re-propose using ONLY object/fixture names listed in "
                + "the AVAILABLE BUILDING BLOCKS catalog above. Respond "
                + "again as JSON.\n"
            )
            result = query_llm_json(system_prompt, retry_prompt)

        proposal = self._build_novel_proposal(result)
        scored = self.compute_curiosity_scores([proposal], skill_context)
        if scored:
            proposal["curiosity_score"] = scored[0].get("curiosity_score", 0.25)
            proposal["novelty"] = scored[0].get("novelty", 0.5)
            proposal["learnability"] = scored[0].get("learnability", 0.5)
        return proposal

    # ------------------------------------------------------------------
    # MolmoSpaces novel task proposal
    # ------------------------------------------------------------------

    def _propose_novel_molmospaces(
        self,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch MolmoSpaces novel proposals to the configured proposer mode."""
        if self._molmospaces_proposer_mode == "open":
            return self._propose_novel_molmospaces_open(
                scene_context,
                skill_context,
            )
        if self._molmospaces_proposer_mode == "playtime":
            return self._propose_novel_molmospaces_playtime(
                scene_context,
                skill_context,
            )
        if self._molmospaces_proposer_mode in {
            "benchmark_order",
            "catalog_sequential",
            "sequential",
        }:
            return self._propose_catalog_molmospaces_sequential(scene_context)
        return self._propose_catalog_molmospaces(
            scene_context,
            skill_context,
        )

    def _propose_catalog_molmospaces(
        self,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Propose a task from the MolmoSpaces benchmark catalog (legacy mode)."""
        prompt_path = Path("rats/prompts/task_proposer_novel_molmospaces.txt")
        prompt_template = prompt_path.read_text()

        history_str = "None yet." if not self._task_history else json.dumps(
            self._task_history[-10:], indent=2
        )

        # Build catalog text from available tasks (these come from the
        # MolmoSpaces benchmark catalog via discover_molmospaces_tasks).
        catalog_entries = []
        for t in self._available_tasks:
            catalog_entries.append({
                "canonical_task_id": t.get("canonical_task_id", t.get("activity_name", "")),
                "task_family": t.get("task_family", "unknown"),
                "language": t.get("language", ""),
                "objects": t.get("objects", []),
                "scene_family": t.get("scene_family", ""),
                "benchmark": t.get("benchmark", ""),
            })
        catalog_text = json.dumps(catalog_entries, indent=2) if catalog_entries else "No catalog available."

        user_prompt = prompt_template.replace(
            "{skill_context}", json.dumps(skill_context, indent=2)
        ).replace(
            "{task_history}", history_str
        ).replace(
            "{catalog}", catalog_text
        )

        system_prompt = (
            "You are a curiosity-driven task proposer for a robot learning system. "
            "Propose a NOVEL manipulation task. Respond only in valid JSON."
        )

        result = query_llm_json(system_prompt, user_prompt)

        # Parse the response
        canonical_task_id = result.get("canonical_task_id", "")
        language = result.get("language", "")

        # Validate: canonical_task_id must be in available tasks
        valid_ids = {
            t.get("canonical_task_id", t.get("activity_name", ""))
            for t in self._available_tasks
        }
        if canonical_task_id not in valid_ids:
            # Fallback: pick an untried task or the first available
            attempted = {h["activity_name"] for h in self._task_history}
            canonical_task_id = ""
            for t in self._available_tasks:
                tid = t.get("canonical_task_id", t.get("activity_name", ""))
                if tid not in attempted:
                    canonical_task_id = tid
                    break
            if not canonical_task_id and self._available_tasks:
                t = self._available_tasks[0]
                canonical_task_id = t.get("canonical_task_id", t.get("activity_name", ""))

        # Find matching task metadata
        task_meta = next(
            (t for t in self._available_tasks
             if t.get("canonical_task_id", t.get("activity_name", "")) == canonical_task_id),
            {},
        )
        if not language:
            language = task_meta.get("language", canonical_task_id)

        proposal = {
            "mode": "novel",
            "activity_name": canonical_task_id,
            "canonical_task_id": canonical_task_id,
            "language": language,
            "task_family": result.get("task_family", task_meta.get("task_family", "")),
            "objects": result.get("objects", task_meta.get("objects", [])),
            "scene_family": result.get("scene_family", task_meta.get("scene_family", "")),
            "benchmark": result.get("benchmark", task_meta.get("benchmark", "")),
            "scene_model": task_meta.get("scene_model", scene_context.get("scene_model", "unknown")),
            "activity_definition_id": 0,
            "goal_conditions": language,
            "reasoning": result.get("reasoning", ""),
            "expected_new_skills": result.get("expected_new_skills", []),
            "novelty_score": result.get("novelty_score", 0.5),
            "difficulty_estimate": result.get("difficulty_estimate", "medium"),
        }

        # Score the proposal
        scored = self.compute_curiosity_scores([proposal], skill_context)
        if scored:
            proposal["curiosity_score"] = scored[0].get("curiosity_score", 0.25)
            proposal["novelty"] = scored[0].get("novelty", 0.5)
            proposal["learnability"] = scored[0].get("learnability", 0.5)

        return proposal

    def _propose_catalog_molmospaces_sequential(
        self,
        scene_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the next MolmoSpaces catalog task in benchmark order.

        This mode is for benchmark sweeps where the evaluator should cover
        every JSON episode exactly once before cycling, instead of asking an
        LLM proposer to choose among catalog entries.
        """
        if not self._available_tasks:
            raise ValueError("MolmoSpaces benchmark_order mode requires a non-empty task catalog")

        attempted = {h["activity_name"] for h in self._task_history}
        task_meta = next(
            (
                t for t in self._available_tasks
                if t.get("canonical_task_id", t.get("activity_name", "")) not in attempted
            ),
            None,
        )
        if task_meta is None:
            task_meta = self._available_tasks[len(self._task_history) % len(self._available_tasks)]

        canonical_task_id = task_meta.get("canonical_task_id", task_meta.get("activity_name", ""))
        language = task_meta.get("language", canonical_task_id)
        return {
            "mode": "benchmark_order",
            "activity_name": canonical_task_id,
            "canonical_task_id": canonical_task_id,
            "language": language,
            "task_family": task_meta.get("task_family", ""),
            "objects": task_meta.get("objects", []),
            "scene_family": task_meta.get("scene_family", ""),
            "benchmark": task_meta.get("benchmark", ""),
            "scene_model": task_meta.get("scene_model", scene_context.get("scene_model", "unknown")),
            "activity_definition_id": 0,
            "goal_conditions": language,
            "reasoning": "Deterministic MolmoSpaces benchmark_order selection.",
            "expected_new_skills": [],
            "novelty_score": 0.5,
            "difficulty_estimate": "benchmark",
        }

    # ------------------------------------------------------------------
    # MolmoSpaces open-mode novel proposer (inventory-driven)
    # ------------------------------------------------------------------

    # Single-robot-base reach budget for pick_and_place pairs. The Franka
    # FR3 has ~0.85 m max-reach from base; with the bridge re-placing the
    # base between grasp and place phases we get a bit more, but procthor
    # rooms regularly span 5–10 m so the LLM otherwise picks targets in
    # different rooms (mug at one end, bed at the other). 1.5 m is the
    # ceiling for "the same workspace can serve both phases".
    _PICK_PLACE_MAX_XY_M: float = 1.5

    @staticmethod
    def _lookup_inventory_position(
        inventory: dict[str, Any], internal_name: str | None,
    ) -> tuple[float, float, float] | None:
        """Return the (x, y, z) world position recorded in the inventory.

        Walks pickables / receptacles / articulations (the only categories the
        bridge populates positions for). Returns None when the entry is not
        found or has no position field.
        """
        if not internal_name:
            return None
        for kind in (
            *TaskProposer._PLAYTIME_TARGET_SECTIONS,
            *TaskProposer._PLAYTIME_PLACEABLE_SECTIONS,
        ):
            for entry in inventory.get(kind, []) or []:
                if entry.get("internal_name") != internal_name:
                    continue
                pos = entry.get("position")
                if not isinstance(pos, (list, tuple)) or len(pos) < 3:
                    return None
                try:
                    return (float(pos[0]), float(pos[1]), float(pos[2]))
                except (TypeError, ValueError):
                    return None
        return None

    # Curriculum gates for the open-mode MolmoSpaces proposer. Mirror of
    # the LIBERO `_curriculum_stage`/`_curriculum_hint` pair but expressed
    # in MolmoSpaces task types instead of BDDL predicates.
    _MOLMOSPACES_CURRICULUM_HINTS: dict[int, str] = {
        1: (
            "STAGE 1 (0-2 successes so far): propose ONLY a `pick` task on a "
            "small object that has has_grasp_file=true and is sitting on a "
            "receptacle. No open / close / nav."
        ),
        2: (
            "STAGE 2 (3-9 successes): you may propose `pick`, `pick_and_place` "
            "(target sitting on a clear receptacle), or `open` on a drawer or "
            "cabinet. No `close` (requires the joint to be pre-opened) or `nav`."
        ),
        3: (
            "STAGE 3 (10+ successes): all task types are unlocked, including "
            "`close`, `nav`, and cross-room targets."
        ),
    }

    def _curriculum_stage_molmospaces(self) -> int:
        """Curriculum stage based on MolmoSpaces task successes.

        Counted from the same `_task_history` the LIBERO curriculum uses
        but with the curriculum knob defaulting on for MolmoSpaces (the
        upstream sampler is fragile enough that gating early proposals
        to `pick` is the right default even without an explicit
        curriculum=True flag at construction).
        """
        successes = sum(
            1 for h in self._task_history
            if h.get("success") and h.get("env_type", "molmospaces") == "molmospaces"
        )
        if successes < 3:
            return 1
        if successes < 10:
            return 2
        return 3

    _DEFAULT_PLAYTIME_INTERACTIONS = [
        "touch", "tap", "push", "pull", "slide", "roll", "lift", "drop",
        "shake", "stack", "knock_over", "place_on", "place_in", "open",
        "close",
    ]

    @staticmethod
    def _category_from_internal(internal: str | None) -> str:
        if not internal:
            return ""
        head = str(internal).split("_", 1)[0]
        return head.lower()

    @classmethod
    def _affordance_keys_for_action(cls, action: str) -> tuple[str, ...]:
        """Mirror of PlaytimeMemory._affordances_for_action without the
        import dependency, so the proposer can score candidates even when
        the memory object isn't injected (e.g. unit tests)."""
        action = (action or "").strip().lower()
        tokens = action.replace("-", "_").replace(" ", "_")
        if (
            action in {"push", "tap", "touch", "slide", "knock_over", "nudge", "press", "poke"}
            or any(tok in tokens for tok in ("push", "tap", "touch", "slide", "nudge", "press", "poke", "contact"))
        ):
            return (
                "moves_under_light_contact", "slides_on_surface", "tippable",
                "contact_sensitive", "rolls", "resists_motion",
            )
        if (
            action in {"lift", "pick", "place_in", "place_on", "stack", "lower"}
            or any(tok in tokens for tok in ("lift", "pick", "place", "stack", "lower", "grasp"))
        ):
            return ("graspable_topdown", "tippable", "contact_sensitive", "rolls")
        if (
            action in {"open", "close", "pull", "tug", "wiggle", "articulate"}
            or any(tok in tokens for tok in ("open", "close", "pull", "tug", "wiggle", "drawer", "door", "lid", "hinge", "handle", "seam"))
        ):
            return ("articulated_movable", "resists_motion", "contact_sensitive")
        if action in {"shake", "drop", "roll"} or any(tok in tokens for tok in ("shake", "drop", "roll", "release")):
            return ("graspable_topdown", "tippable", "contact_sensitive")
        # Fallback: score against the full key set.
        return (
            "moves_under_light_contact", "rolls", "tippable",
            "resists_motion", "contact_sensitive", "graspable_topdown",
            "articulated_movable", "slides_on_surface",
        )

    @classmethod
    def _canonical_playtime_action(cls, raw: dict[str, Any] | None) -> str:
        """Map freeform action text to a stable action family for scoring.

        ``interaction_type`` stays backward-compatible for normal playtime
        runs. In freeform proposal mode, the LLM may write a phrase like
        "press the hamper lid lip and wiggle lightly"; this helper derives a
        coarse action family so Piaget memory keys and grounded verification
        still have a meaningful schema.
        """
        raw = raw or {}
        known = cls._PLAYTIME_KNOWN_ACTIONS
        # In freeform mode the proposer LLM is asked to classify its own
        # phrase while it generates the proposal.  Prefer that explicit
        # classification over brittle substring matching; the heuristic below
        # is only a compatibility fallback for older traces/tests or malformed
        # model outputs.
        for key in (
            "interaction_family",
            "canonical_interaction_type",
            "canonical_interaction",
            "action_family",
        ):
            classified = str(raw.get(key) or "").strip().lower()
            classified = classified.replace("-", "_").replace(" ", "_")
            if classified in known:
                return classified
        explicit = str(raw.get("interaction_type") or "").strip().lower()
        if explicit in known:
            return explicit
        text = " ".join(
            str(raw.get(k) or "")
            for k in (
                "interaction_type",
                "interaction_text",
                "language",
                "exploration_question",
                "expected_observation",
            )
        ).strip().lower()
        compact = text.replace("-", "_")
        words = set(re.findall(r"[a-z0-9_]+", compact))

        def phrase_match(needle: str) -> bool:
            needle = needle.strip().lower().replace("-", "_")
            if " " in needle:
                return re.search(rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])", compact) is not None
            return needle in words

        rules: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("place_in", ("place in", "put in", "drop into", "inside", "contain")),
            ("place_on", ("place on", "put on", "set on", "onto", "support")),
            ("stack", ("stack", "balance on top")),
            ("open", ("open", "pull open", "lid", "hinge", "handle", "drawer", "door")),
            ("close", ("close", "push closed", "shut")),
            ("pull", ("pull", "tug", "draw out")),
            ("wiggle", ("wiggle", "jiggle", "rock")),
            ("push", ("push", "press", "nudge", "poke", "tap", "touch", "contact")),
            ("slide", ("slide", "drag")),
            ("rotate", ("rotate", "twist", "turn")),
            ("roll", ("roll",)),
            ("lift", ("lift", "pick up", "raise")),
            ("lower", ("lower", "set down")),
            ("drop", ("drop", "release")),
            ("shake", ("shake",)),
        )
        for action, needles in rules:
            if any(phrase_match(needle) for needle in needles):
                return action
        fallback = str(raw.get("interaction_type") or "touch").strip().lower()
        fallback = re.sub(r"[^a-z0-9_]+", "_", fallback).strip("_")
        return fallback or "touch"

    @staticmethod
    def _resolve_belief(aff: dict[str, Any] | None) -> tuple[float, int]:
        """Collapse one affordance bucket to (confidence, n_total).

        Returns (0.0, 0) when there's no evidence yet — the caller treats
        that as a high-prior unknown (max curiosity).
        """
        if not isinstance(aff, dict):
            return 0.0, 0
        pos = float(aff.get("supporting_weight", 0.0) or 0.0)
        neg = float(aff.get("contradicting_weight", 0.0) or 0.0)
        n_pos = int(aff.get("n_supporting", 0) or 0)
        n_neg = int(aff.get("n_contradicting", 0) or 0)
        total_w = pos + neg
        n_total = n_pos + n_neg
        if total_w <= 0:
            return 0.0, n_total
        confidence = max(pos, neg) / total_w
        return min(confidence, 0.95), n_total

    _PLAYTIME_SIMPLE_VERBS = (
        "touch", "tap", "poke", "press", "push", "pull", "tug",
        "wiggle", "slide", "nudge", "rotate", "lift", "lower",
        "drop", "shake",
    )
    # Compound (pick + place) and articulation verbs are deferred until
    # the toddler has a basic affordance model — i.e. until the playtime
    # memory has at least a few observations to build on.
    _PLAYTIME_COMPOUND_VERBS = ("place_on", "place_in", "stack", "roll")
    _PLAYTIME_ARTICULATED_VERBS = ("open", "close")
    _PLAYTIME_KNOWN_ACTIONS = frozenset({
        "touch", "tap", "poke", "press", "push", "pull", "slide",
        "rotate", "lift", "lower", "drop", "shake", "place_on",
        "place_in", "stack", "open", "close", "wiggle", "roll",
        "knock_over", "nudge", "tug",
    })

    @staticmethod
    def _count_playtime_history_entries(
        task_history: list[dict[str, Any]],
    ) -> int:
        """How many recent rows are actual playtime attempts."""
        return sum(
            1 for h in (task_history or [])
            if isinstance(h, dict)
            and (
                isinstance(h.get("playtime_metadata"), dict)
                and h.get("playtime_metadata")
                or str(h.get("activity_name") or "").startswith("molmospaces:playtime:")
            )
        )

    @classmethod
    def _playtime_curriculum_hint(
        cls,
        task_history: list[dict[str, Any]],
        playtime_memory_obj: Any,
        *,
        cold_start_threshold: int = 1,
        warmup_threshold: int = 3,
    ) -> str:
        """Difficulty-cap hint for early-iteration playtime proposals.

        At iteration 1, the robot has zero prior observations, the skill
        library is bare, and a compound verb like ``place_in`` requires
        a working pick+place stack to even have a chance. The hint
        nudges the LLM to defer compound and articulation verbs until
        the playtime memory has a few simple-contact observations.

        The score-driven argmax can still override this preference if
        the LLM emits a much higher-IG compound candidate, but at cold
        start IG is uniform (everything is unobserved), so the hint
        wording is what actually carries the bias.
        """
        n_hist = cls._count_playtime_history_entries(task_history)
        n_mem = 0
        if playtime_memory_obj is not None:
            try:
                n_mem = len(getattr(playtime_memory_obj, "entries", []) or [])
            except Exception:
                n_mem = 0
        n = max(n_hist, n_mem)

        articulated = ", ".join(cls._PLAYTIME_ARTICULATED_VERBS)

        if n < cold_start_threshold:
            return (
                "CURRICULUM STAGE 1 (hard constraint, 0 prior playtime "
                "attempts so far):\n"
                "- Propose ONLY a single-step pick-and-place task: exactly "
                "ONE pickable object moved to ONE container or surface. "
                "Use `place_on` or `place_in` as the interaction_type.\n"
                "- Forbidden: contact-only verbs (touch, tap, poke, press, "
                "push, pull, tug, slide, wiggle, nudge, rotate, lift, "
                "lower, drop, shake) — they produce no usable skill and "
                "waste an iteration.\n"
                f"- Forbidden: articulation verbs ({articulated}), stack, "
                "and any multi-step 'and then', 'and close', 'and turn on' "
                "compounding.\n"
                "- Pick (target, container) pairs the seed library's "
                "grasp+place primitives can realistically handle — a "
                "graspable item + an open receptacle/surface visible in "
                "the live inventory."
            )
        if n < warmup_threshold:
            return (
                f"CURRICULUM STAGE 2 ({n} prior playtime attempt(s) so far):\n"
                "- Continue to propose single-step pick-and-place "
                "(`place_on` / `place_in`); you may combine novel "
                "(target, container) pairs not yet attempted.\n"
                "- Still forbidden: contact-only verbs (touch, tap, poke, "
                "press, push, pull, tug, slide, wiggle, nudge, rotate, "
                "lift, lower, drop, shake), articulation verbs "
                f"({articulated}), stack, and any multi-step compounding."
            )
        return (
            "Curriculum: free exploration. All allowed verbs are fair "
            "game; the curiosity-loss ranker will pick the candidate with "
            "the highest expected information gain."
        )

    @staticmethod
    def _verb_coverage_hint(
        task_history: list[dict[str, Any]],
        allowed_interactions: list[str],
        *,
        window: int = 20,
        top_k: int = 3,
    ) -> str:
        """Short prompt-injectable string flagging under-explored verbs.

        Counts interaction-verb usage across the trailing ``window``
        playtime attempts and surfaces the bottom-``top_k`` verbs (with
        priority to verbs never attempted) so the LLM can prefer them
        when sampling K candidates. Permissive language: a hint, not a
        constraint, so the loss-driven argmax can still pick a
        higher-IG candidate that uses a more-frequent verb.
        """
        if not allowed_interactions:
            return ""
        counts: dict[str, int] = {
            str(v).strip().lower(): 0
            for v in allowed_interactions
            if str(v).strip()
        }
        if not counts:
            return ""
        recent = [
            h for h in (task_history or [])
            if isinstance(h, dict)
            and (
                isinstance(h.get("playtime_metadata"), dict)
                and h.get("playtime_metadata")
                or str(h.get("activity_name") or "").startswith("molmospaces:playtime:")
            )
        ][-int(max(1, window)):]
        for entry in recent:
            meta = entry.get("playtime_metadata") or {}
            verb = str(meta.get("interaction_type") or "").strip().lower()
            if verb in counts:
                counts[verb] += 1
        if not recent:
            return (
                "Verb coverage: no prior playtime attempts yet — any "
                "sensorimotor verb in the allowed list is reasonable."
            )
        ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
        untried = [v for v, c in ranked if c == 0]
        k = max(1, int(top_k))
        if untried:
            picks = untried[:k]
            return (
                f"Verb coverage (last {len(recent)} attempts): never tried: "
                f"{', '.join(picks)}. Prefer one of these in at least one "
                f"of your sampled candidates unless an obviously better "
                f"target/affordance experiment exists."
            )
        picks = [v for v, _ in ranked[:k]]
        counts_str = ", ".join(f"{v} ({counts[v]})" for v in picks)
        return (
            f"Verb coverage (last {len(recent)} attempts): least-used "
            f"verbs are {counts_str}. Prefer one of these in at least one "
            f"of your sampled candidates unless an obviously better "
            f"target/affordance experiment exists."
        )

    # LIBERO-aligned scoring: ``base = novelty * frontier`` (product
    # composition) via agents.curiosity_scoring.score_candidate. No
    # Piaget bonuses/penalties layer on top (the legacy affordance +
    # variation scoring is no longer used).

    @staticmethod
    def _format_playtime_curiosity_components(
        curiosity: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a stable trace payload of LIBERO curiosity fields."""
        raw = curiosity.get("playtime_curiosity_raw")
        if raw is None:
            raw = curiosity.get("score_breakdown", {})
        return {
            "curiosity_mode": curiosity.get("curiosity_mode", "formula"),
            "novelty_score": curiosity.get("novelty_score", 0.0),
            "frontier_score": curiosity.get("frontier_score", 0.0),
            "competence_estimate": curiosity.get("competence_estimate", 0.0),
            "failure_penalty": curiosity.get("failure_penalty", 0.0),
            "llm_novelty_score": curiosity.get("llm_novelty_score"),
            "llm_frontier_score": curiosity.get("llm_frontier_score"),
            "formula_novelty_score": curiosity.get("formula_novelty_score"),
            "formula_frontier_score": curiosity.get("formula_frontier_score"),
            "objects_projection": curiosity.get("objects_projection", []),
            "required_skills_projection": curiosity.get(
                "required_skills_projection", []
            ),
            "raw": raw,
            "weights": curiosity.get("weights", {}),
        }

    @classmethod
    def _select_target_by_curiosity_internal(
        cls,
        *,
        taskable_target_internal: list[str],
        allowed_verbs: list[str],
        skill_lookup: Any,
        history_counts: dict[tuple[str, str], int],
        task_history: list[dict[str, Any]],
    ) -> tuple[str | None, dict[str, float]]:
        """Stage-A target selection: pick the (target, verb)-marginal argmax.

        Runs LIBERO's ``score_candidate`` for every (target × verb) pair
        in the inventory × allowed_interactions cross-product, then for
        each target takes the **max-over-verbs** score and returns the
        target with the highest such max.

        This is cheap (no LLM) and lets the bridge anchor the robot near
        the curiosity-chosen target BEFORE the LLM picks a verb/secondary
        in the post-anchor scene. Without this, the LLM picked verb +
        secondary against the bridge's round-robin anchor (often a
        different, less-curious target).

        Returns ``(best_target_internal_name, per_target_scores_dict)``.
        Returns ``(None, {})`` when the cross-product is empty or any
        of the required inputs is missing.
        """
        from rats.agents.curiosity_scoring import score_candidate

        if not taskable_target_internal or not allowed_verbs:
            return None, {}
        if skill_lookup is None or history_counts is None:
            return None, {}

        per_target: dict[str, float] = {}
        best_target: str | None = None
        best_score = -float("inf")
        for target in taskable_target_internal:
            best_for_this_target = -float("inf")
            for verb in allowed_verbs:
                cand = {
                    "objects": [target],
                    "required_skills": [verb],
                    "language": f"{verb} {target}",
                    "fixtures": [],
                    "retry_bonus_score": 0.0,
                }
                score_candidate(
                    cand,
                    mode="formula",
                    skill_lookup=skill_lookup,
                    history_counts=history_counts,
                    task_history=task_history,
                    retry_bonus_weight=0.0,
                    failure_penalty_weight=0.3,
                    score_composition="product",
                )
                s = float(cand.get("final_score", 0.0) or 0.0)
                if s > best_for_this_target:
                    best_for_this_target = s
            per_target[target] = round(best_for_this_target, 4)
            if best_for_this_target > best_score:
                best_score = best_for_this_target
                best_target = target
        return best_target, per_target

    @classmethod
    def _compute_playtime_curiosity(
        cls,
        proposal: dict[str, Any],
        playtime_memory_obj: Any,  # kept for API compat; unused under alignment
        task_history: list[dict[str, Any]],
        *,
        weights: dict[str, float] | None = None,
        skill_lookup: Any = None,
        history_counts: dict[tuple[str, str], int] | None = None,
        mode: str = "formula",
    ) -> dict[str, Any]:
        """Score a playtime candidate using LIBERO's curiosity formula.

        Delegates to ``agents.curiosity_scoring.score_candidate`` so
        playtime and LIBERO pick the same scoring philosophy:

          novelty   = mean(1/sqrt(N(o,s)+1)) over (object, skill) pairs
          competence= mean(Wilson-LB) over required_skills
          frontier  = 4·c·(1-c) Goldilocks
          base      = novelty × frontier   (product composition)
          final     = base − w_fail · failure_penalty

        Retry bonus is always 0 here (the playtime proposer has no
        retry bank). Failure penalty uses LIBERO's Jaccard-overlap
        formulation against ``task_history``.

        The playtime candidate is projected onto LIBERO's (objects,
        required_skills) schema:
          object  = target_internal_name (or target_category fallback)
          skills  = [interaction_type] + any api_primitives_used hints

        Legacy fallback (when ``skill_lookup`` or ``history_counts`` is
        missing): returns a fixed mid-range score so callers still get
        a well-formed dict. The lifelong-loop playtime path passes both
        lookups, so production paths always go through LIBERO scoring.
        """
        from rats.agents.curiosity_scoring import (
            compute_recent_failure_penalty,
            score_candidate,
        )

        # Tunable knobs. Honor caller-supplied weights only for keys
        # that LIBERO's score_candidate exposes; ignore the legacy
        # piaget_curiosity_weights aliases (info/variation/freeform/
        # adult_penalty) since those signals are no longer used.
        w = {
            "retry_bonus_weight": 0.0,
            "failure_penalty_weight": 0.3,
            "score_composition": "product",
        }
        if weights:
            for k, v in weights.items():
                k = str(k)
                if k in w and not isinstance(w[k], str):
                    try:
                        w[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                elif k == "score_composition" and isinstance(v, str):
                    w[k] = v

        # Project playtime candidate onto LIBERO schema.
        pt = proposal.get("_playtime") or {}
        target_obj = (
            pt.get("target_internal_name")
            or proposal.get("playtime_target_category")
            or (proposal.get("objects") or [""])[0]
        )
        verb = (
            pt.get("interaction_type")
            or proposal.get("interaction_type")
            or ""
        )
        required_skills: list[str] = [verb] if verb else []
        for prim in (proposal.get("api_primitives_used") or []):
            if isinstance(prim, str) and prim:
                required_skills.append(prim)
        cand_view: dict[str, Any] = {
            "objects": [str(target_obj)] if target_obj else [],
            "required_skills": required_skills,
            "language": proposal.get("language", ""),
            "activity_name": proposal.get("activity_name", ""),
            "fixtures": [],
            "retry_bonus_score": 0.0,
        }
        # Pass through LLM-emitted scores when present so
        # ``score_candidate(mode="llm", ...)`` finds them on the view.
        for k in ("llm_novelty_score", "llm_frontier_score", "llm_rationale"):
            if k in proposal:
                cand_view[k] = proposal[k]
        # Normalize mode: 'llm' only if both lookups + at least one LLM
        # score are present; otherwise fall to 'formula' to preserve
        # numeric ranking.
        active_mode = "llm" if (
            str(mode).lower() == "llm"
            and (
                cand_view.get("llm_novelty_score") is not None
                or cand_view.get("llm_frontier_score") is not None
            )
        ) else "formula"

        if skill_lookup is not None and history_counts is not None:
            # Full LIBERO scoring.
            score_candidate(
                cand_view,
                mode=active_mode,
                skill_lookup=skill_lookup,
                history_counts=history_counts,
                task_history=task_history,
                retry_bonus_weight=float(w["retry_bonus_weight"]),
                failure_penalty_weight=float(w["failure_penalty_weight"]),
                score_composition=str(w["score_composition"]),
            )
            final_score = float(cand_view.get("final_score", 0.0) or 0.0)
            breakdown = cand_view.get("score_breakdown", {}) or {}
            novelty = float(cand_view.get("novelty_score", 0.0) or 0.0)
            frontier = float(cand_view.get("frontier_score", 0.0) or 0.0)
            competence = float(cand_view.get("competence_estimate", 0.0) or 0.0)
            failure_penalty = float(cand_view.get("failure_penalty", 0.0) or 0.0)
        else:
            # Lookups missing: cannot compute LIBERO score. Return neutral
            # ranking value so downstream argmax doesn't bias on noise.
            # The legacy weighted-sum is intentionally NOT resurrected
            # here — the rest of the system has been switched over.
            failure_penalty = compute_recent_failure_penalty(
                cand_view, task_history,
            )
            final_score = max(0.0, 0.5 - w["failure_penalty_weight"] * failure_penalty)
            breakdown = {"note": "skill_lookup or history_counts missing; using neutral 0.5 base"}
            novelty = frontier = competence = 0.5

        return {
            "playtime_curiosity_score": round(final_score, 4),
            "curiosity_mode": active_mode,  # 'llm' | 'formula' actually used
            # LIBERO-aligned fields (the load-bearing ones).
            "novelty_score": round(novelty, 4),
            "frontier_score": round(frontier, 4),
            "competence_estimate": round(competence, 4),
            "failure_penalty": round(failure_penalty, 4),
            "llm_novelty_score": cand_view.get("llm_novelty_score"),
            "llm_frontier_score": cand_view.get("llm_frontier_score"),
            "formula_novelty_score": cand_view.get("formula_novelty_score"),
            "formula_frontier_score": cand_view.get("formula_frontier_score"),
            "objects_projection": cand_view["objects"],
            "required_skills_projection": required_skills,
            "weights": w,
            "score_breakdown": breakdown,
        }

    # ------------------------------------------------------------------
    # Unsuitability filter: oversize + reachability
    # ------------------------------------------------------------------
    #
    # Two failure modes the proposer should catch BEFORE the policy
    # writer / executor wastes an iteration on a target the robot can't
    # physically interact with:
    #
    # 1. Oversized assets: category or asset_uid known to be wider than
    #    the Franka gripper. See molmospaces_catalog.is_oversized_pick_target.
    # 2. Out-of-reach: target position is farther than the robot's
    #    Franka envelope from the current robot base. We check
    #    Euclidean distance from the live robot_base_pose to the
    #    inventory's reported object position.
    #
    # The check is applied to the LLM's chosen target during validation
    # (and also to the secondary for compound verbs, with the oversize
    # rule relaxed since the secondary is typically a receptacle/surface
    # we don't need to grasp).

    # Default single-base reach envelope for playtime target selection.
    # This is deliberately a proposal/verifier gate, not a hard IK limit:
    # downstream motion planning may still fail if a particular approach pose
    # is unreachable. Tunable via ``molmospaces.playtime.max_reach_m``.
    _DEFAULT_PLAYTIME_MAX_REACH_M = 1.5
    _PLAYTIME_TARGET_SECTIONS = ("pickables", "articulations")
    _PLAYTIME_PLACEABLE_SECTIONS = ("placeables", "receptacles")

    @classmethod
    def _force_include_anchored_benchmark_targets(
        cls,
        inventory: dict[str, Any],
    ) -> None:
        """Add bridge-pinned benchmark targets missing from taskable inventory.

        MolmoSpaces benchmark episodes can pin a target object and recorded
        camera while ``describe_scene_inventory`` only reports that object in a
        room's raw ``object_names`` list. Playtime target selection is driven by
        the structured ``pickables`` / ``articulations`` lists, so such targets
        otherwise disappear before prompt building. This creates a small
        synthetic inventory row so the normal grounding, validation, and
        fallback paths can still select the benchmark anchor.
        """
        if not isinstance(inventory, dict):
            return
        anchor = inventory.get("anchored_target")
        if not isinstance(anchor, dict):
            return

        task_type = str(anchor.get("task_type") or "").strip().lower()
        primary_names: list[tuple[str, str]] = []

        joint_name = str(anchor.get("joint_name") or "").strip()
        pickup_name = str(anchor.get("pickup_obj_name") or "").strip()
        if task_type in {"open", "close"}:
            if joint_name:
                primary_names.append((joint_name, "articulations"))
            if pickup_name:
                primary_names.append((pickup_name, "articulations"))
        elif pickup_name:
            primary_names.append((pickup_name, "pickables"))
        elif joint_name:
            primary_names.append((joint_name, "articulations"))

        place_name = str(anchor.get("place_receptacle_name") or "").strip()
        secondary_names = [(place_name, "placeables")] if place_name else []

        for internal_name, section in (*primary_names, *secondary_names):
            if not internal_name or cls._inventory_has_internal_name(
                inventory, internal_name,
            ):
                continue
            inventory.setdefault(section, []).append(
                cls._synthetic_anchor_inventory_entry(
                    inventory,
                    internal_name,
                    section=section,
                    anchor=anchor,
                )
            )

    @classmethod
    def _inventory_has_internal_name(
        cls,
        inventory: dict[str, Any],
        internal_name: str,
    ) -> bool:
        for section in (*cls._PLAYTIME_TARGET_SECTIONS, *cls._PLAYTIME_PLACEABLE_SECTIONS):
            for entry in inventory.get(section, []) or []:
                if isinstance(entry, dict) and entry.get("internal_name") == internal_name:
                    return True
        return False

    @classmethod
    def _synthetic_anchor_inventory_entry(
        cls,
        inventory: dict[str, Any],
        internal_name: str,
        *,
        section: str,
        anchor: dict[str, Any],
    ) -> dict[str, Any]:
        category = cls._category_phrase_from_internal(internal_name)
        room = cls._room_for_internal_name(inventory, internal_name) or "unknown"
        entry: dict[str, Any] = {
            "internal_name": internal_name,
            "category": category,
            "room": room,
            "source": "anchored_benchmark_target",
            "anchored_benchmark_target": True,
        }
        if section == "articulations":
            joint_name = str(anchor.get("joint_name") or internal_name).strip()
            joint: dict[str, Any] = {"name": joint_name, "type": "unknown"}
            if anchor.get("joint_index") is not None:
                joint["index"] = anchor.get("joint_index")
            entry["joints"] = [joint]
        return entry

    @staticmethod
    def _room_for_internal_name(
        inventory: dict[str, Any],
        internal_name: str,
    ) -> str | None:
        for room in inventory.get("rooms", []) or []:
            if not isinstance(room, dict):
                continue
            if internal_name not in (room.get("object_names") or []):
                continue
            raw = str(room.get("name") or "").strip()
            if not raw:
                return None
            parts = [
                part for part in raw.split("_")
                if part and part != "room" and not part.isdigit()
            ]
            label = " ".join(parts) or raw
            return label.replace("livingroom", "living room")
        return None

    @staticmethod
    def _category_phrase_from_internal(internal_name: str) -> str:
        raw = str(internal_name or "").strip()
        if raw.startswith("place_receptacle/"):
            return "placement receptacle"
        head = raw.split("_", 1)[0].strip()
        if "/" in head:
            head = head.rsplit("/", 1)[-1]
        lowered = head.lower()
        if lowered.startswith("obja") and len(lowered) > 4:
            lowered = lowered[4:]
        phrase_overrides = {
            "cellulartelephone": "cellular telephone",
            "decorativecornerpiece": "decorative corner piece",
            "decorativemushroom": "decorative mushroom",
            "decorativewindmill": "decorative windmill",
            "firedepartmentconnection": "fire department connection",
            "portablecomputer": "portable computer",
            "roboticdog": "robotic dog",
            "robotichead": "robotic head",
            "spinningtop": "spinning top",
            "tennisracket": "tennis racket",
        }
        if lowered in phrase_overrides:
            return phrase_overrides[lowered]
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", lowered).strip().lower()
        return spaced or "object"

    @staticmethod
    def _get_robot_base_xyz(env: Any) -> tuple[float, float, float] | None:
        """Best-effort robot base xyz from the latest observation."""
        try:
            from rats.agents.molmospaces_scene_grounder import _capture_observation
        except Exception:
            return None
        try:
            obs = _capture_observation(env)
        except Exception:
            return None
        if not isinstance(obs, dict):
            return None
        pose = obs.get("robot_base_pose")
        if pose is None:
            return None
        try:
            x = float(pose[0]); y = float(pose[1]); z = float(pose[2])
        except (TypeError, ValueError, IndexError):
            return None
        if not all(math.isfinite(v) for v in (x, y, z)):
            return None
        return (x, y, z)

    @classmethod
    def _build_playtime_unsuitable_check(
        cls,
        inventory: dict[str, Any],
        robot_base_xyz: tuple[float, float, float] | None,
        max_reach_m: float,
    ):
        """Return a callable ``(internal_name, role) -> reason str``.

        ``role`` is ``"target"`` (apply oversize + reach check) or
        ``"secondary"`` (apply reach check only — we do not need to
        grasp the receptacle).
        """
        from rats.agents.molmospaces_catalog import is_oversized_pick_target

        meta_by_internal: dict[str, dict[str, Any]] = {}
        for section in (
            *cls._PLAYTIME_TARGET_SECTIONS,
            *cls._PLAYTIME_PLACEABLE_SECTIONS,
        ):
            for entry in (inventory.get(section) or []):
                if not isinstance(entry, dict):
                    continue
                internal = entry.get("internal_name")
                if not internal:
                    continue
                meta_by_internal[internal] = {
                    "category": str(entry.get("category") or ""),
                    "asset_uid": str(entry.get("asset_uid") or ""),
                    "position": entry.get("position") or [0.0, 0.0, 0.0],
                    "section": section,
                }

        def check(internal: str, *, role: str = "target") -> str:
            if not internal:
                return ""
            meta = meta_by_internal.get(internal)
            if not meta:
                return ""
            if role == "target" and is_oversized_pick_target(
                meta["category"], meta["asset_uid"]
            ):
                return (
                    f"oversized for the gripper "
                    f"(category={meta['category']!r}, "
                    f"asset={meta['asset_uid']!r})"
                )
            if robot_base_xyz is not None:
                pos = meta["position"]
                try:
                    px = float(pos[0]); py = float(pos[1]); pz = float(pos[2])
                except (TypeError, ValueError, IndexError):
                    return ""
                rx, ry, rz = robot_base_xyz
                dist = math.sqrt(
                    (px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2
                )
                if dist > max_reach_m:
                    return (
                        f"out of reach (3D dist={dist:.2f}m > "
                        f"max_reach={max_reach_m:.2f}m from robot base)"
                    )
            return ""

        return check

    @classmethod
    def _playtime_taskable_target_names(
        cls,
        inventory: dict[str, Any],
        *,
        benchmark_task_meta: dict[str, Any] | None = None,
    ) -> set[str]:
        """Primary playtime targets that RATS can ground as object tasks.

        Main targets are limited to actual free-bodied pickables or explicit
        articulations. Receptacles/placeables are intentionally excluded here:
        they are valid *secondary* placement/support targets, but allowing them
        as primary targets caused static drawers/dishwashers to be proposed as
        openable objects. The current benchmark open/close target is included
        as an explicit articulation source because some MolmoSpaces versions
        under-report the live ``articulations`` list for pinned benchmark
        scenes even though benchmark metadata supplies the joint.
        """
        names: set[str] = set()

        def add(name: Any) -> None:
            internal = str(name or "").strip()
            if not internal:
                return
            names.add(internal)
            names.add(cls._root_object_internal_name(internal))

        for section in cls._PLAYTIME_TARGET_SECTIONS:
            for entry in inventory.get(section, []) or []:
                if isinstance(entry, dict):
                    add(entry.get("internal_name"))

        benchmark_task_meta = benchmark_task_meta or {}
        task_kind = str(
            benchmark_task_meta.get("task_family")
            or (benchmark_task_meta.get("metadata") or {}).get("task_type")
            or ""
        ).strip().lower()
        if task_kind in {"open", "close", "open_articulated_object"}:
            for obj in benchmark_task_meta.get("objects") or []:
                add(obj)
        return {name for name in names if name}

    @classmethod
    def _playtime_placeable_target_names(cls, inventory: dict[str, Any]) -> set[str]:
        """Secondary placement/support targets.

        MolmoSpaces currently exposes these mostly as ``receptacles``; keep a
        ``placeables`` alias for newer/alternate inventory payloads.
        """
        names: set[str] = set()
        for section in cls._PLAYTIME_PLACEABLE_SECTIONS:
            for entry in inventory.get(section, []) or []:
                if not isinstance(entry, dict):
                    continue
                internal = str(entry.get("internal_name") or "").strip()
                if internal:
                    names.add(internal)
                    names.add(cls._root_object_internal_name(internal))
        return {name for name in names if name}

    _ARTICULATED_OBJECT_CATEGORY_HINTS = {
        "appliance",
        "cabinet",
        "chest",
        "chestofdrawers",
        "closet",
        "dishwasher",
        "door",
        "drawer",
        "dresser",
        "hamper",
        "lid",
        "microwave",
        "microwaveoven",
        "oven",
        "refrigerator",
        "sidetable",
        "toilet",
        "washer",
    }

    @staticmethod
    def _root_object_internal_name(internal_name: str) -> str:
        """Best-effort map from an articulated link/joint name to its root object.

        MolmoSpaces object/link names commonly look like
        ``category_assetuid_instance_link_variant_extra``. Open-task benchmark
        targets and high-level inventory entries usually use the root
        ``..._instance_0_0`` name, while articulation rows may point at a
        link/joint such as ``..._instance_2_0_joint_1``. This helper lets the
        articulation-focus filter accept either representation.
        """
        parts = str(internal_name or "").split("_")
        for i in range(1, len(parts) - 2):
            if (
                parts[i].isdigit()
                and parts[i + 1].isdigit()
                and parts[i + 2].isdigit()
            ):
                return "_".join(parts[: i + 1] + ["0", "0"])
        return str(internal_name or "")

    @classmethod
    def _articulated_playtime_target_names(
        cls,
        inventory: dict[str, Any],
        *,
        benchmark_task_meta: dict[str, Any] | None = None,
    ) -> set[str]:
        """Return internal names allowed when articulated-object focus is on."""
        focus: set[str] = set()

        def add(name: Any) -> None:
            internal = str(name or "").strip()
            if not internal:
                return
            focus.add(internal)
            focus.add(cls._root_object_internal_name(internal))

        # Direct articulation rows and their root objects are the strongest
        # signal that an object exposes an articulated affordance.
        for entry in inventory.get("articulations", []) or []:
            if isinstance(entry, dict):
                add(entry.get("internal_name"))

        # Open/close benchmark metadata names the object whose articulation is
        # being evaluated (e.g. side table, dishwasher, dresser). Keep it
        # eligible even if the live inventory classifies the root as a
        # receptacle rather than an articulation row.
        benchmark_task_meta = benchmark_task_meta or {}
        task_kind = str(
            benchmark_task_meta.get("task_family")
            or (benchmark_task_meta.get("metadata") or {}).get("task_type")
            or ""
        ).strip().lower()
        if task_kind in {"open", "close", "open_articulated_object"}:
            for obj in benchmark_task_meta.get("objects") or []:
                add(obj)

        return {name for name in focus if name}

    @staticmethod
    def _playtime_freeform_proposal_block(
        *,
        enabled: bool,
        allowed_interactions: list[str],
    ) -> str:
        if not enabled:
            return (
                "Freeform proposal mode is OFF. Choose `interaction_type` from "
                "the configured allowed interactions list unless explicitly "
                "permitted by the config."
            )
        examples = ", ".join(allowed_interactions[:8])
        return (
            "Freeform proposal mode is ON. The allowed interactions list is "
            "only a vocabulary prior, not a closed menu. You may invent a "
            "short, concrete, safe interactive action phrase when that better "
            "matches the visible scene and Piaget curiosity objective.\n"
            "- Also write `interaction_family` as the single closest coarse "
            "action family while you generate the proposal. Choose one of the "
            "configured seed interactions when possible (for example `nudge`, "
            "`slide`, `push`, `lift`, `place_on`, `place_in`, `stack`, "
            "`open`). This classification is used for verifier selection and "
            "curiosity scoring.\n"
            "- Write `interaction_type` as the freeform action phrase, e.g. "
            "\"pull the drawer handle a little then release\", \"wiggle the "
            "hamper lid lip while lifting lightly\", \"open the drawer a crack "
            "then push it back\", or \"slide the cup partway then stop\".\n"
            "- Classify by the intended action, not by substrings inside other "
            "words: `slide` / `nudge` a potato is not `open` just because "
            "`slide` contains the letters `lid`.\n"
            "- Avoid standalone infant-like `touch`, `tap`, or `poke` actions "
            "when the scene supports a richer 3-4-year-old style manipulation.\n"
            "- The ranker uses `interaction_family` for affordance keys and "
            "Piaget information gain, so keep the phrase and family "
            "consistent, gentle, bounded, and directly observable.\n"
            f"- Useful seed vocabulary: {examples}."
        )

    def _propose_novel_molmospaces(
        self,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch MolmoSpaces novel proposals to the configured proposer mode."""
        if self._molmospaces_proposer_mode == "open":
            return self._propose_novel_molmospaces_open(
                scene_context,
                skill_context,
            )
        if self._molmospaces_proposer_mode == "playtime":
            return self._propose_novel_molmospaces_playtime(
                scene_context,
                skill_context,
            )
        if self._molmospaces_proposer_mode in {
            "benchmark_order",
            "catalog_sequential",
            "sequential",
        }:
            return self._propose_catalog_molmospaces_sequential(scene_context)
        return self._propose_catalog_molmospaces(
            scene_context,
            skill_context,
        )

    def _propose_novel_molmospaces_playtime(
        self,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Propose a Piaget sensorimotor-style exploratory task."""
        from rats.agents.molmospaces_scene_grounder import ground_inventory
        from rats.loop.molmospaces_utils import extract_molmospaces_scene_inventory

        env = scene_context.get("_env")
        if env is None:
            raise RuntimeError(
                "_propose_novel_molmospaces_playtime requires the live env in "
                "scene_context['_env']; the lifelong loop must inject it."
            )
        inventory = extract_molmospaces_scene_inventory(env)
        self._force_include_anchored_benchmark_targets(inventory)
        grounded = ground_inventory(
            env,
            inventory,
            enabled=self._molmospaces_vlm_grounding,
            use_geometric_visibility_gate=self._molmospaces_geometric_visibility_gate,
            cache=self._molmospaces_grounding_cache,
        )
        visible_map = grounded.get("visible") or {}
        # Look-around retry: when the initial camera angle had nothing
        # useful in frame (v7 iter 1 root cause — 0/115 inventory items
        # visible because the post-house-switch agentview happened to
        # point at a wall), retry the grounder a few times with a fresh
        # image capture each time. Subsequent calls bypass the per-image
        # cache because the fingerprint changes if the env's render is
        # non-deterministic / re-randomized between calls; otherwise we
        # accept the all-False result and let the deterministic fallback
        # kick in (which will request a house-switch via switch_house=True
        # when allowed).
        if visible_map and not any(visible_map.values()):
            for attempt in range(1, 4):
                logger.info(
                    "  playtime grounder: 0/%d visible on attempt %d; retrying",
                    len(visible_map), attempt,
                )
                regrounded = ground_inventory(
                    env,
                    inventory,
                    enabled=self._molmospaces_vlm_grounding,
                    use_geometric_visibility_gate=self._molmospaces_geometric_visibility_gate,
                    cache=None,  # bypass cache so the VLM re-evaluates fresh frames
                )
                regrounded_visible = regrounded.get("visible") or {}
                if any(regrounded_visible.values()):
                    logger.info(
                        "  playtime grounder: retry %d found %d visible",
                        attempt,
                        sum(1 for v in regrounded_visible.values() if v),
                    )
                    grounded = regrounded
                    visible_map = regrounded_visible
                    break
        display_to_internal: dict[str, str] = {
            phrase: internal
            for internal, phrase in (grounded.get("display_names") or {}).items()
            if phrase
        }

        # Build the physical unsuitability check (oversize + reach) before
        # rendering the prompt inventory. The proposer should only see
        # visible/reachable candidates, rather than a full simulator inventory
        # plus an "excluded items" appendix.
        try:
            max_reach_m = float(
                self._molmospaces_playtime_config.get(
                    "max_reach_m", self._DEFAULT_PLAYTIME_MAX_REACH_M,
                )
            )
        except (TypeError, ValueError):
            max_reach_m = self._DEFAULT_PLAYTIME_MAX_REACH_M
        robot_base_xyz = self._get_robot_base_xyz(env)
        # The bridge's sampler already placed the robot near a specific
        # anchored target (pinned_pickup_obj_name / place_receptacle_name /
        # joint_name) when the house was sampled. RATS-side unsuitable_check
        # would otherwise reject items by distance even though the bridge
        # guarantees the anchored ones are reachable. Whitelist them so the
        # reach filter doesn't false-reject items the sampler is staked on.
        anchored = inventory.get("anchored_target") or {}
        anchored_internal_names: set[str] = {
            str(name)
            for name in (
                anchored.get("pickup_obj_name"),
                anchored.get("place_receptacle_name"),
                anchored.get("joint_name"),
            )
            if name
        }
        base_unsuitable_check = self._build_playtime_unsuitable_check(
            inventory, robot_base_xyz, max_reach_m,
        )

        def unsuitable_check(internal: str, *, role: str = "target") -> str:
            if internal in anchored_internal_names:
                # Bridge-anchored — skip the reach gate.
                return ""
            return base_unsuitable_check(internal, role=role)
        visible_display_names: set[str] = {
            phrase
            for internal, phrase in (grounded.get("display_names") or {}).items()
            if phrase and bool(visible_map.get(internal, True))
        }
        # Force-include the bridge-anchored items in the visible set: bridge
        # guarantees reachability (and typically visibility — robot was
        # placed adjacent to the anchor), so the grounder VLM marking it
        # invisible (e.g. wrist-only visibility, or grounder false-negative)
        # should not exclude it from the candidate pool.
        for anchored_internal in anchored_internal_names:
            anchored_phrase = (grounded.get("display_names") or {}).get(anchored_internal)
            if anchored_phrase:
                grounded.setdefault("visible", {})[anchored_internal] = True
                visible_map[anchored_internal] = True
                visible_display_names.add(anchored_phrase)
        prompt_inventory = self._annotate_inventory_for_prompt(
            inventory,
            grounded,
            visible_only=True,
            unsuitable_check=unsuitable_check,
        )
        benchmark_fallback_meta = self._current_benchmark_task_meta(scene_context)
        focus_articulated_objects = bool(
            self._molmospaces_playtime_config.get("focus_articulated_objects", False)
        )
        articulated_focus_names: set[str] | None = None
        articulated_focus_display_names: list[str] = []
        taskable_target_names = self._playtime_taskable_target_names(
            inventory,
            benchmark_task_meta=benchmark_fallback_meta,
        )
        placeable_target_names = self._playtime_placeable_target_names(inventory)
        internal_to_display = {v: k for k, v in display_to_internal.items()}
        taskable_target_display_names = sorted(
            display
            for internal, display in (
                (name, internal_to_display.get(name, ""))
                for name in taskable_target_names
            )
            if display
            and display in visible_display_names
            and not unsuitable_check(internal, role="target")
        )
        # Visible+reachable receptacles/placeables, used by the no_llm
        # random baseline to supply a valid secondary for place verbs.
        # (Stage-A never narrows or re-grounds for no-context modes, so
        # this stays valid through to the candidate-generation site.)
        placeable_target_display_names = sorted(
            display
            for internal, display in (
                (name, internal_to_display.get(name, ""))
                for name in placeable_target_names
            )
            if display
            and display in visible_display_names
            and not unsuitable_check(internal, role="secondary")
        )
        if focus_articulated_objects:
            articulated_focus_names = self._articulated_playtime_target_names(
                inventory,
                benchmark_task_meta=benchmark_fallback_meta,
            )
            articulated_focus_names &= taskable_target_names
            articulated_focus_display_names = sorted(
                display
                for internal, display in (
                    (name, internal_to_display.get(name, ""))
                    for name in articulated_focus_names
                )
                if display
                and display in visible_display_names
                and not unsuitable_check(internal, role="target")
            )

        allowed_interactions = [
            str(x).strip().lower()
            for x in (
                self._molmospaces_playtime_config.get("allowed_interactions")
                or self._DEFAULT_PLAYTIME_INTERACTIONS
            )
            if str(x).strip()
        ]

        # Curiosity / proposer mode. Read EARLY (before Stage-A and the
        # prompt build) because the two history-free baselines change
        # both. See the K-candidate winner-selection block below for the
        # full mode reference.
        #   "formula" / "llm" / "random" — LLM proposer WITH task history
        #       in the prompt + Stage-A history-driven target anchoring;
        #       differ only in winner selection (argmax-formula /
        #       argmax-llm / uniform-random over the K-candidate pool).
        #   "llm_nocontext" — LLM proposer with ALL history stripped (no
        #       {task_history}, no verb-coverage / curriculum hints, no
        #       Stage-A anchoring); K candidates, uniform-random winner.
        #       Isolates "history vs no-history" against "random" mode.
        #   "no_llm" — no LLM call at all; uniform-random (verb × object)
        #       drawn from the same visible+reachable+taskable pool. Floor
        #       baseline. Also history-free (no Stage-A, no prompt).
        curiosity_mode = str(
            self._molmospaces_playtime_config.get("curiosity_mode", "formula")
        ).lower()
        if curiosity_mode not in (
            "formula", "llm", "random", "llm_nocontext", "no_llm",
        ):
            curiosity_mode = "formula"
        # History-free baselines: drop prompt history AND the Stage-A
        # history-driven target anchoring (Stage-A keys off per-object
        # attempt counts, which is exactly the "history" we're ablating).
        no_context = curiosity_mode in ("llm_nocontext", "no_llm")
        use_llm = curiosity_mode != "no_llm"
        random_pick = curiosity_mode in ("random", "llm_nocontext", "no_llm")

        # Stage A: target selection by (target, verb)-joint curiosity.
        # When enabled, score every (target × verb) pair in the visible
        # taskable inventory via LIBERO's score_candidate, pick the
        # max-over-verbs argmax target, and have the bridge anchor the
        # robot adjacent to it BEFORE the LLM picks a verb + secondary.
        # Without this, the LLM proposed against the bridge's round-robin
        # pin (often a different, less-curious target), and the auto-
        # anchor in _rebind_molmospaces_playtime only fired afterwards
        # so the proposer never saw the post-anchor scene.
        two_stage_target = bool(
            self._molmospaces_playtime_config.get(
                "two_stage_target_selection", True
            )
        ) and not no_context
        forced_target_internal: str | None = None
        forced_target_display: str | None = None
        stage_a_per_target_scores: dict[str, float] = {}
        if (
            two_stage_target
            and isinstance(skill_context, dict)
            and skill_context.get("_skill_lookup") is not None
            and skill_context.get("_object_skill_counts") is not None
        ):
            if focus_articulated_objects and articulated_focus_names:
                stage_a_pool_internal = sorted(articulated_focus_names)
            else:
                stage_a_pool_internal = sorted(taskable_target_names)
            stage_a_pool_internal = [
                internal for internal in stage_a_pool_internal
                if internal_to_display.get(internal, "") in visible_display_names
                and not unsuitable_check(internal, role="target")
            ]
            if stage_a_pool_internal:
                forced_target_internal, stage_a_per_target_scores = (
                    self._select_target_by_curiosity_internal(
                        taskable_target_internal=stage_a_pool_internal,
                        allowed_verbs=allowed_interactions,
                        skill_lookup=skill_context["_skill_lookup"],
                        history_counts=skill_context["_object_skill_counts"],
                        task_history=self._task_history,
                    )
                )
                if forced_target_internal:
                    forced_target_display = internal_to_display.get(
                        forced_target_internal
                    )
                    current_anchor = (anchored or {}).get("pickup_obj_name")
                    if current_anchor != forced_target_internal:
                        anchor_fn = getattr(env, "anchor_to_pickup", None)
                        if callable(anchor_fn):
                            try:
                                anchor_fn(forced_target_internal)
                                logger.info(
                                    "  playtime stage-A: anchored '%s' "
                                    "(was '%s', score=%.4f)",
                                    forced_target_internal,
                                    current_anchor,
                                    stage_a_per_target_scores.get(
                                        forced_target_internal, 0.0
                                    ),
                                )
                                anchored = {
                                    **(anchored or {}),
                                    "pickup_obj_name": forced_target_internal,
                                }
                                # Re-ground from the new pose: images shown to the LLM are post-anchor, but visibility/reach state above was pre-anchor.
                                regrounded = ground_inventory(
                                    env,
                                    inventory,
                                    enabled=self._molmospaces_vlm_grounding,
                                    use_geometric_visibility_gate=self._molmospaces_geometric_visibility_gate,
                                    cache=None,
                                )
                                grounded = regrounded
                                visible_map = grounded.get("visible") or {}
                                display_to_internal = {
                                    phrase: internal
                                    for internal, phrase in (
                                        grounded.get("display_names") or {}
                                    ).items()
                                    if phrase
                                }
                                internal_to_display = {
                                    v: k for k, v in display_to_internal.items()
                                }
                                visible_display_names = {
                                    phrase
                                    for internal_, phrase in (
                                        grounded.get("display_names") or {}
                                    ).items()
                                    if phrase
                                    and bool(visible_map.get(internal_, True))
                                }
                                anchored_internal_names = {
                                    str(name)
                                    for name in (
                                        anchored.get("pickup_obj_name"),
                                        anchored.get("place_receptacle_name"),
                                        anchored.get("joint_name"),
                                    )
                                    if name
                                }
                                for anchored_internal in anchored_internal_names:
                                    p = (grounded.get("display_names") or {}).get(
                                        anchored_internal
                                    )
                                    if p:
                                        grounded.setdefault("visible", {})[
                                            anchored_internal
                                        ] = True
                                        visible_map[anchored_internal] = True
                                        visible_display_names.add(p)
                                # Reassigning rebinds the names the unsuitable_check closure resolves at call time.
                                robot_base_xyz = self._get_robot_base_xyz(env)
                                base_unsuitable_check = (
                                    self._build_playtime_unsuitable_check(
                                        inventory, robot_base_xyz, max_reach_m,
                                    )
                                )
                                forced_target_display = internal_to_display.get(
                                    forced_target_internal
                                )
                                prompt_inventory = self._annotate_inventory_for_prompt(
                                    inventory,
                                    grounded,
                                    visible_only=True,
                                    unsuitable_check=unsuitable_check,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "  playtime stage-A anchor_to_pickup "
                                    "failed: %s", exc,
                                )
                                forced_target_internal = None
                                forced_target_display = None
            # Narrow target whitelists to the single Stage-A pick so
            # both the prompt's HARD RULE block and the validator
            # naturally reject candidates that drift to a different
            # target. Receptacle/secondary lists are unchanged.
            if forced_target_internal:
                taskable_target_names = {forced_target_internal}
                taskable_target_display_names = (
                    [forced_target_display] if forced_target_display else []
                )
                if focus_articulated_objects and articulated_focus_names is not None:
                    if forced_target_internal in articulated_focus_names:
                        articulated_focus_names = {forced_target_internal}
                        articulated_focus_display_names = (
                            [forced_target_display]
                            if forced_target_display
                            else []
                        )

        allow_freeform = bool(
            self._molmospaces_playtime_config.get("allow_freeform_interactions", True)
        )
        requested_freeform_proposal = bool(
            self._molmospaces_playtime_config.get("freeform_proposal", False)
        )
        verify_with_vlm_only = bool(
            self._molmospaces_playtime_config.get("verify_with_vlm_only", False)
        )
        stateful_checker_enabled = bool(
            self._molmospaces_playtime_config.get("stateful_checker_enabled", False)
        )
        freeform_proposal = requested_freeform_proposal and (
            verify_with_vlm_only or stateful_checker_enabled
        )
        if requested_freeform_proposal and not (verify_with_vlm_only or stateful_checker_enabled):
            logger.warning(
                "  playtime freeform_proposal requested but verify_with_vlm_only "
                "and stateful_checker_enabled are false; disabling freeform proposal "
                "for grounded verifier compatibility",
            )
        if freeform_proposal:
            allow_freeform = True
        freeform_block = self._playtime_freeform_proposal_block(
            enabled=freeform_proposal,
            allowed_interactions=allowed_interactions,
        )
        safe_rules = str(
            self._molmospaces_playtime_config.get(
                "safe_interaction_rules",
                "Use gentle, reversible, bounded actions. Avoid damage, throwing, "
                "pushing objects off support surfaces, and mobile-base navigation.",
            )
        )
        switches_left = (
            self._molmospaces_house_switch_max_per_run
            - self._molmospaces_house_switches_used
        )
        if self._molmospaces_allow_house_switching and switches_left > 0:
            house_switch_block = (
                f"House-switching is ENABLED. You may set switch_house=true "
                f"if no useful safe play task remains. {switches_left} "
                f"switches remain in this run."
            )
        else:
            house_switch_block = (
                "House-switching is DISABLED for this run; do NOT set "
                "switch_house=true."
            )

        # PlaytimeMemory removed in the unified --play-mode refactor. The
        # rewritten prompt no longer has a {playtime_memory} placeholder.
        # Strip the (now-unused) playtime keys from the prompt skill_context
        # if the loop somehow still injects them, so json.dumps doesn't
        # blow up on the live ``PlaytimeMemory`` object.
        skill_context_for_prompt: Any = skill_context
        if isinstance(skill_context, dict):
            skill_context_for_prompt = {
                k: v for k, v in skill_context.items()
                if k not in (
                    "playtime_memory",
                    "sensorimotor_memory",
                    "_playtime_memory_obj",
                    # LIBERO-aligned curiosity scoring inputs (callables /
                    # tuples-keyed dict) — not serializable as prompt text
                    # and irrelevant to the LLM's task proposal anyway.
                    "_skill_lookup",
                    "_object_skill_counts",
                )
            }

        prompt_template = Path(
            "rats/prompts/task_proposer_molmospaces_playtime_sensorimotor.txt"
        ).read_text()
        if no_context:
            # History-free baseline: blank every history-derived block so
            # each iteration looks like a fresh iteration-1 to the LLM
            # ("free memory every time"). The prompt's "review RECENT TASK
            # HISTORY" instructions become no-ops when history is empty.
            history_str = "None yet."
            verb_coverage_hint = ""
            playtime_curriculum_hint = ""
        else:
            history_str = "None yet." if not self._task_history else json.dumps(
                self._task_history[-10:], indent=2
            )
            verb_coverage_hint = self._verb_coverage_hint(
                self._task_history,
                allowed_interactions,
            )
            playtime_curriculum_hint = self._playtime_curriculum_hint(
                self._task_history,
                skill_context.get("_playtime_memory_obj")
                if isinstance(skill_context, dict) else None,
            )
        visibility_instruction_block = (
            "You are also given the live agentview and wrist images of the "
            "scene. The text inventory below has already been filtered to "
            "items that the current cameras marked visible and that are within "
            "the robot's playtime reach filter for their role."
        )
        visibility_hard_rule_block = (
            "- Display names MUST come from the filtered inventory below. "
            "Anything not listed was hidden, out of reach, or physically "
            "unsuitable, and must not be used as a target or secondary object."
        )
        user_prompt = (
            prompt_template
            .replace("{allowed_interactions}", json.dumps(allowed_interactions))
            .replace("{freeform_proposal_block}", freeform_block)
            .replace("{safe_interaction_rules}", safe_rules)
            .replace("{visibility_instruction_block}", visibility_instruction_block)
            .replace("{visibility_hard_rule_block}", visibility_hard_rule_block)
            .replace("{inventory}", json.dumps(prompt_inventory, indent=2))
            .replace("{skill_context}", json.dumps(skill_context_for_prompt, indent=2))
            .replace("{playtime_curriculum_hint}", playtime_curriculum_hint)
            .replace("{verb_coverage_hint}", verb_coverage_hint or "(no signal yet)")
            .replace("{task_history}", history_str)
            .replace("{house_switch_block}", house_switch_block)
        )
        if self._include_eval_task_context:
            if self._eval_task_context_block is None:
                benchmark_dir = (
                    self._molmospaces_playtime_config.get(
                        "eval_task_context_benchmark_dir"
                    )
                    or self._molmospaces_playtime_config.get("eval_benchmark_dir")
                )
                try:
                    from rats.loop.molmospaces_utils import (
                        format_molmospaces_eval_tasks_block,
                    )
                    self._eval_task_context_block = (
                        format_molmospaces_eval_tasks_block(benchmark_dir)
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to build MolmoSpaces eval-task context: %s",
                        exc,
                    )
                    self._eval_task_context_block = ""
            if self._eval_task_context_block:
                user_prompt = f"{self._eval_task_context_block}\n\n{user_prompt}"
        # Anchored-target hint. The bridge's sampler placed the robot
        # adjacent to a specific pickable/articulation when the house was
        # sampled — that item is guaranteed reachable for this iter. Tell
        # the LLM which item it is and that picking it is the safest bet.
        # (Not a hard rule — the LLM can still propose a different visible
        # taskable target if it has good reason to; this just prevents the
        # LLM from missing the bridge's free hint.)
        anchored_lines: list[str] = []
        anchor_pickup = anchored.get("pickup_obj_name")
        anchor_place = anchored.get("place_receptacle_name")
        anchor_joint = anchored.get("joint_name")
        display_names_map = grounded.get("display_names") or {}
        if anchor_pickup:
            anchored_lines.append(
                f"  - pickup target: {anchor_pickup}"
                + (f"  (display: {display_names_map[anchor_pickup]!r})" if display_names_map.get(anchor_pickup) else "")
            )
        if anchor_place:
            anchored_lines.append(
                f"  - placement receptacle: {anchor_place}"
                + (f"  (display: {display_names_map[anchor_place]!r})" if display_names_map.get(anchor_place) else "")
            )
        if anchor_joint:
            anchored_lines.append(
                f"  - articulated joint: {anchor_joint}"
                + (f"  (display: {display_names_map[anchor_joint]!r})" if display_names_map.get(anchor_joint) else "")
            )
        if anchored_lines:
            user_prompt += (
                "\n\n# Anchored target (bridge sampler-pinned; reach-guaranteed)\n"
                "The MolmoSpaces bridge placed the robot adjacent to the items "
                "below when this house was sampled. Reach is guaranteed; the rest "
                "of the inventory may be physically out of arm range. Strongly "
                "prefer one of these as your `target_object_display_name` (and "
                "the receptacle/joint variant as `secondary_object_display_name` "
                "when relevant) unless you have a clear reason to pick another "
                "visible item.\n"
                + "\n".join(anchored_lines)
            )
        if taskable_target_display_names:
            user_prompt += (
                "\n\n# Taskable target restriction (HARD RULE)\n"
                "The main `target_object_display_name` MUST be a free-bodied "
                "pickable object or an explicit articulated object. Choose it "
                "only from this taskable target list:\n"
                + "\n".join(
                    f"  - {name}" for name in taskable_target_display_names
                )
                + "\nReceptacles/placeables such as counters, bowls, drawers "
                "without joints, pots, appliances, and tables may be used only "
                "as `secondary_object_display_name` for place/support tasks, "
                "not as the main target."
            )
        else:
            user_prompt += (
                "\n\n# Taskable target restriction (HARD RULE)\n"
                "No pickable or explicit articulated main target was resolved "
                "from the current inventory. Do not invent a receptacle or "
                "static fixture as a main target."
            )
        if focus_articulated_objects:
            if articulated_focus_display_names:
                focus_target_label = "visible/reachable articulated-object targets"
                focus_block = (
                    "\n\n# Articulated-object focus (HARD RULE)\n"
                    "This config has molmospaces.playtime.focus_articulated_objects=true. "
                    "The main `target_object_display_name` MUST be one of these "
                    f"{focus_target_label}:\n"
                    + "\n".join(
                        f"  - {name}" for name in articulated_focus_display_names
                    )
                    + "\nYou may use an ordinary inventory object/surface only as "
                    "`secondary_object_display_name` when needed."
                )
            else:
                missing_target_label = "visible/reachable articulated-object target"
                focus_block = (
                    "\n\n# Articulated-object focus (HARD RULE)\n"
                    "This config has molmospaces.playtime.focus_articulated_objects=true, "
                    f"but no {missing_target_label} was resolved. Do not "
                    "invent non-articulated targets."
                )
            user_prompt += focus_block
        env_verifier_feedback = str(
            scene_context.get("environment_verifier_feedback") or ""
        ).strip()
        if env_verifier_feedback:
            user_prompt += (
                "\n\n# Environment verifier rejection (HARD FEEDBACK)\n"
                f"{env_verifier_feedback}\n"
                "Do not repeat the rejected target. Choose a different object "
                "that is clearly visible in the current agentview/wrist images."
            )
        # Capture the live agentview + wrist images so the proposer LLM
        # can self-verify visibility instead of trusting only the text
        # inventory. The grounder's visibility flags can still be
        # wrong (esp. on edge-of-frame items); a multimodal proposer
        # can sanity-check before committing.
        proposer_images: list[str] = []
        if use_llm:
            try:
                from rats.agents.molmospaces_scene_grounder import _capture_image  # type: ignore[attr-defined]
                for kind in ("agent", "wrist"):
                    img = _capture_image(env, kind=kind)
                    if img:
                        proposer_images.append(img)
            except Exception as exc:
                logger.debug("playtime proposer image capture failed: %s", exc)
        system_prompt = (
            "You are a curiosity-driven task proposer for a robot learning system. "
            "Propose a NOVEL manipulation task. Respond only in valid JSON."
        )
        trace: dict[str, Any] = {
            "proposer": "molmospaces_playtime",
            "developmental_stage": self._molmospaces_playtime_config.get(
                "developmental_stage", "sensorimotor"
            ),
            "curiosity": self._molmospaces_curiosity,
            "allowed_interactions": allowed_interactions,
            "allow_freeform_interactions": allow_freeform,
            "freeform_proposal": freeform_proposal,
            "requested_freeform_proposal": requested_freeform_proposal,
            "include_eval_task_context": self._include_eval_task_context,
            "verify_with_vlm_only": verify_with_vlm_only,
            "focus_articulated_objects": focus_articulated_objects,
            "inventory_visibility_gate": True,
            "articulated_focus_names": sorted(articulated_focus_names or []),
            "articulated_focus_display_names": articulated_focus_display_names,
            "taskable_target_names": sorted(taskable_target_names),
            "taskable_target_display_names": taskable_target_display_names,
            "placeable_target_names": sorted(placeable_target_names),
            "house_switches_left": switches_left,
            "two_stage_target_selection": two_stage_target,
            "stage_a_forced_target_internal": forced_target_internal,
            "stage_a_forced_target_display": forced_target_display,
            "stage_a_per_target_scores": stage_a_per_target_scores,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "inventory": inventory,
            "grounded_inventory": grounded,
            "prompt_inventory": prompt_inventory,
            "verb_coverage_hint": verb_coverage_hint,
            "playtime_curriculum_hint": playtime_curriculum_hint,
            "history": self._task_history[-10:],
            "attempts": [],
        }
        if benchmark_fallback_meta:
            trace["benchmark_fallback_task"] = {
                "canonical_task_id": benchmark_fallback_meta.get("canonical_id")
                or benchmark_fallback_meta.get("canonical_task_id"),
                "task_family": benchmark_fallback_meta.get("task_family"),
                "language": benchmark_fallback_meta.get("language"),
                "objects": benchmark_fallback_meta.get("objects"),
            }

        # Sample-K-and-rank: ask the LLM for K diverse candidates in a
        # single call (input tokens — inventory, memory, images — are paid
        # once; only output tokens scale with K). Each candidate is
        # validated, scored with the Piaget curiosity loss, and the
        # argmax wins. K==1 falls back to the original single-proposal
        # flow for backwards compatibility.
        sample_k_raw = self._molmospaces_playtime_config.get(
            "proposal_count",
            self._molmospaces_playtime_config.get("sample_k", 5),
        )
        try:
            sample_k = int(sample_k_raw)
        except (TypeError, ValueError):
            sample_k = 5
        sample_k = max(1, min(8, sample_k))
        trace["sample_k"] = sample_k
        trace["proposal_count"] = sample_k

        trace["robot_base_xyz"] = (
            list(robot_base_xyz) if robot_base_xyz is not None else None
        )
        trace["max_reach_m"] = max_reach_m

        # Keep the trace auditable, but do not append excluded objects to the
        # prompt. The prompt inventory above is the filtered allowed set.
        unsuitable_lines: list[str] = []
        for internal, display in (
            (i, d) for d, i in display_to_internal.items() if i
        ):
            why = unsuitable_check(internal, role="target")
            if why:
                unsuitable_lines.append(f"  - {display}: {why}")
        trace["unsuitable_targets"] = unsuitable_lines
        trace["user_prompt"] = user_prompt

        if bool(scene_context.get("_force_playtime_deterministic_fallback")):
            logger.warning(
                "  playtime proposer forced to deterministic fallback after "
                "environment-verifier rejections",
            )
            fb_raw = self._fallback_playtime_proposal(
                inventory,
                display_to_internal,
                allowed_interactions,
                benchmark_task_meta=benchmark_fallback_meta,
                visible_display_names=visible_display_names,
                unsuitable_check=unsuitable_check,
                target_allowlist=(
                    articulated_focus_names
                    if articulated_focus_names is not None
                    else taskable_target_names
                ),
                allow_switch_house=(
                    self._molmospaces_allow_house_switching
                    and switches_left > 0
                ),
            )
            proposal = self._build_playtime_proposal_dict(
                fb_raw,
                inventory=inventory,
                display_to_internal=display_to_internal,
                scene_context=scene_context,
                freeform_proposal=freeform_proposal,
            )
            proposal["playtime_curiosity_score"] = 0.0
            proposal["curiosity_score"] = 0.25
            proposal["novelty"] = proposal.get("novelty_score", 0.4)
            proposal["learnability"] = 0.5
            trace["attempts"].append({
                "kind": "forced_environment_verifier_fallback",
                "result": fb_raw,
                "evaluated": [{
                    "idx": 0,
                    "veto": "",
                    "score": 0.0,
                    "is_switch_house": proposal.get("_request_house_switch", False),
                }],
            })
            proposal["_proposer_trace"] = trace
            self.last_proposal_trace = trace
            return proposal

        # Curiosity / proposer mode was resolved early (see the block right
        # after allowed_interactions). Selection reference:
        #   "formula" — programmatic novelty*frontier argmax over K (WITH
        #               history).
        #   "llm"     — LLM-emitted per-candidate novelty/frontier argmax
        #               over K (WITH history).
        #   "random"  — uniform-random pick over K (WITH history). Scoring
        #               is still computed/logged for the audit trail.
        #   "llm_nocontext" — uniform-random pick over K, but the proposer
        #               prompt has all history stripped (history ablation).
        #   "no_llm"  — no LLM call; K random (verb × object) candidates
        #               drawn from the visible+reachable+taskable pool,
        #               uniform-random pick. Floor baseline.
        if sample_k > 1:
            user_prompt = (
                user_prompt
                + f"\n\n# Sample {sample_k} candidates\n"
                f"Instead of returning ONE proposal, return {sample_k} "
                "DIVERSE candidate tasks under a top-level \"candidates\" "
                "list. Each candidate must satisfy the schema above and "
                "the hard rules. Diversify across at least one of: "
                "target object, secondary object/receptacle, interaction "
                "verb/freeform action phrase, sensorimotor motive. Avoid near-duplicate "
                "candidates and avoid an exact duplicate of any entry in "
                "the recent task history. Output format:\n"
                "{\n  \"candidates\": [\n    { ...candidate 1... },\n"
                "    { ...candidate 2... },\n    ...\n  ]\n}"
            )
            if curiosity_mode == "llm":
                # Mirror the LIBERO score-with-llm prompt block (see
                # agents/task_proposer.py::propose_novel_candidates).
                user_prompt += (
                    "\n\nEach candidate ADDITIONALLY emits:\n"
                    "    `llm_novelty_score` (float in [0, 1]),\n"
                    "    `llm_frontier_score` (float in [0, 1]),\n"
                    "    `llm_rationale` (short string).\n"
                    "Definitions (use these, do NOT remap to easy/medium/hard):\n"
                    "  novelty: high when this (object, verb) interaction\n"
                    "    has rarely been attempted by this agent before\n"
                    "    (look at RECENT TASK HISTORY).\n"
                    "  frontier: high when the task is near the robot's\n"
                    "    current competence boundary — NOT already\n"
                    "    mastered, NOT obviously impossible, learnable\n"
                    "    with 1-2 new or refined skills.\n"
                )
            trace["user_prompt"] = user_prompt

        piaget_weights = self._molmospaces_playtime_config.get(
            "piaget_curiosity_weights",
        )
        playtime_memory_obj = (
            skill_context.get("_playtime_memory_obj")
            if isinstance(skill_context, dict) else None
        )
        # LIBERO-aligned scoring inputs. ``skill_lookup`` resolves a skill
        # name to a Wilson lower-bound competence; ``history_counts``
        # maps (object, skill) → attempt count. When BOTH are passed in
        # via skill_context, _compute_playtime_curiosity falls back to
        # LIBERO's ``novelty * frontier`` base score (see
        # agents/curiosity_scoring.score_candidate). Optional: legacy
        # callers without these still get the original weighted_sum.
        aligned_skill_lookup = (
            skill_context.get("_skill_lookup")
            if isinstance(skill_context, dict) else None
        )
        aligned_history_counts = (
            skill_context.get("_object_skill_counts")
            if isinstance(skill_context, dict) else None
        )

        def evaluate_candidates(raw_response: Any) -> list[dict[str, Any]]:
            """Validate, build, and score each candidate from one LLM call.

            Returns a list of dicts, one per candidate, each with the
            built proposal (or None on veto), the Piaget curiosity
            components, and the veto reason. Candidates are NOT yet
            sorted; the caller picks argmax.
            """
            extracted: list[dict[str, Any]] = []
            if isinstance(raw_response, dict):
                cands = raw_response.get("candidates")
                if isinstance(cands, list) and cands:
                    extracted = [c for c in cands if isinstance(c, dict)]
                else:
                    # LLM ignored the K instruction and returned a single
                    # proposal — fall back to treating it as 1 candidate.
                    extracted = [raw_response]
            elif isinstance(raw_response, list):
                extracted = [c for c in raw_response if isinstance(c, dict)]
            evaluated: list[dict[str, Any]] = []
            for idx, raw in enumerate(extracted):
                veto_reason = self._validate_playtime_proposal(
                    raw,
                    display_to_internal=display_to_internal,
                    allowed_interactions=allowed_interactions,
                    allow_freeform=allow_freeform,
                    freeform_proposal=freeform_proposal,
                    allow_house_switch=(
                        self._molmospaces_allow_house_switching
                        and switches_left > 0
                    ),
                    visible_display_names=visible_display_names,
                    unsuitable_check=unsuitable_check,
                    articulated_focus_names=articulated_focus_names,
                    taskable_target_names=taskable_target_names,
                    placeable_target_names=placeable_target_names,
                )
                if veto_reason:
                    evaluated.append({
                        "idx": idx,
                        "raw": raw,
                        "veto": veto_reason,
                        "proposal": None,
                        "score": 0.0,
                        "piaget": None,
                    })
                    continue
                p = self._build_playtime_proposal_dict(
                    raw,
                    inventory=inventory,
                    display_to_internal=display_to_internal,
                    scene_context=scene_context,
                    freeform_proposal=freeform_proposal,
                )
                # Switch-house responses cannot be Piaget-scored (no
                # target object / no affordance card to query). Give
                # them a deterministic 0 so any valid play wins, but
                # keep the candidate so it can still be picked when
                # every other candidate failed validation.
                if p.get("_request_house_switch"):
                    evaluated.append({
                        "idx": idx,
                        "raw": raw,
                        "veto": "",
                        "proposal": p,
                        "score": 0.0,
                        "piaget": None,
                        "is_switch_house": True,
                    })
                    continue
                # When LLM scoring is enabled, harvest the per-candidate
                # ``llm_novelty_score`` / ``llm_frontier_score`` /
                # ``llm_rationale`` fields off the raw LLM response so
                # _compute_playtime_curiosity can pass them to
                # score_candidate(mode="llm"). Invalid / out-of-range
                # values are clipped to [0, 1] downstream.
                if curiosity_mode == "llm" and isinstance(raw, dict):
                    for k in ("llm_novelty_score", "llm_frontier_score", "llm_rationale"):
                        if k in raw and k not in p:
                            p[k] = raw[k]
                piaget = self._compute_playtime_curiosity(
                    p,
                    playtime_memory_obj,
                    self._task_history,
                    weights=piaget_weights
                    if isinstance(piaget_weights, dict) else None,
                    skill_lookup=aligned_skill_lookup,
                    history_counts=aligned_history_counts,
                    mode=curiosity_mode,
                )
                evaluated.append({
                    "idx": idx,
                    "raw": raw,
                    "veto": "",
                    "proposal": p,
                    "score": piaget["playtime_curiosity_score"],
                    "piaget": piaget,
                })
            return evaluated

        if curiosity_mode == "no_llm":
            # Floor baseline: no LLM. Synthesize K random (verb × object)
            # candidates from the visible+reachable+taskable pool and feed
            # them through the SAME validate/build/score path as the LLM
            # candidates, so the only difference vs the LLM modes is "who
            # picked the (verb, object)".
            import random as _random
            rng = _random
            no_llm_raws = [
                self._random_playtime_raw_proposal(
                    allowed_interactions=allowed_interactions,
                    taskable_display_names=taskable_target_display_names,
                    placeable_display_names=placeable_target_display_names,
                    rng=rng,
                    allow_switch_house=(
                        self._molmospaces_allow_house_switching
                        and switches_left > 0
                    ),
                )
                for _ in range(sample_k)
            ]
            result: Any = {"candidates": no_llm_raws}
        else:
            try:
                result = query_llm_json(
                    system_prompt, user_prompt,
                    images=proposer_images or None,
                )
            except Exception as exc:
                logger.warning(
                    "  playtime proposer LLM call failed after provider "
                    "fallbacks; skipping LLM proposal for this attempt: %s",
                    exc,
                )
                result = {"candidates": []}
        evaluated = evaluate_candidates(result)
        trace["attempts"].append({
            "kind": "no_llm_random" if curiosity_mode == "no_llm" else "llm",
            "result": result,
            "evaluated": [
                {
                    "idx": c["idx"],
                    "veto": c["veto"],
                    "score": c["score"],
                    "is_switch_house": c.get("is_switch_house", False),
                    "components": (c.get("piaget") or {}).get(
                        "playtime_curiosity_components"
                    )
                    if c.get("piaget")
                    else {
                        "affordance_information_gain": (c.get("piaget") or {}).get(
                            "affordance_information_gain"
                        ),
                        "variation_bonus": (c.get("piaget") or {}).get(
                            "variation_bonus"
                        ),
                    } if c.get("piaget") else None,
                }
                for c in evaluated
            ],
        })
        valid = [c for c in evaluated if c["proposal"] is not None]

        if not valid and use_llm:
            # First call returned nothing usable — re-prompt once with
            # the dominant veto reason. Mirrors the prior single-shot
            # retry behavior. Skipped for no_llm (there is no LLM to
            # re-prompt; the deterministic fallback below handles it).
            veto_summary = "; ".join(
                sorted({c["veto"] for c in evaluated if c["veto"]})
            ) or "no usable candidate returned"
            retry_prompt = (
                user_prompt
                + f"\n\nPREVIOUS PROPOSAL(S) REJECTED:\n  reasons: {veto_summary}\n"
                + (
                    "Re-propose using only display names from the filtered "
                    "visible/reachable inventory and obey the taskable target "
                    "list plus articulated-object focus list. "
                )
                + "JSON only."
            )
            try:
                result = query_llm_json(
                    system_prompt, retry_prompt,
                    images=proposer_images or None,
                )
            except Exception as exc:
                logger.warning(
                    "  playtime proposer retry failed after provider "
                    "fallbacks; skipping retry for this attempt: %s",
                    exc,
                )
                result = {"candidates": []}
            evaluated_retry = evaluate_candidates(result)
            trace["retry_prompt"] = retry_prompt
            trace["attempts"].append({
                "kind": "llm_retry",
                "result": result,
                "evaluated": [
                    {
                        "idx": c["idx"],
                        "veto": c["veto"],
                        "score": c["score"],
                        "is_switch_house": c.get("is_switch_house", False),
                    }
                    for c in evaluated_retry
                ],
            })
            valid = [c for c in evaluated_retry if c["proposal"] is not None]
            evaluated = evaluated_retry

        if not valid:
            logger.warning(
                "  playtime proposer produced no usable candidate; "
                "using deterministic fallback",
            )
            fb_raw = self._fallback_playtime_proposal(
                inventory,
                display_to_internal,
                allowed_interactions,
                benchmark_task_meta=benchmark_fallback_meta,
                visible_display_names=visible_display_names,
                unsuitable_check=unsuitable_check,
                target_allowlist=(
                    articulated_focus_names
                    if articulated_focus_names is not None
                    else taskable_target_names
                ),
                allow_switch_house=(
                    self._molmospaces_allow_house_switching
                    and switches_left > 0
                ),
            )
            fb_proposal = self._build_playtime_proposal_dict(
                fb_raw,
                inventory=inventory,
                display_to_internal=display_to_internal,
                scene_context=scene_context,
                freeform_proposal=freeform_proposal,
            )
            fb_piaget = None
            fb_score = 0.0
            if not fb_proposal.get("_request_house_switch"):
                fb_piaget = self._compute_playtime_curiosity(
                    fb_proposal,
                    playtime_memory_obj,
                    self._task_history,
                    weights=piaget_weights
                    if isinstance(piaget_weights, dict) else None,
                )
                fb_score = fb_piaget["playtime_curiosity_score"]
            fb_entry = {
                "idx": 0,
                "raw": fb_raw,
                "veto": "",
                "proposal": fb_proposal,
                "score": fb_score,
                "piaget": fb_piaget,
            }
            valid = [fb_entry]
            trace["attempts"].append({
                "kind": "deterministic_fallback",
                "result": fb_raw,
                "evaluated": [{
                    "idx": 0, "veto": "", "score": fb_score,
                    "is_switch_house": fb_proposal.get("_request_house_switch", False),
                }],
            })

        # Pick the winner. For the random-pick modes (``random``,
        # ``llm_nocontext``, ``no_llm``) we ignore the score and pick
        # uniformly from the non-vetoed candidate pool — this isolates
        # "is the candidate POOL useful" from "is the SCORING useful".
        # For ``formula`` and ``llm`` we pick argmax (ties broken by
        # index, purely cosmetic).
        if random_pick:
            import random as _random
            valid.sort(key=lambda c: int(c["idx"]))  # stable order for logging
            winner = _random.choice(valid)
            logger.info(
                "  playtime sample-K (%s, random pick): K=%d, picked idx=%d "
                "uniformly from %d non-vetoed candidates",
                curiosity_mode, sample_k, winner["idx"], len(valid),
            )
        else:
            valid.sort(key=lambda c: (-float(c["score"]), int(c["idx"])))
            winner = valid[0]
        proposal = winner["proposal"]
        winner_piaget = winner["piaget"]

        if proposal.get("_request_house_switch"):
            self._molmospaces_house_switches_used += 1
            self._molmospaces_grounding_cache.clear()

        # Existing skill-novelty score (kept for backwards compat with
        # downstream consumers that read curiosity_score / novelty /
        # learnability — see compute_curiosity_scores).
        scored = self.compute_curiosity_scores([proposal], skill_context)
        if scored:
            proposal["curiosity_score"] = scored[0].get("curiosity_score", 0.25)
            proposal["novelty"] = scored[0].get("novelty", 0.5)
            proposal["learnability"] = scored[0].get("learnability", 0.5)

        # Surface the playtime curiosity score on the winning proposal.
        # House-switch responses get a placeholder 0 and no components
        # block, since affordance_information_gain has no target.
        if winner_piaget is not None:
            proposal["playtime_curiosity_score"] = winner_piaget["playtime_curiosity_score"]
            proposal["playtime_curiosity_components"] = (
                self._format_playtime_curiosity_components(winner_piaget)
            )
            trace["piaget_curiosity"] = winner_piaget
            logger.info(
                "  playtime sample-K: K=%d, picked idx=%d, score=%.3f "
                "(novelty=%.2f, frontier=%.2f, competence=%.2f); rejected=%d",
                sample_k,
                winner["idx"],
                winner_piaget["playtime_curiosity_score"],
                winner_piaget.get("novelty_score", 0.0),
                winner_piaget.get("frontier_score", 0.0),
                winner_piaget.get("competence_estimate", 0.0),
                sum(1 for c in evaluated if c["proposal"] is None),
            )
        else:
            proposal["playtime_curiosity_score"] = 0.0
            logger.info(
                "  playtime sample-K: K=%d, picked house-switch (no scoreable target)",
                sample_k,
            )

        # Trace every candidate the proposer saw this turn (winner +
        # losers + rejects), so offline analysis can correlate
        # Piaget-loss components with verifier success and tune the
        # mixing weights.
        trace["candidates"] = [
            {
                "idx": c["idx"],
                "veto": c["veto"],
                "score": c["score"],
                "is_winner": c["idx"] == winner["idx"] and c["veto"] == "",
                "is_switch_house": c.get("is_switch_house", False),
                "raw": c["raw"],
                "components": (
                    self._format_playtime_curiosity_components(c["piaget"])
                    if c.get("piaget") else None
                ),
            }
            for c in evaluated
        ]
        trace["winner_idx"] = winner["idx"]
        trace["final_llm_or_fallback_result"] = winner["raw"]
        trace["proposal"] = dict(proposal)
        self.last_proposal_trace = trace
        # NOTE on the dropped ``_playtime`` / ``interaction_type`` strip.
        # A previous refactor (90b0fb4f, "unified --play-mode") stripped
        # these two keys from the returned proposal to keep schema parity
        # with the LIBERO play path. Downstream code in lifelong_loop's
        # merge_keys block + per_step_verifier + failure_diagnoser /
        # feedback_generator was trying to read them anyway, so it kept
        # seeing ``_playtime: {}`` (empty dict) for every iter — verb /
        # target_display_name / success_criteria_nl all lost. iteration_
        # *.json shows the symptom (``_playtime: {}`` everywhere even
        # though the LLM emitted real playtime metadata).
        #
        # Schema parity with LIBERO was never required — those keys are
        # extra fields on the proposal dict, LIBERO downstream consumers
        # don't read them. Returning the full proposal here lets the
        # playtime-aware downstream agents see the verb + target +
        # success-criteria the LLM actually committed to.
        return proposal

    def _current_benchmark_task_meta(
        self,
        scene_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Return currently loaded benchmark metadata for deterministic fallback."""
        descriptor = dict(scene_context.get("task_descriptor") or {})
        current_id = (
            scene_context.get("current_activity")
            or descriptor.get("canonical_id")
            or descriptor.get("canonical_task_id")
            or descriptor.get("activity_name")
        )
        task_meta: dict[str, Any] = {}
        if current_id:
            task_meta = next(
                (
                    dict(t)
                    for t in self._available_tasks
                    if (
                        t.get("canonical_task_id", t.get("activity_name", ""))
                        == current_id
                        or t.get("canonical_id") == current_id
                    )
                ),
                {},
            )
        merged = {**task_meta, **descriptor}
        if not merged.get("objects") and task_meta.get("objects"):
            merged["objects"] = task_meta.get("objects")
        if not merged.get("metadata") and task_meta.get("metadata"):
            merged["metadata"] = task_meta.get("metadata")
        return merged

    @staticmethod
    def _validate_playtime_proposal(
        result: dict[str, Any],
        *,
        display_to_internal: dict[str, str],
        allowed_interactions: list[str],
        allow_freeform: bool,
        freeform_proposal: bool = False,
        allow_house_switch: bool = True,
        visible_display_names: set[str] | None = None,
        unsuitable_check=None,
        articulated_focus_names: set[str] | None = None,
        taskable_target_names: set[str] | None = None,
        placeable_target_names: set[str] | None = None,
    ) -> str:
        if bool(result.get("switch_house", False)):
            if allow_house_switch:
                return ""
            return "switch_house=true but house switching is disabled"
        interaction_raw = str(result.get("interaction_type") or "").strip()
        interaction = interaction_raw.lower()
        if not interaction:
            return "missing interaction_type"
        if freeform_proposal:
            if len(interaction_raw) > 96:
                return "freeform interaction_type is too long; keep it to one short action phrase"
            supplied_family = str(
                result.get("interaction_family")
                or result.get("canonical_interaction_type")
                or result.get("canonical_interaction")
                or result.get("action_family")
                or ""
            ).strip().lower().replace("-", "_").replace(" ", "_")
            if supplied_family and supplied_family not in TaskProposer._PLAYTIME_KNOWN_ACTIONS:
                return f"interaction_family {supplied_family!r} is not a supported playtime action family"
        elif interaction not in allowed_interactions and not allow_freeform:
            return f"interaction_type {interaction!r} not allowed"
        lang_blob = " ".join(
            str(result.get(k) or "")
            for k in ("language", "exploration_question", "expected_observation")
        ).lower()
        unsafe_words = ("throw", "smash", "break", "damage", "off counter", "off table")
        if any(word in lang_blob for word in unsafe_words):
            return "unsafe/destructive play language"
        if any(word in lang_blob for word in ("navigate", "drive base", "move base")):
            return "navigation/mobile-base play tasks are not supported by fixed Franka"
        target = str(result.get("target_object_display_name") or "")
        if not target:
            return "missing target_object_display_name"
        if target not in display_to_internal:
            return f"target_object_display_name {target!r} not in inventory"
        target_internal = display_to_internal.get(target, "")
        if taskable_target_names is not None and target_internal not in taskable_target_names:
            return (
                f"target_object_display_name {target!r} is not taskable as a "
                "main target; choose a pickable object or explicit articulation"
            )
        if articulated_focus_names is not None and target_internal not in articulated_focus_names:
            return (
                f"target_object_display_name {target!r} is not an articulated "
                "object target; pick a target from the articulated-object focus list"
            )
        if visible_display_names is not None and target not in visible_display_names:
            return (
                f"target_object_display_name {target!r} is not visible in either "
                "camera; pick a target from the visible inventory"
            )
        if unsuitable_check is not None:
            why = unsuitable_check(target_internal, role="target")
            if why:
                return (
                    f"target_object_display_name {target!r} is unsuitable: {why}"
                )
        secondary = str(result.get("secondary_object_display_name") or "")
        if secondary and secondary not in display_to_internal:
            return f"secondary_object_display_name {secondary!r} not in inventory"
        if secondary and placeable_target_names is not None and taskable_target_names is not None:
            secondary_internal = display_to_internal.get(secondary, "")
            if (
                secondary_internal not in placeable_target_names
                and secondary_internal not in taskable_target_names
            ):
                return (
                    f"secondary_object_display_name {secondary!r} is not a "
                    "placeable/receptacle, pickable, or explicit articulation"
                )
        if (
            secondary
            and visible_display_names is not None
            and secondary not in visible_display_names
        ):
            return (
                f"secondary_object_display_name {secondary!r} is not visible "
                "in either camera"
            )
        if secondary and unsuitable_check is not None:
            why = unsuitable_check(
                display_to_internal.get(secondary, ""), role="secondary"
            )
            if why:
                return (
                    f"secondary_object_display_name {secondary!r} is "
                    f"unsuitable: {why}"
                )
        canonical = TaskProposer._canonical_playtime_action(result)
        if canonical in {"place_in", "place_on", "stack"} and not secondary:
            return (
                f"interaction_type {interaction!r} requires "
                "secondary_object_display_name so grounded verification can "
                "check the final object relation"
            )
        criteria = result.get("success_criteria_nl") or []
        if not isinstance(criteria, list) or not criteria:
            return "success_criteria_nl must be a non-empty list"
        if canonical in {"open", "pull", "tug"} and articulated_focus_names is not None:
            criteria_blob = " ".join(str(x).lower() for x in criteria)
            resistance_as_success = any(
                phrase in criteria_blob
                for phrase in (
                    "resist",
                    "resistance",
                    "no movement",
                    "no-motion",
                    "does not move",
                    "doesn't move",
                    "not move",
                    "stays still",
                    "stay still",
                    "stays closed",
                    "remain closed",
                    "remains closed",
                )
            )
            motion_required = any(
                token in criteria_blob
                for token in (
                    "move",
                    "moves",
                    "moved",
                    "movement",
                    "displace",
                    "displacement",
                    "open",
                    "opens",
                    "opened",
                    "slide",
                    "slides",
                    "shift",
                    "joint",
                    "articulation",
                    "angle",
                )
            )
            if resistance_as_success:
                return (
                    "pull/open articulated tasks must not list resistance "
                    "or no-movement as success; require measurable target motion"
                )
            if not motion_required:
                return (
                    "pull/open articulated tasks must include measurable "
                    "target or articulation motion in success_criteria_nl"
                )
        return ""

    @staticmethod
    def _playtime_interaction_template(
        interaction: str,
        target_display: str,
        secondary_display: str | None = None,
    ) -> tuple[str, str, str, list[str], list[str]]:
        """Verb-appropriate NL fields for a synthesized playtime proposal.

        Returns ``(language, exploration_question, expected_observation,
        success_criteria_nl, result_state_fields)`` shaped so the result
        passes ``_validate_playtime_proposal`` (non-empty criteria, at
        least one measurable-state-change criterion, no resistance-as-
        success phrasing on pull/open). Used by the no_llm random
        baseline; mirrors the verb families handled in
        ``_fallback_playtime_proposal``.
        """
        v = (interaction or "").strip().lower()
        if v in {"open", "pull", "tug", "slide"}:
            language = (
                f"I want to see what happens if I gently {v} the {target_display}."
            )
            expected = (
                "The drawer/door/part may move a small measurable amount."
            )
            criteria = [
                f"the robot makes gentle contact with {target_display} or its handle/edge",
                f"{target_display} visibly moves/opens by a small measurable amount",
            ]
            fields = [
                "contact_made",
                "articulation_displacement",
                "target_motion_magnitude",
                "policy_implication",
            ]
        elif v in {"close", "push"}:
            language = (
                f"I'm curious what happens if I gently {v} the {target_display}."
            )
            expected = "The object/part may move, shift, or resist contact."
            criteria = [
                f"the robot makes gentle contact with {target_display}",
                f"{target_display} visibly moves/shifts by a small measurable amount",
            ]
            fields = [
                "contact_made",
                "object_moved",
                "movement_magnitude",
                "policy_implication",
            ]
        elif v in {"lift", "drop"}:
            direction = "rises" if v == "lift" else "lowers"
            language = (
                f"I want to {v} the {target_display} a little and watch what happens."
            )
            expected = "The object may rise/lower slightly, then settle."
            criteria = [
                f"the robot grasps or contacts {target_display}",
                f"{target_display} visibly {direction} by a small measurable amount",
            ]
            fields = [
                "object_moved",
                "height_change",
                "object_stability",
                "policy_implication",
            ]
        elif v in {"place_on", "place_in", "stack"} and secondary_display:
            rel = {
                "place_on": "onto",
                "place_in": "into",
                "stack": "on top of",
            }[v]
            language = f"I want to put the {target_display} {rel} the {secondary_display}."
            expected = f"The {target_display} should end up {rel} the {secondary_display}."
            criteria = [
                f"the robot picks up {target_display}",
                f"{target_display} ends up {rel} {secondary_display} in the final frames",
            ]
            fields = [
                "object_moved",
                "target_secondary_relation",
                "object_stability",
                "policy_implication",
            ]
        else:
            language = f"I'm curious how the {target_display} responds to a gentle {v}."
            expected = "The object may move, rotate, wobble, or resist contact."
            criteria = [
                f"the robot visibly makes gentle contact with {target_display}",
                f"{target_display} visibly moves by a small measurable amount",
            ]
            fields = [
                "object_moved",
                "movement_magnitude",
                "object_stability",
                "policy_implication",
            ]
        exploration = f"How does {target_display} respond to a gentle {v}?"
        return language, exploration, expected, criteria, fields

    @classmethod
    def _random_playtime_raw_proposal(
        cls,
        *,
        allowed_interactions: list[str],
        taskable_display_names: list[str],
        placeable_display_names: list[str],
        rng,
        allow_switch_house: bool = False,
    ) -> dict[str, Any]:
        """One uniform-random (verb × object) playtime proposal, no LLM.

        Draws a target from the already visible/reachable/taskable pool
        and a verb from ``allowed_interactions``. Place verbs additionally
        draw a receptacle from the placeable pool; if none is available
        the verb is re-drawn from the non-place verbs so the result still
        passes validation. Returns a raw dict in the same shape the LLM
        proposer emits (consumed by ``_build_playtime_proposal_dict`` and
        ``_validate_playtime_proposal``).
        """
        place_verbs = {"place_on", "place_in", "stack"}
        if not taskable_display_names:
            if allow_switch_house:
                return {
                    "switch_house": True,
                    "reasoning": (
                        "Random baseline (no-LLM): no visible/reachable "
                        "taskable target in this house; switching houses."
                    ),
                    "interaction_type": (
                        allowed_interactions[0] if allowed_interactions else "push"
                    ),
                    "target_object_display_name": "",
                    "secondary_object_display_name": None,
                    "language": "No taskable target available; switching houses.",
                    "expected_new_skills": ["abort empty-scene iter via house switch"],
                    "novelty_score": 0.0,
                    "difficulty_estimate": "medium",
                }
            # Empty target → vetoed by validator → deterministic fallback.
            return {
                "switch_house": False,
                "interaction_type": (
                    allowed_interactions[0] if allowed_interactions else "push"
                ),
                "target_object_display_name": "",
                "secondary_object_display_name": None,
                "language": "No taskable target available.",
                "success_criteria_nl": ["no out-of-view object is selected"],
                "novelty_score": 0.0,
                "difficulty_estimate": "easy",
            }
        target = rng.choice(taskable_display_names)
        verbs = list(allowed_interactions) or ["push"]
        verb = rng.choice(verbs)
        secondary: str | None = None
        if verb in place_verbs:
            secondary_pool = [
                d for d in placeable_display_names if d and d != target
            ]
            if secondary_pool:
                secondary = rng.choice(secondary_pool)
            else:
                # No receptacle to place onto/into — re-draw a non-place verb.
                non_place = [v for v in verbs if v not in place_verbs]
                verb = rng.choice(non_place) if non_place else "push"
        language, exploration, expected, criteria, fields = (
            cls._playtime_interaction_template(verb, target, secondary)
        )
        return {
            "switch_house": False,
            "reasoning": (
                f"Random baseline (no-LLM): uniformly drew verb '{verb}' "
                f"× object '{target}'."
            ),
            "sensorimotor_motive": "cause_effect",
            "interaction_type": verb,
            "target_object_display_name": target,
            "secondary_object_display_name": secondary,
            "language": language,
            "exploration_question": exploration,
            "expected_observation": expected,
            "success_criteria_nl": criteria,
            "result_state_fields": fields,
            "expected_new_skills": [f"observe {target} response to {verb}"],
            "novelty_score": 0.4,
            "difficulty_estimate": "easy",
        }

    @staticmethod
    def _fallback_playtime_proposal(
        inventory: dict[str, Any],
        display_to_internal: dict[str, str],
        allowed_interactions: list[str],
        *,
        benchmark_task_meta: dict[str, Any] | None = None,
        visible_display_names: set[str] | None = None,
        unsuitable_check=None,
        target_allowlist: set[str] | None = None,
        allow_switch_house: bool = False,
    ) -> dict[str, Any]:
        internal_to_display = {v: k for k, v in display_to_internal.items()}
        benchmark_task_meta = benchmark_task_meta or {}
        benchmark_objects = [
            str(obj)
            for obj in (benchmark_task_meta.get("objects") or [])
            if str(obj).strip()
        ]
        if benchmark_objects:
            target_internal = benchmark_objects[0]
            target_display = internal_to_display.get(target_internal)
            if target_display and (
                target_allowlist is None or target_internal in target_allowlist
            ):
                family = str(
                    benchmark_task_meta.get("task_family")
                    or (benchmark_task_meta.get("metadata") or {}).get("task_type")
                    or ""
                ).strip().lower()
                preferred_by_family = {
                    "open": ["open", "pull", "tug", "wiggle"],
                    "close": ["close", "push"],
                    "pick": ["lift", "pull"],
                    "pick_and_place": ["place_on", "place_in", "lift"],
                }
                preferred = preferred_by_family.get(family, [])
                interaction = next(
                    (verb for verb in preferred if verb in allowed_interactions),
                    allowed_interactions[0] if allowed_interactions else "open",
                )
                language = str(benchmark_task_meta.get("language") or "").strip()
                if not language:
                    language = f"{interaction.capitalize()} {target_display}."
                exploration = (
                    f"What happens if I try the benchmark action '{language}' "
                    f"on {target_display}?"
                )
                if interaction in {"open", "pull", "tug"}:
                    expected = (
                        "The drawer, door, or articulated part should open or "
                        "move a small measurable amount; resistance-only is "
                        "an observation/failure for this pull/open attempt."
                    )
                    criteria = [
                        f"the robot makes contact with {target_display} or its handle/front edge",
                        f"{target_display} visibly opens or moves by a small measurable amount",
                        "the action remains gentle and bounded",
                    ]
                    fields = [
                        "contact_made",
                        "articulation_displacement",
                        "target_motion_magnitude",
                        "policy_implication",
                    ]
                elif interaction == "wiggle":
                    expected = (
                        "The drawer, door, or articulated part may wiggle a "
                        "small amount, visibly resist, or stay still."
                    )
                    criteria = [
                        f"the robot makes contact with {target_display} or its handle/front edge",
                        f"{target_display} visibly wiggles/moves or clearly resists the bounded wiggle",
                        "the action remains gentle and bounded",
                    ]
                    fields = [
                        "contact_made",
                        "articulation_displacement",
                        "resistance_observed",
                        "policy_implication",
                    ]
                elif interaction in {"close", "push"}:
                    expected = (
                        "The articulated part may move toward closed, resist, or stay still."
                    )
                    criteria = [
                        f"the robot makes contact with {target_display}",
                        f"{target_display} moves toward the requested benchmark state or clearly resists",
                    ]
                    fields = [
                        "contact_made",
                        "articulation_displacement",
                        "resistance_observed",
                        "policy_implication",
                    ]
                else:
                    expected = "The object may move, lift slightly, or resist contact."
                    criteria = [
                        f"the robot visibly interacts with {target_display}",
                        "the final frames show the object's response or resistance",
                    ]
                    fields = [
                        "object_moved",
                        "movement_magnitude",
                        "object_stability",
                        "policy_implication",
                    ]
                return {
                    "switch_house": False,
                    "reasoning": (
                        "Deterministic fallback: use the currently loaded "
                        "benchmark task target instead of inventing a new target."
                    ),
                    "sensorimotor_motive": "means_end",
                    "interaction_type": interaction,
                    "target_object_display_name": target_display,
                    "secondary_object_display_name": None,
                    "language": language,
                    "exploration_question": exploration,
                    "expected_observation": expected,
                    "success_criteria_nl": criteria,
                    "result_state_fields": fields,
                    "expected_new_skills": [
                        f"benchmark fallback {interaction} on {target_display}"
                    ],
                    "novelty_score": 0.4,
                    "difficulty_estimate": "benchmark",
                }
        pickables = inventory.get("pickables", []) or []
        articulations = inventory.get("articulations", []) or []
        # When the optional inventory visibility gate is on, prefer only
        # items marked visible. In the default mode, visible_display_names is
        # None and fallback selection uses inventory/focus/unsuitable filters;
        # the MolmoSpaces environment verifier owns target visibility later.
        candidates = list(pickables) + list(articulations)
        chosen: dict[str, Any] = {}
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            internal = entry.get("internal_name", "")
            if target_allowlist is not None and internal not in target_allowlist:
                continue
            display = internal_to_display.get(internal, "")
            if visible_display_names is not None and display not in visible_display_names:
                continue
            # Skip oversized / out-of-reach targets when the validator's
            # unsuitable_check is wired through.
            if unsuitable_check is not None and unsuitable_check(internal, role="target"):
                continue
            chosen = entry
            break
        if not chosen and candidates and visible_display_names is None:
            chosen = candidates[0] if isinstance(candidates[0], dict) else {}
        if not chosen:
            # No usable target in this scene. If the caller has
            # house-switching budget left, request a switch instead of
            # returning a defer/no-op proposal — the loop has plumbing
            # (_request_house_switch) that will rotate to a fresh procthor
            # house, giving the next iter a different camera roll. This
            # is the iter-1 fix for v7's "0/115 visible" paralysis.
            if allow_switch_house:
                return {
                    "switch_house": True,
                    "reasoning": (
                        "Deterministic fallback: no suitable visible target "
                        "in this house; requesting a house switch to retry "
                        "with a different scene."
                    ),
                    "interaction_type": allowed_interactions[0] if allowed_interactions else "touch",
                    "target_object_display_name": "",
                    "secondary_object_display_name": None,
                    "language": "No safe visible playtime target available; switching houses.",
                    "expected_new_skills": ["abort empty-scene iter via house switch"],
                    "novelty_score": 0.0,
                    "difficulty_estimate": "medium",
                }
            return {
                "switch_house": False,
                "reasoning": (
                    "Deterministic fallback: no suitable fallback target "
                    "was available, so no object-specific task can be selected."
                ),
                "sensorimotor_motive": "cause_effect",
                "interaction_type": allowed_interactions[0] if allowed_interactions else "touch",
                "target_object_display_name": "",
                "secondary_object_display_name": None,
                "language": "No safe visible playtime target was available.",
                "exploration_question": "What should the robot do when no target is visible?",
                "expected_observation": "No manipulation should be attempted.",
                "success_criteria_nl": ["no unsafe out-of-view object is selected"],
                "result_state_fields": ["policy_implication"],
                "expected_new_skills": ["defer when no visible target exists"],
                "novelty_score": 0.0,
                "difficulty_estimate": "hard",
            }
        internal = chosen.get("internal_name", "object")
        display = internal_to_display.get(internal, internal)
        interaction = (
            "push" if "push" in allowed_interactions
            else (allowed_interactions[0] if allowed_interactions else "touch")
        )
        return {
            "switch_house": False,
            "reasoning": "Deterministic fallback: test a simple visible cause-effect interaction.",
            "sensorimotor_motive": "cause_effect",
            "interaction_type": interaction,
            "target_object_display_name": display,
            "secondary_object_display_name": None,
            "language": f"Gently {interaction} the {display} and observe what changes.",
            "exploration_question": f"How does {display} respond to a gentle {interaction}?",
            "expected_observation": "The object may move, rotate, wobble, or resist contact.",
            "success_criteria_nl": [
                f"the robot visibly makes gentle contact with {display}",
                "the final frames show the object's response or resistance",
            ],
            "result_state_fields": [
                "object_moved",
                "movement_magnitude",
                "object_stability",
                "contact_sensitivity",
                "policy_implication",
            ],
            "expected_new_skills": [f"observe object response to {interaction}"],
            "novelty_score": 0.4,
            "difficulty_estimate": "easy",
        }

    @staticmethod
    def _build_playtime_proposal_dict(
        result: dict[str, Any],
        *,
        inventory: dict[str, Any],
        display_to_internal: dict[str, str],
        scene_context: dict[str, Any],
        freeform_proposal: bool = False,
    ) -> dict[str, Any]:
        switch_house = bool(result.get("switch_house", False))
        raw_interaction = str(result.get("interaction_type") or "touch").strip()
        canonical_interaction = TaskProposer._canonical_playtime_action(result)
        interaction = canonical_interaction if freeform_proposal else raw_interaction.lower()
        interaction_slug = re.sub(r"[^a-z0-9_]+", "_", interaction).strip("_") or "touch"
        scene_family = str(inventory.get("house_index") or "house")
        if switch_house:
            language = result.get("language") or "Switch to a different house for play."
            activity = f"molmospaces:playtime:switch_house:{scene_family}"
            return {
                "mode": "novel",
                "activity_name": activity,
                "canonical_task_id": activity,
                "language": language,
                "task_family": "playtime_switch_house",
                "scene_family": scene_family,
                "benchmark": str(inventory.get("scene_dataset", "")),
                "scene_model": scene_context.get("scene_model", "molmospaces_scene"),
                "objects": [],
                "activity_definition_id": 0,
                "goal_conditions": language,
                "reasoning": result.get("reasoning", ""),
                "expected_new_skills": result.get("expected_new_skills", []),
                "novelty_score": result.get("novelty_score", 0.5),
                "difficulty_estimate": result.get("difficulty_estimate", "medium"),
                "_request_house_switch": True,
                "_molmospaces_requested_task_type": None,
                "_molmospaces_proposer_mode": "playtime",
                "_playtime": {
                    "interaction_type": interaction,
                    "interaction_text": raw_interaction if freeform_proposal else None,
                    "interaction_family": interaction,
                    "freeform_proposal": bool(freeform_proposal),
                },
            }
        target_display = str(result.get("target_object_display_name") or "")
        secondary_display = str(result.get("secondary_object_display_name") or "")
        target_internal = display_to_internal.get(target_display)
        secondary_internal = display_to_internal.get(secondary_display) if secondary_display else None
        target_initial_entry = TaskProposer._find_inventory_entry(inventory, target_internal)
        secondary_initial_entry = TaskProposer._find_inventory_entry(inventory, secondary_internal)
        language = result.get("language") or f"{raw_interaction.capitalize()} {target_display}"
        slug_src = f"{scene_family}:{interaction}:{target_internal}:{secondary_internal}:{language}"
        short_hash = hashlib.sha1(slug_src.encode("utf-8")).hexdigest()[:8]
        target_token = (target_internal or target_display or "object").split("/")[-1]
        activity = f"molmospaces:playtime:{interaction_slug}:{scene_family}:{target_token}:{short_hash}"
        objects = [o for o in (target_internal, secondary_internal) if o]
        playtime = {
            "interaction_type": interaction,
            "interaction_text": raw_interaction if freeform_proposal else None,
            "interaction_family": interaction,
            "freeform_proposal": bool(freeform_proposal),
            "sensorimotor_motive": result.get("sensorimotor_motive", "cause_effect"),
            "exploration_question": result.get("exploration_question", ""),
            "expected_observation": result.get("expected_observation", ""),
            "success_criteria_nl": result.get("success_criteria_nl", []),
            "result_state_fields": result.get("result_state_fields", []),
            "target_internal_name": target_internal,
            "target_display_name": target_display,
            "secondary_internal_name": secondary_internal,
            "secondary_display_name": secondary_display or None,
            "target_initial_entry": target_initial_entry,
            "secondary_initial_entry": secondary_initial_entry,
            "verifier": "grounded",
            "verification_spec": {
                "strategy": TaskProposer._playtime_verification_strategy(interaction),
                "interaction_type": interaction,
                "interaction_text": raw_interaction if freeform_proposal else None,
                "target_internal_name": target_internal,
                "secondary_internal_name": secondary_internal,
                "success_criteria_nl": result.get("success_criteria_nl", []),
            },
        }
        return {
            "mode": "novel",
            "activity_name": activity,
            "canonical_task_id": activity,
            "language": language,
            "task_family": f"playtime_{interaction_slug}",
            "scene_family": scene_family,
            "benchmark": str(inventory.get("scene_dataset", "")),
            "scene_model": scene_context.get("scene_model", "molmospaces_scene"),
            "objects": objects,
            "activity_definition_id": 0,
            "goal_conditions": language,
            "reasoning": result.get("reasoning", ""),
            "expected_new_skills": result.get("expected_new_skills", []),
            "novelty_score": result.get("novelty_score", 0.5),
            "difficulty_estimate": result.get("difficulty_estimate", "medium"),
            "_request_house_switch": False,
            "_molmospaces_open_spec": None,
            "_molmospaces_proposer_mode": "playtime",
            "_playtime": playtime,
        }

    @staticmethod
    def _find_inventory_entry(
        inventory: dict[str, Any],
        internal_name: str | None,
    ) -> dict[str, Any] | None:
        if not internal_name:
            return None
        for section in (
            *TaskProposer._PLAYTIME_TARGET_SECTIONS,
            *TaskProposer._PLAYTIME_PLACEABLE_SECTIONS,
        ):
            for entry in inventory.get(section, []) or []:
                if isinstance(entry, dict) and entry.get("internal_name") == internal_name:
                    out = dict(entry)
                    out["inventory_section"] = section
                    return out
        return None

    @staticmethod
    def _playtime_verification_strategy(interaction: str) -> str:
        interaction = str(interaction or "").strip().lower()
        if interaction in {"place_in", "place_on", "stack"}:
            return "target_final_relation_to_secondary"
        if interaction in {"push", "slide", "roll", "knock_over", "nudge", "rotate"}:
            return "target_displacement_or_explicit_resistance"
        if interaction in {"touch", "tap", "poke", "press"}:
            return "targeted_contact_attempt"
        if interaction in {"lift", "lower", "drop", "shake"}:
            return "grasp_or_target_state_change"
        if interaction in {"open", "close", "pull", "tug", "wiggle"}:
            return "articulation_joint_or_target_state_change"
        return "generic_grounded_state_change"

    def _propose_novel_molmospaces_open(
        self,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Propose a typed task built from the live house inventory.

        Flow:
          1. Pull the live inventory from the bridge.
          2. Ground the inventory with a single VLM call so the LLM
             sees human-readable display names + per-target difficulty
             cues. Cached per (house_index, scene_dataset).
          3. Render the open-mode prompt (curriculum hint, schemas,
             inventory, history).
          4. Validate the LLM's choice against the schema, the
             curriculum stage, and the unreliable-pick blocklist;
             re-roll once on rejection.
          5. Return a proposal dict the lifelong loop converts into a
             `set_task_from_spec` call (or a `request_new_house` call
             if `switch_house` is true and the YAML allows it).
        """
        from rats.agents.molmospaces_catalog import (
            TASK_TYPE_SCHEMAS,
            build_molmospaces_taskschema_text,
            is_blocklisted_pick_target,
            task_type_allowed_at_stage,
        )
        from rats.agents.molmospaces_scene_grounder import ground_inventory
        from rats.loop.molmospaces_utils import extract_molmospaces_scene_inventory

        env = scene_context.get("_env")
        if env is None:
            raise RuntimeError(
                "_propose_novel_molmospaces_open requires the live env in "
                "scene_context['_env']; the lifelong loop must inject it."
            )
        inventory = extract_molmospaces_scene_inventory(env)
        grounded = ground_inventory(
            env,
            inventory,
            enabled=self._molmospaces_vlm_grounding,
            use_geometric_visibility_gate=self._molmospaces_geometric_visibility_gate,
            cache=self._molmospaces_grounding_cache,
        )
        # display_name -> internal_name reverse map for validation
        display_to_internal: dict[str, str] = {
            phrase: internal
            for internal, phrase in (grounded.get("display_names") or {}).items()
        }
        # Annotated inventory for the prompt (display_name + difficulty_cue
        # baked in so the LLM doesn't have to cross-reference two blocks).
        prompt_inventory = self._annotate_inventory_for_prompt(inventory, grounded)

        stage = self._curriculum_stage_molmospaces()
        supported_types = [
            tt for tt in self._molmospaces_allowed_task_types
            if tt in TASK_TYPE_SCHEMAS
        ]
        allowed_types = [
            tt for tt in supported_types if task_type_allowed_at_stage(tt, stage)
        ]
        forced_task_type: str | None = None
        if self._molmospaces_forced_task_type_sequence:
            forced_task_type = self._molmospaces_forced_task_type_sequence[
                len(self._task_history) % len(self._molmospaces_forced_task_type_sequence)
            ]
            if forced_task_type in TASK_TYPE_SCHEMAS and forced_task_type in supported_types:
                allowed_types = [forced_task_type]
            else:
                logger.warning(
                    "  ignoring unsupported forced MolmoSpaces task type %r",
                    forced_task_type,
                )
                forced_task_type = None
        if not allowed_types:
            # If the configured supported set only contains later-stage tasks,
            # fall back to the first supported type rather than producing an
            # empty prompt. This mostly helps smoke configs.
            allowed_types = supported_types[:1] or ["pick"]

        curriculum_hint = self._MOLMOSPACES_CURRICULUM_HINTS.get(stage, "")
        if forced_task_type:
            curriculum_hint = (
                f"SMOKE TEST OVERRIDE: propose exactly one `{forced_task_type}` "
                "task from the live inventory. Ignore normal curriculum stage "
                "progression for this run, but still satisfy the task schema "
                "and inventory constraints."
            )

        switches_left = (
            self._molmospaces_house_switch_max_per_run
            - self._molmospaces_house_switches_used
        )
        if self._molmospaces_allow_house_switching and switches_left > 0:
            house_switch_block = (
                f"House-switching is ENABLED. You may set switch_house=true "
                f"if no useful task remains in this house. {switches_left} "
                f"switches remain in this run."
            )
        else:
            house_switch_block = (
                "House-switching is DISABLED for this run; do NOT set "
                "switch_house=true."
            )

        prompt_template = Path("rats/prompts/task_proposer_novel_molmospaces_open.txt").read_text()
        history_str = "None yet." if not self._task_history else json.dumps(
            self._task_history[-10:], indent=2
        )
        user_prompt = (
            prompt_template
            .replace("{curriculum_hint}", curriculum_hint)
            .replace("{task_type_schemas}", build_molmospaces_taskschema_text())
            .replace("{inventory}", json.dumps(prompt_inventory, indent=2))
            .replace("{skill_context}", json.dumps(skill_context, indent=2))
            .replace("{task_history}", history_str)
            .replace("{house_switch_block}", house_switch_block)
            .replace("{allowed_task_types}", json.dumps(allowed_types))
        )
        system_prompt = (
            "You are a curiosity-driven task proposer for a robot learning system. "
            "Propose a NOVEL manipulation task. Respond only in valid JSON."
        )

        trace: dict[str, Any] = {
            "proposer": "molmospaces_open",
            "stage": stage,
            "allowed_task_types": allowed_types,
            "configured_supported_task_types": supported_types,
            "forced_task_type": forced_task_type,
            "house_switches_left": switches_left,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "inventory": inventory,
            "grounded_inventory": grounded,
            "prompt_inventory": prompt_inventory,
            "history": self._task_history[-10:],
            "attempts": [],
        }

        result = query_llm_json(system_prompt, user_prompt)
        veto = self._validate_open_proposal(
            result,
            inventory=inventory,
            display_to_internal=display_to_internal,
            allowed_task_types=allowed_types,
            stage=stage,
            schemas=TASK_TYPE_SCHEMAS,
            blocklist_check=is_blocklisted_pick_target,
        )
        trace["attempts"].append({
            "kind": "llm",
            "result": result,
            "veto": veto,
        })
        if veto:
            logger.warning(
                "  open-mode proposer rejected (%s); re-rolling once",
                veto,
            )
            retry_prompt = (
                user_prompt
                + f"\n\nPREVIOUS PROPOSAL REJECTED:\n  reason: {veto}\n"
                + "Re-propose a task that satisfies the rules above. JSON only."
            )
            result = query_llm_json(system_prompt, retry_prompt)
            veto = self._validate_open_proposal(
                result,
                inventory=inventory,
                display_to_internal=display_to_internal,
                allowed_task_types=allowed_types,
                stage=stage,
                schemas=TASK_TYPE_SCHEMAS,
                blocklist_check=is_blocklisted_pick_target,
            )
            trace["retry_prompt"] = retry_prompt
            trace["attempts"].append({
                "kind": "llm_retry",
                "result": result,
                "veto": veto,
            })
            if veto:
                logger.warning(
                    "  open-mode proposer second roll also rejected (%s); "
                    "falling back to a deterministic pick", veto,
                )
                result = self._fallback_open_proposal(
                    inventory, display_to_internal, allowed_types,
                )
                trace["attempts"].append({
                    "kind": "deterministic_fallback",
                    "result": result,
                    "veto": "",
                })

        proposal = self._build_open_proposal_dict(
            result,
            inventory=inventory,
            display_to_internal=display_to_internal,
            scene_context=scene_context,
        )
        if proposal.get("_request_house_switch"):
            self._molmospaces_house_switches_used += 1
            # Drop the cache so the next iteration re-grounds the new house.
            self._molmospaces_grounding_cache.clear()

        scored = self.compute_curiosity_scores([proposal], skill_context)
        if scored:
            proposal["curiosity_score"] = scored[0].get("curiosity_score", 0.25)
            proposal["novelty"] = scored[0].get("novelty", 0.5)
            proposal["learnability"] = scored[0].get("learnability", 0.5)
        trace["final_llm_or_fallback_result"] = result
        trace["proposal"] = proposal
        self.last_proposal_trace = trace
        return proposal

    @staticmethod
    def _annotate_inventory_for_prompt(
        inventory: dict[str, Any],
        grounded: dict[str, Any],
        *,
        split_visibility: bool = False,
        visible_only: bool = False,
        unsuitable_check=None,
    ) -> dict[str, Any]:
        """Return a copy of `inventory` with display_name, difficulty_cue,
        and visible merged into each item dict so the LLM only has to
        scan one block.

        When ``visible_only`` is true, hidden and physically unsuitable items
        are omitted entirely so the proposer prompt does not show a full
        simulator inventory plus an excluded-item appendix.

        When ``split_visibility`` is true, each kind is split into a
        ``<kind>`` block (visible items the proposer may target) and a
        ``<kind>_out_of_view`` block (items present in the simulator but
        not in either camera). The playtime proposer only uses the split
        form when its legacy inventory visibility gate is explicitly enabled;
        by default a later MolmoSpaces environment verifier owns visibility.
        """
        names = grounded.get("display_names") or {}
        cues = grounded.get("difficulty_cues") or {}
        visible_map = grounded.get("visible") or {}
        out: dict[str, Any] = {
            "house_index": inventory.get("house_index"),
            "scene_dataset": inventory.get("scene_dataset"),
            "rooms": inventory.get("rooms", []),
        }
        for kind in (
            *TaskProposer._PLAYTIME_TARGET_SECTIONS,
            *TaskProposer._PLAYTIME_PLACEABLE_SECTIONS,
        ):
            visible_items: list[dict[str, Any]] = []
            hidden_items: list[dict[str, Any]] = []
            for entry in inventory.get(kind, []) or []:
                if not isinstance(entry, dict):
                    continue
                internal = entry.get("internal_name", "")
                # Default to visible when grounder returned nothing for
                # this item (e.g. fallback path with vision disabled)
                # so legacy non-playtime paths are unchanged.
                is_visible = bool(visible_map.get(internal, True))
                role = (
                    "target"
                    if kind in TaskProposer._PLAYTIME_TARGET_SECTIONS
                    else "secondary"
                )
                unsuitable_reason = (
                    unsuitable_check(internal, role=role)
                    if unsuitable_check is not None else ""
                )
                if visible_only and ((not is_visible) or unsuitable_reason):
                    continue
                annotated = {
                    k: v
                    for k, v in entry.items()
                    if k not in {"position", "bbox", "bounds", "aabb", "asset_uid"}
                } | {
                    "display_name": names.get(internal, ""),
                    "difficulty_cue": cues.get(internal, ""),
                    "visible": is_visible,
                }
                if unsuitable_reason:
                    annotated["excluded_reason"] = unsuitable_reason
                if is_visible or not split_visibility:
                    visible_items.append(annotated)
                else:
                    hidden_items.append(annotated)
            out[kind] = visible_items
            if split_visibility:
                out[f"{kind}_out_of_view"] = hidden_items
        return out

    @staticmethod
    def _validate_open_proposal(
        result: dict[str, Any],
        *,
        inventory: dict[str, Any],
        display_to_internal: dict[str, str],
        allowed_task_types: list[str],
        stage: int,
        schemas: dict[str, dict[str, Any]],
        blocklist_check,
    ) -> str:
        """Reject malformed / out-of-curriculum / out-of-inventory proposals."""
        switch_house = bool(result.get("switch_house", False))
        task_type = str(result.get("task_type", "")).lower()

        if switch_house:
            # When asking for a new house we don't validate the task fields.
            return ""

        if task_type not in allowed_task_types:
            return (
                f"task_type {task_type!r} not allowed at stage {stage}; "
                f"allowed: {allowed_task_types}"
            )
        schema = schemas[task_type]
        for required_key in schema["required"]:
            if not result.get(required_key):
                return f"missing required key {required_key!r} for task_type {task_type}"

        # Resolve display_name -> internal_name for each named target.
        target_internal: str | None = None
        if result.get("target_object_display_name"):
            disp = str(result["target_object_display_name"])
            if disp not in display_to_internal:
                return f"target_object_display_name {disp!r} not in inventory"
            target_internal = display_to_internal[disp]
        if result.get("place_receptacle_display_name"):
            disp = str(result["place_receptacle_display_name"])
            if disp not in display_to_internal:
                return f"place_receptacle_display_name {disp!r} not in inventory"
        joint_internal: str | None = None
        if result.get("joint_object_display_name"):
            disp = str(result["joint_object_display_name"])
            if disp not in display_to_internal:
                return f"joint_object_display_name {disp!r} not in inventory"
            joint_internal = display_to_internal[disp]

        # Category-level checks against the schema's permitted set.
        permitted = schema.get("target_categories") or ()
        if permitted:
            check_internal = joint_internal or target_internal
            if check_internal:
                category = ""
                # Find category from the inventory entry.
                for kind in (
                    *TaskProposer._PLAYTIME_TARGET_SECTIONS,
                    *TaskProposer._PLAYTIME_PLACEABLE_SECTIONS,
                ):
                    for entry in inventory.get(kind, []) or []:
                        if entry.get("internal_name") == check_internal:
                            category = (entry.get("category") or "").lower()
                            break
                    if category:
                        break
                if category and not any(p in category for p in permitted):
                    return (
                        f"category {category!r} not in {task_type}'s permitted "
                        f"set {sorted(permitted)}"
                    )

        # Pick-reliability blocklist.
        if task_type in ("pick", "pick_and_place") and target_internal:
            for entry in inventory.get("pickables", []) or []:
                if entry.get("internal_name") == target_internal:
                    if blocklist_check(entry.get("category", "")):
                        return (
                            f"target category {entry.get('category')!r} is on "
                            f"the unreliable-pick blocklist"
                        )
                    break

        # Single-base-pose reach gate for pick_and_place. The Franka arm is
        # bolted to a fixed base that the bridge places once per task; if the
        # target object and the place receptacle are too far apart in XY, no
        # single base pose can serve both grasp and place. Gate at
        # _PICK_PLACE_MAX_XY_M (a bit past one Franka workspace radius — the
        # bridge can re-place the base mid-attempt but only marginally).
        if (
            task_type == "pick_and_place"
            and target_internal
            and result.get("place_receptacle_display_name")
        ):
            place_internal = display_to_internal.get(
                str(result["place_receptacle_display_name"])
            )
            tgt_pos = TaskProposer._lookup_inventory_position(
                inventory, target_internal,
            )
            recp_pos = TaskProposer._lookup_inventory_position(
                inventory, place_internal,
            )
            if tgt_pos is not None and recp_pos is not None:
                dx = float(tgt_pos[0]) - float(recp_pos[0])
                dy = float(tgt_pos[1]) - float(recp_pos[1])
                xy_dist = (dx * dx + dy * dy) ** 0.5
                if xy_dist > TaskProposer._PICK_PLACE_MAX_XY_M:
                    return (
                        f"target {target_internal!r} and receptacle "
                        f"{place_internal!r} are {xy_dist:.2f} m apart in XY; "
                        f"max single-base-pose reach is "
                        f"{TaskProposer._PICK_PLACE_MAX_XY_M:.2f} m"
                    )

        # joint_index range check for open / close.
        if task_type in ("open", "close"):
            if joint_internal is None:
                return "joint_object_display_name required for open/close"
            try:
                ji = int(result.get("joint_index"))
            except (TypeError, ValueError):
                return "joint_index must be an integer"
            n_joints = 0
            for entry in inventory.get("articulations", []) or []:
                if entry.get("internal_name") == joint_internal:
                    n_joints = len(entry.get("joints", []) or [])
                    break
            if ji < 0 or ji >= n_joints:
                return (
                    f"joint_index {ji} out of range for {joint_internal!r} "
                    f"(0..{n_joints - 1})"
                )
        return ""

    @staticmethod
    def _fallback_open_proposal(
        inventory: dict[str, Any],
        display_to_internal: dict[str, str],
        allowed_task_types: list[str],
    ) -> dict[str, Any]:
        """Pick the first feasible (task_type, target) when both LLM rolls fail.

        Conservative: prefers `pick` on the first pickable that has a grasp
        file, then `pick_and_place` on the first (pickable, receptacle)
        pair, then `open` on the first articulation. Returns a dict shaped
        like the LLM's JSON so `_build_open_proposal_dict` can consume it.
        """
        pickables = [
            p for p in inventory.get("pickables", []) or []
            if p.get("has_grasp_file")
        ]
        receptacles = inventory.get("receptacles", []) or []
        articulations = inventory.get("articulations", []) or []
        # internal -> display reverse lookup
        internal_to_display = {v: k for k, v in display_to_internal.items()}

        if "pick" in allowed_task_types and pickables:
            target = pickables[0]
            return {
                "switch_house": False,
                "task_type": "pick",
                "target_object_display_name": internal_to_display.get(
                    target["internal_name"], target["internal_name"]
                ),
                "language": f"Pick up {internal_to_display.get(target['internal_name'], target.get('category', 'object'))}",
                "reasoning": "Deterministic fallback after LLM rejections.",
                "expected_new_skills": [],
                "novelty_score": 0.2,
                "difficulty_estimate": "easy",
            }
        if "pick_and_place" in allowed_task_types and pickables and receptacles:
            # Pair the first pickable with the nearest receptacle in XY so the
            # deterministic fallback honors the same single-base-pose reach
            # gate the validator enforces. Falls back to receptacles[0] when
            # no positions are recorded.
            target = pickables[0]
            tgt_pos = target.get("position") or [0.0, 0.0, 0.0]
            best_dest = receptacles[0]
            best_dist = float("inf")
            for cand in receptacles:
                cand_pos = cand.get("position") or [0.0, 0.0, 0.0]
                try:
                    dx = float(cand_pos[0]) - float(tgt_pos[0])
                    dy = float(cand_pos[1]) - float(tgt_pos[1])
                except (TypeError, ValueError):
                    continue
                d = (dx * dx + dy * dy) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_dest = cand
            dest = best_dest
            return {
                "switch_house": False,
                "task_type": "pick_and_place",
                "target_object_display_name": internal_to_display.get(
                    target["internal_name"], target["internal_name"]
                ),
                "place_receptacle_display_name": internal_to_display.get(
                    dest["internal_name"], dest["internal_name"]
                ),
                "language": f"Place {target.get('category', 'object')} on {dest.get('category', 'surface')}",
                "reasoning": "Deterministic fallback after LLM rejections.",
                "expected_new_skills": [],
                "novelty_score": 0.2,
                "difficulty_estimate": "medium",
            }
        if any(tt in allowed_task_types for tt in ("open", "close")) and articulations:
            task_type = "open" if "open" in allowed_task_types else "close"
            art = articulations[0]
            return {
                "switch_house": False,
                "task_type": task_type,
                "joint_object_display_name": internal_to_display.get(
                    art["internal_name"], art["internal_name"]
                ),
                "joint_index": 0,
                "language": f"{task_type.capitalize()} the {art.get('category', 'fixture')}",
                "reasoning": "Deterministic fallback after LLM rejections.",
                "expected_new_skills": [],
                "novelty_score": 0.2,
                "difficulty_estimate": "medium",
            }
        # Nothing usable in this house; ask for a switch.
        return {
            "switch_house": True,
            "task_type": "pick",
            "language": "Switch to a different house — current one has no actionable targets.",
            "reasoning": "Deterministic fallback after LLM rejections; no feasible task in inventory.",
            "expected_new_skills": [],
            "novelty_score": 0.0,
            "difficulty_estimate": "easy",
        }

    @staticmethod
    def _build_open_proposal_dict(
        result: dict[str, Any],
        *,
        inventory: dict[str, Any],
        display_to_internal: dict[str, str],
        scene_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert validated LLM JSON into the proposal dict the lifelong
        loop expects. Stamps `_request_house_switch` + the resolved
        `internal_name`s for `set_task_from_spec`.
        """
        switch_house = bool(result.get("switch_house", False))
        task_type = str(result.get("task_type") or "pick").lower()
        scene_family = str(inventory.get("house_index") or "house")
        if switch_house:
            language = result.get("language") or "Switch to a new house and explore."
            activity = f"molmospaces:open:switch_house:{scene_family}"
            return {
                "mode": "novel",
                "activity_name": activity,
                "canonical_task_id": activity,
                "language": language,
                "task_family": "switch_house",
                "scene_family": str(scene_family),
                "benchmark": str(inventory.get("scene_dataset", "")),
                "scene_model": scene_context.get("scene_model", "molmospaces_scene"),
                "objects": [],
                "activity_definition_id": 0,
                "goal_conditions": language,
                "reasoning": result.get("reasoning", ""),
                "expected_new_skills": result.get("expected_new_skills", []),
                "novelty_score": result.get("novelty_score", 0.5),
                "difficulty_estimate": result.get("difficulty_estimate", "medium"),
                "_request_house_switch": True,
                # Preserve the requested smoke/curriculum family so a house
                # switch can reinitialize the bridge to that sampler before
                # drawing the first task in the new house. Without this, a
                # forced `open` switch after a failed pick_and_place iteration
                # would keep the pick_and_place sampler active.
                "_molmospaces_requested_task_type": task_type,
                "_molmospaces_open_spec": None,
                "_molmospaces_proposer_mode": "open",
            }
        # Resolve display names back to internal names for the bridge.
        target_internal = display_to_internal.get(
            str(result.get("target_object_display_name") or "")
        )
        place_internal = display_to_internal.get(
            str(result.get("place_receptacle_display_name") or "")
        )
        joint_internal = display_to_internal.get(
            str(result.get("joint_object_display_name") or "")
        )
        joint_index = result.get("joint_index")
        try:
            joint_index = int(joint_index) if joint_index is not None else None
        except (TypeError, ValueError):
            joint_index = None
        language = result.get("language") or task_type.capitalize()
        # Stable id so failure_memory / curiosity scoring can dedupe.
        target_token = (target_internal or joint_internal or "task").split("/")[-1]
        activity = (
            f"molmospaces:open:{task_type}:{scene_family}:"
            f"{target_token}:{joint_index if joint_index is not None else 'na'}"
        )
        objects = [o for o in (target_internal, place_internal, joint_internal) if o]
        return {
            "mode": "novel",
            "activity_name": activity,
            "canonical_task_id": activity,
            "language": language,
            "task_family": task_type,
            "scene_family": str(scene_family),
            "benchmark": str(inventory.get("scene_dataset", "")),
            "scene_model": scene_context.get("scene_model", "molmospaces_scene"),
            "objects": objects,
            "activity_definition_id": 0,
            "goal_conditions": language,
            "reasoning": result.get("reasoning", ""),
            "expected_new_skills": result.get("expected_new_skills", []),
            "novelty_score": result.get("novelty_score", 0.5),
            "difficulty_estimate": result.get("difficulty_estimate", "medium"),
            "_request_house_switch": False,
            "_molmospaces_open_spec": {
                "task_type": task_type,
                "target_internal_name": target_internal,
                "place_receptacle_internal_name": place_internal,
                "joint_internal_name": joint_internal,
                "joint_index": joint_index,
            },
            "_molmospaces_proposer_mode": "open",
        }

    # ------------------------------------------------------------------
    # Catalog-based proposal (select from predefined tasks)
    # ------------------------------------------------------------------

    def _propose_catalog(
        self,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Select a task from the predefined catalog."""
        prompt_template = Path("rats/prompts/task_proposer.txt").read_text()

        # Format task history
        history_str = "None yet." if not self._task_history else json.dumps(
            self._task_history[-10:], indent=2  # last 10 tasks
        )

        # Format available tasks -- MUST only contain activity_names from the catalog
        if self._available_tasks:
            task_entries = [
                {
                    "activity_name": task["activity_name"],
                    "instruction": task.get("goal_conditions")
                    or task.get("language")
                    or task["activity_name"].replace("_", " "),
                }
                for task in self._available_tasks
            ]
            tasks_str = json.dumps(task_entries, indent=2)
        else:
            tasks_str = "No catalog available. Propose based on scene context."

        # Phase 3.1: Compute curiosity scores for available tasks
        curiosity_hint = ""
        if self._available_tasks and len(self._available_tasks) > 1:
            scored_tasks = self.compute_curiosity_scores(self._available_tasks, skill_context)
            top_scored = scored_tasks[:5]
            if top_scored:
                score_lines = []
                for t in top_scored:
                    name = t.get("activity_name", "?")
                    score_lines.append(
                        f"  {name}: curiosity={t.get('curiosity_score', 0):.2f} "
                        f"(novelty={t.get('novelty', 0):.2f}, learnability={t.get('learnability', 0):.2f})"
                    )
                curiosity_hint = (
                    "\nCURIOSITY SCORES (prefer high-scoring tasks in the 'zone of proximal development'):\n"
                    + "\n".join(score_lines)
                )

        prompt_skill_context = {
            key: value
            for key, value in skill_context.items()
            if not str(key).startswith("_")
        }
        user_prompt = prompt_template.replace(
            "{skill_context}", json.dumps(prompt_skill_context, indent=2)
        ).replace(
            "{task_history}", history_str
        ).replace(
            "{available_tasks}", tasks_str
        )

        if curiosity_hint:
            user_prompt += curiosity_hint
        curriculum_hint = self._curriculum_hint()
        if curriculum_hint:
            user_prompt += "\n\n" + curriculum_hint

        # Add current activity context so the LLM knows what's loaded
        current_activity = scene_context.get("current_activity", "")
        if current_activity:
            user_prompt += (
                f"\n\nCURRENT SCENE: {scene_context.get('scene_model', 'unknown')}"
                f"\nCURRENTLY LOADED TASK: {current_activity}"
                f"\nYou MUST select a task from the AVAILABLE TASKS list above. "
                f"Do NOT invent task names. The selected_task must exactly match "
                f"one of the listed activity names."
            )

        system_prompt = (
            "You are a curiosity-driven task proposer for a robot learning system. "
            "Propose a NOVEL manipulation task. Respond only in valid JSON."
        )

        result = query_llm_json(system_prompt, user_prompt)

        # Build structured proposal
        selected_task = result.get("selected_task", "")

        # Validate: selected task MUST be in the available tasks list
        valid_names = {t["activity_name"] for t in self._available_tasks}
        if selected_task not in valid_names:
            # LLM hallucinated a task name -- find closest match or fall back
            attempted = {h["activity_name"] for h in self._task_history}
            selected_task = ""
            for t in self._available_tasks:
                if t["activity_name"] not in attempted:
                    selected_task = t["activity_name"]
                    break
            if not selected_task and self._available_tasks:
                selected_task = self._available_tasks[0]["activity_name"]

        # Find matching task metadata from catalog
        task_meta = next(
            (t for t in self._available_tasks if t["activity_name"] == selected_task),
            {},
        )

        proposal = {
            "mode": "catalog",
            "activity_name": selected_task,
            "scene_model": task_meta.get("scene_model", scene_context.get("scene_model", "unknown")),
            "activity_definition_id": task_meta.get("activity_definition_id", 0),
            "goal_conditions": task_meta.get("goal_conditions", selected_task.replace("_", " ")),
            "reasoning": result.get("reasoning", ""),
            "expected_new_skills": result.get("expected_new_skills", []),
            "novelty_score": result.get("novelty_score", 0.5),
            "difficulty_estimate": result.get("difficulty_estimate", "medium"),
        }
        return proposal

    def record_task_outcome(
        self,
        activity_name: str,
        success: bool,
        retries: int,
        skills_learned: list[str],
        *,
        failure_reason: str = "",
        env_created: bool = True,
        language: str = "",
        objects_used: list[str] | None = None,
        fixtures_used: list[str] | None = None,
        goal_predicates: list | None = None,
        play_mode: bool = False,
        play_verb: str = "",
        play_target: str = "",
        resulting_state: dict[str, Any] | None = None,
        playtime_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record task outcome with rich failure context for adaptive proposals."""
        entry: dict[str, Any] = {
            "activity_name": activity_name,
            "language": language or activity_name.replace("_", " "),
            "success": success,
            "retries": retries,
            "skills_learned": skills_learned,
            "failure_reason": failure_reason,
            "env_created": env_created,
            "objects_used": objects_used or [],
            "fixtures_used": fixtures_used or [],
            "goal_predicates": goal_predicates or [],
        }
        # Only emit play_* fields when actually in play-mode. Empty defaults
        # were always serialized into every history entry — observed in the
        # LIBERO smoke run as five always-empty fields per non-play task,
        # which add ~30% noise to the task_history block in the proposer's
        # prompt. Keep the structured fields when they carry real values.
        if play_mode or play_verb or play_target or resulting_state or playtime_metadata:
            entry["play_mode"] = play_mode
            entry["play_verb"] = play_verb
            entry["play_target"] = play_target
            entry["resulting_state"] = resulting_state or {}
            entry["playtime_metadata"] = playtime_metadata or {}
        self._task_history.append(entry)

    # ------------------------------------------------------------------
    # Phase 3.1: Curiosity Scoring (novelty x learnability)
    # ------------------------------------------------------------------

    def compute_curiosity_scores(
        self,
        candidates: list[dict[str, Any]],
        skill_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Score candidate tasks by novelty x learnability.

        For each candidate:
          - Decompose into required subskills (heuristic from task name/objects)
          - novelty = fraction of subskills NOT in library
          - learnability = fraction of subskills that ARE in library
          - score = novelty * learnability (zone of proximal development)

        Works for both LIBERO and BEHAVIOR tasks.

        Returns candidates sorted by score (highest first), each with
        curiosity_score, novelty, learnability fields added.
        """
        library_skills = set()
        for s in skill_context.get("primitive_names", []):
            library_skills.add(s.lower())
            # Also add individual words from primitive names for fuzzy matching
            for w in s.lower().replace("_", " ").split():
                if len(w) > 3:
                    library_skills.add(w)
        for s in skill_context.get("learned_skills", []):
            library_skills.add(s["name"].lower())
            for w in s["name"].lower().replace("_", " ").split():
                if len(w) > 3:
                    library_skills.add(w)
            # Also add description keywords as implicit coverage
            desc_words = s.get("description", "").lower().split()
            for w in desc_words:
                if len(w) > 4:
                    library_skills.add(w)

        scored = []
        for candidate in candidates:
            subskills = self._decompose_subskills(candidate)
            if not subskills:
                scored.append({**candidate, "novelty": 0.5, "learnability": 0.5, "curiosity_score": 0.25})
                continue

            covered = sum(1 for s in subskills if s.lower() in library_skills)
            total = len(subskills)
            learnability = covered / total
            novelty = 1.0 - learnability
            score = novelty * learnability

            scored.append({
                **candidate,
                "novelty": round(novelty, 3),
                "learnability": round(learnability, 3),
                "curiosity_score": round(score, 3),
            })

        scored.sort(key=lambda x: x["curiosity_score"], reverse=True)

        # Log scores
        for c in scored[:5]:
            name = c.get("activity_name", c.get("language", "?"))[:40]
            logger.info(
                f"  Curiosity: {name} -> "
                f"N={c['novelty']:.2f} L={c['learnability']:.2f} "
                f"score={c['curiosity_score']:.3f}"
            )

        return scored

    def _decompose_subskills(self, task: dict[str, Any]) -> list[str]:
        """Heuristic decomposition of a task into required subskills.

        Uses task name, objects, and goal predicates to infer needed skills.
        """
        subskills: list[str] = []
        name = task.get("activity_name", "").lower().replace("_", " ")
        language = task.get("language", name).lower()
        text = f"{name} {language}"

        # Action keywords -> subskills
        action_map = {
            "pick": ["grasp", "navigate", "observe"],
            "place": ["grasp", "navigate", "place", "observe"],
            "put": ["grasp", "navigate", "place", "observe"],
            "open": ["navigate", "observe", "open_fixture"],
            "close": ["navigate", "observe", "close_fixture"],
            "turn on": ["navigate", "observe", "actuate"],
            "turnon": ["navigate", "observe", "actuate"],
            "stack": ["grasp", "navigate", "place", "observe", "stack"],
            "clean": ["grasp", "navigate", "place", "observe"],
            "move": ["grasp", "navigate", "place", "observe"],
            "carry": ["grasp", "navigate", "place", "observe"],
            "load": ["grasp", "navigate", "place", "observe"],
            "store": ["grasp", "navigate", "place", "observe", "open_fixture"],
        }

        for keyword, skills in action_map.items():
            if keyword in text:
                subskills.extend(skills)

        # Objects add observation/grasping subskills
        objects = task.get("objects", [])
        for obj in objects:
            subskills.append(f"handle_{obj}")

        # Goal predicates add specific subskills
        for pred in task.get("goal", task.get("goal_predicates", [])):
            if isinstance(pred, list) and pred:
                pred_name = pred[0].lower()
                if pred_name == "on":
                    subskills.append("place")
                elif pred_name == "in":
                    subskills.extend(["place", "open_fixture"])
                elif pred_name in ("turnon", "open", "close"):
                    subskills.append(pred_name)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in subskills:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        # Fallback: if no subskills detected, use generic ones
        if not unique:
            unique = ["observe", "navigate", "grasp", "place"]

        return unique

    def get_task_history(self) -> list[dict[str, Any]]:
        return list(self._task_history)
