"""Memory Curator: periodically prunes / merges / rewrites distilled lessons.

Complements FailureMemory's distiller. Distillation ADDS lessons from
observed failure groups; over time it produces many near-duplicates and
vague prose ("verify before grasping") that poison retrieval. The curator
runs every N iterations, reads current lessons + usage stats + recent
outcomes, and emits MERGE / DELETE / REWRITE / NOOP actions to keep the
library small, specific, and load-bearing.

Invoked by loop/lifelong_loop.py every `curate_every` iterations. It follows
the same run-level model override as the rest of RATS unless a caller passes a
model explicitly.

All actions are logged to `<output_dir>/memory/curator_history.json` for
audit / debugging.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("rats.memory_curator")


class MemoryCurator:
    def __init__(
        self,
        prompt_path: str | Path = "rats/prompts/memory_curator.txt",
        skill_prompt_path: str | Path = "rats/prompts/skill_curator.txt",
        model: str | None = None,
        max_lessons_shown: int = 40,
        max_skills_shown: int = 60,
        max_actions_per_call: int = 5,
        max_skill_actions_per_call: int = 8,
    ) -> None:
        self.prompt_path = Path(prompt_path)
        self.skill_prompt_path = Path(skill_prompt_path)
        # Follow the run-level model by default. `scripts/run_rats.py --model`
        # sets both RATS_LLM_MODEL and RATS_CURATOR_MODEL so curation cannot
        # silently use a different model during the same experiment.
        from rats.agents.base_agent import get_default_model

        self.model = (
            model
            or os.getenv("RATS_CURATOR_MODEL")
            or os.getenv("RATS_LLM_MODEL")
            or get_default_model()
        )
        self.max_lessons_shown = max_lessons_shown
        self.max_skills_shown = max_skills_shown
        self.max_actions_per_call = max_actions_per_call
        # Skill curation needs more actions per call: a single v10b-style
        # library can easily carry 5-8 near-duplicates to collapse in one
        # sweep, and throttling to 5 strands half of them for another tick.
        self.max_skill_actions_per_call = max_skill_actions_per_call

    def curate(
        self,
        failure_memory,  # FailureMemory instance; typed loosely to avoid import cycle
        recent_iteration_outcomes: list[dict[str, Any]],
        history_path: Path | None = None,
    ) -> dict[str, Any]:
        """Inspect lessons + recent outcomes, apply actions, persist audit log.

        Returns summary dict with counts of actions taken.
        """
        lessons = list(failure_memory._lessons or [])
        if not lessons:
            return {"actions_applied": 0, "reason": "no lessons yet"}

        # Build prompt payload
        try:
            system_prompt = self.prompt_path.read_text()
        except Exception as e:
            logger.warning(f"curator prompt missing: {e}")
            return {"actions_applied": 0, "reason": "prompt missing"}

        # Index source episodes by ID for cheap lookup when building the
        # lesson payload below. Each distilled lesson carries an
        # `evidence` list of episode_ids (see failure_memory._distill_
        # one_group ~line 534); surfacing those episodes' actual content
        # is what lets the curator do *evidence-based* DELETE / MERGE /
        # REWRITE as the prompt requires.
        episode_by_id = {
            ep.get("episode_id"): ep
            for ep in (failure_memory._episodes or [])
            if isinstance(ep, dict) and ep.get("episode_id")
        }

        def _episode_summary(ep_id: str) -> dict[str, Any] | None:
            ep = episode_by_id.get(ep_id)
            if not ep:
                return None
            return {
                "episode_id": ep_id,
                "task_name": ep.get("task_name", ""),
                "failure_category": ep.get("failure_category", ""),
                # Full diagnosis (storage caps live upstream); top_k=3
                # source episodes per lesson keeps total token cost bounded.
                "diagnosis_summary": ep.get("diagnosis_summary", ""),
                "objects_involved": ep.get("objects_involved", []) or [],
            }

        # Include usage stats + keep the most-active/most-recent lessons first
        lesson_payload = []
        for l in sorted(
            lessons,
            key=lambda x: (x.get("times_applied", 0), x.get("confidence", 0)),
            reverse=True,
        )[: self.max_lessons_shown]:
            evidence_ids = list(l.get("evidence", []) or [])[:5]
            source_eps = [
                summary
                for ep_id in evidence_ids[:3]  # 3 per lesson keeps tokens bounded
                if (summary := _episode_summary(ep_id)) is not None
            ]
            # Per-application history — appended on every
            # retrieve_for_policy_writer -> attempt-resolved cycle.
            # Lets the curator see "applied 5 times, all failed" with
            # actual per-iteration evidence instead of relying on the
            # times_applied / times_helped aggregates alone.
            recent_apps = list(l.get("recent_applications", []) or [])[-5:]
            lesson_payload.append({
                "lesson_id": l.get("lesson_id"),
                "condition": l.get("condition", ""),
                "antipattern": l.get("antipattern", ""),
                "remedy": l.get("remedy", ""),
                "applicable_objects": l.get("applicable_to", {}).get("objects", []),
                "applicable_actions": l.get("applicable_to", {}).get("actions", []),
                "times_applied": l.get("times_applied", 0),
                "times_helped": l.get("times_helped", 0),
                "source": l.get("source", "distilled"),
                # NEW — evidence surface for the curator.
                "source_episode_ids": evidence_ids,
                "source_episodes": source_eps,
                "recent_applications": recent_apps,
            })

        user_prompt = (
            f"CURRENT DISTILLED LESSONS ({len(lessons)} total, showing top "
            f"{len(lesson_payload)} by usage):\n"
            + json.dumps(lesson_payload, indent=2)
            + "\n\n"
            + "RECENT ITERATION OUTCOMES (most recent last):\n"
            + json.dumps(recent_iteration_outcomes[-15:], indent=2)
            + "\n\n"
            + f"Propose up to {self.max_actions_per_call} curation actions. "
              "Prefer MERGE over DELETE when the underlying insight is correct "
              "but duplicated. Use REWRITE to replace vague prose with primitive-"
              "named specifics when evidence supports it. NOOP is valid and "
              "preferred to busywork."
        )

        try:
            from rats.agents.base_agent import query_llm_json
            # gpt-5.4 is a reasoning model — early experiments showed it
            # burning the whole max_tokens budget on internal reasoning and
            # returning empty content (finish_reason=length). Give it 6000
            # so reasoning + JSON output both fit.
            result = query_llm_json(
                system_prompt, user_prompt,
                model=self.model, max_tokens=6000, temperature=0.2,
            )
        except Exception as e:
            logger.warning(f"curator LLM call failed (non-fatal): {e}")
            return {"actions_applied": 0, "reason": f"llm_error: {e}"}

        actions = result.get("actions", []) or []
        if not isinstance(actions, list):
            return {"actions_applied": 0, "reason": "malformed response"}

        # Apply each action
        summary = {"merge": 0, "delete": 0, "rewrite": 0, "noop": 0, "errors": 0}
        applied_record: list[dict[str, Any]] = []
        for action in actions[: self.max_actions_per_call]:
            if not isinstance(action, dict):
                summary["errors"] += 1
                continue
            op = (action.get("op") or "").upper()
            try:
                if op == "MERGE":
                    ok = self._apply_merge(failure_memory, action)
                    summary["merge"] += int(ok)
                    if ok:
                        applied_record.append(action)
                elif op == "DELETE":
                    n = self._apply_delete(failure_memory, action)
                    summary["delete"] += n
                    if n:
                        applied_record.append(action)
                elif op == "REWRITE":
                    ok = self._apply_rewrite(failure_memory, action)
                    summary["rewrite"] += int(ok)
                    if ok:
                        applied_record.append(action)
                elif op == "NOOP":
                    summary["noop"] += 1
                else:
                    logger.info(f"  curator: unknown op {op!r}, skipping")
                    summary["errors"] += 1
            except Exception as e:
                logger.warning(f"  curator action failed ({op}): {e}")
                summary["errors"] += 1

        if summary["merge"] or summary["delete"] or summary["rewrite"]:
            failure_memory._save_lessons()

        # Audit log
        if history_path is not None:
            try:
                history_path.parent.mkdir(parents=True, exist_ok=True)
                existing: list[dict[str, Any]] = []
                if history_path.exists():
                    try:
                        existing = json.loads(history_path.read_text())
                    except Exception:
                        existing = []
                existing.append({
                    "timestamp": time.time(),
                    "model": self.model,
                    "summary": summary,
                    "actions": applied_record,
                })
                history_path.write_text(json.dumps(existing, indent=2))
            except Exception as e:
                logger.debug(f"  curator history write failed: {e}")

        summary["actions_applied"] = (
            summary["merge"] + summary["delete"] + summary["rewrite"]
        )
        return summary

    # ------------------------------------------------------------------
    # Low-level action handlers; mutate failure_memory._lessons in place.
    # ------------------------------------------------------------------

    def _apply_merge(self, fm, action: dict[str, Any]) -> bool:
        from_ids = action.get("from_ids") or []
        new_lesson_fields = action.get("new_lesson") or {}
        if not from_ids or not new_lesson_fields:
            return False
        lessons = fm._lessons
        kept = [l for l in lessons if l.get("lesson_id") not in from_ids]
        removed = [l for l in lessons if l.get("lesson_id") in from_ids]
        if not removed:
            return False
        # Build new merged lesson
        import uuid as _uuid
        condition = (new_lesson_fields.get("condition") or "").strip()
        remedy = (new_lesson_fields.get("remedy") or "").strip()
        antipattern = (new_lesson_fields.get("antipattern") or "").strip()
        if not (condition and remedy):
            return False
        description = f"WHEN {condition} | WRONG: {antipattern} | DO: {remedy}"[:600]
        merged_applied = sum(l.get("times_applied", 0) for l in removed)
        merged_helped = sum(l.get("times_helped", 0) for l in removed)
        evidence = []
        for l in removed:
            evidence.extend(l.get("evidence", []) or [])
        new_lesson = {
            "lesson_id": f"les_m_{_uuid.uuid4().hex[:8]}",
            "description": description,
            "condition": condition,
            "antipattern": antipattern,
            "remedy": remedy,
            "applicable_to": {
                "objects": new_lesson_fields.get("applicable_objects", []) or [],
                "actions": new_lesson_fields.get("applicable_actions", []) or [],
                "task_types": [],
            },
            "evidence": evidence,
            "confidence": max((l.get("confidence", 0.5) for l in removed), default=0.5),
            "times_applied": merged_applied,
            "times_helped": merged_helped,
            "source": "curator_merge",
        }
        kept.append(new_lesson)
        fm._lessons = kept
        return True

    def _apply_delete(self, fm, action: dict[str, Any]) -> int:
        ids = set(action.get("lesson_ids") or [])
        if not ids:
            return 0
        before = len(fm._lessons)
        fm._lessons = [l for l in fm._lessons if l.get("lesson_id") not in ids]
        return before - len(fm._lessons)

    def _apply_rewrite(self, fm, action: dict[str, Any]) -> bool:
        lid = action.get("lesson_id")
        new_fields = action.get("new_fields") or {}
        if not lid or not new_fields:
            return False
        for l in fm._lessons:
            if l.get("lesson_id") == lid:
                condition = (new_fields.get("condition") or l.get("condition", "")).strip()
                remedy = (new_fields.get("remedy") or l.get("remedy", "")).strip()
                antipattern = (new_fields.get("antipattern") or l.get("antipattern", "")).strip()
                if not (condition and remedy):
                    return False
                l["condition"] = condition
                l["remedy"] = remedy
                l["antipattern"] = antipattern
                l["description"] = (
                    f"WHEN {condition} | WRONG: {antipattern} | DO: {remedy}"
                )[:600]
                applied_to = new_fields.get("applicable_objects")
                actions = new_fields.get("applicable_actions")
                if applied_to is not None or actions is not None:
                    app = l.setdefault("applicable_to", {})
                    if applied_to is not None:
                        app["objects"] = applied_to
                    if actions is not None:
                        app["actions"] = actions
                l["source"] = "curator_rewrite"
                return True
        return False

    # ------------------------------------------------------------------
    # Skill-library curation. Complements the lesson curator above: the
    # same LLM + audit log pattern, but against SkillLibrary's learned
    # skills. Exists because ingest-time semantic dedup (library.add_skill)
    # only fires against the current candidate — once a near-duplicate
    # slips in under a slightly different name, nothing looks at it again
    # and the library sprawls (v10b: 29 learned skills, ~6 distinct
    # procedures, Planner drowning in variants).
    # ------------------------------------------------------------------

    def curate_skills(
        self,
        skill_library,  # SkillLibrary instance; typed loosely to avoid import cycle
        recent_iteration_outcomes: list[dict[str, Any]],
        history_path: Path | None = None,
        current_iteration: int | None = None,
    ) -> dict[str, Any]:
        """Inspect learned skills, collapse near-duplicates, record audit log.

        Sends FULL code (not just preview) so the curator can also author
        REWRITE actions that lift hardcoded literals to parameters.
        """
        skills = skill_library.get_learned_skills_for_curator(include_full_code=True)
        if not skills:
            return {"actions_applied": 0, "reason": "no learned skills yet"}

        try:
            system_prompt = self.skill_prompt_path.read_text()
        except Exception as e:
            logger.warning(f"skill curator prompt missing: {e}")
            return {"actions_applied": 0, "reason": "prompt missing"}

        # Rank by usage/rediscovery first so the slice we send the LLM is the
        # most load-bearing subset when the library is bigger than the budget.
        ranked = sorted(
            skills,
            key=lambda s: (
                int(s.get("usage_count", 0)),
                int(s.get("rediscovery_count", 0)),
                int(s.get("success_count", 0)),
            ),
            reverse=True,
        )[: self.max_skills_shown]

        user_prompt = (
            f"CURRENT LEARNED SKILLS ({len(skills)} total, showing top "
            f"{len(ranked)} by usage):\n"
            + json.dumps(ranked, indent=2)
            + "\n\n"
            + "RECENT ITERATION OUTCOMES (most recent last):\n"
            + json.dumps(recent_iteration_outcomes[-15:], indent=2)
            + "\n\n"
            + f"Propose up to {self.max_skill_actions_per_call} curation "
              "actions (MERGE / DEPRECATE / REWRITE / NOOP). Prefer MERGE "
              "when the same procedure has been extracted under multiple "
              "names. Use REWRITE when a single skill is correct but tied "
              "to a hardcoded object/literal/offset that, lifted to a "
              "parameter with a default, would make the skill reusable on "
              "new tasks. NOOP is valid and preferred to busywork, but if "
              "you see >2 skills in the same name-family (e.g. "
              "pick_X_with_verification for different X) you should be "
              "merging — and if you see a clearly hardcoded skill (e.g. "
              "object_name=\"butter\" baked in), you should be rewriting."
        )

        try:
            from rats.agents.base_agent import query_llm_json
            result = query_llm_json(
                system_prompt, user_prompt,
                model=self.model, max_tokens=6000, temperature=0.2,
            )
        except Exception as e:
            logger.warning(f"skill curator LLM call failed (non-fatal): {e}")
            return {"actions_applied": 0, "reason": f"llm_error: {e}"}

        actions = result.get("actions", []) or []
        if not isinstance(actions, list):
            return {"actions_applied": 0, "reason": "malformed response"}

        summary = {
            "merge": 0,
            "deprecate": 0,
            "rewrite": 0,
            "noop": 0,
            "errors": 0,
        }
        applied_record: list[dict[str, Any]] = []
        lifecycle_events: list[dict[str, Any]] = []
        for action in actions[: self.max_skill_actions_per_call]:
            if not isinstance(action, dict):
                summary["errors"] += 1
                continue
            op = (action.get("op") or "").upper()
            try:
                if op == "MERGE":
                    dup = (action.get("duplicate") or "").strip()
                    canon = (action.get("canonical") or "").strip()
                    rationale = (action.get("rationale") or "").strip()
                    ok = skill_library.mark_duplicate(
                        dup,
                        canon,
                        rationale,
                        iteration=current_iteration,
                        source="skill_curator",
                    )
                    summary["merge"] += int(ok)
                    if ok:
                        applied_record.append(action)
                        lifecycle_events.append({
                            "iteration": current_iteration,
                            "action": "deprecated",
                            "skill": dup,
                            "duplicate_of": canon,
                            "reason": rationale,
                            "source": "skill_curator",
                        })
                elif op == "DEPRECATE":
                    name = (action.get("skill") or "").strip()
                    reason = (action.get("reason") or "").strip()
                    ok = skill_library.deprecate_by_curator(
                        name,
                        reason,
                        iteration=current_iteration,
                        source="skill_curator",
                    )
                    summary["deprecate"] += int(ok)
                    if ok:
                        applied_record.append(action)
                        lifecycle_events.append({
                            "iteration": current_iteration,
                            "action": "deprecated",
                            "skill": name,
                            "reason": reason,
                            "source": "skill_curator",
                        })
                elif op == "REWRITE":
                    name = (action.get("skill") or "").strip()
                    new_code = action.get("new_code") or ""
                    new_desc = (action.get("new_description") or "").strip()
                    rationale = (action.get("rationale") or "").strip()
                    ok = skill_library.rewrite_skill_code(
                        name, new_code, new_desc, rationale,
                    )
                    summary["rewrite"] += int(ok)
                    if ok:
                        # Don't echo the full new_code into audit; keep the
                        # log compact. The library already saved the
                        # previous body to rewrite_history.
                        applied_record.append({
                            "op": "REWRITE",
                            "skill": name,
                            "rationale": rationale,
                            "new_description": new_desc,
                        })
                elif op == "NOOP":
                    summary["noop"] += 1
                else:
                    logger.info(
                        f"  skill curator: unknown op {op!r}, skipping"
                    )
                    summary["errors"] += 1
            except Exception as e:
                logger.warning(f"  skill curator action failed ({op}): {e}")
                summary["errors"] += 1

        # Audit log — written to the *skill* curator history so it doesn't
        # tangle with the lesson curator's audit stream.
        if history_path is not None:
            try:
                history_path.parent.mkdir(parents=True, exist_ok=True)
                existing: list[dict[str, Any]] = []
                if history_path.exists():
                    try:
                        existing = json.loads(history_path.read_text())
                    except Exception:
                        existing = []
                existing.append({
                    "timestamp": time.time(),
                    "model": self.model,
                    "summary": summary,
                    "actions": applied_record,
                })
                history_path.write_text(json.dumps(existing, indent=2))
            except Exception as e:
                logger.debug(f"  skill curator history write failed: {e}")

        summary["actions_applied"] = (
            summary["merge"] + summary["deprecate"] + summary["rewrite"]
        )
        summary["lifecycle_events"] = lifecycle_events
        return summary
