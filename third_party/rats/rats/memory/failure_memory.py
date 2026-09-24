"""Failure Memory: persistent memory of past failures for RATS.

Two-layer memory:
  Layer 1 -- Failure Episode Store (raw, factual)
    Records every terminal failure with structured metadata.
  Layer 2 -- Distilled Lessons (compressed, actionable)
    Periodically LLM-compressed rules from raw episodes.

Retrieval is keyword/tag-based (no embeddings). All methods degrade
gracefully when the store is empty (first run).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("rats.failure_memory")


class FailureMemory:
    """Persistent failure memory for the RATS lifelong loop."""

    def __init__(self, storage_dir: str | Path) -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._episodes_path = self._storage_dir / "episodes.json"
        self._lessons_path = self._storage_dir / "lessons.json"

        self._episodes: list[dict[str, Any]] = []
        self._lessons: list[dict[str, Any]] = []
        self._episodes_since_last_distill = 0
        self._last_distill_iteration = 0

        self._load()

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def _load(self) -> None:
        if self._episodes_path.exists():
            try:
                self._episodes = json.loads(self._episodes_path.read_text())
            except Exception:
                self._episodes = []
        if self._lessons_path.exists():
            try:
                self._lessons = json.loads(self._lessons_path.read_text())
            except Exception:
                self._lessons = []

    def _save_episodes(self) -> None:
        self._episodes_path.write_text(json.dumps(self._episodes, indent=2))

    def _save_lessons(self) -> None:
        self._lessons_path.write_text(json.dumps(self._lessons, indent=2))

    # ----------------------------------------------------------------
    # 1B.1  Record failures
    # ----------------------------------------------------------------

    def record_failure(
        self,
        *,
        task_name: str,
        scene: str = "",
        objects_involved: list[str] | None = None,
        failure_category: str = "unknown",
        diagnosis_summary: str = "",
        code_snippet: str = "",
        failed_step: str = "",
        approaches_tried: list[str] | None = None,
        retry_count: int = 0,
        max_reward: float = 0.0,
        iteration: int | None = None,
    ) -> str:
        """Record a terminal failure episode.

        Returns the episode_id.
        """
        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        # FIX (storage truncations were too aggressive): previously
        # `diagnosis_summary[:300]` and `code_snippet[:500]` truncated at
        # the storage layer, so EVERY downstream consumer (planner, distill,
        # retry prompt) inherited the loss — they couldn't recover what was
        # never persisted. A typical diagnosis is 500-2000 chars and a
        # typical policy is 1500-3000 chars, so 300/500 amounted to "throw
        # away most of the signal at write time."
        # New caps are deliberately above the realistic upper end of each
        # field so the natural shape of the content is preserved; only
        # truly pathological inputs get touched.
        episode = {
            "episode_id": episode_id,
            "task_name": task_name,
            "scene": scene,
            "objects_involved": objects_involved or [],
            "failure_category": failure_category,
            # origin/main went one step further than the audit on this
            # branch — they dropped the 8000-char caps entirely. Same
            # direction as the audit ("don't truncate downstream of
            # storage either"); accept the untrimmed version and keep
            # their new `failed_step` field.
            "failed_step": failed_step,
            "diagnosis_summary": diagnosis_summary,
            "code_snippet": code_snippet,
            "approaches_tried": approaches_tried or [],
            "retry_count": retry_count,
            "max_reward": max_reward,
            # Bind failures to the development transaction that produced
            # them. Resume can then discard an interrupted iteration without
            # relying on task names or wall-clock heuristics.
            "iteration": iteration,
            "timestamp": time.time(),
        }
        self._episodes.append(episode)
        self._episodes_since_last_distill += 1
        self._save_episodes()
        logger.info(
            f"Recorded failure episode {episode_id}: {task_name} "
            f"({failure_category}), {len(self._episodes)} total"
        )
        return episode_id

    # ----------------------------------------------------------------
    # 1B.2  Retrieval for Policy Writer
    # ----------------------------------------------------------------

    @staticmethod
    def _format_lesson_metadata(
        lesson: dict[str, Any],
        current_iteration: int | None = None,
    ) -> str:
        """Render a one-line metadata footer for a lesson.

        Includes confidence, times_applied / times_helped (the empirical
        reliability the curator has been tracking), and staleness signals
        (last_helped iteration, distilled iteration). The formatter is
        defensive about missing fields so older lessons that lack
        `distilled_at_iteration` or `recent_applications` still render.

        Layout: ``(confidence=0.80, applied=5× helped=1×, last_helped=iter4 (2 iters ago), distilled=iter2 (4 iters ago))``
        """
        bits: list[str] = []
        try:
            conf = float(lesson.get("confidence", 0.0))
            bits.append(f"confidence={conf:.2f}")
        except Exception:
            pass
        applied = int(lesson.get("times_applied", 0) or 0)
        helped = int(lesson.get("times_helped", 0) or 0)
        bits.append(f"applied={applied}× helped={helped}×")

        # last_helped_iteration is derived from recent_applications
        # (commit 9861587a added that field). Look for the most recent
        # entry with success=True.
        last_helped = None
        for entry in reversed(lesson.get("recent_applications") or []):
            if isinstance(entry, dict) and entry.get("success"):
                last_helped = entry.get("iteration")
                break

        def _stale_suffix(iter_at: Any) -> str:
            if iter_at is None or current_iteration is None:
                return ""
            try:
                delta = int(current_iteration) - int(iter_at)
            except Exception:
                return ""
            if delta <= 0:
                return ""
            return f" ({delta} iters ago)"

        if last_helped is not None:
            bits.append(f"last_helped=iter{last_helped}{_stale_suffix(last_helped)}")
        else:
            bits.append("last_helped=never")

        distilled = lesson.get("distilled_at_iteration")
        if distilled is not None:
            bits.append(f"distilled=iter{distilled}{_stale_suffix(distilled)}")

        return "(" + ", ".join(bits) + ")"

    def record_outcome_for_last_served_lessons(
        self,
        *,
        success: bool,
        iteration: int | None = None,
        task_name: str = "",
        attempt_idx: int | None = None,
    ) -> None:
        """After an attempt resolves, record the outcome on every lesson
        that was surfaced in the most recent retrieve_for_policy_writer
        call. This is the only signal the MemoryCurator has for "does
        this lesson actually help downstream attempts."

        Two things happen per affected lesson:

          1. Aggregate counter ``times_helped`` is incremented on
             success (legacy behaviour).
          2. Per-attempt entry appended to ``recent_applications`` —
             ``{iteration, task_name, attempt_idx, success}`` — capped
             at the last 10 entries. Lets the curator see "applied 5
             times, all failed" vs "applied 5 times, helped 4×" with
             real per-call evidence rather than aggregate counts.

        Soft heuristic: "helped" still means "was shown right before
        an attempt that then succeeded", not "causally caused the
        success."
        """
        ids = getattr(self, "_last_served_lesson_ids", None) or []
        if not ids:
            return
        application_entry = {
            "iteration": iteration,
            "task_name": task_name,
            "attempt_idx": attempt_idx,
            "success": bool(success),
        }
        for lesson in self._lessons:
            if lesson.get("lesson_id") not in ids:
                continue
            if success:
                lesson["times_helped"] = lesson.get("times_helped", 0) + 1
            recent = lesson.setdefault("recent_applications", [])
            recent.append(application_entry)
            # Cap to last 10 to keep lessons.json bounded.
            if len(recent) > 10:
                lesson["recent_applications"] = recent[-10:]
        self._save_lessons()

    def retrieve_for_policy_writer(
        self,
        task_name: str,
        objects: list[str] | None = None,
        failure_category: str | None = None,
        top_k: int = 3,
        *,
        current_iteration: int | None = None,
    ) -> str:
        """Retrieve relevant past failures formatted for the Policy Writer.

        Ranking (in priority order):
          1. Object overlap (shared objects with the new task)
          2. Failure category match
          3. Task-type similarity (shared prefix/words in task name)

        Returns a compact string (~300 tokens) ready for prompt injection,
        or empty string if no relevant failures found.
        """
        if not self._episodes and not self._lessons:
            return ""

        objects_lower = [o.lower() for o in (objects or [])]
        task_words = set(task_name.lower().replace("_", " ").split())

        # --- Layer 2: distilled lessons (prepended when relevant) ---
        lesson_lines: list[str] = []
        self._last_served_lesson_ids: list[str] = []
        if self._lessons:
            ranked_lessons: list[tuple[int, dict]] = []
            for lesson in self._lessons:
                applicable = lesson.get("applicable_to", {})
                lesson_objs = {o.lower() for o in applicable.get("objects", [])}
                overlap = len(set(objects_lower) & lesson_objs)
                if overlap > 0:
                    ranked_lessons.append((overlap, lesson))
            ranked_lessons.sort(key=lambda x: x[0], reverse=True)
            if ranked_lessons:
                lesson_lines.append("--- DISTILLED LESSONS FROM PRIOR RUNS ---")
                for _, lesson in ranked_lessons[:top_k]:
                    # FIX: was [:160], which lost the DO half of every
                    # WHEN/WRONG/DO lesson. Distilled lessons are LLM-bounded
                    # (~1-2k chars upper end) so emitting the full text is
                    # safe — top_k=3 keeps the total bounded.
                    desc = lesson.get("description", "")
                    if desc:
                        lesson_lines.append(f"- {desc}")
                        # Metadata footer — confidence + applied/helped
                        # counters + staleness signals. Let the LLM weigh
                        # which lessons are reliable vs stale priors.
                        meta = self._format_lesson_metadata(lesson, current_iteration)
                        lesson_lines.append(f"    {meta}")
                    lid = lesson.get("lesson_id")
                    if lid:
                        self._last_served_lesson_ids.append(lid)
                        # Count this serving as an "application" of the lesson
                        lesson["times_applied"] = lesson.get("times_applied", 0) + 1

        # --- Layer 1: raw relevant episodes ---
        scored: list[tuple[float, dict]] = []
        for ep in self._episodes:
            score = 0.0
            ep_objects = [o.lower() for o in ep.get("objects_involved", [])]

            overlap = len(set(objects_lower) & set(ep_objects))
            score += overlap * 3.0

            if failure_category and ep.get("failure_category") == failure_category:
                score += 2.0

            ep_words = set(ep["task_name"].lower().replace("_", " ").split())
            word_overlap = len(task_words & ep_words)
            score += word_overlap * 1.0

            if score > 0:
                scored.append((score, ep))

        episode_lines: list[str] = []
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            episode_lines.append("--- RAW FAILURE EPISODES (most relevant) ---")
            for i, (_, ep) in enumerate(scored[:top_k], 1):
                # NOTE: this branch dropped its earlier [:120] diag slice
                # and [-300:] snippet tail (audit on this branch);
                # origin/main went further and now picks the specific
                # failed-step segment when available. Take origin/main's
                # version — it's strictly more informative than the full
                # snippet when the diagnoser identified a single step.
                diag = ep.get("diagnosis_summary", "no diagnosis")
                episode_lines.append(
                    f"{i}. Task \"{ep['task_name']}\": "
                    f"{ep['failure_category']} -- {diag}"
                )
                snippet = ep.get("code_snippet", "")
                if snippet:
                    failed_step = str(ep.get("failed_step") or "")
                    step_code = _extract_step_segment(snippet, failed_step) or snippet
                    label = (
                        f"diagnosed step {failed_step}"
                        if failed_step else
                        "full failed code"
                    )
                    episode_lines.append(
                        f"   Failed code ({label}):\n```python\n{step_code}\n```"
                    )

        if not lesson_lines and not episode_lines:
            return ""

        out: list[str] = []
        if lesson_lines:
            out.extend(lesson_lines)
        if episode_lines:
            out.extend(episode_lines)
        out.append(
            "Do NOT repeat the failed approaches above. Apply the lessons and try a different strategy."
        )
        return "\n".join(out)

    # ----------------------------------------------------------------
    # 1B.3  Retrieval for Planner
    # ----------------------------------------------------------------

    def get_lessons_for_planner(
        self,
        objects: list[str] | None = None,
        action_types: list[str] | None = None,
        top_k: int = 5,
        *,
        current_iteration: int | None = None,
    ) -> str:
        """Return distilled lessons relevant to the given objects/actions.

        Prefers Layer 2 (distilled lessons) when available, falls back to
        summarizing raw episodes. Returns compact string (~150 tokens).
        """
        objects_lower = {o.lower() for o in (objects or [])}
        actions_lower = {a.lower() for a in (action_types or [])}

        # Try distilled lessons first
        if self._lessons:
            relevant = []
            for lesson in self._lessons:
                applicable = lesson.get("applicable_to", {})
                lesson_objs = {o.lower() for o in applicable.get("objects", [])}
                lesson_acts = {a.lower() for a in applicable.get("actions", [])}
                overlap = len(objects_lower & lesson_objs) + len(actions_lower & lesson_acts)
                if overlap > 0:
                    relevant.append((overlap, lesson))
            relevant.sort(key=lambda x: x[0], reverse=True)
            if relevant:
                lines = ["KNOWN CONSTRAINTS FROM EXPERIENCE:"]
                for _, lesson in relevant[:top_k]:
                    # FIX: was [:400], originally [:80]. Distilled lessons
                    # are LLM-bounded by the distill call's max_tokens — so
                    # the realistic upper bound is ~2k chars. With top_k=3
                    # that's ~6k chars in the planner prompt, which is fine
                    # against an 8k-32k+ context model. Emit the full text.
                    lines.append(f"- {lesson['description']}")
                    # Metadata footer — let the planner downweight lessons
                    # with low helped/applied or stale last_helped.
                    meta = self._format_lesson_metadata(lesson, current_iteration)
                    lines.append(f"    {meta}")
                lines.append("Plan around these constraints.")
                return "\n".join(lines)

        # Fallback: summarize raw episodes
        if not self._episodes:
            return ""

        # Group by failure category for the relevant objects
        relevant_eps = []
        for ep in self._episodes:
            ep_objs = {o.lower() for o in ep.get("objects_involved", [])}
            if ep_objs & objects_lower:
                relevant_eps.append(ep)

        if not relevant_eps:
            return ""

        # Aggregate by category
        cat_counts: dict[str, int] = {}
        cat_summaries: dict[str, str] = {}
        for ep in relevant_eps:
            cat = ep.get("failure_category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            if cat not in cat_summaries:
                # FIX: was [:60], then [:240]. Storage truncations now cap
                # diagnosis_summary at 8000 chars, so the realistic upper
                # bound from a single episode is fine to emit untrimmed —
                # we already top_k=3 the categories.
                cat_summaries[cat] = ep.get("diagnosis_summary", "")

        lines = ["KNOWN CONSTRAINTS FROM EXPERIENCE:"]
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:top_k]:
            lines.append(f"- {cat} occurred {count}x: {cat_summaries[cat]}")
        lines.append("Plan around these constraints.")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # 1B.5  Lesson distillation
    # ----------------------------------------------------------------

    def maybe_distill(self, iteration: int) -> bool:
        """Check if distillation is needed and run if so.

        Triggers every 10 iterations or when 5+ new episodes accumulated.
        Returns True if distillation ran.
        """
        should_distill = (
            self._episodes_since_last_distill >= 2
            or (iteration > 0 and iteration % 5 == 0 and self._episodes_since_last_distill > 0)
        )
        if not should_distill:
            return False

        return self.distill_lessons(iteration=iteration)

    def distill_lessons(self, iteration: int | None = None) -> bool:
        """Compress raw failure episodes into actionable lessons.

        Groups episodes by object/category overlap, sends to LLM for
        rule extraction, deduplicates against existing lessons.

        ``iteration`` (when provided) is stamped onto every new lesson
        as ``distilled_at_iteration`` so downstream prompts can render
        per-lesson staleness alongside confidence and reliability.
        """
        if len(self._episodes) < 2:
            return False

        groups = self._group_episodes()
        if not groups:
            return False

        new_lessons = []
        for group_key, episodes in groups.items():
            if len(episodes) < 2:
                continue
            lesson = self._distill_one_group(group_key, episodes, iteration=iteration)
            if lesson and not self._is_duplicate_lesson(lesson):
                new_lessons.append(lesson)

        if new_lessons:
            self._lessons.extend(new_lessons)
            self._save_lessons()
            self._episodes_since_last_distill = 0
            if iteration is not None:
                self._last_distill_iteration = int(iteration)
            logger.info(f"Distilled {len(new_lessons)} new lessons (total: {len(self._lessons)})")
            return True

        self._episodes_since_last_distill = 0
        if iteration is not None:
            self._last_distill_iteration = int(iteration)
        return False

    def _group_episodes(self) -> dict[str, list[dict]]:
        """Group episodes by overlapping objects or failure categories."""
        groups: dict[str, list[dict]] = {}

        # Group by failure category
        for ep in self._episodes:
            cat = ep.get("failure_category", "unknown")
            if cat not in ("unknown", "none"):
                groups.setdefault(f"cat:{cat}", []).append(ep)

        # Group by most common object
        obj_eps: dict[str, list[dict]] = {}
        for ep in self._episodes:
            for obj in ep.get("objects_involved", []):
                obj_eps.setdefault(obj, []).append(ep)
        for obj, eps in obj_eps.items():
            if len(eps) >= 2:
                groups[f"obj:{obj}"] = eps

        return groups

    # Tool inventory for distillation prompts. The distiller is told these
    # functions exist so the lessons it produces reference real call sites
    # instead of vague prose ("verify the grasp"). Kept small / focused.
    _DISTILL_TOOL_INVENTORY = (
        "Available primitives the policy writer can call:\n"
        "  segment_sam3_text_prompt(rgb, text_prompt) -> [{'mask','score', 'box'}]\n"
        "  segment_sam3_point_prompt(rgb, (x,y))       -> [{'mask','score'}]\n"
        "  point_prompt_molmo(rgb, text)               -> {text: (x,y) | (None,None)}\n"
        "  plan_grasp(point_cloud, ...)                -> grasp poses\n"
        "  goto_pose(world_pos, quat, z_approach=...)  -> moves the arm\n"
        "  open_gripper() / close_gripper()\n"
        "  get_observation()                           -> obs dict (agentview + wrist cam)\n"
        "  mask_to_world_points(mask, depth, K, T)     -> (N,3) world points\n"
        "  pixel_to_world_point(u,v,z, K, T)           -> (3,) world point\n"
        "  verify_object_identity(rgb, (x,y), expected) -> {verified, confidence, actual}\n"
        "  inspect_at_wrist(world_pos, hover_height=0.10) -> {wrist:{rgb,depth,...}, agentview:{...}}\n"
        "Wrist cam: obs['robot0_eye_in_hand']['images']['rgb'] gives a close-up; agent-view "
        "obs['agentview'] is fixed.\n"
    )

    def _distill_one_group(
        self, group_key: str, episodes: list[dict],
        *,
        iteration: int | None = None,
    ) -> dict[str, Any] | None:
        """Distill one group of episodes into a code-shaped lesson via LLM.

        The lesson the LLM produces is required to:
          - state a CONDITION (when does the lesson apply)
          - point to a CONCRETE code-level fix (referencing real primitives)
          - explicitly say what NOT to do (the failed pattern)
        This is much more useful for the next iteration's policy writer than
        a generic "verify the grasp before placing" sentence.
        """
        # FIX (failed_code_tail was useless boilerplate): previously took
        # the last 300 chars of code_snippet, which is almost always the
        # same `try: RESULT["success"] = ...; except ...; return True`
        # scaffold every attempt produces. The LLM saw five identical
        # "tails" across five different failures and had nothing to
        # distinguish them.
        #
        # New strategy: emit the FULL code body of each failed attempt.
        # Storage now caps each snippet at 8000 chars (see record_failure),
        # and we cap episodes[:5] below, so the whole prompt is bounded
        # at ≤40k chars of code + diagnosis — well under the distill call's
        # 65k-token output budget and the 1M-token Gemini input window.
        # Truncating arbitrarily here was the original sin: it lost the
        # signal we wanted the distill LLM to reason over.

        summaries = []
        for ep in episodes[:5]:  # cap episodes for prompt size
            code_body = ep.get("code_snippet") or ""
            diag = ep.get("diagnosis_summary", "")
            summaries.append(
                f"- task={ep['task_name']!r} "
                f"objects={ep.get('objects_involved', [])} "
                f"category={ep['failure_category']} "
                f"retries={ep.get('retry_count')}\n"
                f"  diagnosis: {diag}\n"
                f"  failed_code:\n"
                f"```python\n{code_body}\n```"
            )

        system_prompt = (
            "You convert RATS robot-failure episodes into ACTIONABLE lessons "
            "for the next policy-writer attempt. A good lesson names a CONCRETE "
            "trigger condition, points at SPECIFIC primitives by name, and shows "
            "the failed antipattern so the LLM avoids repeating it. Generic prose "
            "like 'verify the grasp' is useless — say HOW (which function, with "
            "what arguments, in what order). Respond only in valid JSON."
        )
        user_prompt = (
            f"Group key: [{group_key}]\n"
            f"{len(summaries)} failure episode(s):\n\n"
            + "\n\n".join(summaries)
            + "\n\n"
            + self._DISTILL_TOOL_INVENTORY
            + "\n\nProduce ONE lesson with this exact JSON schema:\n"
            "{\n"
            '  "condition": "<one-line trigger: when should the policy writer apply this>",\n'
            '  "antipattern": "<what the failed code did wrong, referencing a primitive>",\n'
            '  "remedy": "<concrete code-level fix; reference primitives by exact name; '
            '            show 1-3 lines of pseudo-Python if it helps>",\n'
            '  "applicable_objects": ["<obj>", ...],\n'
            '  "applicable_actions": ["grasp" | "pick" | "place" | "open" | "close" | ...],\n'
            '  "confidence": <float 0-1>\n'
            "}\n"
            "Do NOT produce vague advice like 'verify the grasp' — the remedy must "
            "name actual functions (verify_object_identity / inspect_at_wrist / "
            "segment_sam3_text_prompt etc.) or actual variables to check (gripper_pos, "
            "z-height after lift, mask score, etc.)."
        )

        try:
            from rats.agents.base_agent import query_llm_json
            # 512 tokens was too tight — even gpt-5.5 with reasoning_effort=low
            # routinely consumed all 512 on reasoning, leaving 0 for the JSON
            # response (12/15 distill calls in the LIBERO smoke run truncated
            # to empty content). Pinned to 65536 — the cross-provider ceiling
            # (Gemini-3 Pro = 65536, Claude 4.x = 64k, gpt-5 = 128k) — so
            # reasoning + the small JSON schema always fit.
            result = query_llm_json(system_prompt, user_prompt, max_tokens=65536)
            condition = (result.get("condition") or "").strip()
            antipattern = (result.get("antipattern") or "").strip()
            remedy = (result.get("remedy") or "").strip()
            if not (condition and remedy):
                return None

            # Compose a compact rendering of the lesson that downstream
            # retrieval (retrieve_for_policy_writer / get_lessons_for_planner)
            # serves verbatim. Three-line shape so the policy writer's
            # context shows trigger / wrong / right at a glance.
            # FIX: was [:600] which clipped the DO clause for any lesson
            # with a multi-step fix. The distill LLM is now budgeted to
            # produce ~2-3k char lessons (max_tokens=65536, schema is
            # small), so emit the full WHEN/WRONG/DO triple.
            description = (
                f"WHEN {condition} | WRONG: {antipattern} | "
                f"DO: {remedy}"
            )

            return {
                "lesson_id": f"les_{uuid.uuid4().hex[:8]}",
                "description": description,
                "condition": condition,
                "antipattern": antipattern,
                "remedy": remedy,
                "applicable_to": {
                    "objects": result.get("applicable_objects", []),
                    "actions": result.get("applicable_actions", []),
                    "task_types": [],
                },
                "evidence": [ep["episode_id"] for ep in episodes[:5]],
                "confidence": float(result.get("confidence", 0.5)),
                "times_applied": 0,
                "times_helped": 0,
                # Lets downstream prompts render per-lesson staleness
                # (current_iteration - distilled_at_iteration). Older
                # lessons that lack this field default to None at
                # retrieval time, which the formatter renders as
                # "distilled=?".
                "distilled_at_iteration": int(iteration) if iteration is not None else None,
            }
        except Exception as e:
            logger.debug(f"Lesson distillation failed for {group_key}: {e}")
            return None

    def _is_duplicate_lesson(self, new_lesson: dict) -> bool:
        """Check if a lesson is semantically duplicate of existing ones.

        Compare on the structured (condition + remedy) fields when present,
        falling back to the description otherwise. Stricter than the previous
        word-overlap check so that 5 lessons all saying "verify the grasp"
        don't all survive — only the first.
        """
        def _key(lesson: dict) -> str:
            return (
                (lesson.get("condition", "") + " " + lesson.get("remedy", ""))
                .strip()
                .lower()
                or lesson.get("description", "").lower()
            )

        new_key = _key(new_lesson)
        new_words = set(new_key.split())
        if not new_words:
            return False
        for existing in self._lessons:
            existing_words = set(_key(existing).split())
            if not existing_words:
                continue
            denom = max(len(new_words), len(existing_words))
            jaccard = len(new_words & existing_words) / denom
            # Stricter: 0.55 jaccard against the larger set (was 0.7 against
            # the new only, which let near-duplicates through).
            if jaccard > 0.55:
                return True
        return False

    # ----------------------------------------------------------------
    # 1B.6  Cross-run merge
    # ----------------------------------------------------------------

    def merge_from(self, other_path: str | Path) -> int:
        """Merge episodes from another failure memory directory.

        Deduplicates by episode_id. Returns count of new episodes added.
        """
        other_dir = Path(other_path)
        other_episodes_path = other_dir / "episodes.json"
        other_lessons_path = other_dir / "lessons.json"
        added = 0

        if other_episodes_path.exists():
            try:
                other_episodes = json.loads(other_episodes_path.read_text())
                existing_ids = {ep["episode_id"] for ep in self._episodes}
                for ep in other_episodes:
                    if ep.get("episode_id") not in existing_ids:
                        self._episodes.append(ep)
                        added += 1
                if added:
                    self._save_episodes()
            except Exception as e:
                logger.warning(f"Could not merge episodes from {other_path}: {e}")

        if other_lessons_path.exists():
            try:
                other_lessons = json.loads(other_lessons_path.read_text())
                existing_ids = {l["lesson_id"] for l in self._lessons}
                new_count = 0
                for lesson in other_lessons:
                    if lesson.get("lesson_id") not in existing_ids:
                        self._lessons.append(lesson)
                        new_count += 1
                if new_count:
                    self._save_lessons()
                    logger.info(f"Merged {new_count} lessons from {other_path}")
            except Exception as e:
                logger.warning(f"Could not merge lessons from {other_path}: {e}")

        logger.info(f"Merged {added} new episodes from {other_path}")
        return added

    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------

    def get_failure_stats(self) -> dict[str, Any]:
        """Return summary statistics about failure memory."""
        cat_counts: dict[str, int] = {}
        for ep in self._episodes:
            cat = ep.get("failure_category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {
            "total_episodes": len(self._episodes),
            "total_lessons": len(self._lessons),
            "failure_categories": cat_counts,
            "episodes_since_last_distill": self._episodes_since_last_distill,
        }

    @property
    def episode_count(self) -> int:
        return len(self._episodes)


def _truncate_code(code: str, max_chars: int = 500) -> str:
    """Truncate code to the key section (last N chars, which is usually the action)."""
    if len(code) <= max_chars:
        return code
    return "...\n" + code[-max_chars:]


_STEP_INDEX_RE = re.compile(r"\d+")


def _step_index(step_id: str) -> str:
    if not step_id:
        return ""
    m = _STEP_INDEX_RE.search(str(step_id))
    return m.group(0) if m else ""


def _extract_step_segment(code: str, step_id: str) -> str:
    """Return the complete code block owned by the diagnosed plan step."""
    if not code or not step_id:
        return ""
    idx = _step_index(step_id)
    if not idx:
        return ""
    idx_in_line = re.compile(rf"(?<!\d){idx}(?!\d)")
    header_re = re.compile(r"^\s*#\s*step\b[^\n]*", re.IGNORECASE | re.MULTILINE)
    match_pos: int | None = None
    next_step_pos: int | None = None
    for m in header_re.finditer(code):
        line = code[m.start():m.end()]
        owns_idx = bool(idx_in_line.search(line))
        if match_pos is None:
            if owns_idx:
                match_pos = m.start()
            continue
        if not owns_idx:
            next_step_pos = m.start()
            break
    if match_pos is None:
        return ""
    end = next_step_pos if next_step_pos is not None else len(code)
    return code[match_pos:end].strip("\n")
