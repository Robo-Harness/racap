"""Planner: THE sole skill retrieval owner.

Reads the full Skill Library, selects relevant skills for each plan step,
and outputs a structured plan with selected_skills[] and new_skill_needed flags.
Policy Writer NEVER touches the library directly - it only sees what Planner hands it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rats.agents.base_agent import image_to_data_url, query_llm_json
from rats.agents.experience_card import append_experience_card

logger = logging.getLogger("rats.planner")
from skill_library.initial_primitives import get_primitive_docs


def _annotate_planner_image(rgb: Any, label: str) -> Any:
    """Stamp a small label on the top-left of the planner's initial image.

    Best-effort: if PIL or numpy isn't available, return the original
    image. The label removes ambiguity about which capture phase the
    image represents when it shows up in agent_io/ later.
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except Exception:
        return rgb
    try:
        if isinstance(rgb, np.ndarray):
            img = Image.fromarray(rgb)
        elif isinstance(rgb, Image.Image):
            img = rgb.copy()
        else:
            return rgb
        draw = ImageDraw.Draw(img)
        # Plain font — don't depend on system fonts being available.
        pad = 4
        text_box = (pad, pad, pad + 7 * len(label) + 8, pad + 14)
        draw.rectangle(text_box, fill=(0, 0, 0, 200))
        draw.text((pad + 4, pad + 1), label, fill=(255, 255, 255))
        return np.asarray(img) if isinstance(rgb, np.ndarray) else img
    except Exception:
        return rgb


def _format_learned_skills_by_tier(learned: list[dict[str, Any]]) -> str:
    """Render learned skills as verified/experimental groups with empirical
    reliability footprints.

    The library returns skills already sorted by (tier, Wilson lower bound),
    so we just split into two groups and preserve order. Each entry is a
    JSON object with an added ``reliability`` field (only when the skill has
    actually been used — zero-usage skills show nothing so we don't mislead
    with SR=0). No must-use language anywhere; the visual hierarchy alone
    is the nudge.
    """
    if not learned:
        return "No learned skills yet."

    def _fmt(s: dict[str, Any]) -> dict[str, Any]:
        n = int(s.get("usage_count", 0))
        sc = int(s.get("success_count", 0))
        entry: dict[str, Any] = {
            "name": s["name"],
            "description": s["description"],
            "code": s["code"],
            "preconditions": s.get("preconditions", []),
            "effects": s.get("effects", []),
        }
        # Provenance — populated when the skill was extracted on this
        # branch's extraction prompt (feedback_generator.py +
        # prompts/feedback_generator.txt). Older skills may lack one or
        # more of these; omit empty keys so the planner JSON stays tight.
        src = (s.get("source_task") or "").strip()
        if src:
            entry["source_task"] = src
        why = (s.get("extraction_rationale") or "").strip()
        if why:
            entry["why_extracted"] = why
        ex = (s.get("usage_example") or "").strip() or (s.get("example_code") or "").strip()
        if ex:
            entry["usage_example"] = ex
        if n > 0:
            entry["reliability"] = (
                f"{sc}/{n} successes (SR={s.get('success_rate', 0.0):.2f})"
            )
        return entry

    verified = [_fmt(s) for s in learned if s.get("tier") == "verified"]
    experimental = [
        _fmt(s) for s in learned if s.get("tier", "experimental") == "experimental"
    ]

    parts: list[str] = []
    if verified:
        parts.append(
            f"## Verified skills ({len(verified)}) — "
            f"proven on prior iterations (success_rate ≥ 0.5, uses ≥ 3)\n"
            + json.dumps(verified, indent=2)
        )
    if experimental:
        parts.append(
            f"## Experimental skills ({len(experimental)}) — "
            f"proposed but not yet validated empirically\n"
            + json.dumps(experimental, indent=2)
        )
    return "\n\n".join(parts) if parts else "No learned skills yet."


def _normalize_prediction_card(
    raw: dict[str, Any] | None,
    task_proposal: dict[str, Any],
) -> dict[str, Any]:
    raw = raw or {}
    default_prob = task_proposal.get("learnability", task_proposal.get("novelty_score", 0.5))
    try:
        prob = float(raw.get("predicted_success_probability", default_prob))
    except Exception:
        prob = float(default_prob or 0.5)
    prob = max(0.0, min(1.0, prob))
    return {
        "predicted_success_probability": prob,
        "predicted_bottleneck_step": str(raw.get("predicted_bottleneck_step", "") or ""),
        "prediction_reasoning": str(raw.get("prediction_reasoning", "") or ""),
    }


class Planner:
    def plan(
        self,
        task_proposal: dict[str, Any],
        all_skills: list[dict[str, Any]],
        scene_context: dict[str, Any],
        *,
        failure_lessons: str = "",
        initial_rgb: Any = None,
    ) -> dict[str, Any]:
        """Create a step-by-step plan with selected skills for each step.

        Args:
            task_proposal: From TaskProposer (activity_name, goal_conditions, etc.)
            all_skills: From SkillLibrary.get_full_skills_for_planner()
            scene_context: Scene info (available_functions, api_docs, object_scope)
            failure_lessons: Optional constraints from past failures (from FailureMemory).

        Returns:
            Plan dict with steps, each containing selected skills and code.
        """
        prompt_template = Path("rats/prompts/planner.txt").read_text()
        failure_lessons = append_experience_card(failure_lessons)

        # Separate primitives and learned skills
        primitives = [s for s in all_skills if s.get("is_primitive", False)]
        learned = [s for s in all_skills if not s.get("is_primitive", False)]

        learned_section = _format_learned_skills_by_tier(learned)

        # Use actual available primitives from scene context (env-aware),
        # falling back to static docs only if scene context is empty.
        api_docs = scene_context.get("api_docs", "")
        available_fns = scene_context.get("available_functions", [])
        if api_docs:
            primitive_list = api_docs
        elif available_fns:
            primitive_list = "\n".join(f"- {fn}()" for fn in available_fns)
        else:
            primitive_list = get_primitive_docs()

        user_prompt = prompt_template.replace(
            "{task_description}", task_proposal.get("goal_conditions", task_proposal["activity_name"])
        ).replace(
            "{goal_conditions}", task_proposal.get("goal_conditions", "")
        ).replace(
            "{object_scope}", json.dumps(scene_context.get("object_scope", {}))
        ).replace(
            "{selected_skills}", learned_section
        ).replace(
            "{primitive_list}", primitive_list
        ).replace(
            "{failure_lessons}", failure_lessons
        )
        # Note: prediction_card spec is now baked into the template's stable
        # header so the prefix bytes match across iterations and the prompt
        # cache can keep its hit rate high.

        system_prompt = "You are a robot task planner. Decompose tasks into concrete steps. Respond only in valid JSON."

        # Optional initial agentview RGB so the planner can see the scene
        # state before committing to a step order (e.g. notice the microwave
        # door is closed so opening-first is required). Initial RGB is what
        # a real robot would observe at deployment — not privileged.
        #
        # FIX (planner image was unannotated): the IO log used to save the
        # raw RGB with no phase label, so reviewers couldn't tell whether
        # the renderer was showing the actual reset state or a stale
        # fallback. Stamp the image with "INITIAL SCENE (planner)" so it's
        # unambiguous when re-rendered in agent_io/.
        images = None
        if initial_rgb is not None:
            annotated = _annotate_planner_image(initial_rgb, "INITIAL SCENE (planner)")
            url = image_to_data_url(annotated)
            if url:
                images = [url]

        result = query_llm_json(system_prompt, user_prompt, images=images)

        steps = result.get("steps", [])
        if not steps:
            # Fallback: create a basic plan
            steps = self._create_fallback_plan(task_proposal, scene_context)

        # Attach selected skill code to each step
        skill_map = {s["name"]: s for s in all_skills}
        for step in steps:
            selected = []
            for skill_name in step.get("relevant_skills", []):
                if skill_name in skill_map:
                    selected.append(skill_map[skill_name])
                elif skill_name not in ("", None):
                    logger.warning(
                        f"  Planner referenced skill '{skill_name}' not found in library "
                        f"(available: {[s['name'] for s in all_skills if not s.get('is_primitive')]})"
                    )
            step["selected_skill_details"] = selected

        prediction_card = _normalize_prediction_card(
            result.get("prediction_card", {}), task_proposal,
        )
        return {
            "task_id": f"{task_proposal['activity_name']}:0",
            "steps": steps,
            "all_selected_skill_names": [
                name
                for step in steps
                for name in step.get("relevant_skills", [])
            ],
            "prediction_card": prediction_card,
        }

    def refine_plan(
        self,
        old_plan: dict[str, Any],
        task_proposal: dict[str, Any],
        all_skills: list[dict[str, Any]],
        scene_context: dict[str, Any],
        *,
        plan_issue_reason: str,
        prior_attempts: list[dict[str, Any]] | None = None,
        failure_lessons: str = "",
        initial_rgb: Any = None,
    ) -> dict[str, Any]:
        """Rewrite a plan that the diagnoser flagged as structurally broken.

        Triggered only when ``diagnosis["plan_issue"]`` is True — the
        trajectory showed a step-ordering / prerequisite problem that
        code-level retries cannot fix. The caller is responsible for
        clearing any mechanism-A preserved segments before running code
        against the new plan; step_ids may have shifted.

        Args:
            old_plan: The plan that just failed (same shape as `plan()` returns).
            task_proposal: Same as for `plan()`.
            all_skills: Same as for `plan()`.
            scene_context: Same as for `plan()`.
            plan_issue_reason: The diagnoser's 1-2 sentence structural
                diagnosis describing what's wrong with the old plan.
            prior_attempts: Summary of previous attempts in this iteration.
                Each entry should have ``{attempt_idx, failure_mode,
                policy_feedback, failed_step, plan_issue_reason,
                visual_predicate_status, code}``. The refine prompt uses
                ALL of these — without ``failed_step`` and
                ``visual_predicate_status`` the model can't tell which
                plan-index needs reworking vs which already-✓ sub-behavior
                to leave alone. Capped at the last 4 attempts for token
                budget.
            failure_lessons: Carried over from ``plan()``.

        Returns:
            New plan dict, same shape as ``plan()`` output.
        """
        prompt_template = Path("rats/prompts/planner_refine.txt").read_text()

        primitives = [s for s in all_skills if s.get("is_primitive", False)]
        learned = [s for s in all_skills if not s.get("is_primitive", False)]
        learned_section = _format_learned_skills_by_tier(learned)

        api_docs = scene_context.get("api_docs", "")
        available_fns = scene_context.get("available_functions", [])
        if api_docs:
            primitive_list = api_docs
        elif available_fns:
            primitive_list = "\n".join(f"- {fn}()" for fn in available_fns)
        else:
            primitive_list = get_primitive_docs()

        # Render the old plan as a compact ordered list.
        old_plan_lines = []
        for step in old_plan.get("steps", []):
            sid = step.get("id", "?")
            desc = step.get("description", "")
            old_plan_lines.append(f"  {sid}: {desc}")
        old_plan_text = "\n".join(old_plan_lines) or "(no prior plan)"

        # FIX (refine_plan was using a 3-line abridgment of each prior
        # attempt — only attempt_idx + failure_mode + policy_feedback[:300]).
        # That's too thin for a structural refine: the diagnoser already
        # produces failed_step, plan_issue_reason, and per-step visual
        # verdicts; without them the refine model is reordering a plan
        # without visibility into what's already working visually,
        # which is the load-bearing signal for "keep the ✓ steps as-is,
        # rework only the ✗ ones."
        #
        # Now emit per attempt:
        #   - attempt_idx + failure_mode
        #   - failed_step             (which plan step the diagnoser blamed)
        #   - plan_issue_reason       (prior structural diagnosis, if any)
        #   - visual_predicate_status (✓/✗ per step from the vision LLM)
        #   - policy_feedback         (full, no slice)
        #   - code                    (full body in a python fence)
        # `prior_attempts[-4:]` window is the same bound used elsewhere.
        prior_txt = "(no prior attempts)"
        if prior_attempts:
            blocks = []
            for pa in prior_attempts[-4:]:
                idx = pa.get("attempt_idx", "?")
                fmode = str(pa.get("failure_mode", "") or "")
                fstep = str(pa.get("failed_step", "") or "")
                pir = str(pa.get("plan_issue_reason", "") or "")
                pfb = str(pa.get("policy_feedback", "") or "")
                vps = pa.get("visual_predicate_status") or []
                vps_lines = []
                for e in vps:
                    mark = "✓" if e.get("visually_satisfied") else "✗"
                    sid = e.get("step_id") or e.get("description") or "?"
                    ev = str(e.get("evidence", "") or "")
                    vps_lines.append(f"      {mark} {sid} — {ev}")
                vps_txt = "\n".join(vps_lines) if vps_lines else "      (no per-step verdict)"
                pcode = str(pa.get("code", "") or "")
                code_block = pcode if pcode else "(no code)"
                parts = [
                    f"  attempt {idx} (failure_mode={fmode}, failed_step={fstep or 'unknown'}):",
                    f"    diagnoser_said: {pfb}",
                ]
                if pir:
                    parts.append(f"    prior_structural_diagnosis: {pir}")
                parts.append(f"    visual_per_step_verdict:\n{vps_txt}")
                parts.append(f"    code:\n```python\n{code_block}\n```")
                blocks.append("\n".join(parts))
            prior_txt = "\n\n".join(blocks)

        user_prompt = prompt_template.replace(
            "{task_description}", task_proposal.get("goal_conditions", task_proposal["activity_name"])
        ).replace(
            "{goal_conditions}", task_proposal.get("goal_conditions", "")
        ).replace(
            "{object_scope}", json.dumps(scene_context.get("object_scope", {}))
        ).replace(
            "{old_plan}", old_plan_text
        ).replace(
            "{plan_issue_reason}", plan_issue_reason or "(no structural diagnosis provided)"
        ).replace(
            "{prior_attempts_summary}", prior_txt
        ).replace(
            "{selected_skills}", learned_section
        ).replace(
            "{primitive_list}", primitive_list
        ).replace(
            "{failure_lessons}", failure_lessons
        )
        # Note: prediction_card spec is now baked into the template's stable
        # header so the prefix bytes match across iterations and the prompt
        # cache can keep its hit rate high.

        system_prompt = (
            "You are a robot task planner revising a broken plan. "
            "Respond only in valid JSON."
        )
        images = None
        if initial_rgb is not None:
            url = image_to_data_url(initial_rgb)
            if url:
                images = [url]
        result = query_llm_json(system_prompt, user_prompt, images=images)
        steps = result.get("steps", [])
        if not steps:
            # Fall back to the old plan if refinement failed — better than
            # losing the whole iteration. The loop will just retry under the
            # old plan (re-generating code) which is the pre-refine behavior.
            logger.warning("  refine_plan returned no steps; falling back to old plan")
            return old_plan

        skill_map = {s["name"]: s for s in all_skills}
        for step in steps:
            selected = []
            for skill_name in step.get("relevant_skills", []):
                if skill_name in skill_map:
                    selected.append(skill_map[skill_name])
                elif skill_name not in ("", None):
                    logger.warning(
                        f"  refine_plan referenced skill '{skill_name}' not found"
                    )
            step["selected_skill_details"] = selected

        prediction_card = _normalize_prediction_card(
            result.get("prediction_card", {}), task_proposal,
        )
        return {
            "task_id": f"{task_proposal['activity_name']}:refined",
            "steps": steps,
            "all_selected_skill_names": [
                name
                for step in steps
                for name in step.get("relevant_skills", [])
            ],
            "prediction_card": prediction_card,
        }

    def _create_fallback_plan(
        self, task_proposal: dict[str, Any], scene_context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Create a basic observe-act-verify plan when LLM fails."""
        goal = task_proposal.get("goal_conditions", task_proposal["activity_name"])
        available_fns = set(scene_context.get("available_functions", []))

        # Pick env-appropriate primitives for the fallback skeleton
        if "plan_grasp" in available_fns and "get_object_pose" not in available_fns:
            # Non-privileged LIBERO API
            return [
                {
                    "id": "step-1",
                    "description": "Observe the scene and segment target objects using vision.",
                    "relevant_skills": ["get_observation", "segment_sam3_text_prompt"],
                    "skill_code": "",
                    "new_skill_needed": False,
                    "notes": "Use SAM3 for perception, mask_to_world_points for 3D localization.",
                },
                {
                    "id": "step-2",
                    "description": f"Execute interaction to achieve: {goal}",
                    "relevant_skills": ["plan_grasp", "goto_pose", "close_gripper", "open_gripper"],
                    "skill_code": "",
                    "new_skill_needed": True,
                    "notes": "Use plan_grasp for grasp candidates, decompose_transform for pose extraction.",
                },
                {
                    "id": "step-3",
                    "description": "Verify goal conditions are met.",
                    "relevant_skills": ["get_observation"],
                    "skill_code": "",
                    "new_skill_needed": False,
                    "notes": "Collect evidence for verifier.",
                },
            ]
        elif "get_observation" in available_fns:
            # Privileged LIBERO API
            return [
                {
                    "id": "step-1",
                    "description": "Observe the scene and locate target objects.",
                    "relevant_skills": ["get_observation", "get_object_pose"],
                    "skill_code": "",
                    "new_skill_needed": False,
                    "notes": "Use deterministic perception.",
                },
                {
                    "id": "step-2",
                    "description": f"Execute interaction to achieve: {goal}",
                    "relevant_skills": ["sample_grasp_pose", "goto_pose", "close_gripper", "open_gripper"],
                    "skill_code": "",
                    "new_skill_needed": True,
                    "notes": "May need new skill composition.",
                },
                {
                    "id": "step-3",
                    "description": "Verify goal conditions are met.",
                    "relevant_skills": ["get_observation"],
                    "skill_code": "",
                    "new_skill_needed": False,
                    "notes": "Collect evidence for verifier.",
                },
            ]
        else:
            # BEHAVIOR / R1Pro default
            return [
                {
                    "id": "step-1",
                    "description": "Observe the scene and locate target objects.",
                    "relevant_skills": ["get_env_observation", "get_object_pose"],
                    "skill_code": "",
                    "new_skill_needed": False,
                    "notes": "Use deterministic perception.",
                },
                {
                    "id": "step-2",
                    "description": f"Execute interaction to achieve: {goal}",
                    "relevant_skills": ["navigate_to_pose", "sample_grasp_pose", "grasp_object"],
                    "skill_code": "",
                    "new_skill_needed": True,
                    "notes": "May need new skill composition.",
                },
                {
                    "id": "step-3",
                    "description": "Verify goal conditions are met.",
                    "relevant_skills": ["get_env_observation", "check_object_in_hand"],
                    "skill_code": "",
                    "new_skill_needed": False,
                    "notes": "Collect evidence for verifier.",
                },
            ]
