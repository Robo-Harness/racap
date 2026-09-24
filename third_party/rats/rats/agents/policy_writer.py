"""Policy Writer: generates executable code-as-policy from the Planner's output.

Forked from CaP-Agent0's code generation approach. Accepts structured plan +
pre-selected skill code (from Planner) instead of raw task description.
Implements retry path with stderr + failure diagnosis feedback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import logging

import os

from rats.agents.base_agent import (
    DEFAULT_MAX_TOKENS,
    canonicalize_model_name,
    extract_python_code,
    query_llm_text,
)
from rats.agents.experience_card import append_experience_card


# Policy_writer defaults to Gemini-3-Pro for code generation, matching the
# failure_diagnoser / verifier defaults. Routes through Vertex AI when
# google-genai + ADC are present (rats.llm.genai_backend); otherwise the
# existing Gemini-proxy / Google-direct fallback applies. Planner, task
# proposer, etc. stay on base_agent's global default (gpt-5.5).
DEFAULT_MODEL = "google/gemini-3.1-pro-preview"


def _policy_writer_model() -> str | None:
    """Per-agent model for policy_writer.

    Defaults to Gemini-3-Pro (``DEFAULT_MODEL``) for code generation, so the
    writer / diagnoser / verifier all run on Gemini while planner and the
    rest stay on base_agent's global default.

    Override with ``RATS_POLICY_WRITER_MODEL`` to swap only the writer
    (e.g. ``openai/gpt-5.5``) without touching ``RATS_LLM_MODEL``. Set it to
    ``global`` / ``default`` / ``inherit`` to fall back to base_agent's
    global default — in that case this returns ``None`` so call sites use
    get_default_model().

    User-typed bare ids (e.g. ``gemini-3-pro-preview``) are canonicalized
    here so they don't silently fall through to direct OpenAI routing later.
    """
    val = os.environ.get("RATS_POLICY_WRITER_MODEL", "").strip() or DEFAULT_MODEL
    if val.lower() in {"global", "default", "inherit"}:
        return None
    return canonicalize_model_name(val)

logger = logging.getLogger("rats.policy_writer")


def _runtime_marker_requirement(env_type: str) -> str:
    # Per-step verifier needs frame-range slices keyed on plan-step id.
    # Without markers in the generated code, exec_history only carries
    # `timeline_kind="learned_skill"` entries with `policy_step_id=None`,
    # which means `_build_step_frame_segments` produces an empty list and
    # the per-step verifier reports "no step media available" for every
    # step (observed across the entire libero_main_30iter run before this
    # fix). Enable for envs whose sandbox actually injects step_context
    # into the exec globals (see rats/envs/tasks/base.py).
    if env_type not in ("molmospaces", "libero"):
        return ""
    return (
        '- Wrap the executable body for EACH plan step in the runtime marker context\n'
        '  `with step_context("<step id>", "<step description>", step_index=<0-based index>):`.\n'
        '  This is a logging marker, not a robot primitive. It lets the verifier save\n'
        '  exact per-step API traces, motion frame ranges, and intermediate artifacts.'
    )


def _format_skill_reference(skill_detail: dict[str, Any]) -> str:
    name = skill_detail.get("name", "")
    description = skill_detail.get("description", "")
    preconditions = skill_detail.get("preconditions", [])
    effects = skill_detail.get("effects", [])
    dependent_skills = skill_detail.get("dependent_skills", [])
    api_primitives_used = skill_detail.get("api_primitives_used", [])
    code = skill_detail.get("code", "")

    lines = [f"### {name}"]
    if code:
        import re

        match = re.search(r"^\s*(async\s+def|def)\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:", code, re.MULTILINE)
        if match:
            lines.append(f"Signature: {match.group(0).strip()}")
    if description:
        lines.append(f"Description: {description}")
    if preconditions:
        lines.append(f"Preconditions: {preconditions}")
    if effects:
        lines.append(f"Effects: {effects}")
    if dependent_skills:
        lines.append(f"Depends on: {dependent_skills}")
    if api_primitives_used:
        lines.append(f"API primitives used: {api_primitives_used}")
    return "\n".join(lines)


class PolicyWriter:
    def __init__(self, max_retries: int = 5, ensemble_n: int = 0) -> None:
        self.max_retries = max_retries
        # ensemble_n > 0: generate N candidates at different temperatures, pick best
        self.ensemble_n = ensemble_n
        # Populated by write()/write_step() for run-artifact persistence.
        self.last_retry_context_text = ""
        self.last_retry_diagnostic_context: dict[str, Any] = {}
        self.last_retry_diagnostic_image_count = 0
        self.last_user_prompt_text = ""

    def _format_retry_diagnostic_context(
        self,
        retry_feedback: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Render diagnoser-supplied raw artifacts for retry prompts."""
        if not retry_feedback:
            return "", []
        context = retry_feedback.get("diagnostic_context") or {}
        if not isinstance(context, dict):
            return "", []

        sections: list[str] = []
        artifact_summary = str(context.get("artifact_summary") or "").strip()
        if artifact_summary:
            sections.append(
                "PERCEPTION / GRASP / MOTION DIAGNOSTIC SUMMARY:\n"
                + artifact_summary[:2500]
            )

        raw_numeric = str(context.get("raw_numeric_artifacts") or "").strip()
        if raw_numeric:
            sections.append(
                "RAW NUMERIC ARTIFACT EXCERPTS AND FILE PATHS:\n"
                + raw_numeric[:4500]
            )

        staleness = str(context.get("pointcloud_staleness") or "").strip()
        if staleness:
            sections.append(
                "POINTCLOUD STALENESS WARNING:\n"
                + staleness[:2000]
            )

        images: list[str] = []
        image_lines: list[str] = []
        for idx, image in enumerate(context.get("artifact_images") or [], start=1):
            if not isinstance(image, dict):
                continue
            data_url = str(image.get("data_url") or "")
            if not data_url:
                continue
            images.append(data_url)
            bits = [f"image {idx}"]
            for key in ("event", "camera", "label", "path"):
                value = str(image.get(key) or "").strip()
                if value:
                    bits.append(f"{key}={value}")
            image_lines.append("  - " + "; ".join(bits))
        if image_lines:
            sections.append(
                "DIAGNOSTIC VISUALIZATION IMAGES ATTACHED TO THIS WRITER CALL:\n"
                + "\n".join(image_lines)
            )

        if not sections:
            return "", images

        block = (
            "\nDIAGNOSTIC ARTIFACTS FROM THE FAILED RUN "
            "(non-privileged; produced by the policy/runtime APIs):\n"
            f"{context.get('selection', '')}\n"
            + "\n\n".join(sections)
            + "\nUse these to repair perception, grasp selection, and motion "
            "targets. Do not blindly hard-code stale old coordinates; if the "
            "staleness warning or diagnosis says the object may have moved, "
            "recompute the pointcloud from a fresh observation before using "
            "numeric targets.\n"
        )
        return block, images

    def write(
        self,
        plan: dict[str, Any],
        scene_context: dict[str, Any],
        *,
        retry_feedback: dict[str, Any] | None = None,
        failure_context: str = "",
        success_context: str = "",
        subagent_directive: str = "",
    ) -> str:
        """Generate executable Python code from plan + selected skills.

        Args:
            plan: From Planner (steps with selected skills).
            scene_context: Scene info (available_functions, api_docs, object_scope).
            retry_feedback: Optional feedback from previous failed attempt
                           (stderr, diagnosis, previous_code).
            failure_context: Optional advisory lessons from past failures on
                            similar tasks (from FailureMemory). Distilled
                            "WHEN/WRONG/DO" lessons with confidence metadata;
                            rendered under "LESSONS FROM PAST FAILURES".
            subagent_directive: Optional HARD constraint for parallel sub-
                            agent workers (assigned approach + retry history
                            directive). Rendered under "SUB-AGENT DIRECTIVE"
                            with priority #0 — overrides all other guidance.
                            Empty for the normal single-agent loop.

        Returns:
            Python code string ready for execution.
        """
        self.last_retry_context_text = ""
        self.last_retry_diagnostic_context = {}
        self.last_retry_diagnostic_image_count = 0
        self.last_user_prompt_text = ""
        failure_context = append_experience_card(failure_context)
        prompt_template = Path("rats/prompts/policy_writer.txt").read_text()
        env_type = scene_context.get("env_type", "")

        # The template has several bullet points that reference non-default
        # primitives (`grasp_with_wrist_closeloop`, `plan_grasp`, etc.). If
        # those primitives aren't in the current env's available_functions,
        # the guidance is dead text — burns tokens and risks misleading the
        # model into expecting helpers it can't call. Drop the bullets when
        # none of the conditional primitives are present.
        _available = set(scene_context.get("available_functions") or [])
        _conditional_names = {
            "grasp_with_wrist_closeloop",
            "grasp_object_with_verification",
            "plan_grasp",
            "inspect_at_wrist",
            "verify_object_identity",
            "verify_step",
            "verify_holding_before_release",
            "confirm_held_object_is_lifted_before_transfer",
        }
        if not (_conditional_names & _available):
            kept_lines: list[str] = []
            for ln in prompt_template.splitlines():
                if any(n in ln for n in _conditional_names):
                    continue
                kept_lines.append(ln)
            prompt_template = "\n".join(kept_lines)

        # Format plan steps
        step_lines = []
        for step_idx, step in enumerate(plan.get("steps", [])):
            skills = ", ".join(step.get("relevant_skills", []))
            if env_type == "molmospaces":
                step_lines.append(
                    f"- index {step_idx}, id {step.get('id', '?')}: {step.get('description', '')} "
                    f"| skills: {skills} | notes: {step.get('notes', '')}"
                )
            else:
                step_lines.append(
                    f"- {step.get('id', '?')}: {step.get('description', '')} "
                    f"| skills: {skills} | notes: {step.get('notes', '')}"
                )

        # Show only skill references to the model. The actual code is injected
        # later at execution time, not exposed in the prompt.
        skill_ref_parts = []
        seen_skills: set[str] = set()
        for step in plan.get("steps", []):
            for skill_detail in step.get("selected_skill_details", []):
                name = skill_detail.get("name", "")
                if (not skill_detail.get("is_primitive", False)
                        and name not in seen_skills):
                    seen_skills.add(name)
                    skill_ref_parts.append(_format_skill_reference(skill_detail))

        skill_code = (
            "\n\n".join(skill_ref_parts)
            if skill_ref_parts
            else "No learned helper functions available yet."
        )

        # Turn mode: env state from the previous turn persists into this
        # call. Swap "RETRY CONTEXT" framing for "PREVIOUS TURN" framing
        # so the writer continues from the current state rather than
        # rewriting the whole task as a fresh attempt.
        task_in_progress = bool(
            retry_feedback and retry_feedback.get("task_in_progress")
        )

        # Build retry context — pure-visual feedback only (no sim state leaks).
        retry_context = ""
        retry_diagnostic_images: list[str] = []
        if retry_feedback:
            diagnostic_context = retry_feedback.get("diagnostic_context") or {}
            self.last_retry_diagnostic_context = (
                diagnostic_context if isinstance(diagnostic_context, dict) else {}
            )
            # MolmoSpaces now uses a compact per-step VLM feedback generator.
            # Keep raw perception/motion artifacts out of the policy-writer
            # prompt even if a legacy caller accidentally includes them.
            if env_type == "molmospaces":
                retry_diagnostic_block, retry_diagnostic_images = "", []
                self.last_retry_diagnostic_context = {}
            else:
                retry_diagnostic_block, retry_diagnostic_images = (
                    self._format_retry_diagnostic_context(retry_feedback)
                )
            self.last_retry_diagnostic_image_count = len(retry_diagnostic_images)
            # Per-predicate visual verdicts (from vision LLM looking at a
            # time-ordered filmstrip + wrist camera). NOT simulator truth;
            # treat as the diagnoser's best visual inference.
            vps = retry_feedback.get("visual_predicate_status", []) or []
            visual_pred_block = ""
            if vps:
                # VPS entries are per plan step (agent-public info). step_id
                # comes from the plan the agent wrote — safe to render. We
                # used to strip BDDL predicate strings here; now the VPS
                # schema itself carries no symbolic ground-truth info.
                lines = []
                for entry in vps:
                    mark = "✓" if entry.get("visually_satisfied") else "✗"
                    sid = entry.get("step_id") or "?"
                    ev = entry.get("evidence", "")
                    lines.append(f"  [{mark}] {sid} — {ev}")
                visual_pred_block = (
                    "VISUAL PER-STEP VERDICT (from vision LLM, not sim):\n"
                    + "\n".join(lines)
                    + "\nFocus on steps currently marked ✗ without undoing ✓.\n"
                )

            # Mechanism A: preserved sub-step code from the previous attempt.
            # Each entry is a code region that the diagnoser attributed to a
            # ✓ predicate. The policy_writer should keep the INTENT of these
            # segments and focus the retry on the remaining ✗ predicates.
            preserved = retry_feedback.get("preserved_code_segments", []) or []
            preserved_block = ""
            if preserved:
                chunks = []
                for seg in preserved:
                    # Predicate string withheld — see note in visual_pred_block
                    # above. The plan step id and evidence are enough to tell
                    # the writer which sub-behavior to keep.
                    chunks.append(
                        f"--- Preserved segment ({seg.get('step_id','?')}) ---\n"
                        f"why preserved: {seg.get('evidence','')}\n"
                        f"```python\n{seg.get('code_snippet','')}\n```"
                    )
                preserved_block = (
                    "PRESERVED CODE SEGMENTS (these sub-behaviors appear to "
                    "have satisfied a goal on the previous attempt — KEEP "
                    "THEIR INTENT; rewrite only to improve clarity or to "
                    "integrate with the rest. Focus new code on the goals "
                    "still marked ✗):\n"
                    + "\n\n".join(chunks)
                    + "\n"
                )

            # Structured retry-edit scale from the diagnoser. Renders an
            # explicit banner at the TOP of retry_context so the writer
            # sees the directive before the prose diagnosis. None means
            # the diagnoser didn't set the field (older runs / cached
            # responses) — fall back to the prose hint already inside
            # `diagnosis`. Argument-level says "keep call sequence,
            # change only named args"; rewrite-needed says "the API or
            # structure is wrong, you may rewrite".
            edit_scale = retry_feedback.get("edit_scale")
            if edit_scale == "argument_level":
                edit_scale_banner = (
                    "## EDIT SCALE: argument-level\n"
                    "Preserve PREVIOUS CODE's call sequence (same primitives, "
                    "same helpers, same order). Change ONLY the arguments "
                    "named in the diagnosis below (e.g. target text, "
                    "verify_label, pose/offset, approach axis, release "
                    "height, retry count, gate threshold). Do not refactor "
                    "structure or swap APIs unless the diagnosis demands it.\n\n"
                )
            elif edit_scale == "rewrite_needed":
                edit_scale_banner = (
                    "## EDIT SCALE: rewrite-needed\n"
                    "Arguments alone cannot fix this. You may swap the "
                    "primitive/API, restructure the call sequence, or "
                    "rewrite helper bodies — guided by the diagnosis below. "
                    "Still reuse any PRESERVED CODE SEGMENTS verbatim, and "
                    "don't discard a working sub-behavior just because the "
                    "overall attempt failed elsewhere.\n\n"
                )
            else:
                edit_scale_banner = ""

            if task_in_progress:
                # The env was NOT reset before this call — the effects of
                # the previous turn's code are persisted in the simulator.
                # Tell the writer to inspect the current state and to
                # CONTINUE rather than restart. Verifier short-circuits
                # if the task is already satisfied, but the writer should
                # also not issue redundant motions when it can tell.
                retry_context = (
                    f"\n--- PREVIOUS TURN CONTEXT (turn {retry_feedback.get('attempt', '?')}) ---\n"
                    f"{edit_scale_banner}"
                    f"The environment was NOT reset before this turn. The "
                    f"effects of the code below are still in place — if it "
                    f"grasped, the gripper is still holding; if it moved an "
                    f"object, that object is at its new pose; if a joint was "
                    f"actuated, it is at its new angle. Re-inspect the scene "
                    f"with a fresh observation rather than assuming a clean "
                    f"reset.\n"
                    f"PREVIOUS TURN STDERR: {retry_feedback.get('stderr', 'none')}\n"
                    f"DIAGNOSIS OF WHAT'S LEFT TO DO (vision LLM over the "
                    f"trajectory + code + plan): "
                    f"{retry_feedback.get('diagnosis', 'none')}\n"
                    f"LAST FAILED STEP (or 'none' if all steps look done): "
                    f"{retry_feedback.get('failed_step', 'unknown')}\n"
                    f"FAILURE MODE: {retry_feedback.get('failure_mode', 'unknown')}\n"
                    f"{visual_pred_block}"
                    f"{retry_diagnostic_block}"
                    f"PREVIOUS TURN CODE:\n```python\n{retry_feedback.get('previous_code', '')}\n```\n"
                    f"Continue from the current persisted state. The "
                    f"previous turn was not accepted as task-complete; this "
                    f"retry would not be running otherwise. Skip sub-actions "
                    f"whose effects are clearly already in place (checking "
                    f"via observation when in doubt), but do NOT return a "
                    f"no-op / RESULT-only response just because the scene "
                    f"looks close. Inspect when uncertain, then execute the "
                    f"smallest concrete corrective action needed for the "
                    f"goal.\n"
                )
            else:
                # FIX (multi-attempt history): if the lifelong_loop
                # supplied an OLDER attempt code (the one before the
                # most recent), show it so the writer sees two
                # attempts of context — current PREVIOUS CODE plus the
                # one before. The diagnoser's prose critique only
                # covers the most recent attempt; including the older
                # code lets the writer notice "I already tried that
                # pattern" without relying on memory of an earlier
                # turn. Bounded: only one older attempt is shown,
                # never a full history.
                older_code = retry_feedback.get("older_code") or ""
                older_block = ""
                if older_code:
                    older_block = (
                        "OLDER ATTEMPT (one before PREVIOUS CODE; "
                        "shown so you can avoid re-trying the same "
                        "pattern this iteration already discarded):\n"
                        f"```python\n{older_code}\n```\n"
                    )
                retry_context = (
                    f"\n--- RETRY CONTEXT (attempt {retry_feedback.get('attempt', '?')}) ---\n"
                    f"{edit_scale_banner}"
                    f"Previous code FAILED. Here is the feedback:\n"
                    f"STDERR: {retry_feedback.get('stderr', 'none')}\n"
                    f"FAILURE DIAGNOSIS (vision LLM over trajectory filmstrip + code + plan): "
                    f"{retry_feedback.get('diagnosis', 'none')}\n"
                    f"FAILED STEP: {retry_feedback.get('failed_step', 'unknown')}\n"
                    f"FAILURE MODE: {retry_feedback.get('failure_mode', 'unknown')}\n"
                    f"{visual_pred_block}"
                    f"{preserved_block}"
                    f"{retry_diagnostic_block}"
                    f"{older_block}"
                    f"PREVIOUS CODE:\n```python\n{retry_feedback.get('previous_code', '')}\n```\n"
                    f"Fix the issues identified above.\n"
                )
            self.last_retry_context_text = retry_context

        available_functions = ", ".join(scene_context.get("available_functions", []))
        api_docs = scene_context.get("api_docs", "")

        # Use safe string replacement (not .format()) because LLM-generated
        # plan steps may contain { } characters that break Python formatting
        user_prompt = prompt_template
        # Environment-specific patterns (BEHAVIOR needs R1Pro guidance)
        env_specific = ""
        if env_type == "behavior" or "find_object_base_rotate" in (api_docs or ""):
            env_specific = (
                "KEY PATTERNS for R1Pro (use as needed):\n"
                "- find_object_base_rotate('<object>') to locate objects outside FOV (call ONCE per object)\n"
                "- find_object_torso_rotate('<object>') to re-localize BEFORE grasping (required after navigation)\n"
                "- sample_grasp_pose('<object>') returns MULTIPLE candidates — loop with check_object_in_hand()\n"
                "- Use SHORT NATURAL LANGUAGE object names (e.g., 'red radio'), NOT BDDL synsets\n"
            )
        elif env_type == "libero" and "plan_grasp" in available_functions and "get_object_pose" not in available_functions:
            # Non-privileged LIBERO API: vision-based perception, no ground-truth poses
            env_specific = (
                "KEY PATTERNS for non-privileged Franka LIBERO API:\n"
                "- There is NO get_object_pose() or sample_grasp_pose(). You must use vision.\n"
                "\n"
                "PREFERRED APPROACH (GraspNet / plan_grasp first):\n"
                "```python\n"
                "import numpy as np\n"
                "obs = get_observation()\n"
                "cam = obs['agentview']\n"
                "rgb, depth = cam['images']['rgb'], cam['images']['depth']\n"
                "K, T = cam['intrinsics'], cam['pose_mat']\n"
                "if depth.ndim == 3: depth = depth[:, :, 0]  # squeeze channel dim\n"
                "\n"
                "# 1. Segment object\n"
                "masks = segment_sam3_text_prompt(rgb, '<object_name>')\n"
                "if not masks:  # fallback to Molmo pointing\n"
                "    pt = point_prompt_molmo(rgb, '<object_name>')\n"
                "    if pt:\n"
                "        masks = segment_sam3_point_prompt(rgb, list(pt.values())[0])\n"
                "mask = max(masks, key=lambda m: m.get('score', 0))['mask']\n"
                "\n"
                "# 2. Ask GraspNet for candidate grasp poses instead of inventing offsets\n"
                "grasps_cam, scores = plan_grasp(depth, K, mask)\n"
                "best_idx = int(np.argmax(scores))\n"
                "grasp_world = T @ grasps_cam[best_idx]\n"
                "grasp_pos, grasp_quat = decompose_transform(grasp_world)\n"
                "\n"
                "# 3. Execute the selected planner pose\n"
                "open_gripper()\n"
                "goto_pose(grasp_pos, grasp_quat, z_approach=0.12)\n"
                "close_gripper()\n"
                "goto_pose(np.asarray(grasp_pos) + np.array([0.0, 0.0, 0.20]), grasp_quat)\n"
                "```\n"
                "\n"
                "IMPORTANT:\n"
                "- Prefer plan_grasp(depth, intrinsics, mask) or plan_grasp_from_point_clouds(...) for grasp poses.\n"
                "- Do NOT hard-code contact z offsets such as object_top_z + 0.015, zmax - 0.025, or top_z - 0.04 as the primary grasp target.\n"
                "- Use manually constructed centroid/top-down grasps only as an explicit fallback when plan_grasp returns no valid candidates or raises a no-candidate error.\n"
                "- If you use that fallback, record it in RESULT, e.g. RESULT['grasp_fallback'] = 'centroid_top_down'.\n"
                "- depth may have shape (H,W,1) - squeeze to (H,W) before indexing.\n"
                "- plan_grasp returns camera-frame transforms; transform with camera pose (T @ grasp_cam) before decompose_transform().\n"
                "- z_approach in goto_pose controls the approach height before final descent.\n"
                "- inspect_at_wrist(pos) returns close['wrist'] with BOTH close['wrist']['rgb'] and close['wrist']['images']['rgb']; generic camera helpers that read cam['images']['rgb'] can safely consume it.\n"
                "- NEVER use while True or while <condition> loops. Use for _ in range(max_retries) instead.\n"
                "- NEVER catch ValueError or broad exceptions from goto_pose/close_gripper calls — let them propagate.\n"
            )
        elif env_type == "molmospaces":
            task_kind = str(scene_context.get("molmospaces_task_kind", "pick_and_place"))
            env_specific = _molmospaces_recipe_for_kind(
                task_kind,
                scene_context.get("available_functions", []),
            )
            playtime_context = str(scene_context.get("playtime_context") or "").strip()
            if playtime_context:
                env_specific += (
                    "\n\nSENSORIMOTOR / PLAYTIME MEMORY (visual observations, not simulator truth):\n"
                    f"{playtime_context}\n"
                    "Use these as object-affordance hints. For example, if an object moved a lot "
                    "when lightly pushed, approach it more vertically and avoid lateral pre-contact.\n"
                )

        replacements = {
            "{scene_model}": scene_context.get("scene_model", "unknown"),
            "{goal_conditions}": scene_context.get("goal_conditions_nl", plan.get("task_id", "")),
            "{object_scope}": json.dumps(scene_context.get("object_scope", {})),
            "{plan_steps}": "\n".join(step_lines),
            "{available_functions}": available_functions or "(see API docs below)",
            "{api_docs}": api_docs or "See available functions above.",
            "{skill_code}": skill_code,
            "{success_context}": success_context,
            "{subagent_directive}": subagent_directive,
            "{failure_context}": failure_context,
            "{retry_context}": retry_context,
            "{runtime_marker_requirement}": _runtime_marker_requirement(env_type),
            "{env_specific_patterns}": env_specific,
        }
        for placeholder, value in replacements.items():
            user_prompt = user_prompt.replace(placeholder, str(value))
        self.last_user_prompt_text = user_prompt

        system_prompt = (
            "You are a deterministic policy writer for a code-as-policy robot system. "
            "Write ONLY Python code. Do not use while True loops. "
            "Use only the provided imported primitive functions and learned helper functions."
        )
        # BEHAVIOR-1K needs explicit guidance for R1Pro primitives
        env_type = scene_context.get("env_type", "")
        if env_type == "behavior" or "find_object_base_rotate" in (api_docs or ""):
            system_prompt += (
                " ALWAYS use find_object_base_rotate to locate objects and "
                "find_object_torso_rotate to re-localize before grasping."
            )

        def _generate_once(prompt: str, *, images: list[str] | None = None) -> str:
            try:
                if images:
                    response = query_llm_text(
                        system_prompt,
                        prompt,
                        images=images,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        model=_policy_writer_model(),
                    )
                else:
                    response = query_llm_text(
                        system_prompt, prompt, max_tokens=DEFAULT_MAX_TOKENS,
                        model=_policy_writer_model(),
                    )
            except Exception as e:
                logger.error(f"LLM query failed: {e}")
                return ""

            logger.debug(f"Raw LLM response ({len(response)} chars): {response[:200]!r}")
            code = extract_python_code(response)
            if not code.strip():
                code = response.strip()
                logger.debug(f"Using raw response as code ({len(code)} chars)")
            # FIX (soft-constraint enforcement): the prompt says "keep code
            # SHORT (~20-40 lines, ≤80 lines)" but models routinely return
            # 60-100+ lines. We don't reject — that would burn extra LLM
            # calls — but we DO log a WARNING with the offending count so the
            # constraint stops being silent. If a future change wants to make
            # this hard (reject + retry), this is the natural place.
            non_empty = sum(1 for ln in code.splitlines() if ln.strip())
            if non_empty > 80:
                logger.warning(
                    "  policy_writer response exceeded soft cap: %d non-empty "
                    "lines (target 20-40, must-not-exceed 80). Consider "
                    "tightening the prompt or making the cap a hard reject.",
                    non_empty,
                )
            return code

        if self.ensemble_n > 1:
            code = self._generate_ensemble(
                system_prompt,
                user_prompt,
                self.ensemble_n,
                images=retry_diagnostic_images,
            )
        else:
            code = _generate_once(user_prompt, images=retry_diagnostic_images)

        if not code.strip():
            logger.warning(
                "Policy writer produced empty code; retrying once with explicit ```python``` requirement."
            )
            retry_hint = (
                "\n\nIMPORTANT: Output exactly ONE ```python fenced block with complete executable code. "
                "Do not leave the code block empty. Set RESULT as required."
            )
            code = _generate_once(user_prompt + retry_hint, images=retry_diagnostic_images)

        if not code.strip():
            logger.error(
                "Policy writer still empty after retry; raw response may be reasoning-only or truncated. "
                "Check llm_max_tokens in rats/config/default.yaml and API usage (reasoning vs output tokens)."
            )

        return code

    def write_step(
        self,
        *,
        plan: dict[str, Any],
        scene_context: dict[str, Any],
        step_idx: int,
        step: dict[str, Any],
        committed_step_codes: dict[int, str],
        step_retry_feedback: dict[str, Any] | None = None,
        edit_scale: str = "fresh",
        failure_context: str = "",
        success_context: str = "",
        learned_skill_names: list[str] | None = None,
        force_committed_step_indices: set[int] | None = None,
    ) -> str:
        """Generate code for ONE plan step (multiturn-reset mode).

        Returns the BODY of the step only — no ``def main(env):``, no
        ``with step_context(...):``, no ``RESULT = ...``. The multiturn
        orchestrator wraps this body in a step_context block and concatenates
        it with the already-committed prior step bodies into one executable
        ``main(env)``.

        Args:
            plan: Full planner output (used for plan_summary).
            scene_context: Scene info (env_type, available_functions, api_docs).
            step_idx: 0-based index of the step being written.
            step: The plan step dict for this step.
            committed_step_codes: ``{step_idx -> body_code}`` for prior steps
                that already passed per-step verification. Rendered as
                read-only context.
            step_retry_feedback: Output of
                ``MultiturnResetExecutor._build_step_feedback`` from the previous
                retry of THIS step. ``None`` on the first try.
            edit_scale: ``"fresh" | "argument_level" | "rewrite_needed"``.
            failure_context: Past-failure lessons (same string the main
                ``write`` consumes).
            success_context: Past-success snippets.
            learned_skill_names: Available learned-skill function names so the
                writer can call them by name.
        """
        self.last_retry_context_text = ""
        self.last_retry_diagnostic_context = {}
        self.last_retry_diagnostic_image_count = 0
        self.last_user_prompt_text = ""

        env_type = scene_context.get("env_type", "")
        api_docs = scene_context.get("api_docs", "") or ""
        available = scene_context.get("available_functions") or []
        if not isinstance(available, list):
            available = list(available) if isinstance(available, (set, tuple)) else []

        forced_set: set[int] = set(force_committed_step_indices or set())

        plan_steps = plan.get("steps") or []
        plan_summary_lines = []
        for k, s in enumerate(plan_steps):
            if k == step_idx:
                marker = ">>>"
            elif k < step_idx:
                # Distinguish PS-verified ("done") from force-committed
                # ("forced") — the writer must not assume a forced step
                # left the env in the expected state.
                marker = "[forced]" if k in forced_set else "[done]"
            else:
                marker = "     "
            sid = s.get("id") or s.get("step_id") or f"step-{k + 1}"
            plan_summary_lines.append(
                f"  {marker} {k + 1}. ({sid}) {s.get('description', s.get('goal', ''))}"
            )
        plan_summary = "\n".join(plan_summary_lines) or "(plan has no steps)"

        prior_blocks: list[str] = []
        for k in range(step_idx):
            committed = committed_step_codes.get(k, "")
            prior_step = plan_steps[k] if k < len(plan_steps) else {}
            sid = prior_step.get("id") or prior_step.get("step_id") or f"step-{k + 1}"
            desc = prior_step.get("description") or prior_step.get("goal") or ""
            if not committed.strip():
                prior_blocks.append(
                    f"### Step {k + 1} ({sid}): {desc}\n(no committed code — step skipped or empty)"
                )
            elif k in forced_set:
                # Force-committed: PS never approved this step's visual
                # goal, but its retry budget ran out and the code was
                # committed so the rest of the plan could execute.
                # Flag the writer so it doesn't treat the prior env state
                # as guaranteed-correct.
                prior_blocks.append(
                    f"### Step {k + 1} ({sid}) [FORCE-COMMITTED — UNVERIFIED]: {desc}\n"
                    f"WARNING: this step exhausted its per-step retry budget "
                    f"without a PS-succeeded verdict. The code below ran but "
                    f"the visual goal was not confirmed. Do not assume it "
                    f"left the env in the expected state; if your current "
                    f"step depends on this step's effect, re-observe / "
                    f"re-grasp / re-localise before acting on that assumption.\n"
                    f"```python\n{committed}\n```"
                )
            else:
                prior_blocks.append(
                    f"### Step {k + 1} ({sid}): {desc}\n```python\n{committed}\n```"
                )
        prior_steps_block = "\n\n".join(prior_blocks) if prior_blocks else (
            "(this is the first step — env was just reset)"
        )

        feedback_block = ""
        if step_retry_feedback:
            pf = str(step_retry_feedback.get("policy_feedback") or "").strip()
            ps_reason = str(step_retry_feedback.get("ps_reason") or "").strip()
            ps_status = str(step_retry_feedback.get("ps_status") or "").strip()
            feedback_block = (
                "==== PREVIOUS RETRY FEEDBACK FOR THIS STEP ====\n"
                f"per-step verifier status: {ps_status or 'unknown'}\n"
                f"per-step verifier reason: {ps_reason or '(none)'}\n"
                f"corrective note: {pf or '(none)'}\n"
            )

        if edit_scale == "argument_level":
            edit_scale_block = (
                "ARGUMENT-LEVEL FIX: keep the same call sequence and primitives "
                "as the previous attempt, change only specific named arguments "
                "(e.g. grasp_z_offset, target_quaternion, sam3 prompt string). "
                "Do not restructure the step."
            )
        elif edit_scale == "rewrite_needed":
            edit_scale_block = (
                "REWRITE NEEDED: the previous approach for this step is wrong. "
                "Pick a different primitive or call sequence. Do not just tweak "
                "an argument."
            )
        else:
            edit_scale_block = (
                "FRESH ATTEMPT: no prior code for this step. Write the most "
                "direct approach using the available primitives + learned skills."
            )

        # Surface available primitives + learned skills compactly
        learned_names_str = ", ".join(learned_skill_names or []) or "(none)"
        scene_lines = [
            f"env_type: {env_type or 'unknown'}",
            f"available_function_count: {len(available)}",
            f"learned_skill_names: {learned_names_str}",
        ]
        scene_block = "\n".join(scene_lines) + "\n\nAPI DOCS (excerpt):\n" + (api_docs[:6000] or "(none)")

        if failure_context:
            scene_block += "\n\nLESSONS FROM PAST FAILURES:\n" + failure_context[:2000]
        if success_context:
            scene_block += "\n\nLESSONS FROM PAST SUCCESSES:\n" + success_context[:2000]

        template = Path("rats/prompts/policy_writer_single_step.txt").read_text()
        user_prompt = template.format(
            step_number=step_idx + 1,
            prior_step_count=step_idx,
            task_description=plan.get("task_description") or plan.get("goal") or "(no task description)",
            plan_summary=plan_summary,
            prior_steps_block=prior_steps_block,
            step_description=step.get("description") or step.get("goal") or "",
            step_skills=", ".join(step.get("relevant_skills") or []) or "(none)",
            step_notes=step.get("notes") or "(none)",
            feedback_block=feedback_block,
            scene_block=scene_block,
            edit_scale=edit_scale,
            edit_scale_block=edit_scale_block,
        )

        system_prompt = (
            "You are writing one step of a multi-step robot policy. The "
            "multiturn orchestrator runs each step in isolation against a "
            "fresh env reset (committed prior steps replay first). Output "
            "ONLY the body of the current step — no def main, no "
            "with step_context wrapper, no RESULT assignment. The orchestrator "
            "wraps your code. Keep it focused and short."
        )

        self.last_user_prompt_text = user_prompt
        try:
            response = query_llm_text(
                system_prompt,
                user_prompt,
                max_tokens=DEFAULT_MAX_TOKENS,
                model=_policy_writer_model(),
            )
        except Exception as exc:
            logger.error(f"write_step LLM call failed: {exc}")
            return ""

        code = extract_python_code(response)
        if not code.strip():
            code = response.strip()
        return code

    def _generate_ensemble(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int,
        *,
        images: list[str] | None = None,
    ) -> str:
        """Generate N candidates at different temperatures, pick best via LLM synthesis.

        Mirrors CaP-Agent0's parallel ensemble: generate diverse candidates,
        then ask the LLM to pick/synthesize the best one.
        """
        import concurrent.futures

        temps = [0.2 + i * (1.4 / max(n - 1, 1)) for i in range(n)]  # 0.2 to 1.6
        logger.info(f"  Ensemble: generating {n} candidates at temps {[f'{t:.1f}' for t in temps]}")

        candidates: list[str] = []

        def _gen(temp: float) -> str:
            try:
                if images:
                    resp = query_llm_text(
                        system_prompt,
                        user_prompt,
                        images=images,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        temperature=temp,
                        model=_policy_writer_model(),
                    )
                else:
                    resp = query_llm_text(
                        system_prompt,
                        user_prompt,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        temperature=temp,
                        model=_policy_writer_model(),
                    )
                code = extract_python_code(resp)
                return code if code.strip() else resp.strip()
            except Exception as e:
                logger.warning(f"  Ensemble candidate failed (temp={temp:.1f}): {e}")
                return ""

        # Generate candidates in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_gen, t) for t in temps]
            for f in concurrent.futures.as_completed(futures):
                code = f.result()
                if code.strip():
                    candidates.append(code)

        if not candidates:
            logger.warning("  Ensemble: all candidates empty")
            return ""
        if len(candidates) == 1:
            return candidates[0]

        # Ask LLM to pick the best candidate
        logger.info(f"  Ensemble: {len(candidates)} candidates, synthesizing best")
        selection_prompt = (
            "You are selecting the best robot control code from multiple candidates.\n"
            "Pick the candidate most likely to succeed. Prefer code that:\n"
            "1. Uses the correct API functions\n"
            "2. Has bounded retries (for loops, not while)\n"
            "3. Is concise\n"
            "4. Sets RESULT correctly\n\n"
        )
        for i, c in enumerate(candidates):
            selection_prompt += f"--- CANDIDATE {i+1} ---\n```python\n{c[:2000]}\n```\n\n"
        selection_prompt += "Output ONLY the best candidate's Python code, no explanation."

        try:
            resp = query_llm_text(
                system_prompt, selection_prompt, max_tokens=DEFAULT_MAX_TOKENS,
                model=_policy_writer_model(),
            )
            best = extract_python_code(resp)
            if best.strip():
                return best
        except Exception as e:
            logger.warning(f"  Ensemble synthesis failed: {e}")

        return min(candidates, key=len)  # fallback: shortest = simplest


# ----------------------------------------------------------------------
# MolmoSpaces recipe dispatch.
#
# The MolmoSpaces benchmark mixes pick / pick-and-place tasks (which want a
# closed-loop top-down grasp) with articulated open/close tasks (horizontal
# handle grasp + multi-waypoint pull/push along world X). Rather than
# giving the policy writer one catch-all prompt, we dispatch on
# ``scene_context["molmospaces_task_kind"]`` (derived in
# ``loop/molmospaces_utils.py::extract_molmospaces_scene_context``).
# ----------------------------------------------------------------------


_MOLMOSPACES_COMMON_HEADER_BASE = (
    "KEY PATTERNS for non-privileged Franka MolmoSpaces API:\n"
    "- All goto_pose / solve_ik inputs are WORLD frame. solve_ik internally calls\n"
    "  world_pose_to_robot_base_frame(), so you must NOT pre-convert to base frame.\n"
    "- The arm is capped at max_joint_step_rad per sim step (see YAML — typically\n"
    "  0.015–0.08 rad), so blocking moves take hundreds of sim steps. Do not add\n"
    "  hand-rolled while-loops hoping to force faster motion — let goto_pose block.\n"
)

_MOLMOSPACES_SPEED_HEADER = (
    "- RATS-only speed control: prefer numeric speeds before clutter/contact\n"
    "  motions, e.g. set_arm_speed(0.35) for 35% of YAML speed or\n"
    "  set_arm_speed(max_joint_step_rad=0.02) for a direct rad/step cap. Presets\n"
    "  set_arm_speed('slow') / set_arm_speed('very_slow') are good starting\n"
    "  points. Call set_arm_speed('normal') to restore the YAML default later.\n"
)

_MOLMOSPACES_CAMERA_HEADER = (
    "- Cameras: 'agentview' + 'robot0_eye_in_hand'. obs[cam]['intrinsics'] (3x3)\n"
    "  and obs[cam]['pose_mat'] (4x4, camera-to-world) feed mask_to_world_points /\n"
    "  pixel_to_world_point directly.\n"
    "\n"
)


def _molmospaces_available_set(available_functions: object) -> set[str]:
    if isinstance(available_functions, str):
        return {part.strip() for part in available_functions.split(",") if part.strip()}
    try:
        return {str(part).strip() for part in available_functions if str(part).strip()}  # type: ignore[arg-type]
    except TypeError:
        return set()


def _molmospaces_common_header(available_functions: object) -> str:
    available = _molmospaces_available_set(available_functions)
    parts = [_MOLMOSPACES_COMMON_HEADER_BASE]
    if "set_arm_speed" in available:
        parts.append(_MOLMOSPACES_SPEED_HEADER)
    parts.append(_MOLMOSPACES_CAMERA_HEADER)
    return "".join(parts)


def _molmospaces_common_footer(available_functions: object) -> str:
    available = _molmospaces_available_set(available_functions)
    mask_verify_note = (
        "- SAM3 mask validation: before mask_to_world_points(...), check top SAM3 candidates with\n"
        "  vlm_verify(rgb, '<target object>', mask=candidate['mask']) and use the first verified mask;\n"
        "  do not blindly trust SAM3 rank 1 when similar handles/appliances are visible.\n"
        if "vlm_verify" in available
        else ""
    )
    progress_note = (
        "- Use observations, inspect_at_wrist(...), and RESULT fields for in-code progress signals.\n"
        if "inspect_at_wrist" in available
        else "- Use observations and RESULT fields for in-code progress signals.\n"
    )
    return (
        "\nIMPORTANT:\n"
        "- Pass WORLD-frame positions/quaternions to goto_pose/solve_ik. Do NOT call\n"
        "  world_pose_to_robot_base_frame yourself — the API does it internally and\n"
        "  double-conversion will send the arm to the wrong place.\n"
        "- Use rotation_matrix_to_quaternion() for quaternion, NOT a raw tuple.\n"
        + mask_verify_note
        + progress_note
        + "- NEVER use while True or while <condition> loops. Use for _ in range(N).\n"
        "- NEVER catch ValueError or broad exceptions from goto_pose/close_gripper\n"
        "  — let them propagate so the outer retry sees the real failure mode.\n"
    )


def _molmospaces_pick_recipe(available_functions: object = ()) -> str:
    """Closed-loop grasp recipe (pick / pick_and_place)."""
    available = _molmospaces_available_set(available_functions)
    if "grasp_with_wrist_closeloop" not in available:
        locate = (
            "Use search_and_locate_object('<object_name>', max_views=5) for the object cloud/centroid, "
            "then plan_grasp_from_point_clouds(...) and execute with goto_pose/open_gripper/close_gripper."
            if "search_and_locate_object" in available
            else "Use primitive perception: get_observation() → point_prompt_molmo/segment_sam3_* → "
            "mask_to_world_points or pixel_to_world_point → plan_grasp_from_point_clouds(...) → "
            "goto_pose/open_gripper/close_gripper."
        )
        return (
            _molmospaces_common_header(available_functions)
            + "PICK / PICK-AND-PLACE: the closed-loop grasp helper is not available "
            "unless the MolmoSpaces helper+wrist gate is opted in.\n"
            f"{locate}\n"
            "Set RESULT from observed completion evidence and avoid calling unavailable helpers.\n"
            + _molmospaces_common_footer(available_functions)
        )
    return (
        _molmospaces_common_header(available_functions)
        + "RECOMMENDED APPROACH — prefer the closed-loop grasp primitive:\n"
        "```python\n"
        "# Best path: one call that does Molmo → SAM3 → wrist-cam refinement →\n"
        "# grasp → post-lift visual confirmation, with retries. Use this unless\n"
        "# you need an atypical (non-top-down) grasp.\n"
        "r = grasp_with_wrist_closeloop('<object_name>',\n"
        "                               verify_label='<tight_identity_label>',\n"
        "                               max_retries=2)\n"
        "RESULT['grasp'] = r['success']\n"
        "if not r['success']:\n"
        "    goto_home_joint_position()\n"
        "```\n"
        "\n"
        "FALLBACK (manual top-down grasp — only when grasp_with_wrist_closeloop is\n"
        "not appropriate, e.g. sideways handle or drawer pull):\n"
        "```python\n"
        "import numpy as np\n"
        "# Finding-object stage: actively sweep the wrist camera if the first\n"
        "# view is missing/partial. The helper keeps any good single-view cloud\n"
        "# (agentview OR wrist); it only fails when all views fail.\n"
        "loc = search_and_locate_object('<object_name>', max_views=5)\n"
        "assert loc['success'], 'object search returned no points'\n"
        "pts = loc['points_3d']\n"
        "grasp_pos = loc['centroid']\n"
        "\n"
        "# Top-down quaternion (wxyz)\n"
        "grasp_quat = rotation_matrix_to_quaternion(\n"
        "    np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64))\n"
        "\n"
        "open_gripper()\n"
        "goto_pose(grasp_pos + np.array([0,0,0.15]), grasp_quat)   # hover\n"
        "goto_pose(grasp_pos,                       grasp_quat)     # descend\n"
        "close_gripper()\n"
        "goto_pose(grasp_pos + np.array([0,0,0.10]), grasp_quat)   # lift\n"
        "# Confirm from a fresh observation or wrist inspection before declaring success\n"
        "RESULT['grasp'] = True\n"
        "```\n"
        "- Prefer search_and_locate_object first when object visibility is uncertain;\n"
        "  fuse_object_world_points is also relaxed and proceeds with either\n"
        "  agentview or wrist when the other view is not valid.\n"
        + _molmospaces_common_footer(available_functions)
    )


def _molmospaces_open_recipe(available_functions: object = ()) -> str:
    """Drawer / cabinet opening recipe.

    Key differences from the pick recipe:
    - Handles are horizontal, not top-down. Use select_top_down_grasp with
      ``vertical_threshold=0.3`` — returns None when no grasp is vertical
      enough, in which case fall back to argmax(scores).
    - After closing the gripper, DO NOT lift. Pull backward along world -X
      in several short waypoints; a single long goto_pose overshoots the
      per-step joint cap and stalls.
    - Post-condition is the joint angle of the drawer/door, not an object
      pose. Use before/after observations plus the verifier's ``joint_position``
      evidence field as your signals.
    """
    available = _molmospaces_available_set(available_functions)
    needed = {"fuse_object_world_points", "select_grasp_along_direction", "select_horizontal_grasp"}
    if not needed.issubset(available):
        return (
            _molmospaces_common_header(available_functions)
            + "TASK TYPE: ARTICULATED OPEN. The default API intentionally hides "
            "horizontal grasp-selection helpers. Build the handle point cloud, "
            "plan candidate grasps, choose a physically safe side grasp from the "
            "returned matrices/scores yourself, approach from outside the cabinet "
            "face, close on the handle, and pull in short world-frame waypoints. "
            "Do not call unavailable helper functions.\n"
            + _molmospaces_common_footer(available_functions)
        )
    return (
        _molmospaces_common_header(available_functions)
        + "TASK TYPE: ARTICULATED OPEN (drawer / cabinet / oven / fridge /\n"
        "microwave / doorway).\n"
        "\n"
        "CRITICAL: select_top_down_grasp filters FOR top-down grasps. Lowering\n"
        "its vertical_threshold only widens the top-down cone; it does NOT flip\n"
        "the filter to horizontal. Using it here makes the arm dive onto the\n"
        "handle from above and collide with the cabinet face. Use one of the\n"
        "two horizontal selectors below instead. Do NOT use\n"
        "grasp_with_wrist_closeloop either — it assumes top-down.\n"
        "\n"
        "RECOMMENDED APPROACH — direction-aligned grasp + multi-waypoint pull:\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "# 1) Localise the handle. Fused perception now accepts a valid\n"
        "#    agentview OR wrist cloud when the other view misses.\n"
        "pts = fuse_object_world_points('<handle_name>', use_multiview=True)\n"
        "assert len(pts) > 0, 'handle segmentation returned no points'\n"
        "grasps, scores = plan_grasp(pts)\n"
        "\n"
        "# 2) Pick a grasp whose approach axis points INTO the cabinet face,\n"
        "#    i.e. anti-parallel to the pull direction. For a drawer that pulls\n"
        "#    along world -X (robot stands at +X, reaches toward the drawer),\n"
        "#    the gripper approach should point in -X.\n"
        "#    First preference: direction-aligned (physically meaningful).\n"
        "#    Fallback: any horizontal grasp.\n"
        "#    Last resort: highest-scoring candidate (likely top-down — expect\n"
        "#    collisions; at that point prefer re-perception over executing).\n"
        "approach_dir = np.array([-1.0, 0.0, 0.0])  # robot at +X, drawer at -X\n"
        "grasp_pose, _ = select_grasp_along_direction(\n"
        "    grasps, scores, approach_dir, alignment_threshold=0.6)\n"
        "if grasp_pose is None:\n"
        "    grasp_pose, _ = select_horizontal_grasp(\n"
        "        grasps, scores, horizontal_threshold=0.7)\n"
        "if grasp_pose is None:\n"
        "    idx = int(np.argmax(scores))\n"
        "    grasp_pose = grasps[idx]\n"
        "grasp_pos, grasp_quat = decompose_transform(grasp_pose)\n"
        "\n"
        "# 3) Pre-grasp ~15 cm in front of the face along +X so the joint-space\n"
        "#    interpolation from the current pose doesn't sweep through the\n"
        "#    cabinet. goto_pose has no collision-aware planner — it just runs\n"
        "#    IK and interpolates in joint space — so err on the side of a\n"
        "#    larger standoff.\n"
        "open_gripper()\n"
        "goto_pose(grasp_pos + np.array([0.15, 0, 0]), grasp_quat)  # pre-grasp\n"
        "goto_pose(grasp_pos,                          grasp_quat)  # on handle\n"
        "close_gripper()\n"
        "\n"
        "# 4) Pull backward along world -X in 5-10 short waypoints of 1-3 cm.\n"
        "#    One big goto_pose overshoots max_joint_step_rad and stalls.\n"
        "n_steps = 8\n"
        "pull_dx = -0.02  # 2 cm per waypoint -> ~16 cm total drawer travel\n"
        "for i in range(1, n_steps + 1):\n"
        "    tgt = grasp_pos + np.array([pull_dx * i, 0, 0])\n"
        "    goto_pose(tgt, grasp_quat)\n"
        "\n"
        "open_gripper()\n"
        "# Set RESULT from observed before/after state; final verifier checks joint evidence.\n"
        "RESULT['open'] = True\n"
        "```\n"
        "\n"
        "DIRECTION NOTES:\n"
        "- World-frame convention: the robot base front is +X. For articulated\n"
        "  opens where the front-facing sampler placed the robot opposite the\n"
        "  joint's opening side, -X is the direction the handle is pulled.\n"
        "- If ``verification['predicate_status'][0]['evidence']['joint_axis']``\n"
        "  is available, prefer the world-frame joint axis as the pull\n"
        "  direction instead of assuming ±X.\n"
        "\n"
        "GOAL SIGNAL:\n"
        "- The verifier synthesises a single 'opened past threshold' predicate\n"
        "  for opening tasks. The underlying truth is the articulation joint\n"
        "  angle — exposed in verification['predicate_status'][0]['evidence']\n"
        "  ['joint_position'] once get_task_info is plumbed.\n"
        + _molmospaces_common_footer(available_functions)
    )


def _molmospaces_close_recipe(available_functions: object = ()) -> str:
    """Drawer / cabinet closing recipe — mirror of open, but pushing inward."""
    available = _molmospaces_available_set(available_functions)
    needed = {"fuse_object_world_points", "select_grasp_along_direction", "select_horizontal_grasp"}
    if not needed.issubset(available):
        return (
            _molmospaces_common_header(available_functions)
            + "TASK TYPE: ARTICULATED CLOSE. The default API intentionally hides "
            "horizontal grasp-selection helpers. Use primitive perception and "
            "motion only; either push the face directly or choose a physically "
            "safe handle grasp from raw grasp candidates, then close in short "
            "world-frame waypoints. Do not call unavailable helpers.\n"
            + _molmospaces_common_footer(available_functions)
        )
    return (
        _molmospaces_common_header(available_functions)
        + "TASK TYPE: ARTICULATED CLOSE (drawer / cabinet / oven / fridge /\n"
        "microwave / doorway).\n"
        "Closing is often easier than opening — you can push the face instead\n"
        "of grasping the handle — but grasping and pushing from the handle is\n"
        "more robust for doors that bounce open.\n"
        "\n"
        "CRITICAL: ``select_top_down_grasp`` filters FOR top-down. Do not use\n"
        "it for horizontal handles; use select_grasp_along_direction or\n"
        "select_horizontal_grasp. See the open recipe for the rationale.\n"
        "\n"
        "RECOMMENDED APPROACH — direction-aligned grasp + multi-waypoint push:\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "pts = fuse_object_world_points('<handle_name>', use_multiview=True)\n"
        "# fuse_object_world_points proceeds with either agentview or wrist if\n"
        "# only one view detects the handle.\n"
        "assert len(pts) > 0, 'handle segmentation returned no points'\n"
        "grasps, scores = plan_grasp(pts)\n"
        "\n"
        "# Drawer is currently OPEN, so the handle is sticking out toward\n"
        "# the robot at +X. The gripper still approaches INTO the cabinet\n"
        "# face, so the approach direction is -X (same as the open recipe).\n"
        "approach_dir = np.array([-1.0, 0.0, 0.0])\n"
        "grasp_pose, _ = select_grasp_along_direction(\n"
        "    grasps, scores, approach_dir, alignment_threshold=0.6)\n"
        "if grasp_pose is None:\n"
        "    grasp_pose, _ = select_horizontal_grasp(\n"
        "        grasps, scores, horizontal_threshold=0.7)\n"
        "if grasp_pose is None:\n"
        "    idx = int(np.argmax(scores))\n"
        "    grasp_pose = grasps[idx]\n"
        "grasp_pos, grasp_quat = decompose_transform(grasp_pose)\n"
        "\n"
        "open_gripper()\n"
        "# Pre-grasp at +X (outside the face); on-handle is the grasp point\n"
        "# itself; then push inward.\n"
        "goto_pose(grasp_pos + np.array([0.15, 0, 0]), grasp_quat)  # pre-grasp\n"
        "goto_pose(grasp_pos,                          grasp_quat)  # on handle\n"
        "close_gripper()\n"
        "\n"
        "# Push inward along -X in small waypoints.\n"
        "n_steps = 8\n"
        "push_dx = -0.02\n"
        "for i in range(1, n_steps + 1):\n"
        "    tgt = grasp_pos + np.array([push_dx * i, 0, 0])\n"
        "    goto_pose(tgt, grasp_quat)\n"
        "\n"
        "open_gripper()\n"
        "# Set RESULT from observed before/after state; final verifier checks joint evidence.\n"
        "RESULT['close'] = True\n"
        "```\n"
        "\n"
        "GOAL SIGNAL:\n"
        "- Same as opening, but the verifier expects the complementary 'closed'\n"
        "  predicate. In OpeningTask the reward is (1 - percent_open) for close\n"
        "  tasks, so full-closure drives verification['success'] true.\n"
        + _molmospaces_common_footer(available_functions)
    )


def _molmospaces_playtime_recipe(available_functions: object = ()) -> str:
    speed_hint = (
        "- Prefer slow arm speed near contact: set_arm_speed('slow') or set_arm_speed('very_slow').\n"
        if "set_arm_speed" in _molmospaces_available_set(available_functions)
        else "- Move gently with small bounded goto_pose waypoints near contact.\n"
    )
    return (
        _molmospaces_common_header(available_functions)
        + "PLAYTIME / SENSORIMOTOR TASK PATTERN:\n"
        "- This is a grounded-state-verified exploratory action, not a built-in simulator reward task.\n"
        "- First call get_observation(), localize the named object with Molmo/SAM3 or fused points.\n"
        "- Execute a small, gentle, bounded interaction that makes the intended contact/action visible.\n"
        + speed_hint
        + "- For push/tap/touch: approach from a safe hover, move a few cm in the intended direction, retreat, observe.\n"
        "- For lift/drop/shake: keep height low over a support surface; avoid throwing or dropping off tables.\n"
        "- Set RESULT with what you observed if possible, but grounded before/after object and robot state are the acceptance gate.\n"
        + _molmospaces_common_footer(available_functions)
    )


def _molmospaces_recipe_for_kind(task_kind: str, available_functions: object = ()) -> str:
    kind = (task_kind or "").lower()
    if kind == "playtime":
        return _molmospaces_playtime_recipe(available_functions)
    if kind == "open":
        return _molmospaces_open_recipe(available_functions)
    if kind == "close":
        return _molmospaces_close_recipe(available_functions)
    # pick, pick_and_place, nav, other → the closed-loop grasp recipe is the
    # most useful fallback because it's the only one with a do-everything
    # helper (grasp_with_wrist_closeloop).
    return _molmospaces_pick_recipe(available_functions)
