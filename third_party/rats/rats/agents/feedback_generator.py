"""Feedback Generator: control router and decision engine.

Aggregates execution feedback, decides next action, triggers downstream effects.
NOT a training module - no gradients, no RL updates.

Decision logic:
  - Success -> extract skills -> add to library -> next task
  - Retry (< MAX_RETRIES) -> package feedback -> send to Policy Writer
  - Skip (>= MAX_RETRIES) -> log failure -> next task
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
from pathlib import Path
from typing import Any

from rats.agents.base_agent import image_to_data_url, query_llm_json, video_file_to_data_url


_STEP_INDEX_RE = re.compile(r"\d+")


def _step_index(step_id: str) -> str:
    """Pull the integer index out of any step_id naming variant.

    Diagnoser and writer disagreed on naming convention, which was
    silently breaking Mechanism A: the plan + writer use ``step-3`` /
    ``# step-3:`` while the diagnoser invents ``step_3_pick_pan``.
    Matching on the bare integer bridges the two without forcing a
    rename on either side.
    """
    if not step_id:
        return ""
    m = _STEP_INDEX_RE.search(step_id)
    return m.group(0) if m else ""


def _extract_step_segment(code: str, step_id: str) -> str:
    """Extract the code region attributed to a given plan step_id.

    Matches step headers by *integer index*, not by full string, so all
    of these point to the same block:
        # step-3: Pick the pan      # writer convention
        # Step 3 Pick the pan       # alt writer convention
        # step_3_pick_pan           # diagnoser-renamed id

    Returns the block from the matching header up to the next ``# step
    <N>`` header or EOF. Empty string if no header carries the index.
    """
    if not code or not step_id:
        return ""
    idx = _step_index(step_id)
    if not idx:
        return ""
    # Find the comment header line that "owns" this step. We accept any
    # line of the form `# step <stuff>` whose <stuff> contains the
    # target index `<idx>` with non-digit boundaries on both sides
    # (avoids matching "step-30" when looking for "3"). The flexible
    # middle matches combined headers like `# step-2/3:` and `# step-6/7:`
    # that the writer occasionally emits.
    idx_in_line = re.compile(rf"(?<!\d){idx}(?!\d)")
    header_re = re.compile(r"^\s*#\s*step\b[^\n]*", re.IGNORECASE | re.MULTILINE)
    match_pos: int | None = None
    match_end: int = 0
    next_step_pos: int | None = None
    for m in header_re.finditer(code):
        line = code[m.start():m.end()]
        owns_idx = bool(idx_in_line.search(line))
        if match_pos is None:
            if owns_idx:
                match_pos = m.start()
                match_end = m.end()
            continue
        # We already have a match; the very next `# step <N>` header
        # (whose N is NOT the same idx) terminates the segment.
        if not owns_idx:
            next_step_pos = m.start()
            break
    if match_pos is None:
        return ""
    end = next_step_pos if next_step_pos is not None else len(code)
    return code[match_pos:end].strip("\n")


def _coerce_self_report_bool(value: Any) -> bool | None:
    """Convert explicit step self-report scalars to bool.

    Policy RESULT dicts often also carry structured runtime data such as
    world points, masks, or bounding boxes. Those must not be treated as
    truth values: numpy raises on multi-element arrays, and container
    truthiness would turn diagnostic metadata into fake step success.
    """
    if isinstance(value, bool):
        return value
    if value is None or isinstance(value, dict):
        return None

    # numpy scalar / ndarray support without making numpy a hard import.
    size = getattr(value, "size", None)
    item = getattr(value, "item", None)
    if size is not None and callable(item):
        try:
            if int(size) != 1:
                return None
            return _coerce_self_report_bool(item())
        except Exception:
            return None

    if isinstance(value, (list, tuple, set)):
        return None

    if isinstance(value, bytes):
        try:
            value = value.decode()
        except Exception:
            return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "success", "succeeded", "ok", "passed", "1"}:
            return True
        if normalized in {"false", "no", "n", "failure", "failed", "fail", "0"}:
            return False
        return None

    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return bool(numeric)

    return None


class FeedbackGenerator:
    def __init__(
        self,
        max_retries: int = 5,
        *,
        model: str | None = None,
        molmospaces_model: str = "google/gemini-3.1-pro-preview",
        molmospaces_max_tokens: int = 65536,
        molmospaces_max_frames: int = 16,
    ) -> None:
        self.max_retries = max_retries
        resolved = (
            model
            if model is not None
            else os.getenv("RATS_FEEDBACK_GENERATOR_MODEL", "")
        )
        resolved = str(resolved or "").strip()
        self.model = (
            None
            if resolved.lower() in {"", "default", "global", "inherit"}
            else resolved
        )
        self.molmospaces_model = molmospaces_model
        self.molmospaces_max_tokens = molmospaces_max_tokens
        self.molmospaces_max_frames = molmospaces_max_frames

    def generate(
        self,
        execution_result: dict[str, Any],
        verification: dict[str, Any],
        diagnosis: dict[str, Any],
        attempt: int,
        *,
        plan: dict[str, Any] | None = None,
        code: str = "",
        task_language: str = "",
        existing_skills: list[dict[str, Any]] | None = None,
        multiturn_reset_partial_success: bool = False,
    ) -> dict[str, Any]:
        """Route feedback based on execution results.

        Returns:
            Dict with:
              - action: "success" | "retry" | "skip"
              - message: str
              - new_skills: list[dict] (if success, extracted skills)
              - retry_package: dict (if retry, for Policy Writer)

        ``multiturn_reset_partial_success``: when True, the rollout
        reached a verifier-success state but at least one plan step was
        force-committed without a PS-succeeded verdict. Per Option B
        semantics (metrics still success, skill library writes blocked),
        the LLM skill-extraction call is skipped — the unverified step
        codes must not become reusable library skills.
        """
        # SUCCESS
        if verification.get("success"):
            if multiturn_reset_partial_success:
                # Skip the expensive LLM skill-extraction call; the
                # rollout contained unverified force-committed steps and
                # the lifelong loop will block library writes anyway.
                return {
                    "action": "success",
                    "message": (
                        "Task completed successfully (multiturn-reset "
                        "partial_success: force-committed step(s) "
                        "unverified — skill extraction skipped)."
                    ),
                    "new_skills": [],
                    "retry_package": None,
                }
            new_skills = self._extract_skills(
                code, plan, execution_result,
                task_language=task_language,
                existing_skills=existing_skills,
            )
            return {
                "action": "success",
                "message": "Task completed successfully.",
                "new_skills": new_skills,
                "retry_package": None,
            }

        # RETRY — pure-visual feedback only.
        #
        # We deliberately do NOT forward verifier symbolic outputs
        # (state_hint, predicate_status, satisfied/unsatisfied_conditions,
        # verifier_*). Those read the simulator's ground-truth state and
        # would leak privileged info into the policy writer's retry prompt.
        # The retry loop is driven entirely by the FailureDiagnoser's
        # vision-LLM output (which only sees RGB + code + plan +
        # on-model affordance hints — no sim state).
        if attempt < self.max_retries and diagnosis.get("confidence", 0) >= 0.3:
            vps = diagnosis.get("visual_predicate_status", []) or []
            # Mechanism A: harvest "preserved" code — segments from this
            # attempt's code that the diagnoser marked visually ✓ for a
            # given plan step (with end-of-trajectory stability per the
            # diagnoser prompt). These go to the next attempt's
            # policy_writer with "keep this, don't redo what already
            # worked" guidance, so retries focus on the ✗ steps instead
            # of rediscovering good sub-sequences.
            #
            # Ground-truth guard (P1): a bare VLM ✓ is unreliable — we've
            # seen the diagnoser confidently claim a sub-behavior
            # succeeded when the sim's reward says otherwise. Cross-check
            # against the policy writer's OWN self-report in
            # `user_result` (the final RESULT dict from the executed
            # code). Only preserve a segment when BOTH the VLM ✓ AND the
            # code's own gate fired for that step. This is
            # non-privileged: `user_result` comes from the code's runtime
            # variables, not from simulator state.
            #
            # If `user_result` is missing / not a dict (older executions,
            # mock env, or code that didn't follow the RESULT convention),
            # degrade to VLM-only behavior so we don't silently turn off
            # mechanism A on inputs that never opted into the cross-check.
            user_result = execution_result.get("user_result")
            has_self_report = isinstance(user_result, dict) and bool(user_result)

            # Pre-index user_result and plan step.id by integer index so the
            # cross-check matches even when the diagnoser renamed step_ids
            # ("step_3_pick_pan") and the plan/RESULT use plain ids
            # ("step-3" / "step_3_pick"). Without this the self-report
            # guard silently rejected every preserved segment in iter 3 /
            # iter 4 because of the naming mismatch.
            #
            # Distinguish "explicit False" from "missing key": the writer
            # often uses ad-hoc step keys (e.g. RESULT has step_3_grasp_pan
            # but no step_2_*). A missing key means "no self-report signal"
            # — fall through to VLM only. An explicit False means the
            # agent's own gate said it failed — veto preservation.
            self_report_by_idx: dict[str, bool] = {}
            for k, v in (user_result or {}).items():
                idx = _step_index(str(k))
                self_report = _coerce_self_report_bool(v)
                if not idx or self_report is None:
                    # Skip nested details and structured runtime metadata
                    # such as np.array world points / masks / bboxes. We only
                    # want explicit top-level step_<n> scalar self-reports.
                    continue
                # Last write per index wins; if multiple keys collide, any
                # truthy one is enough to keep preservation alive.
                self_report_by_idx[idx] = (
                    self_report_by_idx.get(idx, False) or self_report
                )

            plan_ids_by_idx: dict[str, str] = {}
            if plan:
                for s in plan.get("steps", []):
                    sid = str(s.get("id") or "")
                    idx = _step_index(sid)
                    if idx and idx not in plan_ids_by_idx:
                        plan_ids_by_idx[idx] = sid

            preserved: list[dict[str, Any]] = []
            for entry in vps:
                if not entry.get("visually_satisfied"):
                    continue
                raw_id = str(entry.get("step_id") or "")
                idx = _step_index(raw_id)
                if not idx:
                    continue
                # Veto only when the agent's RESULT carries an explicit
                # False for this step's index. Missing keys = no signal.
                if has_self_report and idx in self_report_by_idx \
                        and not self_report_by_idx[idx]:
                    continue
                snippet = _extract_step_segment(code, raw_id)
                if not snippet:
                    continue
                # Pin canonical plan id when we can — the policy_writer
                # prompt re-renders these alongside the new plan, and a
                # mismatched id ("step_3_pick_pan" vs "step-3") would
                # confuse the writer about which plan step is preserved.
                preserved.append({
                    "step_id": plan_ids_by_idx.get(idx, raw_id),
                    "description": entry.get("description", ""),
                    "evidence": entry.get("evidence", ""),
                    "code_snippet": snippet,
                })

            # If the diagnoser flagged a structural plan problem, signal
            # the outer loop to call planner.refine_plan before next code
            # attempt, and clear preserved_code_segments — step_ids may
            # shift after re-plan, so preserving across a replan is unsafe.
            replan = bool(diagnosis.get("plan_issue", False))
            plan_issue_reason = str(diagnosis.get("plan_issue_reason", "") or "")
            if replan:
                preserved = []

            # Sub-agent isolation: when the diagnoser identifies a
            # specific sub-behavior that keeps failing and can be
            # practiced in isolation, flag it. Outer loop will spawn a
            # SubAgent (its own mini retry loop + skill extraction) and
            # add any learned skill to the library before the next main
            # attempt. Has priority over replan — if we can learn the
            # sub-skill first, the existing plan might start working.
            subagent_target = diagnosis.get("subagent_skill_target") or None
            subagent_reason = str(
                diagnosis.get("subagent_skill_target_reason", "") or ""
            )

            retry_package = {
                "attempt": attempt + 1,
                "stderr": execution_result.get("stderr", ""),
                "diagnosis": diagnosis.get("policy_feedback", ""),
                "failed_step": diagnosis.get("failed_step", ""),
                "failure_mode": diagnosis.get("failure_mode", ""),
                "previous_code": code,
                # Per-plan-step visual verdicts from the diagnoser. Each
                # entry: {step_id, description, visually_satisfied, evidence}.
                # step_id comes from the agent's own plan — no privileged
                # info leaks into retry_package via this field.
                "visual_predicate_status": vps,
                # Inner-loop partial-success preservation (Mechanism A).
                "preserved_code_segments": preserved,
                # Replan signal: when True, lifelong_loop must call
                # planner.refine_plan(...) and update `plan` before the
                # policy writer runs. plan_issue_reason carries the
                # diagnoser's 1-2 sentence structural diagnosis.
                "replan": replan,
                "plan_issue_reason": plan_issue_reason,
                # Sub-agent spawn signal. When non-null, lifelong_loop
                # should run a SubAgent focused on practicing the named
                # sub-behavior in isolation and add any learned skill to
                # the library before the next main attempt.
                "subagent_skill_target": subagent_target,
                "subagent_skill_target_reason": subagent_reason,
                # Parallel-sub-agent approach directives forwarded
                # straight from the diagnoser. The lifelong_loop
                # decides whether to dispatch in parallel based on
                # this list's length.
                "subagent_approaches": diagnosis.get(
                    "subagent_approaches", [],
                ) or [],
                # Non-privileged runtime artifacts collected by the robot API:
                # pointcloud/grasp/motion summaries, raw numeric excerpts, and
                # visualization images. These are produced by the policy's own
                # perception calls, not simulator ground truth, and let the
                # policy writer repair perception failures instead of seeing
                # only a generic stderr/code_bug string.
                "diagnostic_context": diagnosis.get("diagnostic_context") or {},
                # Retry-edit scale from the diagnoser. One of
                # "argument_level" | "rewrite_needed" | None. The policy
                # writer renders an explicit banner above the diagnosis
                # block when set, so the next attempt knows whether to
                # preserve the call sequence or rewrite. None means no
                # banner (older runs / cached responses) and the writer
                # falls back to the prose hint in `diagnosis`.
                "edit_scale": diagnosis.get("edit_scale"),
            }
            return {
                "action": "retry",
                "message": diagnosis.get("policy_feedback", "Retrying with feedback."),
                "new_skills": [],
                "retry_package": retry_package,
            }

        # SKIP (max retries exceeded or low confidence diagnosis)
        return {
            "action": "skip",
            "message": f"Skipping after {attempt} attempts. Last failure: {diagnosis.get('failure_reason', 'unknown')}",
            "new_skills": [],
            "retry_package": None,
        }

    def generate_molmospaces(
        self,
        execution_result: dict[str, Any],
        verification: dict[str, Any],
        attempt: int,
        *,
        plan: dict[str, Any] | None = None,
        code: str = "",
        task_in_progress: bool = False,
        prior_attempts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate MolmoSpaces retry feedback without the legacy diagnoser.

        This is the MolmoSpaces-only simplified path:

            execution + per-step VLM verifier -> retry feedback -> policy writer

        The feedback writer may inspect trajectory frames, the final verifier
        result, the executed code, and the compact per-step verifier summary.
        It does *not* receive or forward raw pointclouds, segmentation masks,
        grasp candidates, artifact images, or numeric diagnostic files. The
        returned retry_package intentionally leaves ``diagnostic_context`` empty
        so PolicyWriter cannot attach those intermediate artifacts.
        """
        if verification.get("success"):
            new_skills = self._extract_skills(code, plan, execution_result)
            diagnosis = {
                "visual_success": True,
                "failed_step": None,
                "failure_reason": "",
                "policy_feedback": "Task verifier accepted the MolmoSpaces attempt.",
                "confidence": verification.get("confidence", 1.0),
                "failure_mode": "none",
                "visual_predicate_status": self._visual_status_from_per_step(
                    plan,
                    self._per_step_payload(execution_result),
                ),
                "diagnosis_source": "molmospaces_feedback_generator",
            }
            return {
                "action": "success",
                "message": "Task completed successfully.",
                "new_skills": new_skills,
                "retry_package": None,
                "diagnosis": diagnosis,
            }

        per_step_payload = self._per_step_payload(execution_result)
        if self._is_timeout_failure(execution_result):
            fallback_diagnosis = self._timeout_molmospaces_diagnosis(
                execution_result,
                verification,
                plan=plan,
                per_step_payload=per_step_payload,
            )
        else:
            fallback_diagnosis = self._fallback_molmospaces_diagnosis(
                execution_result,
                verification,
                plan=plan,
                per_step_payload=per_step_payload,
            )

        if attempt >= self.max_retries:
            return {
                "action": "skip",
                "message": (
                    "Skipping after "
                    f"{attempt} attempts. Last failure: "
                    f"{fallback_diagnosis.get('failure_reason', 'unknown')}"
                ),
                "new_skills": [],
                "retry_package": None,
                "diagnosis": fallback_diagnosis,
            }

        diagnosis = self._query_molmospaces_retry_feedback(
            execution_result,
            verification,
            plan=plan,
            code=code,
            per_step_payload=per_step_payload,
            task_in_progress=task_in_progress,
            prior_attempts=prior_attempts,
            fallback=fallback_diagnosis,
        )
        confidence = self._safe_float(diagnosis.get("confidence"), 0.0)
        if confidence < 0.3:
            # Keep retry loop actionable even when the VLM is uncertain.
            diagnosis = fallback_diagnosis

        vps = diagnosis.get("visual_predicate_status") or []
        if not vps:
            vps = self._visual_status_from_per_step(plan, per_step_payload)
            diagnosis["visual_predicate_status"] = vps

        preserved = self._preserved_segments_from_visual_status(
            execution_result,
            plan,
            code,
            vps,
        )
        replan = bool(diagnosis.get("plan_issue", False))
        if replan:
            preserved = []

        retry_package = {
            "attempt": attempt + 1,
            "stderr": execution_result.get("stderr", ""),
            "diagnosis": diagnosis.get("policy_feedback", ""),
            "failed_step": diagnosis.get("failed_step", ""),
            "failure_mode": diagnosis.get("failure_mode", ""),
            "previous_code": code,
            "visual_predicate_status": vps,
            "preserved_code_segments": preserved,
            "replan": replan,
            "plan_issue_reason": str(diagnosis.get("plan_issue_reason", "") or ""),
            "subagent_skill_target": None,
            "subagent_skill_target_reason": "",
            "subagent_approaches": [],
            # Deliberately empty for MolmoSpaces: do not pass segmentation
            # masks, pointclouds, grasp candidates, artifact images, or raw
            # numeric diagnostics into PolicyWriter.
            "diagnostic_context": {},
            "feedback_source": "molmospaces_per_step_vlm",
            "edit_scale": diagnosis.get("edit_scale"),
        }
        return {
            "action": "retry",
            "message": diagnosis.get("policy_feedback", "Retrying with feedback."),
            "new_skills": [],
            "retry_package": retry_package,
            "diagnosis": diagnosis,
        }

    def _query_molmospaces_retry_feedback(
        self,
        execution_result: dict[str, Any],
        verification: dict[str, Any],
        *,
        plan: dict[str, Any] | None,
        code: str,
        per_step_payload: dict[str, Any],
        task_in_progress: bool,
        prior_attempts: list[dict[str, Any]] | None,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        images, videos, media_manifest = self._molmospaces_feedback_media(execution_result)
        prompt_plan = self._compact_plan(plan)
        prompt_per_step = self._prompt_safe_payload(self._compact_per_step_payload(per_step_payload))
        prompt_verification = self._prompt_safe_payload(self._compact_verification(verification))
        prompt_prior = self._prompt_safe_payload(self._compact_prior_attempts(prior_attempts))
        prompt_exec = {
            "stdout_tail": str(execution_result.get("stdout", "") or "")[-1200:],
            "stderr_tail": str(execution_result.get("stderr", "") or "")[-1200:],
            "success": execution_result.get("success"),
            "task_completed": execution_result.get("task_completed"),
        }
        prompt_exec = self._prompt_safe_payload(prompt_exec)
        system_prompt = (
            "You are the MolmoSpaces RATS retry feedback generator. Produce "
            "the exact feedback package content needed for the next robot "
            "policy-writing attempt. Use only: the task plan, executed code, "
            "stdout/stderr, final verifier result, compact per-step verifier "
            "summary, prior-attempt summaries, and attached robot trajectory "
            "video. Do not ask to inspect raw segmentation masks, pointclouds, "
            "grasp candidates, artifact files, or hidden simulator state. Do "
            "not include artifact paths or raw coordinates in the policy "
            "feedback. If final verification says the task failed, give a "
            "concrete corrective action; do not output no-op feedback. Respond "
            "only with JSON."
        )
        user_prompt = (
            "TASK / PLAN:\n"
            f"{json.dumps(prompt_plan, indent=2, sort_keys=True)}\n\n"
            "EXECUTION SUMMARY:\n"
            f"{json.dumps(prompt_exec, indent=2, sort_keys=True)}\n\n"
            "FINAL TASK VERIFIER RESULT:\n"
            f"{json.dumps(prompt_verification, indent=2, sort_keys=True)}\n\n"
            "PER-STEP VLM VERIFIER SUMMARY:\n"
            f"{json.dumps(prompt_per_step, indent=2, sort_keys=True)}\n\n"
            "PRIOR ATTEMPTS IN THIS ITERATION:\n"
            f"{json.dumps(prompt_prior, indent=2, sort_keys=True)}\n\n"
            "MEDIA MANIFEST (media attached in this order; video entries are full motion clips):\n"
            f"{json.dumps(media_manifest, indent=2, sort_keys=True)}\n\n"
            "EXECUTED POLICY CODE:\n"
            f"```python\n{code[-12000:]}\n```\n\n"
            f"TASK_IN_PROGRESS_WITHOUT_ENV_RESET: {bool(task_in_progress)}\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            "  \"visual_success\": boolean,\n"
            "  \"failed_step\": \"step id or 'verification'\",\n"
            "  \"failure_reason\": \"short reason\",\n"
            "  \"failure_mode\": \"grasp_failure|navigation_error|wrong_object|wrong_affordance|collision|code_bug|timeout|nothing_happened|partial_completion|none\",\n"
            "  \"confidence\": number between 0 and 1,\n"
            "  \"plan_issue\": boolean,\n"
            "  \"plan_issue_reason\": \"only if plan_issue is true\",\n"
            "  \"visual_predicate_status\": [\n"
            "    {\"step_id\": \"plan step id\", \"description\": \"plan step description\", \"visually_satisfied\": boolean, \"evidence\": \"visual/per-step evidence\"}\n"
            "  ],\n"
            "  \"policy_feedback\": \"concrete code-level instruction for the next attempt; name the step/object feature/motion change; do not mention raw artifacts\"\n"
            "}\n"
        )
        try:
            # ``reasoning_effort="low"`` is explicit. Without it the call
            # falls back to the global default ("medium"), which on
            # Gemini-3.1-pro can burn most of the max_tokens budget on
            # internal reasoning before emitting the JSON answer — the
            # output then gets truncated mid-object. The retry-feedback
            # task is structured extraction (verifier verdict + per-step
            # summary → policy_feedback JSON), not open-ended reasoning,
            # so "low" is appropriate and leaves the full budget for the
            # actual output.
            parsed = query_llm_json(
                system_prompt,
                user_prompt,
                images=images,
                videos=videos,
                model=self.molmospaces_model,
                max_tokens=self.molmospaces_max_tokens,
                reasoning_effort="low",
            )
        except Exception as exc:
            out = dict(fallback)
            out["policy_feedback"] = (
                f"{out.get('policy_feedback', '')} "
                f"(MolmoSpaces retry-feedback VLM call failed: {type(exc).__name__}: {exc})"
            ).strip()
            out["diagnosis_source"] = "molmospaces_feedback_generator_fallback"
            return out

        if not isinstance(parsed, dict):
            parsed = {}
        out = dict(fallback)
        for key in (
            "visual_success",
            "failed_step",
            "failure_reason",
            "failure_mode",
            "confidence",
            "plan_issue",
            "plan_issue_reason",
            "visual_predicate_status",
            "policy_feedback",
        ):
            if key in parsed and parsed.get(key) not in (None, ""):
                out[key] = parsed.get(key)
        out["confidence"] = max(
            0.0,
            min(1.0, self._safe_float(out.get("confidence"), 0.0)),
        )
        out["diagnosis_source"] = "molmospaces_feedback_generator"
        out["feedback_input_image_manifest"] = media_manifest
        out["feedback_input_media_manifest"] = media_manifest

        # Conservative guard: final task verifier failed, so do not allow a
        # no-op or "already done" policy feedback to reach PolicyWriter.
        feedback = str(out.get("policy_feedback", "") or "").lower()
        failed_final = not bool(verification.get("success"))
        noopish = (
            "no corrective action" in feedback
            or "no further action" in feedback
            or "no-op" in feedback
            or str(out.get("failure_mode", "")).lower() == "none"
        )
        if failed_final and noopish:
            return fallback
        return out

    def _fallback_molmospaces_diagnosis(
        self,
        execution_result: dict[str, Any],
        verification: dict[str, Any],
        *,
        plan: dict[str, Any] | None,
        per_step_payload: dict[str, Any],
    ) -> dict[str, Any]:
        vps = self._visual_status_from_per_step(plan, per_step_payload)
        failed = next((entry for entry in vps if not entry.get("visually_satisfied")), None)
        failed_step = (failed or {}).get("step_id") or "verification"
        reason = (
            (failed or {}).get("evidence")
            or verification.get("short_reason")
            or execution_result.get("stderr")
            or "Final verifier did not accept the task."
        )
        mode = "code_bug" if execution_result.get("stderr") else "partial_completion"
        return {
            "visual_success": False,
            "failed_step": failed_step,
            "failure_reason": str(reason)[:500],
            "policy_feedback": (
                f"Retry from {failed_step}: {str(reason)[:350]}. "
                "Re-observe the current scene, then execute a concrete corrective "
                "motion for the failed step rather than returning RESULT only."
            ),
            "confidence": 0.55,
            "failure_mode": mode,
            "visual_predicate_status": vps,
            "plan_issue": False,
            "plan_issue_reason": "",
            "diagnosis_source": "molmospaces_feedback_generator_fallback",
        }

    @staticmethod
    def _is_timeout_failure(execution_result: dict[str, Any]) -> bool:
        if execution_result.get("timeout"):
            return True
        artifacts = execution_result.get("artifacts")
        if isinstance(artifacts, dict) and artifacts.get("timeout"):
            return True
        text = " ".join(
            str(execution_result.get(k) or "")
            for k in ("stderr", "timeout_message")
        ).lower()
        return "timeout" in text or "timed out" in text

    def _timeout_molmospaces_diagnosis(
        self,
        execution_result: dict[str, Any],
        verification: dict[str, Any],
        *,
        plan: dict[str, Any] | None,
        per_step_payload: dict[str, Any],
    ) -> dict[str, Any]:
        vps = self._visual_status_from_per_step(plan, per_step_payload)
        stderr = str(execution_result.get("stderr") or execution_result.get("timeout_message") or "")
        budget = (
            execution_result.get("timeout_seconds")
            or (execution_result.get("artifacts") or {}).get("timeout_seconds")
            or "the execution budget"
        )
        reason = (
            verification.get("short_reason")
            or stderr
            or f"Policy execution timed out after {budget}."
        )
        return {
            "visual_success": False,
            "failed_step": "timeout",
            "failure_reason": str(reason),
            "policy_feedback": (
                f"The MolmoSpaces policy timed out before verification could "
                f"accept the task (budget={budget}). Diagnose this as a "
                "timeout, not a generic code bug: remove blocking waits, bound "
                "all retry/search loops, add explicit phase time budgets, and "
                "after each long perception/motion call check whether progress "
                "was made before continuing. If a motion call may block, split "
                "it into shorter waypoints or choose a simpler corrective "
                "motion for the failed step."
            ),
            "confidence": 0.75,
            "failure_mode": "timeout",
            "visual_predicate_status": vps,
            "plan_issue": False,
            "plan_issue_reason": "",
            "diagnosis_source": "molmospaces_timeout_feedback_generator",
        }

    @staticmethod
    def _per_step_payload(execution_result: dict[str, Any]) -> dict[str, Any]:
        artifacts = execution_result.get("artifacts", {}) or {}
        payload = artifacts.get("per_step_verification")
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _compact_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(plan, dict):
            return {"task_id": "", "steps": []}
        return {
            "task_id": plan.get("task_id", ""),
            "steps": [
                {
                    "id": step.get("id") or step.get("step_id") or f"step-{idx + 1}",
                    "description": step.get("description", ""),
                    "notes": step.get("notes", ""),
                }
                for idx, step in enumerate(plan.get("steps", []) or [])
                if isinstance(step, dict)
            ],
        }

    @staticmethod
    def _compact_verification(verification: dict[str, Any]) -> dict[str, Any]:
        keep = {
            "success",
            "task_completed",
            "short_reason",
            "observed_effect",
            "confidence",
        }
        return {k: verification.get(k) for k in keep if k in verification}

    @staticmethod
    def _compact_prior_attempts(
        prior_attempts: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for attempt in (prior_attempts or [])[-4:]:
            if not isinstance(attempt, dict):
                continue
            out.append({
                "attempt_idx": attempt.get("attempt_idx"),
                "policy_feedback": str(attempt.get("policy_feedback", "") or "")[:800],
                "failure_mode": attempt.get("failure_mode", ""),
                "visual_predicate_status": attempt.get("visual_predicate_status", []) or [],
            })
        return out

    @staticmethod
    def _compact_per_step_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict) or not payload:
            return {"enabled": False, "steps": [], "summary_text": ""}
        return {
            "enabled": bool(payload.get("enabled", False)),
            "verifier_backend": payload.get("verifier_backend"),
            "verifier_model": payload.get("verifier_model"),
            "summary_text": str(payload.get("summary_text", "") or "")[:3000],
            "steps": [
                {
                    "step_id": step.get("step_id"),
                    "goal": step.get("goal"),
                    "success": step.get("success"),
                    "status": step.get("status"),
                    "confidence": step.get("confidence"),
                    "reason": str(step.get("reason", "") or "")[:500],
                    "evidence_keys": step.get("evidence_keys", []) or [],
                }
                for step in (payload.get("steps") or [])
                if isinstance(step, dict)
            ],
        }

    def _visual_status_from_per_step(
        self,
        plan: dict[str, Any] | None,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        plan_steps = self._compact_plan(plan).get("steps", [])
        by_idx: dict[str, dict[str, Any]] = {}
        for step in (payload.get("steps") or []):
            if not isinstance(step, dict):
                continue
            idx = _step_index(str(step.get("step_id", "")))
            if idx:
                by_idx[idx] = step
        out: list[dict[str, Any]] = []
        for i, plan_step in enumerate(plan_steps, start=1):
            sid = str(plan_step.get("id") or f"step-{i}")
            matched = by_idx.get(_step_index(sid), {})
            success = bool(matched.get("success", False))
            reason = str(matched.get("reason") or matched.get("status") or "No per-step verdict available.")
            out.append({
                "step_id": sid,
                "description": plan_step.get("description", ""),
                "visually_satisfied": success,
                "evidence": reason,
            })
        if out:
            return out
        for step in (payload.get("steps") or []):
            if isinstance(step, dict):
                out.append({
                    "step_id": step.get("step_id", ""),
                    "description": step.get("goal", ""),
                    "visually_satisfied": bool(step.get("success", False)),
                    "evidence": str(step.get("reason", "") or step.get("status", "")),
                })
        return out

    def _preserved_segments_from_visual_status(
        self,
        execution_result: dict[str, Any],
        plan: dict[str, Any] | None,
        code: str,
        visual_predicate_status: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        user_result = execution_result.get("user_result")
        has_self_report = isinstance(user_result, dict) and bool(user_result)
        self_report_by_idx: dict[str, bool] = {}
        for k, v in (user_result or {}).items():
            idx = _step_index(str(k))
            self_report = _coerce_self_report_bool(v)
            if not idx or self_report is None:
                continue
            self_report_by_idx[idx] = self_report_by_idx.get(idx, False) or self_report

        plan_ids_by_idx: dict[str, str] = {}
        if plan:
            for s in plan.get("steps", []):
                sid = str(s.get("id") or "")
                idx = _step_index(sid)
                if idx and idx not in plan_ids_by_idx:
                    plan_ids_by_idx[idx] = sid

        preserved: list[dict[str, Any]] = []
        for entry in visual_predicate_status or []:
            if not entry.get("visually_satisfied"):
                continue
            raw_id = str(entry.get("step_id") or "")
            idx = _step_index(raw_id)
            if not idx:
                continue
            if has_self_report and idx in self_report_by_idx and not self_report_by_idx[idx]:
                continue
            snippet = _extract_step_segment(code, raw_id)
            if not snippet:
                continue
            preserved.append({
                "step_id": plan_ids_by_idx.get(idx, raw_id),
                "description": entry.get("description", ""),
                "evidence": entry.get("evidence", ""),
                "code_snippet": snippet,
            })
        return preserved

    def _molmospaces_feedback_media(
        self,
        execution_result: dict[str, Any],
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        images: list[str] = []
        videos: list[str] = []
        manifest: list[dict[str, Any]] = []

        def add(label: str, frame: Any, **meta: Any) -> None:
            url = image_to_data_url(frame)
            if not url:
                return
            images.append(url)
            record = {"media_index": len(manifest) + 1, "type": "image", "label": label}
            record.update({k: v for k, v in meta.items() if v is not None})
            manifest.append(record)

        frames = (
            execution_result.get("trajectory_video_frames")
            or execution_result.get("full_trajectory_frames")
            or execution_result.get("trajectory_frames")
            or []
        )
        if frames:
            video_url, video_meta = self._frames_to_video_data_url(
                list(frames),
                label="trajectory_video",
            )
            if video_url:
                videos.append(video_url)
                manifest.append(
                    {
                        "media_index": len(manifest) + 1,
                        "type": "video",
                        **video_meta,
                    }
                )
            else:
                add("trajectory_start", frames[0], frame_index=1, frame_count=len(frames))
                add("trajectory_end", frames[-1], frame_index=len(frames), frame_count=len(frames))
        else:
            add("before_agentview", execution_result.get("before_frame"))
            add("after_agentview", execution_result.get("after_frame"))
        add("after_wrist", execution_result.get("after_wrist_frame"))
        return images, videos, manifest

    @staticmethod
    def _frames_to_video_data_url(
        frames: list[Any],
        *,
        label: str,
        fps: int = 20,
    ) -> tuple[str | None, dict[str, Any]]:
        if not frames:
            return None, {"label": label, "frame_count": 0, "fps": fps}
        import tempfile

        try:
            import imageio.v2 as imageio
        except Exception:
            try:
                import imageio  # type: ignore
            except Exception:
                return None, {"label": label, "frame_count": len(frames), "fps": fps}
        tmp_name = ""
        try:
            arrs = []
            for frame in frames:
                if frame is None:
                    continue
                arrs.append(frame)
            if not arrs:
                return None, {"label": label, "frame_count": 0, "fps": fps}
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_name = tmp.name
            imageio.mimsave(tmp_name, arrs, fps=fps)
            return video_file_to_data_url(tmp_name), {
                "label": label,
                "frame_count": len(arrs),
                "fps": fps,
            }
        except Exception:
            return None, {"label": label, "frame_count": len(frames), "fps": fps}
        finally:
            if tmp_name:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except Exception:
                    pass

    def _prompt_safe_payload(self, value: Any, key: str = "") -> Any:
        lk = key.lower()
        rawish = any(
            token in lk
            for token in (
                "pointcloud",
                "point_cloud",
                "joints",
                "joint",
                "qpos",
                "pose",
                "pose_mat",
                "intrinsics",
                "extrinsics",
                "depth",
                "centroid",
                "center",
                "world_point",
                "world_points",
                "normalized_vector",
                "raw",
                "privileged",
                "grounded_state",
                "state_trace",
            )
        )
        pathish = lk in {"path", "paths"} or lk.endswith("_path") or lk.endswith("_file") or lk.endswith("_dir")
        if pathish or rawish:
            return None
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for child_key, child in value.items():
                redacted = self._prompt_safe_payload(child, str(child_key))
                if redacted not in (None, {}, []):
                    out[str(child_key)] = redacted
            return out
        if isinstance(value, (list, tuple)):
            if not value:
                return []
            if lk in {"steps", "visual_predicate_status", "prior_attempts", "evidence_keys"}:
                return [
                    redacted
                    for v in value
                    if (redacted := self._prompt_safe_payload(v, key)) not in (None, {}, [])
                ]
            numeric_count = sum(isinstance(x, numbers.Number) and not isinstance(x, bool) for x in value)
            if numeric_count >= 3 and numeric_count >= max(3, len(value) // 2):
                return None
            if len(value) > 16 and any(isinstance(x, (list, tuple, dict)) for x in value):
                return None
            return [
                redacted
                for v in value
                if (redacted := self._prompt_safe_payload(v, key)) not in (None, {}, [])
            ]
        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            if any(token in lk for token in ("joint", "pose", "position", "center", "centroid", "point", "depth", "delta", "dist")):
                return None
        if isinstance(value, str):
            text = re.sub(
                r"(?i)\b(?:raw|path|file|artifact|json|npz)=((?:/|outputs/|rats/outputs/|\S*/outputs/)[^\s,;]+)",
                "",
                value,
            )
            text = re.sub(
                r"(?i)(?:/workspace/[^\s,;]+|/tmp/[^\s,;]+|outputs/[^\s,;]+|rats/outputs/[^\s,;]+)",
                "",
                text,
            )
            text = re.sub(
                r"\[(?:\s*-?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*,){2,}\s*-?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*\]",
                "",
                text,
            )
            text = re.sub(r"\b\w+=($|\s)", " ", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
            return text or None
        return value

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return default
        return out if math.isfinite(out) else default

    def _extract_skills(
        self,
        code: str,
        plan: dict[str, Any] | None,
        execution_result: dict[str, Any],
        *,
        task_language: str = "",
        existing_skills: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract reusable skills from successfully executed code.

        Uses LLM to identify sub-functions that could be added to the skill library.
        """
        if not code.strip():
            return []

        prompt_template = Path("rats/prompts/feedback_generator.txt").read_text()
        task_desc = ""
        goal = ""
        if plan:
            task_desc = plan.get("task_id", "")
        # The actual task natural-language goal. Prefer the passed-in
        # task_language (the task proposer's `language` field), and only
        # fall back to plan.steps[0].description when the caller didn't
        # supply it — the step-1 description is the planner's framing
        # of the first step, not the task goal, so using it here renders
        # the wrong thing in the extraction prompt (observed in the LIBERO
        # smoke run: 0089's GOAL was the porcelain mug step-1 description
        # instead of "put the porcelain mug on the plate").
        if task_language:
            goal = task_language
        elif plan and plan.get("steps"):
            goal = plan["steps"][0].get("description", task_desc)

        # Inline the existing learned skills so the LLM can avoid
        # duplicating them. Primitives are documented elsewhere and don't
        # need to be in this prompt — only learned skills do.
        existing_block = "(none)"
        if existing_skills:
            entries = []
            for s in existing_skills[:30]:
                if s.get("is_primitive"):
                    continue
                name = s.get("name", "")
                desc = (s.get("description", "") or "")[:160]
                if name:
                    entries.append(f"- {name}: {desc}")
            if entries:
                existing_block = "\n".join(entries)

        user_prompt = prompt_template.replace(
            "{task_description}", str(task_desc or "")
        ).replace(
            "{goal_conditions}", str(goal or "")
        ).replace(
            "{existing_skills}", existing_block
        ).replace(
            "{code}", str(code or "")
        ).replace(
            "{reward}", str(execution_result.get("reward", 0.0) or 0.0)
        )

        system_prompt = (
            "You extract reusable robot skills from successful code. "
            "Respond only in valid JSON."
        )

        try:
            kwargs: dict[str, Any] = {}
            if self.model:
                kwargs["model"] = self.model
            result = query_llm_json(system_prompt, user_prompt, **kwargs)
            extracted = result.get("extracted_skills", []) or []
            # Belt-and-braces: enforce the ≤2 cap in case the LLM ignores
            # the prompt instruction.
            extracted = extracted[:2]
            # AST gate. Apply the same perception-verify anti-pattern check
            # that ``policy_quality_checker`` uses on policy code: reject
            # any candidate skill whose body binds a vlm_verify /
            # verify_object_identity result to a name and then never reads
            # ``name["verified"]`` in the same scope. Audit on the existing
            # learned-skill library found 3/10 skills baked in that
            # anti-pattern — they were extracted before the gate landed on
            # the policy_writer side, so future "skill reuse" was
            # propagating the bug into new policy code. Gating at
            # extraction time prevents new bad skills from entering the
            # library.
            from rats.agents.policy_quality_checker import PolicyQualityChecker
            filtered: list[dict[str, Any]] = []
            for sk in extracted:
                if not isinstance(sk, dict):
                    continue
                code_str = str(sk.get("code") or "")
                if code_str.strip():
                    issues = PolicyQualityChecker._check_perception_verify_consumed(
                        code_str
                    )
                    if issues:
                        logger.warning(
                            "Skill extraction: rejecting candidate '%s' for "
                            "perception-verify anti-pattern: %s",
                            sk.get("name", "<unnamed>"),
                            issues[0][:200],
                        )
                        continue
                filtered.append(sk)
            extracted = filtered
            # Stamp provenance fields onto each extracted skill so the
            # library + planner can later answer "where did this come
            # from, and what's a concrete invocation that worked?".
            # The LLM is asked for `extraction_rationale` and
            # `usage_example`; `source_task` we know directly (the
            # task whose success triggered this extraction).
            for sk in extracted:
                if not isinstance(sk, dict):
                    continue
                if task_language and not sk.get("source_task"):
                    sk["source_task"] = task_language
                # Best-effort: leave keys present even if empty so the
                # library record shape is stable.
                sk.setdefault("extraction_rationale", "")
                sk.setdefault("usage_example", "")
                # Structured typing + shape — see prompts/feedback_generator.txt.
                # Older runs that pre-date this field still parse, since the
                # downstream skill_library.add_skill setdefault's both.
                sk.setdefault("params", [])
                sk.setdefault("returns", {"type": "", "shape": "", "description": ""})
            return extracted
        except Exception:
            return []
