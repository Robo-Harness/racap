"""Planner verifier: gate the planner's visual + structural correctness.

Runs after ``Planner.plan`` returns, before the policy writer touches the
plan. The same agentview image the planner saw is shown to a VLM along
with the task statement, object scope, and the plan it produced. The
VLM looks for four classes of plan-level mistake:

- ``visual_misperception``: the plan asserts a scene state the image
  contradicts (door already open, object not present, wrong colour).
- ``missing_prerequisite``: a dependent step has no preceding enabling
  step (place-into-drawer without opening it first).
- ``wrong_ordering``: steps are present but in an impossible order
  (lift before grasp, place before pick).
- ``object_scope_mismatch``: plan references an object outside
  ``object_scope`` or not visible, or a skill not in the available
  list.

The verdict is consumed by ``lifelong_loop`` as a one-shot gate: if
``verdict == "fail"`` with non-trivial confidence, the loop calls
``Planner.refine_plan`` once with the verifier's ``summary_for_refine_plan``
fed in as ``plan_issue_reason``. No verify-refine loop — one round,
then accept whatever the refinement produces.

This module mirrors the structure of ``per_step_verifier.py``
(config dataclass + class with ``enabled`` switch + JSON-only LLM call)
but is intentionally simpler: no event-trace assignment, no privileged
state, just (image, plan) → verdict.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rats.agents.base_agent import image_to_data_url, query_llm_json

logger = logging.getLogger("rats.planner_verifier")


@dataclass(frozen=True)
class PlannerVerifierConfig:
    enabled: bool = True
    save_artifacts: bool = True
    model: str = "google/gemini-3.1-pro-preview"
    max_tokens: int = 8192
    # Confidence threshold for triggering a refine_plan. Below this we
    # log the verdict but do not gate. Calibrated so an "ambiguous"
    # verdict (typically confidence < 0.6) does NOT trigger a refine.
    min_fail_confidence: float = 0.6


class PlannerVerifier:
    """One-shot pre-execution gate over (initial agentview, plan)."""

    schema_version = "rats_planner_verification_v1"

    def __init__(
        self,
        *,
        enabled: bool = True,
        save_artifacts: bool = True,
        model: str | None = "google/gemini-3.1-pro-preview",
        max_tokens: int = 8192,
        min_fail_confidence: float = 0.6,
    ) -> None:
        self.config = PlannerVerifierConfig(
            enabled=bool(enabled),
            save_artifacts=bool(save_artifacts),
            model=str(model or "google/gemini-3.1-pro-preview"),
            max_tokens=int(max_tokens or 8192),
            min_fail_confidence=float(min_fail_confidence),
        )

    def verify_plan(
        self,
        *,
        task_proposal: dict[str, Any],
        plan: dict[str, Any],
        scene_context: dict[str, Any],
        all_skills: list[dict[str, Any]] | None = None,
        initial_rgb: Any = None,
        output_dir: str | Path | None = None,
        iteration: int | None = None,
        phase: str = "initial",
    ) -> dict[str, Any]:
        """Run the verifier. Returns a verdict dict, never raises.

        Args:
            task_proposal: TaskProposer output (activity_name, goal_conditions, ...).
            plan: Planner.plan output (steps, all_selected_skill_names, ...).
            scene_context: object_scope, available_functions, api_docs.
            all_skills: Optional full library passed to the planner — used
                to extend the available-skills block with learned skill
                names so the verifier doesn't false-flag the planner's
                legit references to learned skills as object_scope_mismatch.
                If None, only scene_context.available_functions is shown.
            initial_rgb: The same agentview RGB the planner saw. If None,
                the verdict is forced to "ambiguous" — there is no image
                to verify against.
            output_dir: Optional directory for the verdict JSON artifact.
            iteration: Iteration number, for artifact filename.
            phase: "initial" (first pass) or "refined" (after refine_plan).

        Returns:
            Dict with keys:
              - enabled (bool)
              - schema_version (str)
              - verdict ("pass" | "ambiguous" | "fail")
              - confidence (0..1)
              - issues (list of issue dicts)
              - summary_for_refine_plan (str)
              - should_refine (bool) — convenience, True iff verdict=="fail"
                AND confidence >= min_fail_confidence
              - error (str, optional) — set on parse / API errors
        """
        if not self.config.enabled:
            return {
                "enabled": False,
                "schema_version": self.schema_version,
                "verdict": "pass",
                "confidence": 0.0,
                "issues": [],
                "summary_for_refine_plan": "",
                "should_refine": False,
            }

        if initial_rgb is None:
            verdict = {
                "enabled": True,
                "schema_version": self.schema_version,
                "verdict": "ambiguous",
                "confidence": 0.0,
                "issues": [],
                "summary_for_refine_plan": "",
                "should_refine": False,
                "error": "no_initial_rgb",
                "phase": phase,
                "model": self.config.model,
            }
            self._maybe_save_artifact(verdict, output_dir, iteration, phase)
            return verdict

        user_prompt = self._build_user_prompt(
            task_proposal, plan, scene_context, all_skills=all_skills,
        )
        # image_to_data_url can raise on unusual shapes / dtypes / IO
        # errors as well as return falsy. Treat both as the same
        # degraded path — don't silently fall through to a text-only
        # LLM call (the verifier would hallucinate a verdict against
        # no image at all).
        image_url: str | None = None
        try:
            image_url = image_to_data_url(initial_rgb)
        except Exception as exc:
            logger.debug(f"PlannerVerifier image encode failed: {exc}")
            image_url = None
        if not image_url:
            verdict = {
                "enabled": True,
                "schema_version": self.schema_version,
                "verdict": "ambiguous",
                "confidence": 0.0,
                "issues": [],
                "summary_for_refine_plan": "",
                "should_refine": False,
                "error": "image_encoding_failed",
                "phase": phase,
                "model": self.config.model,
            }
            self._maybe_save_artifact(verdict, output_dir, iteration, phase)
            return verdict
        images = [image_url]

        system_prompt = (
            "You are a strict visual + structural verifier for a robot "
            "task plan. Inspect the provided agentview image and the "
            "plan. Report visual_misperception, missing_prerequisite, "
            "wrong_ordering, and object_scope_mismatch findings. "
            "Respond only with the JSON schema requested."
        )

        try:
            raw = query_llm_json(
                system_prompt,
                user_prompt,
                images=images,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            logger.warning(f"PlannerVerifier LLM call failed: {exc}")
            verdict = {
                "enabled": True,
                "schema_version": self.schema_version,
                "verdict": "ambiguous",
                "confidence": 0.0,
                "issues": [],
                "summary_for_refine_plan": "",
                "should_refine": False,
                "error": f"llm_error: {type(exc).__name__}: {exc}",
                "phase": phase,
                "model": self.config.model,
            }
            self._maybe_save_artifact(verdict, output_dir, iteration, phase)
            return verdict

        verdict = self._normalize_verdict(raw)
        verdict["phase"] = phase
        verdict["model"] = self.config.model
        self._maybe_save_artifact(verdict, output_dir, iteration, phase)
        return verdict

    def _build_user_prompt(
        self,
        task_proposal: dict[str, Any],
        plan: dict[str, Any],
        scene_context: dict[str, Any],
        *,
        all_skills: list[dict[str, Any]] | None = None,
    ) -> str:
        template = Path("rats/prompts/planner_verifier.txt").read_text()

        task_description = (
            task_proposal.get("goal_conditions")
            or task_proposal.get("activity_name")
            or ""
        )
        goal_conditions = task_proposal.get("goal_conditions", "")
        object_scope = scene_context.get("object_scope", {}) or {}

        # Skill names only — keep the prompt short. Detailed signatures
        # live in the planner's prompt; the verifier just needs to spot
        # references to names that don't exist. We merge:
        #   - primitives from scene_context.available_functions
        #   - learned-skill names from all_skills (if provided), so the
        #     verifier doesn't false-flag the planner's legitimate
        #     references to learned skills as object_scope_mismatch
        available_skills: set[str] = set()
        for fn in scene_context.get("available_functions", []) or []:
            if fn:
                available_skills.add(str(fn))
        if all_skills:
            for skill in all_skills:
                name = skill.get("name") if isinstance(skill, dict) else None
                if name:
                    available_skills.add(str(name))
        skills_block = ", ".join(sorted(available_skills)) or "(none)"

        plan_steps_block = self._format_plan_steps(plan.get("steps") or [])

        return (
            template
            .replace("{task_description}", str(task_description))
            .replace("{goal_conditions}", str(goal_conditions))
            .replace("{object_scope}", json.dumps(object_scope, ensure_ascii=False))
            .replace("{available_skills}", skills_block)
            .replace("{plan_steps_block}", plan_steps_block)
        )

    @staticmethod
    def _format_plan_steps(steps: list[dict[str, Any]]) -> str:
        if not steps:
            return "(plan has zero steps)"
        lines: list[str] = []
        for step in steps:
            sid = step.get("id") or step.get("step_id") or "step-?"
            desc = step.get("description") or step.get("goal") or ""
            relevant = step.get("relevant_skills") or []
            new_skill = bool(step.get("new_skill_needed", False))
            notes = step.get("notes") or ""
            lines.append(
                f"- {sid}: {desc}\n"
                f"    relevant_skills: {list(relevant)}\n"
                f"    new_skill_needed: {new_skill}\n"
                f"    notes: {notes}"
            )
        return "\n".join(lines)

    def _normalize_verdict(self, raw: Any) -> dict[str, Any]:
        """Coerce the LLM JSON into the documented schema with defaults."""
        verdict_value = "ambiguous"
        confidence = 0.0
        scene_summary = ""
        issues_out: list[dict[str, Any]] = []
        summary_for_refine = ""

        if isinstance(raw, dict):
            v = str(raw.get("verdict") or "").strip().lower()
            if v in ("pass", "fail", "ambiguous"):
                verdict_value = v
            try:
                confidence = float(raw.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            scene_summary = str(raw.get("scene_summary") or "")
            for issue in raw.get("issues") or []:
                if not isinstance(issue, dict):
                    continue
                kind = str(issue.get("kind") or "").strip().lower()
                if kind not in (
                    "visual_misperception",
                    "missing_prerequisite",
                    "wrong_ordering",
                    "object_scope_mismatch",
                ):
                    # Tolerate unknown kinds by keeping them but tagging.
                    kind = kind or "other"
                step_id = issue.get("step_id")
                if step_id in (None, "", "null"):
                    step_id = None
                else:
                    step_id = str(step_id)
                issues_out.append({
                    "kind": kind,
                    "step_id": step_id,
                    "evidence": str(issue.get("evidence") or ""),
                    "suggested_fix": str(issue.get("suggested_fix") or ""),
                })
            summary_for_refine = str(raw.get("summary_for_refine_plan") or "")

        # Fallback: if the LLM gave us issues but no summary, synthesize
        # one from the issues so refine_plan gets actionable text instead
        # of an empty plan_issue_reason (the planner would otherwise come
        # back with the same plan).
        if not summary_for_refine and issues_out:
            parts = []
            for issue in issues_out:
                kind = issue.get("kind") or "issue"
                step = issue.get("step_id") or "across plan"
                fix = issue.get("suggested_fix") or issue.get("evidence") or ""
                parts.append(f"[{kind} @ {step}] {fix}")
            summary_for_refine = " ".join(parts)

        # Require a non-empty reason string. The synthesis above
        # guarantees this whenever issues_out is non-empty, so the
        # explicit `or issues_out` check from earlier drafts was dead.
        should_refine = (
            verdict_value == "fail"
            and confidence >= self.config.min_fail_confidence
            and bool(summary_for_refine)
        )
        return {
            "enabled": True,
            "schema_version": self.schema_version,
            "verdict": verdict_value,
            "confidence": confidence,
            "scene_summary": scene_summary,
            "issues": issues_out,
            "summary_for_refine_plan": summary_for_refine,
            "should_refine": should_refine,
        }

    def _maybe_save_artifact(
        self,
        verdict: dict[str, Any],
        output_dir: str | Path | None,
        iteration: int | None,
        phase: str,
    ) -> None:
        if not (self.config.save_artifacts and output_dir):
            return
        try:
            base = Path(output_dir)
            base.mkdir(parents=True, exist_ok=True)
            iter_tag = f"iter{int(iteration):03d}" if iteration is not None else "iter_unknown"
            fname = f"{iter_tag}_planner_verifier_{phase}.json"
            (base / fname).write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"PlannerVerifier artifact save failed: {e}")
