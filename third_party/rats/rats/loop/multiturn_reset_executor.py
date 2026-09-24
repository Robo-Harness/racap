"""Multiturn-reset step-by-step executor (LIBERO-only, opt-in).

Default RATS flow writes the entire policy in one shot per attempt: writer
emits N-step code, executor runs the whole thing, then a single failure
diagnoser tries to figure out which step broke. Per-step verifier reports
correct localization 86%+ of the time but its verdict only reaches policy
writer through diagnoser, so retries discard already-working sub-behavior
("regression on retry" pattern in libero_main_30iter).

Multiturn mode flips the loop: run each plan step in isolation, ask
per_step_verifier whether THAT step succeeded, and only advance once the
step is committed. On step failure, reset the env, replay the committed
prior steps, and ask the writer to rewrite ONLY the current step (with
argument_level / rewrite_needed framing from the verifier verdict). If a
step burns through its retry budget without progress, signal a plan-level
escalation so the outer attempt loop can re-plan.

The orchestrator returns an `execution_result`-shaped dict so downstream
agents (final verifier, feedback_generator, skill_proposer) don't need to
know multiturn-reset mode happened.

Default off; only wired for LIBERO (deterministic reset). BEHAVIOR's
non-resettable physical env would break the replay model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rats.agents.base_agent import extract_python_code

logger = logging.getLogger("rats.multiturn_reset_executor")


@dataclass
class MultiturnResetConfig:
    enabled: bool = False
    max_step_retries: int = 10
    # When a single step has burned through this many consecutive retries
    # whose per_step verdicts share the same reason snippet, declare
    # stagnation early instead of running out the full budget.
    stagnation_window: int = 3


@dataclass
class StepRetryRecord:
    retry: int
    edit_scale: str
    step_code: str
    ps_status: str | None
    ps_confidence: float | None
    ps_reason: str
    exec_success: bool
    exec_stderr_tail: str
    feedback_summary: str = ""


@dataclass
class MultiturnResetRunResult:
    status: str  # "success" | "partial_success" | "step_stagnation" | "preamble_failure" | "skipped"
    final_code: str
    execution_result: dict[str, Any]
    per_step_verifications: list[dict[str, Any]] = field(default_factory=list)
    step_retry_log: dict[int, list[StepRetryRecord]] = field(default_factory=dict)
    stuck_step_idx: int | None = None
    stuck_step_id: str | None = None
    stuck_reason: str = ""
    committed_step_codes: dict[int, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    total_step_retries: int = 0
    notes: list[str] = field(default_factory=list)
    # Steps that exhausted their PS retry budget without a "succeeded"
    # verdict and were force-committed so the rest of the plan could run.
    # These codes are UNVERIFIED — downstream prompts/extractors MUST mark
    # them as such (no caching as successful_code, no skill extraction).
    force_committed_indices: list[int] = field(default_factory=list)
    # Steps for which the writer never produced usable code (skipped
    # entirely; even less trusted than force-committed).
    skipped_indices: list[int] = field(default_factory=list)
    # Per-retry quality-checker results across the whole run. Each entry:
    # {step_idx, retry, approved, tier1_violations, tier2_issues}.
    step_quality_results: list[dict[str, Any]] = field(default_factory=list)


class MultiturnResetExecutor:
    """Per-step executor with env-reset rollback (LIBERO-only).

    Hands every step through ``per_step_verifier`` for an independent
    "did this step succeed" verdict, commits passing step code as fixed
    prior context, and retries failing steps with the verifier's verdict
    as feedback.
    """

    def __init__(
        self,
        *,
        executor: Any,
        policy_writer: Any,
        per_step_verifier: Any,
        env_resetter: Callable[[], None],
        config: MultiturnResetConfig | None = None,
        # Optional injected helpers so we can reuse LifelongLoop's
        # existing exec_history + step_frame_segments machinery without
        # duplicating it. When provided, MultiturnResetExecutor wraps
        # each executor.execute with the same video-capture + api-logging
        # context the legacy path uses, then asks the builder to slice
        # the buffer into per-step segments that per_step_verifier
        # consumes. Without them per_step_verifier short-circuits to
        # "no step media available" on every retry.
        frame_segment_builder: Callable[..., list] | None = None,
        api_logging_enable_fn: Callable[[Any], list] | None = None,
        api_logging_restore_fn: Callable[[list], None] | None = None,
        # PolicyQualityChecker (Tier-1 AST/API gate + Tier-2 advisory).
        # When provided, runs per-step on the writer's output BEFORE the
        # exec call. Tier-1 violations consume a retry with the violation
        # text surfaced as next-retry feedback; Tier-2 issues are folded
        # into the next-retry advisory. Optional so unit tests + legacy
        # callers without a checker still work.
        quality_checker: Any | None = None,
    ) -> None:
        self.executor = executor
        self.policy_writer = policy_writer
        self.per_step_verifier = per_step_verifier
        self.env_resetter = env_resetter
        self.config = config or MultiturnResetConfig()
        self.frame_segment_builder = frame_segment_builder
        self.api_logging_enable_fn = api_logging_enable_fn
        self.api_logging_restore_fn = api_logging_restore_fn
        self.quality_checker = quality_checker

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def run(
        self,
        *,
        env: Any,
        plan: dict[str, Any],
        scene_context: dict[str, Any],
        skill_preamble: str = "",
        failure_context: str = "",
        success_context: str = "",
        iteration: int = 0,
        attempt_in_iter: int = 0,
        output_dir: Path | None = None,
        learned_skill_names: list[str] | None = None,
    ) -> MultiturnResetRunResult:
        start = time.time()
        steps = list(plan.get("steps") or [])
        if not steps:
            return MultiturnResetRunResult(
                status="skipped",
                final_code="",
                execution_result={"success": False, "stderr": "multiturn: empty plan"},
                notes=["empty plan"],
            )

        committed: dict[int, str] = {}
        retry_log: dict[int, list[StepRetryRecord]] = {}
        ps_verdicts: list[dict[str, Any]] = []
        final_exec: dict[str, Any] = {}
        final_code = ""
        total_retries = 0
        # When a step exhausts its retry budget without a PS-succeeded
        # verdict, we force-commit its last attempt and continue to the
        # next step so the rest of the plan can at least execute. Track
        # which step_idx's were force-committed (lost real verification)
        # vs naturally committed (PS approved) so the run result can be
        # classified as partial_success and the outer loop knows to plan-
        # rewrite for the next attempt.
        force_committed: list[int] = []
        skipped_steps: list[int] = []
        # Per-retry PolicyQualityChecker results. Tier-1 violations block
        # exec (counted as a retry); Tier-2 issues are advisory and don't
        # block. Surfaced in MultiturnResetRunResult.step_quality_results
        # so the lifelong-loop artifact write can replace the legacy
        # always-approved stub with real evidence.
        step_quality_results: list[dict[str, Any]] = []
        available_functions = (
            scene_context.get("available_functions") if scene_context else None
        )
        goal_text = (
            scene_context.get("goal_conditions_nl") if scene_context else ""
        ) or ""
        learned_skill_names_for_qc = list(learned_skill_names or [])

        for step_idx, step in enumerate(steps):
            step_id = self._step_id(step, step_idx)
            logger.info(
                "Multiturn-reset step %d/%d (%s): %s",
                step_idx + 1, len(steps), step_id,
                (step.get("description") or step.get("goal") or "")[:120],
            )

            step_retries: list[StepRetryRecord] = []
            last_feedback: dict[str, Any] | None = None
            step_succeeded = False
            stuck_reason = ""

            for retry in range(self.config.max_step_retries):
                edit_scale = self._next_edit_scale(last_feedback, retry, step_retries)

                # 1. Reset env so the replay starts from a deterministic init
                try:
                    self.env_resetter()
                except Exception as exc:
                    logger.exception(
                        "Multiturn-reset: env reset failed at step %d retry %d: %s",
                        step_idx, retry, exc,
                    )
                    stuck_reason = f"env_reset_exception: {exc}"
                    break

                # 2. Ask writer for THIS step's code only, given prior committed
                #    steps + the failure framing from the previous retry.
                # `force_committed_step_indices` flags which prior committed
                # codes are UNVERIFIED (PS never approved them) so the writer
                # prompt can render them as "[forced]" instead of "[done]"
                # and warn the writer not to assume those steps left the env
                # in the expected state.
                try:
                    step_code = self.policy_writer.write_step(
                        plan=plan,
                        scene_context=scene_context,
                        step_idx=step_idx,
                        step=step,
                        committed_step_codes=committed,
                        step_retry_feedback=last_feedback,
                        edit_scale=edit_scale,
                        failure_context=failure_context,
                        success_context=success_context,
                        learned_skill_names=learned_skill_names or [],
                        force_committed_step_indices=set(force_committed),
                    )
                except Exception as exc:
                    logger.exception(
                        "Multiturn-reset: policy_writer.write_step crashed at step %d retry %d: %s",
                        step_idx, retry, exc,
                    )
                    step_retries.append(StepRetryRecord(
                        retry=retry, edit_scale=edit_scale, step_code="",
                        ps_status=None, ps_confidence=None,
                        ps_reason=f"writer_exception: {exc}",
                        exec_success=False, exec_stderr_tail=str(exc)[-500:],
                    ))
                    total_retries += 1
                    continue

                step_code = self._normalize_writer_output(step_code or "")
                if not step_code.strip():
                    step_retries.append(StepRetryRecord(
                        retry=retry, edit_scale=edit_scale, step_code="",
                        ps_status=None, ps_confidence=None,
                        ps_reason="writer_returned_empty",
                        exec_success=False, exec_stderr_tail="",
                    ))
                    total_retries += 1
                    continue

                # 2b. PolicyQualityChecker gate (Tier-1 blocks; Tier-2 advisory).
                # Run on the JUST-WRITTEN step code (not the composed full
                # code) — Tier-1 violations like banned imports, dangerous
                # primitives, or syntactically wrong API use should be
                # caught here before we burn an env reset + exec on them.
                # Tier-2 issues (semantic / advisory) are recorded but
                # don't block.
                if self.quality_checker is not None:
                    try:
                        qc = self.quality_checker.check(
                            step_code,
                            available_functions=available_functions,
                            goal=goal_text,
                            learned_skill_names=learned_skill_names_for_qc,
                            # Per-step body: skip full-policy-only rules
                            # (RESULT assignment + pick-and-place progression).
                            # The orchestrator's _compose_full_code adds
                            # RESULT after the body, and a pick/grasp in
                            # step k legitimately has its place/release in
                            # step k+1's body. Other Tier-1 rules still
                            # fire (banned primitives, env.<method>,
                            # while True, etc.).
                            full_policy_check=False,
                        )
                    except Exception as exc:
                        logger.exception(
                            "Multiturn-reset: quality_checker.check crashed at "
                            "step %d retry %d: %s",
                            step_idx, retry, exc,
                        )
                        qc = {
                            "approved": True,
                            "feedback": f"quality_checker_exception: {exc}",
                            "tier1_violations": [],
                            "tier2_issues": [],
                            "_checker_error": True,
                        }
                    # PolicyQualityChecker returns `tier1_issues` (not
                    # `tier1_violations`); accept either field name for
                    # robustness against future renames.
                    qc_tier1 = (
                        qc.get("tier1_issues")
                        or qc.get("tier1_violations")
                        or []
                    )
                    qc_record = {
                        "step_idx": step_idx,
                        "retry": retry,
                        "approved": bool(qc.get("approved", True)),
                        "tier1_violations": list(qc_tier1),
                        "tier2_issues": list(qc.get("tier2_issues", []) or []),
                        "feedback": str(qc.get("feedback") or "")[:1000],
                    }
                    step_quality_results.append(qc_record)
                    if not qc_record["approved"]:
                        # Tier-1 violation: consume a retry, surface as
                        # next-retry feedback for the writer. Don't compose
                        # / execute this candidate.
                        tier1_text = (
                            "; ".join(qc_record["tier1_violations"])
                            or qc_record["feedback"]
                            or "Tier-1 quality violation"
                        )
                        directive = (
                            "PolicyQualityChecker rejected the step (Tier-1): "
                            f"{tier1_text[:800]}. Rewrite this step without that "
                            "violation; keep the same step intent."
                        )
                        step_retries.append(StepRetryRecord(
                            retry=retry, edit_scale="rewrite_needed",
                            step_code=step_code,
                            ps_status="rejected_by_quality_checker",
                            ps_confidence=0.0,
                            ps_reason=tier1_text[:500],
                            exec_success=False,
                            exec_stderr_tail="(skipped: blocked by quality_checker)",
                            feedback_summary=directive[:500],
                        ))
                        last_feedback = {
                            "edit_scale": "rewrite_needed",
                            "policy_feedback": directive,
                            "ps_status": "rejected_by_quality_checker",
                            "ps_reason": tier1_text[:500],
                            "source": "policy_quality_checker_tier1",
                            "edit_scale_source": "policy_quality_checker_tier1",
                            "feedback_source": "policy_quality_checker_tier1",
                        }
                        logger.warning(
                            "  MT-RESET step %d retry %d/%d BLOCKED by quality_checker "
                            "(Tier-1): %s",
                            step_idx + 1, retry + 1, self.config.max_step_retries,
                            tier1_text[:200],
                        )
                        total_retries += 1
                        continue
                    if qc_record["tier2_issues"]:
                        logger.info(
                            "  MT-RESET step %d retry %d quality advisory "
                            "(Tier-2, non-blocking): %s",
                            step_idx + 1, retry + 1,
                            "; ".join(qc_record["tier2_issues"])[:200],
                        )

                # 3. Compose full executable (skill_preamble + committed prior
                #    steps + current candidate step), then execute.
                full_code = self._compose_full_code(
                    steps=steps, committed=committed, step_idx=step_idx,
                    step_code=step_code, skill_preamble=skill_preamble,
                )
                final_code = full_code
                exec_result = self._execute(
                    full_code, env, scene_context,
                    step_idx=step_idx, retry=retry, step=step,
                )
                final_exec = exec_result

                # 4. Per-step verify just THIS step
                ps_out = self._verify_single_step(
                    exec_result=exec_result, plan=plan, step=step,
                    full_code=full_code, output_dir=output_dir,
                    iteration=iteration, attempt_in_iter=attempt_in_iter,
                    step_idx=step_idx, retry=retry,
                )
                ps_step = self._extract_single_step_verdict(ps_out)

                exec_success = bool(exec_result.get("success"))
                exec_stderr = (exec_result.get("stderr") or "").strip()
                ps_status = str(ps_step.get("status") or "").lower()
                ps_conf = self._safe_float(ps_step.get("confidence"))
                ps_reason = str(ps_step.get("reason") or "")

                record = StepRetryRecord(
                    retry=retry, edit_scale=edit_scale, step_code=step_code,
                    ps_status=ps_status, ps_confidence=ps_conf,
                    ps_reason=ps_reason,
                    exec_success=exec_success,
                    exec_stderr_tail=exec_stderr[-500:],
                )
                step_retries.append(record)
                total_retries += 1

                # Per-retry trace line so iter logs / Monitor filters can
                # follow exactly which step+retry+verdict each LLM call
                # produced. Format is grep-friendly:
                #   MT-RESET step S retry R/N edit=X exec_ok=Y ps=Z conf=C reason='...'
                logger.info(
                    "  MT-RESET step %d retry %d/%d edit=%s exec_ok=%s "
                    "ps=%s conf=%.2f reason=%r",
                    step_idx + 1, retry + 1, self.config.max_step_retries,
                    edit_scale, exec_success,
                    ps_status or "?", ps_conf, ps_reason[:200],
                )

                # Commit decision: PS verdict is the source of truth (the
                # user's spec for multiturn-reset). "succeeded" → commit
                # regardless of exec_ok. Common case where exec_ok=False
                # but ps=succeeded: a perception-only step where the
                # writer's own assert raised AFTER the goal was already
                # observably met — the env state matches the goal, so
                # advancing is safe and re-running the buggy code 10
                # times never produces new visual evidence.
                # Ambiguous + failed both go to the retry path.
                if ps_status == "succeeded":
                    committed[step_idx] = step_code
                    ps_verdicts.append(ps_step)
                    record.feedback_summary = "committed"
                    step_succeeded = True
                    log_tail = "" if exec_success else " [exec stderr ignored — PS approves visual goal]"
                    logger.info(
                        "  MT-RESET step %d COMMITTED after %d retries (ps_conf=%.2f)%s",
                        step_idx + 1, retry + 1, ps_conf, log_tail,
                    )
                    break

                # Step failed — formulate next-retry framing from PS verdict
                last_feedback = self._build_step_feedback(
                    step=step, step_code=step_code, ps_step=ps_step,
                    exec_result=exec_result, prior_retries=step_retries,
                )
                record.feedback_summary = str(last_feedback.get("policy_feedback") or "")[:200]
                # Log the directive that will drive the NEXT retry so the
                # log shows the agent's plan, not just the verdict.
                if last_feedback.get("policy_feedback"):
                    logger.info(
                        "  MT-RESET step %d retry %d/%d → next directive (edit_scale=%s, "
                        "source=%s): %s",
                        step_idx + 1, retry + 1, self.config.max_step_retries,
                        last_feedback.get("edit_scale"),
                        last_feedback.get("source"),
                        str(last_feedback["policy_feedback"])[:200],
                    )

            retry_log[step_idx] = step_retries

            if not step_succeeded:
                # All `max_step_retries` retries failed. Per spec: do NOT abort
                # the whole attempt. Force-commit the last retry's code (so
                # the env state progresses to allow steps k+1..N to at least
                # be attempted). Track which steps were force-committed so
                # the outer attempt loop can escalate to plan_rewrite even
                # when execution reached the end.
                #
                # SAFETY: never force-commit a Tier-1-rejected retry's code
                # (that's exactly what PolicyQualityChecker blocked). If
                # ALL retries were QC-rejected, fall through to the skip
                # branch — running unsafe code as a prior-step prefix
                # for subsequent steps would silently defeat the gate.
                safe_records = [
                    r for r in step_retries
                    if r.ps_status != "rejected_by_quality_checker"
                ]
                qc_blocked_count = len(step_retries) - len(safe_records)
                last_record = safe_records[-1] if safe_records else None
                last_code = (last_record.step_code if last_record else "") or ""
                if last_code.strip():
                    committed[step_idx] = last_code
                    force_committed.append(step_idx)
                    qc_note = (
                        f" ({qc_blocked_count} retries blocked by quality_checker, "
                        f"using last non-blocked retry's code)"
                        if qc_blocked_count else ""
                    )
                    logger.warning(
                        "  MT-RESET step %d FORCE-COMMITTED after %d retries "
                        "exhausted (no PS-succeeded verdict)%s. Advancing to "
                        "step %d so the remaining plan can still execute; "
                        "the outer attempt loop will request plan rewrite.",
                        step_idx + 1, self.config.max_step_retries, qc_note, step_idx + 2,
                    )
                else:
                    # Either the writer kept returning empty code, or every
                    # retry was Tier-1-rejected. Skip this step entirely;
                    # downstream steps may fail (missing prerequisite) but
                    # we never replay unsafe code as a committed prefix.
                    skipped_steps.append(step_idx)
                    skip_reason = (
                        f"all {qc_blocked_count} retries blocked by quality_checker"
                        if qc_blocked_count and qc_blocked_count == len(step_retries)
                        else "no usable code produced"
                    )
                    logger.warning(
                        "  MT-RESET step %d SKIPPED after %d retries "
                        "(%s). Advancing to step %d.",
                        step_idx + 1, self.config.max_step_retries,
                        skip_reason, step_idx + 2,
                    )

        # All steps have been processed (some natural-succeeded, some force-
        # committed, some skipped). Classify the run:
        # - success: every step naturally PS-succeeded
        # - partial_success: at least one step was force-committed or skipped;
        #   final exec reached the end of the plan but some step never had
        #   a PS-succeeded verdict. Still escalate to plan_rewrite via the
        #   post-diagnoser hook (same path as step_stagnation).
        if force_committed or skipped_steps:
            status = "partial_success"
            stuck_step_idx = (force_committed + skipped_steps)[0]
            stuck_step_id = self._step_id(steps[stuck_step_idx], stuck_step_idx)
            stuck_reason = (
                f"max_step_retries_exhausted at step(s) "
                f"{sorted(force_committed + skipped_steps)} (force_committed="
                f"{force_committed}, skipped={skipped_steps})"
            )
        else:
            status = "success"
            stuck_step_idx = None
            stuck_step_id = None
            stuck_reason = ""

        return MultiturnResetRunResult(
            status=status,
            final_code=final_code,
            execution_result=final_exec,
            per_step_verifications=ps_verdicts,
            step_retry_log=retry_log,
            stuck_step_idx=stuck_step_idx,
            stuck_step_id=stuck_step_id,
            stuck_reason=stuck_reason,
            committed_step_codes=committed,
            elapsed_seconds=time.time() - start,
            total_step_retries=total_retries,
            force_committed_indices=list(force_committed),
            skipped_indices=list(skipped_steps),
            step_quality_results=step_quality_results,
        )

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------
    @staticmethod
    def _step_id(step: dict[str, Any], idx: int) -> str:
        return str(step.get("id") or step.get("step_id") or f"step-{idx + 1}")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    def _next_edit_scale(
        self,
        last_feedback: dict[str, Any] | None,
        retry: int,
        history: list[StepRetryRecord],
    ) -> str:
        """Pick edit_scale for the upcoming retry.

        Normal path: per_step_verifier emits edit_scale alongside its verdict,
        ``_build_step_feedback`` puts it on ``last_feedback["edit_scale"]``,
        and we just honor it. The "fresh"/"argument_level" defaults only
        fire on the very first attempt (no PS verdict yet) or when PS omits
        the field (older response shape) — purely defensive.
        """
        if last_feedback and last_feedback.get("edit_scale"):
            return str(last_feedback["edit_scale"])
        return "fresh" if (retry == 0 and not history) else "argument_level"

    def _is_stagnating(self, history: list[StepRetryRecord]) -> bool:
        window = self.config.stagnation_window
        if len(history) < window:
            return False
        recent = history[-window:]
        first = recent[0]
        for r in recent[1:]:
            # Stagnating = same ps_status AND short-prefix of ps_reason matches
            if r.ps_status != first.ps_status:
                return False
            if r.ps_reason[:120].strip() != first.ps_reason[:120].strip():
                return False
        return True

    def _build_step_feedback(
        self,
        *,
        step: dict[str, Any],
        step_code: str,
        ps_step: dict[str, Any],
        exec_result: dict[str, Any],
        prior_retries: list[StepRetryRecord],
    ) -> dict[str, Any]:
        """Convert a PS-failed verdict into next-retry framing.

        The per-step verifier itself emits ``edit_scale`` and
        ``corrective_action`` alongside the success/fail verdict (the same
        VLM call that judged this step also produces the retry directive).
        Multiturn-reset just hands those fields to the writer — no second
        LLM call required. When PS didn't populate them (legacy / older VLM
        response missing the new keys), fall back to PS ``reason`` text
        with a safe ``argument_level`` default; the writer prompt's
        edit_scale block already tolerates either tag.
        """
        ps_status = str(ps_step.get("status") or "").lower()
        ps_reason = str(ps_step.get("reason") or "")
        ps_edit_scale = str(ps_step.get("edit_scale") or "").strip().lower()
        ps_corrective = str(ps_step.get("corrective_action") or "").strip()
        unsatisfied = ps_step.get("unsatisfied_conditions") or []
        stderr_tail = (exec_result.get("stderr") or "")[-500:]

        # edit_scale comes straight from PS. If PS punted, default to
        # argument_level — writer can still consume an arg-tune directive
        # safely; rewrite_needed is reserved for cases PS explicitly flagged
        # as structurally wrong.
        if ps_edit_scale in ("argument_level", "rewrite_needed"):
            edit_scale = ps_edit_scale
            edit_scale_source = "per_step_verifier"
        else:
            edit_scale = "argument_level"
            edit_scale_source = "per_step_verifier_default"

        # policy_feedback prefers PS's explicit corrective_action. If that's
        # empty (older PS run / VLM omission), assemble one from the verdict
        # fields PS DID return.
        if ps_corrective:
            policy_feedback = ps_corrective
            feedback_source = "per_step_verifier"
        else:
            parts = []
            if ps_reason:
                parts.append(f"Per-step verifier said: {ps_reason}")
            if unsatisfied:
                parts.append(
                    "Unsatisfied conditions: "
                    + "; ".join(str(c) for c in unsatisfied if c)
                )
            if stderr_tail.strip():
                parts.append(f"Stderr tail: {stderr_tail.strip()[-300:]}")
            parts.append(
                f"This step has now failed {len(prior_retries)} time(s). "
                "Change something concrete (named argument value, primitive choice, "
                "or missing precondition call). Do not repeat what already failed."
            )
            policy_feedback = " ".join(parts)
            feedback_source = "per_step_verifier_assembled"

        # Composite source: "per_step_verifier" iff both fields came from PS
        # verbatim; otherwise "per_step_verifier_partial" (at least one was
        # defaulted/assembled from PS sub-fields).
        if edit_scale_source == "per_step_verifier" and feedback_source == "per_step_verifier":
            source = "per_step_verifier"
        elif edit_scale_source == "per_step_verifier_default" and feedback_source == "per_step_verifier_assembled":
            source = "per_step_verifier_default"
        else:
            source = "per_step_verifier_partial"

        return {
            "edit_scale": edit_scale,
            "policy_feedback": policy_feedback,
            "ps_status": ps_status,
            "ps_reason": ps_reason,
            "source": source,
            "edit_scale_source": edit_scale_source,
            "feedback_source": feedback_source,
        }

    def _execute(
        self,
        code: str,
        env: Any,
        scene_context: dict[str, Any],
        *,
        step_idx: int = 0,
        retry: int = 0,
        step: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run ``executor.execute`` with the same video/exec_history capture
        the legacy path uses, so per_step_verifier finds step_frame_segments
        and a motion video to verify against.

        Per retry we ``enable_video_capture(True, clear=True)`` — multiturn-
        reset already re-runs the env from a fresh reset on each retry, so
        the prior retry's frames are throwaway. The current retry's buffer
        will contain frames for both the committed-prior-steps replay AND
        the new step k; ``_build_step_frame_segments`` slices them by the
        ``step_context`` markers the composed code already includes.
        """
        low_level = getattr(env, "low_level_env", env)
        recording_frames = hasattr(low_level, "get_video_frame_count")

        # 1. Enable video capture, clear buffer (fresh per retry)
        if hasattr(low_level, "enable_video_capture"):
            try:
                low_level.enable_video_capture(True, clear=True)
            except Exception:
                pass
        turn_frame_start = (
            low_level.get_video_frame_count() if recording_frames else 0
        )

        # 2. Wire api-logging + execution_logger context so exec_history
        #    records the per-step boundaries from `with step_context(...)`.
        api_logging_states: list = []
        if self.api_logging_enable_fn is not None:
            try:
                api_logging_states = self.api_logging_enable_fn(env)
            except Exception as exc:
                logger.debug("Multiturn-reset: api_logging_enable failed: %s", exc)
                api_logging_states = []

        exec_history = None
        try:
            from rats.utils import execution_logger
            execution_logger.init_execution_context(
                code_block_index=step_idx * 100 + retry,
                emit_callback=None,
                frame_count_provider=(
                    low_level.get_video_frame_count if recording_frames else None
                ),
                frame_fps=20.0 if recording_frames else None,
            )
        except Exception as exc:
            logger.warning("Multiturn-reset: execution_logger init failed: %s", exc)

        # 3. Run the policy
        try:
            try:
                exec_result = self.executor.execute(code, env, scene_context)
            except Exception as exc:
                logger.exception("Multiturn-reset: executor.execute raised: %s", exc)
                exec_result = {
                    "success": False, "stdout": "",
                    "stderr": f"executor exception: {exc}",
                    "reward": None, "task_completed": False, "artifacts": {},
                }
        finally:
            try:
                from rats.utils import execution_logger as _el
                exec_history = _el.finalize_execution_context()
            except Exception as exc:
                logger.warning("Multiturn-reset: execution_logger finalize failed: %s", exc)
            if api_logging_states and self.api_logging_restore_fn is not None:
                try:
                    self.api_logging_restore_fn(api_logging_states)
                except Exception as exc:
                    logger.debug(
                        "Multiturn-reset: api_logging_restore failed: %s", exc,
                    )

        # 4. Pull per-turn frames from the buffer and attach to exec_result
        #    so per_step_verifier finds them. Mirrors lifelong_loop's
        #    post-exec block (line ~3232) but scoped to this single retry.
        turn_frame_end = -1
        n_frames = 0
        n_history_steps = 0
        n_policy_step_events = 0
        n_segments = 0
        # ALWAYS log exec_history shape so we can debug independent of
        # whether frame extraction succeeded.
        if exec_history is not None:
            history_steps_all = list(getattr(exec_history, "steps", []) or [])
            n_history_steps = len(history_steps_all)
            n_policy_step_events = sum(
                1 for s in history_steps_all
                if str(getattr(s, "timeline_kind", "") or "") == "policy_step"
            )
        if recording_frames:
            try:
                turn_frame_end = low_level.get_video_frame_count()
                if (
                    turn_frame_end > turn_frame_start
                    and hasattr(low_level, "get_video_frames_range")
                ):
                    turn_frames = low_level.get_video_frames_range(
                        turn_frame_start, turn_frame_end,
                    )
                    if turn_frames:
                        turn_frames_list = list(turn_frames)
                        n_frames = len(turn_frames_list)
                        exec_result["trajectory_frame_count"] = n_frames
                        exec_result["trajectory_video_frames"] = turn_frames_list
                        if self.frame_segment_builder is not None:
                            try:
                                segs = self.frame_segment_builder(
                                    exec_history,
                                    turn_frame_start,
                                    turn_frame_end,
                                    turn_frames_list,
                                )
                                exec_result["step_frame_segments"] = segs
                                n_segments = len(segs) if isinstance(segs, list) else 0
                            except Exception as exc:
                                logger.warning(
                                    "Multiturn-reset: frame_segment_builder failed: %s",
                                    exc,
                                )
            except Exception as exc:
                logger.warning("Multiturn-reset: frame extraction failed: %s", exc)

        # Fallback: perception-only steps and crashes that happen before
        # any env.step() tick the sim leave the video buffer empty. PS
        # short-circuits when it has zero step media, so for THIS step we
        # synthesise a single segment from the executor's before/after
        # frames. PS then has at least the static scene to inspect — better
        # than auto-failing every perception-only step.
        if not exec_result.get("step_frame_segments") and step is not None:
            fallback = [
                f for f in (
                    exec_result.get("before_frame"),
                    exec_result.get("after_frame"),
                )
                if f is not None
            ]
            if fallback:
                sid = self._step_id(step, step_idx)
                exec_result["step_frame_segments"] = [{
                    "policy_step_id": sid,
                    "policy_step_index": step_idx,
                    "policy_step_goal": str(
                        step.get("description") or step.get("goal") or sid
                    ),
                    "frame_count": len(fallback),
                    "sampled_frame_count": len(fallback),
                    "sampled_frames": fallback,
                    "synthetic_fallback": True,
                }]
                # Also expose as trajectory frames so any consumer falling
                # back to full_trajectory_frames also finds them.
                exec_result.setdefault("trajectory_video_frames", fallback)
                n_segments = 1

        # Attach the execution_history's API events to exec_result so
        # PerStepVerifier's ``MAIN-FUNCTION API CALLS DURING THIS STEP``
        # section is populated for LIBERO too — rats/envs/tasks/base.py
        # only fills ``info["api_call_trace"]`` for MolmoSpaces (the only
        # env that binds primitives through ``_wrap_api_function``). Filter
        # by ``step_idx`` because in MT-reset each verify_attempt is
        # scoped to a single plan step.
        if exec_history is not None:
            try:
                from rats.agents.per_step_verifier import (
                    extract_exec_history_api_events,
                )
                step_api_events = extract_exec_history_api_events(
                    exec_history, policy_step_index=step_idx,
                )
                if step_api_events:
                    exec_result["api_timeline_events"] = step_api_events
            except Exception as exc:
                logger.debug(
                    "Multiturn-reset: api_timeline_events extract failed: %s",
                    exc,
                )

        logger.info(
            "  MT-RESET capture diag step=%d retry=%d: frame_range=(%d,%d) "
            "n_frames=%d exec_history_steps=%d policy_step_events=%d segments=%d",
            step_idx + 1, retry + 1, turn_frame_start, turn_frame_end,
            n_frames, n_history_steps, n_policy_step_events, n_segments,
        )
        return exec_result

    def _verify_single_step(
        self,
        *,
        exec_result: dict[str, Any],
        plan: dict[str, Any],
        step: dict[str, Any],
        full_code: str,
        output_dir: Path | None,
        iteration: int,
        attempt_in_iter: int,
        step_idx: int,
        retry: int,
    ) -> dict[str, Any]:
        per_step_dir = None
        if output_dir is not None:
            per_step_dir = (
                output_dir / f"step_{step_idx + 1:02d}" / f"retry_{retry:02d}" / "per_step"
            )
        try:
            return self.per_step_verifier.verify_attempt(
                exec_result,
                plan=plan,
                step=step,
                output_dir=per_step_dir,
                iteration=iteration,
                attempt=attempt_in_iter,
                code=full_code,
            )
        except Exception as exc:
            logger.exception("Multiturn-reset: per_step_verifier crashed: %s", exc)
            return {
                "enabled": True,
                "steps": [{
                    "step_id": self._step_id(step, step_idx),
                    "status": "failed",
                    "success": False,
                    "confidence": 0.0,
                    "reason": f"per_step_verifier exception: {exc}",
                }],
            }

    @staticmethod
    def _extract_single_step_verdict(ps_out: dict[str, Any]) -> dict[str, Any]:
        steps = ps_out.get("steps") or []
        if not steps:
            return {"status": "failed", "success": False, "confidence": 0.0,
                    "reason": "verifier returned no step verdicts"}
        return steps[0] if isinstance(steps[0], dict) else {}

    def _compose_full_code(
        self,
        *,
        steps: list[dict[str, Any]],
        committed: dict[int, str],
        step_idx: int,
        step_code: str,
        skill_preamble: str = "",
    ) -> str:
        """Assemble TOP-LEVEL executable code: skill defs + per-step blocks.

        CodeExecEnvBase._exec_user_code runs the code via ``exec(code, globals)``,
        so the per-step blocks must be at module top level (not wrapped in
        ``def main(env):``). ``env`` is already bound in exec_globals by
        CodeExecEnvBase._init_exec_globals. Each step (committed or candidate)
        is wrapped in a ``with step_context(step_id, description, step_index=k):``
        block so per_step_verifier's slicer and execution_logger's
        policy_step events line up with the plan steps.
        """
        lines: list[str] = []
        if skill_preamble.strip():
            lines.append(skill_preamble.rstrip())
            lines.append("")

        for k, s in enumerate(steps):
            if k > step_idx:
                break
            body = committed.get(k) if k < step_idx else step_code
            if body is None:
                continue
            sid = self._step_id(s, k)
            desc = (s.get("description") or s.get("goal") or "")
            desc_safe = desc.replace('"', "'").replace("\n", " ")[:200]
            lines.append("")
            lines.append(f"# === Step {k + 1}: {desc_safe[:80]} ===")
            lines.append(
                f'with step_context("{sid}", "{desc_safe}", step_index={k}):'
            )
            body_lines = (body or "").splitlines() or ["pass"]
            indented_any = False
            for ln in body_lines:
                if ln.strip():
                    indented_any = True
                    lines.append("    " + ln)
                else:
                    lines.append("")
            if not indented_any:
                lines.append("    pass")

        lines.append("")
        lines.append("RESULT = {'success': True}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _normalize_writer_output(code: str) -> str:
        """Reuse base_agent.extract_python_code to unwrap any ```python ...```
        fences the writer may have emitted; fall back to the raw text when
        no fence was present (extract_python_code returns empty in that case).
        """
        if not code.strip():
            return ""
        extracted = extract_python_code(code)
        return (extracted or code).strip()

    # ----------------------------------------------------------------
    # Persistence helpers (used by lifelong_loop when storing into iter JSON)
    # ----------------------------------------------------------------
    @staticmethod
    def record_to_dict(rec: StepRetryRecord) -> dict[str, Any]:
        return {
            "retry": rec.retry,
            "edit_scale": rec.edit_scale,
            "ps_status": rec.ps_status,
            "ps_confidence": rec.ps_confidence,
            "ps_reason": rec.ps_reason[:1000],
            "exec_success": rec.exec_success,
            "exec_stderr_tail": rec.exec_stderr_tail[:500],
            "feedback_summary": rec.feedback_summary[:500],
            "step_code_chars": len(rec.step_code),
        }

    @classmethod
    def result_to_dict(cls, result: MultiturnResetRunResult) -> dict[str, Any]:
        return {
            "status": result.status,
            "stuck_step_idx": result.stuck_step_idx,
            "stuck_step_id": result.stuck_step_id,
            "stuck_reason": result.stuck_reason,
            "committed_step_count": len(result.committed_step_codes),
            "total_step_retries": result.total_step_retries,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "per_step_verifications_count": len(result.per_step_verifications),
            "step_retry_log": {
                str(k): [cls.record_to_dict(r) for r in v]
                for k, v in result.step_retry_log.items()
            },
            "notes": list(result.notes),
            "force_committed_indices": list(result.force_committed_indices),
            "skipped_indices": list(result.skipped_indices),
            "step_quality_results": list(result.step_quality_results),
        }
