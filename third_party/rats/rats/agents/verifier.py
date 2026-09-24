"""Verifier: programmatic check of goal-state conditions + LLM analysis.

Two layers:

1. **Symbolic verifier** — uses LIBERO's parsed_problem["goal_state"] +
   _eval_predicate to evaluate each goal predicate individually against the
   simulator's true state. Returns per-predicate satisfied/unsatisfied lists
   plus a one-line state_hint. This is ground truth.

2. **LLM analyst** — if any predicate is unsatisfied AND the caller passes the
   policy code (and optional attempt_history), the verifier asks an LLM to
   reason over (goal predicates + per-predicate state + code + history) and
   produce:
     - root_cause_predicate (which predicate is the root failure)
     - code_antipattern (what the code did wrong, code-specific)
     - fix_suggestion (concrete next-attempt fix, naming real primitives)
     - fix_pseudo_code (1-3 line snippet)
     - generalizable_lesson (condition / antipattern / remedy that lifelong_loop
       pushes to failure_memory so future iterations on different tasks reuse it)

For LIBERO, the simulator goal predicates are the authoritative outcome
label.  Structured and visual checks remain useful diagnostic signals, but
they may not override a failed native predicate.  This matters during
self-play: a visual false positive must never be promoted into the skill
library as a successful experience.

This is RATS's verifier, NOT CaP-X's "did reward >= 0.99" check. Built to
direct future iterations and attempts via failure_memory, not just gate
the current one.

Prompt template lives at `prompts/verifier.txt`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("rats.verifier")


class Verifier:
    DEFAULT_MODEL = "google/gemini-3.1-pro-preview"

    def __init__(
        self,
        prompt_path: str | Path = "rats/prompts/verifier.txt",
        *,
        model: str | None = None,
    ) -> None:
        self.prompt_path = Path(prompt_path)
        resolved = (
            model
            if model is not None
            else os.getenv("RATS_VERIFIER_MODEL", self.DEFAULT_MODEL)
        )
        resolved = str(resolved or "").strip()
        self.model = (
            None
            if resolved.lower() in {"", "default", "global", "inherit"}
            else resolved
        )

    def _query_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        max_tokens: int,
    ) -> dict[str, Any]:
        from rats.agents.base_agent import query_llm_json

        kwargs: dict[str, Any] = {"images": images, "max_tokens": max_tokens}
        if videos:
            kwargs["videos"] = videos
        if self.model:
            kwargs["model"] = self.model
        try:
            return query_llm_json(system_prompt, user_prompt, **kwargs)
        except TypeError as exc:
            # Some tests monkeypatch query_llm_json with a narrower signature.
            if "unexpected keyword argument 'model'" not in str(exc):
                raise
            kwargs.pop("model", None)
            return query_llm_json(system_prompt, user_prompt, **kwargs)

    def verify(
        self,
        execution_result: dict[str, Any],
        task_proposal: dict[str, Any],
        env: Any | None = None,
        code: str = "",
        attempt_history: list[dict[str, Any]] | None = None,
        failure_memory_view: str = "",
        plan: dict[str, Any] | None = None,
        artifact_dir: str | Path | None = None,
        artifact_prefix: str = "verifier",
    ) -> dict[str, Any]:
        """Check whether goal-state conditions are satisfied.

        Args:
            execution_result: From Executor (success, reward, task_completed, stdout, stderr).
            task_proposal: Task info with goal conditions.
            env: Optional. The CaP-Gym wrapper env. If it carries a LIBERO
                low_level_env with parsed_problem["goal_state"] +
                _eval_predicate, we evaluate every predicate individually so
                the caller can see which goals are still unsatisfied.
            plan: Optional planner output ({"steps":[...]}). When supplied we
                emit per-step reachability feedback by keyword-matching each
                step's description to the synthesized predicate list.

        Returns:
            Dict with:
              - success: bool
              - reward: float
              - task_completed: bool
              - satisfied_conditions: list[str]
              - unsatisfied_conditions: list[str]
              - predicate_status: list[dict]      (per-predicate, when env supports it)
              - plan_step_feedback: list[dict]    (per-plan-step reached/not, when plan given)
              - state_hint: str                   (one-line hint for the next attempt)
              - evidence: dict
        """
        reward = float(execution_result.get("reward", 0.0) or 0.0)
        task_completed = bool(execution_result.get("task_completed", False))
        exec_success = bool(execution_result.get("success", False))

        native_success = task_completed or reward >= 0.99
        structured_custom_result: dict[str, Any] | None = None
        visual_custom_result: dict[str, Any] | None = None
        structured_custom_success = False
        visual_custom_success = False
        save_artifacts = artifact_dir is not None
        visual_frame_path = (
            Path(artifact_dir) / f"{artifact_prefix}_vlm_frame.png"
            if artifact_dir is not None else None
        )

        # Treat task-state verification as an OR over three success signals:
        #   1. the simulator's native reward/task_completed result,
        #   2. the deterministic structured custom verifier, and
        #   3. a VLM verifier over the terminal frame.
        # Keep the execution-success gate for fallback/custom verification.
        # When the simulator exposes its benchmark predicates directly, those
        # predicates are the outcome authority below.  In particular, a
        # post-goal helper error (a final observation, retreat, or an action
        # issued after LIBERO has terminated the successful episode) must not
        # turn an already-satisfied benchmark state into a false negative.
        has_structured_cv = bool(task_proposal.get("custom_verifier_code"))
        if exec_success and has_structured_cv and (save_artifacts or not native_success):
            structured_custom_result = self._run_custom_verifier(
                task_proposal["custom_verifier_code"], env,
            )
            structured_custom_success = self._truthy_bool(
                structured_custom_result.get("success"),
            )
            logger.info(f"  Structured custom verifier result: {structured_custom_result}")
        elif exec_success and (save_artifacts or not native_success):
            logger.debug("  No structured custom verifier code available for this task")

        if exec_success and (
            save_artifacts or (not native_success and not structured_custom_success)
        ):
            visual_custom_result = self._run_visual_verifier(
                execution_result,
                task_proposal,
                artifact_image_path=visual_frame_path,
            )
            visual_custom_success = self._truthy_bool(visual_custom_result.get("success"))
            logger.info(f"  Visual custom verifier result: {visual_custom_result}")

        verified = exec_success and (
            native_success or structured_custom_success or visual_custom_success
        )
        custom_override = bool(
            exec_success
            and not native_success
            and (structured_custom_success or visual_custom_success)
        )
        custom_override_source = (
            "structured"
            if structured_custom_success
            else "visual" if visual_custom_success else ""
        )

        # Try to pull a per-predicate breakdown out of the LIBERO env.
        predicate_status: list[dict[str, Any]] = []
        unsat_predicate_descs: list[str] = []
        sat_predicate_descs: list[str] = []
        has_native_predicate_api = False
        try:
            # FIX (LIBERO predicate probe): the chain
            #   env.low_level_env.handle.env
            # lands on OffScreenRenderEnv (a LIBERO wrapper), which does NOT
            # expose parsed_problem / _eval_predicate. The actual problem
            # object that owns those is OffScreenRenderEnv.env (e.g.
            # `Libero_Tabletop_Manipulation`). The previous code stopped at
            # the wrapper and silently logged a `[]/[]` result. Walk through
            # `.env` and `.unwrapped` wrappers until we either find the
            # predicate API or run out of layers.
            low = getattr(env, "low_level_env", env)  # CaP-Gym wraps the sim
            handle = getattr(low, "handle", None) if low is not None else None
            sim_env = getattr(handle, "env", None) if handle is not None else None
            parsed = getattr(sim_env, "parsed_problem", None)
            eval_pred = getattr(sim_env, "_eval_predicate", None)
            if (parsed is None or not callable(eval_pred)) and sim_env is not None:
                visited = {id(sim_env)}
                cur = sim_env
                for _ in range(8):  # bounded — LIBERO has at most 1-2 wrappers
                    nxt = getattr(cur, "env", None) or getattr(cur, "unwrapped", None)
                    if nxt is None or id(nxt) in visited:
                        break
                    visited.add(id(nxt))
                    cur = nxt
                    parsed = getattr(cur, "parsed_problem", None)
                    eval_pred = getattr(cur, "_eval_predicate", None)
                    if parsed is not None and callable(eval_pred):
                        sim_env = cur
                        break
            if parsed is not None and callable(eval_pred):
                has_native_predicate_api = True
                for state in parsed.get("goal_state", []) or []:
                    desc = "[" + " ".join(str(s) for s in state) + "]"
                    try:
                        ok = bool(eval_pred(state))
                    except Exception as e:
                        ok = False
                        logger.warning(f"  predicate eval failed for {desc}: {e}")
                    predicate_status.append({"predicate": desc, "satisfied": ok})
                    (sat_predicate_descs if ok else unsat_predicate_descs).append(desc)
            elif sim_env is not None:
                # Even after wrapper-unwrapping we still couldn't find the
                # predicate API. Surface the chain so future debugging doesn't
                # need to rediscover the structure.
                logger.warning(
                    "  per-predicate probe couldn't reach _eval_predicate; "
                    "chain ended at %r (parsed_problem=%s, _eval_predicate=%s)",
                    type(sim_env).__name__,
                    "present" if parsed is not None else "missing",
                    "callable" if callable(eval_pred) else "missing",
                )
        except Exception as e:
            logger.warning(f"  per-predicate verifier probe failed: {e}")

        # MolmoSpaces probe. Consumes the task's get_info() (piped through
        # bridge → server → remote → FrankaMolmoSpacesEnv.get_task_info) so
        # each synthesized subgoal gets an independent satisfied bit rather
        # than all sharing the single judge_success() verdict.
        if not predicate_status:
            try:
                predicate_status = self._probe_molmospaces_predicates(
                    env, verified=verified,
                )
                for p in predicate_status:
                    (sat_predicate_descs if p["satisfied"] else unsat_predicate_descs).append(
                        p["predicate"]
                    )
            except Exception as e:
                logger.debug(f"  molmospaces verifier probe failed: {e}")

        # LIBERO outcome authority.  The visual verifier is deliberately
        # permissive because it is meant to explain failures and propose
        # repairs.  It is not a replacement for the benchmark's exact goal
        # predicates (containment, contact, articulation threshold, etc.).
        # Previously these signals were OR'ed, so a terminal RGB frame that
        # merely looked correct could label a native failure as success and
        # poison skill extraction.  When the native predicate API is
        # available, require every benchmark predicate to be satisfied.
        native_predicate_authority = bool(has_native_predicate_api and predicate_status)
        strict_benchmark = os.getenv("RATS_VERIFIER_STRICT_BENCHMARK", "0") == "1"
        native_predicate_success: bool | None = None
        diagnostic_disagreement = False
        if native_predicate_authority:
            native_predicate_success = all(
                bool(entry.get("satisfied")) for entry in predicate_status
            )
            diagnostic_disagreement = bool(
                (structured_custom_success or visual_custom_success)
                != native_predicate_success
            )
            # Native benchmark state is the label.  Do not gate it on whether
            # the generated Python returned cleanly: LIBERO can terminate as
            # soon as the goal becomes true, so harmless post-success code can
            # raise even though the task is natively complete.
            verified = bool(native_predicate_success)
            # Custom verifiers remain in evidence as diagnostics, but are not
            # allowed to claim an override in a LIBERO benchmark episode.
            custom_override = False
            custom_override_source = ""

        # MolmoSpaces no-op guard: some articulated tasks can report success
        # immediately after reset (for example a laptop/drawer already past the
        # open threshold). Do not mark an attempt successful when no trajectory
        # frames beyond the initial frame were captured and the scalar reward is
        # still below the normal completion threshold. This prevents smoke runs
        # from labeling a no-motion policy as a success.
        try:
            low = getattr(env, "low_level_env", env)
            get_desc = getattr(low, "get_task_descriptor", None)
            descriptor = get_desc() if callable(get_desc) else {}
            is_molmospaces = str((descriptor or {}).get("canonical_id", "")).startswith(
                "molmospaces:"
            )
        except Exception:
            is_molmospaces = False
        frames = execution_result.get("trajectory_frames") or []
        no_motion_guard = bool(
            is_molmospaces
            and verified
            and len(frames) <= 1
            and reward < 0.99
        )
        if no_motion_guard:
            verified = False
            if predicate_status:
                for entry in predicate_status:
                    entry["satisfied"] = False
                    ev = entry.setdefault("evidence", {})
                    if isinstance(ev, dict):
                        ev["no_motion_guard"] = True
                        ev["trajectory_frame_count"] = len(frames)
                        ev["reason"] = (
                            "task was already satisfied at/near reset; "
                            "no robot motion was captured"
                        )
                sat_predicate_descs = []
                unsat_predicate_descs = [p["predicate"] for p in predicate_status]

        # Per-plan-step feedback (optional). Maps each planner step to its
        # best-matching predicate by keyword overlap and records whether the
        # matched predicate is satisfied. Gives the retry prompt a direct
        # "step-2 ‘place bowl on plate’ not reached" signal instead of a bare
        # predicate list.
        plan_step_feedback: list[dict[str, Any]] = []
        if plan and predicate_status:
            try:
                plan_step_feedback = self._match_plan_steps_to_predicates(
                    plan, predicate_status,
                )
            except Exception as e:
                logger.debug(f"  plan-step feedback failed: {e}")

        # Build a one-line state hint the policy_writer's retry prompt can
        # paste verbatim. When we have per-predicate data this is much more
        # actionable than the old "task incomplete" string.
        if predicate_status:
            if not unsat_predicate_descs:
                state_hint = "All goal predicates are satisfied."
            else:
                done_part = (
                    f"Already satisfied: {', '.join(sat_predicate_descs)}. "
                    if sat_predicate_descs else ""
                )
                if custom_override:
                    state_hint = (
                        f"{custom_override_source.title()} custom verifier confirms task "
                        f"completion even though strict predicates remain unsatisfied: "
                        f"{', '.join(unsat_predicate_descs)}."
                    )
                else:
                    state_hint = (
                        f"{done_part}Still NOT satisfied: {', '.join(unsat_predicate_descs)}. "
                        f"Address those predicates next without undoing the satisfied ones."
                    )
        elif custom_override:
            state_hint = (
                f"{custom_override_source.title()} custom verifier confirms task completion "
                "(native simulator verifier did not mark the task complete)."
            )
        else:
            # Fallback: no env-level introspection available
            state_hint = (
                "Goal verifier reports task NOT completed (reward < 1.0). "
                "Re-examine your code's pick/place/release sequence."
                if not verified else "Goal satisfied."
            )

        # Goal description (from proposer or activity name) for legacy callers.
        goal_label = task_proposal.get(
            "goal_conditions",
            task_proposal.get("activity_name", "task_complete"),
        )
        if predicate_status:
            satisfied = sat_predicate_descs
            unsatisfied = unsat_predicate_descs
        else:
            satisfied = [goal_label] if verified else []
            unsatisfied = [] if verified else [goal_label]

        evidence = {
            "reward": reward,
            "task_completed": task_completed,
            "execution_success": exec_success,
            "custom_verifier_override": custom_override,
            "no_motion_guard": no_motion_guard,
            "custom_verifier_override_source": custom_override_source or None,
            "outcome_authority": (
                "native_predicates" if native_predicate_authority else "verifier_vote"
            ),
            "native_predicate_success": native_predicate_success,
            "native_success_overrode_execution_error": bool(
                native_predicate_authority
                and native_predicate_success
                and not exec_success
            ),
            "diagnostic_disagreement": diagnostic_disagreement,
            "control_feedback_policy": (
                "binary_native_outcome_only"
                if strict_benchmark and native_predicate_authority
                else "standard"
            ),
            "native_verifier": {
                "attempted": True,
                "success": native_success,
                "reward": reward,
                "task_completed": task_completed,
                "threshold": 0.99,
            },
            "verifier_votes": {
                "native": native_success,
                "structured_custom": structured_custom_success,
                "visual_custom": visual_custom_success,
            },
            "structured_custom_verifier": structured_custom_result,
            "visual_custom_verifier": visual_custom_result,
            "stdout_snippet": (execution_result.get("stdout", ""))[:500],
            "stderr_snippet": (execution_result.get("stderr", ""))[:500],
        }
        if exec_success and not verified and reward > 0:
            evidence["progress"] = reward

        # Strict benchmark isolation: native predicates are private evaluator
        # state.  They may determine the binary benchmark label and may be
        # retained for offline auditing, but they must never reach policy
        # repair, failure-memory distillation, planning, or a diagnostic LLM.
        # Keep the exact breakdown under evidence so experiment analysis can
        # still prove how the label was obtained, then redact every
        # control-facing field below.
        privileged_feedback_redacted = bool(
            strict_benchmark and native_predicate_authority
        )
        if privileged_feedback_redacted:
            evidence["private_native_audit"] = {
                "predicate_status": predicate_status,
                "satisfied_conditions": sat_predicate_descs,
                "unsatisfied_conditions": unsat_predicate_descs,
                "plan_step_feedback": plan_step_feedback,
            }
            evidence["privileged_feedback_redacted"] = True

        # Layer 2: LLM analysis (only on failure, only when caller gave us code)
        llm_analysis: dict[str, Any] | None = None
        if not verified and code and not privileged_feedback_redacted:
            llm_analysis = self._llm_analyze(
                task_proposal=task_proposal,
                predicate_status=predicate_status,
                satisfied=sat_predicate_descs,
                unsatisfied=unsat_predicate_descs,
                code=code,
                attempt_history=attempt_history or [],
                failure_memory_view=failure_memory_view,
            )

        observed_effect = ""
        if isinstance(visual_custom_result, dict):
            observed_effect = str(visual_custom_result.get("observed_effect") or "").strip()

        public_satisfied = [] if privileged_feedback_redacted else satisfied
        public_unsatisfied = [] if privileged_feedback_redacted else unsatisfied
        public_predicate_status = [] if privileged_feedback_redacted else predicate_status
        public_plan_step_feedback = [] if privileged_feedback_redacted else plan_step_feedback
        public_state_hint = (
            (
                "Independent task verifier reports task NOT completed. "
                "Use only the public visual trajectory and runtime errors to "
                "decide the next corrective action."
            )
            if privileged_feedback_redacted and not verified
            else "Goal satisfied."
            if privileged_feedback_redacted
            else state_hint
        )

        result = {
            "success": verified,
            "reward": reward,
            "task_completed": task_completed,
            "satisfied_conditions": public_satisfied,
            "unsatisfied_conditions": public_unsatisfied,
            "predicate_status": public_predicate_status,
            "plan_step_feedback": public_plan_step_feedback,
            "state_hint": public_state_hint,
            "llm_analysis": llm_analysis,
            "evidence": evidence,
            "observed_effect": observed_effect,
        }
        if artifact_dir is not None:
            artifact_paths = self._save_verifier_artifacts(
                artifact_dir=Path(artifact_dir),
                artifact_prefix=artifact_prefix,
                result=result,
                task_proposal=task_proposal,
                native_success=native_success,
                structured_custom_result=structured_custom_result,
                structured_custom_success=structured_custom_success,
                has_structured_cv=has_structured_cv,
                visual_custom_result=visual_custom_result,
                visual_custom_success=visual_custom_success,
                visual_frame_path=visual_frame_path,
            )
            result["artifact_paths"] = artifact_paths
            result["evidence"]["artifact_paths"] = artifact_paths
        return result

    # ------------------------------------------------------------------
    # Layer 2: LLM-driven code+state+trajectory analyst.
    # ------------------------------------------------------------------

    def _llm_analyze(
        self,
        task_proposal: dict[str, Any],
        predicate_status: list[dict[str, Any]],
        satisfied: list[str],
        unsatisfied: list[str],
        code: str,
        attempt_history: list[dict[str, Any]],
        failure_memory_view: str = "",
    ) -> dict[str, Any] | None:
        """Reason over (goal, state, code, history, failure_memory) -> root cause + fix."""
        try:
            system_prompt = self.prompt_path.read_text()
        except Exception as e:
            logger.warning(f"  verifier prompt missing at {self.prompt_path}: {e}")
            return None

        goal_lang = (
            task_proposal.get("language")
            or task_proposal.get("goal_conditions")
            or task_proposal.get("activity_name")
            or "?"
        )
        goal_predicates = (
            task_proposal.get("goal_predicates")
            or task_proposal.get("goal")
            or []
        )

        # Truncate code so the prompt stays bounded
        code_snip = code 

        history_lines = []
        for h in (attempt_history or [])[-5:]:  # last 5 attempts at most
            history_lines.append(
                f"- attempt {h.get('attempt')}: success={h.get('success')} "
                f"unsatisfied={h.get('unsatisfied_conditions', [])} "
                f"failure_mode={h.get('failure_mode', '?')}"
            )
        history_block = "\n".join(history_lines) if history_lines else "(this is attempt 0, no history)"

        fm_block = ""
        if failure_memory_view:
            fm_block = (
                "\nRELEVANT DISTILLED LESSONS FROM PRIOR RUNS (use them to "
                "avoid proposing fixes that were already tried and failed, "
                "and to keep your new lesson consistent with the existing set):\n"
                f"{failure_memory_view}\n"
            )

        user_prompt = (
            f"TASK GOAL (natural language): {goal_lang}\n"
            f"TASK GOAL (BDDL predicates): {goal_predicates}\n\n"
            f"PER-PREDICATE STATE (ground truth from simulator):\n"
            f"  satisfied:   {satisfied}\n"
            f"  unsatisfied: {unsatisfied}\n\n"
            f"ATTEMPT HISTORY (so don't repeat what's already failed):\n"
            f"{history_block}\n"
            f"{fm_block}"
            f"\nCODE THAT JUST RAN:\n```python\n{code_snip}\n```\n"
        )

        try:
            # 600 tokens was too tight — Gemini-3.1-pro-preview consistently
            # spent 570+ on reasoning, leaving ~30 for the JSON. 100% of
            # tier-2 verifier calls in the LIBERO smoke run truncated mid-JSON
            # (always cut off after `"root_cause_predicate": "['On', 'milk_`).
            # Pinned to 65536 — the cross-provider ceiling (Gemini-3 Pro = 65536,
            # Claude 4.x = 64k, gpt-5 = 128k) — so reasoning + JSON always fit.
            result = self._query_llm_json(system_prompt, user_prompt, max_tokens=65536)
        except Exception as e:
            logger.warning(f"  verifier LLM analysis failed (non-fatal): {e}")
            return None

        # Gemini-3.1-pro occasionally wraps the JSON object in a single-
        # element list (``[{...}]``) or returns a bare list of partial
        # results; coerce to dict before .get(). Observed iter4 of the
        # core200 llm play arm — without this guard the AttributeError
        # propagates up through Verifier.verify, kills the whole iter,
        # and the iter's task_proposal / skill-extraction signal is lost.
        if isinstance(result, list):
            result = next(
                (x for x in result if isinstance(x, dict)),
                None,
            )
        if not isinstance(result, dict):
            logger.warning(
                "  verifier LLM analysis returned %s instead of dict; "
                "treating as 'no diagnosis available'",
                type(result).__name__,
            )
            return None

        if result.get("task_satisfied"):
            return {"task_satisfied": True}

        return {
            "root_cause_predicate": result.get("root_cause_predicate"),
            "code_antipattern": result.get("code_antipattern"),
            "fix_suggestion": result.get("fix_suggestion"),
            "fix_pseudo_code": result.get("fix_pseudo_code", ""),
            "generalizable_lesson": result.get("generalizable_lesson") or {},
            "confidence": float(result.get("confidence", 0.5)),
        }

    # ------------------------------------------------------------------
    # Custom verifier execution
    # ------------------------------------------------------------------

    def _run_custom_verifier(
        self, code_str: str, env: Any,
    ) -> dict[str, Any]:
        """Execute a structured custom verifier function.

        The structured verifier provides relaxed success checks for novel BDDL
        tasks where LIBERO's built-in predicates are too strict (e.g. On/In
        predicates that should use target bounds instead of center distance).
        """
        try:
            # Unwrap CaP-Gym layers to get the raw LIBERO domain env
            # Chain: CaPGymEnv -> low_level_env -> handle -> OffScreenRenderEnv -> env (domain)
            low = getattr(env, "low_level_env", env)
            handle = getattr(low, "handle", None) if low is not None else None
            sim_env = getattr(handle, "env", None) if handle is not None else None
            # OffScreenRenderEnv wraps the domain env via .env attribute
            # Keep unwrapping until we find obj_body_id
            for _ in range(5):
                if sim_env is None:
                    break
                if hasattr(sim_env, "obj_body_id"):
                    break
                sim_env = getattr(sim_env, "env", None)
            if sim_env is None or not hasattr(sim_env, "obj_body_id"):
                return {"success": False, "details": "could not unwrap to domain env with obj_body_id"}

            namespace: dict[str, Any] = {}
            exec(code_str, namespace)
            verify_fn = namespace.get("custom_verify")
            if not callable(verify_fn):
                return {"success": False, "details": "no custom_verify() found"}

            result = verify_fn(sim_env)
            if not isinstance(result, dict):
                return {"success": False, "details": f"returned {type(result)}"}
            return result
        except Exception as e:
            logger.debug(f"  custom verifier exec failed: {e}")
            return {"success": False, "details": f"exec error: {e}"}

    # MolmoSpaces predicate synthesis.
    # ------------------------------------------------------------------

    # Task-family → ordered list of subgoal predicate templates.
    # {obj} / {target} expand from the descriptor's referral_expressions /
    # objects list. Families not listed here fall through to a single
    # "<language>" predicate derived from task_descriptor.language.
    _MOLMO_FAMILY_TEMPLATES: dict[str, list[str]] = {
        "pick": ["object '{obj}' is lifted clear of its support"],
        "pick_and_place": [
            "object '{obj}' is grasped and lifted",
            "object '{obj}' is placed on/at target '{target}'",
        ],
        "pick_and_place_next_to": [
            "object '{obj}' is grasped and lifted",
            "object '{obj}' is placed next to target '{target}'",
        ],
        "opening": ["articulated target '{target}' is opened past the success threshold"],
        "closing": ["articulated target '{target}' is closed past the success threshold"],
        "nav": ["robot base is within tolerance of goal waypoint"],
    }

    @classmethod
    def _probe_molmospaces_predicates(
        cls,
        env: Any | None,
        *,
        verified: bool,
    ) -> list[dict[str, Any]]:
        """Per-subgoal synthesis for MolmoSpaces tasks.

        Consumes ``FrankaMolmoSpacesEnv.get_task_info()`` (which wraps the
        upstream ``task.get_info()`` payload) so each synthesized subgoal
        gets an independent satisfied bit:

        - pick          → success
        - pick_and_place → (1) robot_contact OR supported_by_receptacle,
                           (2) supported_by_receptacle AND position_error<thr
        - pick_and_place_next_to → same, position-only check on leg 2
        - opening       → success (joint_position past threshold)
        - closing       → success
        - nav           → success

        When ``get_task_info()`` is unavailable or empty, falls back to the
        aggregate ``verified`` bit for every predicate.
        """
        if env is None:
            return []
        low = getattr(env, "low_level_env", env)
        get_desc = getattr(low, "get_task_descriptor", None)
        if not callable(get_desc):
            return []
        try:
            descriptor = get_desc() or {}
        except Exception:
            return []
        canonical_id = str(descriptor.get("canonical_id", ""))
        if not canonical_id.startswith("molmospaces:"):
            return []

        task_family = str(descriptor.get("task_family", "")) or (
            canonical_id.split(":")[3] if canonical_id.count(":") >= 4 else ""
        )
        language = str(descriptor.get("language", "")).strip()

        metadata = descriptor.get("metadata") or {}
        refs = metadata.get("referral_expressions") or {}
        obj = (
            refs.get("pickup_obj_name")
            or refs.get("object_name")
            or (descriptor.get("objects") or [None])[0]
            or "target"
        )
        target = (
            refs.get("place_receptacle_name")
            or refs.get("receptacle_name")
            or refs.get("articulated_obj_name")
            or (descriptor.get("objects") or [None, None])[1]
            or "target"
        )

        info: dict[str, Any] = {}
        get_info = getattr(low, "get_task_info", None)
        if callable(get_info):
            try:
                info = dict(get_info() or {})
            except Exception:
                info = {}

        templates = cls._MOLMO_FAMILY_TEMPLATES.get(task_family)
        if templates:
            preds = [t.format(obj=obj, target=target) for t in templates]
        elif language:
            preds = [language]
        else:
            preds = [f"task '{canonical_id}' reports success"]

        bits = cls._derive_molmo_subgoal_bits(task_family, info, verified, len(preds))
        results: list[dict[str, Any]] = []
        for pred, bit in zip(preds, bits):
            entry: dict[str, Any] = {"predicate": pred, "satisfied": bool(bit)}
            results.append(entry)

        # Attach a diagnostic snapshot to the first predicate so Layer 2 can
        # see the raw numerical evidence without expanding every row.
        if results and info:
            results[0]["evidence"] = {
                k: info[k]
                for k in (
                    "success", "position_error", "rotation_error",
                    "joint_position", "robot_contact",
                    "supported_by_receptacle", "supported_by_carry_forward",
                    "carry_forward_pos_diff", "carry_forward_rot_diff",
                    "receptacle_pos_displacement", "episode_step",
                )
                if k in info
            }
        return results

    @staticmethod
    def _derive_molmo_subgoal_bits(
        task_family: str,
        info: dict[str, Any],
        verified: bool,
        n_predicates: int,
    ) -> list[bool]:
        """Map upstream get_info() fields to independent per-subgoal bits."""
        if not info:
            return [verified] * n_predicates

        overall = bool(info.get("success", verified))

        if task_family == "pick":
            return [overall][:n_predicates] or [overall]

        if task_family in ("pick_and_place", "pick_and_place_next_to"):
            placed = bool(info.get("supported_by_receptacle", False))
            contacted = bool(info.get("robot_contact", False))
            # Subgoal 1 (grasp-and-lift): true if the robot touched it, or if
            # downstream success made contact moot.
            lifted_bit = contacted or placed or overall
            placed_bit = placed or overall
            bits = [lifted_bit, placed_bit]
            if n_predicates > 2:
                bits += [overall] * (n_predicates - 2)
            return bits[:n_predicates]

        if task_family in ("opening", "closing"):
            return [overall] * n_predicates

        if task_family == "nav":
            return [overall] * n_predicates

        return [overall] * n_predicates

    @staticmethod
    def _match_plan_steps_to_predicates(
        plan: dict[str, Any],
        predicate_status: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Annotate each planner step with reached / matched-predicate.

        Matches by keyword overlap between step description and predicate
        text. A step with no keyword hits falls through to the last
        predicate (terminal subgoal). The output is intentionally small and
        prompt-friendly — one dict per step.
        """
        steps = plan.get("steps") or []
        if not steps or not predicate_status:
            return []

        # Keyword → predicate-matching hints. Order matters: first hit wins.
        keyword_groups: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
            (("grasp", "pick", "lift", "pick up", "grab"),
             ("lift", "grasp", "pick")),
            (("place", "put", "drop", "set down", "release"),
             ("place", "placed", "on", "next to")),
            (("open", "pull", "slide out"), ("open",)),
            (("close", "push in", "shut"), ("close",)),
            (("navigate", "move to", "go to", "approach"), ("robot base", "waypoint")),
        ]

        def _find_predicate(desc: str) -> tuple[int, str]:
            desc_low = desc.lower()
            for step_kws, pred_kws in keyword_groups:
                if any(kw in desc_low for kw in step_kws):
                    for idx, p in enumerate(predicate_status):
                        if any(pk in p["predicate"].lower() for pk in pred_kws):
                            return idx, "keyword"
            return len(predicate_status) - 1, "fallback"

        out: list[dict[str, Any]] = []
        for step in steps:
            desc = str(step.get("description", "")).strip()
            sid = str(step.get("id", "")) or f"step-{len(out) + 1}"
            if not desc:
                out.append({
                    "step_id": sid, "description": "",
                    "reached": None, "matched_predicate": None,
                    "match_method": "no-description",
                })
                continue
            idx, method = _find_predicate(desc)
            pred = predicate_status[idx]
            out.append({
                "step_id": sid,
                "description": desc,
                "reached": bool(pred["satisfied"]),
                "matched_predicate": pred["predicate"],
                "match_method": method,
            })
        return out

    def _run_visual_verifier(
        self,
        execution_result: dict[str, Any],
        task_proposal: dict[str, Any],
        *,
        artifact_image_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Ask a VLM whether the robot's action achieved the task goal.

        Prefers feeding the FULL rollout as a video (Gemini-3-Pro native video
        input) so the verdict comes from the motion — did the gripper actually
        grasp / move / open / place the object — instead of a single terminal
        frame, which false-positives whenever the goal state merely looks
        satisfied in the last frame. Falls back to a multi-frame sequence, then
        to the single terminal frame, when video/frames are unavailable.
        """
        frames = execution_result.get("trajectory_frames") or []
        terminal = frames[-1] if frames else execution_result.get("after_frame")
        if terminal is None:
            return {
                "success": False,
                "confidence": 0.0,
                "details": "no frames available",
            }

        saved_image_path = None
        if artifact_image_path is not None:
            saved_image_path = self._save_frame_image(terminal, Path(artifact_image_path))

        try:
            from rats.agents.base_agent import image_to_data_url

            goal = (
                task_proposal.get("language")
                or task_proposal.get("goal_conditions")
                or task_proposal.get("goal_conditions_nl")
                or task_proposal.get("activity_name")
                or "the specified robot task"
            )
            objects = task_proposal.get("objects") or []
            fixtures = task_proposal.get("fixtures") or []
            object_block = ", ".join(str(o) for o in objects) if objects else "(not specified)"
            fixture_block = ", ".join(str(f) for f in fixtures) if fixtures else "(not specified)"

            # Strongest available visual evidence: a video of the whole rollout >
            # a sampled frame sequence > the single terminal frame.
            images: list[str] | None = None
            videos: list[str] | None = None
            video_url = self._frames_to_video_url(frames) if len(frames) >= 2 else None
            if video_url:
                videos = [video_url]
                media_desc = (
                    "A VIDEO of the full robot execution (start -> motion -> end) "
                    "is provided. Watch the whole motion before deciding."
                )
            else:
                sample = (
                    self._sample_frames(frames, max_frames=8)
                    if len(frames) >= 2
                    else [terminal]
                )
                urls = [u for u in (image_to_data_url(f) for f in sample) if u]
                if not urls:
                    return {
                        "success": False,
                        "confidence": 0.0,
                        "details": "frames could not be encoded",
                        "image_path": str(saved_image_path) if saved_image_path else None,
                    }
                images = urls
                media_desc = (
                    f"{len(urls)} frames sampled in time order (first = start, "
                    "last = final state) are provided."
                    if len(urls) > 1
                    else "Only the final frame after execution is provided."
                )

            system_prompt = (
                "You are a STRICT visual task-completion verifier for a robot "
                "manipulation benchmark. Judge from the FULL motion shown, not a "
                "single static frame. Mark success true ONLY when the visual "
                "evidence clearly shows the goal was achieved BY the robot's "
                "action during THIS rollout: the gripper must visibly grasp / "
                "move / open / close / place the relevant object, and the final "
                "state must satisfy the goal. Mark success false if the object "
                "did not visibly move, the gripper missed or only collided "
                "without securing the object, the relevant change is not visible "
                "or occluded, or the goal state already held at the start. Do not "
                "infer success from code or intent. Respond only as JSON."
            )
            user_prompt = (
                f"TASK GOAL: {goal}\n"
                f"OBJECTS: {object_block}\n"
                f"FIXTURES/TARGETS: {fixture_block}\n\n"
                f"{media_desc}\n"
                "Return JSON exactly like:\n"
                "{\n"
                '  "success": false,\n'
                '  "confidence": 0.0,\n'
                '  "evidence": "what in the motion supports the verdict",\n'
                '  "observed_effect": "what visibly changed in the scene due to the robot action"\n'
                "}\n"
            )
            result = self._query_llm_json(
                system_prompt,
                user_prompt,
                images=images,
                videos=videos,
                max_tokens=16384,
            )
            confidence = self._safe_float(result.get("confidence"), default=0.5)
            evidence = str(result.get("evidence") or result.get("details") or "")
            observed_effect = str(result.get("observed_effect") or "").strip()
            return {
                "success": self._truthy_bool(result.get("success", False)),
                "confidence": confidence,
                "details": evidence,
                "observed_effect": observed_effect,
                "image_path": str(saved_image_path) if saved_image_path else None,
                "used_video": bool(videos),
                "raw": result,
            }
        except Exception as e:
            logger.warning(f"  visual verifier failed (non-fatal): {e}")
            return {
                "success": False,
                "confidence": 0.0,
                "details": str(e),
                "image_path": str(saved_image_path) if saved_image_path else None,
            }

    def _save_verifier_artifacts(
        self,
        *,
        artifact_dir: Path,
        artifact_prefix: str,
        result: dict[str, Any],
        task_proposal: dict[str, Any],
        native_success: bool,
        structured_custom_result: dict[str, Any] | None,
        structured_custom_success: bool,
        has_structured_cv: bool,
        visual_custom_result: dict[str, Any] | None,
        visual_custom_success: bool,
        visual_frame_path: Path | None,
    ) -> dict[str, str]:
        """Persist a compact verifier trace for post-hoc manual inspection."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifact_dir / f"{artifact_prefix}.json"

        payload = {
            "artifact_prefix": artifact_prefix,
            "final_success": bool(result.get("success")),
            "task": {
                "activity_name": task_proposal.get("activity_name"),
                "goal": (
                    task_proposal.get("language")
                    or task_proposal.get("goal_conditions")
                    or task_proposal.get("goal_conditions_nl")
                ),
                "objects": task_proposal.get("objects", []),
                "fixtures": task_proposal.get("fixtures", []),
            },
            "votes": {
                "native": {
                    "attempted": True,
                    "success": bool(native_success),
                    "reward": result.get("reward"),
                    "task_completed": result.get("task_completed"),
                    "threshold": 0.99,
                },
                "structured_custom": {
                    "attempted": bool(has_structured_cv),
                    "success": bool(structured_custom_success),
                    "result": structured_custom_result,
                    "available": bool(has_structured_cv),
                },
                "visual_custom": {
                    "attempted": visual_custom_result is not None,
                    "success": bool(visual_custom_success),
                    "result": visual_custom_result,
                    "image_path": (
                        str(visual_frame_path)
                        if visual_frame_path is not None and visual_frame_path.exists()
                        else None
                    ),
                },
            },
            "predicate_status": result.get("predicate_status", []),
            "satisfied_conditions": result.get("satisfied_conditions", []),
            "unsatisfied_conditions": result.get("unsatisfied_conditions", []),
            "state_hint": result.get("state_hint", ""),
            "evidence": result.get("evidence", {}),
        }
        try:
            json_path.write_text(
                json.dumps(payload, indent=2, default=self._json_default),
            )
        except Exception as e:
            logger.warning(f"  failed to save verifier artifact JSON: {e}")
            return {}
        logger.info(f"  Verifier artifacts saved: {json_path}")
        paths = {"json": str(json_path)}
        frame_path = payload["votes"]["visual_custom"].get("image_path")
        if frame_path:
            paths["vlm_frame"] = str(frame_path)
        return paths

    @staticmethod
    def _save_frame_image(frame: Any, path: Path) -> Path | None:
        try:
            import base64
            import shutil

            import numpy as np
            from PIL import Image

            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(frame, str):
                if frame.startswith("data:"):
                    raw = frame.split(",", 1)[-1]
                    path.write_bytes(base64.b64decode(raw))
                    return path
                src = Path(frame)
                if src.exists():
                    shutil.copy2(src, path)
                    return path
                return None

            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(path)
            return path
        except Exception as e:
            logger.warning(f"  failed to save visual verifier frame: {e}")
            return None

    @staticmethod
    def _terminal_frame(execution_result: dict[str, Any]) -> Any | None:
        filmstrip = execution_result.get("trajectory_frames") or []
        if filmstrip:
            return filmstrip[-1]
        return execution_result.get("after_frame")

    @staticmethod
    def _sample_frames(frames: list[Any], *, max_frames: int) -> list[Any]:
        """Evenly sample at most ``max_frames`` frames, preserving time order."""
        if not frames:
            return []
        n = len(frames)
        if n <= max_frames:
            return list(frames)
        import numpy as np

        idx = np.linspace(0, n - 1, max_frames).round().astype(int)
        return [frames[int(i)] for i in idx]

    def _frames_to_video_url(
        self, frames: list[Any], *, max_frames: int = 24, fps: int = 6
    ) -> str | None:
        """Encode sampled trajectory frames into an mp4 data URL for the VLM.

        Returns None on any failure so the caller can fall back to frames.
        """
        sample = self._sample_frames(frames, max_frames=max_frames)
        if len(sample) < 2:
            return None
        import os
        import tempfile

        import numpy as np

        norm: list[Any] = []
        for fr in sample:
            arr = np.asarray(fr)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            norm.append(arr)
        tmp_path = None
        try:
            import imageio

            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            imageio.mimsave(tmp_path, norm, fps=fps)
            from rats.agents.base_agent import video_file_to_data_url

            return video_file_to_data_url(tmp_path)
        except Exception as e:
            logger.warning(f"  visual verifier video encode failed; using frames: {e}")
            return None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _safe_float(value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _truthy_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _json_default(value: Any) -> Any:
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return {
                    "type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
        except Exception:
            pass
        if isinstance(value, Path):
            return str(value)
        return str(value)
