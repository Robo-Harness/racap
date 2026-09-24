"""SubAgent: focused retry loop for a single sub-behavior.

Spawned by the outer lifelong loop when the FailureDiagnoser identifies
that the same sub-behavior keeps failing across main-task attempts and
can plausibly be practiced in isolation (env reset to initial state).
The SubAgent runs a short retry loop on a single-step "plan" derived
from the diagnoser's NL description of the sub-behavior, judges success
by the diagnoser's visual per-step verdict, and on success extracts the
code as a reusable skill for the main task's skill library.

Non-privileged: the SubAgent reuses the main pipeline's FailureDiagnoser
and Executor so it inherits the same non-priv input surface. It does
NOT consult verifier symbolic state, BDDL predicates, or simulator
ground truth. Success is judged from the visual filmstrip.

Diversity-vs-main-loop
----------------------
The whole point of spawning a sub-agent is to explore approaches the
main loop did not. Observed empirically: on the OpenAI direct path
``temperature`` is dropped (reasoning models only consume
``reasoning_effort``), and the ensemble "pick best" selector further
collapses variance. A generic "try something different" hint barely
moves the needle — reasoning models gravitate back to the same
primitive family.

What actually works is **prompt-level diversity with concrete prior
evidence**: every retry injects the full code + primitive set of every
earlier sub-agent attempt into the policy writer's ``subagent_directive``
slot (a HARD constraint, priority #0 in the writer template — NOT the
advisory ``failure_context`` lessons slot, which the template demotes
to priority #2/#3), together with an explicit directive that the next
attempt MUST use a different subset of the available primitives. The model can then see
which family to pivot away from without us hardcoding any task-specific
menu. Cross-attempt context (prior_attempts) is also threaded into the
diagnoser so its verdict accounts for what earlier attempts achieved.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import imageio
import numpy as np

from rats.agents.base_agent import query_llm_json
from rats.agents.policy_writer import PolicyWriter

logger = logging.getLogger("rats.subagent")


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len] or "subagent"


class SubAgent:
    """Run a focused retry loop to learn a single sub-behavior.

    Usage:
        sub_agent = SubAgent(max_retries=6, ensemble_n=3)
        result = sub_agent.run(
            subgoal="...NL description of what to achieve...",
            env=env,
            scene_context=scene_context,
            diagnoser=main_diagnoser,
            executor=main_executor,
            reset_env=lambda: env.reset(),
            video_dir=output_dir,
            video_tag="iter007_subagent0",
            parent_task_name="rd6_put_mug_on_tray",
        )
        if result["success"]:
            new_skill = result["skill"]   # ready to add_skill(...)
    """

    def __init__(
        self,
        max_retries: int = 6,
        ensemble_n: int = 1,
    ) -> None:
        self.max_retries = max_retries
        self.ensemble_n = ensemble_n
        # Dedicated policy writer. ensemble_n defaults to 1 because the
        # ensemble "pick best" path collapses variance on reasoning
        # models (empirically verified on the mw_mug_subagent_v5 run:
        # 6 attempts all used the same primitive set). The real
        # diversity lever is the prompt-level directive built from
        # prior attempts' code — see _build_sub_agent_directive.
        self._policy_writer = PolicyWriter(max_retries=1, ensemble_n=ensemble_n)

    def run(
        self,
        subgoal: str,
        env: Any,
        scene_context: dict[str, Any],
        *,
        diagnoser: Any,
        executor: Any,
        reset_env: Callable[[], None],
        video_dir: Path | None = None,
        video_tag: str = "subagent",
        parent_task_name: str | None = None,
        approach_directive: str = "",
    ) -> dict[str, Any]:
        """Retry the sub-behavior up to ``max_retries`` times.

        Each attempt: reset env → policy_writer writes code for a
        single-step plan → executor runs → diagnoser judges visually.
        Success = the (sole) VPS entry has ``visually_satisfied: true``.

        ``parent_task_name`` is preserved onto the extracted skill's
        ``source_task`` so provenance stays readable (the subgoal is a
        free-form NL string and makes a poor identifier).

        ``approach_directive`` is a one-line physical-strategy hint
        that pins this sub-agent to a specific approach (top-down vs.
        side grasp, Molmo-pointing vs. SAM3-text, etc.). Used by the
        parallel-sub-agent orchestrator to spawn N sub-agents on the
        same subgoal with N different approaches; each subprocess gets
        one directive. Inlined into the writer's ``subagent_directive``
        slot (priority #0, HARD constraint) on every attempt so the
        writer sees it from attempt 1 onward and doesn't drift into a
        sibling worker's strategy.

        Returns a dict with:
          - success: bool
          - code: str | None (the successful code)
          - attempts: int (how many tries were used)
          - skill: dict | None (extracted skill ready for the library)
          - history: list[dict]  # per-attempt record for traceability
        """
        plan = self._build_single_step_plan(subgoal)
        prior_attempts_buf: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        retry_feedback: dict[str, Any] | None = None

        for attempt in range(self.max_retries):
            logger.info(
                f"[subagent:{video_tag}] attempt {attempt + 1}/{self.max_retries} "
                f"targeting: {subgoal[:120]!r}"
            )

            reset_env()

            # On retries, push the writer toward a different approach
            # than what we have already tried. The sub-agent directive
            # is the policy writer's dedicated HARD-constraint slot
            # (priority #0 in prompts/policy_writer.txt PRIORITY ORDER) —
            # not the advisory ``failure_context`` lessons slot. Earlier
            # versions of this code routed the assigned approach through
            # ``failure_context``, which the writer template labels
            # "LESSONS FROM PAST FAILURES (advisory — NOT ground truth)"
            # and demotes to priority #2/#3, so the parallel-sub-agent
            # commit could silently get overridden by the diagnoser.
            history_directive = (
                self._build_sub_agent_directive(
                    history,
                    scene_context.get("available_functions", []) or [],
                )
                if history else ""
            )
            # Pin the assigned approach in front of the history-based
            # directive so the writer can never silently drift to a
            # sibling worker's strategy. Each parallel sub-agent
            # subprocess is supposed to commit to one approach; without
            # this, the writer sees only the prior attempts and may
            # converge on the same retry strategy as its siblings.
            if approach_directive:
                approach_block = (
                    "## SUB-AGENT DIRECTIVE (HARD — overrides all other guidance below)\n"
                    "\n"
                    "You are one of several parallel sub-agents practicing "
                    "the SAME subgoal with DIFFERENT physical strategies. "
                    "Stay committed to the strategy below. Sibling "
                    "sub-agents are exploring other strategies in parallel; "
                    "duplicating their approach defeats the parallelism.\n"
                    "\n"
                    f"  Strategy: {approach_directive}\n"
                )
                subagent_directive = (
                    approach_block + "\n" + history_directive
                    if history_directive else approach_block
                )
            elif history_directive:
                # No assigned approach (e.g. unit-test path) but we do
                # have a retry-history nudge — still a current-attempt
                # directive, not a past-failure lesson.
                subagent_directive = (
                    "## SUB-AGENT DIRECTIVE (HARD — overrides all other guidance below)\n"
                    "\n"
                    + history_directive
                )
            else:
                subagent_directive = ""

            code = self._policy_writer.write(
                plan, scene_context,
                retry_feedback=retry_feedback,
                failure_context="",
                subagent_directive=subagent_directive,
                success_context="",
            )

            # Enable per-attempt video capture on the low-level env.
            low = getattr(env, "low_level_env", env)
            if hasattr(low, "enable_video_capture"):
                try:
                    low.enable_video_capture(True, clear=True)
                except Exception:
                    pass

            execution_result = executor.execute(code, env, scene_context)

            # Save per-attempt video, just like the main retry loop
            # (so videos of sub-agent probes are inspectable alongside
            # main-iteration videos).  Sub-agent attempts are single-shot,
            # not multi-turn, so each attempt directory contains one
            # capx-style ``combined.mp4``:
            #
            #   iter002_subagent0_attempt3_failed/
            #       combined.mp4
            if hasattr(low, "get_video_frames") and video_dir is not None:
                try:
                    frames = low.get_video_frames(clear=True)
                    if frames and len(frames) > 1:
                        pending_dir = (
                            video_dir / f"{video_tag}_attempt{attempt}_pending"
                        )
                        if pending_dir.exists():
                            shutil.rmtree(pending_dir)
                        pending_dir.mkdir(parents=True, exist_ok=True)
                        path = pending_dir / "combined.mp4"
                        imageio.mimsave(str(path), frames, fps=20)
                        execution_result["video_path"] = str(path)
                        execution_result["video_dir"] = str(pending_dir)
                        logger.info(
                            f"[subagent:{video_tag}]   video: "
                            f"{pending_dir.name}/{path.name} "
                            f"({len(frames)} frames)"
                        )
                        # Sub-agent probes need spatial reasoning frames,
                        # not just reset+terminal. A pure before/after pair
                        # left the diagnoser saying "gripper missed the
                        # target" with no actionable delta — the user
                        # wanted "approach was 3cm too far in -x" instead.
                        # Anchor first + last (the reset / terminal cues
                        # the prompt asks for) and sample N-2 motion
                        # frames evenly in between so the diagnoser can
                        # see HOW the arm got there.
                        execution_result["trajectory_frames"] = (
                            self._sample_filmstrip(frames, k=6)
                        )
                        try:
                            from rats.utils.video_utils import _encode_video_base64

                            video_fps = max(
                                1,
                                int(os.environ.get("RATS_DIAGNOSER_VIDEO_FPS", "20") or 20),
                            )
                            execution_result["trajectory_video_data_url"] = (
                                _encode_video_base64(frames, fps=video_fps)
                            )
                            execution_result["trajectory_video_frame_count"] = len(frames)
                            execution_result["trajectory_video_fps"] = video_fps
                        except Exception as e:
                            logger.debug(
                                f"[subagent:{video_tag}]   video encode failed: {e}"
                            )
                except Exception as e:
                    logger.debug(f"[subagent:{video_tag}]   video save failed: {e}")

            # Two-tier diagnosis:
            #   Tier 1 — OUTCOME. Reset + terminal only, minimal prompt.
            #       Motion frames were biasing the outcome call (observed
            #       on iter007_subagent1_attempt5: the grasp succeeded
            #       but the combined prompt latched onto the descent-
            #       phase frames and called partial_completion).
            #   Tier 2 — CRITIQUE. Full filmstrip, only runs on Tier-1
            #       failure, produces the spatial delta for retry.
            # Build a reduced execution_result for Tier 1: keep wrist +
            # stdout/stderr but pass only reset+terminal as the filmstrip.
            full_filmstrip = execution_result.get("trajectory_frames") or []
            tier1_exec = dict(execution_result)
            if len(full_filmstrip) >= 2:
                tier1_exec["trajectory_frames"] = [
                    full_filmstrip[0], full_filmstrip[-1],
                ]
            for key in (
                "trajectory_video_data_url",
                "trajectory_video_frame_count",
                "trajectory_video_fps",
            ):
                tier1_exec.pop(key, None)

            tier1 = diagnoser.diagnose(
                tier1_exec,
                scene_context,
                plan=plan,
                code=code,
                goal_predicates=[],
                affordance_hints={},
                prior_attempts=None,  # Tier 1 is fresh per-attempt
                mode="subagent_outcome",
            )
            tier1_vps = tier1.get("visual_predicate_status") or []
            sub_success = bool(
                tier1_vps and tier1_vps[0].get("visually_satisfied")
            )
            video_path_str = execution_result.get("video_path")
            if video_path_str and video_dir is not None:
                try:
                    path = Path(video_path_str)
                    status = "succeeded" if sub_success else "failed"
                    final_dir = video_dir / f"{video_tag}_attempt{attempt}_{status}"
                    final_path = final_dir / "combined.mp4"
                    current_dir = path.parent
                    if current_dir != final_dir:
                        if final_dir.exists():
                            shutil.rmtree(final_dir)
                        if current_dir.exists() and path.name == "combined.mp4":
                            current_dir.replace(final_dir)
                        else:
                            final_dir.mkdir(parents=True, exist_ok=True)
                            path.replace(final_path)
                    execution_result["video_path"] = str(final_path)
                    execution_result["video_dir"] = str(final_dir)
                    logger.info(
                        f"[subagent:{video_tag}]   video outcome: "
                        f"{final_dir.name}/{final_path.name}"
                    )
                except Exception as e:
                    logger.debug(f"[subagent:{video_tag}]   video rename failed: {e}")
            tier1_evidence = (
                str(tier1_vps[0].get("evidence", "") or "")
                if tier1_vps else ""
            )

            if sub_success:
                # Happy path: skip Tier 2 entirely. Cheaper + the skill
                # gets extracted before any second-guessing can flip it.
                diag = tier1
                vps = tier1_vps
            else:
                # Tier 2: the critique call. Full filmstrip, cross-
                # attempt context, Tier-1 evidence inlined so the
                # critic can corroborate or flag disagreement.
                tier2 = diagnoser.diagnose(
                    execution_result,
                    scene_context,
                    plan=plan,
                    code=code,
                    goal_predicates=[],
                    affordance_hints={},
                    prior_attempts=(prior_attempts_buf[-4:] or None),
                    mode="subagent_critique",
                    tier1_evidence=tier1_evidence,
                )
                # Merge: outcome stays Tier-1's (False), but the VPS
                # evidence and everything else (policy_feedback,
                # failure_mode, ...) comes from Tier 2 where the
                # spatial reasoning happens. Preserve the Tier-1
                # visually_satisfied bit though — a rogue Tier-2 parse
                # setting it True would orphan us back into "success".
                tier2_vps = tier2.get("visual_predicate_status") or []
                if tier2_vps:
                    tier2_vps[0]["visually_satisfied"] = False
                    # Keep step_id = step-1 so downstream lookups work.
                    if not tier2_vps[0].get("step_id"):
                        tier2_vps[0]["step_id"] = "step-1"
                    vps = tier2_vps
                else:
                    vps = tier1_vps
                diag = dict(tier2)
                diag["visual_success"] = False
                diag["visual_predicate_status"] = vps
                if tier2.get("tier1_disagreement"):
                    logger.info(
                        f"[subagent:{video_tag}] tier2 disagrees with tier1 "
                        f"outcome — motion frames suggest success. "
                        f"Honoring tier1 verdict but logging disagreement."
                    )

            history.append({
                "attempt": attempt,
                "code": code,
                "success": sub_success,
                "evidence": (vps[0].get("evidence") if vps else "") or "",
                "failure_mode": diag.get("failure_mode"),
                "policy_feedback": diag.get("policy_feedback", ""),
                "video_path": execution_result.get("video_path"),
            })

            # Record this attempt for next diagnose()'s cross-attempt
            # context (mirrors lifelong_loop's own prior_attempts buffer).
            last_frame = None
            frames_now = execution_result.get("trajectory_frames") or []
            if frames_now:
                last_frame = frames_now[-1]
            prior_attempts_buf.append({
                "attempt_idx": attempt,
                "code": code,
                "policy_feedback": diag.get("policy_feedback", ""),
                "failure_mode": diag.get("failure_mode", ""),
                "visual_predicate_status": vps,
                "last_frame": last_frame,
            })

            if sub_success:
                logger.info(
                    f"[subagent:{video_tag}] SUCCESS on attempt {attempt + 1}"
                )
                # `_extract_skill` was previously called here, but
                # lifelong_loop.py:3156-3170 explicitly does NOT persist
                # subagent-extracted skills to the library (the previous
                # design produced wrappers that either hallucinated calls
                # or parameterised away the values that made the script
                # succeed). The main loop only reuses the verbatim winning
                # script for the current iteration's success_context.
                # Spending an LLM call to produce output nothing reads is
                # pure waste — three such calls in the LIBERO smoke run
                # cost ~30s with zero downstream effect.
                return {
                    "success": True,
                    "code": code,
                    "attempts": attempt + 1,
                    "skill": None,
                    "history": history,
                    "video_tag": video_tag,
                }

            # Prepare retry_feedback for the next attempt. Single-step
            # plan → mechanism-A preservation is vacuous; leave empty.
            retry_feedback = {
                "attempt": attempt + 1,
                "stderr": (execution_result.get("stderr", "") or "")[:3000],
                "diagnosis": diag.get("policy_feedback", ""),
                "failed_step": diag.get("failed_step", "step-1"),
                "failure_mode": diag.get("failure_mode", ""),
                "previous_code": code,
                "visual_predicate_status": vps,
                "preserved_code_segments": [],
                "replan": False,
                "plan_issue_reason": "",
                "subagent_skill_target": None,
                "subagent_skill_target_reason": "",
                "diagnostic_context": diag.get("diagnostic_context") or {},
                "edit_scale": diag.get("edit_scale"),
            }

        logger.info(
            f"[subagent:{video_tag}] exhausted {self.max_retries} attempts; no success"
        )
        return {
            "success": False,
            "code": None,
            "attempts": self.max_retries,
            "skill": None,
            "history": history,
            "video_tag": video_tag,
        }

    @staticmethod
    def _sample_filmstrip(frames: list[Any], k: int = 6) -> list[Any]:
        """Pick ``k`` frames evenly across ``frames`` with first + last anchored.

        The diagnoser needs both anchors (reset / terminal cues are encoded
        in the prompt) AND motion frames in between to reason about
        approach direction and timing. Even sampling avoids over-weighting
        the descent phase that triggered the v8 iter1 false-failure
        (mug lifted in frame 115 but diagnoser anchored on frames 0-98).
        """
        if not frames:
            return []
        n = len(frames)
        if n <= k:
            return list(frames)
        # Pick k integer indices spanning [0, n-1] with both endpoints.
        step = (n - 1) / (k - 1)
        idxs = sorted({int(round(i * step)) for i in range(k)})
        # Round-tripping through a set can drop a duplicate when n is
        # small relative to k; backfill with adjacent unique indices so
        # the caller still gets a stable count.
        while len(idxs) < k:
            for cand in range(n):
                if cand not in idxs:
                    idxs.append(cand)
                    break
            idxs.sort()
        return [frames[i] for i in idxs[:k]]

    @staticmethod
    def _build_single_step_plan(subgoal: str) -> dict[str, Any]:
        return {
            "task_id": f"subagent:{_slug(subgoal)}",
            "steps": [
                {
                    "id": "step-1",
                    "description": subgoal,
                    "relevant_skills": [],
                    "selected_skill_details": [],
                    "skill_code": "",
                    "new_skill_needed": True,
                    "notes": (
                        "Focused sub-agent attempt. Env has been reset to "
                        "its initial state. Write code to achieve ONLY the "
                        "subgoal above; do not attempt the broader task. "
                        "You are encouraged to try approaches the main "
                        "loop would not — this is an exploration probe."
                    ),
                }
            ],
            "all_selected_skill_names": [],
        }

    @staticmethod
    def _extract_used_primitives(
        code: str, available_functions: list[str],
    ) -> list[str]:
        """Return the subset of ``available_functions`` called in ``code``.

        Uses a word-boundary regex so substring collisions (e.g.
        ``close_gripper`` inside ``verified_close_gripper``) don't
        inflate the set. Agent-public info only: we're inspecting code
        the agent itself wrote.
        """
        if not code or not available_functions:
            return []
        used: set[str] = set()
        for fn in available_functions:
            if not fn:
                continue
            if re.search(rf"(?<![\w.]){re.escape(fn)}\s*\(", code):
                used.add(fn)
        return sorted(used)

    @classmethod
    def _build_sub_agent_directive(
        cls,
        history: list[dict[str, Any]],
        available_functions: list[str],
    ) -> str:
        """Build the prompt block that drives per-attempt diversity.

        The core signal is the **diagnoser's physical observation** for
        each prior attempt — the prose it wrote describing what the arm
        actually did and why it didn't achieve the subgoal. That prose
        is the richest information available about what went wrong; we
        inline it verbatim (no truncation of the critique itself) and
        ask the writer to respond to the observation rather than treat
        it as a vague failure report.

        We also inline each prior attempt's code + primitive set so the
        writer sees concretely what has been tried — but we do NOT
        prescribe which primitives to use next or hard-code a fix. The
        diagnoser is the one that identified the physical problem;
        the writer decides how to address it.

        Kept under ~4 prior attempts and ~3000 chars per code block to
        stay within policy_writer's context budget.
        """
        lines = [
            "## SUB-AGENT EXPLORATION DIRECTIVE",
            "",
            "You are in a focused sub-agent retry loop on a single sub-behavior.",
            "Each attempt below was a full, fresh try at the subgoal from the",
            "same reset initial state, and they all failed.",
            "",
            "The richest signal you have is the diagnoser's OBSERVATION of each",
            "attempt — the prose below each attempt describes what the arm",
            "physically did and why it did not achieve the subgoal. Read those",
            "observations carefully and make sure your next attempt addresses",
            "the specific physical issue the diagnoser identified. Repeating",
            "the same physical approach with slightly different tooling will",
            "reproduce the same failure.",
            "",
            "Prior sub-agent attempts (most recent last):",
            "",
        ]
        # Show up to 4 most recent attempts, but DROP the code block for
        # the most recent one — its full code already appears in the
        # policy_writer's separate PREVIOUS CODE section. Repeating it
        # here just added 25–40 duplicated lines per retry prompt (observed
        # 2-3x duplication in the LIBERO smoke run). Keep the diagnoser
        # critique for the most recent attempt — that's the part the
        # writer needs to respond to.
        window = history[-4:]
        last_idx = len(window) - 1
        for i, h in enumerate(window):
            code = (h.get("code") or "").strip()
            mode = (h.get("failure_mode") or "").strip() or "unknown"
            # Diagnoser evidence + full policy_feedback are the physical
            # critique — inline untruncated (they are the point).
            ev = (h.get("evidence") or "").strip()
            pf = (h.get("policy_feedback") or "").strip()
            prims = cls._extract_used_primitives(code, available_functions)
            lines.append(f"--- Attempt {h['attempt']} ---")
            lines.append(f"failure_mode: {mode}")
            if ev:
                lines.append("diagnoser observation (what the arm actually did):")
                lines.append(ev)
            if pf and pf != ev:
                lines.append("diagnoser critique / guidance for next attempt:")
                lines.append(pf)
            lines.append(f"primitives called: {prims or '(none detected)'}")
            if i == last_idx:
                lines.append("code: (see PREVIOUS CODE block below — same as most-recent attempt)")
            else:
                # FIX: removed the prior `code[:3000]` slice — same pattern
                # as the failure_memory truncation cleanup. The writer
                # needs to see what's already been tried in full;
                # truncating mid-policy hides exactly the pattern we want
                # it to avoid. history[-4:] already caps at 4 attempts and
                # `policy_writer.py:455` warns when single-policy length
                # exceeds 80 lines, so the total stays bounded by those
                # upstream limits rather than an arbitrary char count
                # here.
                lines.append("code:")
                lines.append("```python")
                lines.append(code)
                lines.append("```")
            lines.append("")
        lines.append(
            "When you write the next attempt, treat the diagnoser observations "
            "above as physical facts you must respond to. If every prior attempt "
            "failed the same way, that pattern is telling you something specific "
            "about the scene or the interaction — adapt accordingly."
        )
        lines.append("")
        return "\n".join(lines)

    def _extract_skill(
        self,
        code: str,
        subgoal: str,
        scene_context: dict[str, Any],
        *,
        parent_task_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Wrap a successful sub-agent code blob into a reusable skill.

        Uses a dedicated LLM prompt (``prompts/subagent_extract_skill.txt``)
        to turn the ad-hoc script into a named function with a real
        docstring, parameters, and pre/post conditions.

        Seeds ``usage_count=1`` / ``success_count=1`` so the tier system
        has one empirical positive observation on day one (Wilson lower
        bound ≈ 0.21 instead of 0), reflecting the fact that the sub-agent
        itself just proved the behavior works from reset state.

        Returns a skill dict ready for ``SkillLibrary.add_skill`` or None
        on extraction failure.
        """
        if not code.strip():
            return None
        try:
            prompt_path = Path("rats/prompts/subagent_extract_skill.txt")
            template = prompt_path.read_text()
        except OSError as e:
            logger.warning(f"[subagent] skill extraction prompt missing: {e}")
            return None

        user_prompt = template.replace(
            "{subgoal}", subgoal
        ).replace(
            "{api_docs}", scene_context.get("api_docs", "") or ""
        ).replace(
            "{code}", code
        )
        system_prompt = (
            "You turn robot control scripts into named reusable skill "
            "functions with real docstrings. Respond only in valid JSON."
        )

        try:
            parsed = query_llm_json(system_prompt, user_prompt)
        except Exception as e:
            logger.warning(f"[subagent] skill extraction LLM call failed: {e}")
            return None

        name = str(parsed.get("name", "") or "").strip()
        wrapped = str(parsed.get("code", "") or "").strip()
        description = str(parsed.get("description", "") or subgoal).strip()
        if not name or not wrapped:
            logger.warning("[subagent] skill extraction returned incomplete JSON")
            return None

        source = (
            f"subagent:{parent_task_name}"
            if parent_task_name else
            f"subagent:{_slug(subgoal)}"
        )
        # Preserve the sub-agent's RAW successful script. The parameterized
        # wrapper above is useful for cross-scene reuse, but in practice the
        # main agent regularly guesses the parameter values wrong (SAM3
        # prompt strings, pull directions) and crashes. Keeping the
        # verbatim working invocation lets the main agent copy the exact
        # args that succeeded when the scene matches. Truncation is soft
        # — 4000 chars covers every sub-agent script observed so far.
        example_code = (code or "").strip()
        # if len(example_code) > 4000:
        #     example_code = example_code[:4000] + "\n# ... (truncated)"

        return {
            "name": name,
            "description": description,
            "code": wrapped,
            "api_primitives_used": list(parsed.get("api_primitives_used", []) or []),
            "preconditions": list(parsed.get("preconditions", []) or []),
            "effects": list(parsed.get("effects", []) or []),
            "source_task": source,
            # Raw successful script from the sub-agent that produced this
            # skill. Surfaced to downstream callers as a "verified working
            # invocation, copy these args if the scene matches" reference.
            "example_code": example_code,
            # One empirical positive: the sub-agent proved this works
            # from reset state just now. Wilson lower-bound ≈ 0.21 puts
            # the skill ahead of never-used siblings in planner ranking.
            "usage_count": 1,
            "success_count": 1,
            "success_rate": 1.0,
        }
