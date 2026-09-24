"""Skill Proposer: propose new helper skills (treated as primitive-like tools) from failure patterns.

Complements FeedbackGenerator (which only extracts skills on SUCCESS). The Skill
Proposer is invoked after iterations that *failed*, when the failure memory has
accumulated enough episodes/lessons to suggest a missing capability. It asks the
LLM to write a new helper function whose body composes existing primitives, and
returns a skill dict ready to be inserted into the library.

Crucially, this is the path by which RATS can introduce *new* tools at runtime —
not just remix the ones that already exist in the seed library or extract the
exact code that solved a past task. Proposed skills:
  - get `is_primitive: false` (they're code, not external API wrappers)
  - get `source_task: 'proposed_from_failures'` for provenance
  - go through SkillLibrary.add_skill which de-duplicates against existing skills

Trigger policy lives in lifelong_loop; this file just exposes the agent.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from rats.agents.base_agent import query_llm_json

logger = logging.getLogger("rats.skill_proposer")


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """Cut ``text`` to at most ``max_chars`` characters at the latest
    natural section boundary at or below the limit.

    Tries each separator in order from most-coarse to most-fine:
      - ``"\\n\\n"`` (paragraph / function-block boundary in combined_doc
        output and section breaks in failure_summary)
      - ``"\\n--- "`` (markdown ``--- TITLE ---`` section dividers used
        by failure_memory.retrieve_for_policy_writer)
      - ``"\\n- "`` (list-item boundary in learned_summary and
        DISTILLED LESSONS lists)
      - ``"\\n"`` (line boundary as last structural option)

    Falls back to a raw character cut only when none of those separators
    exists below the limit — that is, when the input is one
    unbroken blob. The point is to avoid decapitating a function
    signature, a JSON block, or a markdown bullet mid-stream so the
    downstream LLM never sees half an API entry.
    """
    if not text or len(text) <= max_chars:
        return text
    suffix = "\n# ... (truncated at section boundary; not all content shown)"
    # Find the latest natural boundary at-or-below the char limit; try
    # coarsest -> finest. Then add the suffix only if it fits without
    # pushing past max_chars; otherwise return the boundary-cut text
    # alone. This keeps the cut clean even when max_chars is tight.
    for sep in ("\n\n", "\n--- ", "\n- ", "\n"):
        cut = text.rfind(sep, 0, max_chars)
        if cut > 0:
            trimmed = text[:cut]
            if len(trimmed) + len(suffix) <= max_chars:
                return trimmed + suffix
            return trimmed
    return text[:max_chars]


SYSTEM_PROMPT = (
    "You are a robotics skill proposer. Given a list of repeated failure modes "
    "from past task attempts and the current set of available primitives + learned "
    "skills, propose 0-2 NEW helper functions that, if added to the library, would "
    "let future attempts avoid these failures. Each helper must be pure Python that "
    "calls only the listed primitives/helpers — no new external imports beyond numpy. "
    "\n\n"
    "CRITICAL API CONSTRAINTS (proposals that violate these will be rejected and fail at runtime):\n"
    "- get_observation() returns a dict with EXACTLY these keys:\n"
    "    obs['agentview']['images']['rgb']              # HxWx3 uint8\n"
    "    obs['agentview']['images']['depth']            # HxW float (meters)\n"
    "    obs['robot0_eye_in_hand']['images']['rgb']     # wrist RGB\n"
    "    obs['robot0_eye_in_hand']['images']['depth']   # wrist depth\n"
    "    obs['robot_joint_pos']                         # ndarray(8,)\n"
    "    obs['robot_cartesian_pos']                     # ndarray(8,) xyz+wxyz+gripper\n"
    "- obs['agentview'] has NO 'intrinsics', NO 'pose_mat', NO 'extrinsics', NO 'K', NO 'T'. "
    "Camera calibration is NOT exposed. Do NOT attempt pixel↔world projection via camera matrices — "
    "it will crash with KeyError. Use get_object_pose(name) for world-frame positions directly.\n"
    "- To read depth: `obs['agentview']['images']['depth']` (NOT `obs['depth']` or `cam['depth']`).\n"
    "- No cv2, no PIL, no torch. numpy only.\n"
    "\nRespond ONLY with valid JSON."
)


USER_TEMPLATE = """\
RECENT FAILURE PATTERN (distilled lessons + most-relevant raw episodes):
{failure_summary}

CURRENT PRIMITIVES (callable from generated code; signatures only):
{primitive_list}

CURRENT LEARNED SKILLS (names/descriptions; do not redefine these):
{learned_summary}

TASK: propose 0, 1, or 2 NEW helper functions that would prevent the failures
above. Each helper should be a small, focused function (10-40 lines) that
composes the primitives or existing learned skills. DO NOT propose a helper that
duplicates an existing skill. If no new helper would help (e.g. failures are
purely perception bugs that no Python composition can fix), return an empty list.

Output JSON of the form:
{{
  "proposed_skills": [
    {{
      "name": "snake_case_function_name",
      "description": "One sentence describing when/why to use this helper.",
      "code": "def snake_case_function_name(...):\\n    ...\\n    return ...",
      "api_primitives_used": ["primitive_name", ...],
      "preconditions": ["..."],
      "effects": ["..."],
      "rationale": "Which failure(s) this helper addresses and how."
    }},
    ...
  ]
}}
"""


class SkillProposer:
    """LLM-driven proposer of new helper skills from observed failure patterns."""

    def __init__(self, max_proposed_per_call: int = 2) -> None:
        self.max_proposed = max_proposed_per_call

    def propose(
        self,
        failure_summary: str,
        primitive_list: str,
        learned_skills: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Ask the LLM for 0-N new skill proposals.

        Args:
            failure_summary: Text summary of recent failures (typically the same
                string returned by FailureMemory.get_lessons_for_planner / a slice
                of retrieve_for_policy_writer).
            primitive_list: API docs of available primitives (same format the
                planner sees in scene_context.api_docs).
            learned_skills: Current learned skills (so the proposer doesn't
                duplicate). List of dicts with at least "name" and "description".

        Returns:
            List of skill dicts, each with name/description/code fields. Empty if
            the LLM judges that no new helper would help, or if parsing failed.
        """
        if not failure_summary.strip():
            return []

        learned_summary = "(none)"
        if learned_skills:
            entries = []
            # Per-skill code cap. Typical learned skills are 900–2200 chars
            # (see skill_library schema); a 2400 cap keeps most skills full
            # while truncating runaway outliers so the prompt stays bounded
            # at ~30 skills × ~2400 chars ≈ 18k tokens of skill code total.
            code_cap = 2400
            for s in learned_skills[:30]:  # cap to keep prompt bounded
                name = s.get("name", "")
                if not name:
                    continue
                desc = s.get("description", "") or ""
                # Render names/descriptions by default; if callers provide code
                # (some future paths may), include it as extra duplicate-avoidance
                # context without promising that code is always present.
                code = s.get("code") or s.get("source_code") or ""
                code = code.strip()
                if len(code) > code_cap:
                    code = code[:code_cap] + "\n# … (truncated)"
                header = f"### {name}"
                if desc:
                    header += f" — {desc}"
                if code:
                    entries.append(f"{header}\n```python\n{code}\n```")
                else:
                    entries.append(f"{header}\n(no code recorded)")
            learned_summary = "\n\n".join(entries) if entries else "(none)"

        # Boundary-aware truncation. The three inputs have natural section
        # separators — function blocks for primitive_list, lessons /
        # episodes for failure_summary, full lines for learned_summary.
        # Cutting at arbitrary char offsets risked decapitating a JSON
        # block, a function signature, or a markdown bullet mid-stream so
        # the proposer saw half an API. We now cut at the latest natural
        # boundary at or below the limit; if none exists, fall back to
        # char-cut as a safety net. Char budgets bumped to fit Gemini's
        # 65k input window — combined ≤ ~36k still leaves plenty of room.
        user_prompt = (
            USER_TEMPLATE
            .replace("{failure_summary}", _truncate_at_boundary(failure_summary, 12000))
            .replace("{primitive_list}", _truncate_at_boundary(primitive_list or "", 16000))
            .replace("{learned_summary}", _truncate_at_boundary(learned_summary, 8000))
        )

        try:
            result = query_llm_json(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.warning(f"SkillProposer LLM query failed: {e}")
            return []

        proposals = result.get("proposed_skills", []) or []
        if not isinstance(proposals, list):
            return []

        # AST gate. Apply the same perception-verify anti-pattern check
        # that ``policy_quality_checker`` uses to reject candidate skills
        # whose body binds a vlm_verify / verify_object_identity result
        # and never reads ``["verified"]``. Audit on the existing
        # learned-skill library (10 skills across v2/v3/v4) found 3
        # baked in that anti-pattern — they were proposed before the
        # gate landed on the policy_writer side, so subsequent "skill
        # reuse" was propagating the bug into new policy code. Gating
        # here prevents new bad skills from entering the library via the
        # SkillProposer (failure-driven proposal) path.
        from rats.agents.policy_quality_checker import PolicyQualityChecker

        out: list[dict[str, Any]] = []
        for p in proposals[: self.max_proposed]:
            if not isinstance(p, dict):
                continue
            name = p.get("name", "").strip()
            code = p.get("code", "").strip()
            if not name or not code or not code.startswith("def "):
                logger.info(f"SkillProposer rejected proposal '{name}' (no name or non-def code)")
                continue
            verify_issues = PolicyQualityChecker._check_perception_verify_consumed(code)
            if verify_issues:
                logger.warning(
                    f"SkillProposer rejected '{name}' for perception-verify "
                    f"anti-pattern: {verify_issues[0][:200]}"
                )
                continue
            out.append({
                "name": name,
                "description": p.get("description", ""),
                "code": code,
                "api_primitives_used": p.get("api_primitives_used", []) or [],
                "preconditions": p.get("preconditions", []) or [],
                "effects": p.get("effects", []) or [],
                "source_task": "proposed_from_failures",
                "is_primitive": False,
                "proposed": True,
                "rationale": p.get("rationale", ""),
            })
        return out
