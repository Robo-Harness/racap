"""Lifelong Loop: main orchestrator wiring all RATS components.

Flow per iteration:
  Environment Creator -> Task Proposer -> Planner -> Policy Writer ->
  Quality Checker -> Executor -> Verifier + Failure Diagnoser ->
  Feedback Generator -> (Skill Library + back to top)
"""

from __future__ import annotations

import ast
import base64
import inspect
import json
import logging
import os
import re
import sys
import time
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rats.utils.json_safety import json_safe as _json_safe_value
from rats.control_flow import RATSEpisodeWallTimeExceeded
from rats.deadline import (
    arm_episode_wall_deadline,
    restore_episode_wall_deadline,
)


_STEP_INDEX_RE = re.compile(r"\d+")


def resolve_libero_episode_key(
    scene_context: dict[str, Any], iteration: int, seed_offset: int
) -> str:
    """Return the public telemetry key without changing the internal trial seed.

    Controlled evaluation registers an exact public key because RATS uses
    1-based trial seeds internally while benchmark manifests use 0-based
    public seeds.  Development runs without a registered key retain RATS's
    ordinary per-iteration naming.
    """

    registered = os.getenv("RATS_REGISTERED_EPISODE_KEY", "").strip()
    if registered:
        return registered
    return (
        f"{scene_context.get('suite_name', 'libero')}/"
        f"{scene_context.get('task_id', 0)}/"
        f"seed{iteration + seed_offset}"
    )


def _step_index(step_id: str) -> str:
    """Extract the numeric index from a step id variant like ``step-3``."""
    if not step_id:
        return ""
    m = _STEP_INDEX_RE.search(step_id)
    return m.group(0) if m else ""


def _extract_step_segment(code: str, step_id: str) -> str:
    """Extract the code block corresponding to a planner step id."""
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


def _extract_called_learned_skills(
    code: str, learned_skill_names: set[str],
) -> list[str]:
    """Find learned-skill function names actually called in the executed code.

    Matches ``name(`` with a word boundary before the name so sub-strings
    like ``close_gripper`` inside ``verified_close_gripper`` don't collide.
    Only returns names present in ``learned_skill_names`` (primitives and
    arbitrary tokens are filtered out). Order is insertion-order of first
    appearance so the lifelong-loop can log a deterministic list.
    """
    if not code or not learned_skill_names:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"(?<![\w.])([A-Za-z_][A-Za-z_0-9]*)\s*\(", code):
        name = match.group(1)
        if name in learned_skill_names and name not in seen:
            seen.add(name)
            found.append(name)
    return found


def _extract_local_function_defs(
    code: str,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map top-level local helper names to their AST definitions."""
    if not code:
        return {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = node
    return defs


def _ordered_call_names_from_statements(statements: list[ast.stmt]) -> list[str]:
    """Collect direct function-call names without descending into nested defs."""
    calls: list[str] = []

    class _Collector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            fn = node.func
            if isinstance(fn, ast.Name):
                calls.append(fn.id)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

    collector = _Collector()
    for stmt in statements:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        collector.visit(stmt)
    return calls


def _collect_reachable_learned_skills(
    statements: list[ast.stmt],
    helper_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    learned_skill_names: set[str],
) -> list[str]:
    """Follow local helper calls and return reachable learned-skill names."""
    queue = list(_ordered_call_names_from_statements(statements))
    visited_helpers: set[str] = set()
    seen_skills: set[str] = set()
    found: list[str] = []
    while queue:
        name = queue.pop(0)
        if name in learned_skill_names:
            if name not in seen_skills:
                seen_skills.add(name)
                found.append(name)
            continue
        helper = helper_defs.get(name)
        if helper is None or name in visited_helpers:
            continue
        visited_helpers.add(name)
        queue.extend(_ordered_call_names_from_statements(helper.body))
    return found


def _extract_reachable_learned_skills(
    code: str, learned_skill_names: set[str],
) -> list[str]:
    """Return learned skills reachable from top-level executable code."""
    if not code or not learned_skill_names:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _extract_called_learned_skills(code, learned_skill_names)
    return _collect_reachable_learned_skills(
        tree.body,
        _extract_local_function_defs(code),
        learned_skill_names,
    )


def _extract_step_reachable_learned_skills(
    code: str, step_id: str, learned_skill_names: set[str],
) -> list[str]:
    """Return learned skills reachable from the given failed plan step."""
    if not code or not step_id or not learned_skill_names:
        return []
    segment = _extract_step_segment(code, step_id)
    if not segment:
        return []
    segment = textwrap.dedent(segment)
    try:
        tree = ast.parse(segment)
    except SyntaxError:
        return _extract_called_learned_skills(segment, learned_skill_names)
    return _collect_reachable_learned_skills(
        tree.body,
        _extract_local_function_defs(code),
        learned_skill_names,
    )


def _diagnosis_claims_no_remaining_work(diagnosis: dict[str, Any]) -> bool:
    """Return true when the visual diagnoser effectively told retry to no-op."""
    feedback = str(diagnosis.get("policy_feedback", "") or "").lower()
    failure_mode = str(diagnosis.get("failure_mode", "") or "").lower()
    feedback_says_noop = (
        "no corrective action needed" in feedback
        or "no corrective action required" in feedback
        or "no further action needed" in feedback
        or "no further action required" in feedback
    )
    return (
        bool(diagnosis.get("visual_success"))
        or failure_mode == "none"
        or feedback_says_noop
    )


def _verifier_diagnoser_mismatch(
    verification: dict[str, Any],
    diagnosis: dict[str, Any],
) -> bool:
    """Detect verifier-failed / diagnoser-no-op contradiction."""
    return (
        not bool(verification.get("success"))
        and _diagnosis_claims_no_remaining_work(diagnosis)
    )


def _build_verifier_challenge(
    verification: dict[str, Any],
    diagnosis: dict[str, Any],
) -> str:
    """Build a non-privileged challenge for a second diagnoser pass.

    Keep this intentionally high-level: the verifier may have access to
    exact symbolic predicates, and the policy/diagnosis path for these runs
    should not receive those predicate internals. The useful information is
    the binary disagreement: retry is happening because independent
    verification did not accept success.
    """
    _ = verification
    prior_feedback = str(diagnosis.get("policy_feedback", "") or "").strip()
    if prior_feedback:
        prior_feedback = f" Previous diagnosis feedback was: {prior_feedback[:400]}"
    return (
        "The task verifier did not accept the previous turn as successful. "
        "The control loop is retrying because completion was not accepted, "
        "so identify a concrete remaining corrective action from the visual "
        "state rather than recommending a no-op/RESULT-only turn."
        f"{prior_feedback}"
    )


def _force_mismatch_retry_diagnosis(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Conservative fallback if a challenged diagnosis still says 'done'."""
    forced = dict(diagnosis)
    forced.update({
        "visual_success": False,
        "failed_step": forced.get("failed_step") or "verification",
        "failure_reason": (
            "Verifier rejected completion while diagnosis reported no "
            "remaining corrective action."
        ),
        "policy_feedback": (
            "The task was not accepted as complete. Do not return a no-op "
            "or RESULT-only turn. Re-inspect the current scene and execute "
            "the smallest concrete corrective action toward the goal; focus "
            "on the final success-critical substep (secure/lift/place/"
            "actuate as appropriate)."
        ),
        "failure_mode": (
            forced.get("failure_mode")
            if str(forced.get("failure_mode", "") or "").lower() not in ("", "none")
            else "partial_completion"
        ),
        "verifier_challenge_forced": True,
    })
    try:
        forced["confidence"] = max(float(forced.get("confidence", 0.0) or 0.0), 0.7)
    except Exception:
        forced["confidence"] = 0.7
    return forced


def _trajectory_filmstrip_sample_count(frame_count: int) -> int:
    """Length-scaled visual trajectory budget for failure diagnosis.

    Long videos and timeouts should not collapse to the same eight images as a
    short attempt. Defaults are environment-overridable for local VLM runs:
      - RATS_TRAJECTORY_ALL_FRAMES_UP_TO: pass every frame up to this length.
      - RATS_TRAJECTORY_MIN_FRAMES: minimum sampled count for longer videos.
      - RATS_TRAJECTORY_FRAME_STRIDE: roughly one selected frame per N frames.
      - RATS_TRAJECTORY_MAX_FRAMES: safety cap for latency/context size.
    """
    if frame_count <= 0:
        return 0

    all_frames_up_to = max(
        1,
        int(os.environ.get("RATS_TRAJECTORY_ALL_FRAMES_UP_TO", "16") or 16),
    )
    if frame_count <= all_frames_up_to:
        return frame_count

    min_frames = max(
        1,
        int(os.environ.get("RATS_TRAJECTORY_MIN_FRAMES", "8") or 8),
    )
    stride = max(
        1,
        int(os.environ.get("RATS_TRAJECTORY_FRAME_STRIDE", "30") or 30),
    )
    max_frames = max(
        min_frames,
        int(os.environ.get("RATS_TRAJECTORY_MAX_FRAMES", "96") or 96),
    )
    scaled = (frame_count + stride - 1) // stride
    return min(frame_count, max_frames, max(min_frames, scaled))


def _sample_trajectory_frames(frames: list[Any]) -> list[Any]:
    """Uniformly sample a length-scaled filmstrip, always preserving endpoints."""
    frame_count = len(frames)
    target = _trajectory_filmstrip_sample_count(frame_count)
    if target <= 0:
        return []
    if target >= frame_count:
        return list(frames)
    if target == 1:
        return [frames[-1]]

    indices = [
        int(round(i * (frame_count - 1) / (target - 1)))
        for i in range(target)
    ]
    deduped: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        idx = max(0, min(frame_count - 1, idx))
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    if deduped[0] != 0:
        deduped.insert(0, 0)
    if deduped[-1] != frame_count - 1:
        deduped.append(frame_count - 1)
    if len(deduped) > target:
        deduped = deduped[: target - 1] + [frame_count - 1]
    return [frames[i] for i in deduped]


def _sample_vlm_verifier_frames(
    frames: list[Any],
    *,
    min_frames: int = 8,
    max_frames: int = 48,
) -> list[Any]:
    """Uniformly sample VLM verifier frames with a hard 8..48 default budget."""
    frame_count = len(frames)
    if frame_count <= 0:
        return []
    min_frames = max(1, int(min_frames or 8))
    max_frames = max(min_frames, int(max_frames or 48))
    if frame_count <= min_frames:
        return list(frames)
    target = min(frame_count, max_frames, max(min_frames, (frame_count + 29) // 30))
    if target >= frame_count:
        return list(frames)
    if target == 1:
        return [frames[-1]]

    indices = [
        int(round(i * (frame_count - 1) / (target - 1)))
        for i in range(target)
    ]
    deduped: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        idx = max(0, min(frame_count - 1, idx))
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    if deduped[0] != 0:
        deduped.insert(0, 0)
    if deduped[-1] != frame_count - 1:
        deduped.append(frame_count - 1)
    if len(deduped) > target:
        deduped = deduped[: target - 1] + [frame_count - 1]
    return [frames[i] for i in deduped]

from rats.agents.failure_diagnoser import FailureDiagnoser
from rats.agents.feedback_generator import FeedbackGenerator
from rats.agents.planner import Planner
from rats.agents.policy_primitive_cache import scope as policy_primitive_cache_scope
from rats.agents.policy_quality_checker import PolicyQualityChecker
from rats.agents.policy_writer import PolicyWriter
from rats.agents.environment_creator import EnvironmentCreator
from rats.agents.memory_curator import MemoryCurator
from rats.agents.multi_turn_decider import MultiTurnDecider, resolve_public_task_goal
from rats.agents.skill_proposer import SkillProposer
from rats.agents.subagent import SubAgent
from rats.agents.task_proposer import TaskProposer
from rats.agents.verifier import Verifier
from rats.agents.environment_verifier import MolmoEnvironmentVerifier
from rats.agents.per_step_verifier import PerStepVerifier
from rats.agents.planner_verifier import PlannerVerifier
from rats.evaluation.metrics import MetricsTracker
from rats.executor.sandbox import Executor
from rats.loop.libero_utils import (
    detect_env_type,
    extract_libero_scene_context,
    parse_libero_activity_name,
    recreate_libero_env,
)
from rats.loop.molmospaces_utils import (
    extract_molmospaces_scene_context,
    recreate_molmospaces_env,
)
from rats.loop.retry_bank import RetryBank
from rats.loop.task_queue import TaskQueue  # deprecated; retained for resume compatibility
from rats.memory.failure_memory import FailureMemory
from rats.memory.playtime_memory import PlaytimeMemory
from skill_library.library import SkillLibrary

logger = logging.getLogger("rats.lifelong_loop")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_subagent_config() -> dict[str, Any]:
    """Read the ``subagent:`` block from rats/config/default.yaml.

    Returns an empty dict if the file or key is missing — callers pass
    that through to ``SubAgent(...)`` which has its own hardcoded
    fallbacks, so missing config is always safe.
    """
    cfg_path = _PROJECT_ROOT / "rats" / "config" / "default.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        with cfg_path.open() as f:
            data = yaml.safe_load(f) or {}
        return data.get("subagent", {}) or {}
    except Exception:
        return {}


def _merge_molmospaces_config(
    base: dict[str, Any],
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge MolmoSpaces proposer config, preserving nested defaults."""
    merged = dict(base)
    override = override or {}
    merged.update(override)
    merged["vlm_grounding"] = {
        **(base.get("vlm_grounding") or {}),
        **(override.get("vlm_grounding") or {}),
    }
    merged["playtime"] = {
        **(base.get("playtime") or {}),
        **(override.get("playtime") or {}),
    }
    merged["environment_verifier"] = {
        **(base.get("environment_verifier") or {}),
        **(override.get("environment_verifier") or {}),
    }
    merged["per_step_verifier"] = {
        **(base.get("per_step_verifier") or {}),
        **(override.get("per_step_verifier") or {}),
    }
    merged["feedback_generator"] = {
        **(base.get("feedback_generator") or {}),
        **(override.get("feedback_generator") or {}),
    }
    return merged


def _normalize_molmospaces_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply cross-field defaults after YAML merging."""
    out = dict(cfg)
    # `curiosity: true` is the user-facing switch for Piaget playtime mode.
    # Keep existing explicit modes intact; only upgrade the safe catalog
    # default to playtime when curiosity is enabled.
    if bool(out.get("curiosity")) and str(out.get("proposer_mode", "catalog")) in {"", "catalog", "open"}:
        out["proposer_mode"] = "playtime"
    return out


def _env_bool_override(name: str) -> bool | None:
    """Parse optional boolean env override; None means no override."""
    raw = os.getenv(name)
    if raw is None:
        return None
    val = str(raw).strip().lower()
    if val == "":
        return None
    return val not in {"0", "false", "no", "off"}


def _load_molmospaces_config(
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read MolmoSpaces proposer config with optional env-YAML overrides.

    Drives the open-mode novel proposer: which mode to use, whether
    house-switching is allowed, the per-run rate limit, and the VLM
    grounder toggle. Base precedence is:

      built-in safe defaults < rats/config/default.yaml::molmospaces < env config::molmospaces

    Passing the env config override lets a run YAML select open/catalog
    proposer mode without editing the repository-global default.
    """
    cfg_path = _PROJECT_ROOT / "rats" / "config" / "default.yaml"
    defaults: dict[str, Any] = {
        "proposer_mode": "catalog",
        "allow_house_switching": True,
        "house_switch_max_per_run": 5,
        # Force-rotate to a new house every N iterations, regardless of
        # what the LLM proposer asked for. 0 = disabled (legacy
        # behavior; the LLM is the only switch trigger). Useful when
        # the proposer is too sticky on a single scene — observed in
        # both playtime and curriculum runs where 50 iters all stayed
        # on house_index=0. Forced switches do NOT consume the
        # `house_switch_max_per_run` budget that gates LLM-driven
        # switches; they're an administrative knob.
        "house_switch_every": 0,
        # Bridge-backed open proposer task types. `nav` exists in the
        # proposer schema as a future concept, but the current MolmoSpaces
        # bridge only instantiates these four task families.
        "allowed_task_types": ["pick", "pick_and_place", "open", "close"],
        # Optional smoke-test override. When non-empty, the proposer cycles
        # through this exact sequence instead of waiting for curriculum
        # successes to unlock later task types.
        "forced_task_type_sequence": [],
        "vlm_grounding": {
            "enabled": True,
            "cache_per_house": True,
            # Keep visibility grounded in the actual agentview/wrist VLM
            # judgment by default. The geometric frustum veto can false-hide
            # large articulated targets when their object center is outside
            # frame even though the drawer/front is visible.
            "geometric_visibility_gate": False,
        },
        "curiosity": False,
        "playtime": {
            "developmental_stage": "sensorimotor",
            "verifier": "vlm",
            "freeform_proposal": False,
            "verify_with_vlm_only": False,
            "stateful_checker_enabled": False,
            "stateful_checker_timeout_seconds": 2.0,
            "final_vlm_on_stateful_fail": False,
            "final_vlm_on_stateful_invalid": False,
            "vlm_verifier_model": "openai/gpt-5.5",
            "vlm_verifier_min_frames": 8,
            "vlm_verifier_max_frames": 48,
            "vlm_verifier_diagnostic_image_limit": 6,
            "record_resulting_state": True,
            "memory_path": "playtime_memory.jsonl",
            "max_recent_memories": 20,
            # Number of diverse task candidates to ask the proposer for before
            # Piaget curiosity ranking. Legacy alias: sample_k.
            "proposal_count": 5,
            # When true, playtime proposal validation only accepts targets that
            # are articulated objects / articulation handles / current open-close
            # benchmark targets. This is a hard target filter; secondaries may
            # still be ordinary support/containment objects.
            "focus_articulated_objects": False,
            # When true, the playtime proposer treats the generic inventory
            # VLM's per-item visible=false labels as a hard target veto and
            # splits the prompt inventory into visible/out_of_view blocks.
            # Default off: target visibility is checked by the narrower
            # MolmoSpaces environment verifier after rebind/reset, which is
            # better aligned with the frame the policy will actually use.
            "inventory_visibility_gate": False,
            "allowed_interactions": [
                "touch", "tap", "push", "pull", "slide", "roll", "lift",
                "drop", "shake", "stack", "knock_over", "place_on",
                "place_in", "open", "close",
            ],
            "allow_freeform_interactions": True,
        },
        "environment_verifier": {
            "enabled": False,
            "provider": "molmo",
            "run_every_iteration": True,
            "max_retries": 4,
            "fail_open_on_error": True,
            "molmo_model": "allenai/Molmo2-8B",
            "molmo_base_url": "http://127.0.0.1:8122/v1",
            "benchmark_reinit_on_fail": True,
        },
        "per_step_verifier": {
            "enabled": True,
            # Default evidence boundary: do not send hidden simulator /
            # privileged state to the VLM; use runtime logs plus media.
            "include_privileged_state": False,
            "save_artifacts": True,
            "model": "google/gemini-3.1-pro-preview",
            "max_tokens": 1600,
            "max_images": 32,
            "max_events_per_step": 96,
            "max_state_entries": 12,
        },
        "feedback_generator": {
            "model": "google/gemini-3.1-pro-preview",
            "max_tokens": 2400,
            "max_frames": 16,
        },
    }
    if not cfg_path.exists():
        return _normalize_molmospaces_config(_merge_molmospaces_config(defaults, override))
    try:
        import yaml
        with cfg_path.open() as f:
            data = yaml.safe_load(f) or {}
        loaded = data.get("molmospaces", {}) or {}
        return _normalize_molmospaces_config(
            _merge_molmospaces_config(
                _merge_molmospaces_config(defaults, loaded),
                override,
            )
        )
    except Exception:
        return _normalize_molmospaces_config(_merge_molmospaces_config(defaults, override))


def _load_scene_compatible_tasks(scene_model: str) -> list[str]:
    """Load tasks compatible with a scene from OmniGibson's task_custom_lists.json."""
    task_list_path = (
        _PROJECT_ROOT / "rats" / "third_party" / "b1k" / "OmniGibson"
        / "omnigibson" / "sampling" / "task_custom_lists.json"
    )
    if not task_list_path.exists():
        return []
    try:
        data = json.loads(task_list_path.read_text())
        return sorted(task for task, cfg in data.items() if scene_model in cfg)
    except Exception:
        return []


def _load_task_template_metadata(scene_model: str, activity_name: str) -> dict | None:
    """Load inst_to_name + robot_poses from pre-sampled task template.

    Returns the task metadata dict or None if no template exists.
    """
    template_dir = (
        _PROJECT_ROOT / "rats" / "third_party" / "b1k" / "joylo"
        / "sampled_task" / activity_name
    )
    template_file = template_dir / f"{scene_model}_task_{activity_name}_0_0_template.json"
    if not template_file.exists():
        return None
    try:
        data = json.loads(template_file.read_text())
        return data.get("metadata", {}).get("task", {})
    except Exception:
        return None


class LifelongLoop:
    """Main RATS orchestrator: runs the lifelong learning loop."""

    def __init__(
        self,
        env: Any,
        *,
        skill_library_path: str = "skill_library/skills.json",
        skill_library_merge_paths: list[str] | None = None,
        skill_library_min_tier: str | None = None,
        playtime_memory_seed_path: str | None = None,
        max_retries_per_task: int = 5,
        execution_timeout: int = 600,
        output_dir: str = "outputs/rats_lifelong",
        available_tasks: list[dict[str, Any]] | None = None,
        fixed_task: bool = False,
        env_type: str | None = None,
        failure_memory_path: str | None = None,
        use_catalog: bool = False,
        curriculum: bool = False,
        curiosity: bool = False,
        no_skill_reuse: bool = False,
        random_order: bool = False,
        no_failure_memory: bool = False,
        save_debug_frames: bool = False,
        ensemble_n: int = 0,
        turns_per_attempt: int = 1,
        attempts_per_iteration: int | None = None,
        molmospaces_config: dict[str, Any] | None = None,
        play_mode: bool = True,
        policy_self_check_max_repairs: int = 2,
        multiturn_reset_mode: bool = False,
        multiturn_reset_max_step_retries: int = 10,
        web_debugger: Any | None = None,
        multi_turn_decision: bool = False,
        # Candidate-based curiosity selection (replaces the old persistent
        # task_queue). ``curiosity_candidate_mode`` ∈ {"none", "formula"}:
        # "formula" scores K fresh + up to K_retry retry-bank candidates per
        # iteration with novelty×frontier and selects the argmax; "none" uses
        # the single-proposal path.
        curiosity_candidate_mode: str = "none",
        proposer_no_context: bool = False,
        proposer_temperature: float | None = None,
        proposer_include_eval_task_context: bool = False,
        num_fresh_candidates: int = 3,
        num_retry_candidates: int = 2,
        retry_bank_size: int = 8,
        retry_bank_ttl: int = 3,
        retry_bonus_weight: float = 0.15,
        failure_penalty_weight: float = 0.10,
        score_composition: str = "product",
        retry_min_pred_success: float = 0.5,
        # Number of warm-up iterations during which the candidate-based
        # curiosity loop is BYPASSED in favor of the legacy single-
        # propose path (`task_proposer.propose()`). During cold start the
        # Goldilocks frontier 4·c·(1−c) systematically rewards compound
        # candidates whose required_skills mix in more unknown names
        # (lowering c toward the 0.5 peak). After this many iterations
        # the skill library has accumulated some Wilson reliability data
        # and the formula starts producing useful signal. Default 0 =
        # candidate mode on from iter 1 (legacy behavior).
        curiosity_warmup_iters: int = 0,
        snapshot_interval: int = 0,
        # Deprecated kwargs kept for back-compat with older launchers /
        # tests; both fold into the unified play_mode now.
        play_iterations: int = 0,
        play_prompt_mode: bool = False,
    ) -> None:
        self.env = env
        self.web_debugger = web_debugger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.web_debugger = web_debugger
        self._no_skill_reuse = no_skill_reuse
        self._no_failure_memory = no_failure_memory
        self._save_debug_frames = save_debug_frames
        # Play mode is the default explore-proposer style — 3-4 year-old
        # curious-child prompt, no warm-up, runs for the entire loop. The
        # old ``play_prompt_mode`` and ``play_iterations`` kwargs are
        # collapsed onto ``play_mode``; ``play_iterations`` is now a
        # no-op (no warm-up flip happens).
        self._play_mode = bool(play_mode or play_prompt_mode)
        self._play_prompt_mode = self._play_mode  # legacy alias
        self._play_iterations = 0  # warm-up retired
        self._proposer_include_eval_task_context = bool(
            proposer_include_eval_task_context
        )
        self._snapshot_interval = max(0, int(snapshot_interval))
        self._curiosity_requested = curiosity
        self._curiosity_enabled = False
        # Two-level structure (matches capx):
        #   iteration  = one task proposal (env reset at iteration boundary)
        #   attempt    = one trial of the proposed task (env reset BETWEEN
        #                attempts within an iteration)
        #   turn       = one policy_writer call inside an attempt (env NOT
        #                reset between turns, so turn 2 builds on turn 1's
        #                grasp and so on)
        #
        # Single-shot legacy mode = 1 turn per attempt, attempts capped by
        # max_retries+1. Turn mode = many turns per attempt, attempts
        # configurable independently.
        self._turns_per_attempt = max(1, int(turns_per_attempt))
        # When attempts_per_iteration is unspecified, default to legacy
        # behavior: max_retries+1 attempts per iteration. When specified,
        # use the explicit count regardless of max_retries.
        if attempts_per_iteration is None:
            self._attempts_per_iteration = max_retries_per_task + 1
        else:
            self._attempts_per_iteration = max(1, int(attempts_per_iteration))
        self._turn_mode = self._turns_per_attempt > 1
        # Per-attempt frame range tracker, reset at each attempt boundary.
        # Each entry is (start, end) into the env's video buffer.
        self._attempt_turn_frame_ranges: list[tuple[int, int]] = []
        self._attempt_timeline_events: list[dict[str, Any]] = []
        self._attempt_viser_frame_start: int | None = None
        if self._turn_mode or attempts_per_iteration is not None:
            logger.info(
                "Iteration loop: %d attempts/iter × %d turns/attempt "
                "(env resets between attempts; not between turns).",
                self._attempts_per_iteration,
                self._turns_per_attempt,
            )
        self._policy_self_check_max_repairs = max(0, int(policy_self_check_max_repairs))
        # Per-task success code cache: {task_name: code_string}
        # When we encounter the same task again, we show the last working code.
        self._successful_code: dict[str, str] = {}
        # Custom verifiers belong to the currently loaded generated LIBERO
        # environment. If a later novel env creation fails and we fall back to
        # this still-loaded env, preserve its relaxed verifier instead of
        # silently reverting to strict reward-only verification.
        self._active_custom_verifier_code = ""
        self._active_custom_verifier_goal = ""
        self._set_api_output_dir(env)

        # Detect environment type: "behavior" or "libero"
        self.env_type = env_type or detect_env_type(env)
        logger.info(f"Environment type: {self.env_type}")
        self._policy_self_check_max_repairs = max(
            0, int(policy_self_check_max_repairs)
        )
        if self._policy_self_check_max_repairs and self._is_physical_env():
            logger.warning(
                "Policy runtime self-check disabled: environment appears "
                "physical/non-resettable."
            )
            self._policy_self_check_max_repairs = 0

        # Each run gets its own skill library copy in the output dir.
        # The source path (skill_library/skills.json) provides the initial
        # primitives; we copy it so runs don't pollute each other.
        #
        # Cross-run knobs: ``skill_library_min_tier`` filters seed skills
        # by their reliability tier (drop unproven extractions before
        # they reach this run); ``skill_library_merge_paths`` lets one
        # run inherit summed usage_count/success_count from prior runs
        # so a skill that worked 5x in playtime + 4x in curriculum starts
        # this run at 9/9 instead of 0/0.
        run_skill_path = self.output_dir / "skills.json"
        self._skill_library_merge_paths = [
            Path(p) for p in (skill_library_merge_paths or []) if p
        ]
        if not run_skill_path.exists():
            src = Path(skill_library_path)
            extras = list(self._skill_library_merge_paths)
            if extras or skill_library_min_tier:
                # Even when the seed file is missing, the merge helper
                # gracefully returns an empty list and the resulting
                # SkillLibrary will fall back to build_initial_skills().
                from skill_library.library import merge_skill_library_files
                merged = merge_skill_library_files(
                    src, extras, min_tier=skill_library_min_tier,
                )
                if merged:
                    run_skill_path.parent.mkdir(parents=True, exist_ok=True)
                    with run_skill_path.open("w") as fh:
                        json.dump(merged, fh, indent=2)
                    logger.info(
                        "Skill seed: merged %d skills from seed=%s + "
                        "%d extras (min_tier=%s) -> %s",
                        len(merged),
                        src,
                        len(extras),
                        skill_library_min_tier or "<none>",
                        run_skill_path,
                    )
                elif src.exists():
                    import shutil
                    shutil.copy2(src, run_skill_path)
            else:
                if src.exists():
                    import shutil
                    shutil.copy2(src, run_skill_path)
                # else SkillLibrary will initialize from primitives
        self.skill_library = SkillLibrary(storage_path=str(run_skill_path))

        # Failure memory: persistent record of past failures
        if no_failure_memory:
            # Still create the object but it stays empty -- all retrieval returns ""
            self.failure_memory = FailureMemory(str(self.output_dir / "failure_memory_disabled"))
            logger.info("Failure memory: DISABLED (ablation)")
        else:
            fm_dir = failure_memory_path or str(self.output_dir / "failure_memory")
            self.failure_memory = FailureMemory(fm_dir)
            if failure_memory_path:
                run_fm_dir = self.output_dir / "failure_memory"
                if str(run_fm_dir) != failure_memory_path:
                    self.failure_memory = FailureMemory(str(run_fm_dir))
                    self.failure_memory.merge_from(failure_memory_path)
            logger.info(f"Failure memory: {self.failure_memory.episode_count} episodes loaded")

        # Novel-task mode: proposer generates specs, env creator builds a
        # runnable artifact/environment for LIBERO.
        # MolmoSpaces uses "novel" proposer mode (selects from benchmark catalog)
        # but the catalog execution path (rebind, not env-create).
        self.novel_tasks = (self.env_type == "libero" and not fixed_task and not use_catalog)
        _molmospaces_explore = (self.env_type == "molmospaces" and not fixed_task and not use_catalog)
        # Standard mode after any Libero play warm-up: novel for LIBERO /
        # MolmoSpaces explore, catalog rebind otherwise.
        self._post_play_mode = "novel" if (self.novel_tasks or _molmospaces_explore) else "catalog"
        self._curiosity_enabled = bool(
            self._curiosity_requested and self.env_type == "libero" and self.novel_tasks
        )
        # Play mode is the default explore-proposer style. No warm-up — the
        # mode set here runs for the entire loop. ``self._post_play_mode``
        # is now only used as the fallback when play mode is explicitly
        # disabled via --no-play-mode.
        if self.env_type == "libero" and self._play_mode:
            proposer_mode = "play"
        else:
            proposer_mode = self._post_play_mode
        self._available_tasks_full = available_tasks or []
        # MolmoSpaces-only knobs. Defaults come from rats/config/default.yaml,
        # and per-run env YAML may override them via a top-level
        # `molmospaces:` block. Cached on self so other parts of the loop
        # (e.g. the rebind path) can consult the same values.
        self._molmospaces_cfg = _load_molmospaces_config(molmospaces_config)
        # Stash early so the task_proposer_kwargs build below (and getattr
        # fallback in legacy paths) can read it without re-plumbing.
        self._proposer_no_context = bool(proposer_no_context)
        self._proposer_temperature = (
            float(proposer_temperature) if proposer_temperature is not None else None
        )
        self._proposer_include_eval_task_context = bool(
            proposer_include_eval_task_context
        )
        if self.env_type == "molmospaces":
            logger.info(
                "MolmoSpaces proposer config: mode=%s, house_switching=%s, "
                "house_switch_max=%s, allowed_task_types=%s, forced_sequence=%s, "
                "vlm_grounding=%s",
                self._molmospaces_cfg.get("proposer_mode", "catalog"),
                self._molmospaces_cfg.get("allow_house_switching", True),
                self._molmospaces_cfg.get("house_switch_max_per_run", 5),
                self._molmospaces_cfg.get("allowed_task_types"),
                self._molmospaces_cfg.get("forced_task_type_sequence"),
                (self._molmospaces_cfg.get("vlm_grounding") or {}).get("enabled", True),
            )
            if self._molmospaces_cfg.get("curiosity"):
                logger.info(
                    "MolmoSpaces playtime curiosity: %s",
                    self._molmospaces_cfg.get("playtime") or {},
                )
        # --play-mode also controls MolmoSpaces: when on, force the
        # molmospaces proposer into "playtime" (which now uses the same
        # 3-4 year-old framing as the libero play prompt). When off,
        # fall back to whatever the molmospaces YAML config requests
        # (default "catalog"). This wires --play-mode / --no-play-mode
        # to both env types from a single CLI knob.
        if self.env_type == "molmospaces":
            molmospaces_proposer_mode = (
                "playtime" if self._play_mode
                else str(self._molmospaces_cfg.get("proposer_mode", "catalog"))
            )
        else:
            molmospaces_proposer_mode = str(
                self._molmospaces_cfg.get("proposer_mode", "catalog")
            )
        task_proposer_kwargs = {
            "available_tasks": available_tasks,
            "mode": proposer_mode,
            "random_order": random_order,
            "curriculum": curriculum,
            "no_context": bool(getattr(self, "_proposer_no_context", False)),
            "proposer_temperature": getattr(self, "_proposer_temperature", None),
            "include_eval_task_context": bool(
                getattr(self, "_proposer_include_eval_task_context", False)
            ),
            "molmospaces_proposer_mode": molmospaces_proposer_mode,
            "molmospaces_allow_house_switching": bool(
                self._molmospaces_cfg.get("allow_house_switching", True)
            ),
            "molmospaces_house_switch_max_per_run": int(
                self._molmospaces_cfg.get("house_switch_max_per_run", 5)
            ),
            "molmospaces_vlm_grounding": bool(
                (self._molmospaces_cfg.get("vlm_grounding") or {}).get("enabled", True)
            ),
            "molmospaces_geometric_visibility_gate": bool(
                (self._molmospaces_cfg.get("vlm_grounding") or {}).get(
                    "geometric_visibility_gate", False
                )
            ),
            "molmospaces_allowed_task_types": list(
                self._molmospaces_cfg.get("allowed_task_types") or []
            ),
            "molmospaces_forced_task_type_sequence": list(
                self._molmospaces_cfg.get("forced_task_type_sequence") or []
            ),
            "molmospaces_curiosity": bool(self._molmospaces_cfg.get("curiosity")),
            "molmospaces_playtime_config": dict(
                self._molmospaces_cfg.get("playtime") or {}
            ),
            "include_eval_task_context": self._proposer_include_eval_task_context,
        }
        try:
            signature = inspect.signature(TaskProposer)
            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
            if not accepts_kwargs:
                task_proposer_kwargs = {
                    key: value
                    for key, value in task_proposer_kwargs.items()
                    if key in signature.parameters
                }
        except (TypeError, ValueError):
            pass
        self.task_proposer = TaskProposer(**task_proposer_kwargs)

        # Candidate-based curiosity selection. ``curiosity_candidate_mode``
        # is "formula" for LIBERO explore (set automatically by run_rats) and
        # "none" everywhere else; "formula" supersedes the legacy persistent
        # task_queue, which is only constructed when the candidate path is off.
        self._candidate_mode = str(curiosity_candidate_mode or "none").lower()
        if self._candidate_mode not in {"none", "formula"}:
            logger.warning(
                "Unknown curiosity_candidate_mode=%r; defaulting to 'none'",
                curiosity_candidate_mode,
            )
            self._candidate_mode = "none"
        self._num_fresh_candidates = max(1, int(num_fresh_candidates))
        self._num_retry_candidates = max(0, int(num_retry_candidates))
        self._retry_bank_size = max(1, int(retry_bank_size))
        self._retry_bank_ttl = max(1, int(retry_bank_ttl))
        self._retry_bonus_weight = float(retry_bonus_weight)
        self._failure_penalty_weight = float(failure_penalty_weight)
        self._score_composition = (
            "weighted_sum" if str(score_composition).lower() == "weighted_sum"
            else "product"
        )
        self._retry_min_pred_success = float(retry_min_pred_success)
        self._curiosity_warmup_iters = max(0, int(curiosity_warmup_iters))

        self.retry_bank: RetryBank | None = None
        self._obj_skill_counts: dict[tuple[str, str], int] = {}
        self._obj_skill_counts_path: str | None = None
        if self._candidate_mode == "formula":
            from rats.agents.curiosity_scoring import load_object_skill_counts
            self.retry_bank = RetryBank(
                self.output_dir / "retry_bank" / "items.json",
                max_size=self._retry_bank_size,
                default_ttl=self._retry_bank_ttl,
                min_predicted_success=self._retry_min_pred_success,
            )
            self._obj_skill_counts_path = str(
                self.output_dir / "candidate_state" / "object_skill_counts.json"
            )
            self._obj_skill_counts = load_object_skill_counts(
                self._obj_skill_counts_path
            )
            logger.info(
                "Task proposer: CANDIDATE selection ON "
                "(mode=%s, composition=%s, fresh=%d, retry=%d, "
                "retry_bonus_w=%.2f, fail_pen_w=%.2f, warmup_iters=%d)",
                self._candidate_mode, self._score_composition,
                self._num_fresh_candidates, self._num_retry_candidates,
                self._retry_bonus_weight, self._failure_penalty_weight,
                self._curiosity_warmup_iters,
            )

        # Legacy persistent task_queue: deprecated and superseded by the
        # formula candidate scorer. ``_curiosity_enabled`` is always False
        # now, so this stays None for fresh runs (kept for resume compat).
        self.task_queue = (
            TaskQueue(self.output_dir / "task_queue" / "tasks.json")
            if self._curiosity_enabled and self._candidate_mode == "none"
            else None
        )
        if self._play_mode:
            logger.info(
                "Task proposer: PLAY mode ON for the entire run "
                "(3-4 year-old curious-child prompt, unified across "
                "LIBERO + MolmoSpaces). Disable with --no-play-mode."
            )
        elif self.env_type == "libero":
            logger.info(
                f"Task proposer: PLAY mode OFF (--no-play-mode); using "
                f"'{self._post_play_mode}' mode."
            )
        if curriculum:
            logger.info("Task proposer: CURRICULUM mode ON (stage-gated by success count)")
        if self._curiosity_enabled:
            logger.info("Task proposer: CURIOSITY queue ON (LIBERO explore only)")
        # Environment Creator handles task setup for ALL env types:
        # - BEHAVIOR: template injection + configure_behavior_task (catalog tasks)
        # - LIBERO: novel BDDL generation + env instantiation
        self.env_creator = EnvironmentCreator(
            env_type=self.env_type,
            bddl_output_dir=self.output_dir / "generated_bddl",
        ) if not fixed_task else None
        self.planner = Planner()
        # Focused sub-skill learner. Spawned mid-iteration when the
        # diagnoser identifies a specific sub-behavior that needs
        # isolated practice. Runs its own short retry loop on a
        # single-step plan, reuses diagnoser / executor, and on success
        # extracts a reusable skill for the library. Owns its own
        # PolicyWriter with a higher ensemble_n than the main loop — the
        # whole point is to explore variants the deterministic main path
        # did not discover. Settings come from rats/config/default.yaml under
        # the `subagent:` key.
        _sub_cfg = _load_subagent_config()
        self.sub_agent = SubAgent(
            max_retries=int(_sub_cfg.get("max_retries", 6)),
            ensemble_n=int(_sub_cfg.get("ensemble_n", 3)),
        )
        self._subagent_enabled = os.getenv(
            "RATS_SUBAGENT_DISABLED", ""
        ).strip().lower() not in ("1", "true", "yes", "on")
        if not self._subagent_enabled:
            logger.info("SubAgent DISABLED via RATS_SUBAGENT_DISABLED env var")
        self.policy_writer = PolicyWriter(max_retries=max_retries_per_task, ensemble_n=ensemble_n)
        self.quality_checker = PolicyQualityChecker()
        self.executor = Executor(timeout_seconds=execution_timeout)
        # Opt-in CaP-X-style intra-attempt decider. When enabled AND
        # turns_per_attempt > 1, after each non-terminal turn we run a
        # single fast LLM call that returns FINISH | REGENERATE+code.
        # REGENERATE short-circuits the heavy verifier→diagnoser→writer
        # pipeline for the next turn and feeds the new code straight into
        # the executor. FINISH falls through to the existing pipeline so
        # skill extraction still runs on the final turn of the attempt.
        self._multi_turn_decision_enabled = bool(multi_turn_decision)
        self.multi_turn_decider = (
            MultiTurnDecider() if self._multi_turn_decision_enabled else None
        )
        if self._multi_turn_decision_enabled:
            logger.info(
                "Multi-turn decider ENABLED (CaP-X-style intra-attempt "
                "FINISH/REGENERATE gate)."
            )
        self.verifier = Verifier()
        play_cfg = self._molmospaces_cfg.get("playtime") or {}
        env_verifier_cfg = self._molmospaces_cfg.get("environment_verifier") or {}
        per_step_cfg = self._molmospaces_cfg.get("per_step_verifier") or {}
        # Default-on for molmospaces (its plan-step shape and event-trace
        # API were designed for this verifier) AND for libero (the
        # privileged-API plan steps also expose enough state for the
        # event-assignment + LLM verdict pipeline to work). For BEHAVIOR
        # and other envs, leave it off until they emit a similar trace.
        # Any config can still override via `per_step_verifier: { enabled: ... }`.
        # Multiturn-reset is the only consumer of per_step_verifier's
        # retry-directive fields (edit_scale, corrective_action). Gate them
        # at construction so default LIBERO/MolmoSpaces runs don't pay the
        # extra ~150 output tokens per VLM call for fields they ignore.
        _multiturn_reset_will_be_enabled = (
            bool(multiturn_reset_mode) and self.env_type == "libero"
        )
        per_step_enabled_override = _env_bool_override("RATS_PER_STEP_VERIFIER_ENABLED")
        self.per_step_verifier = PerStepVerifier(
            enabled=(
                bool(per_step_enabled_override)
                if per_step_enabled_override is not None
                else bool(per_step_cfg.get(
                    "enabled", self.env_type in ("molmospaces", "libero"),
                ))
            ),
            include_privileged_state=bool(per_step_cfg.get("include_privileged_state", False)),
            save_artifacts=bool(per_step_cfg.get("save_artifacts", True)),
            max_events_per_step=int(per_step_cfg.get("max_events_per_step", 96) or 96),
            max_state_entries=int(per_step_cfg.get("max_state_entries", 12) or 12),
            model=(
                os.getenv("RATS_PER_STEP_VERIFIER_MODEL")
                or per_step_cfg.get("model")
                or os.getenv("RATS_LLM_MODEL")
                or "google/gemini-3.1-pro-preview"
            ),
            max_tokens=int(per_step_cfg.get("max_tokens", 1600) or 1600),
            max_images=int(per_step_cfg.get("max_images", 32) or 32),
            include_retry_directive=_multiturn_reset_will_be_enabled,
        )

        # Multiturn-reset (step-by-step) executor — LIBERO-only opt-in. When
        # enabled, _run_one_iteration routes each attempt's policy-execution
        # block through MultiturnResetExecutor instead of the single-shot writer
        # path. Disabled for non-LIBERO envs since BEHAVIOR's physical env
        # can't safely env.reset() between step retries.
        from rats.loop.multiturn_reset_executor import MultiturnResetConfig, MultiturnResetExecutor
        self._multiturn_reset_enabled = (
            bool(multiturn_reset_mode) and self.env_type == "libero"
        )
        if multiturn_reset_mode and not self._multiturn_reset_enabled:
            logger.warning(
                "Multiturn-reset mode requested but env_type=%s (not 'libero'); ignored.",
                self.env_type,
            )
        self._multiturn_reset_max_step_retries = max(1, int(multiturn_reset_max_step_retries))
        self.multiturn_reset_executor: MultiturnResetExecutor | None = None
        if self._multiturn_reset_enabled:
            self.multiturn_reset_executor = MultiturnResetExecutor(
                executor=self.executor,
                policy_writer=self.policy_writer,
                per_step_verifier=self.per_step_verifier,
                env_resetter=self._reset_env,
                config=MultiturnResetConfig(
                    enabled=True,
                    max_step_retries=self._multiturn_reset_max_step_retries,
                ),
                # Reuse the legacy frame-capture + api-logging helpers
                # so per_step_verifier finds step_frame_segments instead
                # of short-circuiting with "no step media available".
                frame_segment_builder=self._build_step_frame_segments,
                api_logging_enable_fn=self._enable_api_execution_logging,
                api_logging_restore_fn=self._restore_api_execution_logging,
                # Plumb the PolicyQualityChecker so the per-step writer
                # output goes through the same Tier-1 AST/API gate the
                # legacy single-shot path uses. Without this, the stub
                # artifact in _run_attempt_via_multiturn_reset was
                # claiming quality was approved without ever running
                # the check.
                quality_checker=self.quality_checker,
            )
            logger.info(
                "Multiturn-reset mode ENABLED (max_step_retries=%d).",
                self._multiturn_reset_max_step_retries,
            )
        # Pre-execution plan gate. Same VLM family as the per-step
        # verifier but scoped to (initial agentview, plan) — checks
        # visual misperception, missing prerequisites, wrong ordering,
        # and object-scope mismatch. Default-on for the envs whose
        # planner is fed an agentview image (LIBERO + molmospaces);
        # other envs skip it. Bounded verify→refine loop: a failing
        # verdict triggers refine_plan, then re-verify, then re-refine
        # if still failing, up to ``planner_verifier.max_refines``
        # passes (default 5). Was a one-shot gate; bumped because one
        # refine is too few for plans that need a cascade of fixes.
        planner_verifier_cfg = self._molmospaces_cfg.get("planner_verifier") or {}
        planner_enabled_override = _env_bool_override("RATS_PLANNER_VERIFIER_ENABLED")
        self.planner_verifier = PlannerVerifier(
            enabled=(
                bool(planner_enabled_override)
                if planner_enabled_override is not None
                else bool(planner_verifier_cfg.get(
                    "enabled", self.env_type in ("molmospaces", "libero"),
                ))
            ),
            save_artifacts=bool(planner_verifier_cfg.get("save_artifacts", True)),
            model=(
                os.getenv("RATS_PLANNER_VERIFIER_MODEL")
                or planner_verifier_cfg.get("model")
                or os.getenv("RATS_LLM_MODEL")
                or "google/gemini-3.1-pro-preview"
            ),
            max_tokens=int(planner_verifier_cfg.get("max_tokens", 8192) or 8192),
            min_fail_confidence=float(
                planner_verifier_cfg.get("min_fail_confidence", 0.6) or 0.6
            ),
        )
        # Active env verifier (one impl per env type). Both impls inherit
        # from agents.environment_verifier.EnvironmentVerifier and expose
        # ``.verify(env, task_proposal, **kw) -> {suitable, ok, reason, ...}``.
        # For molmospaces this is the post-rebind Molmo-grounding probe;
        # the libero counterpart lives inside EnvironmentCreator (called
        # at instantiate time, not as a separate loop-level agent).
        self.environment_verifier = MolmoEnvironmentVerifier(
            enabled=bool(env_verifier_cfg.get("enabled", False)),
            provider=str(env_verifier_cfg.get("provider", "molmo")),
            max_retries=int(env_verifier_cfg.get("max_retries", 4) or 4),
            fail_open_on_error=bool(env_verifier_cfg.get("fail_open_on_error", True)),
            molmo_model=str(env_verifier_cfg.get("molmo_model", "allenai/Molmo2-8B")),
            molmo_base_url=str(
                env_verifier_cfg.get("molmo_base_url", "http://127.0.0.1:8122/v1")
            ),
            molmo_api_key=env_verifier_cfg.get("molmo_api_key"),
            output_dir=self.output_dir,
        )
        # Back-compat alias for anything still reading the old attribute name.
        self.molmospaces_environment_verifier = self.environment_verifier
        self.failure_diagnoser = FailureDiagnoser()
        # The feedback generator's "retry vs skip" cutoff is keyed off a
        # flat retry counter. Across the nested attempt × turn loops the
        # total code-gen budget per iteration is attempts × turns, so
        # cap it at that to avoid premature "skip" decisions before the
        # outer loops have finished their budgets.
        feedback_max_retries = max(
            max_retries_per_task,
            self._attempts_per_iteration * self._turns_per_attempt,
        )
        feedback_cfg = self._molmospaces_cfg.get("feedback_generator") or {}
        self.feedback_generator = FeedbackGenerator(
            max_retries=feedback_max_retries,
            model=(
                os.getenv("RATS_FEEDBACK_GENERATOR_MODEL")
                or os.getenv("RATS_LLM_MODEL")
            ),
            molmospaces_model=(
                os.getenv("RATS_FEEDBACK_GENERATOR_MODEL")
                or feedback_cfg.get("model")
                or os.getenv("RATS_LLM_MODEL")
                or "google/gemini-3.1-pro-preview"
            ),
            molmospaces_max_tokens=int(feedback_cfg.get("max_tokens", 65536) or 65536),
            molmospaces_max_frames=int(feedback_cfg.get("max_frames", 16) or 16),
        )
        self.skill_proposer = SkillProposer()
        self.memory_curator = MemoryCurator()
        # PlaytimeMemory was an extra archive of (object, interaction,
        # outcome) tuples that the molmospaces playtime proposer used to
        # feed back into its prompt as "prior sensorimotor observations".
        # The unified --play-mode treats LIBERO and MolmoSpaces play
        # identically — a prompt-only flavour on the standard novel
        # pipeline — so this side archive is no longer needed. Kept as
        # None so any straggling getattr-based caller is a no-op rather
        # than an AttributeError.
        self.playtime_memory = None
        self.playtime_memory_seed = None
        self.metrics = MetricsTracker()
        self._iters_since_skill_proposal = 0
        self._iters_since_curate = 0
        # Curate lessons every 5 iterations; tunable.
        self._curate_every = int(os.getenv("RATS_CURATE_EVERY", "5"))
        # Rolling iteration history for MemoryCurator's context.
        self._iter_history_for_curator: list[dict[str, Any]] = []

        self.max_retries = max_retries_per_task
        self.fixed_task = fixed_task
        self._iteration = 0
        self._failed_rebind_tasks: set[str] = set()
        self._validated_tasks: list[str] | None = None
        self._bootstrap_activity: str | None = None
        self._resumed_results: list[dict[str, Any]] = []  # loaded from previous run
        self._skip_completed_iteration_results: dict[int, dict[str, Any]] = {}
        self._active_custom_verifier_code = ""
        self._active_custom_verifier_goal = ""
        # Per-iteration init-failure flag set by _reset_env. When True the
        # iteration loop should skip executor / verifier work and record
        # the iteration as a soft init failure.
        self._last_reset_failed: bool = False
        self._last_reset_error: str = ""
        self._molmospaces_env_verifier_pending_reasons: list[str] = []
        # MolmoSpaces scene-first playtime: a shuffled list of
        # house_index values to walk deterministically. Populated lazily
        # the first time _pick_next_playtime_scene runs; ``None`` means
        # scene-first mode is disabled or the bridge hasn't been
        # interrogated yet.
        self._playtime_scene_pool: list[int] | None = None
        self._playtime_scene_pool_cursor: int = 0

    def _set_api_output_dir(self, env: Any) -> None:
        """Point API debug image saves at the run output directory."""
        for api in getattr(env, "_apis", {}).values():
            if hasattr(api, "output_dir"):
                api.output_dir = self.output_dir
        debugger = getattr(self, "web_debugger", None)
        if debugger is not None:
            try:
                debugger.set_env(env)
            except Exception:
                pass

    def _queue_molmospaces_environment_check(self, reason: str) -> None:
        """Remember that the current MolmoSpaces task/env pair needs a guard check."""
        if self.env_type != "molmospaces":
            return
        reason = str(reason or "").strip()
        if not reason:
            return
        pending = getattr(self, "_molmospaces_env_verifier_pending_reasons", None)
        if pending is None:
            self._molmospaces_env_verifier_pending_reasons = []
            pending = self._molmospaces_env_verifier_pending_reasons
        if reason not in pending:
            pending.append(reason)

    def _consume_molmospaces_environment_check_reasons(self) -> list[str]:
        reasons = list(getattr(self, "_molmospaces_env_verifier_pending_reasons", []) or [])
        self._molmospaces_env_verifier_pending_reasons = []
        return reasons

    def _verify_molmospaces_environment_if_needed(
        self,
        task_proposal: dict[str, Any],
        scene_context: dict[str, Any],
        *,
        reasons: list[str] | None = None,
        attempt: int = 0,
    ) -> dict[str, Any] | None:
        """Run the pre-flight Molmo verifier on each iteration and risk event."""
        if self.env_type != "molmospaces":
            return None
        verifier = getattr(self, "environment_verifier", None)
        enabled = bool(verifier is not None and getattr(verifier, "enabled", False))
        all_reasons: list[str] = []
        for reason in list(reasons or []) + self._consume_molmospaces_environment_check_reasons():
            if reason and reason not in all_reasons:
                all_reasons.append(reason)
        run_every = bool(
            (self._molmospaces_cfg.get("environment_verifier") or {}).get(
                "run_every_iteration", True
            )
        )
        if run_every and enabled and "iteration_preflight" not in all_reasons:
            all_reasons.insert(0, "iteration_preflight")
        if not all_reasons:
            return None
        if not enabled:
            return {
                "enabled": False,
                "suitable": True,
                "reason": "disabled",
                "reasons": all_reasons,
            }
        logger.info(
            "  MolmoSpaces environment verifier: checking task target after %s",
            ", ".join(all_reasons),
        )
        return verifier.verify(
            self.env,
            task_proposal,
            scene_context,
            iteration=self._iteration,
            attempt=attempt,
            reasons=all_reasons,
        )

    @staticmethod
    def _environment_verifier_feedback(result: dict[str, Any]) -> str:
        task = result.get("task") or {}
        return (
            "your previous proposal was rejected by the environment verifier: "
            f"Molmo could not find the proposed target in the current agentview. "
            f"Previous task: {task.get('language') or task.get('activity_name')}. "
            f"Queries tried: {result.get('queries') or []}. "
            "Choose another clearly visible object in the current scene."
        )

    def _in_play_phase(self) -> bool:
        """Retired: unified play mode no longer tags skills with a ``play:``
        provenance prefix.

        The original LIBERO ``--play-mode`` (since deleted in favor of the
        prompt-only sibling) was a warm-up phase that tagged extracted
        skills with ``source_task="play:<activity>"``. The new unified play
        mode descends from ``--play-prompt-mode``, which never applied
        that prefix — it was strictly a prompt swap on the standard novel
        pipeline. This helper now always returns False so the prefix path
        is dead. Kept as a stub so any straggling caller still resolves.
        """
        return False

    @staticmethod
    def _normalize_goal_for_verifier(goal: str) -> str:
        return re.sub(r"\s+", " ", str(goal or "").strip().lower().replace("_", " "))

    def _remember_active_custom_verifier(self, code: str, goal: str) -> None:
        self._active_custom_verifier_code = code or ""
        self._active_custom_verifier_goal = self._normalize_goal_for_verifier(goal)

    def _active_custom_verifier_for_goal(self, goal: str) -> str:
        if not getattr(self, "_active_custom_verifier_code", ""):
            return ""
        if not getattr(self, "_active_custom_verifier_goal", ""):
            return ""
        if self._active_custom_verifier_goal != self._normalize_goal_for_verifier(goal):
            return ""
        return self._active_custom_verifier_code

    def _is_physical_env(self) -> bool:
        """Best-effort guard for policy self-check reset safety."""
        low_level = getattr(self.env, "low_level_env", self.env)
        cls = low_level.__class__
        module = getattr(cls, "__module__", "").lower()
        name = getattr(cls, "__name__", "").lower()
        env_type = str(getattr(self, "env_type", "") or "").lower()
        if env_type == "molmospaces":
            # MolmoSpaces runs through a resettable simulator/remote bridge in
            # this branch; keep self-check available there unless config turns
            # it off. The reset path has MolmoSpaces-specific recovery logic.
            return False
        return (
            "franka_real" in module
            or name == "frankareallowlevel"
            or ("real" in module and "sim" not in module)
        )

    def _quality_check_policy(
        self,
        code: str,
        scene_context: dict[str, Any],
        learned_skill_names: list[str],
    ) -> dict[str, Any]:
        return self.quality_checker.check(
            code,
            available_functions=scene_context.get("available_functions"),
            goal=scene_context.get("goal_conditions_nl", ""),
            learned_skill_names=learned_skill_names,
        )

    def _run_policy_runtime_self_check(
        self,
        exec_code: str,
        scene_context: dict[str, Any],
        *,
        runtime_self_check_enabled: bool = True,
        attempt: int | None = None,
        repair_index: int = 0,
    ) -> dict[str, Any]:
        """Run candidate policy once before official delivery.

        This catches Python/API runtime crashes before the official execution
        attempt is recorded, then resets the env so official execution starts
        from the canonical state. It must be disabled for non-resettable
        physical envs and for in-progress MolmoSpaces turn state.
        """
        if self._policy_self_check_max_repairs <= 0 or not runtime_self_check_enabled:
            return {
                "enabled": False,
                "passed": True,
                "reason": (
                    "disabled_by_context"
                    if not runtime_self_check_enabled
                    else "disabled_by_config"
                ),
            }

        logger.info("Step 4b: Policy Runtime Self-Check")
        low_level = getattr(self.env, "low_level_env", self.env)
        result: dict[str, Any] = {}
        video_artifacts: dict[str, Any] = {}
        try:
            if hasattr(low_level, "enable_video_capture"):
                try:
                    low_level.enable_video_capture(False, clear=True)
                except Exception:
                    pass
            self._reset_env()
            if hasattr(low_level, "enable_video_capture"):
                try:
                    low_level.enable_video_capture(True, clear=True)
                except Exception:
                    pass
            if self.web_debugger is not None:
                with self.web_debugger.execution_interrupt_scope():
                    result = self.executor.execute(exec_code, self.env, scene_context)
            else:
                result = self.executor.execute(exec_code, self.env, scene_context)
        finally:
            if hasattr(low_level, "get_video_frames"):
                try:
                    frames = low_level.get_video_frames(clear=True)
                    video_artifacts = self._save_policy_self_check_video(
                        frames,
                        attempt=attempt,
                        repair_index=repair_index,
                        status=(
                            "passed"
                            if bool(result.get("success")) and not (result.get("stderr") or "").strip()
                            else "failed"
                        ),
                    )
                except Exception:
                    pass
            if hasattr(low_level, "enable_video_capture"):
                try:
                    low_level.enable_video_capture(False, clear=False)
                except Exception:
                    pass
            self._reset_env()

        stderr = (result.get("stderr") or "").strip()
        stdout = result.get("stdout") or ""
        reward = result.get("reward")
        task_completed = result.get("task_completed")
        passed = bool(result.get("success")) and not stderr
        return {
            "enabled": True,
            "passed": passed,
            "success": bool(result.get("success")),
            "timeout": bool(result.get("timeout")),
            "timeout_message": result.get("timeout_message"),
            "timeout_seconds": result.get("timeout_seconds"),
            "reward": float(reward) if reward is not None else None,
            "task_completed": bool(task_completed) if task_completed is not None else None,
            "stdout_snippet": stdout[:2000],
            "stderr_snippet": stderr[:4000],
            "stderr": stderr,
            "user_result": result.get("user_result"),
            "artifacts": result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {},
            "video_artifacts": video_artifacts,
            "video_path": video_artifacts.get("video_path"),
            "skills_video_path": video_artifacts.get("skills_video_path"),
            "api_diagnostics_summary": (
                result.get("artifacts", {})
                .get("info", {})
                .get("api_diagnostics_summary")
            ),
        }

    def _build_policy_self_check_feedback(
        self,
        *,
        attempt: int,
        code: str,
        self_check: dict[str, Any],
        previous_retry_feedback: dict[str, Any] | None,
        scene_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prior = ""
        if previous_retry_feedback:
            prior = (
                "\nOriginal retry context from the official previous attempt:\n"
                f"- failed_step: {previous_retry_feedback.get('failed_step', 'unknown')}\n"
                f"- failure_mode: {previous_retry_feedback.get('failure_mode', 'unknown')}\n"
                f"- diagnosis: {previous_retry_feedback.get('diagnosis', '')}\n"
            )
        stderr = self_check.get("stderr") or self_check.get("stderr_snippet") or ""
        is_timeout = self_check.get("timeout") or "timeout" in str(stderr).lower()
        is_motion_abort = "MotionAbort" in stderr

        if is_timeout or is_motion_abort:
            video_data_url = self._self_check_video_to_data_url(self_check)
            diag_result = {
                "success": False,
                "stdout": self_check.get("stdout_snippet", ""),
                "stderr": stderr,
                "timeout": bool(is_timeout),
                "timeout_message": self_check.get("timeout_message") or (stderr if is_timeout else ""),
                "timeout_seconds": self_check.get("timeout_seconds"),
                "artifacts": self_check.get("artifacts") or {},
                "trajectory_video_data_url": video_data_url,
                "trajectory_frames": [],
                "vlm_verifier_frames": [],
            }
            if is_timeout:
                default_feedback = (
                    "Policy runtime self-check timed out before official "
                    "execution. Add bounded loops/timeouts, remove blocking "
                    "waits, and gate long motion/perception phases on runtime "
                    "feedback before retrying."
                )
                failure_mode = "timeout"
                prefix = "Policy runtime self-check timed out before official execution. "
            else:
                default_feedback = (
                    "Policy runtime self-check hit MolmoSpacesMotionAbort: the "
                    "robot arm stalled with no measurable progress toward target. "
                    "This typically means the commanded pose causes the gripper "
                    "to collide with scene geometry (table, object body, cabinet "
                    "face) on the way to or at the target. Choose a different "
                    "approach angle, use a pre-approach waypoint further from "
                    "obstacles, or select a grasp candidate whose approach "
                    "direction has more clearance."
                )
                failure_mode = "navigation_error"
                prefix = "Policy runtime self-check hit MotionAbort (arm stalled). "
            try:
                diag = self.failure_diagnoser.diagnose(
                    diag_result,
                    scene_context or {},
                    plan=None,
                    code=code,
                    goal_predicates=[],
                    affordance_hints={},
                    prior_attempts=None,
                )
            except Exception:
                diag = {
                    "failed_step": "runtime_self_check",
                    "failure_mode": failure_mode,
                    "policy_feedback": default_feedback,
                }
            return {
                "attempt": attempt + 1,
                "stderr": stderr,
                "diagnosis": (
                    prefix
                    + str(diag.get("policy_feedback", default_feedback))
                    + prior
                ),
                "failed_step": diag.get("failed_step", "runtime_self_check"),
                "failure_mode": diag.get("failure_mode", failure_mode),
                "previous_code": code,
            }
        return {
            "attempt": attempt + 1,
            "stderr": stderr,
            "diagnosis": (
                "Policy runtime self-check crashed before official execution. "
                "Fix the Python/API runtime error using the exact stderr below, "
                "then keep pursuing the task strategy from the plan and any "
                "original retry context. Do not hand in code until this self-check "
                "runs without stderr."
                f"{prior}"
            ),
            "failed_step": "runtime_self_check",
            "failure_mode": "code_bug",
            "previous_code": code,
        }

    def _self_check_video_to_data_url(self, self_check: dict[str, Any]) -> str | None:
        """Load the self-check video file and return a base64 data URL for the diagnoser."""
        video_path = self_check.get("video_path") or (
            self_check.get("video_artifacts") or {}
        ).get("video_path")
        if not video_path:
            return None
        from pathlib import Path
        p = Path(video_path)
        if not p.exists():
            return None
        try:
            import base64
            data = p.read_bytes()
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:video/mp4;base64,{b64}"
        except Exception:
            return None

    def _save_policy_self_check_video(
        self,
        frames: list[Any] | None,
        *,
        attempt: int | None,
        repair_index: int = 0,
        status: str = "failed",
    ) -> dict[str, Any]:
        """Persist a human-readable video for resettable runtime self-checks."""
        artifacts: dict[str, Any] = {
            "iteration": self._iteration,
            "attempt": attempt,
            "repair_index": repair_index,
            "status": status,
        }
        if not frames:
            return artifacts
        try:
            import imageio
        except Exception as exc:
            artifacts["video_error"] = f"imageio_unavailable: {exc}"
            return artifacts

        safe_status = re.sub(r"[^A-Za-z0-9_-]+", "_", str(status or "failed"))
        attempt_label = "unknown" if attempt is None else str(attempt)
        suffix = (
            f"attempt{attempt_label}_{safe_status}"
            if repair_index <= 0
            else f"attempt{attempt_label}_self_repair{repair_index}_{safe_status}"
        )
        video_path = self.output_dir / f"iter{self._iteration:03d}_policy_self_check_{suffix}.mp4"
        try:
            imageio.mimsave(str(video_path), frames, fps=20)
            artifacts["video_path"] = str(video_path)
            artifacts["frame_count"] = len(frames)
            overlay_path = video_path.with_name(f"{video_path.stem}_skills.mp4")
            overlay_ok = self._write_skill_overlay_video(
                overlay_path,
                frames,
                [],
                frame_offset=0,
                fps=20.0,
            )
            if overlay_ok:
                artifacts["skills_video_path"] = str(overlay_path)
            logger.info(
                "  Saved policy self-check video: %s (%d frames)%s",
                video_path.name,
                len(frames),
                f" + {overlay_path.name}" if overlay_ok else "",
            )
        except Exception as exc:
            artifacts["video_error"] = f"{type(exc).__name__}: {exc}"
            logger.debug("  Policy self-check video save failed: %s", exc)
        return artifacts

    def _self_check_and_repair_policy(
        self,
        *,
        code: str,
        plan: dict[str, Any],
        scene_context: dict[str, Any],
        retry_feedback: dict[str, Any] | None,
        failure_context: str,
        success_context: str,
        skill_preamble: str,
        learned_skill_names: list[str],
        attempt: int,
        iteration_data: dict[str, Any],
        runtime_self_check_enabled: bool = True,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Quality-check, runtime self-check, and repair code before delivery."""
        repairs_used = 0
        max_repairs = self._policy_self_check_max_repairs
        current_retry_feedback = retry_feedback

        while True:
            logger.info(
                "Step 4: Quality Check"
                + (f" (self-repair {repairs_used})" if repairs_used else "")
            )
            quality = self._quality_check_policy(
                code,
                scene_context,
                learned_skill_names,
            )
            quality_key = (
                f"quality_attempt_{attempt}"
                if repairs_used == 0
                else f"quality_attempt_{attempt}_self_repair_{repairs_used}"
            )
            iteration_data[quality_key] = quality

            if not quality["approved"]:
                return code, quality, {
                    "passed": False,
                    "reason": "quality_check",
                    "repairs_used": repairs_used,
                }

            if quality.get("tier2_issues"):
                logger.info(f"  Advisory: {quality['tier2_issues']}")

            exec_code = f"{skill_preamble}\n\n{code}" if skill_preamble else code
            self_check = self._run_policy_runtime_self_check(
                exec_code,
                scene_context,
                runtime_self_check_enabled=runtime_self_check_enabled,
                attempt=attempt,
                repair_index=repairs_used,
            )
            self_check_key = (
                f"policy_self_check_attempt_{attempt}"
                if repairs_used == 0
                else f"policy_self_check_attempt_{attempt}_self_repair_{repairs_used}"
            )
            iteration_data[self_check_key] = self_check

            if self_check.get("passed", False):
                return code, quality, {
                    "passed": True,
                    "self_check": self_check,
                    "repairs_used": repairs_used,
                }

            repair_feedback = self._build_policy_self_check_feedback(
                attempt=attempt,
                code=code,
                self_check=self_check,
                previous_retry_feedback=current_retry_feedback,
                scene_context=scene_context,
            )
            logger.warning(
                "  Policy runtime self-check failed: "
                f"{(self_check.get('stderr_snippet') or '')[:500]}"
            )
            if repairs_used >= max_repairs:
                return code, quality, {
                    "passed": False,
                    "reason": "runtime_self_check",
                    "self_check": self_check,
                    "retry_feedback": repair_feedback,
                    "repairs_used": repairs_used,
                }

            repairs_used += 1
            logger.info(
                f"Step 3b: Policy Self-Repair "
                f"({repairs_used}/{max_repairs})"
            )
            code = self.policy_writer.write(
                plan,
                scene_context,
                retry_feedback=repair_feedback,
                failure_context=failure_context,
                success_context=success_context,
            )
            iteration_data[f"code_attempt_{attempt}_self_repair_{repairs_used}"] = code
            current_retry_feedback = repair_feedback

    @staticmethod
    def _is_no_safe_playtime_result(result: dict[str, Any]) -> bool:
        proposal = result.get("task_proposal") or {}
        playtime = proposal.get("_playtime") or {}
        language = str(
            proposal.get("language")
            or proposal.get("goal_conditions")
            or result.get("language")
            or ""
        ).lower()
        no_safe_language = (
            "no safe visible playtime target" in language
            or "no safe playtime target" in language
            or "no taskable target available" in language
        )
        missing_target = not (
            str(playtime.get("target_internal_name") or "").strip()
            or str(playtime.get("target_display_name") or "").strip()
            or str(proposal.get("play_target") or "").strip()
        )
        return no_safe_language or (
            str(proposal.get("_molmospaces_proposer_mode") or "") == "playtime"
            and missing_target
        )

    def resume_from(
        self,
        prev_output_dir: str | Path,
        *,
        skip_completed: bool = False,
    ) -> int:
        """Load state from a previous run to resume from.

        Loads completed iteration results (for task history), skill library,
        failure memory, and the last successful code snippets. Partial/hung
        iteration artifacts are intentionally ignored unless an
        ``iteration_NNN.json`` exists and parses cleanly.
        When ``skip_completed`` is true, completed playtime scenes are retained
        as skipped history but the known no-safe-target fallback iterations are
        left open so they can be rerun after proposer fixes.
        Returns the number of completed iterations found.
        """
        prev_path = Path(prev_output_dir)
        prev_dir = (
            prev_path.parent
            if prev_path.is_file() or prev_path.suffix == ".json"
            else prev_path
        )
        loaded: list[tuple[int, dict[str, Any], str]] = []
        skipped: list[str] = []

        def _iteration_number(path: Path) -> int | None:
            m = re.match(r"iteration_(\d+)\.json$", path.name)
            if not m:
                return None
            return int(m.group(1))

        def _latest_code_attempt(result: dict[str, Any]) -> str | None:
            attempts: list[tuple[int, str]] = []
            for key, value in result.items():
                m = re.match(r"code_attempt_(\d+)$", str(key))
                if m and isinstance(value, str) and value.strip():
                    attempts.append((int(m.group(1)), value))
            if not attempts:
                return None
            return max(attempts, key=lambda item: item[0])[1]

        # Load previous iteration results for task history
        for f in sorted(prev_dir.glob("iteration_*.json")):
            iter_num = _iteration_number(f)
            if iter_num is None:
                continue
            try:
                d = json.loads(f.read_text())
                if not isinstance(d, dict):
                    raise ValueError("iteration file did not contain a JSON object")
                loaded.append((iter_num, d, f.read_text()))
            except Exception as e:
                skipped.append(f"{f.name}: {e}")
                logger.warning(f"Could not load {f}: {e}")

        loaded.sort(key=lambda item: item[0])
        rerun_no_safe_iters: list[int] = []
        skipped_completed_iters: list[int] = []
        for iter_num, d, raw_text in loaded:
            rerun_no_safe = skip_completed and self._is_no_safe_playtime_result(d)
            try:
                tp = d.get("task_proposal", {})
                if rerun_no_safe:
                    rerun_no_safe_iters.append(iter_num)
                else:
                    if skip_completed:
                        self._skip_completed_iteration_results[iter_num] = d
                        skipped_completed_iters.append(iter_num)
                    else:
                        self._resumed_results.append(d)
                    # Replay task outcomes into the proposer for retained
                    # completed/skipped scenes. No-safe fallback iterations are
                    # intentionally omitted so the rerun starts with the fixed
                    # benchmark anchor, not a stale no-target pseudo-task.
                    self.task_proposer.record_task_outcome(
                        tp.get("activity_name", "unknown"),
                        d.get("success", False),
                        d.get("total_attempts", 0),
                        d.get("skills_learned", []),
                        failure_reason=(
                            d.get("failure_reason")
                            or d.get("failure_category")
                            or d.get("_env_error")
                            or ""
                        ),
                        env_created=bool(d.get("_env_created", True)),
                        language=tp.get("language", tp.get("goal_conditions", "")),
                        objects_used=tp.get("objects", []),
                        fixtures_used=tp.get("fixtures", []),
                        goal_predicates=tp.get("goal", []),
                        play_mode=bool(tp.get("play_mode", False)),
                        play_verb=tp.get("play_verb", ""),
                        play_target=tp.get("play_target", ""),
                        playtime_metadata=tp.get("_playtime") or {},
                    )
                if d.get("success"):
                    activity = tp.get("activity_name") or tp.get("canonical_task_id")
                    code = _latest_code_attempt(d)
                    if activity and code:
                        self._successful_code[str(activity)] = code
                # Copy iteration file to new output dir
                dst = self.output_dir / f"iteration_{iter_num:03d}.json"
                if not rerun_no_safe and not dst.exists():
                    dst.write_text(raw_text)
            except Exception as e:
                logger.warning(
                    "Could not replay resumed iteration %03d from %s: %s",
                    iter_num,
                    prev_dir,
                    e,
                )

        # Copy skill library from previous run (skip copy if resuming into the
        # same directory — in that case the file IS already at run_skills and
        # copying would raise SameFileError).
        prev_skills = prev_dir / "skills.json"
        run_skills = self.output_dir / "skills.json"
        if prev_skills.exists():
            if prev_skills.resolve() != run_skills.resolve():
                import shutil
                shutil.copy2(prev_skills, run_skills)
            self.skill_library = SkillLibrary(storage_path=str(run_skills))
            logger.info(f"Loaded skill library: {len(self.skill_library.get_all_skill_names())} skills")

        # Merge failure memory from previous run
        prev_fm = prev_dir / "failure_memory"
        if prev_fm.exists():
            merged = self.failure_memory.merge_from(str(prev_fm))
            logger.info(f"Merged {merged} failure episodes from previous run")

        completed_iters = [item[0] for item in loaded]
        completed_count = len(completed_iters)
        max_completed = max(completed_iters, default=0)

        # Surface likely hung/partial artifacts so the user knows resume is
        # intentionally starting after the last completed iteration file.
        partial_iters: set[int] = set()
        for child in prev_dir.rglob("*"):
            if not child.is_file():
                continue
            for pat in (r"iter(?:ation)?[_-]?(\d+)", r"iteration_(\d+)"):
                m = re.search(pat, child.name)
                if m and int(m.group(1)) > max_completed:
                    partial_iters.add(int(m.group(1)))
        if partial_iters:
            logger.warning(
                "Ignoring partial artifacts from incomplete iteration(s): %s",
                ", ".join(str(i) for i in sorted(partial_iters)),
            )

        if skip_completed:
            self._iteration = 0
            logger.info(
                "Skip-completed resume from %s: %d completed iteration file(s), "
                "%d skipped, %d no-safe playtime iter(s) queued for rerun",
                prev_dir,
                completed_count,
                len(skipped_completed_iters),
                len(rerun_no_safe_iters),
            )
            if rerun_no_safe_iters:
                logger.info(
                    "No-safe playtime iterations to rerun: %s",
                    ", ".join(str(i) for i in rerun_no_safe_iters),
                )
        else:
            self._iteration = max_completed
            logger.info(
                "Resumed from %s: %d completed iteration file(s), next is %d",
                prev_dir,
                completed_count,
                max_completed + 1,
            )
        if skipped:
            logger.warning("Skipped %d malformed iteration file(s): %s", len(skipped), "; ".join(skipped))
        return max_completed

    def run(self, num_iterations: int = 10) -> dict[str, Any]:
        """Run the lifelong learning loop for N iterations.

        Each iteration: propose task -> plan -> write code -> execute -> evaluate -> feedback

        Returns:
            Summary dict with all iteration results and final metrics.
        """
        # _run_one_iteration historically reset the simulator unconditionally
        # at its end to prepare the next task. Record the actual run boundary
        # so the terminal iteration can preserve its final state instead of
        # creating an unregistered post-episode reset.
        self._run_terminal_iteration = int(num_iterations)
        results = list(self._resumed_results)
        iter_times: list[float] = []
        run_started_at = time.monotonic()
        episode_wall_time_seconds = float(
            os.getenv("RATS_EPISODE_WALL_TIME_SECONDS", "0") or 0
        )
        deadline_state = (
            arm_episode_wall_deadline(episode_wall_time_seconds)
            if episode_wall_time_seconds > 0
            else None
        )
        remaining = num_iterations - self._iteration
        logger.info(f"Starting lifelong loop: {remaining} iterations remaining (from {self._iteration + 1} to {num_iterations})")
        if self.web_debugger is not None:
            self.web_debugger.status(
                "RATS lifelong loop starting",
                details=f"Iterations remaining: `{remaining}`\n\nOutput: `{self.output_dir}`",
            )

        # Pre-validate which tasks can actually be rebound in this scene
        if not self.fixed_task and self._validated_tasks is None:
            if self.env_type == "libero":
                # LIBERO: all tasks in the current suite are valid (env recreation)
                self._init_libero_validated_tasks()
            elif self.env_type == "molmospaces":
                self._init_molmospaces_validated_tasks()
            elif self.env_type == "behavior":
                # BEHAVIOR: lightweight prompt-only rebinding — all scene-compatible
                # tasks are valid since we don't touch OmniGibson's task system.
                # Try multiple sources for scene name
                low_level = getattr(self.env, "low_level_env", self.env)
                og_env = getattr(low_level, "env", None)
                scene_name = None
                if og_env:
                    task_obj = getattr(og_env, "task", None)
                    scene_name = getattr(task_obj, "scene_name", None) if task_obj else None
                    if not scene_name:
                        scene_obj = getattr(og_env, "scene", None)
                        scene_name = getattr(scene_obj, "scene_model", None) if scene_obj else None
                if scene_name:
                    scene_tasks = _load_scene_compatible_tasks(scene_name)
                    self._validated_tasks = scene_tasks if scene_tasks else None
                if not self._validated_tasks:
                    # Fallback: use all tasks from proposer's available list
                    self._validated_tasks = [t["activity_name"] for t in self.task_proposer._available_tasks]
                if not self._validated_tasks:
                    self._validated_tasks = [self._bootstrap_activity]
                logger.info(f"BEHAVIOR exploration: {len(self._validated_tasks)} tasks validated (prompt-only rebind, scene={scene_name})")
            else:
                self._validate_available_tasks()

        start_iter = self._iteration
        for i in range(start_iter, num_iterations):
            if self.web_debugger is not None:
                self.web_debugger.raise_if_stopped()
            self._iteration = i + 1
            # Play mode is permanent for the run (no warm-up flip). The
            # task proposer's mode is set once at startup; nothing changes
            # mid-loop. MolmoSpaces playtime continues to be controlled by
            # the MolmoSpaces proposer config and proposal metadata.
            # ETA calculation
            if iter_times:
                avg_time = sum(iter_times) / len(iter_times)
                remaining = num_iterations - i
                eta_s = avg_time * remaining
                eta_str = f"  ETA: {eta_s/60:.1f}min ({avg_time:.0f}s/iter)"
            else:
                eta_str = ""
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {self._iteration}/{num_iterations}{eta_str}")
            logger.info(f"{'='*60}")
            if self._iteration in self._skip_completed_iteration_results:
                result = self._skip_completed_iteration_results[self._iteration]
                results.append(result)
                logger.info(
                    "Skipping completed iteration %03d from resume cache",
                    self._iteration,
                )
                continue
            if self.web_debugger is not None:
                self.web_debugger.status(
                    f"RATS iteration {self._iteration}/{num_iterations}",
                    details=eta_str.strip() or None,
                    running=True,
                )

            try:
                result = self._run_one_iteration()
                # The policy has returned. Artifact serialization, plotting,
                # and report generation are measured separately and must not
                # retroactively turn an on-time policy result into a timeout.
                restore_episode_wall_deadline(deadline_state)
                deadline_state = None
                results.append(result)
                iter_times.append(result.get("elapsed_seconds", 0))
                self._save_iteration_result(result)
                self._maybe_snapshot()
                self._log_iteration_summary(result)
            except RATSEpisodeWallTimeExceeded as e:
                # Policy time is over. Disarm before serializing potentially
                # large videos so artifact I/O is not mistaken for policy
                # runtime and cannot be interrupted by the periodic alarm.
                restore_episode_wall_deadline(deadline_state)
                deadline_state = None
                logger.error(str(e))
                timeout_media: dict[str, Any] = {}
                low_level = getattr(self.env, "low_level_env", self.env)
                try:
                    timeout_media = self._save_attempt_videos(
                        low_level=low_level,
                        attempt_in_iter=0,
                        attempt=None,
                        status="timeout",
                    )
                except Exception as video_exc:
                    timeout_media = {
                        "video_error": f"{type(video_exc).__name__}: {video_exc}"
                    }

                # A deadline can fire during a long first model request before
                # any motion frames exist. Retain a one-frame final-state video
                # rather than accepting an evidence-less timeout episode.
                if not any(self.output_dir.rglob("*.mp4")):
                    try:
                        import imageio

                        render = getattr(self.env, "render", None)
                        frame = render(mode="rgb_array") if callable(render) else None
                        if frame is not None:
                            fallback_path = (
                                self.output_dir
                                / f"iter{self._iteration:03d}_timeout_state.mp4"
                            )
                            imageio.mimsave(str(fallback_path), [frame], fps=1)
                            timeout_media["fallback_video_path"] = str(fallback_path)
                    except Exception as fallback_exc:
                        timeout_media["fallback_video_error"] = (
                            f"{type(fallback_exc).__name__}: {fallback_exc}"
                        )

                # Agent-I/O files are normally committed after a model call
                # returns. If the very first request is still in flight at the
                # deadline, preserve that fact explicitly rather than either
                # failing the evidence gate or inventing token usage / an
                # actual routed model identity that the provider never sent.
                agent_io_raw = os.getenv("RATS_AGENT_IO_DIR", "").strip()
                if agent_io_raw:
                    agent_io_dir = Path(agent_io_raw)
                    agent_io_dir.mkdir(parents=True, exist_ok=True)
                    if not any(agent_io_dir.glob("*.json")):
                        timeout_call = {
                            "schema_version": 1,
                            "index": 0,
                            "caller": "episode_deadline.in_flight_request",
                            "model": os.getenv("RATS_LLM_MODEL", ""),
                            "actual_model": None,
                            "request_id": None,
                            "episode_key": os.getenv("RATS_EPISODE_KEY", ""),
                            "timestamp": time.time(),
                            "elapsed_s": episode_wall_time_seconds,
                            "request": {
                                "status": "in_flight_at_episode_deadline"
                            },
                            "response": {
                                "content": None,
                                "reasoning": None,
                                "usage": None,
                                "finish_reason": "episode_wall_time_exceeded",
                            },
                            "usage_qualification": (
                                "Provider response was not received before the "
                                "registered deadline; token usage and actual "
                                "routed model are unknown, not zero."
                            ),
                        }
                        timeout_call_path = (
                            agent_io_dir / "0000_episode_deadline.json"
                        )
                        timeout_call_path.write_text(
                            json.dumps(timeout_call, indent=2),
                            encoding="utf-8",
                        )
                        timeout_media["in_flight_model_telemetry"] = str(
                            timeout_call_path
                        )

                timeout_result = {
                    "iteration": self._iteration,
                    "success": False,
                    "timed_out": True,
                    "wall_time_limit_seconds": episode_wall_time_seconds,
                    "elapsed_seconds": episode_wall_time_seconds,
                    "artifact_elapsed_seconds_at_flush": max(
                        0.0, time.monotonic() - run_started_at
                    ),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "feedback_action": "timeout",
                    "total_attempts": 1,
                    "task_proposal": {
                        "activity_name": self._bootstrap_activity or "registered_task"
                    },
                    "timeout_media": timeout_media,
                }
                results.append(timeout_result)
                self._save_iteration_result(timeout_result)
                self._log_iteration_summary(timeout_result)
                break
            except Exception as e:
                logger.exception(
                    "Iteration %s failed with exception: %s",
                    self._iteration,
                    e,
                )
                if self.web_debugger is not None:
                    self.web_debugger.error(f"Iteration {self._iteration} failed: {type(e).__name__}: {e}")
                error_result = {
                    "iteration": self._iteration,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "success": False,
                }
                results.append(error_result)
                try:
                    self._save_iteration_result(error_result)
                except Exception as save_exc:
                    logger.debug(f"  failed to save error iteration result: {save_exc}")

        restore_episode_wall_deadline(deadline_state)
        summary = self._build_summary(results)
        self._save_summary(summary)
        if self.web_debugger is not None:
            self.web_debugger.complete(summary)
        return summary

    def _run_one_iteration(self) -> dict[str, Any]:
        """Run a single iteration of the lifelong loop."""
        start_time = time.time()
        iteration_data: dict[str, Any] = {"iteration": self._iteration}
        learned_count_before = self.skill_library.get_learned_skill_count()
        selected_queue_task_id: str | None = None
        selected_queue_active = False

        def _queue_summary(entry: dict[str, Any] | None) -> dict[str, Any] | None:
            if not entry:
                return None
            return {
                "task_id": entry.get("task_id"),
                "language": entry.get("language"),
                "novelty": entry.get("novelty"),
                "learnability": entry.get("learnability"),
                "base_score": entry.get("base_score"),
                "predicted_success_probability": entry.get("predicted_success_probability"),
                "predicted_bottleneck_step": entry.get("predicted_bottleneck_step"),
                "prediction_reasoning": entry.get("prediction_reasoning"),
                "surprise_score": entry.get("surprise_score"),
                "failure_count": entry.get("failure_count"),
                "penalty": entry.get("penalty"),
                "current_score": entry.get("current_score"),
                "first_proposed_iteration": entry.get("first_proposed_iteration"),
                "last_attempt_iteration": entry.get("last_attempt_iteration"),
                "times_selected": entry.get("times_selected"),
            }

        # ---- 0. Forced periodic house switch (MolmoSpaces only) ----
        # Must run BEFORE _get_scene_context so the proposer reads the
        # NEW house's inventory and proposes a target that the executor
        # will actually run against. If the switch happened later
        # (after the proposer had already chosen a target from the OLD
        # house), the proposer↔executor task would mismatch.
        if self.env_type == "molmospaces":
            # Scene-first playtime: deterministic shuffled-sequential
            # walk over benchmark.json scenes. Takes precedence over the
            # legacy modulo-N periodic switch, so the per-iter scene is
            # purely a function of (iter_idx, scene_pool_seed) rather
            # than whatever the LLM proposer would have asked for.
            scene_first_switch = self._pick_next_playtime_scene()
            if scene_first_switch is not None:
                iteration_data["_forced_house_switch"] = scene_first_switch
                self._queue_molmospaces_environment_check("scene_first_switch")
            else:
                forced_switch = self._maybe_force_periodic_house_switch()
                if forced_switch:
                    iteration_data["_forced_house_switch"] = forced_switch
                    self._queue_molmospaces_environment_check("periodic_house_switch")

        # ---- 1. Get scene context ----
        scene_context = self._get_scene_context()
        if self.env_type == "libero":
            seed_offset = int(os.getenv("RATS_SEED_OFFSET", "0") or 0)
            os.environ["RATS_EPISODE_KEY"] = resolve_libero_episode_key(
                scene_context, self._iteration, seed_offset
            )
        iteration_data["scene_context"] = {
            "scene_model": scene_context.get("scene_model", "unknown"),
            "activity_name": scene_context.get("activity_name"),
        }
        if self.web_debugger is not None:
            self.web_debugger.capture_env_frame(
                self.env,
                f"Iteration {self._iteration} initial observation",
            )

        # ---- 2. Task Proposer ----
        current_activity = scene_context.get("activity_name") or "unknown_task"
        current_scene = scene_context.get("scene_model", "unknown")

        # Remember the bootstrap activity for restoration after failed rebinds
        if self._bootstrap_activity is None:
            self._bootstrap_activity = current_activity

        if self.fixed_task:
            # Fixed task mode: use the task already loaded in the environment
            task_proposal = self._build_proposal_from_env(current_activity, current_scene, scene_context)
            logger.info(f"Step 1: Fixed task mode - using: {current_activity}")
        elif (
            self.novel_tasks
            and self.env_creator is not None
            and self._candidate_mode == "formula"
            and self._iteration > self._curiosity_warmup_iters
        ):
            logger.info(
                "Step 1: Candidate-Based Curiosity Selection "
                "(mode=%s)", self._candidate_mode,
            )
            task_proposal = self._run_candidate_selection(
                scene_context=scene_context,
                current_activity=current_activity,
                iteration_data=iteration_data,
            )
            # Environment Creator: generate BDDL and instantiate env.
            try:
                result = self.env_creator.create_from_proposal(
                    task_proposal, old_env=self.env,
                )
                self.env = result["env"]
                self._set_api_output_dir(self.env)
                scene_context = result["scene_context"]
                task_proposal["activity_name"] = result["activity_name"]
                if result.get("bddl_path"):
                    scene_context["bddl_path"] = result["bddl_path"]
                if result.get("custom_verifier_code"):
                    task_proposal["custom_verifier_code"] = result["custom_verifier_code"]
                    self._remember_active_custom_verifier(
                        result["custom_verifier_code"],
                        task_proposal.get("goal_conditions")
                        or task_proposal.get("language")
                        or scene_context.get("goal_conditions_nl", ""),
                    )
                iteration_data["_env_created"] = True
                logger.info(f"  Created novel env: {result.get('bddl_path', 'N/A')}")
                self._reset_env()
            except Exception as e:
                logger.warning(f"  Novel env creation failed: {e}")
                logger.info("  Falling back to current env")
                iteration_data["_env_created"] = False
                iteration_data["_env_error"] = str(e)
                task_proposal = self._build_proposal_from_env(
                    current_activity, current_scene, scene_context,
                )
                self._reset_env()
        elif (
            self.novel_tasks
            and self.env_creator is not None
            and self._curiosity_enabled
            and self._candidate_mode == "none"
        ):
            # Novel task mode: proposer generates a candidate batch, the
            # task queue keeps the strongest unresolved tasks, and the top
            # queued task becomes the current execution target.
            logger.info("Step 1: Novel Task Proposal + Environment Creation")

            skill_context = self.skill_library.get_context_for_task_proposer()
            candidate_proposals = self.task_proposer.propose_novel_candidates(
                dict(scene_context, current_activity=current_activity),
                skill_context,
                num_candidates=5,
            )
            inserted_entries = self.task_queue.insert_candidates(
                candidate_proposals,
                iteration=self._iteration,
                top_k=3,
            )
            selected_entry = self.task_queue.select_top()
            if selected_entry is None:
                fallback_proposal = self.task_proposer.propose(
                    dict(scene_context, current_activity=current_activity),
                    skill_context,
                )
                inserted_entries = self.task_queue.insert_candidates(
                    [fallback_proposal],
                    iteration=self._iteration,
                    top_k=1,
                )
                selected_entry = self.task_queue.select_top()
            if selected_entry is None:
                raise RuntimeError("task queue is empty after candidate generation")

            selected_queue_active = True
            selected_queue_task_id = str(selected_entry.get("task_id") or "")
            selected_entry = self.task_queue.mark_selected(
                selected_queue_task_id, self._iteration,
            ) or selected_entry
            task_proposal = dict(selected_entry.get("task_spec") or {})
            for key in (
                "novelty", "learnability", "base_score", "surprise_score",
                "failure_count", "penalty", "current_score",
                "first_proposed_iteration", "last_attempt_iteration",
                "times_selected",
            ):
                if key in selected_entry:
                    task_proposal[key] = selected_entry[key]
            logger.info(
                f"  Selected from queue: {task_proposal.get('language', task_proposal.get('activity_name', '?'))} "
                f"(score={selected_entry.get('current_score', 0):.3f})"
            )
            iteration_data["task_queue"] = {
                "generated_candidates": [
                    {
                        "language": c.get("language", c.get("activity_name", "")),
                        "novelty": c.get("novelty"),
                        "learnability": c.get("learnability"),
                        "base_score": c.get("curiosity_score", c.get("base_score")),
                    }
                    for c in candidate_proposals
                ],
                "inserted": [_queue_summary(e) for e in inserted_entries],
                "selected": _queue_summary(selected_entry),
                "queue_size": len(self.task_queue),
                "top_snapshot": self.task_queue.top_snapshot(),
            }

            # Environment Creator: generate BDDL and instantiate env
            try:
                result = self.env_creator.create_from_proposal(
                    task_proposal, old_env=self.env,
                )
                self.env = result["env"]
                self._set_api_output_dir(self.env)
                scene_context = result["scene_context"]
                task_proposal["activity_name"] = result["activity_name"]
                if result.get("bddl_path"):
                    scene_context["bddl_path"] = result["bddl_path"]
                if result.get("custom_verifier_code"):
                    task_proposal["custom_verifier_code"] = result["custom_verifier_code"]
                    self._remember_active_custom_verifier(
                        result["custom_verifier_code"],
                        task_proposal.get("goal_conditions")
                        or task_proposal.get("language")
                        or scene_context.get("goal_conditions_nl", ""),
                    )
                iteration_data["_env_created"] = True
                logger.info(f"  Created novel env: {result.get('bddl_path', 'N/A')}")
                self._reset_env()
            except Exception as e:
                logger.warning(f"  Novel env creation failed: {e}")
                logger.info("  Falling back to current env")
                iteration_data["_env_created"] = False
                iteration_data["_env_error"] = str(e)
                task_proposal = self._build_proposal_from_env(
                    current_activity, current_scene, scene_context,
                )
                # Reset the (still-alive) env before policy runs. Without
                # this, the sim state persists from the previous iter's
                # terminal frame, so any goal predicate that the previous
                # iter satisfied (mug-on-plate, milk-on-bowl, microwave-
                # open, etc.) is STILL true at this iter's "initial" state.
                # The verifier then short-circuits to ✅ even if policy
                # never touched anything. Observed in libero_main_30iter
                # iter 26-28 (three consecutive fake "milk on black bowl"
                # successes inherited from iter 25's terminal state).
                self._reset_env()
        elif self.novel_tasks and self.env_creator is not None:
            if (
                self._candidate_mode == "formula"
                and self._iteration <= self._curiosity_warmup_iters
            ):
                logger.info(
                    "Step 1: Novel Task Proposal (WARMUP %d/%d — candidate "
                    "mode '%s' deferred until iter %d)",
                    self._iteration, self._curiosity_warmup_iters,
                    self._candidate_mode, self._curiosity_warmup_iters + 1,
                )
            else:
                logger.info("Step 1: Novel Task Proposal + Environment Creation")

            skill_context = self.skill_library.get_context_for_task_proposer()
            task_proposal = self.task_proposer.propose(
                dict(scene_context, current_activity=current_activity),
                skill_context,
            )
            logger.info(
                f"  Proposed novel task: {task_proposal.get('language', task_proposal.get('activity_name', '?'))}"
            )

            try:
                result = self.env_creator.create_from_proposal(
                    task_proposal, old_env=self.env,
                )
                self.env = result["env"]
                self._set_api_output_dir(self.env)
                scene_context = result["scene_context"]
                task_proposal["activity_name"] = result["activity_name"]
                if result.get("bddl_path"):
                    scene_context["bddl_path"] = result["bddl_path"]
                if result.get("custom_verifier_code"):
                    task_proposal["custom_verifier_code"] = result["custom_verifier_code"]
                    self._remember_active_custom_verifier(
                        result["custom_verifier_code"],
                        task_proposal.get("goal_conditions")
                        or task_proposal.get("language")
                        or scene_context.get("goal_conditions_nl", ""),
                    )
                iteration_data["_env_created"] = True
                logger.info(f"  Created novel env: {result.get('bddl_path', 'N/A')}")
                self._reset_env()
            except Exception as e:
                logger.warning(f"  Novel env creation failed: {e}")
                logger.info("  Falling back to current env")
                iteration_data["_env_created"] = False
                iteration_data["_env_error"] = str(e)
                task_proposal = self._build_proposal_from_env(
                    current_activity, current_scene, scene_context,
                )
                # See companion fallback above (curiosity branch): without
                # this reset the verifier inherits the previous iter's
                # terminal sim state and may short-circuit to ✅ on a
                # predicate that was satisfied by the LAST iter, not this
                # one. Reset returns the (reused) env to its BDDL :init
                # state so the verifier judges what THIS iter actually did.
                self._reset_env()
        else:
            molmospaces_proposer_mode = str(
                self._molmospaces_cfg.get("proposer_mode", "catalog")
            )
            if self.env_type == "molmospaces" and molmospaces_proposer_mode == "open":
                logger.info("Step 1: Task Proposal (MolmoSpaces open inventory)")
            elif self.env_type == "molmospaces" and molmospaces_proposer_mode == "playtime":
                logger.info("Step 1: Task Proposal (MolmoSpaces Piaget playtime)")
            else:
                logger.info("Step 1: Task Proposal (catalog)")

            # Give Task Proposer only tasks that are validated rebindable.
            # For MolmoSpaces, preserve full metadata (language, objects, etc.)
            # so the proposer can build a catalog for the LLM.
            available = self._validated_tasks or [current_activity]
            if self.env_type in {"molmospaces", "libero"} and self._available_tasks_full:
                available_ids = set(available)
                self.task_proposer.set_available_tasks([
                    t for t in self._available_tasks_full
                    if t.get("canonical_task_id", t.get("activity_name", "")) in available_ids
                ])
            else:
                self.task_proposer.set_available_tasks([
                    {"activity_name": t, "scene_model": current_scene}
                    for t in available
                ])
            if self.env_type == "molmospaces" and molmospaces_proposer_mode == "open":
                logger.info(
                    "  %d bridge task descriptor(s) available for fallback; "
                    "open proposer will use live inventory",
                    len(available),
                )
            else:
                logger.info(f"  {len(available)} validated tasks available")

            # Tell proposer what's currently loaded
            scene_context_for_proposer = dict(scene_context)
            scene_context_for_proposer["current_activity"] = current_activity
            # Open-mode MolmoSpaces proposer needs the live env to pull
            # the inventory + agentview frame for VLM grounding. Catalog
            # / LIBERO paths ignore this key.
            if self.env_type == "molmospaces":
                scene_context_for_proposer["_env"] = self.env

            skill_context = self.skill_library.get_context_for_task_proposer()
            # Unified --play-mode no longer wires PlaytimeMemory into the
            # MolmoSpaces playtime proposer. The proposer's prompt also
            # dropped the ``{playtime_memory}`` placeholder. Past playtime
            # observations are not surfaced as proposer-prompt context
            # anymore — both libero play and molmospaces playtime see
            # the same skill_library + task_history channels.
            #
            # LIBERO-aligned curiosity scoring (only used when playtime
            # proposer is active). When BOTH a Wilson-LB skill lookup
            # and an (object, skill) attempt-count dict are populated,
            # ``_compute_playtime_curiosity`` switches its base score
            # from the legacy ``info_gain`` weighted sum to LIBERO's
            # ``novelty * frontier`` (see agents/curiosity_scoring.py).
            # These keys are skipped at prompt-render time (see
            # ``_propose_novel_molmospaces_playtime`` which strips
            # ``_*`` keys before json.dumps).
            try:
                from rats.agents.curiosity_scoring import make_skill_lookup as _make_sl
                skill_context["_skill_lookup"] = _make_sl(self.skill_library)
            except Exception:
                pass
            obj_skill_counts = getattr(self, "_obj_skill_counts", None)
            if obj_skill_counts is not None:
                skill_context["_object_skill_counts"] = obj_skill_counts
            task_proposal = self.task_proposer.propose(
                scene_context_for_proposer, skill_context,
            )
            proposed_activity = task_proposal["activity_name"]
            logger.info(f"  Proposed task: {proposed_activity}")
            if task_proposal.get("_molmospaces_proposer_mode") == "open":
                logger.info(
                    "  MolmoSpaces open spec: %s",
                    task_proposal.get("_molmospaces_open_spec")
                    or {"request_house_switch": task_proposal.get("_request_house_switch")},
                )
                artifact_paths = self._save_molmospaces_open_proposal_artifacts(
                    task_proposal,
                    trace=getattr(self.task_proposer, "last_proposal_trace", None),
                )
                if artifact_paths:
                    iteration_data["task_proposal_artifacts"] = artifact_paths
            elif task_proposal.get("_molmospaces_proposer_mode") == "playtime":
                logger.info(
                    "  MolmoSpaces playtime task: %s",
                    task_proposal.get("_playtime") or {},
                )
                artifact_paths = self._save_molmospaces_playtime_proposal_artifacts(
                    task_proposal,
                    trace=getattr(self.task_proposer, "last_proposal_trace", None),
                )
                if artifact_paths:
                    iteration_data["task_proposal_artifacts"] = artifact_paths

            # Rebind / recreate environment for proposed task
            if self.env_type == "libero":
                rebind_status = self._rebind_libero_env(task_proposal)
            elif self.env_type == "molmospaces":
                rebind_status = self._rebind_molmospaces_env(task_proposal)
            elif self.env_type == "behavior":
                # BEHAVIOR: use in-place configure_behavior_task() for same-scene rebinding.
                # Creating a new OmniGibson instance fails ("Simulator must be stopped").
                rebind_status = self._rebind_env_to_task(task_proposal)
            elif self.env_creator is not None:
                rebind_status = self._rebind_via_env_creator(task_proposal)
            else:
                rebind_status = self._rebind_env_to_task(task_proposal)
            if rebind_status == "same_task":
                logger.info(f"  Task {proposed_activity} already loaded, proceeding")
            elif rebind_status == "success":
                scene_context = self._get_scene_context()
                logger.info(
                    "  Rebound to %s",
                    scene_context.get("activity_name") or proposed_activity,
                )
            else:
                # Should rarely happen since we pre-validated, but handle gracefully
                self._failed_rebind_tasks.add(proposed_activity)
                if self._validated_tasks and proposed_activity in self._validated_tasks:
                    self._validated_tasks.remove(proposed_activity)
                logger.warning(f"  Rebind failed for {proposed_activity}, restoring bootstrap")
                scene_context = self._get_scene_context()

            # Rebind can switch houses/tasks (especially MolmoSpaces open-mode
            # house-switch proposals).  Refresh the per-iteration scene fields
            # before building/saving the canonical task proposal; otherwise the
            # JSON records keep the bootstrap scene (e.g. house_0) while the
            # videos and environment are already on the rebound house.
            current_activity = scene_context.get("activity_name") or current_activity
            current_scene = scene_context.get("scene_model") or current_scene
            if self.env_type == "libero":
                seed_offset = int(os.getenv("RATS_SEED_OFFSET", "0") or 0)
                os.environ["RATS_EPISODE_KEY"] = resolve_libero_episode_key(
                    scene_context, self._iteration, seed_offset
                )
            iteration_data["scene_context"] = {
                "scene_model": current_scene,
                "activity_name": current_activity,
            }

            # ALWAYS read goal from the CURRENT env state (task-adaptive, not cached)
            proposer_fields = dict(task_proposal)  # preserve LLM reasoning + metadata

            # Playtime switch_house follow-up. The LLM playtime proposer
            # can bail with ``_request_house_switch: True`` when the
            # initial scene has no usable target (escape valve added in
            # commit a31bab86). The rebind handler above already swapped
            # the scene for us.
            #
            # We MUST re-invoke the LLM playtime proposer on the freshly-
            # rebound scene; the alternative (falling through to
            # ``_build_proposal_from_env``) adopts the bridge's bootstrap
            # CANONICAL benchmark task — which is data leakage when the
            # playtime benchmark overlaps with the eval set, and is also
            # off-script for the planner/policy_writer/verifier that
            # expect a playtime-mode proposal.
            #
            # Loop up to MAX_REPROPOSE_RETRIES times: each attempt that
            # returns another switch_house triggers a fresh rebind +
            # re-propose. If we still don't have a valid playtime task
            # after the retry budget (or the proposer keeps erroring),
            # we raise — outer iter handler catches and marks the iter
            # failed. Crucially, NO canonical-benchmark-task fallback.
            MAX_REPROPOSE_RETRIES = 3
            re_proposed = False
            if (
                self.env_type == "molmospaces"
                and molmospaces_proposer_mode == "playtime"
                and proposer_fields.get("_request_house_switch")
                and rebind_status in ("success", "same_task")
            ):
                last_exc: Exception | None = None
                for attempt in range(MAX_REPROPOSE_RETRIES):
                    try:
                        scene_context_for_proposer = dict(scene_context)
                        scene_context_for_proposer["current_activity"] = current_activity
                        scene_context_for_proposer["_env"] = self.env
                        repropose = self.task_proposer.propose(
                            scene_context_for_proposer,
                            skill_context,
                        )
                    except Exception as exc:
                        last_exc = exc
                        logger.warning(
                            "  Post-switch_house playtime re-proposal "
                            "attempt %d/%d failed (%s: %s)",
                            attempt + 1, MAX_REPROPOSE_RETRIES,
                            type(exc).__name__, exc,
                        )
                        break  # don't keep retrying on hard exception
                    if repropose is None:
                        logger.warning(
                            "  Post-switch_house re-propose attempt %d/%d "
                            "returned None; aborting retry loop",
                            attempt + 1, MAX_REPROPOSE_RETRIES,
                        )
                        break
                    # Persist the re-proposal artifacts alongside the
                    # original switch_house pseudo-proposal so traces show
                    # both halves of the dance.
                    artifact_paths = self._save_molmospaces_playtime_proposal_artifacts(
                        repropose,
                        trace=getattr(self.task_proposer, "last_proposal_trace", None),
                    )
                    if artifact_paths:
                        existing = dict(iteration_data.get("task_proposal_artifacts") or {})
                        key = "post_switch_house_reproposal" if attempt == 0 \
                            else f"post_switch_house_reproposal_{attempt + 1}"
                        existing[key] = artifact_paths
                        iteration_data["task_proposal_artifacts"] = existing
                    if repropose.get("_request_house_switch"):
                        # Proposer asks to switch AGAIN — perform a rebind
                        # and try one more re-propose (up to budget).
                        logger.info(
                            "  Re-propose attempt %d/%d asked for another "
                            "switch_house; switching and trying again",
                            attempt + 1, MAX_REPROPOSE_RETRIES,
                        )
                        rebind_again = self._rebind_molmospaces_env(repropose)
                        if rebind_again not in ("success", "same_task"):
                            logger.warning(
                                "  Switch_house chain rebind failed (%s); "
                                "aborting retry loop",
                                rebind_again,
                            )
                            break
                        scene_context = self._get_scene_context()
                        current_activity = scene_context.get("activity_name") or current_activity
                        current_scene = scene_context.get("scene_model") or current_scene
                        continue
                    # Got a non-switch playtime proposal; bind it.
                    rebind_repropose = self._rebind_molmospaces_env(repropose)
                    if rebind_repropose not in ("success", "same_task"):
                        logger.warning(
                            "  Post-switch_house re-proposal rebind failed "
                            "(%s); aborting retry loop",
                            rebind_repropose,
                        )
                        break
                    task_proposal = repropose
                    proposer_fields = dict(task_proposal)
                    scene_context = self._get_scene_context()
                    current_activity = scene_context.get("activity_name") or current_activity
                    current_scene = scene_context.get("scene_model") or current_scene
                    iteration_data["scene_context"] = {
                        "scene_model": current_scene,
                        "activity_name": current_activity,
                    }
                    re_proposed = True
                    logger.info(
                        "  Post-switch_house playtime re-proposal succeeded "
                        "(attempt %d/%d): %r",
                        attempt + 1, MAX_REPROPOSE_RETRIES,
                        repropose.get("language"),
                    )
                    break
                if not re_proposed:
                    raise RuntimeError(
                        f"Post-switch_house re-propose exhausted "
                        f"{MAX_REPROPOSE_RETRIES} attempts without a valid "
                        f"playtime task. Refusing to fall back to the "
                        f"bridge's canonical benchmark task (would be "
                        f"data leakage). Skipping this iteration. "
                        f"Last exception: {last_exc!r}"
                    )
            # Merge proposer metadata that _build_proposal_from_env doesn't set.
            # After a house-switch proposal, do not copy the pseudo-task's
            # canonical id / task_family / language back onto the real task
            # sampled in the new house; the env descriptor is the source of
            # truth for planning and verification from here on.
            if proposer_fields.get("_request_house_switch"):
                merge_keys = ("mode", "difficulty_estimate", "curiosity_score")
            else:
                merge_keys = (
                    "canonical_task_id", "language", "task_family",
                    "objects", "scene_family", "benchmark", "mode",
                    "difficulty_estimate", "curiosity_score",
                    "_molmospaces_proposer_mode", "_playtime",
                )
            for key in merge_keys:
                if key in proposer_fields and key not in task_proposal:
                    task_proposal[key] = proposer_fields[key]

            # Persist the auto-sampled task that the bridge actually drew
            # after a switch_house. Without this, task_proposals/ and
            # generated_molmospaces_specs/ only ever show the LLM's
            # switch_house pseudo-proposal and not the open/close/pick task
            # that ran on the new house.
            if (
                self.env_type == "molmospaces"
                and proposer_fields.get("_request_house_switch")
                and rebind_status in ("success", "same_task")
            ):
                post_switch_paths = self._save_molmospaces_post_switch_artifacts(
                    task_proposal,
                    scene_context,
                    switch_artifact_paths=iteration_data.get("task_proposal_artifacts"),
                    requested_task_type=proposer_fields.get(
                        "_molmospaces_requested_task_type"
                    ),
                )
                if post_switch_paths:
                    existing = dict(iteration_data.get("task_proposal_artifacts") or {})
                    existing.update(post_switch_paths)
                    iteration_data["task_proposal_artifacts"] = existing

            env_verifier_result = self._verify_molmospaces_environment_if_needed(
                task_proposal,
                scene_context,
                attempt=0,
            )
            if env_verifier_result is not None:
                iteration_data.setdefault("environment_verifier", []).append(env_verifier_result)
            if env_verifier_result is not None and not bool(
                env_verifier_result.get("suitable", True)
            ):
                if (
                    self.env_type == "molmospaces"
                    and molmospaces_proposer_mode == "playtime"
                ):
                    recovered = self._retry_molmospaces_playtime_after_env_rejection(
                        scene_context=scene_context,
                        skill_context=skill_context,
                        rejected_result=env_verifier_result,
                        iteration_data=iteration_data,
                    )
                    if recovered is not None:
                        scene_context = recovered["scene_context"]
                        task_proposal = recovered["task_proposal"]
                        current_activity = scene_context.get("activity_name") or current_activity
                        current_scene = scene_context.get("scene_model") or current_scene
                        iteration_data["scene_context"] = {
                            "scene_model": current_scene,
                            "activity_name": current_activity,
                        }
                elif (
                    self.env_type == "molmospaces"
                    and molmospaces_proposer_mode == "benchmark_order"
                    and bool(
                        (self._molmospaces_cfg.get("environment_verifier") or {}).get(
                            "benchmark_reinit_on_fail", True
                        )
                    )
                ):
                    if self._reinitialize_molmospaces_current_task(task_proposal):
                        scene_context = self._get_scene_context()
                        current_activity = scene_context.get("activity_name") or current_activity
                        current_scene = scene_context.get("scene_model") or current_scene
                        task_proposal = self._build_proposal_from_env(
                            current_activity,
                            current_scene,
                            scene_context,
                            reasoning=task_proposal.get("reasoning", ""),
                            expected_new_skills=task_proposal.get("expected_new_skills", []),
                            novelty_score=task_proposal.get("novelty_score", 0.5),
                        )
                        for key in (
                            "canonical_task_id", "language", "task_family", "objects",
                            "scene_family", "benchmark", "mode", "difficulty_estimate",
                            "curiosity_score",
                        ):
                            if key in proposer_fields and key not in task_proposal:
                                task_proposal[key] = proposer_fields[key]
                        iteration_data["scene_context"] = {
                            "scene_model": current_scene,
                            "activity_name": current_activity,
                        }
                        retry_result = self._verify_molmospaces_environment_if_needed(
                            task_proposal,
                            scene_context,
                            reasons=["benchmark_reinit_after_environment_rejection"],
                            attempt=1,
                        )
                        if retry_result is not None:
                            iteration_data.setdefault("environment_verifier", []).append(
                                retry_result
                            )
                        if retry_result is not None and not bool(
                            retry_result.get("suitable", True)
                        ):
                            self._last_reset_failed = True
                            self._last_reset_error = (
                                "environment verifier rejected benchmark task "
                                "after pinned reinitialization"
                            )
                    else:
                        self._last_reset_failed = True
                        self._last_reset_error = (
                            "environment verifier rejected benchmark task and "
                            "pinned reinitialization failed"
                        )

        iteration_data["task_proposal"] = task_proposal
        if self.web_debugger is not None:
            self.web_debugger.task_proposed(
                self._iteration,
                task_proposal,
                scene_context,
            )

        # ---- 3. Filter out unavailable/forbidden functions from scene context ----
        blocked_fns = set(self.quality_checker._server_blocked.keys())
        # Note: find_object_base_rotate and find_object_torso_rotate are ALLOWED
        # -- they are essential for locating objects before grasping (used by oracle)
        #
        # A blocked primitive (graspnet/SAM3 server down) may still have a
        # privileged-API variant registered under the same name. If it appears
        # in `available_functions`, the API is reachable — don't strip it.
        # Without this guard the privileged LIBERO env loses sample_grasp_pose
        # from both `available_functions` AND its api_docs signature line, even
        # though FrankaLiberoPrivilegedApi.sample_grasp_pose works fine.
        registered_fns = set(scene_context.get("available_functions") or [])
        effectively_blocked = blocked_fns - registered_fns
        if scene_context.get("available_functions"):
            scene_context["available_functions"] = [
                fn for fn in scene_context["available_functions"]
                if fn not in effectively_blocked
            ]
        # Strip forbidden function names from API docs so LLM never sees them.
        # Use word-boundary matching: a substring match removes the function's
        # own signature line (where the name is followed by `(`), AND any other
        # docstring line that happens to mention the blocked name — e.g. a
        # cross-reference inside another primitive's docstring.
        api_docs = scene_context.get("api_docs", "")
        if effectively_blocked and api_docs:
            import re as _re_strip
            patterns = {
                fn: _re_strip.compile(r"\b" + _re_strip.escape(fn) + r"\b")
                for fn in effectively_blocked
            }
            kept_lines: list[str] = []
            for line in api_docs.splitlines():
                if any(p.search(line) for p in patterns.values()):
                    continue
                kept_lines.append(line)
            api_docs = "\n".join(kept_lines)
        scene_context["api_docs"] = api_docs

        # ---- 4. Planner ----
        logger.info("Step 2: Planning")
        all_skills = self.skill_library.get_full_skills_for_planner()
        # Retrieve failure lessons for planner (1B.3)
        task_objects = (
            list(scene_context.get("object_scope", {}).keys())
            or task_proposal.get("objects", [])
        )
        # PlaytimeMemory was used here to inject "prior playtime
        # observations" plus a read-only seed-memory block into
        # scene_context["playtime_context"] for the policy writer.
        # Unified --play-mode dropped this side channel; the policy
        # writer no longer receives a separate playtime-context blob.
        failure_lessons = self.failure_memory.get_lessons_for_planner(
            objects=task_objects,
            current_iteration=self._iteration,
        )
        # Capture the initial agentview RGB so the planner can see the
        # scene before committing to a step order (door closed, clutter,
        # etc.). Reused for refine_plan later in this iteration — the
        # env resets between attempts, so initial state is invariant.
        # Non-privileged: a real robot sees this at deployment too.
        initial_rgb = None
        try:
            low = getattr(self.env, "low_level_env", self.env)
            render = getattr(low, "render", None) or getattr(self.env, "render", None)
            if callable(render):
                try:
                    initial_rgb = render(mode="rgb_array")
                except TypeError:
                    initial_rgb = render()
        except Exception as e:
            logger.debug(f"  initial agentview capture failed: {e}")

        plan = self.planner.plan(
            task_proposal, all_skills, scene_context,
            failure_lessons=failure_lessons,
            initial_rgb=initial_rgb,
        )

        # ---- 4b. Pre-execution plan gate (PlannerVerifier) ----
        # The planner has been observed to misread the agentview image
        # (asserts a state that isn't visible, references objects that
        # aren't in the scene, schedules a place before a pick, etc.).
        # We run a VLM verifier on (initial_rgb, plan). A high-confidence
        # "fail" triggers ONE refine_plan call here, before the policy
        # writer runs. We do not re-verify the refined plan — keeping
        # this a one-shot gate avoids verify/refine oscillation.
        plan_verifier_artifact_dir = self.output_dir / "planner_verifier"
        try:
            planner_verdict = self.planner_verifier.verify_plan(
                task_proposal=task_proposal,
                plan=plan,
                scene_context=scene_context,
                all_skills=all_skills,
                initial_rgb=initial_rgb,
                output_dir=plan_verifier_artifact_dir,
                iteration=self._iteration,
                phase="initial",
            )
        except Exception as e:
            logger.warning(f"  PlannerVerifier crashed: {e}; skipping gate")
            planner_verdict = {
                "enabled": False,
                "verdict": "pass",
                "confidence": 0.0,
                "issues": [],
                "summary_for_refine_plan": "",
                "should_refine": False,
                "error": f"crash: {type(e).__name__}: {e}",
            }
        iteration_data["planner_verifier_initial"] = planner_verdict

        if planner_verdict.get("enabled") and planner_verdict.get("verdict") != "pass":
            logger.info(
                f"  PlannerVerifier verdict={planner_verdict.get('verdict')} "
                f"confidence={planner_verdict.get('confidence'):.2f} "
                f"issues={len(planner_verdict.get('issues') or [])}"
            )

        # Multi-pass verify→refine loop. Was previously a one-shot gate
        # (1 verify + 0-or-1 refine + 1 telemetry verify) on the rationale
        # that verify/refine oscillation would burn LLM calls without
        # progress. In practice 1 refine is too few for plans that need a
        # cascade of small fixes (wrong skill ordering + bad object
        # binding + missing intermediate step all at once); each refine
        # tends to address one issue at a time. Now we keep iterating
        # while the verifier still says ``should_refine`` is True, capped
        # at ``planner_verifier.max_refines`` (default 5). On every
        # iteration we save the refined plan + its post-verify verdict
        # so the trace shows the full progression.
        max_refines = int(
            (self._molmospaces_cfg.get("planner_verifier") or {}).get(
                "max_refines", 5,
            )
            or 5
        )
        refine_count = 0
        current_verdict = planner_verdict
        while (
            current_verdict.get("should_refine")
            and refine_count < max_refines
        ):
            reason = current_verdict.get("summary_for_refine_plan") or ""
            logger.info(
                f"  PlannerVerifier triggering refine_plan "
                f"(pass {refine_count + 1}/{max_refines}). Reason: "
                f"{reason[:160]}"
            )
            try:
                refined = self.planner.refine_plan(
                    old_plan=plan,
                    task_proposal=task_proposal,
                    all_skills=all_skills,
                    scene_context=scene_context,
                    plan_issue_reason=reason,
                    prior_attempts=[],
                    failure_lessons=failure_lessons,
                    initial_rgb=initial_rgb,
                )
            except Exception as e:
                logger.warning(
                    f"  PlannerVerifier-triggered refine_plan failed: {e}; "
                    "keeping current plan"
                )
                break
            if not (refined and refined.get("steps")):
                logger.info(
                    "  refine_plan returned no steps; stopping verify→refine loop"
                )
                break
            # Snapshot the plan we just had so the trace shows the full
            # progression (initial → refined_1 → refined_2 → …). Only
            # snapshot the very-first plan once.
            if refine_count == 0:
                iteration_data["plan_initial_before_verifier"] = plan
            plan = refined
            refine_count += 1
            iteration_data[f"plan_refined_by_verifier_{refine_count}"] = plan
            # ``plan_refined_by_verifier`` keeps pointing at the LATEST
            # refined plan for downstream callers that expect a single key.
            iteration_data["plan_refined_by_verifier"] = plan
            logger.info(
                f"  Plan refined by verifier (pass {refine_count}): "
                f"{len(plan.get('steps', []))} steps"
            )
            # Re-verify the refined plan. Result drives the next loop
            # iteration: if it passes (should_refine=False), we exit; if
            # it still wants refinement, we loop until max_refines.
            try:
                post_verdict = self.planner_verifier.verify_plan(
                    task_proposal=task_proposal,
                    plan=plan,
                    scene_context=scene_context,
                    all_skills=all_skills,
                    initial_rgb=initial_rgb,
                    output_dir=plan_verifier_artifact_dir,
                    iteration=self._iteration,
                    phase=f"refined_{refine_count}",
                )
            except Exception as e:
                logger.debug(
                    f"  post-refine verifier pass {refine_count} failed: {e}"
                )
                break
            iteration_data[f"planner_verifier_refined_{refine_count}"] = post_verdict
            # Keep the legacy single-key alias too.
            iteration_data["planner_verifier_refined"] = post_verdict
            current_verdict = post_verdict
        if refine_count >= max_refines and current_verdict.get("should_refine"):
            logger.info(
                f"  PlannerVerifier hit max_refines ({max_refines}); "
                "proceeding with the latest refined plan even though "
                "verdict still says should_refine"
            )

        if selected_queue_active and selected_queue_task_id and self.task_queue is not None:
            updated_entry = self.task_queue.update_prediction(
                selected_queue_task_id,
                plan.get("prediction_card", {}) or {},
            )
            tq = iteration_data.setdefault("task_queue", {})
            tq["selected"] = _queue_summary(updated_entry or {"task_id": selected_queue_task_id})
            tq["queue_size"] = len(self.task_queue)
            tq["top_snapshot"] = self.task_queue.top_snapshot()
        # Persist the (potentially refined) plan after the gate.
        iteration_data["plan"] = plan
        logger.info(f"  Plan: {len(plan.get('steps', []))} steps")
        if self.web_debugger is not None:
            self.web_debugger.plan_ready(self._iteration, plan)

        # ---- 5. Execute with retry loop ----
        attempt = 0
        retry_feedback = None
        final_result = None
        execution_result: dict[str, Any] = {"success": False, "reward": 0, "task_completed": False}
        diagnosis: dict[str, Any] | None = None
        code = ""
        # FIX (policy_writer multi-attempt history): keep the last N
        # generated codes so the retry prompt can show the writer how the
        # last two attempts looked (in addition to the diagnoser's prose
        # critique of each). Capped at the last 2 — see
        # POLICY_HISTORY_DEPTH below — to keep retry prompts bounded while
        # still letting the writer see ONE attempt before the most recent.
        code_history: list[str] = []
        POLICY_HISTORY_DEPTH = 2
        # Multi-turn decider hand-off: when REGENERATE fires on turn N,
        # the new code lives here so turn N+1's policy_writer step uses
        # it directly instead of issuing a fresh write() call. Cleared
        # the moment it is consumed.
        pending_multi_turn_code: str | None = None
        generated_skill_artifacts: list[dict[str, Any]] = []
        accepted_feedback_skill_names: list[str] = []
        accepted_proposed_skills: list[dict[str, Any]] = []
        rejected_skill_records: list[dict[str, Any]] = []
        successful_called_skills: list[str] = []
        failed_called_skills: list[str] = []
        failed_skill_step = ""
        skill_lifecycle_events: list[dict[str, Any]] = []

        # Accumulator for per-attempt context fed back into the Failure
        # Diagnoser. Lives only for THIS iteration. Each entry is a dict:
        # {attempt_idx, code, policy_feedback, failure_mode,
        #  visual_predicate_status, last_frame}. The diagnoser uses this
        # to avoid re-attributing failure to a sub-behavior that visibly
        # worked in an earlier attempt of the same task (e.g. grasp
        # succeeded in attempt 0 → don't flag grasp in attempt 1 unless
        # the new images show fresh regression).
        prior_attempts_for_diag: list[dict[str, Any]] = []

        # Retrieve failure context for policy writer (1B.2)
        failure_context = self.failure_memory.retrieve_for_policy_writer(
            task_name=task_proposal["activity_name"],
            objects=task_objects,
            current_iteration=self._iteration,
        )

        # Inject successful code from past attempts on the same task
        success_context = ""
        task_name = task_proposal["activity_name"]
        logger.info(f"  Success code cache: {list(self._successful_code.keys())}")
        if task_name in self._successful_code:
            past_code = self._successful_code[task_name]
            logger.info(f"  Injecting past successful code for {task_name} ({len(past_code)} chars)")
            success_context = (
                f"\n--- CODE THAT PREVIOUSLY SUCCEEDED ON THIS TASK ---\n"
                f"The following code completed \"{task_name}\" successfully.\n"
                f"Reuse or adapt it. Note: it uses only primitive API functions.\n"
                f"```python\n{past_code}\n```\n"
            )
        else:
            logger.info(f"  No past successful code for {task_name}")

        # Cross-task reliability evidence: surface which learned skills have
        # actually paid off in past iterations. Pure observational data —
        # no "must-use" language — so the policy writer can gravitate toward
        # skills like verified_pick_with_hold_check on its own. Hidden when
        # no skill has been used yet (nothing useful to show).
        try:
            rel = self.skill_library.get_reliability_summary(top_k=6)
            top = [s for s in rel.get("top_by_wilson", []) if s.get("usage_count", 0) > 0]
            if top:
                tier_counts = rel.get("tier_counts", {}) or {}
                tier_str = ", ".join(
                    f"{t}={n}" for t, n in sorted(tier_counts.items())
                )
                lines = [
                    "",
                    "--- LEARNED SKILL RELIABILITY (from prior iterations in this run) ---",
                    f"Tier counts: {tier_str or '(none yet)'}. "
                    "These are empirical outcomes when each skill was actually "
                    "called in executed code — use as evidence for which "
                    "helpers have held up in practice.",
                    "Top by Wilson lower-bound score:",
                ]
                for s in top:
                    lines.append(
                        f"  - {s['name']} "
                        f"[{s['tier']}] "
                        f"{s['success_count']}/{s['usage_count']} "
                        f"(SR={s['success_rate']:.2f}, "
                        f"Wilson={s['wilson_score']:.2f}) — "
                        f"{(s.get('description') or '')[:100]}"
                    )
                success_context = success_context + "\n".join(lines) + "\n"
        except Exception as e:
            logger.debug(f"  reliability summary injection failed (non-fatal): {e}")

        # Inject planner-selected learned skills first, then ALL OTHER learned
        # skills as a fallback preamble so the policy code never NameErrors on
        # a sibling helper that the planner happened not to list (e.g. utility
        # functions like get_topdown_quaternion that the LLM keeps calling).
        # Planner remains the sole retrieval *owner* for prompt-time selection
        # (CLAUDE.md architecture rule #1); this only affects exec-scope safety.
        learned_skill_defs: list[str] = []
        learned_skill_names: list[str] = []
        _seen_skill_names: set[str] = set()
        for step in plan.get("steps", []):
            for sd in step.get("selected_skill_details", []):
                sname = sd.get("name", "")
                if (not sd.get("is_primitive", False)
                        and sd.get("code")
                        and sname not in _seen_skill_names):
                    _seen_skill_names.add(sname)
                    learned_skill_names.append(sname)
                    learned_skill_defs.append(sd["code"])
        # Fallback inject every remaining learned skill so the LLM's policy can
        # call any sibling helper without NameError, even if the planner did
        # not list it under any step's relevant_skills.
        for s in self.skill_library.get_full_skills_for_planner():
            sname = s.get("name", "")
            if (not s.get("is_primitive", False)
                    and s.get("code")
                    and sname
                    and sname not in _seen_skill_names):
                _seen_skill_names.add(sname)
                learned_skill_defs.append(s["code"])
        # Dependency closure: a wrapper's transitive deps may include
        # tier='deprecated' skills which `get_full_skills_for_planner`
        # filters out by default. Force those into the preamble too —
        # without them the wrapper NameErrors at exec time on a sibling
        # helper that the planner correctly ranked low but the wrapper
        # still needs to call. Includes deprecated skills here only;
        # ranking for prompt-time selection stays the same.
        seed_names = list(_seen_skill_names)
        closure = self.skill_library.collect_dependency_closure(seed_names)
        if closure:
            by_name = {
                s.get("name"): s
                for s in self.skill_library.get_full_skills_for_planner(
                    include_deprecated=True,
                )
            }
            for dep_name in closure:
                if dep_name in _seen_skill_names:
                    continue
                s = by_name.get(dep_name)
                if not s or not s.get("code") or s.get("is_primitive"):
                    continue
                _seen_skill_names.add(dep_name)
                learned_skill_defs.append(s["code"])
                logger.debug(
                    f"  Injected dep '{dep_name}' (tier={s.get('tier')}) "
                    f"required by an in-scope wrapper"
                )
        skill_preamble = "\n\n".join(learned_skill_defs)

        # Alias common LLM typos of skill names to their real definitions so
        # policy code that hits the typo doesn't die at Quality Check.
        # Observed repeatedly: place_object_in_* instead of place_object_into_*
        # (LLM drops the second "to" by analogy with `place_object_in_container`).
        alias_lines = []
        defined = set(_seen_skill_names)
        _aliases = [
            ("place_object_in_container_topdown",          "place_object_into_container_topdown"),
            ("place_object_in_container_with_localization", "place_object_into_container_with_localization"),
        ]
        for typo, real in _aliases:
            if typo not in defined and real in defined:
                alias_lines.append(f"{typo} = {real}")
                defined.add(typo)
        if alias_lines:
            skill_preamble = skill_preamble + "\n\n# --- skill-name aliases (LLM typo fallbacks) ---\n" + "\n".join(alias_lines) + "\n"
            learned_skill_names = list(learned_skill_names) + [a for a, _ in _aliases if a in defined]
            _seen_skill_names.update(a for a, _ in _aliases if a in defined)

        # Runtime instrumentation for learned skills. The older overlay
        # fallback only knew "this turn's code references skill X", so it
        # colored the whole turn. Wrapping the injected learned functions lets
        # execution_logger timestamp the actual function call duration in both
        # web-ui and benchmark/no-web-ui runs.
        instrumented_skill_names = sorted(
            name for name in _seen_skill_names if str(name).isidentifier()
        )
        if instrumented_skill_names:
            skill_iteration_map = self._learned_skill_iteration_map()
            wrap_lines = [
                "# --- RATS learned-skill usage instrumentation ---",
                "try:",
                "    from rats.utils.execution_logger import log_step as _rats_log_step, log_step_update as _rats_log_step_update",
                "except Exception:",
                "    _rats_log_step = None",
                "    _rats_log_step_update = None",
                "def _rats_wrap_learned_skill(_rats_fn, _rats_name, _rats_iter=None):",
                "    import functools as _rats_functools",
                "    @_rats_functools.wraps(_rats_fn)",
                "    def _rats_wrapper(*args, **kwargs):",
                "        if _rats_log_step is not None:",
                "            _rats_log_step(",
                "                'Learned Skill Usage',",
                "                f'Running learned skill: {_rats_name}',",
                "                highlight=True,",
                "                timeline_kind='learned_skill',",
                "                timeline_label=_rats_name,",
                "            )",
                "        try:",
                "            return _rats_fn(*args, **kwargs)",
                "        finally:",
                "            if _rats_log_step_update is not None:",
                "                _rats_log_step_update(",
                "                    text=f'Finished learned skill: {_rats_name}'",
                "                )",
                "    return _rats_wrapper",
            ]
            for name in instrumented_skill_names:
                wrap_lines.extend([
                    f"if callable(globals().get({name!r})):",
                    "    "
                    + f"{name} = _rats_wrap_learned_skill("
                    + f"{name}, {name!r}, {skill_iteration_map.get(name)!r})",
                ])
            skill_preamble = (
                skill_preamble
                + "\n\n"
                + "\n".join(wrap_lines)
                + "\n"
            )

        if learned_skill_names:
            logger.info(f"  Learned skills selected by planner: {learned_skill_names}")
        logger.info(f"  Total learned skills available in exec scope: {len(_seen_skill_names)}")

        # Two-level loop: outer attempt × inner turn. Flat counter `attempt`
        # is now the GLOBAL step index across all attempts × turns; the
        # attempt-in-iteration and turn-in-attempt indices are derived.
        #
        #   step_idx 0..T-1   = attempt 0, turns 0..T-1   (no reset between)
        #   step_idx T..2T-1  = attempt 1, turns 0..T-1   (env reset at boundary)
        #   ...
        #
        # legacy single-shot is the T=1 special case: every step crosses an
        # attempt boundary, env resets every step — same as before.
        total_budget = self._attempts_per_iteration * self._turns_per_attempt
        last_failed_usage_context: dict[str, Any] | None = None
        while attempt < total_budget:
            low_level = getattr(self.env, "low_level_env", self.env)
            turn_in_attempt = attempt % self._turns_per_attempt
            attempt_in_iter = attempt // self._turns_per_attempt
            is_first_turn_of_attempt = (turn_in_attempt == 0)
            is_last_turn_of_attempt = (turn_in_attempt == self._turns_per_attempt - 1)
            # task_in_progress: True iff this turn started on a non-reset env
            # (i.e. carries persisted state from the previous turn within
            # the same attempt). The first turn of every attempt always
            # starts on a fresh reset (iteration-start reset for attempt 0,
            # explicit _reset_env() for later attempts), so it's False
            # there even in nested mode.
            task_in_progress = not is_first_turn_of_attempt

            # 5a. Policy Writer
            if self._turn_mode or self._attempts_per_iteration > 1:
                attempt_label = (
                    f"attempt {attempt_in_iter + 1}/{self._attempts_per_iteration} "
                    f"turn {turn_in_attempt + 1}/{self._turns_per_attempt}"
                )
            else:
                attempt_label = f"attempt {attempt + 1}"

            # ---- Multiturn branch (LIBERO-only, opt-in) ----
            # When enabled, the MultiturnResetExecutor replaces the legacy
            # writer→quality→self-check→executor→per-step block: each plan
            # step runs in isolation with env-reset rollback. Variables that
            # 5a-5d would have populated are pre-set here so the rest of the
            # iteration loop (verifier, diagnoser, feedback generator, skill
            # extraction) runs unchanged on the multiturn output.
            multiturn_reset_active = False
            multiturn_reset_result_dict: dict[str, Any] | None = None
            if (
                self._multiturn_reset_enabled
                and self.multiturn_reset_executor is not None
                and is_first_turn_of_attempt
            ):
                logger.info(
                    f"Step 3-5 (multiturn-reset): step-by-step rollout for {attempt_label}"
                )
                mt_outcome = self._run_attempt_via_multiturn_reset(
                    plan=plan,
                    scene_context=scene_context,
                    skill_preamble=skill_preamble,
                    failure_context=failure_context,
                    success_context=success_context,
                    attempt=attempt,
                    attempt_in_iter=attempt_in_iter,
                    iteration_data=iteration_data,
                    learned_skill_names=list(_seen_skill_names),
                )
                code = mt_outcome["code"]
                exec_code = mt_outcome["exec_code"]
                execution_result = mt_outcome["execution_result"]
                quality = mt_outcome["quality"]
                policy_ready = mt_outcome["policy_ready"]
                per_step_verification = mt_outcome["per_step_verification"]
                multiturn_reset_result_dict = mt_outcome["multiturn_reset_result_dict"]
                multiturn_reset_active = True
                code_history.append(code)
                if len(code_history) > POLICY_HISTORY_DEPTH:
                    code_history = code_history[-POLICY_HISTORY_DEPTH:]

            if not multiturn_reset_active:
                logger.info(f"Step 3: Policy Writing ({attempt_label})")
            if task_in_progress and retry_feedback is not None:
                retry_feedback = dict(retry_feedback)
                retry_feedback["task_in_progress"] = True
                # preserved_code_segments are sub-blocks of a prior turn's
                # code that the diagnoser thinks already succeeded. In
                # single-shot mode (or at attempt boundaries) the env was
                # reset and these need re-running; within an attempt the
                # env state already reflects them, so re-running would
                # re-pick / re-place. Strip.
                retry_feedback["preserved_code_segments"] = []
            elif retry_feedback is not None:
                # Crossing an attempt boundary OR legacy single-shot: env
                # was reset just before this call, so the "task in progress"
                # framing does not apply.
                retry_feedback = dict(retry_feedback)
                retry_feedback["task_in_progress"] = False
            if multiturn_reset_active:
                # Multiturn already built `code` step-by-step; skip writer.
                pass
            elif pending_multi_turn_code is not None:
                # CaP-X-style fast path: the multi-turn decider produced
                # this code on the previous turn. Skip policy_writer
                # entirely so we don't burn an LLM call regenerating
                # from scratch.
                code = pending_multi_turn_code
                pending_multi_turn_code = None
                logger.info(
                    "  Using multi-turn-decider-supplied code "
                    f"({len(code)} chars); skipping PolicyWriter for this turn."
                )
            else:
                code = self.policy_writer.write(
                    plan, scene_context,
                    retry_feedback=retry_feedback,
                    failure_context=failure_context,
                    success_context=success_context,
                )
            if not multiturn_reset_active:
                self._save_policy_writer_retry_artifacts(
                    iteration_data,
                    attempt,
                    retry_feedback,
                    attempt_in_iter=attempt_in_iter,
                    turn_in_attempt=turn_in_attempt,
                    task_in_progress=task_in_progress,
                )
            iteration_data[f"code_attempt_{attempt}"] = code
            # Track up to POLICY_HISTORY_DEPTH generated codes so the
            # NEXT retry's retry_package can include not just the most
            # recent attempt but the one before it. The writer sees
            # patterns it has already tried twice and avoids them.
            code_history.append(code)
            if len(code_history) > POLICY_HISTORY_DEPTH:
                code_history = code_history[-POLICY_HISTORY_DEPTH:]

            # FIX (self-check/Step-5 doubling): Step 4b
            # (`_self_check_and_repair_policy`) runs the policy once to
            # catch crashes, then Step 5 runs it again for the official
            # attempt. Every LLM-using primitive the policy invokes
            # (verify_object_identity, point_prompt_molmo) was being
            # called TWICE per attempt with bit-identical inputs (env
            # reset to same state between the two executions). Open a
            # primitive-cache scope here; the cache clears at the early-
            # exit branches below or right after Step 5 returns.
            attempt_cache = policy_primitive_cache_scope()
            attempt_cache.__enter__()

            # 5b. Quality Checker + optional runtime self-check/self-repair.
            # Runtime self-check resets the environment around a dry run, so
            # skip it when this turn is already carrying state from a previous
            # turn inside the same attempt.
            if self.web_debugger is not None:
                self.web_debugger.raise_if_stopped()
                self.web_debugger.status(
                    "Checking generated policy before execution",
                    details=(
                        f"Iteration {self._iteration}, {attempt_label}: running "
                        "quality checks and the resettable policy runtime self-check. "
                        "This dry run may take a while for MolmoSpaces motion calls."
                    ),
                    running=True,
                )
            if not multiturn_reset_active:
                code, quality, policy_ready = self._self_check_and_repair_policy(
                    code=code,
                    plan=plan,
                    scene_context=scene_context,
                    retry_feedback=retry_feedback,
                    failure_context=failure_context,
                    success_context=success_context,
                    skill_preamble=skill_preamble,
                    learned_skill_names=list(_seen_skill_names),
                    attempt=attempt,
                    iteration_data=iteration_data,
                    runtime_self_check_enabled=not task_in_progress,
                )
            iteration_data[f"code_attempt_{attempt}"] = code

            # In multiturn-reset mode the inline per-step gate already
            # consumed Tier-1 retries against the writer (and either
            # naturally committed after a fix, force-committed safe code,
            # or skipped the step). The aggregated quality_attempt[N]
            # artifact this attempt reports approved=False when ANY
            # retry was blocked — that's informational, not actionable
            # ("rewind the whole attempt"). The legacy quality-fail
            # handler below is for legacy single-shot output and would
            # short-circuit verifier + diagnoser + feedback_generator
            # for an attempt that actually executed end-to-end, so skip
            # it when MT-RESET is active.
            if not multiturn_reset_active and not quality["approved"]:
                logger.warning(f"  Quality check BLOCKED: {quality['feedback']}")
                last_failed_usage_context = None
                if self.env_type == "molmospaces" and self.web_debugger is not None:
                    self.web_debugger.status(
                        "Quality check blocked policy execution",
                        details=str(quality.get("feedback", ""))[:4000],
                    )
                # Tier 1 failure doesn't count as execution attempt
                retry_feedback = {
                    "attempt": attempt + 1,
                    "stderr": quality["feedback"],
                    "diagnosis": "Code rejected by quality checker. Fix the issues.",
                    "failed_step": "quality_check",
                    "previous_code": code,
                }
                # Quality fail does not record frames, but if this was the
                # last turn of an attempt we still want to flush any
                # accumulated frames from prior turns to disk and reset
                # for the next attempt.
                if is_last_turn_of_attempt:
                    self._save_attempt_videos(
                        low_level=getattr(self.env, "low_level_env", self.env),
                        attempt_in_iter=attempt_in_iter,
                        attempt=attempt,
                        status="failed",
                    )
                    if (attempt + 1) < total_budget:
                        self._reset_env()
                # Release primitive-cache scope before `continue` — we
                # never reached Step 5 but the self-check may have
                # populated entries; clear them.
                try:
                    attempt_cache.__exit__(None, None, None)
                except Exception:
                    pass
                attempt += 1
                continue

            if not policy_ready.get("passed", False):
                self_check = policy_ready.get("self_check") or {}
                logger.warning(
                    "  Policy runtime self-check BLOCKED: %s",
                    (self_check.get("stderr_snippet") or "")[:500],
                )
                last_failed_usage_context = None
                if self.env_type == "molmospaces" and self.web_debugger is not None:
                    self.web_debugger.status(
                        "Policy runtime self-check blocked official execution",
                        details=(self_check.get("stderr_snippet") or "")[:4000],
                    )
                retry_feedback = policy_ready.get("retry_feedback") or {
                    "attempt": attempt + 1,
                    "stderr": self_check.get("stderr") or self_check.get("stderr_snippet") or "",
                    "diagnosis": "Code rejected by policy runtime self-check.",
                    "failed_step": "runtime_self_check",
                    "previous_code": code,
                }
                iteration_data[f"policy_self_check_blocked_attempt_{attempt}"] = {
                    "reason": policy_ready.get("reason"),
                    "repairs_used": policy_ready.get("repairs_used", 0),
                    "self_check": self_check,
                }
                if is_last_turn_of_attempt:
                    self._save_attempt_videos(
                        low_level=getattr(self.env, "low_level_env", self.env),
                        attempt_in_iter=attempt_in_iter,
                        attempt=attempt,
                        status="failed",
                    )
                    if (attempt + 1) < total_budget:
                        self._reset_env()
                # Self-check failed before Step 5 — release the cache
                # scope so the next attempt starts fresh.
                try:
                    attempt_cache.__exit__(None, None, None)
                except Exception:
                    pass
                attempt += 1
                continue

            debug_block_idx = None
            if self.web_debugger is not None:
                self.web_debugger.raise_if_stopped()
                debug_block_idx = self.web_debugger.start_attempt(
                    iteration=self._iteration,
                    attempt=attempt,
                    code=code,
                    label="RATS Policy Attempt",
                )
                self.web_debugger.step(
                    debug_block_idx,
                    "Quality Check",
                    quality.get("feedback", "approved"),
                    highlight=not bool(quality.get("approved", False)),
                )

            # 5c. Executor — prepend learned skill definitions so they are in scope
            if not multiturn_reset_active:
                logger.info("Step 5: Execution")
                exec_code = f"{skill_preamble}\n\n{code}" if skill_preamble else code

            # Enable video recording. In nested mode the buffer must
            # persist across turns within an attempt so we can slice
            # per-turn videos and a combined attempt video at the end —
            # only clear it at the start of each attempt's first turn.
            # Multiturn already ran its own video capture per step retry, so
            # skip this block to avoid clearing those frames.
            if not multiturn_reset_active and hasattr(low_level, "enable_video_capture"):
                try:
                    low_level.enable_video_capture(
                        True, clear=is_first_turn_of_attempt,
                    )
                except Exception:
                    pass

            # Track per-turn frame ranges so we can slice the combined
            # attempt buffer into one video per turn at the attempt's end.
            # Reset the list at each attempt boundary.
            if is_first_turn_of_attempt:
                self._attempt_turn_frame_ranges = []
                self._attempt_timeline_events = []
                mark_fn = getattr(self.web_debugger, "mark_viser_recording", None)
                self._attempt_viser_frame_start = (
                    mark_fn() if callable(mark_fn) else None
                )
            recording_frames = hasattr(low_level, "get_video_frame_count")
            turn_frame_start = (
                low_level.get_video_frame_count() if recording_frames else 0
            )

            # If the env's last reset failed (set by _reset_env on a wedged
            # MolmoSpaces bridge after _try_recover_molmospaces_env also
            # could not bring it back), env.step(code) will either raise or
            # emit stale observations from the previous task. Skip the
            # executor call entirely and synthesise an init-failure
            # result; the verifier sees a failed attempt and the iteration
            # records the cause as the bridge error rather than mis-
            # attributing it to the policy code.
            exec_history = None
            if multiturn_reset_active:
                # execution_result already produced by MultiturnResetExecutor;
                # skip the legacy executor.execute path entirely. Video
                # capture / api_logging / exec_history are also skipped —
                # multiturn-reset ran its own per-step videos and the per-step
                # verifier verdicts already drove the rollout decisions.
                pass
            elif getattr(self, "_last_reset_failed", False):
                logger.warning(
                    "  Skipping executor: previous env.reset failed (%s); "
                    "marking attempt as init_failed",
                    getattr(self, "_last_reset_error", "unknown") or "unknown",
                )
                execution_result = {
                    "success": False,
                    "stdout": "",
                    "stderr": (
                        "init_failed: env.reset did not produce a usable "
                        "task state — "
                        f"{getattr(self, '_last_reset_error', 'unknown')}"
                    ),
                    "reward": 0.0,
                    "task_completed": False,
                    "user_result": None,
                    "before_frame": None,
                    "after_frame": None,
                    "before_wrist_frame": None,
                    "after_wrist_frame": None,
                    "terminated": False,
                    "truncated": True,
                    "artifacts": {"init_failed": True},
                }
            else:
                from rats.utils import execution_logger

                frame_provider = (
                    low_level.get_video_frame_count if recording_frames else None
                )
                emit_callback = None
                if self.web_debugger is not None and debug_block_idx is not None:
                    emit_callback = self.web_debugger.execution_step_callback(
                        debug_block_idx,
                    )
                api_logging_states = self._enable_api_execution_logging(self.env)
                execution_logger.init_execution_context(
                    code_block_index=(
                        debug_block_idx if debug_block_idx is not None else attempt
                    ),
                    emit_callback=emit_callback,
                    frame_count_provider=frame_provider,
                    frame_fps=20.0 if recording_frames else None,
                )
                if self.web_debugger is not None and debug_block_idx is not None:
                    self.web_debugger.step(
                        debug_block_idx,
                        "Execution",
                        "Running policy in the RATS environment.",
                        highlight=True,
                        frame_start=turn_frame_start if recording_frames else None,
                        start_s=(
                            round(turn_frame_start / 20.0, 3)
                            if recording_frames
                            else None
                        ),
                        timeline_kind="execution",
                        timeline_label="policy",
                    )
                try:
                    if self.web_debugger is not None:
                        with self.web_debugger.execution_interrupt_scope():
                            execution_result = self.executor.execute(
                                exec_code, self.env, scene_context,
                            )
                    else:
                        execution_result = self.executor.execute(
                            exec_code, self.env, scene_context,
                        )
                finally:
                    exec_history = execution_logger.finalize_execution_context()
                    self._restore_api_execution_logging(api_logging_states)
            # Step 5 done — release the primitive-cache scope opened
            # before Step 4b. From here on the attempt path is
            # verification/diagnosis/feedback, none of which call the
            # cached primitives, so holding the cache open longer just
            # delays freeing the memory.
            try:
                attempt_cache.__exit__(None, None, None)
            except Exception:
                pass
            api_diag_summary = (
                execution_result.get("artifacts", {})
                .get("info", {})
                .get("api_diagnostics_summary")
            )
            if api_diag_summary:
                logger.info(f"  API diagnostics: {api_diag_summary}")
                if self.web_debugger is not None:
                    self.web_debugger.step(
                        debug_block_idx,
                        "API Diagnostics",
                        str(api_diag_summary),
                    )
            if self.web_debugger is not None:
                try:
                    self.web_debugger.publish_env(self.env, reason="after_execution")
                except Exception:
                    pass

            # Record this turn's frame range and sample a filmstrip from
            # ONLY this turn's frames for the diagnoser (so vision LLM
            # judges what THIS turn's code did, not the cumulative
            # attempt). Don't clear / dump the buffer here — it accumulates
            # until the attempt finishes, when _save_attempt_videos slices
            # the combined buffer into per-turn .mp4s + a combined .mp4.
            if recording_frames:
                turn_frame_end = low_level.get_video_frame_count()
                self._attempt_turn_frame_ranges.append(
                    (turn_frame_start, turn_frame_end)
                )
                if exec_history is not None:
                    for step in exec_history.steps:
                        event = step.to_dict()
                        event.update({
                            "iteration": self._iteration,
                            "attempt": attempt_in_iter,
                            "turn": turn_in_attempt,
                            "block_index": debug_block_idx,
                        })
                        self._attempt_timeline_events.append(event)
                try:
                    if turn_frame_end > turn_frame_start and hasattr(
                        low_level, "get_video_frames_range",
                    ):
                        turn_frames = low_level.get_video_frames_range(
                            turn_frame_start, turn_frame_end,
                        )
                        if turn_frames:
                            turn_frames_list = list(turn_frames)
                            execution_result["trajectory_frame_count"] = len(turn_frames_list)
                            execution_result["trajectory_video_frames"] = turn_frames_list
                            execution_result["trajectory_frames"] = (
                                _sample_trajectory_frames(turn_frames_list)
                            )
                            execution_result["vlm_verifier_frames"] = (
                                _sample_vlm_verifier_frames(
                                    turn_frames_list,
                                    min_frames=8,
                                    max_frames=48,
                                )
                            )
                            try:
                                from rats.utils.video_utils import _encode_video_base64

                                video_fps = max(
                                    1,
                                    int(os.environ.get("RATS_DIAGNOSER_VIDEO_FPS", "20") or 20),
                                )
                                execution_result["trajectory_video_data_url"] = (
                                    _encode_video_base64(turn_frames_list, fps=video_fps)
                                )
                                execution_result["trajectory_video_frame_count"] = len(turn_frames_list)
                                execution_result["trajectory_video_fps"] = video_fps
                            except Exception as video_exc:
                                logger.debug(
                                    "  Per-turn trajectory video encoding failed: %s",
                                    video_exc,
                                )
                            if self.env_type in ("molmospaces", "libero"):
                                # `_build_step_frame_segments` is generic — it
                                # filters exec_history.steps on
                                # timeline_kind=="policy_step" which is
                                # written by the env-agnostic
                                # policy_step_context. Libero only gets
                                # those entries when its policy code uses
                                # `with step_context(...)` markers; without
                                # them this just yields [] and per-step
                                # verifier falls back gracefully.
                                execution_result["step_frame_segments"] = (
                                    self._build_step_frame_segments(
                                        exec_history,
                                        turn_frame_start,
                                        turn_frame_end,
                                        turn_frames_list,
                                    )
                                )
                            # Also flatten exec_history's API-kind events so
                            # PerStepVerifier can render them in the
                            # ``MAIN-FUNCTION API CALLS DURING THIS STEP``
                            # prompt section. rats/envs/tasks/base.py only
                            # binds primitives through _wrap_api_function for
                            # MolmoSpaces, so LIBERO's info["api_call_trace"]
                            # stays empty even when SAM3 / goto_pose / etc.
                            # clearly ran. Each event carries policy_step_index,
                            # so _marked_event_step_index will route it.
                            try:
                                from rats.agents.per_step_verifier import (
                                    extract_exec_history_api_events,
                                )
                                api_events = extract_exec_history_api_events(
                                    exec_history,
                                )
                                if api_events:
                                    execution_result["api_timeline_events"] = api_events
                            except Exception as api_exc:
                                logger.debug(
                                    "  api_timeline_events extract failed: %s",
                                    api_exc,
                                )
                except Exception as e:
                    logger.debug(f"  Per-turn filmstrip extraction failed: {e}")
            # LIBERO's task_completed is a numpy.bool_, which breaks
            # json.dumps and causes the whole dict to fall back to str()
            # in _save_iteration_result — masking user_result as a blob.
            # Coerce to native Python types at the persist boundary.
            _tc = execution_result.get("task_completed")
            _rw = execution_result.get("reward")
            execution_history_artifacts = self._save_execution_history_artifacts(
                iteration_data,
                exec_history,
                attempt,
            )
            iteration_data[f"execution_attempt_{attempt}"] = {
                "success": bool(execution_result["success"]),
                "reward": float(_rw) if _rw is not None else None,
                "task_completed": bool(_tc) if _tc is not None else None,
                "stderr_snippet": (execution_result.get("stderr", ""))[:2000],
                "user_result": execution_result.get("user_result"),
                "api_diagnostics_summary": api_diag_summary,
                "api_diagnostics": (
                    execution_result.get("artifacts", {})
                    .get("info", {})
                    .get("api_diagnostics")
                ),
            }
            if execution_history_artifacts:
                iteration_data[f"execution_attempt_{attempt}"][
                    "execution_history"
                ] = execution_history_artifacts

            # ---- CaP-X-style intra-attempt decider (opt-in) ----
            # When the multi-turn decider is enabled AND we are NOT on the
            # last turn of the attempt AND execution didn't hard-fail or
            # wedge the env, we ask a single fast LLM call: FINISH or
            # REGENERATE+code. REGENERATE skips per-step verifier +
            # verifier + diagnoser + feedback for this turn and feeds the
            # new code straight into the next turn's executor. FINISH
            # falls through to the existing pipeline so skill extraction,
            # failure memory, and verifier confirmation still run.
            if (
                self.multi_turn_decider is not None
                and self._turns_per_attempt > 1
                and not is_last_turn_of_attempt
                and execution_result.get("success", False)
                and not execution_result.get("terminated", False)
                and not execution_result.get("truncated", False)
                and not execution_result.get("timeout", False)
            ):
                task_goal_text = resolve_public_task_goal(
                    task_proposal,
                    scene_context,
                )
                logger.info("Step 5b: Multi-turn decider (FINISH vs REGENERATE)")
                decision = self.multi_turn_decider.decide(
                    executed_code=code,
                    stdout=execution_result.get("stdout", ""),
                    stderr=execution_result.get("stderr", ""),
                    after_frame=execution_result.get("after_frame"),
                    task_goal=task_goal_text,
                )
                iteration_data[f"multi_turn_decision_attempt_{attempt}"] = {
                    "action": decision.get("action"),
                    "has_new_code": bool(decision.get("new_code")),
                    "raw_excerpt": (decision.get("raw") or "")[:500],
                }
                logger.info(
                    f"  Multi-turn decider -> {decision.get('action', '?').upper()}"
                )
                if decision.get("action") == "regenerate" and decision.get("new_code"):
                    pending_multi_turn_code = decision["new_code"]
                    # The decider's new code supersedes any leftover
                    # retry feedback from an earlier turn — clear it so
                    # the next turn's quality/self-check don't see stale
                    # diagnoser context that no longer applies.
                    retry_feedback = None
                    # Skip per-step verifier, verifier, diagnoser, and
                    # feedback for this turn. The next turn will pick up
                    # the new code via the policy_writer bypass at the
                    # top of the loop.
                    try:
                        attempt_cache.__exit__(None, None, None)
                    except Exception:
                        pass
                    attempt += 1
                    logger.info(
                        f"  Continuing to turn {turn_in_attempt + 2}/"
                        f"{self._turns_per_attempt} of attempt "
                        f"{attempt_in_iter + 1} with multi-turn-decider "
                        f"code (no env reset; task in progress)..."
                    )
                    continue
                # FINISH (or REGENERATE without parseable code) -> fall
                # through to the heavy pipeline below.

            # Post-execution per-step verification. This is deliberately
            # separate from policy-authored RESULT: the verifier input is the
            # plan step goal plus runtime output artifacts/logs only. It does
            # not inspect generated policy code and does not trust RESULT as an
            # evidence source. With MolmoSpaces, grounded before/after
            # object/joint state is allowed as verifier-only evidence.
            # Skipped when multiturn-reset already produced a per-step verdict.
            if multiturn_reset_active:
                # `per_step_verification` and the iteration_data entry are
                # already populated by _run_attempt_via_multiturn_reset; ensure
                # execution_result also carries the artifact for the
                # diagnoser/feedback generator to consume downstream.
                execution_result.setdefault("artifacts", {})[
                    "per_step_verification"
                ] = per_step_verification
                if per_step_verification.get("summary_text"):
                    logger.info(
                        "  Per-step verification (multiturn-reset): %s",
                        str(per_step_verification.get("summary_text", "")).replace("\n", " | "),
                    )
            else:
                try:
                    per_step_dir = (
                        self.output_dir
                        / f"iteration_{self._iteration:03d}"
                        / f"attempt_{attempt:02d}"
                        / "per_step"
                    )
                    per_step_verification = self.per_step_verifier.verify_attempt(
                        execution_result,
                        plan=plan,
                        output_dir=per_step_dir,
                        iteration=self._iteration,
                        attempt=attempt,
                        attempt_in_iter=attempt_in_iter,
                        turn_in_attempt=turn_in_attempt,
                        code=code,
                    )
                    execution_result.setdefault("artifacts", {})[
                        "per_step_verification"
                    ] = per_step_verification
                    iteration_data[f"per_step_verification_attempt_{attempt}"] = {
                        "enabled": per_step_verification.get("enabled", False),
                        "artifact_dir": per_step_verification.get("artifact_dir"),
                        "summary_text": per_step_verification.get("summary_text"),
                        "steps": per_step_verification.get("steps", []),
                    }
                    if per_step_verification.get("summary_text"):
                        logger.info(
                            "  Per-step verification: %s",
                            str(per_step_verification.get("summary_text", "")).replace("\n", " | "),
                        )
                except Exception as exc:
                    logger.debug("  Per-step verification failed (non-fatal): %s", exc)
                    iteration_data[f"per_step_verification_attempt_{attempt}"] = {
                        "enabled": False,
                        "error": str(exc),
                    }

            # Save before/after frames only if explicitly enabled
            if self._save_debug_frames:
                try:
                    import imageio
                    bf = execution_result.get("before_frame")
                    af = execution_result.get("after_frame")
                    if bf is not None:
                        imageio.imwrite(str(self.output_dir / f"iter{self._iteration:03d}_attempt{attempt}_before.png"), bf)
                    if af is not None:
                        imageio.imwrite(str(self.output_dir / f"iter{self._iteration:03d}_attempt{attempt}_after.png"), af)
                except Exception:
                    pass

            # 5d. Verifier (programmatic predicate check + LLM analysis)
            logger.info("Step 6: Verification")
            # Build short attempt-history summary so the verifier's LLM doesn't
            # propose a fix that was already tried on a prior attempt of THIS
            # iteration.
            attempt_history: list[dict[str, Any]] = []
            for prev in range(attempt):
                v_prev = iteration_data.get(f"verification_attempt_{prev}") or {}
                d_prev = iteration_data.get(f"diagnosis_attempt_{prev}") or {}
                attempt_history.append({
                    "attempt": prev,
                    "success": v_prev.get("success"),
                    "unsatisfied_conditions": v_prev.get("unsatisfied_conditions"),
                    "failure_mode": d_prev.get("failure_mode"),
                })
            # Pull a short lessons snippet for the verifier so it can cross-
            # check its proposed fix against already-known patterns (avoid
            # producing a duplicate lesson, and avoid re-proposing a remedy
            # that's already in the library as a known-to-fail antipattern).
            fm_view = ""
            try:
                fm_view = self.failure_memory.get_lessons_for_planner(
                    objects=task_objects, top_k=4,
                    current_iteration=self._iteration,
                )
            except Exception:
                fm_view = ""
            verification = self.verifier.verify(
                execution_result, task_proposal,
                env=self.env, code=code, attempt_history=attempt_history,
                failure_memory_view=fm_view,
                plan=plan,
                artifact_dir=self.output_dir / "verifier_artifacts",
                artifact_prefix=(
                    f"iter{self._iteration:03d}_attempt{attempt_in_iter:02d}"
                    f"_turn{turn_in_attempt:02d}_step{attempt:02d}"
                ),
            )
            logger.info(f"  Verified: {verification['success']}")
            if self.web_debugger is not None:
                self.web_debugger.verification(debug_block_idx, verification)

            if self.env_type == "molmospaces":
                # MolmoSpaces keeps per-attempt skill timeline events for the
                # web/Viser replay path. Keep this branch separate so LIBERO
                # retains origin/main's final-success-only accounting below.
                try:
                    learned_name_set = {
                        s["name"]
                        for s in self.skill_library.get_full_skills_for_planner(
                            include_deprecated=True,
                        )
                        if not s.get("is_primitive", False)
                    }
                    called = _extract_called_learned_skills(code, learned_name_set)
                    if called:
                        if verification.get("success"):
                            successful_called_skills = called
                        skill_lifecycle_events.extend(
                            self.skill_library.record_usage(
                                called,
                                success=bool(verification.get("success")),
                                iteration=self._iteration,
                                source="molmospaces_attempt_code",
                            )
                        )
                        logger.info(
                            f"  Skill usage recorded ({'✓' if verification.get('success') else '✗'}): "
                            f"{', '.join(called[:8])}"
                            + (f" (+{len(called)-8} more)" if len(called) > 8 else "")
                        )
                        skill_label = ", ".join(called[:4]) + (
                            f" +{len(called) - 4}" if len(called) > 4 else ""
                        )
                        start_frame = turn_frame_start if recording_frames else None
                        end_frame = turn_frame_end if recording_frames else None
                        timed_event_exists = any(
                            event.get("timeline_kind") == "learned_skill"
                            and event.get("attempt") == attempt_in_iter
                            and event.get("turn") == turn_in_attempt
                            for event in self._attempt_timeline_events
                            if isinstance(event, dict)
                        )
                        if timed_event_exists:
                            timeline_event = {"start_s": None, "end_s": None}
                        else:
                            timeline_event = self._append_learned_skill_timeline_event(
                                called,
                                start_frame=start_frame,
                                end_frame=end_frame,
                                attempt=attempt_in_iter,
                                turn=turn_in_attempt,
                                block_index=debug_block_idx,
                            )
                        if self.web_debugger is not None and not timed_event_exists:
                            self.web_debugger.step(
                                debug_block_idx,
                                "Learned Skill Usage",
                                f"Policy referenced learned skill(s): {', '.join(called)}",
                                highlight=True,
                                frame_start=start_frame,
                                frame_end=end_frame,
                                start_s=timeline_event["start_s"],
                                end_s=timeline_event["end_s"],
                                timeline_kind="learned_skill",
                                timeline_label=skill_label,
                            )
                except Exception as e:
                    logger.debug(f"  record_usage failed (non-fatal): {e}")
            else:
                if verification.get("success"):
                    try:
                        learned_name_set = {
                            s["name"]
                            for s in self.skill_library.get_full_skills_for_planner(
                                include_deprecated=True,
                            )
                            if not s.get("is_primitive", False)
                        }
                        # On MT-RESET partial_success: extract skills only
                        # from PS-verified step bodies (not force-committed
                        # ones). A force-committed step's call to skill X
                        # was never visually confirmed to have worked, so
                        # crediting X with success=True purely because
                        # verifier reward=1 is a false positive in X's
                        # reliability ledger (some OTHER step likely
                        # accomplished the goal). Fall back to full code
                        # for normal success / non-MT-RESET runs.
                        mt_status_for_skills = (execution_result.get("artifacts") or {}).get(
                            "multiturn_reset_status"
                        )
                        skill_extract_source = code
                        extract_label = "final_success_code"
                        if mt_status_for_skills == "partial_success":
                            verified_code = (
                                (execution_result.get("artifacts") or {})
                                .get("multiturn_reset_natural_committed_code") or ""
                            )
                            skill_extract_source = verified_code
                            extract_label = "natural_committed_code_partial_success"
                        called = _extract_reachable_learned_skills(
                            skill_extract_source, learned_name_set,
                        )
                        if called:
                            successful_called_skills = called
                            skill_lifecycle_events.extend(
                                self.skill_library.record_usage(
                                    called,
                                    success=True,
                                    iteration=self._iteration,
                                    source=extract_label,
                                )
                            )
                            logger.info(
                                f"  Successful skill usage recorded: {', '.join(called[:8])}"
                                + (f" (+{len(called)-8} more)" if len(called) > 8 else "")
                                + (
                                    " [partial_success: PS-verified steps only]"
                                    if mt_status_for_skills == "partial_success" else ""
                                )
                            )
                    except Exception as e:
                        logger.debug(f"  record_usage failed (non-fatal): {e}")
            if verification.get("predicate_status"):
                pred_compact = " | ".join(
                    f"{p['predicate']}={'✓' if p['satisfied'] else '✗'}"
                    for p in verification["predicate_status"]
                )
                logger.info(f"  Predicates: {pred_compact}")
            if verification.get("plan_step_feedback"):
                steps_compact = " | ".join(
                    f"{s['step_id']}={'✓' if s.get('reached') else '✗'}"
                    for s in verification["plan_step_feedback"]
                )
                logger.info(f"  Plan steps: {steps_compact}")
            if not verification["success"] and verification.get("state_hint"):
                logger.info(f"  State hint: {verification['state_hint'][:300]}")
            llm_analysis = verification.get("llm_analysis") or {}
            if llm_analysis.get("fix_suggestion"):
                logger.info(
                    f"  Verifier root cause: {llm_analysis.get('root_cause_predicate')} "
                    f"| antipattern: {(llm_analysis.get('code_antipattern') or '')[:160]}"
                )
                logger.info(
                    f"  Verifier fix suggestion: {(llm_analysis.get('fix_suggestion') or '')[:300]}"
                )
            iteration_data[f"verification_attempt_{attempt}"] = {
                "success": verification.get("success"),
                "reward": verification.get("reward"),
                "task_completed": verification.get("task_completed"),
                "predicate_status": verification.get("predicate_status"),
                "satisfied_conditions": verification.get("satisfied_conditions"),
                "unsatisfied_conditions": verification.get("unsatisfied_conditions"),
                "state_hint": verification.get("state_hint"),
                "llm_analysis": llm_analysis,
                "details": verification.get("evidence"),
                "observed_effect": verification.get("observed_effect"),
                "confidence": verification.get("confidence"),
                "short_reason": verification.get("short_reason")
                or (verification.get("evidence") or {}).get("short_reason"),
            }
            # Per-turn record. The actual video files (per-turn + combined)
            # are written once per attempt — see _save_attempt_videos
            # below, called at attempt boundaries. Tracking the verifier
            # outcome here so the saved attempt folder reflects whether
            # ANY turn within the attempt satisfied the task (vs all
            # turns having failed).
            attempt_key = f"execution_attempt_{attempt}"
            if isinstance(iteration_data.get(attempt_key), dict):
                iteration_data[attempt_key]["turn_frame_range"] = (
                    self._attempt_turn_frame_ranges[-1]
                    if self._attempt_turn_frame_ranges else None
                )
            # Push the verifier's generalizable_lesson into failure_memory so
            # future iterations on different tasks can reuse the insight.
            gen = (llm_analysis.get("generalizable_lesson") or {}) if llm_analysis else {}
            if gen and gen.get("condition") and gen.get("remedy") and not self._no_failure_memory:
                try:
                    lesson = {
                        "lesson_id": f"les_v_{uuid.uuid4().hex[:8]}",
                        "description": (
                            f"WHEN {gen.get('condition')} | "
                            f"WRONG: {gen.get('antipattern','')} | "
                            f"DO: {gen.get('remedy')}"
                        )[:600],
                        "condition": gen.get("condition", ""),
                        "antipattern": gen.get("antipattern", ""),
                        "remedy": gen.get("remedy", ""),
                        "applicable_to": {
                            "objects": gen.get("applicable_objects", []) or [],
                            "actions": gen.get("applicable_actions", []) or [],
                            "task_types": [],
                        },
                        "evidence": [],
                        "confidence": float(llm_analysis.get("confidence", 0.5)),
                        "times_applied": 0,
                        "times_helped": 0,
                        "source": "verifier_llm",
                    }
                    if not self.failure_memory._is_duplicate_lesson(lesson):
                        self.failure_memory._lessons.append(lesson)
                        self.failure_memory._save_lessons()
                        logger.info(f"  Verifier lesson added to failure_memory")
                except Exception as e:
                    logger.debug(f"  failed to push verifier lesson: {e}")

            feedback = None
            # 5e. Failure Diagnoser / MolmoSpaces feedback synthesis.
            #
            # MolmoSpaces uses the simplified post per-step-verifier path:
            # code + trajectory frames + compact per-step VLM summary ->
            # retry_package. The legacy FailureDiagnoser is kept for LIBERO /
            # BEHAVIOR, where raw diagnostic artifact routing remains part of
            # the current feedback design.
            #
            # Single diagnoser path. Was previously env-type-split: LIBERO /
            # BEHAVIOR went through ``failure_diagnoser.diagnose``; MolmoSpaces
            # went through a parallel ``feedback_generator.generate_molmospaces``
            # that bundled diagnosis + retry-package + skill-extraction. The
            # MolmoSpaces sibling was added because the playtime grounded
            # verifier carried more pre-digested signal than BDDL predicate
            # checking, so it felt natural to write a thinner analyzer. In
            # practice the two paths drifted (different prompts, different
            # retry-package shape, two places to fix bugs) and the user
            # asked for unification.
            #
            # Now every env goes through the same ``failure_diagnoser.diagnose``
            # + ``feedback_generator.generate`` two-stage path. LIBERO behavior
            # is unchanged (we did not modify the diagnoser); MolmoSpaces is
            # rerouted into it. The playtime-success shortcut below skips the
            # diagnoser LLM call when the grounded verifier already accepted
            # the attempt — same canned diagnosis the legacy molmospaces path
            # produced on success.
            logger.info("Step 7: Diagnosis")
            if (
                task_proposal.get("_molmospaces_proposer_mode") == "playtime"
                and verification.get("success")
            ):
                diagnosis = {
                    "visual_success": True,
                    "failed_step": None,
                    "failure_reason": "",
                    "policy_feedback": "Grounded playtime verifier accepted the exploratory action.",
                    "confidence": verification.get("confidence", 1.0),
                    "failure_mode": "none",
                    "visual_predicate_status": [],
                }
            else:
                # Surface the verifier's `observed_effect` to the diagnoser
                # so both vision passes share what visibly happened (the
                # verifier and diagnoser are independent VLM calls on
                # different sub-samples of the same trajectory).
                aff_hints = dict(task_proposal.get("affordance_hints") or {})
                if task_proposal.get("_molmospaces_proposer_mode") == "playtime":
                    obs_effect = (verification.get("observed_effect") or "").strip()
                    if obs_effect:
                        aff_hints["_verifier_observation"] = obs_effect
                diagnosis = self.failure_diagnoser.diagnose(
                    execution_result,
                    scene_context,
                    plan=plan,
                    code=code,
                    goal_predicates=task_proposal.get("goal") or task_proposal.get("goal_predicates") or [],
                    affordance_hints=aff_hints,
                    prior_attempts=prior_attempts_for_diag or None,
                    # Tell the diagnoser to phrase policy_feedback as
                    # "what's left from current state" rather than a clean
                    # re-attempt critique — non-trivial in turn mode because
                    # the env did not reset, so completed sub-actions should
                    # not be re-suggested. True only on turns >= 1 within
                    # an attempt; the first turn of any attempt ran on a
                    # fresh reset.
                    task_in_progress=task_in_progress,
                )
            diagnosis_challenged = False
            pre_challenge_diagnosis: dict[str, Any] | None = None
            # Mismatch-challenge gate: was previously gated to ``env_type !=
            # "molmospaces"`` because MolmoSpaces wasn't running the
            # diagnoser. With the unified path above, MolmoSpaces ALSO
            # goes through ``failure_diagnoser.diagnose`` now, so the
            # mismatch challenge can fire for every env.
            if _verifier_diagnoser_mismatch(verification, diagnosis):
                diagnosis_challenged = True
                pre_challenge_diagnosis = {
                    "failure_mode": diagnosis.get("failure_mode"),
                    "policy_feedback": diagnosis.get("policy_feedback"),
                    "failed_step": diagnosis.get("failed_step"),
                    "visual_success": diagnosis.get("visual_success"),
                    "confidence": diagnosis.get("confidence"),
                }
                verifier_challenge = _build_verifier_challenge(
                    verification, diagnosis,
                )
                logger.info(
                    "  Verifier/diagnoser mismatch: verifier rejected "
                    "completion but diagnosis reported no remaining work; "
                    "re-querying diagnoser with challenge."
                )
                diagnosis = self.failure_diagnoser.diagnose(
                    execution_result,
                    scene_context,
                    plan=plan,
                    code=code,
                    goal_predicates=task_proposal.get("goal")
                    or task_proposal.get("goal_predicates")
                    or [],
                    affordance_hints=task_proposal.get("affordance_hints") or {},
                    prior_attempts=prior_attempts_for_diag or None,
                    task_in_progress=task_in_progress,
                    verifier_challenge=verifier_challenge,
                )
                if _verifier_diagnoser_mismatch(verification, diagnosis):
                    logger.info(
                        "  Challenged diagnosis still reported no remaining "
                        "work; applying conservative retry feedback."
                    )
                    diagnosis = _force_mismatch_retry_diagnosis(diagnosis)
            diagnoser_input_images = self._save_diagnoser_input_image_artifacts(
                iteration_data,
                diagnosis,
                attempt,
            )
            diagnoser_input_videos = self._save_diagnoser_input_video_artifacts(
                iteration_data,
                diagnosis,
                attempt,
            )
            diagnosis.pop("diagnoser_input_images", None)
            diagnosis.pop("diagnoser_input_videos", None)
            if diagnoser_input_images:
                diagnosis["diagnoser_input_images"] = diagnoser_input_images
            if diagnoser_input_videos:
                diagnosis["diagnoser_input_videos"] = diagnoser_input_videos
            if self.env_type == "molmospaces" and self.web_debugger is not None:
                self.web_debugger.diagnosis(debug_block_idx, diagnosis)

            # Multiturn-reset: when a step stagnates after exhausting its
            # per-step retry budget, the policy-level retry loop alone won't
            # help — the same plan re-running would just re-stagnate the
            # same step. Force the diagnoser's plan_issue flag so the
            # existing refine_plan pathway fires before the next attempt.
            # (Maps requirement 2: "如果当前步一直失败没有起色，再考虑重新修改plan".)
            # Trigger plan-rewrite on BOTH step_stagnation (old name; no
            # longer emitted by orchestrator after the force-commit fix but
            # kept here for back-compat) AND partial_success (new: one or
            # more steps had their retry budget exhausted and were force-
            # committed so the rest of the plan could still execute, but
            # the plan needs rewriting before the next attempt because
            # those stuck steps will likely fail again with the same plan).
            mt_status = (execution_result.get("artifacts") or {}).get("multiturn_reset_status")
            # On verifier-success + partial_success, do NOT force plan_issue.
            # Per Option B semantics chosen by the user: when the FullTask
            # verifier reports reward=1 / task_completed=True, the iteration
            # is treated as a successful outcome (metrics still record
            # success). The downstream gates block skill/code caching but
            # the loop terminates the attempt — no "next attempt" exists,
            # so emitting "forcing plan_issue=True for next attempt" would
            # contradict the immediately-following "Action: success" log.
            # The hook still fires for verifier-fail + partial_success,
            # which is where plan refinement actually matters.
            mt_should_replan = mt_status in ("step_stagnation", "partial_success") and not bool(
                verification.get("success")
            )
            if mt_should_replan:
                stuck_step_id = (
                    (execution_result.get("artifacts") or {}).get("multiturn_reset_stuck_step_id")
                    or "?"
                )
                stuck_reason = (
                    (execution_result.get("artifacts") or {}).get("multiturn_reset_stuck_reason")
                    or "max_step_retries_exhausted"
                )
                # Full force-committed + skipped step lists so the
                # planner's refinement prompt can see EVERY unverified
                # step, not just the first one stuck_step_id points to.
                forced_step_ids = list(
                    (execution_result.get("artifacts") or {}).get(
                        "multiturn_reset_force_committed_step_ids"
                    ) or []
                )
                skipped_step_ids = list(
                    (execution_result.get("artifacts") or {}).get(
                        "multiturn_reset_skipped_step_ids"
                    ) or []
                )
                unverified_summary_parts = []
                if forced_step_ids:
                    unverified_summary_parts.append(
                        "force-committed (PS never approved, code is UNVERIFIED): "
                        + ", ".join(forced_step_ids)
                    )
                if skipped_step_ids:
                    unverified_summary_parts.append(
                        "skipped (writer never produced usable code): "
                        + ", ".join(skipped_step_ids)
                    )
                unverified_summary = (
                    "; ".join(unverified_summary_parts)
                    if unverified_summary_parts
                    else f"first stuck step '{stuck_step_id}'"
                )
                if mt_status == "partial_success":
                    forced_reason = (
                        f"multiturn-reset: partial success — one or more steps "
                        f"exhausted their per-step retry budget without a "
                        f"PS-succeeded verdict and were force-committed so the "
                        f"rest of the plan could still execute. {unverified_summary}. "
                        f"Reason: {stuck_reason}. Plan rewrite required so the "
                        f"next attempt can restructure around the unverified "
                        f"step(s); treat their code as UNVERIFIED, not as a "
                        f"validated building block."
                    )
                else:
                    forced_reason = (
                        f"multiturn-reset: step '{stuck_step_id}' stagnated "
                        f"({stuck_reason}); plan rewrite required so the next "
                        f"attempt can route around this step."
                    )
                diagnosis["plan_issue"] = True
                # Don't clobber an existing plan_issue_reason if the legacy
                # diagnoser already supplied one — append.
                prior_reason = str(diagnosis.get("plan_issue_reason") or "").strip()
                diagnosis["plan_issue_reason"] = (
                    f"{forced_reason}\nDiagnoser added: {prior_reason}"
                    if prior_reason else forced_reason
                )
                # failed_step is what refine_plan uses to know which plan
                # index to rework. Use the stuck step's id from multiturn.
                if not diagnosis.get("failed_step"):
                    diagnosis["failed_step"] = stuck_step_id
                logger.info(
                    "  Multiturn-reset %s: forcing plan_issue=True for "
                    "next attempt (first stuck step=%s).",
                    mt_status, stuck_step_id,
                )

            iteration_data[f"diagnosis_attempt_{attempt}"] = {
                "failure_mode": diagnosis.get("failure_mode"),
                "policy_feedback": diagnosis.get("policy_feedback"),
                "failed_step": diagnosis.get("failed_step"),
                "confidence": diagnosis.get("confidence"),
                "diagnosis_summary": diagnosis.get("diagnosis_summary"),
                "visual_predicate_status": diagnosis.get("visual_predicate_status") or [],
                "verifier_challenge_requery": diagnosis_challenged,
                "pre_challenge_diagnosis": pre_challenge_diagnosis,
                "verifier_challenge_forced": bool(
                    diagnosis.get("verifier_challenge_forced", False)
                ),
                # Plan-refinement trigger + reason. Persisted so the trace
                # md / post-hoc analysis can show which diagnoser call
                # caused each refined plan to appear.
                "plan_issue": bool(diagnosis.get("plan_issue", False)),
                "plan_issue_reason": diagnosis.get("plan_issue_reason", "") or "",
                # Reflects whether the post-diagnoser hook actually forced
                # plan_issue=True. On Option B semantics, partial_success +
                # verifier-success skips the force (iteration succeeded; no
                # next attempt to replan for). So the field tracks the
                # hook firing, not just MT-RESET status.
                "multiturn_reset_forced_plan_issue": mt_should_replan,
                "multiturn_reset_partial_success": mt_status == "partial_success",
                "diagnoser_input_images": diagnoser_input_images,
                "diagnoser_input_videos": diagnoser_input_videos,
            }
            # Append this attempt to the prior_attempts buffer BEFORE the
            # retry loop reuses it. last_frame is the last element of the
            # filmstrip (== end-of-episode agentview) when available.
            _film = execution_result.get("trajectory_frames") or []
            _last_frame = _film[-1] if _film else execution_result.get("after_frame")
            prior_attempts_for_diag.append({
                "attempt_idx": attempt,
                "code": code,
                "policy_feedback": diagnosis.get("policy_feedback", ""),
                "failure_mode": diagnosis.get("failure_mode", ""),
                # FIX: failed_step and plan_issue_reason were absent from
                # the record even though the diagnosis dict carries them.
                # refine_plan needs failed_step to know which plan-index to
                # rework, and plan_issue_reason to inherit prior structural
                # diagnoses (so refine doesn't re-propose the same reorder
                # that already failed).
                "failed_step": str(diagnosis.get("failed_step") or ""),
                "plan_issue_reason": str(diagnosis.get("plan_issue_reason") or ""),
                "visual_predicate_status": diagnosis.get("visual_predicate_status") or [],
                "last_frame": _last_frame,
            })
            if not verification["success"]:
                last_failed_usage_context = {
                    "attempt": attempt,
                    "code": code,
                    "failed_step": str(diagnosis.get("failed_step") or ""),
                }
                logger.info(f"  Failure mode: {diagnosis.get('failure_mode', 'unknown')}")
                # Log full feedback (was 100-char truncated). Future log-scrapers
                # (scripts/backfill_diagnosis_from_log.py) recover the same text.
                logger.info(f"  Feedback: {diagnosis.get('policy_feedback', '')[:500]}")
                # Per-attempt failure recording: each failed attempt becomes its
                # own episode in failure_memory. Previously only the LAST attempt
                # of the iteration was recorded (after the retry loop), so 5
                # different failure modes from 6 retries collapsed into 1
                # episode and the next iteration's policy writer never saw the
                # earlier approaches that didn't work.
                if not self._no_failure_memory:
                    try:
                        self.failure_memory.record_failure(
                            task_name=task_proposal["activity_name"],
                            scene=task_proposal.get("scene_model", ""),
                            objects_involved=task_objects,
                            failure_category=diagnosis.get("failure_mode", "unknown"),
                            diagnosis_summary=(
                                f"[attempt {attempt + 1}] "
                                + diagnosis.get("policy_feedback", "")
                            ),
                            code_snippet=code,
                            failed_step=str(diagnosis.get("failed_step") or ""),
                            approaches_tried=[f"attempt {attempt + 1}"],
                            retry_count=attempt,
                            max_reward=float(execution_result.get("reward", 0) or 0),
                            iteration=self._iteration,
                        )
                    except Exception as e:
                        logger.debug(f"  per-attempt record_failure failed (non-fatal): {e}")

            # 5f. Feedback Generator
            logger.info("Step 8: Feedback Generation")
            # Keep origin/main's `if feedback is None` guard (new playtime
            # / per-step verifier path can set `feedback` upstream) AND
            # my task_language + existing_skills arguments (GOAL field
            # + dedup-against-library — see commits 3f60d203 / 74379df1).
            # Detect multiturn-reset partial_success BEFORE calling
            # feedback_generator: when one or more steps were force-
            # committed (PS never approved their code), Option B says
            # metrics still record success but skill extraction is
            # skipped — the unverified step codes should not become
            # reusable library skills. Tell feedback_generator to short-
            # circuit the LLM extraction call so we don't burn tokens
            # generating skills we'd then drop.
            _mt_partial_success = (
                (execution_result.get("artifacts") or {}).get(
                    "multiturn_reset_status"
                )
                == "partial_success"
            )
            if feedback is None:
                feedback = self.feedback_generator.generate(
                    execution_result, verification, diagnosis,
                    attempt=attempt, plan=plan, code=code,
                    task_language=task_proposal.get("language", ""),
                    existing_skills=self.skill_library.get_full_skills_for_planner(),
                    multiturn_reset_partial_success=_mt_partial_success,
                )
            logger.info(f"  Action: {feedback['action']}")
            if self.web_debugger is not None:
                self.web_debugger.step(
                    debug_block_idx,
                    "Feedback Generation",
                    f"Action: `{feedback.get('action')}`",
                    highlight=feedback.get("action") == "success",
                )
                self.web_debugger.finish_attempt(
                    debug_block_idx,
                    execution_result,
                    success=bool(verification.get("success")),
                )

            if feedback["action"] == "success":
                # Save the attempt's videos (per-turn + combined) to disk
                # before the success break. Status reflects the attempt-
                # level outcome that the user will scan for in the output
                # dir.
                attempt_media_artifacts = self._save_attempt_videos(
                    low_level=low_level,
                    attempt_in_iter=attempt_in_iter,
                    attempt=attempt,
                    status="succeeded",
                )
                self._attach_attempt_media_artifacts(
                    iteration_data, attempt, attempt_media_artifacts,
                )
                if task_proposal.get("_molmospaces_proposer_mode") == "playtime":
                    self._record_playtime_outcome(
                        task_proposal=task_proposal,
                        verification=verification,
                        iteration_data=iteration_data,
                        attempt_in_iter=attempt_in_iter,
                    )
                # Remember the successful code for this task type, but
                # ONLY when the rollout was end-to-end PS-verified. On
                # multiturn-reset partial_success, one or more steps were
                # force-committed (PS never approved them) — caching that
                # code as the "winning recipe" would let future
                # iterations replay unverified primitives as if they were
                # known-good. Verifier ground-truth still says success,
                # so metrics record success; we just don't promote the
                # code to a reusable artifact.
                if _mt_partial_success:
                    logger.info(
                        "  Skipping _successful_code cache for "
                        f"{task_proposal['activity_name']}: multiturn-reset "
                        "partial_success (force-committed step(s) unverified)."
                    )
                else:
                    self._successful_code[task_proposal["activity_name"]] = code
                    logger.info(f"  Stored successful code for {task_proposal['activity_name']} ({len(code)} chars)")
                # Bump times_helped on lessons that were most recently shown
                # to this iteration's policy writer — soft heuristic signal
                # for the MemoryCurator. Pass iteration/task/attempt so
                # `recent_applications` carries per-attempt evidence the
                # curator can cite when deciding DELETE / MERGE.
                try:
                    self.failure_memory.record_outcome_for_last_served_lessons(
                        success=True,
                        iteration=self._iteration,
                        task_name=task_proposal.get("activity_name", ""),
                        attempt_idx=attempt,
                    )
                except Exception:
                    pass
                # Extract and add skills (unless no-skill-reuse ablation).
                # Unified play mode descends from the prompt-only sibling
                # which never applied a ``play:`` source-task prefix, so
                # the source-task is just the activity name.
                extracted_source_task = task_proposal["activity_name"]
                for skill in feedback.get("new_skills", []) or []:
                    generated_skill_artifacts.append({
                        "name": skill.get("name", "unnamed"),
                        "description": skill.get("description", ""),
                        "code": skill.get("code", ""),
                        "api_primitives_used": skill.get("api_primitives_used", []),
                        "preconditions": skill.get("preconditions", []),
                        "effects": skill.get("effects", []),
                        "source": "feedback_generator",
                        "source_task": extracted_source_task,
                        "learned_iteration": self._iteration,
                        "stored_in_library": False,
                    })
                if not self._no_skill_reuse:
                    for skill in feedback.get("new_skills", []) or []:
                        original_name = skill.get("name", "unnamed")
                        candidate = {
                            "name": skill.get("name", "unnamed"),
                            "description": skill.get("description", ""),
                            "code": skill.get("code", ""),
                            "api_primitives_used": skill.get("api_primitives_used", []),
                            "preconditions": skill.get("preconditions", []),
                            "effects": skill.get("effects", []),
                            "source_task": extracted_source_task,
                            "learned_iteration": self._iteration,
                        }
                        added = self.skill_library.add_skill(
                            candidate,
                            available_functions=scene_context.get(
                                "available_functions",
                            ),
                        )
                        stored_name = candidate.get("name", original_name)
                        for generated in reversed(generated_skill_artifacts):
                            if (
                                generated.get("source") == "feedback_generator"
                                and generated.get("name") == original_name
                                and generated.get("source_task") == extracted_source_task
                            ):
                                generated["stored_in_library"] = bool(added)
                                if added:
                                    generated["stored_name"] = stored_name
                                else:
                                    generated["rejected_reason"] = "duplicate_or_invalid"
                                break
                        if added:
                            if stored_name not in accepted_feedback_skill_names:
                                accepted_feedback_skill_names.append(stored_name)
                            # Credit the extraction with one success: the
                            # skill was distilled from a verifier-accepted
                            # task that just succeeded, so its very first
                            # observation is positive. Without this it
                            # ships at usage=0/0/success_rate=0 and the
                            # Wilson-sorted planner deprioritises it for
                            # iterations until an unrelated task happens
                            # to call it. record_usage skips primitives
                            # automatically.
                            try:
                                skill_lifecycle_events.extend(
                                    self.skill_library.record_usage(
                                        [stored_name],
                                        success=True,
                                        iteration=self._iteration,
                                        source="skill_extraction",
                                    )
                                )
                            except Exception as e:
                                logger.debug(
                                    f"  failed to credit new skill "
                                    f"{stored_name!r} with extraction "
                                    f"success (non-fatal): {e}"
                                )
                            logger.info(
                                f"  New skill added: {stored_name}"
                            )
                        else:
                            rejected_skill_records.append({
                                "name": original_name,
                                "source": "feedback_generator",
                                "source_task": extracted_source_task,
                                "reason": "duplicate_or_invalid",
                            })
                else:
                    logger.info("  Skill reuse DISABLED (ablation) -- not storing skills")

                final_result = feedback
                break

            elif feedback["action"] == "retry":
                retry_feedback = feedback["retry_package"]
                # Inject the attempt-before-the-most-recent code so the
                # writer sees POLICY_HISTORY_DEPTH attempts of context
                # (most recent is already in `previous_code`). With
                # depth=2 and ≥2 entries in code_history, code_history[-2]
                # is the older attempt.
                if len(code_history) >= 2:
                    retry_feedback = dict(retry_feedback)
                    retry_feedback["older_code"] = code_history[-2]
                attempt += 1
                attempt_video_flushed = False

                # Sub-skill isolation: when the diagnoser targeted a
                # specific sub-behavior for isolated practice, spawn a
                # SubAgent BEFORE considering plan refinement. If it
                # learns the skill, the next main attempt may succeed
                # under the existing plan — no replan needed.
                sub_target = retry_feedback.get("subagent_skill_target")
                if sub_target and is_last_turn_of_attempt and self._subagent_enabled:
                    # SubAgent.run() resets the same env and clears/consumes
                    # the same low-level video buffer for its own per-attempt
                    # recordings. Flush the parent attempt first; otherwise
                    # the normal iterXXX_attemptY_*.mp4 / attempt folder is
                    # lost and the output dir appears to contain only
                    # sub-agent folders.
                    attempt_media_artifacts = self._save_attempt_videos(
                        low_level=low_level,
                        attempt_in_iter=attempt_in_iter,
                        attempt=attempt,
                        status="failed",
                    )
                    self._attach_attempt_media_artifacts(
                        iteration_data, attempt, attempt_media_artifacts,
                    )
                    attempt_video_flushed = True
                    sub_reason = retry_feedback.get(
                        "subagent_skill_target_reason", ""
                    )
                    sub_approaches = retry_feedback.get(
                        "subagent_approaches", [],
                    ) or []
                    logger.info(
                        f"  Diagnoser targeted sub-skill for isolation: "
                        f"{sub_target[:120]!r} (reason: {sub_reason[:120]!r}) "
                        f"approaches={len(sub_approaches)}"
                    )
                    sub_video_tag = (
                        f"iter{self._iteration:03d}_subagent"
                        f"{iteration_data.get('_subagent_count', 0)}"
                    )
                    iteration_data["_subagent_count"] = (
                        iteration_data.get("_subagent_count", 0) + 1
                    )
                    # Parallel path: when the diagnoser proposed multiple
                    # distinct approaches AND we have a BDDL path that
                    # worker subprocesses can rebuild from, dispatch them
                    # in parallel. First success kills the rest. Falls
                    # back to single-approach inline run when either
                    # condition isn't met.
                    bddl_path = scene_context.get("bddl_path")
                    use_parallel = bool(sub_approaches) and bool(bddl_path)
                    sub_result: dict[str, Any]
                    parallel_results: list[dict[str, Any]] | None = None
                    parallel_failed = False
                    try:
                        if use_parallel:
                            parallel_results = self._dispatch_parallel_subagents(
                                bddl_path=bddl_path,
                                subgoal=sub_target,
                                approaches=sub_approaches,
                                scene_context=scene_context,
                                parent_task_name=task_proposal.get("activity_name") or "",
                                video_tag_prefix=sub_video_tag,
                            )
                            # Pick first successful winner (canonical)
                            winner = next(
                                (r for r in parallel_results if r.get("success")),
                                None,
                            )
                            sub_result = {
                                "success": bool(winner),
                                "code": (winner or {}).get("code", ""),
                                "attempts": (winner or {}).get("attempts_used"),
                                "history": [],
                            }
                        else:
                            sub_result = self.sub_agent.run(
                                subgoal=sub_target,
                                env=self.env,
                                scene_context=scene_context,
                                diagnoser=self.failure_diagnoser,
                                executor=self.executor,
                                reset_env=self._reset_env,
                                video_dir=self.output_dir,
                                video_tag=sub_video_tag,
                                parent_task_name=task_proposal.get("activity_name"),
                            )
                    except Exception as e:
                        # Parallel/sequential dispatch crashed — log loud and
                        # fall back to the inline sequential sub-agent so we
                        # don't silently drop a sub-agent opportunity. We
                        # discovered an `import sys` miss this way; without
                        # the fallback, every parallel dispatch swallowed
                        # the error and the next attempt got NO injected
                        # script (32 silent failures across iter 1-4).
                        logger.warning(
                            f"  Parallel sub-agent dispatch failed "
                            f"({type(e).__name__}: {e}); falling back to "
                            f"inline sequential sub-agent run."
                        )
                        parallel_failed = True
                        sub_result = {"success": False, "code": "", "history": []}

                    if parallel_failed:
                        try:
                            sub_result = self.sub_agent.run(
                                subgoal=sub_target,
                                env=self.env,
                                scene_context=scene_context,
                                diagnoser=self.failure_diagnoser,
                                executor=self.executor,
                                reset_env=self._reset_env,
                                video_dir=self.output_dir,
                                video_tag=sub_video_tag,
                                parent_task_name=task_proposal.get("activity_name"),
                            )
                        except Exception as e:
                            logger.warning(
                                f"  Sequential fallback also failed: "
                                f"{type(e).__name__}: {e}"
                            )

                    try:
                        iteration_data[f"subagent_{sub_video_tag}"] = {
                            "subgoal": sub_target,
                            "reason": sub_reason,
                            "approaches": sub_approaches,
                            "parallel": use_parallel,
                            "parallel_results": parallel_results,
                            "success": sub_result.get("success"),
                            "attempts": sub_result.get("attempts"),
                            "history": sub_result.get("history", []),
                        }
                        if sub_result.get("success"):
                            # Sub-agent successes are TASK-SPECIFIC, not
                            # general skills. The previous design extracted
                            # an LLM-parameterised wrapper and added it to
                            # the persistent library, but in practice
                            # extraction either (a) hallucinated calls
                            # (validator-rejected) or (b) parameterised
                            # away the exact arg values that made the
                            # original script succeed, so the main loop
                            # could not actually reuse it. Instead we keep
                            # the verbatim winning sub-agent script,
                            # surface it in the writer's success_context
                            # for THIS iteration only, and let the writer
                            # adapt/inline it for the matching plan step.
                            # No library pollution, no LLM rewrite step.
                            raw_code = (sub_result.get("code") or "").strip()
                            if raw_code:
                                # Tie the winning script to the exact plan
                                # step the diagnoser flagged. Without this
                                # the writer sees a generic "here's a
                                # script" block and regenerates its own
                                # version anyway — losing the offsets /
                                # SAM3 prompts / axis choices that
                                # actually worked. With the step_id
                                # attached, the prompt can issue a direct
                                # "at step X, call this verbatim" order.
                                failed_step_id = str(
                                    retry_feedback.get("failed_step") or ""
                                ).strip() or "?"
                                failed_step_desc = ""
                                _plan_steps = (plan or {}).get("steps", []) if isinstance(plan, dict) else (plan or [])
                                for _st in _plan_steps:
                                    if not isinstance(_st, dict):
                                        continue
                                    if str(_st.get("id") or _st.get("step_id") or "") == failed_step_id:
                                        failed_step_desc = str(
                                            _st.get("description", "")
                                        )
                                        break
                                logger.info(
                                    f"  Sub-agent winning script captured "
                                    f"({len(raw_code)} chars) for step "
                                    f"{failed_step_id}; injecting as "
                                    f"MANDATORY reuse for this iteration "
                                    f"(NOT added to library)"
                                )
                                fresh_block_lines = [
                                    "",
                                    f"--- SUB-AGENT WINNING SCRIPT FOR "
                                    f"PLAN STEP {failed_step_id} ---",
                                    (
                                        f"Step {failed_step_id} description: "
                                        f"{failed_step_desc}"
                                        if failed_step_desc
                                        else f"Failed step id: {failed_step_id}"
                                    ),
                                    f"Sub-agent subgoal (equivalent): {sub_target}",
                                    "",
                                    "MANDATORY REUSE INSTRUCTION:",
                                    "- The Python block below was just run "
                                    "end-to-end from the SAME reset state "
                                    "as this attempt, and it succeeded at "
                                    "the sub-skill above. Its exact "
                                    "numerical offsets (hover Z, descend Z, "
                                    "any x/y biasing), SAM3/Molmo prompt "
                                    "strings, axis choices, and retry "
                                    "counts are what made it work — the "
                                    "previous attempts' regenerated "
                                    "equivalents did NOT work.",
                                    f"- When you emit code for plan step "
                                    f"{failed_step_id}, you MUST inline "
                                    f"this block verbatim (or call it as "
                                    f"a helper) and MUST NOT rewrite the "
                                    f"geometry, the drop-point math, the "
                                    f"prompt strings, or the hover/descend "
                                    f"Z offsets. Preserve every numeric "
                                    f"value below. You may only add the "
                                    f"glue needed to connect it to other "
                                    f"plan steps (variables it consumes "
                                    f"from prior steps or writes into "
                                    f"RESULT keys).",
                                    "- If you find yourself writing fresh "
                                    "pixel-to-world math, fresh SAM3 "
                                    "prompts, or fresh hover/descend "
                                    "numbers for this step, STOP and "
                                    "paste the block instead.",
                                    "",
                                    "```python",
                                    raw_code,
                                    "```",
                                ]
                                success_context = (
                                    success_context
                                    + "\n".join(fresh_block_lines)
                                    + "\n"
                                )
                    except Exception as e:
                        logger.warning(f"  SubAgent run failed: {e}")
                elif sub_target:
                    logger.info(
                        "  Diagnoser requested sub-skill isolation, but this "
                        "is not an attempt boundary; deferring to normal "
                        "within-attempt retry so the in-progress env state is "
                        "not reset by a sub-agent."
                    )

                # Plan-level refinement. When the diagnoser flagged a
                # structural problem (plan_issue), rewrite the plan via
                # Planner.refine_plan before the next code attempt.
                # retry_package has already cleared preserved_code_segments
                # in this case — step_ids may shift after a replan.
                #
                # Only run at attempt boundaries (about to reset env). On
                # within-attempt turn transitions the env state already
                # reflects partial plan progress, and a refined plan would
                # issue new step ids that don't line up with what
                # physically happened.
                if retry_feedback.get("replan") and is_last_turn_of_attempt:
                    reason = retry_feedback.get("plan_issue_reason", "")
                    logger.info(
                        f"  Diagnoser flagged plan_issue; refining plan. "
                        f"Reason: {reason[:160]}"
                    )
                    try:
                        new_plan = self.planner.refine_plan(
                            old_plan=plan,
                            task_proposal=task_proposal,
                            all_skills=all_skills,
                            scene_context=scene_context,
                            plan_issue_reason=reason,
                            prior_attempts=prior_attempts_for_diag,
                            failure_lessons=failure_lessons,
                            # Same agentview we captured at plan time.
                            # Env resets between attempts → initial state
                            # is invariant within an iteration.
                            initial_rgb=initial_rgb,
                        )
                        if new_plan and new_plan.get("steps"):
                            plan = new_plan
                            iteration_data[f"plan_refined_attempt_{attempt}"] = plan
                            logger.info(
                                f"  Refined plan: {len(plan.get('steps', []))} steps"
                            )
                    except Exception as e:
                        logger.warning(f"  refine_plan failed: {e}; keeping old plan")

                if is_last_turn_of_attempt:
                    # End of an attempt: save the attempt's accumulated
                    # videos (per-turn + combined) to disk, then reset
                    # env for next attempt's turn 0. Skip the reset on
                    # the very last attempt (we're about to exit the
                    # iteration loop) — the iteration epilogue handles
                    # that, but we still flush the videos here.
                    if not attempt_video_flushed:
                        attempt_media_artifacts = self._save_attempt_videos(
                            low_level=low_level,
                            attempt_in_iter=attempt_in_iter,
                            attempt=attempt,
                            status="failed",
                        )
                        self._attach_attempt_media_artifacts(
                            iteration_data, attempt, attempt_media_artifacts,
                        )
                    will_have_next_attempt = (attempt + 1 < total_budget)
                    if will_have_next_attempt:
                        logger.info(
                            f"  Attempt {attempt_in_iter + 1} exhausted its "
                            f"{self._turns_per_attempt} turn(s); resetting env "
                            f"for attempt {attempt_in_iter + 2}."
                        )
                        # Reset env between attempts. Without this, the
                        # episode may have exhausted its step limit (e.g.
                        # 4000 MuJoCo steps), causing all subsequent code
                        # to fail with "executing action in terminated
                        # episode". reset() preserves the current task
                        # — only the episode state is re-initialized.
                        self._reset_env()
                else:
                    # Within-attempt turn transition: do NOT reset.
                    # Persisted env state lets turn N+1 build on turn N.
                    if execution_result.get("terminated") or execution_result.get("truncated"):
                        # The env is already wedged — further turns within
                        # this attempt can't step it. Save what we have
                        # and fast-forward to the next attempt boundary.
                        attempt_media_artifacts = self._save_attempt_videos(
                            low_level=low_level,
                            attempt_in_iter=attempt_in_iter,
                            attempt=attempt,
                            status="failed",
                        )
                        self._attach_attempt_media_artifacts(
                            iteration_data, attempt, attempt_media_artifacts,
                        )
                        skip_to_next_attempt = self._turns_per_attempt - turn_in_attempt - 1
                        logger.info(
                            f"  Env terminated/truncated mid-attempt; "
                            f"skipping {skip_to_next_attempt} remaining "
                            f"turn(s) and resetting for next attempt."
                        )
                        attempt += skip_to_next_attempt
                        if attempt + 1 < total_budget:
                            self._reset_env()
                    else:
                        logger.info(
                            f"  Continuing to turn {turn_in_attempt + 2}/"
                            f"{self._turns_per_attempt} of attempt "
                            f"{attempt_in_iter + 1} (no env reset; task in progress)..."
                        )
                continue

            else:  # skip
                # Save whatever we have for this attempt before bailing
                # out of the iteration loop.
                attempt_media_artifacts = self._save_attempt_videos(
                    low_level=low_level,
                    attempt_in_iter=attempt_in_iter,
                    attempt=attempt,
                    status="failed",
                )
                self._attach_attempt_media_artifacts(
                    iteration_data, attempt, attempt_media_artifacts,
                )
                final_result = feedback
                break

        if final_result is None:
            final_result = {
                "action": "skip",
                "message": f"Max retries ({self.max_retries}) exceeded",
                "new_skills": [],
            }

        # Record outcome with rich failure context
        success = final_result["action"] == "success"
        # Record the per-iteration lesson-application outcome. We already
        # call this on the success path (with full success=True) at the
        # success branch above; on the failure path it lets the
        # MemoryCurator see "applied N times, helped 0" instead of
        # only the aggregate.
        if not success:
            try:
                self.failure_memory.record_outcome_for_last_served_lessons(
                    success=False,
                    iteration=self._iteration,
                    task_name=task_proposal.get("activity_name", ""),
                    attempt_idx=attempt - 1 if attempt > 0 else 0,
                )
            except Exception:
                pass
        if not success and last_failed_usage_context:
            try:
                failed_step = str(last_failed_usage_context.get("failed_step") or "")
                if failed_step:
                    learned_name_set = {
                        s["name"]
                        for s in self.skill_library.get_full_skills_for_planner(
                            include_deprecated=True,
                        )
                        if not s.get("is_primitive", False)
                    }
                    failed_called = _extract_step_reachable_learned_skills(
                        str(last_failed_usage_context.get("code") or ""),
                        failed_step,
                        learned_name_set,
                    )
                    if failed_called:
                        failed_called_skills = failed_called
                        failed_skill_step = failed_step
                        skill_lifecycle_events.extend(
                            self.skill_library.record_usage(
                                failed_called,
                                success=False,
                                iteration=self._iteration,
                                source="final_failed_step",
                            )
                        )
                        logger.info(
                            f"  Failed-step skill usage recorded ({failed_step}): "
                            f"{', '.join(failed_called[:8])}"
                            + (
                                f" (+{len(failed_called)-8} more)"
                                if len(failed_called) > 8 else ""
                            )
                        )
                    else:
                        logger.info(
                            f"  Failed-step skill usage skipped: no reachable "
                            f"learned skills found for {failed_step!r}"
                        )
            except Exception as e:
                logger.debug(
                    f"  failed-step record_usage failed (non-fatal): {e}"
                )
        skills_learned = list(accepted_feedback_skill_names)
        skill_growth = self.skill_library.get_learned_skill_count() > learned_count_before
        if selected_queue_active and selected_queue_task_id and self.task_queue is not None:
            queue_meta = iteration_data.setdefault("task_queue", {})
            queue_success = bool(success and iteration_data.get("_env_created", True))
            if queue_success:
                removed_entry = self.task_queue.remove(selected_queue_task_id)
                queue_meta["queue_update"] = {
                    "action": "remove_on_success",
                    "selected_after_update": _queue_summary(removed_entry),
                }
                if skill_growth:
                    updated_skill_context = self.skill_library.get_context_for_task_proposer()
                    self.task_queue.rescore_all(
                        task_proposer=self.task_proposer,
                        skill_context=updated_skill_context,
                        reset_penalties=True,
                    )
                    queue_meta["rescored"] = True
                else:
                    queue_meta["rescored"] = False
                queue_meta["queue_size_after_update"] = len(self.task_queue)
                queue_meta["top_snapshot_after_update"] = self.task_queue.top_snapshot()
            else:
                pred_prob = float(
                    ((plan.get("prediction_card", {}) or {}).get("predicted_success_probability", 0.5))
                    if isinstance(plan, dict) else 0.5
                )
                failed_entry = self.task_queue.record_failure(
                    selected_queue_task_id,
                    predicted_success_probability=pred_prob,
                    iteration=self._iteration,
                )
                queue_meta["queue_update"] = {
                    "action": "requeue_on_failure",
                    "selected_after_update": _queue_summary(failed_entry),
                }
                queue_meta["rescored"] = False
                queue_meta["queue_size_after_update"] = len(self.task_queue)
                queue_meta["top_snapshot_after_update"] = self.task_queue.top_snapshot()

        # Determine failure reason from diagnosis
        failure_reason = ""
        failure_category = "unknown"
        diagnosis_summary = ""
        if not success:
            if not iteration_data.get("_env_created", True):
                failure_reason = f"env_creation_failed: {iteration_data.get('_env_error', 'unknown')}"
                failure_category = "env_creation_failed"
            elif diagnosis:
                failure_category = diagnosis.get("failure_mode", "unknown")
                diagnosis_summary = diagnosis.get("policy_feedback", "")
                failure_reason = failure_category
                if diagnosis_summary:
                    # FIX: was [:100] then [:300], both cut mid-sentence.
                    # `failure_reason` lands in the task_proposer's
                    # task_history block; an entry is per-task per-iter, so
                    # the size is bounded by the history window (top-K
                    # entries in the prompt), not by this slice. Emit the
                    # full diagnosis — the proposer can use it to avoid
                    # re-proposing the same failure pattern.
                    failure_reason += f": {diagnosis_summary}"

        # Record env-creation failures (the only kind not already captured by
        # the per-attempt path inside the retry loop above). Per-attempt
        # execution failures are recorded as they happen at Step 7.
        if (
            not success
            and not self._no_failure_memory
            and not iteration_data.get("_env_created", True)
        ):
            self.failure_memory.record_failure(
                task_name=task_proposal["activity_name"],
                scene=task_proposal.get("scene_model", ""),
                objects_involved=task_objects,
                failure_category="env_creation_failed",
                diagnosis_summary=iteration_data.get("_env_error", ""),
                code_snippet="",
                approaches_tried=[],
                retry_count=0,
                max_reward=0.0,
                iteration=self._iteration,
            )

        self.task_proposer.record_task_outcome(
            task_proposal["activity_name"], success, attempt, skills_learned,
            failure_reason=failure_reason,
            env_created=iteration_data.get("_env_created", True),
            language=task_proposal.get("language", task_proposal.get("goal_conditions", "")),
            objects_used=task_proposal.get("objects", []),
            fixtures_used=task_proposal.get("fixtures", []),
            goal_predicates=task_proposal.get("goal", []),
            play_mode=bool(task_proposal.get("play_mode", False)),
            play_verb=task_proposal.get("play_verb", ""),
            play_target=task_proposal.get("play_target", ""),
            playtime_metadata=task_proposal.get("_playtime") or {},
        )

        # Candidate-mode bookkeeping: update (object, skill) attempt
        # counter and the retry bank. Both are no-ops when candidate mode
        # is off — they only carry signal for the new curiosity scorer.
        if self._candidate_mode == "formula":
            from rats.agents.curiosity_scoring import (
                save_object_skill_counts,
                update_object_skill_counts,
            )
            update_object_skill_counts(
                self._obj_skill_counts,
                task_proposal.get("objects") or [],
                task_proposal.get("required_skills") or [],
            )
            save_object_skill_counts(
                self._obj_skill_counts, self._obj_skill_counts_path,
            )
            if self.retry_bank is not None:
                pred_prob = None
                # ``plan`` is set earlier in the iteration; re-extract its
                # planner-emitted predicted_success_probability the same
                # way the legacy task_queue path did.
                try:
                    if isinstance(plan, dict):
                        pred_prob = (
                            (plan.get("prediction_card", {}) or {})
                            .get("predicted_success_probability")
                        )
                except NameError:  # plan not bound (env-create failure path)
                    pred_prob = None
                rb_meta: dict[str, Any] = {
                    "size_before": len(self.retry_bank),
                    "pred_prob_seen": pred_prob,
                    "action": "noop",
                }
                if success:
                    if task_proposal.get("candidate_type") == "retry_derived":
                        rid = task_proposal.get("source_retry_id")
                        if self.retry_bank.mark_resolved(rid):
                            rb_meta["action"] = "resolved"
                            rb_meta["resolved_retry_id"] = rid
                else:
                    should, why = self.retry_bank.should_add(
                        success=False,
                        failure_reason=failure_reason,
                        predicted_success_probability=pred_prob,
                    )
                    rb_meta["should_add_reason"] = why
                    if should:
                        added = self.retry_bank.add(
                            task_spec={
                                "language": task_proposal.get("language", ""),
                                "objects": task_proposal.get("objects") or [],
                                "fixtures": task_proposal.get("fixtures") or [],
                                "goal": task_proposal.get("goal") or [],
                                "scene_type": task_proposal.get("scene_type", ""),
                            },
                            failure_reason=failure_reason,
                            diagnosis_summary=diagnosis_summary,
                            predicted_success_probability=float(pred_prob or 0.0),
                            iteration=self._iteration,
                        )
                        rb_meta["action"] = "added"
                        rb_meta["added_retry_id"] = added.get("retry_id")
                # TTL decay once per iteration, regardless of outcome.
                self.retry_bank.decay_ttl()
                rb_meta["size_after"] = len(self.retry_bank)
                rb_meta["snapshot_after"] = self.retry_bank.snapshot()
                iteration_data["retry_bank"] = rb_meta
        if success:
            skills_reused = sorted({
                name for name in successful_called_skills
                if name and name not in skills_learned
            })
            skills_failed: list[str] = []
            skill_usage_source = (
                "final_success_code" if skills_reused else "none"
            )
        else:
            skills_reused = []
            skills_failed = sorted({name for name in failed_called_skills if name})
            skill_usage_source = (
                "final_failed_step" if skills_failed else "none"
            )

        # Reset only when another iteration will actually consume the fresh
        # state. A terminal reset is not preparation: in a one-episode
        # evaluation it is a second hidden simulator episode and destroys the
        # authoritative final state before the evaluator can serialize it.
        if self._should_reset_for_next_iteration():
            self._reset_env()

        # Maybe distill failure lessons (1B.5)
        if self.failure_memory.maybe_distill(self._iteration):
            logger.info("  Failure lessons distilled")

        # Maybe curate lessons (MemoryCurator): throttled periodic pass over
        # the failure-memory lesson set. Merges duplicates, deletes vague/
        # never-helped lessons, rewrites prose into primitive-specific
        # guidance. Uses a stronger LLM (gpt-5.4 by default) because the
        # reasoning is subtle and the prompt context is already long.
        # Trigger is based on the absolute iteration number (which IS
        # persisted across --resume) rather than an in-process counter —
        # restart-heavy sessions used to miss every curator window because
        # the in-process counter reset to 0 on each restart.
        self._iters_since_curate += 1
        iter_mod = self._iteration > 0 and self._iteration % self._curate_every == 0
        if (
            not self._no_failure_memory
            and (self._iters_since_curate >= self._curate_every or iter_mod)
            and len(self.failure_memory._lessons) >= 3
        ):
            self._iters_since_curate = 0
            recent = [
                {
                    "iteration": d.get("iteration"),
                    "task": d.get("task_proposal", {}).get("activity_name"),
                    "success": d.get("success"),
                    "new_skills": d.get("skills_learned", []),
                    "unsatisfied": (d.get(f"verification_attempt_{d.get('total_attempts', 1) - 1}", {})
                                    or {}).get("unsatisfied_conditions", []),
                }
                for d in self._iter_history_for_curator[-15:]
            ]
            try:
                history_path = self.output_dir / "memory" / "curator_history.json"
                summary = self.memory_curator.curate(
                    failure_memory=self.failure_memory,
                    recent_iteration_outcomes=recent,
                    history_path=history_path,
                )
                logger.info(
                    f"  MemoryCurator: merge={summary.get('merge',0)} "
                    f"delete={summary.get('delete',0)} "
                    f"rewrite={summary.get('rewrite',0)} "
                    f"noop={summary.get('noop',0)} "
                    f"(lessons now: {len(self.failure_memory._lessons)})"
                )
                iteration_data["curator_summary"] = summary
            except Exception as e:
                logger.warning(f"  MemoryCurator failed (non-fatal): {e}")

            # Skill curation rides the same cadence. Separate toggle so it
            # can be disabled independently (e.g. when reproducing an
            # earlier run's exact library state). Only runs when the
            # library has grown past a small threshold — nothing to
            # collapse in 2-3 skills, and the LLM tends to over-act.
            if os.getenv("RATS_SKILL_CURATE", "1") != "0":
                try:
                    learned_n = self.skill_library.get_learned_skill_count()
                    if learned_n >= 4:
                        skill_hist = (
                            self.output_dir / "memory"
                            / "skill_curator_history.json"
                        )
                        skill_summary = self.memory_curator.curate_skills(
                            skill_library=self.skill_library,
                            recent_iteration_outcomes=recent,
                            history_path=skill_hist,
                            current_iteration=self._iteration,
                        )
                        skill_lifecycle_events.extend(
                            skill_summary.get("lifecycle_events", []) or []
                        )
                        logger.info(
                            f"  SkillCurator: merge="
                            f"{skill_summary.get('merge', 0)} "
                            f"deprecate={skill_summary.get('deprecate', 0)} "
                            f"noop={skill_summary.get('noop', 0)} "
                            f"(learned now: "
                            f"{self.skill_library.get_learned_skill_count()})"
                        )
                        iteration_data["skill_curator_summary"] = skill_summary
                except Exception as e:
                    logger.warning(
                        f"  SkillCurator failed (non-fatal): {e}"
                    )

        # Maybe propose new helper skills from observed failures.
        # Trigger on failed iterations only, throttled to once per 2 failures, and
        # only when there is enough failure context for the LLM to reason about.
        self._iters_since_skill_proposal += 1
        if (
            not success
            and not self._no_skill_reuse
            and self._iters_since_skill_proposal >= 2
            and self.failure_memory.episode_count >= 2
        ):
            self._iters_since_skill_proposal = 0
            try:
                fail_summary = self.failure_memory.retrieve_for_policy_writer(
                    task_name=task_proposal["activity_name"],
                    objects=task_objects,
                    top_k=4,
                    current_iteration=self._iteration,
                )
                primitive_list = scene_context.get("api_docs", "") or "\n".join(
                    f"- {fn}()" for fn in scene_context.get("available_functions", [])
                )
                learned_now = [
                    {"name": s["name"], "description": s.get("description", "")}
                    for s in self.skill_library.get_full_skills_for_planner()
                    if not s.get("is_primitive", False)
                ]
                proposals = self.skill_proposer.propose(
                    failure_summary=fail_summary,
                    primitive_list=primitive_list,
                    learned_skills=learned_now,
                )
                for p in proposals:
                    original_name = p.get("name", "unnamed")
                    p = {**p, "learned_iteration": self._iteration}
                    added = self.skill_library.add_skill(p)
                    stored_name = p.get("name", original_name)
                    generated_skill_artifacts.append({
                        "name": original_name,
                        "description": p.get("description", ""),
                        "code": p.get("code", ""),
                        "api_primitives_used": p.get("api_primitives_used", []),
                        "preconditions": p.get("preconditions", []),
                        "effects": p.get("effects", []),
                        "rationale": p.get("rationale", ""),
                        "source": "skill_proposer",
                        "source_task": "proposed_from_failures",
                        "learned_iteration": self._iteration,
                        "stored_in_library": bool(added),
                        "stored_name": stored_name if added else "",
                        "rejected_reason": "" if added else "duplicate_or_invalid",
                    })
                    if added:
                        accepted_proposed_skills.append({
                            "name": stored_name,
                            "original_name": original_name,
                            "rationale": p.get("rationale", ""),
                        })
                        logger.info(
                            f"  SkillProposer added new skill: {stored_name} "
                            f"(rationale: {p.get('rationale','')[:120]})"
                        )
                    else:
                        rejected_skill_records.append({
                            "name": original_name,
                            "source": "skill_proposer",
                            "source_task": "proposed_from_failures",
                            "reason": "duplicate_or_invalid",
                        })
                if accepted_proposed_skills:
                    iteration_data["proposed_skills"] = accepted_proposed_skills
                rejected_proposals = [
                    r for r in rejected_skill_records
                    if r.get("source") == "skill_proposer"
                ]
                if rejected_proposals:
                    iteration_data["rejected_proposed_skills"] = rejected_proposals
            except Exception as e:
                logger.warning(f"  SkillProposer failed (non-fatal): {e}")

        proposed_skill_names = [
            p.get("name", "") for p in accepted_proposed_skills if p.get("name")
        ]
        skills_added = sorted({
            name for name in [*skills_learned, *proposed_skill_names] if name
        })
        all_skill_records = self.skill_library.get_full_skills_for_planner(
            include_deprecated=True,
        )
        active_learned_skill_count = sum(
            1 for s in all_skill_records
            if not s.get("is_primitive", False) and s.get("tier") != "deprecated"
        )
        if rejected_skill_records:
            iteration_data["skills_rejected"] = rejected_skill_records
        if skill_lifecycle_events:
            iteration_data["skill_lifecycle_events"] = skill_lifecycle_events

        self.metrics.record_iteration(
            task=task_proposal["activity_name"],
            success=success,
            retries=attempt,
            skills_added=len(skills_added),
            reward=execution_result.get("reward", 0) if execution_result else 0,
            skills_reused=skills_reused,
            skills_failed=skills_failed,
            active_skill_count=active_learned_skill_count,
        )

        elapsed = time.time() - start_time
        total_attempts_used = min(attempt + 1, total_budget)
        iteration_data.update({
            "success": success,
            "total_attempts": total_attempts_used,
            "feedback_action": final_result["action"],
            "skills_learned": skills_learned,
            "skills_added": skills_added,
            "skills_reused": skills_reused,
            "skills_failed": skills_failed,
            "skill_failed_step": failed_skill_step,
            "skill_usage_source": skill_usage_source,
            "elapsed_seconds": elapsed,
            "skill_library_size": len(self.skill_library.get_all_skill_names()),
            "learned_skill_count": self.skill_library.get_learned_skill_count(),
            "active_learned_skill_count": active_learned_skill_count,
            "failure_memory_episodes": self.failure_memory.episode_count,
            "task_queue_size": len(self.task_queue) if self.task_queue is not None else 0,
        })

        # Feed rolling iteration history (for MemoryCurator + Verifier FM view)
        self._iter_history_for_curator.append({
            "iteration": iteration_data.get("iteration"),
            "task_proposal": {
                "activity_name": task_proposal.get("activity_name"),
                "objects": task_objects,
            },
            "success": success,
            "skills_learned": skills_learned,
            "total_attempts": total_attempts_used,
        })
        # Keep at most the last 30 so the context stays bounded
        if len(self._iter_history_for_curator) > 30:
            self._iter_history_for_curator = self._iter_history_for_curator[-30:]

        # Save human-readable trace file for this iteration
        artifact_paths = self._save_playtime_iteration_artifacts(
            iteration_data,
            task_proposal,
            plan,
            generated_skill_artifacts=generated_skill_artifacts,
        )
        if artifact_paths:
            iteration_data["iteration_artifacts"] = artifact_paths
        reused_skill_artifacts = self._save_reused_learned_skill_artifacts(
            iteration_data,
        )
        if reused_skill_artifacts:
            iteration_data["reused_learned_skill_artifacts"] = (
                reused_skill_artifacts
            )
        self._save_iteration_trace(iteration_data, task_proposal, plan,
                                    success_context, failure_context)

        return iteration_data

    def _is_playtime_proposal(self, proposal: dict[str, Any]) -> bool:
        """Return true for MolmoSpaces playtime iterations."""
        return proposal.get("_molmospaces_proposer_mode") == "playtime"

    @staticmethod
    def _artifact_slug(value: str, fallback: str = "artifact") -> str:
        """Filesystem-safe stem for generated artifact files."""
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
        return slug or fallback

    @staticmethod
    def _enable_api_execution_logging(env: Any) -> list[tuple[Any, Any]]:
        """Temporarily enable API image logging for execution artifacts."""
        states: list[tuple[Any, Any]] = []
        for api in getattr(env, "_apis", {}).values():
            enable = getattr(api, "enable_webui", None)
            if not callable(enable):
                continue
            previous = getattr(api, "_webui_enabled", None)
            try:
                enable(True)
                states.append((api, previous))
            except Exception:
                pass
        return states

    @staticmethod
    def _restore_api_execution_logging(states: list[tuple[Any, Any]]) -> None:
        for api, previous in states:
            enable = getattr(api, "enable_webui", None)
            if not callable(enable):
                continue
            try:
                enable(bool(previous) if previous is not None else False)
            except Exception:
                pass

    @staticmethod
    def _format_python_for_artifact(code: str, *, preserve_comments: bool) -> str:
        """Return deterministic Python text for an output artifact.

        Policy-writer artifacts preserve comments and exact structure because
        they are debugging evidence. Generated skill artifacts are normalized
        with ``ast.unparse`` when possible so the saved reusable function files
        are compact and consistently indented.
        """
        normalized = (code or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return ""
        if preserve_comments:
            return "\n".join(line.rstrip() for line in normalized.splitlines()) + "\n"
        try:
            return ast.unparse(ast.parse(normalized)).strip() + "\n"
        except SyntaxError:
            return "\n".join(line.rstrip() for line in normalized.splitlines()) + "\n"

    def _write_python_artifact(
        self,
        path: Path,
        *,
        code: str,
        metadata: dict[str, Any],
        preserve_comments: bool,
    ) -> None:
        """Write a Python artifact with a comment header and normalized body."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Auto-generated RATS playtime artifact.",
            "# This file is archival/debug output; robot API functions are",
            "# supplied as globals by the RATS execution environment.",
        ]
        for key, value in metadata.items():
            if value in (None, "", []):
                continue
            safe_value = json.dumps(self._json_safe(value), sort_keys=True)
            lines.append(f"# {key}: {safe_value}")
        body = self._format_python_for_artifact(
            code,
            preserve_comments=preserve_comments,
        )
        path.write_text("\n".join(lines) + "\n\n" + body)

    def _save_reused_learned_skill_artifacts(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist Python files for learned playtime skills reused this iteration."""
        try:
            iteration = int(data.get("iteration", self._iteration))
            iter_dir = self.output_dir / f"iteration_{iteration:03d}"
            skill_dir = iter_dir / "reused_playtime_skills"
            raw_skills = getattr(self.skill_library, "_skills", None)
            source_skills = (
                list(raw_skills)
                if isinstance(raw_skills, list)
                else self.skill_library.get_full_skills_for_planner(
                    include_deprecated=True,
                )
            )
            skill_by_name = {
                str(skill.get("name")): skill
                for skill in source_skills
                if skill.get("name") and not skill.get("is_primitive", False)
            }
            learned_names = set(skill_by_name)
            reused: dict[str, dict[str, Any]] = {}
            for key in sorted(
                (
                    k for k in data
                    if k.startswith("code_attempt_") and data.get(k)
                ),
                key=lambda k: int(k.rsplit("_", 1)[-1]),
            ):
                flat_step = int(key.rsplit("_", 1)[-1])
                code = str(data.get(key) or "")
                called = _extract_called_learned_skills(code, learned_names)
                for name in called:
                    skill = skill_by_name.get(name)
                    if not skill or not skill.get("code"):
                        continue
                    source_task = str(skill.get("source_task") or "")
                    is_playtime_skill = (
                        "playtime" in source_task
                        or source_task.startswith("play:")
                    )
                    if not is_playtime_skill:
                        continue
                    entry = reused.setdefault(
                        name,
                        {
                            "skill": skill,
                            "flat_steps": [],
                            "attempts": [],
                        },
                    )
                    entry["flat_steps"].append(flat_step)
                    entry["attempts"].append({
                        "flat_step": flat_step,
                        "attempt_in_iteration": (
                            flat_step // max(1, self._turns_per_attempt)
                        ),
                        "turn_in_attempt": (
                            flat_step % max(1, self._turns_per_attempt)
                        ),
                    })

            if not reused:
                return {}

            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_iteration_map = self._learned_skill_iteration_map()
            files: list[dict[str, Any]] = []
            used_stems: set[str] = set()
            for name, entry in sorted(reused.items()):
                skill = entry["skill"]
                stem = self._artifact_slug(name, "learned_skill")
                if stem in used_stems:
                    suffix = 2
                    base = stem
                    while f"{base}_{suffix}" in used_stems:
                        suffix += 1
                    stem = f"{base}_{suffix}"
                used_stems.add(stem)
                path = skill_dir / f"{stem}.py"
                self._write_python_artifact(
                    path,
                    code=str(skill.get("code") or ""),
                    preserve_comments=False,
                    metadata={
                        "iteration": iteration,
                        "skill": name,
                        "artifact_kind": "reused_playtime_skill",
                        "source_task": skill.get("source_task"),
                        "learned_iteration": (
                            skill.get("learned_iteration")
                            or skill_iteration_map.get(name)
                        ),
                        "tier": skill.get("tier"),
                        "usage_count": skill.get("usage_count"),
                        "success_count": skill.get("success_count"),
                        "reused_in": entry["attempts"],
                        "description": skill.get("description"),
                        "api_primitives_used": skill.get("api_primitives_used"),
                    },
                )
                files.append({
                    "skill": name,
                    "file": str(path),
                    "flat_steps": entry["flat_steps"],
                    "attempts": entry["attempts"],
                    "learned_iteration": (
                        skill.get("learned_iteration")
                        or skill_iteration_map.get(name)
                    ),
                })

            manifest = {
                "iteration": iteration,
                "directory": str(skill_dir),
                "files": files,
            }
            (skill_dir / "manifest.json").write_text(json.dumps(
                self._json_safe(manifest),
                indent=2,
            ))
            logger.info(
                "  Reused playtime skill artifacts: %s (%d skill(s))",
                skill_dir,
                len(files),
            )
            return manifest
        except Exception as e:
            logger.debug(f"  reused skill artifact save failed: {e}")
            return {}

    def _run_attempt_via_multiturn_reset(
        self,
        *,
        plan: dict[str, Any],
        scene_context: dict[str, Any],
        skill_preamble: str,
        failure_context: str,
        success_context: str,
        attempt: int,
        attempt_in_iter: int,
        iteration_data: dict[str, Any],
        learned_skill_names: list[str],
    ) -> dict[str, Any]:
        """Run one attempt's writer→exec→per-step block via MultiturnResetExecutor.

        Returns the variables the legacy 5a-5d path would have set
        (``code``, ``execution_result``, ``quality``, ``policy_ready``,
        ``per_step_verification``, ...) plus a ``multiturn_reset_result_dict``
        suitable for persistence in ``iteration_data``.

        Side effects: writes ``multiturn_reset_attempt_{attempt}``,
        ``per_step_verification_attempt_{attempt}``, and a stub
        ``quality_attempt_{attempt}`` / ``policy_self_check_attempt_{attempt}``
        into ``iteration_data`` so report rendering still finds them.
        """
        from rats.loop.multiturn_reset_executor import MultiturnResetExecutor

        out_dir = (
            self.output_dir
            / f"iteration_{self._iteration:03d}"
            / f"attempt_{attempt_in_iter:02d}"
            / "multiturn"
        )
        try:
            mt = self.multiturn_reset_executor.run(
                env=self.env,
                plan=plan,
                scene_context=scene_context,
                skill_preamble=skill_preamble,
                failure_context=failure_context,
                success_context=success_context,
                iteration=self._iteration,
                attempt_in_iter=attempt_in_iter,
                output_dir=out_dir,
                learned_skill_names=learned_skill_names,
            )
        except Exception as exc:
            logger.exception("Multiturn-reset run failed; falling back to no-op attempt: %s", exc)
            mt = None

        if mt is None:
            # Fail-safe: synthesise a minimal failure so the rest of the
            # iteration loop can still run + record + retry.
            execution_result = {
                "success": False,
                "stdout": "",
                "stderr": "multiturn_reset_internal_error",
                "reward": 0.0,
                "task_completed": False,
                "artifacts": {},
            }
            multiturn_reset_result_dict = {"status": "internal_error"}
            code = ""
            per_step_verification = {
                "enabled": True, "summary_text": "multiturn-reset internal error",
                "steps": [], "multiturn_reset": True,
            }
        else:
            execution_result = mt.execution_result or {
                "success": False, "stdout": "", "stderr": "multiturn-reset produced no exec result",
                "reward": 0.0, "task_completed": False, "artifacts": {},
            }
            code = mt.final_code
            multiturn_reset_result_dict = MultiturnResetExecutor.result_to_dict(mt)

            # Synthesize a per_step_verification matching the legacy shape.
            combined = list(mt.per_step_verifications)
            if mt.status == "step_stagnation" and mt.stuck_step_idx is not None:
                last_records = mt.step_retry_log.get(mt.stuck_step_idx) or []
                last = last_records[-1] if last_records else None
                if last is not None:
                    combined.append({
                        "step_id": mt.stuck_step_id or f"step-{mt.stuck_step_idx + 1}",
                        "status": last.ps_status or "failed",
                        "success": False,
                        "confidence": last.ps_confidence or 0.0,
                        "reason": last.ps_reason or mt.stuck_reason or "step stagnated",
                    })
            summary_lines = ["Per-step verification (multiturn-reset mode):"]
            for s in combined:
                ok = bool(s.get("success") or s.get("status") == "succeeded")
                mark = "✓" if ok else "✗"
                summary_lines.append(
                    f"- {mark} {s.get('step_id', '?')}: {s.get('status', '?')} "
                    f"conf={s.get('confidence', '?')} — {(s.get('reason') or '')[:160]}"
                )
            per_step_verification = {
                "enabled": True,
                "schema_version": "rats_per_step_verification_v1",
                "verifier_backend": "vlm",
                "summary_text": "\n".join(summary_lines),
                "steps": combined,
                "multiturn_reset": True,
                "multiturn_reset_status": mt.status,
                "stuck_step_idx": mt.stuck_step_idx,
                "stuck_step_id": mt.stuck_step_id,
            }

        # Tag execution_result so the post-diagnoser hook in
        # _run_one_iteration can detect step_stagnation and force the
        # legacy plan-refinement pathway (requirement 2: "if a step keeps
        # failing, re-plan before the next attempt").
        execution_result.setdefault("artifacts", {})["multiturn_reset_status"] = (
            multiturn_reset_result_dict or {}
        ).get("status")
        execution_result["artifacts"]["multiturn_reset_stuck_step_id"] = (
            multiturn_reset_result_dict or {}
        ).get("stuck_step_id")
        execution_result["artifacts"]["multiturn_reset_stuck_step_idx"] = (
            multiturn_reset_result_dict or {}
        ).get("stuck_step_idx")
        execution_result["artifacts"]["multiturn_reset_stuck_reason"] = (
            multiturn_reset_result_dict or {}
        ).get("stuck_reason")
        # Surface force-committed + skipped step lists so:
        # (a) the post-diagnoser hook can enrich plan_issue_reason with
        #     the full list (not just the first stuck step), and
        # (b) the success-branch can intercept partial_success to skip
        #     _successful_code caching + skill extraction (force-committed
        #     step code is UNVERIFIED — PS never approved it).
        execution_result["artifacts"]["multiturn_reset_force_committed_indices"] = list(
            (multiturn_reset_result_dict or {}).get("force_committed_indices") or []
        )
        execution_result["artifacts"]["multiturn_reset_skipped_indices"] = list(
            (multiturn_reset_result_dict or {}).get("skipped_indices") or []
        )
        # Resolve the force-committed indices to step IDs (for prompt text).
        plan_steps = list(plan.get("steps") or [])
        def _step_id_at(idx: int) -> str:
            if 0 <= idx < len(plan_steps):
                s = plan_steps[idx]
                return str(s.get("id") or s.get("step_id") or f"step-{idx + 1}")
            return f"step-{idx + 1}"
        execution_result["artifacts"]["multiturn_reset_force_committed_step_ids"] = [
            _step_id_at(i)
            for i in execution_result["artifacts"]["multiturn_reset_force_committed_indices"]
        ]
        execution_result["artifacts"]["multiturn_reset_skipped_step_ids"] = [
            _step_id_at(i)
            for i in execution_result["artifacts"]["multiturn_reset_skipped_indices"]
        ]
        # Build the "natural-committed only" code (concatenation of PS-
        # verified step bodies only, skipping force-committed and
        # skipped steps). Used by the skill-reliability tracker on
        # partial_success so that record_usage(success=True) only
        # credits skills that ran in PS-verified context. Without this,
        # a force-committed step's `os.system`-like primitive would
        # get a +1 success credit purely because the verifier saw
        # reward=1 (likely due to some OTHER step in the same plan
        # accomplishing the goal). That's a false positive in the
        # skill reliability ledger.
        if mt is not None:
            forced_set = set(execution_result["artifacts"]["multiturn_reset_force_committed_indices"])
            skipped_set = set(execution_result["artifacts"]["multiturn_reset_skipped_indices"])
            natural_only_parts: list[str] = []
            for idx in sorted(mt.committed_step_codes.keys()):
                if idx in forced_set or idx in skipped_set:
                    continue
                body = mt.committed_step_codes.get(idx, "")
                if body.strip():
                    natural_only_parts.append(body)
            execution_result["artifacts"]["multiturn_reset_natural_committed_code"] = (
                "\n\n".join(natural_only_parts)
            )
        else:
            execution_result["artifacts"]["multiturn_reset_natural_committed_code"] = ""

        # Persist iteration_data fields the legacy path would have set so
        # the report renderer + resume logic still find them.
        iteration_data[f"multiturn_reset_attempt_{attempt}"] = multiturn_reset_result_dict
        iteration_data[f"per_step_verification_attempt_{attempt}"] = {
            "enabled": per_step_verification.get("enabled", False),
            "summary_text": per_step_verification.get("summary_text"),
            "steps": per_step_verification.get("steps", []),
            "multiturn_reset": True,
            "multiturn_reset_status": (multiturn_reset_result_dict or {}).get("status"),
        }
        # Real per-step quality results from the inline gate, aggregated
        # across all retries this attempt. Replaces the prior stub that
        # always wrote approved=True — see commit history for the bug
        # this fixes (writer output bypassed PolicyQualityChecker even
        # though the artifact claimed otherwise).
        qc_results = list((multiturn_reset_result_dict or {}).get("step_quality_results") or [])
        tier1_all: list[str] = []
        tier2_all: list[str] = []
        any_tier1_blocked = False
        for qr in qc_results:
            if not qr.get("approved", True):
                any_tier1_blocked = True
            for v in (qr.get("tier1_violations") or []):
                tier1_all.append(f"step{qr.get('step_idx')}-retry{qr.get('retry')}: {v}")
            for it in (qr.get("tier2_issues") or []):
                tier2_all.append(f"step{qr.get('step_idx')}-retry{qr.get('retry')}: {it}")
        iteration_data[f"quality_attempt_{attempt}"] = {
            # Aggregate "approved" is True if the attempt actually ran
            # (i.e. MT-RESET produced executable composed code, even if
            # some retries got Tier-1 blocked along the way). The legacy
            # gate at line 3235 uses approved=False to mean "abort the
            # whole attempt, never executed", which doesn't fit
            # MT-RESET's per-step semantics. Tier-1 detail is preserved
            # in tier1_violations + had_tier1_blocks for traceability;
            # the run-report renderer also reads `approved` to decide
            # whether to show the BLOCKED card or the exec result.
            "approved": True,
            "had_tier1_blocks": any_tier1_blocked,
            "feedback": (
                "multiturn-reset inline per-step quality gate"
                + (f" — {len(tier1_all)} Tier-1 block(s) (handled inline)" if tier1_all else "")
                + (f" — {len(tier2_all)} Tier-2 advisory" if tier2_all else "")
            ),
            "tier1_violations": tier1_all,
            "tier2_issues": tier2_all,
            "per_retry_results": qc_results,
            "multiturn_reset": True,
        }
        # Runtime self-check is intentionally a no-op in multiturn-reset
        # mode: every retry is itself a real env.execute + per-step
        # verifier judgment, which subsumes the legacy self-check's job
        # of "catch Python/API runtime crashes before official execution".
        # Be honest about why instead of writing passed=True.
        iteration_data[f"policy_self_check_attempt_{attempt}"] = {
            "enabled": False,
            "passed": True,
            "reason": (
                "subsumed by multiturn-reset: each retry is a real "
                "env.execute + per-step verifier judgment, so a separate "
                "runtime self-check would just re-execute the same code"
            ),
            "multiturn_reset": True,
        }
        # Synthesize an execution_attempt entry up-front; the legacy 5e
        # path adds richer fields but multiturn-mode persistence is
        # entirely on this stub.
        iteration_data[f"execution_attempt_{attempt}"] = {
            "success": execution_result.get("success"),
            "task_completed": execution_result.get("task_completed"),
            "reward": execution_result.get("reward"),
            "stderr_snippet": (execution_result.get("stderr") or "")[:2000],
            "user_result": execution_result.get("user_result"),
            "multiturn_reset": True,
            "multiturn_reset_status": (multiturn_reset_result_dict or {}).get("status"),
            "multiturn_reset_committed_steps": (multiturn_reset_result_dict or {}).get("committed_step_count"),
            "multiturn_reset_total_retries": (multiturn_reset_result_dict or {}).get("total_step_retries"),
        }

        return {
            "code": code,
            "exec_code": code,
            "execution_result": execution_result,
            "quality": iteration_data[f"quality_attempt_{attempt}"],
            "policy_ready": {
                "passed": True,
                "self_check": iteration_data[f"policy_self_check_attempt_{attempt}"],
                "repairs_used": 0,
            },
            "per_step_verification": per_step_verification,
            "multiturn_reset_result_dict": multiturn_reset_result_dict,
        }

    def _save_policy_writer_retry_artifacts(
        self,
        data: dict[str, Any],
        attempt: int,
        retry_feedback: dict[str, Any] | None,
        *,
        attempt_in_iter: int | None = None,
        turn_in_attempt: int | None = None,
        task_in_progress: bool = False,
    ) -> dict[str, Any]:
        """Persist the exact retry text and diagnostic context sent to writer.

        ``diagnosis_attempt_N.policy_feedback`` records only the diagnoser's
        core suggestion. This artifact records the expanded retry block that
        PolicyWriter injected into its prompt, plus the raw diagnostic context
        package (including artifact image data URLs when present) in a separate
        JSON file. Keeping these out of ``iteration_NNN.json`` avoids bloating
        the main run ledger while preserving exact retry evidence on disk.
        """
        if not retry_feedback:
            return {}

        retry_text = str(
            getattr(self.policy_writer, "last_retry_context_text", "") or ""
        )
        full_prompt = str(
            getattr(self.policy_writer, "last_user_prompt_text", "") or ""
        )
        diagnostic_context = getattr(
            self.policy_writer, "last_retry_diagnostic_context", {},
        ) or {}
        if not retry_text and not full_prompt and not diagnostic_context:
            return {}

        try:
            iteration = int(data.get("iteration", self._iteration))
            iter_dir = self.output_dir / f"iteration_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            if attempt_in_iter is not None:
                prefix = f"attempt{int(attempt_in_iter):02d}"
                if turn_in_attempt is not None:
                    prefix += f"_turn{int(turn_in_attempt):02d}"
                prefix += f"_step{int(attempt):02d}"
            else:
                prefix = f"step{int(attempt):02d}"
            retry_path = iter_dir / f"policy_writer_retry_{prefix}.md"
            prompt_path = iter_dir / f"policy_writer_prompt_{prefix}.txt"
            context_path = iter_dir / f"diagnostic_context_{prefix}.json"

            header_lines = [
                f"# Policy writer retry context — iteration {iteration}, step {attempt}",
                "",
                f"- **flat_step:** `{attempt}`",
                f"- **attempt_in_iteration:** `{attempt_in_iter}`",
                f"- **turn_in_attempt:** `{turn_in_attempt}`",
                f"- **task_in_progress:** `{bool(task_in_progress)}`",
                f"- **failure_mode:** `{retry_feedback.get('failure_mode', '')}`",
                f"- **failed_step:** `{retry_feedback.get('failed_step', '')}`",
                f"- **diagnostic_image_count:** `{getattr(self.policy_writer, 'last_retry_diagnostic_image_count', 0)}`",
                "",
                "## Exact retry block inserted into policy-writer prompt",
                "",
                "```text",
                retry_text.rstrip(),
                "```",
                "",
            ]
            retry_path.write_text("\n".join(header_lines))

            if full_prompt:
                prompt_path.write_text(full_prompt)

            context_payload = {
                "iteration": iteration,
                "flat_step": attempt,
                # Back-compat: `attempt` remains the flat loop step.
                "attempt": attempt,
                "attempt_in_iteration": attempt_in_iter,
                "turn_in_attempt": turn_in_attempt,
                "task_in_progress": bool(task_in_progress),
                "diagnostic_image_count": getattr(
                    self.policy_writer,
                    "last_retry_diagnostic_image_count",
                    0,
                ),
                "diagnostic_context": diagnostic_context,
            }
            context_path.write_text(json.dumps(
                self._json_safe(context_payload),
                indent=2,
            ))

            record = {
                "flat_step": attempt,
                # Back-compat: `attempt` remains the flat loop step.
                "attempt": attempt,
                "attempt_in_iteration": attempt_in_iter,
                "turn_in_attempt": turn_in_attempt,
                "task_in_progress": bool(task_in_progress),
                "retry_text_file": str(retry_path),
                "full_prompt_file": str(prompt_path) if full_prompt else "",
                "diagnostic_context_file": str(context_path),
                "diagnostic_image_count": getattr(
                    self.policy_writer,
                    "last_retry_diagnostic_image_count",
                    0,
                ),
            }
            data.setdefault("policy_writer_retry_artifacts", []).append(record)
            logger.info(
                "  Policy-writer retry artifacts: %s, %s, %s",
                retry_path,
                prompt_path if full_prompt else "(no full prompt)",
                context_path,
            )
            return record
        except Exception as e:
            logger.debug(f"  policy-writer retry artifact save failed: {e}")
            return {}

    @staticmethod
    def _diagnoser_image_extension(data_url: str) -> tuple[str, str, str] | None:
        match = re.match(
            r"^data:(image/[A-Za-z0-9.+-]+);base64,(.*)$",
            data_url,
            flags=re.DOTALL,
        )
        if not match:
            return None
        mime = match.group(1).lower()
        ext = {
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
        }.get(mime, "png")
        return mime, ext, match.group(2)

    @staticmethod
    def _diagnoser_video_extension(data_url: str) -> tuple[str, str, str] | None:
        match = re.match(
            r"^data:(video/[A-Za-z0-9.+-]+);base64,(.*)$",
            data_url,
            flags=re.DOTALL,
        )
        if not match:
            return None
        mime = match.group(1).lower()
        ext = {
            "video/mp4": "mp4",
            "video/mpeg": "mpeg",
            "video/webm": "webm",
            "video/quicktime": "mov",
        }.get(mime, "mp4")
        return mime, ext, match.group(2)

    def _save_diagnoser_input_image_artifacts(
        self,
        data: dict[str, Any],
        diagnosis: dict[str, Any],
        attempt: int,
    ) -> list[dict[str, Any]]:
        """Persist VLM images sent to FailureDiagnoser without bloating iteration JSON."""
        raw_records = diagnosis.get("diagnoser_input_images") or []
        if not isinstance(raw_records, list) or not raw_records:
            return []

        try:
            iteration = int(data.get("iteration", self._iteration))
            image_dir = (
                self.output_dir
                / f"iteration_{iteration:03d}"
                / "diagnoser_input_images"
            )
            image_dir.mkdir(parents=True, exist_ok=True)

            saved: list[dict[str, Any]] = []
            for idx, raw in enumerate(raw_records, start=1):
                if not isinstance(raw, dict):
                    continue
                data_url = str(raw.get("data_url") or "")
                parsed = self._diagnoser_image_extension(data_url)
                if parsed is None:
                    continue
                mime, ext, payload = parsed
                try:
                    image_bytes = base64.b64decode(payload, validate=False)
                except Exception:
                    continue
                image_index = int(raw.get("image_index") or idx)
                path = image_dir / f"attempt_{attempt:02d}_image_{image_index:02d}.{ext}"
                path.write_bytes(image_bytes)

                record = {
                    key: self._json_safe(value)
                    for key, value in raw.items()
                    if key != "data_url"
                }
                record.update({
                    "mime_type": mime,
                    "file": str(path),
                    "relative_file": str(path.relative_to(self.output_dir)),
                })
                saved.append(record)
            return saved
        except Exception as e:
            logger.debug(f"  diagnoser input image artifact save failed: {e}")
            return []

    def _save_diagnoser_input_video_artifacts(
        self,
        data: dict[str, Any],
        diagnosis: dict[str, Any],
        attempt: int,
    ) -> list[dict[str, Any]]:
        """Persist VLM videos sent to FailureDiagnoser without bloating iteration JSON."""
        raw_records = diagnosis.get("diagnoser_input_videos") or []
        if not isinstance(raw_records, list) or not raw_records:
            return []

        try:
            iteration = int(data.get("iteration", self._iteration))
            video_dir = (
                self.output_dir
                / f"iteration_{iteration:03d}"
                / "diagnoser_input_videos"
            )
            video_dir.mkdir(parents=True, exist_ok=True)

            saved: list[dict[str, Any]] = []
            for idx, raw in enumerate(raw_records, start=1):
                if not isinstance(raw, dict):
                    continue
                data_url = str(raw.get("data_url") or "")
                parsed = self._diagnoser_video_extension(data_url)
                if parsed is None:
                    continue
                mime, ext, payload = parsed
                try:
                    video_bytes = base64.b64decode(payload, validate=False)
                except Exception:
                    continue
                video_index = int(raw.get("video_index") or idx)
                path = video_dir / f"attempt_{attempt:02d}_video_{video_index:02d}.{ext}"
                path.write_bytes(video_bytes)

                record = {
                    key: self._json_safe(value)
                    for key, value in raw.items()
                    if key != "data_url"
                }
                record.update({
                    "mime_type": mime,
                    "file": str(path),
                    "relative_file": str(path.relative_to(self.output_dir)),
                })
                saved.append(record)
            return saved
        except Exception as e:
            logger.debug(f"  diagnoser input video artifact save failed: {e}")
            return []

    @staticmethod
    def _execution_image_extension(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\xff\xd8"):
            return "jpg"
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return "webp"
        return "jpg"

    def _save_execution_history_artifacts(
        self,
        data: dict[str, Any],
        exec_history: Any,
        attempt: int,
    ) -> dict[str, Any]:
        """Persist API execution logger images for post-hoc WebUI/debugging."""
        steps = list(getattr(exec_history, "steps", []) or [])
        if not steps:
            return {}

        try:
            iteration = int(data.get("iteration", self._iteration))
            history_dir = (
                self.output_dir
                / f"iteration_{iteration:03d}"
                / "execution_history"
                / f"attempt_{attempt:02d}"
            )
            history_dir.mkdir(parents=True, exist_ok=True)

            saved_steps: list[dict[str, Any]] = []
            total_images = 0
            for fallback_idx, step in enumerate(steps):
                step_index = int(getattr(step, "step_index", fallback_idx))
                tool_name = str(getattr(step, "tool_name", "") or "Execution Step")
                step_slug = self._artifact_slug(
                    f"step_{step_index:02d}_{tool_name}",
                    f"step_{step_index:02d}",
                )
                image_records: list[dict[str, Any]] = []
                for image_idx, image_b64 in enumerate(
                    list(getattr(step, "images", []) or []),
                    start=1,
                ):
                    if not isinstance(image_b64, str) or not image_b64.strip():
                        continue
                    payload = (
                        image_b64.split(",", 1)[-1]
                        if image_b64.startswith("data:")
                        else image_b64
                    )
                    try:
                        image_bytes = base64.b64decode(payload, validate=False)
                    except Exception:
                        continue
                    ext = self._execution_image_extension(image_bytes)
                    image_path = history_dir / f"{step_slug}_image_{image_idx:02d}.{ext}"
                    image_path.write_bytes(image_bytes)
                    image_records.append({
                        "image_index": image_idx,
                        "file": str(image_path),
                        "relative_file": str(image_path.relative_to(self.output_dir)),
                    })
                total_images += len(image_records)
                saved_steps.append({
                    "step_index": step_index,
                    "tool_name": tool_name,
                    "text": str(getattr(step, "text", "") or ""),
                    "timestamp": str(getattr(step, "timestamp", "") or ""),
                    "highlight": bool(getattr(step, "highlight", False)),
                    "image_count": len(image_records),
                    "images": image_records,
                })

            manifest = {
                "directory": str(history_dir),
                "relative_directory": str(history_dir.relative_to(self.output_dir)),
                "step_count": len(saved_steps),
                "image_count": total_images,
                "steps": saved_steps,
            }
            manifest_path = history_dir / "execution_history.json"
            manifest_path.write_text(json.dumps(self._json_safe(manifest), indent=2))
            manifest["manifest_file"] = str(manifest_path)
            manifest["relative_manifest_file"] = str(
                manifest_path.relative_to(self.output_dir)
            )
            return manifest
        except Exception as e:
            logger.debug(f"  execution history artifact save failed: {e}")
            return {}

    @staticmethod
    def _execution_image_extension(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\xff\xd8"):
            return "jpg"
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return "webp"
        return "jpg"

    def _save_execution_history_artifacts(
        self,
        data: dict[str, Any],
        exec_history: Any,
        attempt: int,
    ) -> dict[str, Any]:
        """Persist API execution logger images for post-hoc WebUI/debugging."""
        steps = list(getattr(exec_history, "steps", []) or [])
        if not steps:
            return {}

        try:
            iteration = int(data.get("iteration", self._iteration))
            history_dir = (
                self.output_dir
                / f"iteration_{iteration:03d}"
                / "execution_history"
                / f"attempt_{attempt:02d}"
            )
            history_dir.mkdir(parents=True, exist_ok=True)

            saved_steps: list[dict[str, Any]] = []
            total_images = 0
            for fallback_idx, step in enumerate(steps):
                step_index = int(getattr(step, "step_index", fallback_idx))
                tool_name = str(getattr(step, "tool_name", "") or "Execution Step")
                step_slug = self._artifact_slug(
                    f"step_{step_index:02d}_{tool_name}",
                    f"step_{step_index:02d}",
                )
                image_records: list[dict[str, Any]] = []
                for image_idx, image_b64 in enumerate(
                    list(getattr(step, "images", []) or []),
                    start=1,
                ):
                    if not isinstance(image_b64, str) or not image_b64.strip():
                        continue
                    payload = (
                        image_b64.split(",", 1)[-1]
                        if image_b64.startswith("data:")
                        else image_b64
                    )
                    try:
                        image_bytes = base64.b64decode(payload, validate=False)
                    except Exception:
                        continue
                    ext = self._execution_image_extension(image_bytes)
                    image_path = history_dir / f"{step_slug}_image_{image_idx:02d}.{ext}"
                    image_path.write_bytes(image_bytes)
                    image_records.append({
                        "image_index": image_idx,
                        "file": str(image_path),
                        "relative_file": str(image_path.relative_to(self.output_dir)),
                    })
                total_images += len(image_records)
                saved_steps.append({
                    "step_index": step_index,
                    "tool_name": tool_name,
                    "text": str(getattr(step, "text", "") or ""),
                    "timestamp": str(getattr(step, "timestamp", "") or ""),
                    "highlight": bool(getattr(step, "highlight", False)),
                    "frame_start": getattr(step, "frame_start", None),
                    "frame_end": getattr(step, "frame_end", None),
                    "timeline_kind": getattr(step, "timeline_kind", None),
                    "timeline_label": getattr(step, "timeline_label", None),
                    "policy_step_id": getattr(step, "policy_step_id", None),
                    "policy_step_index": getattr(step, "policy_step_index", None),
                    "policy_step_goal": getattr(step, "policy_step_goal", None),
                    "policy_step_marker_index": getattr(step, "policy_step_marker_index", None),
                    "image_count": len(image_records),
                    "images": image_records,
                })

            manifest = {
                "directory": str(history_dir),
                "relative_directory": str(history_dir.relative_to(self.output_dir)),
                "step_count": len(saved_steps),
                "image_count": total_images,
                "steps": saved_steps,
            }
            manifest_path = history_dir / "execution_history.json"
            manifest_path.write_text(json.dumps(self._json_safe(manifest), indent=2))
            manifest["manifest_file"] = str(manifest_path)
            manifest["relative_manifest_file"] = str(
                manifest_path.relative_to(self.output_dir)
            )
            return manifest
        except Exception as e:
            logger.debug(f"  execution history artifact save failed: {e}")
            return {}

    def _build_step_frame_segments(
        self,
        exec_history: Any,
        turn_frame_start: int,
        turn_frame_end: int,
        turn_frames: list[Any],
    ) -> list[dict[str, Any]]:
        """Build exact per-policy-step frame slices from execution markers."""
        steps = list(getattr(exec_history, "steps", []) or [])
        if not steps or not turn_frames:
            return []
        segments: list[dict[str, Any]] = []
        for fallback_idx, step in enumerate(steps):
            if str(getattr(step, "timeline_kind", "") or "") != "policy_step":
                continue
            start = getattr(step, "frame_start", None)
            end = getattr(step, "frame_end", None)
            try:
                start_i = int(start) if start is not None else int(turn_frame_start)
            except Exception:
                start_i = int(turn_frame_start)
            try:
                end_i = int(end) if end is not None else int(start_i)
            except Exception:
                end_i = int(start_i)
            start_i = max(int(turn_frame_start), min(int(turn_frame_end), start_i))
            end_i = max(start_i, min(int(turn_frame_end), end_i))
            local_start = max(0, start_i - int(turn_frame_start))
            local_end = max(local_start + 1, end_i - int(turn_frame_start))
            local_end = min(len(turn_frames), local_end)
            raw_segment = turn_frames[local_start:local_end]
            if not raw_segment and 0 <= local_start < len(turn_frames):
                raw_segment = [turn_frames[local_start]]
            sampled = _sample_trajectory_frames(raw_segment) if raw_segment else []
            segments.append(
                {
                    "policy_step_id": getattr(step, "policy_step_id", None),
                    "policy_step_index": getattr(step, "policy_step_index", None),
                    "policy_step_goal": getattr(step, "policy_step_goal", None),
                    "policy_step_marker_index": getattr(step, "policy_step_marker_index", fallback_idx),
                    "frame_start": start_i,
                    "frame_end": end_i,
                    "local_frame_start": local_start,
                    "local_frame_end": local_end,
                    "frame_count": len(raw_segment),
                    "sampled_frame_count": len(sampled),
                    "sampled_frames": sampled,
                }
            )
        return segments


    def _save_playtime_iteration_artifacts(
        self,
        data: dict[str, Any],
        proposal: dict[str, Any],
        plan: dict[str, Any],
        *,
        generated_skill_artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Write playtime per-iteration folder artifacts.

        For each playtime iteration this creates ``iteration_###/`` with:
        - raw policy-writer code attempts as ``policy_attempt_XX.py``;
        - generated reusable skills as ``generated_skills/<skill>.py``;
        - ``proposed_task.json`` / ``proposed_task.md``;
        - ``summary.json`` / ``summary.md``.
        """
        if not self._is_playtime_proposal(proposal):
            return {}

        try:
            iteration = int(data.get("iteration", self._iteration))
            iter_dir = self.output_dir / f"iteration_{iteration:03d}"
            skill_dir = iter_dir / "generated_skills"
            iter_dir.mkdir(parents=True, exist_ok=True)
            skill_dir.mkdir(parents=True, exist_ok=True)

            safe_proposal = self._json_safe(proposal)
            proposed_task_json = iter_dir / "proposed_task.json"
            proposed_task_json.write_text(json.dumps(safe_proposal, indent=2))

            proposed_task_md = iter_dir / "proposed_task.md"
            proposed_task_md.write_text("\n".join([
                f"# Proposed task for iteration {iteration}",
                "",
                f"- **Activity:** `{proposal.get('activity_name', '')}`",
                f"- **Goal:** {proposal.get('goal_conditions') or proposal.get('language', '')}",
                f"- **Scene:** `{proposal.get('scene_model', '')}`",
                f"- **Mode:** `{proposal.get('mode', '')}`",
                f"- **Task family:** `{proposal.get('task_family', '')}`",
                f"- **Reasoning:** {proposal.get('reasoning', '')}",
                f"- **Expected new skills:** {proposal.get('expected_new_skills', [])}",
                "",
            ]))

            policy_files: list[str] = []
            code_attempt_keys = sorted(
                (
                    key for key in data
                    if key.startswith("code_attempt_") and data.get(key)
                ),
                key=lambda key: int(key.rsplit("_", 1)[-1]),
            )
            for key in code_attempt_keys:
                attempt_idx = int(key.rsplit("_", 1)[-1])
                path = iter_dir / f"policy_attempt_{attempt_idx:02d}.py"
                self._write_python_artifact(
                    path,
                    code=str(data.get(key) or ""),
                    preserve_comments=True,
                    metadata={
                        "iteration": iteration,
                        "attempt": attempt_idx,
                        "task": proposal.get("activity_name"),
                        "goal": proposal.get("goal_conditions")
                        or proposal.get("language", ""),
                    },
                )
                policy_files.append(path.name)

            skill_files: list[dict[str, str]] = []
            used_skill_stems: set[str] = set()
            for skill in generated_skill_artifacts:
                code = str(skill.get("code") or "").strip()
                if not code:
                    continue
                name = str(skill.get("name") or "unnamed_skill")
                stem = self._artifact_slug(name, "unnamed_skill")
                if stem in used_skill_stems:
                    suffix = 2
                    base = stem
                    while f"{base}_{suffix}" in used_skill_stems:
                        suffix += 1
                    stem = f"{base}_{suffix}"
                used_skill_stems.add(stem)
                path = skill_dir / f"{stem}.py"
                self._write_python_artifact(
                    path,
                    code=code,
                    preserve_comments=False,
                    metadata={
                        "iteration": iteration,
                        "skill": name,
                        "source": skill.get("source"),
                        "source_task": skill.get("source_task"),
                        "learned_iteration": skill.get("learned_iteration", iteration),
                        "stored_in_library": skill.get("stored_in_library"),
                        "rationale": skill.get("rationale"),
                        "description": skill.get("description"),
                        "api_primitives_used": skill.get("api_primitives_used"),
                    },
                )
                skill_files.append({"skill": name, "file": f"generated_skills/{path.name}"})

            summary = {
                "iteration": iteration,
                "activity_name": proposal.get("activity_name"),
                "goal": proposal.get("goal_conditions")
                or proposal.get("language", ""),
                "success": data.get("success"),
                "total_attempts": data.get("total_attempts"),
                "feedback_action": data.get("feedback_action"),
                "elapsed_seconds": data.get("elapsed_seconds"),
                "skills_learned": data.get("skills_learned", []),
                "proposed_skills": data.get("proposed_skills", []),
                "policy_files": policy_files,
                "policy_writer_retry_artifacts": data.get(
                    "policy_writer_retry_artifacts", [],
                ),
                "verifier_artifacts": data.get("verifier_artifacts", []),
                "generated_skill_files": skill_files,
            }
            (iter_dir / "summary.json").write_text(json.dumps(
                self._json_safe(summary),
                indent=2,
            ))

            summary_lines = [
                f"# Iteration {iteration} summary",
                "",
                f"- **Task:** `{summary['activity_name']}`",
                f"- **Goal:** {summary['goal']}",
                f"- **Success:** {summary['success']}",
                f"- **Attempts:** {summary['total_attempts']}",
                f"- **Feedback action:** `{summary['feedback_action']}`",
                f"- **Elapsed:** {float(summary.get('elapsed_seconds') or 0):.1f}s",
                f"- **Skills learned:** {summary['skills_learned']}",
                f"- **Proposed skills:** {summary['proposed_skills']}",
                "",
                "## Policy writer code",
            ]
            if policy_files:
                summary_lines.extend(f"- [{name}]({name})" for name in policy_files)
            else:
                summary_lines.append("- None")
            retry_artifacts = data.get("policy_writer_retry_artifacts", []) or []
            summary_lines.extend(["", "## Policy writer retry context"])
            if retry_artifacts:
                for entry in retry_artifacts:
                    retry_name = Path(str(entry.get("retry_text_file", ""))).name
                    prompt_name = Path(str(entry.get("full_prompt_file", ""))).name
                    context_name = Path(str(entry.get("diagnostic_context_file", ""))).name
                    bits = [
                        (
                            f"attempt {entry.get('attempt_in_iteration')} "
                            f"turn {entry.get('turn_in_attempt')} "
                            f"(step {entry.get('flat_step', entry.get('attempt'))})"
                        )
                    ]
                    if retry_name:
                        bits.append(f"[retry text]({retry_name})")
                    if prompt_name:
                        bits.append(f"[full prompt]({prompt_name})")
                    if context_name:
                        bits.append(f"[diagnostic context]({context_name})")
                    summary_lines.append("- " + " · ".join(bits))
            else:
                summary_lines.append("- None")
            verifier_artifacts = data.get("verifier_artifacts", []) or []
            summary_lines.extend(["", "## Verifier artifacts"])
            if verifier_artifacts:
                for entry in verifier_artifacts:
                    summary_name = Path(str(entry.get("summary", ""))).name
                    prompt_name = Path(str(entry.get("prompt", ""))).name
                    output_name = Path(str(entry.get("output_json", ""))).name
                    reason = str(entry.get("short_reason") or "").strip()
                    bits = [
                        (
                            f"{entry.get('phase', 'attempt')} "
                            f"attempt {entry.get('attempt_in_iteration')} "
                            f"turn {entry.get('turn_in_attempt')} "
                            f"(step {entry.get('flat_step', entry.get('attempt'))})"
                        )
                    ]
                    if summary_name:
                        bits.append(f"[summary](verifier/{summary_name})")
                    if prompt_name:
                        bits.append(f"[prompt](verifier/{prompt_name})")
                    if output_name:
                        bits.append(f"[output](verifier/{output_name})")
                    if reason:
                        bits.append(f"reason: {reason[:180]}")
                    summary_lines.append("- " + " · ".join(bits))
            else:
                summary_lines.append("- None")
            summary_lines.extend(["", "## Generated skill code"])
            if skill_files:
                summary_lines.extend(
                    f"- `{entry['skill']}` → [{entry['file']}]({entry['file']})"
                    for entry in skill_files
                )
            else:
                summary_lines.append("- None")
            summary_lines.append("")
            (iter_dir / "summary.md").write_text("\n".join(summary_lines))

            return {
                "iteration_dir": str(iter_dir),
                "summary": str(iter_dir / "summary.md"),
                "proposed_task": str(proposed_task_json),
                "policy_files": [str(iter_dir / name) for name in policy_files],
                "generated_skill_files": [
                    str(skill_dir / Path(entry["file"]).name)
                    for entry in skill_files
                ],
            }
        except Exception as e:
            logger.debug(f"  playtime iteration artifact save failed: {e}")
            return {}

    def _save_iteration_trace(
        self, data: dict, proposal: dict, plan: dict,
        success_ctx: str, failure_ctx: str,
    ) -> None:
        """Save a human-readable trace of the full agent pipeline for this iteration."""
        try:
            trace_path = self.output_dir / f"trace_iter{data['iteration']:03d}.md"
            lines = [
                f"# Iteration {data['iteration']}",
                f"**Task:** {proposal.get('activity_name', '?')}",
                f"**Goal:** {proposal.get('goal_conditions', '')}",
                f"**Success:** {data.get('success')} | Attempts: {data.get('total_attempts')}",
                f"**Elapsed:** {data.get('elapsed_seconds', 0):.1f}s",
                f"**Library:** {data.get('skill_library_size')} ({data.get('learned_skill_count')} learned)",
                "",
                "## 1. Task Proposal",
                f"- Reasoning: {proposal.get('reasoning', 'N/A')[:200]}",
                f"- Expected new skills: {proposal.get('expected_new_skills', [])}",
                "",
                "## 2. Plan (initial)",
            ]
            # When the pre-execution verifier rewrote the plan, render
            # the ORIGINAL planner output here (so the "initial" section
            # always reflects what the planner first emitted, even if a
            # refine happened before the policy writer saw the plan).
            initial_plan = (
                data.get("plan_initial_before_verifier")
                or data.get("plan")
                or plan
            )
            for step in initial_plan.get("steps", []):
                skills = step.get("relevant_skills", [])
                lines.append(f"- **{step.get('id', '?')}:** {step.get('description', '')}")
                if skills:
                    lines.append(f"  Skills: {skills}")

            # Pre-execution verifier verdict and (if applicable) the
            # plan rewrite it triggered. Rendered between the initial
            # plan and any diagnoser-driven refines, since this gate
            # fires BEFORE attempt 0 runs.
            initial_verdict = data.get("planner_verifier_initial") or {}
            if initial_verdict.get("enabled"):
                lines += ["", "## 2a. Planner Verifier (pre-execution gate)"]
                lines.append(
                    f"- **Verdict:** {initial_verdict.get('verdict', '?')} "
                    f"(confidence {float(initial_verdict.get('confidence') or 0.0):.2f})"
                )
                err = initial_verdict.get("error")
                if err:
                    lines.append(f"- **Error:** {err}")
                scene_summary = (initial_verdict.get("scene_summary") or "").strip()
                if scene_summary:
                    lines.append(f"- **Scene (what the VLM saw):** {scene_summary}")
                for issue in initial_verdict.get("issues") or []:
                    lines.append(
                        f"  - `{issue.get('kind', '?')}` @ "
                        f"{issue.get('step_id', 'across plan')}: "
                        f"{issue.get('evidence', '')} → {issue.get('suggested_fix', '')}"
                    )
                if initial_verdict.get("should_refine"):
                    lines.append(
                        f"- **Refine reason fed to planner:** "
                        f"{(initial_verdict.get('summary_for_refine_plan') or '').strip()}"
                    )

            refined_by_verifier = data.get("plan_refined_by_verifier") or {}
            if refined_by_verifier.get("steps"):
                lines += ["", "## 2b. Plan (refined by verifier before execution)"]
                for step in refined_by_verifier.get("steps", []):
                    skills = step.get("relevant_skills", [])
                    lines.append(
                        f"- **{step.get('id', '?')}:** {step.get('description', '')}"
                    )
                    if skills:
                        lines.append(f"  Skills: {skills}")
                refined_verdict = data.get("planner_verifier_refined") or {}
                if refined_verdict.get("enabled"):
                    lines.append(
                        f"- **Post-refine re-verify:** "
                        f"{refined_verdict.get('verdict', '?')} "
                        f"(confidence "
                        f"{float(refined_verdict.get('confidence') or 0.0):.2f})"
                    )

            # Plan revisions. When the diagnoser flagged plan_issue on
            # attempt N, feedback_generator set replan=True and lifelong_loop
            # called planner.refine_plan BEFORE attempt N+1 was written.
            # Surface each refined plan here along with the structural
            # reason the diagnoser gave, so a reader can see both the
            # progression of plans and why each one was rewritten.
            refined_attempts = sorted(
                int(k.rsplit("_", 1)[-1])
                for k in data.keys()
                if k.startswith("plan_refined_attempt_")
            )
            for n in refined_attempts:
                refined = data.get(f"plan_refined_attempt_{n}") or {}
                # The diagnosis that TRIGGERED this refine is on the attempt
                # immediately prior (attempt n-1 in 0-indexed terms; since
                # `attempt` was incremented before refine, the diagnosis
                # lives at index n-1 of the 0-indexed diagnosis attempts).
                prev_diag = data.get(f"diagnosis_attempt_{n - 1}") or {}
                reason = prev_diag.get("plan_issue_reason", "") or "(no reason captured)"
                lines += ["", f"## 2.{n}. Plan (refined before attempt {n + 1})"]
                lines.append(f"**Trigger (diagnoser on attempt {n}):** {reason[:400]}")
                lines.append("")
                for step in refined.get("steps", []):
                    skills = step.get("relevant_skills", [])
                    lines.append(
                        f"- **{step.get('id', '?')}:** {step.get('description', '')}"
                    )
                    if skills:
                        lines.append(f"  Skills: {skills}")

            if success_ctx:
                lines += ["", "## 3. Success Context (from past)", success_ctx[:300] + "..."]
            if failure_ctx:
                lines += ["", "## 4. Failure Context (from past)", failure_ctx[:300] + "..."]

            lines.append("")
            lines.append("## 5. Code Attempts")
            for a in range(10):
                code = data.get(f"code_attempt_{a}", "")
                if not code:
                    break
                qual = data.get(f"quality_attempt_{a}", {})
                exec_r = data.get(f"execution_attempt_{a}", {})
                lines.append(f"\n### Attempt {a+1}")
                if isinstance(qual, dict) and not qual.get("approved", True):
                    lines.append(f"**BLOCKED:** {qual.get('feedback', '')[:150]}")
                else:
                    if isinstance(exec_r, dict):
                        lines.append(f"**Result:** reward={exec_r.get('reward')}, completed={exec_r.get('task_completed')}")
                        if exec_r.get("stderr_snippet"):
                            lines.append(f"**Error:** {exec_r['stderr_snippet'][:200]}")
                lines.append(f"```python\n{code}\n```")

            # Sub-agent runs. Spawned mid-iteration when the diagnoser
            # isolates a sub-behavior; each run has its own retry budget
            # and extracts a reusable skill on success. Rendered in plan
            # order by the tag suffix (subagent0 → subagent1 → ...).
            subagent_keys = sorted(k for k in data.keys() if k.startswith("subagent_"))
            if subagent_keys:
                lines += ["", "## 5a. Sub-Agent Runs"]
                for k in subagent_keys:
                    sa = data.get(k) or {}
                    # Tag like "iter001_subagent0" → render just the suffix.
                    tag = k.replace("subagent_", "", 1)
                    ok_badge = "✅ SUCCESS" if sa.get("success") else "❌ exhausted"
                    lines.append(
                        f"\n### {tag} — {ok_badge} "
                        f"(attempts: {sa.get('attempts', '?')})"
                    )
                    subgoal = (sa.get("subgoal") or "").strip()
                    reason = (sa.get("reason") or "").strip()
                    if subgoal:
                        lines.append(f"**Subgoal:** {subgoal}")
                    if reason:
                        lines.append(f"**Why spawned:** {reason}")
                    for h in sa.get("history", []) or []:
                        att_idx = h.get("attempt")
                        att_ok = h.get("success")
                        att_fm = h.get("failure_mode") or ""
                        ev = (h.get("evidence") or "").strip()
                        pf = (h.get("policy_feedback") or "").strip()
                        att_badge = "✓" if att_ok else "✗"
                        lines.append(
                            f"\n<details><summary>attempt {att_idx} — "
                            f"{att_badge} {att_fm}</summary>"
                        )
                        if ev:
                            lines.append(f"\n**Diagnoser observation:** {ev}")
                        if pf and pf != ev:
                            lines.append(f"\n**Diagnoser critique:** {pf}")
                        code = h.get("code") or ""
                        if code:
                            lines.append("\n```python")
                            lines.append(code)
                            lines.append("```")
                        lines.append("\n</details>")

            if data.get("skills_learned"):
                lines += ["", "## 6. Skills Extracted", str(data["skills_learned"])]

            # Video references
            videos = list(self.output_dir.glob(f"iter{data['iteration']:03d}_*.mp4"))
            if videos:
                lines += ["", "## 7. Videos"]
                for v in videos:
                    lines.append(f"- [{v.name}]({v.name})")

            trace_path.write_text("\n".join(lines))
        except Exception as e:
            logger.debug(f"  Trace save failed: {e}")

    def _get_scene_context(self) -> dict[str, Any]:
        """Extract scene context from the environment."""
        # LIBERO environments use a dedicated extractor
        if self.env_type == "libero":
            return extract_libero_scene_context(self.env)
        if self.env_type == "molmospaces":
            return extract_molmospaces_scene_context(self.env)

        # Try to use CaP-X runtime extraction (BEHAVIOR-1K)
        try:
            from rats.rats.runtime import extract_behavior_scene_context
            ctx = extract_behavior_scene_context(self.env)
            return ctx.model_dump()
        except Exception:
            pass

        # Fallback: basic context from env attributes
        low_level = getattr(self.env, "low_level_env", self.env)
        apis = getattr(self.env, "_apis", {})
        available_functions: list[str] = []
        api_docs_parts: list[str] = []
        for api in apis.values():
            if hasattr(api, "functions") and callable(api.functions):
                available_functions.extend(list(api.functions().keys()))
            if hasattr(api, "combined_doc") and callable(api.combined_doc):
                api_docs_parts.append(api.combined_doc())

        # Get the task prompt — try multiple sources
        task_prompt = (
            getattr(self.env, "_task_prompt", "")
            or getattr(self.env, "prompt", "")
            or getattr(type(self.env), "prompt", "")
        )

        return {
            "env_type": "behavior",
            "scene_model": getattr(low_level, "task_name", "unknown_scene"),
            "activity_name": getattr(low_level, "task_name", None),
            # `low_level.task_relevant_obj` is BEHAVIOR-1K's privileged
            # list of scene objects relevant to the current activity. Not
            # something a non-priv vision-only agent has. Return empty
            # to match the LIBERO path; downstream consumers fall back
            # to NL-goal-derived objects.
            "object_scope": {},
            "goal_conditions_nl": task_prompt,
            "available_functions": sorted(set(available_functions)),
            "api_docs": "\n\n".join(api_docs_parts),
        }

    @staticmethod
    def _normalize_goal_for_verifier(goal: str) -> str:
        return re.sub(r"\s+", " ", str(goal or "").strip().lower().replace("_", " "))

    def _remember_active_custom_verifier(self, code: str, goal: str) -> None:
        self._active_custom_verifier_code = code or ""
        self._active_custom_verifier_goal = self._normalize_goal_for_verifier(goal)

    def _active_custom_verifier_for_goal(self, goal: str) -> str:
        if not self._active_custom_verifier_code:
            return ""
        if not self._active_custom_verifier_goal:
            return ""
        if self._active_custom_verifier_goal != self._normalize_goal_for_verifier(goal):
            return ""
        return self._active_custom_verifier_code

    def _run_candidate_selection(
        self,
        *,
        scene_context: dict[str, Any],
        current_activity: str,
        iteration_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate K fresh + up-to-K_retry candidates, score, select max.

        Writes a full audit trail into ``iteration_data["candidate_selection"]``
        so downstream analysis can compare LLM vs formula scoring side by
        side and recover which retry-bank items were sampled.

        Returns the winning candidate dict, which the caller then hands to
        the Environment Creator the same way the single-proposal path does.
        """
        from rats.agents.curiosity_scoring import (
            compute_retry_bonus,
            make_skill_lookup,
            score_candidate,
        )

        skill_context = self.skill_library.get_context_for_task_proposer()
        scene_for_proposer = dict(scene_context, current_activity=current_activity)

        fresh = self.task_proposer.propose_novel_candidates(
            scene_for_proposer,
            skill_context,
            num_candidates=self._num_fresh_candidates,
            request_required_skills=True,
        )
        # `propose_novel_candidates` falls back to a single propose() when
        # all K candidates fail validation. Tag the single result so it
        # still flows through the scorer.
        for c in fresh:
            c.setdefault("candidate_type", "fresh")
            c.setdefault("required_skills", [])

        retry_items: list[dict[str, Any]] = []
        retry_candidates: list[dict[str, Any]] = []
        if self.retry_bank is not None and self._num_retry_candidates > 0:
            retry_items = self.retry_bank.sample(
                self._num_retry_candidates, iteration=self._iteration,
            )
            if retry_items:
                retry_candidates = self.task_proposer.propose_retry_candidates(
                    retry_items,
                    scene_for_proposer,
                    skill_context,
                )

        candidates: list[dict[str, Any]] = list(fresh) + list(retry_candidates)
        if not candidates:
            # Total proposer failure — fall back to the legacy single propose
            # so the iteration doesn't crash. Score it through the same path
            # for consistent logging.
            single = self.task_proposer.propose(scene_for_proposer, skill_context)
            single.setdefault("candidate_type", "fresh")
            single.setdefault("required_skills", [])
            candidates = [single]

        skill_lookup = make_skill_lookup(self.skill_library)
        task_history = self.task_proposer.get_task_history()
        for cand in candidates:
            if cand.get("candidate_type") == "retry_derived":
                ttl = int(cand.get("retry_ttl", 0))
                mttl = max(1, int(cand.get("retry_max_ttl", self._retry_bank_ttl)))
                cand["retry_bonus_score"] = compute_retry_bonus(
                    {
                        "diagnosable": cand.get("retry_diagnosable", False),
                        "surprise_score": cand.get("retry_surprise_score", 0.0),
                        "ttl": ttl,
                    },
                    max_ttl=mttl,
                )
            else:
                cand["retry_bonus_score"] = 0.0
            score_candidate(
                cand,
                mode="formula",
                skill_lookup=skill_lookup,
                history_counts=self._obj_skill_counts,
                task_history=task_history,
                retry_bonus_weight=self._retry_bonus_weight,
                failure_penalty_weight=self._failure_penalty_weight,
                score_composition=self._score_composition,
            )

        # Selection: argmax over final_score (novelty×frontier base + retry
        # bonus − failure penalty). Ties fall back to candidate order, which
        # the proposer emits simplest-first.
        candidates.sort(
            key=lambda c: float(c.get("final_score", 0.0) or 0.0),
            reverse=True,
        )
        for i, c in enumerate(candidates):
            c["selected"] = (i == 0)
        selected = candidates[0]

        # Audit trail — keep light fields only.
        loggable_keys = (
            "language", "candidate_type", "objects", "fixtures",
            "required_skills", "llm_novelty_score", "llm_frontier_score",
            "llm_rationale", "formula_novelty_score", "formula_frontier_score",
            "competence_estimate", "novelty_score", "frontier_score",
            "retry_bonus_score", "failure_penalty", "final_score",
            "score_breakdown", "selected", "source_retry_id",
            "source_failure_reason", "source_failure_category",
            "retry_ttl", "retry_max_ttl",
        )
        iteration_data["candidate_selection"] = {
            "mode": self._candidate_mode,
            "composition": self._score_composition,
            "retry_bonus_weight": self._retry_bonus_weight,
            "failure_penalty_weight": self._failure_penalty_weight,
            "fresh_count": sum(
                1 for c in candidates if c.get("candidate_type") == "fresh"
            ),
            "retry_count": sum(
                1 for c in candidates if c.get("candidate_type") == "retry_derived"
            ),
            "retry_bank_size_before": (
                len(self.retry_bank) if self.retry_bank is not None else 0
            ),
            "retry_items_sampled": [
                {
                    "retry_id": it.get("retry_id"),
                    "language": it.get("language"),
                    "failure_category": it.get("failure_category"),
                    "surprise_score": it.get("surprise_score"),
                    "ttl": it.get("ttl"),
                }
                for it in retry_items
            ],
            "candidates": [
                {k: c.get(k) for k in loggable_keys if k in c}
                for c in candidates
            ],
            "selected": {
                "language": selected.get("language"),
                "candidate_type": selected.get("candidate_type"),
                "final_score": selected.get("final_score"),
            },
        }
        logger.info(
            "  Selected candidate: %s (type=%s, final=%.3f, N=%.2f, F=%.2f%s)",
            selected.get("language", "?")[:80],
            selected.get("candidate_type", "fresh"),
            float(selected.get("final_score", 0.0) or 0.0),
            float(selected.get("novelty_score", 0.0) or 0.0),
            float(selected.get("frontier_score", 0.0) or 0.0),
            (
                f", retry_bonus={selected.get('retry_bonus_score', 0):.2f}"
                if selected.get("candidate_type") == "retry_derived" else ""
            ),
        )
        return selected

    def _build_proposal_from_env(
        self, activity: str, scene: str, scene_context: dict[str, Any], **extra,
    ) -> dict[str, Any]:
        """Build a task proposal dict with the goal from the current env state.

        The goal always comes from the env's task_prompt (set by CaP-X env class),
        making it adaptive to whatever task is currently loaded — not hardcoded.
        """
        if scene_context.get("env_type") == "molmospaces":
            goal = (
                scene_context.get("goal_conditions_nl")
                or scene_context.get("task_prompt")
                or activity.replace("_", " ")
            )
        else:
            goal = (
                scene_context.get("task_prompt")
                or scene_context.get("goal_conditions_nl")
                or activity.replace("_", " ")
            )
        # No goal/goal_predicates keys — those would carry BDDL symbolic
        # checklist into the diagnoser which is privileged info a baseline
        # non-priv agent wouldn't have. The diagnoser now uses plan steps
        # (agent-generated) as its visual-verdict unit.
        proposal = {
            "activity_name": activity,
            "scene_model": scene,
            "activity_definition_id": 0,
            "goal_conditions": goal,
            # `language` carries the human-readable task instruction. Without
            # it, the report renderer falls back to `activity_name` which for
            # the LIBERO BDDL-from-handle path is "novel_task0" — see
            # environment_creator._create_libero_env_from_bddl (suite_name +
            # task_id are hardcoded). Setting language here lets the report
            # show the actual goal whenever the fallback fires (e.g. after a
            # BDDL validation failure where novel env creation excepted).
            "language": goal,
            "expected_new_skills": extra.get("expected_new_skills", []),
            "novelty_score": extra.get("novelty_score", 0.5),
            "reasoning": extra.get("reasoning", ""),
        }
        # MolmoSpaces: pull canonical identity from the live env descriptor so
        # post-switch_house iterations (where the proposer's switch_house
        # pseudo-fields are intentionally dropped during merge) still have
        # language / task_family / scene_family / benchmark / objects /
        # canonical_task_id wired up. Without this, iteration_*.json,
        # record_task_outcome, and the curator history all see Nones for the
        # auto-sampled task that actually ran.
        if scene_context.get("env_type") == "molmospaces":
            descriptor = scene_context.get("task_descriptor") or {}
            molmospaces_fields = {
                "canonical_task_id": descriptor.get("canonical_id") or activity,
                "language": descriptor.get("language") or goal,
                "task_family": descriptor.get("task_family"),
                "scene_family": descriptor.get("scene_family") or scene,
                "benchmark": descriptor.get("benchmark"),
                "variant": descriptor.get("variant"),
                "objects": list(descriptor.get("objects") or []),
            }
            for key, value in molmospaces_fields.items():
                if value is not None and value != "":
                    proposal[key] = value
        custom_verifier_code = (
            extra.get("custom_verifier_code")
            or self._active_custom_verifier_for_goal(goal)
        )
        if custom_verifier_code:
            proposal["custom_verifier_code"] = custom_verifier_code
        return proposal

    def _validate_available_tasks(self) -> None:
        """At startup, probe which scene tasks can actually be rebound.

        Tries configure_behavior_task for each candidate, records which succeed,
        then restores the bootstrap task. This avoids wasting time on doomed
        rebinds during the main loop.
        """
        scene_context = self._get_scene_context()
        current_scene = scene_context.get("scene_model", "unknown")
        scene_tasks = _load_scene_compatible_tasks(current_scene)
        if not scene_tasks:
            self._validated_tasks = [self._bootstrap_activity] if self._bootstrap_activity else []
            return

        logger.info(f"Validating {len(scene_tasks)} scene-compatible tasks...")
        validated = []
        low_level = getattr(self.env, "low_level_env", self.env)

        scene = getattr(getattr(low_level, "env", None), "scene", None)
        for task_name in scene_tasks:
            if task_name == self._bootstrap_activity:
                validated.append(task_name)
                continue
            # Check if template exists (fast, no sim needed)
            tmeta = _load_task_template_metadata(current_scene, task_name)
            if tmeta and tmeta.get("inst_to_name"):
                validated.append(task_name)
                logger.info(f"  {task_name}: OK (template found)")
            else:
                self._failed_rebind_tasks.add(task_name)
                logger.debug(f"  {task_name}: FAILED (no template)")

        # Restore bootstrap task after probing
        self._restore_bootstrap_task()
        self._validated_tasks = validated
        logger.info(
            f"Validated {len(validated)}/{len(scene_tasks)} tasks: {validated}"
        )

    def _rebind_via_env_creator(self, task_proposal: dict[str, Any]) -> str:
        """Rebind using the Environment Creator (works for both BEHAVIOR and LIBERO).

        Returns "success", "same_task", or "failed".
        """
        low_level = getattr(self.env, "low_level_env", self.env)
        current_activity = getattr(low_level, "task_name", None)
        proposed = task_proposal["activity_name"]

        if current_activity == proposed:
            return "same_task"

        try:
            result = self.env_creator.create_from_proposal(task_proposal, old_env=self.env)
            # Both BEHAVIOR and LIBERO may return a new env object
            new_env = result.get("env")
            if new_env is not None and new_env is not self.env:
                self.env = new_env
                self._set_api_output_dir(self.env)
            self._remember_active_custom_verifier(
                result.get("custom_verifier_code", ""),
                (result.get("scene_context") or {}).get("goal_conditions_nl")
                or task_proposal.get("language")
                or task_proposal.get("goal_conditions", ""),
            )
            return "success"
        except Exception as e:
            logger.warning(f"  Env creator failed for {proposed}: {e}")
            # Don't try to restore bootstrap — env may be destroyed
            return "failed"

    def _rebind_env_to_task(self, task_proposal: dict[str, Any]) -> str:
        """Rebind environment to a different BEHAVIOR activity.

        Strategy: keep OmniGibson's internal task fixed (too many failure modes
        with configure_behavior_task), but change the task name and prompt that
        the LLM sees. Since all rooms are loaded, all scene objects exist and
        the LLM can manipulate anything — we just redirect its goal.

        Returns:
            "success" - rebound to a new task
            "same_task" - proposed task is already loaded (proceed with it)
            "failed" - rebind failed (BDDL objects missing, etc.)
        """
        low_level = getattr(self.env, "low_level_env", self.env)
        current_activity = getattr(low_level, "task_name", None)
        proposed_activity = task_proposal["activity_name"]
        proposed_def_id = task_proposal.get("activity_definition_id", 0)

        if current_activity == proposed_activity:
            return "same_task"

        # Lightweight rebind: change the task name and prompt without touching
        # OmniGibson's internal task system (which has obs space, wrapped_obj,
        # and TRO file issues). All rooms are loaded so the LLM can manipulate
        # any object — we just change its goal.
        try:
            low_level.task_name = proposed_activity
            # Update task prompt so planner/policy writer see the new task
            new_prompt = f"Complete the task: {proposed_activity.replace('_', ' ')}"
            self.env._task_prompt = new_prompt
            if hasattr(self.env, "prompt"):
                self.env.prompt = new_prompt
            logger.info(f"  Rebound environment to: {proposed_activity}")
            return "success"
        except Exception as e:
            logger.warning(f"  Could not rebind environment: {e}")
            return "failed"

    # ---- LIBERO task switching (env recreation) ----

    def _init_molmospaces_validated_tasks(self) -> None:
        """Populate validated tasks from the bridge-reported MolmoSpaces catalog."""
        scene_ctx = self._get_scene_context()
        descriptor = scene_ctx.get("task_descriptor", {})
        low_level = getattr(self.env, "low_level_env", self.env)
        tasks = low_level.list_task_descriptors()
        self._validated_tasks = [task["canonical_id"] for task in tasks]
        self._bootstrap_activity = descriptor.get("canonical_id", self._bootstrap_activity)
        if str(self._molmospaces_cfg.get("proposer_mode", "catalog")) == "open":
            logger.info(
                "MolmoSpaces open exploration: %d live task descriptor(s) "
                "available for fallback in benchmark %s",
                len(self._validated_tasks),
                descriptor.get("benchmark", "unknown"),
            )
        else:
            logger.info(
                "MolmoSpaces exploration: %d tasks in benchmark %s",
                len(self._validated_tasks),
                descriptor.get("benchmark", "unknown"),
            )

    def _rebind_molmospaces_env(self, task_proposal: dict[str, Any]) -> str:
        """Switch MolmoSpaces tasks, preferring in-place bridge rebinding.

        Three paths in order of preference:
          1. Open-mode proposer asked for a house switch -> call
             ``request_new_house`` on the bridge.
          2. Open-mode proposer chose a typed task spec -> call
             ``set_task_from_spec`` so the bridge instantiates exactly
             the LLM's pick.
          3. Catalog mode -> the legacy ``set_task(canonical_id)`` path
             that selects from the benchmark catalog.
        """
        proposed_activity = task_proposal["activity_name"]
        scene_ctx = self._get_scene_context()
        current_activity = scene_ctx.get("activity_name")
        low_level = getattr(self.env, "low_level_env", self.env)

        # Playtime mode: arbitrary sensorimotor tasks are prompt-only
        # overlays, not bridge-backed MolmoSpaces reward tasks.
        if task_proposal.get("_molmospaces_proposer_mode") == "playtime":
            return self._rebind_molmospaces_playtime(task_proposal)

        # Open-mode: house switch
        if task_proposal.get("_request_house_switch"):
            try:
                from rats.loop.molmospaces_utils import request_molmospaces_new_house
                requested_task_type = task_proposal.get("_molmospaces_requested_task_type")
                request_molmospaces_new_house(
                    self.env,
                    task_type=str(requested_task_type) if requested_task_type else None,
                )
                if not self._reset_env(recover_on_failure=False):
                    logger.warning(
                        "  MolmoSpaces house-switch reset failed; "
                        "not falling back to a stale task"
                    )
                    return "failed"
                self._set_api_output_dir(self.env)
                self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)
                self._queue_molmospaces_environment_check("proposal_house_switch")
                return "success"
            except Exception as e:
                logger.warning(
                    "  MolmoSpaces house-switch request failed: %s; "
                    "not falling back to a stale task",
                    e,
                )
                return "failed"

        # Open-mode: typed task spec
        spec = task_proposal.get("_molmospaces_open_spec")
        if spec:
            task_type = str(spec.get("task_type", "pick"))
            try:
                from rats.loop.molmospaces_utils import apply_molmospaces_task_spec
                logger.info("  Applying MolmoSpaces open task spec via set_task_from_spec")
                apply_molmospaces_task_spec(
                    self.env,
                    task_type=task_type,
                    target_internal_name=spec.get("target_internal_name"),
                    place_receptacle_internal_name=spec.get(
                        "place_receptacle_internal_name"
                    ),
                    joint_internal_name=spec.get("joint_internal_name"),
                    joint_index=spec.get("joint_index"),
                )
            except Exception as e:
                logger.warning(
                    "  MolmoSpaces set_task_from_spec failed for %s: %s; "
                    "trying a compatible-house fallback instead of a stale task",
                    proposed_activity,
                    e,
                )
                return self._fallback_molmospaces_open_task_type(task_type)

            if self._reset_env(recover_on_failure=False):
                self._set_api_output_dir(self.env)
                self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)
                return "success"

            logger.warning(
                "  MolmoSpaces reset failed after applying open task spec for %s; "
                "trying a compatible-house fallback instead of a stale task",
                proposed_activity,
            )
            return self._fallback_molmospaces_open_task_type(task_type)

        if proposed_activity == current_activity:
            return "same_task"

        set_task = getattr(low_level, "set_task", None)
        if callable(set_task):
            try:
                set_task(proposed_activity)
                if not self._reset_env():
                    return "failed"
                self._set_api_output_dir(self.env)
                self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)
                return "success"
            except Exception as e:
                logger.warning(
                    "  MolmoSpaces in-place task switch failed for %s: %s; falling back to env recreation",
                    proposed_activity,
                    e,
                )

        try:
            self._close_molmospaces_env_socket_for_recovery()
            new_env = recreate_molmospaces_env(self.env, proposed_activity)
            self.env = new_env
            self._set_api_output_dir(self.env)
            self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)
            self._queue_molmospaces_environment_check("env_recreated_for_task_switch")
            return "success"
        except Exception as e:
            logger.warning(f"  MolmoSpaces env recreation failed for {proposed_activity}: {e}")
            return "failed"

    def _init_playtime_scene_pool(self) -> list[int] | None:
        """Build a shuffled list of episode indices from the playtime
        benchmark dir.

        Reads ``benchmark.json`` once and shuffles the episode order
        deterministically with ``playtime.scene_pool_seed`` (default 42).
        We track ``episode_index`` (not ``house_index``) so the bridge's
        ``resample_task`` can switch ``scene_dataset`` per episode — that
        path goes through ``_build_benchmark_config`` which reads the
        episode's own ``scene_dataset`` and ``data_split`` instead of
        sticking with whatever the bridge was bootstrapped with. This is
        what makes a mixed ithor + procthor-objaverse playtime set
        actually interleave instead of getting stuck on one dataset.

        Returns the pool of episode indices, or ``None`` if the benchmark
        dir / json can't be found. Cached on ``self._playtime_scene_pool``.
        """
        if self._playtime_scene_pool is not None:
            return self._playtime_scene_pool
        # Find benchmark dir off the active bridge.
        # ``FrankaMolmoSpacesEnv`` exposes the configured benchmark dir as
        # ``low.benchmark_dir`` directly. The remote bridge wrapper keeps a
        # mirrored copy on ``self._init_kwargs['benchmark_dir']`` after the
        # init RPC. Try both because catalogs sometimes wrap the env in an
        # extra adapter that hides the attribute.
        bench_dir: str | None = None
        try:
            low = getattr(self.env, "low_level_env", self.env)
            bench_dir = getattr(low, "benchmark_dir", None)
            if not bench_dir:
                real_bridge = getattr(low, "_real_bridge", None)
                if real_bridge is not None:
                    bench_dir = getattr(real_bridge, "benchmark_dir", None)
                    if not bench_dir:
                        init_kwargs = getattr(real_bridge, "_init_kwargs", None) or {}
                        bench_dir = init_kwargs.get("benchmark_dir")
        except Exception:
            bench_dir = None
        if not bench_dir:
            logger.warning(
                "  Playtime scene-first: bridge has no benchmark_dir; "
                "scene-first mode disabled"
            )
            return None
        from pathlib import Path
        import json as _json
        bench_path = Path(bench_dir) / "benchmark.json"
        if not bench_path.exists():
            logger.warning(
                "  Playtime scene-first: benchmark.json not found at %s",
                bench_path,
            )
            return None
        try:
            episodes = _json.loads(bench_path.read_text())
        except Exception as exc:
            logger.warning(
                "  Playtime scene-first: failed to parse %s: %s",
                bench_path, exc,
            )
            return None
        if not episodes:
            logger.warning(
                "  Playtime scene-first: no episodes in %s", bench_path,
            )
            return None
        playtime_cfg = self._molmospaces_cfg.get("playtime") or {}
        seed = int(playtime_cfg.get("scene_pool_seed", 42))
        import random as _random
        rng = _random.Random(seed)
        # Pool of episode indices (0..N-1) shuffled deterministically.
        # Using episode_index keeps each (scene_dataset, house_index)
        # pairing intact in the bridge's resample_task path.
        ordered = list(range(len(episodes)))
        rng.shuffle(ordered)
        self._playtime_scene_pool = ordered
        # Build a sidecar of (idx, dataset, house) for human-readable
        # logging.
        first_10 = [
            (
                ordered[i],
                episodes[ordered[i]].get("scene_dataset"),
                episodes[ordered[i]].get("house_index"),
            )
            for i in range(min(10, len(ordered)))
        ]
        logger.info(
            "  Playtime scene-first: built episode pool with %d eps "
            "(seed=%d). First 10 (ep_idx, dataset, house): %s",
            len(ordered), seed, first_10,
        )
        return ordered

    def _pick_next_playtime_scene(self) -> dict[str, Any] | None:
        """Deterministically force the next house from the shuffled scene
        pool.

        Triggered every iter when
        ``molmospaces.playtime.scene_first.enabled`` is true. Picks the
        next ``house_index`` from the shuffled pool (round-robin), asks
        the bridge for that specific house, and resets the env so the
        proposer + executor see the same scene.

        Returns a descriptor dict for the iteration record on success,
        ``None`` if disabled / failed (in which case the iter falls back
        to whatever scene the bridge was already on).
        """
        playtime_cfg = self._molmospaces_cfg.get("playtime") or {}
        scene_first = playtime_cfg.get("scene_first") or {}
        if not bool(scene_first.get("enabled", False)):
            return None

        pool = self._init_playtime_scene_pool()
        if not pool:
            return None

        iter_idx = int(self._iteration)
        offset = int(scene_first.get("start_offset", 0))
        idx_in_pool = (iter_idx + offset) % len(pool)
        target_episode = pool[idx_in_pool]

        prev_house: Any = None
        try:
            low_level = getattr(self.env, "low_level_env", self.env)
            prev_house = getattr(low_level, "house_index", None)
        except Exception:
            prev_house = None

        # Use resample_task(episode_index=...) instead of
        # request_new_house(house_index=...). The episode_index path
        # goes through _build_benchmark_task which respects each
        # episode's scene_dataset (so ithor and procthor-objaverse
        # episodes can interleave inside one bridge); request_new_house
        # only retunes house_index against the bridge's bootstrap
        # dataset, which is what was wedging scene-first when the pool
        # mixed two datasets.
        try:
            low_level = getattr(self.env, "low_level_env", self.env)
            real_bridge = getattr(low_level, "_real_bridge", None)
            resample_fn = getattr(real_bridge or low_level, "resample_task", None)
            if not callable(resample_fn):
                logger.warning(
                    "  Scene-first: bridge has no resample_task RPC; "
                    "skipping switch (iter %d)",
                    iter_idx,
                )
                return None
            resample_fn(episode_index=int(target_episode))
        except Exception as e:
            logger.warning(
                "  Scene-first resample_task RPC failed at iter %d "
                "(episode_index=%s, %s: %s); staying on current house",
                iter_idx, target_episode, type(e).__name__, e,
            )
            return None

        if not self._reset_env(recover_on_failure=True):
            logger.warning(
                "  Scene-first house switch at iter %d: reset_env returned "
                "False; iter will run against whatever state the env is in",
                iter_idx,
            )
            return None

        try:
            self._set_api_output_dir(self.env)
        except Exception:
            pass
        self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)

        try:
            self.task_proposer._molmospaces_grounding_cache.clear()
        except Exception:
            pass

        new_house: Any = None
        try:
            low_level = getattr(self.env, "low_level_env", self.env)
            new_house = getattr(low_level, "house_index", None)
        except Exception:
            pass

        logger.info(
            "  Scene-first iter %d: pool[%d] -> episode_index=%s, "
            "house_index %s -> %s",
            iter_idx, idx_in_pool, target_episode, prev_house, new_house,
        )
        return {
            "iteration": iter_idx,
            "scene_first": True,
            "pool_index": idx_in_pool,
            "episode_index": target_episode,
            "house_index": new_house,
            "prev_house_index": prev_house,
        }

    def _maybe_force_periodic_house_switch(self) -> dict[str, Any] | None:
        """Force-rotate to a new MolmoSpaces house every N iterations.

        Triggers when ``molmospaces.house_switch_every`` (in YAML) is a
        positive integer and the current iteration index is a non-zero
        multiple of it. Runs BEFORE the proposer reads the env, so the
        proposer sees the new house's inventory and the executor will
        run on the same house — no proposer↔executor mismatch.

        Idempotent on failure: if the switch RPC raises or the reset
        fails, the loop stays on the current house and the iteration
        proceeds normally; the warning is logged so post-mortem can
        find it.

        Returns a small descriptor dict for the iteration record on
        successful switches, ``None`` otherwise.

        Note: when ``playtime.scene_first.enabled`` is true,
        ``_pick_next_playtime_scene`` runs first and this periodic-switch
        path is skipped (the scene pool is authoritative).
        """
        every = int(self._molmospaces_cfg.get("house_switch_every") or 0)
        if every <= 0:
            return None
        iter_idx = int(self._iteration)
        # Don't switch at iter 0 (the bridge was just initialized).
        if iter_idx <= 0 or iter_idx % every != 0:
            return None

        try:
            from rats.loop.molmospaces_utils import request_molmospaces_new_house
        except Exception as e:
            logger.warning(
                "  Periodic house switch unavailable (import failed: %s); skipping",
                e,
            )
            return None

        prev_house: Any = None
        try:
            low_level = getattr(self.env, "low_level_env", self.env)
            prev_house = getattr(low_level, "house_index", None)
        except Exception:
            prev_house = None

        try:
            request_molmospaces_new_house(self.env)
        except Exception as e:
            logger.warning(
                "  Periodic house switch RPC failed at iter %d (%s: %s); "
                "staying on current house",
                iter_idx, type(e).__name__, e,
            )
            return None

        # The reset is what actually loads the new scene; without it,
        # subsequent calls would still observe the old house's frame.
        # Use the recover_on_failure path so a transient bridge hiccup
        # gets one retry instead of leaving the loop wedged.
        if not self._reset_env(recover_on_failure=True):
            logger.warning(
                "  Periodic house switch at iter %d: reset_env returned False; "
                "the iteration will run against whatever state the env is in",
                iter_idx,
            )
            return None

        try:
            self._set_api_output_dir(self.env)
        except Exception as e:
            logger.debug(f"  _set_api_output_dir after periodic switch failed: {e}")
        self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)

        # Drop the per-house grounding cache so the new house gets a
        # fresh display_names / visibility pass. Otherwise the proposer
        # would see stale (house, dataset) → grounded mappings if the
        # cache key happened to collide.
        try:
            self.task_proposer._molmospaces_grounding_cache.clear()
        except Exception:
            pass

        new_house: Any = None
        try:
            low_level = getattr(self.env, "low_level_env", self.env)
            new_house = getattr(low_level, "house_index", None)
        except Exception:
            pass

        logger.info(
            "  Periodic house switch at iter %d: house_index %s -> %s "
            "(every=%d)",
            iter_idx, prev_house, new_house, every,
        )
        return {
            "iteration": iter_idx,
            "every": every,
            "prev_house_index": prev_house,
            "new_house_index": new_house,
        }

    def _fallback_molmospaces_open_task_type(self, task_type: str) -> str:
        """Recover an invalid open-mode concrete spec by sampling a valid task type.

        Open-mode proposals can pick object/receptacle combinations that are
        not physically placeable in the current house. Falling back to the
        catalog path can silently continue on the stale previous task, so use
        the bridge's house/task-type resampling path instead and only report
        success if the new task actually resets.
        """
        allow_switching = bool(self._molmospaces_cfg.get("allow_house_switching", True))
        if not allow_switching:
            logger.warning("  MolmoSpaces compatible-house fallback disabled by config")
            return "failed"
        try:
            from rats.loop.molmospaces_utils import request_molmospaces_new_house

            request_molmospaces_new_house(self.env, task_type=task_type)
            if not self._reset_env(recover_on_failure=False):
                logger.warning(
                    "  MolmoSpaces compatible-house fallback reset failed for task_type=%s",
                    task_type,
                )
                return "failed"
            self._set_api_output_dir(self.env)
            self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)
            self._queue_molmospaces_environment_check("compatible_house_fallback")
            logger.info(
                "  Rebound MolmoSpaces open task via compatible-house fallback "
                "(task_type=%s)",
                task_type,
            )
            return "success"
        except Exception as e:
            logger.warning(
                "  MolmoSpaces compatible-house fallback failed for task_type=%s: %s",
                task_type,
                e,
            )
            return "failed"

    def _rebind_molmospaces_playtime(self, task_proposal: dict[str, Any]) -> str:
        """Install a prompt-only playtime descriptor without set_task_from_spec."""
        if task_proposal.get("_request_house_switch"):
            try:
                from rats.loop.molmospaces_utils import request_molmospaces_new_house
                request_molmospaces_new_house(self.env)
                self._reset_env()
                self._set_api_output_dir(self.env)
                self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)
                self._queue_molmospaces_environment_check("playtime_house_switch")
                return "success"
            except Exception as e:
                logger.warning("  MolmoSpaces playtime house switch failed: %s", e)
                return "failed"

        descriptor = {
            "canonical_id": task_proposal.get("canonical_task_id")
            or task_proposal.get("activity_name"),
            "language": task_proposal.get("language")
            or task_proposal.get("goal_conditions", ""),
            "task_family": task_proposal.get("task_family")
            or f"playtime_{(task_proposal.get('_playtime') or {}).get('interaction_type', 'touch')}",
            "scene_family": task_proposal.get("scene_family"),
            "benchmark": task_proposal.get("benchmark"),
            "variant": "playtime",
            "objects": task_proposal.get("objects", []),
            "metadata": {
                "task_type": "playtime",
                "playtime": task_proposal.get("_playtime") or {},
            },
        }
        # If proposer picked a target object different from the bridge's
        # currently-anchored one, ask the bridge to RE-ANCHOR (re-place
        # the robot near the new target). Before this, the rebind was
        # prompt-only: bridge stayed pointed at its round-robin pick
        # while the policy was told to interact with a different
        # object, causing systematic perception/IK failures whenever
        # the proposer's curiosity argmax disagreed with bridge's
        # round-robin. Now (target × verb) is the joint curiosity unit
        # and the bridge physically follows.
        low_level = getattr(self.env, "low_level_env", self.env)
        anchor_fn = getattr(low_level, "anchor_to_pickup", None)
        get_anchor_fn = getattr(low_level, "get_anchored_task_target", None)
        target_name = (task_proposal.get("_playtime") or {}).get("target_internal_name")
        if callable(anchor_fn) and callable(get_anchor_fn) and target_name:
            try:
                current_anchor = (get_anchor_fn() or {}).get("pickup_obj_name")
            except Exception:
                current_anchor = None
            if current_anchor and current_anchor != target_name:
                try:
                    anchor_fn(target_name)
                    logger.info(
                        "  Bridge anchor switched: %s -> %s",
                        current_anchor, target_name,
                    )
                except Exception as exc:
                    logger.warning(
                        "  anchor_to_pickup(%s) failed: %s; staying on %s",
                        target_name, exc, current_anchor,
                    )
        low_level = getattr(self.env, "low_level_env", self.env)
        base_canonical_task_id = getattr(low_level, "canonical_task_id", None)
        existing_playtime = getattr(low_level, "_playtime_task_descriptor", None)
        if isinstance(existing_playtime, dict):
            base_canonical_task_id = (
                (existing_playtime.get("metadata") or {}).get("base_canonical_task_id")
                or base_canonical_task_id
            )
        if base_canonical_task_id:
            descriptor.setdefault("metadata", {})["base_canonical_task_id"] = str(
                base_canonical_task_id
            )
        setter = getattr(low_level, "set_playtime_task_descriptor", None)
        try:
            if callable(setter):
                setter(descriptor)
            else:
                setattr(low_level, "_playtime_task_descriptor", descriptor)
            prompt = str(descriptor.get("language") or "")
            if hasattr(self.env, "_compose_live_task_prompt"):
                prompt = self.env._compose_live_task_prompt(prompt)
            if hasattr(self.env, "_task_prompt"):
                self.env._task_prompt = prompt
            if hasattr(self.env, "prompt"):
                self.env.prompt = prompt
            self._reset_env()
            self._set_api_output_dir(self.env)
            self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)
            logger.info("  Installed MolmoSpaces playtime prompt overlay")
            return "success"
        except Exception as e:
            # Previously this swallowed any exception (including reset
            # crashes) into a one-line warning, leaving the run pointed
            # at whatever stale state the env was in. Log the type and
            # traceback so post-mortem can find the root cause without
            # re-running.
            logger.warning(
                "  MolmoSpaces playtime prompt overlay failed (%s: %s)",
                type(e).__name__, e, exc_info=True,
            )
            return "failed"

    def _retry_molmospaces_playtime_after_env_rejection(
        self,
        *,
        scene_context: dict[str, Any],
        skill_context: dict[str, Any],
        rejected_result: dict[str, Any],
        iteration_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Re-prompt playtime proposal when Molmo cannot see the target.

        The retry budget is the verifier config's max_retries. The final retry
        bypasses the LLM and asks TaskProposer for its deterministic fallback,
        which is benchmark-target-first when a benchmark descriptor exists.
        """
        cfg = self._molmospaces_cfg.get("environment_verifier") or {}
        max_retries = max(1, int(cfg.get("max_retries", 4) or 4))
        previous_result = rejected_result
        for retry_idx in range(1, max_retries + 1):
            force_fallback = retry_idx >= max_retries
            feedback = self._environment_verifier_feedback(previous_result)
            try:
                current_activity = scene_context.get("activity_name") or "unknown_task"
                scene_context_for_proposer = dict(scene_context)
                scene_context_for_proposer["current_activity"] = current_activity
                scene_context_for_proposer["_env"] = self.env
                scene_context_for_proposer["environment_verifier_feedback"] = feedback
                if force_fallback:
                    scene_context_for_proposer["_force_playtime_deterministic_fallback"] = True

                logger.warning(
                    "  Re-prompting MolmoSpaces playtime proposer after "
                    "environment-verifier rejection (%d/%d, fallback=%s)",
                    retry_idx,
                    max_retries,
                    force_fallback,
                )
                proposal = self.task_proposer.propose(
                    scene_context_for_proposer,
                    skill_context,
                )
                artifact_paths = self._save_molmospaces_playtime_proposal_artifacts(
                    proposal,
                    trace=getattr(self.task_proposer, "last_proposal_trace", None),
                )
                if artifact_paths:
                    existing = dict(iteration_data.get("task_proposal_artifacts") or {})
                    existing[f"environment_retry_{retry_idx}"] = artifact_paths
                    iteration_data["task_proposal_artifacts"] = existing

                rebind_status = self._rebind_molmospaces_env(proposal)
                if rebind_status not in ("success", "same_task"):
                    previous_result = {
                        "suitable": False,
                        "task": {
                            "activity_name": proposal.get("activity_name"),
                            "language": proposal.get("language"),
                        },
                        "queries": [],
                        "reason": f"rebind_failed:{rebind_status}",
                    }
                    continue

                refreshed_context = self._get_scene_context()
                current_activity = refreshed_context.get("activity_name") or current_activity
                current_scene = refreshed_context.get("scene_model") or scene_context.get(
                    "scene_model", "unknown"
                )
                proposer_fields = dict(proposal)
                rebuilt = self._build_proposal_from_env(
                    current_activity,
                    current_scene,
                    refreshed_context,
                    reasoning=proposal.get("reasoning", ""),
                    expected_new_skills=proposal.get("expected_new_skills", []),
                    novelty_score=proposal.get("novelty_score", 0.5),
                )
                merge_keys = (
                    "canonical_task_id", "language", "task_family", "objects",
                    "scene_family", "benchmark", "mode", "difficulty_estimate",
                    "curiosity_score", "_molmospaces_proposer_mode", "_playtime",
                )
                for key in merge_keys:
                    if key in proposer_fields and key not in rebuilt:
                        rebuilt[key] = proposer_fields[key]

                check_result = self._verify_molmospaces_environment_if_needed(
                    rebuilt,
                    refreshed_context,
                    reasons=[f"playtime_reproposal_after_env_rejection_{retry_idx}"],
                    attempt=retry_idx,
                )
                if check_result is not None:
                    iteration_data.setdefault("environment_verifier", []).append(check_result)
                if check_result is None or bool(check_result.get("suitable", True)):
                    return {
                        "scene_context": refreshed_context,
                        "task_proposal": rebuilt,
                    }
                previous_result = check_result
            except Exception as exc:
                logger.warning(
                    "  MolmoSpaces playtime env-verifier retry %d failed (%s: %s)",
                    retry_idx,
                    type(exc).__name__,
                    exc,
                )
                previous_result = {
                    "suitable": False,
                    "task": {},
                    "queries": [],
                    "reason": f"retry_exception:{type(exc).__name__}",
                }

        self._last_reset_failed = True
        self._last_reset_error = (
            "environment verifier rejected playtime proposal after retry budget"
        )
        return None

    def _init_libero_validated_tasks(self) -> None:
        """Populate validated tasks from the current LIBERO suite.

        All tasks in the same suite are valid since we recreate the env.
        """
        scene_ctx = self._get_scene_context()
        suite_name = scene_ctx.get("suite_name", "libero_spatial")
        from rats.loop.libero_utils import LIBERO_SUITE_SIZES

        n_tasks = LIBERO_SUITE_SIZES.get(suite_name, 10)
        self._validated_tasks = [
            f"{suite_name}_task{tid}" for tid in range(n_tasks)
        ]
        # Give the catalog proposer the public natural-language instruction,
        # not just opaque labels such as ``libero_90_task37``.  This is the
        # information every deployed policy receives and is required for
        # meaningful curiosity/curriculum selection; no predicates, poses, or
        # simulator state are exposed.
        try:
            from libero import benchmark

            suite = benchmark.get_benchmark_dict(help=False)[suite_name]()
            self._available_tasks_full = [
                {
                    "activity_name": f"{suite_name}_task{task_id}",
                    "scene_model": suite_name,
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "activity_definition_id": task_id,
                    "goal_conditions": str(suite.get_task(task_id).language),
                    "language": str(suite.get_task(task_id).language),
                }
                for task_id in range(n_tasks)
            ]
        except Exception as exc:
            logger.warning("Could not attach LIBERO task language to catalog: %s", exc)
            self._available_tasks_full = [
                {
                    "activity_name": name,
                    "scene_model": suite_name,
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "activity_definition_id": task_id,
                    "goal_conditions": name.replace("_", " "),
                }
                for task_id, name in enumerate(self._validated_tasks)
            ]
        self._libero_suite = suite_name
        logger.info(
            f"LIBERO exploration: {len(self._validated_tasks)} tasks in suite {suite_name}"
        )

    def _rebind_libero_env(self, task_proposal: dict[str, Any]) -> str:
        """Switch LIBERO to a different task by recreating the environment.

        Returns "success", "same_task", or "failed".
        """
        proposed_activity = task_proposal["activity_name"]
        scene_ctx = self._get_scene_context()
        current_activity = scene_ctx.get("activity_name")

        if proposed_activity == current_activity:
            return "same_task"

        suite_name, task_id = parse_libero_activity_name(proposed_activity)
        try:
            new_env = recreate_libero_env(self.env, suite_name, task_id)
            # Swap env reference everywhere
            self.env = new_env
            self._set_api_output_dir(self.env)
            self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)
            self._remember_active_custom_verifier("", "")
            return "success"
        except Exception as e:
            logger.warning(f"  LIBERO env recreation failed for {proposed_activity}: {e}")
            return "failed"

    def _restore_bootstrap_task(self) -> None:
        """Restore environment to the bootstrap task label."""
        if not self._bootstrap_activity:
            return
        try:
            low_level = getattr(self.env, "low_level_env", self.env)
            low_level.task_name = self._bootstrap_activity
            prompt = f"Complete the task: {self._bootstrap_activity.replace('_', ' ')}"
            self.env._task_prompt = prompt
            if hasattr(self.env, "prompt"):
                self.env.prompt = prompt
            logger.info(f"  Restored env to bootstrap task: {self._bootstrap_activity}")
        except Exception as e2:
            logger.warning(f"  Could not restore bootstrap task: {e2}")

    def _dispatch_parallel_subagents(
        self,
        *,
        bddl_path: str,
        subgoal: str,
        approaches: list[str],
        scene_context: dict[str, Any],
        parent_task_name: str,
        video_tag_prefix: str,
        poll_interval: float = 2.0,
        terminate_grace: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Spawn one ``run_subagent_worker.py`` subprocess per approach.

        Each subprocess builds its own LIBERO env from ``bddl_path`` and
        runs ``SubAgent.run`` on ``(subgoal, approach_directive)``. We
        poll each subprocess's output JSON file; the first one to
        report ``status: done`` AND ``success: true`` wins. The rest
        are SIGTERM-then-SIGKILLed so we don't burn vision-server
        budget on already-superseded work.

        Returns a list aligned with ``approaches``; each entry is the
        worker's final output JSON (whether success, cancelled, or
        error).

        Why subprocesses instead of threads: MuJoCo envs aren't safe to
        share across threads in the same process, and we need each
        worker to truly reset and own its env independently. Vision
        servers (SAM3 / GraspNet / Molmo / pyroki) are HTTP and
        already serialize internally via Semaphore(1), so concurrent
        workers don't corrupt them — they just queue.
        """
        import json as _json
        import signal as _signal
        import subprocess as _sp
        import time as _time

        # Persist scene_context to disk so workers can load it. Drop
        # known non-JSON-serializable fields. We keep the
        # available_functions list, api_docs, goal_conditions_nl, etc.
        scene_for_worker: dict[str, Any] = {}
        for k, v in (scene_context or {}).items():
            try:
                _json.dumps(v, default=str)
                scene_for_worker[k] = v
            except (TypeError, ValueError):
                # Skip whatever can't survive a round-trip.
                continue
        scene_dump = self.output_dir / f"{video_tag_prefix}_scene_context.json"
        scene_dump.write_text(_json.dumps(scene_for_worker, default=str))

        # Read the API set + privileged flag off the active env so each
        # worker rebuilds an env with the same surface area.
        api_names = list(getattr(self.env, "_apis", {}).keys())
        if not api_names:
            api_names = ["FrankaLiberoApiReducedSkillLibrary"]
        cfg_obj = getattr(self.env, "cfg", None)
        privileged = bool(getattr(cfg_obj, "privileged", False)) if cfg_obj else False

        skill_lib_path = getattr(self.skill_library, "_storage_path", None)
        if skill_lib_path is None:
            raise RuntimeError(
                "skill_library has no _storage_path; cannot dispatch "
                "parallel sub-agents (each worker needs to load it)."
            )

        worker_script = str(
            Path(__file__).resolve().parent.parent.parent
            / "scripts" / "run_subagent_worker.py"
        )
        procs: list[_sp.Popen] = []
        out_paths: list[Path] = []
        for i, approach in enumerate(approaches):
            out_path = self.output_dir / f"{video_tag_prefix}_w{i}.json"
            video_tag = f"{video_tag_prefix}_w{i}"
            cmd = [
                sys.executable,
                worker_script,
                "--bddl-path", str(bddl_path),
                "--apis", ",".join(api_names),
                "--subgoal", subgoal,
                "--approach", approach,
                "--scene-context-json", str(scene_dump),
                "--skill-library", str(skill_lib_path),
                "--max-retries", str(self.sub_agent.max_retries),
                "--video-dir", str(self.output_dir),
                "--video-tag", video_tag,
                "--output-json", str(out_path),
                "--parent-task-name", parent_task_name or "",
                "--execution-timeout", str(self.executor.timeout_seconds),
                # Pin the worker's env to the same seeded placement
                # the parent's _reset_env uses. Otherwise the worker
                # practices on a different scene and its winning script
                # is moot for the parent's main attempts.
                "--iteration-seed", str(self._iteration),
            ]
            if privileged:
                cmd.append("--privileged")
            logger.info(
                f"  [parallel-subagent] spawn worker {i} "
                f"approach={approach[:80]!r}"
            )
            procs.append(_sp.Popen(cmd))
            out_paths.append(out_path)

        # Poll until any worker reports success or all exit.
        winner_idx: int | None = None
        while True:
            all_done = True
            for i, (proc, out_path) in enumerate(zip(procs, out_paths)):
                if proc.poll() is None:
                    all_done = False
                if out_path.exists():
                    try:
                        data = _json.loads(out_path.read_text())
                    except (_json.JSONDecodeError, OSError):
                        # Concurrent writer / partial file — try later.
                        continue
                    if data.get("status") == "done" and data.get("success"):
                        winner_idx = i
                        break
            if winner_idx is not None or all_done:
                break
            _time.sleep(poll_interval)

        if winner_idx is not None:
            logger.info(
                f"  [parallel-subagent] worker {winner_idx} won; "
                f"terminating {sum(1 for p in procs if p.poll() is None)} sibling(s)"
            )
            for i, proc in enumerate(procs):
                if i != winner_idx and proc.poll() is None:
                    try:
                        proc.send_signal(_signal.SIGTERM)
                    except OSError:
                        pass
            deadline = _time.time() + terminate_grace
            for proc in procs:
                while proc.poll() is None and _time.time() < deadline:
                    _time.sleep(0.2)
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except OSError:
                        pass

        # Collect final outputs.
        results: list[dict[str, Any]] = []
        for i, (proc, out_path) in enumerate(zip(procs, out_paths)):
            try:
                proc.wait(timeout=10)
            except _sp.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            entry: dict[str, Any]
            if out_path.exists():
                try:
                    entry = _json.loads(out_path.read_text())
                except _json.JSONDecodeError:
                    entry = {
                        "status": "corrupt",
                        "approach": approaches[i],
                        "success": False,
                    }
            else:
                entry = {
                    "status": "missing",
                    "approach": approaches[i],
                    "success": False,
                }
            results.append(entry)
        return results

    def _append_learned_skill_timeline_event(
        self,
        called: list[str],
        *,
        start_frame: int | None,
        end_frame: int | None,
        attempt: int | None,
        turn: int | None,
        block_index: int | None = None,
    ) -> dict[str, Any]:
        """Append a video timeline event for learned/injected skill usage."""
        skill_label = ", ".join(called[:4]) + (
            f" +{len(called) - 4}" if len(called) > 4 else ""
        )
        skill_iteration_map = self._learned_skill_iteration_map()
        skill_iterations = {
            name: skill_iteration_map[name]
            for name in called
            if name in skill_iteration_map
        }
        timeline_event = {
            "tool_name": "Learned Skill Usage",
            "text": f"Policy referenced learned skill(s): {', '.join(called)}",
            "images": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step_index": -1,
            "highlight": True,
            "frame_start": start_frame,
            "frame_end": end_frame,
            "start_s": (
                round(start_frame / 20.0, 3)
                if start_frame is not None
                else None
            ),
            "end_s": (
                round(end_frame / 20.0, 3)
                if end_frame is not None
                else None
            ),
            "timeline_kind": "learned_skill",
            "timeline_label": skill_label,
            "iteration": self._iteration,
            "attempt": attempt,
            "turn": turn,
            "block_index": block_index,
            "skills": called,
            "skill_iterations": skill_iterations,
        }
        self._attempt_timeline_events.append(timeline_event)
        return timeline_event

    def _learned_skill_iteration_map(self) -> dict[str, int]:
        """Best-effort map of learned skill name -> iteration learned.

        New runs persist ``learned_iteration`` into skills.json. For older runs,
        infer from iteration_*/generated_skills/<skill>.py in either this run or
        merged playtime-run directories so backfilled/benchmark overlay videos
        can still display ``(iter N)`` when the artifact exists.
        """
        mapping: dict[str, int] = {}

        def _put(name: Any, value: Any, *, overwrite: bool = False) -> None:
            skill_name = str(name or "").strip()
            if not skill_name:
                return
            try:
                iteration = int(value)
            except (TypeError, ValueError):
                return
            if overwrite or skill_name not in mapping:
                mapping[skill_name] = iteration

        try:
            raw_skills = getattr(self.skill_library, "_skills", None)
            source_skills = (
                list(raw_skills)
                if isinstance(raw_skills, list)
                else self.skill_library.get_full_skills_for_planner(
                    include_deprecated=True,
                )
            )
            for skill in source_skills:
                if skill.get("is_primitive", False):
                    continue
                value = (
                    skill.get("learned_iteration")
                    or skill.get("created_iteration")
                    or skill.get("source_iteration")
                )
                _put(skill.get("name"), value)
        except Exception:
            pass

        search_roots: list[Path] = []
        output_dir = getattr(self, "output_dir", None)
        if output_dir is not None:
            search_roots.append(Path(output_dir))
        for merge_path in getattr(self, "_skill_library_merge_paths", []) or []:
            path = Path(merge_path)
            search_roots.append(path.parent if path.is_file() else path)

        try:
            for root in search_roots:
                for skill_file in root.glob("iteration_*/generated_skills/*.py"):
                    match = re.search(r"iteration_(\d+)", skill_file.parent.parent.name)
                    if match:
                        iteration = match.group(1)
                        skill_name = skill_file.stem
                        try:
                            head = skill_file.read_text(errors="ignore")[:1000]
                            skill_match = re.search(
                                r"^# skill:\s*[\"']?([^\"'\n]+)[\"']?",
                                head,
                                flags=re.MULTILINE,
                            )
                            if skill_match:
                                skill_name = skill_match.group(1).strip()
                        except Exception:
                            pass
                        _put(skill_name, iteration)
        except Exception:
            pass

        return mapping

    def _write_skill_overlay_video(
        self,
        path: Path,
        frames: list[Any],
        timeline_events: list[dict[str, Any]],
        *,
        frame_offset: int = 0,
        fps: float = 20.0,
    ) -> bool:
        """Write an MP4 with one highlighted current skill/API label."""
        if not frames:
            return False
        try:
            import imageio
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            logger.debug(f"  Skill overlay video deps unavailable: {exc}")
            return False

        learned_iterations = self._learned_skill_iteration_map()

        def _event_bounds(event: dict[str, Any]) -> tuple[int | None, int | None]:
            try:
                start = event.get("frame_start")
                end = event.get("frame_end")
                if start is None or end is None:
                    return None, None
                return int(start), int(end)
            except (TypeError, ValueError):
                return None, None

        def _active(event: dict[str, Any], frame_idx: int) -> bool:
            start, end = _event_bounds(event)
            if start is None or end is None:
                return False
            return start <= frame_idx < max(start + 1, end)

        def _event_sort_key(event: dict[str, Any]) -> tuple[int, int]:
            start, _ = _event_bounds(event)
            try:
                step = int(event.get("step_index", 0) or 0)
            except (TypeError, ValueError):
                step = 0
            return (start if start is not None else -1, step)

        def _truncate(text: str, max_chars: int = 52) -> str:
            text = " ".join(str(text or "").split())
            if len(text) <= max_chars:
                return text
            iter_marker = " (iter "
            if iter_marker in text:
                prefix, suffix = text.rsplit(iter_marker, 1)
                suffix = iter_marker + suffix
                max_prefix = max(8, max_chars - len(suffix) - 1)
                return prefix[:max_prefix].rstrip() + "…" + suffix
            return text[: max(1, max_chars - 1)].rstrip() + "…"

        def _learned_skill_label(event: dict[str, Any]) -> str:
            skills = [str(s).strip() for s in event.get("skills") or [] if str(s).strip()]
            skill = skills[0] if skills else str(event.get("timeline_label") or "").strip()
            if not skill:
                return ""
            event_iterations = event.get("skill_iterations") or {}
            learned_iter = event_iterations.get(skill) or learned_iterations.get(skill)
            if learned_iter is not None:
                return f"{skill} (iter {learned_iter})"
            return skill

        def _api_label(event: dict[str, Any]) -> str:
            label = str(
                event.get("timeline_label")
                or event.get("tool_name")
                or event.get("name")
                or ""
            ).strip()
            return label

        def _current_label(frame_idx: int) -> tuple[str, str]:
            active_events = [
                event for event in timeline_events
                if isinstance(event, dict) and _active(event, frame_idx)
            ]
            if not active_events:
                return "", ""
            active_events.sort(key=_event_sort_key)

            # Learned-skill events are the only ones annotated as learned, so
            # prefer them when active. This keeps primitive/helper API events
            # from receiving a misleading ``(iter N)`` suffix.
            for event in reversed(active_events):
                if event.get("timeline_kind") == "learned_skill":
                    label = _learned_skill_label(event)
                    if label:
                        return _truncate(label), "skill"

            for event in reversed(active_events):
                if event.get("timeline_kind") != "learned_skill":
                    label = _api_label(event)
                    if label:
                        return _truncate(label), "api"
            return "", ""

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        annotated: list[Any] = []
        for local_idx, frame in enumerate(frames):
            global_idx = frame_offset + local_idx
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr[..., :3]).convert("RGBA")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            label, label_kind = _current_label(global_idx)
            if label:
                pad_x = 8
                pad_y = 6
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w = max(80, len(label) * 10)
                    text_h = 18
                x0, y0 = 8, 8
                x1 = min(img.size[0] - 8, x0 + text_w + 2 * pad_x)
                y1 = min(img.size[1] - 8, y0 + text_h + 2 * pad_y)
                text_fill = (
                    (255, 230, 80, 255)
                    if label_kind == "skill"
                    else (255, 255, 255, 255)
                )
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=(0, 0, 0, 170),
                    outline=(255, 255, 255, 160),
                )
                draw.text((x0 + pad_x, y0 + pad_y), label, fill=text_fill, font=font)

            annotated.append(np.asarray(Image.alpha_composite(img, overlay).convert("RGB")))

        try:
            imageio.mimsave(str(path), annotated, fps=fps)
            return True
        except Exception as exc:
            logger.debug(f"  Skill overlay video save failed ({path}): {exc}")
            return False

    def _save_attempt_videos(
        self,
        *,
        low_level: Any,
        attempt_in_iter: int,
        attempt: int | None = None,
        status: str,
    ) -> dict[str, Any]:
        """Flush an attempt's accumulated frames to disk, capx-style.

        Output layout when ``turns_per_attempt > 1``:

            iter{NNN}_attempt{M}_{status}/
                turn_00.mp4
                turn_01.mp4
                ...
                combined.mp4

        In legacy single-shot mode (``turns_per_attempt == 1``) the
        attempt has exactly one turn, so we keep the historical flat
        filename ``iter{NNN}_attempt{N}_{status}.mp4`` to avoid breaking
        downstream tooling that scans for those files. ``attempt_in_iter``
        is the user-facing 0-based attempt index for the current
        iteration, NOT the global flat ``attempt`` step counter.

        Clears the env's frame buffer when done; the next attempt will
        start with an empty buffer at its first ``enable_video_capture``.
        """
        artifacts: dict[str, Any] = {
            "iteration": self._iteration,
            "attempt_in_iter": attempt_in_iter,
            "attempt": attempt,
            "status": status,
        }
        if not hasattr(low_level, "get_video_frames"):
            self._export_attempt_viser_recording(artifacts, attempt_in_iter, attempt, status)
            return artifacts
        try:
            all_frames = low_level.get_video_frames(clear=True)
        except Exception as e:
            logger.debug(f"  get_video_frames failed: {e}")
            self._export_attempt_viser_recording(artifacts, attempt_in_iter, attempt, status)
            return artifacts
        if not all_frames:
            self._attempt_turn_frame_ranges = []
            self._attempt_timeline_events = []
            self._export_attempt_viser_recording(artifacts, attempt_in_iter, attempt, status)
            return artifacts

        import imageio
        ranges = list(self._attempt_turn_frame_ranges)
        timeline_events = list(getattr(self, "_attempt_timeline_events", []) or [])

        def _write_timeline(path: Path) -> None:
            try:
                payload = {
                    "iteration": self._iteration,
                    "attempt": attempt_in_iter,
                    "status": status,
                    "fps": 20.0,
                    "frame_count": len(all_frames),
                    "turn_frame_ranges": [
                        {"turn": idx, "frame_start": start, "frame_end": end}
                        for idx, (start, end) in enumerate(ranges)
                    ],
                    "events": timeline_events,
                }
                path.write_text(json.dumps(payload, indent=2, default=str))
            except Exception as e:
                logger.debug(f"  Attempt timeline save failed: {e}")

        # Drop the trailing lists so the next attempt starts fresh.
        self._attempt_turn_frame_ranges = []
        self._attempt_timeline_events = []

        if self._turns_per_attempt <= 1:
            # Legacy single-shot: one .mp4 per attempt, flat filename.
            video_path = (
                self.output_dir
                / f"iter{self._iteration:03d}_attempt{attempt_in_iter}_{status}.mp4"
            )
            try:
                imageio.mimsave(str(video_path), all_frames, fps=20)
                overlay_path = video_path.with_name(
                    f"{video_path.stem}_skills.mp4"
                )
                overlay_ok = self._write_skill_overlay_video(
                    overlay_path,
                    all_frames,
                    timeline_events,
                    frame_offset=0,
                    fps=20.0,
                )
                _write_timeline(video_path.with_suffix(".timeline.json"))
                artifacts.update({
                    "video_path": str(video_path),
                    "skills_video_path": str(overlay_path) if overlay_ok else None,
                    "timeline_path": str(video_path.with_suffix(".timeline.json")),
                })
                logger.info(
                    f"  Saved attempt video: {video_path.name} "
                    f"({len(all_frames)} frames)"
                    + (f" + {overlay_path.name}" if overlay_ok else "")
                )
            except Exception as e:
                logger.debug(f"  Attempt video save failed: {e}")
            self._export_attempt_viser_recording(artifacts, attempt_in_iter, attempt, status)
            return artifacts

        # Nested mode: capx-style folder.
        attempt_dir = (
            self.output_dir
            / f"iter{self._iteration:03d}_attempt{attempt_in_iter}_{status}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        artifacts["attempt_dir"] = str(attempt_dir)
        for i, (start, end) in enumerate(ranges):
            if start >= end or end > len(all_frames):
                continue
            turn_frames = all_frames[start:end]
            if not turn_frames:
                continue
            turn_path = attempt_dir / f"turn_{i:02d}.mp4"
            try:
                imageio.mimsave(
                    str(turn_path),
                    turn_frames, fps=20,
                )
                self._write_skill_overlay_video(
                    attempt_dir / f"turn_{i:02d}_skills.mp4",
                    turn_frames,
                    timeline_events,
                    frame_offset=start,
                    fps=20.0,
                )
            except Exception as e:
                logger.debug(f"  Per-turn video save failed (turn {i}): {e}")
        try:
            combined_path = attempt_dir / "combined.mp4"
            imageio.mimsave(
                str(combined_path), all_frames, fps=20,
            )
            overlay_ok = self._write_skill_overlay_video(
                attempt_dir / "combined_skills.mp4",
                all_frames,
                timeline_events,
                frame_offset=0,
                fps=20.0,
            )
            _write_timeline(attempt_dir / "timeline.json")
            artifacts.update({
                "combined_video_path": str(combined_path),
                "combined_skills_video_path": str(attempt_dir / "combined_skills.mp4") if overlay_ok else None,
                "timeline_path": str(attempt_dir / "timeline.json"),
            })
        except Exception as e:
            logger.debug(f"  Combined attempt video save failed: {e}")
        self._export_attempt_viser_recording(artifacts, attempt_in_iter, attempt, status)
        logger.info(
            f"  Saved attempt videos to {attempt_dir.name}/ "
            f"({len(ranges)} turn(s), {len(all_frames)} total frames"
            f"{', skills overlay' if 'overlay_ok' in locals() and overlay_ok else ''})"
        )
        return artifacts

    def _export_attempt_viser_recording(
        self,
        artifacts: dict[str, Any],
        attempt_in_iter: int,
        attempt: int | None,
        status: str,
    ) -> None:
        """Persist the slice of MolmoSpaces Viser snapshots for this attempt."""
        export_fn = getattr(self.web_debugger, "export_viser_recording", None)
        if not callable(export_fn):
            self._attempt_viser_frame_start = None
            return
        mark_fn = getattr(self.web_debugger, "mark_viser_recording", None)
        frame_end = mark_fn() if callable(mark_fn) else None
        frame_start = self._attempt_viser_frame_start
        self._attempt_viser_frame_start = None
        if frame_start is None:
            return

        detailed_attempt_dir = None
        if attempt is not None:
            detailed_attempt_dir = (
                self.output_dir
                / f"iteration_{self._iteration:03d}"
                / f"attempt_{attempt:02d}"
            )
        primary_dir = Path(
            artifacts.get("attempt_dir")
            or detailed_attempt_dir
            or (
                self.output_dir
                / f"iter{self._iteration:03d}_attempt{attempt_in_iter}_{status}_artifacts"
            )
        )
        metadata = {
            "iteration": self._iteration,
            "attempt_in_iter": attempt_in_iter,
            "attempt": attempt,
            "status": status,
            "artifact_kind": "attempt_viser_recording",
        }
        try:
            exported = export_fn(
                primary_dir,
                frame_start=frame_start,
                frame_end=frame_end,
                metadata_extra=metadata,
            )
        except Exception as exc:
            logger.debug("  Attempt Viser recording export failed: %s", exc)
            return
        if not exported:
            return
        exported.update({
            "frame_start": frame_start,
            "frame_end": frame_end,
            "attempt_dir": str(primary_dir),
        })
        artifacts["viser_recording"] = exported
        if detailed_attempt_dir is not None and detailed_attempt_dir != primary_dir:
            try:
                detailed_attempt_dir.mkdir(parents=True, exist_ok=True)
                link_path = detailed_attempt_dir / "viser_recording_link.json"
                link_path.write_text(json.dumps(exported, indent=2, default=str))
                exported["detailed_attempt_link_path"] = str(link_path)
            except Exception as exc:
                logger.debug("  Attempt Viser recording link save failed: %s", exc)

    @staticmethod
    def _attach_attempt_media_artifacts(
        iteration_data: dict[str, Any],
        attempt: int,
        artifacts: dict[str, Any] | None,
    ) -> None:
        """Attach saved video/Viser paths to the persisted execution attempt."""
        if not artifacts:
            return
        record = iteration_data.get(f"execution_attempt_{attempt}")
        if not isinstance(record, dict):
            return
        record["attempt_artifacts"] = artifacts
        if artifacts.get("viser_recording"):
            record["viser_recording"] = artifacts["viser_recording"]

    def _reset_env(self, *, recover_on_failure: bool = True) -> bool:
        """Reset environment for next task.

        Passes the current iteration number as seed so that each iteration
        uses a different init state (important for LIBERO eval where each
        task has 50 init states).

        Returns True on success and False on failure. Callers should check
        the return value before stepping the env — for MolmoSpaces, a
        failed reset leaves the underlying bridge in a state where calling
        ``env.step(code)`` will raise (or worse, silently use stale
        observations from the previous task). The flag also lets the
        iteration mark itself as a soft init failure rather than counting
        the policy as having "failed".

        ``recover_on_failure`` is disabled for normal task-rebind paths where
        a reset failure usually means "this proposed task/spec is invalid for
        the active house"; those callers should sample a compatible house or
        mark the init failed rather than silently recreating/stale-falling back.
        """
        # Sticky flag the iteration loop reads to skip executor / verifier
        # work when init is broken. Cleared on every successful reset.
        self._last_reset_failed = False
        try:
            seed_offset = int(os.getenv("RATS_SEED_OFFSET", "0") or 0)
            self.env.reset(seed=self._iteration + seed_offset)
            return True
        except TypeError:
            # Fallback for envs that don't accept seed
            try:
                self.env.reset()
                return True
            except Exception as e:
                logger.warning(f"  Environment reset failed: {e}")
                self._last_reset_failed = True
                self._last_reset_error = str(e)
                return False
        except Exception as e:
            logger.warning(f"  Environment reset failed: {e}")
            self._last_reset_failed = True
            self._last_reset_error = str(e)
            # MolmoSpaces-specific recovery: if the remote bridge is
            # wedged, re-create the env once. The new bridge process
            # gets a fresh sampler with empty placement caches.
            if recover_on_failure and self.env_type == "molmospaces":
                if self._try_recover_molmospaces_env():
                    self._last_reset_failed = False
                    return True
            return False

    def _should_reset_for_next_iteration(self) -> bool:
        """Return whether the current run has another registered iteration.

        Direct callers that bypass :meth:`run` keep the legacy reset behavior;
        normal runs set ``_run_terminal_iteration`` at their entry point.
        """

        terminal_iteration = getattr(self, "_run_terminal_iteration", None)
        return terminal_iteration is None or self._iteration < terminal_iteration

    def _close_molmospaces_env_socket_for_recovery(self) -> None:
        """Close the current MolmoSpaces bridge before opening a replacement.

        ``scripts/mlspaces_server.py`` serves one client connection at a time.
        If RATS creates a second ``RemoteMolmoSpacesBridge`` while the old
        socket is still open, the server can remain blocked servicing the old
        idle connection and never accept the new one. Closing the current env
        first forces the server back to ``accept()`` before recovery/recreation.
        """
        try:
            close_fn = getattr(self.env, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception as e:
            logger.debug("  Ignoring MolmoSpaces env close failure before recovery: %s", e)

    def _try_recover_molmospaces_env(self) -> bool:
        """Attempt to recover from a wedged MolmoSpaces env.

        Re-creates the env on the current task (bootstrap or last
        proposed). Best-effort: if this fails too, the iteration's reset
        check will skip executor work and the task proposer will see a
        soft init failure recorded against the activity.
        """
        try:
            try:
                scene_ctx = self._get_scene_context()
            except Exception as e:
                logger.warning(
                    "  Could not read MolmoSpaces scene context before recovery: %s",
                    e,
                )
                scene_ctx = {}
            current_task = scene_ctx.get("activity_name") or self._bootstrap_activity
            if scene_ctx.get("molmospaces_task_kind") == "playtime":
                descriptor = scene_ctx.get("task_descriptor") or {}
                metadata = descriptor.get("metadata") or {}
                current_task = (
                    metadata.get("base_canonical_task_id")
                    or self._bootstrap_activity
                    or current_task
                )
            if not current_task:
                return False
            self._close_molmospaces_env_socket_for_recovery()
            new_env = recreate_molmospaces_env(self.env, current_task)
            self.env = new_env
            self._set_api_output_dir(self.env)
            self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)
            self._queue_molmospaces_environment_check("env_recovered_after_reset_failure")
            logger.info(
                f"  Recovered MolmoSpaces env after wedged reset "
                f"(task={current_task})"
            )
            return True
        except Exception as e:
            logger.warning(f"  MolmoSpaces env recovery failed: {e}")
            return False

    def _reinitialize_molmospaces_current_task(
        self,
        task_proposal: dict[str, Any],
    ) -> bool:
        """Recreate MolmoSpaces with the current benchmark task pinned."""
        current_task = (
            task_proposal.get("canonical_task_id")
            or task_proposal.get("activity_name")
            or self._bootstrap_activity
        )
        if not current_task:
            return False
        try:
            self._close_molmospaces_env_socket_for_recovery()
            self.env = recreate_molmospaces_env(self.env, str(current_task))
            self._set_api_output_dir(self.env)
            self.executor = Executor(timeout_seconds=self.executor.timeout_seconds)
            self._queue_molmospaces_environment_check("benchmark_reinitialized_current_task")
            logger.info("  Reinitialized MolmoSpaces env with task pinned: %s", current_task)
            return True
        except Exception as e:
            logger.warning(
                "  MolmoSpaces pinned-task reinitialization failed for %s: %s",
                current_task,
                e,
            )
            self._last_reset_failed = True
            self._last_reset_error = f"environment verifier reinit failed: {e}"
            return False

    def _save_iteration_result(self, result: dict[str, Any]) -> None:
        """Save iteration result to disk, then refresh report.md live.

        Re-rendering after each iteration means a long run is browsable
        in-flight: `tail -n 50 outputs/<run>/report.md` or opening it in
        VSCode shows the live cumulative-SR table, latest verifier analysis,
        and the matplotlib charts up to the most-recent iter.
        """
        path = self.output_dir / f"iteration_{result['iteration']:03d}.json"
        serializable = self._json_safe(result)
        with path.open("w") as f:
            json.dump(serializable, f, indent=2, allow_nan=False)

        # Persist exact policy code files plus skill/API usage graphs as live
        # run artifacts. This is best-effort and deliberately non-fatal: the
        # benchmark result JSON remains the source of truth if export fails.
        try:
            from scripts.export_attempt_artifacts import export_attempt_artifacts

            export_attempt_artifacts(self.output_dir)
        except Exception as e:
            logger.debug(f"  attempt artifact export failed: {e}")

        # Best-effort live markdown refresh. Non-fatal if it fails (the
        # renderer imports matplotlib which can fail on weird envs).
        try:
            from scripts.render_run_md import render as _render_md
            (self.output_dir / "report.md").write_text(_render_md(self.output_dir))
        except Exception as e:
            logger.debug(f"  live markdown refresh failed: {e}")

    def _maybe_snapshot(self) -> None:
        """Snapshot skill library + failure memory at periodic intervals."""
        if self._snapshot_interval <= 0:
            return
        if self._iteration % self._snapshot_interval != 0:
            return
        snap_dir = self.output_dir / "snapshots" / f"iter{self._iteration:03d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        skills_src = self.output_dir / "skills.json"
        if skills_src.exists():
            shutil.copy2(skills_src, snap_dir / "skills.json")
        fm_src = self.output_dir / "failure_memory"
        if fm_src.is_dir():
            dst = snap_dir / "failure_memory"
            if not dst.exists():
                shutil.copytree(fm_src, dst)
        logger.info(
            "  Snapshot iter %d -> %s", self._iteration, snap_dir,
        )

    def _json_safe(self, value: Any) -> Any:
        """Convert proposal/debug payloads into strict JSON-serializable values."""
        return _json_safe_value(value)

    def _save_molmospaces_playtime_proposal_artifacts(
        self,
        task_proposal: dict[str, Any],
        *,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Persist playtime proposer prompt, response, and prompt-only spec."""
        if task_proposal.get("_molmospaces_proposer_mode") != "playtime":
            return {}

        proposal_dir = self.output_dir / "task_proposals"
        spec_dir = self.output_dir / "generated_molmospaces_specs"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        spec_dir.mkdir(parents=True, exist_ok=True)

        iter_tag = f"iter{self._iteration:03d}"
        activity = str(task_proposal.get("activity_name") or "molmospaces_playtime")
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", activity).strip("_") or "task"

        safe_trace = self._json_safe(trace or {})
        safe_proposal = self._json_safe(task_proposal)
        final_result = safe_trace.get("final_llm_or_fallback_result") if isinstance(safe_trace, dict) else None
        if final_result is None and isinstance(safe_trace, dict):
            attempts = safe_trace.get("attempts") or []
            if attempts:
                final_result = attempts[-1].get("result")

        trace_path = proposal_dir / f"{iter_tag}_molmospaces_playtime_trace.json"
        proposal_path = proposal_dir / f"{iter_tag}_molmospaces_playtime_proposal.json"
        response_path = proposal_dir / f"{iter_tag}_molmospaces_playtime_response.json"
        prompt_path = proposal_dir / f"{iter_tag}_molmospaces_playtime_prompt.txt"
        spec_path = spec_dir / f"{iter_tag}_{slug}.json"

        trace_path.write_text(json.dumps(safe_trace, indent=2))
        proposal_path.write_text(json.dumps(safe_proposal, indent=2))
        response_path.write_text(json.dumps(self._json_safe(final_result or {}), indent=2))

        prompt_parts: list[str] = []
        if isinstance(trace, dict):
            if trace.get("system_prompt"):
                prompt_parts.append("## SYSTEM PROMPT\n")
                prompt_parts.append(str(trace["system_prompt"]))
            if trace.get("user_prompt"):
                prompt_parts.append("\n\n## USER PROMPT\n")
                prompt_parts.append(str(trace["user_prompt"]))
            if trace.get("retry_prompt"):
                prompt_parts.append("\n\n## RETRY PROMPT\n")
                prompt_parts.append(str(trace["retry_prompt"]))
        prompt_path.write_text("".join(prompt_parts) if prompt_parts else "(prompt unavailable)\n")

        runtime_spec = {
            "artifact_version": "1.0",
            "env_type": "molmospaces",
            "format": "playtime_prompt_overlay_task_spec",
            "canonical_task_id": task_proposal.get("canonical_task_id") or activity,
            "activity_name": activity,
            "language_goal": task_proposal.get("language")
            or task_proposal.get("goal_conditions", ""),
            "task_family": task_proposal.get("task_family"),
            "scene_family": task_proposal.get("scene_family"),
            "benchmark_or_catalog": task_proposal.get("benchmark"),
            "objects": task_proposal.get("objects", []),
            "bridge_rebind": {"method": "prompt_overlay", "kwargs": {}},
            "playtime": task_proposal.get("_playtime") or {},
            "request_house_switch": bool(task_proposal.get("_request_house_switch")),
            "verifier": "vlm",
            "generator_metadata": {
                "source": "TaskProposer._propose_novel_molmospaces_playtime",
                "iteration": self._iteration,
                "mode": task_proposal.get("mode", "novel"),
                "reasoning": task_proposal.get("reasoning", ""),
                "difficulty_estimate": task_proposal.get("difficulty_estimate"),
                "novelty_score": task_proposal.get("novelty_score"),
                "curiosity_score": task_proposal.get("curiosity_score"),
                "proposal_trace_path": str(trace_path),
                "proposal_prompt_path": str(prompt_path),
                "proposal_response_path": str(response_path),
            },
            "validation_status": "proposed",
        }
        spec_path.write_text(json.dumps(self._json_safe(runtime_spec), indent=2))
        logger.info("  Saved MolmoSpaces playtime proposal trace: %s", trace_path)
        return {
            "trace": str(trace_path),
            "prompt": str(prompt_path),
            "response": str(response_path),
            "proposal": str(proposal_path),
            "spec": str(spec_path),
        }

    def _record_playtime_outcome(
        self,
        *,
        task_proposal: dict[str, Any],
        verification: dict[str, Any],
        iteration_data: dict[str, Any],
        attempt_in_iter: int,
    ) -> None:
        """No-op: playtime-specific archival removed during play-mode unification.

        Earlier MolmoSpaces playtime runs wrote per-iteration entries to
        ``playtime_memory.jsonl`` so the next proposer pass could read
        category-level affordances. The unified play mode emits the same
        proposal schema as the LIBERO play path (standard novel keys
        only — no ``_playtime`` metadata, no ``interaction_type``), so
        there is no extra archive to write here. Kept as a stub so any
        legacy caller that still invokes ``_record_playtime_outcome``
        is a no-op rather than an AttributeError.
        """
        return

    def _save_molmospaces_open_proposal_artifacts(
        self,
        task_proposal: dict[str, Any],
        *,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Persist MolmoSpaces open-proposer prompts, LLM JSON, and bridge spec.

        The open proposer does not use ``EnvironmentCreator`` (it rebinds the
        live bridge via ``set_task_from_spec`` / ``request_new_house``), so the
        legacy ``generated_molmospaces_specs`` directory would otherwise remain
        empty. This method writes the equivalent proposed runtime artifact plus
        the proposal-step debug log under the run output directory.
        """
        if task_proposal.get("_molmospaces_proposer_mode") != "open":
            return {}

        proposal_dir = self.output_dir / "task_proposals"
        spec_dir = self.output_dir / "generated_molmospaces_specs"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        spec_dir.mkdir(parents=True, exist_ok=True)

        iter_tag = f"iter{self._iteration:03d}"
        activity = str(task_proposal.get("activity_name") or "molmospaces_open")
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", activity).strip("_") or "task"

        safe_trace = self._json_safe(trace or {})
        safe_proposal = self._json_safe(task_proposal)
        final_result = safe_trace.get("final_llm_or_fallback_result") if isinstance(safe_trace, dict) else None
        if final_result is None and isinstance(safe_trace, dict):
            attempts = safe_trace.get("attempts") or []
            if attempts:
                final_result = attempts[-1].get("result")

        trace_path = proposal_dir / f"{iter_tag}_molmospaces_open_trace.json"
        proposal_path = proposal_dir / f"{iter_tag}_molmospaces_open_proposal.json"
        response_path = proposal_dir / f"{iter_tag}_molmospaces_open_response.json"
        prompt_path = proposal_dir / f"{iter_tag}_molmospaces_open_prompt.txt"
        spec_path = spec_dir / f"{iter_tag}_{slug}.json"

        trace_path.write_text(json.dumps(safe_trace, indent=2))
        proposal_path.write_text(json.dumps(safe_proposal, indent=2))
        response_path.write_text(json.dumps(self._json_safe(final_result or {}), indent=2))

        prompt_parts: list[str] = []
        if isinstance(trace, dict):
            system_prompt = trace.get("system_prompt")
            user_prompt = trace.get("user_prompt")
            retry_prompt = trace.get("retry_prompt")
            if system_prompt:
                prompt_parts.append("## SYSTEM PROMPT\n")
                prompt_parts.append(str(system_prompt))
            if user_prompt:
                prompt_parts.append("\n\n## USER PROMPT\n")
                prompt_parts.append(str(user_prompt))
            if retry_prompt:
                prompt_parts.append("\n\n## RETRY PROMPT\n")
                prompt_parts.append(str(retry_prompt))
        prompt_path.write_text("".join(prompt_parts) if prompt_parts else "(prompt unavailable)\n")

        open_spec = task_proposal.get("_molmospaces_open_spec")
        if task_proposal.get("_request_house_switch"):
            bridge_call = {
                "method": "request_new_house",
                "kwargs": {
                    "house_index": None,
                    "task_type": task_proposal.get("_molmospaces_requested_task_type"),
                },
            }
        else:
            bridge_call = {
                "method": "set_task_from_spec",
                "kwargs": dict(open_spec or {}),
            }
        runtime_spec = {
            "artifact_version": "1.0",
            "env_type": "molmospaces",
            "format": "open_runtime_task_spec",
            "canonical_task_id": task_proposal.get("canonical_task_id") or activity,
            "activity_name": activity,
            "language_goal": task_proposal.get("language")
            or task_proposal.get("goal_conditions", ""),
            "task_family": task_proposal.get("task_family"),
            "scene_family": task_proposal.get("scene_family"),
            "benchmark_or_catalog": task_proposal.get("benchmark"),
            "objects": task_proposal.get("objects", []),
            "bridge_rebind": bridge_call,
            "open_spec": open_spec,
            "request_house_switch": bool(task_proposal.get("_request_house_switch")),
            "generator_metadata": {
                "source": "TaskProposer._propose_novel_molmospaces_open",
                "iteration": self._iteration,
                "mode": task_proposal.get("mode", "novel"),
                "reasoning": task_proposal.get("reasoning", ""),
                "difficulty_estimate": task_proposal.get("difficulty_estimate"),
                "novelty_score": task_proposal.get("novelty_score"),
                "curiosity_score": task_proposal.get("curiosity_score"),
                "proposal_trace_path": str(trace_path),
                "proposal_prompt_path": str(prompt_path),
                "proposal_response_path": str(response_path),
            },
            "validation_status": "proposed",
        }
        spec_path.write_text(json.dumps(self._json_safe(runtime_spec), indent=2))

        logger.info("  Saved MolmoSpaces proposal trace: %s", trace_path)
        logger.info("  Generated MolmoSpaces open spec: %s", spec_path)
        return {
            "trace": str(trace_path),
            "prompt": str(prompt_path),
            "response": str(response_path),
            "proposal": str(proposal_path),
            "spec": str(spec_path),
        }

    def _save_molmospaces_post_switch_artifacts(
        self,
        rebuilt_proposal: dict[str, Any],
        scene_context: dict[str, Any],
        *,
        switch_artifact_paths: dict[str, str] | None = None,
        requested_task_type: str | None = None,
    ) -> dict[str, str]:
        """Persist the auto-sampled task that the bridge drew after a switch_house.

        ``_save_molmospaces_open_proposal_artifacts`` runs *before* the rebind,
        so the LLM's switch_house pseudo-proposal is the only thing it can
        capture. After ``request_new_house`` returns, the bridge has sampled
        a real task in the new house — this method writes a sibling
        ``iter00X_molmospaces_post_switch_*`` artifact set so the proposals /
        specs directories show what actually ran.
        """
        if rebuilt_proposal.get("env_type") and rebuilt_proposal.get("env_type") != "molmospaces":
            return {}
        if scene_context.get("env_type") != "molmospaces":
            return {}

        proposal_dir = self.output_dir / "task_proposals"
        spec_dir = self.output_dir / "generated_molmospaces_specs"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        spec_dir.mkdir(parents=True, exist_ok=True)

        iter_tag = f"iter{self._iteration:03d}"
        descriptor = scene_context.get("task_descriptor") or {}
        activity = (
            rebuilt_proposal.get("activity_name")
            or descriptor.get("canonical_id")
            or "molmospaces_post_switch"
        )
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(activity)).strip("_") or "task"

        proposal_path = proposal_dir / f"{iter_tag}_molmospaces_post_switch_proposal.json"
        spec_path = spec_dir / f"{iter_tag}_post_switch_{slug}.json"

        proposal_path.write_text(
            json.dumps(self._json_safe(rebuilt_proposal), indent=2)
        )

        runtime_spec = {
            "artifact_version": "1.0",
            "env_type": "molmospaces",
            "format": "post_switch_runtime_task_spec",
            "canonical_task_id": (
                rebuilt_proposal.get("canonical_task_id")
                or descriptor.get("canonical_id")
                or activity
            ),
            "activity_name": activity,
            "language_goal": (
                rebuilt_proposal.get("language")
                or descriptor.get("language")
                or rebuilt_proposal.get("goal_conditions", "")
            ),
            "task_family": (
                rebuilt_proposal.get("task_family")
                or descriptor.get("task_family")
            ),
            "scene_family": (
                rebuilt_proposal.get("scene_family")
                or descriptor.get("scene_family")
            ),
            "benchmark_or_catalog": (
                rebuilt_proposal.get("benchmark")
                or descriptor.get("benchmark")
            ),
            "variant": (
                rebuilt_proposal.get("variant")
                or descriptor.get("variant")
            ),
            "objects": list(
                rebuilt_proposal.get("objects")
                or descriptor.get("objects")
                or []
            ),
            "metadata": descriptor.get("metadata") or {},
            "requested_task_type": requested_task_type,
            "auto_sampled_after_switch_house": True,
            "generator_metadata": {
                "source": "_rebind_molmospaces_env(request_new_house)",
                "iteration": self._iteration,
                "switch_proposal_artifacts": dict(switch_artifact_paths or {}),
            },
            "validation_status": "auto_sampled",
        }
        spec_path.write_text(json.dumps(self._json_safe(runtime_spec), indent=2))

        logger.info("  Saved MolmoSpaces post-switch proposal: %s", proposal_path)
        logger.info("  Generated MolmoSpaces post-switch spec: %s", spec_path)
        return {
            "post_switch_proposal": str(proposal_path),
            "post_switch_spec": str(spec_path),
        }

    def _log_iteration_summary(self, result: dict[str, Any]) -> None:
        """Log a summary of the iteration."""
        logger.info(f"\n--- Iteration {result['iteration']} Summary ---")
        logger.info(f"  Task: {result.get('task_proposal', {}).get('activity_name', 'unknown')}")
        logger.info(f"  Success: {result.get('success', False)}")
        records = self.metrics.get_records()
        completed = sum(1 for r in records if r.get("success"))
        total = len(records)
        if total:
            logger.info(f"  Running success rate: {completed}/{total} ({completed / total:.1%})")
        logger.info(f"  Attempts: {result.get('total_attempts', 0)}")
        logger.info(f"  Skills learned: {result.get('skills_learned', [])}")
        logger.info(f"  Library size: {result.get('skill_library_size', 0)} ({result.get('learned_skill_count', 0)} learned)")
        logger.info(f"  Elapsed: {result.get('elapsed_seconds', 0):.1f}s")

    def _build_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Build final summary of all iterations."""
        total = len(results)
        successes = sum(1 for r in results if r.get("success"))
        return {
            "total_iterations": total,
            "successful_iterations": successes,
            "failed_iterations": sum(1 for r in results if not r.get("success")),
            "success_rate": successes / total if total else 0.0,
            "final_skill_library_size": len(self.skill_library.get_all_skill_names()),
            "learned_skills": self.skill_library.get_learned_skill_count(),
            "failure_memory": self.failure_memory.get_failure_stats(),
            "metrics": self.metrics.get_summary(),
            "iterations": results,
        }

    def _save_summary(self, summary: dict[str, Any]) -> None:
        """Save final summary to disk."""
        path = self.output_dir / "lifelong_summary.json"
        serializable = self._json_safe(summary)
        with path.open("w") as f:
            json.dump(serializable, f, indent=2, allow_nan=False)
        logger.info(f"\nSummary saved to {path}")
