"""Single-trial execution for CaP-X environments.

This module handles single trial execution including code generation,
multi-turn decisions, and visual feedback. It contains the core trial
loop extracted from launch.py, covering:

- Initial code generation and oracle code handling
- Code block execution with multi-turn regeneration
- Visual feedback capture and image/video differencing
- Trial artifact saving (code, logs, per-turn videos, combined video)
"""

from __future__ import annotations

import base64
import copy
import re
import gc
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rats.envs.configs.instantiate import instantiate
from rats.envs.tasks.base import CodeExecutionEnvBase

from rats.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_ensemble as _query_model_ensemble,
    query_single_model_ensemble as _query_single_model_ensemble,
)
from rats.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _build_multi_turn_decision_prompt_legacy,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from rats.utils.video_utils import _encode_video_base64, _write_video

# Use TYPE_CHECKING to avoid circular imports for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rats.envs.launch import LaunchArgs


MULTITURN_LIMIT = 10


def _load_rats_runtime_helpers():
    """Import embedded RATS runtime helpers only when that path is enabled.

    Plain Cap-X launch runs do not need :mod:`rats.rats.runtime`; importing it
    eagerly leaks optional RATS/loop dependencies into the baseline path.
    """
    from rats.rats.runtime import (
        apply_task_proposal_to_prompt,
        build_task_proposal_bundle,
        rebind_behavior_task_from_proposal,
        run_rats_episode_on_env,
    )

    return (
        apply_task_proposal_to_prompt,
        build_task_proposal_bundle,
        rebind_behavior_task_from_proposal,
        run_rats_episode_on_env,
    )

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        "-" * 100,
    ])
    return lines


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str:
    """Return the trial output directory path used for video saving."""
    return os.path.join(
        config["output_dir"],
        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}"
        f"_taskcompleted_{int(info_step.get('task_completed', False))}",
    )


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")


# ---------------------------------------------------------------------------
# Visual feedback and image differencing
# ---------------------------------------------------------------------------

def _capture_initial_visual_feedback(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: LaunchArgs,
    visual_differencing_args: ModelQueryArgs,
) -> tuple[list, list[str], str]:
    """Capture the initial environment image and optionally describe it.

    Returns:
        (visual_feedback_imgs, visual_feedback_base64_history, task_description)
    """
    visual_feedback_imgs: list = []
    visual_feedback_base64_history: list[str] = []
    task_description = ""

    use_wrist = config.get("use_wrist_camera", False)

    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
        or config.get("use_video_differencing", False)
    )
    if not (needs_visual and hasattr(env, "render")):
        return visual_feedback_imgs, visual_feedback_base64_history, task_description

    initial_base64, initial_img = _get_visual_feedback(env)
    visual_feedback_imgs.append(initial_img)
    visual_feedback_base64_history.append(initial_base64)
    task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])

    # Also capture wrist camera image for multiview initial description
    initial_wrist_base64 = None
    if use_wrist and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            initial_wrist_base64 = (
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )
            visual_feedback_imgs.append(pil_wrist)

    # Append image to the prompt for VLM visual feedback
    if config["use_visual_feedback"]:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nIncluded below is an image of the initial state of the environment."
        )
        obs["full_prompt"][-1]["content"].append(
            {"type": "image_url", "image_url": {"url": initial_base64}}
        )
        if initial_wrist_base64 is not None:
            obs["full_prompt"][-1]["content"].append(
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                }
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_wrist_base64}}
            )

    # Image differencing: ask a VLM to describe the initial scene
    if config["use_img_differencing"] or config.get("use_video_differencing", False):
        description = _describe_initial_scene(
            visual_differencing_args, task_description, initial_base64,
            wrist_image_base64=initial_wrist_base64,
        )
        feedback = f"The initial state of the environment is described as follows:\n{description}"
        obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{feedback}"
        if args.debug:
            print(description)

    return visual_feedback_imgs, visual_feedback_base64_history, task_description


def _describe_initial_scene(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    image_base64: str,
    wrist_image_base64: str | None = None,
) -> str:
    """Query a VLM to describe the initial environment state."""
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the goal of the "
                "task in mind. You should try to provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    if wrist_image_base64 is not None:
        user_content.extend([
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": wrist_image_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of the "
                "environment with the goal of the task in mind. You should try to provide "
                "objective information and no assumptions. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


def _get_visual_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    visual_feedback_base64_history: list[str],
    wrist_base64_history: list[str] | None = None,
) -> str | None:
    """Query a VLM to describe what changed between the two most recent frames.

    Args:
        wrist_base64_history: Optional history of wrist camera images.  When provided
            and has >=2 entries, the before/after wrist images are included in the prompt.
    """
    if len(visual_feedback_base64_history) < 2:
        return None

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the difference between the current state of the "
                "environment and the previous state of the environment with the "
                "goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code.."
            ),
        },
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
    ]

    if wrist_base64_history and len(wrist_base64_history) >= 2:
        user_content.extend([
            {"type": "text", "text": "Previous state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-2]}},
            {"type": "text", "text": "Current state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-1]}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between the "
                "current state of the environment and the previous state of the environment "
                "with the goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Video differencing
# ---------------------------------------------------------------------------

def _get_video_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    turn_frames: list[np.ndarray],
    wrist_turn_frames: list[np.ndarray] | None = None,
) -> str | None:
    """Query a VLM with a video of the turn execution to describe what happened.

    Args:
        visual_differencing_args: Model query args for the VDM model.
        task_description: The task goal.
        turn_frames: RGB frames from the main camera for this turn.
        wrist_turn_frames: RGB frames from the wrist camera for this turn (optional).

    Returns:
        Text description of the execution, or None if no frames.
    """
    if not turn_frames:
        return None

    video_base64 = _encode_video_base64(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "The following video shows the robot executing code in the "
                "environment from the main camera view. Describe what happened "
                "during execution, including what actions the robot took, how "
                "the objects in the scene changed, and whether the task appears "
                "to have been completed. Provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera video:"},
        {"type": "image_url", "image_url": {"url": video_base64}},
    ]

    if wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the "
                    "robot's wrist-mounted camera (eye-in-hand view), providing "
                    "a close-up perspective of the gripper and objects being "
                    "manipulated."
                ),
            },
            {"type": "text", "text": "Wrist camera video:"},
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes robot execution "
                "videos. You describe what happened during the robot's code "
                "execution, what actions were taken, how the environment "
                "changed, and whether the task appears to have been completed. "
                "Provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Initial code generation
# ---------------------------------------------------------------------------

def _query_initial_code(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
) -> tuple[str, str | None, dict | None]:
    """Query the model for the initial code generation.

    Returns:
        (raw_code, reasoning, ensemble_data)
    """
    # Save the initial prompt
    with open(os.path.join(config["output_dir"], "initial_prompt.txt"), "w") as f:
        f.write(str(obs["full_prompt"]))

    ensemble_data = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTIMODEL ENSEMBLE QUERY")
            out = _query_model_ensemble(args, obs["full_prompt"], is_multiturn=False)
        else:
            print("RUNNING SINGLE MODEL ENSEMBLE QUERY")
            out = _query_single_model_ensemble(args, obs["full_prompt"], args.model, is_multiturn=False)
        ensemble_data = {
            "ensemble_candidates_txt": out["ensemble_candidates_txt"],
            "ensemble_synthesis_txt": out["ensemble_synthesis_txt"],
        }
    else:
        out = _query_model(args, obs["full_prompt"])

    return out["content"], out["reasoning"], ensemble_data


# ---------------------------------------------------------------------------
# Multi-turn decision handling
# ---------------------------------------------------------------------------

def _handle_multi_turn_step(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    args: LaunchArgs,
    config: dict[str, Any],
    visual_differencing_args: ModelQueryArgs,
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    task_description: str,
    visual_feedback_imgs: list,
    visual_feedback_base64_history: list[str],
    stderr_history: list[str],
    turn_frames: list[np.ndarray] | None = None,
    wrist_turn_frames: list[np.ndarray] | None = None,
    wrist_base64_history: list[str] | None = None,
) -> tuple[str, str | None, str | None, dict | None, list | None]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry)
        where decision is "regenerate", "finish", or "continue".
    """
    use_wrist = config.get("use_wrist_camera", False)

    executed_code = "\n".join(code_blocks[:code_block_idx])
    # task_goal substitution: some multi_turn_prompt templates include a
    # `{task_goal}` placeholder for an "anchor" reminder (e.g. "REMINDER —
    # original task goal: {task_goal}\nDo NOT drift..."). When the template
    # uses that placeholder and we don't pass it through, .format() raises
    # KeyError('task_goal') and the trial fails before any artifact is saved.
    # Pass task_description here; .format() ignores kwargs not referenced by
    # the template, so templates that don't use {task_goal} are unaffected.
    complete_multi_turn_prompt = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=info_step["stdout"],
        console_stderr=info_step["stderr"],
        task_goal=task_description if task_description is not None else "",
    )

    if info_step["stderr"] != "":
        stderr_history.append(info_step["stderr"])

    # Capture visual feedback if applicable
    visual_feedback_base64 = None
    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
    )
    if needs_visual and hasattr(env, "render"):
        vf_base64, vf_img = _get_visual_feedback(env)
        visual_feedback_imgs.append(vf_img)
        visual_feedback_base64_history.append(vf_base64)

        # Also capture wrist camera snapshot for image-based multiview
        if use_wrist and hasattr(env, "render_wrist") and wrist_base64_history is not None:
            wrist_result = _get_visual_feedback(env, use_wrist_camera=True)
            if wrist_result[0] is not None and isinstance(wrist_result[0], list) and len(wrist_result[0]) > 1:
                wrist_base64_history.append(wrist_result[0][1])  # index 1 = wrist image

    # Determine differencing feedback
    differencing_feedback = None
    is_video_feedback = False

    if config.get("use_video_differencing") and turn_frames:
        # Video-based differencing: pass video of this turn to VDM
        differencing_feedback = _get_video_differencing_feedback(
            visual_differencing_args, task_description, turn_frames, wrist_turn_frames,
        )
        is_video_feedback = True
    elif config["use_img_differencing"] and len(visual_feedback_base64_history) >= 2:
        # Image-based differencing: pass before/after images to VDM
        differencing_feedback = _get_visual_differencing_feedback(
            visual_differencing_args, task_description, visual_feedback_base64_history,
            wrist_base64_history=wrist_base64_history,
        )

    # Only pass visual feedback to prompt if visual_feedback is enabled
    if not config["use_visual_feedback"]:
        visual_feedback_base64 = None
    elif needs_visual and hasattr(env, "render"):
        visual_feedback_base64 = visual_feedback_base64_history[-1] if visual_feedback_base64_history else None

    # Build decision prompt
    if args.use_legacy_multi_turn_decision_prompt:
        print("Using legacy multi-turn decision prompt")
        decision_prompt = _build_multi_turn_decision_prompt_legacy(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )
    else:
        decision_prompt = _build_multi_turn_decision_prompt(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )

    # Query model
    multiturn_ensemble_entry = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTITURN MULTIMODEL ENSEMBLE QUERY")
            content = _query_model_ensemble(args, decision_prompt, is_multiturn=True)
        else:
            print("RUNNING MULTITURN SINGLE MODEL ENSEMBLE QUERY")
            content = _query_single_model_ensemble(args, decision_prompt, args.model, is_multiturn=True)
        multiturn_ensemble_entry = {
            "ensemble_candidates_txt": content.get("ensemble_candidates_txt", ""),
            "ensemble_synthesis_txt": content.get("ensemble_synthesis_txt", ""),
        }
    else:
        content = _query_model(args, decision_prompt)

    reasoning = content["reasoning"]
    decision, new_code = _parse_multi_turn_decision(content["content"])

    return decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt


# ---------------------------------------------------------------------------
# Core single-trial execution
# ---------------------------------------------------------------------------

def _run_single_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
    trial_start_time = time.time()

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)

    # --- 1. Reset environment ---
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    # Reset the SIGALRM timer AFTER env.reset() so the timeout only covers
    # actual task execution, not scene loading / cuRobo JIT compilation.
    import signal
    remaining = signal.alarm(0)  # cancel current alarm
    if remaining > 0:
        signal.alarm(1000)  # restart fresh 1000s from now
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)
    _inject_seed_skill_library(env, obs, config, args)

    if config.get("rats_enabled", False) and config.get("rats_use_orchestrator", False):
        *_, run_rats_episode_on_env = _load_rats_runtime_helpers()
        rats_result = run_rats_episode_on_env(env, config, trial=trial)
        draft = rats_result["draft"]
        execution = rats_result["execution"]
        verification = rats_result["verification"]
        diagnosis = rats_result["diagnosis"]
        feedback = rats_result["feedback"]
        info_step = {
            "sandbox_rc": 0 if execution.success else 1,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "task_completed": execution.task_completed,
        }
        reward = execution.reward or 0.0
        terminated = bool(execution.artifacts.get("terminated", reward == 1.0))
        truncated = bool(execution.artifacts.get("truncated", False))
        final_code = draft.code
        log_lines = _build_log_lines(final_code, info_step, reward, terminated, truncated, 0, 1 if feedback.action == "success" else 0, 1)
        all_responses = [{
            "decision": "rats_orchestrator",
            "proposal": rats_result["proposal"].model_dump(),
            "plan": rats_result["plan"].model_dump(),
            "writer_prompt": rats_result.get("writer_prompt"),
            "quality": rats_result["quality"].model_dump(),
            "verification": verification.model_dump(),
            "diagnosis": diagnosis.model_dump(),
            "feedback": feedback.model_dump(),
        }]
        code_path = _save_trial_artifacts(
            config, trial, info_step["sandbox_rc"], reward,
            info_step.get("task_completed", False), final_code, final_code,
            all_responses, log_lines, [],
        )
        return TrialSummary(
            trial=trial,
            success=execution.success,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            sandbox_rc=info_step["sandbox_rc"],
            log="\n".join(log_lines),
            task_completed=info_step.get("task_completed", None),
            code_path=code_path,
            num_regenerations=0,
            num_finishes=1 if feedback.action == "success" else 0,
            num_code_blocks=1,
        )

    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)
    elif use_video_diff and hasattr(env, "enable_video_capture"):
        # Video differencing needs frame recording even without record_video
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

    # --- Shared trial state ---
    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    num_regenerations = 0
    num_finishes = 0
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    sandbox_rc_override = None
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []

    # Per-turn frame tracking (for video differencing and per-turn video saving)
    turn_frame_ranges: list[tuple[int, int]] = []

    # Wrist camera base64 history for image-based multiview differencing
    wrist_base64_history: list[str] | None = [] if use_wrist else None

    visual_differencing_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    if config["use_img_differencing"] or use_video_diff:
        assert visual_differencing_args.model in VLM_MODELS, (
            "Image/video differencing model must be in the list of VLM models"
        )

    if config.get("rats_enabled", False):
        (
            apply_task_proposal_to_prompt,
            build_task_proposal_bundle,
            rebind_behavior_task_from_proposal,
            _,
        ) = _load_rats_runtime_helpers()
        proposal_bundle = build_task_proposal_bundle(env, config, trial=trial)
        if proposal_bundle is not None:
            proposal = proposal_bundle["proposal"]
            rebound = rebind_behavior_task_from_proposal(env, proposal)
            if rebound:
                obs, _ = env.reset(
                    options={"trial": proposal.preferred_instance_id},
                    seed=proposal.preferred_instance_id,
                )
                obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
                _patch_libero_goal(env, obs)
            apply_task_proposal_to_prompt(obs, proposal)
            if partial_artifacts is not None:
                partial_artifacts["rats_proposal"] = proposal.model_dump()
                partial_artifacts["rats_rebound"] = rebound

    # --- 2. Capture initial visual feedback ---
    visual_feedback_imgs, visual_feedback_base64_history, task_description = (
        _capture_initial_visual_feedback(env, obs, config, args, visual_differencing_args)
    )

    # Seed wrist base64 history with initial wrist image
    if use_wrist and wrist_base64_history is not None and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            wrist_base64_history.append(
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )

    # --- 3. Initial code generation ---
    if config["use_oracle_code"]:
        raw_code = env.oracle_code
        with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
            f.write(raw_code)
        reasoning = None
        ensemble_data = None
    else:
        raw_code, reasoning, ensemble_data = _query_initial_code(args, config, obs)

    # Initialize partial artifacts for timeout recovery
    if partial_artifacts is not None:
        partial_artifacts.update({
            "raw_code": raw_code,
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": num_regenerations,
            "num_finishes": num_finishes,
            "num_code_blocks": 0,
            "ensemble_data": ensemble_data,
            "multiturn_ensemble_data": multiturn_ensemble_data,
        })

    # Parse initial code into blocks
    initial_blocks = _extract_code(raw_code)
    code_blocks.extend(initial_blocks)
    code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "decision": "initial",
        "initial_prompt": copy.deepcopy(obs["full_prompt"]),
        "reasoning": reasoning if reasoning is not None else "",
    })

    with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
        json.dump(all_responses, f)

    if args.debug:
        with open(os.path.join(config["output_dir"], "code_init.txt"), "w") as f:
            f.write("\n".join(initial_blocks))

    # --- 4. Execute code blocks (with optional multi-turn) ---
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    code_block_idx = 0
    max_multi_turns = max(1, int(config.get("multi_turn_limit", MULTITURN_LIMIT)))

    # Track whether we're recording frames (for video diff or record_video)
    recording_frames = (
        (config["record_video"] or use_video_diff)
        and hasattr(env, "get_video_frame_count")
    )

    while code_block_idx < len(code_blocks) and code_block_idx < max_multi_turns:
        code = code_blocks[code_block_idx]
        code_block_idx += 1

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        obs_next, reward, terminated, truncated, info_step = env.step(code)

        # Record frame index after step
        frame_end = env.get_video_frame_count() if recording_frames else 0
        turn_frame_ranges.append((frame_start, frame_end))

        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

        obs = obs_next

        # Multi-turn decision
        if multi_turn_prompt and code_block_idx < max_multi_turns:
            if "terminated episode" in info_step["stderr"]:
                truncated = True
                break

            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if use_video_diff and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, code_block_idx, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                turn_frames=turn_frames,
                wrist_turn_frames=wrist_turn_frames,
                wrist_base64_history=wrist_base64_history,
            )

            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision == "regenerate":
                print("Model chose to regenerate code")
                new_blocks = _extract_code(new_code)
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "decision": "regenerate",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(
                    [{"generation": num_regenerations + 1, "regenerated": True,
                      "regenerated_at_idx": code_block_idx}]
                    * len(new_blocks)
                )
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations

            elif decision == "finish":
                all_responses.append({
                    "decision": "finish",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                })
                print("Model chose to finish")
                num_finishes += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_finishes"] = num_finishes
                break

        print(f"Code block {code_block_idx} done")
        print(f"Number of code blocks: {len(code_blocks)}")

        # Save intermediate artifacts (code, logs) per code block
        final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
        _save_trial_artifacts(
            config, trial, info_step["sandbox_rc"], reward,
            info_step.get("task_completed", False), final_code, raw_code,
            all_responses, ["-" * 100, "Generated program:", final_code],
            visual_feedback_imgs,
        )

        # Only save intermediate video if NOT doing per-turn saving
        # (per-turn saving is deferred to after the loop to avoid clearing the buffer)
        if not recording_frames:
            _save_trial_video(
                env, config, trial, info_step, reward, len(code_blocks),
                suffix_extra=str(len(code_blocks)),
            )

    print("Code blocks done")

    # --- 5. Build final summary ---
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
    num_code_blocks = len(code_blocks)

    if partial_artifacts is not None:
        partial_artifacts["final_code"] = final_code
        partial_artifacts["num_code_blocks"] = num_code_blocks

    # Override sandbox_rc for terminated-episode stderr
    if "executing action in terminated episode" in info_step["stderr"]:
        sandbox_rc_override = 0
    if sandbox_rc_override is not None:
        info_step["sandbox_rc"] = sandbox_rc_override

    stderr = "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        stderr_override=stderr,
    )

    code_path = _save_trial_artifacts(
        config, trial, info_step["sandbox_rc"], reward,
        info_step.get("task_completed", False), final_code, raw_code,
        all_responses, log_lines, visual_feedback_imgs,
        ensemble_data=ensemble_data,
        multiturn_ensemble_data=multiturn_ensemble_data,
    )

    # Save per-turn and combined videos
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(
            env, config, trial, info_step, reward, turn_frame_ranges,
        )
    else:
        _save_trial_video(env, config, trial, info_step, reward, num_code_blocks)

    success = info_step["sandbox_rc"] == 0

    # --- Evolving skill library integration (opt-in) ---
    if config.get("evolve_skill_library", False) and info_step.get("task_completed", False):
        try:
            from rats.skills import SkillLibrary

            skill_lib_path = config.get("skill_library_path", None)
            skill_lib = SkillLibrary(path=skill_lib_path)
            task_name = config.get("task_name", f"trial_{trial}")
            new_skills = skill_lib.extract_from_code(final_code, task_name=task_name)
            skill_lib.save()
            if new_skills:
                print(f"[SkillLibrary] Extracted {len(new_skills)} new skill(s): {new_skills}")
        except Exception as exc:
            print(f"[SkillLibrary] Skill extraction failed: {exc}")

    print(f"Trial {trial} took {time.time() - trial_start_time:.2f} seconds")

    gc.collect()

    return TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )


def _select_relevant_seed_skills(
    candidate_skills: list[dict[str, Any]],
    full_prompt_text: str,
    config: dict[str, Any],
    args: "LaunchArgs",
) -> list[dict[str, Any]]:
    """LLM-based filter: keep only skills directly useful for the current task.

    Mirrors the planner-level skill selection that RATS does
    (``agents/planner.py:Planner.plan`` returns ``relevant_skills`` per
    step). For the CaP-X seed-library path we don't decompose the task
    into steps — we just ask the LLM which subset of the seeded library
    is worth showing to the policy writer. Dumping all 24+ helpers in
    the prompt has been observed to hurt sandbox success because the
    policy writer wastes attention on irrelevant signatures.

    Returns the original list verbatim on any failure (parse error,
    empty selection, LLM error) so a flaky selector never silently
    deletes the library.
    """
    if not candidate_skills:
        return []
    model = (
        config.get("seed_skill_library_select_model")
        or getattr(args, "model", None)
        or "openai/gpt-5.5"
    )
    # Compact per-skill descriptor: name + signature + first doc line.
    listing_lines: list[str] = []
    for s in candidate_skills:
        doc = (s.get("description") or "").strip()
        first_doc = doc.splitlines()[0][:160] if doc else ""
        listing_lines.append(f"- {s['name']}{s['signature']}  -- {first_doc}")
    skill_listing = "\n".join(listing_lines)

    # Use only the tail of the prompt text so the selector sees the
    # actual task language (which lives at the end of full_prompt) but
    # we don't burn tokens echoing the giant API-doc block. 4000 chars
    # is enough for any MolmoSpaces task prompt with margin.
    tail = full_prompt_text[-4000:] if len(full_prompt_text) > 4000 else full_prompt_text

    # Default top-K = 6 matches the v2 planner_max_selected convention
    # in capx-baseline. Without an explicit cap the selector prompt used
    # soft language ("pick only directly useful ones"), and selectors
    # (Claude / Gemini) read that as "be very conservative" — observed
    # result: 2-3 skills kept out of 24-45, policy writer underequipped.
    # An explicit top_k pins both the prompt phrasing (exactly N) AND
    # the post-hoc hard cap. Cap at the candidate count so we never ask
    # for more skills than exist.
    top_k = config.get("seed_skill_library_select_top_k")
    if top_k is None:
        top_k = 6
    top_k_int = min(int(top_k), len(candidate_skills))
    selector_user_prompt = (
        "You are a skill planner. The policy writer LLM will receive a "
        "list of learned skill helpers along with the task. "
        f"Pick the {top_k_int} MOST RELEVANT skills (no more, no fewer).\n\n"
        "TASK CONTEXT (verbatim tail of the policy prompt):\n"
        "-----\n"
        f"{tail}\n"
        "-----\n\n"
        "CANDIDATE LEARNED SKILLS:\n"
        f"{skill_listing}\n\n"
        "Respond with JSON only, no commentary, in the form:\n"
        '  {"selected": ["skill_name_1", "skill_name_2", ...]}\n'
        f"The `selected` list MUST have exactly {top_k_int} entries.\n"
        "Use skill names exactly as listed. If unsure, prefer to "
        "include rather than exclude."
    )
    sel_args = ModelQueryArgs(
        model=model,
        server_url=getattr(args, "server_url", "http://127.0.0.1:8110/chat/completions"),
        api_key=getattr(args, "api_key", None),
        temperature=float(config.get("seed_skill_library_select_temperature", 0.0)),
        max_tokens=int(config.get("seed_skill_library_select_max_tokens", 32768)),
        reasoning_effort=str(config.get("seed_skill_library_select_reasoning_effort", "low")),
        debug=bool(getattr(args, "debug", False)),
    )
    prompt_msg = [{"role": "user", "content": [{"type": "text", "text": selector_user_prompt}]}]
    # Persist selector I/O to a per-call JSONL so we can audit the raw LLM
    # response when the kept count looks wrong (e.g. Gemini under-delivering
    # vs top_k). Path comes from config; falls back to stderr-only logging.
    io_log_path = config.get("seed_skill_library_select_io_log")
    try:
        result = _query_model(sel_args, prompt_msg)
        content = result["content"] if isinstance(result, dict) else str(result)
    except Exception as exc:
        print(f"[seed_skill_library] selector LLM call failed ({exc}); falling back to all skills")
        return candidate_skills
    if io_log_path:
        try:
            from datetime import datetime
            Path(io_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(io_log_path, "a") as f:
                f.write(json.dumps({
                    "ts": datetime.utcnow().isoformat(),
                    "model": model,
                    "top_k": top_k_int,
                    "n_candidates": len(candidate_skills),
                    "prompt": selector_user_prompt,
                    "response": content,
                }) + "\n")
        except Exception as exc:
            print(f"[seed_skill_library] failed to write io log {io_log_path}: {exc}")
    # Be lenient about extra text. Gemini-3.1-pro consistently wraps
    # JSON in ``` ```json ... ``` ``` fences, and sometimes the closing
    # fence is missing when the response gets truncated. Try in order:
    #   (1) strip leading fence + optional language tag
    #   (2) strict json.loads
    #   (3) balanced-brace scan around "selected"
    #   (4) regex-extract array literal after "selected"
    selected_names: list[str] | None = None
    stripped = content.strip()
    # Drop an opening ``` (with optional language tag) and a trailing ```.
    stripped = re.sub(r"^```[a-zA-Z]*\s*\n", "", stripped)
    stripped = re.sub(r"\n```\s*$", "", stripped)
    stripped = stripped.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and isinstance(parsed.get("selected"), list):
            selected_names = [str(n) for n in parsed["selected"]]
    except Exception:
        # (3) balanced-brace scan around "selected"
        idx = stripped.find('"selected"')
        if idx >= 0:
            start = stripped.rfind("{", 0, idx)
            if start >= 0:
                depth = 0
                for i in range(start, len(stripped)):
                    if stripped[i] == "{":
                        depth += 1
                    elif stripped[i] == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                parsed = json.loads(stripped[start:i + 1])
                                selected_names = [str(n) for n in parsed.get("selected", [])]
                            except Exception:
                                selected_names = None
                            break
    if not selected_names:
        # (4) Last-ditch: pull out the array literal even if the surrounding
        # object braces are missing/truncated. e.g. "selected": ["a", "b", ...
        arr = re.search(
            r'"selected"\s*:\s*\[\s*((?:"[^"]+"\s*,?\s*)+)',
            stripped,
            re.DOTALL,
        )
        if arr:
            try:
                selected_names = re.findall(r'"([^"]+)"', arr.group(1))
            except Exception:
                selected_names = None
    if not selected_names:
        print(f"[seed_skill_library] selector returned unparseable output, using all skills. Raw head: {content[:300]!r}")
        return candidate_skills
    name_set = set(selected_names)
    # Preserve the LLM's selected order (top-K cares about ranking, not just
    # set membership) by iterating selected_names rather than candidate_skills.
    by_name = {s["name"]: s for s in candidate_skills}
    ordered_filtered = [by_name[n] for n in selected_names if n in by_name]
    # Enforce the configured top-K hard cap — LLM occasionally overshoots.
    if len(ordered_filtered) > top_k_int:
        ordered_filtered = ordered_filtered[:top_k_int]
    # Debug: when the LLM under-delivers (returns far fewer than top_k),
    # log the raw response so we can see why. Useful for the runE batch
    # where the selector started keeping 1/47 instead of 6/47 — same
    # parser, same prompt, but production calls returned tiny selections.
    if len(ordered_filtered) < max(1, top_k_int // 2):
        head = content[:600].replace("\n", " | ")
        print(f"[seed_skill_library] DEBUG: selector under-delivered "
              f"({len(ordered_filtered)} vs requested {top_k_int}). "
              f"Raw head: {head!r}")
    if not ordered_filtered:
        print(f"[seed_skill_library] selector picked nothing matching catalog (asked for {selected_names[:6]}…), using all skills")
        return candidate_skills
    kept_names = {s["name"] for s in ordered_filtered}
    dropped = [s["name"] for s in candidate_skills if s["name"] not in kept_names]
    print(
        f"[seed_skill_library] selector kept {len(ordered_filtered)}/{len(candidate_skills)} skills "
        f"(model={model}, top_k={top_k_int}); dropped: {dropped}"
    )
    return ordered_filtered


def _inject_seed_skill_library(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: "LaunchArgs | None" = None,
) -> None:
    """Load a RATS-format ``skills.json`` and seed it as extra primitives.

    When ``config['seed_skill_library_path']`` is set, this:
    1. reads the RATS skills.json (or CaP-X SkillLibrary shape)
    2. filters to learned (non-primitive) skills
    3. (optional) if ``config['seed_skill_library_select']`` is true,
       runs an LLM-based selector that drops skills unrelated to the
       current task — mirrors the planner-level skill selection RATS
       does. Requires ``args`` so we can reuse the trial's LLM client.
    4. appends each helper's signature + description to the existing
       primitives list in the prompt, matching ``ApiBase.combined_doc``'s
       format so they look like additional primitive entries
    5. ``exec()`` each helper's code into the env's exec globals so
       generated policy code can invoke them by name

    Designed to make a CaP-X baseline run on equal footing with a
    RATS run whose library was populated by playtime: same set of
    helpers available, but no other RATS machinery (planner,
    verifier-driven retries, failure-memory retrieval).

    Failures along this chain log a warning and otherwise no-op so a
    misconfigured ``seed_skill_library_path`` doesn't crash the run.
    """
    path = config.get("seed_skill_library_path")
    if not path:
        return
    p = Path(path)
    if not p.exists():
        print(f"[seed_skill_library] path does not exist: {path}; skipping")
        return
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        print(f"[seed_skill_library] failed to parse {path}: {exc}; skipping")
        return
    # Tolerate two on-disk shapes:
    #   - RATS shape: list[dict]
    #   - CaP-X SkillLibrary shape: {"skills": {name: {...}}}
    if isinstance(data, dict) and "skills" in data:
        records = list(data["skills"].values())
    elif isinstance(data, list):
        records = data
    else:
        print(f"[seed_skill_library] unexpected shape in {path}; skipping")
        return
    learned: list[dict[str, Any]] = []
    for s in records:
        if not isinstance(s, dict):
            continue
        if s.get("is_primitive"):
            continue
        code = (s.get("code") or s.get("source_code") or "").strip()
        if not code or not code.startswith("def "):
            continue
        # Extract the function signature from the ``def`` line so the
        # entry visually matches combined_doc's ``name(sig)`` format.
        first = code.splitlines()[0].strip()  # "def foo(arg, ...):"
        sig = "(…)"
        try:
            after_def = first[len("def "):].rstrip(":")
            open_paren = after_def.index("(")
            close_paren = after_def.rindex(")")
            sig = after_def[open_paren:close_paren + 1]
        except (ValueError, IndexError):
            pass
        learned.append({
            "name": s.get("name", "") or first[len("def "):].split("(", 1)[0],
            "signature": sig,
            "description": s.get("description", ""),
            "code": code,
        })
    if not learned:
        return

    # Optional planner-style filter: cut the seed library down to the
    # subset relevant for the current task before injecting. Opt-in via
    # config; requires ``args`` so the selector can reuse the trial's
    # LLM client. Failures fall back to the full library.
    if config.get("seed_skill_library_select") and args is not None:
        full_prompt_text = obs["full_prompt"][-1]["content"][0]["text"]
        learned = _select_relevant_seed_skills(learned, full_prompt_text, config, args)
        if not learned:
            return

    # 1. Extend the existing API/primitives section in the prompt.
    # ``_get_complete_prompt`` writes "...\nAPIs:\n<combined_doc>"; we
    # append our entries to the end so they appear inline with the
    # existing primitives. Format extends ApiBase.combined_doc with a
    # full Source block since seeded helpers are LEARNED Python — we have
    # their bodies and exposing them lets the LLM verify assumptions,
    # adapt patterns, and inline-tweak rather than treat them as opaque.
    #     name(signature)
    #       Doc:
    #         <docstring lines>
    #       Source:
    #         <full code body>
    extras: list[str] = []
    for s in learned:
        extras.append(f"{s['name']}{s['signature']}")
        if s["description"]:
            extras.append("  Doc:")
            extras.extend(f"    {ln}" for ln in s["description"].splitlines())
        if s["code"]:
            extras.append("  Source:")
            extras.extend(f"    {ln}" for ln in s["code"].splitlines())
        extras.append("")
    obs["full_prompt"][-1]["content"][0]["text"] += "\n" + "\n".join(extras).rstrip() + "\n"

    # 2. (optional) exec into the env's namespace so policy code can
    # call the seeded helpers by name without re-defining them. This is
    # how the original seed_skill_library worked, but it diverges from
    # capx-baseline's v2 planner mode (which only shows the source
    # block as documentation and lets the LLM inline-borrow or call
    # whatever it wants — the body is NOT pre-bound). To match v2
    # planner exactly, set seed_skill_library_inject_namespace: false.
    # Default stays True to preserve existing behavior for older yamls.
    if not bool(config.get("seed_skill_library_inject_namespace", True)):
        print("[seed_skill_library] namespace injection disabled (v2-planner-equivalent mode); docs only")
        return
    low_level = getattr(env, "low_level_env", None)
    exec_globals = getattr(env, "_exec_globals", None)
    if exec_globals is None and low_level is not None:
        exec_globals = getattr(low_level, "_exec_globals", None)
    if not isinstance(exec_globals, dict):
        print("[seed_skill_library] env has no _exec_globals attr; docs injected but namespace not seeded")
        return
    for s in learned:
        try:
            exec(s["code"], exec_globals, exec_globals)  # noqa: S102
        except Exception as exc:
            print(f"[seed_skill_library] failed to define {s['name']}: {exc}")

    print(f"[seed_skill_library] seeded {len(learned)} helper(s) from {path}")

    # Inject LIBERO-compatibility shims (hidden from the LLM prompt).
    # These let LIBERO-trained skills call get_object_3d_points_and_masks_from_language,
    # get_object_pose, and sample_grasp_pose without modification, by decomposing
    # them into MolmoSpaces primitives already present in exec_globals.
    _inject_libero_compat_shims(exec_globals)


_LIBERO_COMPAT_SHIMS = '''
import numpy as np

def get_object_3d_points_and_masks_from_language(text_prompt, use_multiview=True):
    """LIBERO-compat shim: segment object by text and return 3D points + masks."""
    obs = get_observation()
    cam_name = "agentview"
    rgb = obs[cam_name]["images"]["rgb"]
    depth = obs[cam_name]["images"]["depth"]
    intrinsics = obs[cam_name]["intrinsics"]
    extrinsics = obs[cam_name]["pose_mat"]

    molmo_result = point_prompt_molmo(rgb, text_prompt)
    pixel = molmo_result.get(text_prompt, (None, None))
    if pixel[0] is None or pixel[1] is None:
        return {"agentview_mask": None, "wrist_mask": None, "points_3d": np.zeros((0, 3)), "agentview_score": 0.0, "wrist_score": None}

    masks = segment_sam3_point_prompt(rgb, (float(pixel[0]), float(pixel[1])))
    if not masks:
        return {"agentview_mask": None, "wrist_mask": None, "points_3d": np.zeros((0, 3)), "agentview_score": 0.0, "wrist_score": None}

    best = max(masks, key=lambda m: m.get("score", 0.0))
    mask = best["mask"]
    score = best.get("score", 0.0)
    points_3d = mask_to_world_points(mask, depth, intrinsics, extrinsics)

    return {"agentview_mask": mask, "wrist_mask": None, "points_3d": points_3d, "agentview_score": score, "wrist_score": None}


def get_object_pose(object_name, use_multiview=True):
    """LIBERO-compat shim: get object center + orientation via segmentation + OBB."""
    result = get_object_3d_points_and_masks_from_language(object_name, use_multiview=use_multiview)
    points_3d = result["points_3d"]
    if len(points_3d) == 0:
        return None, None
    obb = get_oriented_bounding_box_from_3d_points(points_3d)
    position = np.array(obb["center"])
    R = np.array(obb["R"])
    quaternion_wxyz = rotation_matrix_to_quaternion(R)
    return position, quaternion_wxyz


def sample_grasp_pose(object_name, use_multiview=True):
    """LIBERO-compat shim: plan a grasp for a named object, return (position, quaternion_wxyz)."""
    result = get_object_3d_points_and_masks_from_language(object_name, use_multiview=use_multiview)
    pc_segment = result["points_3d"]
    if len(pc_segment) == 0:
        raise ValueError(f"Could not segment object '{object_name}'")

    obs = get_observation()
    depth = obs["agentview"]["images"]["depth"]
    intrinsics = obs["agentview"]["intrinsics"]
    extrinsics = obs["agentview"]["pose_mat"]
    pc_cam = depth_to_point_cloud(depth, intrinsics)
    ones = np.ones((len(pc_cam), 1))
    pc_full = (extrinsics @ np.concatenate([pc_cam, ones], axis=1).T).T[:, :3]
    pc_full = subsample_point_cloud(pc_full, max_points=20000)
    pc_segment = subsample_point_cloud(pc_segment, max_points=10000)

    grasp_tfs, grasp_scores = plan_grasp_from_point_clouds(pc_full, pc_segment, label=object_name)
    selected, score = select_top_down_grasp(grasp_tfs, grasp_scores)
    if selected is None:
        best_idx = np.argmax(grasp_scores)
        selected = grasp_tfs[best_idx]
    position, quaternion_wxyz = decompose_transform(selected)
    return position, quaternion_wxyz
'''


def _inject_libero_compat_shims(exec_globals: dict) -> None:
    """Exec LIBERO-compatibility shims into the env namespace (hidden from prompt)."""
    shim_names = ("get_object_3d_points_and_masks_from_language", "get_object_pose", "sample_grasp_pose")
    already = all(name in exec_globals for name in shim_names)
    if already:
        return
    try:
        exec(_LIBERO_COMPAT_SHIMS, exec_globals, exec_globals)  # noqa: S102
        print(f"[seed_skill_library] injected LIBERO-compat shims: {', '.join(shim_names)}")
    except Exception as exc:
        print(f"[seed_skill_library] failed to inject LIBERO-compat shims: {exc}")


def _patch_libero_goal(env: CodeExecutionEnvBase, obs: dict[str, Any]) -> None:
    """Inject the LIBERO task language into the prompt template if applicable."""
    low_level_env = getattr(env, "low_level_env", None)
    if low_level_env is None or not hasattr(low_level_env, "handle"):
        return
    handle = low_level_env.handle
    if (
        hasattr(handle, "task_language")
        and "libero_environment_goal" in obs["full_prompt"][-1]["content"][0]["text"]
    ):
        goal = getattr(handle, "task_language")
        obs["full_prompt"][-1]["content"][0]["text"] = (
            obs["full_prompt"][-1]["content"][0]["text"].format(
                libero_environment_goal=goal
            )
        )
