from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rats.llm.client import ModelQueryArgs
from rats.rats.registry import build_default_registry
from rats.rats.schemas import BehaviorSceneContext, ExecutionRecord, FeedbackAction, TaskProposal
from rats.loop.libero_utils import detect_env_type


def _normalize_nl_field(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def _assert_behavior_runtime_supported(env: Any) -> None:
    env_type = detect_env_type(env)
    if env_type != "behavior":
        raise ValueError(
            f"Embedded RATS runtime only supports behavior envs today; got env_type={env_type}. "
            "Use the standalone scripts/run_rats.py MolmoSpaces/LIBERO path instead."
        )


def extract_behavior_scene_context(
    env: Any,
    *,
    fixed_scene_model: str | None = None,
    activity_instance_id: int | None = None,
) -> BehaviorSceneContext:
    low_level_env = getattr(env, "low_level_env", env)
    og_env = getattr(low_level_env, "env", None)
    task = getattr(og_env, "task", None)

    scene_model = (
        fixed_scene_model
        or getattr(task, "scene_name", None)
        or getattr(getattr(og_env, "scene", None), "scene_model", None)
        or getattr(low_level_env, "task_name", None)
        or "unknown_scene"
    )
    object_scope = {}
    raw_scope = getattr(task, "object_scope", None) or getattr(low_level_env, "task_relevant_obj", None) or {}
    for key, value in raw_scope.items():
        synset = getattr(value, "synset", None)
        if synset is None and isinstance(value, str):
            synset = value
        object_scope[str(key)] = synset or str(key)

    apis = getattr(env, "_apis", {})
    api_docs = []
    available_functions: list[str] = []
    for api in apis.values():
        if hasattr(api, "combined_doc"):
            api_docs.append(api.combined_doc())
        if hasattr(api, "functions"):
            available_functions.extend(list(api.functions().keys()))

    task_metadata = {}
    scene = getattr(og_env, "scene", None)
    if scene is not None and hasattr(scene, "get_task_metadata"):
        for key in ("inst_to_name", "robot_poses"):
            value = scene.get_task_metadata(key)
            if value is not None:
                task_metadata[key] = value

    return BehaviorSceneContext(
        scene_model=scene_model,
        # Prefer low_level_env.task_name — lightweight rebinding updates this
        # without touching OmniGibson's internal task.activity_name.
        activity_name=getattr(low_level_env, "task_name", None) or getattr(task, "activity_name", None),
        activity_definition_id=getattr(task, "activity_definition_id", None),
        activity_instance_id=activity_instance_id,
        object_scope=object_scope,
        initial_conditions_nl=_normalize_nl_field(getattr(task, "activity_natural_language_initial_conditions", None)),
        goal_conditions_nl=_normalize_nl_field(getattr(task, "activity_natural_language_goal_conditions", None)),
        task_prompt=getattr(env, "_task_prompt", None),
        api_docs="\n\n".join(doc for doc in api_docs if doc),
        available_functions=sorted(set(available_functions)),
        task_metadata=task_metadata,
    )




def dump_rats_writer_artifacts(config: dict[str, Any], trial: int, *, proposal: TaskProposal, plan, writer_prompt, draft, quality) -> None:
    output_dir = config.get("output_dir")
    if not output_dir:
        return
    trial_dir = Path(output_dir) / f"rats_writer_debug_trial_{trial:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "proposal.json").write_text(json.dumps(proposal.model_dump(), indent=2))
    plan_dump = plan.model_dump() if hasattr(plan, "model_dump") else plan
    (trial_dir / "plan.json").write_text(json.dumps(plan_dump, indent=2))
    (trial_dir / "writer_prompt.json").write_text(json.dumps(writer_prompt, indent=2))
    (trial_dir / "draft.py").write_text(draft.code)
    quality_dump = quality.model_dump() if hasattr(quality, "model_dump") else quality
    (trial_dir / "quality.json").write_text(json.dumps(quality_dump, indent=2))


def build_registry_from_runtime_config(config: dict[str, Any]):
    if config.get("rats_enabled", False):
        model_args = ModelQueryArgs(
            model=config.get("rats_writer_model") or config.get("model") or "google/gemini-3.1-pro-preview",
            server_url=config.get("rats_writer_server_url") or config.get("server_url") or "http://127.0.0.1:8110/chat/completions",
            api_key=config.get("rats_writer_api_key"),
            temperature=float(config.get("rats_writer_temperature", 0.2)),
            max_tokens=int(config.get("rats_writer_max_tokens", 4096)),
            reasoning_effort=config.get("rats_writer_reasoning_effort", "medium"),
            debug=bool(config.get("debug", False)),
        )
        diagnoser_args = ModelQueryArgs(
            model=config.get("rats_diagnoser_model") or config.get("visual_differencing_model") or model_args.model,
            server_url=config.get("rats_diagnoser_server_url") or config.get("visual_differencing_model_server_url") or model_args.server_url,
            api_key=config.get("rats_diagnoser_api_key") or config.get("visual_differencing_model_api_key"),
            temperature=float(config.get("rats_diagnoser_temperature", 0.2)),
            max_tokens=int(config.get("rats_diagnoser_max_tokens", 1024)),
            reasoning_effort=config.get("rats_diagnoser_reasoning_effort", model_args.reasoning_effort),
            debug=bool(config.get("debug", False)),
        )
        return build_default_registry(
            fixed_scene_model=config.get("rats_scene_model"),
            model_query_args=model_args,
            diagnoser_model_query_args=diagnoser_args,
        )
    return build_default_registry(fixed_scene_model=config.get("rats_scene_model"))


def build_task_proposal_bundle(env: Any, config: dict[str, Any], *, trial: int | None = None) -> dict[str, Any] | None:
    if not config.get("rats_enabled", False):
        return None

    _assert_behavior_runtime_supported(env)
    registry = build_registry_from_runtime_config(config)
    scene_context = extract_behavior_scene_context(
        env,
        fixed_scene_model=config.get("rats_scene_model"),
        activity_instance_id=trial,
    )
    skill_summary = registry.skill_library.summarize()
    proposal = registry.task_proposer.propose(scene_context, skill_summary)
    return {
        "scene_context": scene_context,
        "skill_summary": skill_summary,
        "proposal": proposal,
    }


def apply_task_proposal_to_prompt(obs: dict[str, Any], proposal: TaskProposal) -> None:
    prompt = obs["full_prompt"][-1]["content"][0]["text"]
    proposal_json = json.dumps(proposal.model_dump(), indent=2)
    obs["full_prompt"][-1]["content"][0]["text"] = (
        f"{prompt}\n\nRATS task proposal metadata:\n```json\n{proposal_json}\n```"
    )


def rebind_behavior_task_from_proposal(env: Any, proposal: TaskProposal) -> bool:
    _assert_behavior_runtime_supported(env)
    low_level_env = getattr(env, "low_level_env", env)
    if not hasattr(low_level_env, "configure_behavior_task"):
        return False
    activity_name = getattr(low_level_env, "task_name", None)
    activity_definition_id = getattr(getattr(getattr(low_level_env, "env", None), "task", None), "activity_definition_id", None)
    if activity_name == proposal.activity_name and activity_definition_id == proposal.activity_definition_id:
        return False
    low_level_env.configure_behavior_task(
        activity_name=proposal.activity_name,
        activity_definition_id=proposal.activity_definition_id,
    )
    return True


def run_rats_episode_on_env(env: Any, config: dict[str, Any], *, trial: int | None = None) -> dict[str, Any]:
    _assert_behavior_runtime_supported(env)
    registry = build_registry_from_runtime_config(config)
    scene_context = extract_behavior_scene_context(
        env,
        fixed_scene_model=config.get("rats_scene_model"),
        activity_instance_id=trial,
    )
    skill_summary = registry.skill_library.summarize()
    proposal = registry.task_proposer.propose(scene_context, skill_summary)
    rebound = rebind_behavior_task_from_proposal(env, proposal)
    if rebound and hasattr(env, "reset"):
        env.reset(options={"trial": proposal.preferred_instance_id}, seed=proposal.preferred_instance_id)
        scene_context = extract_behavior_scene_context(
            env,
            fixed_scene_model=config.get("rats_scene_model"),
            activity_instance_id=proposal.preferred_instance_id,
        )

    plan = registry.planner.plan(proposal, skill_summary)
    writer_prompt = registry.policy_writer.build_prompt(plan, scene_context)
    draft = registry.policy_writer.write(plan, scene_context)
    quality = registry.policy_quality_checker.check(draft)
    dump_rats_writer_artifacts(config, trial or 0, proposal=proposal, plan=plan, writer_prompt=writer_prompt, draft=draft, quality=quality)

    if config.get("rats_stop_after_writer", False):
        execution = ExecutionRecord(
            success=False,
            stdout="writer_only_mode",
            stderr="",
            reward=0.0,
            task_completed=False,
            artifacts={"writer_only": True},
        )
        verification = registry.verifier.verify(execution, proposal)
        diagnosis = registry.failure_diagnoser.diagnose(execution, scene_context)
        feedback = FeedbackAction(action="failure", message="Stopped after writer for inspection.")
        return {
            "scene_context": scene_context,
            "proposal": proposal,
            "plan": plan,
            "writer_prompt": writer_prompt,
            "draft": draft,
            "quality": quality,
            "execution": execution,
            "verification": verification,
            "diagnosis": diagnosis,
            "feedback": feedback,
            "rebound": rebound,
        }

    if not quality.approved:
        execution = ExecutionRecord(
            success=False,
            stdout="",
            stderr=quality.feedback,
            reward=None,
            task_completed=False,
            artifacts={},
        )
        verification = registry.verifier.verify(execution, proposal)
        diagnosis = registry.failure_diagnoser.diagnose(execution, scene_context)
        feedback = registry.feedback_generator.generate(execution, verification, diagnosis)
        return {
            "scene_context": scene_context,
            "proposal": proposal,
            "plan": plan,
            "writer_prompt": writer_prompt,
            "draft": draft,
            "quality": quality,
            "execution": execution,
            "verification": verification,
            "diagnosis": diagnosis,
            "feedback": feedback,
            "rebound": rebound,
        }

    execution = registry.executor.execute(draft, env, scene_context)
    verification = registry.verifier.verify(execution, proposal)
    diagnosis = registry.failure_diagnoser.diagnose(execution, scene_context)
    feedback = registry.feedback_generator.generate(execution, verification, diagnosis)
    return {
        "scene_context": scene_context,
        "proposal": proposal,
        "plan": plan,
        "writer_prompt": writer_prompt,
        "draft": draft,
        "quality": quality,
        "execution": execution,
        "verification": verification,
        "diagnosis": diagnosis,
        "feedback": feedback,
        "rebound": rebound,
    }
