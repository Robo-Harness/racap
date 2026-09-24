from __future__ import annotations

from dataclasses import dataclass

from rats.llm.client import ModelQueryArgs
from rats.rats.backends import make_diagnoser_query_backend, make_policy_query_backend
from rats.rats.environment_creator import EnvironmentCreator
from rats.rats.executor_agent import ExecutorAgent
from rats.rats.failure_diagnoser import FailureDiagnoser
from rats.rats.feedback_generator import FeedbackGenerator
from rats.rats.planner_agent import PlannerAgent
from rats.rats.policy_quality_checker import PolicyQualityChecker
from rats.rats.policy_writer_agent import PolicyWriterAgent
from rats.rats.skill_library_agent import SkillLibraryAgent
from rats.rats.task_proposer import TaskProposer
from rats.rats.verifier_agent import VerifierAgent


@dataclass(slots=True)
class RatsRegistry:
    environment_creator: EnvironmentCreator
    task_proposer: TaskProposer
    skill_library: SkillLibraryAgent
    planner: PlannerAgent
    policy_writer: PolicyWriterAgent
    policy_quality_checker: PolicyQualityChecker
    executor: ExecutorAgent
    verifier: VerifierAgent
    failure_diagnoser: FailureDiagnoser
    feedback_generator: FeedbackGenerator


def build_default_registry(
    *,
    fixed_scene_model: str | None = None,
    policy_query_backend=None,
    diagnoser_query_backend=None,
    model_query_args: ModelQueryArgs | None = None,
    diagnoser_model_query_args: ModelQueryArgs | None = None,
) -> RatsRegistry:
    if policy_query_backend is None and model_query_args is not None:
        policy_query_backend = make_policy_query_backend(model_query_args)
    if diagnoser_query_backend is None and diagnoser_model_query_args is not None:
        diagnoser_query_backend = make_diagnoser_query_backend(diagnoser_model_query_args)
    return RatsRegistry(
        environment_creator=EnvironmentCreator(fixed_scene_model=fixed_scene_model),
        task_proposer=TaskProposer(),
        skill_library=SkillLibraryAgent(),
        planner=PlannerAgent(),
        policy_writer=PolicyWriterAgent(query_backend=policy_query_backend),
        policy_quality_checker=PolicyQualityChecker(),
        executor=ExecutorAgent(),
        verifier=VerifierAgent(),
        failure_diagnoser=FailureDiagnoser(query_backend=diagnoser_query_backend),
        feedback_generator=FeedbackGenerator(),
    )
