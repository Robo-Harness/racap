from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BehaviorSceneContext(BaseModel):
    scene_model: str
    activity_name: str | None = None
    activity_definition_id: int | None = None
    activity_instance_id: int | None = None
    object_scope: dict[str, str] = Field(default_factory=dict)
    initial_conditions_nl: str | None = None
    goal_conditions_nl: str | None = None
    task_prompt: str | None = None
    api_docs: str | None = None
    available_functions: list[str] = Field(default_factory=list)
    task_metadata: dict[str, Any] = Field(default_factory=dict)
    rgb: Any | None = None


class TaskSummary(BaseModel):
    natural_language_goal: str
    object_scope_synsets: list[str] = Field(default_factory=list)
    initial_condition_summary: list[str] = Field(default_factory=list)
    goal_condition_summary: list[str] = Field(default_factory=list)


class SceneEvidence(BaseModel):
    visible_objects: list[str] = Field(default_factory=list)
    room_type_hypothesis: list[str] = Field(default_factory=list)
    compatibility_score: float = 0.0
    reasoning: str = ""


class LearningMetadata(BaseModel):
    required_primitives: list[str] = Field(default_factory=list)
    novelty_score: float = 0.0
    learnability_score: float = 0.0
    failure_risk: list[str] = Field(default_factory=list)


class TaskProposal(BaseModel):
    scene_model: str
    proposal_mode: str
    activity_name: str
    activity_definition_id: int = 0
    candidate_instance_ids: list[int] = Field(default_factory=list)
    preferred_instance_id: int = 0
    task_summary: TaskSummary
    scene_evidence: SceneEvidence
    learning_metadata: LearningMetadata


class SkillSummary(BaseModel):
    total_skills: int = 0
    promoted_skills: list[str] = Field(default_factory=list)
    docs: str = ""


class PlanStep(BaseModel):
    id: str
    description: str
    relevant_skills: list[str] = Field(default_factory=list)
    notes: str = ""


class PlanBundle(BaseModel):
    task_id: str
    retrieval_mode: Literal["precise", "diverse"] = "precise"
    steps: list[PlanStep] = Field(default_factory=list)


class PolicyDraft(BaseModel):
    code: str
    reasoning: str = ""
    skills_used: list[str] = Field(default_factory=list)


class QualityCheckResult(BaseModel):
    approved: bool
    feedback: str = ""


class ExecutionRecord(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    reward: float | None = None
    task_completed: bool | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    success: bool
    satisfied_conditions: list[str] = Field(default_factory=list)
    unsatisfied_conditions: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class DiagnosisResult(BaseModel):
    visual_success: bool = False
    failed_step: str | None = None
    failure_reason: str = ""
    policy_feedback: str = ""
    confidence: float = 0.0


class FeedbackAction(BaseModel):
    action: Literal["retry", "success", "failure", "noop"] = "noop"
    message: str = ""
    next_task_signal: bool = False
    new_skills: list[str] = Field(default_factory=list)
