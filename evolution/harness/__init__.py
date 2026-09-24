"""Machine-enforced, evidence-first evolution harness for RACaP."""

from .curriculum import Curriculum, default_curriculum
from .scheduler import CurriculumScheduler, SchedulerConfig, StageDecision
from .schema import CriticReport, EpisodeRef, Metrics, StageSpec
from .selection import SelectionDecision, compare

__all__ = [
    "CriticReport",
    "Curriculum",
    "CurriculumScheduler",
    "EpisodeRef",
    "Metrics",
    "SelectionDecision",
    "SchedulerConfig",
    "StageDecision",
    "StageSpec",
    "compare",
    "default_curriculum",
]
