from __future__ import annotations

from rats.rats.schemas import SkillSummary


class SkillLibraryAgent:
    def __init__(self) -> None:
        self._skills: list[str] = []

    def summarize(self) -> SkillSummary:
        return SkillSummary(
            total_skills=len(self._skills),
            promoted_skills=list(self._skills),
            docs="\n".join(self._skills),
        )

    def register(self, new_skills: list[str]) -> None:
        for skill in new_skills:
            if skill not in self._skills:
                self._skills.append(skill)
