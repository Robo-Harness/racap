from __future__ import annotations

import re
from typing import Any

from rats.envs.tasks.base import CodeExecutionEnvBase
from rats.integrations.base_api import register_api

from rats.integrations.franka.molmospaces import FrankaMolmoSpacesApi, FrankaMolmoSpacesPrivilegedApi


register_api("FrankaMolmoSpacesApi", FrankaMolmoSpacesApi)
register_api("FrankaMolmoSpacesPrivilegedApi", FrankaMolmoSpacesPrivilegedApi)


def _register_control_api() -> None:
    try:
        from rats.integrations.franka.molmospaces_control import FrankaMolmoSpacesControlApi
        from rats.integrations.franka.molmospaces_reduced import FrankaMolmoSpacesApiReduced
        from rats.integrations.franka.molmospaces_reduced_skill_library import (
            FrankaMolmoSpacesApiReducedSkillLibrary,
        )
        register_api(
            "FrankaMolmoSpacesControlApi",
            lambda env: FrankaMolmoSpacesControlApi(env, use_sam3=True),
        )
        register_api(
            "FrankaMolmoSpacesApiReduced",
            lambda env: FrankaMolmoSpacesApiReduced(env, use_sam3=True),
        )
        register_api(
            "FrankaMolmoSpacesApiReducedSkillLibrary",
            FrankaMolmoSpacesApiReducedSkillLibrary,
        )
    except Exception:
        pass


_register_control_api()


class FrankaMolmoSpacesCodeEnv(CodeExecutionEnvBase):
    """Generic code-execution wrapper for MolmoSpaces bridge envs."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._sync_task_prompt_from_env()

    def _compose_live_task_prompt(self, live_task_goal: str) -> str:
        template = (self._task_prompt_template or "").strip()
        live_task_goal = live_task_goal.strip()
        if not template:
            return live_task_goal

        placeholder_values = {
            "live_task": live_task_goal,
            "task_language": live_task_goal,
            "goal": live_task_goal,
            "task_prompt": live_task_goal,
        }
        if any(f"{{{key}}}" in template for key in placeholder_values):
            try:
                return template.format(**placeholder_values)
            except Exception:
                pass

        if re.search(r"(?im)^Goal:\s*.*$", template):
            return re.sub(
                r"(?im)^Goal:\s*.*$",
                lambda _match: f"Goal: {live_task_goal}",
                template,
                count=1,
            )

        if re.search(r"(?im)^Your goal is.*$", template):
            return re.sub(
                r"(?im)^Your goal is.*$",
                lambda _match: f"Your goal is: {live_task_goal}",
                template,
                count=1,
            )

        return f"{template}\n\nLive task goal: {live_task_goal}"

    def _get_live_task_descriptor(self) -> dict[str, Any]:
        if hasattr(self.low_level_env, "get_task_descriptor"):
            descriptor = self.low_level_env.get_task_descriptor()
            if isinstance(descriptor, dict):
                return descriptor
        return {}

    def _sync_task_prompt_from_env(self, fallback_task_prompt: str | None = None) -> None:
        descriptor = self._get_live_task_descriptor()
        live_task_goal = descriptor.get("language") or fallback_task_prompt
        if isinstance(live_task_goal, str) and live_task_goal:
            self._set_task_prompt(self._compose_live_task_prompt(live_task_goal))

    def _update_task_prompt_from_reset(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
    ) -> None:
        _ = obs
        fallback = info.get("task_prompt") if isinstance(info.get("task_prompt"), str) else None
        self._sync_task_prompt_from_env(fallback_task_prompt=fallback)


__all__ = ["FrankaMolmoSpacesCodeEnv"]
