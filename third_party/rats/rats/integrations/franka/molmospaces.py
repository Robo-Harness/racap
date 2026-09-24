from __future__ import annotations

from typing import Any

from rats.envs.base import BaseEnv
from rats.integrations.base_api import ApiBase


class FrankaMolmoSpacesApi(ApiBase):
    """Prompt-visible helpers for MolmoSpaces phase-1 bridge environments."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_task_descriptor": self.get_task_descriptor,
            "list_available_tasks": self.list_available_tasks,
            "switch_task": self.switch_task,
            "get_runtime_summary": self.get_runtime_summary,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the current MolmoSpaces observation bundle.

        Returns:
            Observation dictionary with agentview and wrist-camera RGB/depth,
            robot state placeholders, and the active task descriptor.
        """
        return self._env.get_observation()

    def get_task_descriptor(self) -> dict[str, Any]:
        """Return the currently active canonical MolmoSpaces task descriptor."""
        return self._env.get_task_descriptor()

    def list_available_tasks(self) -> list[dict[str, Any]]:
        """List canonical task descriptors available from the bridge catalog."""
        return self._env.list_task_descriptors()

    def switch_task(self, canonical_task_id: str) -> dict[str, Any]:
        """Recreate the active MolmoSpaces session for a canonical task id.

        Args:
            canonical_task_id: Stable id of the form
                ``molmospaces:{benchmark}:{scene_family}:{task_family}:{variant}``.

        Returns:
            The descriptor of the newly active task.
        """
        return self._env.set_task(canonical_task_id)

    def get_runtime_summary(self) -> dict[str, Any]:
        """Return bridge/session bookkeeping for debugging and verification."""
        return self._env.runtime_summary()


class FrankaMolmoSpacesPrivilegedApi(FrankaMolmoSpacesApi):
    """Privileged helpers for simulator-side MolmoSpaces bookkeeping."""

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        fns.update(
            {
                "complete_current_task": self.complete_current_task,
                "append_runtime_note": self.append_runtime_note,
            }
        )
        return fns

    def complete_current_task(self, note: str = "") -> None:
        """Mark the current task as solved for privileged smoke or bridge tests."""
        self._env.mark_task_complete(note=note or None)

    def append_runtime_note(self, note: str) -> None:
        """Attach a debug note to the active bridge session."""
        self._env.append_runtime_note(note)
