from __future__ import annotations

from typing import Any

from rats.rats.schemas import BehaviorSceneContext, ExecutionRecord, PolicyDraft


class ExecutorAgent:
    def _capture_frame(self, env: object):
        render_fn = getattr(env, "render", None)
        if callable(render_fn):
            try:
                return render_fn(mode="rgb_array")
            except TypeError:
                return render_fn()
            except Exception:
                return None
        return None

    def execute(self, draft: PolicyDraft, env: object, context: BehaviorSceneContext | None = None) -> ExecutionRecord:
        before_frame = self._capture_frame(env)
        step_fn = getattr(env, "step", None)
        if callable(step_fn):
            try:
                obs, reward, terminated, truncated, info = step_fn(draft.code)
                after_frame = self._capture_frame(env)
            except Exception as exc:  # pragma: no cover - defensive path
                return ExecutionRecord(
                    success=False,
                    stdout="",
                    stderr=str(exc),
                    reward=None,
                    task_completed=False,
                    artifacts={
                        "scene_model": context.scene_model if context is not None else None,
                        "simulated": False,
                        "exception": exc.__class__.__name__,
                        "before_frame": before_frame,
                        "after_frame": self._capture_frame(env),
                    },
                )

            sandbox_rc = info.get("sandbox_rc", 0)
            success = sandbox_rc == 0
            return ExecutionRecord(
                success=success,
                stdout=info.get("stdout", ""),
                stderr=info.get("stderr", ""),
                reward=float(reward) if reward is not None else None,
                task_completed=info.get("task_completed"),
                artifacts={
                    "scene_model": context.scene_model if context is not None else None,
                    "simulated": False,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                    "observation": obs,
                    "before_frame": before_frame,
                    "after_frame": after_frame,
                },
            )

        return ExecutionRecord(
            success=True,
            stdout="No-op executor stub.",
            stderr="",
            reward=1.0,
            task_completed=True,
            artifacts={
                "scene_model": context.scene_model if context is not None else None,
                "simulated": True,
                "before_frame": before_frame,
                "after_frame": before_frame,
            },
        )
