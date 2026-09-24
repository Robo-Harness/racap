"""Executor / Sandbox: wraps CaP-Gym execution with frame capture and timeout.

Reuses CaP-Gym's execution infrastructure (env.step(code)) directly.
Adds: first/last frame capture, stderr routing, execution timeout.
"""

from __future__ import annotations

import signal
import time
import traceback
from typing import Any

from rats.control_flow import RATSExecutionTimeout


class Executor:
    def __init__(self, timeout_seconds: int = 600) -> None:
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        code: str,
        env: Any,
        scene_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute generated code in the CaP-Gym environment.

        Args:
            code: Python code string to execute.
            env: CaP-Gym environment with step() method.
            scene_context: Optional scene metadata.

        Returns:
            Dict with: success, stdout, stderr, reward, task_completed,
                       before_frame, after_frame, artifacts.
        """
        before_frame = self._capture_frame(env)
        before_wrist_frame = self._capture_wrist_frame(env)
        grounded_before = self._capture_grounded_state(env)
        self._start_state_trace(env)

        step_fn = getattr(env, "step", None)
        if not callable(step_fn):
            # No real environment - return simulated result
            return self._simulated_execution(code, before_frame, scene_context)

        # Set timeout
        old_handler = None
        old_alarm_seconds = 0.0
        old_alarm_interval = 0.0
        replaced_alarm = False
        alarm_started = 0.0
        try:
            def _timeout_handler(signum, frame):
                raise RATSExecutionTimeout(
                    f"Execution timed out after {self.timeout_seconds}s"
                )

            old_handler = signal.getsignal(signal.SIGALRM)
            old_alarm_seconds, old_alarm_interval = signal.getitimer(
                signal.ITIMER_REAL
            )
            # An episode deadline has higher priority than this per-execution
            # watchdog.  If it will fire first, leave both its timer and its
            # handler untouched.  Otherwise install the shorter sandbox timer
            # and later restore the episode timer after subtracting elapsed
            # wall time.  Restoring the stale pre-execution duration was the
            # bug that allowed repeated code blocks to extend a 1000 s episode.
            if not old_alarm_seconds or old_alarm_seconds > self.timeout_seconds:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, float(self.timeout_seconds))
                alarm_started = time.monotonic()
                replaced_alarm = True
        except (ValueError, AttributeError):
            pass  # Not on Unix or not main thread

        try:
            obs, reward, terminated, truncated, info = step_fn(code)
            after_frame = self._capture_frame(env)
            after_wrist_frame = self._capture_wrist_frame(env)
            grounded_after = self._capture_grounded_state(env)

            sandbox_rc = info.get("sandbox_rc", 0)
            return {
                "success": sandbox_rc == 0,
                "stdout": info.get("stdout", ""),
                "stderr": info.get("stderr", ""),
                "reward": float(reward) if reward is not None else None,
                "task_completed": info.get("task_completed", False),
                # Policy writer's own self-reported per-step outcome dict.
                # Populated by CodeExecutionEnvBase from the code's final
                # RESULT namespace variable. Non-privileged: comes from the
                # code's own gates, not the simulator's reward.
                "user_result": info.get("user_result"),
                "before_frame": before_frame,
                "after_frame": after_frame,
                "before_wrist_frame": before_wrist_frame,
                "after_wrist_frame": after_wrist_frame,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "artifacts": {
                    "scene_model": scene_context.get("scene_model") if scene_context else None,
                    "info": info,
                    "observation": obs,
                    "grounded_state": {
                        "before": grounded_before,
                        "after": grounded_after,
                    },
                    "state_trace": self._capture_state_trace(env),
                },
            }
        except RATSExecutionTimeout as e:
            after_frame = self._capture_frame(env)
            after_wrist_frame = self._capture_wrist_frame(env)
            grounded_after = self._capture_grounded_state(env)
            timeout_info = self._collect_runtime_diagnostics(env)
            return {
                "success": False,
                "stdout": "",
                # Keep stderr empty so downstream diagnosers don't classify
                # this as a code_bug — the timeout flag below is the
                # canonical signal.
                "stderr": "",
                "reward": None,
                "task_completed": False,
                "before_frame": before_frame,
                "after_frame": after_frame,
                "before_wrist_frame": before_wrist_frame,
                "after_wrist_frame": after_wrist_frame,
                "terminated": False,
                "truncated": True,
                "timeout": True,
                "timeout_seconds": self.timeout_seconds,
                "timeout_message": str(e),
                "artifacts": {
                    "timeout": True,
                    "timeout_seconds": self.timeout_seconds,
                    "info": timeout_info,
                    "grounded_state": {
                        "before": grounded_before,
                        "after": grounded_after,
                    },
                    "state_trace": self._capture_state_trace(env),
                },
            }
        except Exception as e:
            after_frame = self._capture_frame(env)
            after_wrist_frame = self._capture_wrist_frame(env)
            grounded_after = self._capture_grounded_state(env)
            error_info = self._collect_runtime_diagnostics(env)
            return {
                "success": False,
                "stdout": "",
                "stderr": traceback.format_exc(),
                "reward": None,
                "task_completed": False,
                "before_frame": before_frame,
                "after_frame": after_frame,
                "before_wrist_frame": before_wrist_frame,
                "after_wrist_frame": after_wrist_frame,
                "terminated": False,
                "truncated": False,
                "artifacts": {
                    "exception": e.__class__.__name__,
                    "info": error_info,
                    "grounded_state": {
                        "before": grounded_before,
                        "after": grounded_after,
                    },
                    "state_trace": self._capture_state_trace(env),
                },
            }
        finally:
            # Restore an outer deadline without giving back time spent inside
            # this sandbox execution.  When the outer timer was already the
            # earlier deadline we never touched it, so no restoration occurs.
            if replaced_alarm:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)
                    if old_alarm_seconds > 0.0:
                        elapsed = max(0.0, time.monotonic() - alarm_started)
                        remaining = max(1e-6, old_alarm_seconds - elapsed)
                        signal.setitimer(
                            signal.ITIMER_REAL,
                            remaining,
                            old_alarm_interval,
                        )
                except (ValueError, AttributeError):
                    pass

    def _collect_runtime_diagnostics(self, env: Any) -> dict[str, Any]:
        """Best-effort API diagnostics even when env.step raises/times out."""
        info: dict[str, Any] = {}
        collect_fn = getattr(env, "_collect_api_runtime_diagnostics", None)
        if callable(collect_fn):
            try:
                api_diagnostics, api_diagnostics_summary = collect_fn()
                if api_diagnostics:
                    info["api_diagnostics"] = api_diagnostics
                if api_diagnostics_summary:
                    info["api_diagnostics_summary"] = api_diagnostics_summary
            except Exception:
                pass
        return info

    def _start_state_trace(self, env: Any) -> None:
        low = getattr(env, "low_level_env", None) or getattr(env, "_env", None) or env
        fn = getattr(low, "start_state_trace", None)
        if callable(fn):
            try:
                fn("execution_start")
                return
            except Exception:
                pass
        clear_fn = getattr(low, "clear_state_trace", None)
        if callable(clear_fn):
            try:
                clear_fn()
            except Exception:
                pass

    def _capture_state_trace(self, env: Any) -> dict[str, Any]:
        low = getattr(env, "low_level_env", None) or getattr(env, "_env", None) or env
        fn = getattr(low, "get_state_trace", None)
        if callable(fn):
            try:
                trace = fn(clear=True)
                if isinstance(trace, dict):
                    return self._json_safe(trace)
            except Exception as exc:
                return {"error": str(exc)}
        return {}

    def _capture_grounded_state(self, env: Any) -> dict[str, Any]:
        """Best-effort privileged/grounded state snapshot before/after code.

        This is intentionally generic: if the low-level environment exposes
        MolmoSpaces inventory/object positions, task info, task descriptors, or
        robot state, capture the JSON-safe subset. Verifiers can then compare
        before/after state without depending on policy self-report or VLM
        success labels.
        """
        low = getattr(env, "low_level_env", None) or getattr(env, "_env", None) or env
        out: dict[str, Any] = {}
        for key, method_name in (
            ("inventory", "describe_scene_inventory"),
            ("task_info", "get_task_info"),
            ("task_descriptor", "get_task_descriptor"),
        ):
            fn = getattr(low, method_name, None)
            if callable(fn):
                try:
                    val = fn()
                    if val is not None:
                        out[key] = self._json_safe(val)
                except Exception:
                    pass

        obs = None
        for owner in (env, low):
            if owner is None:
                continue
            for attr in ("get_observation", "_get_observation", "build_observation"):
                fn = getattr(owner, attr, None)
                if callable(fn):
                    try:
                        obs = fn()
                        break
                    except Exception:
                        continue
            if obs is not None:
                break
        if isinstance(obs, dict):
            robot: dict[str, Any] = {}
            for key in ("robot_cartesian_pos", "robot_joint_pos", "robot_base_pose"):
                if key in obs:
                    robot[key] = self._json_safe(obs[key])
            if robot:
                out["robot"] = robot
        return out

    def _json_safe(self, value: Any) -> Any:
        """Convert common numpy containers/scalars into JSON-ish values."""
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if hasattr(value, "tolist"):
            try:
                return self._json_safe(value.tolist())
            except Exception:
                pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _capture_frame(self, env: Any):
        """Capture current frame from environment."""
        render_fn = getattr(env, "render", None)
        if callable(render_fn):
            try:
                return render_fn(mode="rgb_array")
            except TypeError:
                try:
                    return render_fn()
                except Exception:
                    return None
            except Exception:
                return None
        return None

    def _capture_wrist_frame(self, env: Any):
        """Capture current wrist-camera RGB from env observation if available.

        LIBERO obs carries `obs["robot0_eye_in_hand"]["images"]["rgb"]`. This
        gives the view from the gripper looking down — sharper than the fixed
        agentview for judging grasp alignment ('is the target directly under
        the gripper?').
        """
        # Try several paths to find the observation object
        obs = None
        for attr in ("get_observation", "_get_observation"):
            fn = getattr(env, attr, None)
            if callable(fn):
                try:
                    obs = fn()
                    break
                except Exception:
                    continue
        if obs is None:
            low = getattr(env, "low_level_env", None) or getattr(env, "_env", None)
            if low is not None:
                for attr in ("get_observation", "_get_observation"):
                    fn = getattr(low, attr, None)
                    if callable(fn):
                        try:
                            obs = fn()
                            break
                        except Exception:
                            continue
        if not isinstance(obs, dict):
            return None
        try:
            return obs["robot0_eye_in_hand"]["images"]["rgb"]
        except (KeyError, TypeError):
            return None

    def _simulated_execution(
        self,
        code: str,
        before_frame: Any,
        scene_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Simulated execution when no real environment is available."""
        # Try to compile the code to check for syntax errors
        try:
            compile(code, "<policy>", "exec")
        except SyntaxError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"SyntaxError: {e.msg} at line {e.lineno}",
                "reward": None,
                "task_completed": False,
                "before_frame": before_frame,
                "after_frame": before_frame,
                "terminated": False,
                "truncated": False,
                "artifacts": {"simulated": True},
            }

        return {
            "success": True,
            "stdout": "Simulated execution (no real environment).",
            "stderr": "",
            "reward": 0.0,
            "task_completed": False,
            "before_frame": before_frame,
            "after_frame": before_frame,
            "terminated": False,
            "truncated": False,
            "artifacts": {"simulated": True},
        }
