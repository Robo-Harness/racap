import contextlib
import functools
import io
import inspect
import json
import math
import os
import sys
import time
import traceback
from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, SupportsFloat, Tuple, Union

import numpy as np
from gymnasium import Env, spaces

from rats.control_flow import RATSControlFlowAbort
from rats.envs.base import BaseEnv, ObsType, get_env
from rats.envs.configs.instantiate import instantiate as cfg_instantiate
from rats.envs.configs.loader import DictLoader
from rats.integrations.base_api import ApiBase, get_api


def _record_experiment_reset(env: Any, seed: int | None) -> None:
    """Append a credential-free simulator-episode event when instrumentation is enabled."""

    target = os.getenv("RATS_SIM_EPISODE_TELEMETRY_PATH", "").strip()
    if not target:
        return
    handle = getattr(env, "handle", None)
    init_states = getattr(handle, "init_states", None)
    # Prefer the index recorded by the simulator after it has applied all
    # adapter-specific conventions.  FrankaLiberoEnv accepts a 1-based trial
    # seed and therefore applies ``seed - 1``; recomputing the index here from
    # the raw seed used to mislabel trial 1 as init state 1 even though state 0
    # was actually loaded.
    init_state_index = getattr(env, "_current_init_state_index", None)
    if init_state_index is None and init_states is not None:
        init_state_index = (0 if seed is None else int(seed)) % len(init_states)
    record = {
        "event": "environment_reset",
        "time": time.time(),
        "pid": os.getpid(),
        "episode_key": os.getenv("RATS_EPISODE_KEY", ""),
        "task_prompt": str(getattr(handle, "task_language", "") or ""),
        "seed": seed,
        "init_state_index": init_state_index,
        "suite": getattr(env, "suite_name", ""),
        "task_id": getattr(env, "task_id", None),
        "environment_class": type(env).__name__,
    }
    path = os.path.abspath(target)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _check_experiment_reset_budget() -> None:
    target = os.getenv("RATS_SIM_EPISODE_TELEMETRY_PATH", "").strip()
    maximum = int(os.getenv("RATS_SIM_EPISODE_MAX", "0") or 0)
    if not target or maximum <= 0:
        return
    path = os.path.abspath(target)
    completed = 0
    try:
        with open(path, encoding="utf-8") as handle:
            completed = sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        pass
    if completed >= maximum:
        raise SystemExit(
            f"RATS simulator-episode budget reached: {completed}/{maximum}"
        )


def _native_predicates(env: Any) -> list[dict[str, Any]]:
    current = getattr(getattr(env, "handle", None), "env", None)
    visited: set[int] = set()
    for _ in range(8):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        parsed = getattr(current, "parsed_problem", None)
        evaluate = getattr(current, "_eval_predicate", None)
        if parsed is not None and callable(evaluate):
            rows = []
            for state in parsed.get("goal_state", []) or []:
                try:
                    satisfied = bool(evaluate(state))
                except Exception:
                    satisfied = False
                rows.append(
                    {
                        "predicate": [str(value) for value in state],
                        "satisfied": satisfied,
                    }
                )
            return rows
        current = getattr(current, "env", None) or getattr(current, "unwrapped", None)
    return []


def _record_experiment_native_state(
    env: Any,
    reward: float,
    terminated: bool,
    step_count: int,
    *,
    event: str = "native_state_after_code",
    api_name: str = "",
) -> None:
    target = os.getenv("RATS_NATIVE_TELEMETRY_PATH", "").strip()
    if not target:
        return
    low_level = getattr(env, "low_level_env", env)
    simulator = getattr(low_level, "handle", None)
    record = {
        "event": event,
        "time": time.time(),
        "pid": os.getpid(),
        "episode_key": os.getenv("RATS_EPISODE_KEY", ""),
        "reward": float(reward),
        "native_success": bool(terminated),
        "native_predicates": _native_predicates(env),
        "code_step": step_count,
        "api_name": api_name,
        "simulator_steps": int(getattr(env, "_sim_step_count", 0)),
        "simulator_suite": getattr(simulator, "suite_name", None),
        "simulator_task_id": getattr(simulator, "task_id", None),
    }
    path = os.path.abspath(target)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class Tee(io.TextIOBase):
    """This allows streaming stdout and stderr to both the console and a buffer
    (enables breakpointing for debugging!)
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


@dataclass
class CodeExecEnvConfig:
    """Configuration for a code-execution environment.

    Attributes:
        low_level: A constructed low-level env or a YAML path to its config.
        apis: List of API names to expose to user code (e.g., "graspnet-real").
        prompt: Task instruction for the agent.
        multi_turn_prompt: Instruction for the agent to regenerate the code for multi-turn.
    """

    low_level: Env | str
    apis: list[str]
    prompt: str | None = None
    task_only_prompt: str | None = None
    multi_turn_prompt: str | None = None
    oracle_code: str | None = None
    privileged: bool = False
    enable_render: bool = True
    viser_debug: bool = False


def _stderr_indicates_environment_timeout(stderr: str, exception_type: str = "") -> bool:
    """Return True when a user-code exception means the env/RPC is poisoned.

    CodeExecutionEnvBase intentionally catches user-code exceptions and reports
    them as sandbox failures so RATS can give feedback.  Remote simulator
    timeouts are different: after the RPC times out the client has already
    dropped/reconnected the socket and the attempt should not continue as if
    the same simulator state were healthy.  Keep this detector conservative so
    ordinary user-code errors still remain non-terminal feedback.
    """

    if not stderr:
        return False
    if "Timed out waiting for mlspaces server response" in stderr:
        return True
    if "Execution timed out after" in stderr:
        return True
    timeout_exception = exception_type in {"TimeoutError", "TimeoutException"}
    if timeout_exception and any(token in stderr for token in ("mlspaces", "MolmoSpaces", "RPC")):
        return True
    return False


class SimpleExecutor:
    """Minimal in-process code executor with full imports allowed.

    Executes user code with globals: env (low-level env), APIS (name->api), INPUTS, RESULT.
    The user code may import any installed package and can interact with `env` directly
    for closed-loop control.
    """

    def __init__(self, env: BaseEnv, apis: dict[str, ApiBase]) -> None:
        self._env = env
        self._apis = apis

    def run(self, code: str, *, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        g: dict[str, Any] = {
            "__name__": "__main__",
            "env": self._env,
            "APIS": self._apis,
            "INPUTS": inputs or {},
            "RESULT": None,
        }
        try:
            exec(code, g, g)
            return {"ok": True, "result": g.get("RESULT")}
        except RATSControlFlowAbort:
            raise
        except BaseException as exc:  # defensive; propagate minimal info
            return {"ok": False, "error": repr(exc)}


class CodeExecutionEnvBase(Env):
    """High-level env that runs Python code and interacts with a low-level env."""

    prompt: str | None = None
    regenerate_prompt: str | None = None

    def __init__(self, cfg: CodeExecEnvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.low_level_env: BaseEnv = self._build_low_level(
            cfg.low_level, cfg.privileged, cfg.enable_render, cfg.viser_debug
        )  # type: ignore[assignment]
        # Create APIs once; maximize sharing inside a worker via lru_cache in get_api
        self._apis: dict[str, ApiBase] = {n: get_api(n)(self.low_level_env) for n in cfg.apis}
        self._is_molmospaces_runtime = self._detect_molmospaces_runtime(cfg)
        # for api in self._apis.values():
        #     api.set_env(self.low_level_env)
        self._executor = SimpleExecutor(self.low_level_env, self._apis)
        self._step_count = 0
        self.action_space = spaces.Text(max_length=4096)
        self.observation_space = spaces.Dict({"task_prompt": spaces.Text(max_length=4096)})

        # Prompt priority: YAML config (cfg.prompt) overrides the class attribute (self.prompt).
        # The class attribute serves as the single source of truth for the default task prompt.
        # YAML configs should only set prompt when they need to override the class default
        # (e.g., multi-turn variants that add extra instructions).
        self._task_prompt_template = cfg.prompt if cfg.prompt is not None else self.prompt
        self._task_prompt = self._task_prompt_template

        # Oracle code: YAML config overrides class attribute
        if cfg.oracle_code is not None:
            self.oracle_code = cfg.oracle_code
        self._system_prompt = (
            "You are a helpful assistant that generates Python code to directly solve the task."
        )
        self._refresh_full_prompt()

        # Persistent execution namespace to retain variables across steps
        self._exec_globals: dict[str, Any] = {}
        self._api_call_trace: list[dict[str, Any]] = []
        self._init_exec_globals()

    # Functions that can be overridden by subclasses
    def compute_reward(self) -> float:
        """
        Computes the reward for the current state by delegating to the
        low-level environment.

        Returns:
            float: The reward at the current base simulator state.
        """
        return self.low_level_env.compute_reward()

    # ---- Private methods ----
    def _get_complete_prompt(self) -> str:
        """
        Get the complete prompt for the task.
        Returns:
            str: The complete prompt for the task.
        """
        docs = []
        for _name, api in self._apis.items():
            text = api.combined_doc()
            # NOTE: we need to discuss this further down the line
            # docs.append(f"- {name}:\n{text.strip()}")
            docs.append(f"\n{text.strip()}")
        return f"{self._task_prompt}\nAPIs:\n" + "\n".join(docs)

    def _refresh_full_prompt(self) -> None:
        self._full_prompt = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": [{"type": "text", "text": self._get_complete_prompt()}]},
        ]

    def _set_task_prompt(self, task_prompt: str | None) -> None:
        self._task_prompt = task_prompt
        self._refresh_full_prompt()

    def _update_task_prompt_from_reset(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
    ) -> None:
        _ = obs
        task_prompt = info.get("task_prompt")
        if isinstance(task_prompt, str) and task_prompt:
            self._set_task_prompt(task_prompt)

    def _exec_user_code(self, code: str) -> dict[str, Any]:
        obs = self._get_observation()
        # Update dynamic obs while retaining previously defined variables
        self._exec_globals["obs"] = obs
        self._exec_globals["env"] = self.low_level_env
        self._exec_globals["APIS"] = self._apis
        self._clear_api_runtime_diagnostics()
        self._api_call_trace = []
        # Ensure API helper functions remain bound/current.  Preserve the
        # origin/main LIBERO path by binding raw primitives outside MolmoSpaces;
        # MolmoSpaces alone needs wrappers for per-step verifier traces.
        for api_name, api in self._apis.items():
            for fn_name, fn in api.functions().items():
                selected = (
                    self._wrap_api_function(api_name, fn_name, fn)
                    if self._is_molmospaces_runtime
                    else fn
                )
                if os.getenv("RATS_NATIVE_TELEMETRY_PATH", "").strip():
                    selected = self._wrap_experiment_api(fn_name, selected)
                self._exec_globals[fn_name] = selected

        stdout_buffer = io.StringIO()
        tee_out = Tee(sys.stdout, stdout_buffer)
        stderr_buffer = io.StringIO()
        tee_err = Tee(sys.stderr, stderr_buffer)
        ok = True
        exception_type = ""
        exception_message = ""
        try:
            with (
                contextlib.redirect_stdout(tee_out),
                contextlib.redirect_stderr(tee_err),
            ):
                exec(code, self._exec_globals, self._exec_globals)
        except RATSControlFlowAbort:
            # Deadlines and provider/quota aborts are infrastructure control
            # flow, not policy failures.  Let the outer executor/checkpoint
            # wrapper handle them instead of serializing them as stderr.
            raise
        except BaseException as exc:  # defensive; propagate minimal info
            ok = False
            exception_type = type(exc).__name__
            exception_message = str(exc)
            # Always print full traceback to the redirected stderr (tee -> console and buffer)
            traceback.print_exc(file=tee_err)

        return {
            "ok": ok,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "result": self._exec_globals.get("RESULT"),
            "exception_type": exception_type,
            "exception_message": exception_message,
        }

    def _wrap_experiment_api(
        self, fn_name: str, fn: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Record evaluator-only progress after each public API call."""

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            finally:
                reward = float(self.compute_reward())
                _record_experiment_native_state(
                    self.low_level_env,
                    reward,
                    reward == 1.0,
                    self._step_count,
                    event="native_state_after_api",
                    api_name=fn_name,
                )

        return wrapped

    def _init_exec_globals(self) -> None:
        """
        Initialize the persistent globals dictionary for user code execution.
        This is called at construction time and on reset to avoid leakage across episodes.
        """
        g: dict[str, Any] = {
            "__name__": "__main__",
            "env": self.low_level_env,
            "APIS": self._apis,
            # Populated per-step/reset; keep reference stable across execs
            "INPUTS": {},
            # Users can set and reuse RESULT across steps if desired
            "RESULT": None,
            # Pre-bind common modules so user code and learned-skill
            # function defs work without an explicit import. Two failure
            # modes are fixed at once:
            #   1. LLMs (esp. terser ones like Gemini-flash) often emit
            #      `np.array(...)` without an `import numpy as np`,
            #      crashing exec immediately.
            #   2. Auto-extracted learned skills sometimes carry
            #      `np.ndarray` in type annotations or `np.array(...)`
            #      in default arguments. Those expressions are evaluated
            #      at function-DEFINITION time, i.e. while exec'ing the
            #      skill_preamble, BEFORE the LLM's own `import numpy as
            #      np` line at the top of its block has run. A late
            #      `import numpy as np` inside the function body cannot
            #      rescue the def-time evaluation.
            # Pre-binding here resolves both — the LLM may still write
            # `import numpy as np`, that's idempotent.
            "np": np,
            "numpy": np,
            "math": math,
            "time": time,
            # typing exports for learned skills that carry Python type hints.
            # Same def-time-evaluation issue as np.ndarray (see comment above):
            # when the skill_preamble exec's a skill like
            # `def f(...) -> dict[str, Any]:`, the `Any` is looked up at def
            # time, before any `from typing import Any` the LLM might emit.
            # Pre-binding here avoids NameError and matches the np pattern.
            "Any": Any,
            "Optional": Optional,
            "Union": Union,
            "Tuple": Tuple,
            "List": List,
            "Dict": Dict,
            "Callable": Callable,
        }

        @contextmanager
        def step_context(
            step_id: str,
            step_goal: str = "",
            step_index: int | None = None,
        ):
            """Delimit one planned policy step for runtime artifact logging."""
            from rats.utils.execution_logger import policy_step_context

            with policy_step_context(
                str(step_id),
                str(step_goal or step_id),
                step_index=step_index,
            ):
                yield

        def begin_step(
            step_id: str,
            step_goal: str = "",
            step_index: int | None = None,
        ) -> dict[str, Any]:
            from rats.utils.execution_logger import begin_policy_step

            return begin_policy_step(
                str(step_id),
                str(step_goal or step_id),
                step_index=step_index,
            )

        def end_step(step_id: str | None = None) -> dict[str, Any] | None:
            from rats.utils.execution_logger import end_policy_step

            return end_policy_step(str(step_id) if step_id is not None else None)

        # Step markers (step_context/begin_step/end_step) are env-agnostic:
        # they write into execution_logger which records frame_start/end per
        # marker. Used by the per-step verifier to slice trajectory frames
        # into step-scope segments. Inject unconditionally — for envs that
        # don't carry policy_step markers in their generated code (legacy
        # path before policy_writer was taught the marker requirement),
        # the helpers are just unused names in scope; no harm.
        g.update(
            {
                "step_context": step_context,
                "begin_step": begin_step,
                "end_step": end_step,
            }
        )
        # Bind helper functions from APIs into the global namespace for convenience.
        # Outside MolmoSpaces, bind exactly the raw functions as origin/main did.
        for api in self._apis.values():
            for fn_name, fn in api.functions().items():
                g[fn_name] = (
                    self._wrap_api_function(api.__class__.__name__, fn_name, fn)
                    if self._is_molmospaces_runtime
                    else fn
                )
        self._exec_globals = g

    def _detect_molmospaces_runtime(self, cfg: CodeExecEnvConfig) -> bool:
        api_names = [str(name).lower() for name in (cfg.apis or [])]
        if any("molmospaces" in name for name in api_names):
            return True
        env_obj = self.low_level_env
        env_text = (
            f"{env_obj.__class__.__module__}."
            f"{env_obj.__class__.__name__}"
        ).lower()
        return "molmospaces" in env_text or "mlspaces" in env_text

    def _wrap_api_function(self, api_name: str, fn_name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Trace primitive/helper calls without relying on policy-authored RESULT.

        The trace is intentionally about runtime outputs: arguments are compact
        summaries, returns are compact summaries, and before/after robot/frame
        snapshots come from the environment.  Generated policy code and line
        contents are not recorded here.
        """
        try:
            signature = inspect.signature(fn)
        except Exception:
            signature = None

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            policy_step = self._current_policy_step_context()
            api_obj = getattr(fn, "__self__", None)
            diagnostic_event_start = self._runtime_diagnostic_event_count(api_obj)
            event: dict[str, Any] = {
                "event_id": len(self._api_call_trace) + 1,
                "api_name": str(api_name),
                "function_name": str(fn_name),
                "kind": "primitive_or_helper",
                "start_time": round(time.time(), 6),
                "start_frame": self._video_frame_count(),
                "args_summary": self._summarize_call_args(signature, args, kwargs),
                "robot_before": self._robot_snapshot(),
                "status": "running",
            }
            if policy_step:
                event.update(policy_step)
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                event.update(
                    {
                        "status": "exception",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
                raise
            else:
                event.update(
                    {
                        "status": "ok",
                        "return_summary": self._summarize_value(result),
                    }
                )
                return result
            finally:
                event.update(
                    {
                        "end_time": round(time.time(), 6),
                        "end_frame": self._video_frame_count(),
                        "robot_after": self._robot_snapshot(),
                    }
                )
                if policy_step:
                    self._tag_new_runtime_diagnostics(
                        api_obj,
                        diagnostic_event_start,
                        policy_step,
                    )
                self._api_call_trace.append(self._json_safe(event))

        if signature is not None:
            wrapped.__signature__ = signature  # type: ignore[attr-defined]
        return wrapped

    @staticmethod
    def _runtime_diagnostic_event_count(api_obj: Any) -> int | None:
        get_fn = getattr(api_obj, "get_runtime_diagnostics", None)
        if not callable(get_fn):
            return None
        try:
            data = get_fn()
        except Exception:
            return None
        events = data.get("events") if isinstance(data, dict) else None
        return len(events) if isinstance(events, list) else None

    @staticmethod
    def _tag_new_runtime_diagnostics(
        api_obj: Any,
        start_count: int | None,
        policy_step: dict[str, Any],
    ) -> None:
        """Attach the active policy-step marker to diagnostics emitted by an API call.

        Some integrations record rich image/mask/point-cloud artifacts in their
        own runtime diagnostics instead of returning them from the primitive.
        The wrapper owns the call boundary, so it can deterministically tag only
        the diagnostics appended during this primitive/helper call.
        """
        if start_count is None:
            return
        get_fn = getattr(api_obj, "get_runtime_diagnostics", None)
        if not callable(get_fn):
            return
        try:
            data = get_fn()
        except Exception:
            return
        events = data.get("events") if isinstance(data, dict) else None
        if not isinstance(events, list):
            return
        keep = {
            "policy_step_id",
            "policy_step_index",
            "policy_step_goal",
            "policy_step_marker_index",
            "policy_step_start_frame",
            "policy_step_start_s",
        }
        marker = {k: policy_step.get(k) for k in keep if k in policy_step}
        if not marker:
            return
        for event in events[start_count:]:
            if isinstance(event, dict):
                for key, value in marker.items():
                    event.setdefault(key, value)

    def _current_policy_step_context(self) -> dict[str, Any]:
        try:
            from rats.utils.execution_logger import get_current_step_context

            context = get_current_step_context()
        except Exception:
            context = None
        if not isinstance(context, dict) or not context:
            return {}
        keep = {
            "policy_step_id",
            "policy_step_index",
            "policy_step_goal",
            "policy_step_marker_index",
            "policy_step_start_frame",
            "policy_step_start_s",
        }
        return {k: context.get(k) for k in keep if k in context}

    def _video_frame_count(self) -> int | None:
        fn = getattr(self.low_level_env, "get_video_frame_count", None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                return None
        return None

    def _robot_snapshot(self) -> dict[str, Any]:
        try:
            obs = self.low_level_env.get_observation()
        except Exception:
            return {}
        out: dict[str, Any] = {}
        for key in ("robot_cartesian_pos", "robot_joint_pos", "robot_base_pose"):
            if isinstance(obs, dict) and key in obs:
                out[key] = self._summarize_value(obs.get(key))
        return out

    def _summarize_call_args(
        self,
        signature: inspect.Signature | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if signature is not None:
            try:
                bound = signature.bind_partial(*args, **kwargs)
                return {
                    str(k): self._summarize_value(v)
                    for k, v in bound.arguments.items()
                    if k != "self"
                }
            except Exception:
                pass
        return {
            "args": [self._summarize_value(v) for v in args],
            "kwargs": {str(k): self._summarize_value(v) for k, v in kwargs.items()},
        }

    def _summarize_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            summary: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if value.size <= 32:
                summary["values"] = self._json_safe(value)
            elif np.issubdtype(value.dtype, np.number):
                finite = value[np.isfinite(value)]
                if finite.size:
                    summary.update(
                        {
                            "min": float(np.min(finite)),
                            "max": float(np.max(finite)),
                            "mean": float(np.mean(finite)),
                        }
                    )
                    if value.ndim == 2 and value.shape[1] == 3:
                        summary["centroid"] = self._json_safe(np.mean(value, axis=0))
            return summary
        if isinstance(value, np.generic):
            return self._json_safe(value.item())
        if isinstance(value, dict):
            return {
                str(k): self._summarize_value(v)
                for idx, (k, v) in enumerate(value.items())
                if idx < 24
            }
        if isinstance(value, (list, tuple)):
            if len(value) > 32:
                return {"type": type(value).__name__, "length": len(value)}
            return [self._summarize_value(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > 500:
                return value[:500] + "..."
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return {"type": type(value).__name__, "repr": repr(value)[:200]}

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return self._json_safe(value.item())
            return [self._json_safe(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return self._json_safe(value.item())
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return repr(value)

    def _build_low_level(
        self, src: Env | str, privileged: bool = False, enable_render: bool = True, viser_debug: bool = False
    ) -> BaseEnv:
        """
        Builds the low level environment from the given source.
        Args:
            src: Env | str: the source of the low level environment
        Returns:
            BaseEnv: the low level environment
        """
        if isinstance(src, str):
            if src.endswith(".yaml") or src.endswith(".yml"):
                cfg = DictLoader.load(src)
                if isinstance(cfg, dict) and "_target_" in cfg:
                    return cfg_instantiate(cfg)  # type: ignore[no-any-return]
                return cfg  # type: ignore[return-value]
            else:
                return get_env(src, privileged=privileged, enable_render=enable_render, viser_debug=viser_debug)
        return src

    def _get_observation(self) -> dict[str, Any]:
        """
        Gets the observation of the environment. This should be consistent for all environments, where observation from low level environment
        along with the full prompt is returned
        Returns:
            Dict[str, Any]: The observation of the environment.
        """
        obs = self.low_level_env.get_observation()
        obs.update({"full_prompt": self._full_prompt})
        return obs

    def _clear_api_runtime_diagnostics(self) -> None:
        for api in self._apis.values():
            clear_fn = getattr(api, "clear_runtime_diagnostics", None)
            if callable(clear_fn):
                clear_fn()

    def _collect_api_runtime_diagnostics(self) -> tuple[dict[str, Any], str]:
        diagnostics: dict[str, Any] = {}
        summary_parts: list[str] = []
        for name, api in self._apis.items():
            get_fn = getattr(api, "get_runtime_diagnostics", None)
            if callable(get_fn):
                data = get_fn()
                if data and (data.get("events") or data.get("last_error")):
                    diagnostics[name] = data
            summary_fn = getattr(api, "get_runtime_diagnostics_summary", None)
            if callable(summary_fn):
                summary = summary_fn()
                if summary:
                    summary_parts.append(f"{name}: {summary}")
        return diagnostics, " | ".join(summary_parts)

    # ---- Public facing methods ----
    # Public facing methods that should be consistent for all environments
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Args:
            seed: The seed to reset the environment with.
            options: The options to reset the environment with.
        Returns:
            tuple[ObsType, dict[str, Any]]: A tuple containing the observation and info.
        """
        self._step_count = 0
        _check_experiment_reset_budget()
        obs, info = self.low_level_env.reset(seed=seed, options=options)
        _record_experiment_reset(self.low_level_env, seed)
        _record_experiment_native_state(self.low_level_env, 0.0, False, 0)
        self._update_task_prompt_from_reset(obs, info)
        obs.update(self._get_observation())
        # Reinitialize globals for a fresh episode and prime INPUTS with the reset observation
        self._init_exec_globals()
        self._exec_globals["INPUTS"] = obs
        info.update({"task_prompt": self._task_prompt})
        return obs, info

    def step(self, action: str) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Default implementation: execute code with helpers, report reward and logs.
        Subclasses can override hooks to customize inputs and helper bindings.
        """
        self._step_count += 1
        exec_result = self._exec_user_code(action)
        obs = self._get_observation()
        # Force viser 3D view update after code execution so the scene
        # reflects the final state (sim substep updates may have been skipped).
        if hasattr(self.low_level_env, "viser_debug") and self.low_level_env.viser_debug:
            self.low_level_env._update_viser_server()
        reward = self.compute_reward()
        if hasattr(self.low_level_env, "task_completed"):
            task_completed = self.low_level_env.task_completed()
        else:
            task_completed = None
        terminated = reward == 1.0
        _record_experiment_native_state(
            self.low_level_env, float(reward), bool(terminated), self._step_count
        )

        truncated = getattr(self.low_level_env, "_sim_step_count", 0) >= getattr(
            self.low_level_env, "max_steps", 999999
        )  # type: ignore[arg-type]
        env_poisoned = _stderr_indicates_environment_timeout(
            str(exec_result.get("stderr") or ""),
            str(exec_result.get("exception_type") or ""),
        )
        if env_poisoned:
            truncated = True

        if not exec_result["ok"] and exec_result["stderr"] == "":
            print("Uhh we shouldn't be here, sandbox return code 1 but stderr appears empty")
            # import pdb; pdb.set_trace()
            raise RuntimeError("Sandbox return code 1 but stderr appears empty")

        info = {
            "sandbox_rc": 0 if exec_result["ok"] else 1,
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
            "exception_type": exec_result.get("exception_type", ""),
            "exception_message": exec_result.get("exception_message", ""),
            "task_prompt": self._task_prompt,
            "task_completed": task_completed,
            "env_poisoned": env_poisoned,
            # The user code's final RESULT namespace value (usually a dict
            # mapping plan step_id -> bool). This is the policy writer's own
            # self-report — non-privileged, non-sim-derived. RATS's
            # feedback_generator cross-checks it against the VLM diagnoser's
            # VPS verdicts so mechanism A won't preserve code that the
            # writer's own gates said failed.
            "user_result": exec_result.get("result"),
        }
        api_diagnostics, api_diagnostics_summary = self._collect_api_runtime_diagnostics()
        if api_diagnostics:
            info["api_diagnostics"] = api_diagnostics
        if api_diagnostics_summary:
            info["api_diagnostics_summary"] = api_diagnostics_summary
        if self._api_call_trace:
            info["api_call_trace"] = list(self._api_call_trace)
        return obs, reward, bool(terminated), bool(truncated), info

    def render(self, mode: str = "rgb_array"):
        return self.low_level_env.render(mode=mode)

    def render_wrist(self) -> np.ndarray | None:
        if hasattr(self.low_level_env, "render_wrist"):
            return self.low_level_env.render_wrist()
        return None

    # Video passthrough for demo compatibility
    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        import inspect

        sig = inspect.signature(self.low_level_env.enable_video_capture)
        if "wrist_camera" in sig.parameters:
            self.low_level_env.enable_video_capture(
                enabled, clear=clear, wrist_camera=wrist_camera
            )
        else:
            self.low_level_env.enable_video_capture(enabled, clear=clear)

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        return self.low_level_env.get_video_frames(clear=clear)

    def get_video_frame_count(self) -> int:
        if hasattr(self.low_level_env, "get_video_frame_count"):
            return self.low_level_env.get_video_frame_count()
        if hasattr(self.low_level_env, "_frame_buffer"):
            return len(self.low_level_env._frame_buffer)
        return 0

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        if hasattr(self.low_level_env, "get_video_frames_range"):
            return self.low_level_env.get_video_frames_range(start, end)
        if hasattr(self.low_level_env, "_frame_buffer"):
            return [f.copy() for f in self.low_level_env._frame_buffer[start:end]]
        return []

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        if hasattr(self.low_level_env, "get_wrist_video_frames"):
            return self.low_level_env.get_wrist_video_frames(clear=clear)
        if hasattr(self.low_level_env, "_wrist_frame_buffer"):
            frames = [f.copy() for f in self.low_level_env._wrist_frame_buffer]
            if clear:
                self.low_level_env._wrist_frame_buffer.clear()
            return frames
        return []

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        if hasattr(self.low_level_env, "get_wrist_video_frames_range"):
            return self.low_level_env.get_wrist_video_frames_range(start, end)
        if hasattr(self.low_level_env, "_wrist_frame_buffer"):
            return [f.copy() for f in self.low_level_env._wrist_frame_buffer[start:end]]
        return []

    def close(self) -> None:
        for api in self._apis.values():
            close_fn = getattr(api, "close", None)
            if callable(close_fn):
                close_fn()
        close_fn = getattr(self.low_level_env, "close", None)
        if callable(close_fn):
            close_fn()
        super().close()


# Use user's BaseEnv for low-level envs

_EXEC_ENV_FACTORIES: dict[str, Callable[[], CodeExecutionEnvBase]] = {}


def register_exec_env(name: str, factory: Callable[[], CodeExecutionEnvBase]) -> None:
    _EXEC_ENV_FACTORIES[name] = factory


def get_exec_env(name: str) -> Callable[[], CodeExecutionEnvBase]:
    if name not in _EXEC_ENV_FACTORIES:
        raise KeyError(f"Execution Environment '{name}' not registered")
    return _EXEC_ENV_FACTORIES[name]


def list_exec_envs() -> list[str]:
    return list(_EXEC_ENV_FACTORIES.keys())


_CONFIG_FACTORIES: dict[str, CodeExecEnvConfig] = {}


def register_config(name: str, factory: CodeExecEnvConfig) -> None:
    _CONFIG_FACTORIES[name] = factory


def get_config(name: str) -> CodeExecEnvConfig:
    if name not in _CONFIG_FACTORIES:
        raise KeyError(f"Configuration '{name}' not registered")
    return _CONFIG_FACTORIES[name]


def list_configs() -> list[str]:
    return list(_CONFIG_FACTORIES.keys())


__all__ = [
    "register_exec_env",
    "get_exec_env",
    "list_exec_envs",
    "register_config",
    "get_config",
    "list_configs",
    "SimpleExecutor",
    "CodeExecEnvConfig",
    "CodeExecutionEnvBase",
]
