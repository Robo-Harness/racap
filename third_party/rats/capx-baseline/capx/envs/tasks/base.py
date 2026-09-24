import contextlib
import functools
import io
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, SupportsFloat

import numpy as np
from gymnasium import Env, spaces

from capx.control_flow import CapXControlFlowAbort
from capx.envs.base import BaseEnv, ObsType, get_env
from capx.envs.configs.instantiate import instantiate as cfg_instantiate
from capx.envs.configs.loader import DictLoader
from capx.integrations.base_api import ApiBase, get_api
from capx.skills.rats_adapter import format_external_skills_for_prompt


def _controlled_experience_card() -> str:
    configured = os.getenv("CONTROLLED_ONE_SHOT_EXPERIENCE_CARD", "").strip()
    if not configured:
        return ""
    try:
        with open(os.path.expanduser(configured), encoding="utf-8") as handle:
            return handle.read().strip()[:4_000]
    except (OSError, UnicodeError):
        return ""


def _record_experiment_reset(env: Any, seed: int | None) -> None:
    """Append a credential-free simulator-episode event when instrumentation is enabled."""

    target = os.getenv("CAPX_SIM_EPISODE_TELEMETRY_PATH", "").strip()
    if not target:
        return
    handle = getattr(env, "handle", None)
    init_states = getattr(handle, "init_states", None)
    init_state_index = getattr(env, "_current_init_state_index", None)
    if init_state_index is None and init_states is not None:
        init_state_index = (0 if seed is None else int(seed)) % len(init_states)
    record = {
        "event": "environment_reset",
        "time": time.time(),
        "pid": os.getpid(),
        "episode_key": os.getenv("CAPX_EPISODE_KEY", ""),
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
    target = os.getenv("CAPX_NATIVE_TELEMETRY_PATH", "").strip()
    if not target:
        return
    record = {
        "event": event,
        "time": time.time(),
        "pid": os.getpid(),
        "episode_key": os.getenv("CAPX_EPISODE_KEY", ""),
        "reward": float(reward),
        "native_success": bool(terminated),
        "native_predicates": _native_predicates(env),
        "code_step": step_count,
        "api_name": api_name,
        "simulator_steps": int(getattr(env, "_sim_step_count", 0)),
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
    external_skill_library_path: str | None = None
    external_skill_library_mode: str = "direct"
    external_skill_library_max_skills: int = 12
    external_skill_library_include_code: bool = True
    external_skill_library_include_primitives: bool = False
    external_skill_planner_max_selected: int = 6
    external_skill_planner_include_code: bool = False
    external_skill_policy_include_code: bool = True
    external_skill_planner_model: str | None = None
    # When True, exec()s each external skill's body into the user-code
    # execution namespace at every _exec_user_code call. The policy writer
    # can then CALL the skill by name directly — no copy/adapt needed.
    # Mirrors RATS proper's `skill_preamble` injection model. The prompt
    # intro is also rewritten to claim skills are pre-loaded (see
    # rats_adapter.format_skill_entries_for_prompt's ``injected_into_scope``).
    external_skill_inject_into_namespace: bool = False


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
        self._task_prompt = cfg.prompt if cfg.prompt is not None else self.prompt

        # Oracle code: YAML config overrides class attribute
        if cfg.oracle_code is not None:
            self.oracle_code = cfg.oracle_code
        self._system_prompt = (
            "You are a helpful assistant that generates Python code to directly solve the task."
        )
        experience_card = _controlled_experience_card()
        if experience_card:
            self._system_prompt += (
                "\n\nONE-TRIAL TARGET-DOMAIN EXPERIENCE CARD "
                "(advisory; current visual evidence wins):\n"
                + experience_card
            )
        self._full_prompt = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": [{"type": "text", "text": self._get_complete_prompt()}]},
        ]

        # Persistent execution namespace to retain variables across steps
        self._exec_globals: dict[str, Any] = {}
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

        prompt = f"{self._task_prompt}\nAPIs:\n" + "\n".join(docs)
        skill_mode = (self.cfg.external_skill_library_mode or "direct").strip().lower()
        if self.cfg.external_skill_library_path and skill_mode in {"direct", "append"}:
            skill_context = format_external_skills_for_prompt(
                self.cfg.external_skill_library_path,
                max_skills=self.cfg.external_skill_library_max_skills,
                include_code=self.cfg.external_skill_library_include_code,
                include_primitives=self.cfg.external_skill_library_include_primitives,
                injected_into_scope=getattr(
                    self.cfg, "external_skill_inject_into_namespace", False,
                ),
            )
            if skill_context:
                prompt += "\n\n" + skill_context
        return prompt

    def _exec_user_code(self, code: str) -> dict[str, Any]:
        obs = self._get_observation()
        # Update dynamic obs while retaining previously defined variables
        self._exec_globals["obs"] = obs
        self._exec_globals["env"] = self.low_level_env
        self._exec_globals["APIS"] = self._apis
        # Ensure API helper functions remain bound/current
        for api in self._apis.values():
            for fn_name, fn in api.functions().items():
                selected = fn
                if os.getenv("CAPX_NATIVE_TELEMETRY_PATH", "").strip():
                    selected = self._wrap_experiment_api(fn_name, selected)
                self._exec_globals[fn_name] = selected

        # When external_skill_inject_into_namespace is on, also pre-bind
        # the RATS-learned skill function defs into globals. The policy
        # writer's prompt was rewritten to claim these are pre-loaded;
        # actually loading them here closes the loop so the LLM can call
        # them by name without copy/adapt. Skills are exec'd into globals,
        # so cross-references between learned skills (one skill calling
        # another) resolve at call time. Idempotent: re-injection
        # overwrites the same names, which is what we want when a skill
        # body has been edited between turns.
        if getattr(self.cfg, "external_skill_inject_into_namespace", False) and self.cfg.external_skill_library_path:
            try:
                from capx.skills.rats_adapter import (
                    inject_external_skills_into_namespace,
                )
                inject_external_skills_into_namespace(
                    self.cfg.external_skill_library_path,
                    self._exec_globals,
                    max_skills=self.cfg.external_skill_library_max_skills,
                    include_primitives=self.cfg.external_skill_library_include_primitives,
                )
            except Exception as exc:
                # Don't crash the trial if injection fails — log and proceed.
                print(f"[ExternalSkills] Injection failed: {exc}")

        stdout_buffer = io.StringIO()
        tee_out = Tee(sys.stdout, stdout_buffer)
        stderr_buffer = io.StringIO()
        tee_err = Tee(sys.stderr, stderr_buffer)
        ok = True
        try:
            with (
                contextlib.redirect_stdout(tee_out),
                contextlib.redirect_stderr(tee_err),
            ):
                exec(code, self._exec_globals, self._exec_globals)
        except CapXControlFlowAbort:
            # The registered episode deadline is evaluator/runtime control
            # flow, not a policy exception.  It must reach the trial wrapper
            # so the current native state and video are finalized exactly at
            # the deadline rather than becoming stderr followed by another
            # generated-code turn.
            raise
        except BaseException:  # defensive; propagate minimal info
            ok = False
            # Always print full traceback to the redirected stderr (tee -> console and buffer)
            traceback.print_exc(file=tee_err)

        return {
            "ok": ok,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "result": self._exec_globals.get("RESULT"),
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
        }
        # Bind helper functions from APIs into the global namespace for convenience
        for api in self._apis.values():
            for fn_name, fn in api.functions().items():
                g[fn_name] = fn
        self._exec_globals = g

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
        obs, info = self.low_level_env.reset(seed=seed, options=options)
        _record_experiment_reset(self.low_level_env, seed)
        _record_experiment_native_state(self.low_level_env, 0.0, False, 0)
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

        if not exec_result["ok"] and exec_result["stderr"] == "":
            print("Uhh we shouldn't be here, sandbox return code 1 but stderr appears empty")
            # import pdb; pdb.set_trace()
            raise RuntimeError("Sandbox return code 1 but stderr appears empty")

        info = {
            "sandbox_rc": 0 if exec_result["ok"] else 1,
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
            "task_prompt": self._task_prompt,
            "task_completed": task_completed,
        }
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
