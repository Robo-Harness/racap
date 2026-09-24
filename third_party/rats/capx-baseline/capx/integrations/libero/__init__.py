from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


@dataclass
class LiberoHandle:
    env: Any
    suite_name: str
    task_id: int
    task_language: str
    init_states: Any

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.env.seed(seed)
        obs = self.env.reset()
        init_state_index = None
        if self.init_states is not None:
            init_state_index = (0 if seed is None else int(seed)) % len(self.init_states)
            state_obs = self.env.set_init_state(self.init_states[init_state_index])
            if state_obs is not None:
                obs = state_obs
        return obs, {"init_state_index": init_state_index}

    def step(self, action: list[float]) -> tuple[Any, float, bool, dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return obs, float(reward), bool(done), info


def _extract_language_from_bddl(bddl_path: str) -> str | None:
    try:
        with open(bddl_path, "r") as f:
            content = f.read()
        match = re.search(r"\(:language\s+(.*?)\)", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Warning: Could not extract language from {bddl_path}: {e}")
    return None


def _select_task_language(
    *, public_task_language: str, bddl_path: str, custom_bddl: bool
) -> str:
    """Select the public instruction without trusting stale official BDDL text.

    Some released LIBERO BDDL files contain a ``:language`` field that does
    not describe their own native goal (notably LIBERO-90 tasks 17, 84, and
    85).  The benchmark task metadata is the public instruction authority for
    official suites.  A controlled custom BDDL has no benchmark metadata of
    its own, so only that case intentionally takes its language from the file.
    """

    public = str(public_task_language or "").strip()
    extracted = _extract_language_from_bddl(bddl_path)
    if custom_bddl and extracted:
        return extracted
    return public or extracted or ""


def load_libero_task(
    suite_name: str,
    task_id: int,
    cam_w: int = 128,
    cam_h: int = 128,
    controller: str = "OSC_POSE",
    horizon: int = 1000,
    control_freq: int = 20,
    camera_depths: bool = True,
) -> LiberoHandle:
    """Load a LIBERO task using OffScreenRenderEnv.

    Reference: https://github.com/Lifelong-Robot-Learning/LIBERO
    """
    # Prefer vendored third_party/LIBERO if present, then fall back to installed package
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    vendor_root = os.path.normpath(os.path.join(here, "..", "..", "third_party", "LIBERO-PRO"))
    if os.path.isdir(vendor_root) and vendor_root not in sys.path:
        sys.path.append(vendor_root)
    try:
        from libero import benchmark  # type: ignore[import-not-found]
        from libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]
        from libero.utils import get_libero_path  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "LIBERO not available; add submodule or run `uv sync --extra libero`."
        ) from e
    import os

    # setting help=True will print the available benchmarks
    benchmark_dict = benchmark.get_benchmark_dict(help=False)
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)

    # Controlled-comparison tasks may reuse an official scene and initial
    # state while changing only the public language and native goal.  Keeping
    # these paths explicit avoids mutating the installed LIBERO benchmark and
    # makes the exact evaluation asset hashable in the experiment manifest.
    custom_bddl = os.environ.get("CONTROLLED_LIBERO_BDDL_FILE", "").strip()
    custom_init = os.environ.get("CONTROLLED_LIBERO_INIT_FILE", "").strip()
    if bool(custom_bddl) != bool(custom_init):
        raise ValueError(
            "CONTROLLED_LIBERO_BDDL_FILE and CONTROLLED_LIBERO_INIT_FILE "
            "must be provided together"
        )
    if custom_bddl and not os.path.isfile(custom_bddl):
        raise FileNotFoundError(f"Controlled LIBERO BDDL not found: {custom_bddl}")
    if custom_init and not os.path.isfile(custom_init):
        raise FileNotFoundError(f"Controlled LIBERO init state not found: {custom_init}")

    bddl_file_path = custom_bddl or os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    if not os.path.exists(bddl_file_path):
        # Fallback: try to locate BDDL files relative to this file
        # This handles cases where get_libero_path returns incorrect relative paths
        here = os.path.dirname(os.path.abspath(__file__))
        # Path: capx/integrations/libero/../../third_party/LIBERO-PRO/libero/libero/bddl_files
        fallback_bddl_root = os.path.abspath(
            os.path.join(here, "..", "..", "third_party", "LIBERO-PRO", "libero", "libero", "bddl_files")
        )
        fallback_path = os.path.join(fallback_bddl_root, task.problem_folder, task.bddl_file)

        if os.path.exists(fallback_path):
            print(f"Found BDDL file at fallback path: {fallback_path}")
            bddl_file_path = fallback_path
        else:
            print(f"Error: BDDL file not found at {bddl_file_path} OR {fallback_path}")

    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": cam_h,
        "camera_widths": cam_w,
        "controller": controller,
        "horizon": horizon,
        "control_freq": control_freq,
        "camera_depths": camera_depths,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    task_language = _select_task_language(
        public_task_language=task.language,
        bddl_path=bddl_file_path,
        custom_bddl=bool(custom_bddl),
    )

    # Handle init states path resolution
    # Libero's get_task_init_states uses get_libero_path("init_states") internally
    # We need to manually load them if the default path fails
    if custom_init:
        import torch

        init_states = torch.load(custom_init)
        print(f"Loaded controlled init states from {custom_init}")
    else:
        try:
            init_states = task_suite.get_task_init_states(task_id)
            print(f"Loaded init states for task {task_id} in suite {suite_name}")
        except (FileNotFoundError, OSError):
            print(f"Warning: Could not load init states for task {task_id} in suite {suite_name}")
            init_states_path = os.path.join(
                get_libero_path("init_states"), task.problem_folder, task.init_states_file
            )
            if not os.path.exists(init_states_path):
                here = os.path.dirname(os.path.abspath(__file__))
                fallback_init_root = os.path.abspath(
                    os.path.join(
                        here, "..", "..", "third_party", "LIBERO-PRO",
                        "libero", "libero", "init_files",
                    )
                )
                fallback_init_path = os.path.join(
                    fallback_init_root, task.problem_folder, task.init_states_file
                )
                if os.path.exists(fallback_init_path):
                    print(f"Found init states file at fallback path: {fallback_init_path}")
                    import torch

                    init_states = torch.load(fallback_init_path)
                else:
                    raise FileNotFoundError(
                        "Init states file not found at "
                        f"{init_states_path} or {fallback_init_path}"
                    )
            else:
                import torch

                init_states = torch.load(init_states_path)

    handle = LiberoHandle(
        env=env,
        suite_name=suite_name,
        task_id=task_id,
        task_language=task_language,
        init_states=init_states,
    )
    return handle
