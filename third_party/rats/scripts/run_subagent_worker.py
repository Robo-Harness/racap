"""Subprocess entrypoint for one parallel sub-agent practice run.

The orchestrator (``loop/lifelong_loop.py:_dispatch_parallel_subagents``)
spawns N of these as separate processes, one per approach the diagnoser
proposed. Each worker:

  1. Builds its OWN ``FrankaLiberoCodeEnv`` from the same BDDL the main
     process uses (parallel processes can't share a MuJoCo env).
  2. Runs ``SubAgent.run`` on the assigned (subgoal, approach_directive)
     for up to ``--max-retries`` tries.
  3. Writes the outcome to ``--output-json``.

When any worker reports success, the orchestrator SIGKILLs the rest.
The worker is therefore designed to be killable at any point — it
incrementally rewrites ``--output-json`` after each attempt so the
parent always has a current view of progress.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

# The vendored robosuite hard-codes /tmp/robosuite.log as its log
# destination; on a shared box that file is owned by whoever launched
# robosuite first, so a fresh subprocess crashes on import. Redirect to
# a per-pid file in /tmp BEFORE any robosuite import runs.
_OrigFH = logging.FileHandler


class _RedirFH(_OrigFH):
    def __init__(self, filename, *a, **k):
        if filename == "/tmp/robosuite.log":
            try:
                user = os.getlogin()
            except OSError:
                user = str(os.getuid())
            filename = os.path.join(
                tempfile.gettempdir(),
                f"robosuite_{user}_{os.getpid()}.log",
            )
        super().__init__(filename, *a, **k)


logging.FileHandler = _RedirFH

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "rats" / "third_party" / "LIBERO-PRO"))


def _write_result(out_path: Path, payload: dict) -> None:
    """Atomic write so the parent never reads a half-written file."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-path", required=True)
    parser.add_argument(
        "--apis", required=True,
        help="comma-separated FrankaLibero* API class names",
    )
    parser.add_argument("--privileged", action="store_true")
    parser.add_argument("--subgoal", required=True)
    parser.add_argument("--approach", required=True)
    parser.add_argument(
        "--scene-context-json", required=True,
        help="path to a JSON file holding the parent's scene_context",
    )
    parser.add_argument("--skill-library", required=True)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--video-tag", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--parent-task-name", default="")
    parser.add_argument("--execution-timeout", type=int, default=600)
    # iteration_seed must match the parent's _reset_env seed so that the
    # worker's MuJoCo placement_initializer samples the SAME object
    # positions as the parent's later main attempts. Without this the
    # worker practices on a randomly-jittered scene and its winning
    # script doesn't transfer back. See loop/lifelong_loop.py probe note.
    parser.add_argument("--iteration-seed", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Pre-write a "running" stub so the parent can poll without races.
    _write_result(out_path, {
        "status": "running",
        "approach": args.approach,
        "subgoal": args.subgoal,
        "video_tag": args.video_tag,
        "success": False,
    })

    try:
        from rats.agents.environment_creator import build_full_libero_env_from_bddl
        from rats.agents.failure_diagnoser import FailureDiagnoser
        from rats.agents.subagent import SubAgent
        from rats.executor.sandbox import Executor
        from skill_library.library import SkillLibrary

        env = build_full_libero_env_from_bddl(
            bddl_path=args.bddl_path,
            api_names=args.apis.split(","),
            privileged=args.privileged,
        )
        # Force the seeded placement initializer immediately. The
        # constructor already calls reset() once but unseeded; that
        # leaves the env in a randomly-jittered state different from the
        # parent's seeded state. Re-reset with the parent's iteration
        # seed so worker and parent share the same MuJoCo placement.
        try:
            env.reset(seed=args.iteration_seed)
        except TypeError:
            env.reset()

        scene_context = json.loads(
            Path(args.scene_context_json).read_text(),
        )

        skill_library = SkillLibrary(storage_path=args.skill_library)
        diagnoser = FailureDiagnoser()
        executor = Executor(timeout_seconds=args.execution_timeout)
        sub_agent = SubAgent(max_retries=args.max_retries)

        def _reset_env() -> None:
            try:
                env.reset(seed=args.iteration_seed)
            except TypeError:
                env.reset()

        result = sub_agent.run(
            subgoal=args.subgoal,
            env=env,
            scene_context=scene_context,
            diagnoser=diagnoser,
            executor=executor,
            reset_env=_reset_env,
            video_dir=Path(args.video_dir),
            video_tag=args.video_tag,
            parent_task_name=args.parent_task_name or None,
            approach_directive=args.approach,
        )

        _write_result(out_path, {
            "status": "done",
            "success": bool(result.get("success")),
            "code": result.get("code") or "",
            "approach": args.approach,
            "subgoal": args.subgoal,
            "attempts_used": result.get("attempts"),
            "video_tag": args.video_tag,
        })
        return 0 if result.get("success") else 1

    except KeyboardInterrupt:
        # Orchestrator sent SIGINT — another sibling probably won.
        _write_result(out_path, {
            "status": "cancelled",
            "approach": args.approach,
            "subgoal": args.subgoal,
            "video_tag": args.video_tag,
            "success": False,
        })
        return 130
    except BaseException as e:
        _write_result(out_path, {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "approach": args.approach,
            "subgoal": args.subgoal,
            "video_tag": args.video_tag,
            "success": False,
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
