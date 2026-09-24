"""Command line interface for reproducible RACaP evolution experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .coder import CodingAgent
from .critic import VisualCritic
from .curriculum import default_curriculum
from .orchestrator import EvolutionOrchestrator
from .runner import EvaluationRunner
from .scheduler import CurriculumScheduler, SchedulerConfig


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RACaP strong-start evolution harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan", help="print the development curriculum (sealed set omitted)"
    )
    plan.add_argument("--stage", default=None)

    run = subparsers.add_parser("run", help="run paired coding/simulator evolution iterations")
    run.add_argument("--experiment", required=True)
    run.add_argument(
        "--stage",
        default="auto",
        help="auto schedules the capability DAG; a concrete stage pins this run",
    )
    run.add_argument(
        "--start-stage",
        default="s0_direct_transport",
        help="initial evidence distribution for a new auto-scheduled experiment",
    )
    run.add_argument("--iterations", type=int, default=1)
    run.add_argument(
        "--runtime-candidates",
        type=int,
        default=None,
        help=(
            "run until this many candidates reach simulator evaluation; model, "
            "patch and contract failures do not consume this experimental budget"
        ),
    )
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--critic-limit", type=int, default=8)
    run.add_argument("--max-steps", type=int, default=8000)
    run.add_argument("--no-video", action="store_true")
    run.add_argument("--stage-min-visits", type=int, default=2)
    run.add_argument("--stage-patience", type=int, default=3)
    run.add_argument("--stage-exploration", type=float, default=0.45)
    run.add_argument(
        "--implementation-repairs",
        type=int,
        default=2,
        help="same-coder compile/test repair rounds before abandoning a proposal",
    )
    run.add_argument(
        "--smoke-episodes",
        type=int,
        default=1,
        help="runtime episodes used only to catch unconnected APIs/exceptions before paired eval",
    )
    run.add_argument(
        "--coder-model", default=os.environ.get("RACAP_CODER_MODEL", "gpt-5.6-sol")
    )
    run.add_argument("--critic-model", default=os.environ.get("RACAP_CRITIC_MODEL", "gpt-5.5"))
    run.add_argument("--runtime-model", default=os.environ.get("RACAP_MODEL", "gpt-5.5"))

    status = subparsers.add_parser(
        "status", help="show the machine-readable state of an experiment"
    )
    status.add_argument("--experiment", required=True)

    args = parser.parse_args(argv)
    racap_root = _root()
    curriculum = default_curriculum()
    if args.command == "plan":
        value = curriculum.get(args.stage).to_dict() if args.stage else curriculum.to_dict()
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return
    experiments = racap_root / "outputs" / "evolution"
    experiment_root = experiments / args.experiment
    if args.command == "status":
        state = experiment_root / "lineage" / "state.json"
        if not state.is_file():
            raise SystemExit(f"experiment state does not exist: {state}")
        print(state.read_text(encoding="utf-8"))
        return

    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative")
    if args.runtime_candidates is not None and args.runtime_candidates < 0:
        raise SystemExit("--runtime-candidates must be non-negative")
    if args.implementation_repairs < 0:
        raise SystemExit("--implementation-repairs must be non-negative")
    if args.smoke_episodes < 0:
        raise SystemExit("--smoke-episodes must be non-negative")
    if args.stage_min_visits < 0:
        raise SystemExit("--stage-min-visits must be non-negative")
    if args.stage_patience < 0:
        raise SystemExit("--stage-patience must be non-negative")
    if args.stage_exploration < 0:
        raise SystemExit("--stage-exploration must be non-negative")
    runner = EvaluationRunner(
        racap_root,
        experiment_root,
        workers=args.workers,
        model=args.runtime_model,
        max_steps=args.max_steps,
        record_rollouts=not args.no_video,
    )
    orchestrator = EvolutionOrchestrator(
        racap_root,
        experiment_root,
        curriculum,
        runner,
        CodingAgent(args.coder_model),
        VisualCritic(args.critic_model),
        critic_limit=args.critic_limit,
        implementation_repairs=args.implementation_repairs,
        smoke_episodes=args.smoke_episodes,
        scheduler=CurriculumScheduler(
            curriculum,
            SchedulerConfig(
                min_visits=args.stage_min_visits,
                patience=args.stage_patience,
                exploration=args.stage_exploration,
            ),
        ),
    )
    state = orchestrator.run(
        args.stage,
        args.iterations,
        start_stage_id=args.start_stage,
        runtime_candidates=args.runtime_candidates,
    )
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
