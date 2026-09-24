"""Subprocess runner for fresh, paired simulator evaluations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .evidence import load_metrics
from .schema import EpisodeRef, EvaluationResult


class EvaluationError(RuntimeError):
    pass


def bind_candidate_memory(environment: dict[str, str], solution_root: Path) -> None:
    """Bind candidate-owned experience files to both runtime ReAct scopes.

    Evolution stores memory beside ``solution/`` rather than in RACaP's global
    operator-memory location.  Explicitly binding it here makes a memory edit a
    real candidate behavior change and prevents a worker's ambient memory
    variables from leaking into paired evaluation.
    """
    keys = ("RACAP_AGENT_MEMORY", "RACAP_FULL_REACT_MEMORY", "RACAP_TRANSPORT_REACT_MEMORY")
    for key in keys:
        environment.pop(key, None)

    memory_root = solution_root.resolve() / "memory"
    shared = memory_root / "experience.md"
    full = memory_root / "full_react.md"
    transport = memory_root / "transport_react.md"
    if shared.is_file():
        environment["RACAP_AGENT_MEMORY"] = str(shared)
    if full.is_file() or shared.is_file():
        environment["RACAP_FULL_REACT_MEMORY"] = str(full if full.is_file() else shared)
    if transport.is_file() or shared.is_file():
        environment["RACAP_TRANSPORT_REACT_MEMORY"] = str(
            transport if transport.is_file() else shared
        )


class EvaluationRunner:
    def __init__(
        self,
        racap_root: Path,
        experiment_root: Path,
        *,
        workers: int = 4,
        model: str = "gpt-5.5",
        max_steps: int = 8000,
        record_rollouts: bool = True,
    ):
        self.racap_root = racap_root.resolve()
        self.experiment_root = experiment_root.resolve()
        self.workers = workers
        self.model = model
        self.max_steps = max_steps
        self.record_rollouts = record_rollouts

    def evaluate(
        self,
        solution_root: Path,
        episodes: tuple[EpisodeRef, ...],
        tag: str,
    ) -> EvaluationResult:
        if not episodes:
            raise EvaluationError("cannot evaluate an empty cohort")
        output_root = self.experiment_root / "rollouts"
        output_dir = output_root / tag
        if output_dir.exists():
            raise EvaluationError(f"refusing to reuse rollout tag: {output_dir}")
        context_path = self.experiment_root / "contexts" / f"{tag}.json"
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.write_text(
            json.dumps(
                {str(episode.task_id): episode.context for episode in episodes if episode.context},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        suite = episodes[0].suite
        if any(episode.suite != suite for episode in episodes):
            raise EvaluationError("split mixed-suite cohorts before evaluation")
        task_ids = tuple(sorted({episode.task_id for episode in episodes}))
        seeds = tuple(sorted({episode.seed for episode in episodes}))
        requested = {(episode.task_id, episode.seed) for episode in episodes}
        product = {(task_id, seed) for task_id in task_ids for seed in seeds}
        if requested != product:
            raise EvaluationError(
                "one evaluator call requires a complete task x seed product; split sparse cohorts"
            )
        command = [
            sys.executable,
            str(self.racap_root / "scripts" / "eval_full_agent.py"),
            "--suite",
            suite,
            "--task-ids",
            *(str(task_id) for task_id in task_ids),
            "--seeds",
            *(str(seed) for seed in seeds),
            "--workers",
            str(min(self.workers, len(episodes))),
            "--model",
            self.model,
            "--max-steps",
            str(self.max_steps),
            "--solution-root",
            str(solution_root.resolve()),
            "--episode-context",
            str(context_path),
            "--tag",
            tag,
            "--output-dir",
            str(output_root),
        ]
        if self.record_rollouts:
            command.append("--record-rollouts")
        log_path = self.experiment_root / "logs" / f"{tag}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(self.racap_root),
                str(self.racap_root / "third_party" / "rats"),
                environment.get("PYTHONPATH", ""),
            )
        )
        bind_candidate_memory(environment, solution_root)
        # Baseline and candidates in one experiment share exact perception
        # replies.  Hosted-model variation must not masquerade as code gain.
        environment.setdefault(
            "RACAP_LLM_CACHE_DIR", str((self.experiment_root / "llm_cache").resolve())
        )
        # Simulator parallelism is cheap; hosted VLM concurrency is externally
        # rate-limited.  The backend enforces this across spawned processes.
        environment.setdefault("RACAP_VLM_MAX_CONCURRENCY", "4")
        environment.setdefault(
            "RACAP_VLM_SLOT_DIR", str((self.experiment_root / "vlm_slots").resolve())
        )

        if environment.get("RACAP_EVAL_HEALTH_PROBE", "1") != "0":
            probe_model = environment.get("RACAP_GROUNDER_MODEL", "gpt-5.5")
            probe = subprocess.run(
                [
                    sys.executable,
                    str(self.racap_root / "scripts" / "probe_vlm.py"),
                    "--model",
                    probe_model,
                ],
                cwd=self.racap_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if probe.returncode:
                raise EvaluationError(
                    "VLM health probe failed before simulator rollout; evaluation "
                    f"was not scored:\n{probe.stdout[-2000:]}"
                )
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=self.racap_root,
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        seconds = time.time() - started
        if process.returncode:
            tail = log_path.read_text(encoding="utf-8")[-4000:]
            raise EvaluationError(f"evaluation exited {process.returncode}:\n{tail}")
        metrics = load_metrics(output_dir, episodes)
        return EvaluationResult(output_dir, metrics, tuple(command), log_path, seconds)
