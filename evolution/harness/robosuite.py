"""Leakage-safe Robosuite curriculum and evaluator for RACaP evolution."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import time
from pathlib import Path

from racap.backends.llm import LLMProviderUnavailableError, LLMQuotaError

from .curriculum import Curriculum
from .evidence import load_metrics
from .prompting import read_prompt
from .runner import EvaluationError, bind_candidate_memory
from .schema import EpisodeRef, EvaluationResult, StageSpec


SUITE = "robosuite_transfer"
TASK_ORDER = (
    "cube_lifting",
    "cube_restack",
    "cube_stack",
    "nut_assembly",
    "spill_wipe",
    "two_arm_handover",
    "two_arm_lift",
)
DEVELOPMENT_TASKS = (
    "cube_lifting",
    "cube_stack",
    "nut_assembly",
    "spill_wipe",
    "two_arm_lift",
)
HELDOUT_TASKS = ("cube_restack", "two_arm_handover")
DEVELOPMENT_SEEDS = (100, 101, 102)
EVALUATION_SEEDS = (0, 1, 2, 3, 4)


def _candidate_source_sha256(root: Path) -> str:
    """Hash candidate-owned source while excluding interpreter/git artifacts."""

    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def episode(task: str, seed: int) -> EpisodeRef:
    if task not in TASK_ORDER:
        raise KeyError(task)
    return EpisodeRef(
        SUITE,
        TASK_ORDER.index(task),
        int(seed),
        {
            "task_name": task,
            "episode_key": f"robosuite/{task}/seed{int(seed)}",
        },
    )


def development_curriculum() -> Curriculum:
    development = tuple(
        episode(task, seed)
        for task in DEVELOPMENT_TASKS
        for seed in DEVELOPMENT_SEEDS
    )
    sealed = tuple(
        [episode(task, seed) for task in HELDOUT_TASKS for seed in DEVELOPMENT_SEEDS]
        + [episode(task, seed) for task in TASK_ORDER for seed in EVALUATION_SEEDS]
    )
    stage = StageSpec(
        id="rs_cross_embodiment",
        title="Robosuite cross-embodiment development",
        capability="general RGB-D manipulation across single- and dual-arm tasks",
        parents=(),
        development=development,
        hints=(
            "Start from the mature LIBERO-90 champion and preserve working behavior.",
            "Use paired visual evidence to isolate embodiment versus strategy failures.",
            "Policy APIs expose physical controls; runtime ReAct owns semantic choices.",
            "Prefer mechanisms shared across tasks and seeds over task-specific scripts.",
        ),
        prompt_file="evolution/prompts/stages/rs_cross_embodiment.md",
        prompt=read_prompt("stages/rs_cross_embodiment.md"),
    )
    curriculum = Curriculum((stage,), sealed)
    curriculum.validate()
    return curriculum


class RobosuiteEvaluationRunner:
    """Run candidate-owned controllers against evaluator-owned Robosuite state."""

    def __init__(
        self,
        racap_root: Path,
        experiment_root: Path,
        *,
        python: Path,
        robosuite_root: Path,
        rats_root: Path,
        capx_root: Path,
        workers: int = 5,
        model: str = "gpt-5.5",
        max_steps: int = 4000,
        record_rollouts: bool = True,
    ) -> None:
        self.racap_root = racap_root.resolve()
        self.experiment_root = experiment_root.resolve()
        # Virtual-environment launchers are commonly symlinks to the system
        # interpreter. Resolving the symlink discards the venv's ``pyvenv.cfg``
        # context and therefore its installed Robosuite dependencies.
        self.python = Path(os.path.abspath(python))
        self.robosuite_root = robosuite_root.resolve()
        self.rats_root = rats_root.resolve()
        self.capx_root = capx_root.resolve()
        self.workers = min(max(1, int(workers)), 5)
        self.model = str(model)
        self.max_steps = int(max_steps)
        self.record_rollouts = bool(record_rollouts)

    def _environment(self, solution_root: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment["CAPX_ENV_STACK"] = "robosuite"
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(self.robosuite_root),
                str(self.racap_root),
                str(self.capx_root),
                str(self.rats_root),
                environment.get("PYTHONPATH", ""),
            )
        )
        bind_candidate_memory(environment, solution_root)
        environment.setdefault(
            "RACAP_LLM_CACHE_DIR", str((self.experiment_root / "llm_cache").resolve())
        )
        environment.setdefault("RACAP_VLM_MAX_CONCURRENCY", "4")
        environment.setdefault(
            "RACAP_VLM_SLOT_DIR", str((self.experiment_root / "vlm_slots").resolve())
        )
        return environment

    def evaluate(
        self,
        solution_root: Path,
        episodes: tuple[EpisodeRef, ...],
        tag: str,
    ) -> EvaluationResult:
        if not episodes:
            raise EvaluationError("cannot evaluate an empty Robosuite cohort")
        if any(item.suite != SUITE for item in episodes):
            raise EvaluationError("Robosuite runner received a mixed-suite cohort")
        task_names = tuple(dict.fromkeys(str(item.context["task_name"]) for item in episodes))
        seeds = tuple(sorted({int(item.seed) for item in episodes}))
        requested = {(str(item.context["task_name"]), int(item.seed)) for item in episodes}
        product = {(task, seed) for task in task_names for seed in seeds}
        if requested != product:
            raise EvaluationError(
                "one Robosuite evaluator call requires a complete task x seed product"
            )
        output_dir = self.experiment_root / "rollouts" / tag
        resume = output_dir.exists()
        command = [
            str(self.python),
            str(self.racap_root / "scripts" / "eval_robosuite_agent.py"),
            "--solution-root",
            str(solution_root.resolve()),
            "--output-dir",
            str(output_dir),
            "--tasks",
            *task_names,
            "--seeds",
            *(str(seed) for seed in seeds),
            "--workers",
            str(min(self.workers, len(episodes))),
            "--model",
            self.model,
            "--max-steps",
            str(self.max_steps),
            "--method-label",
            "racap_rs_evolution_candidate",
            "--protocol-note",
            "Robosuite development-only paired candidate evaluation",
        ]
        if self.record_rollouts:
            command.append("--record-rollouts")
        if resume:
            command.append("--resume")
        log_path = self.experiment_root / "logs" / f"{tag}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = self._environment(solution_root)
        source_sha256_before = _candidate_source_sha256(solution_root)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                # Upstream Robosuite environment constructors resolve their
                # controller JSON relative to the RATS checkout.
                cwd=self.rats_root,
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        seconds = time.time() - started
        source_sha256_after = _candidate_source_sha256(solution_root)
        (output_dir / "frozen_runtime_audit.json").write_text(
            json.dumps(
                {
                    "before_sha256": source_sha256_before,
                    "after_sha256": source_sha256_after,
                    "unchanged": source_sha256_before == source_sha256_after,
                    "solution_root": str(solution_root.resolve()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if source_sha256_before != source_sha256_after:
            raise EvaluationError(
                "candidate source changed during its Robosuite runtime evaluation"
            )
        if process.returncode == 75:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            aborted = output_dir / "ABORTED.json"
            reason = ""
            if aborted.is_file():
                try:
                    reason = str(json.loads(aborted.read_text()).get("reason") or "")
                except (OSError, json.JSONDecodeError):
                    pass
            message = f"Robosuite evaluation paused ({reason or 'provider'}):\n{tail}"
            if reason == "quota_exhausted":
                raise LLMQuotaError(message)
            raise LLMProviderUnavailableError(message)
        if process.returncode:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            raise EvaluationError(
                f"Robosuite evaluation exited {process.returncode}:\n{tail}"
            )
        metrics = load_metrics(output_dir, episodes)
        (output_dir / "evolution_cohort.json").write_text(
            json.dumps(
                {
                    "tag": tag,
                    "tasks": list(task_names),
                    "seeds": list(seeds),
                    "sealed_tasks": list(HELDOUT_TASKS),
                    "sealed_evaluation_seeds": list(EVALUATION_SEEDS),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return EvaluationResult(output_dir, metrics, tuple(command), log_path, seconds)
