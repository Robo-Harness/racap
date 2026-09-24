"""Derive evolution episodes from LIBERO goal predicates.

An episode is one concrete invocation of a Policy API against one task and seed,
plus the native predicate that decides whether it worked. Deriving the arguments
from the bddl goal instead of hand-listing them keeps the training set honest:
the labels handed to the policy are exactly the objects the benchmark scores,
and adding a suite costs nothing.

Only tasks whose entire goal is a single predicate are used. A multi-predicate
task cannot attribute credit to one skill invocation.

Capability variation comes from task and position-perturbation suites.
Seeds index benchmark-provided initial states; they do not imply independent
scene distributions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from racap.envs.paths import bddl_root

_REGION_SUFFIXES = (
    "_contain_region",
    "_cook_region",
    "_heating_region",
    "_top_region",
    "_middle_region",
    "_bottom_region",
    "_back_contain_region",
    "_front_contain_region",
    "_top_side",
    "_init_region",
    "_region",
)


def humanize(entity: str) -> str:
    """Turn a bddl entity name into the natural-language label a VLM can ground.

    ``alphabet_soup_1`` -> ``alphabet soup``;
    ``white_cabinet_1_bottom_region`` -> ``white cabinet``.
    """
    name = entity
    for suffix in _REGION_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(r"_\d+$", "", name)
    return name.replace("_", " ").strip()


# Perturbation strengths that move one object and leave the rest of the scene
# alone. At 0.3 and above LIBERO-PRO resolves the collision it creates by
# teleporting the obstructing object 10 m away, which removes a distractor at
# the same time as it moves the target -- two effects in opposite directions on
# one axis. Those levels are for boundary scanning, not for training.
CLEAN_PERTURBATIONS = ("temp_x0.1", "temp_x0.2", "temp_y0.1", "temp_y0.2")


def variants(suite: str, perturbations: tuple[str, ...] = CLEAN_PERTURBATIONS) -> list[str]:
    """The base suite plus every perturbation of it that is registered."""
    from libero.benchmark import get_benchmark_dict

    known = set(get_benchmark_dict())
    return [suite] + [name for p in perturbations if (name := f"{suite}_{p}") in known]


def region_part(entity: str) -> str:
    """Extract which part of an articulated receptacle a region refers to."""
    for part in ("bottom", "middle", "top", "back", "front", "cook", "heating"):
        if f"_{part}_" in entity or entity.endswith(f"_{part}_region"):
            return part
    return ""


# Where a destination phrase starts inside an instruction. Searched from the
# right, because every instruction here names the target first and the
# destination last: "put the book in the middle on the cabinet shelf" must
# yield the shelf, not the middle.
_DESTINATION_PREPOSITIONS = (" into ", " onto ", " inside ", " in ", " on ", " to ")

# LIBERO-PRO task perturbations rewrite the public ``(:language ...)`` field
# together with the goal, but retain the original BDDL filename.  Upstream
# ``Task.language`` is derived from that filename, so it is stale precisely for
# these suites.  Other perturbations do not change task semantics and continue
# to use the benchmark metadata wording for compatibility with completed runs.
_TASK_PERTURBATION_SUITES = frozenset(
    {
        "libero_spatial_task",
        "libero_goal_task",
        "libero_object_task",
        "libero_10_task",
    }
)


def destination_phrase(instruction: str, fallback: str) -> str:
    """The destination as the instruction words it.

    Humanising the bddl entity loses exactly what distinguishes one of these
    tasks from the next. ``white_cabinet_1_bottom_region`` and
    ``white_cabinet_1_top_region`` both become "white cabinet", so a policy
    handed that label cannot tell the bottom drawer from the top one however
    good its perception is; ``plate_1`` and ``plate_2`` both become "plate" in
    a scene holding two plates; and ``kitchen_table_plate_right_region``
    becomes "kitchen table plate right", which describes nothing.

    The instruction has the missing information and is not privileged -- it is
    what a runtime agent reads before choosing arguments. Taking the phrase
    after the last preposition reproduces the call that agent would make.
    """
    text = " " + " ".join(str(instruction).split()).lower().rstrip(".") + " "
    cut = max(
        (text.rfind(preposition), len(preposition)) for preposition in _DESTINATION_PREPOSITIONS
    )
    if cut[0] < 0:
        return fallback
    phrase = text[cut[0] + cut[1] :].strip()
    return phrase or fallback


@dataclass(frozen=True)
class Episode:
    suite: str
    task_id: int
    seed: int
    skill: str
    pick_label: str
    place_label: str
    place_part: str = ""
    instruction: str = ""
    predicate: tuple[str, ...] = field(default_factory=tuple)
    place_phrase: str = ""

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.task_id}/seed{self.seed}"

    @property
    def destination(self) -> str:
        return self.place_phrase or self.place_label

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "task_id": self.task_id,
            "seed": self.seed,
            "skill": self.skill,
            "pick_label": self.pick_label,
            "place_label": self.place_label,
            "place_part": self.place_part,
            "instruction": self.instruction,
            "predicate": list(self.predicate),
            "place_phrase": self.place_phrase,
        }


@dataclass(frozen=True)
class TaskEpisode:
    """One complete natural-language task, including multi-predicate tasks.

    The predicates are retained only in evaluation metadata.  The full ReAct
    planner receives ``instruction`` and camera images, never this answer-key
    field.
    """

    suite: str
    task_id: int
    seed: int
    instruction: str
    instruction_source: str = "benchmark_task_language"
    predicates: tuple[tuple[str, ...], ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.task_id}/seed{self.seed}"

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "task_id": self.task_id,
            "seed": self.seed,
            "instruction": self.instruction,
            "instruction_source": self.instruction_source,
            "predicates": [list(value) for value in self.predicates],
        }


def _goal_predicates(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    text = path.read_text()
    m = re.search(r"\(:goal(.*?)\n\s*\)\s*\n\)", text, re.S)
    goal = " ".join(m.group(1).split()) if m else ""
    return [
        (g.group(1), tuple(g.group(2).split()))
        for g in re.finditer(r"\((On|In|Open|Close|Turnon|Turnoff)\s+([^()]+)\)", goal)
    ]


def _bddl_language(path: Path) -> str:
    """Read the public natural-language instruction embedded in a BDDL file."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\(:language\s+([^\n\r)]*)\)", text, re.IGNORECASE)
    return " ".join(match.group(1).split()) if match else ""


def task_instruction(suite: str, task: object, path: Path) -> tuple[str, str]:
    """Return the authoritative public instruction and its audited source.

    Task-perturbation suites must use the language written by the perturbator
    into BDDL because their filenames intentionally remain those of the source
    tasks.  The benchmark metadata is authoritative everywhere else.
    """
    benchmark_language = " ".join(str(getattr(task, "language", "")).split())
    if suite in _TASK_PERTURBATION_SUITES:
        language = _bddl_language(path)
        if not language:
            raise ValueError(f"missing (:language ...) in task-perturbation BDDL: {path}")
        return language, "bddl_task_perturbation_language"
    return benchmark_language, "benchmark_task_language"


def build_episodes(
    suite: str,
    *,
    seeds: tuple[int, ...] = (0,),
    skills: tuple[str, ...] = ("pickplace",),
    task_ids: tuple[int, ...] | None = None,
) -> list[Episode]:
    """Enumerate single-predicate tasks of ``suite`` as episodes for ``skills``."""
    from libero.benchmark import get_benchmark_dict

    benchmark = get_benchmark_dict()[suite]()
    episodes: list[Episode] = []
    for task_id in range(benchmark.n_tasks):
        if task_ids is not None and task_id not in task_ids:
            continue
        task = benchmark.get_task(task_id)
        task_path = bddl_root() / suite / task.bddl_file
        instruction, _instruction_source = task_instruction(suite, task, task_path)
        preds = _goal_predicates(task_path)
        if len(preds) != 1:
            continue
        pred, pargs = preds[0]

        if pred in ("In", "On") and len(pargs) == 2:
            skill = "pickplace"
            pick, place = humanize(pargs[0]), humanize(pargs[1])
            part = region_part(pargs[1])
        elif pred in ("Open", "Close") and len(pargs) == 1:
            skill = "open_articulated" if pred == "Open" else "close_articulated"
            pick, place = "", humanize(pargs[0])
            part = region_part(pargs[0])
        elif pred in ("Turnon", "Turnoff") and len(pargs) == 1:
            skill = "press_button"
            pick, place = "", humanize(pargs[0])
            part = ""
        else:
            continue

        if skill not in skills or (skill == "pickplace" and pick == place):
            continue

        episodes.extend(
            Episode(
                suite=suite,
                task_id=task_id,
                seed=seed,
                skill=skill,
                pick_label=pick,
                place_label=place,
                place_part=part,
                instruction=instruction,
                predicate=(pred, *pargs),
                place_phrase=destination_phrase(instruction, place),
            )
            for seed in seeds
        )
    return episodes


def build_task_episodes(
    suite: str,
    *,
    seeds: tuple[int, ...] = (0,),
    task_ids: tuple[int, ...] | None = None,
) -> list[TaskEpisode]:
    """Enumerate complete tasks without dropping multi-predicate goals."""
    from libero.benchmark import get_benchmark_dict

    benchmark = get_benchmark_dict()[suite]()
    episodes: list[TaskEpisode] = []
    for task_id in range(benchmark.n_tasks):
        if task_ids is not None and task_id not in task_ids:
            continue
        task = benchmark.get_task(task_id)
        task_path = bddl_root() / suite / task.bddl_file
        instruction, instruction_source = task_instruction(suite, task, task_path)
        predicates = _goal_predicates(task_path)
        packed = tuple((pred, *args) for pred, args in predicates)
        episodes.extend(
            TaskEpisode(
                suite=suite,
                task_id=task_id,
                seed=seed,
                instruction=instruction,
                instruction_source=instruction_source,
                predicates=packed,
            )
            for seed in seeds
        )
    return episodes


def base_suite(suite: str) -> str:
    """The unperturbed suite a perturbed one was derived from."""
    return re.sub(r"_temp_[xy]\d+(\.\d+)?$", "", suite)


def split_train_eval(
    episodes: list[Episode], *, holdout_every: int = 3
) -> tuple[list[Episode], list[Episode]]:
    """Hold out every ``holdout_every``-th task so the curve is not self-graded.

    The split is by the *unperturbed* task. A perturbation shifts one object by
    a few centimetres and changes nothing else, so putting ``task 0`` in
    training and ``task 0 shifted 7 cm`` in the holdout would be scoring the
    policy on a scene it had just been tuned on.
    """
    tasks = sorted({(base_suite(e.suite), e.task_id) for e in episodes})
    holdout = {t for i, t in enumerate(tasks) if i % holdout_every == holdout_every - 1}
    train = [e for e in episodes if (base_suite(e.suite), e.task_id) not in holdout]
    evalset = [e for e in episodes if (base_suite(e.suite), e.task_id) in holdout]
    return train, evalset
