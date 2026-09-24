"""Register LIBERO-PRO suites that ship as data but never got a benchmark class.

LIBERO-PRO's ``benchmark/__init__.py`` hardcodes a ``libero_suites`` list and a
``libero_task_map`` dict, and only suites named there become loadable. Several
directories under ``bddl_files/`` — most importantly the position-strength
sweep ``libero_{suite}_temp_{x,y}0.1 … 0.5``, which is the natural grid for
capability-boundary scanning — have complete bddl and init_files but no entry,
so loading them raises ``KeyError``.

This module fills in the missing entries at runtime instead of forking
LIBERO-PRO. Call :func:`register_missing_suites` before loading a task.

One consequence of LIBERO-PRO's design is worth knowing: task instructions come
from ``grab_language_from_filename``, not from the bddl ``(:language ...)``
field. Suites whose filenames were truncated therefore carry truncated
instructions. :func:`audit_instructions` reports them.
"""

from __future__ import annotations

import functools
from racap.envs.paths import bddl_root, init_root


@functools.lru_cache(maxsize=1)
def register_missing_suites(require_init_files: bool = True) -> dict[str, str]:
    """Register every bddl directory that has no benchmark class yet.

    Suites without matching ``init_files`` are skipped by default: LIBERO needs
    the init states to place objects, and loading one raises a less obvious
    error later.

    Returns a suite name to status mapping.
    """
    from libero.benchmark import (
        Benchmark,
        Task,
        libero_suites,
        register_benchmark,
        task_maps,
    )
    from libero.benchmark import grab_language_from_filename
    from libero.benchmark.libero_suite_task_map import libero_task_map

    status: dict[str, str] = {}
    bddls_root = bddl_root()
    inits_root = init_root()
    for path in sorted(p for p in bddls_root.iterdir() if p.is_dir()):
        suite = path.name
        if suite in task_maps:
            status[suite] = "already_registered"
            continue

        bddls = sorted(f.stem for f in path.iterdir() if f.suffix == ".bddl")
        if not bddls:
            status[suite] = "skipped_no_bddl"
            continue
        if require_init_files and not (inits_root / suite).is_dir():
            status[suite] = "skipped_no_init_files"
            continue

        libero_task_map[suite] = bddls
        if suite not in libero_suites:
            libero_suites.append(suite)
        task_maps[suite] = {
            name: Task(
                name=name,
                language=grab_language_from_filename(name + ".bddl"),
                problem="Libero",
                problem_folder=suite,
                bddl_file=name + ".bddl",
                init_states_file=name + ".pruned_init",
            )
            for name in bddls
        }

        # register_benchmark keys off ``__name__.lower()``, so the class must be
        # named exactly like the suite -- including dots, which type() allows.
        cls = type(
            suite,
            (Benchmark,),
            {
                "__init__": lambda self, task_order_index=0, _s=suite: (
                    Benchmark.__init__(self, task_order_index=task_order_index),
                    setattr(self, "name", _s),
                    self._make_benchmark(),
                )[0]
            },
        )
        register_benchmark(cls)
        status[suite] = f"registered ({len(bddls)} tasks)"

    return status


def audit_instructions() -> list[dict[str, object]]:
    """Flag suites whose filename-derived instructions were truncated.

    The telltale signature is a whole suite whose task names all sit at the
    exact same length: that is a hard character cap applied when the files were
    generated, not a coincidence. Suites with naturally varying name lengths are
    fine. Only suites with more than one task can show the signature.
    """
    import re

    rows = []
    for path in sorted(p for p in bddl_root().iterdir() if p.is_dir()):
        stems = [
            re.sub(r"^[A-Z0-9_]*SCENE\d*_", "", f.stem)
            for f in sorted(path.iterdir())
            if f.suffix == ".bddl"
        ]
        lengths = {len(s) for s in stems}
        if len(stems) < 2 or len(lengths) != 1:
            continue
        rows.append(
            {
                "suite": path.name,
                "tasks": len(stems),
                "capped_at": lengths.pop(),
                "examples": [s.replace("_", " ") for s in stems[:3]],
            }
        )
    return rows
