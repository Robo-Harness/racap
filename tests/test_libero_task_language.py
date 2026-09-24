from __future__ import annotations

import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = [
    ROOT
    / "third_party"
    / "rats"
    / "capx-baseline"
    / "capx"
    / "integrations"
    / "libero"
    / "__init__.py",
    ROOT
    / "third_party"
    / "rats"
    / "rats"
    / "integrations"
    / "libero"
    / "__init__.py",
]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_official_task_metadata_overrides_stale_bddl_language(
    adapter: Path, tmp_path: Path
) -> None:
    bddl = tmp_path / "task.bddl"
    bddl.write_text(
        "(define (problem x) (:language stack the middle bowl on the front bowl))",
        encoding="utf-8",
    )
    select = runpy.run_path(str(adapter))["_select_task_language"]

    assert select(
        public_task_language="stack the middle bowl on the back bowl",
        bddl_path=str(bddl),
        custom_bddl=False,
    ) == "stack the middle bowl on the back bowl"


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_controlled_custom_bddl_owns_its_instruction(
    adapter: Path, tmp_path: Path
) -> None:
    bddl = tmp_path / "custom.bddl"
    bddl.write_text(
        "(define (problem x) (:language put all objects in the basket))",
        encoding="utf-8",
    )
    select = runpy.run_path(str(adapter))["_select_task_language"]

    assert select(
        public_task_language="unused parent scene instruction",
        bddl_path=str(bddl),
        custom_bddl=True,
    ) == "put all objects in the basket"
