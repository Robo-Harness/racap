from __future__ import annotations

import pandas as pd
import pytest

from experiments.controlled_comparison.consolidate_main_results import (
    _coverage_frame,
    _validate,
)


def _frames(methods: list[str], episodes_per_method: int = 2):
    rows = []
    complete = []
    for method in methods:
        for index in range(episodes_per_method):
            key = f"suite/{index}/seed0"
            rows.append(
                {
                    "method": method,
                    "episode_key": key,
                    "instruction_match": True,
                    "simulator_episode_resets": 1,
                    "actual_models": "gpt-5.5",
                }
            )
            complete.append(
                {
                    "method": method,
                    "episode_key": key,
                    "completed": True,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(complete)


def test_validate_accepts_exact_heterogeneous_table_contract():
    methods = ["capx", "racap_phase1"]
    episodes, complete = _frames(methods)
    _validate(
        episodes,
        complete,
        methods=methods,
        episodes_per_method=2,
        name="test",
    )


def test_validate_rejects_duplicate_episode_evidence():
    methods = ["capx", "racap_phase1"]
    episodes, complete = _frames(methods)
    episodes = pd.concat([episodes, episodes.iloc[[0]]], ignore_index=True)
    with pytest.raises(SystemExit, match="duplicate episode evidence"):
        _validate(
            episodes,
            complete,
            methods=methods,
            episodes_per_method=2,
            name="test",
        )


def test_validate_rejects_string_false_instruction_match():
    methods = ["capx", "racap_phase1"]
    episodes, complete = _frames(methods)
    episodes["instruction_match"] = "True"
    episodes.loc[0, "instruction_match"] = "False"
    with pytest.raises(SystemExit, match="delivered-instruction mismatch"):
        _validate(
            episodes,
            complete,
            methods=methods,
            episodes_per_method=2,
            name="test",
        )


def test_coverage_contract_never_imputes_champion_full_grid():
    coverage = _coverage_frame()
    row = coverage[
        (coverage["method"] == "racap_phase2")
        & (coverage["table"] == "main_four_method")
    ].iloc[0]
    assert row["episodes_expected"] == 0
    assert row["coverage"] == "not run; never imputed"
    pro = coverage[
        (coverage["method"] == "racap_phase2")
        & (coverage["table"] == "pro_five_method")
    ].iloc[0]
    assert pro["episodes_expected"] == 180


def test_coverage_contract_discloses_phase1_libero90_reuse():
    coverage = _coverage_frame()
    row = coverage[
        (coverage["method"] == "racap_phase1")
        & (coverage["table"] == "main_four_method")
    ].iloc[0]
    assert row["episodes_expected"] == 350
    assert "archived LIBERO-90 reused" in row["coverage"]
