import json

from evolution.harness.analysis import compare_rollouts
from evolution.harness.schema import Metrics


def _metrics(tmp_path, name, wins):
    folder = tmp_path / name
    folder.mkdir()
    rows = []
    for index in range(3):
        key = f"libero_90/{index}/seed0"
        rows.append(
            {
                "key": key,
                "native_success": key in wins,
                "agent_success": key in wins,
                "turns": index + 1,
                "artifacts": {"episode_dir": str(folder / f"e{index}")},
                "steps": [{"action": "pickplace"}],
            }
        )
    records = folder / "records.jsonl"
    records.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    all_keys = {row["key"] for row in rows}
    return Metrics(
        3,
        len(wins),
        len(wins),
        2,
        1,
        100,
        3,
        0,
        tuple(sorted(wins)),
        tuple(sorted(all_keys - set(wins))),
        str(records),
        name,
    )


def test_paired_delta_keeps_new_wins_and_regressions_separate(tmp_path):
    old = _metrics(tmp_path, "old", {"libero_90/0/seed0"})
    new = _metrics(tmp_path, "new", {"libero_90/1/seed0"})
    delta = compare_rollouts(old, new)
    assert delta["native_delta"] == 0
    assert delta["new_wins"] == ["libero_90/1/seed0"]
    assert delta["regressions"] == ["libero_90/0/seed0"]
    assert {item["change"] for item in delta["changed_episodes"]} == {
        "new_win",
        "regression",
    }
