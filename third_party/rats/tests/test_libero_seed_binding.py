from __future__ import annotations

from rats.integrations.libero import LiberoHandle, trial_seed_from_public_seed


class _FakeLiberoEnv:
    def __init__(self) -> None:
        self.seed_calls: list[int | None] = []
        self.reset_calls = 0
        self.applied_states: list[object] = []

    def seed(self, seed: int | None) -> None:
        self.seed_calls.append(seed)

    def reset(self) -> str:
        self.reset_calls += 1
        return "default-observation"

    def set_init_state(self, state: object) -> tuple[str, object]:
        self.applied_states.append(state)
        return ("registered-observation", state)


def test_public_seed_maps_to_matching_registered_init_state() -> None:
    states = [object(), object(), object()]
    raw = _FakeLiberoEnv()
    handle = LiberoHandle(
        env=raw,
        suite_name="libero_spatial_swap",
        task_id=0,
        task_language="test task",
        init_states=states,
    )

    for public_seed, expected_index in ((0, 0), (1, 1), (2, 2)):
        trial_seed = trial_seed_from_public_seed(public_seed)
        observation, info = handle.reset(seed=trial_seed)
        assert info["init_state_index"] == expected_index
        assert raw.applied_states[-1] is states[expected_index]
        assert observation == ("registered-observation", states[expected_index])

    assert raw.seed_calls == [1, 2, 3]
    assert raw.reset_calls == 3
    assert len(raw.applied_states) == 3
