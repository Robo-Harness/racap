from __future__ import annotations

from racap.agent.experience import load_runtime_memory, with_runtime_memory


def test_runtime_memory_can_be_overridden_and_reloaded(tmp_path, monkeypatch) -> None:
    memory = tmp_path / "memory.md"
    memory.write_text("first general lesson", encoding="utf-8")
    monkeypatch.setenv("RACAP_AGENT_MEMORY", str(memory))

    assert load_runtime_memory() == "first general lesson"
    memory.write_text("updated between episodes", encoding="utf-8")
    assert load_runtime_memory() == "updated between episodes"


def test_runtime_memory_is_appended_as_advisory_context() -> None:
    prompt = with_runtime_memory("base prompt", "preserve the scene")

    assert prompt.startswith("base prompt")
    assert "not privileged state" in prompt
    assert prompt.endswith("preserve the scene")


def test_full_and_transport_memory_can_be_scoped_independently(
    tmp_path,
    monkeypatch,
) -> None:
    full = tmp_path / "full.md"
    transport = tmp_path / "transport.md"
    shared = tmp_path / "shared.md"
    full.write_text("plan causal prerequisites", encoding="utf-8")
    transport.write_text("recover an empty grasp", encoding="utf-8")
    shared.write_text("shared fallback", encoding="utf-8")
    monkeypatch.setenv("RACAP_AGENT_MEMORY", str(shared))
    monkeypatch.setenv("RACAP_FULL_REACT_MEMORY", str(full))
    monkeypatch.setenv("RACAP_TRANSPORT_REACT_MEMORY", str(transport))

    assert load_runtime_memory(scope="full") == "plan causal prerequisites"
    assert load_runtime_memory(scope="transport") == "recover an empty grasp"
    assert load_runtime_memory() == "shared fallback"


def test_one_trial_card_is_appended_without_replacing_base_memory(
    tmp_path,
    monkeypatch,
) -> None:
    memory = tmp_path / "memory.md"
    card = tmp_path / "card.md"
    memory.write_text("base invariant", encoding="utf-8")
    card.write_text("target-domain observation", encoding="utf-8")
    monkeypatch.setenv("RACAP_AGENT_MEMORY", str(memory))
    monkeypatch.setenv("CONTROLLED_ONE_SHOT_EXPERIENCE_CARD", str(card))

    result = load_runtime_memory()

    assert "base invariant" in result
    assert "ONE-TRIAL TARGET-DOMAIN EXPERIENCE CARD" in result
    assert result.endswith("target-domain observation")
