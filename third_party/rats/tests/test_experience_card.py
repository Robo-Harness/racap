from rats.agents.experience_card import append_experience_card, read_experience_card


def test_missing_card_is_noop(monkeypatch):
    monkeypatch.delenv("CONTROLLED_ONE_SHOT_EXPERIENCE_CARD", raising=False)
    assert read_experience_card() == ""
    assert append_experience_card("base") == "base"


def test_card_is_appended_as_advisory_context(tmp_path, monkeypatch):
    card = tmp_path / "card.md"
    card.write_text("approach from a clearer view", encoding="utf-8")
    monkeypatch.setenv("CONTROLLED_ONE_SHOT_EXPERIENCE_CARD", str(card))

    result = append_experience_card("existing lesson")

    assert result.startswith("existing lesson")
    assert "current visual evidence wins" in result
    assert result.endswith("approach from a clearer view")
