from rats.integrations.franka.libero_reduced_skill_library import _runtime_vlm_model


def test_runtime_vlm_override_has_priority(monkeypatch):
    monkeypatch.setenv("RATS_LLM_MODEL", "gpt-5.5")
    monkeypatch.setenv("RATS_RUNTIME_VLM_MODEL", "runtime-test-model")
    assert _runtime_vlm_model() == "runtime-test-model"


def test_runwide_model_is_used_without_runtime_override(monkeypatch):
    monkeypatch.delenv("RATS_RUNTIME_VLM_MODEL", raising=False)
    monkeypatch.setenv("RATS_LLM_MODEL", "gpt-5.5")
    assert _runtime_vlm_model() == "gpt-5.5"


def test_upstream_default_is_retained_for_unbound_standalone_runs(monkeypatch):
    monkeypatch.delenv("RATS_RUNTIME_VLM_MODEL", raising=False)
    monkeypatch.delenv("RATS_LLM_MODEL", raising=False)
    assert _runtime_vlm_model() == "google/gemini-3.1-pro-preview"
