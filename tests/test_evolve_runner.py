from evolution.harness.runner import bind_candidate_memory


def test_candidate_shared_memory_binds_both_react_scopes(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    shared = memory / "experience.md"
    shared.write_text("tray priors", encoding="utf-8")
    environment = {
        "RACAP_AGENT_MEMORY": "/ambient/shared",
        "RACAP_FULL_REACT_MEMORY": "/ambient/full",
        "RACAP_TRANSPORT_REACT_MEMORY": "/ambient/transport",
    }

    bind_candidate_memory(environment, tmp_path)

    assert environment["RACAP_AGENT_MEMORY"] == str(shared)
    assert environment["RACAP_FULL_REACT_MEMORY"] == str(shared)
    assert environment["RACAP_TRANSPORT_REACT_MEMORY"] == str(shared)


def test_candidate_scoped_memory_overrides_shared_memory(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    shared = memory / "experience.md"
    full = memory / "full_react.md"
    transport = memory / "transport_react.md"
    for path in (shared, full, transport):
        path.write_text(path.name, encoding="utf-8")
    environment = {}

    bind_candidate_memory(environment, tmp_path)

    assert environment["RACAP_AGENT_MEMORY"] == str(shared)
    assert environment["RACAP_FULL_REACT_MEMORY"] == str(full)
    assert environment["RACAP_TRANSPORT_REACT_MEMORY"] == str(transport)


def test_missing_candidate_memory_clears_ambient_memory(tmp_path):
    environment = {
        "RACAP_AGENT_MEMORY": "/ambient/shared",
        "RACAP_FULL_REACT_MEMORY": "/ambient/full",
        "RACAP_TRANSPORT_REACT_MEMORY": "/ambient/transport",
    }

    bind_candidate_memory(environment, tmp_path)

    assert not any(key.startswith("RACAP_") and key.endswith("MEMORY") for key in environment)
