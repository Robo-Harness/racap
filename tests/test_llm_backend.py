import pytest
import json

import racap.backends.llm as llm


@pytest.mark.parametrize("base", [None, "", "localhost:8000", "https://" + "user:password@example.invalid/v1", "https://example.invalid/v1?token=example"])
def test_relay_requires_an_explicit_credential_free_endpoint(monkeypatch, base):
    import requests

    monkeypatch.setenv("RACAP_VAPI_KEY", "test-only")
    if base is None:
        monkeypatch.delenv("RACAP_VAPI_BASE", raising=False)
    else:
        monkeypatch.setenv("RACAP_VAPI_BASE", base)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("must reject before sending credentials"))
    with pytest.raises(llm.LLMError, match="RACAP_VAPI_BASE"):
        llm._via_relay("system", "prompt", [], "test-model", 16, 0.0)


def test_abort_sentinel_stops_before_network(monkeypatch, tmp_path):
    sentinel = tmp_path / "ABORTED.json"
    sentinel.write_text(json.dumps({"reason": "quota_exhausted"}))
    monkeypatch.setenv("RACAP_ABORT_SENTINEL", str(sentinel))
    monkeypatch.setattr(
        llm,
        "_via_relay",
        lambda *args, **kwargs: pytest.fail("network must not run after global abort"),
    )

    with pytest.raises(llm.LLMQuotaError, match="quota_exhausted"):
        llm.ask("system", "prompt", model="gpt-5.5", cache=False)


def test_cache_hit_writes_credential_free_telemetry(monkeypatch, tmp_path):
    telemetry = tmp_path / "calls.jsonl"
    monkeypatch.setenv("RACAP_LLM_TELEMETRY_PATH", str(telemetry))
    monkeypatch.setenv("RACAP_EPISODE_KEY", "libero_90/0/seed0")
    monkeypatch.delenv("RACAP_ABORT_SENTINEL", raising=False)
    monkeypatch.setattr(llm, "_read_cache", lambda _: "cached reply")

    answer = llm.ask("secret system", "secret prompt", model="gpt-5.5")

    assert answer == "cached reply"
    record = json.loads(telemetry.read_text())
    assert record["event"] == "cache_hit"
    assert record["episode_key"] == "libero_90/0/seed0"
    assert "secret system" not in telemetry.read_text()
    assert "secret prompt" not in telemetry.read_text()


def test_telemetry_attests_one_shot_card_without_retaining_text(monkeypatch, tmp_path):
    telemetry = tmp_path / "calls.jsonl"
    card = tmp_path / "card.md"
    card.write_text("private transferable lesson", encoding="utf-8")
    monkeypatch.setenv("RACAP_LLM_TELEMETRY_PATH", str(telemetry))
    monkeypatch.setenv("CONTROLLED_ONE_SHOT_EXPERIENCE_CARD", str(card))
    monkeypatch.delenv("RACAP_ABORT_SENTINEL", raising=False)
    monkeypatch.setattr(llm, "_read_cache", lambda _: "cached reply")

    llm.ask(
        "system with private transferable lesson",
        "ordinary prompt",
        model="gpt-5.5",
    )

    record = json.loads(telemetry.read_text())
    assert record["one_shot_card_readable"] is True
    assert record["one_shot_card_in_system"] is True
    assert record["one_shot_card_in_request"] is True
    assert len(record["one_shot_card_sha256"]) == 64
    assert "private transferable lesson" not in telemetry.read_text()


def test_quota_error_is_not_retried(monkeypatch):
    llm.reset_call_stats()
    calls = 0

    def blocked(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise llm.LLMQuotaError("HTTP 403: insufficient_user_quota")

    monkeypatch.setattr(llm, "_via_relay", blocked)
    with pytest.raises(llm.LLMQuotaError):
        llm.ask(
            "system",
            "prompt",
            model="gpt-5.5",
            attempts=3,
            cache=False,
        )
    assert calls == 1
    assert llm.call_stats()["terminal_failures"] == 1
    assert llm.call_stats()["quota_failures"] == 1


def test_exhausted_transient_retries_are_counted_as_terminal(monkeypatch):
    llm.reset_call_stats()

    def unavailable(*args, **kwargs):
        raise llm.LLMError("temporary gateway failure")

    monkeypatch.setattr(llm, "_via_relay", unavailable)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    with pytest.raises(llm.LLMError, match="failed after 2 attempts"):
        llm.ask("system", "prompt", model="gpt-5.5", attempts=2, cache=False)

    assert llm.call_stats() == {
        "logical_calls": 1,
        "cache_hits": 0,
        "network_attempts": 2,
        "terminal_failures": 1,
        "quota_failures": 0,
    }


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (403, '{"code":"insufficient_user_quota"}', True),
        (403, "need pre-deduct $1; balance is insufficient", True),
        (429, "ordinary rate limit", False),
        (401, "insufficient balance", False),
    ],
)
def test_quota_failure_classification(status, body, expected):
    assert llm._is_quota_failure(status, body) is expected


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    (
        (403, "model gpt-5.6 暂无可用渠道", True),
        (404, "model_not_found", True),
        (429, "temporary rate limit", False),
    ),
)
def test_provider_unavailable_classification(status, body, expected):
    assert llm._is_provider_unavailable(status, body) is expected
