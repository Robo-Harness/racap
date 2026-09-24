import numpy as np
import pytest

from racap.backends import vlm
from racap.backends.llm import LLMError


def test_detect_many_does_not_turn_provider_failure_into_empty_detection(monkeypatch):
    def unavailable(*args, **kwargs):
        raise LLMError("gateway timed out")

    monkeypatch.setattr(vlm, "_ask_json", unavailable)
    with pytest.raises(vlm.PerceptionUnavailableError) as failure:
        vlm.detect_many(np.zeros((32, 32, 3), dtype=np.uint8), ["black bowl"])

    assert failure.value.kind == "provider_unavailable"
    assert failure.value.operation == "detect_many"


def test_valid_found_false_remains_a_semantic_miss(monkeypatch):
    monkeypatch.setattr(vlm, "_ask_json", lambda *args, **kwargs: {"found": False})

    assert vlm.bbox(np.zeros((32, 32, 3), dtype=np.uint8), "occluded bowl") is None


def test_malformed_found_true_is_not_a_semantic_miss(monkeypatch):
    monkeypatch.setattr(vlm, "_ask_json", lambda *args, **kwargs: {"found": True})

    with pytest.raises(vlm.PerceptionUnavailableError, match="no valid box"):
        vlm.bbox(np.zeros((32, 32, 3), dtype=np.uint8), "black bowl")
