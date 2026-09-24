"""Verify the vendored relay cannot silently select a third-party endpoint."""

import os
from pathlib import Path
import subprocess
import sys


def test_vendor_relay_requires_explicit_safe_endpoint():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(root / "third_party/rats"),
        PYTHONDONTWRITEBYTECODE="1",
        RATS_VAPI_KEY="synthetic-test-only",
        RATS_VAPI_URL="",
    )
    script = """
from rats.agents import base_agent
invalid = [
    '', 'not-a-url', 'http://example.invalid/v1',
    'https://' + 'user:password@example.invalid/v1',
    'https://example.invalid/v1?token=example',
    'https://example.invalid/v1#fragment',
]
for value in invalid:
    base_agent.VAPI_URL = value
    try:
        base_agent._get_api_config('openai/test-model')
    except RuntimeError as exc:
        assert 'RATS_VAPI_URL' in str(exc)
    else:
        raise AssertionError('Unsafe or missing endpoint accepted')
for value in ['https://example.invalid/v1', 'http://127.0.0.1:8000/v1']:
    base_agent.VAPI_URL = value
    url, headers = base_agent._get_api_config('openai/test-model')
    assert url == value
"""
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=15)
