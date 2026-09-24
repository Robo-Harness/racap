"""Read-only one-trial context adapter used by controlled transfer experiments."""

from __future__ import annotations

import os
from pathlib import Path

ENV_NAME = "CONTROLLED_ONE_SHOT_EXPERIENCE_CARD"
MAX_CHARS = 4_000


def read_experience_card() -> str:
    configured = os.environ.get(ENV_NAME, "").strip()
    if not configured:
        return ""
    try:
        return Path(configured).expanduser().read_text(encoding="utf-8").strip()[:MAX_CHARS]
    except (OSError, UnicodeError):
        return ""


def append_experience_card(context: str) -> str:
    card = read_experience_card()
    if not card:
        return context
    return (
        str(context).rstrip()
        + "\n\nONE-TRIAL TARGET-DOMAIN EXPERIENCE CARD "
        "(advisory; current visual evidence wins):\n"
        + card
    ).strip()


__all__ = ["ENV_NAME", "append_experience_card", "read_experience_card"]
