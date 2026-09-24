"""Load versioned, human-readable prompts from the unified evolution tree."""

from __future__ import annotations

from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


def read_prompt(relative_path: str) -> str:
    path = PROMPT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"evolution prompt does not exist: {path}")
    return path.read_text(encoding="utf-8").strip()
