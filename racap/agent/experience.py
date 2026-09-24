"""Runtime-loaded, task-agnostic experience for visual ReAct controllers."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "memory.md"
DEFAULT_SCOPED_MEMORY_PATHS = {
    "full": PROJECT_ROOT / "memories" / "full_react.md",
    "transport": PROJECT_ROOT / "memories" / "transport_react.md",
}
SCOPED_MEMORY_ENV = {
    "full": "RACAP_FULL_REACT_MEMORY",
    "transport": "RACAP_TRANSPORT_REACT_MEMORY",
}
MAX_MEMORY_CHARS = 12_000
ONE_SHOT_CARD_ENV = "CONTROLLED_ONE_SHOT_EXPERIENCE_CARD"
MAX_ONE_SHOT_CARD_CHARS = 4_000


def _read_optional_card() -> str:
    configured = os.environ.get(ONE_SHOT_CARD_ENV, "").strip()
    if not configured:
        return ""
    try:
        return Path(configured).expanduser().read_text(encoding="utf-8").strip()[
            :MAX_ONE_SHOT_CARD_CHARS
        ]
    except (OSError, UnicodeError):
        return ""


def load_runtime_memory(
    path: str | os.PathLike[str] | None = None,
    *,
    scope: str | None = None,
) -> str:
    """Read operator-maintained experience without making imports stateful.

    Reloading at episode boundaries is intentional: an operator can update the
    memory between rollouts without restarting long-lived evaluation workers.
    Missing or unreadable memory is a valid empty configuration.
    """
    if scope is not None and scope not in DEFAULT_SCOPED_MEMORY_PATHS:
        raise ValueError(
            f"unknown runtime-memory scope {scope!r}; expected one of "
            f"{sorted(DEFAULT_SCOPED_MEMORY_PATHS)}"
        )
    scoped_env = SCOPED_MEMORY_ENV.get(scope or "", "")
    configured = (
        path
        or (os.environ.get(scoped_env) if scoped_env else None)
        or os.environ.get("RACAP_AGENT_MEMORY")
    )
    if configured:
        memory_path = Path(configured).expanduser()
    elif scope is not None:
        memory_path = DEFAULT_SCOPED_MEMORY_PATHS[scope]
    else:
        memory_path = DEFAULT_MEMORY_PATH
    try:
        text = memory_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        text = ""
    card = _read_optional_card()
    if card:
        text = (
            text
            + "\n\nONE-TRIAL TARGET-DOMAIN EXPERIENCE CARD "
            "(advisory; current visual evidence wins):\n"
            + card
        ).strip()
    return text[: MAX_MEMORY_CHARS + MAX_ONE_SHOT_CARD_CHARS]


def with_runtime_memory(prompt: str, memory: str | None = None) -> str:
    """Append reusable experience to a system prompt when available."""
    content = load_runtime_memory() if memory is None else str(memory).strip()
    if not content:
        return prompt
    return (
        prompt.rstrip()
        + "\n\nRuntime experience (general guidance, not privileged state):\n"
        + content
    )


__all__ = [
    "DEFAULT_MEMORY_PATH",
    "DEFAULT_SCOPED_MEMORY_PATHS",
    "ONE_SHOT_CARD_ENV",
    "load_runtime_memory",
    "with_runtime_memory",
]
