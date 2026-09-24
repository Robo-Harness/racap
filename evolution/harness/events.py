"""Append-only lineage plus atomically replaced current state."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .schema import ExperimentState


class EventStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = root / "events.jsonl"
        self.state_path = root / "state.json"

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        row = {"time": time.time(), "event": event, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return row

    def save_state(self, state: ExperimentState) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(state.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def load_state(self) -> ExperimentState:
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        return ExperimentState(**value)
