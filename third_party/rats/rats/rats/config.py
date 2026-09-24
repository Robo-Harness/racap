from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RatsConfig:
    enabled: bool = False
    fixed_scene_model: str | None = None
    proposer_mode: str = "existing_definition_single_scene"
    primitive_api: str = "legacy"
    max_feedback_retries: int = 0
    writer_model: str | None = None
    writer_server_url: str | None = None
    writer_api_key: str | None = None
    writer_temperature: float = 0.2
    writer_max_tokens: int = 4096
    writer_reasoning_effort: str = "medium"


DEFAULT_RATS_CONFIG = RatsConfig()
