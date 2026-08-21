from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ThreatEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    entity_type: Literal["actor", "location", "organization", "tactic", "malware", "indicator", "threat_type"]
    value: str = Field(..., min_length=1, max_length=256)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = Field(default="local-ner")

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Entity value cannot be blank")
        return cleaned


class ThreatSignal(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    signal_id: str = Field(default_factory=lambda: uuid4().hex)
    source_type: str = Field(default="telegram")
    source_id: str = Field(default="unknown")
    raw_text: str = Field(..., min_length=1, max_length=20000)
    language: str = Field(default="und")
    intent: str = Field(default="unknown")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entities: list[ThreatEntity] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("entities")
    @classmethod
    def validate_entities(cls, entities: list[ThreatEntity]) -> list[ThreatEntity]:
        return entities[:25]


class KafkaEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    message_id: str = Field(default_factory=lambda: uuid4().hex)
    topic: str
    payload: dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
