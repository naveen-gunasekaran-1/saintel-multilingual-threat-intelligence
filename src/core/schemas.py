from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
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


# ---------------------------------------------------------------------------
# Layer 4 multi-agent synthesis state.
#
# These models are the state schema for the LangGraph StateGraph in
# src/layer4_graphrag/synthesis_agent.py. Nodes return partial dict updates;
# `rationale` and `errors` use an additive reducer so each node appends its own
# trace rather than clobbering the previous node's.
# ---------------------------------------------------------------------------


class TriageVerdict(BaseModel):
    """Outcome of the triage node.

    IMPORTANT: `threat_score` is ADVISORY METADATA and never gates the graph.
    The fastText triage model measures recall 0.250 at its production
    threshold (see results/test-033479a/report.md), meaning it discards ~75%
    of real threats. Routing on it would silently destroy the pipeline's
    recall, so `accepted` is decided on structural validity alone.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    accepted: bool
    reason: str = Field(..., min_length=1, max_length=512)
    script: str = "unknown"
    threat_score: float | None = Field(default=None, ge=0.0, le=1.0)


class GraphNeighbour(BaseModel):
    """One entity reached from a seed entity during multi-hop retrieval."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    entity_id: str = Field(..., min_length=1)
    entity_type: str
    value: str = Field(..., min_length=1)
    relation: str
    hops: int = Field(..., ge=1, le=3)
    via_signal_id: str | None = None


class GraphContext(BaseModel):
    """Result of the GraphRAG retrieval step."""

    model_config = ConfigDict(extra="forbid")

    seed_entity_ids: list[str] = Field(default_factory=list)
    neighbours: list[GraphNeighbour] = Field(default_factory=list)
    retrieved: bool = False
    note: str = ""


class SynthesisState(BaseModel):
    """State threaded through the Layer 4 StateGraph."""

    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(..., min_length=1, max_length=20000)
    source_type: str = "telegram"
    source_id: str = "unknown"

    signal: ThreatSignal | None = None
    triage: TriageVerdict | None = None
    graph_context: GraphContext | None = None
    stix_bundle: str | None = None

    rationale: Annotated[list[str], operator.add] = Field(default_factory=list)
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    halted_at: str | None = None
