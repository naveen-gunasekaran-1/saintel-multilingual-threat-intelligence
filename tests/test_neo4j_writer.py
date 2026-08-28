"""Tests for the Neo4j write path.

This module had no coverage at all before its write logic was rebatched, which
is precisely why the rebatch needed tests: the old shape issued one autocommit
query per entity and per entity *pair*, and nothing would have caught a
regression in the graph it produced.

`get_settings()` runs at neo4j_connector import time, so the environment has to
be satisfied before the import -- that coupling is itself a finding, recorded
in the architecture review rather than silently worked around.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("NEO4J_PASSWORD", "test-only-not-a-real-password")
os.environ.setdefault("POSTGRES_DSN", "postgresql://test@localhost:5432/test")

from src.core.identity import entity_id
from src.core.schemas import ThreatEntity, ThreatSignal
from src.layer4_graphrag.neo4j_connector import (
    _ALLOWED_RELATIONS,
    GraphEntityWriter,
    RELATIONSHIP_RULES,
)


def _signal(*pairs: tuple[str, str]) -> ThreatSignal:
    return ThreatSignal(
        raw_text="நாளை DRDO சர்வர்களை முடக்குவோம்.",
        entities=[ThreatEntity(entity_type=t, value=v, confidence=0.9) for t, v in pairs],
    )


class FakeTx:
    """Records every query issued, so the transaction shape is assertable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params) -> None:
        self.calls.append((query, params))


class FakeSession:
    def __init__(self) -> None:
        self.tx = FakeTx()
        self.write_units = 0

    def execute_write(self, unit) -> None:
        self.write_units += 1
        unit(self.tx)


def test_build_rows_emits_one_row_per_entity():
    rows, _ = GraphEntityWriter._build_rows(_signal(("organization", "DRDO"), ("location", "Chennai")))
    assert [r["entity_id"] for r in rows] == ["organization:drdo", "location:chennai"]


def test_entity_ids_match_the_shared_helper():
    """The graph key written here must equal the key GraphRAG looks up."""
    rows, _ = GraphEntityWriter._build_rows(_signal(("organization", "  DRDO ")))
    assert rows[0]["entity_id"] == entity_id("organization", "  DRDO ")


def test_relationship_rows_are_the_cartesian_product_per_rule():
    signal = _signal(("actor", "Grp"), ("organization", "DRDO"), ("organization", "ISRO"))
    _, relations = GraphEntityWriter._build_rows(signal)
    assert len(relations["TARGETS"]) == 2  # one actor x two organizations
    assert {r["target_id"] for r in relations["TARGETS"]} == {"organization:drdo", "organization:isro"}


def test_relationship_confidence_is_the_weaker_of_the_pair():
    signal = ThreatSignal(
        raw_text="x",
        entities=[
            ThreatEntity(entity_type="actor", value="Grp", confidence=0.9),
            ThreatEntity(entity_type="organization", value="DRDO", confidence=0.4),
        ],
    )
    _, relations = GraphEntityWriter._build_rows(signal)
    assert relations["TARGETS"][0]["confidence"] == pytest.approx(0.4)


def test_write_is_a_single_atomic_unit():
    session = FakeSession()
    GraphEntityWriter._upsert_signal_transaction(session, _signal(("organization", "DRDO")))
    assert session.write_units == 1


def test_round_trips_are_bounded_not_quadratic():
    """20 entities used to cost 1 + 20 + up to 400 queries. Now it is batched."""
    pairs = [("organization", f"ORG{i}") for i in range(10)]
    pairs += [("location", f"LOC{i}") for i in range(10)]
    session = FakeSession()
    GraphEntityWriter._upsert_signal_transaction(session, _signal(*pairs))
    # 1 signal + 1 entity batch + 1 batch for the single applicable relation rule
    assert len(session.tx.calls) <= 2 + len(RELATIONSHIP_RULES)
    assert len(session.tx.calls) == 3


def test_signal_with_no_entities_skips_the_entity_batch():
    session = FakeSession()
    GraphEntityWriter._upsert_signal_transaction(session, _signal())
    assert len(session.tx.calls) == 1


def test_every_interpolated_relation_is_allow_listed():
    """The one interpolated identifier in the Cypher must never take free input."""
    signal = _signal(("actor", "Grp"), ("organization", "DRDO"), ("location", "Chennai"),
                     ("tactic", "Phishing"))
    _, relations = GraphEntityWriter._build_rows(signal)
    assert set(relations).issubset(_ALLOWED_RELATIONS)


def test_native_script_survives_into_the_write_parameters():
    session = FakeSession()
    GraphEntityWriter._upsert_signal_transaction(session, _signal(("location", "சென்னை")))
    entity_batch = session.tx.calls[1][1]
    assert entity_batch["rows"][0]["value"] == "சென்னை"
