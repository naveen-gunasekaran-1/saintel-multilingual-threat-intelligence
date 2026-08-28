"""Tests for the Layer 4 multi-agent synthesis graph.

Everything here runs offline: no Kafka, no Neo4j, no model download. That is
the point of dependency injection in SynthesisDeps -- if these tests ever need
infrastructure, the injection has been broken.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from src.core.schemas import GraphContext, GraphNeighbour, ThreatEntity
from src.layer4_graphrag import neo4j_connector
from src.layer4_graphrag.synthesis_agent import (
    MIN_TEXT_LENGTH,
    MULTIHOP_CYPHER,
    SynthesisDeps,
    _entity_id,
    build_synthesis_graph,
    gazetteer_only_extractor,
    neo4j_multihop_retriever,
    run_synthesis,
    state_to_json,
)

TAMIL_MIXED = "நாளை DRDO சர்வர்களை முடக்குவோம். Chennai target."
TAMIL_NATIVE = "கல்பாக்கம் அணு உலை மீது தாக்குதல்."


# ---------------------------------------------------------------- end to end


def test_graph_compiles_and_runs_end_to_end():
    state = run_synthesis(TAMIL_MIXED)
    assert state.triage is not None and state.triage.accepted
    assert state.signal is not None
    assert state.graph_context is not None
    assert state.halted_at is None
    assert not state.errors


def test_all_three_nodes_contribute_to_the_rationale_in_order():
    state = run_synthesis(TAMIL_MIXED)
    assert len(state.rationale) >= 3
    assert state.rationale[0].startswith("triage:")
    assert state.rationale[1].startswith("extraction:")
    assert any(line.startswith("graphrag:") for line in state.rationale)


def test_entities_are_extracted_with_correct_ontological_types():
    """Directive 3: location and organization are never conflated."""
    state = run_synthesis(TAMIL_MIXED)
    found = {(e.entity_type, e.value) for e in state.signal.entities}
    assert ("organization", "DRDO") in found
    assert ("location", "Chennai") in found


def test_native_tamil_survives_the_whole_graph():
    state = run_synthesis(TAMIL_NATIVE)
    values = [e.value for e in state.signal.entities]
    assert "கல்பாக்கம்" in values


# ------------------------------------------------------------- utf-8 directive


@pytest.mark.parametrize("text", [TAMIL_MIXED, TAMIL_NATIVE])
def test_serialized_state_is_native_utf8_not_escaped(text):
    """Directive 2: no \\u0b.. escape sequences anywhere in the output."""
    payload = state_to_json(run_synthesis(text))
    assert "\\u0b" not in payload
    assert "\\u0c" not in payload
    # and the native string is literally present
    assert any(ch for ch in payload if 0x0B80 <= ord(ch) <= 0x0BFF)
    json.loads(payload)  # still valid JSON


# -------------------------------------------------------------------- triage


def test_short_text_is_rejected_and_halts_at_triage():
    state = run_synthesis("ab")
    assert state.triage is not None and not state.triage.accepted
    assert state.halted_at == "triage"
    assert state.signal is None
    assert state.graph_context is None


def test_min_text_length_boundary():
    assert MIN_TEXT_LENGTH == 3
    assert run_synthesis("ab").halted_at == "triage"
    assert run_synthesis("abc").halted_at != "triage"


def test_low_threat_score_does_not_gate_by_default():
    """Regression guard: the 0.250-recall classifier must not route the graph.

    If this test ever fails, someone has wired the triage model back into the
    data path and the pipeline is now silently discarding real threats.
    """
    deps = SynthesisDeps(threat_scorer=lambda _: 0.01)
    state = run_synthesis(TAMIL_MIXED, deps=deps)
    assert state.triage.accepted is True
    assert state.triage.threat_score == pytest.approx(0.01)
    assert state.halted_at is None
    assert "not used to route" in state.rationale[0]


def test_threat_score_gate_works_when_explicitly_enabled():
    deps = SynthesisDeps(
        threat_scorer=lambda _: 0.01, gate_on_threat_score=True, threat_score_threshold=0.6
    )
    state = run_synthesis(TAMIL_MIXED, deps=deps)
    assert state.triage.accepted is False
    assert state.halted_at == "triage"
    assert "recall 0.250" in state.rationale[0]


def test_broken_threat_scorer_does_not_break_the_graph():
    def boom(_: str) -> float:
        raise RuntimeError("model file missing")

    state = run_synthesis(TAMIL_MIXED, deps=SynthesisDeps(threat_scorer=boom))
    assert state.triage.accepted is True
    assert any("threat_scorer raised RuntimeError" in e for e in state.errors)


def test_triage_reports_the_script():
    assert run_synthesis(TAMIL_MIXED).triage.script == "mixed"
    assert run_synthesis(TAMIL_NATIVE).triage.script == "native_tamil"
    assert run_synthesis("DRDO servers offline tomorrow").triage.script == "latin"


# ---------------------------------------------------------------- extraction


def test_no_entities_halts_before_synthesis():
    state = run_synthesis("good morning everyone have a nice day")
    assert state.signal is not None
    assert state.signal.entities == []
    assert state.halted_at == "extraction"
    assert state.graph_context is None


def test_extractor_exception_is_captured_not_raised():
    def boom(_: str):
        raise ValueError("model OOM")

    state = run_synthesis(TAMIL_MIXED, deps=SynthesisDeps(extractor=boom))
    assert state.halted_at == "extraction"
    assert any("ValueError: model OOM" in e for e in state.errors)
    assert state.signal is None


def test_default_extractor_is_offline_safe():
    entities, intent, confidence = gazetteer_only_extractor(TAMIL_MIXED)
    assert entities
    assert intent == "unknown"  # never guessed -- bart-large-mnli is English-only
    assert confidence == 0.0


# ----------------------------------------------------------------- graphrag


def test_retriever_results_land_in_graph_context():
    def retrieve(entities):
        return GraphContext(
            seed_entity_ids=[_entity_id(e.entity_type, e.value) for e in entities],
            neighbours=[
                GraphNeighbour(
                    entity_id="organization:npcil",
                    entity_type="organization",
                    value="NPCIL",
                    relation="CO_OCCURS_WITH",
                    hops=1,
                    via_signal_id="sig-1",
                ),
                GraphNeighbour(
                    entity_id="location:கல்பாக்கம்",
                    entity_type="location",
                    value="கல்பாக்கம்",
                    relation="LOCATED_IN",
                    hops=2,
                ),
            ],
            retrieved=True,
        )

    state = run_synthesis(TAMIL_MIXED, deps=SynthesisDeps(retriever=retrieve))
    assert state.graph_context.retrieved is True
    assert len(state.graph_context.neighbours) == 2
    assert {n.hops for n in state.graph_context.neighbours} == {1, 2}
    assert any("1-hop=1, 2-hop=1" in line for line in state.rationale)


def test_retriever_failure_degrades_instead_of_crashing():
    def boom(_):
        raise ConnectionError("neo4j unreachable")

    state = run_synthesis(TAMIL_MIXED, deps=SynthesisDeps(retriever=boom))
    assert state.graph_context.retrieved is False
    assert "retrieval failed" in state.graph_context.note
    assert any("ConnectionError" in e for e in state.errors)
    assert state.halted_at is None  # degraded, not halted


def test_stix_bundler_is_invoked_and_output_retained():
    state = run_synthesis(
        TAMIL_MIXED, deps=SynthesisDeps(bundler=lambda s: json.dumps({"type": "bundle"}))
    )
    assert json.loads(state.stix_bundle)["type"] == "bundle"


def test_stix_failure_is_captured_not_raised():
    def boom(_):
        raise TypeError("bad object")

    state = run_synthesis(TAMIL_MIXED, deps=SynthesisDeps(bundler=boom))
    assert any("STIX build failed" in e for e in state.errors)
    assert state.stix_bundle is None


# ------------------------------------------------- contract with the writer


def test_entity_id_matches_the_neo4j_writer_exactly():
    """Retrieval seeds must key identically to what neo4j_connector wrote.

    If these two drift, every multi-hop lookup silently returns nothing.
    """
    for entity_type, value in [
        ("organization", "DRDO"),
        ("location", " Chennai "),
        ("location", "கல்பாக்கம்"),
        ("organization", "npcil"),
    ]:
        assert _entity_id(entity_type, value) == neo4j_connector._entity_id(entity_type, value)


def test_multihop_cypher_only_uses_labels_the_writer_creates():
    for token in ("ThreatSignal", "Entity", "CONTAINS_ENTITY"):
        assert token in MULTIHOP_CYPHER
    writer_relations = {relation for _, _, relation in neo4j_connector.RELATIONSHIP_RULES}
    for relation in writer_relations:
        assert relation in MULTIHOP_CYPHER, f"{relation} written but never retrieved"


def test_neo4j_retriever_maps_records_to_both_hop_levels():
    """Exercise the real retriever against a fake session."""

    class FakeSession:
        def __init__(self):
            self.params = None

        def run(self, cypher, **params):
            self.params = params
            return [
                {
                    "h1_id": "organization:npcil",
                    "h1_type": "organization",
                    "h1_value": "NPCIL",
                    "signal_id": "sig-1",
                    "relation": "LOCATED_IN",
                    "h2_id": "location:கல்பாக்கம்",
                    "h2_type": "location",
                    "h2_value": "கல்பாக்கம்",
                },
                {  # duplicate hop-1, must be deduplicated
                    "h1_id": "organization:npcil",
                    "h1_type": "organization",
                    "h1_value": "NPCIL",
                    "signal_id": "sig-2",
                    "relation": None,
                    "h2_id": None,
                    "h2_type": None,
                    "h2_value": None,
                },
            ]

    session = FakeSession()

    @contextmanager
    def factory():
        yield session

    retrieve = neo4j_multihop_retriever(factory, limit=25)
    context = retrieve([ThreatEntity(entity_type="organization", value="DRDO")])

    assert context.retrieved is True
    assert context.seed_entity_ids == ["organization:drdo"]
    assert session.params["seed_ids"] == ["organization:drdo"]
    assert session.params["limit"] == 25
    assert len(context.neighbours) == 2  # deduplicated
    hop1 = [n for n in context.neighbours if n.hops == 1]
    hop2 = [n for n in context.neighbours if n.hops == 2]
    assert hop1[0].value == "NPCIL" and hop1[0].relation == "CO_OCCURS_WITH"
    assert hop2[0].value == "கல்பாக்கம்" and hop2[0].relation == "LOCATED_IN"


# --------------------------------------------------------------------- graph


def test_graph_exposes_the_three_declared_nodes():
    app = build_synthesis_graph()
    nodes = set(app.get_graph().nodes)
    assert {"triage", "extraction", "synthesis"} <= nodes


# ------------------------------------------- integration with the real Layer 5


def test_real_stix_formatter_plugs_into_the_bundler_slot():
    """Not a fake bundler: the actual STIXBundleFormatter, end to end."""
    from src.layer5_output.stix_formatter import STIXBundleFormatter

    state = run_synthesis(
        TAMIL_MIXED, deps=SynthesisDeps(bundler=STIXBundleFormatter().to_json)
    )
    bundle = json.loads(state.stix_bundle)
    assert bundle["type"] == "bundle"
    assert {o["type"] for o in bundle["objects"]} >= {"identity", "location"}


def test_native_tamil_reaches_stix_unescaped():
    """Directive 2 held across the Layer 4 -> Layer 5 boundary.

    stix2.serialize() defaults to ensure_ascii=True, which silently turned
    'கல்பாக்கம்' into '\\u0b95...' before this was fixed. This test is what
    stops that regressing.
    """
    from src.layer5_output.stix_formatter import STIXBundleFormatter

    state = run_synthesis(
        TAMIL_NATIVE, deps=SynthesisDeps(bundler=STIXBundleFormatter().to_json)
    )
    assert "\\u0b" not in state.stix_bundle
    names = [o.get("name") for o in json.loads(state.stix_bundle)["objects"]]
    assert "கல்பாக்கம்" in names
