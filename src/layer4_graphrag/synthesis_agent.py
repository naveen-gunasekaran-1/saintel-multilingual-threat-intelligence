"""Layer 4 — multi-agent synthesis over a LangGraph StateGraph.

Three nodes, run in sequence with early-exit edges:

    triage_node  ->  extraction_node  ->  graphrag_synthesis_node
        |                  |
        +-> END            +-> END        (nothing survived the stage)

Design notes that are load-bearing, not decoration:

**1. The triage node does not route on the threat classifier.**
The fastText signal/noise model measures recall 0.250 at its production
threshold of 0.6 (results/test-033479a/report.md). Gating Layer 3 behind it
would silently discard roughly three quarters of real threats, which is why
README.md records it as deliberately *not* in the data path. This node keeps
that decision: it accepts or rejects on **structural** grounds only, and
attaches any classifier score as advisory metadata for a human. If a future
model earns the gate, flip `gate_on_threat_score` — the plumbing is here, the
default is off, and the reason is written down.

**2. Every dependency is injected.**
`SynthesisDeps` holds callables for extraction, retrieval and bundling. The
default extractor is the dependency-free gazetteer, so the graph imports and
runs with no `.env`, no Kafka, no Neo4j and no 430 MB model download — which
is what makes it testable offline. Production wiring passes real callables in.

**3. Serialization is native UTF-8 throughout.**
Every `json.dumps` here passes `ensure_ascii=False` (project directive 2).
`சென்னை` stays `சென்னை` and never becomes an escape sequence.

**4. The rationale trace is a decision log, not saliency-based XAI.**
Each node appends a plain-language line explaining what it did and why. That
is honest provenance for an analyst; it is *not* an attribution method such as
LIME/SHAP and should not be described as one in a paper.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.core.identity import entity_id as _shared_entity_id
from src.core.schemas import (
    GraphContext,
    GraphNeighbour,
    SynthesisState,
    ThreatEntity,
    ThreatSignal,
    TriageVerdict,
)
from src.layer1_ingestion.normalizer import detect_script, normalize_threat_text
from src.layer3_native_nlp.gazetteer import refine_entities

__all__ = [
    "SynthesisDeps",
    "MIN_TEXT_LENGTH",
    "build_synthesis_graph",
    "run_synthesis",
    "triage_node",
    "extraction_node",
    "graphrag_synthesis_node",
    "gazetteer_only_extractor",
    "neo4j_multihop_retriever",
    "MULTIHOP_CYPHER",
]

# Below this, there is not enough text for any extractor to work with.
MIN_TEXT_LENGTH = 3

# Two-hop retrieval over the schema written by neo4j_connector.py:
#   (:ThreatSignal)-[:CONTAINS_ENTITY]->(:Entity)
#   (:Entity)-[:TARGETS|OPERATES_IN|USES|LOCATED_IN]->(:Entity)
#
# Hop 1 is co-occurrence: entities that appeared in the same signal as a seed.
# Hop 2 follows the typed relationships out of those co-occurring entities.
# Together they answer "what else is this asset associated with", which is the
# question a GraphRAG retrieval step exists to answer.
MULTIHOP_CYPHER = """
MATCH (seed:Entity)
WHERE seed.entity_id IN $seed_ids
MATCH (seed)<-[:CONTAINS_ENTITY]-(s:ThreatSignal)-[:CONTAINS_ENTITY]->(hop1:Entity)
WHERE hop1.entity_id <> seed.entity_id
OPTIONAL MATCH (hop1)-[r]->(hop2:Entity)
WHERE type(r) IN ['TARGETS', 'OPERATES_IN', 'USES', 'LOCATED_IN']
  AND hop2.entity_id <> seed.entity_id
RETURN hop1.entity_id   AS h1_id,
       hop1.entity_type AS h1_type,
       hop1.value       AS h1_value,
       s.signal_id      AS signal_id,
       type(r)          AS relation,
       hop2.entity_id   AS h2_id,
       hop2.entity_type AS h2_type,
       hop2.value       AS h2_value
LIMIT $limit
"""


def _entity_id(entity_type: str, value: str) -> str:
    """Graph key. Shared with neo4j_connector via src.core.identity, which is
    what guarantees the seed ids built here match the ids written there."""
    return _shared_entity_id(entity_type, value)


def gazetteer_only_extractor(text: str) -> tuple[list[ThreatEntity], str, float]:
    """Default extractor: gazetteer repair with no model behind it.

    Returns (entities, intent, confidence). Intent is reported as "unknown"
    rather than guessed: the only intent classifier in the repo is
    bart-large-mnli, which is English-only and produces a near-uniform label
    distribution on Tamil. Emitting "unknown" is more honest than emitting a
    number that carries no information.
    """
    entities = refine_entities(text, [])
    return entities, "unknown", 0.0


@dataclass
class SynthesisDeps:
    """Injected collaborators. Defaults are offline-safe."""

    extractor: Callable[[str], tuple[list[ThreatEntity], str, float]] = gazetteer_only_extractor
    retriever: Callable[[Sequence[ThreatEntity]], GraphContext] | None = None
    bundler: Callable[[ThreatSignal], str | None] | None = None
    threat_scorer: Callable[[str], float] | None = None
    gate_on_threat_score: bool = False
    threat_score_threshold: float = 0.6
    max_neighbours: int = 25
    _: dict[str, Any] = field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def triage_node(state: SynthesisState, deps: SynthesisDeps) -> dict[str, Any]:
    """Validate signal quality. Structural checks gate; the model does not."""
    text = normalize_threat_text(state.raw_text)
    script = detect_script(text)

    score: float | None = None
    if deps.threat_scorer is not None:
        try:
            score = float(deps.threat_scorer(text))
        except Exception as exc:  # a broken scorer must not fail the graph
            return {
                "errors": [f"triage: threat_scorer raised {type(exc).__name__}: {exc}"],
                "triage": TriageVerdict(
                    accepted=True,
                    reason="scorer failed; accepted on structural validity",
                    script=script,
                ),
                "rationale": [f"triage: scorer unavailable, proceeding on structure ({script})"],
            }

    if len(text) < MIN_TEXT_LENGTH:
        return {
            "triage": TriageVerdict(
                accepted=False,
                reason=f"text shorter than {MIN_TEXT_LENGTH} characters after normalization",
                script=script,
                threat_score=score,
            ),
            "halted_at": "triage",
            "rationale": [f"triage: rejected, {len(text)} chars after normalization"],
        }

    # The gate exists but is off by default. See the module docstring.
    if deps.gate_on_threat_score and score is not None and score < deps.threat_score_threshold:
        return {
            "triage": TriageVerdict(
                accepted=False,
                reason=f"threat score {score:.3f} below gate {deps.threat_score_threshold}",
                script=script,
                threat_score=score,
            ),
            "halted_at": "triage",
            "rationale": [
                f"triage: rejected by EXPLICITLY ENABLED score gate ({score:.3f}); "
                "note this classifier measures recall 0.250"
            ],
        }

    advisory = f", advisory threat score {score:.3f} (not used to route)" if score is not None else ""
    return {
        "raw_text": text,
        "triage": TriageVerdict(
            accepted=True,
            reason="structurally valid",
            script=script,
            threat_score=score,
        ),
        "rationale": [f"triage: accepted, script={script}{advisory}"],
    }


def extraction_node(state: SynthesisState, deps: SynthesisDeps) -> dict[str, Any]:
    """Invoke Layer 3 and build the ThreatSignal."""
    try:
        entities, intent, confidence = deps.extractor(state.raw_text)
    except Exception as exc:
        return {
            "errors": [f"extraction: {type(exc).__name__}: {exc}"],
            "halted_at": "extraction",
            "rationale": ["extraction: failed, no signal produced"],
        }

    signal = ThreatSignal(
        source_type=state.source_type,
        source_id=state.source_id,
        raw_text=state.raw_text,
        language=(state.triage.script if state.triage else "und"),
        intent=intent,
        confidence=confidence,
        entities=entities,
    )

    if not entities:
        return {
            "signal": signal,
            "halted_at": "extraction",
            "rationale": ["extraction: no entities found; nothing to synthesize"],
        }

    by_source: dict[str, int] = {}
    for entity in entities:
        by_source[entity.source] = by_source.get(entity.source, 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))
    named = " | ".join(f"{e.entity_type}:{e.value}" for e in entities[:6])

    return {
        "signal": signal,
        "rationale": [f"extraction: {len(entities)} entities [{breakdown}] -> {named}"],
    }


def graphrag_synthesis_node(state: SynthesisState, deps: SynthesisDeps) -> dict[str, Any]:
    """Multi-hop retrieval, then a STIX 2.1 bundle."""
    signal = state.signal
    if signal is None:  # unreachable via the compiled graph; defensive
        return {"errors": ["synthesis: no signal in state"], "halted_at": "synthesis"}

    seed_ids = [_entity_id(e.entity_type, e.value) for e in signal.entities]
    updates: dict[str, Any] = {}
    rationale: list[str] = []
    errors: list[str] = []

    if deps.retriever is None:
        context = GraphContext(
            seed_entity_ids=seed_ids,
            retrieved=False,
            note="no retriever configured; synthesis ran on the current signal alone",
        )
        rationale.append(f"graphrag: no retriever, {len(seed_ids)} seed entities, 0 hops")
    else:
        try:
            context = deps.retriever(signal.entities)
            hop_counts: dict[int, int] = {}
            for neighbour in context.neighbours:
                hop_counts[neighbour.hops] = hop_counts.get(neighbour.hops, 0) + 1
            spread = ", ".join(f"{h}-hop={c}" for h, c in sorted(hop_counts.items())) or "none"
            rationale.append(
                f"graphrag: {len(seed_ids)} seeds -> {len(context.neighbours)} neighbours ({spread})"
            )
        except Exception as exc:
            context = GraphContext(
                seed_entity_ids=seed_ids,
                retrieved=False,
                note=f"retrieval failed: {type(exc).__name__}",
            )
            errors.append(f"synthesis: retrieval failed: {type(exc).__name__}: {exc}")
            rationale.append("graphrag: retrieval failed, degrading to signal-only synthesis")

    updates["graph_context"] = context

    if deps.bundler is not None:
        try:
            bundle = deps.bundler(signal)
            updates["stix_bundle"] = bundle
            rationale.append(
                "stix: bundle built" if bundle else "stix: builder returned no bundle"
            )
        except Exception as exc:
            errors.append(f"synthesis: STIX build failed: {type(exc).__name__}: {exc}")
            rationale.append("stix: build failed")
    else:
        rationale.append("stix: no bundler configured")

    updates["rationale"] = rationale
    if errors:
        updates["errors"] = errors
    return updates


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------


def _after_triage(state: SynthesisState) -> str:
    return "extraction" if (state.triage and state.triage.accepted) else END


def _after_extraction(state: SynthesisState) -> str:
    return "synthesis" if (state.signal and state.signal.entities) else END


def build_synthesis_graph(deps: SynthesisDeps | None = None):
    """Compile the Layer 4 StateGraph. Returns a runnable app."""
    deps = deps or SynthesisDeps()

    graph = StateGraph(SynthesisState)
    graph.add_node("triage", lambda s: triage_node(s, deps))
    graph.add_node("extraction", lambda s: extraction_node(s, deps))
    graph.add_node("synthesis", lambda s: graphrag_synthesis_node(s, deps))

    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", _after_triage, {"extraction": "extraction", END: END})
    graph.add_conditional_edges("extraction", _after_extraction, {"synthesis": "synthesis", END: END})
    graph.add_edge("synthesis", END)

    return graph.compile()


def run_synthesis(
    raw_text: str,
    *,
    deps: SynthesisDeps | None = None,
    source_type: str = "telegram",
    source_id: str = "unknown",
) -> SynthesisState:
    """Run one record end to end and return the final state."""
    app = build_synthesis_graph(deps)
    result = app.invoke(
        SynthesisState(raw_text=raw_text, source_type=source_type, source_id=source_id)
    )
    return SynthesisState.model_validate(result)


# --------------------------------------------------------------------------
# Production wiring (network-bound; not exercised by the offline tests)
# --------------------------------------------------------------------------


def neo4j_multihop_retriever(session_factory, *, limit: int = 25):
    """Build a retriever that runs MULTIHOP_CYPHER against Neo4j.

    `session_factory` is a zero-arg callable returning a context-managed
    neo4j Session — e.g. `Neo4jSessionManager(...).get_session`.
    """

    def retrieve(entities: Sequence[ThreatEntity]) -> GraphContext:
        seed_ids = [_entity_id(e.entity_type, e.value) for e in entities]
        neighbours: list[GraphNeighbour] = []
        seen: set[tuple[str, int]] = set()

        with session_factory() as session:
            for record in session.run(MULTIHOP_CYPHER, seed_ids=seed_ids, limit=limit):
                if record["h1_id"] and (record["h1_id"], 1) not in seen:
                    seen.add((record["h1_id"], 1))
                    neighbours.append(
                        GraphNeighbour(
                            entity_id=record["h1_id"],
                            entity_type=record["h1_type"],
                            value=record["h1_value"],
                            relation="CO_OCCURS_WITH",
                            hops=1,
                            via_signal_id=record["signal_id"],
                        )
                    )
                if record["h2_id"] and (record["h2_id"], 2) not in seen:
                    seen.add((record["h2_id"], 2))
                    neighbours.append(
                        GraphNeighbour(
                            entity_id=record["h2_id"],
                            entity_type=record["h2_type"],
                            value=record["h2_value"],
                            relation=record["relation"] or "RELATED_TO",
                            hops=2,
                            via_signal_id=record["signal_id"],
                        )
                    )

        return GraphContext(seed_entity_ids=seed_ids, neighbours=neighbours, retrieved=True)

    return retrieve


def state_to_json(state: SynthesisState) -> str:
    """Serialize final state as native UTF-8 JSON (directive 2)."""
    return json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    demo = "நாளை DRDO சர்வர்களை முடக்குவோம். Chennai target."
    print(state_to_json(run_synthesis(demo)))
