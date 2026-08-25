# ADR-003: LangGraph multi-agent synthesis for Layer 4

**Status:** Proposed — *decision incomplete* · **Date:** 2026-08-25

> This ADR is written **retrospectively**. The code was added in commit
> `01def71` without an accompanying decision record. It documents what exists
> and names the question that is still open, rather than ratifying a choice
> nobody wrote down.

## Context

Layer 4 originally consisted of `neo4j_connector.py`: a Kafka consumer that
validates a `ThreatSignal` and upserts it into Neo4j. `synthesis_agent.py`
existed in the baseline commit as a **0-byte placeholder**, so a synthesis
component was always part of the intended design.

Commit `01def71` filled it with a LangGraph `StateGraph`:

```
triage_node -> extraction_node -> graphrag_synthesis_node
    |               |
    +-> END         +-> END        (nothing survived the stage)
```

## What was built

Three properties are worth preserving whatever is decided:

1. **Dependencies are injected.** `SynthesisDeps` carries callables for
   extraction, retrieval, and bundling. Defaults are offline-safe — the default
   extractor is the dependency-free gazetteer — so the graph imports and runs
   with no `.env`, no Kafka, no Neo4j, and no 430 MB model download. That is
   what makes it testable, and it is a better arrangement than the imperative
   consumer's module-scope `get_settings()`.

2. **The triage node does not gate on the threat classifier.** It accepts or
   rejects on *structural* grounds only and attaches any classifier score as
   advisory metadata. The reason is written at the point of the decision: the
   classifier measures recall 0.250. `gate_on_threat_score` exists and defaults
   to `False`.

3. **Two-hop GraphRAG retrieval** over the schema `neo4j_connector` writes —
   co-occurrence within a signal, then typed relationships out of those
   neighbours.

## The unresolved problem

**Layer 4 now has two implementations, and nothing routes between them.**

| | `neo4j_connector` | `synthesis_agent` |
|---|---|---|
| Trigger | Kafka consumer on `layer3_entities` | direct call only — tests and `__main__` |
| In the data path | yes | **no** |
| Testability | needs env vars to import | offline-safe |
| Writes to Neo4j | yes | no |

They already shared one rule that had to be kept byte-identical by hand — the
entity-key function — with a comment saying so. That duplication has since been
removed into `src/core/identity.py`, but it is a preview of how this diverges:
two copies of one rule, drifting silently, failing quietly. GraphRAG retrieval
returning zero neighbours looks identical to "no neighbours exist".

## Decision required

One of:

- **A — LangGraph replaces the consumer.** Wire `synthesis_agent` to the Kafka
  topic; `neo4j_connector` becomes the persistence callable injected into it.
  Single data path, better testability. Most work.
- **B — LangGraph is an offline analysis path.** Keep the consumer as
  production; the graph is for research, ablation, and the paper's
  "multi-agent" framing. Document that it is not in the data path, as Layer 2
  already is. Least work, honest, but the divergence risk remains.
- **C — Remove it.** Only if the multi-agent framing is dropped from the paper.
  Not recommended: it is the better-engineered of the two.

**No option is chosen yet.** Until one is, treat any "multi-agent pipeline"
claim in a paper as describing an implemented but *unwired* component, and say
so.

## Consequences of leaving it unresolved

- `langgraph==1.2.11` is a production dependency for code no production path
  reaches.
- The two implementations will drift.
- The paper's architecture section cannot honestly say the multi-agent layer
  processes live data, because it does not.
