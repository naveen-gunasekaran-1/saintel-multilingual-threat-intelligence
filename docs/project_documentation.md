# SAINTEL — Architecture & Module Reference

**S**outh **A**sian **I**nfrastructure — **N**ative **T**hreat **E**ntity **L**abelling.

Status: **research prototype**. Not deployed, not production-hardened.
Last reviewed against the code: 2026-08-25.

---

## 1. What the system does

Ingests open-source text (Telegram, Tor leak sites, threat feeds), extracts
critical-national-infrastructure entities **in native script with no translation
step**, and builds a threat graph with STIX 2.1 export.

The design constraint that shapes everything: multilingual NER tokenizers
fragment mixed-script entities (`DRDO` → `dr` + `##do`) and drop Indic viramas
(`சென்னை` → `செனனை`). Layer 3 exists to detect and repair that damage.

---

## 2. Layer topology

```
L0  darkweb collectors      play_dls_ingest, ransomlook_ingest
L1  surface collection      telegram_scraper, normalizer, capture_sink
L2  edge triage             triage_consumer            [NOT in the data path]
L3  native NLP + repair     entity_extractor, gazetteer
L4  threat graph            neo4j_connector  |  synthesis_agent (LangGraph)
L5  output                  stix_formatter, ui_dashboard
```

Layers are decoupled by Kafka topics. Three consumer groups read the raw topic
independently (triage, extraction, capture) — deliberate fan-out, not a bug.

### Known topology facts

- **`high_intent_signals` has no consumer.** Layer 2 publishes to it; nothing
  subscribes. This is intentional (see §5) but the producer still runs.
- **Layer 4 exists twice.** `neo4j_connector` (imperative Kafka consumer) and
  `synthesis_agent` (LangGraph StateGraph). Neither calls the other. See
  `adr-003-langgraph-layer4.md`.
- **Collectors bypass settings.** `src/layer0_scrapers/*` hardcode
  `raw_threat_stream`; `config.py` defaults to `raw_osint_stream`. They agree
  only because `.env` sets the former. See §7.

---

## 3. Data contract

Every message on the bus is a Pydantic v2 model with `extra="forbid"`, so
malformed payloads fail at the boundary rather than corrupting the graph.

| Model | Purpose |
|---|---|
| `ThreatEntity` | typed entity: `actor \| location \| organization \| tactic \| malware \| indicator \| threat_type`, value, confidence, source |
| `ThreatSignal` | one processed message: raw_text, language, intent, entities (capped at 25) |
| `KafkaEnvelope` | transport wrapper |
| `TriageVerdict`, `GraphNeighbour`, `GraphContext`, `SynthesisState` | LangGraph Layer 4 state |

`ThreatEntity.source` records provenance: `cni-gazetteer-*`, `transformers-ner`,
or `heuristic`. Ablation and error analysis depend on it.

---

## 4. Module reference

### `src/core/`

| Module | Responsibility |
|---|---|
| `config.py` | `Settings` dataclass from env. **Raises `RuntimeError` without `NEO4J_PASSWORD` and `POSTGRES_DSN`.** |
| `logger.py` | JSON formatter; collects `extra={...}` into the payload |
| `schemas.py` | all Pydantic contracts |
| `identity.py` | `entity_id()` — the single graph-key rule, shared by both Layer 4 implementations |
| `messaging.py` | Kafka config factories, `encode()`/`dumps()` (native UTF-8), `extract_text()` |

### `src/layer1_ingestion/`

| Module | Responsibility |
|---|---|
| `normalizer.py` | NFKC + zero-width strip; `detect_script()` |
| `telegram_scraper.py` | collection from `config/telegram_targets.txt`; `--sink archive\|kafka` |
| `capture_sink.py` | Kafka → `data/raw/YYYY-MM-DD.jsonl`, idempotent on `topic:partition:offset` |

### `src/layer3_native_nlp/`

| Module | Responsibility |
|---|---|
| `gazetteer.py` | **dependency-free** repair logic. Imports only schemas, so it loads with no `.env` and no infrastructure — this is what makes offline benchmarking possible. |
| `entity_extractor.py` | fine-tuned XLM-R-base (ADR-004) + zero-shot intent + Kafka consumer |

`refine_entities()` runs four tiers. Measured contribution (dev split, n=44):

| Tier | F1 without it | Verdict |
|---|---|---|
| raw-text recovery | 0.298 | load-bearing |
| fragment cleanup | 0.491 | contributes |
| skeleton (virama) match | 0.553 | **zero effect** |
| fuzzy match | 0.553 | **zero effect** |

Baseline hybrid is 0.553. The last two tiers change zero records
(McNemar b=0, c=0). This is reported rather than hidden.

### `src/layer4_graphrag/`

| Module | Responsibility |
|---|---|
| `neo4j_connector.py` | Kafka → Neo4j. Batched `UNWIND` writes inside one `execute_write`; retrying buffer |
| `synthesis_agent.py` | LangGraph: triage → extraction → graphrag_synthesis, dependency-injected |

### `src/layer5_output/` and `src/ui_dashboard/`

`stix_formatter.py` builds STIX 2.1 bundles (Identity, Location, AttackPattern,
ThreatActor, Relationship), serialized with `ensure_ascii=False`.
`app.py` is the Streamlit console: live feed, graph explorer, STIX export.

---

## 5. Components deliberately excluded from the data path

**Triage classifier.** Trained on 13 hand-written lines; measures recall 0.250
at its 0.6 threshold. Routing Layer 3 behind it would discard roughly three
quarters of real threats. It is wired *out*, not tuned.

**Intent classifier.** `bart-large-mnli` is English-only; on Tamil its label
distribution flattens toward uniform. Its output is not a prediction.
`synthesis_agent` returns `"unknown"` instead, which is the honest value.

---

## 6. Reproducing the results

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in real values
pytest tests/ -q              # 153 assertions, no infrastructure needed

python scripts/build_eval_seed.py
python scripts/build_finetune_corpus.py
python scripts/finetune_model.py --base xlm-roberta-base --slug xlmr  # ~13 min on Apple MPS
python scripts/evaluate_pipeline.py --split test --arms all
```

Deterministically seeded (20260821); re-running produces byte-identical
`metrics.json`. Outputs land in `results/<split>-<git-sha>/`.

Batch-process a collected corpus without Kafka:

```bash
python scripts/run_batch_pipeline.py --out results/batch-full
```

---

## 7. Known defects

Recorded rather than silently carried. Each changes behaviour to fix, so none
were fixed during the code-quality pass.

1. **Topic-name mismatch** — collectors hardcode `raw_threat_stream`, config
   defaults to `raw_osint_stream`. Deploy without `KAFKA_RAW_TOPIC` set and the
   pipeline goes silent with no error. The k8s Secret does not carry it.
2. **Poison messages are committed** — `neo4j_connector`'s validation handler
   commits the offset on failure. No dead-letter topic; the message is lost.
3. **Buffer eviction is silent data loss** — `deque(maxlen=1000)` drops the
   oldest signal during a sustained Neo4j outage.
4. **Per-message `producer.flush(10.0)`** in `entity_extractor` caps throughput
   at broker round-trip latency.
5. **Triage consumer auto-commits** — the original config omitted
   `enable.auto.commit`, inheriting `True`. Now explicit with a comment;
   flipping it requires adding commit calls first.
6. **Module-scope `get_settings()`** in four modules — importing them requires
   full infrastructure config.
7. **20 files hand-roll a `sys.path` bootstrap** — the fix is packaging.

**Fixed, not carried:** `_ner_entities()` raised on an empty-string entity
value (some tokenizers' `aggregation_strategy="simple"` can yield a span that
strips to empty), and the surrounding exception handler discarded every other
entity already found in that record — not just the empty one. Found on the
real 433-record Telegram archive after ADR-004 (93/433 records affected).
Fixed by skipping the empty value instead of constructing it; regression test
in `tests/test_entity_extractor.py`.

---

## 8. Testing

153 assertions across 9 modules, all offline, no infrastructure required.

| Module | Covers |
|---|---|
| `test_gazetteer.py` | repair tiers, thresholds, non-matches |
| `test_capture_sink.py` | record shape, idempotency, fsync |
| `test_neo4j_writer.py` | batched write shape, relation allow-list, round-trip bound |
| `test_synthesis_agent.py` | LangGraph nodes, early-exit edges, injected deps |
| `test_native_utf8.py` | **parametrized guard**: every `json.dumps` in `src/` must declare `ensure_ascii` |
| `test_normalizer.py`, `test_schemas.py`, `test_logger.py` | contracts |

The UTF-8 guard is self-extending: it discovers new files automatically.
