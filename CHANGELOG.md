# Changelog

Research prototype. Versions mark meaningful states of the work, not releases.

## [Unreleased] — 2026-08-25

### Added
- `src/core/identity.py` — single definition of the graph entity key, shared by
  both Layer 4 implementations (previously duplicated with a "must stay
  byte-identical" comment)
- `src/core/messaging.py` — Kafka config factories, native-UTF-8 `encode()`,
  and `extract_text()` (the text-field chain previously differed between stages)
- `tests/test_neo4j_writer.py` — 9 assertions on a module that had none
- `scripts/telegram_login.py`, `scripts/verify_targets.py` — one-time auth and
  pre-collection target verification
- `scripts/run_batch_pipeline.py` — batch extraction over the raw archive, no
  Kafka required
- `scripts/harvest_literature.py` + `docs/litreview/` — 535 Crossref-verified
  journal articles, 92 SCIE candidates, BibTeX with DOIs and ISSNs
- `config/telegram_targets.txt` — collection targets, committed for reproducibility
- `docs/collection_protocol.md`, `docs/ethics_statement.md`,
  `docs/adr-003-langgraph-layer4.md`, `CHANGELOG.md`

### Changed
- **Neo4j writes batched** — was one autocommit query per entity and per entity
  *pair* (`1 + N + N²`); now three to six `UNWIND` batches in a single
  transaction. A 20-entity signal went from up to 421 round trips to 3, and is
  now atomic.
- `telegram_scraper.py` reads targets from a file, supports `--sink archive`,
  records provenance, and **exits non-zero rather than scraping placeholders**
- `stix_formatter.py` converted from tabs to spaces (verified AST-identical)
- Ad-hoc dev scripts moved to `scripts/devtools/` — `test_*` names alongside a
  real `tests/` directory invited running the wrong thing

### Removed
- `src/layer3_native_nlp/intent_classifier.py` — 0 bytes, imported nowhere

### Fixed
- macOS certificate failure in the literature harvester (`urllib` has no root
  certs on the python.org build; switched to `requests`)

### Known issues
Seven defects documented in `docs/project_documentation.md` §7. Each changes
behaviour to fix and was deliberately left alone during a code-quality pass.

---

## [0.5.0] — 2026-08-23 — Darkweb collection restored
Installed `beautifulsoup4` and `stem`; restored the `[socks]` extra on
`requests` that had been dropped during Phase 1 pinning, silently removing all
SOCKS5 support. Verified against live Tor.

## [0.4.0] — 2026-08-22 — Native script across all output paths
`ensure_ascii=False` was missing at 6 of 8 `json.dumps` sites plus
`stix2.serialize()` — STIX exported `சென்னை` as escape sequences. Fixed and
guarded by a parametrized test that discovers new files automatically.

## [0.3.0] — 2026-08-21 — ADR-001 reversed by its own experiment
ADR-001 rejected fine-tuning **without testing it**. Tested: base IndicNER
F1 0.186 → fine-tuned 0.758 (McNemar b=0, c=14, p<0.05); held-out entities
0.875 against the gazetteer's structural 0.000. ADR-002 supersedes it. Both
documents retained.

## [0.2.0] — 2026-08-21 — Ground truth and first real numbers
Extracted `gazetteer.py` as a dependency-free module so benchmarking can run
offline. Built frozen dev/test splits and a harness with ablations, bootstrap
CIs, and McNemar tests. Ablation found two of four repair tiers change zero
records; reported rather than hidden.

## [0.1.0] — 2026-08-21 — Version control and security hygiene
`.gitignore` before `git init`, so no secret ever entered history. Pinned all
dependencies to the versions results were produced on. Rotated credentials.

## [0.0.0] — Baseline
Six-layer pipeline imported under version control: 2,399 lines, zero tests,
zero evaluation data.
