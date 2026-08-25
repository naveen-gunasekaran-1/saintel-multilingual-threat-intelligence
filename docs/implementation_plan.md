# Implementation Plan & Status

Live status document. What is done, what is not, and what blocks publication.
Last updated: 2026-08-25.

---

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 1 | Security hygiene, version control, dependency pinning | **Done** |
| 2 | Ground truth, benchmark harness, first real numbers | **Done** |
| 3 | Architecture decision (rules vs fine-tuning) | **Done** — ADR-001 → ADR-002 |
| 4 | Topology defects, double-write removal, logger fix, test coverage | **Done** |
| 5 | GPU binding, fine-tuned model in production, Docker | **Done** |
| 6 | Real corpus collection | **Started — yield insufficient** |
| 7 | Annotation, κ, retrain on real data | **Not started** |
| 8 | Paper | **Not started** |

## What exists

- Six-layer pipeline, all layers implemented, four engineering directives verified holding
- 150 offline assertions; benchmark harness with ablations, bootstrap CIs, McNemar
- Fine-tuned IndicNER: F1 **0.866** overall, **0.875** on held-out entities (synthetic)
- Frozen dev/test splits, deterministically seeded and reproducible
- Collection tooling: target verification, archiving, batch extraction, PII-redacting sampler
- 92 Crossref-verified literature candidates with DOIs and ISSNs

## The blocking item

**A corpus containing actual CNI entity mentions.**

Run 1 collected 433 real Telegram records and produced **zero** CNI-relevant
entities (`collection_protocol.md` §5). The corpus is Tamil political news.
Recency-driven collection from general news does not reach the target domain.

Options, in order of expected yield:

1. **Query-driven collection** — search Tamil news archives for infrastructure
   terminology (கல்பாக்கம், அணுமின் நிலையம், துறைமுகம்) rather than pulling
   recent posts. Highest yield per record, no ethics gate.
2. **Corrected Telegram targets** — fix the three bad handles; decide whether
   joining the two `NO_HISTORY` channels is acceptable (currently excluded on
   ethical grounds).
3. **Accept the language gap** — collect from English-language hacktivist
   channels for threat register, and reframe the contribution accordingly.

## Publication paths

| Path | Deliverable | Blocked on |
|---|---|---|
| **A — Systems paper** | The pipeline itself, demonstrated end to end | Nothing. Achievable now. |
| **B — Field-degradation study** | Synthetic F1 0.866 vs 22% structural junk and 96% virama loss on 433 real records | Writing only. Data exists. |
| **C — Full contribution** | Fine-tuned hybrid validated on real annotated Tamil CNI text | Corpus (above) + annotation + κ |

Path B is the shortest route to a defensible result and discards nothing.
Path C is the strongest and needs months.

## Outstanding non-research items

- [ ] Resolve ADR-003 — two Layer 4 implementations, neither wired to the other
- [ ] Institutional ethics determination
- [ ] Retention period for the raw archive
- [ ] Second annotator for κ
- [ ] Seven known defects in `project_documentation.md` §7 (each changes behaviour)
