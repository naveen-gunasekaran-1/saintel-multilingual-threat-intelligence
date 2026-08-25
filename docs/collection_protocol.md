# Data Collection Protocol

This document exists because "how did you choose your sources?" is the first
question asked of any OSINT corpus, and an answer reconstructed after the fact
is not a protocol. It records what was actually done, including what failed.

Last updated: 2026-08-25.

---

## 1. Sources

| Layer | Source | Access | Status |
|---|---|---|---|
| L1 | Public Telegram channels | Telethon, MTProto | **Active** |
| L0 | Tor ransomware leak site | SOCKS5 `127.0.0.1:9050` | Verified, not run for corpus |
| L0 | RansomLook API | public HTTPS, no auth | Verified, not run for corpus |

Only **public** channels are collected. No private groups, no invite-only
content, no accounts created to gain access, no interaction with any channel —
collection is read-only.

## 2. Target selection

Targets live in `config/telegram_targets.txt`, which is **committed on
purpose**: a source list that is not published is not reproducible.

Selection criteria, applied in order:

1. **Language relevance** — channels publishing in Tamil, since the
   contribution concerns Tamil-script entity extraction.
2. **Domain relevance** — channels reporting on infrastructure, security
   incidents, or threat activity.
3. **Public accessibility** — readable without joining, without payment.

Every target must pass `scripts/verify_targets.py` before collection. That tool
exists because the first run failed silently on two thirds of its list.

## 3. Verification before collection

```bash
python scripts/verify_targets.py --file config/telegram_targets.txt
```

Four verdicts:

| Verdict | Meaning | Action |
|---|---|---|
| `OK` | resolves, recent posts carry text | collect |
| `NO_HANDLE` | username does not exist | fix the handle |
| `NO_HISTORY` | resolves, returns nothing | requires joining — **excluded** |
| `NO_TEXT` | media-only posts | unusable for NLP |

`NO_HISTORY` channels are excluded rather than joined. Joining a threat-actor
channel is participation, not observation, and changes the ethical and legal
position of the collection.

## 4. Collection

```bash
python src/layer1_ingestion/telegram_scraper.py --limit 200 --sink archive
```

- Archives to `data/raw/YYYY-MM-DD.jsonl`, one JSON record per message.
- Idempotency key `telegram:<channel>:<message_id>` — reruns deduplicate.
- Provenance recorded per record: `t.me/<channel> @ <ISO timestamp>`.
- Native UTF-8 preserved; **nothing is translated at any stage**.
- The archive is gitignored and dockerignored.

## 5. Collection log

### Run 1 — 2026-08-25

9 targets configured, **3 produced data**, 433 messages.

| Channel | Messages | Outcome |
|---|---|---|
| News18TamilNadu | 200 | collected (limit reached) |
| falconfeedsio | 196 | collected |
| PolimerNewsOfficial | 37 | collected |
| ThanthiTVOfficial | 0 | `NO_HANDLE` |
| daily_dark_web | 0 | `NO_HANDLE` |
| mysteriousteambd_real | 0 | `NO_HANDLE` |
| puthiyathalaimuraitv | 0 | `NO_TEXT` — media-only |
| darkwebinformer | 0 | `NO_HISTORY` — requires joining |
| dragonforceio | 0 | `NO_HISTORY` — requires joining |

**Composition:** 54.3% mixed script, 45.5% Latin, 0.2% native Tamil by
`detect_script()`. By Tamil character density, only **42 of 433 records (9.7%)
carry substantive Tamil** (>30% Tamil characters).

**Yield against purpose: zero.** Batch extraction over all 433 records produced
**0 CNI-relevant entity mentions**. The corpus is Tamil political and general
news; the gazetteer fired 26 times, only ever on `Chennai`/`சென்னை`.

**Conclusion recorded at the time:** recent-post collection from general Tamil
news does not yield critical-infrastructure content at usable density. Future
collection should be query-driven against infrastructure terminology rather
than recency-driven.

## 6. Annotation

Not yet performed. When it is:

- Sample via `scripts/sample_for_annotation.py --n 300` — deterministic,
  stratified by script, PII-redacted (email, URL, phone, ID patterns) on output.
- Label per `tests/data/annotation_guidelines.md`.
- **A second annotator labels 15–20%**; report Cohen's κ separately for entity
  spans and triage label. Sole-annotator ground truth is a standing reviewer
  objection and cannot be retrofitted.

## 7. Retention

- Raw archive stays local; gitignored, dockerignored, never published.
- Only **aggregate statistics and redacted samples** appear in any output.
- Delete the raw archive when the annotated corpus is frozen; the annotated
  set, not the archive, is the research artifact.

## 8. Outstanding before publication

- [ ] Institutional ethics approval or a documented determination that none is required
- [ ] Retention period agreed and recorded above
- [ ] Second annotator identified, κ budgeted
- [ ] Corpus containing actual CNI entity mentions (§5 shows the current one does not)
