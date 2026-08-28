# SAINTEL Evaluation Set — Annotation Guidelines

Version 0.1 · Tamil-first · applies to `eval_set.dev.json` and `eval_set.test.json`

## Why this document exists

Two annotators must produce the same labels for the same message, or Cohen's κ
is meaningless and the benchmark cannot be defended. Every rule below exists
because a real case was ambiguous.

---

## 1. The two labels are independent

The single most common annotation error is conflating these.

- **`entities`** — *what things are named in this text?* Purely linguistic.
- **`triage_label`** — *does this message warrant analyst attention?*

"Travelling to **Chennai** for my cousin's wedding" contains a **correct
location entity** (`Chennai`) and is **`noise`**. Annotate both accordingly.
Never suppress an entity because the message is benign, and never mark a
message `signal` merely because a CNI term appears in it.

## 2. `triage_label`

| Label | Criterion |
|---|---|
| `signal` | States, plans, claims, or credibly threatens hostile action; or reports operational security incidents |
| `noise` | Everything else — including all ordinary discussion of CNI places and organisations |

Decision rules:

- **Public news about a CNI org is `noise`.** "ISRO launched a satellite today"
  is journalism, not intelligence.
- **Intent, not vocabulary.** Words like *attack*, *target*, *shut down* appear
  constantly in sports, gaming, and business talk. Label the meaning.
- **Boasting/claiming a past breach is `signal`.** Leak-site posts count.
- **When genuinely torn, label `noise`** and set `notes`. A CTI system's cost
  of a false positive is analyst time and, for a system that profiles regional
  language, real harm to real people. Bias toward `noise`.

## 3. `intent_label`

One of: `cyberattack`, `disinformation`, `infrastructure_disruption`,
`physical_threat`, `social_engineering`, `surveillance`, `security_alert`,
`benign`.

Every `noise` record is `benign`. Only `signal` records take a threat intent.
Choose the *primary* intent; do not multi-label.

## 4. Entity annotation

Types: `actor`, `location`, `organization`, `tactic`, `malware`, `indicator`,
`threat_type` (constrained by `ThreatEntity` in `src/core/schemas.py`).

- **Annotate the surface form as written**, in its original script. If the text
  says `சென்னை`, the entity value is `சென்னை`, not `Chennai`.
- **Ontological strictness.** `Avadi`/`Chennai` are `location`. `DRDO`/`ISRO`
  are `organization`. Never conflate. A facility *name* is an organization; the
  *place* it sits in is a location.
- **Offsets** (`start`/`end`) are computed automatically by
  `scripts/build_eval_seed.py`; annotators supply values only.
- **Do not annotate** generic nouns. `city`, `port`, `airport` are not
  locations. (The current heuristic extractor emits these — that is a defect
  the benchmark is designed to expose, not a labelling convention.)

## 5. `gazetteer_slice` — the field the experiment turns on

| Value | Meaning |
|---|---|
| `in_gazetteer` | Every CNI entity is one of the 11 terms in `CNI_GAZETTEER` |
| `held_out` | Contains ≥1 real CNI entity **deliberately absent** from the gazetteer |
| `none` | No CNI entity at all |

**This is not bookkeeping.** If every CNI entity were in the gazetteer, a
lookup table would score ~100% and the benchmark would prove nothing. The
`held_out` slice is where gazetteer recall is 0 by construction and only a
model can score — it answers whether the transformer generalises past a dict.

Held-out entities in use: HAL, NPCIL, BARC, Sriharikota, Kudankulam, Trombay,
Tarapur, Ordnance Factory Medak, Manali refinery, Ennore port.
**Do not add any of these to `CNI_GAZETTEER`** — doing so silently invalidates
every reported number.

## 6. `script` and `language`

`script`: `native_tamil` · `romanized` · `mixed` · `latin`
`language`: `ta` · `bn` · `hi` · `en` · `code_switched`

`mixed` means native script and Latin in the same message (e.g. Tamil prose
containing `DRDO`). This is the tokenizer-fragmentation case the repair logic
targets — annotate it accurately, the per-script breakdown depends on it.

## 7. Provenance and ethics

- `source`: `synthetic` (authored) or `harvested` (real capture).
- `provenance`: for harvested records, channel/URL plus capture date.
- **Never store victim PII.** Ransomware leak posts: record the claiming group
  and the fact of the claim; redact individual names, addresses, documents.
- Harvested records need a defensible collection note for the paper's ethics
  statement.

## 8. Tamil verification

Records carry `tamil_verified`. Anything authored by a non-fluent contributor
starts `false` and must be reviewed by a fluent reader before use in reported
results. **Do not report numbers over unverified Tamil.**

## 9. Inter-annotator agreement

A second fluent annotator independently labels a 15–20% subset. Report Cohen's
κ separately for `triage_label` and for entity spans. Adjudicate disagreements
by discussion and record the outcome. κ < 0.6 on either means the guidelines
are inadequate — fix them and re-annotate rather than proceeding.
