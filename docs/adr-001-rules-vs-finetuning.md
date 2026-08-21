# ADR-001: Curated gazetteer over transformer fine-tuning for CNI entity extraction

**Status:** Accepted · **Date:** 2026-08-21 · **Supersedes:** the "zero-translation
native NLP" framing in the original project description

## Context

Phase 3 was to decide between (a) investing in domain fine-tuning of
`ai4bharat/IndicNER` via PEFT/LoRA, or (b) treating a curated gazetteer plus
rules as the primary extractor. The plan deferred the decision until the
Phase 2 benchmark produced numbers. It has.

Measured on the dev split (81-record seed set, 44 dev / 37 test, synthetic,
Tamil pending fluent verification). Full output: `results/dev-<sha>/`.

| Arm | P | R | F1 | held_out F1 | native_tamil F1 |
|---|---|---|---|---|---|
| gazetteer alone | **1.000** | 0.658 | **0.794** | 0.000 | 0.769 |
| hybrid (production) | 0.464 | 0.684 | 0.553 | 0.077 | 0.385 |
| transformer alone | 0.054 | 0.079 | 0.064 | 0.077 | **0.000** |
| heuristic regex | 0.136 | 0.079 | 0.100 | 0.000 | 0.000 |

Paired McNemar: gazetteer vs hybrid b=8 c=1 χ²=4.00 (**p<0.05**);
gazetteer vs transformer b=20 c=1 χ²=15.43 (**p<0.05**).

## Decision

**Adopt the curated gazetteer as the primary CNI entity extractor. Do not
invest in PEFT/LoRA fine-tuning of IndicNER.** Remove the transformer from the
production entity path pending evidence it adds value.

## Rationale

**1. The transformer is not neutral — it is harmful.** Adding IndicNER to the
gazetteer moves F1 from 0.794 to 0.553 and precision from 1.000 to 0.464, at
p<0.05. The audit's earlier claim that it "contributes nothing" was too
generous.

**2. It does not generalise, which was the only argument for keeping it.** The
`held_out` slice exists precisely to test this: CNI entities (NPCIL, BARC,
Sriharikota, Kudankulam, Trombay, Tarapur, Ennore, HAL) deliberately absent
from the gazetteer, where gazetteer recall is 0.000 by construction. If the
model had value, it would appear there. It scores F1 0.077 — indistinguishable
from noise. A lookup table's known weakness is not offset by the model.

**3. It scores 0.000 on native Tamil and romanized text.** Its only non-zero
script is Latin (0.140). A model that fails entirely on native script cannot
support a native-script thesis.

**4. Fine-tuning has no viable substrate.** PEFT/LoRA would need thousands of
labelled native-script CTI examples; the project has 81 synthetic records and
no captured corpus. Base performance of 0.064 F1 is not a promising starting
point, and IndicNER is a general PER/LOC/ORG model with no threat ontology.

**5. Curation cost is linear and interpretable.** Each gazetteer entry is one
line, deterministic, auditable, and yields precision 1.000. For a *fixed,
enumerable* target set — Indian CNI assets number in the low hundreds — this is
the correct engineering shape. Adding the 8 held-out terms would raise
`held_out` F1 from 0.077 toward ~1.0; that is arithmetic, not a research
finding, and it is exactly the point.

## Consequences

- The heuristic regex extractor should also be removed or rewritten: F1 0.100,
  and it emits generic nouns (`city`, `port`, `airport`) as location *values*,
  which is graph pollution.
- **The repair logic's contribution is unmeasurable and must not be claimed.**
  `abl_no_skeleton` and `abl_no_fuzzy` are byte-identical to `hybrid`
  (McNemar b=0, c=0). The virama-normalisation and fuzzy-matching tiers change
  no records, because raw-text recovery already resolves those cases and
  fragment cleanup deletes what remains. Removing raw-text recovery drops F1
  0.553 → 0.298 — it is the only component carrying weight.
- The paper's contribution is now a **negative result**: a widely-used Indic
  NER model is decisively beaten by a curated gazetteer on code-switched CNI
  entity extraction, with an ablation isolating why. This is publishable and
  true; the original "zero-translation architecture" framing is neither novel
  nor supported.

## Honest limitations

- **The gazetteer's 0.794 is a composition effect.** It scores 0.962 on
  `in_gazetteer` and 0.000 on both other slices; the dev split is 50%
  `in_gazetteer` by entity count. Under different traffic the overall figure
  moves. The *comparison* is unaffected — the transformer is worse on every
  slice and every script — but the absolute number is not a field estimate.
- Synthetic, self-authored eval data. Real harvested traffic will differ.
- Tamil records are `tamil_verified: false`. **Do not publish these numbers
  until a fluent reader has verified them.**
- Scoring is value-level, not span-level, because the pipeline discards
  character offsets. Gold data stores them; this upgrades once Phase 4 fixes
  the pipeline.
- n=44 dev; CIs are wide (gazetteer F1 95% CI 0.667–0.889).

## Revisit if

A fine-tuned or alternative multilingual model (MuRIL, XLM-R, a CTI-domain
model) beats the gazetteer on the `held_out` slice on verified, harvested data.
That is the only evidence that should reopen this.
