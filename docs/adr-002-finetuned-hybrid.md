# ADR-002: Fine-tuned IndicNER + gazetteer hybrid as the production extractor

**Status:** Accepted · **Date:** 2026-08-21 · **Supersedes:** ADR-001

## Context

ADR-001 rejected fine-tuning on the grounds that base IndicNER scored F1 0.064,
did not generalise (held_out F1 0.077), and had no viable training substrate.
That decision was made *without testing it*. It has now been tested and is
wrong.

A synthetic training corpus (4,000 sentences, 4,622 spans) was generated using
**only in-gazetteer entities** plus non-CNI distractors. The eight held-out CNI
entities used by the evaluation set — NPCIL, BARC, Sriharikota, Kudankulam,
Trombay, Tarapur, Ennore, HAL — never appear in training; a programmatic
leakage check enforces this. IndicNER was then fine-tuned for token
classification (3 epochs, lr 3e-5, MPS, ~10 min) reusing its existing BIO head.

## Results — `test` split (37 records, untouched until this evaluation)

| Arm | P | R | F1 | **held_out F1** | native_tamil F1 |
|---|---|---|---|---|---|
| gazetteer | 1.000 | 0.667 | 0.800 | **0.000** | 1.000 |
| transformer (base) | 0.151 | 0.242 | 0.186 | 0.210 | 0.000 |
| hybrid (base+gaz) | 0.444 | 0.727 | 0.552 | 0.210 | 0.750 |
| finetuned | 0.758 | 0.758 | 0.758 | **0.875** | 0.333 |
| **finetuned_hybrid** | 0.853 | 0.879 | **0.866** | **0.875** | 0.923 |

Fine-tuning moved base IndicNER from F1 0.186 to 0.758 (McNemar b=0, c=14,
χ²=12.07, **p<0.05**). Dev split agrees: 0.064 → 0.667.

## Decision

**Adopt `finetuned_hybrid` — fine-tuned IndicNER with gazetteer refinement — as
the production entity extractor.** Retain the gazetteer; it is not redundant.

## Rationale

**1. Fine-tuning buys exactly the generalisation ADR-001 said it could not.**
On held-out CNI entities the model never saw, F1 is **0.875** against the
gazetteer's structural **0.000**. This is the question the held_out slice was
built to answer, and the answer reversed. The model learned "a CNI organisation
occupies this role in this context" rather than memorising the 11 strings.

**2. The two components are complementary, not competing.** Fine-tuned alone
collapses on native Tamil (0.333); the gazetteer alone is perfect there (1.000)
but scores 0.000 on anything outside its list. Combined: 0.923 on native Tamil
and 0.875 on held-out. Each covers the other's structural blind spot.

**3. The GPU infrastructure is justified again.** ADR-001 implied deleting the
k8s GPU manifests. With a transformer back in the production path, Phase 5
(`device=-1` → CUDA, the torch/base-image conflict) becomes live work again.

## Consequences

- `device=-1` in `entity_extractor.py` must be fixed; the model now does real work.
- The extractor should load `data/models/indicner-cni-ft`, not `ai4bharat/IndicNER`.
- The fine-tuned model is gitignored (~430 MB) but fully reproducible:
  `build_finetune_corpus.py` and `finetune_indicner.py` are deterministically
  seeded (20260821).
- ADR-001's finding that the **repair tiers** (skeleton, fuzzy) have no
  measurable effect still stands — it was not affected by this experiment.

## Honest limitations — read before citing any of this

**1. Training and evaluation data were both authored by the same process.**
Generalisation is demonstrated over **entity identity** (held-out strings, a
real and non-trivial result) but **not over linguistic register**. Both corpora
use similar sentence shapes. Real hacktivist text — slang, misspellings,
irregular syntax, unseen genres — is not represented. The 0.875 is an upper
bound, and the true field number is likely well below it.

**2. Error analysis shows positional pattern-learning.** The model emits
`location:biryani`, `location:kaaikari`, `location:marina` — unfamiliar tokens
in familiar template positions. That is template overfitting, visible despite
distractor entities in training.

**3. Overall superiority is NOT statistically significant on test.**
gazetteer vs finetuned_hybrid: b=3, c=5, χ²=0.125, p>0.05 at n=37. The
*held_out* difference (0.875 vs 0.000) is categorical and needs no test; the
*overall* 0.866 vs 0.800 does not survive one.

**4. `tactic` entities are structurally unlearnable.** IndicNER's head has only
LOC/ORG/PER, so `tactic:reconnaissance` and `tactic:exfiltrated` are missed by
construction. Extracting tactics needs an extended label set.

**5. Virama corruption persists.** The model still emits `செனனை` for `சென்னை`
— which is why the gazetteer repair stage remains necessary in the hybrid.

**6. Tamil records remain `tamil_verified: false`.** Do not publish these
numbers until a fluent reader verifies them.

## Revisit when

Real harvested data exists. The single highest-value experiment is retraining
on harvested text and re-running this comparison — that is what converts an
upper bound into a field estimate.
