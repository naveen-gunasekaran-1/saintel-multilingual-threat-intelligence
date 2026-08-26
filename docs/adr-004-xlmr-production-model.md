# ADR-004: XLM-RoBERTa-base replaces IndicNER as the production extractor

**Status:** Accepted · **Date:** 2026-08-26 · **Supersedes:** ADR-002

## Context

ADR-002 adopted fine-tuned `ai4bharat/IndicNER` as the production entity
extractor on the strength of one number: held-out F1 0.875. `docs/model_comparison.md`
then ran the same fine-tuning procedure — identical corpus, label set,
hyperparameters, seed — against four alternative encoders, including
`xlm-roberta-base`, the model the originally approved architecture specified
for this layer before it was substituted for IndicNER without a decision
record (see ADR-003 for the parallel situation in Layer 4).

That comparison found IndicNER is **not** the best performer on aggregate
metrics. This ADR is the decision to switch, made explicitly rather than by
silent substitution — the same failure mode ADR-003 documents for Layer 4 is
not being repeated here.

## Results — test split, n=37, hybrid arm (model + gazetteer repair)

| Metric | IndicNER (previous) | XLM-R-base (new) | Δ |
|---|---|---|---|
| Overall F1 | 0.866 | **0.889** | +0.023 |
| Overall P / R | 0.853 / 0.879 | 0.933 / 0.849 | — |
| **held_out F1** | **0.875** | 0.800 | **−0.075** |
| native_tamil F1 | 0.923 | 0.923 | 0 |
| in_gazetteer F1 | 0.936 | 0.957 | +0.021 |
| Inference | 41.5 ms/record | 20.4 ms/record | 2× faster |
| Params | 166.8M | 277.5M | larger, but no gated-repo dependency |

McNemar (paired, n=37): b=1, c=2, χ²=0.00, **not significant** (p>0.05).

## Decision

**Adopt fine-tuned `xlm-roberta-base` as the production entity extractor.**
`FINETUNED_MODEL_DIR` now resolves to `data/models/xlmr-cni-ft`;
`BASE_NER_MODEL` fallback is `xlm-roberta-base`.

## The honest trade-off — read before citing either number

**This decision trades away the metric ADR-002 was accepted on.**
`held_out` F1 — CNI entities absent from both the gazetteer and the training
corpus, the one slice that tests generalisation rather than memorisation —
drops from 0.875 to 0.800. The gain is +0.023 overall F1, which is **not
statistically significant** at n=37 (McNemar χ²=0.00).

If the paper's headline claim is specifically "the fine-tuned model
generalises to unseen CNI entities," **this switch weakens that claim's best
evidence.** If the claim is broader — "a fine-tuned multilingual encoder plus
gazetteer repair outperforms both the gazetteer alone and the base
transformer" — this switch strengthens it marginally, and adds two more
substantial advantages documented below.

This ADR does not resolve which framing the paper should take. That is a
judgement call for whoever is writing it, and it should be made knowingly,
against these numbers, not discovered after the fact.

## Rationale for the switch despite the held_out cost

**1. It matches the approved architecture.** The originally reviewed and
signed-off design specified XLM-RoBERTa for this layer. IndicNER was
substituted at some point before this project's version-control history
begins, with no record of why. Reverting closes that gap between the
architecture diagram and the shipped system (see ADR-003, which documents the
same class of undocumented substitution still open in Layer 4).

**2. No gated-repo dependency.** `ai4bharat/IndicNER` requires an
authenticated Hugging Face token; `xlm-roberta-base` does not. This removes
`HF_TOKEN` as a hard requirement for reproducing the entity-extraction path
end to end — a real reproducibility improvement, independent of accuracy.

**3. An unplanned finding in `docs/model_comparison.md` §4 favours the base
tokenizer, not just this checkpoint.** IndicNER's *raw* (pre-repair)
`native_tamil` F1 is 0.333; XLM-R's is 0.923, identical to its hybrid score —
its tokenizer already preserves the Tamil virama that IndicNER's drops,
verified manually on a held-out record. Some of the fragmentation problem
this project characterises may be specific to IndicNER's WordPiece
vocabulary rather than inherent to the task. That does not invalidate the
paper's motivation section — the failure is still real and still measured —
but it changes which encoder should carry the fix.

**A consequence that must be stated plainly: the gazetteer's complementarity
claim, central to ADR-002's rationale, is weaker for this pairing.**

| Split | finetuned F1 | finetuned+gazetteer F1 | Gazetteer's marginal gain |
|---|---|---|---|
| test (n=37) | 0.889 | 0.889 | **0.000 — identical on every slice and script** |
| dev (n=44), native_tamil | 0.875 | 0.941 | +0.066, smaller than IndicNER's |

With IndicNER, the gazetteer recovered a collapse (native_tamil 0.333 to
0.923, +0.590). With XLM-R, on the test split it recovers **nothing
measurable** — XLM-R's own tokenizer already gets Tamil entities right, so
there is usually nothing left to repair. Retain the gazetteer regardless: it
is what makes `held_out` non-zero for the gazetteer-only arm, it backs the
raw-text-recovery tier the ablations already show is load-bearing, and dev
shows a real if small gain. But the two-component complementarity narrative
in ADR-002 and the README describes IndicNER's behaviour, not XLM-R's, and
must be rewritten for this pairing rather than reused with new numbers
substituted in — reusing it would overstate what the gazetteer now does.

**4. Faster inference, larger model notwithstanding.** 20.4ms vs 41.5ms per
record on identical hardware.

## Consequences

- `docs/model_comparison.md`, `entity_extractor.py`, and
  `scripts/evaluate_pipeline.py` now treat `xlm-roberta-base` as the
  reproduction target. `scripts/finetune_indicner.py` still works and still
  reproduces the ADR-002 IndicNER numbers exactly — it is not deleted,
  because those numbers are cited in this document and must remain
  reproducible.
- `results/dev-1469a64/` and `results/test-1469a64/` are the new headline
  numbers. Prior `results/*-<sha>/` directories are not deleted; they are the
  evidence this ADR's comparison table cites.
- Any paper text already written against ADR-002's 0.866/0.875 must be
  checked against this document before submission — the honest overall
  number is now 0.889, and the honest held_out number is now 0.800.
- **The README's headline table, architecture notes, and model references
  need updating to match** — tracked as the next step after this ADR, not
  bundled into it, so the decision and its documentation land as separable,
  reviewable commits.

## Revisit when

A real harvested Tamil corpus exists (`docs/collection_protocol.md`). Whether
IndicNER's tokenizer disadvantage or XLM-R's held_out shortfall dominates on
real text — as opposed to the templated synthetic corpus both were fine-tuned
and measured on — is untested and is the single highest-value experiment to
run once real data exists.
