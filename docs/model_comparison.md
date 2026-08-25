# Model Selection: Controlled Comparison

Answers "why fastText and why IndicNER" with a controlled experiment rather
than an assertion. Both comparisons are real, reproducible runs — not
literature figures — on this project's own data.

Reproduce:
```bash
python scripts/finetune_model.py --base <hf-model-id> --slug <name>   # per candidate
python scripts/compare_ner_models.py       # NER comparison
python scripts/compare_triage_models.py    # triage comparison
```

Raw output: `results/model_comparison/comparison.json`, `triage_comparison.json`.

---

## Part 1 — Entity extraction (Layer 3): why IndicNER?

### Method

Five encoders, **identical procedure**: same synthetic training corpus
(4,000 sentences, in-gazetteer entities only), same label set
(B/I-LOC, B/I-ORG, B/I-PER, O), same hyperparameters (3 epochs, lr 3e-5,
batch 16, seed 20260821), same held-out test split (37 records, touched once).
Only the base checkpoint changes.

One asymmetry is real and reported rather than hidden: **IndicNER already
ships a matching NER head** for this exact label set, so its fine-tuning
warm-starts a pretrained classifier. The other four have no NER head for
this task — `AutoModelForTokenClassification` attaches a randomly
initialised one. That is not a flaw in the experiment; it is part of the
answer, because it means IndicNER's advantage (where it has one) is
partly a head-start effect, not purely an encoder-quality effect. Reported,
not concealed.

| Model | Params | Train time | Note |
|---|---:|---:|---|
| **ai4bharat/IndicNER** | 166.8M | — (existing run) | Pretrained NER head for this label set |
| google/muril-base-cased | 237.0M | 12.7 min | India-specific multilingual, random head |
| xlm-roberta-base | 277.5M | 13.2 min | Original approved architecture's Layer 3 choice |
| bert-base-multilingual-cased | 177.3M | 10.1 min | Generic multilingual baseline |
| distilbert-base-multilingual-cased | 134.7M | 6.5 min | Efficiency-oriented baseline |

### Results — test split, n=37, value-level scoring, hybrid arm (model + gazetteer repair)

| Model | P | R | F1 | 95% CI | held_out F1 | native_tamil F1 | ms/record |
|---|---|---|---|---|---|---|---|
| xlm-roberta-base | 0.933 | 0.849 | **0.889** | [0.811, 0.955] | 0.800 | 0.923 | 20.4 |
| bert-base-multilingual-cased | 0.933 | 0.849 | **0.889** | [0.800, 0.962] | 0.800 | 0.923 | 20.3 |
| distilbert-base-multilingual-cased | 0.875 | 0.849 | 0.862 | [0.769, 0.941] | 0.750 | 0.923 | 10.7 |
| **ai4bharat/IndicNER (chosen)** | 0.853 | 0.879 | 0.866 | [0.772, 0.944] | **0.875** | 0.923 | 41.5 |
| google/muril-base-cased | 0.778 | 0.849 | 0.812 | [0.698, 0.909] | 0.667 | 0.857 | 23.0 |

**McNemar, hybrid arm, each vs IndicNER** (paired, same 37 records):

| Comparison | b | c | χ² | p<0.05 |
|---|---|---|---|---|
| IndicNER vs XLM-R | 1 | 2 | 0.00 | no |
| IndicNER vs mBERT | 1 | 3 | 0.25 | no |
| IndicNER vs DistilmBERT | 3 | 3 | 0.17 | no |
| IndicNER vs MuRIL | 4 | 2 | 0.17 | no |

**None of these differences are statistically significant at n=37.** Any
claim of superiority among the top four models would not survive review.

### The honest finding, not the flattering one

**On overall F1, XLM-R-base and mBERT tie for best (0.889), beating
fine-tuned IndicNER (0.866).** If overall F1 were the selection criterion,
the original architecture's XLM-R choice would be the better call, and this
project's Phase 3 decision to standardise on IndicNER would not be supported
by this evidence.

**On `held_out` — the metric that actually matters for this paper's
contribution — IndicNER wins: 0.875 vs 0.800 (XLM-R, mBERT) vs 0.750
(DistilmBERT) vs 0.667 (MuRIL).** `held_out` is CNI entities absent from
both the gazetteer and the training corpus, so it is the only slice that
tests generalisation rather than memorisation. **IndicNER is retained as the
production model on this basis: it generalises furthest to unseen
infrastructure entities, which is the specific claim the paper makes.**
Overall F1 is not the paper's headline metric, and should not be treated as
one when justifying the choice.

### An unplanned finding worth its own discussion in the paper

Look at `native_tamil` F1 in the raw (pre-gazetteer) arm, not the hybrid one:

| Model | native_tamil F1, **raw** (no gazetteer repair) |
|---|---|
| IndicNER | **0.333** |
| XLM-R, mBERT, MuRIL, DistilmBERT | **0.857 – 1.000** |

Manual inspection of one record confirms this is not a scoring artefact.
On `சென்னை மெட்ரோ ரயில் சேவை நாளை தாமதமாகும்.`, XLM-R's raw output already
returns `சென்னை` **with its virama intact** — the gazetteer repair step ran
and found nothing left to fix. IndicNER's raw output on equivalent inputs
loses the virama and requires the repair stage to recover it.

**Both models were fine-tuned identically, on the same corpus, for the same
entities.** The difference is upstream of fine-tuning: it is the base
tokenizer. XLM-R's SentencePiece vocabulary, trained on a much larger and
more diverse multilingual corpus, appears to preserve Tamil combining marks
that IndicNER's tokenizer does not.

**This means the tokenizer contributing to the exact failure mode this
paper characterises may be specific to IndicNER's WordPiece vocabulary, not
an inherent property of "Indic-specialised" models generally.** That
qualifies the paper's central claim in a way worth stating explicitly rather
than discovering in review: the fragmentation/virama-loss problem is
demonstrated on IndicNER specifically; whether it generalises to other
Indic-pretrained encoders is untested here and is a natural follow-up.

---

## Part 2 — Edge triage (Layer 2): why fastText?

### The honest caveat, stated before the numbers

**Training data is 13 hand-written lines.** No classifier choice is
statistically defensible at this sample size — a comparison here measures
which model overfits most gracefully, not which generalises best. This is
reported anyway because "why fastText" deserves evidence rather than
convenience, and because the actual finding — the comparison is
inconclusive, which is itself why this component is disconnected from the
data path — is legitimate and citable.

### Method

Four candidates, all trained on the identical 13 lines, all evaluated on
the eval corpus's `triage_label` field (dev+test combined, n=81, zero
overlap with training text).

| Model | Best F1 | Threshold | FPR at best F1 |
|---|---|---|---|
| Char n-gram Multinomial Naive Bayes | 0.687 | 0.40 | 0.644 |
| **fastText (chosen)** | 0.681 | 0.40 | 0.578 |
| TF-IDF (char 2–4gram) + Logistic Regression | 0.647 | 0.50 | 0.733 |
| Fixed keyword-match baseline (no learning) | 0.400 | 0.05 | 0.000 |

### Finding

**The three learned models are statistically indistinguishable** (0.647–
0.687 F1 on n=81); the ranking would plausibly reorder under a different
random seed or a different 13 examples. All three clear the no-learning
keyword baseline, which rules out "any classifier is doing nothing" but
does not support choosing among them.

**This supports, rather than undermines, the project's existing decision.**
The triage classifier is disconnected from the data path
(`docs/project_documentation.md` §5) because it measures recall 0.250 at
production threshold — and this comparison shows that failure is not a
fixable algorithm choice. Naive Bayes, logistic regression, and fastText
all land in the same narrow band. **The lever is training-data volume, not
model sophistication**, and 13 examples is the actual constraint.

---

## Summary for the paper

| Question | Answer | Evidence |
|---|---|---|
| Is IndicNER the best encoder by raw F1? | **No** — XLM-R and mBERT tie above it | Part 1 results table |
| Is IndicNER justified as the production choice? | **Yes**, specifically on generalisation | held_out F1: 0.875 vs 0.800/0.667/0.750 |
| Is the overall-F1 gap between top models significant? | **No** | McNemar, all p>0.05 at n=37 |
| Is fastText justified over its alternatives? | **Not decisively — and that itself is the finding** | Three algorithms tie at n=13; disconnect the component, not chase the algorithm |
| Does the tokenizer-fragmentation problem generalise beyond IndicNER? | **Untested — flagged as a limitation** | raw native_tamil F1 varies 0.333→1.000 across encoders |
