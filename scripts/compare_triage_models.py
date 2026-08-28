"""Compare fastText against 3 alternative triage classifiers.

Read the sample-size caveat before the table: training data is 13 hand-written
lines. No classifier choice is statistically justified on 13 examples -- any
comparison at this n measures which model overfits most gracefully, not which
generalises best. This is run and reported anyway because "why fastText"
deserves an answer grounded in evidence rather than convenience, and because
the honest answer -- the comparison is inconclusive at this sample size, which
is itself why the classifier is disconnected from the data path -- is a
legitimate methodological finding for the paper.

All four candidates train on the SAME 13 lines and are scored on the SAME
held-out set: the eval corpus's triage_label field (dev+test combined, n=81),
which the training data has zero overlap with.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_pipeline import prf

TRAIN_PATH = ROOT / "data" / "datasets" / "training_data.txt"
DATA_DIR = ROOT / "tests" / "data"


def load_train() -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for line in TRAIN_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        label, text = line.split(" ", 1)
        labels.append(label.replace("__label__", ""))
        texts.append(text)
    return texts, labels


def load_eval() -> list[dict]:
    records = []
    for split in ("dev", "test"):
        records += json.loads((DATA_DIR / f"eval_set.{split}.json").read_text(encoding="utf-8"))
    return [r for r in records if "triage_label" in r]


def score_at_threshold(rows: list[dict], threshold: float) -> dict:
    tp = sum(1 for x in rows if x["signal_conf"] > threshold and x["gold"] == "signal")
    fp = sum(1 for x in rows if x["signal_conf"] > threshold and x["gold"] == "noise")
    fn = sum(1 for x in rows if x["signal_conf"] <= threshold and x["gold"] == "signal")
    tn = sum(1 for x in rows if x["signal_conf"] <= threshold and x["gold"] == "noise")
    s = prf(tp, fp, fn)
    s["fpr"] = round(fp / (fp + tn), 4) if (fp + tn) else 0.0
    s["threshold"] = threshold
    return s


def best_and_at06(rows: list[dict]) -> dict:
    sweep = [score_at_threshold(rows, i / 20) for i in range(1, 20)]
    at06 = min(sweep, key=lambda s: abs(s["threshold"] - 0.6))
    best = max(sweep, key=lambda s: s["f1"])
    return {"at_threshold_0.6": at06, "best_threshold": best}


# --------------------------------------------------------------- candidates --

def run_fasttext(train_texts, train_labels, eval_records) -> list[dict]:
    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None
    tmp = ROOT / "data" / "datasets" / "_triage_compare_tmp.txt"
    tmp.write_text("\n".join(f"__label__{l} {t}" for l, t in zip(train_labels, train_texts)), encoding="utf-8")
    model = fasttext.train_supervised(str(tmp), epoch=25, lr=1.0, wordNgrams=2, verbose=0)
    tmp.unlink()
    rows = []
    for r in eval_records:
        lab, conf = model.predict(r["text"].replace("\n", " ").strip())
        label = lab[0].replace("__label__", "")
        signal_conf = float(conf[0]) if label == "signal" else 1.0 - float(conf[0])
        rows.append({"gold": r["triage_label"], "signal_conf": signal_conf})
    return rows


def run_tfidf_logreg(train_texts, train_labels, eval_records) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X = vec.fit_transform(train_texts)
    y = [1 if l == "signal" else 0 for l in train_labels]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    Xe = vec.transform([r["text"] for r in eval_records])
    probs = clf.predict_proba(Xe)[:, 1]
    return [{"gold": r["triage_label"], "signal_conf": float(p)} for r, p in zip(eval_records, probs)]


def run_naive_bayes(train_texts, train_labels, eval_records) -> list[dict]:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X = vec.fit_transform(train_texts)
    y = [1 if l == "signal" else 0 for l in train_labels]
    clf = MultinomialNB().fit(X, y)
    Xe = vec.transform([r["text"] for r in eval_records])
    probs = clf.predict_proba(Xe)[:, 1]
    return [{"gold": r["triage_label"], "signal_conf": float(p)} for r, p in zip(eval_records, probs)]


KEYWORDS = {"urgent", "alert", "threat", "attack", "breach", "coordinat",
           "reconnaissance", "restricted", "emergency", "protest", "disrupt",
           "intercept", "unauthorized", "leaked", "exfiltrat", "compromise"}


def run_keyword_baseline(train_texts, train_labels, eval_records) -> list[dict]:
    """Simplest possible baseline: fraction of threat-adjacent keywords present.
    No training beyond the fixed keyword list -- the floor any learned model
    must beat to justify its complexity."""
    rows = []
    for r in eval_records:
        t = r["text"].lower()
        hits = sum(1 for k in KEYWORDS if k in t)
        conf = min(1.0, hits / 3)
        rows.append({"gold": r["triage_label"], "signal_conf": conf})
    return rows


CANDIDATES = [
    ("fasttext", "fastText (current choice)", run_fasttext),
    ("tfidf_logreg", "TF-IDF (char 2-4gram) + Logistic Regression", run_tfidf_logreg),
    ("naive_bayes", "Char n-gram Multinomial Naive Bayes", run_naive_bayes),
    ("keyword_baseline", "Fixed keyword-match baseline (no learning)", run_keyword_baseline),
]


def main() -> int:
    train_texts, train_labels = load_train()
    eval_records = load_eval()
    print(f"train n={len(train_texts)}  eval n={len(eval_records)} "
          f"(gold: {Counter(r['triage_label'] for r in eval_records)})\n")

    results = {}
    for slug, label, fn in CANDIDATES:
        t0 = time.time()
        rows = fn(train_texts, train_labels, eval_records)
        elapsed = time.time() - t0
        scores = best_and_at06(rows)
        results[slug] = {"label": label, "train_seconds": round(elapsed, 4), **scores}
        print(f"{slug:18s} F1@0.6={scores['at_threshold_0.6']['f1']:.3f}  "
              f"best_F1={scores['best_threshold']['f1']:.3f} "
              f"@thr={scores['best_threshold']['threshold']}  "
              f"FPR@best={scores['best_threshold']['fpr']:.3f}")

    out = ROOT / "results" / "model_comparison" / "triage_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "train_n": len(train_texts), "eval_n": len(eval_records),
        "caveat": "13 training examples. No result here is statistically "
                  "reliable; comparison is reported to show the choice was "
                  "evidence-checked, not to claim significance.",
        "models": results,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
