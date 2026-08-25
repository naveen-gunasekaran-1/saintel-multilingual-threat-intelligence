"""Score every fine-tuned candidate model on the SAME frozen test split and
produce one comparison table -- the artifact that justifies "why IndicNER"
in the paper rather than asserting it.

Two arms per model, matching production usage:
    raw     the fine-tuned model alone
    hybrid  fine-tuned model + gazetteer repair (refine_entities)

Scoring is value-level (entity_type, casefolded value), identical to
evaluate_pipeline.py -- reused from there rather than reimplemented, so the
two harnesses cannot silently diverge on what "correct" means.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_pipeline import bootstrap_f1, key_set, mcnemar, prf, score_entities
from src.core.schemas import ThreatEntity
from src.layer3_native_nlp.gazetteer import refine_entities

MODELS_DIR = ROOT / "data" / "models"
DATA_DIR = ROOT / "tests" / "data"

CANDIDATES = [
    ("indicner",    "ai4bharat/IndicNER (current choice)"),
    ("muril",       "google/muril-base-cased"),
    ("xlmr",        "xlm-roberta-base (original design)"),
    ("mbert",       "bert-base-multilingual-cased"),
    ("distilmbert", "distilbert-base-multilingual-cased"),
]

TYPE_MAP = {"loc": "location", "location": "location", "gpe": "location",
            "org": "organization", "organization": "organization",
            "per": "actor", "person": "actor", "actor": "actor"}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return -1


def load_pipeline(model_dir: Path):
    from transformers import pipeline
    return pipeline("ner", model=str(model_dir), tokenizer=str(model_dir),
                    aggregation_strategy="simple", device=pick_device())


def run_ner(nlp, text: str) -> list[tuple[str, str, float]]:
    try:
        out = nlp(text)
    except Exception:
        return []
    results = []
    for item in out:
        tag = item.get("entity_group", "").lower()
        mapped = TYPE_MAP.get(tag, "threat_type")
        value = item.get("word", "").strip()
        if value:
            results.append((mapped, value, float(item.get("score", 0.5))))
    return results


def load_split(name: str) -> list[dict]:
    return json.loads((DATA_DIR / f"eval_set.{name}.json").read_text(encoding="utf-8"))


def score_arm(records, predict_fn) -> dict:
    per_record, per_slice, per_script, correctness = [], {}, {}, []
    for rec in records:
        gold = key_set((e["type"], e["value"]) for e in rec.get("entities", []))
        pred = key_set(predict_fn(rec))
        per_record.append((gold, pred))
        correctness.append(gold == pred)
        sl = rec.get("gazetteer_slice", "none")
        per_slice.setdefault(sl, []).append((gold, pred))
        sc = rec.get("script", "unknown")
        per_script.setdefault(sc, []).append((gold, pred))

    overall = score_entities(per_record)
    ci = bootstrap_f1(per_record)
    if ci:
        overall.update(ci)
    return {
        "overall": overall,
        "by_slice": {k: score_entities(v)["f1"] for k, v in per_slice.items()},
        "by_script": {k: score_entities(v)["f1"] for k, v in per_script.items()},
        "record_exact_acc": round(sum(correctness) / len(correctness), 4) if correctness else 0.0,
        "_correctness": correctness,
    }


def main() -> int:
    test = load_split("test")
    results = {}

    for slug, label in CANDIDATES:
        model_dir = MODELS_DIR / f"{slug}-cni-ft"
        if not (model_dir / "config.json").exists():
            print(f"SKIP {slug}: not trained yet ({model_dir})")
            continue

        meta_path = model_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        print(f"Loading {slug} ({label})...", flush=True)
        t0 = time.time()
        nlp = load_pipeline(model_dir)
        load_s = time.time() - t0

        t0 = time.time()
        raw_scores = score_arm(test, lambda r, nlp=nlp: [(t, v) for t, v, _ in run_ner(nlp, r["text"])])
        infer_s = (time.time() - t0) / max(len(test), 1)

        def hybrid_predict(rec, nlp=nlp):
            raw = [ThreatEntity(entity_type=t, value=v, confidence=c, source="transformers-ner")
                   for t, v, c in run_ner(nlp, rec["text"])]
            return [(e.entity_type, e.value) for e in refine_entities(rec["text"], raw)]

        hybrid_scores = score_arm(test, hybrid_predict)

        results[slug] = {
            "label": label,
            "n_params": meta.get("n_params"),
            "train_seconds": meta.get("train_seconds"),
            "load_seconds": round(load_s, 2),
            "ms_per_record": round(infer_s * 1000, 1),
            "raw": {k: v for k, v in raw_scores.items() if k != "_correctness"},
            "hybrid": {k: v for k, v in hybrid_scores.items() if k != "_correctness"},
        }
        results[slug]["_correctness_hybrid"] = hybrid_scores["_correctness"]
        print(f"  {slug}: hybrid F1={hybrid_scores['overall']['f1']}  "
              f"held_out={hybrid_scores['by_slice'].get('held_out', 'n/a')}  "
              f"params={meta.get('n_params', '?')}")
        del nlp

    # Pairwise McNemar, hybrid arm, vs the current production choice
    if "indicner" in results:
        base_correct = results["indicner"]["_correctness_hybrid"]
        for slug in results:
            if slug == "indicner":
                continue
            results[slug]["mcnemar_vs_indicner"] = mcnemar(base_correct, results[slug]["_correctness_hybrid"])

    for r in results.values():
        r.pop("_correctness_hybrid", None)

    out_dir = ROOT / "results" / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(
        json.dumps({"split": "test", "n_records": len(test), "models": results}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nWrote {out_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
