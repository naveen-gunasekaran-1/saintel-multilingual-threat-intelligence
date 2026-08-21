"""SAINTEL benchmark harness.

Answers one question the project has never been able to answer: does the
transformer contribute anything the gazetteer does not?

Runs offline. No Kafka, no Neo4j. The transformer arm needs IndicNER locally
(cached or downloadable); every other arm is stdlib + fastText.

  python scripts/evaluate_pipeline.py --split dev --arms fast
  python scripts/evaluate_pipeline.py --split dev --arms all

MEASUREMENT CAVEAT: entity scoring is VALUE-level -- (entity_type, casefolded
value) set comparison -- not span-level. The pipeline's ThreatEntity carries no
character offsets, so true span P/R/F1 is impossible against it today. The gold
data already stores start/end, so this upgrades for free once the pipeline
preserves offsets (Phase 4). Do not describe these numbers as span-level.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.schemas import ThreatEntity
from src.layer3_native_nlp.gazetteer import refine_entities

DATA_DIR = ROOT / "tests" / "data"
BOOTSTRAP_N = 2000


# ------------------------------------------------------------------ scoring --

def key_set(pairs):
    return {(t, v.casefold()) for t, v in pairs}


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def score_entities(per_record: list[tuple[set, set]]) -> dict:
    tp = fp = fn = 0
    for gold, pred in per_record:
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    return prf(tp, fp, fn)


def bootstrap_f1(per_record, n=BOOTSTRAP_N):
    """Deterministic bootstrap CI. LCG instead of `random` so results are
    reproducible without seeding global state."""
    if not per_record:
        return None
    m, state, N = len(per_record), 12345, len(per_record)
    f1s = []
    for _ in range(n):
        tp = fp = fn = 0
        for _ in range(m):
            state = (1103515245 * state + 12345) % (2 ** 31)
            gold, pred = per_record[state % N]
            tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
        s = prf(tp, fp, fn)
        f1s.append(s["f1"])
    f1s.sort()
    return {"f1_ci95_low": round(f1s[int(0.025 * n)], 4),
            "f1_ci95_high": round(f1s[int(0.975 * n)], 4)}


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Exact-ish McNemar on paired per-record correctness."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    if b + c == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "significant_p05": False,
                "note": "arms are identical on every record"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": round(chi2, 4),
            "significant_p05": chi2 > 3.841,
            "note": "chi2 > 3.841 => p < 0.05 (df=1)"}


# --------------------------------------------------------------------- arms --

def arm_gazetteer(rec, _ctx):
    return [(e.entity_type, e.value) for e in refine_entities(rec["text"], [])]


def arm_heuristic(rec, ctx):
    return [(e.entity_type, e.value) for e in ctx["heuristic"](rec["text"])]


def arm_transformer(rec, ctx):
    return [(t, v) for t, v in ctx["ner"](rec["text"])]


def arm_hybrid(rec, ctx):
    raw = [ThreatEntity(entity_type=t, value=v, confidence=c, source="transformers-ner")
           for t, v, c in ctx["ner_conf"](rec["text"])]
    return [(e.entity_type, e.value) for e in refine_entities(rec["text"], raw)]


def make_ablation(**flags):
    def arm(rec, ctx):
        raw = [ThreatEntity(entity_type=t, value=v, confidence=c, source="transformers-ner")
               for t, v, c in ctx["ner_conf"](rec["text"])]
        return [(e.entity_type, e.value) for e in refine_entities(rec["text"], raw, **flags)]
    return arm


ENTITY_ARMS = {
    "gazetteer":  (arm_gazetteer,  False),
    "heuristic":  (arm_heuristic,  False),
    "transformer": (arm_transformer, True),
    "hybrid":     (arm_hybrid,     True),
    "abl_no_skeleton":  (make_ablation(enable_skeleton=False), True),
    "abl_no_fuzzy":     (make_ablation(enable_fuzzy=False), True),
    "abl_no_fragclean": (make_ablation(enable_fragment_cleanup=False), True),
    "abl_no_recovery":  (make_ablation(enable_raw_text_recovery=False), True),
}
FAST_ARMS = ["gazetteer", "heuristic"]


# ------------------------------------------------------------------ context --

def build_context(need_ner: bool) -> dict:
    ctx = {}
    from src.layer3_native_nlp.entity_extractor import EntityExtractor
    shell = object.__new__(EntityExtractor)
    ctx["heuristic"] = shell._heuristic_entities

    if not need_ner:
        return ctx

    import os, warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers import pipeline
    ner = pipeline("ner", model="ai4bharat/IndicNER",
                   aggregation_strategy="simple", device=-1)

    def _map(item):
        g = (item.get("entity_group") or "").lower()
        if g in {"loc", "location", "gpe"}:
            return "location"
        if g in {"org", "organization"}:
            return "organization"
        if g in {"per", "person", "actor"}:
            return "actor"
        if g in {"tactic", "technique"}:
            return "tactic"
        return "threat_type"

    cache: dict[str, list] = {}

    def run(text):
        if text not in cache:
            cache[text] = [(_map(i), (i.get("word") or "").strip(), float(i.get("score", 0.5)))
                           for i in ner(text) if (i.get("word") or "").strip()]
        return cache[text]

    ctx["ner_conf"] = run
    ctx["ner"] = lambda t: [(a, b) for a, b, _ in run(t)]
    return ctx


# ------------------------------------------------------------------- triage --

def eval_triage(records, model_path):
    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None
    model = fasttext.load_model(model_path)

    rows = []
    for r in records:
        lab, conf = model.predict(r["text"].replace("\n", " ").strip())
        label = lab[0].replace("__label__", "")
        signal_conf = float(conf[0]) if label == "signal" else 1.0 - float(conf[0])
        rows.append({"id": r["id"], "gold": r["triage_label"],
                     "signal_conf": round(signal_conf, 4), "notes": r.get("notes", "")})

    sweep = []
    for i in range(1, 20):
        th = i / 20
        tp = sum(1 for x in rows if x["signal_conf"] > th and x["gold"] == "signal")
        fp = sum(1 for x in rows if x["signal_conf"] > th and x["gold"] == "noise")
        fn = sum(1 for x in rows if x["signal_conf"] <= th and x["gold"] == "signal")
        tn = sum(1 for x in rows if x["signal_conf"] <= th and x["gold"] == "noise")
        s = prf(tp, fp, fn)
        s["threshold"] = round(th, 2)
        s["fpr"] = round(fp / (fp + tn), 4) if (fp + tn) else 0.0
        sweep.append(s)

    at06 = next(s for s in sweep if s["threshold"] == 0.6)
    best = max(sweep, key=lambda s: s["f1"])
    worst_fp = sorted([x for x in rows if x["gold"] == "noise"],
                      key=lambda x: -x["signal_conf"])[:8]
    return {"at_production_threshold_0.6": at06, "best_threshold_by_f1": best,
            "sweep": sweep, "worst_false_positives": worst_fp}


# --------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--arms", default="fast", help="'fast', 'all', or comma list")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--skip-triage", action="store_true")
    args = ap.parse_args()

    path = DATA_DIR / f"eval_set.{args.split}.json"
    if not path.exists():
        print(f"missing {path}; run scripts/build_eval_seed.py first", file=sys.stderr)
        return 1
    records = json.loads(path.read_text(encoding="utf-8"))

    if args.arms == "fast":
        names = FAST_ARMS
    elif args.arms == "all":
        names = list(ENTITY_ARMS)
    else:
        names = [n.strip() for n in args.arms.split(",") if n.strip()]
    unknown = [n for n in names if n not in ENTITY_ARMS]
    if unknown:
        print(f"unknown arms: {unknown}", file=sys.stderr)
        return 1

    ctx = build_context(need_ner=any(ENTITY_ARMS[n][1] for n in names))

    gold_by_id = {r["id"]: key_set([(e["type"], e["value"]) for e in r["entities"]])
                  for r in records}
    results, correctness, errors = {}, {}, []

    for name in names:
        fn_arm, _ = ENTITY_ARMS[name]
        overall, by_slice, by_script, flags = [], defaultdict(list), defaultdict(list), []
        for r in records:
            gold = gold_by_id[r["id"]]
            pred = key_set(fn_arm(r, ctx))
            overall.append((gold, pred))
            by_slice[r["gazetteer_slice"]].append((gold, pred))
            by_script[r["script"]].append((gold, pred))
            flags.append(gold == pred)
            if gold != pred:
                errors.append({"arm": name, "id": r["id"], "text": r["text"],
                               "slice": r["gazetteer_slice"], "script": r["script"],
                               "missed": sorted(f"{t}:{v}" for t, v in gold - pred),
                               "spurious": sorted(f"{t}:{v}" for t, v in pred - gold)})
        correctness[name] = flags
        entry = {"overall": score_entities(overall)}
        entry["overall"].update(bootstrap_f1(overall) or {})
        entry["by_gazetteer_slice"] = {k: score_entities(v) for k, v in sorted(by_slice.items())}
        entry["by_script"] = {k: score_entities(v) for k, v in sorted(by_script.items())}
        entry["exact_record_accuracy"] = round(sum(flags) / len(flags), 4)
        results[name] = entry
        o = entry["overall"]
        print(f"  {name:18} P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}")

    significance = {}
    for a, b in [("gazetteer", "hybrid"), ("gazetteer", "transformer"),
                 ("transformer", "hybrid"), ("hybrid", "abl_no_fuzzy"),
                 ("hybrid", "abl_no_skeleton")]:
        if a in correctness and b in correctness:
            significance[f"{a}_vs_{b}"] = mcnemar(correctness[a], correctness[b])

    triage = None
    if not args.skip_triage:
        from src.core.config import get_settings
        try:
            triage = eval_triage(records, get_settings().model_path)
            t = triage["at_production_threshold_0.6"]
            print(f"  {'triage@0.6':18} P={t['precision']:.3f} R={t['recall']:.3f} "
                  f"F1={t['f1']:.3f} FPR={t['fpr']:.3f}")
        except Exception as exc:
            triage = {"error": str(exc)}

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        sha = "unknown"

    payload = {
        "split": args.split, "n_records": len(records), "arms": names,
        "git_sha": sha, "python": platform.python_version(),
        "scoring": "value-level (entity_type, casefolded value); NOT span-level",
        "entity_arms": results, "significance_mcnemar": significance, "triage": triage,
    }

    out = Path(args.out) / f"{args.split}-{sha}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    with (out / "errors.jsonl").open("w", encoding="utf-8") as fh:
        for e in errors:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(render_report(payload), encoding="utf-8")
    print(f"\nwrote {out}/metrics.json, report.md, errors.jsonl ({len(errors)} errors)")
    return 0


def render_report(p: dict) -> str:
    L = [f"# SAINTEL benchmark — `{p['split']}` split", "",
         f"{p['n_records']} records · git `{p['git_sha']}` · {p['scoring']}", "",
         "## Entity extraction", "",
         "| arm | P | R | F1 | F1 95% CI | exact-record acc |", "|---|---|---|---|---|---|"]
    for name, e in p["entity_arms"].items():
        o = e["overall"]
        ci = f"{o.get('f1_ci95_low','?')}–{o.get('f1_ci95_high','?')}"
        L.append(f"| `{name}` | {o['precision']:.3f} | {o['recall']:.3f} | **{o['f1']:.3f}** "
                 f"| {ci} | {e['exact_record_accuracy']:.3f} |")
    L += ["", "## By gazetteer slice — the actual experiment", "",
          "`held_out` contains CNI entities deliberately absent from the gazetteer, "
          "so gazetteer recall is 0 there by construction.", "",
          "| arm | in_gazetteer F1 | held_out F1 | none F1 |", "|---|---|---|---|"]
    for name, e in p["entity_arms"].items():
        s = e["by_gazetteer_slice"]
        g = lambda k: f"{s[k]['f1']:.3f}" if k in s else "—"
        L.append(f"| `{name}` | {g('in_gazetteer')} | {g('held_out')} | {g('none')} |")
    L += ["", "## By script", "", "| arm | latin | romanized | native_tamil | mixed |",
          "|---|---|---|---|---|"]
    for name, e in p["entity_arms"].items():
        s = e["by_script"]
        g = lambda k: f"{s[k]['f1']:.3f}" if k in s else "—"
        L.append(f"| `{name}` | {g('latin')} | {g('romanized')} | {g('native_tamil')} | {g('mixed')} |")
    if p.get("significance_mcnemar"):
        L += ["", "## Significance (McNemar, paired)", "",
              "| comparison | b | c | chi2 | p<0.05 |", "|---|---|---|---|---|"]
        for k, v in p["significance_mcnemar"].items():
            L.append(f"| {k} | {v['b']} | {v['c']} | {v['chi2']} | "
                     f"{'**yes**' if v['significant_p05'] else 'no'} |")
    t = p.get("triage")
    if t and "error" not in t:
        a, b = t["at_production_threshold_0.6"], t["best_threshold_by_f1"]
        L += ["", "## Triage classifier", "",
              "| threshold | P | R | F1 | FPR |", "|---|---|---|---|---|",
              f"| 0.6 (production) | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | {a['fpr']:.3f} |",
              f"| {b['threshold']} (best F1) | {b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} | {b['fpr']:.3f} |",
              "", "### Highest-confidence false positives", "",
              "| id | signal conf | note |", "|---|---|---|"]
        for r in t["worst_false_positives"]:
            L.append(f"| {r['id']} | {r['signal_conf']:.3f} | {r['notes'][:60]} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
