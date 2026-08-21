"""Draw a stratified sample from the raw archive into an annotation worksheet.

Bridges data/raw/*.jsonl (real captured traffic) and tests/data/eval_set.*.json
(labelled gold). Emits records in the eval-set schema with labels left blank
for a human to fill, following tests/data/annotation_guidelines.md.

Sampling is deterministic (seeded LCG) so a sample can be regenerated exactly.

Redaction happens HERE, not in the archive: the archive stays verbatim, and
what leaves it for annotation -- and eventually for a public repository -- is
scrubbed of emails, phone numbers, URLs and long digit runs.

  python scripts/sample_for_annotation.py --n 300 --out tests/data/to_annotate.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCHIVE = ROOT / "data" / "raw"

# Unicode blocks -> script label used by the eval schema.
_BLOCKS = {
    "tamil": (0x0B80, 0x0BFF),
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "telugu": (0x0C00, 0x0C7F),
}
_REDACTIONS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"https?://\S+"), "[URL]"),
    (re.compile(r"\b(?:\+?\d[\d\s-]{8,}\d)\b"), "[PHONE]"),
    (re.compile(r"\b\d{9,}\b"), "[ID]"),
]


def detect_script(text: str) -> str:
    indic = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in _BLOCKS.values()))
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if indic and latin:
        return "mixed"
    if indic:
        for name, (lo, hi) in _BLOCKS.items():
            if any(lo <= ord(ch) <= hi for ch in text):
                return f"native_{name}"
    return "latin"


def redact(text: str) -> tuple[str, list[str]]:
    applied = []
    for pattern, token in _REDACTIONS:
        text, n = pattern.subn(token, text)
        if n:
            applied.append(f"{token}x{n}")
    return text, applied


def extract_text(rec: dict) -> str:
    p = rec.get("payload")
    if isinstance(p, dict):
        for k in ("text", "content", "raw_text", "message"):
            v = p.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return (rec.get("raw") or "").strip()


def load_archive(base: Path) -> list[dict]:
    rows = []
    if not base.exists():
        return rows
    for shard in sorted(base.glob("*.jsonl")):
        for line in shard.open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def lcg(seed: int):
    s = seed
    while True:
        s = (1103515245 * s + 12345) % (2 ** 31)
        yield s


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify by detected script so native-script text is not swamped by
    Latin, which would otherwise dominate any real Telegram corpus."""
    buckets: dict[str, list[dict]] = {}
    seen_text: set[str] = set()
    for r in rows:
        t = extract_text(r)
        if not t or t in seen_text:
            continue
        seen_text.add(t)
        buckets.setdefault(detect_script(t), []).append(r)

    rnd = lcg(seed)
    out, keys = [], sorted(buckets)
    per = max(1, n // max(len(keys), 1))
    for k in keys:
        pool = sorted(buckets[k], key=lambda r: r["capture_id"])
        if len(pool) > per:
            picked, idxs = [], set()
            while len(idxs) < per:
                idxs.add(next(rnd) % len(pool))
            picked = [pool[i] for i in sorted(idxs)]
        else:
            picked = pool
        out.extend(picked)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--out", default=str(ROOT / "tests" / "data" / "to_annotate.json"))
    ap.add_argument("--no-redact", action="store_true",
                    help="keep verbatim text (NEVER use for anything published)")
    args = ap.parse_args()

    rows = load_archive(Path(args.archive))
    if not rows:
        print(f"archive empty: {args.archive}\n"
              "Run the capture sink and let real traffic accumulate first:\n"
              "  python src/layer1_ingestion/capture_sink.py", file=sys.stderr)
        return 1

    picked = sample(rows, args.n, args.seed)
    out = []
    for i, r in enumerate(picked, 1):
        text = extract_text(r)
        text, applied = (text, []) if args.no_redact else redact(text)
        out.append({
            "id": f"har-{i:04d}",
            "text": text,
            "script": detect_script(text),
            "language": "",                 # annotator fills
            "source": "harvested",
            "provenance": r.get("capture_id", ""),
            "captured_at": r.get("captured_at", ""),
            "redactions": applied,
            "gazetteer_slice": "",          # annotator fills
            "triage_label": "",             # annotator fills
            "intent_label": "",             # annotator fills
            "entities": [],                 # annotator fills
            "tamil_verified": False,
            "notes": "",
        })

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    from collections import Counter
    print(f"archive {len(rows)} records -> sampled {len(out)} -> {args.out}")
    print("  by script:", dict(Counter(r["script"] for r in out)))
    print(f"  redacted : {sum(1 for r in out if r['redactions'])}")
    print("\nNext: label per tests/data/annotation_guidelines.md, have a second")
    print("annotator do 20% for Cohen's kappa, then merge into eval_set.*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
