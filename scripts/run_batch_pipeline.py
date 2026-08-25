"""Run Layer 1 + Layer 3 over the raw archive as a batch.

The streaming path (Kafka -> entity_extractor -> layer3_entities -> Neo4j)
needs infrastructure up. This runs the same extraction logic directly over
data/raw/*.jsonl so a collected corpus can be analysed without standing up a
broker, and so the run is reproducible from the archive alone.

Output goes to results/batch-<timestamp>/:
    signals.jsonl  one ThreatSignal per input record, native UTF-8
    summary.json   aggregate counts only -- no message text
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.messaging import dumps
from src.layer1_ingestion.normalizer import detect_script, normalize_threat_text


def load_archive(pattern: str) -> list[dict]:
    """Read archive shards, de-duplicating on capture_id."""
    seen: set[str] = set()
    records: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cid = record.get("capture_id")
                if cid in seen:
                    continue
                seen.add(cid)
                records.append(record)
    return records


def tamil_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if 0x0B80 <= ord(ch) <= 0x0BFF) / len(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-extract entities from the raw archive.")
    parser.add_argument("--input", default="data/raw/*.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = all records")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = load_archive(args.input)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"No records matched {args.input}", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or ROOT / "results" / f"batch-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(records)} unique records. Loading models...", flush=True)
    from src.layer3_native_nlp.entity_extractor import EntityExtractor

    extractor = EntityExtractor()
    print("Models ready. Extracting...", flush=True)

    by_type: Counter = Counter()
    by_source: Counter = Counter()
    by_channel: defaultdict = defaultdict(Counter)
    scripts: Counter = Counter()
    values: Counter = Counter()
    with_entities = 0

    signals_path = out_dir / "signals.jsonl"
    with signals_path.open("w", encoding="utf-8") as out:
        for i, record in enumerate(records, 1):
            payload = record.get("payload") or {}
            raw = payload.get("text") or record.get("raw") or ""
            text = normalize_threat_text(raw)
            if not text:
                continue

            channel = payload.get("channel", "unknown")
            signal = extractor.extract(
                text,
                source_type=payload.get("platform", "Telegram"),
                source_id=str(payload.get("id", record.get("capture_id", "unknown"))),
            )

            scripts[detect_script(text)] += 1
            if signal.entities:
                with_entities += 1
            for entity in signal.entities:
                by_type[entity.entity_type] += 1
                by_source[entity.source] += 1
                by_channel[channel][entity.entity_type] += 1
                values[f"{entity.entity_type}:{entity.value}"] += 1

            row = signal.model_dump(mode="json")
            row["_capture_id"] = record.get("capture_id")
            row["_channel"] = channel
            row["_tamil_ratio"] = round(tamil_ratio(text), 3)
            out.write(dumps(row) + "\n")

            if i % 50 == 0:
                print(f"  {i}/{len(records)}", flush=True)

    summary = {
        "generated_at": stamp,
        "records_in": len(records),
        "records_with_entities": with_entities,
        "entity_total": sum(by_type.values()),
        "by_entity_type": dict(by_type.most_common()),
        "by_extractor_source": dict(by_source.most_common()),
        "by_script": dict(scripts.most_common()),
        "by_channel": {k: dict(v) for k, v in by_channel.items()},
        "top_values": dict(values.most_common(40)),
    }
    (out_dir / "summary.json").write_text(dumps(summary) + "\n", encoding="utf-8")

    print(f"\nWrote {signals_path}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"  {with_entities}/{len(records)} records produced at least one entity")
    print(f"  {sum(by_type.values())} entities total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
