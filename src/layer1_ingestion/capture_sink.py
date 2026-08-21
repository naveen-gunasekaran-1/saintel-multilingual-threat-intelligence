"""Raw message archive: append every ingested message to disk, verbatim.

Nothing in this pipeline has ever persisted raw input. Scrapers produce
straight to Kafka and exit, so messages age out with broker retention and are
gone. That makes it impossible to re-run an experiment on the same input, to
build a corpus, or to hand a reviewer the data behind a number.

This consumer is the corpus source. It is deliberately dumb: it does not parse,
normalise, classify, or interpret. It records what arrived and where it came
from, so annotation and evaluation can draw from real traffic later.

SENSITIVITY: this archive will contain whatever the scrapers ingested,
including victim PII from ransomware leak sites. data/raw/ is gitignored and
dockerignored. Redaction happens at annotation time (see
scripts/sample_for_annotation.py), never by silently dropping data here --
an archive you have quietly altered is not an archive.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.logger import get_logger

ARCHIVE_DIR = ROOT / "data" / "raw"
logger = get_logger(__name__)


def build_record(
    raw_bytes: bytes,
    *,
    topic: str,
    partition: int,
    offset: int,
    kafka_timestamp_ms: int | None = None,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Build one archive record. Pure -- no Kafka, no filesystem.

    `raw` always holds the exact bytes as UTF-8 (replacing undecodable bytes),
    so the original is recoverable even when the payload is not valid JSON.
    `payload` is the parsed form when parseable, otherwise None.
    """
    text = raw_bytes.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            payload = None
    except (ValueError, TypeError):
        payload = None

    return {
        "capture_id": f"{topic}:{partition}:{offset}",
        "topic": topic,
        "partition": partition,
        "offset": offset,
        "kafka_timestamp": (
            datetime.fromtimestamp(kafka_timestamp_ms / 1000, tz=timezone.utc).isoformat()
            if kafka_timestamp_ms else None
        ),
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
        "raw": text,
        "payload": payload,
    }


def archive_path(captured_at: str, base: Path = ARCHIVE_DIR) -> Path:
    """Daily shard, so files stay manageable and are easy to sample by date."""
    return base / f"{captured_at[:10]}.jsonl"


def load_seen_ids(base: Path = ARCHIVE_DIR) -> set[str]:
    """Capture ids already on disk, so restarts do not duplicate records.

    Kafka offsets are unique per (topic, partition), which makes capture_id a
    natural idempotency key across redeliveries.
    """
    seen: set[str] = set()
    if not base.exists():
        return seen
    for shard in sorted(base.glob("*.jsonl")):
        for line in shard.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["capture_id"])
            except (ValueError, KeyError):
                continue
    return seen


def append_record(record: dict[str, Any], base: Path = ARCHIVE_DIR) -> Path:
    """Append one record, flushed and fsynced before returning.

    The caller commits the Kafka offset only after this returns, so a crash
    costs a duplicate (deduped by capture_id) rather than a lost message.
    """
    base.mkdir(parents=True, exist_ok=True)
    path = archive_path(record["captured_at"], base)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        import os

        os.fsync(fh.fileno())
    return path


def main() -> int:
    from confluent_kafka import Consumer, KafkaException

    from src.core.config import get_settings

    settings = get_settings()
    consumer = Consumer({
        "bootstrap.servers": settings.kafka_broker_url,
        "group.id": "saintel-raw-capture-group",   # independent of other consumers
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([settings.kafka_raw_topic])

    seen = load_seen_ids()
    logger.info("Raw capture sink started",
                extra={"topic": settings.kafka_raw_topic,
                       "archive": str(ARCHIVE_DIR),
                       "already_archived": len(seen)})
    written = skipped = 0
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaException._PARTITION_EOF:
                    continue
                logger.error("Kafka consumer error", extra={"error": str(msg.error())})
                continue

            ts = None
            try:
                kind, value = msg.timestamp()
                ts = value if kind else None
            except Exception:
                pass

            record = build_record(msg.value() or b"", topic=msg.topic(),
                                  partition=msg.partition(), offset=msg.offset(),
                                  kafka_timestamp_ms=ts)
            if record["capture_id"] in seen:
                skipped += 1
                consumer.commit(message=msg, asynchronous=False)
                continue

            append_record(record)
            seen.add(record["capture_id"])
            written += 1
            consumer.commit(message=msg, asynchronous=False)

            if written % 100 == 0:
                logger.info("Archive progress",
                            extra={"written": written, "duplicates_skipped": skipped})
    except KeyboardInterrupt:
        logger.info("Capture sink stopped",
                    extra={"written": written, "duplicates_skipped": skipped})
    finally:
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
