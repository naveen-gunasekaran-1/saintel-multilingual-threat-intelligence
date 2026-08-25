"""Telegram collection for SAINTEL.

Targets are read from a file (default `config/telegram_targets.txt`) rather
than hardcoded, so the source list is reviewable, reproducible, and publishable
alongside the paper. The collector refuses to run with an empty target list.

Two sinks:
  --sink kafka    publish to the raw topic (needs `docker compose up -d`)
  --sink archive  append straight to data/raw/YYYY-MM-DD.jsonl (no infra)

`archive` exists so corpus collection is not blocked on standing up Kafka, and
so a collection run can be replayed into the pipeline later.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from telethon import TelegramClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.logger import get_logger
from src.layer1_ingestion.capture_sink import append_record

logger = get_logger(__name__)

# Load .env explicitly: this module no longer imports config at module scope
# (that pulled in Kafka/Neo4j settings and made the collector unimportable
# without full infrastructure), so nothing else populates the environment.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass

API_ID = int(os.getenv("TELEGRAM_API_ID", "0") or "0")
API_HASH = os.getenv("TELEGRAM_API_HASH", "")
SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "saintel_scraper")
DEFAULT_TARGETS_FILE = ROOT / "config" / "telegram_targets.txt"


def load_targets(path: Path | None = None) -> list[str]:
    """Read channel usernames, one per line. Blank lines and # comments ignored."""
    target_file = path or Path(os.getenv("TELEGRAM_TARGETS_FILE", str(DEFAULT_TARGETS_FILE)))
    if not target_file.exists():
        return []
    targets: list[str] = []
    for line in target_file.read_text(encoding="utf-8").splitlines():
        cleaned = line.split("#", 1)[0].strip().lstrip("@")
        if cleaned.startswith("https://t.me/"):
            cleaned = cleaned[len("https://t.me/"):]
        elif cleaned.startswith("t.me/"):
            cleaned = cleaned[len("t.me/"):]
        if cleaned:
            targets.append(cleaned)
    return targets


def build_message_record(message, channel: str) -> dict:
    """Archive record for one Telegram message.

    Shape mirrors capture_sink.build_record so both collection paths produce a
    single homogeneous archive. `capture_id` is the idempotency key: a rerun
    over the same channel re-emits identical ids and dedupes on load.
    """
    captured_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "id": message.id,
        "platform": "Telegram",
        "channel": channel,
        "text": message.text,
        "timestamp": message.date.isoformat(),
        "provenance": f"t.me/{channel} @ {message.date.isoformat()}",
        "is_noise": False,
    }
    return {
        "capture_id": f"telegram:{channel}:{message.id}",
        "topic": "telegram",
        "partition": 0,
        "offset": message.id,
        "kafka_timestamp": None,
        "captured_at": captured_at,
        "raw": json.dumps(payload, ensure_ascii=False),
        "payload": payload,
    }


async def scrape_channel(client, channel: str, limit: int, emit) -> int:
    """Scrape recent messages from one channel. Returns the count emitted."""
    logger.info("Scanning Telegram channel", extra={"channel": channel, "limit": limit})
    count = 0
    try:
        async for message in client.iter_messages(channel, limit=limit):
            if not message.text:
                continue
            emit(build_message_record(message, channel))
            count += 1
    except Exception as exc:  # pragma: no cover - network-bound path
        logger.exception("Telegram scrape failed", extra={"channel": channel, "error": str(exc)})
    logger.info("Channel complete", extra={"channel": channel, "messages": count})
    return count


def make_emitter(sink: str):
    """Return (emit_fn, finalise_fn) for the chosen sink."""
    if sink == "archive":
        return append_record, lambda: None

    from confluent_kafka import Producer

    from src.core.config import get_settings

    settings = get_settings()
    producer = Producer({"bootstrap.servers": settings.kafka_broker_url})
    topic = settings.kafka_raw_topic

    def emit(record: dict) -> None:
        producer.produce(topic, value=record["raw"].encode("utf-8"))

    return emit, producer.flush


async def main() -> int:
    parser = argparse.ArgumentParser(description="Collect public Telegram messages for SAINTEL.")
    parser.add_argument("--limit", type=int, default=200, help="messages per channel (default 200)")
    parser.add_argument("--targets", type=Path, default=None, help="override targets file")
    parser.add_argument("--sink", choices=("archive", "kafka"), default="archive",
                        help="archive to data/raw (default) or publish to Kafka")
    args = parser.parse_args()

    if not API_ID or not API_HASH:
        logger.error("Missing TELEGRAM_API_ID or TELEGRAM_API_HASH")
        return 1

    targets = load_targets(args.targets)
    if not targets:
        logger.error(
            "No collection targets configured",
            extra={"file": str(args.targets or DEFAULT_TARGETS_FILE)},
        )
        print(
            "\nNo targets configured. Populate config/telegram_targets.txt with the\n"
            "channels your documented discovery process selected, one per line.\n"
            "Refusing to run against placeholder channels.\n",
            file=sys.stderr,
        )
        return 2

    logger.info("Collection starting", extra={"targets": len(targets), "sink": args.sink})
    emit, finalise = make_emitter(args.sink)

    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()
    logger.info("Telegram client started")

    total = 0
    for channel in targets:
        total += await scrape_channel(client, channel, args.limit, emit)

    finalise()
    await client.disconnect()
    logger.info("Collection complete", extra={"channels": len(targets), "messages": total})
    print(f"Collected {total} messages from {len(targets)} channels -> sink={args.sink}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
