import asyncio
import json
import os
import sys
from pathlib import Path

from confluent_kafka import Producer
from telethon import TelegramClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import get_settings
from src.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__, level=settings.log_level)

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "")
SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "saintel_scraper")

conf = {'bootstrap.servers': settings.kafka_broker_url}
producer = Producer(conf)
KAFKA_TOPIC = settings.kafka_raw_topic


def delivery_report(err, msg):
    """Callback for Kafka to confirm message delivery."""
    if err is not None:
        logger.error("Kafka delivery failed", extra={"error": str(err)})
    else:
        logger.info("Telegram message pushed to Kafka", extra={"topic": msg.topic()})


async def scrape_channel(client, channel_username, limit=10):
    """Scrapes recent messages from a target channel and pushes to Kafka."""
    logger.info("Scanning Telegram channel", extra={"channel": channel_username, "limit": limit})

    try:
        async for message in client.iter_messages(channel_username, limit=limit):
            if not message.text:
                continue

            msg_data = {
                "id": message.id,
                "platform": "Telegram",
                "channel": channel_username,
                "text": message.text,
                "timestamp": message.date.isoformat(),
                "is_noise": False,
            }
            logger.info("Scraped Telegram message", extra={"message_id": msg_data["id"], "channel": channel_username})
            producer.produce(
                KAFKA_TOPIC,
                value=json.dumps(msg_data).encode('utf-8'),
                callback=delivery_report,
            )
    except Exception as exc:  # pragma: no cover - network-bound path
        logger.exception("Telegram scrape failed", extra={"channel": channel_username, "error": str(exc)})


async def main():
    if not API_ID or not API_HASH:
        logger.error("Missing TELEGRAM_API_ID or TELEGRAM_API_HASH environment variables")
        raise SystemExit(1)

    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()
    logger.info("Telegram client started successfully")

    target_channels = ['telegram', 'durov']
    for channel in target_channels:
        await scrape_channel(client, channel, limit=5)

    producer.flush()
    logger.info("Telegram scraping complete")


if __name__ == '__main__':
    asyncio.run(main())