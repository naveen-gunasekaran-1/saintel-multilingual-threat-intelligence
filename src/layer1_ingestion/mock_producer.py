import json
import random
import sys
import time
from pathlib import Path

from confluent_kafka import Producer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import get_settings
from src.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__, level=settings.log_level)

conf = {'bootstrap.servers': settings.kafka_broker_url}
producer = Producer(conf)


def delivery_report(err, msg):
    """Called once for each message produced to indicate delivery result."""
    if err is not None:
        logger.error("Kafka delivery failed", extra={"error": str(err)})
    else:
        logger.info(
            "Kafka message produced",
            extra={"topic": msg.topic(), "partition": msg.partition()},
        )


def simulate_scraping():
    topic_name = settings.kafka_raw_topic
    logger.info("Starting simulated scraper", extra={"topic": topic_name})

    sample_messages = [
        {"platform": "Telegram", "text": "Buy cheap crypto here! Link below.", "is_noise": True},
        {"platform": "DarkWeb", "text": "Khela hobe tonight at the main square.", "is_noise": False},
        {"platform": "Telegram", "text": "Aattam ini than arambam. Coordinates received.", "is_noise": False},
        {"platform": "Telegram", "text": "Good morning everyone, have a blessed day.", "is_noise": True},
        {"platform": "DarkWeb", "text": "Looking for a new laptop, any recommendations?", "is_noise": True},
    ]

    try:
        for i in range(10):
            msg_data = random.choice(sample_messages)
            msg_data['id'] = i
            producer.produce(
                topic_name,
                value=json.dumps(msg_data).encode('utf-8'),
                callback=delivery_report,
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        logger.warning("Producer interrupted by user")
    finally:
        logger.info("Flushing Kafka producer buffer")
        producer.flush()


if __name__ == '__main__':
    simulate_scraping()