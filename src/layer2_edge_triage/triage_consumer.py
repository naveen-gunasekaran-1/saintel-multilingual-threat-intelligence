import json
import sys
from pathlib import Path

import fasttext
from confluent_kafka import Consumer, KafkaException, Producer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import get_settings
from src.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__, level=settings.log_level)

conf = {
    'bootstrap.servers': settings.kafka_broker_url,
    'group.id': 'saintel-edge-triage-group',
    'auto.offset.reset': 'earliest',
}

consumer = Consumer(conf)
consumer.subscribe([settings.kafka_raw_topic])
producer = Producer({'bootstrap.servers': settings.kafka_broker_url})


def delivery_report(err, msg):
    if err is not None:
        logger.error("Kafka delivery failed for triage signal", extra={"error": str(err)})
    else:
        logger.info(
            "High-intent signal emitted to Kafka",
            extra={"topic": msg.topic(), "partition": msg.partition()},
        )


logger.info("Loading Edge AI model", extra={"model_path": settings.model_path})
try:
    model = fasttext.load_model(settings.model_path)
    logger.info("Edge AI model loaded successfully")
except Exception as exc:
    logger.exception("Unable to load fastText model; train triage model first", extra={"error": str(exc)})
    raise SystemExit(1) from exc

logger.info("Triage consumer is running and listening to Kafka", extra={"topic": settings.kafka_raw_topic})

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaException._PARTITION_EOF:
                continue
            logger.error("Kafka error encountered", extra={"error": str(msg.error())})
            break

        try:
            raw_value = msg.value().decode('utf-8')
            record = json.loads(raw_value)
            text = record.get('text', '') or record.get('content', '')
            if not text:
                text = raw_value
            source_id = str(record.get('id', 'unknown'))
            platform = str(record.get('platform', 'telegram'))
        except Exception:
            text = msg.value().decode('utf-8', errors='ignore')
            source_id = 'unknown'
            platform = 'telegram'

        clean_text = text.replace('\n', ' ').strip()
        if not clean_text:
            continue

        predictions = model.predict(clean_text)
        label = predictions[0][0].replace('__label__', '')
        confidence = float(predictions[1][0])

        snippet = clean_text[:50]
        if label == 'signal' and confidence > 0.6:
            payload = {
                'id': source_id,
                'platform': platform,
                'text': clean_text,
                'confidence': round(confidence, 4),
                'label': label,
            }
            producer.produce(
                settings.kafka_signal_topic,
                value=json.dumps(payload).encode('utf-8'),
                callback=delivery_report,
            )
            producer.poll(0)
            logger.info(
                "High-intent signal detected",
                extra={"label": label, "confidence": round(confidence, 4), "snippet": snippet},
            )
        else:
            logger.info(
                "Signal dropped as noise",
                extra={"label": label, "confidence": round(confidence, 4), "snippet": snippet},
            )
except KeyboardInterrupt:
    logger.warning("Stopping consumer")
finally:
    producer.flush()
    consumer.close()