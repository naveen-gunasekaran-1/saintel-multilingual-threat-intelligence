# Save this as scripts/test_native_script.py and run it
import json
from confluent_kafka import Producer
from datetime import datetime, timezone
from uuid import uuid4

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "raw_threat_stream"

# Native Tamil Threat Payload
native_payload = "நமது இலக்கு Avadi மற்றும் சென்னை. DRDO சர்வர்களை முடக்க வேண்டும்."

producer = Producer({'bootstrap.servers': KAFKA_BROKER})

message = {
    "signal_id": f"native_test_{uuid4().hex[:8]}",
    "source_type": "telegram_hacktivist",
    "source_id": "regional_ops",
    "raw_text": native_payload,
    "language": "ta",
    "timestamp": datetime.now(timezone.utc).isoformat()
}

producer.produce(KAFKA_TOPIC, value=json.dumps(message).encode('utf-8'))
producer.flush()

print(f"[+] Native payload published to Kafka: {native_payload}")
print("[+] Check your Streamlit Graph Explorer to see the native Unicode nodes!")