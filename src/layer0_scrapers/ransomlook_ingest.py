import json
import logging
from uuid import uuid4
from datetime import datetime, timezone

import requests
from confluent_kafka import Producer

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.logger import get_logger
from src.layer1_ingestion.normalizer import normalize_threat_text

# Setup logger using centralized logger
logger = get_logger(__name__)

# Constants
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "raw_threat_stream"
RANSOMLOOK_API_URL = "https://www.ransomlook.io/api/posts?days=1"

def create_kafka_producer():
    return Producer({'bootstrap.servers': KAFKA_BROKER})

def scrape_ransomlook():
    producer = create_kafka_producer()
    
    try:
        response = requests.get(RANSOMLOOK_API_URL, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle whether the API returns a list or a nested dictionary
        posts = []
        if isinstance(data, list):
            posts = data
        elif isinstance(data, dict):
            posts = data.get("posts", []) or data.get("recent", []) or [data]
        
        for post in posts:
            if not isinstance(post, dict):
                continue
                
            # Extract data mapping
            group_name = post.get("group_name", "Unknown Group")
            title = post.get("title", "")
            description = post.get("description", "")
            
            raw_text = f"Group: {group_name}\nTitle: {title}\nDescription: {description}"
            
            # Normalize text before ingestion
            normalized_text = normalize_threat_text(raw_text)
            
            # Format as RawThreatMessage
            message = {
                "message_id": uuid4().hex,
                "source": "ransomlook_api",
                "raw_text": normalized_text,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Publish to Kafka
            producer.produce(
                KAFKA_TOPIC,
                value=json.dumps(message, ensure_ascii=False).encode('utf-8')
            )
            logger.info(f"Published message for group: {group_name}")
            
        producer.flush()
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch from RansomLook API: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in RansomLook scraper: {e}")

if __name__ == "__main__":
    scrape_ransomlook()
