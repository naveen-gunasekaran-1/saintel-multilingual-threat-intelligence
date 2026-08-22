import json
import logging
import time
from uuid import uuid4
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from stem import Signal
from stem.control import Controller
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
TOR_PROXY = "socks5h://127.0.0.1:9050"
TOR_CONTROL_PORT = 9051

# The exact Play DLS Onion Mirror
PLAY_DLS_BASE_URL = "http://ipi4tiumgzjsym6pyuzrfqrtwskokxokqannmd6sa24shvr7x5kxdvqd.onion/index.php?page="

def renew_tor_ip():
    """Sends a NEWNYM signal to the Tor control port to get a new IP address."""
    try:
        with Controller.from_port(port=TOR_CONTROL_PORT) as controller:
            controller.authenticate()  # Relies on CookieAuthentication=1 in torrc
            controller.signal(Signal.NEWNYM)
            logger.info("Tor circuit rotated. New exit node requested.")
            time.sleep(5) # Wait for circuit to establish
    except Exception as e:
        logger.error(f"Failed to rotate Tor circuit: {e}")

def create_kafka_producer():
    return Producer({'bootstrap.servers': KAFKA_BROKER})

def scrape_play_dls(total_pages=5):
    session = requests.Session()
    # Route all traffic through local Tor SOCKS5 proxy
    session.proxies = {
        'http': TOR_PROXY,
        'https': TOR_PROXY
    }

    producer = create_kafka_producer()
    
    for page_num in range(1, total_pages + 1):
        url = f"{PLAY_DLS_BASE_URL}{page_num}"
        
        try:
            logger.info(f"Scraping Play DLS Page {page_num} over Tor...")
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse the specific tags: <th> with class 'News'
            news_items = soup.find_all('th', class_='News')
            
            for item in news_items:
                # Extract the victim's name directly from the <th> text
                victim_name = item.find(text=True)
                victim_name = victim_name.strip() if victim_name else "Unknown"
                
                # Extract the nested <div> tags for location, links, and dates
                divs = [div.get_text(strip=True) for div in item.find_all('div')]
                details = " | ".join(divs)
                
                # Construct the raw text payload
                raw_text = f"Play ransomware claims victim {victim_name}. Details: {details}"
                
                # Strict NFKC Normalization & Zero-width space stripping
                normalized_text = normalize_threat_text(raw_text)
                
                # Package into the standard SAINTEL schema
                message = {
                    "signal_id": f"play_dls_p{page_num}_{uuid4().hex[:8]}",
                    "source_type": "darkweb_dls",
                    "source_id": "play_ransomware",
                    "raw_text": normalized_text,
                    "language": "en",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Publish to Kafka
                producer.produce(
                    KAFKA_TOPIC,
                    value=json.dumps(message, ensure_ascii=False).encode('utf-8')
                )
                logger.info(f"Published darkweb threat for victim: {victim_name}")
                
            producer.flush()
            
            # Execute Tor Circuit Rotation every 4 pages to avoid rate limits
            if page_num % 4 == 0:
                renew_tor_ip()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to scrape Play DLS Page {page_num}: {e}")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error in scraper on Page {page_num}: {e}")

if __name__ == "__main__":
    scrape_play_dls(total_pages=5)