from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, KafkaException, Producer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
hf_token = os.getenv("HF_TOKEN")

from src.core.config import get_settings
from src.core.logger import get_logger
from src.core.schemas import ThreatEntity, ThreatSignal
from src.layer1_ingestion.normalizer import normalize_threat_text
from src.layer4_graphrag.neo4j_connector import GraphEntityWriter, Neo4jSessionManager

settings = get_settings()
logger = get_logger(__name__, level=settings.log_level)

_DEFAULT_LABELS = [
    "cyberattack",
    "disinformation",
    "infrastructure disruption",
    "physical threat",
    "social engineering",
    "surveillance",
    "security alert",
]

# Gazetteer constants and matching logic are defined in gazetteer.py, which is
# deliberately free of config/Kafka/Neo4j imports so tests and the benchmark
# harness can use it offline. Re-exported here for backwards compatibility.
from src.layer3_native_nlp.gazetteer import (  # noqa: E402
    CNI_GAZETTEER,
    FRAGMENT_CLEANUP_SOURCES as _FRAGMENT_CLEANUP_SOURCES,
    FUZZY_MATCH_MIN_LENGTH as _FUZZY_MATCH_MIN_LENGTH,
    FUZZY_MATCH_THRESHOLD as _FUZZY_MATCH_THRESHOLD,
    MIN_SUBSTRING_MATCH_LENGTH as _MIN_SUBSTRING_MATCH_LENGTH,
    match_gazetteer_term,
    normalize_indic_skeleton,
    refine_entities,
)



class EntityExtractor:
    def __init__(self, model_name: str | None = None, zero_shot_model_name: str | None = None):
        # Changed to IndicNER for South Asian NLP Optimization
        self.model_name = model_name or "ai4bharat/IndicNER"
        self.zero_shot_model_name = zero_shot_model_name or settings.zero_shot_model_name
        self.ner_pipeline = None
        self.zero_shot_pipeline = None
        self._load_models()

    def _load_models(self) -> None:
        try:
            from transformers import pipeline

            # Load model directly into pipeline as per best practices
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
                token=hf_token,
                device=-1,
            )
            self.zero_shot_pipeline = pipeline(
                "zero-shot-classification",
                model=self.zero_shot_model_name,
                device=-1,
            )
            logger.info(
                "Multilingual NLP models loaded successfully",
                extra={"ner_model": self.model_name, "zero_shot_model": self.zero_shot_model_name},
            )
        except Exception as exc:  # pragma: no cover - runtime dependency issue should be handled gracefully
            logger.error(
                "Transformer models unavailable; fatal exception during initialization. Falling back to heuristic extraction.",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            self.ner_pipeline = None
            self.zero_shot_pipeline = None

    def _heuristic_entities(self, text: str) -> list[ThreatEntity]:
        entities: list[ThreatEntity] = []
        seen: set[str] = set()

        actor_patterns = [
            r"\b(?:hacker|attacker|threat actor|bad actor|cyber actor|group)\b",
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b(?=\s+(?:group|cell|network|operator))",
        ]
        for pattern in actor_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(0).strip()
                key = f"actor:{value.lower()}"
                if value and key not in seen:
                    seen.add(key)
                    entities.append(ThreatEntity(entity_type="actor", value=value, confidence=0.71, source="heuristic"))

        location_patterns = [
            r"\b(?:near|in|at|from|towards)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            r"\b(?:city|town|region|country|district|airport|port|square|hub)\b",
        ]
        for pattern in location_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(0).strip()
                key = f"location:{value.lower()}"
                if value and key not in seen:
                    seen.add(key)
                    entities.append(ThreatEntity(entity_type="location", value=value, confidence=0.68, source="heuristic"))

        tactic_patterns = [
            r"\b(?:phishing|credential harvesting|ransomware|ddos|malware|exfiltration|reconnaissance|lateral movement|spoofing)\b",
            r"\b(?:credential|network|infrastructure|access|disruption)\b",
        ]
        for pattern in tactic_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(0).strip()
                key = f"tactic:{value.lower()}"
                if value and key not in seen:
                    seen.add(key)
                    entities.append(ThreatEntity(entity_type="tactic", value=value, confidence=0.72, source="heuristic"))

        return entities[:15]

    def _ner_entities(self, text: str) -> list[ThreatEntity]:
        if self.ner_pipeline is None:
            return self._heuristic_entities(text)

        try:
            results = self.ner_pipeline(text)
            entities: list[ThreatEntity] = []
            for item in results:
                entity_type = "threat_type"
                if any(tag.lower() in {"loc", "location", "gpe"} for tag in [item.get("entity_group", "")]):
                    entity_type = "location"
                elif any(tag.lower() in {"org", "organization"} for tag in [item.get("entity_group", "")]):
                    entity_type = "organization"
                elif any(tag.lower() in {"per", "person", "actor"} for tag in [item.get("entity_group", "")]):
                    entity_type = "actor"
                elif item.get("entity_group", "").lower() in {"tactic", "technique"}:
                    entity_type = "tactic"

                entities.append(
                    ThreatEntity(
                        entity_type=entity_type,
                        value=item.get("word", "").strip(),
                        confidence=float(item.get("score", 0.5)),
                        source="transformers-ner",
                    )
                )
            return entities[:15]
        except Exception as exc:  # pragma: no cover - network-safe fallback
            logger.warning("NER inference failed, using heuristic fallback", extra={"error": str(exc)})
            return self._heuristic_entities(text)

    def _intent(self, text: str) -> tuple[str, float]:
        if self.zero_shot_pipeline is None:
            return ("unknown", 0.5)

        try:
            result = self.zero_shot_pipeline(text, candidate_labels=_DEFAULT_LABELS)
            intent = result["labels"][0]
            confidence = float(result["scores"][0])
            return intent, confidence
        except Exception as exc:  # pragma: no cover - network-safe fallback
            logger.warning("Zero-shot classification failed", extra={"error": str(exc)})
            return ("unknown", 0.5)

    # Gazetteer logic lives in gazetteer.py so it can be imported without
    # config/Kafka/Neo4j. These wrappers keep the existing method API.
    def _normalize_indic_skeleton(self, text: str) -> str:
        return normalize_indic_skeleton(text)

    def _match_gazetteer_term(self, value: str) -> tuple[str, str, str] | None:
        return match_gazetteer_term(value)

    def refine_entities(self, raw_text: str, entities: list[ThreatEntity]) -> list[ThreatEntity]:
        return refine_entities(raw_text, entities)

    def extract(self, text: str, *, source_type: str = "telegram", source_id: str = "unknown") -> ThreatSignal:
        entities = self._ner_entities(text)
        intent, confidence = self._intent(text)

        if not entities:
            entities = self._heuristic_entities(text)

        entities = self.refine_entities(text, entities)

        payload = ThreatSignal(
            source_type=source_type,
            source_id=source_id,
            raw_text=text,
            language="und",
            intent=intent,
            confidence=max(0.0, min(confidence, 1.0)),
            entities=entities,
        )
        return payload


def _consume_signal_topic() -> None:
    consumer = Consumer(
        {
            'bootstrap.servers': settings.kafka_broker_url,
            'group.id': 'saintel-raw-threat-group',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
        }
    )
    consumer.subscribe([settings.kafka_raw_topic])
    producer = Producer({'bootstrap.servers': settings.kafka_broker_url})

    logger.info("Entity extraction consumer started", extra={"topic": settings.kafka_raw_topic})
    extractor = EntityExtractor()
    graph_writer = GraphEntityWriter(
        Neo4jSessionManager(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    )

    try:
        while True:
            message = consumer.poll(timeout=1.0)
            if message is None:
                continue
            if message.error():
                if message.error().code() == KafkaException._PARTITION_EOF:
                    continue
                logger.error("Kafka consumer error", extra={"error": str(message.error())})
                break

            try:
                payload = json.loads(message.value().decode('utf-8'))
                raw_text = payload.get('text') or payload.get('content') or payload.get('raw_text') or ''
                if not raw_text:
                    continue

                # 1. Normalize threat text to preserve Unicode integrity natively
                normalized_text = normalize_threat_text(raw_text)

                source_type = payload.get('source_type', payload.get('platform', 'telegram'))
                source_id = str(payload.get('source_id', payload.get('id', 'unknown')))
                
                # 2. Extract Entities & 3. Refine Entities
                signal = extractor.extract(normalized_text, source_type=source_type, source_id=source_id)

                # 4. Write Native nodes to Neo4j
                graph_writer.persist_signal(signal)

                # 5. Commit offset and log
                producer.produce(
                    settings.kafka_entity_topic,
                    value=json.dumps(signal.model_dump(mode='json')).encode('utf-8'),
                )
                
                native_count = sum(1 for e in signal.entities if e.source.startswith("transformers"))
                logger.info(
                    "Threat entity payload emitted to graph",
                    extra={"topic": settings.kafka_entity_topic, "signal_id": signal.signal_id, "entities_found": len(signal.entities), "native_entities": native_count},
                )
                
                consumer.commit(message=message, asynchronous=False)
            except Exception as exc:  # pragma: no cover - resilience path
                logger.exception("Entity extraction failed for message", extra={"error": str(exc)})
    except KeyboardInterrupt:
        logger.info("Entity extractor interrupted")
    finally:
        graph_writer.session_manager.close()
        producer.flush()
        consumer.close()


if __name__ == '__main__':
    _consume_signal_topic()
