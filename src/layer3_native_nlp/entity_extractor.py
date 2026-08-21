from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
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

# Canonical name + entity type for critical-national-infrastructure terms.
# See EntityExtractor._match_gazetteer_term for how fragmented/corrupted
# NER output (e.g. "dr", "ava", "செனனை") is resolved back to these values.
CNI_GAZETTEER: dict[str, tuple[str, str]] = {
    # Locations
    "avadi": ("Avadi", "location"),
    "ஆவடி": ("ஆவடி", "location"),
    "chennai": ("Chennai", "location"),
    "சென்னை": ("சென்னை", "location"),
    "kalpakkam": ("Kalpakkam", "location"),
    "கல்பாக்கம்": ("கல்பாக்கம்", "location"),
    "vizag": ("Vizag", "location"),
    "விசாகப்பட்டினம்": ("விசாகப்பட்டினம்", "location"),
    # Organizations
    "drdo": ("DRDO", "organization"),
    "isro": ("ISRO", "organization"),
    "bhel": ("BHEL", "organization"),
}

# South Asian virama/halant marks. The transformer tokenizer's decode step
# can drop these when reassembling subword fragments (e.g. "சென்னை" loses
# its pulli and decodes as "செனனை"), so gazetteer matching normalizes both
# sides down to a virama-free "skeleton" before comparing.
_INDIC_VIRAMAS = "்्্్"  # Tamil, Devanagari, Bengali, Telugu
_INDIC_VIRAMA_TRANSLATION = str.maketrans("", "", _INDIC_VIRAMAS)

_FUZZY_MATCH_MIN_LENGTH = 4
_FUZZY_MATCH_THRESHOLD = 0.85
_MIN_SUBSTRING_MATCH_LENGTH = 3

# Pre-gazetteer sources eligible for fragment cleanup once a canonical
# gazetteer entity for the same term is present elsewhere in the signal.
_FRAGMENT_CLEANUP_SOURCES = {"transformers-ner", "heuristic"}


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

    def _normalize_indic_skeleton(self, text: str) -> str:
        """Strip South Asian virama/halant marks and casefold for comparison.

        The transformer tokenizer's decode step can drop a virama when
        reassembling subword fragments (e.g. "சென்னை" loses its pulli and
        decodes as "செனனை"), so both sides of a gazetteer comparison are
        reduced to this virama-free skeleton before checking equality.
        """
        return text.translate(_INDIC_VIRAMA_TRANSLATION).lower().strip()

    def _match_gazetteer_term(self, value: str) -> tuple[str, str, str] | None:
        """Match a single entity value against the CNI gazetteer.

        Tries, in order: (1) exact match on the raw value, plus a substring
        match in either direction — but substring containment only applies
        when `len(value) >= _MIN_SUBSTRING_MATCH_LENGTH` (3), since a 1-2
        character fragment (e.g. "dr") is a substring of almost anything and
        must instead rely on the raw-text recovery pass in `refine_entities`;
        (2) a virama-insensitive match via `_normalize_indic_skeleton`, which
        catches diacritic corruption like "செனனை" -> "சென்னை" regardless of
        length, since it's an exact match once normalized; (3) a fuzzy match
        via `difflib.SequenceMatcher` for fragments of length >= 4 that are
        merely similar, not identical. Returns
        (canonical_value, entity_type, "exact" | "fuzzy"), or None.
        """
        value_cf = value.casefold()
        allow_substring = len(value) >= _MIN_SUBSTRING_MATCH_LENGTH

        for key, (canonical, entity_type) in CNI_GAZETTEER.items():
            key_cf = key.casefold()
            if value_cf == key_cf:
                return canonical, entity_type, "exact"
            if allow_substring and (value_cf in key_cf or key_cf in value_cf):
                return canonical, entity_type, "exact"

        value_skeleton = self._normalize_indic_skeleton(value)
        if value_skeleton:
            for key, (canonical, entity_type) in CNI_GAZETTEER.items():
                if value_skeleton == self._normalize_indic_skeleton(key):
                    return canonical, entity_type, "exact"

        if len(value) >= _FUZZY_MATCH_MIN_LENGTH:
            for key, (canonical, entity_type) in CNI_GAZETTEER.items():
                key_skeleton = self._normalize_indic_skeleton(key)
                ratio = SequenceMatcher(None, value_skeleton, key_skeleton).ratio()
                if ratio >= _FUZZY_MATCH_THRESHOLD:
                    return canonical, entity_type, "fuzzy"

        return None

    def refine_entities(self, raw_text: str, entities: list[ThreatEntity]) -> list[ThreatEntity]:
        """Reconcile NER output against the CNI gazetteer.

        Every extracted entity is checked directly against the gazetteer
        (exact/substring, then virama-insensitive, then fuzzy), so a
        fragment or diacritic-corrupted value resolves to its canonical
        term in place without ever needing to equal a gazetteer key
        outright. Terms the NER model missed entirely are still recovered
        from `raw_text` so a mention isn't lost for lack of an extracted
        entity to reconcile.
        """
        for entity in entities:
            value = entity.value.strip()
            if not value:
                continue
            match = self._match_gazetteer_term(value)
            if match is None:
                continue
            canonical, entity_type, match_kind = match
            entity.value = canonical
            entity.entity_type = entity_type
            entity.source = f"cni-gazetteer-{match_kind}"

        normalized_text = raw_text.lower()
        present = {(entity.entity_type, entity.value.casefold()) for entity in entities}
        for key, (canonical, entity_type) in CNI_GAZETTEER.items():
            term_key = (entity_type, canonical.casefold())
            if key in normalized_text and term_key not in present:
                entities.append(
                    ThreatEntity(
                        entity_type=entity_type,
                        value=canonical,
                        confidence=0.9,
                        source="cni-gazetteer-exact",
                    )
                )
                present.add(term_key)

        # Collapse duplicate canonical entities produced when multiple NER
        # fragments (e.g. "dr" and "do") both resolve to the same term.
        deduped: dict[tuple[str, str], ThreatEntity] = {}
        for entity in entities:
            dedupe_key = (entity.entity_type, entity.value.casefold())
            current = deduped.get(dedupe_key)
            if current is None or entity.confidence > current.confidence:
                deduped[dedupe_key] = entity
        entities = list(deduped.values())

        # Fragment cleanup: a raw NER or heuristic fragment too short to
        # trigger its own substring match (e.g. "dr", below
        # _MIN_SUBSTRING_MATCH_LENGTH) can still be an orphaned subword of a
        # term the gazetteer recovered by other means (raw-text scan, a
        # sibling fragment's fuzzy match). Once that canonical entity exists,
        # drop the leftover fragment so it doesn't persist as a separate,
        # meaningless graph node. Only pre-gazetteer sources are eligible —
        # canonical gazetteer entities themselves are never touched.
        canonical_values = {
            entity.value.casefold() for entity in entities if entity.source.startswith("cni-gazetteer")
        }
        cleaned: list[ThreatEntity] = []
        for entity in entities:
            if entity.source in _FRAGMENT_CLEANUP_SOURCES:
                fragment_value = entity.value.strip().casefold()
                if fragment_value and any(
                    fragment_value == canonical or fragment_value in canonical
                    for canonical in canonical_values
                ):
                    continue
            cleaned.append(entity)
        return cleaned

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
