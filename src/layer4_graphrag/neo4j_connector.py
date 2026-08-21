from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, TypeVar

from confluent_kafka import Consumer, KafkaException
from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import Neo4jError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import get_settings
from src.core.logger import get_logger
from src.core.schemas import ThreatEntity, ThreatSignal

settings = get_settings()
logger = get_logger(__name__, level=settings.log_level)

T = TypeVar("T")
MAX_RETRIES = 4
MAX_BUFFERED_SIGNALS = 1000

RELATIONSHIP_RULES: tuple[tuple[str, str, str], ...] = (
    ("actor", "organization", "TARGETS"),
    ("actor", "location", "OPERATES_IN"),
    ("actor", "tactic", "USES"),
    ("organization", "location", "LOCATED_IN"),
)


def _entity_id(entity_type: str, value: str) -> str:
    return f"{entity_type}:{value.strip().casefold()}"


class Neo4jSessionManager:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Driver | None = None

    def _connect(self) -> None:
        if self._driver is not None:
            return
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    max_connection_lifetime=300,
                    max_connection_pool_size=50,
                )
                driver.verify_connectivity()
                self._driver = driver
                logger.info("Neo4j connection established", extra={"uri": self.uri})
                return
            except Exception as exc:
                logger.warning(
                    "Neo4j connection failed; retrying",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(2 ** (attempt - 1))

    def get_session(self) -> Session:
        self._connect()
        if self._driver is None:
            raise RuntimeError("Neo4j driver is unavailable")
        return self._driver.session(database=self.database)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


class GraphEntityWriter:
    def __init__(
        self,
        session_manager: Neo4jSessionManager,
        max_buffered_signals: int = MAX_BUFFERED_SIGNALS,
    ) -> None:
        self.session_manager = session_manager
        self.max_buffered_signals = max_buffered_signals
        self._pending_signals: Deque[ThreatSignal] = deque(maxlen=max_buffered_signals)

    def _with_retry(self, operation: Callable[[Session], T]) -> T:
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with self.session_manager.get_session() as session:
                    return operation(session)
            except (Neo4jError, OSError, RuntimeError) as exc:
                last_error = exc
                logger.warning(
                    "Neo4j operation failed; retrying",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                self.session_manager.close()
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError("Neo4j operation failed after retries") from last_error

    @staticmethod
    def _upsert_signal_transaction(session: Session, signal: ThreatSignal) -> None:
        session.run(
            """
            MERGE (s:ThreatSignal {signal_id: $signal_id})
            SET s.source_type = $source_type,
                s.source_id = $source_id,
                s.raw_text = $raw_text,
                s.language = $language,
                s.intent = $intent,
                s.confidence = $confidence,
                s.created_at = $created_at
            """,
            signal_id=signal.signal_id,
            source_type=signal.source_type,
            source_id=signal.source_id,
            raw_text=signal.raw_text,
            language=signal.language,
            intent=signal.intent,
            confidence=signal.confidence,
            created_at=signal.created_at.isoformat(),
        )

        for entity in signal.entities:
            session.run(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.entity_type = $entity_type,
                    e.value = $value,
                    e.confidence = $confidence,
                    e.source = $source
                WITH e
                MATCH (s:ThreatSignal {signal_id: $signal_id})
                MERGE (s)-[:CONTAINS_ENTITY]->(e)
                """,
                entity_id=_entity_id(entity.entity_type, entity.value),
                entity_type=entity.entity_type,
                value=entity.value,
                confidence=entity.confidence,
                source=entity.source,
                signal_id=signal.signal_id,
            )

        entities_by_type: dict[str, list[ThreatEntity]] = {}
        for entity in signal.entities:
            entities_by_type.setdefault(entity.entity_type, []).append(entity)

        for source_type, target_type, relation in RELATIONSHIP_RULES:
            for source in entities_by_type.get(source_type, []):
                for target in entities_by_type.get(target_type, []):
                    session.run(
                        f"""
                        MATCH (source:Entity {{entity_id: $source_id}})
                        MATCH (target:Entity {{entity_id: $target_id}})
                        MERGE (source)-[r:{relation}]->(target)
                        SET r.last_seen = $last_seen,
                            r.confidence = $confidence,
                            r.signal_id = $signal_id
                        """,
                        source_id=_entity_id(source.entity_type, source.value),
                        target_id=_entity_id(target.entity_type, target.value),
                        last_seen=signal.created_at.isoformat(),
                        confidence=min(source.confidence, target.confidence),
                        signal_id=signal.signal_id,
                    )

    def persist_signal(self, signal: ThreatSignal) -> bool:
        validated_signal = ThreatSignal.model_validate(signal)
        try:
            self._with_retry(lambda session: self._upsert_signal_transaction(session, validated_signal))
            self.flush_buffer()
            logger.info(
                "Threat signal persisted to Neo4j",
                extra={"signal_id": validated_signal.signal_id, "entity_count": len(validated_signal.entities)},
            )
            return True
        except Exception as exc:
            if len(self._pending_signals) == self.max_buffered_signals:
                logger.error("Neo4j signal buffer full; oldest signal will be discarded")
            self._pending_signals.append(validated_signal)
            logger.exception(
                "Neo4j unavailable; threat signal buffered",
                extra={"signal_id": validated_signal.signal_id, "buffer_size": len(self._pending_signals), "error": str(exc)},
            )
            return False

    def flush_buffer(self) -> int:
        flushed = 0
        while self._pending_signals:
            pending = self._pending_signals[0]
            try:
                self._with_retry(lambda session: self._upsert_signal_transaction(session, pending))
            except Exception as exc:
                logger.warning(
                    "Neo4j buffer flush paused",
                    extra={"buffer_size": len(self._pending_signals), "error": str(exc)},
                )
                break
            self._pending_signals.popleft()
            flushed += 1
        return flushed

    @property
    def buffered_count(self) -> int:
        return len(self._pending_signals)


def _consume_entities() -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_broker_url,
            "group.id": "saintel-graph-persistence-group",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([settings.kafka_entity_topic])
    writer = GraphEntityWriter(
        Neo4jSessionManager(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    )

    logger.info("Neo4j persistence consumer started", extra={"topic": settings.kafka_entity_topic})
    try:
        while True:
            message = consumer.poll(timeout=1.0)
            if message is None:
                writer.flush_buffer()
                continue
            if message.error():
                if message.error().code() == KafkaException._PARTITION_EOF:
                    continue
                logger.error("Kafka consumer error", extra={"error": str(message.error())})
                continue

            try:
                signal = ThreatSignal.model_validate_json(message.value().decode("utf-8"))
                if writer.persist_signal(signal):
                    consumer.commit(message=message, asynchronous=False)
                else:
                    logger.warning("Kafka offset retained for buffered signal", extra={"signal_id": signal.signal_id})
            except Exception as exc:
                logger.exception("Invalid or failed entity payload", extra={"error": str(exc)})
                consumer.commit(message=message, asynchronous=False)
    except KeyboardInterrupt:
        logger.info("Neo4j persistence consumer interrupted")
    finally:
        writer.session_manager.close()
        consumer.close()


if __name__ == "__main__":
    _consume_entities()
