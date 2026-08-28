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
from src.core.identity import entity_id as _shared_entity_id
from src.core.logger import get_logger
from src.core.messaging import consumer_config
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

# Guards the one interpolated identifier in the Cypher below.
_ALLOWED_RELATIONS = frozenset(relation for _, _, relation in RELATIONSHIP_RULES)


def _entity_id(entity_type: str, value: str) -> str:
    # Delegates to src.core.identity: this rule and synthesis_agent's copy
    # had to stay byte-identical by hand. Now they cannot drift.
    return _shared_entity_id(entity_type, value)


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

    # Cypher kept at module level so the query plan is cached by the driver
    # and the shape is reviewable without reading the builder logic.
    _SIGNAL_CYPHER = """
        MERGE (s:ThreatSignal {signal_id: $signal_id})
        SET s.source_type = $source_type,
            s.source_id    = $source_id,
            s.raw_text     = $raw_text,
            s.language     = $language,
            s.intent       = $intent,
            s.confidence   = $confidence,
            s.created_at   = $created_at
        """

    _ENTITY_CYPHER = """
        MATCH (s:ThreatSignal {signal_id: $signal_id})
        UNWIND $rows AS row
        MERGE (e:Entity {entity_id: row.entity_id})
        SET e.entity_type = row.entity_type,
            e.value       = row.value,
            e.confidence  = row.confidence,
            e.source      = row.source
        MERGE (s)-[:CONTAINS_ENTITY]->(e)
        """

    # `relation` is interpolated, never parameterised: Cypher does not accept a
    # bound parameter in relationship-type position. It is safe here because the
    # only values it can take are the four literals in RELATIONSHIP_RULES, which
    # is a module constant -- never user input. The assertion below enforces that
    # invariant so a future edit cannot turn this into an injection point.
    _RELATION_CYPHER = """
        UNWIND $rows AS row
        MATCH (source:Entity {{entity_id: row.source_id}})
        MATCH (target:Entity {{entity_id: row.target_id}})
        MERGE (source)-[r:{relation}]->(target)
        SET r.last_seen  = $last_seen,
            r.confidence = row.confidence,
            r.signal_id  = $signal_id
        """

    @classmethod
    def _build_rows(
        cls, signal: ThreatSignal
    ) -> tuple[list[dict], dict[str, list[dict]]]:
        """Flatten a signal into UNWIND-ready rows. Pure -- no I/O, so testable."""
        entity_rows = [
            {
                "entity_id": _entity_id(entity.entity_type, entity.value),
                "entity_type": entity.entity_type,
                "value": entity.value,
                "confidence": entity.confidence,
                "source": entity.source,
            }
            for entity in signal.entities
        ]

        by_type: dict[str, list[ThreatEntity]] = {}
        for entity in signal.entities:
            by_type.setdefault(entity.entity_type, []).append(entity)

        relation_rows: dict[str, list[dict]] = {}
        for source_type, target_type, relation in RELATIONSHIP_RULES:
            rows = [
                {
                    "source_id": _entity_id(source.entity_type, source.value),
                    "target_id": _entity_id(target.entity_type, target.value),
                    "confidence": min(source.confidence, target.confidence),
                }
                for source in by_type.get(source_type, [])
                for target in by_type.get(target_type, [])
            ]
            if rows:
                relation_rows.setdefault(relation, []).extend(rows)

        return entity_rows, relation_rows

    @classmethod
    def _upsert_signal_transaction(cls, session: Session, signal: ThreatSignal) -> None:
        """Write one signal, its entities and their relationships atomically.

        Previously this issued one autocommit `session.run` per entity and one
        per entity *pair*, so a 25-entity signal cost 1 + 25 + up to 625 network
        round trips and could leave a half-written signal behind on failure.
        It is now three-to-six UNWIND batches inside a single explicit
        transaction: the same graph, one atomic unit, bounded round trips.
        """
        entity_rows, relation_rows = cls._build_rows(signal)
        created_at = signal.created_at.isoformat()

        def _unit(tx) -> None:
            tx.run(
                cls._SIGNAL_CYPHER,
                signal_id=signal.signal_id,
                source_type=signal.source_type,
                source_id=signal.source_id,
                raw_text=signal.raw_text,
                language=signal.language,
                intent=signal.intent,
                confidence=signal.confidence,
                created_at=created_at,
            )
            if entity_rows:
                tx.run(cls._ENTITY_CYPHER, signal_id=signal.signal_id, rows=entity_rows)
            for relation, rows in relation_rows.items():
                assert relation in _ALLOWED_RELATIONS, f"unknown relation {relation!r}"
                tx.run(
                    cls._RELATION_CYPHER.format(relation=relation),
                    rows=rows,
                    last_seen=created_at,
                    signal_id=signal.signal_id,
                )

        session.execute_write(_unit)

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
        consumer_config("saintel-graph-persistence-group", broker=settings.kafka_broker_url)
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
