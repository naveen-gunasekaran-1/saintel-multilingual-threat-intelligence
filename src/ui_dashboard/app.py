from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st
from confluent_kafka import Consumer, KafkaException, TopicPartition
from confluent_kafka.admin import AdminClient
from neo4j import Driver, GraphDatabase

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import Settings, get_settings
from src.core.logger import get_logger
from src.core.schemas import ThreatSignal
from src.layer5_output.stix_formatter import STIXBundleFormatter

logger = get_logger(__name__)

try:
    settings: Settings | None = get_settings()
except Exception as exc:
    settings = None
    logger.exception("Dashboard configuration could not be loaded", extra={"error": str(exc)})


@st.cache_resource(show_spinner=False)
def neo4j_driver() -> Driver:
    if settings is None:
        raise RuntimeError("Dashboard configuration is unavailable")
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_lifetime=300,
        max_connection_pool_size=20,
    )


def neo4j_health() -> tuple[bool, str]:
    try:
        driver = neo4j_driver()
        driver.verify_connectivity()
        return True, "connected"
    except Exception as exc:
        logger.warning("Neo4j health check failed", extra={"error": str(exc)})
        return False, "unavailable"


def kafka_health() -> tuple[bool, str]:
    if settings is None:
        return False, "configuration unavailable"
    try:
        admin = AdminClient({"bootstrap.servers": settings.kafka_broker_url})
        admin.list_topics(timeout=2.0)
        return True, "connected"
    except Exception as exc:
        logger.warning("Kafka health check failed", extra={"error": str(exc)})
        return False, "unavailable"


def postgres_health() -> tuple[bool, str]:
    if settings is None:
        return False, "configuration unavailable"
    try:
        import psycopg

        with psycopg.connect(settings.postgres_dsn, connect_timeout=2) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        return True, "connected"
    except Exception as exc:
        logger.warning("PostgreSQL health check failed", extra={"error": str(exc)})
        return False, "unavailable"


def fetch_recent_signals(limit: int = 50) -> list[ThreatSignal]:
    """Read recent validated signals from the Layer 3 Kafka output topic."""
    if settings is None:
        return []

    consumer: Consumer | None = None
    try:
        consumer = Consumer(
            {
                "bootstrap.servers": settings.kafka_broker_url,
                "group.id": f"saintel-dashboard-{int(time.time() * 1000)}",
                "auto.offset.reset": "latest",
                "enable.auto.commit": False,
            }
        )
        metadata = consumer.list_topics(topic=settings.kafka_entity_topic, timeout=2.0)
        topic = metadata.topics.get(settings.kafka_entity_topic)
        if topic is None:
            return []

        assignments: list[TopicPartition] = []
        for partition_id in topic.partitions:
            partition = TopicPartition(settings.kafka_entity_topic, partition_id)
            _, high_watermark = consumer.get_watermark_offsets(partition, timeout=2.0)
            assignments.append(
                TopicPartition(
                    settings.kafka_entity_topic,
                    partition_id,
                    max(0, high_watermark - limit),
                )
            )
        consumer.assign(assignments)

        signals: list[ThreatSignal] = []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(signals) < limit:
            message = consumer.poll(timeout=0.25)
            if message is None:
                continue
            if message.error():
                if message.error().code() == KafkaException._PARTITION_EOF:
                    continue
                logger.warning("Kafka feed message error", extra={"error": str(message.error())})
                continue
            try:
                signals.append(ThreatSignal.model_validate_json(message.value().decode("utf-8")))
            except Exception as exc:
                logger.warning("Skipping malformed threat signal", extra={"error": str(exc)})

        signals.sort(key=lambda signal: signal.created_at, reverse=True)
        return signals[:limit]
    except Exception as exc:
        logger.warning("Threat feed unavailable", extra={"error": str(exc)})
        return []
    finally:
        if consumer is not None:
            consumer.close()


def fetch_graph(limit: int = 150) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if settings is None:
        return [], []
    query = """
    MATCH (source)-[relationship]->(target)
    WHERE (source:Entity OR source:ThreatSignal) AND (target:Entity OR target:ThreatSignal)
    RETURN source, relationship, target
    LIMIT $limit
    """
    try:
        driver = neo4j_driver()
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        with driver.session(database="neo4j") as session:
            for record in session.run(query, limit=limit):
                source = dict(record["source"])
                target = dict(record["target"])
                relationship = record["relationship"]
                source_id = str(source.get("entity_id") or source.get("signal_id"))
                target_id = str(target.get("entity_id") or target.get("signal_id"))
                nodes[source_id] = {
                    "id": source_id,
                    "label": str(source.get("value") or source.get("intent") or source_id),
                    "type": str(source.get("entity_type") or "signal"),
                }
                nodes[target_id] = {
                    "id": target_id,
                    "label": str(target.get("value") or target.get("intent") or target_id),
                    "type": str(target.get("entity_type") or "signal"),
                }
                edges.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "label": str(relationship.type),
                    }
                )
        return list(nodes.values()), edges
    except Exception as exc:
        logger.warning("Graph query failed", extra={"error": str(exc)})
        return [], []


def display_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    if not nodes:
        st.info("No graph relationships are currently available.")
        return
    try:
        from streamlit_agraph import Config, Edge, Node, agraph

        palette = {
            "actor": "#e45756",
            "organization": "#4c78a8",
            "location": "#59a14f",
            "tactic": "#f28e2b",
            "signal": "#8e6bbf",
        }
        graph_nodes = [
            Node(
                id=node["id"],
                label=node["label"],
                size=24 if node["type"] == "actor" else 18,
                color=palette.get(node["type"], "#8d99ae"),
            )
            for node in nodes
        ]
        graph_edges = [
            Edge(source=edge["source"], target=edge["target"], label=edge["label"], type="CURVE_SMOOTH")
            for edge in edges
        ]
        agraph(
            nodes=graph_nodes,
            edges=graph_edges,
            config=Config(width="100%", height=620, directed=True, physics=True),
        )
    except ImportError:
        st.warning("Graph visualization dependency is unavailable; showing graph data instead.")
        st.dataframe(nodes, use_container_width=True, hide_index=True)
        st.dataframe(edges, use_container_width=True, hide_index=True)
    except Exception as exc:
        logger.warning("Graph visualization failed", extra={"error": str(exc)})
        st.warning("The graph could not be rendered. The relationship data remains available below.")
        st.dataframe(edges, use_container_width=True, hide_index=True)


def render_sidebar() -> None:
    st.sidebar.title("SAINTEL Analyst Console")
    st.sidebar.caption("Human-in-the-loop threat intelligence workspace")
    if st.sidebar.button("Refresh data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    st.sidebar.divider()
    st.sidebar.subheader("Pipeline health")
    for name, check in (("Neo4j", neo4j_health), ("Kafka", kafka_health), ("PostgreSQL", postgres_health)):
        healthy, status = check()
        icon = "🟢" if healthy else "🔴"
        st.sidebar.write(f"{icon} {name}: {status}")
    st.sidebar.caption(f"Environment: {settings.app_env if settings else 'configuration unavailable'}")


def render_feed(signals: list[ThreatSignal]) -> None:
    if not signals:
        st.info("No processed threat signals are available from Kafka yet.")
        return
    for signal in signals:
        with st.container(border=True):
            left, middle, right = st.columns([2, 2, 1])
            left.markdown(f"**{signal.intent}**")
            middle.caption(f"{signal.source_type} · {signal.created_at.isoformat()}")
            right.metric("Confidence", f"{signal.confidence:.0%}")
            st.write(signal.raw_text[:500])
            if signal.entities:
                st.caption("Entities: " + ", ".join(f"{entity.entity_type}: {entity.value}" for entity in signal.entities))


def render_export(signals: list[ThreatSignal]) -> None:
    if not signals:
        st.info("No threat signals are available for export.")
        return
    signal_map = {signal.signal_id: signal for signal in signals}
    selected_id = st.selectbox("Threat signal", options=list(signal_map), format_func=lambda value: f"{value} · {signal_map[value].intent}")
    selected = signal_map[selected_id]
    payload = STIXBundleFormatter().to_json(selected)
    if payload is None:
        st.warning("This signal could not be converted into a valid STIX 2.1 bundle.")
        return
    st.download_button(
        "Download STIX 2.1 JSON",
        data=payload,
        file_name=f"saintel-{selected.signal_id}.json",
        mime="application/json",
        use_container_width=True,
    )
    st.json(payload)


def main() -> None:
    st.set_page_config(page_title="SAINTEL Analyst Console", page_icon="S", layout="wide")
    st.title("SAINTEL Analyst Console")
    st.caption("South Asian Infrastructure — Native Threat Entity Labelling")
    if settings is None:
        st.error("Configuration is unavailable. Check the required environment variables in .env.")
        st.stop()

    render_sidebar()
    signals = fetch_recent_signals()
    st.session_state["signals"] = signals
    feed_tab, graph_tab, export_tab = st.tabs(["Live Feed", "Graph Explorer", "STIX Export"])
    with feed_tab:
        st.subheader("Recent processed signals")
        render_feed(signals)
    with graph_tab:
        st.subheader("Threat relationship graph")
        nodes, edges = fetch_graph()
        display_graph(nodes, edges)
    with export_tab:
        st.subheader("CTI export console")
        render_export(signals)


if __name__ == "__main__":
    main()
