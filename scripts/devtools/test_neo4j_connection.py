from __future__ import annotations

import sys
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import get_settings
from src.core.logger import get_logger


def main() -> int:
    settings = get_settings()
    logger = get_logger(__name__, level=settings.log_level)
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_lifetime=300,
    )

    try:
        driver.verify_connectivity()
        with driver.session(database="neo4j") as session:
            result = session.run("RETURN 1 AS health")
            health = result.single()
        if health is None or health["health"] != 1:
            logger.error("Neo4j health query returned an unexpected result")
            return 1
        logger.info("Neo4j connection test passed", extra={"uri": settings.neo4j_uri})
        return 0
    except Exception as exc:
        logger.exception("Neo4j connection test failed", extra={"uri": settings.neo4j_uri, "error": str(exc)})
        return 1
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())