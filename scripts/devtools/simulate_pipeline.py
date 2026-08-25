import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import get_settings
from src.core.logger import get_logger
from src.layer1_ingestion.normalizer import normalize_threat_text
from src.layer3_native_nlp.entity_extractor import EntityExtractor
from src.layer4_graphrag.neo4j_connector import GraphEntityWriter, Neo4jSessionManager

settings = get_settings()
logger = get_logger(__name__, level="INFO")

def run_simulation():
    # 1. Define a sample threat message with South Asian code-switching
    # Example Tanglish: "Play Ransomware admin said avanoda network access kedachudchu for the new target in Mumbai."
    # With a zero-width space inserted (\u200b) to test normalizer
    raw_message = "Play Ransomware admin said avanoda \u200bnetwork access kedachudchu for the new target in Mumbai."
    logger.info(f"Original raw message: {repr(raw_message)}")

    # 2. Test Normalization
    normalized_msg = normalize_threat_text(raw_message)
    logger.info(f"Normalized message: {repr(normalized_msg)}")

    # 3. Test NLP Extraction
    logger.info("Initializing IndicNER Entity Extractor...")
    extractor = EntityExtractor()
    
    logger.info("Extracting entities...")
    signal = extractor.extract(normalized_msg, source_type="test_simulation", source_id="sim_123")
    
    logger.info("Extracted Signal:")
    print(json.dumps(signal.model_dump(mode='json'), indent=2))

    # 4. Test Graph Mapping (Neo4j)
    logger.info("Connecting to Neo4j...")
    session_manager = Neo4jSessionManager(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    graph_writer = GraphEntityWriter(session_manager)
    
    try:
        logger.info("Persisting signal to Graph...")
        graph_writer.persist_signal(signal)
        
        # Verify the nodes were written
        with session_manager.get_session() as session:
            result = session.run(
                "MATCH (s:ThreatSignal {signal_id: $signal_id})-[:CONTAINS_ENTITY]->(e:Entity) "
                "RETURN s.signal_id AS signal_id, e.value AS entity_value, e.entity_type AS entity_type",
                signal_id=signal.signal_id
            )
            logger.info("Verification Query Results:")
            for record in result:
                print(f" - Signal [{record['signal_id']}] -> Entity [{record['entity_value']}] (Type: {record['entity_type']})")
                
    except Exception as e:
        logger.error(f"Failed during Neo4j interaction: {e}")
    finally:
        session_manager.close()
        logger.info("Simulation completed.")

if __name__ == "__main__":
    run_simulation()
