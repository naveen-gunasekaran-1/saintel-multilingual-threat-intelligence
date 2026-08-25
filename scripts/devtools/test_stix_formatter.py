from __future__ import annotations

import json
import sys
from pathlib import Path

from stix2 import parse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.logger import get_logger
from src.core.schemas import ThreatEntity, ThreatSignal
from src.layer5_output.stix_formatter import STIXBundleFormatter

logger = get_logger(__name__)


def main() -> int:
    signal = ThreatSignal(
        signal_id="stix-runner-sample",
        source_type="telegram",
        source_id="sample-channel",
        raw_text="APT Example targets Acme Corp in Paris using phishing",
        intent="cyberattack",
        confidence=0.93,
        entities=[
            ThreatEntity(entity_type="actor", value="APT Example", confidence=0.91),
            ThreatEntity(entity_type="organization", value="Acme Corp", confidence=0.89),
            ThreatEntity(entity_type="location", value="Paris", confidence=0.87),
            ThreatEntity(entity_type="tactic", value="phishing", confidence=0.85),
        ],
    )

    formatter = STIXBundleFormatter()
    payload = formatter.to_json(signal)
    if payload is None:
        logger.error("STIX formatter returned no bundle")
        return 1

    parsed = parse(payload, version="2.1")
    document = json.loads(payload)
    object_types = [item["type"] for item in document["objects"]]
    relationship_count = object_types.count("relationship")
    if parsed.type != "bundle" or relationship_count != 4:
        logger.error(
            "Unexpected STIX bundle contents",
            extra={"object_types": object_types, "relationship_count": relationship_count},
        )
        return 1

    malformed = formatter.to_json({"invalid": "payload"})
    if malformed is not None:
        logger.error("Malformed input was not rejected")
        return 1

    logger.info(
        "STIX formatter validation passed",
        extra={"object_count": len(object_types), "relationship_count": relationship_count},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
