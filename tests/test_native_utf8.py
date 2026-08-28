"""Guards the project's native-UTF-8 directive.

Every string leaving this system -- Kafka payloads, STIX bundles, log lines --
must carry native script literally, not as \\u0b.. ASCII escapes.

Six of eight json.dumps call sites originally used the default
ensure_ascii=True, so every Tamil message on Kafka and every exported STIX
bundle was escaped. Functionally lossless, but it violates the stated
directive, inflates payloads ~38%, and renders the flagship CTI artefact
unreadable to a human analyst.
"""

import json
import re
from pathlib import Path

import pytest

from src.core.logger import JsonFormatter, get_logger
from src.core.schemas import ThreatEntity, ThreatSignal
from src.layer5_output.stix_formatter import STIXBundleFormatter

TAMIL = "சென்னை"
ESCAPE_RE = re.compile(r"\\u0[9ab][0-9a-f]{2}")   # Indic block escapes
SRC = Path(__file__).resolve().parents[1] / "src"


def test_stix_bundle_keeps_native_script():
    payload = STIXBundleFormatter().to_json(ThreatSignal(
        raw_text="x", intent="cyberattack", confidence=0.9,
        entities=[ThreatEntity(entity_type="location", value=TAMIL, confidence=0.9)]))
    assert TAMIL in payload
    assert not ESCAPE_RE.search(payload), "STIX export contains ASCII escapes"


def test_stix_bundle_still_parses():
    payload = STIXBundleFormatter().to_json(ThreatSignal(
        raw_text="x", entities=[ThreatEntity(entity_type="location", value=TAMIL)]))
    assert json.loads(payload)["type"] == "bundle"


def test_signal_roundtrips_through_kafka_encoding():
    sig = ThreatSignal(raw_text=f"{TAMIL} தாக்குதல்",
                       entities=[ThreatEntity(entity_type="location", value=TAMIL)])
    wire = json.dumps(sig.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
    assert not ESCAPE_RE.search(wire.decode("utf-8"))
    back = ThreatSignal.model_validate_json(wire.decode("utf-8"))
    assert back.entities[0].value == TAMIL


def test_native_encoding_is_smaller():
    msg = {"text": "நமது இலக்கு DRDO"}
    assert (len(json.dumps(msg, ensure_ascii=False).encode("utf-8"))
            < len(json.dumps(msg).encode("utf-8")))


def test_log_lines_keep_native_script():
    import io
    lg = get_logger("utf8probe")
    buf = io.StringIO(); lg.handlers[0].stream = buf
    lg.info("entity", extra={"value": TAMIL})
    assert TAMIL in buf.getvalue()


@pytest.mark.parametrize("path", sorted(
    p for p in SRC.rglob("*.py") if "json.dumps" in p.read_text(encoding="utf-8")))
def test_every_json_dumps_declares_ensure_ascii(path):
    """Any new json.dumps in src/ must be explicit about ensure_ascii.

    Fails on the default rather than trusting review to catch it.
    """
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "json.dumps(" in line and "ensure_ascii" not in line:
            pytest.fail(f"{path.relative_to(SRC.parent)}:{i} json.dumps without "
                        f"ensure_ascii=False -> {line.strip()}")
