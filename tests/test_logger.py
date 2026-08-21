"""Regression tests for structured logging.

The formatter previously read only `record.extra_fields`, a key no call site
in this repo sets, so every extra={...} field on every log line was dropped.
"""

import io
import json

from src.core.logger import JsonFormatter, get_logger


def emit(**kwargs):
    logger = get_logger(f"t{abs(hash(str(kwargs)))}")
    buf = io.StringIO()
    logger.handlers[0].stream = buf
    logger.info("event happened", extra=kwargs)
    return json.loads(buf.getvalue().strip())


def test_flat_extra_fields_are_emitted():
    out = emit(signal_id="abc123", confidence=0.97)
    assert out["signal_id"] == "abc123"
    assert out["confidence"] == 0.97


def test_core_fields_present():
    out = emit(k="v")
    assert out["level"] == "INFO"
    assert out["message"] == "event happened"
    assert "timestamp" in out and "logger" in out


def test_legacy_extra_fields_dict_still_supported():
    out = emit(extra_fields={"legacy": True})
    assert out["legacy"] is True


def test_native_script_not_ascii_escaped():
    # Directive: native UTF-8 must survive; no \u0b.. escaping.
    out = emit(value="சென்னை")
    assert out["value"] == "சென்னை"


def test_non_serialisable_value_does_not_crash():
    out = emit(obj=object())
    assert isinstance(out["obj"], str)


def test_standard_attrs_not_leaked_into_payload():
    out = emit(k="v")
    for noisy in ("args", "msg", "pathname", "lineno", "processName"):
        assert noisy not in out
