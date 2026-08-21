"""Tests for the Pydantic contracts in src/core/schemas.py."""

import pytest
from pydantic import ValidationError

from src.core.schemas import ThreatEntity, ThreatSignal


class TestThreatEntity:
    def test_valid(self):
        e = ThreatEntity(entity_type="location", value="சென்னை", confidence=0.9)
        assert e.value == "சென்னை" and e.entity_type == "location"

    @pytest.mark.parametrize("bad", ["place", "LOCATION", "", "person"])
    def test_entity_type_is_closed(self, bad):
        # Ontological strictness: the Literal must reject anything off-list.
        with pytest.raises(ValidationError):
            ThreatEntity(entity_type=bad, value="x")

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_confidence_bounds(self, bad):
        with pytest.raises(ValidationError):
            ThreatEntity(entity_type="actor", value="x", confidence=bad)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_value_rejected(self, blank):
        with pytest.raises(ValidationError):
            ThreatEntity(entity_type="actor", value=blank)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ThreatEntity(entity_type="actor", value="x", unexpected=1)

    def test_whitespace_stripped(self):
        assert ThreatEntity(entity_type="actor", value="  APT  ").value == "APT"


class TestThreatSignal:
    def test_defaults(self):
        s = ThreatSignal(raw_text="hello")
        assert s.signal_id and s.intent == "unknown" and s.entities == []
        assert s.created_at.tzinfo is not None, "created_at must be tz-aware"

    def test_raw_text_required(self):
        with pytest.raises(ValidationError):
            ThreatSignal()

    def test_entities_silently_truncated_at_25(self):
        # Documents a LOSSY behaviour: entities beyond 25 vanish with no
        # warning. Relevant to recall if a message is entity-dense.
        s = ThreatSignal(
            raw_text="x",
            entities=[ThreatEntity(entity_type="actor", value=f"a{i}") for i in range(40)],
        )
        assert len(s.entities) == 25

    def test_native_script_survives_json_roundtrip(self):
        # Directive: native UTF-8 must not become \u0b.. escapes.
        s = ThreatSignal(raw_text="சென்னை தாக்குதல்",
                         entities=[ThreatEntity(entity_type="location", value="சென்னை")])
        restored = ThreatSignal.model_validate_json(s.model_dump_json())
        assert restored.raw_text == "சென்னை தாக்குதல்"
        assert restored.entities[0].value == "சென்னை"
