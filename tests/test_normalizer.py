"""Tests for Layer 1 normalization."""

import pytest

from src.layer1_ingestion.normalizer import normalize_threat_text


class TestZeroWidthStripping:
    @pytest.mark.parametrize("ch,name", [
        ("​", "ZWSP"), ("‌", "ZWNJ"),
        ("‍", "ZWJ"), ("﻿", "BOM/ZWNBSP"),
    ])
    def test_each_zero_width_char_removed(self, ch, name):
        assert normalize_threat_text(f"DR{ch}DO") == "DRDO", f"{name} survived"

    def test_evasion_via_interleaved_zero_width(self):
        # Adversarial: zero-width chars inserted to break gazetteer matching.
        assert normalize_threat_text("D​R‌D‍O") == "DRDO"

    def test_zero_width_inside_tamil(self):
        assert normalize_threat_text("சென்​னை") == "சென்னை"


class TestNfkcNormalization:
    def test_fullwidth_folded(self):
        assert normalize_threat_text("ＤＲＤＯ") == "DRDO"

    def test_native_script_preserved(self):
        # Directive: no transliteration, no translation.
        for s in ("சென்னை", "ஆவடி", "கல்பாக்கம்"):
            assert normalize_threat_text(s) == s

    def test_tamil_virama_preserved(self):
        # NFKC must NOT strip the pulli -- that is the corruption Layer 3
        # exists to repair, and introducing it here would be self-inflicted.
        assert "்" in normalize_threat_text("சென்னை")


class TestEdgeCases:
    @pytest.mark.parametrize("value", ["", None])
    def test_falsy_input(self, value):
        assert normalize_threat_text(value) == ""

    def test_whitespace_trimmed(self):
        assert normalize_threat_text("  DRDO  ") == "DRDO"

    def test_only_zero_width_collapses_to_empty(self):
        assert normalize_threat_text("​‌") == ""

    def test_idempotent(self):
        once = normalize_threat_text("நமது இலக்கு Avadi​ மற்றும் சென்னை.")
        assert normalize_threat_text(once) == once
