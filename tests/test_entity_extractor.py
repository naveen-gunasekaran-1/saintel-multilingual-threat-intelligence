"""Regression coverage for EntityExtractor._ner_entities.

Bug found on the real 433-record Telegram archive (2026-08-26): a NER
pipeline can emit an item whose word strips to empty (a whitespace- or
punctuation-only aggregated span -- observed with XLM-R's SentencePiece
merges). Constructing ThreatEntity(value="") raises inside Pydantic's
min_length=1 validator, and the broad except around the whole extraction loop
caught it -- discarding every OTHER entity already found in that record, not
just the empty one. 93 of 433 real records fell back to heuristic-only for
this reason before the fix.

get_settings() runs at entity_extractor's import time, so the environment has
to be satisfied before the import -- same coupling noted in
tests/test_neo4j_writer.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("NEO4J_PASSWORD", "test-only-not-a-real-password")
os.environ.setdefault("POSTGRES_DSN", "postgresql://test@localhost:5432/test")

from src.layer3_native_nlp.entity_extractor import EntityExtractor


class FakePipeline:
    """Stands in for the HF ner pipeline: returns exactly what it's given."""

    def __init__(self, items):
        self.items = items

    def __call__(self, text):
        return self.items


def _extractor(pipeline_items) -> EntityExtractor:
    ex = object.__new__(EntityExtractor)
    ex.ner_pipeline = FakePipeline(pipeline_items)
    return ex


def test_empty_value_item_is_skipped_not_fatal():
    items = [
        {"entity_group": "LOC", "word": "Chennai", "score": 0.9},
        {"entity_group": "ORG", "word": "  ", "score": 0.8},  # strips to empty
        {"entity_group": "ORG", "word": "DRDO", "score": 0.95},
    ]
    result = _extractor(items)._ner_entities("irrelevant, pipeline is faked")
    values = {e.value for e in result}
    assert values == {"Chennai", "DRDO"}
    assert all(e.source == "transformers-ner" for e in result)


def test_all_empty_values_yields_empty_list_not_a_crash():
    items = [{"entity_group": "LOC", "word": "", "score": 0.5}]
    result = _extractor(items)._ner_entities("x")
    assert result == []


def test_a_genuinely_broken_pipeline_still_falls_back_to_heuristic():
    class ExplodingPipeline:
        def __call__(self, text):
            raise RuntimeError("simulated inference failure")

    ex = object.__new__(EntityExtractor)
    ex.ner_pipeline = ExplodingPipeline()
    result = ex._ner_entities("DRDO servers near Chennai will be attacked")
    assert any(e.source == "heuristic" for e in result)
