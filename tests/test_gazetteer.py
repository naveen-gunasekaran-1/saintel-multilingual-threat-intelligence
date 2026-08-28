"""First real test suite in this repository.

Covers src/layer3_native_nlp/gazetteer.py, which the Phase 2 benchmark
depends on. If these assertions are wrong, every benchmark number is wrong.

Deliberately imports only `gazetteer`, never `entity_extractor` -- the latter
calls get_settings() at module scope and pulls in Kafka and Neo4j.
"""

import pytest

from src.core.schemas import ThreatEntity
from src.layer3_native_nlp.gazetteer import (
    CNI_GAZETTEER,
    MIN_SUBSTRING_MATCH_LENGTH,
    match_gazetteer_term,
    normalize_indic_skeleton,
    refine_entities,
)


def ent(value, entity_type="actor", confidence=0.4, source="transformers-ner"):
    return ThreatEntity(entity_type=entity_type, value=value, confidence=confidence, source=source)


def pairs(entities):
    return {(e.entity_type, e.value) for e in entities}


# ---------------------------------------------------------------- skeleton --

class TestNormalizeIndicSkeleton:
    @pytest.mark.parametrize("virama,script", [
        ("்", "Tamil"), ("्", "Devanagari"),
        ("্", "Bengali"), ("్", "Telugu"),
    ])
    def test_strips_each_virama(self, virama, script):
        assert virama not in normalize_indic_skeleton(f"a{virama}b"), f"{script} virama survived"

    def test_chennai_corruption_collapses_to_same_skeleton(self):
        # The exact bug: IndicNER's decode drops the pulli from சென்னை.
        assert normalize_indic_skeleton("சென்னை") == normalize_indic_skeleton("செனனை")

    def test_casefolds_and_strips(self):
        assert normalize_indic_skeleton("  DRDO  ") == "drdo"

    def test_empty_input(self):
        assert normalize_indic_skeleton("") == ""


# ----------------------------------------------------------------- matching --

class TestMatchGazetteerTerm:
    @pytest.mark.parametrize("value,canonical,etype", [
        ("DRDO", "DRDO", "organization"),
        ("drdo", "DRDO", "organization"),
        ("Chennai", "Chennai", "location"),
        ("சென்னை", "சென்னை", "location"),
        ("ஆவடி", "ஆவடி", "location"),
    ])
    def test_exact_match(self, value, canonical, etype):
        assert match_gazetteer_term(value) == (canonical, etype, "exact")

    def test_substring_at_minimum_length_matches(self):
        # 'ava' is 3 chars == the floor, so substring matching applies.
        assert len("ava") == MIN_SUBSTRING_MATCH_LENGTH
        assert match_gazetteer_term("ava") == ("Avadi", "location", "exact")

    @pytest.mark.parametrize("fragment", ["dr", "ch", "is", "a", ""])
    def test_short_fragments_rejected(self, fragment):
        # Below the floor a fragment must NOT match, or 'dr' would resolve to
        # DRDO on two characters. Recovery for these happens via raw-text scan.
        assert len(fragment) < MIN_SUBSTRING_MATCH_LENGTH
        assert match_gazetteer_term(fragment) is None

    def test_virama_corrupted_tamil_recovered(self):
        # The headline repair case.
        assert match_gazetteer_term("செனனை") == ("சென்னை", "location", "exact")

    def test_fuzzy_match_on_misspelling(self):
        result = match_gazetteer_term("Kalpakam")  # one 'k' dropped
        assert result == ("Kalpakkam", "location", "fuzzy")

    @pytest.mark.parametrize("value", ["cricket", "wedding", "laptop", "zzzzzz"])
    def test_unrelated_terms_do_not_match(self, value):
        assert match_gazetteer_term(value) is None

    def test_every_gazetteer_key_matches_itself(self):
        # Guards against a future entry that cannot resolve to itself.
        for key, (canonical, etype) in CNI_GAZETTEER.items():
            assert match_gazetteer_term(key) == (canonical, etype, "exact"), key

    def test_ablation_flags_disable_tiers(self):
        assert match_gazetteer_term("ava", enable_substring=False) is None
        assert match_gazetteer_term("செனனை", enable_skeleton=False, enable_fuzzy=False) is None
        assert match_gazetteer_term("Kalpakam", enable_fuzzy=False) is None


# ------------------------------------------------------------------ refine --

class TestRefineEntities:
    def test_fragments_resolve_and_canonicals_recovered(self):
        out = refine_entities(
            "DRDO facility near Avadi, Chennai reported activity",
            [ent("dr"), ent("ava"), ent("செனனை")],
        )
        assert pairs(out) == {
            ("organization", "DRDO"), ("location", "Avadi"),
            ("location", "Chennai"), ("location", "சென்னை"),
        }
        assert not any(e.value in {"dr", "ava"} for e in out), "fragment leaked into output"

    def test_does_not_mutate_input(self):
        # Critical for benchmarking: arms must not contaminate each other.
        original = [ent("dr")]
        refine_entities("DRDO facility", original)
        assert original[0].value == "dr"
        assert original[0].source == "transformers-ner"

    def test_unrelated_entities_pass_through_untouched(self):
        out = refine_entities("Unrelated report about xy", [ent("xy"), ent("zz", source="heuristic")])
        assert pairs(out) == {("actor", "xy"), ("actor", "zz")}

    def test_raw_text_recovery_finds_missed_term(self):
        # Model returned nothing; the term is still in the text.
        out = refine_entities("Report mentions ISRO facility", [])
        assert ("organization", "ISRO") in pairs(out)

    def test_raw_text_recovery_can_be_disabled(self):
        assert refine_entities("Report mentions ISRO", [], enable_raw_text_recovery=False) == []

    def test_deduplicates_keeping_highest_confidence(self):
        out = refine_entities(
            "DRDO",
            [ent("DRDO", "organization", 0.30), ent("DRDO", "organization", 0.95)],
            enable_raw_text_recovery=False,
        )
        assert len(out) == 1
        assert out[0].confidence == 0.95

    def test_fragment_cleanup_covers_heuristic_source(self):
        out = refine_entities("DRDO facility", [ent("dr", source="heuristic")])
        assert not any(e.value == "dr" for e in out)

    def test_fragment_cleanup_can_be_disabled(self):
        out = refine_entities("DRDO facility", [ent("dr")], enable_fragment_cleanup=False)
        assert ("actor", "dr") in pairs(out)

    def test_canonical_gazetteer_entities_never_dropped(self):
        out = refine_entities("ISRO facility", [ent("ISRO", "organization", 0.95, "cni-gazetteer-exact")])
        assert pairs(out) == {("organization", "ISRO")}

    def test_empty_inputs(self):
        assert refine_entities("", []) == []


class TestKnownLimitations:
    """Documents current behaviour that is arguably wrong.

    These assert what the code does today so a future fix trips the test
    deliberately, rather than silently changing benchmark numbers.
    """

    def test_three_char_substring_is_promiscuous(self):
        # 'isr' resolves to ISRO purely by containment. Acceptable for a
        # curated 11-term list; would misfire badly on a larger gazetteer.
        assert match_gazetteer_term("isr") == ("ISRO", "organization", "exact")

    def test_benign_mention_still_yields_entity(self):
        # Entity extraction is correct here; whether the message is a THREAT
        # is the triage layer's job. Recorded to keep the boundary explicit.
        out = refine_entities("Travelling to Chennai for a wedding", [])
        assert ("location", "Chennai") in pairs(out)
