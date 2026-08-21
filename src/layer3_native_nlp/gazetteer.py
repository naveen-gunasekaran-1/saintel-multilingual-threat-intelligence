"""CNI gazetteer matching and NER-output repair.

Deliberately dependency-free apart from `src.core.schemas`: this module imports
no config, no Kafka, and no Neo4j, so it can be imported on a machine with no
`.env` and no running infrastructure. That is what makes offline benchmarking
possible -- `entity_extractor` calls `get_settings()` at module scope and
transitively imports the Neo4j connector, so it cannot be imported by a test
or benchmark harness.

Background: IndicNER's subword tokenizer fragments mixed-script gazetteer terms
("DRDO" -> "dr" + "##do", "Avadi" -> "ava" + "##di") and its decode step can
drop South Asian viramas ("சென்னை" -> "செனனை"). The functions here resolve
those degraded spans back to canonical CNI entities.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from src.core.schemas import ThreatEntity

__all__ = [
    "CNI_GAZETTEER",
    "FUZZY_MATCH_MIN_LENGTH",
    "FUZZY_MATCH_THRESHOLD",
    "MIN_SUBSTRING_MATCH_LENGTH",
    "FRAGMENT_CLEANUP_SOURCES",
    "normalize_indic_skeleton",
    "match_gazetteer_term",
    "refine_entities",
]

# Canonical name + entity type for critical-national-infrastructure terms.
CNI_GAZETTEER: dict[str, tuple[str, str]] = {
    # Locations
    "avadi": ("Avadi", "location"),
    "ஆவடி": ("ஆவடி", "location"),
    "chennai": ("Chennai", "location"),
    "சென்னை": ("சென்னை", "location"),
    "kalpakkam": ("Kalpakkam", "location"),
    "கல்பாக்கம்": ("கல்பாக்கம்", "location"),
    "vizag": ("Vizag", "location"),
    "விசாகப்பட்டினம்": ("விசாகப்பட்டினம்", "location"),
    # Organizations
    "drdo": ("DRDO", "organization"),
    "isro": ("ISRO", "organization"),
    "bhel": ("BHEL", "organization"),
}

# South Asian virama/halant marks: Tamil, Devanagari, Bengali, Telugu.
INDIC_VIRAMAS = "்्্్"
_INDIC_VIRAMA_TRANSLATION = str.maketrans("", "", INDIC_VIRAMAS)

# NOTE: these three values were chosen by hand during implementation and have
# never been tuned against data. Phase 2 sweeps them on the dev split only.
FUZZY_MATCH_MIN_LENGTH = 4
FUZZY_MATCH_THRESHOLD = 0.85
MIN_SUBSTRING_MATCH_LENGTH = 3

# Pre-gazetteer sources eligible for fragment cleanup.
FRAGMENT_CLEANUP_SOURCES = frozenset({"transformers-ner", "heuristic"})


def normalize_indic_skeleton(text: str) -> str:
    """Strip South Asian virama/halant marks and casefold for comparison."""
    return text.translate(_INDIC_VIRAMA_TRANSLATION).lower().strip()


def match_gazetteer_term(
    value: str,
    *,
    enable_substring: bool = True,
    enable_skeleton: bool = True,
    enable_fuzzy: bool = True,
    fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD,
    fuzzy_min_length: int = FUZZY_MATCH_MIN_LENGTH,
    min_substring_length: int = MIN_SUBSTRING_MATCH_LENGTH,
) -> tuple[str, str, str] | None:
    """Resolve one entity value to a canonical CNI term.

    Three tiers, in order: exact (plus substring containment, gated on
    `min_substring_length` so 1-2 char fragments cannot match promiscuously),
    then virama-insensitive skeleton equality, then fuzzy ratio.

    Returns (canonical_value, entity_type, "exact" | "fuzzy"), or None.
    The keyword flags exist so the benchmark can ablate each tier.
    """
    value_cf = value.casefold()
    allow_substring = enable_substring and len(value) >= min_substring_length

    for key, (canonical, entity_type) in CNI_GAZETTEER.items():
        key_cf = key.casefold()
        if value_cf == key_cf:
            return canonical, entity_type, "exact"
        if allow_substring and (value_cf in key_cf or key_cf in value_cf):
            return canonical, entity_type, "exact"

    value_skeleton = normalize_indic_skeleton(value)

    if enable_skeleton and value_skeleton:
        for key, (canonical, entity_type) in CNI_GAZETTEER.items():
            if value_skeleton == normalize_indic_skeleton(key):
                return canonical, entity_type, "exact"

    if enable_fuzzy and len(value) >= fuzzy_min_length:
        for key, (canonical, entity_type) in CNI_GAZETTEER.items():
            ratio = SequenceMatcher(None, value_skeleton, normalize_indic_skeleton(key)).ratio()
            if ratio >= fuzzy_threshold:
                return canonical, entity_type, "fuzzy"

    return None


def refine_entities(
    raw_text: str,
    entities: list[ThreatEntity],
    *,
    enable_substring: bool = True,
    enable_skeleton: bool = True,
    enable_fuzzy: bool = True,
    enable_raw_text_recovery: bool = True,
    enable_fragment_cleanup: bool = True,
    fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD,
    fuzzy_min_length: int = FUZZY_MATCH_MIN_LENGTH,
    min_substring_length: int = MIN_SUBSTRING_MATCH_LENGTH,
) -> list[ThreatEntity]:
    """Reconcile NER output against the CNI gazetteer.

    Four stages, each independently ablatable for the benchmark:
      1. per-entity resolution to canonical terms
      2. raw-text recovery of terms the model missed entirely
      3. deduplication on (entity_type, value)
      4. cleanup of leftover fragments subsumed by a canonical entity

    Does NOT mutate the input entities -- it works on copies. The previous
    in-place version would have let one benchmark arm contaminate the next
    when several arms run over the same records.
    """
    working = [entity.model_copy(deep=True) for entity in entities]

    # 1. Per-entity gazetteer resolution.
    for entity in working:
        value = entity.value.strip()
        if not value:
            continue
        match = match_gazetteer_term(
            value,
            enable_substring=enable_substring,
            enable_skeleton=enable_skeleton,
            enable_fuzzy=enable_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
            fuzzy_min_length=fuzzy_min_length,
            min_substring_length=min_substring_length,
        )
        if match is None:
            continue
        canonical, entity_type, match_kind = match
        entity.value = canonical
        entity.entity_type = entity_type
        entity.source = f"cni-gazetteer-{match_kind}"

    # 2. Recover gazetteer terms present in the text but missed by the model.
    if enable_raw_text_recovery:
        normalized_text = raw_text.lower()
        present = {(e.entity_type, e.value.casefold()) for e in working}
        for key, (canonical, entity_type) in CNI_GAZETTEER.items():
            term_key = (entity_type, canonical.casefold())
            if key in normalized_text and term_key not in present:
                working.append(
                    ThreatEntity(
                        entity_type=entity_type,
                        value=canonical,
                        confidence=0.9,
                        source="cni-gazetteer-exact",
                    )
                )
                present.add(term_key)

    # 3. Collapse duplicates, keeping the highest-confidence instance.
    deduped: dict[tuple[str, str], ThreatEntity] = {}
    for entity in working:
        dedupe_key = (entity.entity_type, entity.value.casefold())
        current = deduped.get(dedupe_key)
        if current is None or entity.confidence > current.confidence:
            deduped[dedupe_key] = entity
    working = list(deduped.values())

    # 4. Drop raw fragments subsumed by a canonical gazetteer entity.
    if not enable_fragment_cleanup:
        return working

    canonical_values = {
        e.value.casefold() for e in working if e.source.startswith("cni-gazetteer")
    }
    cleaned: list[ThreatEntity] = []
    for entity in working:
        if entity.source in FRAGMENT_CLEANUP_SOURCES:
            fragment_value = entity.value.strip().casefold()
            if fragment_value and any(
                fragment_value == canonical or fragment_value in canonical
                for canonical in canonical_values
            ):
                continue
        cleaned.append(entity)
    return cleaned
