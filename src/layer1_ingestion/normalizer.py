"""Layer 1 text normalization.

Runs before any tokenizer sees the text. Zero-width characters are a direct
contributor to the subword fragmentation that Layer 3 then has to repair, so
removing them here is cheaper than fixing the damage downstream.
"""

import re
import unicodedata

# U+200B zero-width space, U+200C ZWNJ, U+200D ZWJ, U+FEFF BOM/ZWNBSP.
#
# Tradeoff, deliberately taken: ZWNJ and ZWJ are orthographically meaningful in
# Devanagari and Bengali, where they control conjunct and half-form rendering.
# Stripping them can alter how a string displays. For entity matching it is the
# right call -- it collapses variant encodings of the same word into one form,
# and adversarial insertion of zero-width characters is a known evasion trick.
# If rendering fidelity ever matters, preserve the original in raw_text and
# normalize only the matching key.
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")


def normalize_threat_text(text: str) -> str:
    """NFKC-normalize and strip zero-width characters.

    Native scripts are preserved as-is; nothing is transliterated or
    translated (project directive: zero pre-translation).
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH.sub("", normalized)
    return normalized.strip()


# Unicode blocks -> the script label used by the eval-set schema.
# Single source of truth: scripts/sample_for_annotation.py and the Layer 4
# triage node both import from here, so the label vocabulary cannot drift
# between the annotation worksheet and the runtime pipeline.
SCRIPT_BLOCKS: dict[str, tuple[int, int]] = {
    "tamil": (0x0B80, 0x0BFF),
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "telugu": (0x0C00, 0x0C7F),
}


def detect_script(text: str) -> str:
    """Classify text as mixed / native_<script> / latin.

    Deliberately coarse: it answers "which tokenizer failure mode applies",
    not "which language is this". `mixed` is the interesting case -- it is
    exactly where subword fragmentation of Latin acronyms inside Indic prose
    occurs.
    """
    indic = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in SCRIPT_BLOCKS.values()))
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if indic and latin:
        return "mixed"
    if indic:
        for name, (lo, hi) in SCRIPT_BLOCKS.items():
            if any(lo <= ord(ch) <= hi for ch in text):
                return f"native_{name}"
    return "latin"
