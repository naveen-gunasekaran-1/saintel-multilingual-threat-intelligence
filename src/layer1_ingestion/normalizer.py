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
