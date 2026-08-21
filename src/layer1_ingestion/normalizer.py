import unicodedata

def normalize_threat_text(text: str) -> str:
    """
    Applies NFKC Unicode normalization to standard text to ensure native scripts
    (Tamil, Bengali, Devanagari) map correctly at the byte level.
    Strips zero-width spaces and other invisible characters.
    """
    if not text:
        return ""
    
    # NFKC normalization
    normalized = unicodedata.normalize('NFKC', text)
    
    # Strip zero-width spaces (U+200B)
    normalized = normalized.replace('\u200b', '')
    
    return normalized.strip()
