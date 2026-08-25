import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layer3_native_nlp.entity_extractor import EntityExtractor

print("[*] Initializing EntityExtractor...")
extractor = EntityExtractor()

sample_text = "நமது இலக்கு Avadi மற்றும் சென்னை. DRDO சர்வர்களை முடக்க வேண்டும்."
print(f"\n[*] Processing text: {sample_text}\n")

# Run extraction
signal = extractor.extract(sample_text)

print(f"[+] ThreatSignal object returned: {type(signal).__name__}")

# Inspect all fields on the returned object
signal_data = getattr(signal, '__dict__', {})
print("\n--- ThreatSignal Fields ---")
for key, value in signal_data.items():
    if key != 'raw_text':
        print(f"  {key}: {value}")

# Extract entities list
entities = getattr(signal, 'entities', getattr(signal, 'threat_entities', []))

print(f"\n[+] Extracted Entities Count: {len(entities)}")
for e in entities:
    val = getattr(e, 'value', getattr(e, 'name', str(e)))
    e_type = getattr(e, 'entity_type', getattr(e, 'type', 'N/A'))
    e_src = getattr(e, 'source', 'N/A')
    print(f"  - Entity: '{val}' | Type: {e_type} | Source: {e_src}")