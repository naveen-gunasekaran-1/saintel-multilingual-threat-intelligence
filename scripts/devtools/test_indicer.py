import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
hf_token = os.getenv("HF_TOKEN")

print("Loading IndicNER model... (This may take a minute if downloading)")

try:
    # Load the model directly and pass the token
    ner_pipeline = pipeline("ner", model="ai4bharat/IndicNER", aggregation_strategy="simple", token=hf_token)
    
    # Our native test string
    native_text = "நமது இலக்கு Avadi மற்றும் சென்னை. DRDO சர்வர்களை முடக்க வேண்டும்."
    
    print(f"\nProcessing Text: {native_text}")
    
    # Run the extraction
    entities = ner_pipeline(native_text)
    
    print("\n--- Extraction Results ---")
    if not entities:
        print("No entities found by the model.")
    for entity in entities:
        print(f"Entity: {entity['word']} | Label: {entity['entity_group']} | Score: {entity['score']:.4f}")
        
except Exception as e:
    print(f"\n[!] Model Loading Failed: {e}")