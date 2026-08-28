import sys
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import BASE_DIR


def train_model():
    # Anchor to the repo root, not the cwd. Relative paths here previously
    # produced a second 763 MiB model tree under src/layer2_edge_triage/
    # whenever this script was run from anywhere but the repo root.
    dataset_dir = BASE_DIR / "data" / "datasets"
    model_dir = BASE_DIR / "data" / "models"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_file = str(dataset_dir / "training_data.txt")
    model_file = str(model_dir / "edge_triage_model.bin")
    
    # Sample training data for edge triage (Noise vs Signal)
    data = [
        "__label__noise Check out this funny meme https://t.co/xyz",
        "__label__noise Buy cheap crypto tokens now 100x gains",
        "__label__noise Good morning everyone have a nice day",
        "__label__noise Good morning everyone, have a blessed day.",
        "__label__noise Hello world test message",
        "__label__noise Click here to win free prizes",
        "__label__noise Looking for a new laptop, any recommendations?",
        "__label__signal Urgent alert regarding regional security deployment",
        "__label__signal Protest scheduled tomorrow near city central hub",
        "__label__signal Unverified reports of infrastructure disruption in sector 4",
        "__label__signal Threat actor network communications intercepted on node B",
        "__label__signal Emergency broadcast restricted access protocol activated",
        "__label__signal Khela hobe tonight at the main square.",
        "__label__signal Aattam ini than arambam. Coordinates received."
    ]
    
    with open(data_file, "w", encoding="utf-8") as f:
        f.write("\n".join(data))
    print(f"Training dataset generated at: {data_file}")
    
    print("Training fastText edge triage model...")
    model = fasttext.train_supervised(input=data_file, epoch=25, lr=1.0, wordNgrams=2)
    
    model.save_model(model_file)
    print(f"Model successfully trained and saved to: {model_file}")

if __name__ == "__main__":
    train_model()