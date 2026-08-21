import fasttext
import os

def train_model():
    # Define paths relative to project root
    dataset_dir = "data/datasets"
    model_dir = "data/models"
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    data_file = os.path.join(dataset_dir, "training_data.txt")
    model_file = os.path.join(model_dir, "edge_triage_model.bin")
    
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