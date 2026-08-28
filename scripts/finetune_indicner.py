"""Fine-tune ai4bharat/IndicNER for CNI entity extraction.

Plain PyTorch: no accelerate/datasets/Trainer, so the pinned environment in
requirements.txt stays reproducible.

Reuses IndicNER's existing BIO label set (B/I-LOC, B/I-ORG, B/I-PER, O) so the
classification head starts from pretrained weights rather than random init.

Trains ONLY on in-gazetteer entities. Evaluation on held-out CNI entities is a
genuine generalisation test -- see scripts/build_finetune_corpus.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

# ai4bharat/IndicNER is a GATED repo: weights need an authenticated token even
# though the config/tokenizer resolve from cache without one.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
HF_TOKEN = os.getenv("HF_TOKEN") or None
DATA = ROOT / "data" / "finetune"
OUT = ROOT / "data" / "models" / "indicner-cni-ft"

BASE = "ai4bharat/IndicNER"
LABELS = ["B-LOC", "B-ORG", "B-PER", "I-LOC", "I-ORG", "I-PER", "O"]
L2I = {l: i for i, l in enumerate(LABELS)}
MAX_LEN, BATCH, EPOCHS, LR, SEED = 128, 16, 3, 3e-5, 20260821


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NerDataset(Dataset):
    def __init__(self, path: Path, tok):
        self.rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_offsets_mapping=True)
        labels = []
        for j, (s, e) in enumerate(enc["offset_mapping"]):
            if enc["attention_mask"][j] == 0 or s == e:
                labels.append(-100)          # padding / special tokens
                continue
            tag = "O"
            for ent in r["entities"]:
                if s >= ent["start"] and e <= ent["end"]:
                    # B- on the token that starts the span, I- thereafter.
                    tag = ("B-" if s == ent["start"] else "I-") + ent["label"]
                    break
            labels.append(L2I[tag])
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(labels),
        }


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_sum = n = correct = total = 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out = model(**b)
        loss_sum += out.loss.item(); n += 1
        pred = out.logits.argmax(-1)
        mask = b["labels"] != -100
        # Score only entity tokens: 'O' dominates and would hide everything.
        ent = mask & (b["labels"] != L2I["O"])
        correct += (pred[ent] == b["labels"][ent]).sum().item()
        total += ent.sum().item()
    return loss_sum / max(n, 1), correct / max(total, 1)


def main() -> int:
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(LABELS),
        id2label={i: l for l, i in L2I.items()}, label2id=L2I,
        token=HF_TOKEN,
    ).to(device)

    tr = DataLoader(NerDataset(DATA / "train.jsonl", tok), batch_size=BATCH, shuffle=True)
    dv = DataLoader(NerDataset(DATA / "dev.jsonl", tok), batch_size=BATCH)
    print(f"train batches {len(tr)}  dev batches {len(dv)}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(tr) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)

    l0, a0 = evaluate(model, dv, device)
    print(f"before training      dev_loss {l0:.4f}  entity_token_acc {a0:.4f}")

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        run = 0.0
        for i, b in enumerate(tr, 1):
            b = {k: v.to(device) for k, v in b.items()}
            out = model(**b)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run += out.loss.item()
            if i % 50 == 0:
                print(f"  epoch {ep} step {i}/{len(tr)} loss {run/i:.4f}", flush=True)
        l, a = evaluate(model, dv, device)
        print(f"epoch {ep}  train_loss {run/len(tr):.4f}  dev_loss {l:.4f}  entity_token_acc {a:.4f}")

    print(f"trained in {time.time()-t0:.1f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    model.cpu().save_pretrained(OUT)
    tok.save_pretrained(OUT)
    (OUT / "TRAINING_NOTES.md").write_text(
        "# Fine-tuned IndicNER (CNI)\n\n"
        f"Base: {BASE}\nEpochs: {EPOCHS}  LR: {LR}  Batch: {BATCH}  Seed: {SEED}\n\n"
        "Trained ONLY on in-gazetteer entities (Avadi, Chennai, Kalpakkam, Vizag,\n"
        "DRDO, ISRO, BHEL + Tamil forms) plus non-CNI distractors.\n\n"
        "Held-out CNI entities (NPCIL, BARC, Sriharikota, Kudankulam, Trombay,\n"
        "Tarapur, Ennore, HAL) never appear in training. Performance on the\n"
        "eval set's held_out slice is therefore a real generalisation test.\n\n"
        "Training data is SYNTHETIC and template-generated. Scores on similarly\n"
        "generated data measure pattern memorisation, not field performance.\n",
        encoding="utf-8")
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
