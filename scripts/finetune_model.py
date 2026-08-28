"""Fine-tune ANY encoder for CNI entity extraction -- generalises
finetune_indicner.py so the model choice can be justified by controlled
comparison rather than asserted.

Same procedure, same data, same seed for every candidate: that is what makes
the comparison mean something. Only the base checkpoint changes.

One asymmetry is real and reported, not hidden: ai4bharat/IndicNER already
ships a matching B/I-LOC,ORG,PER,O head, so training warm-starts a pretrained
classifier. The other candidates have no NER head for this label set, so
AutoModelForTokenClassification attaches a randomly-initialised one. That is
not a flaw in the experiment -- it is part of the answer to "why IndicNER":
domain-specific pretraining includes a head start the generic encoders do not
have.

    python scripts/finetune_model.py --base google/muril-base-cased --slug muril
"""

from __future__ import annotations

import argparse
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
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
HF_TOKEN = os.getenv("HF_TOKEN") or None
DATA = ROOT / "data" / "finetune"

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
                labels.append(-100)
                continue
            tag = "O"
            for ent in r["entities"]:
                if s >= ent["start"] and e <= ent["end"]:
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
        ent = mask & (b["labels"] != L2I["O"])
        correct += (pred[ent] == b["labels"][ent]).sum().item()
        total += ent.sum().item()
    return loss_sum / max(n, 1), correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="HF model id, e.g. google/muril-base-cased")
    parser.add_argument("--slug", required=True, help="short name for the output dir")
    args = parser.parse_args()

    out_dir = ROOT / "data" / "models" / f"{args.slug}-cni-ft"

    torch.manual_seed(SEED)
    device = pick_device()
    print(f"[{args.slug}] base={args.base} device={device}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base, token=HF_TOKEN)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(LABELS),
        id2label={i: l for l, i in L2I.items()}, label2id=L2I,
        token=HF_TOKEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    tr = DataLoader(NerDataset(DATA / "train.jsonl", tok), batch_size=BATCH, shuffle=True)
    dv = DataLoader(NerDataset(DATA / "dev.jsonl", tok), batch_size=BATCH)
    print(f"[{args.slug}] params={n_params:,} train_batches={len(tr)} dev_batches={len(dv)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(tr) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)

    l0, a0 = evaluate(model, dv, device)
    print(f"[{args.slug}] before training  dev_loss={l0:.4f}  entity_token_acc={a0:.4f}", flush=True)

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
        l, a = evaluate(model, dv, device)
        print(f"[{args.slug}] epoch {ep}  train_loss={run/len(tr):.4f}  dev_loss={l:.4f}  entity_token_acc={a:.4f}", flush=True)

    train_seconds = time.time() - t0
    print(f"[{args.slug}] trained in {train_seconds:.1f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.cpu().save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "TRAINING_NOTES.md").write_text(
        f"# Fine-tuned {args.base} (CNI)\n\n"
        f"Base: {args.base}\nEpochs: {EPOCHS}  LR: {LR}  Batch: {BATCH}  Seed: {SEED}\n"
        f"Parameters: {n_params:,}\nTraining time: {train_seconds:.1f}s on {device}\n\n"
        "Same corpus, same procedure, same seed as every other model in this "
        "comparison -- see docs/model_comparison.md.\n",
        encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({
        "base_model": args.base, "slug": args.slug, "n_params": n_params,
        "train_seconds": round(train_seconds, 1), "device": str(device),
        "dev_entity_token_acc": round(a, 4),
    }, indent=2), encoding="utf-8")
    print(f"[{args.slug}] saved -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
