"""
Train BERT token-classification NER on resume annotations (Dataturks JSONL).

  python ner/train.py

Paths are resolved from this file; weights are saved under ``ner/model/model-state.bin``.
Edit the config block below to change hyperparameters or paths.
"""
import os
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, BertForTokenClassification, BertTokenizerFast

from utils import (
    convert_goldparse,
    idx2tag,
    ResumeDataset,
    tag2idx,
    train_and_val_model,
    trim_entity_spans,
)

# ----- config -----
# MAX_LEN: each resume is truncated to this many WordPieces (BERT cap 512). Gold spans
# entirely beyond truncation are never supervised; for long CVs consider chunked
# training or windowing to match inference (ner/server/utils sliding predict).
EPOCHS = 5
TRAIN_BATCH = 8
VAL_BATCH = 4
LR = 5e-5
MAX_LEN = 500
MODEL_NAME = "bert-base-uncased"
MAX_GRAD_NORM = 1.0


def main() -> None:
    ner_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(ner_dir)

    data_path = os.path.join(ner_dir, "data", "Resumes.json")
    vocab_path = os.path.join(ner_dir, "vocab", "vocab.txt")
    model_dir = os.path.join(ner_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, "model-state.bin")

    if not os.path.isfile(data_path):
        print(
            f"Missing training file: {data_path}\n"
            "Put the Kaggle Resume NER JSONL at ner/data/Resumes.json.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isfile(vocab_path):
        print(f"Missing vocab file: {vocab_path}", file=sys.stderr)
        sys.exit(1)

    raw = convert_goldparse(data_path)
    if not raw:
        print("No training examples loaded (empty file or parse error).", file=sys.stderr)
        sys.exit(1)

    data = trim_entity_spans(raw)
    train_data, val_data = data[:180], data[180:]

    tokenizer = BertTokenizerFast(vocab_path, lowercase=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = torch.cuda.is_available()

    train_d = ResumeDataset(train_data, tokenizer, tag2idx, MAX_LEN)
    val_d = ResumeDataset(val_data, tokenizer, tag2idx, MAX_LEN)

    train_dl = DataLoader(
        train_d,
        sampler=RandomSampler(train_d),
        batch_size=TRAIN_BATCH,
        num_workers=0,
        pin_memory=pin,
    )
    val_dl = DataLoader(
        val_d,
        batch_size=VAL_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(tag2idx),
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)

    print(f"Device: {device} | Train {len(train_data)} | Val {len(val_data)} | Out {output_path}")

    train_and_val_model(
        model,
        tokenizer,
        optimizer,
        EPOCHS,
        idx2tag,
        tag2idx,
        MAX_GRAD_NORM,
        device,
        train_dl,
        val_dl,
    )

    torch.save({"model_state_dict": model.state_dict()}, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
