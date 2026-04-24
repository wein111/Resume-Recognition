"""
NER demo: ``ner/model/model-state.bin`` + ``ner/vocab/vocab.txt`` on ``ner/demo/Resume - Ayush Srivastava.pdf``.

From repository root:

  python ner/test.py

Writes ``ner/demo/demo_entities_YYYYMMDD_HHMMSS.json`` (full merged entities + summary).
"""
import io
import json
import os
import sys
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from ner.server.model_loader import load_resume_ner_model
from ner.server.utils import (
    entity_label_counts,
    idx2tag,
    merge_entities_by_offsets,
    predict,
    preprocess_data,
)

NER_DIR = os.path.join(_ROOT, "ner")
MODEL_PATH = os.path.join(NER_DIR, "model", "model-state.bin")
VOCAB_PATH = os.path.join(NER_DIR, "vocab", "vocab.txt")
DEMO_DIR = os.path.join(NER_DIR, "demo")
DEMO_PDF = os.path.join(DEMO_DIR, "Resume - Ayush Srivastava.pdf")
MAX_LEN = 500


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(MODEL_PATH):
        print(f"Missing weights: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(DEMO_PDF):
        print(f"Missing demo PDF: {DEMO_PDF}", file=sys.stderr)
        sys.exit(1)

    print("Loading model…")
    model, tokenizer = load_resume_ner_model(MODEL_PATH, VOCAB_PATH, device)

    with open(DEMO_PDF, "rb") as f:
        text = preprocess_data(io.BytesIO(f.read()))
    print(f"Extracted text length: {len(text)} chars")

    entities = predict(model, tokenizer, idx2tag, device, text, MAX_LEN)
    merged = merge_entities_by_offsets(entities, text, verbose=False)
    summary = entity_label_counts(merged)

    print("\nEntity counts (merged spans):")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    preview = merged[:25]
    print(f"\nFirst {len(preview)} merged entities:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DEMO_DIR, f"demo_entities_{stamp}.json")
    os.makedirs(DEMO_DIR, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "pdf": os.path.basename(DEMO_PDF),
        "entity_summary": summary,
        "entities": merged,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full result → {out_path}")


if __name__ == "__main__":
    main()
