"""
Compare two NER model checkpoints on demo PDFs.

From repository root:
  python ner/compare_models_demo.py

Optional:
  python ner/compare_models_demo.py --pdf "Resume - Ayush Srivastava.pdf"
"""

import argparse
import io
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ner.server.model_loader import load_resume_ner_model
from ner.server.utils import (
    entity_label_counts,
    idx2tag,
    merge_entities_by_offsets,
    predict,
    preprocess_data,
)

NER_DIR = os.path.join(_ROOT, "ner")
MODEL_DIR = os.path.join(NER_DIR, "model")
VOCAB_PATH = os.path.join(NER_DIR, "vocab", "vocab.txt")
DEMO_DIR = os.path.join(NER_DIR, "demo")
MAX_LEN = 500

MODEL_FILES = [
    "model-state.bin",
    "model-state-origin.bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run demo PDFs with two model checkpoints and compare outputs."
    )
    parser.add_argument(
        "--pdf",
        default="",
        help="Optional single PDF filename under ner/demo/ to test.",
    )
    return parser.parse_args()


def collect_demo_pdfs(single_pdf_name: str) -> List[str]:
    if single_pdf_name:
        pdf_path = os.path.join(DEMO_DIR, single_pdf_name)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"Demo PDF not found: {pdf_path}")
        return [pdf_path]

    pdfs = [
        os.path.join(DEMO_DIR, name)
        for name in sorted(os.listdir(DEMO_DIR))
        if name.lower().endswith(".pdf")
    ]
    if not pdfs:
        raise FileNotFoundError(f"No demo PDFs found in: {DEMO_DIR}")
    return pdfs


def available_models() -> Dict[str, str]:
    found = {}
    for filename in MODEL_FILES:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.isfile(path):
            found[filename] = path
    if len(found) < 2:
        raise FileNotFoundError(
            f"Need both model files in {MODEL_DIR}: {', '.join(MODEL_FILES)}"
        )
    return found


def run_one_pdf(model, tokenizer, pdf_path: str) -> Dict:
    with open(pdf_path, "rb") as f:
        text = preprocess_data(io.BytesIO(f.read()))
    entities = predict(model, tokenizer, idx2tag, DEVICE, text, MAX_LEN)
    merged = merge_entities_by_offsets(entities, text, verbose=False)
    return {
        "pdf": os.path.basename(pdf_path),
        "text_length": len(text),
        "entity_summary": entity_label_counts(merged),
        "entity_count": len(merged),
        "entities": merged,
    }


def main() -> None:
    args = parse_args()
    models = available_models()
    pdf_paths = collect_demo_pdfs(args.pdf)

    print(f"Device: {DEVICE}")
    print(f"Demo PDFs: {len(pdf_paths)}")

    all_results = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(DEVICE),
        "models": {},
    }

    for model_name, model_path in models.items():
        print(f"\nLoading model: {model_name}")
        model, tokenizer = load_resume_ner_model(model_path, VOCAB_PATH, DEVICE)
        per_pdf_results = []

        for pdf_path in pdf_paths:
            print(f"  Running on: {os.path.basename(pdf_path)}")
            one = run_one_pdf(model, tokenizer, pdf_path)
            print(f"    Merged entities: {one['entity_count']}")
            per_pdf_results.append(one)

        all_results["models"][model_name] = per_pdf_results

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DEMO_DIR, f"demo_compare_models_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved comparison result -> {out_path}")


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
