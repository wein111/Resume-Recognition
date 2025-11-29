import os
import io
import json
import argparse
import re
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))        # Resume-NER-master/
NER_DIR = os.path.join(ROOT_DIR, "NER")                      # Resume-NER-master/NER


VOCAB_PATH = os.path.join(NER_DIR, "vocab", "vocab.txt")
MODEL_PATH = os.path.join(ROOT_DIR, "model-state.bin")  

from NER.server.utils import preprocess_data, predict, idx2tag


from skillnormalizer.skill_normalizer.core import SkillNormalizer

BASE = ROOT_DIR
VOCAB_JSON = os.path.join(BASE, "skillnormalizer", "skill_normalizer", "vocab.json")
ALIAS_JSON = os.path.join(BASE, "skillnormalizer", "skill_normalizer", "alias.json")

SKILL_NORMALIZER = SkillNormalizer(
    VOCAB_JSON,
    ALIAS_JSON,
    fuzzy_cutoff=88,
    min_conf=0.70
)



def quick_extract_skills(entities):

    out = []
    for e in entities:
        if str(e.get("label", "")).lower() != "skills":
            continue
        text = e.get("text", "").strip()
        if not text:
            continue

        if ":" in text:
            text = text.split(":", 1)[1]

        parts = re.split(r"[;,/|•]", text)
        for p in parts:
            p = p.strip(" .:-\t\n")
            if p and len(p) > 1:
                out.append(p)


    seen, result = set(), []
    for x in out:
        key = x.lower()
        if key not in seen:
            seen.add(key)
            result.append(x)
    return result

MAX_LEN = 500
NUM_LABELS = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔧 Loading model & tokenizer...")

STATE_DICT = torch.load(MODEL_PATH, map_location=DEVICE)
TOKENIZER = BertTokenizerFast(VOCAB_PATH, lowercase=True)

model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    state_dict=STATE_DICT["model_state_dict"],
    num_labels=NUM_LABELS
)
model.to(DEVICE)
model.eval()


def merge_entities_by_offsets(entities, full_text):
    merged = []
    current = None

    for ent in entities:
        label = ent.get("entity", "")

        if label == "O":
            if current:
                merged.append(current)
                current = None
            continue

        if label.startswith("B-"):
            if current:
                merged.append(current)
            entity_type = label[2:]
            current = {
                "label": entity_type,
                "start": ent["start"],
                "end": ent["end"],
                "text": ent["text"]
            }

        elif label.startswith("I-"):
            entity_type = label[2:]
            if current and current["label"] == entity_type:
                current["end"] = ent["end"]
                current["text"] = full_text[current["start"]:current["end"]]
            else:
                if current:
                    merged.append(current)
                current = {
                    "label": entity_type,
                    "start": ent["start"],
                    "end": ent["end"],
                    "text": ent["text"]
                }

        else:
            if current:
                merged.append(current)
            current = {
                "label": label,
                "start": ent["start"],
                "end": ent["end"],
                "text": ent["text"]
            }

    if current:
        merged.append(current)

    return merged



def process_pdf(pdf_path, output_dir):
    print(f"\n📄 Processing: {pdf_path}")


    with open(pdf_path, "rb") as f:
        data = io.BytesIO(f.read())
    text = preprocess_data(data)


    entities = predict(model, TOKENIZER, idx2tag, DEVICE, text, MAX_LEN)


    merged = merge_entities_by_offsets(entities, text)


    skills_raw = quick_extract_skills(merged)


    normalized, audit = SKILL_NORMALIZER.normalize(skills_raw)


    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(output_dir, base + "_normalized.json")

    result = {
        "skills_raw": skills_raw,
        "skills_normalized": normalized,
        "audit": audit
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved normalized skills → {out_path}")



def main():
    parser = argparse.ArgumentParser(description="Batch NER + Skill Normalization for resume PDFs")
    parser.add_argument("--input_dir", type=str, default="input", help="input directory relative to project root")
    parser.add_argument("--output_dir", type=str, default="output", help="output directory relative to project root")

    args = parser.parse_args()

    input_dir = os.path.join(ROOT_DIR, args.input_dir)
    output_dir = os.path.join(ROOT_DIR, args.output_dir)

    print("📂 INPUT  DIR:", input_dir)
    print("📂 OUTPUT DIR:", output_dir)

    if not os.path.isdir(input_dir):
        print("❌ Input directory does not exist.")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠ No PDF files found in input directory.")
        return

    print(f"🔍 Found {len(pdf_files)} PDF file(s).")

    for name in pdf_files:
        process_pdf(os.path.join(input_dir, name), output_dir)


if __name__ == "__main__":
    main()