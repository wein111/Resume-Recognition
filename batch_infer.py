import os
import io
import json
import torch
import re
from typing import List, Dict

from transformers import BertTokenizerFast, BertForTokenClassification
from NER.server.utils import preprocess_data, predict, idx2tag
from skillnormalizer.skill_normalizer.core import SkillNormalizer


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


NER_DIR = os.path.join(ROOT_DIR, "NER")
VOCAB_PATH = os.path.join(NER_DIR, "vocab", "vocab.txt")
MODEL_PATH = os.path.join(ROOT_DIR, "model-state.bin")


VOCAB_JSON = os.path.join(ROOT_DIR, "skillnormalizer", "skill_normalizer", "vocab.json")
ALIAS_JSON = os.path.join(ROOT_DIR, "skillnormalizer", "skill_normalizer", "alias.json")

SKILL_NORMALIZER = SkillNormalizer(
    VOCAB_JSON,
    ALIAS_JSON,
    fuzzy_cutoff=88,
    min_conf=0.70
)

JOB_POSTINGS_PATH = os.path.join(
    ROOT_DIR,
    "Job-ingestion",
    "output",
    "job_postings_extracted.jsonl"
)


INPUT_DIR = os.path.join(ROOT_DIR, "input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

SKILL_NORMALIZER = SkillNormalizer(
    VOCAB_JSON,
    ALIAS_JSON,
    fuzzy_cutoff=88,
    min_conf=0.70
)

# ====== Job postings (Task 4, from Job-ingestion) ======


def quick_extract_skills(entities: List[Dict]) -> List[str]:
    """
    Extract raw skill strings from merged NER entities (label == 'Skills').
    """
    out = []
    for e in entities:
        if str(e.get("label", "")).lower() != "skills":
            continue
        text = e.get("text", "").strip()
        if not text:
            continue

        # Remove section titles like "Programming Languages: C++, Python"
        if ":" in text:
            text = text.split(":", 1)[1]

        # Split by common delimiters
        parts = re.split(r"[;,/|•]", text)
        for p in parts:
            p = p.strip(" .:-\t\n")
            if p and len(p) > 1:
                out.append(p)

    # Deduplicate while keeping order
    seen, result = set(), []
    for x in out:
        key = x.lower()
        if key not in seen:
            seen.add(key)
            result.append(x)
    return result


# ====== NER model loading ======
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


def merge_entities_by_offsets(entities: List[Dict], full_text: str) -> List[Dict]:
    """
    Merge BIO-tagged token-level entities into span-level entities.
    """
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


# ====== Job-ingestion helpers: load jobs + match jobs ======
def load_jobs(path: str) -> List[Dict]:
    """
    Load job postings with skills_norm from Job-ingestion output JSONL.
    Each line is a JSON object with at least `company_norm`, `title_clean`,
    `location_clean`, and `skills_norm`.
    """
    if not os.path.exists(path):
        print(f"⚠ Job postings file not found: {path}")
        return []

    jobs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"📥 Loaded {len(jobs)} job postings from {path}")
    return jobs


def match_jobs(resume_skills: List[str], jobs: List[Dict], top_k: int = 20) -> List[Dict]:
    """
    Simple overlap-based job matching:
    - score = number of overlapping canonical skills
    - return top_k jobs sorted by score
    """
    if not jobs:
        return []

    resume_set = set(s for s in resume_skills if s)
    scored = []

    for job in jobs:
        job_skills = set(job.get("skills_norm") or [])
        overlap = resume_set & job_skills
        score = len(overlap)
        if score == 0:
            continue

        job_view = {
            "company_norm": job.get("company_norm"),
            "title_clean": job.get("title_clean"),
            "location_clean": job.get("location_clean"),
            "skills_norm": job.get("skills_norm"),
            "score": score,
            "matched_skills": sorted(list(overlap))
        }
        scored.append(job_view)

    scored.sort(key=lambda j: j["score"], reverse=True)
    return scored[:top_k]


# ====== Main per-PDF pipeline: NER + Skill Normalization + Job Matching ======
def process_pdf(pdf_path: str, output_dir: str, jobs: List[Dict]):
    print(f"\n📄 Processing: {pdf_path}")

    # 1) PDF -> text
    with open(pdf_path, "rb") as f:
        data = io.BytesIO(f.read())
    text = preprocess_data(data)

    # 2) NER prediction
    entities = predict(model, TOKENIZER, idx2tag, DEVICE, text, MAX_LEN)

    # 3)
    merged = merge_entities_by_offsets(entities, text)

    # 4)
    skills_raw = quick_extract_skills(merged)

    # 5) 
    normalized, audit = SKILL_NORMALIZER.normalize(skills_raw)
    skills_canonical = [n["canonical"] for n in normalized]

    # 6) 
    matched_jobs = match_jobs(skills_canonical, jobs, top_k=20)

    # 7) 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    # a) 
    norm_path = os.path.join(OUTPUT_DIR, base + "_normalized.json")
    result = {
        "skills_raw": skills_raw,
        "skills_normalized": normalized,
        "skills_canonical": skills_canonical,
        "audit": audit
    }
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # b) 
    matches_path = os.path.join(OUTPUT_DIR, base + "_job_matches.json")
    with open(matches_path, "w", encoding="utf-8") as f:
        json.dump(matched_jobs, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved normalized skills → {norm_path}")
    print(f"✅ Saved job matches      → {matches_path}")
    if matched_jobs:
        top = matched_jobs[0]
        print(f"   Top match: [{top['score']}] {top['company_norm']} - {top['title_clean']} @ {top['location_clean']}")
    else:
        print("   No matching jobs found.")


def main():
    print("📂 INPUT  DIR:", INPUT_DIR)
    print("📂 OUTPUT DIR:", OUTPUT_DIR)
    print("📄 JOB DATA  :", JOB_POSTINGS_PATH)

    if not os.path.isdir(INPUT_DIR):
        print("❌ Input directory does not exist. Please create 'input' folder and put PDFs inside.")
        return

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠ No PDF files found in input directory.")
        return


    jobs = load_jobs(JOB_POSTINGS_PATH)
    if not jobs:
        print("⚠ No job postings loaded. Did you run Job-ingestion to produce job_postings_extracted.jsonl?")
        return

    print(f"🔍 Found {len(pdf_files)} PDF file(s).")

    for name in pdf_files:
        process_pdf(
            os.path.join(INPUT_DIR, name),
            OUTPUT_DIR,
            jobs)


if __name__ == "__main__":
    main()