import io
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from skill_normalizer.skill_normalizer.core import SkillNormalizer
from skill_normalizer.skill_normalizer.from_ner import (
    expand_skill_fragments,
    raw_skills_from_entities,
)

from job_ingestion.resume_job_match import (
    load_job_postings_jsonl,
    rank_jobs_by_skill_overlap,
)
from ner.server.model_loader import load_resume_ner_model
from ner.server.utils import (
    preprocess_data,
    predict,
    idx2tag,
    merge_entities_by_offsets,
    entity_label_counts,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

NER_DIR = os.path.join(ROOT_DIR, "ner")
VOCAB_PATH = os.path.join(NER_DIR, "vocab", "vocab.txt")
_MODEL_IN_NER = os.path.join(NER_DIR, "model", "model-state.bin")
MODEL_PATH = (
    _MODEL_IN_NER
    if os.path.isfile(_MODEL_IN_NER)
    else os.path.join(ROOT_DIR, "model-state.bin")
)

VOCAB_JSON = os.path.join(ROOT_DIR, "skill_normalizer", "skill_normalizer", "vocab.json")
ALIAS_JSON = os.path.join(ROOT_DIR, "skill_normalizer", "skill_normalizer", "alias.json")

SKILL_NORMALIZER = SkillNormalizer(
    VOCAB_JSON,
    ALIAS_JSON,
    fuzzy_cutoff=88,
    min_conf=0.70,
)

JOB_POSTINGS_PATH = os.path.join(
    ROOT_DIR,
    "job_ingestion",
    "output",
    "job_postings_extracted.jsonl",
)

PIPELINE_DIR = os.path.join(ROOT_DIR, "pipeline")
INPUT_DIR = os.path.join(PIPELINE_DIR, "input")
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")

MAX_LEN = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _print_pipeline_paths() -> None:
    print("📂 Pipeline I/O:", PIPELINE_DIR)
    print("📂 INPUT  DIR:", INPUT_DIR)
    print("📂 OUTPUT DIR:", OUTPUT_DIR)
    print("📄 JOB DATA  :", JOB_POSTINGS_PATH)


def _list_pdf_paths_or_none(input_dir: str) -> Optional[List[str]]:
    """Return absolute paths to PDFs under input_dir, or None if unusable."""
    if not os.path.isdir(input_dir):
        print(
            "❌ Input directory does not exist. Create pipeline/input/ and put PDF resumes there."
        )
        return None
    names = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not names:
        print("⚠ No PDF files found in pipeline/input/.")
        return None
    return [os.path.join(input_dir, n) for n in names]


def _load_jobs_or_none(job_jsonl_path: str) -> Optional[List[Dict]]:
    jobs = load_job_postings_jsonl(job_jsonl_path)
    if not jobs:
        print(
            f"⚠ No job postings loaded from {job_jsonl_path}. "
            "Run job_ingestion to produce job_postings_extracted.jsonl."
        )
        return None
    print(f"📥 Loaded {len(jobs)} job postings")
    return jobs


def _load_ner_model_or_none() -> Optional[Tuple[torch.nn.Module, object]]:
    print("🔧 Loading model & tokenizer...")
    try:
        return load_resume_ner_model(MODEL_PATH, VOCAB_PATH, DEVICE)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return None


def process_pdf(pdf_path: str, jobs: List[Dict], model, tokenizer) -> None:
    print(f"\n📄 Processing: {pdf_path}")

    with open(pdf_path, "rb") as f:
        data = io.BytesIO(f.read())
    text = preprocess_data(data)

    entities = predict(model, tokenizer, idx2tag, DEVICE, text, MAX_LEN)
    merged = merge_entities_by_offsets(entities, text, verbose=False)
    entity_summary = entity_label_counts(merged)

    skills_ner_spans = raw_skills_from_entities(merged)
    skill_phrases = expand_skill_fragments(skills_ner_spans)
    normalized, audit = SKILL_NORMALIZER.normalize(skill_phrases)
    skills_canonical = [n["canonical"] for n in normalized]

    matched_jobs = rank_jobs_by_skill_overlap(skills_canonical, jobs, top_k=20)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    entities_path = os.path.join(OUTPUT_DIR, base + "_entities.json")
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    norm_path = os.path.join(OUTPUT_DIR, base + "_normalized.json")
    result: Dict = {
        "pdf_name": os.path.basename(pdf_path),
        "entity_summary": entity_summary,
        "skills_raw": skills_ner_spans,
        "skill_phrases": skill_phrases,
        "skills_normalized": normalized,
        "skills_canonical": skills_canonical,
        "audit": audit,
        "debug_files": {"entities": os.path.basename(entities_path)},
    }
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    matches_path = os.path.join(OUTPUT_DIR, base + "_job_matches.json")
    with open(matches_path, "w", encoding="utf-8") as f:
        json.dump(matched_jobs, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved merged entities     → {entities_path}")
    print(f"✅ Saved normalized skills → {norm_path}")
    print(f"✅ Saved job matches      → {matches_path}")
    if matched_jobs:
        top = matched_jobs[0]
        print(
            f"   Top match: [{top['score']}] {top['company_norm']} - "
            f"{top['title_clean']} @ {top['location_clean']}"
        )
    else:
        print("   No matching jobs found.")


def main():
    _print_pipeline_paths()
    pdf_paths = _list_pdf_paths_or_none(INPUT_DIR)
    if pdf_paths is None:
        return
    jobs = _load_jobs_or_none(JOB_POSTINGS_PATH)
    if jobs is None:
        return
    loaded = _load_ner_model_or_none()
    if loaded is None:
        return
    model, tokenizer = loaded

    print(f"🔍 Found {len(pdf_paths)} PDF file(s).")
    for path in pdf_paths:
        process_pdf(path, jobs, model, tokenizer)


if __name__ == "__main__":
    main()
