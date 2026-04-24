import json
import os
from typing import List, Dict


def load_job_postings_jsonl(path: str) -> List[Dict]:
    """
    Load job postings from job_ingestion JSONL export.
    Each line: JSON with company_norm, title_clean, location_clean, skills_norm, etc.
    """
    if not path or not os.path.exists(path):
        return []

    jobs: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return jobs


def rank_jobs_by_skill_overlap(
    resume_skills: List[str],
    jobs: List[Dict],
    top_k: int = 20,
) -> List[Dict]:
    """
    Overlap-based matching: score = count of shared canonical skills.
    """
    if not jobs:
        return []

    resume_set = set(s for s in resume_skills if s)
    scored: List[Dict] = []

    for job in jobs:
        job_skills = set(job.get("skills_norm") or [])
        overlap = resume_set & job_skills
        score = len(overlap)
        if score == 0:
            continue

        scored.append(
            {
                "company_norm": job.get("company_norm"),
                "title_clean": job.get("title_clean"),
                "location_clean": job.get("location_clean"),
                "skills_norm": job.get("skills_norm"),
                "score": score,
                "matched_skills": sorted(list(overlap)),
            }
        )

    scored.sort(key=lambda j: j["score"], reverse=True)
    return scored[:top_k]
