import json, re, os
from skill_normalizer.core import SkillNormalizer

# ---------- Paths ----------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENTITIES = os.path.join(BASE, "skill-normalizer\data", "entities.json")
VOCAB = os.path.join(BASE, "skill-normalizer\skill_normalizer", "vocab.json")
ALIAS = os.path.join(BASE, "skill-normalizer\skill_normalizer", "alias.json")
OUT_DETAILED = os.path.join(BASE, "skill-normalizer\data", "normalized_detailed.json")
OUT_SIMPLE   = os.path.join(BASE, "skill-normalizer\data", "normalized_for_task4.json")

# ---------- Simple extractor ----------
def quick_extract_skills(entities):
    """Extract skill phrases from Task 2 entities.json"""
    out = []
    for e in entities:
        if str(e.get("label","")).lower() != "skills":
            continue
        text = e.get("text","").strip()
        if not text:
            continue
        # remove section titles like “Programming Languages: ...”
        if ":" in text:
            text = text.split(":",1)[1]
        # split by common delimiters
        parts = re.split(r"[;,/|•]", text)
        for p in parts:
            p = p.strip(" .:-\t\n")
            if p and len(p)>1:
                out.append(p)
    # deduplicate while keeping order
    seen, result = set(), []
    for x in out:
        key = x.lower()
        if key not in seen:
            seen.add(key)
            result.append(x)
    return result

# ---------- Main ----------
def main():
    if not os.path.exists(ENTITIES):
        raise SystemExit(f" Cannot find {ENTITIES}")

    entities = json.load(open(ENTITIES, "r", encoding="utf-8"))
    print(f"Loaded entities.json with {len(entities)} records")

    skills_raw = quick_extract_skills(entities)
    print(f"Extracted {len(skills_raw)} raw skills:", skills_raw[:10], "...")

    n = SkillNormalizer(VOCAB, ALIAS, fuzzy_cutoff=88, min_conf=0.70)
    normalized, audit = n.normalize(skills_raw)

    json.dump({
        "skills_raw": skills_raw,
        "skills_normalized": normalized,
        "audit": audit
    }, open(OUT_DETAILED,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    json.dump({
        "skills": [x["canonical"] for x in normalized]
    }, open(OUT_SIMPLE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"Normalization complete!\n  Saved:\n  {OUT_DETAILED}\n  {OUT_SIMPLE}")
    print("Sample normalized skills:", [x["canonical"] for x in normalized[:10]])

if __name__ == "__main__":
    main()