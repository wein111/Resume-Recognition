import json
from skill_normalizer.core import SkillNormalizer

if __name__ == "__main__":
    n = SkillNormalizer(
        "skill_normalizer/vocab.json",
        "skill_normalizer/alias.json",
        fuzzy_cutoff=90,
        min_conf=0.76
    )
    with open("data/mock_skills.json", "r", encoding="utf-8") as f:
        skills = json.load(f)

    normalized, audit = n.normalize(skills)
    print("=== Normalized ===")
    for item in normalized:
        print(f"{item['from']:20s} -> {item['canonical']:18s} (conf={item['confidence']:.2f}, via={item['method']})")
    print("\n=== Rejected/Audit ===")
    for a in audit:
        print(a)