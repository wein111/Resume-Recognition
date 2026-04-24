from skill_normalizer.core import SkillNormalizer

n = SkillNormalizer(
    "skill_normalizer/vocab.json",
    "skill_normalizer/alias.json",
    fuzzy_cutoff=90,
    min_conf=0.75
)

skills = ["Tensor Flow", "C++", "AWS", "Communication", "Data Science"]
normalized, audit = n.normalize(skills)

print("=== Normalized ===")
for s in normalized:
    print(f"{s['from']:20s} -> {s['canonical']:25s} ({s['method']}, conf={s['confidence']:.2f})")

print("\n=== Rejected ===")
for a in audit:
    print(a)