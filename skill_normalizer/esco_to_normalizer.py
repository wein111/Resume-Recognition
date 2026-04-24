import csv, json, os, re
from collections import defaultdict, Counter

# === CONFIG ===
INPUT_CSV = "data/skills_en.csv"        
OUT_DIR = "skill_normalizer"
FORCE_TYPE = None
ONLY_TECH = False
MIN_LEN = 2

BUILTIN_TECH_ALIAS = {
}

TECH_WHITELIST = {
    "python","java","javascript","typescript","c","c++","go","rust","kotlin","swift","scala","php","ruby",
    "bash","shell","sql","nosql","mysql","postgresql","sqlite","mongodb","redis","elasticsearch","cassandra",
    "hadoop","spark","hive","flink","kafka","airflow","dbt",
    "linux","windows","macos","git","docker","kubernetes","k8s","terraform","ansible","jenkins",
    "html","css","react","next.js","vue","angular","node.js","spring","django","flask","fastapi","rails",
    "aws","gcp","azure","lambda","s3","bigquery","cloud run","ecs","eks","emr",
    "machine learning","deep learning","nlp","data science","ai","cv","transformers","bert","gpt","llm",
    "pytorch","tensorflow","keras","sklearn","xgboost","lightgbm","hugging face","opencv"
}
SOFT_BLACKLIST = {"communication","leadership","management","teamwork","negotiation","customer","creativity","empathy"}

def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def split_multi(s: str):
    if not s: return []
    parts = re.split(r"[;,|]", s)
    out = []
    for p in parts:
        t = norm_text(p)
        if t and t not in out:
            out.append(t)
    return out

def looks_tech(label: str) -> bool:
    if label in TECH_WHITELIST: return True
    if any(w in label for w in SOFT_BLACKLIST): return False
    patterns = [
        r"\b(c\+\+|node\.js|react|vue|angular|django|flask|fastapi|docker|kubernetes|aws|gcp|azure)\b",
        r"\b(sql|nosql|mysql|postgres|mongodb|redis|spark|hadoop|airflow|tensorflow|pytorch|ml|nlp|ai|cv|gpt)\b",
        r"\b(java(script)?|typescript|python|go|rust|php|bash)\b",
    ]
    for pat in patterns:
        if re.search(pat, label):
            return True
    return False

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)


    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"❌ File not found: {INPUT_CSV}")

    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise SystemExit("❌ CSV is empty!")

    cols = set(rows[0].keys())
    print(f"✅ Loaded {len(rows)} rows, columns: {sorted(cols)}")

    st_counter = Counter(norm_text(r.get("skillType","")) for r in rows)
    print(f"📊 skillType distribution: {dict(st_counter)}")


    has_hidden = "hiddenLabels" in cols
    if has_hidden:
        print("🔎 Found 'hiddenLabels' column; will merge into aliases.")

    vocab = {}
    alias_map = defaultdict(set)

    kept = 0
    for r in rows:
        canon = norm_text(r.get("preferredLabel",""))
        if not canon or len(canon) < MIN_LEN: 
            continue

        stype = FORCE_TYPE  
        aliases = set()

        # altLabels
        for a in split_multi(r.get("altLabels","")):
            if a and a != canon:
                aliases.add(a)


        if has_hidden:
            for a in split_multi(r.get("hiddenLabels","")):
                if a and a != canon:
                    aliases.add(a)

        if ONLY_TECH:
            keep = looks_tech(canon) or any(looks_tech(a) for a in aliases)
            if not keep:
                continue

        vocab[canon] = {"type": stype, "source": "esco"}
        for a in aliases:
            alias_map[canon].add(a)
        kept += 1


    for can, alist in BUILTIN_TECH_ALIAS.items():
        if can not in vocab:
            vocab[can] = {"type": "skill", "source": "custom"}
        merged = set(alias_map.get(can, set())) | {norm_text(x) for x in alist}
        merged.discard(can)
        alias_map[can] = merged

    alias_json = {k: sorted(v) for k, v in alias_map.items() if v}

    vocab_path = os.path.join(OUT_DIR, "vocab.json")
    alias_path = os.path.join(OUT_DIR, "alias.json")
    json.dump(vocab, open(vocab_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(alias_json, open(alias_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n🎉 Done! Generated:\n  - {vocab_path} ({len(vocab)} skills)\n  - {alias_path} ({sum(bool(v) for v in alias_json.values())} with aliases)")
    print("   (altLabels + hiddenLabels merged; builtin aliases added: pytorch/js/k8s/tf/react.js/py3 …)")

if __name__ == "__main__":
    main()