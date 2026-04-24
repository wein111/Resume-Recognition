import pandas as pd
import os, shutil, textwrap
import re, json
try:
    from rapidfuzz import process as rf_process, fuzz
except Exception:
    rf_process = None
    fuzz = None

# Maximum number of human annotations to do for the data
annotation_max = 1000
print("Annotation will be done for rows 0 to", annotation_max)

raw = input(f"Start index to annotate (0..{annotation_max}): ").strip()
try:
    n = int(raw)
except ValueError:
    n = 0

start_idx = max(0, min(n, annotation_max))
print(f"Starting at row {start_idx} ...")



# Canonical vocabulary (starter list; grow as you label)
CANON_SKILLS = [
    "python","java","c++","c","c#","javascript","typescript","html","css",
    "sql","nosql","postgresql","mysql","mongodb","redis",
    "bash","linux","git","docker","kubernetes","aws","gcp","azure",
    "hadoop","spark","hive","airflow",
    "pytorch","tensorflow","scikit-learn","pandas","numpy",
    "react","node.js","vue","angular","django","flask","spring", "e-learning", "algorithms", "data structures","machine learning", "Adobe"
]

# Aliases/variants you want to fold into canonical names
ALIASES = {
    "py": "python",
    "java script": "javascript",
    "js": "javascript",
    "ts": "typescript",
    "html5": "html",
    "css3": "css",
    "c plus plus": "c++",
    "node": "node.js",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "np": "numpy",
    "db": "data structures",
    "database": "data structures",
    "ml": "machine learning"
}

def _normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = re.sub(r"[\,;/|]+", " ", s)            # splitters to spaces
    s = re.sub(r"\s{2,}", " ", s)              # collapse spaces
    return s

# Precompile exact patterns once (handle + and . safely)
_EXACT_PATTERNS = {tok: re.compile(rf"(?<!\w){re.escape(tok)}(?!\w)", re.I)
                   for tok in CANON_SKILLS + list(ALIASES.keys())}

def normalize_user_input_skills(user_text: str,
                                canon=CANON_SKILLS,
                                aliases=ALIASES,
                                use_fuzzy: bool = True,
                                fuzzy_thresh: int = 90) -> list[str]:
    """
    Turn free-text skill input into a deduped list of canonical skills.
    1) Exact/alias matches
    2) Optional fuzzy fallback (RapidFuzz), per token
    """
    t = _normalize_text(user_text)
    if not t:
        return []

    # 1) Exact + alias recognition against the full string
    hits = set()
    for key, rx in _EXACT_PATTERNS.items():
        if rx.search(t):
            canon = aliases.get(key, key)
            if canon in CANON_SKILLS:
                hits.add(canon)

    # 2) Tokenize simple words/phrases and fuzzy-match per token if requested
    if use_fuzzy and rf_process:
        # split on spaces but keep short phrases users tend to type
        # e.g., "c plus plus", "java script"
        parts = re.split(r"\s+", t)
        # join neighboring short tokens to fish for alias patterns
        # basic trick: also consider 2-grams
        grams = set(parts)
        grams.update([" ".join(pair) for pair in zip(parts, parts[1:])])

        for g in grams:
            g = g.strip()
            if not g or len(g) < 2: 
                continue
            # First: alias table exact lookup on the gram
            if g in aliases:
                if aliases[g] in CANON_SKILLS:
                    hits.add(aliases[g])
                continue
            # Then: fuzzy to canonical list
            match = rf_process.extractOne(
                g, CANON_SKILLS + list(aliases.keys()),
                scorer=fuzz.WRatio
            )
            if match and match[1] >= fuzzy_thresh:
                m = match[0]
                # fold aliases to canonical
                canon = aliases.get(m, m)
                if canon in CANON_SKILLS:
                    hits.add(canon)

    return sorted(hits)

df = pd.read_csv("data/it_job_postings_annotated.csv")

# ['jobpost', 'date', 'Title', 'Company', 'AnnouncementCode', 'Term',
#       'Eligibility', 'Audience', 'StartDate', 'Duration', 'Location',
#       'JobDescription', 'JobRequirment', 'RequiredQual', 'Salary',
#       'ApplicationP', 'OpeningDate', 'Deadline', 'Notes', 'AboutC', 'Attach',
#       'Year', 'Month', 'IT']

def _wrap(text, width):
    if pd.isna(text): 
        return ""
    return textwrap.fill(str(text), width=width)

def annotate(df: pd.DataFrame, save_path="data/it_job_postings_annotated.csv"):
    if "skills_annotated" not in df.columns:
        df["skills_annotated"] = pd.NA

    term_width = shutil.get_terminal_size((100, 20)).columns  
    box_width = max(60, min(term_width - 4, 140))            

    for idx in range(start_idx, annotation_max + 1):
        row = df.iloc[idx]
        os.system("cls" if os.name == "nt" else "clear")

        print(f"--- Row {idx} ---\n")
        print("JobDescription:\n" + _wrap(row.get("JobDescription", ""), box_width) + "\n")
        print("JobRequirment:\n"  + _wrap(row.get("JobRequirment", ""),  box_width) + "\n")
        print("RequiredQual:\n"   + _wrap(row.get("RequiredQual", ""),   box_width) + "\n")

        user_input = input("Enter your annotation for this job posting (or 'exit' to stop): ")
        if user_input.lower() == "exit":
            break

        normalized = normalize_user_input_skills(user_input)

        df.at[idx, "skills_annotated_raw"] = user_input
        df.at[idx, "skills_annotated"] = ";".join(normalized)  # or store as JSON list  

    df.to_csv(save_path, index=False)
    print(f"\nDone. File saved to {save_path}")
    return df

annotate(df)

# df["skills_annotated"] = "NaN"

# df.to_csv("data/it_job_postings_annotated.csv", index=False)