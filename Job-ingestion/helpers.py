import pandas as pd
import re
import json
from typing import Optional, Dict, List
import spacy
import pycountry

def clean_csv():
    df = pd.read_csv("data/online-job-postings.csv")
    print(f"Total job postings: {len(df)}")
    df = df[df["IT"] == 1]
    print(f"Total IT job postings: {len(df)}")
    df.to_csv("data/it_job_postings.csv", index=False)
    return df

COMPANY_SUFFIX_PATTERNS = [
    r"\binc\b\.?", r"\bincorporated\b",
    r"\bllc\b\.?", r"\bl\.l\.c\b\.?",
    r"\bltd\b\.?", r"\blimited\b",
    r"\bcorp\b\.?", r"\bcorporation\b",
    r"\bco\b\.?", r"\bcompany\b",
    r"\bplc\b\.?", r"\bgmbh\b", r"\bs\.?a\.?\b",
    r"\bpvt\b", r"\bprivate\b",
]

TITLE_CLEAN_PATTERNS = [
    (r"\(remote\)", ""),        
    (r"\(hybrid\)", ""),
    (r"\s+-\s+.*$", ""),    
    (r"\s{2,}", " ")          
]

Location_model = spacy.load("en_core_web_sm")
COUNTRY_NAMES = {c.name.lower() for c in pycountry.countries}
COUNTRY_CODES = {c.alpha_2.lower() for c in pycountry.countries}
COUNTRY_LOOKUP = COUNTRY_NAMES | COUNTRY_CODES

def normalize_company_name(name: str) -> str:
    """Return a cleaned, title-cased company name without legal suffixes."""
    if not isinstance(name, str) or not name.strip():
        return ""

    cleaned = name.lower().strip()

    cleaned = re.sub(r"[,/]", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    for pat in COMPANY_SUFFIX_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.I)
    cleaned = cleaned.strip()

    parts = []
    for w in cleaned.split():
        if w.isupper() and len(w) <= 4: 
            parts.append(w)
        else:
            parts.append(w.capitalize())

    return " ".join(parts)

def normalize_company(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column: company_norm
    Keeps the original column 'Company' untouched
    """
    df["company_norm"] = df["Company"].fillna("").map(normalize_company_name)
    return df

def smart_title_case(text: str) -> str:
    """Title-case while preserving acronyms like NLP, ML, QA."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return " ".join(
        w if (len(w) <= 3 and w.isupper()) else w.capitalize()
        for w in text.split()
    )

def process_job_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column: title_clean
    """
    titles = df["Title"].fillna("").astype(str)

    for pat, repl in TITLE_CLEAN_PATTERNS:
        titles = titles.str.replace(pat, repl, flags=re.I, regex=True)

    df["title_clean"] = titles.str.strip().map(smart_title_case)

    return df


def clean_locations(text: str) -> str:
    """
    Extracts GPE/LOC entities using spaCy, classifies which are countries
    via pycountry, and outputs: "City, Country | City, Country".
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    txt = text.replace("\n", " ").strip()

    pairs = []
    
    pattern = r"([^,]+?),\s*([A-Za-z][A-Za-z .'\-]{1,}?)(?=[,.;]|\s+and\b|\s+on\b|\s+with\b|$)"
    for m in re.finditer(pattern, txt):
        city = m.group(1).strip().strip(' .;,')
        
        city = re.sub(r"^(?:(?:and|on|in|at|site|client|clients)\b[\s,:\-]*)+", "", city, flags=re.I)
        country_token = m.group(2).strip().strip(' .;,')

        
        try:
            cobj = pycountry.countries.lookup(country_token)
            country_name = cobj.name
        except Exception:
            country_name = country_token

        pairs.append(f"{city}, {country_name}")

    if pairs:
        return " | ".join(dict.fromkeys(pairs))

    doc = Location_model(txt)
    ents = [ent.text.strip() for ent in doc.ents if ent.label_ in ("GPE", "LOC")]

    output = []
    i = 0
    while i < len(ents):
        ent = ents[i]

        is_country = False
        try:
            pycountry.countries.lookup(ent)
            is_country = True
        except Exception:
            is_country = False

        if not is_country:
            if i + 1 < len(ents):
                next_ent = ents[i + 1]
                try:
                    cobj = pycountry.countries.lookup(next_ent)
                    output.append(f"{ent}, {cobj.name}")
                    i += 2
                    continue
                except Exception:
                    pass
            output.append(ent)
            i += 1
        else:
            output.append(ent)
            i += 1

    return " | ".join(dict.fromkeys(output))

def process_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cleaned 'location_clean' column by removing newlines and extra spaces.
    """
    locations = df["Location"].fillna("").map(clean_locations)
    locations = locations.str.split(r"[\r\n]", n=1, regex=True).str[0]
    locations = locations.str.replace(r"\s{2,}", " ", regex=True).str.strip()
    df["location_clean"] = locations
    return df

def pattern_matching(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for pattern matching function to normalize skills.
    """
    with open("data/normalized_for_task4.json") as f: 
        CANON_SKILLS = sorted({s.lower() for s in json.load(f)["skills"]}) 
    
    def extract_and_normalize_skills(text): 
        if not isinstance(text, str): 
            return [] 
        text = text.lower() 
        found = [] 
        for skill in CANON_SKILLS: 
            pattern = r'\b' + re.escape(skill) + r'\b' 
            if re.search(pattern, text): 
                found.append(skill) 
                return list(set(found)) 
            
    df["skills_text"] = ( df[["JobDescription", "JobRequirment", "RequiredQual"]] .fillna("") .agg(" ".join, axis=1) ) 
    df["skills_norm"] = df["skills_text"].map(extract_and_normalize_skills)
    return df

# Testing the functions
if __name__ == "__main__":
    df = clean_csv()
    df = normalize_company(df)
    print(df[["Company", "company_norm"]].head())
    df = process_job_titles(df)
    print(df[["Title", "title_clean"]].head())
    df = process_locations(df)
    print(df['location_clean'].head())
    df = pattern_matching(df)
    print(df[['skills_norm']].head())