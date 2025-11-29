import re

def preprocess(term: str) -> list[str]:
    t = (term or "").strip().lower()
    t = re.sub(r'[\u200b]', '', t)
    t = re.sub(r'[-_/]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()


    t = re.sub(r'(c\+\+)\d{2}', r'\1', t)
    t = re.sub(r'(python)\s*3(\.\d+)?', r'\1', t)


    toks = [x for x in t.split(' ') if x]
    cands = [' '.join(toks)] if toks else []
    if len(toks) > 1:
        cands.extend(toks)

    seen, out = set(), []
    for c in cands:
        if c not in seen:
            out.append(c); seen.add(c)
    return out