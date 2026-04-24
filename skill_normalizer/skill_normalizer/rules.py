import re
from typing import List


def _strip_edge_punct(s: str) -> str:
    """Trim commas / bullets so ``sql,`` can match vocab ``sql``."""
    return re.sub(r"^[\s,;.:·•|'\"]+|[\s,;.:·•|'\"]+$", "", s)


def preprocess(term: str) -> List[str]:
    t = (term or "").strip().lower()
    t = re.sub(r'[\u200b]', '', t)
    t = re.sub(r'[-_/]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = _strip_edge_punct(t)

    t = re.sub(r'(c\+\+)\d{2}', r'\1', t)
    t = re.sub(r'(python)\s*3(\.\d+)?', r'\1', t)

    toks = [_strip_edge_punct(x) for x in t.split(" ") if _strip_edge_punct(x)]
    cands = [' '.join(toks)] if toks else []
    if len(toks) > 1:
        cands.extend(toks)

    seen, out = set(), []
    for c in cands:
        if c not in seen:
            out.append(c); seen.add(c)
    return out