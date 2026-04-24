import re
from typing import Any, Dict, List

# Split NER skill spans that bundle many technologies (comma / slash lists).
_SPLIT_RE = re.compile(r"[,;|，、]|(?:\s+/\s+)")
# Section titles mistaken as skills, e.g. "Programming Languages:"
_HEADER_RE = re.compile(r"^[a-z][a-z0-9\s]{0,40}:$", re.I)
_NOISE = frozenset(
    {
        "skills",
        "skill",
        "tools",
        "tool",
        "cr",
        "sk",
    }
)


def raw_skills_from_entities(merged: List[Dict[str, Any]]) -> List[str]:
    """Return stripped skill strings from merged NER spans.

    Supports both:
    - {"label": "Skills", ...}
    - {"entity": "B-Skills" / "I-Skills", ...}
    """
    out: List[str] = []
    for ent in merged:
        label = ent.get("label")
        if not label:
            entity = ent.get("entity")
            if isinstance(entity, str) and "-" in entity:
                _, _, suffix = entity.partition("-")
                label = suffix
            elif isinstance(entity, str):
                label = entity
        if str(label).strip().lower() != "skills":
            continue
        text = (ent.get("text") or "").strip()
        if text:
            out.append(text)
    return out


def expand_skill_fragments(raw_skills: List[str]) -> List[str]:
    """
    Split comma-/semicolon-separated skill blobs from NER into single-skill phrases.

    NER often predicts one ``Skills`` span for ``Python, C, SQL``; the normalizer
    matches whole strings poorly against the vocab. Splitting recovers coverage.
    """
    out: List[str] = []
    for raw in raw_skills:
        text = (raw or "").strip()
        if not text:
            continue
        low = text.lower()
        if low in _NOISE:
            continue
        if _HEADER_RE.match(text):
            continue
        parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
        chunks = parts if len(parts) > 1 else [text]
        for chunk in chunks:
            chunk = re.sub(r"^[\s·•\uf0b7\-]+", "", chunk)
            chunk = chunk.strip()
            if not chunk:
                continue
            if len(chunk) < 2 and chunk.lower() not in ("c", "r"):
                continue
            if chunk.lower() in _NOISE:
                continue
            if _HEADER_RE.match(chunk):
                continue
            if len(chunk) > 140:
                cut = chunk.split(".")[0].split("")[0].split("•")[0].strip()
                chunk = cut if cut and len(cut) < len(chunk) else chunk[:120].rsplit(" ", 1)[0]
            if len(chunk) < 2 and chunk.lower() not in ("c", "r"):
                continue
            out.append(chunk)
    return out


def skills_for_normalizer(merged: List[Dict[str, Any]]) -> List[str]:
    """Skills strings to pass to ``SkillNormalizer.normalize`` (NER → split phrases)."""
    return expand_skill_fragments(raw_skills_from_entities(merged))
