from typing import List, Dict, Tuple
import json
from rapidfuzz import fuzz, process
from .rules import preprocess

class SkillNormalizer:
    def __init__(self, vocab_path: str, alias_path: str,
                 fuzzy_cutoff: int = 90, min_conf: float = 0.76):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        with open(alias_path, 'r', encoding='utf-8') as f:
            alias_map = json.load(f)

        self.vocab: Dict[str, Dict] = {k.lower(): v for k, v in vocab.items()}
        self.canon_list = list(self.vocab.keys())
        self.alias_rev = {}
        for can, aliases in alias_map.items():
            for a in aliases:
                self.alias_rev[a.lower()] = can.lower()

        self.fuzzy_cutoff = fuzzy_cutoff
        self.min_conf = min_conf

    def _alias_map(self, t: str):
        can = self.alias_rev.get(t)
        if can:
            return {"canonical": can, "confidence": 0.98, "method": "alias"}
        return None

    def _fuzzy_map(self, t: str):
        m = process.extractOne(
            t, self.canon_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.fuzzy_cutoff
        )
        if m:
            return {"canonical": m[0], "confidence": m[1]/100.0, "method": "fuzzy"}
        return None

    def normalize_one(self, raw: str, context: Dict = None) -> Dict:
        best = {"canonical": None, "from": raw, "confidence": 0.0, "method": None, "notes": ""}
        for cand in preprocess(raw):

            hit = self._alias_map(cand)
            if hit:
                best.update(hit)
                best["from"] = raw
                return best

            hit = self._fuzzy_map(cand)
            if hit and hit["confidence"] > best["confidence"]:
                best.update(hit); best["from"] = raw

        if best["canonical"] == "c" and context:
            sec = (context.get("section") or "").lower()
            neigh = " ".join(context.get("neighbors", [])).lower()
            if sec == "education" and any(k in neigh for k in ["grade", "gpa", "score"]):
                return {"canonical": None, "from": raw, "confidence": 0.0, "method": "reject", "notes": "likely grade"}

        if best["confidence"] >= self.min_conf:
            return best
        return {"canonical": None, "from": raw, "confidence": 0.0, "method": "reject", "notes": "low_conf"}

    def normalize(self, terms: List[str], contexts: List[Dict] = None):
        contexts = contexts or [{}]*len(terms)
        out, audit = [], []
        best_by_can = {}
        for t, ctx in zip(terms, contexts):
            res = self.normalize_one(t, ctx)
            if res["canonical"]:

                k = res["canonical"]
                if k not in best_by_can or res["confidence"] > best_by_can[k]["confidence"]:
                    best_by_can[k] = res
            else:
                audit.append({"from": res["from"], "decision": "reject", "reason": res.get("notes","")})
        return list(best_by_can.values()), audit