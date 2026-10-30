"""Human Phenotype Ontology loader (hp.obo, release recorded in `data_version`)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.config import HPO_OBO

PHENOTYPIC_ABNORMALITY = "HP:0000118"


class HPO:
    def __init__(self, path: Path = HPO_OBO):
        self.terms: dict[str, dict] = {}
        self.data_version = None
        cur = None
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.startswith("data-version:"):
                self.data_version = line.split(":", 1)[1].strip()
            if line == "[Term]":
                cur = {"synonyms": [], "is_a": [], "obsolete": False}
                continue
            if line.startswith("[") and line != "[Term]":
                cur = None
                continue
            if cur is None or ":" not in line:
                continue
            key, val = line.split(":", 1)
            val = val.strip()
            if key == "id":
                cur["id"] = val
                self.terms[val] = cur
            elif key == "name":
                cur["name"] = val
            elif key == "synonym":
                text = val.split('"')[1]
                scope = val.split('"')[2].split()[0] if len(val.split('"')) > 2 else ""
                cur["synonyms"].append((text, scope))
            elif key == "is_a":
                cur["is_a"].append(val.split("!")[0].strip())
            elif key == "is_obsolete" and val == "true":
                cur["obsolete"] = True
        self.terms = {k: v for k, v in self.terms.items() if not v["obsolete"] and "name" in v}
        self.children: dict[str, list[str]] = {}
        for t, v in self.terms.items():
            for p in v["is_a"]:
                self.children.setdefault(p, []).append(t)

    def ancestors(self, term: str) -> set[str]:
        out, stack = set(), [term]
        while stack:
            for p in self.terms.get(stack.pop(), {}).get("is_a", []):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    def descendants(self, term: str) -> set[str]:
        out, stack = set(), [term]
        while stack:
            for c in self.children.get(stack.pop(), []):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out

    def name(self, term: str) -> str:
        return self.terms.get(term, {}).get("name", term)

    def phenotype_terms(self) -> dict[str, dict]:
        keep = self.descendants(PHENOTYPIC_ABNORMALITY)
        return {k: self.terms[k] for k in keep}


@lru_cache(maxsize=1)
def get_hpo() -> HPO:
    return HPO()
