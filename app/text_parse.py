"""Free-text symptom entry: "I have itching and feeling unwell" -> questionnaire answers.

Rule-based on purpose (offline, fast, every match explainable):
  1. split the sentence into clauses ("," "and" "but" "with" ...);
  2. normalise words (lower case, simple stemming, typo correction against the vocabulary);
  3. match phrases, longest first, from three sources:
       - HPO names and synonyms of the terms each question is grounded to,
       - an everyday / Nigerian-English lexicon mapped to HPO labels ("catarrh", "stooling", "body is hot"),
       - the dataset's own symptom names;
  4. a negation cue earlier in the clause ("no", "not", "without", "don't") marks the symptom as absent;
  5. DDXPlus records pain as one question plus a body location, so pain phrases
     ("headache", "stomach pain") tick the pain question and pre-select the location
     only when exactly one location option fits.
Nothing is ticked silently: the app shows what was understood before the check runs.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

STOP = {"i", "im", "i'm", "ive", "i've", "have", "has", "had", "having", "a", "an", "the", "my", "me", "am", "is",
        "are", "was", "were", "been", "be", "feel", "feeling", "felt", "get", "getting", "got", "some", "very",
        "really", "so", "too", "of", "in", "on", "at", "to", "for", "it", "its", "this", "that", "since", "from",
        "lot", "lots", "bit", "little", "kind", "sort", "also", "just", "always", "sometimes", "now", "today",
        "yesterday", "days", "day", "week", "weeks", "and", "or", "like", "quite", "severe", "mild", "bad", "much"}
NEG = {"no", "not", "never", "without", "dont", "don't", "doesnt", "doesn't", "havent", "haven't", "didnt",
       "didn't", "denies", "deny", "none", "nor", "neither", "free"}
SPLIT = re.compile(r"[,.;!?\n]+|\b(?:and|but|with|also|plus|then)\b", re.I)

# Everyday phrases -> HPO label (first label found in the loaded HPO is used).
LAY: dict[str, list[str]] = {
    "fever": ["Fever"], "temperature": ["Fever"], "high temperature": ["Fever"], "hot body": ["Fever"],
    "body hot": ["Fever"], "feverish": ["Fever"],
    "cough": ["Cough"], "coughing": ["Cough"],
    "itch": ["Pruritus"], "itching": ["Pruritus"], "itchy": ["Pruritus"], "scratching": ["Pruritus"],
    "itchy nose": ["Pruritus"], "itchy throat": ["Pruritus"], "itching nose": ["Pruritus"],
    "unwell": ["Malaise"], "not well": ["Malaise"], "ill": ["Malaise"], "sick": ["Malaise"],
    "under the weather": ["Malaise"], "general weakness": ["Malaise"], "body weakness": ["Malaise"],
    "tired": ["Fatigue"], "tiredness": ["Fatigue"], "exhausted": ["Fatigue"], "weak": ["Fatigue"],
    "weakness": ["Fatigue"], "fatigue": ["Fatigue"], "no energy": ["Fatigue"],
    "vomit": ["Vomiting"], "vomiting": ["Vomiting"], "throwing up": ["Vomiting"], "throw up": ["Vomiting"],
    "nausea": ["Nausea"], "nauseous": ["Nausea"], "feel like vomiting": ["Nausea"],
    "diarrhea": ["Diarrhea"], "diarrhoea": ["Diarrhea"], "stooling": ["Diarrhea"], "running stomach": ["Diarrhea"],
    "loose stool": ["Diarrhea"], "watery stool": ["Diarrhea"],
    "runny nose": ["Rhinorrhea"], "running nose": ["Rhinorrhea"], "catarrh": ["Rhinorrhea"],
    "blocked nose": ["Nasal congestion", "Rhinorrhea"], "stuffy nose": ["Nasal congestion", "Rhinorrhea"],
    "shortness of breath": ["Dyspnea"], "short of breath": ["Dyspnea"], "breathless": ["Dyspnea"],
    "difficulty breathing": ["Dyspnea"], "hard to breathe": ["Dyspnea"], "cant breathe": ["Dyspnea"],
    "sore throat": ["Pharyngalgia", "Sore throat"], "throat pain": ["Pharyngalgia"], "painful throat": ["Pharyngalgia"],
    "dizzy": ["Vertigo"], "dizziness": ["Vertigo"], "lightheaded": ["Presyncope", "Vertigo"],
    "rash": ["Skin rash"], "skin rash": ["Skin rash"], "spots on skin": ["Skin rash"], "rashes": ["Skin rash"],
    "body pain": ["Myalgia"], "body ache": ["Myalgia"], "muscle pain": ["Myalgia"], "aching all over": ["Myalgia"],
    "sweating": ["Hyperhidrosis"], "sweat": ["Hyperhidrosis"],
    "palpitations": ["Palpitations"], "heart racing": ["Palpitations"], "heart beating fast": ["Palpitations"],
    "chills": ["Chills"], "shivering": ["Chills"], "shivers": ["Chills"], "feeling cold": ["Chills"],
    "weight loss": ["Weight loss"], "losing weight": ["Weight loss"],
    "loss of appetite": ["Poor appetite", "Decreased appetite"], "no appetite": ["Poor appetite", "Decreased appetite"],
    "not eating": ["Poor appetite"],
    "coughing blood": ["Hemoptysis"], "cough blood": ["Hemoptysis"], "blood in sputum": ["Hemoptysis"],
    "wheeze": ["Wheezing"], "wheezing": ["Wheezing"],
    "swelling": ["Edema"], "swollen": ["Edema"],
    "swollen glands": ["Lymphadenopathy"], "swollen lymph nodes": ["Lymphadenopathy"],
    "difficulty swallowing": ["Dysphagia"], "hard to swallow": ["Dysphagia"], "pain swallowing": ["Dysphagia"],
    "hoarse": ["Hoarse voice"], "lost my voice": ["Hoarse voice"],
    "confused": ["Confusion"], "confusion": ["Confusion"],
    "fainted": ["Syncope"], "passed out": ["Syncope"], "fainting": ["Syncope"],
    "seizure": ["Seizure"], "convulsion": ["Seizure"], "fits": ["Seizure"],
    "watery eyes": ["Epiphora"], "teary eyes": ["Epiphora"], "red eyes": ["Conjunctival hyperemia"],
    "itchy eyes": ["Ocular pruritus"], "loss of smell": ["Anosmia"], "cant smell": ["Anosmia"],
    "phlegm": ["Productive cough"], "sputum": ["Productive cough"], "mucus": ["Productive cough"],
    "mucus from nose": ["Rhinorrhea"], "nasal discharge": ["Rhinorrhea"], "green mucus nose": ["Rhinorrhea"],
    "yellow mucus nose": ["Rhinorrhea"], "greenish nose": ["Rhinorrhea"], "coloured nasal discharge": ["Rhinorrhea"],
    "heartburn": ["Gastroesophageal reflux"], "bloated": ["Abdominal distention"], "bloating": ["Abdominal distention"],
    "anxious": ["Anxiety"], "anxiety": ["Anxiety"],
    "headache": ["Headache"], "chest pain": ["Chest pain"], "stomach pain": ["Abdominal pain"],
    "stomach ache": ["Abdominal pain"], "belly pain": ["Abdominal pain"], "abdominal pain": ["Abdominal pain"],
    "back pain": ["Back pain"], "joint pain": ["Arthralgia"], "ear pain": ["Ear pain"],
    "numbness": ["Paresthesia"], "tingling": ["Paresthesia"], "pins and needles": ["Paresthesia"],
}

# DDXPlus pain locations: everyday body word -> words that identify the location option(s).
PAIN_WORDS = {"pain", "ache", "aching", "hurt", "hurts", "hurting", "painful", "sore", "headache", "stomachache",
              "bellyache", "earache", "toothache"}
BODY = {"head": ["forehead", "top of the head", "back of head", "temple", "occiput"], "headache": ["forehead", "top of the head", "back of head", "temple", "occiput"],
        "forehead": ["forehead"], "temple": ["temple"], "stomach": ["belly"], "belly": ["belly"], "tummy": ["belly"],
        "abdomen": ["belly"], "abdominal": ["belly"], "stomachache": ["belly"], "bellyache": ["belly"],
        "upper stomach": ["epigastric"], "chest": ["upper chest", "lower chest"], "throat": ["pharynx"],
        "ear": ["ear"], "earache": ["ear"], "eye": ["eye"], "nose": ["nose"], "mouth": ["mouth"],
        "neck": ["back of the neck", "side of the neck"], "jaw": ["under the jaw"], "knee": ["knee"],
        "hip": ["hip"], "shoulder": ["shoulder"], "cheek": ["cheek"], "thigh": ["thigh"], "calf": ["calf"],
        "ankle": ["ankle"], "breast": ["breast"], "testicle": ["testicle"], "side": ["flank", "side of the chest"]}
# DDXPlus has no plain anxiety question (its "Anxiety"-grounded question is fear of dying), so only these phrases reach it.
DDX_DIRECT = {"fear of dying": "E_111", "afraid of dying": "E_111", "going to die": "E_111", "feel like dying": "E_111"}
DDX_SKIP_CONCEPTS = {"Anxiety"}
GENERAL_SITE = {"skin", "body", "general", "all"}
# When several DDXPlus questions share a concept, the general one wins unless the text names something more specific.
DDX_PRIMARY = {"Vomiting": "E_211", "Cough": "E_201", "Rhinorrhea": "E_181", "Pruritus": "E_136", "Edema": "E_151",
               "Skin rash": "E_129", "Paresthesia": "E_177", "Weight loss": "E_162", "Poor appetite": "E_32",
               "Wheezing": "E_112", "Fatigue": "E_89", "Chest pain": "E_53"}
DDX_PAIN_HPO = {"Pain", "Headache", "Chest pain", "Abdominal pain", "Back pain", "Ear pain", "Arthralgia"}


def stem(w: str) -> str:
    w = w.lower().strip("'")
    for a, b in (("iness", "y"), ("ness", ""), ("ies", "y")):
        if w.endswith(a) and len(w) > len(a) + 2:
            w = w[: -len(a)] + b
            break
    if w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    if w.endswith("y") and len(w) > 4 and w[-2] not in "aeiou":
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]                       # runn -> run, itchh -> itch
    return w


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def content(ws: list[str]) -> list[str]:
    return [stem(w) for w in ws if w not in STOP and w not in NEG]


@dataclass
class Phrase:
    toks: tuple[str, ...]
    text: str
    target: str            # question id
    concept: str
    source: str


@dataclass
class Parser:
    phrases: list[Phrase] = field(default_factory=list)
    vocab: set[str] = field(default_factory=set)
    pain_q: str | None = None
    location_q: str | None = None
    location_opts: list[dict] = field(default_factory=list)
    qtext: dict[str, str] = field(default_factory=dict)
    qconcept: dict[str, str] = field(default_factory=dict)
    parent: dict[str, str | None] = field(default_factory=dict)

    @classmethod
    def build(cls, engine) -> "Parser":
        hpo = engine.hpo
        qs = engine.questions
        g = engine.grounding.set_index("feature")
        p = cls()
        p.qtext = {q["id"]: q["text"] for q in qs}
        p.parent = {q["id"]: q["parent"] for q in qs}
        ddx = engine.dataset == "ddxplus"
        by_hpo: dict[str, list[str]] = {}
        for q in qs:
            if q["kind"] != "symptom":
                continue
            feat = q["id"] if q["id"] in g.index else (q["options"][0]["value"] if q["options"] else None)
            hid = g.loc[feat, "hpo_id"] if feat in g.index else None
            if isinstance(hid, str):
                by_hpo.setdefault(hid, []).append(q["id"])
                p.qconcept[q["id"]] = hpo.name(hid)
        if ddx:
            p.pain_q = "E_53"
            loc = next((q for q in qs if q["id"] == "E_55"), None)
            p.location_q, p.location_opts = ("E_55", loc["options"]) if loc else (None, [])

        def add(text: str, target: str, concept: str, source: str):
            t = tuple(content(tokens(text)))
            if t:
                p.phrases.append(Phrase(t, text, target, concept, source))
                p.vocab.update(t)

        def concept_words(hid: str) -> str:
            t = hpo.terms.get(hid, {})
            return " ".join([t.get("name", "")] + [s for s, _ in t.get("synonyms", [])])

        def best_question(cands: list[str], text: str, context: str = "", concept: str = "") -> str:
            """Most word overlap (phrase first, then the concept's HPO synonyms); then fewest extra
            (more specific) words; then a general site; then a top-level question."""
            tt = set(content(tokens(text)))
            ct = set(content(tokens(context)))
            def score(c):
                qt = set(content(tokens(p.qtext[c])))
                primary = ddx and DDX_PRIMARY.get(concept) == c
                return (len(tt & qt), primary, len(ct & qt), -len(qt - tt), bool(qt & GENERAL_SITE), p.parent[c] is None)
            return max(cands, key=score)

        # 1-2: HPO names/synonyms and everyday phrases, both routed through HPO terms
        for hid, cands in by_hpo.items():
            name = hpo.name(hid)
            if ddx and (name in DDX_PAIN_HPO or name in DDX_SKIP_CONCEPTS):
                continue                                    # pain is handled with its location
            names = [name] + [s for s, scope in hpo.terms.get(hid, {}).get("synonyms", []) if scope in ("EXACT", "RELATED")]
            for n in names:
                add(n, best_question(cands, n, concept_words(hid), name), name, "HPO")
        label_to_id = {}
        for hid in by_hpo:
            label_to_id[hpo.name(hid).lower()] = hid
        for phrase, labels in LAY.items():
            for lab in labels:
                hid = label_to_id.get(lab.lower())
                if ddx and lab in DDX_PAIN_HPO:
                    add(phrase, p.pain_q, "Pain", "everyday phrase")
                    break
                if ddx and lab in DDX_SKIP_CONCEPTS:
                    break
                if hid:
                    add(phrase, best_question(by_hpo[hid], phrase, concept_words(hid), hpo.name(hid)), hpo.name(hid), "everyday phrase")
                    break
        if ddx:
            for phrase, q in DDX_DIRECT.items():
                add(phrase, q, "Anxiety (fear of dying)", "everyday phrase")
        # 3: the dataset's own symptom names (Mendeley, phenopackets) or short question labels
        if not ddx:
            for q in qs:
                if q["kind"] == "symptom":
                    add(q["text"], q["id"], q["text"], "dataset symptom name")
        p.phrases.sort(key=lambda ph: -len(ph.toks))
        return p

    # ------------------------------------------------------------------ parse
    def _fix(self, w: str) -> str:
        if w in self.vocab or len(w) < 4:
            return w
        m = difflib.get_close_matches(w, self.vocab, 1, 0.8)
        return m[0] if m else w

    def parse(self, text: str) -> dict:
        found, absent, unrecognised, answers = [], [], [], {}
        tick: list[str] = []
        for clause in [c.strip() for c in SPLIT.split(text or "") if c and c.strip()]:
            raw = tokens(clause)
            seq = [(i, self._fix(stem(w))) for i, w in enumerate(raw) if w not in STOP and w not in NEG]
            neg_pos = [i for i, w in enumerate(raw) if w in NEG]
            used: set[int] = set()
            hits = []
            for ph in self.phrases:
                idx = []
                for t in ph.toks:
                    j = next((i for i, s in seq if s == t and i not in used and i not in idx), None)
                    if j is None:
                        break
                    idx.append(j)
                else:
                    used.update(idx)
                    hits.append((ph, min(idx)))
            # DDXPlus: any remaining pain word -> pain question
            if self.pain_q and not any(h[0].target == self.pain_q for h in hits):
                pw = next((i for i, w in enumerate(raw) if w in PAIN_WORDS and i not in used
                           and not (w == "sore" and "throat" in raw)), None)
                if pw is not None:
                    used.add(pw)
                    hits.append((Phrase((), raw[pw], self.pain_q, "Pain", "pain word"), pw))
            for ph, pos in hits:
                negated = any(n < pos for n in neg_pos)
                q = ph.target
                item = {"phrase": ph.text, "concept": ph.concept, "question_id": q,
                        "question": self.qtext.get(q, q), "source": ph.source, "note": None}
                if negated:
                    absent.append(item)
                    continue
                parent = self.parent.get(q)
                if parent:                                  # a follow-up: tick its parent question
                    if self.qconcept.get(q) != self.qconcept.get(parent):
                        item["note"] = f"then answer the follow-up: {self.qtext.get(q, q)}"
                    q = parent
                    item["question_id"], item["question"] = q, self.qtext.get(q, q)
                if q == self.pain_q and self.location_q:
                    loc = self._locations(raw)
                    if len(loc) == 1:
                        answers.setdefault(self.location_q, [])
                        if loc[0]["value"] not in answers[self.location_q]:
                            answers[self.location_q].append(loc[0]["value"])
                        item["note"] = f"pain location: {loc[0]['label']}"
                    elif loc:
                        item["note"] = "choose where: " + ", ".join(o["label"] for o in loc[:6])
                    else:
                        item["note"] = "choose where the pain is in the follow-up question"
                if q not in tick:
                    tick.append(q)
                    found.append(item)
                elif item["note"]:                          # same question again: keep the extra detail visible
                    prev = next(f for f in found if f["question_id"] in (q, ph.target))
                    prev["phrase"] += f", {ph.text}"
                    prev["note"] = "; ".join(x for x in (prev["note"], item["note"]) if x)
            leftover = [raw[i] for i, s in seq if i not in used and len(raw[i]) > 2]
            if leftover and not hits:
                unrecognised.append(clause)
        return {"symptoms": tick, "answers": answers, "found": found, "absent": absent,
                "unrecognised": unrecognised}

    def _locations(self, raw: list[str]) -> list[dict]:
        words = " ".join(raw)
        keys = [k for k in BODY if re.search(rf"\b{k}\b", words)]
        if "upper stomach" in keys:
            keys = ["upper stomach"]
        out = []
        for k in keys:
            for frag in BODY[k]:
                out += [o for o in self.location_opts if frag in o["label"] and o not in out]
        return out
