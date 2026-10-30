"""Stage 3 – Grounding ("HPO terms").

Every dataset column is mapped to a Human Phenotype Ontology term.

* DDXPlus columns are questions ("Do you have a cough?"), so a curated table names
  the intended HPO term for each symptom evidence.  Names are resolved against the
  loaded hp.obo, never hard-coded IDs, so a renamed or missing term is reported
  instead of silently mis-mapped.
* Mendeley columns are short phrases ("sharp chest pain"): an exact HPO name or
  exact synonym is accepted automatically; the rest use a curated table.  Fuzzy text
  matching proved unreliable ("decreased heart rate" -> Tachycardia), so it is only
  offered as a review suggestion for terms left ungrounded.
* Antecedents (risk factors such as smoking) are not phenotypes and stay ungrounded;
  Stage 3 uses them only as context.
* A reviewer can correct any row through data/hpo/overrides_<dataset>.csv
  (columns: feature, hpo_id).

The resulting table is saved for review: models/<dataset>/grounding.csv.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.config import DATA_DIR
from app.stage3_symbolic.hpo import HPO

# Intended HPO label(s) for each DDXPlus symptom evidence, in order of preference.
DDXPLUS_CURATED: dict[str, list[str]] = {
    "E_91": ["Fever"], "E_55": ["Pain"], "E_53": ["Pain"], "E_57": ["Pain"], "E_54": ["Pain"],
    "E_59": ["Pain"], "E_56": ["Pain"], "E_58": ["Pain"], "E_159": ["Syncope"],
    "E_133": ["Skin rash"], "E_129": ["Skin rash"], "E_130": ["Skin rash"], "E_134": ["Skin rash"],
    "E_132": ["Skin rash"], "E_136": ["Pruritus"], "E_135": ["Skin rash"], "E_131": ["Skin rash"],
    "E_154": ["Pallor"], "E_155": ["Palpitations"], "E_210": ["Hematemesis"], "E_140": ["Melena"],
    "E_51": ["Diarrhea"], "E_75": ["Choking episodes", "Dyspnea"], "E_89": ["Fatigue"],
    "E_114": ["Irritability"], "E_82": ["Presyncope", "Vertigo"], "E_148": ["Nausea"], "E_94": ["Chills"],
    "E_220": ["Pleuritic chest pain", "Chest pain"], "E_161": ["Decreased appetite", "Anorexia"],
    "E_179": ["Hematochezia"], "E_162": ["Weight loss"], "E_173": ["Heartburn", "Gastroesophageal reflux"],
    "E_33": ["Chest pain"], "E_218": ["Exercise intolerance"], "E_93": ["Paresthesia"], "E_66": ["Dyspnea"],
    "E_163": ["Abnormal vaginal discharge"], "E_30": ["Abdominal distention"], "E_127": ["Epiphora"],
    "E_181": ["Rhinorrhea"], "E_88": ["Fatigue"], "E_43": ["Seizure"],
    "E_156": ["Unilateral facial palsy", "Facial palsy"], "E_144": ["Myalgia"], "E_216": ["Pain"],
    "E_201": ["Cough"], "E_217": ["Orthopnea"], "E_215": ["Postprandial abdominal pain", "Abdominal pain"],
    "E_64": ["Exertional dyspnea"], "E_96": ["Weight gain", "Increased body weight"], "E_50": ["Hyperhidrosis"],
    "E_97": ["Sore throat", "Pharyngitis"], "E_9": ["Lymphadenopathy"], "E_76": ["Vertigo"],
    "E_65": ["Dysphagia"], "E_74": ["Conjunctival hyperemia", "Red eye"], "E_205": ["Trismus"],
    "E_63": ["Dysarthria"], "E_128": ["Laryngospasm", "Dyspnea"], "E_190": ["Sialorrhea", "Drooling"],
    "E_39": ["Confusion"], "E_212": ["Hoarse voice"], "E_206": ["Oral ulcer"], "E_52": ["Diplopia"],
    "E_203": ["Paroxysmal cough", "Cough"], "E_38": ["Jaw pain", "Pain"], "E_172": ["Ptosis"],
    "E_84": ["Generalized limb muscle weakness", "Limb muscle weakness"], "E_90": ["Fatigable weakness"],
    "E_211": ["Vomiting"], "E_166": ["Vomiting"], "E_112": ["Wheezing"],
    "E_178": ["Bruising susceptibility", "Abnormal bleeding"], "E_152": ["Edema"], "E_151": ["Edema"],
    "E_45": ["Hemoptysis"], "E_194": ["Inspiratory stridor", "Stridor"],
    "E_83": ["Facial muscle weakness", "Facial palsy"], "E_157": ["Perioral paresthesia", "Paresthesia"],
    "E_176": ["Hemiparesis", "Limb muscle weakness"], "E_214": ["Expiratory wheezing", "Wheezing"],
    "E_32": ["Decreased appetite", "Anorexia"], "E_202": ["Paroxysmal cough", "Cough"],
    "E_170": ["Ocular pruritus", "Pruritus"], "E_77": ["Productive cough"], "E_14": ["Chest pain"],
    "E_169": ["Pruritus"], "E_177": ["Paresthesia"], "E_175": ["Malaise"], "E_174": ["Weight loss"],
    "E_145": ["Menorrhagia"], "E_92": ["Flushing", "Facial erythema"], "E_192": ["Torticollis"],
    "E_188": ["Acholic stools", "Jaundice"], "E_164": ["Arrhythmia"], "E_193": ["Muscle spasm"],
    "E_168": ["Tongue thrusting"], "E_180": ["Oculogyric crisis", "Abnormal eye movement"],
    "E_67": ["Paroxysmal nocturnal dyspnea"], "E_171": ["Depersonalization"], "E_111": ["Anxiety"],
    "E_182": ["Purulent rhinorrhea", "Rhinorrhea"], "E_103": ["Anosmia"], "E_23": ["Sleep apnea"],
}
# Intended HPO label(s) for Mendeley symptom names that are not already exact HPO names/synonyms.
# A name with no fitting HPO term is listed with [] and stays a local term.
MENDELEY_CURATED: dict[str, list[str]] = {
    "dizziness": ["Vertigo"], "difficulty speaking": ["Dysarthria"], "blood in stool": ["Hematochezia"],
    "increased heart rate": ["Tachycardia"], "decreased heart rate": ["Bradycardia"], "weight gain": ["Increased body weight"],
    "difficulty in swallowing": ["Dysphagia"], "itching of skin": ["Pruritus"], "penis pain": ["Pain in penis"],
    "pain in eye": ["Ocular pain"], "eye burns or stings": ["Ocular pain"], "abnormal involuntary movements": ["Involuntary movements"],
    "mass on eyelid": ["Eyelid mass"], "regurgitation": ["Regurgitation", "Gastroesophageal reflux"],
    "regurgitation.1": ["Regurgitation", "Gastroesophageal reflux"], "eyelid swelling": ["Eyelid edema", "Periorbital edema"],
    "swollen eye": ["Periorbital edema", "Eyelid edema"], "leg swelling": ["Lower limb edema", "Edema of the lower limbs", "Edema"],
    "foot or toe swelling": ["Pedal edema"], "arm swelling": ["Upper limb edema", "Edema"], "skin swelling": ["Edema"],
    "upper abdominal pain": ["Epigastric pain", "Abdominal pain"], "lower abdominal pain": ["Lower abdominal pain", "Abdominal pain"],
    "sharp abdominal pain": ["Abdominal pain"], "suprapubic pain": ["Suprapubic pain", "Abdominal pain"],
    "loss of sensation": ["Sensory impairment", "Hypoesthesia"], "hand or finger swelling": ["Finger swelling"],
    "foot or toe pain": ["Foot pain"], "neck swelling": ["Neck swelling", "Anterior neck swelling"], "neck mass": ["Neck mass"],
    "irritable infant": ["Irritability"], "allergic reaction": ["Allergy"], "retention of urine": ["Urinary retention"],
    "lump or mass of breast": ["Breast mass"], "diaper rash": ["Skin rash"], "arm pain": ["Upper limb pain", "Arm pain"],
    "irregular belly button": ["Abnormal umbilicus morphology"], "hand or finger pain": ["Hand pain", "Finger pain"],
    "vaginal itching": ["Vulvovaginal pruritus", "Vulvar pruritus", "Pruritus"], "knee stiffness or tightness": ["Stiff knee", "Joint stiffness"],
    "hip stiffness or tightness": ["Stiff hip", "Joint stiffness"], "arm stiffness or tightness": ["Joint stiffness"],
    "foot or toe stiffness or tightness": ["Joint stiffness"], "back stiffness or tightness": ["Back stiffness", "Spinal rigidity"],
    "painful urination": ["Dysuria"], "abnormal breathing sounds": ["Abnormal breath sound", "Abnormal lung sounds"],
    "sharp chest pain": ["Chest pain"], "burning chest pain": ["Chest pain"], "rib pain": ["Rib pain", "Chest pain"],
    "hurts to breath": ["Pleuritic chest pain", "Chest pain"], "disturbance of memory": ["Memory impairment"],
    "eyelid lesion or rash": ["Abnormal eyelid morphology"], "stomach bloating": ["Abdominal distention"],
    "swollen abdomen": ["Abdominal distention"], "eye deviation": ["Strabismus"], "difficulty eating": ["Feeding difficulties"],
    "infant feeding problem": ["Feeding difficulties"], "mouth dryness": ["Xerostomia"], "mass in scrotum": ["Scrotal mass"],
    "temper problems": ["Temper tantrums", "Aggressive behavior"], "kidney mass": ["Renal mass", "Renal neoplasm"],
    "bleeding or discharge from nipple": ["Abnormal nipple discharge"], "skin lesion": ["Abnormal skin morphology"],
    "abnormal appearing skin": ["Abnormal skin morphology"], "skin growth": ["Skin nodule", "Neoplasm of the skin"],
    "lower body pain": ["Lower limb pain"], "coughing up sputum": ["Productive cough"], "symptoms of eye": ["Abnormality of the eye"],
    "knee swelling": ["Knee swelling", "Joint swelling"], "elbow swelling": ["Joint swelling"],
    "itchiness of eye": ["Ocular pruritus", "Pruritus"], "itchy scalp": ["Pruritus"], "diminished vision": ["Visual impairment", "Reduced visual acuity"],
    "bones are painful": ["Bone pain"], "diminished hearing": ["Hearing impairment"], "mouth pain": ["Oral pain", "Pain"],
    "problems with movement": ["Abnormality of movement", "Movement abnormality"], "anxiety and nervousness": ["Anxiety"],
    "excessive anger": ["Anger", "Aggressive behavior"], "involuntary urination": ["Urinary incontinence"], "side pain": ["Flank pain"],
    "facial pain": ["Facial pain", "Pain"], "lump in throat": ["Globus sensation", "Globus pharyngeus"],
    "throat feels tight": ["Globus sensation", "Globus pharyngeus"], "swollen or red tonsils": ["Enlarged tonsils", "Tonsillitis"],
    "pain or soreness of breast": ["Mastalgia"], "muscle cramps, contractures, or spasms": ["Muscle cramps", "Muscle spasm"],
    "leg cramps or spasms": ["Leg muscle cramps", "Muscle cramps"], "back cramps or spasms": ["Muscle spasm"],
    "lack of growth": ["Growth delay"], "groin mass": ["Inguinal mass", "Groin mass"], "acne or pimples": ["Acne"],
    "sinus congestion": ["Nasal congestion"], "hand or finger lump or mass": ["Finger mass", "Hand mass"],
    "swelling of scrotum": ["Scrotal swelling", "Scrotal edema"], "pain of the anus": ["Anal pain", "Proctalgia"],
    "feeling ill": ["Malaise"], "eye redness": ["Conjunctival hyperemia", "Red eye"], "eye moves abnormally": ["Abnormal eye movement"],
    "throat redness": ["Pharyngeal erythema", "Pharyngitis"], "tongue lesions": ["Abnormal tongue morphology", "Abnormality of the tongue"],
    "abusing alcohol": ["Alcoholism", "Alcohol dependence"], "symptoms of the kidneys": ["Abnormal renal morphology", "Abnormality of the kidney"],
    "heavy menstrual flow": ["Menorrhagia"], "penis redness": ["Penile erythema"], "unusual color or odor to urine": ["Abnormal urine color"],
    "skin dryness, peeling, scaliness, or roughness": ["Dry skin", "Xerosis"], "ache all over": ["Generalized pain", "Myalgia"],
    "depressive or psychotic symptoms": ["Depression", "Psychosis"], "symptoms of the scrotum and testes": ["Abnormality of the testis", "Abnormal testis morphology"],
    "redness in or around nose": ["Facial erythema"], "drug abuse": ["Substance abuse"],
    "irregular appearing nails": ["Abnormal nail morphology", "Nail dysplasia"], "irregular appearing scalp": ["Abnormality of the scalp", "Abnormal scalp morphology"],
    "spots or clouds in vision": ["Vitreous floaters", "Floaters"], "plugged feeling in ear": ["Aural fullness"],
    "early or late onset of menopause": [], "recent pregnancy": [], "problems during pregnancy": [], "smoking problems": [],
    "redness in ear": [], "pulling at ears": [], "postpartum problems of the breast": [], "back mass or lump": [],
    "wrist lump or mass": [], "knee lump or mass": [], "skin on arm or hand looks infected": [], "skin on leg or foot looks infected": [],
}
# Symptom evidences that describe context, not a phenotype: left ungrounded on purpose.
DDXPLUS_NOT_PHENOTYPE = {"E_42", "E_150", "E_221", "E_219", "E_13"}

_NORM = re.compile(r"[^a-z0-9 ]+")


def norm(s: str) -> str:
    return " ".join(_NORM.sub(" ", s.lower()).split())


class Matcher:
    def __init__(self, hpo: HPO):
        self.hpo = hpo
        terms = hpo.phenotype_terms()
        self.exact: dict[str, str] = {}
        rows, ids = [], []
        for tid, t in terms.items():
            self.exact.setdefault(norm(t["name"]), tid)
            rows.append(t["name"]); ids.append(tid)
            for syn, scope in t["synonyms"]:
                if scope == "EXACT":
                    self.exact.setdefault(norm(syn), tid)
                rows.append(syn); ids.append(tid)
        self.ids = np.array(ids)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), preprocessor=norm, sublinear_tf=True)
        self.M = self.vec.fit_transform(rows)

    def by_label(self, label: str) -> str | None:
        return self.exact.get(norm(label))

    def fuzzy(self, text: str) -> tuple[str | None, float]:
        q = self.vec.transform([text])
        sims = linear_kernel(q, self.M).ravel()
        k = int(np.argmax(sims))
        return (str(self.ids[k]), float(sims[k])) if sims[k] > 0 else (None, 0.0)


def build_grounding(dataset: str, features: list[str], meanings: list[str], is_antecedent, hpo: HPO) -> pd.DataFrame:
    m = Matcher(hpo)
    rows = []
    for f, mean, ante in zip(features, meanings, is_antecedent):
        base = f.split("=")[0]
        hid, method, score = None, "no fitting HPO term", 0.0
        if ante:
            method = "antecedent (not a phenotype)"
        elif dataset == "phenopackets":
            hid, method, score = (f, "native HPO (from case report)", 1.0) if f in hpo.terms else (None, "no fitting HPO term", 0.0)
        elif dataset == "ddxplus":
            if base in DDXPLUS_NOT_PHENOTYPE:
                method = "context (not a phenotype)"
            else:
                for lab in DDXPLUS_CURATED.get(base, []):
                    hid = m.by_label(lab)
                    if hid:
                        method, score = "curated", 1.0
                        break
                if hid is None:
                    method = "no fitting HPO term"
        else:
            hid = m.by_label(mean)
            if hid:
                method, score = "exact", 1.0
            elif mean in MENDELEY_CURATED:
                for lab in MENDELEY_CURATED[mean]:
                    hid = m.by_label(lab)
                    if hid:
                        method, score = "curated", 1.0
                        break
                if hid is None:
                    method = "no fitting HPO term"
        sug, sug_score = m.fuzzy(mean) if (not ante and not hid) else (None, 0.0)
        rows.append({"feature": f, "meaning": mean, "hpo_id": hid, "hpo_label": hpo.name(hid) if hid else None,
                     "method": method, "review_suggestion": hpo.name(sug) if (sug and not hid) else None,
                     "suggestion_id": sug if (sug and not hid) else None, "suggestion_score": round(sug_score, 3) if not hid else None})
    df = pd.DataFrame(rows)
    ov = DATA_DIR / "hpo" / f"overrides_{dataset}.csv"
    if ov.exists():
        o = pd.read_csv(ov).set_index("feature")["hpo_id"]
        for i, f in enumerate(df.feature):
            if f in o.index:
                hid = o[f] if isinstance(o[f], str) and o[f] else None
                df.loc[i, ["hpo_id", "hpo_label", "method"]] = [hid, hpo.name(hid) if hid else None, "override"]
    # Ungrounded symptoms keep a local id so reasoning can still use them.
    df["term"] = [h if isinstance(h, str) and h else f"LOCAL:{f}" for h, f in zip(df.hpo_id, df.feature)]
    return df


def save_grounding(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
