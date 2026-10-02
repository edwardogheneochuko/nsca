"""Load the disease-symptom data into a binary symptom matrix (Stage 0 baseline)."""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"

def clean(s):
    return str(s).strip().replace(" ", "_").replace("__", "_")

def load():
    raw = pd.read_csv(DATA / "dataset.csv")
    sym_cols = [c for c in raw.columns if c.startswith("Symptom_")]
    raw["Disease"] = raw["Disease"].str.strip()
    for c in sym_cols:
        raw[c] = raw[c].where(raw[c].notna(), None).map(lambda v: clean(v) if v else None)
    symptoms = sorted({s for c in sym_cols for s in raw[c].dropna()})
    X = pd.DataFrame(0, index=raw.index, columns=symptoms, dtype="uint8")
    for c in sym_cols:
        for i, s in raw[c].dropna().items():
            X.at[i, s] = 1
    y = raw["Disease"]
    desc = pd.read_csv(DATA / "symptom_Description.csv").set_index("Disease")["Description"]
    prec = pd.read_csv(DATA / "symptom_precaution.csv").set_index("Disease")
    desc.index = desc.index.str.strip(); prec.index = prec.index.str.strip()
    return X, y, desc, prec

if __name__ == "__main__":
    X, y, desc, prec = load()
    print("patients:", X.shape[0], "| symptoms:", X.shape[1], "| diseases:", y.nunique())
    print("unique symptom patterns:", X.drop_duplicates().shape[0])
    print(y.value_counts().head(5))
    X.assign(Disease=y).to_csv("data/symptom_matrix.csv", index=False)
