# nsca-backend

Backend for NSCA (Neuro-Symbolic Clustering Architecture), an MSc research prototype.
**Not a clinically validated diagnostic tool.** Dataset associations are not diagnoses.

## Layout
```
api.py              FastAPI routes
engine.py           trains ESOINN on the loaded dataset; identification and held-out evaluation
esoinn.py           ESOINN implementation (see its docstring for interpretive choices)
dataset_loader.py   load, validate, clean, preprocess; consistent feature vectors
extractor.py        rule-based symptom extraction against the dataset's own symptom names
schemas.py          request/response models (frontend types must mirror these)
data/original/      put disease_symptoms.csv here (never modified by the code)
data/processed/     generated: cases.csv, features.npz, feature_names.json, manifest.json
tests/
```

## 1. Dataset (required, not included)
Place a real CSV at `data/original/disease_symptoms.csv` with columns
`Disease, Symptom_1, Symptom_2, ...` (blank cells allowed). A common starting point is the Kaggle
"Disease Symptom Prediction" dataset. Before use, verify provenance, licence, label quality and
suitability, then record them here (and commit the CSV only if the licence permits redistribution):

- Source: _TODO_
- Licence: _TODO_
- Retrieved on: _TODO_

The server refuses to start the model, and the API answers 503 with the reason, if the file is
missing or invalid. Cleaning drops rows with no disease or no symptoms and removes exact duplicate
cases; counts of both are shown in `/model/status`.

## 2. Run
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload
```
Check `http://localhost:8000/model/status` first: it shows whether the dataset loaded, the number of
training rows, diseases and symptom features, rows dropped, duplicates removed, conflicting patterns
(identical symptoms under different diseases), ESOINN node/cluster counts, and a held-out evaluation.

Environment variables: `NSCA_DATASET_PATH`, `NSCA_PROCESSED_DIR`, `NSCA_CORS_ORIGINS` (comma-separated).

## Endpoints
- `GET /health`
- `GET /model/status`
- `POST /symptoms/extract` `{description}`
- `POST /identify` `{description, duration?, severity?, symptoms?}` returns one outcome:
  `known_pattern` (one disease association plus evidence), `potentially_novel_pattern`
  (flagged for human review; not a claim of a new disease) or `inconclusive`.
  No ranking and no list of competing diseases is ever returned.

Feedback storage (`storage.py`, `/feedback`) is not built yet.

## Test
```
pytest
```
Tests use generated placeholder data (`tests/fixtures.py`), never real disease data.

## Known limitations
- The held-out evaluation uses full symptom vectors from the same dataset. It does not measure
  performance on partial or free-text descriptions, and it is not clinical validation.
- Symptom extraction only matches the dataset's own symptom names as phrases. "Skin rash" is found,
  "I have a rash" is not.
- Users who report only some of a disease's symptoms may land outside a cluster's similarity range and be
  flagged `potentially_novel_pattern`. Evaluate this on your real dataset before relying on it.
- Thresholds in `EngineConfig` and `ESOINNConfig` are untuned defaults.
- `duration` and `severity` are echoed back and severity only changes the urgent-care text; the dataset has no such fields.

## Deploy (Render)
Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`. Set `NSCA_CORS_ORIGINS` to the deployed web
frontend origin(s). The processed folder is written best-effort; a read-only filesystem does not stop startup.
