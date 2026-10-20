# NSCA Disease Identification Backend (prototype)

FastAPI research backend for a shared web/mobile client. This is a software scaffold, not a clinically validated diagnostic system. Its bundled symptom vocabulary and broad viral-like demonstration rule are placeholders; they are not SNOMED CT or ICD-11 data and must not be used for medical decisions.

## Windows setup (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Tests

```powershell
python -m pytest -v
```

## Example request

```powershell
$body = @{
  description = "I have fever, cough, sore throat and body aches"
  symptom_duration = "3 days"
  severity = "moderate"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/identify" -Method Post -ContentType "application/json" -Body $body
```

## Important limitations

- The included ESOINN-style incremental graph implementation is a prototype, not a validated reference implementation of ESOINN.
- A single patient input is not sufficient to train a medically meaningful clustering model. Train/evaluate on appropriately curated and ethically sourced data before interpreting clusters.
- No disease codes are fabricated. SNOMED CT and ICD-11 integration must use authorized data/services and traceable mappings.
- A potentially novel pattern is a review flag, not evidence that a new disease exists.
- Do not use this prototype to make health decisions. For urgent or worsening symptoms, seek qualified medical care.
- SQLite stores case responses locally; configure privacy, retention, access controls, and secure deployment before handling real patient data.
