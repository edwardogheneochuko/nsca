# NSCA backend – neuro-symbolic disease identification (Pipeline B)

FastAPI backend that runs each patient through the three stages of the flow chart:

```
Stage 1  Symptom data (binary) ─► ESOINN (online clustering) ─► Model state
Stage 2  CART surrogate ─► Rule extraction ─┐
         Node difference ───────────────────┼─► Fused explanation record
         Exemplars (k-medoids) ─────────────┘
Stage 3  Grounding (HPO) ─► Candidates (top-k) ─► Consistency (exclusion rules) ─► Resolution
                                                                     ├─► Known disease
                                                                     └─► Flag as new
```

## Folder layout

| Path | Flow-chart box |
|---|---|
| `app/data/loaders.py` | Symptom data (binary encoded) |
| `app/stage1_neural/esoinn.py` | ESOINN + Model state (save/load) |
| `app/stage2_xai/surrogate.py` | CART surrogate (refit on trigger) |
| `app/stage2_xai/rules.py` | Rule extraction (tree paths to if-then) |
| `app/stage2_xai/node_difference.py` | Node difference (patient vs winner node) |
| `app/stage2_xai/exemplars.py` | Exemplars (k-medoids per cluster) |
| `app/stage2_xai/fusion.py` | Fused explanation record |
| `app/stage3_symbolic/grounding.py`, `hpo.py` | Grounding (HPO terms) |
| `app/stage3_symbolic/knowledge_base.py`, `candidates.py` | Candidates (top-k diseases) |
| `app/stage3_symbolic/consistency.py` | Consistency (exclusion rules) |
| `app/stage3_symbolic/resolution.py` | Resolution → Known disease / Flag as new |
| `app/engine.py` | Wires the stages together |
| `app/api/main.py` | FastAPI endpoints |
| `scripts/train.py`, `scripts/evaluate.py` | Offline training and evaluation |
| `models/<dataset>/` | Trained model state loaded by the server (~1 MB each) |
| `data/` | Datasets and the HPO release (`hp.obo`, 2026-09-01) |

## Run it

```bash
pip install -r requirements.txt
python -m scripts.train --dataset all        # ~1–2 min per dataset; writes models/
python -m scripts.evaluate --dataset all     # writes reports/evaluation_<dataset>.json
python -m pytest -q tests
uvicorn app.api.main:app --reload            # http://127.0.0.1:8000/docs
```

Render: start command `uvicorn app.api.main:app --host 0.0.0.0 --port $PORT`.
The server needs only `app/`, `models/` and `data/hpo/hp.obo`; the dataset CSVs are for training.
Environment variables: `NSCA_DEFAULT_DATASET` (`ddxplus` or `mendeley`), `CORS_ORIGINS` (comma-separated, default `*`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Loaded models |
| GET | `/model-status?dataset=` | ESOINN size, surrogate fidelity and pending refit triggers, HPO release, thresholds |
| GET | `/questions?dataset=` | Questionnaire for the app: questions, type (`yes_no`, `single`, `multi`, `scale`), answer options, parent question |
| GET | `/symptoms?dataset=&q=` | Flat symptom list with HPO ids |
| POST | `/parse-text` | Free text ("I have itching and feeling unwell") → questions to tick, negations, unrecognised parts |
| POST | `/identify` | Run a patient through all three stages |
| GET | `/clusters?dataset=` | Each ESOINN cluster with its name, size and strongest if-then rules |
| GET | `/flags?dataset=` | Patients flagged as possibly new this session |

`POST /identify` body:

```json
{"dataset": "ddxplus", "symptoms": ["E_91", "E_201", "sore throat"], "detail": "full"}
```

Symptoms can be column ids (`E_91`), plain names (`fever`) or HPO labels (`Cough`).

**DDXPlus in the app (default dataset).** Build the screens from `/questions`. Show a question that has a
`parent` only when the parent was answered yes. Send the `id` of every yes/no question answered yes and the
`value` of every option chosen. Unanswered follow-ups and travel are filled with DDXPlus's default answers
(`fill_defaults: true`, listed back in `filled_defaults`), which is how the training data records them.
On 300 known test patients entered this way, 97 % got the same result as with their full original record.
Response: `outcome` (`known_disease` | `flag_as_new`), `disease`, `confidence`
(`high` | `medium` | `low` | `provisional`), `unknown_symptoms`, a short `summary`, and with
`detail: "full"` every stage's output (`stage1_neural`, `stage2_fused_record`, `stage3_symbolic`).

**Free-text entry** (`app/text_parse.py`) is rule-based so it runs offline and every match is explainable:
HPO names and synonyms, an everyday / Nigerian-English phrase list ("catarrh", "stooling", "my body is hot"),
negation ("no sore throat" is recorded as absent), typo tolerance, and DDXPlus pain locations ("pain in my
stomach" ticks pain and selects "belly"). The app shows what it understood before running the check.

## Datasets

* **DDXPlus** (English, figshare doi:10.6084/m9.figshare.22687585, CC-BY). Validation split used
  for training (≤1,000 per pathology), test split for evaluation (≤300 per pathology).
  Held out as "new": acute otitis media, panic attack, pulmonary embolism.
* **Disease and symptoms dataset 2023** (Kaggle dhivyeshrk; Mendeley DOI 10.17632/2cxccsxydc.1).
  43-disease subset (≥200 unique patients, ≤5 % patterns shared with another disease).
  Held out: breast infection (mastitis), eye alignment disorder, lymphadenitis.

* **Phenopacket-store** (release 0.1.27; Danis et al., *HGG Advances* 2025, 6(1):100371;
  Zenodo 10.5281/zenodo.13168726; repository licence BSD-3-Clause). **Real cases from published case
  reports**, already coded in HPO. 39 diseases with ≥ 50 cases (4,125 cases), split 60/20/20.
  Held out: Cornelia de Lange syndrome 1, mitochondrial DNA depletion syndrome 6, pseudohypoparathyroidism Ia.
  Each case's HPO terms are expanded with their ancestors (below the top two HPO levels) so related
  phenotypes share features. Rebuild with `python -m scripts.build_phenopackets <release folder>`.

Held-out diseases are absent from ESOINN training and from the knowledge base.

## Design decisions and their reasons

* **Jaccard (Tanimoto) distance** instead of Euclidean, because patients are sparse binary vectors.
* **Threshold cap `t_max = 0.7`** (project addition to ESOINN). Without it, a node with no neighbours
  accepts any patient up to its distance to the nearest other node, which chained different diseases
  together (16 clusters for 40 diseases). With the cap: 37 clusters, ARI 0.85.
* **Noise buffer** keeps patients whose node was pruned and who are still outside every cluster in the
  final model, rather than discarding them.
* **Outlier rule** at inference: distance to winner above the 95th percentile of training distances in
  that cluster.
* **CART surrogate** grown to full depth; shallower trees reached only 0.33–0.52 fidelity.
* **Grounding**: exact HPO names/synonyms plus curated tables, all verified against `hp.obo`.
  Fuzzy text matching was tested and rejected ("decreased heart rate" → Tachycardia).
  Review and correct any row through `data/hpo/overrides_<dataset>.csv`.
* **Knowledge base** is learned from training patients of known diseases (not a clinical source); it is
  JSON so a curated clinical profile can replace it.
* **Consistency thresholds** are calibrated on validation patients of known diseases only.

## Results (test patients, single training run, seed 42)

| | Mendeley subset | DDXPlus |
|---|---|---|
| Known diseases / held out | 40 / 3 | 46 / 3 |
| Test patients (known + held out) | 2,406 + 888 | 13,336 + 900 |
| **Stage 1** clusters (true diseases) | 37 (40) | 45 (46) |
| ARI / NMI / purity (known test patients) | 0.858 / 0.957 / 0.887 | 0.881 / 0.962 / 0.898 |
| Outlier rate: known / held out | 4.8 % / 96.4 % | 6.5 % / 100 % |
| **Stage 2** surrogate fidelity (validation) | 0.958 | 0.991 |
| Records reliable: known / held out | 92.7 % / 3.0 % | 93.0 % / 0 % |
| **Stage 3** known patients diagnosed correctly | 94.2 % | 91.9 % |
| True disease among top-3 candidates | 99.2 % | 98.8 % |
| Known patients wrongly flagged as new | 4.6 % | 4.7 % |
| **Held-out patients flagged as new** | **99.1 %** | **85.8 %** |
| Top-3 candidates found in DDXPlus differential | – | 58.0 % |

Per held-out disease (DDXPlus): panic attack 100 %, pulmonary embolism 99.3 %, acute otitis media 58 %
(the rest were called acute rhinosinusitis or laryngitis, its nearest known neighbours).
Flagged cases of one held-out disease spread over 4–11 ESOINN clusters, so the flags do not yet group
into one "new disease" by themselves.

### Real data: phenopacket-store (published case reports)

| | Phenopackets |
|---|---|
| Known diseases / held out | 36 / 3 |
| Test patients (known + held out) | 799 + 200 |
| Stage 1 clusters (true diseases) | 46 (36) |
| ARI / NMI / purity | 0.373 / 0.724 / 0.705 |
| Baseline: agglomerative clustering given the true k | ARI 0.325 |
| Stage 2 surrogate fidelity | 0.829 (below 0.90 → every record marked unreliable) |
| Stage 3 known patients diagnosed correctly | 76.9 % |
| True disease among top-3 candidates | 89.5 % |
| Known patients wrongly flagged as new | 12.5 % |
| Held-out patients flagged as new | 88.5 % |

Real cases are far less tidy: two cases of the same disease share a median of only 26 % of their
features (Jaccard distance 0.74). ESOINN still beats a standard clustering method given the true number
of diseases. No tree setting brought surrogate fidelity above 0.83, so the reliability rule correctly
sent every decision to the symbolic checks alone (all outcomes "low" confidence).

Full numbers: `reports/evaluation_<dataset>.json`.

## Limitations

* DDXPlus and the Mendeley set are synthetic. Phenopacket-store is real but covers rare genetic diseases
  described in publications, not everyday presentations; hospital records are still needed for a clinical claim.
* The knowledge base comes from the same data generator as the patients, which flatters Stage 3.
* HPO's own disease annotations cover mostly rare diseases, so they are not used for candidates.
* ESOINN results vary with stream order (Mendeley ARI 0.71–0.85 over four orders); report several seeds.
* Flagged cases are provisional; no confirmation step is implemented (the flow chart ends at "Flag as new").
* DDXPlus rules and node differences are dominated by antecedent and "answer = No" columns
  (e.g. "traveled out of the country = N"), which makes them hard to read. Training on symptom columns
  only, or hiding "No" answers from explanations, would fix readability; not yet tested.
* Results are from one training run; repeat over several seeds before reporting.
