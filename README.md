# Automated Medical Report Interpretation System (Echocardiography)

End-to-end system that **extracts structured measurements from echocardiography PDF reports** and generates **guideline-based clinical interpretations**, with optional **machine-learning augmentation**, **explainability**, **robustness analysis**, **severity grading**, and **risk stratification**.

> Educational/research project. Outputs are decision-support only and are **not** a medical diagnosis.

---

## What this project does

**Input**: PDF echo report (or JSON measurements)  
**Output**: Structured measurements + human-readable clinical interpretations + optional analytics dashboards.

Pipeline:

```
PDF/JSON → Extraction → Normalization/Validation → Rule Engine → (Optional ML overlay)
       → Severity grading / Risk stratification / Explainability / Sensitivity
       → API/CLI/Web UI outputs
```

---

## Publishable contributions (what you implemented)

1. **PDF information extraction** for echo reports using `pdfplumber` + regex patterns, with table merge and measurement prioritization.  
   - Code: `src/extractor.py`, `src/utils.py`
2. **Guideline-based clinical interpretation engine** (ASE/EACVI-inspired ranges) with **age/sex adjusted thresholds** and severity text generation.  
   - Code: `src/rule_engine.py`
3. **Hybrid inference layer** that always computes rule-based interpretations and **optionally overlays ML predictions** when trained models exist; also tracks **per-category source** (Rule vs ML).  
   - Code: `src/predictor.py`
4. **ML training pipeline** producing a scaler + metadata + per-category models (and scripts for training/testing/comparison).  
   - Code: `src/model_trainer.py`, `train_interpretation_model.py`, `predict_with_ml.py`, `compare_models.py`
5. **Explainable AI tooling**: SHAP summaries/waterfall/dependence + PDP/ICE + feature-importance plots for trained models.  
   - Code: `src/explainability.py`
6. **Robustness + uncertainty quantification**: one-at-a-time perturbation tests, Monte Carlo simulation for measurement error propagation, and global sensitivity analysis.  
   - Code: `src/sensitivity_analysis.py`
7. **Multi-class severity grading** beyond binary outputs (e.g., diastolic dysfunction grading, LVH grading, valvular grading) + severity dashboards.  
   - Code: `src/severity_grading.py`
8. **Clinical risk stratification**: composite cardiovascular risk, heart-failure risk, mortality risk + risk dashboards and recommendations.  
   - Code: `src/risk_stratification.py`
9. **Production interfaces**: Flask REST API + CLI + React frontend, including CORS configuration and containerized deployment.  
   - Code: `src/api.py`, `main.py`, `frontend-react/`, `docker-compose.yml`
10. **Optional LLM narrative generation (Gemini)** for clinician/patient summaries and report Q&A chat, gated by environment flags.  
   - Code: `src/llm_narrator.py`, API routes in `src/api.py`

---

## Repository structure

```
medical_interpreter/
├── main.py                      # CLI entry point
├── src/
│   ├── api.py                   # Flask REST API
│   ├── extractor.py             # PDF extraction
│   ├── utils.py                 # cleaning/normalization/validation helpers
│   ├── rule_engine.py           # guideline-based interpretations
│   ├── predictor.py             # hybrid rule+ML inference
│   ├── model_trainer.py         # ML training utilities
│   ├── explainability.py        # SHAP, PDP/ICE, feature importance
│   ├── sensitivity_analysis.py  # OAT + Monte Carlo + global sensitivity
│   ├── severity_grading.py      # multi-class grading + dashboards
│   ├── risk_stratification.py   # risk scores + dashboards
│   └── llm_narrator.py          # optional Gemini summaries/chat
├── data/                        # PDFs + processed JSON
├── models/                      # trained model artifacts
├── outputs/                     # generated plots/dashboards
├── docs/                        # documentation (quickstart, ML, deployment)
└── frontend-react/              # Vite + React UI
```

---

## Quickstart (local)

### 1) Python setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional (full explainability features):

```bash
pip install shap scipy
```

### 2) CLI usage

```bash
# Extract measurements from a PDF
python main.py extract path/to/report.pdf -o extracted.json

# Interpret (from PDF directly)
python main.py interpret path/to/report.pdf -o interpretation.json

# Interpret (from extracted JSON)
python main.py interpret extracted.json

# Train models (uses real data if available; can generate synthetic otherwise)
python main.py train -d data/processed/ -o models/

# Batch process a folder of PDFs
python main.py batch data/sample_reports/ -o data/processed/
```

### 3) Run the REST API

```bash
python src/api.py
# API: http://localhost:5000
```

### 4) Run the React frontend

```bash
cd frontend-react
npm install
npm run dev
# UI: http://localhost:5173
```

Set backend URL (optional): create `frontend-react/.env.local`

```
VITE_API_BASE=http://localhost:5000
```

---

## API endpoints (backend)

Backend file: `src/api.py`

Common endpoints:

- `GET  /health` — health check
- `POST /api/interpret` — upload PDF (multipart) → interpretation JSON
- `POST /api/interpret/json` — JSON input → interpretation JSON
- `GET  /api/parameters` — supported parameters
- `POST /api/batch` — batch upload PDFs
- `GET  /api/model-metrics` — metrics + confusion matrix (when models exist)

Advanced analytics:

- `POST /api/explainability`
- `POST /api/feature-importance`
- `POST /api/sensitivity-analysis`
- `POST /api/risk-stratification`

LLM (optional):

- `POST /api/narrative` — clinician/patient narratives via Gemini
- `POST /api/chat` — Q&A chat grounded in report context

---

## ML training, artifacts, and evaluation

Docs: `docs/ML_GUIDE.md`

Typical workflow:

```bash
python prepare_training_data.py
python train_interpretation_model.py
python predict_with_ml.py
python compare_models.py
```

### Trained ML models (what’s in this repo)

- **Number of models**: **5** parallel classifiers (one per interpretation category)
- **Categories** (from `models/model_metadata.json`):
  - `LV_FUNCTION`
  - `LV_SIZE`
  - `LV_HYPERTROPHY`
  - `LA_SIZE`
  - `DIASTOLIC_FUNCTION`
- **Feature vector**: 14 features (`age`, `sex` + 12 echo measurements; see `models/model_metadata.json`)

### Model performance

From `MODEL_COMPARISON_OUTPUT.txt` (Version 2 “Expanded Model”, test set **n=325**):

- **Average test accuracy**: **0.981**
- **Macro F1**: **0.978**

Per-category test accuracy (V2):

| Category | Test Accuracy |
|---|---:|
| LV_FUNCTION | 1.000 |
| LV_SIZE | 1.000 |
| LV_HYPERTROPHY | 0.984 |
| LA_SIZE | 1.000 |
| DIASTOLIC_FUNCTION | 0.921 |

(For comparison, Version 1 average test accuracy reported as 0.938 in the same report.)

### Model artifacts

Model artifacts written under `models/` (examples):

- `scaler.pkl`
- `model_*.pkl` (per-category classifiers)
- `model_metadata.json` (feature names, categories, parameters)

---

## Deployment

Docs: `docs/DEPLOYMENT.md`

### Docker

```bash
docker-compose up --build
# http://localhost:5000
```

### Render / Railway / Split deployment
See `render.yaml`, `render.backend.yaml`, and `vercel.json` (frontend).

---

## Configuration (environment variables)

Backend:

- `CORS_ORIGINS` — comma-separated allowed origins for the API
- `MAX_CONTENT_LENGTH` — upload size limit (bytes)

Gemini narration (optional):

- `LLM_ENABLED=true|false`
- `LLM_PROVIDER=gemini`
- `GEMINI_API_KEY=...`
- `GEMINI_MODEL=gemini-2.5-flash` (default)
- `LLM_TIMEOUT_SECONDS=25` (default)

---

## Notes on data, ethics, and safety

- Do not upload sensitive patient-identifying data to public deployments.
- Interpretations are generated automatically and may be incorrect; they are for research/decision support only.

---

## Documentation

- `docs/QUICKSTART.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/ML_GUIDE.md`
- `docs/DEPLOYMENT.md`
- `ARCHITECTURE.md`
- `ADVANCED_FEATURES_GUIDE.md`

---

## Citation (placeholder)

If you publish, consider adding your IEEE citation here.

```bibtex
@misc{medical_interpreter_2026,
  title        = {Automated Medical Report Interpretation System (Echocardiography)},
  author       = {Your Name},
  year         = {2026},
  howpublished = {GitHub repository}
}
```
