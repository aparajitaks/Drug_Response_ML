# Drug Response ML

Interactive dashboard and API for predicting drug response categories from patient review data. Upload a CSV, get predictions with confidence scores, explore charts, and download results. Everything is path-safe (uses `pathlib`) so the project keeps working even if you move the folder.

---

## Problem & Goal

Given basic drug review metadata (drug name, condition, rating, and helpfulness votes), predict the patient's response category (Responder / Non-Responder / Neutral-Mixed). The project ships a trained scikit-learn model, a Streamlit dashboard for analysts, and an optional FastAPI endpoint for programmatic use.

---

## Dataset

- Demo CSV: `ml-service/data/demo_sample.csv` (also downloadable from the dashboard sidebar). Columns include:
	- `uniqueID`: row identifier (not used for prediction)
	- `drugName`: medication name (feature)
	- `condition`: condition being treated (feature)
	- `review`: free-text review (not used by the shipped model)
	- `rating`: numeric user rating 1–10 (feature)
	- `date`: review date (not used by the shipped model)
	- `usefulCount`: helpfulness votes (feature)
	- `response_category`: ground-truth class (0/1/2) for reference
- Feature schema is explicitly defined in `ml-service/feature_schema.json`:
	- `features = ["drugName", "condition", "rating", "usefulCount"]`
- Label mapping in `ml-service/label_mapping.json`:
	- `0 → Non-Responder`
	- `1 → Responder`
	- `2 → Neutral / Mixed Response`

You can swap in your own data as long as these feature columns are present. Extra columns are ignored.

---

## Model

- Stored at `ml-service/models/drug_response_model.pkl` (loaded via `joblib`).
- Trained scikit-learn classification pipeline that ingests the four schema-defined features.
- Supports `predict_proba`, enabling confidence scores in the dashboard.
- Paths are resolved relative to `ml-service` by default, but can be overridden with env vars (`MODEL_PATH`, `LABEL_PATH`, `SCHEMA_PATH`, `DEMO_CSV_PATH`) using absolute or relative paths.

---

## Features

- CSV upload with preview
- Demo dataset download and one-click load
- Schema-driven feature selection with missing-column checks
- Predictions with optional confidence scores
- Label decoding via JSON mapping
- Plotly visualizations (prediction distribution, confidence histogram)
- Top confident cases table
- Downloadable predictions CSV
- Optional FastAPI `/predict` endpoint for programmatic inference

---

## Tech Stack

- Python 3.11+
- Streamlit (dashboard)
- FastAPI + Uvicorn (API)
- scikit-learn, pandas, numpy
- joblib (model serialization)
- plotly (charts)
- python-dotenv (env loading)

---

## Run locally (dashboard)

```bash
git clone https://github.com/aparajitaks/Drug_Response_ML.git
cd Drug_Response_ML/ml-service

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python -m streamlit run dashboard.py
```

Open http://localhost:8501

Environment overrides (optional): set any of `MODEL_PATH`, `LABEL_PATH`, `SCHEMA_PATH`, `DEMO_CSV_PATH` to absolute paths or paths relative to `ml-service`.

### One-liner quickstart (dashboard)

```bash
git clone https://github.com/aparajitaks/Drug_Response_ML.git && \
cd Drug_Response_ML/ml-service && \
python -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
python -m streamlit run dashboard.py
```

If you need a different port (e.g., 8765), add `--server.port 8765` to the Streamlit command.

---

## Run locally (API)

```bash
cd Drug_Response_ML/ml-service
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

- Health check: `GET /`
- Prediction: `POST /predict` with JSON body

```json
{
	"drugName": "cialis",
	"condition": "benign prostatic hyperplasia",
	"rating": 8,
	"usefulCount": 12
}
```

---

## How to demo quickly

1) Run the dashboard (steps above).
2) In the sidebar, click **Download Sample CSV** or **Use Demo Dataset** to load `demo_sample.csv` automatically.
3) View preview, run predictions, inspect charts, and download the annotated CSV.

---

## Project layout

```
Drug_Response_ML/
└── ml-service/
		├── app.py                 # FastAPI inference service
		├── dashboard.py           # Streamlit UI
		├── feature_schema.json    # Feature ordering/selection
		├── label_mapping.json     # Class → human label mapping
		├── requirements.txt       # Python deps
		├── data/
		│   └── demo_sample.csv    # Demo dataset (downloadable)
		└── models/
				└── drug_response_model.pkl  # Trained classifier
```

---

## Notes

- Paths use `pathlib` so moving the project folder won't break file resolution; env vars can still point to custom locations.
- Text reviews are present in the demo CSV but are not consumed by the shipped model.
- For VS Code, pick the interpreter at `ml-service/venv/bin/python` to avoid missing-import warnings.
