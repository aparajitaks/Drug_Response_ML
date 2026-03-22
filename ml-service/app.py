import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

load_dotenv()

app = FastAPI(title="Drug Response ML Inference API")

BASE_DIR = Path(__file__).resolve().parent


def resolve_path(env_value: str | None, default_relative: str) -> Path:
    """Return a path that works if env is absolute or relative to BASE_DIR."""
    if env_value:
        candidate = Path(env_value)
        return candidate if candidate.is_absolute() else BASE_DIR / candidate
    return BASE_DIR / default_relative


MODEL_PATH = resolve_path(os.getenv("MODEL_PATH"), "models/drug_response_model.pkl")
LABEL_PATH = resolve_path(os.getenv("LABEL_PATH"), "label_mapping.json")
SCHEMA_PATH = resolve_path(os.getenv("SCHEMA_PATH"), "feature_schema.json")

allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in allowed_origins else allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = joblib.load(MODEL_PATH)

with LABEL_PATH.open("r") as f:
    label_mapping = json.load(f)

with SCHEMA_PATH.open("r") as f:
    feature_schema = json.load(f)

class PredictionRequest(BaseModel):
    drugName: str
    condition: str
    rating: float
    usefulCount: int

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Drug Response ML API is running"}

@app.post("/predict")
def predict(data: PredictionRequest):
    # Convert input to DataFrame (important for sklearn Pipeline)
    input_df = pd.DataFrame([{
        "drugName": data.drugName,
        "condition": data.condition,
        "rating": data.rating,
        "usefulCount": data.usefulCount
    }])

    if feature_schema and "features" in feature_schema:
        missing = [c for c in feature_schema["features"] if c not in input_df.columns]
        if missing:
            return {"error": f"Missing required features: {missing}"}
        input_df = input_df[feature_schema["features"]]

    # Run prediction
    pred_class = int(model.predict(input_df)[0])

    # Optional: probability (if supported)
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(input_df)[0]))

    return {
        "prediction_class": pred_class,
        "prediction_label": label_mapping[str(pred_class)],
        "confidence": confidence
    }
