from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import os

app = FastAPI(title="Drug Response ML Inference API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "drug_response_model.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "models", "..", "label_mapping.json")
SCHEMA_PATH = os.path.join(BASE_DIR, "models", "..", "feature_schema.json")

model = joblib.load(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    label_mapping = json.load(f)

with open(SCHEMA_PATH, "r") as f:
    feature_schema = json.load(f)

class PredictionRequest(BaseModel):
    drugName: str
    condition: str
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
        "usefulCount": data.usefulCount
    }])

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
