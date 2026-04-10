import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse
from app.services.model_loader import get_model


logger = logging.getLogger(__name__)
router = APIRouter()
DEFAULT_LABEL_MAPPING = {"0": "Non-Responder", "1": "Responder", "2": "Neutral/Mixed"}


def _load_label_mapping() -> dict[str, str]:
    mapping_path = Path(__file__).resolve().parents[3] / "ml-service" / "label_mapping.json"
    try:
        with mapping_path.open("r", encoding="utf-8") as file_obj:
            mapping = json.load(file_obj)
            if isinstance(mapping, dict):
                return {str(k): str(v) for k, v in mapping.items()}
    except Exception:
        logger.warning("Could not load label mapping. Using defaults.")
    return DEFAULT_LABEL_MAPPING


@router.post("", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
        label_mapping = _load_label_mapping()

        input_df = pd.DataFrame(
            [
                {
                    "review": payload.review,
                    "condition": payload.condition,
                    "usefulCount": payload.usefulCount,
                }
            ]
        )

        prediction_class = int(model.predict(input_df)[0])
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(input_df)[0]))
        prediction_label = label_mapping.get(str(prediction_class), "Unknown")

        return PredictionResponse(
            prediction_class=prediction_class,
            prediction_label=prediction_label,
            confidence=confidence,
        )
    except Exception as exc:
        logger.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse
from app.services.model_loader import get_model


logger = logging.getLogger(__name__)
router = APIRouter()
LABEL_MAPPING = {
    "0": "Non-Responder",
    "1": "Responder",
    "2": "Neutral/Mixed",
}


def _load_label_mapping() -> dict[str, str]:
    label_path = Path(__file__).resolve().parents[3] / "ml-service" / "label_mapping.json"
    try:
        import json

        with label_path.open("r", encoding="utf-8") as file_obj:
            mapping = json.load(file_obj)
            if isinstance(mapping, dict):
                return {str(k): str(v) for k, v in mapping.items()}
    except Exception:
        logger.warning("Could not load label mapping file. Using defaults.")
    return LABEL_MAPPING


@router.post("", response_model=PredictionResponse, summary="Predict drug review score")
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
        label_mapping = _load_label_mapping()

        input_df = pd.DataFrame(
            [
                {
                    "review": payload.review,
                    "condition": payload.condition,
                    "usefulCount": payload.usefulCount,
                }
            ]
        )

        prediction_class = int(model.predict(input_df)[0])
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(input_df)[0]))

        prediction_label = label_mapping.get(str(prediction_class), "Unknown")
        return PredictionResponse(
            prediction_class=prediction_class,
            prediction_label=prediction_label,
            confidence=confidence,
        )
    except Exception as exc:
        logger.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
