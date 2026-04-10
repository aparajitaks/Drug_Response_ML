import logging

from fastapi import APIRouter, HTTPException

from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse
from app.services.predictor import run_prediction


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        return PredictionResponse(**run_prediction(payload))
    except Exception as exc:
        logger.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
