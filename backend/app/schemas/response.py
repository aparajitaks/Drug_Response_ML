from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction_class: int
    prediction_label: str
    confidence: float
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction_class: int
    prediction_label: str
    confidence: float


class ReviewAnalysisResponse(BaseModel):
    sentiment: str
    side_effects: list[str]
    benefits: list[str]
    summary: str
