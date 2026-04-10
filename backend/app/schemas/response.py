from pydantic import BaseModel


class ShapExplanationItem(BaseModel):
    feature: str
    direction: str
    impact: str


class PredictionResponse(BaseModel):
    prediction_class: int
    prediction_label: str
    confidence: float
    shap_explanation: list[ShapExplanationItem]
