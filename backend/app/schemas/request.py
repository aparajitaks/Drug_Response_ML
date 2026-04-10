from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    drugName: str = Field(..., min_length=1)
    condition: str = Field(..., min_length=1)
    review: str = Field(..., min_length=1)
    usefulCount: int = Field(..., ge=0)
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    drugName: str = Field(..., min_length=1)
    condition: str = Field(..., min_length=1)
    review: str = Field(..., min_length=1)
    usefulCount: int = Field(..., ge=0)


class ReviewAnalysisRequest(BaseModel):
    review: str = Field(..., min_length=1)
