# © 2026 Андрій Кобізь
#Моделі відповідей API 
from typing import List, Optional

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    # class - зарезервоване слово в Python, тому поле class_, а в JSON "class"
    class_: str = Field(alias="class")
    confidence: float

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_predictions: List[ClassScore]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class HistoryItem(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    created_at: str
    image_url: Optional[str] = None
