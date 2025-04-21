from pydantic import BaseModel
from typing import List, Dict

class LabelInput(BaseModel):
    topic: str
    labeled_data: Dict[str, bool]  # {text: True/False}

class RecommendationResponse(BaseModel):
    topic: str
    recommendations: List[str]
