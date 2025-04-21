from fastapi import APIRouter, HTTPException
from app.recommender import RecommenderEngine
from app.schema import LabelInput, RecommendationResponse

router = APIRouter()
engine = RecommenderEngine()

@router.post("/recommend", response_model=RecommendationResponse)
def recommend_topic(payload: LabelInput):
    if not payload.labeled_data:
        raise HTTPException(status_code=400, detail="Labeled data cannot be empty.")
    try:
        return engine.recommend(payload.topic, payload.labeled_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
