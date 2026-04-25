from fastapi import APIRouter
from pydantic import BaseModel
from services.recommendation import get_recommendations

router = APIRouter()


class AnalysisInput(BaseModel):
    gender: str
    skin_tone: str
    face_shape: str


@router.post("/recommend")
async def recommend(data: AnalysisInput):
    return get_recommendations(data.gender, data.skin_tone, data.face_shape)
