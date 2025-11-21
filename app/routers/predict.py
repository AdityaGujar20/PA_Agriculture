from fastapi import APIRouter
from app.schemas.prediction_schema import PredictionInput
from app.services.prediction_service import predict_values

router = APIRouter()

@router.post("/")
def make_prediction(input_data: PredictionInput):
    return predict_values(input_data.dict())
