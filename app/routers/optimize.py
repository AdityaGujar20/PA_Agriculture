from fastapi import APIRouter
from app.services.optimization_service import optimize_inputs

router = APIRouter()

@router.post("/")
def optimize(base_features: dict):
    """
    base_features = all features EXCEPT fertilizer, irrigation, pesticide
    """
    result = optimize_inputs(base_features)
    return result
