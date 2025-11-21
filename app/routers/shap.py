from fastapi import APIRouter
from app.services.shap_service import shap_summary_plot, shap_bar_plot

router = APIRouter()


@router.get("/summary")
def shap_summary():
    return {"image": shap_summary_plot()}


@router.get("/bar")
def shap_bar():
    return {"image": shap_bar_plot()}
