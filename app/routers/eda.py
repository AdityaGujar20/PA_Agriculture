from fastapi import APIRouter
from app.services.eda_service import generate_eda_summary, generate_eda_plots
from app.services.data_loader import UPLOAD_DIR

router = APIRouter()

def get_latest_uploaded_file():
    files = sorted(
        UPLOAD_DIR.glob("*.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if not files:
        return None
    return str(files[0])

@router.get("/summary")
def eda_summary():
    file_path = get_latest_uploaded_file()
    if not file_path:
        return {"error": "No uploaded file found."}
    return generate_eda_summary(file_path)

@router.get("/plots")
def eda_plots():
    file_path = get_latest_uploaded_file()
    if not file_path:
        return {"error": "No uploaded file found."}
    return generate_eda_plots(file_path)
