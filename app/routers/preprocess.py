from fastapi import APIRouter
from app.services.preprocess_service import handle_missing_values


router = APIRouter()


from app.services.data_loader import UPLOAD_DIR

def get_latest_uploaded_file():
    files = sorted(UPLOAD_DIR.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None

@router.post("/missing")
def missing_value_handler(strategy: str = "mean"):
    file_path = get_latest_uploaded_file()
    if not file_path:
        return {"error": "No uploaded file found."}
    new_path = handle_missing_values(file_path, strategy)
    return {"processed_file": new_path}
