from fastapi import APIRouter
from app.services.model_service import train_model
from pathlib import Path

router = APIRouter()

PROCESSED_DIR = Path("data/processed")

def get_latest_processed_file():
    files = sorted(PROCESSED_DIR.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None

@router.post("/")
def train(target: str = "yield_kg_per_ha", test_size: float = 0.2):
    file_path = get_latest_processed_file()
    if not file_path:
        return {"error": "No processed file found. Run preprocess first."}
    metrics = train_model(file_path, target, test_size)
    return metrics
