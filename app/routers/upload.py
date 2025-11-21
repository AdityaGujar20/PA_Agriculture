from fastapi import APIRouter, UploadFile, File
from app.services.data_loader import save_upload_file


router = APIRouter()


@router.post("/")
async def upload_dataset(file: UploadFile = File(...)):
    path = save_upload_file(file)
    return {"status": "success", "file_path": path}