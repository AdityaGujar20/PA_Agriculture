import shutil
from pathlib import Path


UPLOAD_DIR = Path("data/uploads")


def save_upload_file(upload_file):
    dest = UPLOAD_DIR / upload_file.filename
    with dest.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return str(dest)