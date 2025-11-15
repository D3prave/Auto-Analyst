from pathlib import Path
from typing import Union
import shutil

from ..config import DATA_DIR


def get_dataset_path(dataset_id: str) -> Path:
    """Return the full path for a stored dataset CSV."""
    return DATA_DIR / f"{dataset_id}.csv"


def save_temp_csv(dataset_id: str, file_obj) -> Path:
    """
    Save an uploaded CSV (UploadFile from FastAPI) to disk.
    Returns the path.
    """
    dest_path = get_dataset_path(dataset_id)
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file_obj.file, buffer)
    return dest_path