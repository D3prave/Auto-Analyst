import redis
import pandas as pd
import io
import json
import logging
from app.config import REDIS_URL

r = redis.from_url(REDIS_URL)
logger = logging.getLogger("uvicorn")

def save_dataset(dataset_id: str, df: pd.DataFrame):
    """Serialize dataframe to parquet and store in Redis with 24h expiration."""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    r.set(f"dataset:{dataset_id}:data", buffer.getvalue())
    r.expire(f"dataset:{dataset_id}:data", 86400)

def get_dataset(dataset_id: str) -> pd.DataFrame:
    data = r.get(f"dataset:{dataset_id}:data")
    if not data:
        raise FileNotFoundError("Dataset not found in cache")
    return pd.read_parquet(io.BytesIO(data))

def save_metadata(dataset_id: str, meta: dict):
    r.set(f"dataset:{dataset_id}:meta", json.dumps(meta))
    r.expire(f"dataset:{dataset_id}:meta", 86400)

def get_metadata(dataset_id: str) -> dict:
    raw = r.get(f"dataset:{dataset_id}:meta")
    return json.loads(raw) if raw else {}

def save_plot(dataset_id: str, filename: str, image_bytes: bytes):
    key = f"dataset:{dataset_id}:plot:{filename}"
    r.set(key, image_bytes)
    r.expire(key, 86400)

def get_plot(dataset_id: str, filename: str) -> bytes:
    key = f"dataset:{dataset_id}:plot:{filename}"
    return r.get(key)