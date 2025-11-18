import redis
import pandas as pd
import io
import json
import logging
from app.config import REDIS_URL

# Initialize Redis client
r = redis.from_url(REDIS_URL)

logger = logging.getLogger("uvicorn")

# --- Dataset Management ---

def save_dataset(dataset_id: str, df: pd.DataFrame):
    """Serialize dataframe to parquet and save to Redis."""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    r.set(f"dataset:{dataset_id}:data", buffer.getvalue())
    r.expire(f"dataset:{dataset_id}:data", 86400) # 24h expiration

def get_dataset(dataset_id: str) -> pd.DataFrame:
    """Retrieve dataset from Redis."""
    data = r.get(f"dataset:{dataset_id}:data")
    if not data:
        raise FileNotFoundError("Dataset not found in cache")
    
    buffer = io.BytesIO(data)
    return pd.read_parquet(buffer)

def save_metadata(dataset_id: str, meta: dict):
    """Save metadata (columns, shape, overrides) to Redis."""
    r.set(f"dataset:{dataset_id}:meta", json.dumps(meta))
    r.expire(f"dataset:{dataset_id}:meta", 86400)

def get_metadata(dataset_id: str) -> dict:
    raw = r.get(f"dataset:{dataset_id}:meta")
    if not raw:
        return {}
    return json.loads(raw)

# --- Image/Plot Management ---

def save_plot(dataset_id: str, filename: str, image_bytes: bytes):
    """Save a plot image to Redis."""
    key = f"dataset:{dataset_id}:plot:{filename}"
    r.set(key, image_bytes)
    r.expire(key, 86400)

def get_plot(dataset_id: str, filename: str) -> bytes:
    """Retrieve a plot image from Redis."""
    key = f"dataset:{dataset_id}:plot:{filename}"
    data = r.get(key)
    if not data:
        logger.warning(f"Plot NOT FOUND in Redis: {key}")
        return None
    return data