import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Default to localhost for local dev, but docker-compose overrides this to "redis"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")