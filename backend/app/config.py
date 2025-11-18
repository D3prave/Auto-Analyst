import os
from pathlib import Path

# Base directory for backend
BASE_DIR = Path(__file__).resolve().parent.parent

# Redis Configuration
# Defaults to localhost for local dev, overridden by docker-compose to "redis"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
