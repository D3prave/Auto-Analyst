from pathlib import Path

# Base directory for backend
BASE_DIR = Path(__file__).resolve().parent.parent

# Where uploaded / temp datasets live
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Where plots/reports will live later
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plots directory
PLOTS_DIR = OUTPUT_DIR / "plots"