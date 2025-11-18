import pandas as pd
import numpy as np
from typing import Literal

TaskType = Literal["classification", "regression"]

def detect_task_type(y: pd.Series) -> TaskType:
    s = y.dropna()
    if s.empty: return "regression"
    if pd.api.types.is_numeric_dtype(s):
        # Heuristic: few unique integers usually implies classification
        if s.nunique() <= 20 and np.allclose(s, s.astype(int), equal_nan=True):
            return "classification"
        return "regression"
    return "classification"