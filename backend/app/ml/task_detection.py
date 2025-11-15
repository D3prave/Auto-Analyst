from __future__ import annotations
from typing import Literal

import pandas as pd
import numpy as np


TaskType = Literal["classification", "regression"]


def detect_task_type(y: pd.Series) -> TaskType:
    """
    Decide whether this is a classification or regression task.
    Heuristics:
    - numeric with many unique values -> regression
    - numeric with few unique integer values -> classification
    - non-numeric -> classification
    """
    s = y.dropna()

    if s.empty:
        # default to regression; caller should validate target
        return "regression"

    # If it's numeric
    if pd.api.types.is_numeric_dtype(s):
        unique_vals = s.nunique()
        n = len(s)
        unique_ratio = unique_vals / n if n > 0 else 0.0

        # Few unique integer values (like 0/1, 0-3, etc.) -> classification
        all_int_like = bool(np.allclose(s, s.astype(int), equal_nan=True))

        if all_int_like and unique_vals <= 20 and unique_ratio < 0.5:
            return "classification"

        return "regression"

    # Otherwise we treat as classification
    return "classification"