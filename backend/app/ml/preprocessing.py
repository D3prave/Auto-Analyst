from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.schemas import ColumnType


def build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    overrides: Optional[Dict[str, ColumnType]] = None,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, Dict[str, Any]]:
    """
    Build a ColumnTransformer using overrides when provided.

    - "numeric" -> numeric pipeline
    - "categorical" / "boolean" -> categorical pipeline
    - "id" / "text" -> dropped from features
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe.")

    y = df[target_col]
    X = df.drop(columns=[target_col]).copy()

    numeric_features: List[str] = []
    categorical_features: List[str] = []

    feature_cols = [c for c in df.columns if c != target_col]

    for col in feature_cols:
        override = overrides.get(col) if overrides else None

        if override == "numeric":
            numeric_features.append(col)
        elif override in ("categorical", "boolean"):
            categorical_features.append(col)
        elif override in ("id", "text", "datetime"):
            # skip from modeling for now
            continue
        else:
            # fallback to dtype-based - simple heuristic
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)

    # Convert numeric overrides to numbers where possible
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    return X, y, preprocessor, metadata