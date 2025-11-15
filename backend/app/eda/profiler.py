from __future__ import annotations
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from app.schemas import ColumnType


def infer_column_type(s: pd.Series) -> str:
    """
    Infer a high-level semantic type.
    Types: id, numeric, categorical, boolean, datetime, text, unknown
    """
    if pd.api.types.is_bool_dtype(s):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    # Try parse datetime if it's object-like
    if pd.api.types.is_object_dtype(s):
        sample = s.dropna().astype(str).head(50)
        date_parse_success = 0
        for val in sample:
            try:
                pd.to_datetime(val)
                date_parse_success += 1
            except Exception:
                continue
        if len(sample) > 0 and date_parse_success / len(sample) > 0.7:
            return "datetime"

    if pd.api.types.is_numeric_dtype(s):
        unique = s.dropna().nunique()
        n = len(s.dropna())
        unique_ratio = unique / n if n > 0 else 0.0

        if unique > 0 and unique_ratio > 0.9:
            return "id"
        if unique <= 20:
            return "categorical"
        return "numeric"

    if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
        sample = s.dropna().astype(str)
        if sample.empty:
            return "categorical"
        avg_len = sample.str.len().mean()
        unique = sample.nunique()
        n = len(sample)
        unique_ratio = unique / n if n > 0 else 0.0

        if avg_len > 30 and unique_ratio > 0.5:
            return "text"
        return "categorical"

    return "unknown"


def numeric_summary(s: pd.Series) -> Dict[str, Any]:
    desc = s.describe()
    result = {
        "count": float(desc.get("count", 0.0)),
        "mean": float(desc.get("mean", np.nan)),
        "std": float(desc.get("std", np.nan)),
        "min": float(desc.get("min", np.nan)),
        "q1": float(desc.get("25%", np.nan)),
        "median": float(desc.get("50%", np.nan)),
        "q3": float(desc.get("75%", np.nan)),
        "max": float(desc.get("max", np.nan)),
    }

    q1 = result["q1"]
    q3 = result["q3"]
    if not np.isnan(q1) and not np.isnan(q3):
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((s < lower) | (s > upper)).sum()
        result["iqr"] = float(iqr)
        result["outlier_count"] = int(outliers)
    else:
        result["iqr"] = float("nan")
        result["outlier_count"] = 0

    return result


def categorical_summary(s: pd.Series, top_n: int = 10) -> Dict[str, Any]:
    vc = s.value_counts(dropna=True).head(top_n)
    values = vc.index.astype(str).tolist()
    counts = vc.values.tolist()
    return {
        "top_values": values,
        "top_counts": counts,
        "unique": int(s.nunique(dropna=True)),
    }


def profile_dataset(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, ColumnType]] = None,
    max_unique_for_preview: int = 50,
) -> Dict[str, Any]:
    """
    Produce a JSON-serializable profile.
    If overrides are provided, they override the inferred type.
    """
    n_rows, n_cols = df.shape

    columns_profile: Dict[str, Any] = {}
    for col in df.columns:
        s = df[col]
        inferred = infer_column_type(s)
        override = overrides.get(col) if overrides else None
        col_type = override or inferred

        missing = int(s.isna().sum())
        missing_pct = float(missing / len(s) * 100) if len(s) > 0 else 0.0
        distinct = int(s.nunique(dropna=True))

        col_info: Dict[str, Any] = {
            "name": col,
            "inferred_type": inferred,
            "effective_type": col_type,
            "overridden_type": override,
            "missing_count": missing,
            "missing_pct": missing_pct,
            "distinct_count": distinct,
        }

        if col_type == "numeric":
            col_info["numeric_summary"] = numeric_summary(pd.to_numeric(s, errors="coerce"))
        elif col_type in ("categorical", "boolean"):
            col_info["categorical_summary"] = categorical_summary(s)
        elif col_type == "datetime":
            s_dt = pd.to_datetime(s, errors="coerce")
            col_info["datetime_summary"] = {
                "min": s_dt.min().isoformat() if not s_dt.dropna().empty else None,
                "max": s_dt.max().isoformat() if not s_dt.dropna().empty else None,
            }

        # Suggest plots: skip ids & text
        col_info["plot_suggested"] = col_type not in ("id", "text")

        columns_profile[col] = col_info

    missing_by_row = int((df.isna().sum(axis=1) > 0).sum())
    missing_by_col = df.isna().sum().to_dict()

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns": columns_profile,
        "missing": {
            "rows_with_missing": missing_by_row,
            "missing_by_column": {k: int(v) for k, v in missing_by_col.items()},
        },
    }