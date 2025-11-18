from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from app.schemas import ColumnType

def infer_column_type(s: pd.Series) -> str:
    """Heuristic-based column type inference."""
    if pd.api.types.is_bool_dtype(s): return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s): return "datetime"

    if pd.api.types.is_object_dtype(s):
        # Sample check for datetime strings
        sample = s.dropna().astype(str).head(50)
        try:
            if len(sample) > 0 and (pd.to_datetime(sample, errors='coerce').notna().mean() > 0.7):
                return "datetime"
        except: pass

    if pd.api.types.is_numeric_dtype(s):
        unique = s.dropna().nunique()
        # High cardinality ratio usually implies an ID column
        if unique > 0 and (unique / len(s.dropna()) > 0.9): return "id"
        if unique <= 20: return "categorical"
        return "numeric"

    if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
        sample = s.dropna().astype(str)
        # Long average length and high cardinality imply free text
        if sample.nunique() / len(sample) > 0.5 and sample.str.len().mean() > 30:
            return "text"
        return "categorical"
    return "unknown"

def numeric_summary(s: pd.Series) -> Dict[str, Any]:
    desc = s.describe()
    q1, q3 = desc.get("25%", np.nan), desc.get("75%", np.nan)
    iqr = q3 - q1 if not np.isnan(q1) else np.nan
    return {
        "count": float(desc.get("count", 0)),
        "mean": float(desc.get("mean", np.nan)),
        "std": float(desc.get("std", np.nan)),
        "min": float(desc.get("min", np.nan)),
        "q1": float(q1), "median": float(desc.get("50%", np.nan)), "q3": float(q3),
        "max": float(desc.get("max", np.nan)),
        "iqr": float(iqr),
        "outlier_count": int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()) if not np.isnan(iqr) else 0
    }

def categorical_summary(s: pd.Series, top_n: int = 10) -> Dict[str, Any]:
    vc = s.value_counts(dropna=True).head(top_n)
    return {
        "top_values": vc.index.astype(str).tolist(),
        "top_counts": vc.values.tolist(),
        "unique": int(s.nunique(dropna=True))
    }

def profile_dataset(df: pd.DataFrame, overrides: Optional[Dict[str, ColumnType]] = None) -> Dict[str, Any]:
    profile = {}
    for col in df.columns:
        s = df[col]
        inferred = infer_column_type(s)
        col_type = overrides.get(col, inferred) if overrides else inferred
        
        info = {
            "name": col, "inferred_type": inferred, "effective_type": col_type,
            "overridden_type": overrides.get(col) if overrides else None,
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100),
            "distinct_count": int(s.nunique(dropna=True)),
            "plot_suggested": col_type not in ("id", "text")
        }

        if col_type == "numeric":
            info["numeric_summary"] = numeric_summary(pd.to_numeric(s, errors="coerce"))
        elif col_type in ("categorical", "boolean"):
            info["categorical_summary"] = categorical_summary(s)
        elif col_type == "datetime":
            s_dt = pd.to_datetime(s, errors="coerce")
            info["datetime_summary"] = {"min": s_dt.min().isoformat() if not s_dt.empty else None, "max": s_dt.max().isoformat() if not s_dt.empty else None}
        
        profile[col] = info

    return {
        "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]),
        "columns": profile,
        "missing": {"rows_with_missing": int((df.isna().sum(axis=1) > 0).sum())}
    }