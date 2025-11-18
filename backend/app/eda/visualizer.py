from __future__ import annotations
from typing import Dict, Any, List
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from app.utils.storage import save_plot

def _save_plot_to_redis(dataset_id: str, filename: str) -> str:
    """
    Saves the current matplotlib figure to Redis and returns the API URL.
    """
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    
    buf.seek(0)
    save_plot(dataset_id, filename, buf.getvalue())
    
    # Return the absolute API URL.
    # The browser will resolve this against the current domain (e.g. localhost:3000 or localhost:8000)
    return f"/api/images/{dataset_id}/{filename}"

def generate_numeric_plots(df: pd.DataFrame, dataset_id: str, profile: Dict[str, Any], max_cols: int = 20):
    results = []
    for col, meta in profile["columns"].items():
        if not meta.get("plot_suggested", True):
            continue
        if meta["inferred_type"] != "numeric":
            continue

        s = df[col].dropna()
        if s.empty:
            continue

        col_result = {"column": col, "histogram": None, "boxplot": None}

        # Histogram
        plt.figure(figsize=(5, 4))
        sns.histplot(s, kde=True)
        plt.title(f"Histogram of {col}")
        col_result["histogram"] = _save_plot_to_redis(dataset_id, f"{col}_hist.png")

        # Boxplot
        plt.figure(figsize=(4, 4))
        sns.boxplot(x=s)
        plt.title(f"Boxplot of {col}")
        col_result["boxplot"] = _save_plot_to_redis(dataset_id, f"{col}_box.png")

        results.append(col_result)
        if len(results) >= max_cols:
            break

    return results

def generate_categorical_plots(df: pd.DataFrame, dataset_id: str, profile: Dict[str, Any], max_cols: int = 20, top_n: int = 10):
    results = []
    for col, meta in profile["columns"].items():
        if not meta.get("plot_suggested", True):
            continue
        if meta["inferred_type"] not in ("categorical", "boolean"):
            continue

        s = df[col].astype(str)
        vc = s.value_counts().head(top_n)
        if vc.empty:
            continue

        plt.figure(figsize=(6, 4))
        sns.barplot(x=vc.values, y=vc.index)
        plt.title(f"Top {top_n} values of {col}")
        plt.xlabel("Count")
        
        col_result = {
            "column": col,
            "barplot": _save_plot_to_redis(dataset_id, f"{col}_bar.png"),
            "top_values": vc.index.astype(str).tolist(),
            "top_counts": vc.values.tolist(),
        }
        results.append(col_result)
        if len(results) >= max_cols:
            break

    return results

def generate_correlation_heatmap(df: pd.DataFrame, dataset_id: str, min_cols: int = 2) -> Dict[str, Any] | None:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < min_cols:
        return None

    corr = numeric_df.corr()
    plt.figure(figsize=(max(6, 0.6 * len(corr.columns)), max(5, 0.5 * len(corr.columns))))
    sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")

    path_url = _save_plot_to_redis(dataset_id, "correlation_heatmap.png")

    return {
        "path": path_url,
        "columns": corr.columns.tolist(),
    }

def generate_plots(df: pd.DataFrame, dataset_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "numeric": generate_numeric_plots(df, dataset_id, profile),
        "categorical": generate_categorical_plots(df, dataset_id, profile),
        "correlation_heatmap": generate_correlation_heatmap(df, dataset_id),
    }