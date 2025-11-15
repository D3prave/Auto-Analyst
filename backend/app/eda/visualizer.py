from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

from app.config import PLOTS_DIR, OUTPUT_DIR


def _ensure_dataset_plot_dir(dataset_id: str) -> Path:
    ds_dir = PLOTS_DIR / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    return ds_dir


def _save_fig(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    # Return path relative to OUTPUT_DIR, without leading slash
    # e.g. "plots/<dataset_id>/age_hist.png"
    rel_path = path.relative_to(OUTPUT_DIR)
    return f"outputs/{rel_path}"


def generate_numeric_plots(df: pd.DataFrame, dataset_id: str, profile: Dict[str, Any], max_cols: int = 20):
    ds_dir = _ensure_dataset_plot_dir(dataset_id)

    results = []
    for col, meta in profile["columns"].items():
        if not meta.get("plot_suggested", True):
            continue

        col_type = meta["inferred_type"]
        if col_type != "numeric":
            continue

        s = df[col].dropna()
        if s.empty:
            continue

        col_result = {"column": col, "histogram": None, "boxplot": None}

        # Histogram
        plt.figure(figsize=(5, 4))
        sns.histplot(s, kde=True)
        plt.title(f"Histogram of {col}")
        hist_path = ds_dir / f"{col}_hist.png"
        col_result["histogram"] = _save_fig(hist_path)

        # Boxplot
        plt.figure(figsize=(4, 4))
        sns.boxplot(x=s)
        plt.title(f"Boxplot of {col}")
        box_path = ds_dir / f"{col}_box.png"
        col_result["boxplot"] = _save_fig(box_path)

        results.append(col_result)

        if len(results) >= max_cols:
            break

    return results


def generate_categorical_plots(df: pd.DataFrame, dataset_id: str, profile: Dict[str, Any], max_cols: int = 20, top_n: int = 10):
    ds_dir = _ensure_dataset_plot_dir(dataset_id)

    results = []
    for col, meta in profile["columns"].items():
        if not meta.get("plot_suggested", True):
            continue

        col_type = meta["inferred_type"]
        if col_type not in ("categorical", "boolean"):
            continue

        s = df[col].astype(str)
        vc = s.value_counts().head(top_n)
        if vc.empty:
            continue

        plt.figure(figsize=(6, 4))
        sns.barplot(x=vc.values, y=vc.index)
        plt.title(f"Top {top_n} values of {col}")
        plt.xlabel("Count")
        plt.ylabel(col)

        bar_path = ds_dir / f"{col}_bar.png"
        path_str = _save_fig(bar_path)

        results.append(
            {
                "column": col,
                "barplot": path_str,
                "top_values": vc.index.astype(str).tolist(),
                "top_counts": vc.values.tolist(),
            }
        )

        if len(results) >= max_cols:
            break

    return results



def generate_correlation_heatmap(
    df: pd.DataFrame, dataset_id: str, min_cols: int = 2
) -> Dict[str, Any] | None:
    """
    Generate a correlation heatmap for numeric columns.
    Returns dict with file path and columns, or None if not enough numeric columns.
    """
    ds_dir = _ensure_dataset_plot_dir(dataset_id)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < min_cols:
        return None

    corr = numeric_df.corr()

    plt.figure(figsize=(max(6, 0.6 * len(corr.columns)), max(5, 0.5 * len(corr.columns))))
    sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")

    heatmap_path = ds_dir / "correlation_heatmap.png"
    path_str = _save_fig(heatmap_path)

    return {
        "path": path_str,
        "columns": corr.columns.tolist(),
    }


def generate_plots(df: pd.DataFrame, dataset_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    numeric_plots = generate_numeric_plots(df, dataset_id, profile)
    categorical_plots = generate_categorical_plots(df, dataset_id, profile)
    corr_heatmap = generate_correlation_heatmap(df, dataset_id)

    return {
        "numeric": numeric_plots,
        "categorical": categorical_plots,
        "correlation_heatmap": corr_heatmap,
    }