from __future__ import annotations
from typing import Dict, Any
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Force non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from app.utils.storage import save_plot

def _save(dataset_id: str, filename: str) -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    save_plot(dataset_id, filename, buf.getvalue())
    # Return absolute API path for frontend usage
    return f"/api/images/{dataset_id}/{filename}"

def generate_plots(df: pd.DataFrame, dataset_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    numeric, categorical = [], []
    
    for col, meta in profile["columns"].items():
        if not meta.get("plot_suggested", True): continue
        
        if meta["effective_type"] == "numeric":
            s = df[col].dropna()
            if s.empty: continue
            
            plt.figure(figsize=(5, 4))
            sns.histplot(s, kde=True).set_title(f"Histogram of {col}")
            hist = _save(dataset_id, f"{col}_hist.png")
            
            plt.figure(figsize=(4, 4))
            sns.boxplot(x=s).set_title(f"Boxplot of {col}")
            box = _save(dataset_id, f"{col}_box.png")
            numeric.append({"column": col, "histogram": hist, "boxplot": box})

        elif meta["effective_type"] in ("categorical", "boolean"):
            s = df[col].astype(str)
            vc = s.value_counts().head(10)
            if vc.empty: continue
            
            plt.figure(figsize=(6, 4))
            sns.barplot(x=vc.values, y=vc.index).set_title(f"Top 10 {col}")
            categorical.append({
                "column": col, 
                "barplot": _save(dataset_id, f"{col}_bar.png"),
                "top_values": vc.index.tolist(), "top_counts": vc.values.tolist()
            })

    heatmap = None
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        plt.figure(figsize=(max(6, 0.6 * len(corr)), max(5, 0.5 * len(corr))))
        sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5).set_title("Correlation Heatmap")
        heatmap = {"path": _save(dataset_id, "correlation_heatmap.png"), "columns": corr.columns.tolist()}

    return {"numeric": numeric, "categorical": categorical, "correlation_heatmap": heatmap}