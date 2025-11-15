from __future__ import annotations
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

def evaluate_classification(
    y_true, y_pred, y_proba=None
) -> Dict[str, Any]:
    """Compute common classification metrics."""
    metrics: Dict[str, Any] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    # macro F1 for multi-class
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))

    # ROC AUC if probabilities provided and at least 2 classes
    if y_proba is not None:
        try:
            # If y_proba is shape (n_samples, n_classes)
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr")
                )
            else:
                # binary case
                # ensure we pass probabilities for the positive class
                if y_proba.ndim == 2:
                    proba = y_proba[:, 1]
                else:
                    proba = y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            # it's okay if ROC AUC fails (e.g., single class)
            pass

    return metrics


def evaluate_regression(
    y_true, y_pred
) -> Dict[str, Any]:
    """Compute common regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2),
    }