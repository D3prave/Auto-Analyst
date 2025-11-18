from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

def evaluate_classification(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
    }
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim==2 else y_proba))
        except: pass
    return metrics

def evaluate_regression(y_true, y_pred) -> Dict[str, Any]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred))
    }