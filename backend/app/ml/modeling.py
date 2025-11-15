from __future__ import annotations
from typing import Optional, Dict, Any, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.base import BaseEstimator

from app.schemas import ColumnType
from app.ml.task_detection import detect_task_type, TaskType
from app.ml.preprocessing import build_preprocessor
from app.ml.evaluation import evaluate_classification, evaluate_regression


def _get_candidate_models(task_type: TaskType, random_state: int) -> Dict[str, BaseEstimator]:
    """
    Returns candidate models based on the task type.
    """
    if task_type == "classification":
        return {
            "logreg": LogisticRegression(max_iter=1000),
            "rf": RandomForestClassifier(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
            "gb": GradientBoostingClassifier(random_state=random_state),
        }
    else:
        return {
            "linreg": LinearRegression(),
            "rf": RandomForestRegressor(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
            "gb": GradientBoostingRegressor(random_state=random_state),
        }


def run_baseline_models(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    overrides: Optional[Dict[str, ColumnType]] = None,
) -> Dict[str, Any]:
    """
    Train several candidate models and pick the best one.
    Returns:
      - overall task info
      - best model key + class name
      - best model hyperparameters
      - metrics for each candidate
      - feature importances for the best model (if available)
    """
    X, y, preprocessor, meta = build_preprocessor(df, target_col, overrides)
    task_type: TaskType = detect_task_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task_type == "classification" else None,
    )

    models = _get_candidate_models(task_type, random_state)
    results_per_model: Dict[str, Dict[str, Any]] = {}
    best_model_name: Optional[str] = None
    best_score = -1e9
    best_clf: Optional[Pipeline] = None  # pipeline of best model

    for name, base_model in models.items():
        clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", base_model),
            ]
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if task_type == "classification":
            proba = clf.predict_proba(X_test) if hasattr(base_model, "predict_proba") else None
            metrics = evaluate_classification(y_test, y_pred, proba)
            score = metrics.get("f1_macro", metrics.get("accuracy", 0.0))
        else:
            metrics = evaluate_regression(y_test, y_pred)
            score = -metrics["rmse"]  # smaller RMSE = better model

        results_per_model[name] = {
            "metrics": metrics,
            "class_name": base_model.__class__.__name__,
        }

        if score > best_score:
            best_score = score
            best_model_name = name
            best_clf = clf

    if best_model_name is None or best_clf is None:
        raise RuntimeError("No best model was selected. This should not happen.")

    # Feature importances from *best* model, if available
    feature_importances = None
    try:
        model_step = best_clf.named_steps["model"]
        preprocess_step = best_clf.named_steps["preprocess"]

        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
            feature_names = preprocess_step.get_feature_names_out()
            feature_importances = [
                {"feature": f, "importance": float(i)}
                for f, i in sorted(
                    zip(feature_names, importances), key=lambda x: x[1], reverse=True
                )
            ]
    except Exception:
        pass

    # Model details for the best model
    best_model_step = best_clf.named_steps["model"]
    best_model_class = best_model_step.__class__.__name__
    raw_params = best_model_step.get_params()
    # ensure JSON-serializable
    best_model_params = {
        k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
        for k, v in raw_params.items()
    }

    # Metrics for best model
    best_metrics = results_per_model[best_model_name]["metrics"]

    result: Dict[str, Any] = {
        "task_type": task_type,
        "target": target_col,
        "best_model": best_model_name,          # "logreg", "rf", "gb", etc.
        "best_model_class": best_model_class,   # e.g. "RandomForestClassifier"
        "best_model_params": best_model_params,
        "models": results_per_model,            # all candidate models + metrics
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "metrics": best_metrics,                # metrics of the best model
        "feature_importances": feature_importances,
        "preprocessing": meta,
    }

    return result