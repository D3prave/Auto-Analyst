from typing import Optional, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from app.ml.task_detection import detect_task_type
from app.ml.preprocessing import build_preprocessor
from app.ml.evaluation import evaluate_classification, evaluate_regression
from app.ml.tuning import tune_model


def run_baseline_models(
    df,
    target_col,
    test_size=0.2,
    val_size=0.0,
    random_state=42,
    overrides=None,
    tune_hyperparameters=False,
    tuning_trials=20,
    optimize_metric=None,
):
    X, y, preprocessor, meta = build_preprocessor(df, target_col, overrides)
    task = detect_task_type(y)

    # Default metric if none is provided
    if optimize_metric is None:
        optimize_metric = "accuracy" if task == "classification" else "rmse"

    # Split Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task == "classification" else None,
    )

    # Split Val (if requested)
    if val_size > 0 and (1.0 - test_size) > 0:
        X_train, _, y_train, _ = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size / (1.0 - test_size),
            random_state=random_state,
            stratify=y_temp if task == "classification" else None,
        )
    else:
        X_train, y_train = X_temp, y_temp

    models = (
        {
            "logreg": LogisticRegression(max_iter=1000),
            "rf": RandomForestClassifier(random_state=random_state),
            "gb": GradientBoostingClassifier(random_state=random_state),
        }
        if task == "classification"
        else {
            "linreg": LinearRegression(),
            "rf": RandomForestRegressor(random_state=random_state),
            "gb": GradientBoostingRegressor(random_state=random_state),
        }
    )

    results, best_score, best_name, best_clf = {}, -1e9, None, None

    for name, model in models.items():
        params = {}
        if tune_hyperparameters:
            params = tune_model(
                X_train,
                y_train,
                preprocessor,
                model.__class__,
                task,
                optimize_metric,
                tuning_trials,
                random_state,
            )
            model = model.__class__(
                **(
                    {**params, "random_state": random_state}
                    if "random_state" in model.get_params()
                    else params
                )
            )

        clf = Pipeline([("preprocess", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if task == "classification":
            metrics = evaluate_classification(
                y_test,
                y_pred,
                clf.predict_proba(X_test) if hasattr(model, "predict_proba") else None,
            )

            # Decide which metric to optimize based on optimize_metric
            if optimize_metric == "f1" and "f1_macro" in metrics:
                main_metric_name = "f1_macro"
            elif optimize_metric == "accuracy" and "accuracy" in metrics:
                main_metric_name = "accuracy"
            else:
                # Fallback: prefer F1 if available, otherwise accuracy
                main_metric_name = "f1_macro" if "f1_macro" in metrics else "accuracy"

            score = metrics[main_metric_name]

        else:
            metrics = evaluate_regression(y_test, y_pred)

            # For regression we currently optimize RMSE (lower is better)
            main_metric_name = "rmse"
            score = -metrics[main_metric_name]

        results[name] = {
            "metrics": metrics,
            "class_name": model.__class__.__name__,
            "best_params": params,
        }

        if score > best_score:
            best_score, best_name, best_clf = score, name, clf

    # Extract Feature Importance if possible
    fi = None
    try:
        imps = best_clf.named_steps["model"].feature_importances_
        names = best_clf.named_steps["preprocess"].get_feature_names_out()
        fi = [
            {"feature": f, "importance": float(i)}
            for f, i in sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
        ]
    except:
        pass

    return (
        {
            "task_type": task,
            "target": target_col,
            "best_model": best_name,
            "best_model_class": best_clf.named_steps["model"].__class__.__name__,
            "best_model_params": {
                k: str(v) for k, v in best_clf.named_steps["model"].get_params().items()
            },
            "models": results,
            "metrics": results[best_name]["metrics"],
            "feature_importances": fi,
            "preprocessing": meta,
            "tuned": tune_hyperparameters,
            "n_rows": len(df),
            "n_features": X.shape[1],
        },
        best_clf,
    )
