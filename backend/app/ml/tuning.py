from typing import Dict, Any, Callable
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_squared_error, mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_objective(X, y, preprocessor, model_class, task_type, metric, random_state):
    """Define Optuna objective function based on model type."""
    def objective(trial):
        params = {}
        if model_class in [RandomForestClassifier, RandomForestRegressor]:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 20)
        elif model_class in [GradientBoostingClassifier, GradientBoostingRegressor]:
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
        elif model_class == LogisticRegression:
            params["C"] = trial.suggest_float("C", 1e-4, 1e2, log=True)

        model_params = {k: v for k, v in params.items()}
        if "random_state" in model_class().get_params():
            model_params["random_state"] = random_state
            
        clf = Pipeline([("preprocess", preprocessor), ("model", model_class(**model_params))])
        
        if task_type == "classification":
            cv = StratifiedKFold(3, shuffle=True, random_state=random_state)
            scorer = make_scorer(f1_score, average="macro") if metric == "f1" else make_scorer(accuracy_score)
        else:
            cv = KFold(3, shuffle=True, random_state=random_state)
            scorer = make_scorer(mean_squared_error, greater_is_better=False)

        return cross_val_score(clf, X, y, cv=cv, scoring=scorer).mean()
    return objective

def tune_model(X, y, preprocessor, model_class, task_type, metric="accuracy", n_trials=20, random_state=42):
    metric = metric or ("accuracy" if task_type == "classification" else "rmse")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(get_objective(X, y, preprocessor, model_class, task_type, metric, random_state), n_trials=n_trials)
    return study.best_params