from __future__ import annotations
from typing import Dict, Any, List


def _format_pct(x: float) -> str:
    return f"{x:.1f}%"


def dataset_overview_insight(profile: Dict[str, Any]) -> str:
    n_rows = profile.get("n_rows", 0)
    n_cols = profile.get("n_cols", 0)

    cols = profile.get("columns", {})
    numeric = [c for c, meta in cols.items() if meta.get("inferred_type") == "numeric"]
    categorical = [
        c for c, meta in cols.items() if meta.get("inferred_type") == "categorical"
    ]
    datetime_cols = [
        c for c, meta in cols.items() if meta.get("inferred_type") == "datetime"
    ]

    parts = [
        f"The dataset contains {n_rows} rows and {n_cols} columns.",
        f"There are {len(numeric)} numeric, {len(categorical)} categorical, and {len(datetime_cols)} datetime features.",
    ]

    return " ".join(parts)


def missingness_insight(profile: Dict[str, Any]) -> str:
    missing = profile.get("missing", {})
    missing_by_col = missing.get("missing_by_column", {})

    if not missing_by_col:
        return "There are no missing values reported in the dataset."

    total_rows = profile.get("n_rows", 0)
    if total_rows == 0:
        return "The dataset appears to be empty."

    # Compute max missingness and overall missingness
    max_col = None
    max_count = 0
    total_missing_cells = 0

    for col, count in missing_by_col.items():
        total_missing_cells += count
        if count > max_count:
            max_count = count
            max_col = col

    overall_pct = total_missing_cells / (total_rows * max(len(missing_by_col), 1)) * 100

    if max_col is None or max_count == 0:
        return "The dataset does not contain any missing values."

    max_pct = max_count / total_rows * 100

    msg = (
        f"Missing values are present in the dataset. "
        f"The column with the most missing values is '{max_col}' "
        f"with {max_count} missing entries ({_format_pct(max_pct)} of rows). "
        f"Overall, approximately {_format_pct(overall_pct)} of all cells are missing."
    )

    if max_pct > 50:
        msg += " This column may need special handling (e.g. removal or strong imputation)."

    return msg


def target_balance_insight(
    df_profile: Dict[str, Any], target_col: str | None
) -> str:
    if not target_col:
        return "No target column was specified, so target balance was not analyzed."

    cols = df_profile.get("columns", {})
    target_meta = cols.get(target_col)
    if not target_meta:
        return f"The target column '{target_col}' is not present in the profile."

    inferred_type = target_meta.get("inferred_type")
    if inferred_type not in ("categorical", "boolean"):
        return f"The target '{target_col}' is treated as a continuous variable (regression), so class balance does not apply."

    cat = target_meta.get("categorical_summary", {})
    values = cat.get("top_values", [])
    counts = cat.get("top_counts", [])

    if not values or not counts:
        return f"Could not compute class distribution for target '{target_col}'."

    total = sum(counts)
    parts: List[str] = [f"For the target '{target_col}', the class distribution is: "]
    desc_chunks = []
    for v, c in zip(values, counts):
        pct = c / total * 100 if total > 0 else 0
        desc_chunks.append(f"{v} = {c} ({_format_pct(pct)})")
    parts.append(", ".join(desc_chunks) + ".")

    # Check imbalance
    max_pct = max(c / total * 100 for c in counts) if total > 0 else 0
    min_pct = min(c / total * 100 for c in counts) if total > 0 else 0

    if max_pct > 80 and len(counts) > 1:
        parts.append("The target is highly imbalanced; metrics like accuracy may be misleading.")
    elif max_pct > 65 and len(counts) > 1:
        parts.append("The target shows a moderate imbalance; consider using F1 or AUC in addition to accuracy.")

    return " ".join(parts)


def modeling_insight(modeling: Dict[str, Any] | None) -> str:
    if modeling is None:
        return "No modeling results are available yet. Run the modeling step first."

    task = modeling.get("task_type")
    target = modeling.get("target")
    metrics = modeling.get("metrics", {})

    if task == "classification":
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1_macro")
        roc_auc = metrics.get("roc_auc") or metrics.get("roc_auc_ovr")

        parts = [
            f"A baseline classification model was trained to predict '{target}'."
        ]
        if acc is not None:
            parts.append(f" The accuracy on the hold-out set is {acc:.3f}.")
        if f1 is not None:
            parts.append(f" The macro-averaged F1 score is {f1:.3f}.")
        if roc_auc is not None:
            parts.append(f" The ROC AUC is {roc_auc:.3f}.")

        # simple performance commentary
        if acc is not None:
            if acc < 0.6:
                parts.append(" Model performance is relatively low; consider feature engineering, trying other algorithms, or tuning hyperparameters.")
            elif acc < 0.8:
                parts.append(" Model performance is reasonable but could likely be improved with tuning and additional features.")
            else:
                parts.append(" Model performance appears strong for a baseline model.")

        return "".join(parts)

    elif task == "regression":
        mae = metrics.get("mae")
        rmse = metrics.get("rmse")
        r2 = metrics.get("r2")

        parts = [
            f"A baseline regression model was trained to predict '{target}'."
        ]
        if mae is not None:
            parts.append(f" The mean absolute error (MAE) is {mae:.3f}.")
        if rmse is not None:
            parts.append(f" The root mean squared error (RMSE) is {rmse:.3f}.")
        if r2 is not None:
            parts.append(f" The R² score is {r2:.3f}.")

        if r2 is not None:
            if r2 < 0.3:
                parts.append(" The model explains only a small portion of the variance; more features or a different approach may be needed.")
            elif r2 < 0.7:
                parts.append(" The model explains a moderate amount of variance; there is room for improvement.")
            else:
                parts.append(" The model explains a large portion of the variance for a baseline.")

        return "".join(parts)

    return "Modeling results are available, but the task type could not be interpreted."


def feature_importance_insight(modeling: Dict[str, Any] | None, top_k: int = 5) -> str:
    if modeling is None:
        return "Feature importance could not be analyzed because no modeling results are available."

    fi = modeling.get("feature_importances")
    if not fi:
        return "Feature importance information is not available for this model."

    top = fi[:top_k]
    parts = ["The most influential features for the model appear to be: "]
    parts.append(
        ", ".join([f"{item['feature']} ({item['importance']:.3f})" for item in top]) + "."
    )
    return "".join(parts)


def generate_insights(
    profile: Dict[str, Any],
    modeling: Dict[str, Any] | None = None,
    target_col: str | None = None,
) -> Dict[str, str]:
    """
    High-level function to generate a set of human-readable insights.
    """
    return {
        "overview": dataset_overview_insight(profile),
        "missingness": missingness_insight(profile),
        "target_balance": target_balance_insight(profile, target_col),
        "modeling": modeling_insight(modeling),
        "feature_importance": feature_importance_insight(modeling),
    }