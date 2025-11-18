from typing import Dict, Any

def _format_pct(x: float) -> str:
    return f"{x:.1f}%"

def dataset_overview_insight(profile: Dict[str, Any]) -> str:
    cols = profile.get("columns", {})
    numeric = len([c for c, m in cols.items() if m.get("effective_type") == "numeric"])
    categorical = len([c for c, m in cols.items() if m.get("effective_type") == "categorical"])
    datetime_cols = len([c for c, m in cols.items() if m.get("effective_type") == "datetime"])

    return f"The dataset contains {profile.get('n_rows')} rows and {profile.get('n_cols')} columns. There are {numeric} numeric, {categorical} categorical, and {datetime_cols} datetime features."

def missingness_insight(profile: Dict[str, Any]) -> str:
    missing_counts = {c: m["missing_count"] for c, m in profile.get("columns", {}).items() if m["missing_count"] > 0}
    
    if not missing_counts:
        return "There are no missing values reported in the dataset."
    
    max_col = max(missing_counts, key=missing_counts.get)
    max_count = missing_counts[max_col]
    max_pct = (max_count / profile.get("n_rows", 1)) * 100
    
    msg = f"Missing values are present. The column with the most missing values is '{max_col}' with {max_count} missing entries ({_format_pct(max_pct)})."
    if max_pct > 50:
        msg += " This column may need special handling (e.g. removal or strong imputation)."
    return msg

def target_balance_insight(profile: Dict[str, Any], target_col: str | None) -> str:
    if not target_col: return "No target column specified."
    
    col_meta = profile.get("columns", {}).get(target_col)
    if not col_meta: return f"Target '{target_col}' not found."

    if col_meta.get("effective_type") not in ("categorical", "boolean"):
        return f"The target '{target_col}' is treated as a continuous variable (regression), so class balance does not apply."

    counts = col_meta.get("categorical_summary", {}).get("top_counts", [])
    values = col_meta.get("categorical_summary", {}).get("top_values", [])
    
    if not counts: return "Could not compute class distribution."

    total = sum(counts)
    desc = [f"{v}={c} ({_format_pct(c/total*100)})" for v, c in zip(values, counts)]
    parts = [f"Class distribution for '{target_col}': {', '.join(desc)}."]

    if max(counts)/total > 0.8 and len(counts) > 1:
        parts.append(" The target is highly imbalanced; accuracy may be misleading.")
    
    return "".join(parts)

def modeling_insight(modeling: Dict[str, Any] | None) -> str:
    if not modeling: return "No modeling results are available yet."

    task = modeling.get("task_type")
    metrics = modeling.get("metrics", {})
    
    if task == "classification":
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1_macro", 0)
        parts = [f"A baseline classification model was trained. Best model: {modeling['best_model_class']}."]
        parts.append(f" Accuracy: {acc:.3f}. Macro F1: {f1:.3f}.")
        
        if acc < 0.6: parts.append(" Model performance is relatively low; consider feature engineering.")
        elif acc > 0.85: parts.append(" Model performance appears strong.")
        return "".join(parts)
    
    elif task == "regression":
        rmse = metrics.get("rmse", 0)
        r2 = metrics.get("r2", 0)
        parts = [f"A baseline regression model was trained. Best model: {modeling['best_model_class']}."]
        parts.append(f" RMSE: {rmse:.3f}. R² Score: {r2:.3f}.")
        return "".join(parts)

    return "Model results available."

def feature_importance_insight(modeling: Dict[str, Any] | None, top_k: int = 5) -> str:
    if modeling is None or not modeling.get("feature_importances"):
        return ""

    fi = modeling.get("feature_importances")
    top = fi[:top_k]
    features = ", ".join([f"{item['feature']} ({item['importance']:.3f})" for item in top])
    return f"The most influential features are: {features}."

def generate_insights(profile: Dict[str, Any], modeling: Dict[str, Any] = None, target_col: str = None) -> Dict[str, str]:
    return {
        "overview": dataset_overview_insight(profile),
        "missingness": missingness_insight(profile),
        "target_balance": target_balance_insight(profile, target_col),
        "modeling": modeling_insight(modeling),
        "feature_importance": feature_importance_insight(modeling),
    }