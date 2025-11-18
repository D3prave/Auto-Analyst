from typing import Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.schemas import ColumnType

def build_preprocessor(df: pd.DataFrame, target_col: str, overrides: Optional[Dict[str, ColumnType]] = None) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, Dict[str, Any]]:
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    num_feats, cat_feats = [], []
    for col in X.columns:
        override = overrides.get(col) if overrides else None
        if override == "numeric": num_feats.append(col)
        elif override in ("categorical", "boolean"): cat_feats.append(col)
        elif override in ("id", "text"): continue
        elif pd.api.types.is_numeric_dtype(X[col]): num_feats.append(col)
        else: cat_feats.append(col)

    for col in num_feats:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    
    preprocessor = ColumnTransformer([("num", num_pipe, num_feats), ("cat", cat_pipe, cat_feats)])
    return X, y, preprocessor, {"numeric_features": num_feats, "categorical_features": cat_feats}