from __future__ import annotations
from typing import Dict, Literal, Optional, List, Union
from pydantic import BaseModel, Field

ColumnType = Literal["numeric", "categorical", "datetime", "id", "text", "boolean"]

class EDARequest(BaseModel):
    overrides: Optional[Dict[str, ColumnType]] = None

class FilterCondition(BaseModel):
    column: str
    operator: Literal["==", "!=", ">", "<", ">=", "<="]
    value: Union[str, int, float, bool]

class TransformRequest(BaseModel):
    drop_columns: List[str] = Field(default_factory=list)
    fill_na_strategy: Optional[Literal["mean", "median", "mode", "drop", "constant"]] = None
    fill_na_value: Optional[Union[str, int, float]] = None
    filter_conditions: List[FilterCondition] = Field(default_factory=list)

class ModelingRequest(BaseModel):
    target: str
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    val_size: float = Field(0.0, ge=0.0, lt=1.0)
    random_state: int = 42
    overrides: Optional[Dict[str, ColumnType]] = None
    tune_hyperparameters: bool = False
    tuning_trials: int = Field(20, ge=1, le=100)
    optimize_metric: Optional[str] = None