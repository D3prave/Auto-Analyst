from __future__ import annotations
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

ColumnType = Literal["numeric", "categorical", "datetime", "id", "text", "boolean"]


class EDARequest(BaseModel):
    overrides: Optional[Dict[str, ColumnType]] = None


class ModelingRequest(BaseModel):
    target: str = Field(..., description="Name of the target column in the dataset.")
    test_size: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Fraction of data used for test set."
    )
    random_state: int = Field(default=42, description="Random seed.")
    overrides: Optional[Dict[str, ColumnType]] = None