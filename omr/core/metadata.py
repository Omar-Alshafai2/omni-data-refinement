"""
OMR Core — DatasetMetadata: tracks shape, dtypes, quality stats, and lineage.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class TransformationRecord:
    """Records a single transformation applied to the dataset."""
    step: int
    operation: str
    column: str
    rows_affected: int
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DatasetMetadata:
    """
    Captures essential metadata about a dataset at a point in time.
    Automatically updated by Dataset after every transformation.
    """
    num_rows: int
    num_cols: int
    column_names: List[str]
    dtypes: Dict[str, str]
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    memory_mb: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: Optional[str] = None
    transformations: List[TransformationRecord] = field(default_factory=list)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, source: Optional[str] = None) -> "DatasetMetadata":
        total_cells = df.size
        missing = int(df.isnull().sum().sum())
        return cls(
            num_rows=len(df),
            num_cols=len(df.columns),
            column_names=df.columns.tolist(),
            dtypes={col: str(df[col].dtype) for col in df.columns},
            missing_cells=missing,
            missing_pct=round((missing / total_cells * 100) if total_cells > 0 else 0.0, 2),
            duplicate_rows=int(df.duplicated().sum()),
            memory_mb=round(df.memory_usage(deep=True).sum() / (1024 ** 2), 4),
            source=source,
        )

    def log_transformation(self, operation: str, column: str,
                           rows_affected: int, reason: str) -> None:
        step = len(self.transformations) + 1
        self.transformations.append(TransformationRecord(
            step=step, operation=operation, column=column,
            rows_affected=rows_affected, reason=reason
        ))

    def to_dict(self) -> dict:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "missing_cells": self.missing_cells,
            "missing_pct": self.missing_pct,
            "duplicate_rows": self.duplicate_rows,
            "memory_mb": self.memory_mb,
            "created_at": self.created_at,
            "source": self.source,
        }
