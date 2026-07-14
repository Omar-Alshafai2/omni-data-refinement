"""
OMR Profiling Module — Full statistical profile of the dataset.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing: int
    missing_pct: float
    unique: int
    mean: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    top_values: List[str] = field(default_factory=list)


@dataclass
class ProfilingReport:
    num_rows: int
    num_cols: int
    memory_mb: float
    columns: List[ColumnProfile] = field(default_factory=list)


class ProfilingEngine:
    """Generates column-level statistical statistics."""

    def run(self, df: pd.DataFrame) -> ProfilingReport:
        if df.empty:
            return ProfilingReport(0, 0, 0.0, [])
            
        profiles = []
        num_rows = len(df)
        
        for col in df.columns:
            series = df[col]
            missing = series.isnull().sum()
            missing_pct = (missing / num_rows) * 100
            unique = series.nunique(dropna=True)
            
            mean_val, min_val, max_val = None, None, None
            top_vals = []
            
            if pd.api.types.is_numeric_dtype(series):
                desc = series.describe()
                mean_val = desc.get("mean", None)
                min_val = desc.get("min", None)
                max_val = desc.get("max", None)
            else:
                top_vals = [str(x) for x in series.value_counts().head(3).index.tolist()]
                
            profiles.append(ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                missing=int(missing),
                missing_pct=float(missing_pct),
                unique=int(unique),
                mean=float(mean_val) if mean_val is not None and not np.isnan(mean_val) else None,
                min_val=float(min_val) if min_val is not None and not np.isnan(min_val) else None,
                max_val=float(max_val) if max_val is not None and not np.isnan(max_val) else None,
                top_values=top_vals
            ))
            
        memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        return ProfilingReport(num_rows, len(df.columns), memory_mb, profiles)
