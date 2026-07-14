import pandas as pd
import numpy as np
from ..models import ProfilingReport, ColumnProfile


class ProfilingEngine:
    """
    Generates a comprehensive statistical profile of a dataset.
    Covers data types, missing values, cardinality, distributions, and top values.
    """

    def run(self, df: pd.DataFrame) -> ProfilingReport:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        report = ProfilingReport(num_rows=len(df), num_cols=len(df.columns), memory_mb=memory_mb)

        for col in df.columns:
            series = df[col]
            count   = len(series)
            missing = int(series.isnull().sum())
            missing_pct = (missing / count * 100) if count > 0 else 0.0
            unique  = int(series.nunique())
            dtype   = str(series.dtype)

            # Numeric stats
            mean = median = std = min_val = max_val = None
            top_values = []

            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    mean    = float(non_null.mean())
                    median  = float(non_null.median())
                    std     = float(non_null.std())
                    min_val = float(non_null.min())
                    max_val = float(non_null.max())
            else:
                # Top frequent values for categoricals
                top_values = series.dropna().value_counts().head(3).index.tolist()
                top_values = [str(v) for v in top_values]

            report.columns.append(ColumnProfile(
                name=col, dtype=dtype, count=count,
                missing=missing, missing_pct=round(missing_pct, 2),
                unique=unique, top_values=top_values,
                mean=mean, median=median, std=std,
                min_val=min_val, max_val=max_val
            ))

        return report
