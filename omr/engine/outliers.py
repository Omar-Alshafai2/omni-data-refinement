import pandas as pd
import numpy as np
from typing import List
from ..models import OutlierResult


class OutliersEngine:
    """
    Detects outliers in numeric columns using IQR or Z-Score methods.

    Strategies for handling outliers:
      - "flag"   : Return information only (default).
      - "remove" : Drop rows containing outliers.
      - "cap"    : Winsorize (clip values to the fence bounds).
    """

    def detect(self, df: pd.DataFrame, method: str = "iqr", strategy: str = "flag") -> List[OutlierResult]:
        results = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue

            if method == "iqr":
                outlier_mask = self._iqr_mask(df[col])
            elif method == "zscore":
                outlier_mask = self._zscore_mask(df[col])
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'zscore'.")

            indices = df.index[outlier_mask].tolist()
            if indices:
                values = df.loc[indices, col].tolist()
                results.append(OutlierResult(
                    column=col,
                    method=method,
                    count=len(indices),
                    values=values,
                    indices=indices
                ))

        return results

    def apply(self, df: pd.DataFrame, results: List[OutlierResult], strategy: str = "cap") -> pd.DataFrame:
        """Apply a fix strategy to outliers detected previously."""
        df = df.copy()
        for result in results:
            col = result.column
            if strategy == "remove":
                df = df.drop(index=result.indices, errors="ignore")
            elif strategy == "cap":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _iqr_mask(self, series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (series < lower) | (series > upper)

    def _zscore_mask(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        mean = series.mean()
        std  = series.std()
        if std == 0:
            return pd.Series([False] * len(series), index=series.index)
        z_scores = (series - mean) / std
        return z_scores.abs() > threshold
