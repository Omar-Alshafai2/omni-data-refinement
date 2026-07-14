import pandas as pd
import numpy as np
from typing import Optional, List
from ..models import DriftReport, DriftResult


class DriftEngine:
    """
    Detects data drift between two DataFrames (e.g., training vs production).
    Compares mean, std, and distribution of numeric columns.
    """

    def run(self, train_df: pd.DataFrame, prod_df: pd.DataFrame, threshold_pct: float = 20.0) -> DriftReport:
        """
        Compare two DataFrames for statistical drift.

        Args:
            train_df: The reference (training) DataFrame.
            prod_df:  The production (inference) DataFrame.
            threshold_pct: Percentage change in mean to trigger a warning.
        """
        report = DriftReport()

        common_cols = [c for c in train_df.columns if c in prod_df.columns]
        numeric_cols = train_df[common_cols].select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            train_series = train_df[col].dropna()
            prod_series  = prod_df[col].dropna()

            if len(train_series) == 0 or len(prod_series) == 0:
                continue

            train_mean = float(train_series.mean())
            prod_mean  = float(prod_series.mean())
            train_std  = float(train_series.std())
            prod_std   = float(prod_series.std())

            if train_mean == 0:
                drift_pct = 0.0 if prod_mean == 0 else 100.0
            else:
                drift_pct = abs((prod_mean - train_mean) / train_mean) * 100

            if drift_pct >= 50:
                severity = "High"
            elif drift_pct >= threshold_pct:
                severity = "Medium"
            elif drift_pct >= 5:
                severity = "Low"
            else:
                severity = "None"

            report.add_result(DriftResult(
                column=col,
                train_mean=train_mean,
                prod_mean=prod_mean,
                train_std=train_std,
                prod_std=prod_std,
                drift_pct=round(drift_pct, 2),
                severity=severity
            ))

        return report
