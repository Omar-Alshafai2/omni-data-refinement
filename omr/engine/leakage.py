import pandas as pd
import numpy as np
from typing import Optional
from ..models import LeakageReport, LeakageIssue


class LeakageEngine:
    """
    Detects data leakage risks — features that are suspiciously
    correlated with the target, which could inflate model performance.
    """

    def run(self, df: pd.DataFrame, target_col: str, threshold: float = 0.95) -> LeakageReport:
        report = LeakageReport()

        if target_col not in df.columns:
            return report

        target = df[target_col]

        # Encode target if categorical
        if target.dtype == object or str(target.dtype) == "category":
            target = pd.Categorical(target).codes

        for col in df.columns:
            if col == target_col:
                continue

            series = df[col].copy()

            # Encode feature if categorical
            if series.dtype == object or str(series.dtype) == "category":
                series = pd.Categorical(series).codes

            # Only correlate numeric
            if not pd.api.types.is_numeric_dtype(series):
                continue

            # Drop rows where either is NaN
            combined = pd.DataFrame({"feature": series, "target": target}).dropna()
            if len(combined) < 10:
                continue

            try:
                corr = combined["feature"].corr(combined["target"])
            except Exception:
                continue

            if pd.isna(corr):
                continue

            abs_corr = abs(corr)

            if abs_corr >= threshold:
                risk = "Critical" if abs_corr >= 0.99 else "High"
                report.add_issue(LeakageIssue(
                    feature=col,
                    target=target_col,
                    correlation=round(corr, 4),
                    risk=risk,
                    description=(
                        f"Feature '{col}' has an extremely high correlation ({corr:.4f}) with target '{target_col}'. "
                        f"This may indicate target leakage — the feature may encode the answer directly."
                    )
                ))
            elif abs_corr >= 0.80:
                report.add_issue(LeakageIssue(
                    feature=col,
                    target=target_col,
                    correlation=round(corr, 4),
                    risk="Medium",
                    description=(
                        f"Feature '{col}' is highly correlated ({corr:.4f}) with target '{target_col}'. "
                        f"Review whether this is a legitimate feature or a proxy leakage variable."
                    )
                ))

        return report
