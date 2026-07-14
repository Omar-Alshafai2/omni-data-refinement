import pandas as pd
import numpy as np
from typing import Optional
from ..models import BiasReport, BiasIssue


class BiasEngine:
    """
    Detects bias and fairness issues including class imbalance,
    demographic imbalance, and distribution skew.
    """

    def run(self, df: pd.DataFrame, target_col: Optional[str] = None) -> BiasReport:
        report = BiasReport()

        # Class imbalance in target column
        if target_col and target_col in df.columns:
            self._check_class_imbalance(df, target_col, report)

        # Check all low-cardinality object columns for imbalance
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col == target_col:
                continue
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20:
                self._check_value_imbalance(df, col, report)

        # Check for severely skewed numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            self._check_numeric_skew(df, col, report)

        return report

    def _check_class_imbalance(self, df: pd.DataFrame, col: str, report: BiasReport):
        counts = df[col].value_counts(normalize=True)
        if len(counts) < 2:
            return
        ratio = counts.min() / counts.max()
        if ratio < 0.2:
            majority = counts.idxmax()
            minority = counts.idxmin()
            report.add_issue(BiasIssue(
                column=col,
                issue_type="Class Imbalance",
                description=(
                    f"Severe class imbalance detected. '{majority}' = {counts.max():.1%}, "
                    f"'{minority}' = {counts.min():.1%}."
                ),
                recommendation="Consider oversampling (SMOTE), undersampling, or class-weighted models.",
                context={"majority_class": str(majority), "majority_pct": f"{counts.max():.1%}",
                         "minority_class": str(minority), "minority_pct": f"{counts.min():.1%}"}
            ))

    def _check_value_imbalance(self, df: pd.DataFrame, col: str, report: BiasReport):
        counts = df[col].value_counts(normalize=True)
        if len(counts) < 2:
            return
        ratio = counts.min() / counts.max()
        if ratio < 0.1:
            report.add_issue(BiasIssue(
                column=col,
                issue_type="Value Imbalance",
                description=(
                    f"Highly imbalanced distribution in '{col}'. "
                    f"Dominant value: '{counts.idxmax()}' ({counts.max():.1%})."
                ),
                recommendation="Verify if this imbalance is expected; it may introduce model bias.",
                context={"dominant_value": str(counts.idxmax()), "pct": f"{counts.max():.1%}"}
            ))

    def _check_numeric_skew(self, df: pd.DataFrame, col: str, report: BiasReport):
        series = df[col].dropna()
        if len(series) < 10:
            return
        skew = series.skew()
        if abs(skew) > 2.0:
            direction = "right (positive)" if skew > 0 else "left (negative)"
            report.add_issue(BiasIssue(
                column=col,
                issue_type="Skewed Distribution",
                description=f"Column '{col}' is highly skewed ({direction}, skewness={skew:.2f}).",
                recommendation="Consider applying a log or square-root transformation.",
                context={"skewness": round(skew, 4)}
            ))
