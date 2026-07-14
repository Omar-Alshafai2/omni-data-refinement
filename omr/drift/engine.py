"""
OMR Drift Module — Detects distribution shifts between datasets.
Supports PSI (Population Stability Index), KS Test, and Jensen-Shannon Divergence.
"""
from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np


@dataclass
class DriftResult:
    column: str
    psi: float
    ks_statistic: float
    js_divergence: float
    severity: str


@dataclass
class DriftReport:
    results: List[DriftResult] = field(default_factory=list)


class DriftEngine:
    """Compares two datasets to detect statistical drift."""

    def run(self, df_train: pd.DataFrame, df_prod: pd.DataFrame) -> DriftReport:
        results = []
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in df_prod.columns:
                continue
                
            train_s = df_train[col].dropna()
            prod_s = df_prod[col].dropna()
            
            if len(train_s) == 0 or len(prod_s) == 0:
                continue

            psi_val = self._calculate_psi(train_s, prod_s)
            ks_stat = self._calculate_ks(train_s, prod_s)
            js_div = self._calculate_js(train_s, prod_s)

            # Determine severity primarily by PSI
            if psi_val >= 0.2:
                severity = "High"
            elif psi_val >= 0.1:
                severity = "Medium"
            else:
                severity = "Low" if ks_stat > 0.1 else "None"

            results.append(DriftResult(
                column=col,
                psi=psi_val,
                ks_statistic=ks_stat,
                js_divergence=js_div,
                severity=severity
            ))

        return DriftReport(results)

    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """Calculates Population Stability Index."""
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        
        # Ensure unique breakpoints
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        # Replace 0s to avoid division by zero or log(0)
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def _calculate_ks(self, expected: pd.Series, actual: pd.Series) -> float:
        """Calculates Kolmogorov-Smirnov statistic without scipy."""
        data1 = np.sort(expected)
        data2 = np.sort(actual)
        n1 = len(data1)
        n2 = len(data2)
        if n1 == 0 or n2 == 0: return 0.0
        
        data_all = np.concatenate([data1, data2])
        cdf1 = np.searchsorted(data1, data_all, side='right') / n1
        cdf2 = np.searchsorted(data2, data_all, side='right') / n2
        return float(np.max(np.abs(cdf1 - cdf2)))

    def _calculate_js(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """Calculates Jensen-Shannon Divergence."""
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        
        p, _ = np.histogram(expected, bins=bins, range=(min_val, max_val), density=True)
        q, _ = np.histogram(actual, bins=bins, range=(min_val, max_val), density=True)
        
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        return float(jsd)
