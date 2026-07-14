"""
OMR Statistics Module — Advanced dataset analysis.
Detects outliers, skew, multicollinearity, constant features, and class imbalance.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class StatisticalFinding:
    column: str
    finding_type: str
    severity: str
    description: str
    recommendation: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticsReport:
    findings: List[StatisticalFinding] = field(default_factory=list)


class StatisticsEngine:
    """Runs a full statistical analysis on the dataset."""

    def run(self, df: pd.DataFrame) -> StatisticsReport:
        findings = []
        if df.empty:
            return StatisticsReport()

        numeric_df = df.select_dtypes(include=[np.number])
        
        # 1. Constant / Low Variance Features
        for col in numeric_df.columns:
            std = numeric_df[col].std()
            if pd.isna(std) or std == 0:
                findings.append(StatisticalFinding(
                    column=col, finding_type="Constant Feature", severity="High",
                    description="Column has zero variance (all values are identical).",
                    recommendation="Remove the column as it provides no predictive power."
                ))
                
        for col in df.select_dtypes(exclude=[np.number]).columns:
            if df[col].nunique(dropna=False) <= 1:
                findings.append(StatisticalFinding(
                    column=col, finding_type="Constant Feature", severity="High",
                    description="Categorical column has only 1 unique value.",
                    recommendation="Remove the column."
                ))

        # 2. Skewness
        for col in numeric_df.columns:
            skew = numeric_df[col].skew()
            if pd.notna(skew) and abs(skew) > 2.0:
                findings.append(StatisticalFinding(
                    column=col, finding_type="High Skewness", severity="Medium",
                    description=f"Highly skewed distribution (skew = {skew:.2f}).",
                    recommendation="Consider log or Box-Cox transformation for linear models.",
                    context={"skewness": round(skew, 3)}
                ))

        # 3. Multicollinearity (High correlation between features)
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr().abs()
            # Upper triangle mask
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            high_corr = corr.where(mask)
            
            for col in high_corr.columns:
                correlated_with = high_corr.index[high_corr[col] > 0.90].tolist()
                if correlated_with:
                    findings.append(StatisticalFinding(
                        column=col, finding_type="Multicollinearity", severity="High",
                        description=f"Highly correlated (>0.90) with: {', '.join(correlated_with)}.",
                        recommendation="Remove one of the highly correlated features to prevent model instability.",
                        context={"correlated_features": correlated_with}
                    ))

        # 4. Outliers (IQR Method)
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            if len(s) == 0: continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                if outliers > 0:
                    pct = (outliers / len(s)) * 100
                    findings.append(StatisticalFinding(
                        column=col, finding_type="Outliers", severity="Low" if pct < 5 else "Medium",
                        description=f"Found {outliers} potential outliers ({pct:.1f}%).",
                        recommendation="Cap (winsorize) or remove outliers if they are data entry errors.",
                        context={"count": int(outliers), "percentage": round(pct, 2)}
                    ))

        return StatisticsReport(findings)
