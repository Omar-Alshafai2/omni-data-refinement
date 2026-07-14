"""
OMR Health Module — Calculates overall dataset health.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class Issue:
    column: str
    issue_type: str
    severity: str
    description: str
    recommendation: str

@dataclass
class HealthReport:
    score: float
    completeness: float
    uniqueness: float
    consistency: float
    validity: float
    conformity: float
    issues: List[Issue] = field(default_factory=list)


class HealthEngine:
    """
    Computes a real 0-100 data quality score using five pillars.
    """

    def run(self, df: pd.DataFrame) -> HealthReport:
        if df.empty:
            return HealthReport(100.0, 100.0, 100.0, 100.0, 100.0, 100.0, [])
            
        total_cells = df.size
        num_rows = len(df)
        num_cols = len(df.columns)
        
        issues = []

        # 1. Completeness
        missing_cells = df.isnull().sum().sum()
        completeness = round((1 - missing_cells / total_cells) * 100, 1) if total_cells > 0 else 100.0
        
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                issues.append(Issue(
                    column=col,
                    issue_type="Missing Values",
                    severity="High" if missing / num_rows > 0.2 else "Medium",
                    description=f"Found {missing} missing values ({(missing/num_rows)*100:.1f}%).",
                    recommendation="Impute missing values using mode/median or drop rows if sparse."
                ))

        # 2. Uniqueness
        dup_count = df.duplicated().sum()
        uniqueness = round((1 - dup_count / num_rows) * 100, 1) if num_rows > 0 else 100.0
        if dup_count > 0:
            issues.append(Issue(
                column="Dataset",
                issue_type="Duplicate Rows",
                severity="High",
                description=f"Found {dup_count} exact duplicate rows.",
                recommendation="Remove duplicate rows to prevent data leakage/overfitting."
            ))

        # 3. Consistency
        mixed_cols = 0
        for col in df.columns:
            if df[col].dtype == object:
                types = df[col].dropna().apply(type).unique()
                if len(types) > 1:
                    mixed_cols += 1
                    issues.append(Issue(
                        column=col,
                        issue_type="Mixed Data Types",
                        severity="High",
                        description=f"Column contains mixed types: {[t.__name__ for t in types]}.",
                        recommendation="Cast column to a uniform type (usually str)."
                    ))
        consistency = round((1 - mixed_cols / num_cols) * 100, 1) if num_cols > 0 else 100.0

        # 4. Validity
        invalid_cols = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue # Prevent division by zero logic issues. 
            outlier_pct = ((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).mean()
            if outlier_pct > 0.10:
                invalid_cols += 1
                issues.append(Issue(
                    column=col,
                    issue_type="Validity Risk",
                    severity="Medium",
                    description=f"Over 10% ({outlier_pct*100:.1f}%) of values are extreme outliers.",
                    recommendation="Investigate data collection pipeline or apply capping."
                ))
        validity = round((1 - invalid_cols / len(numeric_cols)) * 100, 1) if len(numeric_cols) > 0 else 100.0

        # 5. Conformity
        suspicious = 0
        object_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in object_cols:
            sample = df[col].dropna().head(50)
            if sample.empty:
                continue
            try:
                pd.to_numeric(sample, errors="raise")
                suspicious += 1
                issues.append(Issue(
                    column=col,
                    issue_type="Type Conformity",
                    severity="Medium",
                    description="Column is object/string but appears to contain numeric data.",
                    recommendation="Cast to float or integer."
                ))
            except (ValueError, TypeError):
                pass
        conformity = round((1 - suspicious / len(object_cols)) * 100, 1) if len(object_cols) > 0 else 100.0

        # Overall: weighted average
        score = (completeness * 0.30 + uniqueness * 0.20 + 
                 consistency * 0.20 + validity * 0.20 + conformity * 0.10)

        return HealthReport(
            score=round(score, 1),
            completeness=completeness,
            uniqueness=uniqueness,
            consistency=consistency,
            validity=validity,
            conformity=conformity,
            issues=issues
        )
