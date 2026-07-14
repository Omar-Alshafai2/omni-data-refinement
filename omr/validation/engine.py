"""
OMR Validation Module — Enforces schemas and business rules.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List
import pandas as pd

from ..schemas.types import ConstraintType


@dataclass
class ValidationIssue:
    column: str
    rule: str
    failing_rows: int
    description: str


@dataclass
class ValidationReport:
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)


class ValidationEngine:
    """
    Validates a dataset against a dictionary of schemas.
    """

    def run(self, df: pd.DataFrame, schema: Dict[str, Any]) -> ValidationReport:
        issues = []
        for col, constraint in schema.items():
            if col not in df.columns:
                issues.append(ValidationIssue(
                    column=col, rule="presence", failing_rows=len(df),
                    description="Column is missing from dataset."
                ))
                continue
                
            if isinstance(constraint, ConstraintType):
                # Use new ConstraintType objects
                errors = constraint.validate(df[col])
                for err in errors:
                    # Extract the failing count from the error string if possible, else 1
                    try:
                        count = int(err.split()[0])
                    except (ValueError, IndexError):
                        count = 1
                    issues.append(ValidationIssue(
                        column=col, rule=constraint.description(),
                        failing_rows=count, description=err
                    ))
            elif isinstance(constraint, dict):
                # Fallback to old dict format for backwards compatibility
                self._validate_legacy_dict(df[col], col, constraint, issues)
                
        return ValidationReport(passed=len(issues) == 0, issues=issues)

    def _validate_legacy_dict(self, series: pd.Series, col: str, rules: dict, issues: list):
        if rules.get("not_null", False):
            nulls = series.isnull().sum()
            if nulls > 0:
                issues.append(ValidationIssue(
                    col, "not_null", int(nulls),
                    f"Found {nulls} missing value(s)."
                ))
                
        non_null = series.dropna()
        if "type" in rules:
            exp_type = rules["type"]
            if exp_type in (int, float):
                numeric = pd.to_numeric(non_null, errors="coerce")
                bad = numeric.isnull().sum()
                if bad > 0:
                    issues.append(ValidationIssue(
                        col, "type", int(bad),
                        f"Found {bad} value(s) that cannot be cast to numeric."
                    ))
                elif exp_type == int:
                    non_int = (numeric % 1 != 0).sum()
                    if non_int > 0:
                        issues.append(ValidationIssue(
                            col, "type_int", int(non_int),
                            f"Found {non_int} value(s) with decimal parts."
                        ))

        if "min" in rules and pd.api.types.is_numeric_dtype(series):
            m = rules["min"]
            bad = (series < m).sum()
            if bad > 0:
                issues.append(ValidationIssue(
                    col, f"min ({m})", int(bad),
                    f"{bad} values are below minimum allowed value of {m}."
                ))

        if "max" in rules and pd.api.types.is_numeric_dtype(series):
            m = rules["max"]
            bad = (series > m).sum()
            if bad > 0:
                issues.append(ValidationIssue(
                    col, f"max ({m})", int(bad),
                    f"{bad} values exceed maximum allowed value of {m}."
                ))

        if "allowed" in rules:
            allowed = set(rules["allowed"])
            bad = ~non_null.isin(allowed)
            count = bad.sum()
            if count > 0:
                issues.append(ValidationIssue(
                    col, f"allowed ({len(allowed)} values)", int(count),
                    f"{count} values are not in the allowed list."
                ))
