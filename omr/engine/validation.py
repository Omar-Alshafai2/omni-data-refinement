import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ..models import ValidationReport, ValidationIssue


class ValidationEngine:
    """
    Validates a DataFrame against a user-defined schema with type and constraint checks.

    Schema format:
    {
        "column_name": {
            "type": int | float | str,   # Optional
            "not_null": True,             # Optional
            "min": 0,                     # Optional (numeric)
            "max": 120,                   # Optional (numeric)
            "regex": r"^[A-Z]",           # Optional (string)
            "allowed": ["A", "B", "C"],   # Optional (categorical)
        }
    }
    """

    def run(self, df: pd.DataFrame, schema: Dict[str, Any]) -> ValidationReport:
        report = ValidationReport()

        for col, rules in schema.items():
            if col not in df.columns:
                report.add_issue(ValidationIssue(
                    column=col,
                    rule="column_exists",
                    description=f"Column '{col}' defined in schema does not exist in dataset.",
                    failing_rows=len(df)
                ))
                continue

            series = df[col]

            # not_null check
            if rules.get("not_null"):
                nulls = series.isnull().sum()
                if nulls > 0:
                    report.add_issue(ValidationIssue(
                        column=col,
                        rule="not_null",
                        description=f"{nulls} null values found; column must be non-null.",
                        failing_rows=int(nulls)
                    ))

            # type check
            if "type" in rules:
                expected_type = rules["type"]
                if expected_type in (int, float):
                    non_numeric = pd.to_numeric(series, errors="coerce").isnull() & series.notnull()
                    count = int(non_numeric.sum())
                    if count > 0:
                        report.add_issue(ValidationIssue(
                            column=col,
                            rule="type",
                            description=f"{count} values are not {expected_type.__name__}.",
                            failing_rows=count
                        ))
                elif expected_type == str:
                    non_str = series.dropna().apply(lambda x: not isinstance(x, str))
                    count = int(non_str.sum())
                    if count > 0:
                        report.add_issue(ValidationIssue(
                            column=col,
                            rule="type",
                            description=f"{count} values are not strings.",
                            failing_rows=count
                        ))

            # min / max checks (numeric only)
            if "min" in rules and pd.api.types.is_numeric_dtype(series):
                failing = (series.dropna() < rules["min"])
                count = int(failing.sum())
                if count > 0:
                    report.add_issue(ValidationIssue(
                        column=col,
                        rule=f"min ({rules['min']})",
                        description=f"{count} values are below minimum allowed value of {rules['min']}.",
                        failing_rows=count
                    ))

            if "max" in rules and pd.api.types.is_numeric_dtype(series):
                failing = (series.dropna() > rules["max"])
                count = int(failing.sum())
                if count > 0:
                    report.add_issue(ValidationIssue(
                        column=col,
                        rule=f"max ({rules['max']})",
                        description=f"{count} values exceed maximum allowed value of {rules['max']}.",
                        failing_rows=count
                    ))

            # allowed values check
            if "allowed" in rules:
                invalid = ~series.dropna().isin(rules["allowed"])
                count = int(invalid.sum())
                if count > 0:
                    report.add_issue(ValidationIssue(
                        column=col,
                        rule="allowed_values",
                        description=f"{count} values not in allowed set: {rules['allowed']}.",
                        failing_rows=count
                    ))

            # regex check
            if "regex" in rules and series.dtype == object:
                import re
                pattern = re.compile(rules["regex"])
                failing = series.dropna().apply(lambda x: not bool(pattern.match(str(x))))
                count = int(failing.sum())
                if count > 0:
                    report.add_issue(ValidationIssue(
                        column=col,
                        rule=f"regex ({rules['regex']})",
                        description=f"{count} values do not match the required pattern.",
                        failing_rows=count
                    ))

        return report
