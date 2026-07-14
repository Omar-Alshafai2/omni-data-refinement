import pandas as pd
from ..models import DiagnosticReport, Issue

class DiagnosticsEngine:
    """
    Analyzes a DataFrame for common data quality issues.
    This is the core of the "one-command dataset diagnosis".
    """
    
    def run(self, df: pd.DataFrame) -> DiagnosticReport:
        """
        Executes all diagnostic checks.
        """
        report = DiagnosticReport(num_rows=len(df), num_cols=len(df.columns))
        
        self._check_missing_values(df, report)
        self._check_duplicates(df, report)
        self._check_mixed_types(df, report)
        
        return report
        
    def _check_missing_values(self, df: pd.DataFrame, report: DiagnosticReport):
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                pct = (count / len(df)) * 100
                severity = "High" if pct > 30 else "Medium" if pct > 5 else "Low"
                report.add_issue(
                    Issue(
                        issue_type="Missing Values",
                        column=col,
                        severity=severity,
                        description=f"{count} missing values ({pct:.2f}% of column).",
                        recommendation="Consider mean/median imputation or removing rows depending on the context.",
                        context={"count": count, "percentage": round(pct, 2)}
                    )
                )

    def _check_duplicates(self, df: pd.DataFrame, report: DiagnosticReport):
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            pct = (dup_count / len(df)) * 100
            severity = "High" if pct > 10 else "Medium"
            report.add_issue(
                Issue(
                    issue_type="Duplicate Rows",
                    column="Dataset",
                    severity=severity,
                    description=f"Found {dup_count} exact duplicate rows ({pct:.2f}% of dataset).",
                    recommendation="Remove exact duplicates unless expected by business logic.",
                    context={"count": dup_count, "percentage": round(pct, 2)}
                )
            )

    def _check_mixed_types(self, df: pd.DataFrame, report: DiagnosticReport):
        # A simple heuristic for mixed types in object columns
        for col in df.columns:
            if df[col].dtype == 'object':
                types = df[col].dropna().apply(type).unique()
                if len(types) > 1:
                    type_names = [t.__name__ for t in types]
                    report.add_issue(
                        Issue(
                            issue_type="Mixed Data Types",
                            column=col,
                            severity="Medium",
                            description=f"Column contains mixed types: {', '.join(type_names)}.",
                            recommendation="Convert column to a uniform type (e.g., string or numeric).",
                            context={"types": type_names}
                        )
                    )
