import pandas as pd
from ..models import DiagnosticReport, Issue, ChangeLog


# Values that represent missing data but may not be NaN
MISSING_INDICATORS = {"", "n/a", "na", "null", "none", "unknown", "nan", "missing", "?", "-"}


class CleaningEngine:
    """
    Automatically resolves issues identified in the DiagnosticReport.
    Tracks every transformation in a ChangeLog for full explainability.
    """

    def run(self, df: pd.DataFrame, report: DiagnosticReport, changelog: ChangeLog) -> pd.DataFrame:
        cleaned_df = df.copy()

        # 1. Normalize string-based missing values before anything else
        cleaned_df = self._normalize_missing(cleaned_df, changelog)

        # 2. Drop duplicates first (reduces row count for subsequent operations)
        for issue in report.issues:
            if issue.issue_type == "Duplicate Rows":
                before = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                dropped = before - len(cleaned_df)
                changelog.log("Dataset", "Dropped duplicate rows", dropped, "Exact duplicate rows detected")

        # 3. Impute missing values, fix types
        for issue in report.issues:
            col = issue.column
            if col not in cleaned_df.columns:
                continue

            if issue.issue_type == "Missing Values":
                missing_before = cleaned_df[col].isnull().sum()
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    median_val = cleaned_df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                    changelog.log(col, f"Imputed with median ({median_val:.4g})", int(missing_before),
                                  "Numeric column had missing values")
                else:
                    cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                    changelog.log(col, "Imputed with 'Unknown'", int(missing_before),
                                  "Categorical column had missing values")

            elif issue.issue_type == "Mixed Data Types":
                cleaned_df[col] = cleaned_df[col].astype(str)
                changelog.log(col, "Cast to string", len(cleaned_df),
                              "Column contained mixed data types")

        # 4. Auto-fix automatic type conversions (string numbers → numeric, dates)
        cleaned_df = self._auto_type_convert(cleaned_df, changelog)

        return cleaned_df

    def _normalize_missing(self, df: pd.DataFrame, changelog: ChangeLog) -> pd.DataFrame:
        """Replace common missing-value placeholders with real NaN."""
        for col in df.columns:
            if df[col].dtype == object:
                mask = df[col].apply(
                    lambda x: str(x).strip().lower() in MISSING_INDICATORS if pd.notna(x) else False
                )
                count = mask.sum()
                if count > 0:
                    df = df.copy()
                    df.loc[mask, col] = None
                    changelog.log(col, "Normalized missing value placeholders → NaN", int(count),
                                  f"Found {count} values like 'N/A', 'null', '' etc.")
        return df

    def _auto_type_convert(self, df: pd.DataFrame, changelog: ChangeLog) -> pd.DataFrame:
        """Attempt to convert object columns that contain numbers or dates."""
        for col in df.columns:
            if df[col].dtype == object:
                # Try numeric conversion
                try:
                    converted = pd.to_numeric(df[col], errors="raise")
                    df[col] = converted
                    changelog.log(col, "Auto-converted to numeric", len(df),
                                  "Column contained string-encoded numbers")
                    continue
                except (ValueError, TypeError):
                    pass

                # Try datetime conversion (only if many values look like dates)
                try:
                    sample = df[col].dropna().head(20)
                    converted = pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    changelog.log(col, "Auto-converted to datetime", len(df),
                                  "Column contained date-like strings")
                except (ValueError, TypeError):
                    pass
        return df
