"""
OMR Cleaning Module — Automatically resolves dataset issues.
"""
from typing import List
import pandas as pd
import numpy as np

from .strategies import MedianImputation, ModeImputation, ConstantImputation, resolve_mixed_types
from ..core.metadata import DatasetMetadata
from ..health.engine import HealthReport


class CleaningEngine:
    """
    Applies automatic fixes to issues detected by the HealthEngine.
    """

    def run(self, df: pd.DataFrame, report: HealthReport, metadata: DatasetMetadata) -> pd.DataFrame:
        """
        Cleans the DataFrame and updates the metadata's transformation log.
        """
        # Replace common missing value strings
        missing_strings = ["n/a", "N/A", "NA", "null", "Null", "NULL", "?", ""]
        df = df.replace(missing_strings, np.nan)

        for issue in report.issues:
            col = issue.column

            if issue.issue_type == "Missing Values":
                missing_count = df[col].isnull().sum()
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = MedianImputation().apply(df[col])
                    metadata.log_transformation(
                        operation="Imputation (Median)",
                        column=col,
                        rows_affected=missing_count,
                        reason="Numeric missing values imputed with median."
                    )
                else:
                    df[col] = ConstantImputation("Unknown").apply(df[col])
                    metadata.log_transformation(
                        operation="Imputation (Constant)",
                        column=col,
                        rows_affected=missing_count,
                        reason="Categorical missing values imputed with 'Unknown'."
                    )

            elif issue.issue_type == "Mixed Data Types":
                affected = len(df)
                df[col] = resolve_mixed_types(df[col])
                metadata.log_transformation(
                    operation="Type Cast (String)",
                    column=col,
                    rows_affected=affected,
                    reason="Mixed types cast to uniform string."
                )

            elif issue.issue_type == "Type Conformity":
                affected = len(df)
                df[col] = pd.to_numeric(df[col], errors="coerce")
                metadata.log_transformation(
                    operation="Type Cast (Numeric)",
                    column=col,
                    rows_affected=affected,
                    reason="String column converted to numeric."
                )

        # Handle duplicates at the end
        if report.uniqueness < 100.0:
            dup_count = df.duplicated().sum()
            df = df.drop_duplicates().reset_index(drop=True)
            metadata.log_transformation(
                operation="Row Deletion",
                column="Dataset",
                rows_affected=dup_count,
                reason="Exact duplicate rows removed."
            )

        return df
