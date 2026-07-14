"""
OMR Monitoring Module — Monitor datasets for drift and quality decay.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd


@dataclass
class Alert:
    check: str
    severity: str
    message: str


class Monitor:
    """Watches a dataset over time for changes in quality or distribution."""
    
    def __init__(self):
        self.reference_dataset = None
        self.schema = None
        self.alerts: List[Alert] = []
        
    def watch(self, dataset) -> None:
        """Sets the baseline dataset for future comparisons."""
        self.reference_dataset = dataset
        self.alerts = []
        
    def set_schema(self, schema: dict) -> None:
        """Sets a schema for validation checks."""
        self.schema = schema
        
    def check(self, df: pd.DataFrame) -> List[Alert]:
        """Runs checks against the baseline and returns any alerts."""
        self.alerts = []
        if self.reference_dataset is None:
            return self.alerts
            
        # 1. Row count anomaly
        ref_rows = self.reference_dataset.metadata.num_rows
        curr_rows = len(df)
        if ref_rows > 0:
            diff = abs(curr_rows - ref_rows) / ref_rows
            if diff > 0.5:
                self.alerts.append(Alert(
                    "Volume Drop/Spike", "High",
                    f"Row count changed by {diff*100:.1f}% ({ref_rows} -> {curr_rows})"
                ))
                
        # 2. Schema check (if configured)
        if self.schema:
            from ..core.dataset import Dataset
            ds = Dataset(df)
            report = ds._validation_engine.run(df, self.schema)
            for issue in report.issues:
                self.alerts.append(Alert(
                    "Schema Violation", "High",
                    f"{issue.column}: {issue.description}"
                ))
                
        # 3. Quick drift check on numeric columns
        import numpy as np
        num_cols = df.select_dtypes(include=[np.number]).columns
        ref_df = self.reference_dataset._df
        for col in num_cols:
            if col in ref_df.columns:
                ref_mean = ref_df[col].mean()
                curr_mean = df[col].mean()
                if ref_mean != 0:
                    drift = abs(curr_mean - ref_mean) / abs(ref_mean)
                    if drift > 0.2:
                        self.alerts.append(Alert(
                            "Mean Shift", "Medium",
                            f"Column '{col}' mean shifted by {drift*100:.1f}%"
                        ))
                        
        return self.alerts
