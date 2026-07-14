"""
OMR Core — The primary Dataset interface.
Replaces OMRAssistant. Connects all 12 modules.
"""
from typing import Dict, Any
import pandas as pd

from .metadata import DatasetMetadata
from ..health.engine import HealthEngine, HealthReport
from ..cleaning.engine import CleaningEngine
from ..profiling.engine import ProfilingEngine
from ..validation.engine import ValidationEngine
from ..statistics.engine import StatisticsEngine
from ..drift.engine import DriftEngine
from ..explainability.engine import ExplainabilityEngine
from ..versioning.system import VersioningSystem
from ..reports.generator import ReportGenerator
from ..pipelines.pipeline import Pipeline
from ..integrations.polars import ensure_pandas
from ..utils.formatting import (
    display_summary, display_health, display_profile, display_validation,
    display_statistics, display_drift, display_changelog
)


class Dataset:
    """
    The primary OMR interface.
    Represents a dataset with built-in quality, validation, and profiling intelligence.
    """

    def __init__(self, data: Any):
        """
        Initializes the Dataset.
        Supports Pandas and Polars DataFrames.
        """
        self._df = ensure_pandas(data).copy()
        self.metadata = DatasetMetadata.from_dataframe(self._df)
        
        # Engines
        self._health_engine       = HealthEngine()
        self._cleaning_engine     = CleaningEngine()
        self._profiling_engine    = ProfilingEngine()
        self._validation_engine   = ValidationEngine()
        self._statistics_engine   = StatisticsEngine()
        self._drift_engine        = DriftEngine()
        self._explain_engine      = ExplainabilityEngine()
        self._version_system      = VersioningSystem()
        self._report_generator    = ReportGenerator()
        
        # State
        self._last_health = None
        self._last_profile = None
        
        # Save initial version
        self._version_system.snapshot(self._df, self.metadata, name="v0 — Initial")

    # ── 1. Core Health & Profiling ───────────────────────────────────────────

    def summary(self) -> "Dataset":
        """Prints a one-line health summary."""
        # Fast compute just the score without deep inspection
        health = self._health_engine.run(self._df)
        display_summary(self.metadata, health.score)
        return self

    def health(self) -> HealthReport:
        """Runs the 5-pillar health check and returns a HealthReport."""
        report = self._health_engine.run(self._df)
        self._last_health = report
        display_health(report)
        return report

    def profile(self) -> "Dataset":
        """Generates a full statistical profile."""
        report = self._profiling_engine.run(self._df)
        self._last_profile = report
        display_profile(report)
        return self

    # ── 2. Cleaning ──────────────────────────────────────────────────────────

    def clean(self) -> "Dataset":
        """Auto-resolves all issues detected by health()."""
        if self._last_health is None:
            self._last_health = self._health_engine.run(self._df)
            
        self._df = self._cleaning_engine.run(self._df, self._last_health, self.metadata)
        
        # Update metadata shape after cleaning (e.g. dropped rows)
        self.metadata.num_rows = len(self._df)
        self.metadata.num_cols = len(self._df.columns)
        self.metadata.missing_cells = int(self._df.isnull().sum().sum())
        
        self._version_system.snapshot(self._df, self.metadata, name="After clean()")
        
        # Clear health state so next call re-evaluates
        self._last_health = None
        return self

    def explain_changes(self) -> "Dataset":
        """Displays the transformation history."""
        display_changelog(self.metadata)
        return self

    # ── 3. Validation & Rules ────────────────────────────────────────────────

    def validate(self, schema: Dict[str, Any]) -> "Dataset":
        """Validates the dataset against a schema of ConstraintTypes."""
        report = self._validation_engine.run(self._df, schema)
        display_validation(report)
        return self

    # ── 4. Advanced Analytics ────────────────────────────────────────────────

    def analyze(self) -> "Dataset":
        """Runs deep statistical analysis (outliers, skew, multicollinearity)."""
        report = self._statistics_engine.run(self._df)
        display_statistics(report)
        return self

    def compare(self, other: "Dataset") -> "Dataset":
        """Detects distribution drift against another dataset (e.g. production)."""
        report = self._drift_engine.run(self._df, other._df)
        display_drift(report)
        return self

    def explain(self, issue: str) -> dict:
        """Explains a data quality issue conceptually and offers fixes."""
        exp = self._explain_engine.run(issue)
        from ..utils.formatting import console
        from rich.panel import Panel
        from rich.text import Text
        t = Text()
        t.append("Definition: ", style="bold cyan"); t.append(f"{exp['definition']}\n\n")
        t.append("Why it matters: ", style="bold yellow"); t.append(f"{exp['why_it_matters']}\n\n")
        t.append("Risks: ", style="bold red"); t.append(f"{exp['risks']}\n\n")
        t.append("Recommended fixes:\n", style="bold green"); t.append(exp['recommended_fixes'])
        console.print(Panel(t, title=f"Explainability — {issue.title()}", border_style="cyan"))
        return exp

    # ── 5. Ops & Versioning ──────────────────────────────────────────────────

    def pipeline(self) -> Pipeline:
        """Returns a fluent Pipeline interface."""
        return Pipeline(self)

    def report(self, format: str = "html", path: str = "omr_report") -> str:
        """Generates a physical report file."""
        if not self._last_health:
            self._last_health = self._health_engine.run(self._df)
        return self._report_generator.generate(self, format=format, path=path)

    def snapshot(self, name: str = "", description: str = "") -> int:
        """Saves a version snapshot."""
        return self._version_system.snapshot(self._df, self.metadata, name, description)

    def rollback(self, version_id: int) -> "Dataset":
        """Restores a previous snapshot."""
        self._df, self.metadata = self._version_system.rollback(version_id)
        self._last_health = None
        self._last_profile = None
        return self

    def export(self) -> pd.DataFrame:
        """Returns the underlying pandas DataFrame."""
        return self._df.copy()
