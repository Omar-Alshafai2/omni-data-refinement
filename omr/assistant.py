import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

from .engine.diagnostics import DiagnosticsEngine
from .engine.cleaning import CleaningEngine
from .engine.profiling import ProfilingEngine
from .engine.validation import ValidationEngine
from .engine.outliers import OutliersEngine
from .engine.bias import BiasEngine
from .engine.leakage import LeakageEngine
from .engine.drift import DriftEngine
from .engine.quality import QualityEngine, CorrelationEngine
from .versioning import VersioningSystem
from .reporting import ReportingEngine
from .models import (
    DiagnosticReport, ProfilingReport, ValidationReport,
    ChangeLog, OutlierResult, BiasReport, LeakageReport, DriftReport
)
from .utils.formatting import console, print_changelog


class OMRAssistant:
    """
    OMRAssistant — Your intelligent data companion.

    The complete data intelligence pipeline:

        assistant = OMRAssistant(df)

        # Understand your data
        assistant.summary()           # Quick one-liner health check
        assistant.profile()           # Full statistical profile
        assistant.quality_score()     # 0-100 quality score with 5-pillar breakdown
        assistant.correlations()      # Feature correlation matrix

        # Clean your data
        assistant.diagnose()          # Detect all issues
        assistant.fix()               # Auto-resolve all issues
        assistant.explain()           # See exactly what changed

        # Validate your data
        assistant.validate(schema)    # Schema + business rule checks

        # Detect advanced issues
        assistant.detect_outliers()   # IQR or Z-Score outlier detection
        assistant.remove_outliers()   # Cap, remove, or flag outliers
        assistant.detect_bias()       # Class imbalance + skew detection
        assistant.detect_leakage()    # Data leakage risk analysis
        assistant.detect_drift()      # Training vs production drift

        # Version and report
        assistant.snapshot()          # Save a version checkpoint
        assistant.rollback()          # Restore a previous version
        assistant.history()           # View all saved versions
        assistant.report()            # Generate HTML/JSON/Markdown report

        # Export
        clean_df = assistant.export()
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the assistant with a pandas DataFrame.

        Args:
            df (pd.DataFrame): The raw dataset to analyze and refine.

        Raises:
            TypeError: If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("OMRAssistant expects a pandas DataFrame.")

        # Work on a copy — the original is never mutated until export()
        self._df = df.copy()

        # Engines
        self._diagnostics_engine = DiagnosticsEngine()
        self._cleaning_engine    = CleaningEngine()
        self._profiling_engine   = ProfilingEngine()
        self._validation_engine  = ValidationEngine()
        self._outliers_engine    = OutliersEngine()
        self._bias_engine        = BiasEngine()
        self._leakage_engine     = LeakageEngine()
        self._drift_engine       = DriftEngine()
        self._quality_engine     = QualityEngine()
        self._corr_engine        = CorrelationEngine()
        self._reporting_engine   = ReportingEngine()
        self._versioning         = VersioningSystem()

        # State
        self._last_report:     Optional[DiagnosticReport]  = None
        self._last_profile:    Optional[ProfilingReport]   = None
        self._last_validation: Optional[ValidationReport]  = None
        self._last_bias:       Optional[BiasReport]        = None
        self._last_leakage:    Optional[LeakageReport]     = None
        self._last_scores:     Optional[dict]              = None
        self._changelog:       ChangeLog                   = ChangeLog()

        # Save initial snapshot
        self._versioning.snapshot(self._df, name="v0 — Initial",
                                  description="Original dataset as loaded.")

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "OMRAssistant":
        """
        Load a CSV file and initialize the assistant directly.

        Args:
            path (str): Path to the CSV file.
            **kwargs: Any additional arguments passed to pandas.read_csv().

        Returns:
            OMRAssistant: A ready-to-use assistant instance.

        Example:
            assistant = OMRAssistant.from_csv("data.csv")
        """
        df = pd.read_csv(path, **kwargs)
        console.print(f"[bold green]Loaded {len(df):,} rows x {len(df.columns)} columns from '{path}'[/bold green]")
        return cls(df)

    @classmethod
    def from_excel(cls, path: str, sheet_name=0, **kwargs) -> "OMRAssistant":
        """
        Load an Excel file and initialize the assistant directly.

        Args:
            path (str): Path to the Excel file (.xlsx or .xls).
            sheet_name: Sheet name or index (default: 0).
            **kwargs: Any additional arguments passed to pandas.read_excel().

        Returns:
            OMRAssistant: A ready-to-use assistant instance.

        Example:
            assistant = OMRAssistant.from_excel("data.xlsx", sheet_name="Sheet1")
        """
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        console.print(f"[bold green]Loaded {len(df):,} rows x {len(df.columns)} columns from '{path}'[/bold green]")
        return cls(df)

    # ─────────────────────────────────────────────────────────────────────────
    # Understanding — Know Your Data
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> "OMRAssistant":
        """
        Prints an instant one-line health summary of the dataset.
        Great for a quick sanity check after loading data.

        Example output:
            [Dataset] 10,000 rows x 15 cols | Missing: 234 (1.6%) |
            Duplicates: 12 | Mixed Types: 2 cols | Quality: 91/100
        """
        rows = len(self._df)
        cols = len(self._df.columns)
        missing = int(self._df.isnull().sum().sum())
        missing_pct = (missing / (rows * cols) * 100) if rows * cols > 0 else 0
        duplicates = int(self._df.duplicated().sum())

        mixed = sum(
            1 for col in self._df.columns
            if self._df[col].dtype == object and len(self._df[col].dropna().apply(type).unique()) > 1
        )

        scores = self._quality_engine.run(self._df)
        overall = scores["Overall"]
        q_color = "green" if overall >= 85 else "yellow" if overall >= 60 else "red"

        from rich.text import Text
        line = Text()
        line.append("[Dataset] ", style="bold cyan")
        line.append(f"{rows:,} rows × {cols} cols", style="white")
        line.append("  |  Missing: ", style="dim")
        miss_color = "green" if missing == 0 else "yellow" if missing_pct < 5 else "red"
        line.append(f"{missing} ({missing_pct:.1f}%)", style=miss_color)
        line.append("  |  Duplicates: ", style="dim")
        line.append(str(duplicates), style="green" if duplicates == 0 else "yellow")
        line.append("  |  Mixed Types: ", style="dim")
        line.append(f"{mixed} col(s)", style="green" if mixed == 0 else "yellow")
        line.append("  |  Quality: ", style="dim")
        line.append(f"{overall}/100", style=f"bold {q_color}")

        console.print(line)
        return self

    def profile(self) -> "OMRAssistant":
        """
        Generates and displays a full statistical profile of the dataset.

        Covers per-column: dtype, missing count & %, unique count,
        mean, median, std, min, max, and top values for categoricals.
        """
        report = self._profiling_engine.run(self._df)
        self._last_profile = report
        report.display()
        return self

    def quality_score(self) -> "OMRAssistant":
        """
        Computes and displays a 0-100 data quality score broken down
        across 5 pillars: Completeness, Uniqueness, Consistency,
        Validity, and Conformity.

        Returns self for method chaining.
        """
        scores = self._quality_engine.run(self._df)
        self._last_scores = scores
        self._quality_engine.display(scores)
        return self

    def correlations(self, target_col: str = None, threshold: float = 0.3) -> "OMRAssistant":
        """
        Displays the correlation matrix for all numeric features.
        If target_col is provided, ranks features by their correlation with it.

        Args:
            target_col (str, optional): Target column to rank correlations against.
            threshold  (float): Minimum absolute correlation to display (default 0.3).

        Returns self for method chaining.
        """
        self._corr_engine.display(self._df, target_col=target_col, threshold=threshold)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Cleaning — Fix Your Data
    # ─────────────────────────────────────────────────────────────────────────

    def diagnose(self) -> "OMRAssistant":
        """
        Runs a comprehensive diagnosis on the dataset.
        Detects: missing values, duplicates, and mixed data types.
        Prints a beautiful, actionable report to the terminal.
        """
        report = self._diagnostics_engine.run(self._df)
        self._last_report = report
        report.display()
        return self

    def fix(self) -> "OMRAssistant":
        """
        Automatically resolves all issues detected in the last diagnose().
        Runs diagnose() automatically first if not already done.

        What it fixes:
          - Normalizes missing-value placeholders (N/A, null, ?, etc.) → NaN
          - Drops exact duplicate rows
          - Imputes numeric missing values with the column median
          - Imputes categorical missing values with 'Unknown'
          - Casts mixed-type columns uniformly to strings
          - Auto-converts string-encoded numbers and dates
        """
        if self._last_report is None:
            console.print("[bold]Running diagnose() first...[/bold]")
            self.diagnose()

        console.print("[bold cyan]Applying automated fixes...[/bold cyan]")
        self._df = self._cleaning_engine.run(self._df, self._last_report, self._changelog)

        self._versioning.snapshot(
            self._df,
            name=f"v{self._versioning._counter} — After fix()",
            description=f"Auto-cleaned. {len(self._changelog.entries)} transformation(s) applied."
        )

        self._last_report = None
        console.print("\n[bold]Done! Updated diagnostic report:[/bold]\n")
        self.diagnose()
        return self

    def explain(self) -> "OMRAssistant":
        """
        Displays a full change log of every transformation applied by fix().
        Shows: step number, column, action taken, rows affected, and reason.
        """
        print_changelog(self._changelog)
        return self

    def clean(self) -> "OMRAssistant":
        """
        Convenience shortcut: runs diagnose() then fix() in one call.
        """
        return self.diagnose().fix()

    # ─────────────────────────────────────────────────────────────────────────
    # Validation — Enforce Rules
    # ─────────────────────────────────────────────────────────────────────────

    def validate(self, schema: Dict[str, Any]) -> "OMRAssistant":
        """
        Validates the dataset against a user-defined schema.

        Schema format:
            {
                "column_name": {
                    "type": int | float | str,   # Type check
                    "not_null": True,             # No missing values allowed
                    "min": 0,                     # Minimum numeric value
                    "max": 120,                   # Maximum numeric value
                    "allowed": ["A", "B", "C"],   # Allowed values
                    "regex": r"^\\d{4}-\\d{2}",  # Regex pattern match
                }
            }

        Args:
            schema (dict): Validation rules per column.
        """
        report = self._validation_engine.run(self._df, schema)
        self._last_validation = report
        report.display()
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Advanced Issue Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_outliers(self, method: str = "iqr") -> List[OutlierResult]:
        """
        Detects outliers in all numeric columns.

        Args:
            method (str): 'iqr' (default, robust for skewed data) or
                          'zscore' (best for normally distributed data).

        Returns:
            List[OutlierResult]: Detected outliers per column.
        """
        results = self._outliers_engine.detect(self._df, method=method)
        if not results:
            console.print("[bold green]No outliers detected.[/bold green]")
        else:
            console.print(f"[bold yellow]Detected outliers in {len(results)} column(s):[/bold yellow]\n")
            for r in results:
                vals = str(r.values[:5])[1:-1] + ("..." if len(r.values) > 5 else "")
                console.print(f"  • [cyan]{r.column}[/cyan]: {r.count} outlier(s) — [{vals}]")
            console.print(
                "\nRun [bold cyan]assistant.remove_outliers()[/bold cyan] to apply a fix strategy."
            )
        return results

    def remove_outliers(self, method: str = "iqr", strategy: str = "cap") -> "OMRAssistant":
        """
        Detects and fixes outliers in all numeric columns.

        Args:
            method   (str): 'iqr' or 'zscore'.
            strategy (str): How to handle detected outliers:
                            'cap'    — Winsorize (clip) to IQR fences (default)
                            'remove' — Drop rows containing outliers
                            'flag'   — Return info only, no changes applied
        """
        results = self._outliers_engine.detect(self._df, method=method)
        if not results:
            console.print("[bold green]No outliers to fix.[/bold green]")
            return self

        if strategy != "flag":
            before = len(self._df)
            self._df = self._outliers_engine.apply(self._df, results, strategy=strategy)
            after = len(self._df)
            for r in results:
                self._changelog.log(r.column, f"Outliers {strategy}ped ({method.upper()})",
                                    r.count, f"Outlier detected via {method.upper()}")
            self._versioning.snapshot(
                self._df,
                name=f"v{self._versioning._counter} — Outliers {strategy}ped",
                description=f"{len(results)} column(s) treated. Strategy: {strategy}."
            )
            console.print(
                f"[bold green]Applied '{strategy}' strategy to {len(results)} column(s). "
                f"Rows: {before} → {after}[/bold green]"
            )
        return self

    def detect_bias(self, target_col: str = None) -> "OMRAssistant":
        """
        Detects bias and fairness issues in the dataset.

        Checks for:
          - Class imbalance in the target column
          - Value distribution imbalance in low-cardinality columns
          - Severely skewed numeric columns (skewness > 2.0)

        Args:
            target_col (str, optional): The label/target column to check for class imbalance.
        """
        report = self._bias_engine.run(self._df, target_col=target_col)
        self._last_bias = report
        report.display()
        return self

    def detect_leakage(self, target_col: str, threshold: float = 0.80) -> "OMRAssistant":
        """
        Detects data leakage — features with suspiciously high correlation
        to the target column that could artificially inflate model performance.

        Args:
            target_col (str): The name of the target/label column.
            threshold  (float): Correlation threshold to flag as medium risk (default 0.80).
        """
        report = self._leakage_engine.run(self._df, target_col=target_col, threshold=threshold)
        self._last_leakage = report
        report.display()
        return self

    def detect_drift(self, production_df: pd.DataFrame, threshold_pct: float = 20.0) -> "OMRAssistant":
        """
        Compares the current (training) dataset to a production DataFrame
        and flags columns where the distribution has significantly shifted.

        Args:
            production_df  (pd.DataFrame): The production/inference dataset.
            threshold_pct  (float): % change in mean to trigger a warning (default 20%).
        """
        report = self._drift_engine.run(self._df, production_df, threshold_pct=threshold_pct)
        report.display()
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Versioning
    # ─────────────────────────────────────────────────────────────────────────

    def snapshot(self, name: str = "", description: str = "") -> int:
        """
        Manually saves a named checkpoint of the current dataset state.

        Args:
            name        (str): A human-readable name for this version.
            description (str): Optional description of what changed.

        Returns:
            int: The version ID assigned to this snapshot.
        """
        vid = self._versioning.snapshot(self._df, name=name, description=description)
        console.print(f"[bold green]Snapshot saved as version {vid}[/bold green]")
        return vid

    def rollback(self, version_id: int) -> "OMRAssistant":
        """
        Restores the dataset to a previously saved snapshot.

        Args:
            version_id (int): The version ID to restore (use history() to see IDs).
        """
        self._df = self._versioning.rollback(version_id)
        self._last_report    = None
        self._last_profile   = None
        self._last_validation = None
        return self

    def history(self) -> "OMRAssistant":
        """
        Displays a table of all saved dataset version snapshots.
        """
        self._versioning.history()
        return self

    def compare_versions(self, version_a: int, version_b: int) -> "OMRAssistant":
        """
        Prints a diff summary between two saved dataset versions.
        Shows row count change, column count change, and missing value delta.
        """
        self._versioning.compare(version_a, version_b)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────────

    def report(self, format: str = "html", path: str = "omr_report") -> str:
        """
        Generates a report file containing diagnostics, profile, and change log.

        Args:
            format (str): Output format — 'html' (default), 'markdown', or 'json'.
            path   (str): Output file path without extension (default: 'omr_report').

        Returns:
            str: The path to the generated report file.
        """
        if self._last_report is None:
            self.diagnose()
        if self._last_profile is None:
            self.profile()

        filepath = self._reporting_engine.generate(
            format=format,
            path=path,
            diagnostic_report=self._last_report,
            profile=self._last_profile,
            validation_report=self._last_validation,
            changelog=self._changelog
        )
        console.print(f"[bold green]Report saved to: {filepath}[/bold green]")
        return filepath

    def generate_report(self, format: str = "html", path: str = "omr_report") -> "OMRAssistant":
        """Chainable alias for report()."""
        self.report(format=format, path=path)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────

    def export(self) -> pd.DataFrame:
        """
        Returns a copy of the current (cleaned) DataFrame.

        Returns:
            pd.DataFrame: The refined dataset, ready for use.
        """
        return self._df.copy()
