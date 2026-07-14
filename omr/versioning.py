"""
OMR Versioning System.

Tracks every transformation applied to the dataset as a named snapshot,
allowing rollback to any previous version.
"""
import pandas as pd
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


class DatasetVersion:
    """A named snapshot of the dataset at a specific point."""
    def __init__(self, version_id: int, name: str, description: str, df: pd.DataFrame):
        self.version_id = version_id
        self.name = name
        self.description = description
        self.df = df.copy()
        self.num_rows = len(df)
        self.num_cols = len(df.columns)


class VersioningSystem:
    """
    Manages dataset snapshots, enabling version history, diff comparisons, and rollback.
    """

    def __init__(self):
        self._versions: List[DatasetVersion] = []
        self._counter = 0

    def snapshot(self, df: pd.DataFrame, name: str = "", description: str = "") -> int:
        """Save a snapshot of the current DataFrame. Returns the version ID."""
        self._counter += 1
        if not name:
            name = f"v{self._counter}"
        version = DatasetVersion(
            version_id=self._counter,
            name=name,
            description=description,
            df=df
        )
        self._versions.append(version)
        return self._counter

    def rollback(self, version_id: int) -> pd.DataFrame:
        """Restore the DataFrame to a specific version."""
        for v in self._versions:
            if v.version_id == version_id:
                console.print(f"[bold cyan]Rolled back to version {version_id}: '{v.name}'[/bold cyan]")
                return v.df.copy()
        raise ValueError(f"Version {version_id} not found. Available: {[v.version_id for v in self._versions]}")

    def history(self):
        """Display the full version history as a table."""
        if not self._versions:
            console.print("[dim]No versions recorded yet.[/dim]")
            return

        table = Table(box=box.ROUNDED, border_style="cyan", show_header=True, header_style="bold cyan")
        table.add_column("ID",          width=4, style="dim")
        table.add_column("Name",        style="bold cyan")
        table.add_column("Rows",        justify="right")
        table.add_column("Columns",     justify="right")
        table.add_column("Description", style="white")

        for v in self._versions:
            table.add_row(str(v.version_id), v.name, str(v.num_rows), str(v.num_cols), v.description)

        console.print(table)

    def compare(self, version_a: int, version_b: int):
        """Print a diff summary between two versions."""
        va = self._get(version_a)
        vb = self._get(version_b)

        rows_diff = vb.num_rows - va.num_rows
        cols_diff = vb.num_cols - va.num_cols

        console.print(f"\n[bold]Comparing v{version_a} ('{va.name}') → v{version_b} ('{vb.name}')[/bold]\n")
        console.print(f"  Rows:    {va.num_rows} → {vb.num_rows}  ({'+' if rows_diff >= 0 else ''}{rows_diff})")
        console.print(f"  Columns: {va.num_cols} → {vb.num_cols}  ({'+' if cols_diff >= 0 else ''}{cols_diff})")

        # Missing value comparison
        ma = va.df.isnull().sum().sum()
        mb = vb.df.isnull().sum().sum()
        mv_diff = mb - ma
        console.print(f"  Missing: {ma} → {mb}  ({'+' if mv_diff >= 0 else ''}{mv_diff})")
        console.print()

    def _get(self, version_id: int) -> DatasetVersion:
        for v in self._versions:
            if v.version_id == version_id:
                return v
        raise ValueError(f"Version {version_id} not found.")
