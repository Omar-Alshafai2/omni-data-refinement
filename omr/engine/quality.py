import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


class QualityEngine:
    """
    Computes a real 0-100 data quality score using five pillars:
      - Completeness  : How free of missing values is the dataset?
      - Uniqueness    : How free of exact duplicates is it?
      - Consistency   : How free of mixed types and format violations is it?
      - Validity      : Are numeric values in sensible ranges?
      - Conformity    : Are column types clean and well-typed?
    """

    def run(self, df: pd.DataFrame) -> dict:
        scores = {}

        # 1. Completeness (missing value rate)
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        scores["Completeness"] = round((1 - missing_cells / total_cells) * 100, 1) if total_cells > 0 else 100.0

        # 2. Uniqueness (duplicate row rate)
        dup_count = df.duplicated().sum()
        scores["Uniqueness"] = round((1 - dup_count / len(df)) * 100, 1) if len(df) > 0 else 100.0

        # 3. Consistency (mixed type columns)
        mixed_cols = 0
        for col in df.columns:
            if df[col].dtype == object:
                types = df[col].dropna().apply(type).unique()
                if len(types) > 1:
                    mixed_cols += 1
        scores["Consistency"] = round((1 - mixed_cols / len(df.columns)) * 100, 1) if len(df.columns) > 0 else 100.0

        # 4. Validity (detect clearly impossible values in numeric cols)
        invalid_cols = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outlier_pct = ((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).mean()
            if outlier_pct > 0.10:  # >10% extreme outliers = validity issue
                invalid_cols += 1
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        scores["Validity"] = round((1 - invalid_cols / numeric_count) * 100, 1) if numeric_count > 0 else 100.0

        # 5. Conformity (are types clean — not all-object where numeric expected)
        suspicious = 0
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().head(50)
            try:
                pd.to_numeric(sample, errors="raise")
                suspicious += 1  # This is a string column that should be numeric
            except (ValueError, TypeError):
                pass
        object_count = len(df.select_dtypes(include=["object"]).columns)
        scores["Conformity"] = round((1 - suspicious / object_count) * 100, 1) if object_count > 0 else 100.0

        # Overall: weighted average
        weights = {
            "Completeness": 0.30,
            "Uniqueness":   0.20,
            "Consistency":  0.20,
            "Validity":     0.20,
            "Conformity":   0.10,
        }
        overall = sum(scores[k] * weights[k] for k in scores)
        scores["Overall"] = round(overall, 1)

        return scores

    def display(self, scores: dict):
        overall = scores["Overall"]
        color = "green" if overall >= 85 else "yellow" if overall >= 60 else "red"

        console.print(Panel.fit(
            f"[bold {color}]Data Quality Score: {overall}/100[/bold {color}]",
            border_style=color
        ))
        console.print()

        table = Table(box=box.ROUNDED, border_style=color, show_header=True, header_style=f"bold {color}")
        table.add_column("Pillar",    style="cyan", no_wrap=True)
        table.add_column("Score",     justify="right")
        table.add_column("Status",    justify="left")
        table.add_column("Weight",    justify="right", style="dim")

        weights = {"Completeness": "30%", "Uniqueness": "20%",
                   "Consistency": "20%", "Validity": "20%", "Conformity": "10%"}

        for pillar, score in scores.items():
            if pillar == "Overall":
                continue
            bar_color = "green" if score >= 85 else "yellow" if score >= 60 else "red"
            bar_len = int(score / 5)
            bar = f"[{bar_color}]{'█' * bar_len}{'░' * (20 - bar_len)}[/{bar_color}]"
            table.add_row(pillar, f"[{bar_color}]{score}[/{bar_color}]", bar, weights[pillar])

        console.print(table)
        console.print()


class CorrelationEngine:
    """
    Computes pairwise correlations between numeric features,
    optionally ranked by correlation with a target column.
    """

    def run(self, df: pd.DataFrame, target_col: str = None, threshold: float = 0.0) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return pd.DataFrame()
        corr_matrix = numeric_df.corr()
        return corr_matrix

    def display(self, df: pd.DataFrame, target_col: str = None, threshold: float = 0.3):
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            console.print("[dim]Not enough numeric columns for correlation analysis.[/dim]")
            return

        corr_matrix = numeric_df.corr()

        if target_col and target_col in corr_matrix.columns:
            # Show correlations ranked by target
            target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)

            console.print(Panel.fit(
                f"[bold cyan]Feature Correlations with '{target_col}'[/bold cyan]",
                border_style="cyan"
            ))
            console.print()

            table = Table(box=box.ROUNDED, border_style="cyan", show_header=True, header_style="bold cyan")
            table.add_column("Feature",     style="white",  no_wrap=True)
            table.add_column("Correlation", justify="right")
            table.add_column("Strength",    justify="left")

            for feat, corr in target_corr.items():
                abs_corr = abs(corr)
                if abs_corr < threshold:
                    continue
                color = "green" if abs_corr >= 0.7 else "yellow" if abs_corr >= 0.4 else "dim"
                direction = "+" if corr >= 0 else "-"
                bar_len = int(abs_corr * 20)
                bar = f"[{color}]{direction}{'█' * bar_len}{'░' * (20 - bar_len)}[/{color}]"
                table.add_row(feat, f"[{color}]{corr:+.4f}[/{color}]", bar)

            console.print(table)
        else:
            # Show full matrix
            console.print(Panel.fit(
                "[bold cyan]Correlation Matrix (Numeric Features)[/bold cyan]",
                border_style="cyan"
            ))
            console.print()

            cols = corr_matrix.columns.tolist()
            table = Table(box=box.SIMPLE, border_style="cyan", show_header=True, header_style="bold cyan")
            table.add_column("", style="cyan", no_wrap=True)
            for col in cols:
                table.add_column(col[:10], justify="right")

            for row_col in cols:
                row = [row_col]
                for col in cols:
                    val = corr_matrix.loc[row_col, col]
                    if row_col == col:
                        row.append("[dim]1.000[/dim]")
                    else:
                        color = "red" if abs(val) >= 0.7 else "yellow" if abs(val) >= 0.4 else "white"
                        row.append(f"[{color}]{val:+.3f}[/{color}]")
                table.add_row(*row)

            console.print(table)

        console.print()
