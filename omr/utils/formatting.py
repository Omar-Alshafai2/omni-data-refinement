"""
OMR Utils — Rich terminal formatting for all report types.
All display logic lives here; engines return data objects, not printed output.
"""
from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from rich import box
import sys

_THEME = Theme({
    "info":    "cyan",
    "warning": "yellow",
    "danger":  "bold red",
    "success": "bold green",
    "header":  "bold magenta",
})

console = Console(theme=_THEME)

BLOCK_FULL = "█"
BLOCK_EMPTY = "░"
CHECK_MARK = "✓"
try:
    if sys.stdout is not None and getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
        BLOCK_FULL = "#"
        BLOCK_EMPTY = "-"
        CHECK_MARK = "OK:"
except Exception:
    pass


def _severity_color(severity: str) -> str:
    return {"High": "danger", "Medium": "warning", "Low": "info"}.get(severity, "white")


# ── Health ───────────────────────────────────────────────────────────────────

def display_health(report) -> None:
    score = report.score
    color = "green" if score >= 85 else "yellow" if score >= 60 else "red"
    console.print(Panel.fit(
        f"[bold {color}]Dataset Health Score: {score:.1f}/100[/bold {color}]",
        border_style=color
    ))
    console.print()

    table = Table(box=box.ROUNDED, border_style=color, show_header=True, header_style=f"bold {color}")
    table.add_column("Pillar",       style="cyan", no_wrap=True)
    table.add_column("Score",        justify="right")
    table.add_column("Visual",       justify="left")
    table.add_column("Weight",       justify="right", style="dim")

    weights = {"Completeness": "30%", "Uniqueness": "20%",
               "Consistency": "20%", "Validity": "20%", "Conformity": "10%"}

    for pillar in ("Completeness", "Uniqueness", "Consistency", "Validity", "Conformity"):
        val = getattr(report, pillar.lower())
        c = "green" if val >= 85 else "yellow" if val >= 60 else "red"
        bar = f"[{c}]{BLOCK_FULL * int(val / 5)}{BLOCK_EMPTY * (20 - int(val / 5))}[/{c}]"
        table.add_row(pillar, f"[{c}]{val:.1f}[/{c}]", bar, weights[pillar])

    console.print(table)

    if report.issues:
        console.print()
        console.print(f"[bold]Detected {len(report.issues)} issue(s):[/bold]")
        console.print()
        for issue in report.issues:
            c = _severity_color(issue.severity)
            content = Text()
            content.append("Problem: ", style="bold")
            content.append(f"{issue.description}\n", style="white")
            content.append("Fix: ", style="bold")
            content.append(issue.recommendation, style="white")
            console.print(Panel(content,
                                title=f"[{c}]{issue.issue_type} — {issue.column}[/{c}]",
                                border_style=c))
    else:
        console.print(f"[success]{CHECK_MARK} No issues detected.[/success]")
    console.print()


# ── Profile ──────────────────────────────────────────────────────────────────

def display_profile(report) -> None:
    console.print(Panel.fit(
        f"[header]Dataset Profile[/header]\n"
        f"Rows: [bold]{report.num_rows:,}[/bold] | Columns: [bold]{report.num_cols}[/bold] | "
        f"Memory: [bold]{report.memory_mb:.2f} MB[/bold]",
        border_style="magenta"
    ))
    console.print()
    table = Table(box=box.ROUNDED, border_style="magenta", show_header=True, header_style="bold magenta")
    table.add_column("Column",   style="cyan", no_wrap=True)
    table.add_column("Type")
    table.add_column("Missing",  justify="right")
    table.add_column("Unique",   justify="right")
    table.add_column("Mean/Top", justify="right")
    table.add_column("Min",      justify="right")
    table.add_column("Max",      justify="right")
    for col in report.columns:
        m = (f"[warning]{col.missing} ({col.missing_pct:.1f}%)[/warning]"
             if col.missing > 0 else "[success]0[/success]")
        mean_s = f"{col.mean:.2f}" if col.mean is not None else (col.top_values[0] if col.top_values else "—")
        table.add_row(col.name, col.dtype, m, str(col.unique), mean_s,
                      str(col.min_val) if col.min_val is not None else "—",
                      str(col.max_val) if col.max_val is not None else "—")
    console.print(table)
    console.print()


# ── Validation ───────────────────────────────────────────────────────────────

def display_validation(report) -> None:
    if report.passed:
        console.print(Panel.fit(f"[success]{CHECK_MARK} All validations passed![/success]", border_style="green"))
        return
    console.print(Panel.fit(
        f"[danger]Validation Failed — {len(report.issues)} Rule(s) Violated[/danger]",
        border_style="red"
    ))
    console.print()
    for issue in report.issues:
        content = Text()
        content.append("Rule: ",         style="bold"); content.append(f"{issue.rule}\n", style="white")
        content.append("Problem: ",      style="bold"); content.append(f"{issue.description}\n", style="white")
        content.append("Rows Failing: ", style="bold"); content.append(str(issue.failing_rows), style="danger")
        console.print(Panel(content, title=f"[danger]Validation — {issue.column}[/danger]", border_style="red"))
        console.print()


# ── Statistics ───────────────────────────────────────────────────────────────

def display_statistics(report) -> None:
    console.print(Panel.fit("[header]Statistical Analysis Report[/header]", border_style="cyan"))
    console.print()
    if not report.findings:
        console.print(f"[success]{CHECK_MARK} No statistical issues detected.[/success]")
        return
    for f in report.findings:
        c = _severity_color(f.severity)
        content = Text()
        content.append("Finding: ",        style="bold"); content.append(f"{f.description}\n", style="white")
        content.append("Recommendation: ", style="bold"); content.append(f.recommendation, style="white")
        if f.context:
            content.append("\nDetails: ", style="bold")
            content.append(", ".join(f"{k}: {v}" for k, v in f.context.items()), style="white")
        console.print(Panel(content, title=f"[{c}]{f.finding_type} — {f.column}[/{c}]", border_style=c))
    console.print()


# ── Drift ────────────────────────────────────────────────────────────────────

def display_drift(report) -> None:
    drifted = [r for r in report.results if r.severity != "None"]
    if not drifted:
        console.print(Panel.fit(f"[success]{CHECK_MARK} No significant drift detected.[/success]", border_style="green"))
        return
    console.print(Panel.fit(
        f"[warning]Drift Report — {len(drifted)} Column(s) Drifted[/warning]",
        border_style="yellow"
    ))
    console.print()
    table = Table(box=box.ROUNDED, border_style="yellow", show_header=True, header_style="bold yellow")
    table.add_column("Column",   style="cyan", no_wrap=True)
    table.add_column("PSI",      justify="right")
    table.add_column("KS Stat",  justify="right")
    table.add_column("JS Div",   justify="right")
    table.add_column("Severity")
    for r in drifted:
        c = "danger" if r.severity == "High" else "warning"
        table.add_row(r.column,
                      f"{r.psi:.4f}", f"{r.ks_statistic:.4f}", f"{r.js_divergence:.4f}",
                      f"[{c}]{r.severity}[/{c}]")
    console.print(table)
    console.print()


# ── Change Log ───────────────────────────────────────────────────────────────

def display_changelog(changelog) -> None:
    if not changelog.transformations:
        console.print("[dim]No transformations recorded.[/dim]")
        return
    console.print(Panel.fit("[header]Transformation Log[/header]", border_style="cyan"))
    console.print()
    table = Table(box=box.SIMPLE, border_style="cyan", show_header=True, header_style="bold cyan")
    table.add_column("#",             style="dim", width=4)
    table.add_column("Column",        style="cyan", no_wrap=True)
    table.add_column("Action",        style="success")
    table.add_column("Rows Affected", justify="right")
    table.add_column("Reason",        style="white")
    for e in changelog.transformations:
        table.add_row(str(e.step), e.column, e.operation, str(e.rows_affected), e.reason)
    console.print(table)
    console.print()


# ── Bias ─────────────────────────────────────────────────────────────────────

def display_bias(report) -> None:
    if not report.issues:
        console.print(Panel.fit(f"[success]{CHECK_MARK} No significant bias detected.[/success]", border_style="green"))
        return
    console.print(Panel.fit(
        f"[warning]Bias Report — {len(report.issues)} Issue(s)[/warning]",
        border_style="yellow"
    ))
    console.print()
    for issue in report.issues:
        content = Text()
        content.append("Type: ", style="bold"); content.append(f"{issue.issue_type}\n", style="white")
        content.append("Problem: ", style="bold"); content.append(f"{issue.description}\n", style="white")
        content.append("Recommendation: ", style="bold"); content.append(issue.recommendation, style="white")
        console.print(Panel(content, title=f"[warning]Bias — {issue.column}[/warning]", border_style="yellow"))
    console.print()


# ── Monitoring ───────────────────────────────────────────────────────────────

def display_alerts(alerts: list) -> None:
    if not alerts:
        console.print(f"[success]{CHECK_MARK} No monitoring alerts.[/success]")
        return
    console.print(Panel.fit(f"[danger]{len(alerts)} Monitoring Alert(s)[/danger]", border_style="red"))
    console.print()
    for alert in alerts:
        c = _severity_color(alert.severity)
        console.print(f"  [{c}]● {alert.check}: {alert.message}[/{c}]")
    console.print()


# ── Summary ──────────────────────────────────────────────────────────────────

def display_summary(metadata, health_score: float) -> None:
    q_color = "green" if health_score >= 85 else "yellow" if health_score >= 60 else "red"
    line = Text()
    line.append("[Dataset] ",         style="bold cyan")
    line.append(f"{metadata.num_rows:,} rows × {metadata.num_cols} cols", style="white")
    line.append("  |  Missing: ",     style="dim")
    mp = metadata.missing_pct
    line.append(f"{metadata.missing_cells} ({mp:.1f}%)",
                style="green" if mp == 0 else "yellow" if mp < 5 else "red")
    line.append("  |  Duplicates: ", style="dim")
    line.append(str(metadata.duplicate_rows),
                style="green" if metadata.duplicate_rows == 0 else "yellow")
    line.append("  |  Health: ",     style="dim")
    line.append(f"{health_score:.1f}/100", style=f"bold {q_color}")
    console.print(line)
