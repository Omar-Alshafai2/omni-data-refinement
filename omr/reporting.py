"""
OMR Reporting Engine.

Generates HTML, Markdown, and JSON reports from OMR diagnostic data.
Uses only Python standard library + rich — no extra dependencies.
"""
import json
import os
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DiagnosticReport, ProfilingReport, ValidationReport, ChangeLog


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ReportingEngine:

    def generate(self,
                 format: str = "html",
                 path: str = "omr_report",
                 diagnostic_report=None,
                 profile=None,
                 validation_report=None,
                 changelog=None) -> str:
        """
        Generate a report file and return the file path.

        Args:
            format: 'html', 'markdown', or 'json'
            path:   Output filename (without extension)
        """
        fmt = format.lower()
        if fmt == "html":
            return self._html(path, diagnostic_report, profile, validation_report, changelog)
        elif fmt in ("markdown", "md"):
            return self._markdown(path, diagnostic_report, profile, validation_report, changelog)
        elif fmt == "json":
            return self._json(path, diagnostic_report, profile, validation_report, changelog)
        else:
            raise ValueError(f"Unknown format '{format}'. Use 'html', 'markdown', or 'json'.")

    # ── HTML ─────────────────────────────────────────────────────────────────

    def _html(self, path, diag, profile, val, changelog) -> str:
        filepath = path if path.endswith(".html") else path + ".html"
        issues_html = self._issues_to_html(diag)
        profile_html = self._profile_to_html(profile)
        changelog_html = self._changelog_to_html(changelog)

        rows = diag.num_rows if diag else "—"
        cols = diag.num_cols if diag else "—"
        issue_count = len(diag.issues) if diag else 0

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OMR Diagnostic Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e2e8f0; line-height: 1.6; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}
  h1 {{ font-size: 2rem; color: #7dd3fc; margin-bottom: 4px; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 32px; font-size: 0.9rem; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
  .stat {{ background: #1e293b; border-radius: 12px; padding: 20px 28px; border-left: 4px solid #7dd3fc; flex: 1; min-width: 140px; }}
  .stat-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #64748b; }}
  .stat-value {{ font-size: 1.8rem; font-weight: 700; color: #e2e8f0; }}
  h2 {{ font-size: 1.2rem; color: #7dd3fc; margin: 32px 0 16px; border-bottom: 1px solid #1e293b; padding-bottom: 8px; }}
  .issue {{ background: #1e293b; border-radius: 10px; padding: 16px 20px; margin-bottom: 12px; border-left: 4px solid; }}
  .issue.High {{ border-color: #f87171; }}
  .issue.Medium {{ border-color: #fbbf24; }}
  .issue.Low {{ border-color: #38bdf8; }}
  .issue-title {{ font-weight: 700; margin-bottom: 4px; }}
  .issue.High .issue-title {{ color: #f87171; }}
  .issue.Medium .issue-title {{ color: #fbbf24; }}
  .issue.Low .issue-title {{ color: #38bdf8; }}
  .issue p {{ color: #94a3b8; font-size: 0.9rem; }}
  .no-issues {{ color: #4ade80; font-weight: 600; padding: 16px; background: #14532d22; border-radius: 8px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; }}
  th {{ background: #0f172a; color: #7dd3fc; text-align: left; padding: 12px 14px; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #0f172a; font-size: 0.88rem; color: #cbd5e1; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
  .badge.warn {{ background: #78350f44; color: #fbbf24; }}
  .badge.ok {{ background: #14532d44; color: #4ade80; }}
  footer {{ margin-top: 48px; text-align: center; color: #334155; font-size: 0.8rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>OMR Diagnostic Report</h1>
  <p class="subtitle">Generated: {_now()}</p>

  <div class="stats">
    <div class="stat"><div class="stat-label">Rows</div><div class="stat-value">{rows:,}</div></div>
    <div class="stat"><div class="stat-label">Columns</div><div class="stat-value">{cols}</div></div>
    <div class="stat"><div class="stat-label">Issues Found</div><div class="stat-value">{issue_count}</div></div>
  </div>

  <h2>🔍 Detected Issues</h2>
  {issues_html}

  <h2>📊 Dataset Profile</h2>
  {profile_html}

  <h2>📋 Change Log</h2>
  {changelog_html}

  <footer>Generated by OMR — Omni Data Refinement</footer>
</div>
</body>
</html>"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        return filepath

    def _issues_to_html(self, diag) -> str:
        if not diag or not diag.issues:
            return '<p class="no-issues">✓ No issues detected. Dataset looks clean!</p>'
        rows = []
        for i in diag.issues:
            rows.append(
                f'<div class="issue {i.severity}">'
                f'<div class="issue-title">{i.issue_type} — {i.column} [{i.severity}]</div>'
                f'<p><strong>Problem:</strong> {i.description}</p>'
                f'<p><strong>Fix:</strong> {i.recommendation}</p>'
                f'</div>'
            )
        return "".join(rows)

    def _profile_to_html(self, profile) -> str:
        if not profile:
            return "<p style='color:#64748b'>Run assistant.profile() to include this section.</p>"
        rows_html = ""
        for c in profile.columns:
            missing_badge = (f'<span class="badge warn">{c.missing} ({c.missing_pct:.1f}%)</span>'
                             if c.missing > 0 else '<span class="badge ok">0</span>')
            mean_str = f"{c.mean:.2f}" if c.mean is not None else (", ".join(c.top_values[:2]) or "—")
            rows_html += (
                f"<tr><td>{c.name}</td><td>{c.dtype}</td><td>{missing_badge}</td>"
                f"<td>{c.unique}</td><td>{mean_str}</td></tr>"
            )
        return (
            "<table><thead><tr><th>Column</th><th>Type</th><th>Missing</th>"
            f"<th>Unique</th><th>Mean / Top</th></tr></thead><tbody>{rows_html}</tbody></table>"
        )

    def _changelog_to_html(self, changelog) -> str:
        if not changelog or not changelog.entries:
            return "<p style='color:#64748b'>No transformations recorded yet. Run assistant.fix() first.</p>"
        rows_html = "".join(
            f"<tr><td>{e.step}</td><td>{e.column}</td><td>{e.action}</td>"
            f"<td>{e.rows_affected}</td><td>{e.reason}</td></tr>"
            for e in changelog.entries
        )
        return (
            "<table><thead><tr><th>#</th><th>Column</th><th>Action</th>"
            f"<th>Rows Affected</th><th>Reason</th></tr></thead><tbody>{rows_html}</tbody></table>"
        )

    # ── Markdown ──────────────────────────────────────────────────────────────

    def _markdown(self, path, diag, profile, val, changelog) -> str:
        filepath = path if path.endswith(".md") else path + ".md"
        lines = [f"# OMR Diagnostic Report", f"\n_Generated: {_now()}_\n"]

        if diag:
            lines += [f"\n## Dataset Overview", f"- **Rows**: {diag.num_rows:,}",
                      f"- **Columns**: {diag.num_cols}", f"\n## Issues ({len(diag.issues)} found)\n"]
            if not diag.issues:
                lines.append("✓ No issues detected.")
            for i in diag.issues:
                lines += [f"### [{i.severity}] {i.issue_type} — `{i.column}`",
                          f"- **Problem**: {i.description}",
                          f"- **Fix**: {i.recommendation}\n"]

        if changelog and changelog.entries:
            lines += ["\n## Change Log\n", "| # | Column | Action | Rows Affected | Reason |",
                      "|---|--------|--------|---------------|--------|"]
            for e in changelog.entries:
                lines.append(f"| {e.step} | {e.column} | {e.action} | {e.rows_affected} | {e.reason} |")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    # ── JSON ──────────────────────────────────────────────────────────────────

    def _json(self, path, diag, profile, val, changelog) -> str:
        filepath = path if path.endswith(".json") else path + ".json"
        data = {"generated_at": _now(), "issues": [], "profile": [], "changelog": []}

        if diag:
            data["rows"] = diag.num_rows
            data["cols"] = diag.num_cols
            data["issues"] = [
                {"type": i.issue_type, "column": i.column, "severity": i.severity,
                 "description": i.description, "recommendation": i.recommendation}
                for i in diag.issues
            ]

        if profile:
            data["profile"] = [
                {"name": c.name, "dtype": c.dtype, "missing": c.missing,
                 "missing_pct": c.missing_pct, "unique": c.unique,
                 "mean": c.mean, "min": c.min_val, "max": c.max_val}
                for c in profile.columns
            ]

        if changelog:
            data["changelog"] = [
                {"step": e.step, "column": e.column, "action": e.action,
                 "rows_affected": e.rows_affected, "reason": e.reason}
                for e in changelog.entries
            ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath
