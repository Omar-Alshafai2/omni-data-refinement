"""
OMR Reports Module — Generates JSON, Markdown, and HTML reports.
"""
import json
from datetime import datetime


class ReportGenerator:
    def generate(self, dataset, format: str = "html", path: str = "omr_report") -> str:
        fmt = format.lower()
        if fmt == "html":
            return self._html(dataset, path)
        elif fmt in ("markdown", "md"):
            return self._markdown(dataset, path)
        elif fmt == "json":
            return self._json(dataset, path)
        else:
            raise ValueError(f"Unknown format '{format}'.")

    def _html(self, dataset, path) -> str:
        filepath = path if path.endswith(".html") else path + ".html"
        health = dataset._last_health
        score = health.score if health else "N/A"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; padding: 40px; }}
  h1 {{ color: #7dd3fc; }}
  .score {{ font-size: 2rem; font-weight: bold; color: #4ade80; }}
</style>
</head>
<body>
  <h1>OMR Dataset Report</h1>
  <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
  <div class="score">Health Score: {score}/100</div>
  <h2>Transformations</h2>
  <ul>
"""
        for t in dataset.metadata.transformations:
            html += f"<li>{t.operation} on {t.column}: {t.reason}</li>"
            
        html += "</ul></body></html>"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        return filepath

    def _markdown(self, dataset, path) -> str:
        filepath = path if path.endswith(".md") else path + ".md"
        health = dataset._last_health
        
        lines = [
            "# OMR Dataset Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"**Health Score:** {health.score if health else 'N/A'}/100",
            "",
            "## Transformations",
        ]
        for t in dataset.metadata.transformations:
            lines.append(f"- **{t.operation}** on `{t.column}`: {t.reason}")
            
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    def _json(self, dataset, path) -> str:
        filepath = path if path.endswith(".json") else path + ".json"
        health = dataset._last_health
        
        data = {
            "metadata": dataset.metadata.to_dict(),
            "health_score": health.score if health else None,
            "transformations": [
                {"step": t.step, "op": t.operation, "col": t.column, "reason": t.reason}
                for t in dataset.metadata.transformations
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath
