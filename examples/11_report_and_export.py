"""
OMR Example 11: Report Generation and Data Export
====================================================
After completing all quality checks and cleaning operations, OMR provides
two output mechanisms:

  .report(format, path)
      Generates a data quality report saved to disk. Supported formats:
        - "html"     : An interactive dashboard viewable in any browser.
        - "markdown" : A formatted report suitable for GitHub wikis or READMEs.
        - "json"     : A machine-readable report for automated pipelines.

  .export()
      Returns the current state of the dataset as a standard pandas DataFrame.
      The returned object is fully compatible with scikit-learn, PyTorch,
      and any other library that accepts a DataFrame or numpy array.
"""

import pandas as pd
from omr import Dataset

df = pd.DataFrame({
    "a": [1, 2, 3, 4],
    "b": ["x", "y", "x", "z"]
})

dataset = Dataset(df)

# Run a health check to populate the internal report data.
# .report() requires at least one analysis method to have been called first.
dataset.health()

# Generate reports in all three formats.
# Each call writes the report to disk and returns the file path.
print("\nGenerating Reports...")
html_path = dataset.report(format="html", path="my_report")
md_path   = dataset.report(format="markdown", path="my_report")
json_path = dataset.report(format="json", path="my_report")

print(f"HTML report saved to  : {html_path}")
print(f"Markdown report saved : {md_path}")
print(f"JSON report saved     : {json_path}")

# Export the dataset as a pandas DataFrame for use in downstream tasks.
# This creates no copy overhead — it returns the underlying DataFrame directly.
print("\nExecuting dataset.export()...\n")
clean_dataframe = dataset.export()

print("Exported type :", type(clean_dataframe))
print(clean_dataframe)
