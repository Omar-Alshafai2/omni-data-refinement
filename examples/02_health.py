"""
OMR Example 02: Health Check
==============================
When the summary health score is low, use `.health()` to get a detailed
breakdown of exactly what is wrong with the data.

.health() evaluates data across five quality pillars:
  - Completeness  : Are there missing values?
  - Uniqueness    : Are there duplicate rows or IDs?
  - Consistency   : Are data types uniform within each column?
  - Validity      : Are values within acceptable ranges?
  - Conformity    : Do values conform to expected patterns?

The method returns a HealthReport object that can be used programmatically
to enforce quality gates in CI/CD pipelines.
"""

import pandas as pd
from omr import Dataset

# A dataset with multiple data quality issues:
#   - customer_id 3 appears twice (duplicate)
#   - age -5 is logically invalid; 120 is an extreme outlier; None is missing
#   - salary column mixes integers and strings ("70k"), causing type inconsistency
df = pd.DataFrame({
    "customer_id": [1, 2, 3, 3, 5],
    "age": [25, 30, -5, 120, None],
    "salary": [50000, 60000, "70k", 80000, 90000],
    "is_active": [1, 0, 1, 0, 1]
})

dataset = Dataset(df)

# Run the full health check across all five quality pillars.
# The output includes a score out of 100, a per-pillar breakdown,
# and a list of specific issues with severity levels and recommended fixes.
print("Executing dataset.health()...\n")
report = dataset.health()

# The report object can be accessed programmatically.
# Use the score to enforce a quality gate: if the score is below 80,
# flag the dataset as unfit and stop the pipeline.
print(f"\n[Programmatic Access] Raw score: {report.score}")
if report.score < 80:
    print("[Pipeline Alert] Health score is too low. Data requires cleaning before use.")
