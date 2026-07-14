"""
=============================================================
  OMR SANDBOX — Write your own code here to test the library
  Run this file directly in PyCharm: Right-click → Run 'sandbox'
=============================================================
"""

import pandas as pd
from omr import Dataset, schemas, Monitor


# =============================================================
# STEP 1: Load your dataset
# =============================================================

df = pd.DataFrame({
    "age":    [25, 30, 35, None, 42, -5, 28, 200],
    "salary": [50000, 75000, None, 62000, 90000, 80000, 55000, 71000],
    "name":   ["Alice", "Bob", "Bob", "Dana", "n/a", "Frank", "Grace", None],
    "score":  [1, "high", "high", 3, 4, 2, "low", 4],
    "churn":  [0, 0, 0, 1, 0, 1, 0, 1],
})

dataset = Dataset(df)


# =============================================================
# STEP 2: Understand your data
# =============================================================

# Quick health snapshot (one line summary)
dataset.summary()

# Full 5-pillar health score report
report = dataset.health()
print(f"Score: {report.score}/100")

# Full statistical profile (dtypes, missing %, mean, std, min, max)
dataset.profile()


# =============================================================
# STEP 3: Clean & Validate
# =============================================================

# Auto-fix missing values, duplicates, and mixed types
dataset.clean()
dataset.explain_changes()

# Validate against business rules using built-in schemas
schema = {
    "age":    schemas.PositiveInteger(min=0, max=120),
    "salary": schemas.PositiveFloat(min=10000),
    "churn":  schemas.OneOf(0, 1),
}
dataset.validate(schema)


# =============================================================
# STEP 4: Advanced Analytics
# =============================================================

# Detect outliers, skewness, multicollinearity, constant features
dataset.analyze()


# =============================================================
# STEP 5: Explainability
# =============================================================

# Rule-based explanations for data science issues
dataset.explain("class_imbalance")
dataset.explain("data_leakage")


# =============================================================
# STEP 6: Drift & Monitoring
# =============================================================

# Compare against production data (PSI, KS Test, JS Divergence)
prod_df = pd.DataFrame({"age": [20, 22, 25, 30], "salary": [40000, 42000, 45000, 50000]})
dataset.compare(Dataset(prod_df))

# Continuous monitoring
monitor = Monitor()
monitor.watch(dataset)
new_data = pd.DataFrame({"age": [85, 90, 95], "salary": [10, 20, 30]})
alerts = monitor.check(new_data)
for alert in alerts:
    print(alert)


# =============================================================
# STEP 7: Versioning & Reporting
# =============================================================

# Save checkpoints
dataset.snapshot(name="v1", description="Cleaned and validated")

# Generate HTML/JSON/Markdown reports
dataset.report(format="html", path="health_report")


# =============================================================
# STEP 8: Export
# =============================================================

clean_df = dataset.export()
print(clean_df)
