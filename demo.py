"""
OMR Full Demo — Run this in PyCharm to test the entire library.

This script demonstrates EVERY major feature of the OMR library.
Run it with: .venv/Scripts/python demo.py
"""
import pandas as pd
from omr import Dataset, schemas, Monitor

print("\n" + "=" * 65)
print("  OMR — Omni Data Refinement  |  Full Feature Demo")
print("=" * 65 + "\n")

# ── Build a realistic messy dataset ──────────────────────────────────────────
df = pd.DataFrame({
    "customer_id":  [1, 2, 3, 3, 4, 5, 6, 7, 8, 9],          # dup rows 2&3
    "name":         ["Alice", "Bob", "Charlie", "Charlie", "Dana", "Eve", "Frank", "Grace", "n/a", None],
    "age":          [25, 30, 35, 35, -5, 42, 28, 150, 33, None],  # -5 and 150 are outliers
    "salary":       [50000, 75000, 90000, 90000, 62000, None, 800000, 55000, 48000, 71000],
    "score":        [1, "high", "high", "high", 3, 4, 2, 5, 3, 4],  # mixed types
    "churn":        [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    "join_date":    ["2020-01-15", "2019-06-20", "2021-03-10", "2021-03-10",
                     "2018-11-05", "2022-07-01", "2023-02-14", "2017-09-30",
                     "2020-12-25", "2021-08-08"],
})

print("ORIGINAL DATASET:")
print(df.to_string())
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Initialize the Dataset
# ─────────────────────────────────────────────────────────────────────────────
dataset = Dataset(df)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Profile — get a full statistical overview
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  STEP 2: profile()")
print("-" * 65)
dataset.profile()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Health — detect all issues
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  STEP 3: health()")
print("-" * 65)
dataset.health()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Clean — auto-resolve all issues
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  STEP 4: clean()")
print("-" * 65)
dataset.clean()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Explain Changes — see what changed and why
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  STEP 5: explain_changes()")
print("-" * 65)
dataset.explain_changes()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Validate — enforce business rules
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  STEP 6: validate()")
print("-" * 65)
dataset.validate({
    "age":    schemas.PositiveInteger(min=0, max=120, not_null=True),
    "salary": schemas.PositiveFloat(not_null=True),
})

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Analyze — advanced statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  STEP 7: analyze()")
print("-" * 65)
dataset.analyze()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Data Drift Detection
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  STEP 8: compare() — simulating production drift")
print("-" * 65)
production_df = dataset.export().copy()
production_df["age"] = production_df["age"] + 18     # Simulate age shift
production_df["salary"] = production_df["salary"] * 2  # Simulate salary shift
dataset.compare(Dataset(production_df))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Data Science Explainability
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  STEP 9: explain() — AI Assistant")
print("-" * 65)
dataset.explain("class_imbalance")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Generate Reports
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  STEP 10: report() — generate HTML + JSON + Markdown reports")
print("-" * 65)
dataset.report(format="html",     path="omr_report")
dataset.report(format="json",     path="omr_report")
dataset.report(format="markdown", path="omr_report")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: Export the clean dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  STEP 11: export()")
print("-" * 65)
clean_df = dataset.export()
print(f"Clean DataFrame: {len(clean_df)} rows × {len(clean_df.columns)} columns")
print(clean_df.to_string())

print("\n" + "=" * 65)
print("  OMR Demo Complete! All features tested successfully.")
print("=" * 65 + "\n")
