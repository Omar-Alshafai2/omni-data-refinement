"""
OMR Example 01: Dataset Summary
================================
The first step when working with any new dataset is getting a quick
overview of its structure and quality. OMR's `.summary()` method provides
a single-line snapshot covering: shape, missing value percentage, duplicate
row count, and an overall heuristic health score.

This replaces the need to chain pandas `.info()`, `.describe()`, and
manual null-checking calls.
"""

import pandas as pd
from omr import Dataset

# A realistic dataset containing missing values and an outlier in 'age'.
df = pd.DataFrame({
    "age": [25, 30, None, 40, 40, 150],  # None is missing; 150 is a likely outlier
    "income": [50000, 60000, 70000, None, None, 90000],  # Two missing values
    "city": ["NY", "LA", "NY", "SF", "SF", None]  # One missing value
})

# Wrap the raw DataFrame in an OMR Dataset.
# This enables all OMR quality, profiling, and validation features.
dataset = Dataset(df)

# Call .summary() to print a compact, single-line overview of the dataset.
# The health score ranges from 0 (poor) to 100 (excellent).
# A green score means the data is safe to use.
# A red score means issues were detected — use .health() to investigate further.
print("Executing dataset.summary()...\n")
dataset.summary()
