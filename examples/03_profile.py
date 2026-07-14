"""
OMR Example 03: Column Profiling
==================================
.profile() generates a unified statistical profile for every column in the
dataset, regardless of data type.

For numeric columns it reports: mean, min, max, standard deviation,
missing count, and unique value count.

For categorical columns it reports: top/most frequent value, unique count,
and missing count.

This is a single-call replacement for pandas .info() + .describe(), which
ignores categorical columns entirely.
"""

import pandas as pd
from omr import Dataset

# A mixed dataset with numeric, categorical, and boolean columns.
# The 'price' column has one missing value to demonstrate null reporting.
df = pd.DataFrame({
    "product_id": [101, 102, 103, 104, 105],
    "category": ["Electronics", "Clothing", "Electronics", "Home", "Clothing"],
    "price": [299.99, 45.50, 899.00, 120.00, None],  # One missing value
    "in_stock": [True, True, False, True, False]
})

dataset = Dataset(df)

# Generate and print the full column profile.
# Each column is displayed as a formatted row in a terminal table showing
# its type, statistical metrics, missing count, and unique value count.
print("Executing dataset.profile()...\n")
dataset.profile()
