"""
OMR Example 06: Statistical Analysis
=======================================
A dataset can be structurally clean (no nulls, no duplicates, correct types)
but still be statistically problematic for machine learning models.

.analyze() runs OMR's statistics engine to detect deeper issues:
  - Outliers       : detected using Z-score and IQR methods
  - Skewness       : identifies heavily skewed distributions that can
                     bias linear and tree-based models
  - Multicollinearity : detects highly correlated feature pairs
  - Class imbalance   : flags target columns with unequal class distribution

Each detected issue includes a severity level and a recommended action.
"""

import pandas as pd
from omr import Dataset

# A dataset with no missing values but with statistical problems:
#   - 'age' contains 150, a clear extreme outlier
#   - 'income' is heavily right-skewed due to the 900000 value
df = pd.DataFrame({
    "age":    [22, 25, 23, 24, 22, 21, 26, 25, 24, 150],
    "income": [40000, 42000, 45000, 41000, 43000, 40000, 44000, 46000, 42000, 900000],
})

dataset = Dataset(df)

# Run the statistical analysis engine.
# The output highlights flagged columns with their detected issue,
# severity, and a concrete recommendation (e.g., "Apply log transform").
print("Executing dataset.analyze()...\n")
dataset.analyze()
