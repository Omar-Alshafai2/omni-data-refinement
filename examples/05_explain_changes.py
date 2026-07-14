"""
OMR Example 05: Explain Changes (Transformation Log)
=======================================================
Every operation performed by .clean() is strictly logged internally.
.explain_changes() prints this transformation log, providing full
auditability of what was changed, which columns were affected, how many
rows were touched, and the reasoning behind each decision.

This is essential in regulated or enterprise environments where data
transformations must be documented and explainable.
"""

import pandas as pd
from omr import Dataset

# A small dataset with a duplicate row and a missing numeric value.
df = pd.DataFrame({
    "id": [1, 2, 2, 4],
    "sales": [100, None, 100, 400]
})

dataset = Dataset(df)

# Run health detection first, then apply automatic cleaning.
# Both steps are required before calling explain_changes().
dataset.health()
dataset.clean()

# Print the transformation log.
# Each entry in the log describes:
#   - Column name
#   - Action taken (e.g., Median Imputation, Duplicate Removal)
#   - Number of rows affected
#   - The reason the action was triggered
print("\nExecuting dataset.explain_changes()...\n")
dataset.explain_changes()
