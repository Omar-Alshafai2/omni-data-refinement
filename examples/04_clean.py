"""
OMR Example 04: Automatic Cleaning
=====================================
.clean() runs OMR's intelligent cleaning engine, which reads the internal
health report and applies a set of automatic fixes:

  - Missing numeric values  : imputed using the column median
  - Missing categorical values : filled with a constant placeholder
  - Duplicate rows          : dropped, keeping the first occurrence
  - Mixed-type columns      : cast to a safe, uniform string type

Call .export() after .clean() to retrieve the cleaned data as a standard
pandas DataFrame.
"""

import pandas as pd
from omr import Dataset

# A dataset with all four common data quality issues:
#   - 'id' column contains a duplicated value (2 appears twice)
#   - 'department' has a missing categorical value
#   - 'salary' has a missing numeric value
#   - 'bonus' mixes integers and a string ("500"), causing a type conflict
df = pd.DataFrame({
    "id": [1, 2, 2, 4, 5],
    "department": ["HR", "IT", "IT", None, "Sales"],
    "salary": [50000, 60000, 60000, None, 90000],
    "bonus": [1000, "500", 500, 2000, 3000]
})

print("BEFORE CLEANING:")
print(df)
print("-" * 50)

dataset = Dataset(df)

# Run the automatic cleaning engine.
# The engine inspects each column's detected issues and applies the
# appropriate fix. No manual configuration is required.
print("\nExecuting dataset.clean()...\n")
dataset.clean()

# Export the cleaned data back to a pandas DataFrame for downstream use.
clean_df = dataset.export()

print("AFTER CLEANING:")
print(clean_df)
print("-" * 50)
# Expected results:
#   - The duplicate row (id=2) has been removed
#   - The missing salary has been filled with the column median
#   - The missing department has been filled with a constant
#   - The 'bonus' column is now uniformly typed as string
