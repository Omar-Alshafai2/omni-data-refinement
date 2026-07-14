"""
OMR Tutorial: Dataset Summary
=============================

This example demonstrates the very first step you should take when receiving 
a new dataset: getting an instant pulse check of your data.

While Pandas provides `.info()`, `.describe()`, or manual null checking, 
OMR simplifies this process into a single, optimized method: `.summary()`.
"""

import pandas as pd
from omr import Dataset

# Let's create a messy, realistic dataset.
print("Loading data...")
df = pd.DataFrame({
    "age": [25, 30, None, 40, 40, 150],  # Includes missing and an outlier
    "income": [50000, 60000, 70000, None, None, 90000], # Includes missing
    "city": ["NY", "LA", "NY", "SF", "SF", None] 
})

# Step 1: Wrap the raw Pandas DataFrame in the OMR Dataset class.
# This adds OMR's advanced profiling and validation capabilities to the data.

dataset = Dataset(df)

# Step 2: Call the `.summary()` method.
# Instead of a massive table, this provides a highly optimized, single-line 
# summary covering the dataset's size, missing data percentage, duplicate rows, 
# and a fast heuristic health score. Ideal for automated logging or Notebook checks.

print("\nExecuting dataset.summary()...\n")
dataset.summary()

# Once the summary is reviewed, you instantly know the shape of your data.
# A high (green) health score means you can proceed safely. 
# A low (red) health score indicates you should investigate further using `.health()`.
