"""
OMR Example 07: Dataset Comparison and Drift Detection
=========================================================
Data drift occurs when the statistical distribution of production data
shifts away from the distribution the model was trained on. This causes
model predictions to silently degrade over time without any visible error.

.compare(other_dataset) detects drift between two datasets by running:
  - Population Stability Index (PSI)
  - Kolmogorov-Smirnov test
  - Jensen-Shannon divergence

Each column receives a drift severity: None, Low, Medium, or High.
A High severity result means the column's distribution has fundamentally
changed and the model should be retrained.
"""

import pandas as pd
from omr import Dataset

# The original training dataset — this represents the data distribution
# the model learned from during training.
training_df = pd.DataFrame({
    "age":    [20, 22, 25, 23, 26],
    "salary": [40000, 42000, 45000, 41000, 46000]
})
training_dataset = Dataset(training_df)

# The incoming production dataset — represents data arriving today.
# 'age' has shifted slightly, but 'salary' has approximately doubled,
# which would cause significant model degradation.
production_df = pd.DataFrame({
    "age":    [21, 23, 26, 24, 28],
    "salary": [80000, 84000, 90000, 82000, 92000]
})
production_dataset = Dataset(production_df)

# Compare the training dataset against the production dataset.
# The drift report shows each column's PSI score and severity.
# If 'salary' is flagged as High, the model must be retrained.
print("Executing training_dataset.compare(production_dataset)...\n")
training_dataset.compare(production_dataset)
