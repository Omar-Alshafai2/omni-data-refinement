"""
OMR Example 10: Versioning and Rollback
=========================================
OMR includes a built-in versioning system that saves in-memory snapshots
of the dataset state. This allows you to experiment freely with cleaning
or transformation operations and undo them instantly without reloading data
from disk.

Workflow:
  1. Call .snapshot() before a risky operation to save the current state.
     It returns a version_id string that references this snapshot.
  2. Apply cleaning or transformation operations.
  3. If the result is unsatisfactory, call .rollback(version_id) to restore
     the dataset to the exact state it was in when the snapshot was taken.
"""

import pandas as pd
from omr import Dataset

# A dataset with one missing value that will be imputed during cleaning.
df = pd.DataFrame({"data": [1, 2, 3, None, 5]})

dataset = Dataset(df)

# Save a snapshot before applying any transformation.
# The 'name' and 'description' parameters are optional labels for the log.
# The returned version_id is required to reference this snapshot later.
print("Taking snapshot...")
version_id = dataset.snapshot(name="Pre-Cleaning", description="Before imputation")
print(f"Saved Version ID: {version_id}")

print("\nBefore Cleaning:")
print(dataset.export())

# Apply the cleaning engine. This will impute the missing value.
dataset.clean()
print("\nAfter Cleaning:")
print(dataset.export())

# Restore the dataset to the pre-cleaning state using the saved version_id.
# This does not reload from disk — it reads from the in-memory snapshot.
print("\nExecuting dataset.rollback()...\n")
dataset.rollback(version_id)

print("After Rollback:")
print(dataset.export())
# The missing value (None) should be restored, confirming the rollback succeeded.
