"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Hello again! Have you ever written a data cleaning script, made a change, 
realized it ruined the dataset, and had to reload the entire CSV from 
scratch?"

"It's a nightmare, especially with massive files. That's why OMR comes 
with a built-in Versioning System. You can take memory snapshots and 
roll back instantly."
=============================================================================
"""

import pandas as pd
from omr import Dataset

df = pd.DataFrame({
    "data": [1, 2, 3, None, 5]
})

dataset = Dataset(df)

# =============================================================================
# HOST (Voiceover):
# "Let's say we're about to do something risky, like a massive imputation. 
# First, we call `dataset.snapshot()`. This saves the exact current state 
# of the data in memory."
# =============================================================================

print("Taking snapshot...")
version_id = dataset.snapshot(name="Pre-Cleaning", description="Before imputation")
print(f"Saved Version ID: {version_id}")

print("\nBefore Cleaning:")
print(dataset.export())

# =============================================================================
# HOST (Voiceover):
# "Now, we clean the dataset."
# =============================================================================

dataset.clean()
print("\nAfter Cleaning:")
print(dataset.export())

# =============================================================================
# HOST (Voiceover):
# "Oh no! We didn't want it to impute that way. We need the raw data back. 
# Instead of reloading from disk, we just call `dataset.rollback(version_id)`."
# =============================================================================

print("\nExecuting dataset.rollback()...\n")
dataset.rollback(version_id)

print("After Rollback:")
print(dataset.export())

# =============================================================================
# HOST (Voiceover):
# "And boom! The missing values are back. The data is exactly as it was. 
# You can experiment freely knowing OMR has your back."
# [SCENE END]
# =============================================================================
