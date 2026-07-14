"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Welcome to part two! In our last video, we used `.summary()` to get a quick 
pulse check. But what if that score is low? What if we need to know *exactly* 
what is wrong with our data before feeding it to a machine learning model?"

"This is where OMR truly shines. We're going to use the `.health()` method.
This doesn't just check for nulls. It evaluates your data across the 
5 Pillars of Data Quality: Completeness, Uniqueness, Consistency, Validity, 
and Conformity."
=============================================================================
"""

import pandas as pd
from omr import Dataset

# Create a dataset with numerous data quality issues
df = pd.DataFrame({
    "customer_id": [1, 2, 3, 3, 5],                 # Duplicated ID
    "age": [25, 30, -5, 120, None],                 # Invalid ages (-5, 120), missing value
    "salary": [50000, 60000, "70k", 80000, 90000],  # Mixed types (int and string)
    "is_active": [1, 0, 1, 0, 1]                    
})

dataset = Dataset(df)

# =============================================================================
# HOST (Voiceover):
# "Here we have a dataset with duplicate rows, missing values, extreme 
# outliers (an age of -5?), and mixed data types in the salary column."
#
# "If we passed this into Scikit-Learn right now, it would crash instantly."
# 
# "Let's call `dataset.health()` and see what the engine discovers."
# =============================================================================

print("\nExecuting dataset.health()...\n")
report = dataset.health()

# =============================================================================
# HOST (Voiceover):
# "Look at that beautiful output! We get a unified Health Score out of 100, 
# and a detailed breakdown across the five quality pillars."
# 
# "But more importantly, OMR lists out specific, actionable issues. It tells 
# you exactly which columns have problems, what the severity is, and gives 
# you a clear recommendation on how to fix it."
#
# "You can even access the underlying report object programmatically to fail 
# a CI/CD pipeline if the score drops below a certain threshold!"
# =============================================================================

print(f"\n[Programmatic Access] The raw score is: {report.score}")
if report.score < 80:
    print("[Pipeline Alert] Health score is too low! Data needs cleaning.")

# =============================================================================
# [SCENE END]
# =============================================================================
