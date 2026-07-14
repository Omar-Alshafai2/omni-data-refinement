"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Welcome back! In our last video, we used `.clean()` to automatically fix 
our messy dataset. But in enterprise environments, 'magic' isn't allowed."

"Your manager or compliance team is going to ask: 'What exactly did the 
algorithm change? How many rows were affected? Why was that decision made?'"

"OMR has you covered. Everything the `.clean()` method does is strictly 
logged. Let's look at `.explain_changes()`."
=============================================================================
"""

import pandas as pd
from omr import Dataset

df = pd.DataFrame({
    "id": [1, 2, 2, 4],
    "sales": [100, None, 100, 400]
})

dataset = Dataset(df)

# First we run health to detect issues, then clean to fix them
dataset.health()
dataset.clean()

# =============================================================================
# HOST (Voiceover):
# "Instead of guessing what just happened to our data, we call 
# `dataset.explain_changes()`. 
#
# "This prints a stunning Transformation Log. It tells us the exact column, 
# the mathematical action taken (like Median Imputation), exactly how many 
# rows were touched, and the logical reason behind it."
#
# "This is complete data lineage and auditability out of the box!"
# =============================================================================

print("\nExecuting dataset.explain_changes()...\n")
dataset.explain_changes()

# =============================================================================
# [SCENE END]
# =============================================================================
