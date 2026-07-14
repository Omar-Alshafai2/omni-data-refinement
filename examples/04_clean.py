"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Alright, this is the big one. In previous videos, we learned how to detect 
issues using `.health()`. But detecting issues is only half the battle. 
Usually, you'd have to spend the next two hours writing custom pandas code 
to impute missing values, drop duplicates, and fix mixed data types."

"Not anymore. OMR comes with an intelligent, one-click cleaning engine."
=============================================================================
"""

import pandas as pd
from omr import Dataset

# We have a terrifyingly messy dataset here
df = pd.DataFrame({
    "id": [1, 2, 2, 4, 5],                        # Duplicates
    "department": ["HR", "IT", "IT", None, "Sales"], # Missing categorical
    "salary": [50000, 60000, 60000, None, 90000],    # Missing numeric
    "bonus": [1000, "500", 500, 2000, 3000]          # Mixed types! (int & str)
})

print("BEFORE CLEANING:")
print(df)
print("-" * 50)

dataset = Dataset(df)

# =============================================================================
# HOST (Voiceover):
# "We simply call `dataset.clean()`. Behind the scenes, the Cleaning Engine 
# reads the health report. It sees the missing numeric values and imputes 
# them using the median. It sees missing text and uses a constant. It finds 
# the duplicate rows and drops them. And it resolves the mixed types!"
# =============================================================================

print("\nExecuting dataset.clean()...\n")
dataset.clean()

# =============================================================================
# HOST (Voiceover):
# "Let's export the data and look at it now."
# =============================================================================

clean_df = dataset.export()

print("AFTER CLEANING:")
print(clean_df)
print("-" * 50)

# =============================================================================
# HOST (Voiceover):
# "Look at that! The duplicate row for ID 2 is gone. The missing salary was 
# filled perfectly with the median. The missing department was imputed. And 
# the 'bonus' column is safely cast to a uniform string type to prevent 
# crashes."
#
# "Hours of data cleaning, reduced to a single method call."
# [SCENE END]
# =============================================================================
