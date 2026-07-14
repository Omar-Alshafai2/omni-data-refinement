"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Hello everyone! Today we're looking at `.profile()`. 

"When you use standard pandas, you usually have to run `.info()` to get 
data types and non-null counts, and then run `.describe()` to get means 
and maximums. And even then, it completely ignores your categorical columns!"

"OMR's `.profile()` method solves this by giving you a unified, comprehensive 
statistical profile of every single column in your dataset, regardless of 
data type."
=============================================================================
"""

import pandas as pd
from omr import Dataset

# Let's create a dataset with both numerical and categorical data
df = pd.DataFrame({
    "product_id": [101, 102, 103, 104, 105],
    "category": ["Electronics", "Clothing", "Electronics", "Home", "Clothing"],
    "price": [299.99, 45.50, 899.00, 120.00, None],
    "in_stock": [True, True, False, True, False]
})

dataset = Dataset(df)

# =============================================================================
# HOST (Voiceover):
# "Watch what happens when we call `dataset.profile()`. 
# It builds a stunning, readable table right in the terminal."
# =============================================================================

print("\nExecuting dataset.profile()...\n")
dataset.profile()

# =============================================================================
# HOST (Voiceover):
# "Notice how it handles everything elegantly. For the 'price' column, it 
# calculates the mean, min, and max. But for the 'category' column, which is 
# text, it automatically calculates the Top/Most Frequent value!"
#
# "It also clearly highlights exactly how many unique values exist and exactly 
# how many missing values are present, along with their percentages."
#
# "This is the ultimate replacement for Pandas `.describe()`."
# [SCENE END]
# =============================================================================
