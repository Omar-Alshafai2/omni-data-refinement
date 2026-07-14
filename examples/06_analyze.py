"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Hello data scientists! Until now, we've focused on basic data quality: 
nulls, duplicates, and types."

"But what if your dataset is clean, but statistically dangerous? What if 
it has hidden extreme outliers, or highly skewed distributions that will 
destroy your linear regression models?"

"This is where OMR steps into advanced analytics with the `.analyze()` method."
=============================================================================
"""

import pandas as pd
from omr import Dataset

# Let's create a dataset that is "clean" (no nulls) but statistically messy
df = pd.DataFrame({
    "age": [22, 25, 23, 24, 22, 21, 26, 25, 24, 150], # 150 is an extreme outlier
    "income": [40000, 42000, 45000, 41000, 43000, 40000, 44000, 46000, 42000, 900000], # Right-skewed
})

dataset = Dataset(df)

# =============================================================================
# HOST (Voiceover):
# "Let's run `dataset.analyze()`. The OMR Statistics Engine is now firing up. 
# It runs Z-score and IQR tests for outliers. It calculates skewness. It 
# looks for multicollinearity and class imbalances."
# =============================================================================

print("\nExecuting dataset.analyze()...\n")
dataset.analyze()

# =============================================================================
# HOST (Voiceover):
# "Look at the report! It explicitly flags the severe skewness in our income 
# column, and highlights the potential outliers. Not only that, it tells us 
# exactly what we should consider doing next, like applying a log transform!"
# 
# "It's like having a senior data scientist looking over your shoulder."
# [SCENE END]
# =============================================================================
