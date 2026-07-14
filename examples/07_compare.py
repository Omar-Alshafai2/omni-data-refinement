"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Welcome back! Imagine you trained an amazing model last year. It was 
perfect. But recently, predictions are failing. Why? Because the world 
changed. Income went up, demographics shifted."

"This is called Data Drift, and it's the silent killer of ML models. Today, 
we're going to use OMR's `.compare()` method to catch it."
=============================================================================
"""

import pandas as pd
from omr import Dataset

# Imagine this is the data we trained our model on last year
training_df = pd.DataFrame({
    "age": [20, 22, 25, 23, 26],
    "salary": [40000, 42000, 45000, 41000, 46000]
})
training_dataset = Dataset(training_df)

# Now imagine this is the data arriving today in production
# Age has shifted slightly, but salary has doubled!
production_df = pd.DataFrame({
    "age": [21, 23, 26, 24, 28],
    "salary": [80000, 84000, 90000, 82000, 92000]
})
production_dataset = Dataset(production_df)

# =============================================================================
# HOST (Voiceover):
# "We simply call `compare()` on our training dataset, and pass in the 
# production dataset. OMR instantly calculates Population Stability Index (PSI), 
# Kolmogorov-Smirnov tests, and Jensen-Shannon divergence."
# =============================================================================

print("\nExecuting training_dataset.compare(production_dataset)...\n")
training_dataset.compare(production_dataset)

# =============================================================================
# HOST (Voiceover):
# "The Drift Report is generated! It clearly shows that 'salary' has suffered 
# High severity drift, meaning its distribution has fundamentally changed. 
# It's time to retrain that model!"
# [SCENE END]
# =============================================================================
