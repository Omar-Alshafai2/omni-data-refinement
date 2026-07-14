"""
OMR Example 08: Built-in Concept Explainer
============================================
When OMR detects an issue such as data leakage or class imbalance, it
surfaces a keyword describing the problem. .explain(keyword) looks up that
keyword and returns a structured explanation covering:

  - Definition    : What the concept means
  - Why it matters: How it affects model training and performance
  - Risks         : Specific failure modes it causes
  - Fixes         : A numbered list of recommended remediation strategies

Supported keywords include: 'class_imbalance', 'data_leakage',
'multicollinearity', 'high_cardinality', 'skewness', and more.
"""

import pandas as pd
from omr import Dataset

# A minimal dataset is needed to initialize the Dataset object.
# The .explain() method does not use the data — it only needs the keyword.
dataset = Dataset(pd.DataFrame({"a": [1]}))

# Look up an explanation for 'class_imbalance'.
# Returns a formatted panel with definition, risks, and recommended fixes.
print("Executing dataset.explain('class_imbalance')...\n")
explanation = dataset.explain("class_imbalance")

# Look up an explanation for 'data_leakage'.
# Data leakage is one of the most common causes of overfitting in ML pipelines.
print("\nExecuting dataset.explain('data_leakage')...\n")
dataset.explain("data_leakage")
