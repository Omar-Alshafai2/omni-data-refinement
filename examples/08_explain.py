"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Hello everyone! Have you ever run a tool, gotten a warning like 'Data Leakage 
Detected', and thought: 'Wait, what exactly does that mean, and how do I fix it?'"

"OMR isn't just an analysis tool; it's an educational framework. With the 
`.explain()` method, OMR acts as your built-in AI Data Science tutor."
=============================================================================
"""

import pandas as pd
from omr import Dataset

# Empty dummy data just to initialize the dataset
dataset = Dataset(pd.DataFrame({"a": [1]}))

# =============================================================================
# HOST (Voiceover):
# "Let's say OMR warned us about 'class_imbalance'. We just type: 
# `dataset.explain('class_imbalance')`."
# =============================================================================

print("\nExecuting dataset.explain('class_imbalance')...\n")
explanation = dataset.explain("class_imbalance")

# =============================================================================
# HOST (Voiceover):
# "OMR returns a beautifully formatted explanation. It gives us the definition, 
# it explains exactly why it matters to our ML models, it outlines the specific 
# risks, and finally, it provides a numbered list of recommended fixes!"
# 
# "Let's try another one: 'data_leakage'."
# =============================================================================

print("\nExecuting dataset.explain('data_leakage')...\n")
dataset.explain("data_leakage")

# =============================================================================
# HOST (Voiceover):
# "Now you don't need to leave your terminal to read documentation or search 
# the web. The knowledge is right at your fingertips."
# [SCENE END]
# =============================================================================
