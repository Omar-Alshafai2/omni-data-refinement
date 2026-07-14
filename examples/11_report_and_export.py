"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Welcome to the finale of our core functionality! We've cleaned, analyzed, 
and validated our data. Now, we need to share these results with our team 
and actually use the data."

"We'll do this using `.report()` and `.export()`."
=============================================================================
"""

import pandas as pd
from omr import Dataset

df = pd.DataFrame({
    "a": [1, 2, 3, 4],
    "b": ["x", "y", "x", "z"]
})

dataset = Dataset(df)
dataset.health() # Run health to populate report data

# =============================================================================
# HOST (Voiceover):
# "Let's say your manager wants to see the data quality score. You can't 
# just send them a terminal screenshot. So, you call `dataset.report()`. 
# 
# You can export it as an HTML dashboard, a Markdown file for GitHub, or 
# a JSON file for your automated pipelines."
# =============================================================================

print("\nGenerating Reports...")
html_path = dataset.report(format="html", path="my_report")
md_path = dataset.report(format="markdown", path="my_report")
json_path = dataset.report(format="json", path="my_report")

print(f"Generated HTML: {html_path}")
print(f"Generated Markdown: {md_path}")
print(f"Generated JSON: {json_path}")

# =============================================================================
# HOST (Voiceover):
# "And finally, we need to pass this pristine data into our Scikit-Learn 
# or PyTorch model. To do that, we simply call `dataset.export()`."
# =============================================================================

print("\nExecuting dataset.export()...\n")
clean_dataframe = dataset.export()

print("Type of exported object:", type(clean_dataframe))
print(clean_dataframe)

# =============================================================================
# HOST (Voiceover):
# "It returns a standard Pandas DataFrame! No vendor lock-in. You clean it 
# with OMR, and use it anywhere."
# [SCENE END]
# =============================================================================
