"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Hello everyone! This is our advanced production tutorial. What happens 
after you deploy your model? Data keeps flowing in every single day."

"If a data pipeline breaks upstream, and a column suddenly becomes 100% 
null, your model will silently make terrible predictions. You need 
Continuous Monitoring. Let's look at the OMR `Monitor` class."
=============================================================================
"""

import pandas as pd
from omr import Dataset, Monitor

# =============================================================================
# HOST (Voiceover):
# "First, we establish a baseline. This is our gold-standard dataset."
# =============================================================================

baseline_df = pd.DataFrame({"sales": [100, 110, 105, 95, 100]})
baseline_dataset = Dataset(baseline_df)

print("Initializing Monitor...")
monitor = Monitor()
monitor.watch(baseline_dataset)

# =============================================================================
# HOST (Voiceover):
# "Now, we put the monitor in our daily cron job or Airflow pipeline. 
# Every day, when a new batch of data arrives, we pass it to `monitor.check()`."
# =============================================================================

# Simulate a broken pipeline where sales suddenly drop to near zero
print("\nNew daily batch arrives (Pipeline is broken!)...")
new_batch = pd.DataFrame({"sales": [1, 0, 2, 0, 1]})

print("Executing monitor.check(new_batch)...\n")
alerts = monitor.check(new_batch)

# =============================================================================
# HOST (Voiceover):
# "The monitor compares the new batch against the baseline. If it detects 
# a massive volume drop, or a huge shift in the mean, it throws an Alert!"
# =============================================================================

for alert in alerts:
    print(f"[{alert.severity} ALERT] {alert.check}: {alert.message}")

# =============================================================================
# HOST (Voiceover):
# "As you can see, OMR caught the Mean Shift immediately. You can pipe these 
# alerts to Slack, PagerDuty, or use them to automatically halt the pipeline 
# before bad data ruins your downstream models."
# [SCENE END]
# =============================================================================
