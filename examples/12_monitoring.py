"""
OMR Example 12: Continuous Production Monitoring
==================================================
In production, data arrives continuously in batches (daily, hourly, etc.).
If an upstream pipeline breaks — for example, a column becomes entirely null
or a distribution shifts dramatically — the model will silently produce
incorrect predictions until the issue is manually discovered.

The OMR Monitor class enables automated quality surveillance:

  monitor.watch(baseline_dataset)
      Registers a baseline dataset. All future batches are compared
      against this reference distribution.

  monitor.check(new_batch_df)
      Compares an incoming DataFrame against the baseline. Returns a list
      of Alert objects, each containing:
        - severity  : the alert level (e.g., "HIGH", "MEDIUM")
        - check     : the name of the check that triggered the alert
        - message   : a human-readable description of the anomaly detected

Integrate monitor.check() into an Airflow DAG, a cron job, or any
data pipeline orchestrator to catch regressions before they reach the model.
"""

import pandas as pd
from omr import Dataset, Monitor

# Establish the baseline: the known-good distribution of the data.
# This is typically the dataset used to train or validate the model.
baseline_df = pd.DataFrame({"sales": [100, 110, 105, 95, 100]})
baseline_dataset = Dataset(baseline_df)

print("Initializing Monitor...")
monitor = Monitor()
monitor.watch(baseline_dataset)

# Simulate a broken upstream pipeline where sales values collapse to near zero.
# In a real system, this new_batch would come from a database query or file read.
print("\nNew daily batch arrives (simulating a broken pipeline)...")
new_batch = pd.DataFrame({"sales": [1, 0, 2, 0, 1]})

# Run the monitoring check.
# The monitor compares the new batch's statistics against the baseline.
# It detects the severe mean shift (from ~100 to ~0.8) and returns alerts.
print("Executing monitor.check(new_batch)...\n")
alerts = monitor.check(new_batch)

# Print each alert returned by the monitor.
# In production, pipe these alerts to Slack, PagerDuty, or an alerting system
# to notify engineers before the bad data propagates to downstream models.
for alert in alerts:
    print(f"[{alert.severity} ALERT] {alert.check}: {alert.message}")
