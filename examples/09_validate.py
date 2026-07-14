"""
=============================================================================
[SCENE START]
HOST (Voiceover): 
"Welcome back! In previous videos, we talked about general data quality. 
But what about Business Logic? What if an age is technically a valid number, 
but in your system, users must be over 18?"

"You need Schema Validation. And OMR's `.validate()` method makes this 
incredibly strict and incredibly easy."
=============================================================================
"""

import pandas as pd
from omr import Dataset, schemas

df = pd.DataFrame({
    "user_id": [1, 2, 3, 4],
    "age": [25, 17, 30, 45],            # 17 is underage
    "email": ["a@a.com", "b.com", "c@c.com", "d@d.com"], # Invalid email 'b.com'
    "status": ["active", "active", "pending", "banned"]  # 'banned' is not allowed
})

dataset = Dataset(df)

# =============================================================================
# HOST (Voiceover):
# "We import `schemas` from OMR. Then we build a dictionary. We map our 
# column names to specific rules. 
# 
# For 'age', we enforce a PositiveInteger with a minimum of 18.
# For 'email', we use the built-in Email regex validator.
# For 'status', we use OneOf to restrict values to an allowed list."
# =============================================================================

business_rules = {
    "age": schemas.PositiveInteger(min=18, max=120),
    "email": schemas.Email(),
    "status": schemas.OneOf("active", "pending")
}

# =============================================================================
# HOST (Voiceover):
# "Now we run `.validate()`. OMR evaluates every single row against these 
# strict rules."
# =============================================================================

print("\nExecuting dataset.validate()...\n")
dataset.validate(business_rules)

# =============================================================================
# HOST (Voiceover):
# "Look at the terminal! It caught the 17-year-old, it caught the badly 
# formatted email, and it caught the invalid 'banned' status. It tells you 
# exactly which rows failed so you can go fix them!"
# [SCENE END]
# =============================================================================
