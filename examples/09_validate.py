"""
OMR Example 09: Schema Validation
====================================
.validate(rules) enforces business logic rules against the dataset.
It checks that column values satisfy the constraints you define, not just
that they are structurally valid.

Rules are defined as a dictionary mapping column names to schema validators
from the `omr.schemas` module:

  - schemas.PositiveInteger(min, max) : enforces a numeric range
  - schemas.Email()                   : validates email format via regex
  - schemas.OneOf(*values)            : restricts to an allowed set of values

.validate() reports exactly which rows failed each rule, the column involved,
and the invalid value found. This is used for enforcing data contracts at
pipeline entry points.
"""

import pandas as pd
from omr import Dataset, schemas

# A user dataset with several business rule violations:
#   - Row 1: age=17, below the minimum allowed age of 18
#   - Row 1: email='b.com', missing the '@' character (invalid format)
#   - Row 3: status='banned', not in the allowed values ('active', 'pending')
df = pd.DataFrame({
    "user_id": [1, 2, 3, 4],
    "age":     [25, 17, 30, 45],
    "email":   ["a@a.com", "b.com", "c@c.com", "d@d.com"],
    "status":  ["active", "active", "pending", "banned"]
})

dataset = Dataset(df)

# Define the business rules as a schema dictionary.
# Each key is a column name; each value is a validator instance.
business_rules = {
    "age":    schemas.PositiveInteger(min=18, max=120),
    "email":  schemas.Email(),
    "status": schemas.OneOf("active", "pending")
}

# Run validation against all defined rules.
# The output lists each failing row with the column name, the rule that
# was violated, and the actual value that caused the failure.
print("Executing dataset.validate()...\n")
dataset.validate(business_rules)
