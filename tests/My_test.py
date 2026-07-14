import pandas as pd
from omr import Dataset, schemas, Monitor

df = pd.DataFrame({
    "age":    [25, 30, 35, None, 42, -5, 28, 200],
    "salary": [50000, 75000, None, 62000, 90000, 80000, 55000, 71000],
    "name":   ["Alice", "Bob", "Bob", "Dana", "n/a", "Frank", "Grace", None],
    "score":  [1, "high", "high", 3, 4, 2, "low", 4],
    "churn":  [0, 0, 0, 1, 0, 1, 0, 1],
})

dataset = Dataset(df)

report = dataset.health()

dataset.clean()

dataset.explain_changes()

schema = {
    "age":    schemas.PositiveInteger(min=0, max=120),
    "salary": schemas.PositiveFloat(min=10000),
    "churn":  schemas.OneOf(0, 1),
}
dataset.validate(schema)

dataset.analyze()

clean_df = dataset.export()

