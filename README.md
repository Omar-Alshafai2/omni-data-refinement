# OMR — Omni Data Refinement

OMR is a pure Python framework for dataset quality, validation, profiling, monitoring, and reliability.

The goal is to become the standard tool developers use immediately after loading a dataset, similar to how Pandas is used for data manipulation. 

OMR is designed to be **Pure Python, Open Source, Modular, Extensible, Enterprise-ready**, and entirely independent of external AI APIs or LLMs.

## Core Vision

Instead of writing dozens of exploratory scripts, you just pass your dataset to OMR.

**Current workflow:**
```python
import pandas as pd
df = pd.read_csv("data.csv")
# Write 50 lines of .isnull(), .duplicated(), and data exploration code...
```

**Desired workflow:**
```python
import pandas as pd
from omr import Dataset

df = pd.read_csv("data.csv")
report = Dataset(df).health()

print(report.score)  # e.g. 87
```

---

## Capabilities

OMR handles 12 complete domains of data intelligence:

1. **Health Engine**: 5-pillar quality score (Completeness, Uniqueness, Consistency, Validity, Conformity).
2. **Cleaning Engine**: Auto-resolution of missing values, duplicates, and type mismatches.
3. **Profiling Engine**: Deep statistical profile of every column.
4. **Validation Engine**: Schema-based validation with typed constraints (`PositiveInteger`, `Email`, etc).
5. **Statistical Engine**: Outlier detection, multicollinearity, skewness, class imbalance.
6. **Drift Engine**: Distribution shift detection using PSI, KS Test, and JS Divergence.
7. **Monitoring System**: Continuous tracking and alerting for data decay.
8. **Explainability**: Rule-based explanations for data science issues.
9. **Versioning**: Snapshots and rollback capabilities for transformations.
10. **Reporting**: Export to HTML, Markdown, or JSON.
11. **Pipelines**: Fluent, chainable API for applying operations.
12. **Plugin Registry**: Extensible via third-party domain-specific packages.

---

## Installation

```bash
pip install omni-data-refinement
```

Dependencies: `pandas`, `numpy`, `rich`. (Optional: `scipy` for KS Drift tests).
Compatible with Pandas, Polars, and NumPy.

---

## Quickstart

### 1. Diagnostic Health Check
Get an immediate quality score and detailed issue list.
```python
import pandas as pd
from omr import Dataset

df = pd.read_csv("messy_data.csv")
dataset = Dataset(df)

# Run health check
report = dataset.health()
print(f"Health Score: {report.score}/100")
```

### 2. Auto-Cleaning
Automatically fix missing values, duplicate rows, and mixed data types.
```python
dataset.clean()
dataset.explain_changes() # View transformation log
```

### 3. Schema Validation
Enforce strict business rules on your dataset.
```python
from omr import schemas

schema = {
    "age": schemas.PositiveInteger(max=120),
    "salary": schemas.PositiveFloat(min=10000),
    "status": schemas.OneOf("active", "inactive"),
    "email": schemas.Email()
}
dataset.validate(schema)
```

### 4. Advanced Analytics & Drift
Detect complex issues or compare against production data.
```python
# Detect outliers, multicollinearity, and zero variance features
dataset.analyze()

# Compare against yesterday's data to detect drift
prod_dataset = Dataset(pd.read_csv("prod_data.csv"))
dataset.compare(prod_dataset)
```

### 5. Fluent API (Pipelines)
Chain commands together.
```python
clean_df = (Dataset(df)
    .health()
    .clean()
    .analyze()
    .export())
```

---

## Architecture (SOLID Principles)

OMR is modular and testable. The architecture is cleanly divided into:
- `omr.core`: The main `Dataset` interface and `DatasetMetadata` state tracking.
- `omr.health`, `omr.cleaning`, `omr.profiling`, `omr.validation`, `omr.statistics`, `omr.drift`, `omr.monitoring`, `omr.explainability`, `omr.versioning`, `omr.pipelines`, `omr.reports`, `omr.plugins`, `omr.integrations`, `omr.utils`, `omr.schemas`.

---

## Extensibility (Plugins)

OMR supports domain-specific extensions.
```python
from omr import register_plugin

@register_plugin("medical")
class MedicalPlugin:
    def validate_ehr(self, df):
        pass
```

---

## License

MIT License. See `LICENSE` for details.
