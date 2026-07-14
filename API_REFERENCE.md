# OMR — Omni Data Refinement: Professional API Reference

Welcome to the definitive API reference for **OMR**. This documentation covers every class, method, argument, and schema available in the library to help you build robust data intelligence pipelines.

---

## Table of Contents
1. [Core API: `Dataset`](#1-core-api-dataset)
   - Core & Quality
   - Cleaning & Transformation
   - Advanced Analytics
   - Schema Validation
   - Ops, Versioning & Export
2. [Validation Schemas: `omr.schemas`](#2-validation-schemas-omrschemas)
3. [Continuous Monitoring: `Monitor`](#3-continuous-monitoring-monitor)

---

## 1. Core API: `Dataset`

The `Dataset` class is the main entry point to OMR. It wraps a Pandas or Polars DataFrame and attaches intelligence engines to it.

```python
from omr import Dataset
import pandas as pd

df = pd.read_csv("data.csv")
dataset = Dataset(df)
```

### Core & Quality

#### `Dataset.summary()`
Prints a highly optimized, single-line summary of your dataset. Ideal for quick pulse checks.
* **Returns:** `Dataset` (returns self for chaining)
* **Under the hood:** Computes essential metadata (shape, total missing cells, duplicate rows) and runs a fast heuristic health score without deep statistical overhead.
* **Example:**
  ```python
  dataset.summary()
  # > [Dataset] 1000 rows × 5 cols | Missing: 12 (1.2%) | Duplicates: 0 | Health: 98.5/100
  ```

#### `Dataset.health()`
Executes the comprehensive 5-Pillar Data Quality assessment (Completeness, Uniqueness, Consistency, Validity, Conformity).
* **Returns:** `HealthReport` object containing the `.score` and `.issues` list.
* **Under the hood:** Spawns a `HealthEngine` that analyzes null density, duplication rates, type uniformity, and logical constraints to compute a weighted health score.
* **Example:**
  ```python
  report = dataset.health()
  print(f"Dataset Health: {report.score}/100")
  ```

#### `Dataset.profile()`
Generates a full statistical profile for every column in the dataset.
* **Returns:** `Dataset` (returns self for chaining)
* **Under the hood:** Extracts data types, missing counts, unique cardinalities, means, modes, minimums, and maximums. Designed to replace Pandas `.info()` and `.describe()`.
* **Example:**
  ```python
  dataset.profile()
  ```

### Cleaning & Transformation

#### `Dataset.clean()`
Automatically resolves all data quality issues detected during the `health()` check.
* **Returns:** `Dataset` (returns self for chaining)
* **Under the hood:** Uses the `CleaningEngine`. It intelligently imputes missing values (using mode for categorical, median/mean for continuous), drops pure duplicate rows, casts mixed data types to uniform types, and standardizes formats.
* **Example:**
  ```python
  dataset.health()  # Identifies the issues
  dataset.clean()   # Fixes them automatically
  ```

#### `Dataset.explain_changes()`
Prints a detailed "Transformation Log" showing exactly what `clean()` (or any other mutation) changed.
* **Returns:** `Dataset` (returns self for chaining)
* **Under the hood:** Reads from `DatasetMetadata.transformations` to provide an auditable log of operations, columns affected, row counts, and the algorithmic reasoning.
* **Example:**
  ```python
  dataset.clean()
  dataset.explain_changes()
  ```

### Advanced Analytics

#### `Dataset.analyze()`
Runs deep statistical analysis to detect complex machine learning hazards.
* **Returns:** `Dataset` (returns self for chaining)
* **Under the hood:** The `StatisticsEngine` detects extreme outliers (via IQR/Z-score), highly skewed distributions, multicollinearity between features, constant/zero-variance features, and class imbalances.
* **Example:**
  ```python
  dataset.analyze()
  ```

#### `Dataset.compare(other: "Dataset")`
Detects data drift by comparing the statistical distributions of the current dataset against a reference dataset.
* **Arguments:**
  * `other` (`Dataset`): The reference dataset (e.g., historical or production data).
* **Returns:** `Dataset` (returns self for chaining)
* **Under the hood:** The `DriftEngine` utilizes Population Stability Index (PSI), Kolmogorov-Smirnov (KS) Tests, and Jensen-Shannon Divergence to detect structural distribution shifts.
* **Example:**
  ```python
  prod_data = Dataset(pd.read_csv("prod.csv"))
  dataset.compare(prod_data)
  ```

#### `Dataset.explain(issue: str)`
Acts as a built-in Data Science tutor.
* **Arguments:**
  * `issue` (`str`): The name of the statistical issue (e.g., `"data_leakage"`, `"class_imbalance"`).
* **Returns:** `dict` containing the explanation keys (`definition`, `why_it_matters`, `risks`, `recommended_fixes`).
* **Example:**
  ```python
  dataset.explain("class_imbalance")
  ```

### Schema Validation

#### `Dataset.validate(schema: Dict[str, Any])`
Enforces strict business rules against the dataset.
* **Arguments:**
  * `schema` (`Dict[str, ConstraintType]`): A dictionary mapping column names to OMR schema constraints.
* **Returns:** `Dataset` (returns self for chaining)
* **Under the hood:** The `ValidationEngine` evaluates every row against the schema constraints and flags exactly which rows violate the rules.
* **Example:**
  ```python
  from omr import schemas
  rules = {
      "age": schemas.PositiveInteger(max=120),
      "email": schemas.Email()
  }
  dataset.validate(rules)
  ```

### Ops, Versioning & Export

#### `Dataset.snapshot(name: str = "", description: str = "")`
Saves a checkpoint of your dataset's current state in memory.
* **Arguments:**
  * `name` (`str`): A short name for the snapshot.
  * `description` (`str`): A detailed description of the current state.
* **Returns:** `int` (The unique Version ID).
* **Example:**
  ```python
  v_id = dataset.snapshot(name="v1", description="Pre-cleaning state")
  ```

#### `Dataset.rollback(version_id: int)`
Reverts the dataset back to a previous snapshot in memory.
* **Arguments:**
  * `version_id` (`int`): The ID of the snapshot to revert to.
* **Returns:** `Dataset` (returns self for chaining)
* **Example:**
  ```python
  dataset.rollback(1)
  ```

#### `Dataset.report(format: str = "html", path: str = "omr_report")`
Exports the health and profiling insights to a physical file.
* **Arguments:**
  * `format` (`str`): The format to export (`"html"`, `"markdown"`, `"json"`). Default: `"html"`.
  * `path` (`str`): The base filename (without extension). Default: `"omr_report"`.
* **Returns:** `str` (The absolute file path of the generated report).
* **Example:**
  ```python
  dataset.report(format="html", path="monthly_health_report")
  ```

#### `Dataset.export()`
Returns the cleaned, refined underlying Pandas DataFrame.
* **Returns:** `pd.DataFrame`
* **Example:**
  ```python
  clean_df = dataset.export()
  model.fit(clean_df)
  ```

---

## 2. Validation Schemas: `omr.schemas`

OMR provides built-in constraint types for strict schema validation. Every schema accepts a `not_null` boolean argument (default: `True`), which dictates whether missing values are permitted.

* **`PositiveInteger(min=1, max=None, not_null=True)`**
  Validates that values are integers ≥ `min` and ≤ `max`.
* **`PositiveFloat(min=0.0, max=None, not_null=True)`**
  Validates that values are floats > `min` and ≤ `max`.
* **`NonNegative(not_null=True)`**
  Validates that all numerical values are ≥ 0.
* **`OneOf(*values, not_null=True)`**
  Validates that values belong to an explicit allowed set.
  *Example:* `OneOf("active", "inactive")`
* **`Email(not_null=True)`**
  Validates standard email address formats via Regex.
* **`Regex(pattern: str, not_null=True)`**
  Validates strings against a custom Regular Expression pattern.
  *Example:* `Regex(r"^\d{4}-\d{2}-\d{2}$")`
* **`NotNull()`**
  Ensures the column contains absolutely no missing or null values.
* **`MinLength(length: int)`**
  Ensures string lengths are at least `length` characters long.
* **`MaxLength(length: int)`**
  Ensures string lengths do not exceed `length` characters.

---

## 3. Continuous Monitoring: `Monitor`

The `Monitor` class is used in production pipelines to track dataset health over time and issue alerts when anomalies occur.

#### `Monitor.watch(dataset: Dataset)`
Registers a baseline dataset to establish the mathematical "norm" (means, volumes, null frequencies).
* **Arguments:**
  * `dataset` (`Dataset`): An OMR Dataset instance representing the baseline.
* **Example:**
  ```python
  from omr import Monitor
  monitor = Monitor()
  monitor.watch(historical_dataset)
  ```

#### `Monitor.check(new_data: pd.DataFrame)`
Evaluates incoming data against the registered baseline to detect sudden drops in volume, massive shifts in column means, or spikes in missing values.
* **Arguments:**
  * `new_data` (`pd.DataFrame`): The new data batch to evaluate.
* **Returns:** `list[Alert]`
* **Example:**
  ```python
  alerts = monitor.check(new_daily_batch)
  for alert in alerts:
      print(f"[{alert.severity}] {alert.check}: {alert.message}")
  ```
