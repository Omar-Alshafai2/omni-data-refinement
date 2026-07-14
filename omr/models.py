from typing import List, Dict, Any, Optional
from .utils.formatting import print_report, print_profile, print_validation, print_bias, print_leakage, print_drift


class Issue:
    """Represents a specific issue found in the dataset."""
    def __init__(self, issue_type: str, column: str, severity: str, description: str, recommendation: str, context: Dict[str, Any] = None):
        self.issue_type = issue_type
        self.column = column
        self.severity = severity
        self.description = description
        self.recommendation = recommendation
        self.context = context or {}


class DiagnosticReport:
    """Holds the results of a dataset diagnosis."""
    def __init__(self, num_rows: int, num_cols: int):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.issues: List[Issue] = []

    def add_issue(self, issue: Issue):
        self.issues.append(issue)

    def display(self):
        print_report(self)


class ColumnProfile:
    """Statistical profile for a single column."""
    def __init__(self, name: str, dtype: str, count: int, missing: int, missing_pct: float,
                 unique: int, top_values: List[Any] = None,
                 mean: float = None, median: float = None, std: float = None,
                 min_val: Any = None, max_val: Any = None):
        self.name = name
        self.dtype = dtype
        self.count = count
        self.missing = missing
        self.missing_pct = missing_pct
        self.unique = unique
        self.top_values = top_values or []
        self.mean = mean
        self.median = median
        self.std = std
        self.min_val = min_val
        self.max_val = max_val


class ProfilingReport:
    """Full statistical profile of a dataset."""
    def __init__(self, num_rows: int, num_cols: int, memory_mb: float, columns: List[ColumnProfile] = None):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.memory_mb = memory_mb
        self.columns: List[ColumnProfile] = columns or []

    def display(self):
        print_profile(self)


class ValidationIssue:
    """Represents a single validation failure."""
    def __init__(self, column: str, rule: str, description: str, failing_rows: int):
        self.column = column
        self.rule = rule
        self.description = description
        self.failing_rows = failing_rows


class ValidationReport:
    """Holds results of schema and rule validation."""
    def __init__(self):
        self.passed: bool = True
        self.issues: List[ValidationIssue] = []

    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)
        self.passed = False

    def display(self):
        print_validation(self)


class ChangeLogEntry:
    """Records a single transformation applied to the dataset."""
    def __init__(self, step: int, column: str, action: str, rows_affected: int, reason: str):
        self.step = step
        self.column = column
        self.action = action
        self.rows_affected = rows_affected
        self.reason = reason


class ChangeLog:
    """Tracks all transformations applied to the dataset."""
    def __init__(self):
        self.entries: List[ChangeLogEntry] = []
        self._step = 0

    def log(self, column: str, action: str, rows_affected: int, reason: str):
        self._step += 1
        self.entries.append(ChangeLogEntry(self._step, column, action, rows_affected, reason))


class OutlierResult:
    """Holds detected outliers for a column."""
    def __init__(self, column: str, method: str, count: int, values: List[Any], indices: List[int]):
        self.column = column
        self.method = method
        self.count = count
        self.values = values
        self.indices = indices


class BiasIssue:
    """Represents a bias or fairness concern."""
    def __init__(self, column: str, issue_type: str, description: str, recommendation: str, context: Dict[str, Any] = None):
        self.column = column
        self.issue_type = issue_type
        self.description = description
        self.recommendation = recommendation
        self.context = context or {}


class BiasReport:
    """Holds all detected bias and fairness issues."""
    def __init__(self):
        self.issues: List[BiasIssue] = []

    def add_issue(self, issue: BiasIssue):
        self.issues.append(issue)

    def display(self):
        print_bias(self)


class LeakageIssue:
    """Represents a data leakage risk."""
    def __init__(self, feature: str, target: str, correlation: float, risk: str, description: str):
        self.feature = feature
        self.target = target
        self.correlation = correlation
        self.risk = risk
        self.description = description


class LeakageReport:
    """Holds all detected leakage risks."""
    def __init__(self):
        self.issues: List[LeakageIssue] = []

    def add_issue(self, issue: LeakageIssue):
        self.issues.append(issue)

    def display(self):
        print_leakage(self)


class DriftResult:
    """Represents drift detected in a single column."""
    def __init__(self, column: str, train_mean: float, prod_mean: float,
                 train_std: float, prod_std: float, drift_pct: float, severity: str):
        self.column = column
        self.train_mean = train_mean
        self.prod_mean = prod_mean
        self.train_std = train_std
        self.prod_std = prod_std
        self.drift_pct = drift_pct
        self.severity = severity


class DriftReport:
    """Holds all detected data drift between two datasets."""
    def __init__(self):
        self.results: List[DriftResult] = []

    def add_result(self, result: DriftResult):
        self.results.append(result)

    def display(self):
        print_drift(self)
