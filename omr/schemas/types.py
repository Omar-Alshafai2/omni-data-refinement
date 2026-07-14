"""
OMR Schema Types — Constraint classes for dataset validation.

Usage:
    from omr.schemas import PositiveInteger, PositiveFloat, Email, OneOf

    schema = {
        "age":    PositiveInteger(max=120),
        "salary": PositiveFloat(),
        "email":  Email(),
        "status": OneOf("active", "inactive"),
    }
    dataset.validate(schema)
"""
from __future__ import annotations
import re
from typing import Any, Optional, Sequence
import pandas as pd


class ConstraintType:
    """Base class for all schema constraint types."""

    def validate(self, series: pd.Series) -> list[str]:
        """
        Validate a pandas Series against this constraint.

        Returns:
            list[str]: A list of human-readable error messages.
                       Empty list means the series passes all checks.
        """
        raise NotImplementedError

    def description(self) -> str:
        """Human-readable description of this constraint."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.description()})"


# ── Numeric Constraints ──────────────────────────────────────────────────────

class PositiveInteger(ConstraintType):
    """Validates that all values are positive integers (> 0).

    Args:
        min (int): Minimum allowed value (default: 1).
        max (int): Maximum allowed value (default: unlimited).
        not_null (bool): If True, missing values are not allowed (default: True).
    """

    def __init__(self, min: int = 1, max: Optional[int] = None, not_null: bool = True):
        self.min = min
        self.max = max
        self.not_null = not_null

    def validate(self, series: pd.Series) -> list[str]:
        errors = []
        if self.not_null:
            nulls = series.isnull().sum()
            if nulls > 0:
                errors.append(f"{nulls} null value(s) found (not_null=True).")

        non_null = series.dropna()
        numeric = pd.to_numeric(non_null, errors="coerce")
        non_numeric = numeric.isnull().sum()
        if non_numeric > 0:
            errors.append(f"{non_numeric} value(s) are not numeric.")
            return errors

        non_int = (numeric % 1 != 0).sum()
        if non_int > 0:
            errors.append(f"{non_int} value(s) are not integers.")

        below_min = (numeric < self.min).sum()
        if below_min > 0:
            errors.append(f"{below_min} value(s) are below minimum {self.min}.")

        if self.max is not None:
            above_max = (numeric > self.max).sum()
            if above_max > 0:
                errors.append(f"{above_max} value(s) exceed maximum {self.max}.")

        return errors

    def description(self) -> str:
        parts = [f"min={self.min}"]
        if self.max is not None:
            parts.append(f"max={self.max}")
        if not self.not_null:
            parts.append("nullable")
        return ", ".join(parts)


class PositiveFloat(ConstraintType):
    """Validates that all values are positive floats (> 0.0).

    Args:
        min (float): Minimum allowed value (default: 0.0, exclusive).
        max (float): Maximum allowed value (default: unlimited).
        not_null (bool): If True, missing values are not allowed (default: True).
    """

    def __init__(self, min: float = 0.0, max: Optional[float] = None, not_null: bool = True):
        self.min = min
        self.max = max
        self.not_null = not_null

    def validate(self, series: pd.Series) -> list[str]:
        errors = []
        if self.not_null:
            nulls = series.isnull().sum()
            if nulls > 0:
                errors.append(f"{nulls} null value(s) found (not_null=True).")

        non_null = series.dropna()
        numeric = pd.to_numeric(non_null, errors="coerce")
        non_numeric = numeric.isnull().sum()
        if non_numeric > 0:
            errors.append(f"{non_numeric} value(s) are not numeric.")
            return errors

        below_min = (numeric <= self.min).sum()
        if below_min > 0:
            errors.append(f"{below_min} value(s) are not positive (≤ {self.min}).")

        if self.max is not None:
            above_max = (numeric > self.max).sum()
            if above_max > 0:
                errors.append(f"{above_max} value(s) exceed maximum {self.max}.")

        return errors

    def description(self) -> str:
        parts = [f"min>{self.min}"]
        if self.max is not None:
            parts.append(f"max={self.max}")
        if not self.not_null:
            parts.append("nullable")
        return ", ".join(parts)


class NonNegative(ConstraintType):
    """Validates that all values are >= 0."""

    def __init__(self, not_null: bool = True):
        self.not_null = not_null

    def validate(self, series: pd.Series) -> list[str]:
        errors = []
        if self.not_null and series.isnull().sum() > 0:
            errors.append(f"{series.isnull().sum()} null value(s) found.")
        numeric = pd.to_numeric(series.dropna(), errors="coerce")
        negative = (numeric < 0).sum()
        if negative > 0:
            errors.append(f"{negative} negative value(s) found.")
        return errors

    def description(self) -> str:
        return "value >= 0"


# ── Categorical Constraints ──────────────────────────────────────────────────

class OneOf(ConstraintType):
    """Validates that all values belong to a set of allowed values.

    Args:
        *values: The allowed values.
        not_null (bool): If True, missing values are not allowed (default: True).

    Example:
        OneOf("active", "inactive", "pending")
    """

    def __init__(self, *values: Any, not_null: bool = True):
        self.values = set(values)
        self.not_null = not_null

    def validate(self, series: pd.Series) -> list[str]:
        errors = []
        if self.not_null and series.isnull().sum() > 0:
            errors.append(f"{series.isnull().sum()} null value(s) found.")
        invalid = ~series.dropna().isin(self.values)
        count = invalid.sum()
        if count > 0:
            sample = series.dropna()[invalid].unique()[:3].tolist()
            errors.append(f"{count} value(s) not in allowed set {sorted(str(v) for v in self.values)[:5]}. Found: {sample}")
        return errors

    def description(self) -> str:
        return f"one of {sorted(str(v) for v in self.values)}"


# ── String / Format Constraints ──────────────────────────────────────────────

class Email(ConstraintType):
    """Validates that all values are valid email addresses."""
    _PATTERN = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

    def __init__(self, not_null: bool = True):
        self.not_null = not_null

    def validate(self, series: pd.Series) -> list[str]:
        errors = []
        if self.not_null and series.isnull().sum() > 0:
            errors.append(f"{series.isnull().sum()} null value(s) found.")
        invalid = series.dropna().apply(lambda x: not bool(self._PATTERN.match(str(x)))).sum()
        if invalid > 0:
            errors.append(f"{invalid} value(s) are not valid email addresses.")
        return errors

    def description(self) -> str:
        return "valid email address"


class Regex(ConstraintType):
    """Validates that all values match a regular expression pattern.

    Args:
        pattern (str): The regex pattern to match.
        not_null (bool): If True, missing values are not allowed (default: True).

    Example:
        Regex(r"^\\d{4}-\\d{2}-\\d{2}$")  # Date format YYYY-MM-DD
    """

    def __init__(self, pattern: str, not_null: bool = True):
        self.pattern = pattern
        self._compiled = re.compile(pattern)
        self.not_null = not_null

    def validate(self, series: pd.Series) -> list[str]:
        errors = []
        if self.not_null and series.isnull().sum() > 0:
            errors.append(f"{series.isnull().sum()} null value(s) found.")
        invalid = series.dropna().apply(lambda x: not bool(self._compiled.match(str(x)))).sum()
        if invalid > 0:
            errors.append(f"{invalid} value(s) do not match pattern '{self.pattern}'.")
        return errors

    def description(self) -> str:
        return f"matches r'{self.pattern}'"


class NotNull(ConstraintType):
    """Validates that the column has no missing values."""

    def validate(self, series: pd.Series) -> list[str]:
        nulls = series.isnull().sum()
        return [f"{nulls} null value(s) found."] if nulls > 0 else []

    def description(self) -> str:
        return "not null"


class MinLength(ConstraintType):
    """Validates minimum string length."""

    def __init__(self, length: int):
        self.length = length

    def validate(self, series: pd.Series) -> list[str]:
        too_short = series.dropna().apply(lambda x: len(str(x)) < self.length).sum()
        return [f"{too_short} value(s) are shorter than {self.length} characters."] if too_short > 0 else []

    def description(self) -> str:
        return f"min length {self.length}"


class MaxLength(ConstraintType):
    """Validates maximum string length."""

    def __init__(self, length: int):
        self.length = length

    def validate(self, series: pd.Series) -> list[str]:
        too_long = series.dropna().apply(lambda x: len(str(x)) > self.length).sum()
        return [f"{too_long} value(s) exceed {self.length} characters."] if too_long > 0 else []

    def description(self) -> str:
        return f"max length {self.length}"


# ── Convenience aliases ───────────────────────────────────────────────────────

Integer   = PositiveInteger
Float     = PositiveFloat
String    = NotNull
