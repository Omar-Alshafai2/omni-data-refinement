"""
OMR Cleaning Strategies — Built-in logic for handling messy data.
"""
import pandas as pd
import numpy as np


class ImputationStrategy:
    def apply(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError


class MedianImputation(ImputationStrategy):
    def apply(self, series: pd.Series) -> pd.Series:
        return series.fillna(series.median())


class ModeImputation(ImputationStrategy):
    def apply(self, series: pd.Series) -> pd.Series:
        mode = series.mode()
        val = mode[0] if not mode.empty else "Unknown"
        return series.fillna(val)


class ConstantImputation(ImputationStrategy):
    def __init__(self, value):
        self.value = value

    def apply(self, series: pd.Series) -> pd.Series:
        return series.fillna(self.value)


def resolve_mixed_types(series: pd.Series) -> pd.Series:
    """Casts mixed types uniformly to string."""
    return series.astype(str)
