"""
OMR Integrations Module — Interoperability with other data tools.
"""
import pandas as pd


def ensure_pandas(data) -> pd.DataFrame:
    """
    Ensures the input is a pandas DataFrame.
    If it's a Polars DataFrame, it converts it seamlessly.
    """
    if isinstance(data, pd.DataFrame):
        return data

    type_name = type(data).__name__
    
    # Check for Polars (without importing polars to avoid hard dependency)
    if type_name == "DataFrame" and data.__class__.__module__.startswith("polars"):
        return data.to_pandas()
        
    # Check for numpy array
    if type_name == "ndarray":
        return pd.DataFrame(data)

    raise TypeError(f"OMR Dataset expects a pandas or polars DataFrame, got {type_name}")
