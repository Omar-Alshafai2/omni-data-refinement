import pytest
import pandas as pd
import numpy as np
from omr import Dataset, schemas

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 3, 5],
        "age": [25, -10, 35, 35, np.nan],
        "salary": [50000, 60000, 70000, 70000, 80000],
        "group": ["A", "B", "A", "A", "C"]
    })

def test_dataset_initialization(sample_df):
    ds = Dataset(sample_df)
    assert ds.metadata.num_rows == 5
    assert ds.metadata.num_cols == 4
    assert ds.metadata.duplicate_rows == 1

def test_dataset_health(sample_df):
    ds = Dataset(sample_df)
    report = ds.health()
    assert report.score < 100
    assert report.completeness < 100
    assert report.uniqueness < 100
    assert len(report.issues) > 0

def test_dataset_clean(sample_df):
    ds = Dataset(sample_df)
    ds.clean()
    
    clean_df = ds.export()
    # Missing value imputed (median of 25, -10, 35, 35 = 30) or dropped duplicates first
    assert len(clean_df) == 4
    assert clean_df.isnull().sum().sum() == 0

def test_dataset_validation(sample_df):
    ds = Dataset(sample_df)
    schema = {
        "age": schemas.PositiveInteger(min=0, max=100)
    }
    # -10 will fail, np.nan will fail (not_null=True by default)
    # Plus one duplicate row might also fail depending on validation order, but let's test engine
    report = ds._validation_engine.run(sample_df, schema)
    assert not report.passed
    assert len(report.issues) > 0

def test_dataset_explainability(sample_df):
    ds = Dataset(sample_df)
    exp = ds.explain("class_imbalance")
    assert "definition" in exp
    assert "why_it_matters" in exp

def test_export_preserves_data(sample_df):
    ds = Dataset(sample_df)
    exported = ds.export()
    pd.testing.assert_frame_equal(sample_df, exported)
    
    exported.iloc[0, 0] = 999
    # Original should not be modified
    assert ds._df.iloc[0, 0] == 1
