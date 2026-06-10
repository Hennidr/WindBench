"""Tests for windbench.data.loader."""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from windbench.data.loader import load_farm, load_all_farms


def test_load_farm_datetime_index(tmp_path, sample_df):
    path = tmp_path / "farm_a.parquet"
    sample_df.to_parquet(path)
    df = load_farm(path, target_col="energy_total")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "energy_total" in df.columns
    assert "prediction_issue_time" in df.columns


def test_load_farm_missing_target(tmp_path, sample_df):
    path = tmp_path / "farm_a.parquet"
    sample_df.to_parquet(path)
    with pytest.raises(ValueError, match="Target column"):
        load_farm(path, target_col="nonexistent")


def test_load_all_farms(tmp_path, sample_df):
    for name in ["alpha", "beta"]:
        sample_df.to_parquet(tmp_path / f"{name}.parquet")
    farms = load_all_farms(tmp_path, target_col="energy_total")
    assert set(farms.keys()) == {"alpha", "beta"}
    for df in farms.values():
        assert isinstance(df.index, pd.DatetimeIndex)


def test_load_all_farms_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_farms(tmp_path, target_col="energy_w")
