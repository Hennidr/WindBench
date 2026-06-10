"""Load per-farm Parquet files into clean DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_KNOWN_META = {
    "production_time", "prediction_issue_time", "turbine_id",
    "windfarm_id", "longitude", "latitude", "lead_time_h",
}


def _process_df(df: pd.DataFrame, target_col: str = "energy_total") -> pd.DataFrame:
    """Aggregate turbines and compute lead_time_h for a raw farm DataFrame."""
    weather_cols = [c for c in df.columns if c not in _KNOWN_META and not c.startswith("energy_")]

    group_keys = [c for c in ["production_time", "prediction_issue_time"] if c in df.columns]
    agg = {"energy_total": "sum"}
    agg.update({c: "mean" for c in weather_cols})

    df = df.groupby(group_keys, sort=True).agg(agg).reset_index()

    if "prediction_issue_time" in df.columns:
        df["lead_time_h"] = (
            pd.to_datetime(df["production_time"], utc=True) - pd.to_datetime(df["prediction_issue_time"], utc=True)
        ).dt.total_seconds() / 3600

    df["production_time"] = pd.to_datetime(df["production_time"], utc=True)
    df = df.set_index("production_time")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available: {list(df.columns)}"
        )

    return df


def load_farm(path: str | Path, target_col: str = "energy_total") -> pd.DataFrame:
    """Load a single farm Parquet file, aggregated to farm level.

    Turbines are merged by summing energy columns and averaging weather columns
    for each (production_time, prediction_issue_time) pair.

    Parameters
    ----------
    path:
        Path to the ``.parquet`` file.
    target_col:
        Name of the production target column.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``production_time`` as index, one row per
        (production_time, prediction_issue_time), farm-level energy, and
        spatially averaged weather features.
    """
    return _process_df(pd.read_parquet(Path(path)), target_col=target_col)


def load_all_farms(
    raw_dir: str | Path,
    target_col: str = "energy_total",
    pattern: str = "*.parquet",
) -> dict[str, pd.DataFrame]:
    """Load all farm Parquet files from a directory.

    Files with a single ``windfarm_id`` are loaded as one farm (name = file stem).
    Files with multiple ``windfarm_id`` values are split per farm
    (name = ``{stem}_farm{id}``).

    Parameters
    ----------
    raw_dir:
        Directory containing ``.parquet`` files.
    target_col:
        Name of the production target column.
    pattern:
        Glob pattern used to find files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{farm_name: DataFrame}`` where *farm_name* is the file stem,
        or ``{stem}_farm{id}`` for multi-farm files.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {raw_dir}")

    farms: dict[str, pd.DataFrame] = {}
    for f in files:
        raw = pd.read_parquet(f)
        if "windfarm_id" in raw.columns and raw["windfarm_id"].nunique() > 1:
            for farm_id, subset in raw.groupby("windfarm_id"):
                name = f"{f.stem}_farm{farm_id}"
                farms[name] = _process_df(subset.copy(), target_col=target_col)
        else:
            farms[f.stem] = _process_df(raw, target_col=target_col)
    return farms
