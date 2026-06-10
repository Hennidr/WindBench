"""Feature engineering and time-safe train/test splitting."""

from __future__ import annotations

from typing import Generator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_nwp_ensemble_features(
    df: pd.DataFrame,
    h: int,
    weather_cols: list[str],
) -> pd.DataFrame:
    """Add NWP ensemble spread features for a given horizon.

    For each production_time, computes the standard deviation of each weather
    variable across all NWP runs with ``lead_time_h >= h`` — i.e. all runs that
    were available at decision time ``t - h``.  High spread signals an
    atmospherically uncertain situation and should widen prediction intervals.

    Parameters
    ----------
    df:
        Full farm DataFrame (all lead times), with ``lead_time_h`` column and
        ``production_time`` as the DatetimeIndex.
    h:
        Forecast horizon in hours.  Only the ``lead_time_h == h`` rows are
        returned; spread is computed across ``lead_time_h >= h``.
    weather_cols:
        Weather feature column names to compute spread for.

    Returns
    -------
    pd.DataFrame
        The ``lead_time_h == h`` rows with ``<col>_spread`` columns appended.
        Spread is 0 where only a single NWP run is available.
    """
    # All NWP runs available at decision time t-h
    df_avail = df[df["lead_time_h"] >= h]

    # Per-production-time std across available runs
    spread = df_avail.groupby(df_avail.index)[weather_cols].std().fillna(0.0)
    spread.columns = [f"{c}_spread" for c in weather_cols]

    # Point-forecast rows (the h-lead run)
    df_h = df[df["lead_time_h"] == h].copy()

    return df_h.join(spread, how="left")


def make_features(
    df: pd.DataFrame,
    target_col: str = "energy_total",
) -> pd.DataFrame:
    """Add cyclical time features to a farm DataFrame.

    Encodes hour-of-day, day-of-week, and day-of-year as sin/cos pairs so
    models receive full temporal context without raw datetime values.

    Parameters
    ----------
    df:
        Farm DataFrame with a ``DatetimeIndex``.
    target_col:
        Name of the production target column (kept as-is, not lagged).

    Returns
    -------
    pd.DataFrame
        DataFrame with cyclical time columns appended. No rows are dropped.
    """
    out = df.copy()
    idx = out.index

    out["hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["doy_sin"]   = np.sin(2 * np.pi * idx.dayofyear / 365)
    out["doy_cos"]   = np.cos(2 * np.pi * idx.dayofyear / 365)

    return out


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    target_col: str = "energy_total",
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Strictly temporal train / validation / test split with horizon shifting.

    The target is shifted ``horizon`` steps into the future so every model
    receives the correct h-step-ahead label.  The last ``horizon`` rows are
    dropped (their future target lies beyond the dataset).

    No-leakage guarantee: training targets end at ``t + horizon`` where
    ``t < n_train``.  Test features start at ``t = n_train + n_val``.  As long
    as ``n_val >= horizon`` (true for any realistic dataset), training targets
    never overlap with test features.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame with a ``DatetimeIndex``.
    target_col:
        Column to use as the forecasting target.
    val_frac:
        Fraction of data for validation.
    test_frac:
        Fraction of data for test.
    horizon:
        Forecast horizon in timesteps.  ``y[t]`` becomes ``original_y[t + horizon]``.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Shift target forward by h; drop the last h rows (NaN future targets)
    y_shifted = df[target_col].shift(-horizon)
    df_h = df.copy()
    df_h[target_col] = y_shifted
    df_h = df_h.iloc[:-horizon] if horizon > 0 else df_h
    df_h = df_h.dropna(subset=[target_col])

    n = len(df_h)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    if n_val < horizon:
        import warnings
        warnings.warn(
            f"Validation set ({n_val} steps) is smaller than horizon ({horizon}). "
            "Consider increasing val_frac to avoid potential target leakage.",
            UserWarning,
            stacklevel=2,
        )

    feature_cols = [c for c in df_h.select_dtypes(include="number").columns if c != target_col]
    X = df_h[feature_cols]
    y = df_h[target_col]

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train : n_train + n_val], y.iloc[n_train : n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val :], y.iloc[n_train + n_val :]

    return X_train, X_val, X_test, y_train, y_val, y_test


def rolling_origin_splits(
    df: pd.DataFrame,
    target_col: str = "energy_total",
    n_splits: int = 5,
    test_size: int | None = None,
    min_train_size: int | None = None,
) -> Generator[tuple, None, None]:
    """Rolling-origin (walk-forward) cross-validation splits.

    Yields ``(X_train, X_test, y_train, y_test)`` tuples with strictly
    increasing training windows and fixed-size test windows.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame.
    target_col:
        Forecasting target column.
    n_splits:
        Number of CV splits.
    test_size:
        Number of test samples per split. Defaults to ``len(df) // (n_splits + 1)``.
    min_train_size:
        Minimum training samples. Defaults to ``test_size``.
    """
    n = len(df)
    if test_size is None:
        test_size = n // (n_splits + 1)
    if min_train_size is None:
        min_train_size = test_size

    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    for i in range(n_splits):
        test_end = n - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        train_end = test_start

        if train_end < min_train_size:
            continue

        yield (
            pd.DataFrame(X[:train_end], columns=feature_cols, index=df.index[:train_end]),
            pd.DataFrame(X[test_start:test_end], columns=feature_cols, index=df.index[test_start:test_end]),
            pd.Series(y[:train_end], index=df.index[:train_end], name=target_col),
            pd.Series(y[test_start:test_end], index=df.index[test_start:test_end], name=target_col),
        )
