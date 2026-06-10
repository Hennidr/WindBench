"""Restructure farm DataFrames into 3-D seq2seq arrays.

The per-horizon benchmark treats each (production_time, lead_time) pair as an
independent sample.  The seq2seq benchmark instead treats each NWP *run*
(identified by prediction_issue_time) as one sample: the model receives all
NWP forecasts for that run (T lead-time steps) and predicts energy at every
step simultaneously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_COLS = ["2d", "2t", "10u", "10v", "100u", "100v", "msl", "tp", "sd", "sf", "cp", "lsp", "sp"]
DEFAULT_LEAD_TIMES = list(range(1, 73))   # h=1..72 (exclude h=0: known at issue time)


def build_seq2seq_arrays(
    df: pd.DataFrame,
    weather_cols: list[str] = WEATHER_COLS,
    lead_times: list[int] | None = None,
    min_lead_coverage: float = 0.8,
    fill_method: str = "interpolate",
    target_col: str = "energy_total",
    n_lags: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Pivot a long farm DataFrame into 3-D seq2seq arrays.

    Parameters
    ----------
    df:
        Output of ``load_farm()``: DatetimeIndex = production_time, columns
        include weather variables, ``target_col``, and ``prediction_issue_time``.
    weather_cols:
        NWP feature column names to include as inputs.
    lead_times:
        Ordered list of integer lead-time hours to include.
        Defaults to ``range(1, 73)``.
    min_lead_coverage:
        Minimum fraction of lead times that must be non-NaN for a run to be
        kept.  Runs at dataset boundaries (only 12 or 24 hours covered) are
        dropped.
    fill_method:
        How to fill isolated NaN gaps within an otherwise complete run.
        ``"interpolate"`` (linear, max 3 steps), ``"ffill"``, or ``"zero"``.
    target_col:
        Energy target column name.
    n_lags:
        Number of hourly observed-production values to append as additional
        input features.  For a run issued at time ``t``, the lags are the
        observed ``target_col`` at ``t-1h, t-2h, ..., t-n_lags*h``.
        These are broadcast as constant values across all T lead-time steps,
        giving every model the same recent-production context.
        Defaults to 0 (disabled).

    Returns
    -------
    X : np.ndarray, shape (n_runs, T, n_features + n_lags)  float32
    y : np.ndarray, shape (n_runs, T)                       float32, NaN where missing
    run_times : pd.DatetimeIndex of prediction_issue_time for each run
    """
    if lead_times is None:
        lead_times = DEFAULT_LEAD_TIMES

    T = len(lead_times)
    df = df.copy()
    df.index.name = "production_time"
    df = df.reset_index()

    # Build production-time energy lookup for lag features (before any filtering)
    if n_lags > 0:
        energy_lookup: pd.Series = (
            df.groupby("production_time")[target_col].first()
        )

    # Ensure prediction_issue_time is datetime (utc=True handles mixed-tz NZ data)
    df["prediction_issue_time"] = pd.to_datetime(df["prediction_issue_time"], utc=True)
    df["production_time"] = pd.to_datetime(df["production_time"], utc=True)
    df["lead_time_h"] = df["lead_time_h"].astype(int)

    # ── Pivot energy target ───────────────────────────────────────────────
    y_pivot = (
        df.pivot_table(
            index="prediction_issue_time",
            columns="lead_time_h",
            values=target_col,
            aggfunc="first",
        )
        .reindex(columns=lead_times)
    )

    # ── Pivot each weather feature ────────────────────────────────────────
    X_pivots: list[np.ndarray] = []
    for col in weather_cols:
        piv = (
            df.pivot_table(
                index="prediction_issue_time",
                columns="lead_time_h",
                values=col,
                aggfunc="mean",
            )
            .reindex(columns=lead_times)
        )
        X_pivots.append(piv.values)   # (n_runs, T)

    X_raw = np.stack(X_pivots, axis=-1).astype(np.float32)   # (n_runs, T, F)
    y_raw = y_pivot.values.astype(np.float32)                  # (n_runs, T)
    run_times = pd.DatetimeIndex(y_pivot.index)

    # ── Drop runs with insufficient coverage ─────────────────────────────
    valid_frac = (~np.isnan(y_raw)).mean(axis=1)               # (n_runs,)
    keep = valid_frac >= min_lead_coverage
    X_raw, y_raw, run_times = X_raw[keep], y_raw[keep], run_times[keep]

    # ── Fill isolated NaN gaps ────────────────────────────────────────────
    def _fill(arr2d: np.ndarray) -> np.ndarray:
        """Fill NaN along axis=1 (lead-time axis) per run."""
        df_tmp = pd.DataFrame(arr2d)
        if fill_method == "interpolate":
            df_tmp = df_tmp.interpolate(axis=1, limit=3)
        elif fill_method == "ffill":
            df_tmp = df_tmp.ffill(axis=1).bfill(axis=1)
        elif fill_method == "zero":
            df_tmp = df_tmp.fillna(0.0)
        return df_tmp.values.astype(np.float32)

    y_raw = _fill(y_raw)
    for f in range(X_raw.shape[-1]):
        X_raw[:, :, f] = _fill(X_raw[:, :, f])

    # ── Append observed-production lags ──────────────────────────────────────
    if n_lags > 0:
        lag_array = np.zeros((len(run_times), n_lags), dtype=np.float32)
        for r, rt in enumerate(run_times):
            for lag in range(n_lags):
                pt = rt - pd.Timedelta(hours=lag + 1)
                val = energy_lookup.get(pt, np.nan)
                lag_array[r, lag] = 0.0 if np.isnan(val) else val
        # Broadcast (n_runs, n_lags) → (n_runs, T, n_lags) and concatenate
        lag_broadcast = np.broadcast_to(
            lag_array[:, np.newaxis, :], (len(run_times), T, n_lags)
        ).copy()
        X_raw = np.concatenate([X_raw, lag_broadcast], axis=-1)

    return X_raw, y_raw, run_times


def build_production_hour_arrays(
    X_runs: np.ndarray,
    y_runs: np.ndarray,
    run_times: pd.DatetimeIndex,
    n_lags: int = 6,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Restructure run arrays into production-hour arrays.

    The seq2seq arrays are organised by NWP run: row = run, column = lead step.
    This function transposes the view: row = production hour, column = which run
    (ordered from freshest to oldest).

    For each unique production hour ``t`` every training run that happens to
    forecast ``t`` contributes one row in the lag axis, sorted ascending by
    lead time so that index 0 = most recent NWP run, index 1 = next most
    recent, etc.

    Parameters
    ----------
    X_runs : (n_runs, T, F)
    y_runs : (n_runs, T)  — may contain NaN
    run_times : DatetimeIndex of NWP issue times, length n_runs
    n_lags : number of lag slots to keep (oldest runs beyond this are dropped)

    Returns
    -------
    X_ph : (n_hours, n_lags, F)  — NaN where run not available for that slot
    y_ph : (n_hours,)            — energy from the most-recent non-NaN run
    ph_times : pd.DatetimeIndex of production hours (sorted)
    ph_leads : (n_hours, n_lags) — actual lead_time_h per slot (NaN if empty)
    """
    from collections import defaultdict

    n_runs, T, F = X_runs.shape
    # ph_dict: production_time → list of (lead_h, weather_features, energy)
    ph_dict: dict = defaultdict(list)

    for r in range(n_runs):
        issue = run_times[r]
        for h in range(T):
            lead = h + 1   # DEFAULT_LEAD_TIMES are 1-indexed
            prod_time = issue + pd.Timedelta(hours=lead)
            ph_dict[prod_time].append(
                (lead, X_runs[r, h, :].copy(), float(y_runs[r, h]))
            )

    ph_times_sorted = sorted(ph_dict.keys())
    n_hours = len(ph_times_sorted)

    X_ph    = np.full((n_hours, n_lags, F), np.nan, dtype=np.float32)
    y_ph    = np.full(n_hours, np.nan, dtype=np.float32)
    ph_leads = np.full((n_hours, n_lags), np.nan, dtype=np.float32)

    for i, pt in enumerate(ph_times_sorted):
        # Sort ascending by lead → freshest (smallest lead) first
        entries = sorted(ph_dict[pt], key=lambda x: x[0])

        # Energy: use most-recent non-NaN run
        for lead, feats, energy in entries:
            if not np.isnan(energy):
                y_ph[i] = energy
                break

        for j, (lead, features, _) in enumerate(entries[:n_lags]):
            X_ph[i, j, :] = features
            ph_leads[i, j] = lead

    return X_ph, y_ph, pd.DatetimeIndex(ph_times_sorted), ph_leads


def production_hour_temporal_split(
    X_ph: np.ndarray,
    y_ph: np.ndarray,
    ph_times: pd.DatetimeIndex,
    ph_leads: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> tuple:
    """Chronological train / val / test split of production-hour arrays."""
    n = len(X_ph)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_val - n_test

    slices = [
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, n),
    ]
    return tuple(
        arr[s]
        for arr in (X_ph, y_ph, ph_times, ph_leads)
        for s in slices
    )


def seq2seq_temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    run_times: pd.DatetimeIndex,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex,
]:
    """Chronological train / val / test split of run arrays.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, times_train, times_val, times_test
    """
    n = len(X)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    X_train, y_train, t_train = X[:n_train], y[:n_train], run_times[:n_train]
    X_val,   y_val,   t_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val], run_times[n_train:n_train+n_val]
    X_test,  y_test,  t_test  = X[n_train+n_val:], y[n_train+n_val:], run_times[n_train+n_val:]

    return X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test
