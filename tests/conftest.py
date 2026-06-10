"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Synthetic NWP farm DataFrame: 100 runs × 24 lead times, raw format.

    Columns: production_time, prediction_issue_time, lead_time_h,
             2t, 2d, 10u, 10v, 100u, 100v, msl, tp, energy_total
    """
    np.random.seed(0)
    n_runs = 100
    n_lead = 24
    base = pd.Timestamp("2023-01-01")

    rows = []
    for r in range(n_runs):
        issue_time = base + pd.Timedelta(hours=12 * r)
        for h in range(1, n_lead + 1):
            prod_time = issue_time + pd.Timedelta(hours=h)
            energy = max(0.0, 5000 + 2000 * np.sin(2 * np.pi * prod_time.hour / 24)
                         + np.random.randn() * 500)
            rows.append({
                "production_time":      prod_time,
                "prediction_issue_time": issue_time,
                "lead_time_h":          h,
                "2t":   280 + np.random.randn(),
                "2d":   275 + np.random.randn(),
                "10u":  np.random.randn() * 5,
                "10v":  np.random.randn() * 5,
                "100u": np.random.randn() * 8,
                "100v": np.random.randn() * 8,
                "msl":  101325 + np.random.randn() * 500,
                "tp":   max(0.0, np.random.randn() * 0.5),
                "energy_total": energy,
            })

    df = pd.DataFrame(rows)
    df["production_time"]       = pd.to_datetime(df["production_time"])
    df["prediction_issue_time"] = pd.to_datetime(df["prediction_issue_time"])
    return df
