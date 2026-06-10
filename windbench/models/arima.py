"""SARIMAX model for wind energy forecasting.

The model is fit on the energy production time series reconstructed from the
training NWP run arrays.  At prediction time, it generates T-step-ahead
forecasts from the end of the training period to cover each test NWP run.

This serves as the *statistical* representative in the benchmark: it exploits
autocorrelation and seasonality in the energy series rather than a learned
mapping from NWP inputs.

Because ARIMA operates on the production-time axis (not the run axis), the
benchmark runner must pass ``run_times`` as a keyword argument to both
:meth:`fit` and :meth:`predict`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from windbench.models.base import BaseSeq2SeqModel


def _build_energy_series(
    y_runs: np.ndarray,
    run_times: pd.DatetimeIndex,
) -> pd.Series:
    """Reconstruct a production-time indexed energy series from run arrays.

    When multiple runs forecast the same production hour, the most-recent
    run's value is kept (smallest lead time), achieved by iterating runs
    chronologically and overwriting.
    """
    n_runs, T = y_runs.shape
    records: dict[pd.Timestamp, float] = {}
    for r in range(n_runs):
        issue = run_times[r]
        for h in range(T):
            prod_time = issue + pd.Timedelta(hours=h + 1)
            val = float(y_runs[r, h])
            if not np.isnan(val):
                records[prod_time] = val
    series = pd.Series(records, dtype=np.float64).sort_index()
    series = series.resample("h").mean()
    series = series.interpolate(method="linear", limit=6)
    return series.dropna()


class ARIMAModel(BaseSeq2SeqModel):
    """SARIMAX model: energy time series → T-step-ahead forecasts per run.

    Parameters
    ----------
    order:
        ARIMA (p, d, q) order.
    seasonal_order:
        Seasonal (P, D, Q, s) order.  Set ``s=0`` to disable seasonality.
    """

    name = "arima"

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 1, 2),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 24),
    ) -> None:
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self._result = None
        self._last_train_time: pd.Timestamp | None = None
        self._T: int | None = None
        self._forecast_cache: pd.Series | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "ARIMAModel":
        run_times: pd.DatetimeIndex | None = kwargs.get("run_times")
        if run_times is None:
            raise ValueError("ARIMAModel requires 'run_times' keyword argument in fit().")

        self._T = y_train.shape[1]
        series = _build_energy_series(y_train, run_times)
        self._last_train_time = series.index[-1]

        try:
            import statsmodels.api as sm
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for ARIMAModel. "
                "Install with: pip install statsmodels"
            ) from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = sm.tsa.SARIMAX(
                endog=series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._result = mod.fit(disp=False)

        self._forecast_cache = None
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model has not been fitted yet.")

        run_times: pd.DatetimeIndex | None = kwargs.get("run_times")
        if run_times is None:
            raise ValueError("ARIMAModel requires 'run_times' keyword argument in predict().")

        T = self._T
        n_test = len(X_test)
        last_run = run_times[-1]
        last_prod_time = last_run + pd.Timedelta(hours=T)
        total_steps = int(
            (last_prod_time - self._last_train_time) / pd.Timedelta(hours=1)
        )

        if total_steps <= 0:
            raise ValueError(
                f"Test production times ({last_prod_time}) are at or before the "
                f"training end ({self._last_train_time})."
            )

        if (
            self._forecast_cache is None
            or len(self._forecast_cache) < total_steps
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._forecast_cache = self._result.forecast(steps=total_steps)

        forecast = self._forecast_cache
        y_pred = np.full((n_test, T), np.nan, dtype=np.float32)
        for r in range(n_test):
            for h in range(T):
                prod_time = run_times[r] + pd.Timedelta(hours=h + 1)
                step = int(
                    (prod_time - self._last_train_time) / pd.Timedelta(hours=1)
                )
                if 1 <= step <= len(forecast):
                    y_pred[r, h] = float(forecast.iloc[step - 1])

        return y_pred
