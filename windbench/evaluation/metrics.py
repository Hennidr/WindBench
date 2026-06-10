"""Forecasting evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN entries from both arrays."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    y_true, y_pred = _clean(np.asarray(y_true, float), np.asarray(y_pred, float))
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = _clean(np.asarray(y_true, float), np.asarray(y_pred, float))
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Mean Absolute Percentage Error (%).

    *eps* guards against division by very small values.
    """
    y_true, y_pred = _clean(np.asarray(y_true, float), np.asarray(y_pred, float))
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> float:
    """Mean Absolute Scaled Error.

    Scales the MAE by the in-sample lag-1 naive forecast error, making it
    interpretable across farms with different energy scales.
    MASE < 1 means the model beats the naive lag-1 forecast.

    Parameters
    ----------
    y_true  : ground-truth test values (1-D, NaN allowed)
    y_pred  : model predictions (1-D, NaN allowed)
    y_train : training targets — either the raw 2-D seq2seq array
              ``(n_runs, T)`` or a 1-D series.  When 2-D, lag-1 differences
              are computed *within each run* (axis=1) to avoid spurious jumps
              at run boundaries.
    """
    arr = np.asarray(y_train, float)
    if arr.ndim == 2:
        # Within-run consecutive differences: (n_runs, T-1)
        diffs = np.diff(arr, axis=1).ravel()
    else:
        diffs = np.diff(arr.ravel())
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return float("nan")
    naive_mae = float(np.mean(np.abs(diffs)))
    if naive_mae == 0:
        return float("nan")
    return mae(y_true, y_pred) / naive_mae


def skill_score(y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray) -> float:
    """Skill score relative to a baseline (higher is better, 1 = perfect).

    ``SS = 1 - RMSE_model / RMSE_baseline``
    """
    r_model = rmse(y_true, y_pred)
    r_base = rmse(y_true, y_baseline)
    if r_base == 0:
        return 0.0
    return float(1.0 - r_model / r_base)


def pinball_loss(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> float:
    """Mean pinball (quantile) loss averaged across all quantile levels."""
    y = np.asarray(y_true, float)
    q_preds = np.asarray(q_preds, float)
    losses = []
    for i, tau in enumerate(quantiles):
        err = y - q_preds[:, i]
        losses.append(np.mean(np.where(err >= 0, tau * err, (tau - 1) * err)))
    return float(np.mean(losses))


def crps_quantile(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> float:
    """CRPS approximated via the pinball loss identity: CRPS = 2 * mean_q(pinball_q)."""
    return 2.0 * pinball_loss(y_true, q_preds, quantiles)


def picp(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    """Prediction Interval Coverage Probability."""
    y = np.asarray(y_true, float)
    return float(np.mean((y >= np.asarray(q_lo)) & (y <= np.asarray(q_hi))))


def mpiw(q_lo: np.ndarray, q_hi: np.ndarray, y_range: float) -> float:
    """Mean Prediction Interval Width, normalised by the target range."""
    width = np.mean(np.asarray(q_hi, float) - np.asarray(q_lo, float))
    return float(width / y_range) if y_range > 0 else float(width)


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_baseline: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    q_preds: np.ndarray | None = None,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> dict[str, float]:
    """Compute all metrics for a single model.

    Parameters
    ----------
    y_true:
        Ground truth values.
    y_pred:
        Point forecast predictions.
    y_baseline:
        Baseline predictions for skill score (e.g. persistence).
    y_train:
        Training target values (flattened) for MASE denominator.
    q_preds:
        Quantile predictions, shape ``(n, len(quantiles))``.
    quantiles:
        Probability levels matching columns of *q_preds*.

    Returns
    -------
    dict with keys: ``mse``, ``rmse``, ``mae``, ``mape``, and optionally
    ``mase``, ``skill_score``, ``crps``.
    """
    y_true = np.asarray(y_true, float)
    result: dict[str, float] = {
        "mse":  mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
    if y_train is not None:
        result["mase"] = mase(y_true, y_pred, y_train)
    if y_baseline is not None:
        result["skill_score"] = skill_score(y_true, y_pred, y_baseline)
    if q_preds is not None:
        q_arr = np.asarray(q_preds, float)
        valid = ~np.isnan(y_true) & ~np.any(np.isnan(q_arr), axis=1)
        result["crps"] = crps_quantile(y_true[valid], q_arr[valid], quantiles)
    return result
