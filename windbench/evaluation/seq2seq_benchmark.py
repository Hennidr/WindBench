"""Unified seq2seq benchmark: train one model per farm, evaluate at multiple horizons.

Model taxonomy
--------------
* **NN** (LSTM, Transformer, TCN, N-BEATS, N-HiTS, NLinear, DLinear): receive
  a 3-D NWP run tensor ``(T, F)`` and produce predictions for all T lead times.
* **Tree** (Random Forest, XGBoost): the same run tensor as NNs, flattened to
  T×F scalar features; multi-output regression over all T lead times.
* **Statistical** (ARIMA): fitted on the reconstructed hourly energy time
  series; forecasts T steps ahead from the training end for each test run.
  Requires ``run_times`` to be forwarded as a keyword argument.

All models implement :class:`~windbench.models.base.BaseSeq2SeqModel` and are
evaluated at the same ``eval_horizons`` (e.g. [1, 6, 24]).

Probabilistic evaluation
------------------------
Split conformal prediction is applied post-hoc using the validation residuals.
No retraining is required — the calibration is model-agnostic.

For each horizon h and coverage level α:
  1. Compute absolute residuals on the val set: |y_val_h - ŷ_val_h|
  2. Conformal quantile: q̂ = quantile(|residuals|, α)
  3. Test intervals: [ŷ - q̂,  ŷ + q̂]

Reported probabilistic metrics: PICP (empirical coverage), MPIW (normalised
width), CRPS (via pinball loss over empirical quantiles from val residuals).
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from windbench.data.seq2seq import (
    DEFAULT_LEAD_TIMES,
    build_seq2seq_arrays,
    seq2seq_temporal_split,
)
from windbench.evaluation.metrics import crps_quantile, evaluate, picp, mpiw
from windbench.models.base import BaseSeq2SeqModel

# Quantile levels used for CRPS estimation from val residuals
_QUANTILE_LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
# Nominal coverage for conformal intervals
_CONFORMAL_COVERAGE = 0.90


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _conformal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    val_residuals: np.ndarray,
    y_range: float,
    coverage: float = _CONFORMAL_COVERAGE,
) -> dict[str, float]:
    """Compute conformal interval metrics for one horizon.

    Parameters
    ----------
    y_true:       Test ground truth (n_test,)
    y_pred:       Point predictions (n_test,)
    val_residuals: Signed residuals y_val - ŷ_val for this horizon
    y_range:      Training target range for MPIW normalisation
    coverage:     Nominal coverage level (default 0.90)
    """
    res = val_residuals[~np.isnan(val_residuals)]
    if len(res) < 5:
        return {}

    abs_res = np.abs(res)
    q_conf = float(np.quantile(abs_res, coverage))
    q_lo = y_pred - q_conf
    q_hi = y_pred + q_conf

    valid = ~np.isnan(y_true)

    # CRPS: build quantile predictions from empirical distribution of val residuals
    q_preds = np.column_stack([
        y_pred + np.quantile(res, q) for q in _QUANTILE_LEVELS
    ])  # (n_test, len(_QUANTILE_LEVELS))

    return {
        f"picp_{int(coverage*100)}": picp(y_true[valid], q_lo[valid], q_hi[valid]),
        f"mpiw_{int(coverage*100)}": mpiw(q_lo[valid], q_hi[valid], y_range),
        "crps": crps_quantile(y_true[valid], q_preds[valid], _QUANTILE_LEVELS),
    }


def _get_model(name: str, **kwargs) -> BaseSeq2SeqModel:
    """Instantiate a model by name (lazy imports)."""
    from windbench.models.deep.lstm import LSTMModel
    from windbench.models.deep.transformer import TransformerModel
    from windbench.models.deep.nbeats import NBEATSModel
    from windbench.models.deep.nhits import NHiTSModel
    from windbench.models.deep.tcn import TCNModel
    from windbench.models.deep.nlinear import NLinearModel
    from windbench.models.deep.dlinear import DLinearModel
    from windbench.models.arima import ARIMAModel
    from windbench.models.trees import RandomForestModel, XGBoostModel, RidgeModel

    registry: dict[str, type[BaseSeq2SeqModel]] = {
        "arima":         ARIMAModel,
        "ridge":         RidgeModel,
        "random_forest": RandomForestModel,
        "xgboost":       XGBoostModel,
        "lstm":          LSTMModel,
        "transformer":   TransformerModel,
        "nbeats":        NBEATSModel,
        "nhits":         NHiTSModel,
        "tcn":           TCNModel,
        "nlinear":       NLinearModel,
        "dlinear":       DLinearModel,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(registry)}"
        )
    return registry[name](**kwargs)


# Keep old name as alias so existing test passes
def _get_seq2seq_model(name: str, **kwargs) -> BaseSeq2SeqModel:
    return _get_model(name, **kwargs)


def run_seq2seq_benchmark(
    farms: dict[str, pd.DataFrame],
    model_configs: list[dict],
    eval_horizons: list[int] | None = None,
    lead_times: list[int] | None = None,
    target_col: str = "energy_total",
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    min_lead_coverage: float = 0.8,
    fill_method: str = "interpolate",
    n_lags: int = 0,
    seed: int = 42,
    results_path: str | Path | None = None,
    full_horizon_path: str | Path | None = None,
    cumulative_path: str | Path | None = None,
    cumulative_window: int = 5,
) -> pd.DataFrame:
    """Train all models on full NWP runs and evaluate at specific horizons.

    Parameters
    ----------
    farms:
        Dict of {farm_name: DataFrame} from ``load_all_farms()``.
    model_configs:
        List of dicts, each with a ``name`` key and model hyperparameters.
    eval_horizons:
        Lead times at which to evaluate (must be in ``lead_times``).
        Defaults to all lead_times.
    lead_times:
        Lead-time hours included in each NWP run sequence.
        Defaults to range(1, 73).
    target_col:
        Energy target column name.
    val_frac, test_frac:
        Fractions for temporal split of NWP runs.
    min_lead_coverage:
        Minimum fraction of lead times populated for a run to be kept.
    fill_method:
        NaN fill strategy: ``"interpolate"``, ``"ffill"``, or ``"zero"``.
    results_path:
        If given, save results CSV here.
    """
    if lead_times is None:
        lead_times = DEFAULT_LEAD_TIMES
    if eval_horizons is None:
        eval_horizons = lead_times

    missing = [h for h in eval_horizons if h not in lead_times]
    if missing:
        raise ValueError(
            f"eval_horizons {missing} not present in lead_times. "
            f"Available range: {min(lead_times)}..{max(lead_times)}"
        )

    weather_cols = ["2d", "2t", "10u", "10v", "100u", "100v", "msl", "tp", "sd", "sf", "cp", "lsp", "sp"]
    records: list[dict] = []
    full_records: list[dict] = []
    cum_records: list[dict] = []

    for farm_name, df in farms.items():
        print(f"\n{'='*60}\nFarm: {farm_name}\n{'='*60}")

        X, y, run_times = build_seq2seq_arrays(
            df,
            weather_cols=[c for c in weather_cols if c in df.columns],
            lead_times=lead_times,
            min_lead_coverage=min_lead_coverage,
            fill_method=fill_method,
            target_col=target_col,
            n_lags=n_lags,
        )
        print(f"  Runs: {len(X)}  T={X.shape[1]}  F={X.shape[2]}")

        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            t_train, t_val, t_test,
        ) = seq2seq_temporal_split(X, y, run_times, val_frac=val_frac, test_frac=test_frac)
        print(f"  Split: train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

        y_train_2d = y_train
        y_range = float(np.nanmax(y_train) - np.nanmin(y_train))

        n_weather_features = X.shape[2] - n_lags if n_lags > 0 else X.shape[2]
        if n_lags > 0:
            lag1_test = X_test[:, 0, n_weather_features].astype(float)
            persistence_baselines: dict[int, np.ndarray] = {h: lag1_test for h in lead_times}
        else:
            persistence_baselines = {}
            for h in lead_times:
                h_idx = lead_times.index(h)
                mean_train = float(np.nanmean(y_train[:, h_idx]))
                persistence_baselines[h] = np.full(len(X_test), mean_train)

        for cfg in model_configs:
            cfg = dict(cfg)
            model_name = cfg.pop("name")
            print(f"\n  [{model_name}] fitting...", end=" ", flush=True)

            try:
                _seed_everything(seed)
                model: BaseSeq2SeqModel = _get_model(model_name, **cfg)

                t0 = time.perf_counter()
                model.fit(
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    run_times=t_train,
                )
                fit_time = time.perf_counter() - t0

                # ── Conformal calibration on val set ──────────────────────
                val_residuals: np.ndarray | None = None
                try:
                    y_val_pred = model.predict(X_val, run_times=t_val)  # (n_val, T)
                    val_residuals = y_val - y_val_pred                  # signed residuals
                except Exception:
                    pass

                t0 = time.perf_counter()
                h_preds = model.predict_at_horizons(
                    X_test, eval_horizons, lead_times,
                    run_times=t_test,
                )
                predict_time = time.perf_counter() - t0

                for h in eval_horizons:
                    h_idx = lead_times.index(h)
                    y_true = y_test[:, h_idx]
                    y_pred = h_preds[h]
                    y_base = persistence_baselines[h]

                    metrics = evaluate(
                        y_true, y_pred,
                        y_baseline=y_base,
                        y_train=y_train_2d,
                    )

                    # Probabilistic metrics via conformal calibration
                    if val_residuals is not None:
                        prob = _conformal_metrics(
                            y_true, y_pred,
                            val_residuals[:, h_idx],
                            y_range,
                        )
                        metrics.update(prob)

                    print(
                        f"  h={h}: RMSE={metrics['rmse']:.1f}"
                        f"  skill={metrics['skill_score']:.3f}"
                        f"  CRPS={metrics.get('crps', float('nan')):.1f}"
                        f"  cov={metrics.get(f'picp_{int(_CONFORMAL_COVERAGE*100)}', float('nan')):.2f}",
                        end="  ", flush=True,
                    )

                    records.append({
                        "farm":    farm_name,
                        "horizon": h,
                        "model":   model_name,
                        **metrics,
                        "fit_time_s":     round(fit_time, 3),
                        "predict_time_s": round(predict_time, 4),
                    })

                print()

                # Full-horizon + cumulative evaluation
                if full_horizon_path is not None or cumulative_path is not None:
                    try:
                        y_pred_full = model.predict(X_test, run_times=t_test)

                        if full_horizon_path is not None:
                            for h in lead_times:
                                h_idx = lead_times.index(h)
                                y_true = y_test[:, h_idx]
                                y_pred = y_pred_full[:, h_idx]
                                y_base = persistence_baselines[h]
                                full_records.append({
                                    "farm":    farm_name,
                                    "horizon": h,
                                    "model":   model_name,
                                    "rmse":    float(np.sqrt(np.nanmean((y_true - y_pred) ** 2))),
                                    "rmse_persistence": float(np.sqrt(np.nanmean((y_true - y_base) ** 2))),
                                })

                        if cumulative_path is not None:
                            W = cumulative_window
                            if n_lags > 0:
                                cum_base = lag1_test * W
                            else:
                                cum_base = None

                            for h_start_idx in range(len(lead_times) - W + 1):
                                idx = slice(h_start_idx, h_start_idx + W)
                                h_start = lead_times[h_start_idx]
                                y_true_cum = np.nansum(y_test[:, idx], axis=1)
                                y_pred_cum = np.nansum(y_pred_full[:, idx], axis=1)

                                if cum_base is not None:
                                    y_base_cum = cum_base
                                else:
                                    mean_w = float(np.nanmean(y_train[:, idx]))
                                    y_base_cum = np.full(len(X_test), mean_w * W)

                                mae_m = float(np.nanmean(np.abs(y_true_cum - y_pred_cum)))
                                mae_b = float(np.nanmean(np.abs(y_true_cum - y_base_cum)))
                                cum_records.append({
                                    "farm":             farm_name,
                                    "horizon_start":    h_start,
                                    "model":            model_name,
                                    "mae":              mae_m,
                                    "mae_persistence":  mae_b,
                                    "skill_score":      float(1.0 - mae_m / mae_b) if mae_b > 0 else 0.0,
                                })
                    except Exception:
                        pass

            except Exception as exc:
                print(f"ERROR -- {exc}")
                for h in eval_horizons:
                    records.append({
                        "farm":    farm_name,
                        "horizon": h,
                        "model":   model_name,
                        "error":   str(exc),
                    })

    results = pd.DataFrame(records)

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")

    if full_horizon_path is not None and full_records:
        full_df = pd.DataFrame(full_records)
        full_horizon_path = Path(full_horizon_path)
        full_horizon_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(full_horizon_path, index=False)
        print(f"Full-horizon results saved to {full_horizon_path}")

    if cumulative_path is not None and cum_records:
        cum_df = pd.DataFrame(cum_records)
        cumulative_path = Path(cumulative_path)
        cumulative_path.parent.mkdir(parents=True, exist_ok=True)
        cum_df.to_csv(cumulative_path, index=False)
        print(f"Cumulative results saved to {cumulative_path}")

    return results
