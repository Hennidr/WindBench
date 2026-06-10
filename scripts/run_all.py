"""Run the benchmark and/or the data-scaling experiment from one config.

Usage
-----
    python scripts/run_all.py --config experiments/configs/full.yaml
    python scripts/run_all.py --config experiments/configs/full.yaml --skip-scaling
    python scripts/run_all.py --config experiments/configs/full.yaml --skip-benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from windbench.data.loader import load_all_farms
from windbench.data.seq2seq import (
    DEFAULT_LEAD_TIMES,
    build_seq2seq_arrays,
    seq2seq_temporal_split,
)
from windbench.evaluation.metrics import mae, mse
from windbench.evaluation.seq2seq_benchmark import (
    _get_model,
    _seed_everything,
    run_seq2seq_benchmark,
)


def run_scaling(
    farms: dict[str, pd.DataFrame],
    model_configs: list[dict],
    fractions: list[float] | None = None,
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
) -> pd.DataFrame:
    """Train each model at multiple training-set sizes and evaluate on fixed test set."""
    if fractions is None:
        fractions = [round(f / 10, 1) for f in range(1, 11)]
    if lead_times is None:
        lead_times = DEFAULT_LEAD_TIMES
    if eval_horizons is None:
        eval_horizons = list(lead_times)

    weather_cols = ["2d", "2t", "10u", "10v", "100u", "100v", "msl", "tp", "sd", "sf", "cp", "lsp", "sp"]
    records: list[dict] = []

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
        print(f"  Total runs: {len(X)}  T={X.shape[1]}  F={X.shape[2]}")

        (
            X_train_full, X_val, X_test,
            y_train_full, y_val, y_test,
            t_train_full, t_val, t_test,
        ) = seq2seq_temporal_split(X, y, run_times, val_frac=val_frac, test_frac=test_frac)
        n_full = len(X_train_full)
        print(f"  Full train={n_full}  val={len(X_val)}  test={len(X_test)}")

        # Persistence baseline: lag_1h (last known production before each test run).
        # Falls back to climatology when n_lags=0.
        n_weather_features = X.shape[2] - n_lags if n_lags > 0 else X.shape[2]
        if n_lags > 0:
            lag1_test = X_test[:, 0, n_weather_features].astype(float)
            persistence_baselines: dict[int, np.ndarray] = {h: lag1_test for h in eval_horizons}
        else:
            persistence_baselines = {}
            for h in eval_horizons:
                h_idx = lead_times.index(h)
                mean_train = float(np.nanmean(y_train_full[:, h_idx]))
                persistence_baselines[h] = np.full(len(X_test), mean_train)

        for frac in fractions:
            n_keep = max(1, int(np.ceil(frac * n_full)))
            X_train = X_train_full[:n_keep]
            y_train = y_train_full[:n_keep]
            t_train = t_train_full[:n_keep]

            print(f"\n  fraction={frac:.1f}  n_train={n_keep}")

            for cfg in model_configs:
                cfg = dict(cfg)
                model_name = cfg.pop("name")
                print(f"    [{model_name}]", end=" ", flush=True)

                try:
                    _seed_everything(seed)
                    model = _get_model(model_name, **cfg)
                    t0 = time.perf_counter()
                    model.fit(
                        X_train, y_train,
                        X_val=X_val, y_val=y_val,
                        run_times=t_train,
                    )
                    fit_time = time.perf_counter() - t0

                    y_pred_full = model.predict(X_test, run_times=t_test)

                    for h_idx, h in enumerate(lead_times):
                        y_true = y_test[:, h_idx]
                        y_pred = y_pred_full[:, h_idx]

                        mse_val = mse(y_true, y_pred)
                        mae_val = mae(y_true, y_pred)
                        records.append({
                            "farm":       farm_name,
                            "fraction":   frac,
                            "n_train":    n_keep,
                            "horizon":    h,
                            "model":      model_name,
                            "mse":        round(mse_val, 3),
                            "mae":        round(mae_val, 3),
                            "fit_time_s": round(fit_time, 3),
                        })
                        if h in eval_horizons:
                            print(f"h={h} mae={mae_val:.1f}", end="  ", flush=True)

                except Exception as exc:
                    print(f"ERROR — {exc}")
                    for h in lead_times:
                        records.append({
                            "farm": farm_name, "fraction": frac,
                            "n_train": n_keep, "horizon": h,
                            "model": model_name, "error": str(exc),
                        })
            print()

    results = pd.DataFrame(records)

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_path, index=False)
        print(f"\nScaling results saved to {results_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WindBench benchmark + scaling experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip the main benchmark")
    parser.add_argument("--skip-scaling",   action="store_true", help="Skip the scaling experiment")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    s2s_cfg  = cfg.get("seq2seq", {})
    sc_cfg   = cfg.get("scaling", {})
    out_cfg  = cfg.get("output", {})

    target_col        = data_cfg.get("target_col", "energy_total")
    farms_filter      = data_cfg.get("farms")
    eval_horizons     = s2s_cfg.get("eval_horizons", [1, 6, 24])
    lead_times        = s2s_cfg.get("lead_times")
    val_frac          = float(s2s_cfg.get("val_frac", 0.1))
    test_frac         = float(s2s_cfg.get("test_frac", 0.2))
    min_lead_coverage = float(s2s_cfg.get("min_lead_coverage", 0.8))
    fill_method       = s2s_cfg.get("fill_method", "interpolate")
    n_lags            = int(s2s_cfg.get("n_lags", 0))
    seed              = int(s2s_cfg.get("seed", 42))
    model_configs     = cfg.get("models", [])

    print(f"Loading data from {data_cfg.get('raw_dir', 'data/raw')} ...")
    farms = load_all_farms(data_cfg.get("raw_dir", "data/raw"), target_col=target_col)
    if farms_filter:
        farms = {k: v for k, v in farms.items() if k in farms_filter}
    print(f"Farms: {list(farms)}\n")

    shared = dict(
        farms=farms,
        model_configs=model_configs,
        eval_horizons=eval_horizons,
        lead_times=lead_times,
        target_col=target_col,
        val_frac=val_frac,
        test_frac=test_frac,
        min_lead_coverage=min_lead_coverage,
        fill_method=fill_method,
        n_lags=n_lags,
        seed=seed,
    )

    if not args.skip_benchmark:
        print("=" * 60)
        print("STEP 1 / 2 — Full benchmark")
        print("=" * 60)
        t0 = time.perf_counter()
        run_seq2seq_benchmark(
            **shared,
            results_path=out_cfg.get("benchmark_path", "experiments/results/seq2seq.csv"),
            full_horizon_path=out_cfg.get("full_horizon_path"),
            cumulative_path=out_cfg.get("cumulative_path"),
        )
        print(f"\nBenchmark finished in {time.perf_counter() - t0:.0f}s")

    if not args.skip_scaling:
        print("\n" + "=" * 60)
        print("STEP 2 / 2 — Data-scaling experiment")
        print("=" * 60)
        t0 = time.perf_counter()
        run_scaling(
            **shared,
            fractions=sc_cfg.get("fractions"),
            results_path=out_cfg.get("scaling_path", "experiments/results/scaling.csv"),
        )
        print(f"\nScaling finished in {time.perf_counter() - t0:.0f}s")

    print("\nAll done.")


if __name__ == "__main__":
    main()
