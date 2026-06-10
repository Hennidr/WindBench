# WindBench

Benchmarking suite for wind energy production forecasting across multiple farms.
All models operate in a **sequence-to-sequence** paradigm: one full NWP run
(T lead-time steps × F weather features) in, energy at all T lead times out.

## Structure

```
data/               Raw Parquet files (gitignored)
windbench/
  data/             Loader, seq2seq array builder
  models/
    base.py         BaseSeq2SeqModel interface
    arima.py        SARIMAX (no-NWP statistical reference)
    trees.py        Ridge, Random Forest, XGBoost
    deep/           LSTM, Transformer, N-BEATS, N-HiTS, TCN, NLinear, DLinear
  evaluation/
    metrics.py      MSE, RMSE, MAE, MAPE, MASE, Skill Score, CRPS
    seq2seq_benchmark.py  Main benchmark runner
experiments/
  configs/          YAML configs
  results/          Output CSVs (gitignored)
notebooks/
  04_results_visualization.ipynb  All paper figures (benchmark + scaling)
scripts/            CLI entry points
tests/              Smoke tests
```

## Quickstart

```bash
# Install dependencies (no package install needed — run from repo root)
pip install -r requirements.txt

# Run benchmark + data-scaling experiment
python scripts/run_all.py --config experiments/configs/full.yaml

# Run benchmark only
python scripts/run_all.py --config experiments/configs/full.yaml --skip-scaling

# Run scaling only
python scripts/run_all.py --config experiments/configs/full.yaml --skip-benchmark

# Launch notebooks
jupyter notebook notebooks/

# Run tests
pytest tests/
pytest tests/ -m "not slow"   # skip deep learning tests
```

## Data Format

Place one Parquet file per farm in `data/raw/`. Each file must contain:
- `production_time` and `prediction_issue_time` datetime columns
- NWP weather columns (all 13 are used if present): `2t`, `2d`, `10u`, `10v`, `100u`, `100v`, `msl`, `tp`, `sd`, `sf`, `cp`, `lsp`, `sp`
- A target column (default: `energy_total`) with hourly production values
- Multi-turbine files are aggregated automatically (energy summed, weather averaged per timestep)
- Files with multiple `windfarm_id` values are split into separate farms automatically

### Training sample structure

Each sample is one NWP run — a 72-step weather forecast issued at a specific time.
With `n_lags: 24` (default), the input tensor has shape **(T=72, F=37)**:

| Features | Count | Description |
|---|---|---|
| NWP weather columns | 13 | NWP forecast at each lead time |
| `lag_1h` … `lag_24h` | 24 | Observed `energy_total` for the 24h prior to run issue time |

The target is `energy_total` at all 72 lead times, shape **(T=72,)**.

The lag features are constant across all T steps (same context regardless of lead time).
They give every model — trees and NNs alike — the same recent-production context,
mirroring what an operational forecaster observes before issuing a forecast.

## Model Taxonomy

All models operate on the same unified input: one NWP run tensor **(T, F)** where
F = 8 NWP features + `n_lags` production-lag features. Trees and the linear model
flatten this to T×F scalar features; NNs process it as a sequence.

| Category   | Model           | Notes |
|------------|-----------------|-------|
| Statistical | `arima`        | Reconstructed energy time series; no NWP input |
| Linear      | `ridge`        | Ridge regression on flattened (T×F); linear NWP baseline |
| Tree        | `random_forest`| Flattened (T×F); one RF per lead time via `MultiOutputRegressor` |
| Tree        | `xgboost`      | Same; one XGB per lead time via `MultiOutputRegressor` |
| Deep NN     | `lstm`         | LSTM over (T, F) sequence |
| Deep NN     | `transformer`  | Transformer encoder with causal mask |
| Deep NN     | `nbeats`       | Generic basis expansion blocks |
| Deep NN     | `nhits`        | Multi-rate hierarchical interpolation |
| Deep NN     | `tcn`          | Dilated causal convolutions |
| Deep NN     | `nlinear`      | Linear with last-step normalization |
| Deep NN     | `dlinear`      | Trend-seasonal decomposition + linear |

Ridge regression establishes the linear NWP skill ceiling — the baseline that all
non-linear models must meaningfully exceed to justify their added complexity.
ARIMA serves as a no-NWP reference only and is excluded from line/scatter plots.

## Persistence Baseline & Skill Score

The **skill score** is computed as:

```
skill_score = 1 − RMSE_model / RMSE_persistence
```

The persistence baseline predicts the **last known production value** before each
NWP run was issued (`lag_1h`), the same constant for all 72 lead times. This is the
operationally honest baseline: it represents what a dispatcher would know at
decision time. The baseline degrades naturally with horizon — `lag_1h` is 1 hour
stale at h=1 and 25 hours stale at h=24 — so skill scores are harder to achieve
at short horizons.

When `n_lags=0`, the baseline falls back to the training-set mean per horizon.

## Output Files

| File | Contents |
|---|---|
| `experiments/results/seq2seq.csv` | One row per (farm, model, horizon) for all 72 lead times; includes probabilistic metrics (`picp_90`, `mpiw_90`, `crps`) |
| `experiments/results/seq2seq_full_horizons.csv` | RMSE at every horizon for line-plot figures |
| `experiments/results/seq2seq_cumulative.csv` | Cumulative 5-hour window MAE/skill for h=1..68 |
| `experiments/results/scaling.csv` | Scaling experiment: per (farm, fraction, model, horizon); columns: `mae`, `mse` |

## Metrics

| Metric         | Description                                                    |
|----------------|----------------------------------------------------------------|
| `rmse`         | Root Mean Squared Error                                        |
| `mae`          | Mean Absolute Error                                            |
| `mape`         | Mean Absolute Percentage Error                                 |
| `mase`         | Mean Absolute Scaled Error (scaled by within-run lag-1 naive) |
| `skill_score`  | 1 − RMSE_model / RMSE_persistence (higher is better)          |
| `crps`         | Continuous Ranked Probability Score (when quantiles provided)  |

All 72 horizons are stored in `seq2seq.csv`. The notebook's summary views (bar charts,
heatmap, LaTeX table) default to h = 1, 6, 24; change `summary_horizons` in the
`load-data` cell to display any subset without re-running the benchmark.

## Reproducibility

All runs are seeded via `seq2seq.seed` in the YAML config (default `42`).
This seeds Python `random`, NumPy, and PyTorch (CPU + CUDA) before each model fit.
Tree models additionally fix `random_state=42` at construction time.

## Adding a New Model

1. Create a class that inherits from `BaseSeq2SeqModel`.
2. Implement `fit(X_train, y_train, **kwargs)` and `predict(X_test, **kwargs) → (n, T)`.
3. Set the `name` class attribute to a unique string.
4. Register it in `windbench/evaluation/seq2seq_benchmark.py` → `_get_model()`.
5. Add an entry to the desired YAML config under `models:`.
