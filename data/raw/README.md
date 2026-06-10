# `data/raw/`

Per-farm Parquet files used by the WindBench loader. Schema:

| column | description |
|---|---|
| `windfarm_id` | farm identifier (int) |
| `production_time` | UTC timestamp of the production hour |
| `prediction_issue_time` | UTC timestamp the NWP run was issued |
| `turbine_id` | per-turbine row identifier (if turbine-resolved) |
| `longitude`, `latitude` | site coordinates |
| `energy_total` | hourly energy production in **kWh** (target) |
| `2t`, `2d`, `10u`, `10v`, `100u`, `100v`, `msl`, `tp`, `sd`, `sf`, `cp`, `lsp`, `sp` | ECMWF NWP weather features |

The loader (`windbench/data/loader.py`) sums `energy_total` across turbines
and averages weather columns to produce one row per
`(production_time, prediction_issue_time)` pair.

## Shipped with the repo

The 10 New Zealand farm files (`nz_farm4.parquet` … `nz_farm13.parquet`,
~200 MB total) are included directly.

## Not shipped — fetch manually

Two files exceed GitHub's 100 MB per-file limit and must be downloaded
separately into this directory:

- **`kelmarsh.parquet`** (~136 MB) — Kelmarsh wind farm, UK.
  Source: Plumley, C. (2022). *Kelmarsh wind farm data*. Zenodo.
  <https://doi.org/10.5281/zenodo.5841833>
- **`penmanshiel.parquet`** (~305 MB) — Penmanshiel wind farm, UK.
  Source: Plumley, C. (2022). *Penmanshiel wind farm data*. Zenodo.
  <https://doi.org/10.5281/zenodo.5946808>

After downloading, place both files in this directory:

```
data/raw/
  kelmarsh.parquet
  penmanshiel.parquet
  nz_farm4.parquet
  ...
```

The benchmark and notebooks will then discover all 12 farms automatically.
