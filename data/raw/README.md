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

## Full dataset on Zenodo

All 12 farm Parquet files are archived together at:

<https://doi.org/10.5281/zenodo.20628054>

This includes the two files that exceed GitHub's 100 MB per-file limit and
are therefore not shipped in this repository:

- **`kelmarsh.parquet`** (~136 MB) — Kelmarsh wind farm, UK
- **`penmanshiel.parquet`** (~305 MB) — Penmanshiel wind farm, UK

## What's in the repo

The 10 New Zealand farm files (`nz_farm4.parquet` … `nz_farm13.parquet`,
~200 MB total) are included directly so the benchmark is runnable out of
the box on a subset.

For the full 12-farm benchmark, download the Zenodo archive and extract
the two missing files into this directory:

```
data/raw/
  kelmarsh.parquet      ← from Zenodo
  penmanshiel.parquet   ← from Zenodo
  nz_farm4.parquet      ← shipped
  ...
  nz_farm13.parquet     ← shipped
```

The benchmark and notebooks will then discover all 12 farms automatically.
