WindBench — notebooks/figures/
==============================

All thesis-facing tables and figures produced by notebooks/04_results_visualization.ipynb.
Tables are saved as both CSV (for inspection) and LaTeX (booktabs, ready to \input{}).
Figures are 300 dpi PDFs.

Source data:
  experiments/results/seq2seq.csv   — main benchmark (multi-farm)
  experiments/results/scaling.csv   — data-scaling experiment (Kelmarsh)

Units:
  Target `energy_total` is in kWh per hourly observation.
  RMSE, MAE, CRPS are therefore in kWh.
  Figure 3 axis and Figure 4 annotations are shown in MWh for readability.
  Skill score, PICP, and the efficiency ratio are dimensionless.
  MPIW@90 in tab2_probabilistic is normalised by the training target range
  (also dimensionless).
  Cross-farm averages of absolute kWh metrics (Appendix A) are dominated by
  the largest farms — use skill score for cross-farm comparison.


Part 1 — Main Benchmark
-----------------------
tab1_headline_skill.{csv,tex}        Headline skill score (vs persistence) per model,
                                     mean ± std across farms, by horizon bin (Short
                                     1-12h, Medium 13-36h, Long 37-72h) and over the
                                     full 1-72h window.
fig1_skill_vs_horizon.pdf            Skill score curves vs lead time, averaged across
                                     farms; ±1σ band for the top model.
fig2_per_farm_heatmap.pdf            Heatmap of skill per (model, farm), averaged over
                                     all 72 lead times. Columns sorted by farm
                                     difficulty.
fig2b_per_farm_skill_vs_horizon.pdf  Small-multiples grid: skill vs horizon, one panel
                                     per farm; shaded bands mark Short/Medium/Long bins.
tab1b_per_farm_best_model.{csv,tex}  Best model per (farm, horizon bin) — for spotting
                                     farms where the cross-farm winner doesn't hold.
tab2_probabilistic.{csv,tex}         PICP@90, MPIW@90, CRPS per model — averaged across
                                     all farms and 72 horizons.


Part 2 — Data-Scaling Experiment (Kelmarsh only)
------------------------------------------------
tab4_sample_efficiency.{csv,tex}     MAE at 10%/50%/100% of training data plus the
                                     efficiency ratio MAE@10% / MAE@100%.
fig3_scaling_mae.pdf                 MAE vs training-set size, one panel per horizon
                                     bin.
fig4_sample_efficiency.pdf           Normalized learning curves — MAE(f)/MAE(100%).
                                     Flat = data-efficient; steep = data-hungry.
                                     Annotations show absolute MAE@100% in kWh/h.


Appendix
--------
appA_benchmark_full.{csv,tex}        Per-model RMSE, MAE, skill score by horizon bin
                                     and over the full 1-72h window (all farms).
appB_per_farm_skill.{csv,tex}        Per-farm skill matrix (model × farm), averaged
                                     over 72 horizons.
appC_scaling_full.{csv,tex}          Full scaling table — MAE per (model, bin, fraction).
appD_per_farm_skill.pdf              Merged PDF: one full-page panel per farm, skill
                                     vs lead time. Sourced from per_farm_skill/.
per_farm_skill/                      Individual per-farm skill-vs-horizon PDFs (one
                                     file per farm).


Naming convention
-----------------
tab*    — table (CSV + TeX)
fig*    — figure (PDF)
app*    — appendix item
Numbers reflect order in the thesis, not creation order.
