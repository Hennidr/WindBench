"""Reusable plot functions for wind benchmarking results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_predictions(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    title: str = "Predictions vs Ground Truth",
    figsize: tuple[int, int] = (14, 4),
) -> plt.Figure:
    """Line plot comparing ground truth and multiple model predictions.

    Parameters
    ----------
    y_true:
        Ground truth time series with a ``DatetimeIndex``.
    predictions:
        Mapping ``{model_name: prediction_array}``.
    title:
        Plot title.
    figsize:
        Figure size in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_true.index, y_true.values, label="Ground truth", color="black", linewidth=1.5, zorder=10)
    for name, pred in predictions.items():
        ax.plot(y_true.index, pred, label=name, alpha=0.75, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (kW)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str = "Error Distribution",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """KDE plot of prediction errors (y_pred - y_true) for each model.

    Parameters
    ----------
    y_true:
        Ground truth array.
    predictions:
        Mapping ``{model_name: prediction_array}``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, pred in predictions.items():
        errors = np.asarray(pred, float) - np.asarray(y_true, float)
        errors = errors[~np.isnan(errors)]
        sns.kdeplot(errors, ax=ax, label=name, fill=True, alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Error (kW)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metric_heatmap(
    results: pd.DataFrame,
    metric: str = "rmse",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    fmt: str = ".2f",
    horizon: int | None = None,
) -> plt.Figure:
    """Heatmap of a metric across farms (rows) and models (columns).

    Parameters
    ----------
    results:
        Tidy results DataFrame from :func:`windbench.evaluation.run_benchmark`.
    metric:
        Column name of the metric to visualise.
    horizon:
        If provided, filter to this forecast horizon before plotting.
        If None and a ``horizon`` column exists, values are averaged across horizons.
    """
    df = results.copy()
    if horizon is not None and "horizon" in df.columns:
        df = df[df["horizon"] == horizon]
        h_label = f" — h={horizon}"
    elif "horizon" in df.columns:
        df = df.groupby(["farm", "model"], as_index=False)[metric].mean()
        h_label = " (avg across horizons)"
    else:
        h_label = ""
    pivot = df.pivot(index="farm", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlGn_r", ax=ax, linewidths=0.5)
    ax.set_title(title or f"{metric.upper()} by farm and model{h_label}")
    ax.set_xlabel("Model")
    ax.set_ylabel("Farm")
    fig.tight_layout()
    return fig


def plot_metric_by_horizon(
    results: pd.DataFrame,
    metric: str = "rmse",
    exclude_models: list[str] | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Line plot of a metric vs forecast horizon, one line per model (averaged across farms).

    Parameters
    ----------
    results:
        Tidy results DataFrame from :func:`windbench.evaluation.run_benchmark`.
        Must contain a ``horizon`` column.
    metric:
        Column name of the metric to plot on the y-axis.
    exclude_models:
        Model names to omit (e.g. ``["persistence"]`` to focus on non-trivial models).
    """
    if "horizon" not in results.columns:
        raise ValueError("Results DataFrame has no 'horizon' column.")

    df = results.copy()
    if exclude_models:
        df = df[~df["model"].isin(exclude_models)]

    summary = df.groupby(["model", "horizon"])[metric].mean().reset_index()
    horizons = sorted(summary["horizon"].unique())

    fig, ax = plt.subplots(figsize=figsize)
    for model, grp in summary.groupby("model"):
        grp = grp.sort_values("horizon")
        ax.plot(grp["horizon"], grp[metric], marker="o", label=model, linewidth=1.8)

    ax.set_xticks(horizons)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_title(f"{metric.upper()} vs Forecast Horizon (avg across farms)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric.upper())
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_horizon_heatmap(
    results: pd.DataFrame,
    metric: str = "skill_score",
    figsize: tuple[int, int] = (8, 6),
    fmt: str = ".3f",
) -> plt.Figure:
    """Heatmap of a metric with models as rows and horizons as columns (averaged across farms).

    Parameters
    ----------
    results:
        Tidy results DataFrame from :func:`windbench.evaluation.run_benchmark`.
    metric:
        Column name of the metric to visualise.
    """
    if "horizon" not in results.columns:
        raise ValueError("Results DataFrame has no 'horizon' column.")

    summary = results.groupby(["model", "horizon"])[metric].mean().reset_index()
    pivot = summary.pivot(index="model", columns="horizon", values=metric)
    pivot.columns = [f"h={h}" for h in pivot.columns]

    cmap = "RdYlGn" if metric == "skill_score" else "RdYlGn_r"
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, linewidths=0.5)
    ax.set_title(f"{metric.upper()} by model and horizon (avg across farms)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Model")
    fig.tight_layout()
    return fig


def plot_skill_scores(
    results: pd.DataFrame,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Grouped bar chart of skill scores per model, averaged across farms.

    Parameters
    ----------
    results:
        Tidy results DataFrame from :func:`windbench.evaluation.run_benchmark`.
    """
    if "skill_score" not in results.columns:
        raise ValueError("'skill_score' column not found in results.")

    summary = results.groupby("model")["skill_score"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(summary["model"], summary["mean"], yerr=summary["std"], capsize=4, alpha=0.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Skill Score vs Persistence Baseline (higher is better)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Skill Score")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig
