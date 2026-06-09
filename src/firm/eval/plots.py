"""Matplotlib / seaborn visualisations for equity curves, drawdowns, etc.

Every public function returns a :class:`matplotlib.figure.Figure` so callers
can further customise or save to disk.  The ``save_all_plots`` helper
generates the standard set and writes PNGs to a given directory.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from firm.portfolio.attribution import PerformanceAttribution

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_equity_curve(
    returns: pd.Series,
    title: str = "Equity Curve",
) -> Figure:
    """Cumulative wealth index (starts at 1.0)."""
    cumulative = (1 + returns).cumprod()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative.index, cumulative.values, linewidth=1.4)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    fig.tight_layout()
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "Drawdown",
) -> Figure:
    """Underwater chart showing peak-to-trough drawdown over time."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.45, color="crimson")
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    fig.tight_layout()
    return fig


def plot_monthly_returns(returns: pd.Series) -> Figure:
    """Heatmap of monthly returns by year."""
    if returns.empty:
        fig, ax = plt.subplots()
        ax.set_title("Monthly Returns (no data)")
        return fig

    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    table = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values}
    )
    pivot = table.pivot_table(index="year", columns="month", values="ret", aggfunc="sum")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][: len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 0.6)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        center=0,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Monthly Returns (%)")
    fig.tight_layout()
    return fig


def plot_strategy_attribution(attribution: PerformanceAttribution) -> Figure:
    """Bar chart of cumulative strategy-level returns."""
    metrics = attribution.get_strategy_metrics()
    if not metrics:
        fig, ax = plt.subplots()
        ax.set_title("Strategy Attribution (no data)")
        return fig

    strategies = list(metrics.keys())
    total_rets = [metrics[s].get("total_return", 0.0) for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["forestgreen" if r >= 0 else "crimson" for r in total_rets]
    ax.barh(strategies, total_rets, color=colours)
    ax.set_xlabel("Total Return")
    ax.set_title("Strategy Attribution")
    fig.tight_layout()
    return fig


def plot_exposure(
    weights_history: list[tuple[datetime, dict[str, float]]],
) -> Figure:
    """Gross and net exposure over time."""
    if not weights_history:
        fig, ax = plt.subplots()
        ax.set_title("Exposure (no data)")
        return fig

    dates = [d for d, _ in weights_history]
    gross = [sum(abs(w) for w in ws.values()) for _, ws in weights_history]
    net = [sum(ws.values()) for _, ws in weights_history]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, gross, label="Gross", linewidth=1.2)
    ax.plot(dates, net, label="Net", linewidth=1.2, linestyle="--")
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5)
    ax.set_title("Portfolio Exposure")
    ax.set_ylabel("Exposure")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
) -> Figure:
    """Rolling annualised Sharpe ratio."""
    if len(returns) < window:
        fig, ax = plt.subplots()
        ax.set_title(f"Rolling Sharpe ({window}d) — insufficient data")
        return fig

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std(ddof=1)
    rolling = (rolling_mean / rolling_std) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling.index, rolling.values, linewidth=1.2)
    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    ax.set_title(f"Rolling Sharpe Ratio ({window}-day)")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Date")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Batch save
# ---------------------------------------------------------------------------

def save_all_plots(
    returns: pd.Series,
    attribution: PerformanceAttribution,
    output_dir: str,
) -> None:
    """Generate and save all standard plots to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plots: dict[str, Figure] = {
        "equity_curve": plot_equity_curve(returns),
        "drawdown": plot_drawdown(returns),
        "monthly_returns": plot_monthly_returns(returns),
        "strategy_attribution": plot_strategy_attribution(attribution),
        "rolling_sharpe": plot_rolling_sharpe(returns),
    }

    for name, fig in plots.items():
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
