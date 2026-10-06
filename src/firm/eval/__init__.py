"""Evaluation: metrics, reports, and visualisations.

``BacktestReport`` is lazily imported to avoid a circular dependency
with ``firm.portfolio.attribution``.
"""

from firm.eval.classification import (
    binary_classification_report,
    brier_score,
    pr_auc,
    reliability_diagram_bins,
)
from firm.eval.metrics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    compute_all_metrics,
    compute_trade_metrics,
    expectancy,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    trade_win_rate,
    turnover,
)
from firm.eval.overfitting import (
    cscv_pbo,
    deflated_sharpe,
    probabilistic_sharpe,
    verdict,
    walk_forward_overfitting,
)
from firm.eval.robustness import MonteCarloAnalyzer
from firm.eval.tca import aggregate_tca, compute_tca_record


def __getattr__(name: str):
    if name == "BacktestReport":
        from firm.eval.reports import BacktestReport
        return BacktestReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BacktestReport",
    "MonteCarloAnalyzer",
    "aggregate_tca",
    "annualized_volatility",
    "binary_classification_report",
    "brier_score",
    "cagr",
    "calmar_ratio",
    "compute_all_metrics",
    "compute_tca_record",
    "compute_trade_metrics",
    "cscv_pbo",
    "deflated_sharpe",
    "expectancy",
    "hit_rate",
    "max_drawdown",
    "pr_auc",
    "probabilistic_sharpe",
    "profit_factor",
    "reliability_diagram_bins",
    "sharpe_ratio",
    "sortino_ratio",
    "total_return",
    "trade_win_rate",
    "turnover",
    "verdict",
    "walk_forward_overfitting",
]
