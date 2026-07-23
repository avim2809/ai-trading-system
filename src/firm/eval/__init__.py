"""Evaluation: metrics, reports, and visualisations.

``BacktestReport`` is lazily imported to avoid a circular dependency
with ``firm.portfolio.attribution``.
"""

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


def __getattr__(name: str):
    if name == "BacktestReport":
        from firm.eval.reports import BacktestReport
        return BacktestReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "annualized_volatility",
    "cagr",
    "calmar_ratio",
    "compute_all_metrics",
    "compute_trade_metrics",
    "expectancy",
    "hit_rate",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "total_return",
    "trade_win_rate",
    "turnover",
    "cscv_pbo",
    "deflated_sharpe",
    "probabilistic_sharpe",
    "verdict",
    "walk_forward_overfitting",
    "MonteCarloAnalyzer",
    "BacktestReport",
]
