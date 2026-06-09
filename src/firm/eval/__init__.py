"""Evaluation: metrics, reports, and visualisations.

``BacktestReport`` is lazily imported to avoid a circular dependency
with ``firm.portfolio.attribution``.
"""

from firm.eval.metrics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    compute_all_metrics,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    turnover,
)


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
    "hit_rate",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "total_return",
    "turnover",
    "BacktestReport",
]
