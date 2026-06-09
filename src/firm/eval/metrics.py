"""Performance and risk metrics: Sharpe, Sortino, max drawdown, etc.

All functions accept a pandas Series of simple daily returns unless stated
otherwise.  Edge cases (empty series, zero variance, zero drawdown) return
``0.0`` rather than raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def total_return(returns: pd.Series) -> float:
    """Cumulative total return: (1+r).prod() - 1."""
    if returns.empty:
        return 0.0
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate assuming 252 trading days per year."""
    if returns.empty:
        return 0.0
    tr = total_return(returns)
    n = len(returns)
    if n == 0 or tr <= -1.0:
        return 0.0
    return float((1 + tr) ** (TRADING_DAYS_PER_YEAR / n) - 1)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized standard deviation of returns."""
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std < 1e-14:
        return 0.0
    return float(std * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio.

    ``(mean(excess) / std(returns)) * sqrt(periods_per_year)``
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    std = returns.std(ddof=1)
    if std < 1e-14:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio (downside deviation in denominator)."""
    if len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    downside_std = float(np.sqrt((downside**2).mean()))
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline (returned as a positive number)."""
    if returns.empty:
        return 0.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    mdd = drawdowns.min()
    return float(abs(mdd))


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR / |max drawdown|.  Returns 0 when drawdown is zero."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(cagr(returns) / mdd)


def turnover(weights_history: list[dict[str, float]]) -> float:
    """Average absolute weight change per rebalance.

    Parameters
    ----------
    weights_history:
        List of weight dictionaries (one per rebalance period).
    """
    if len(weights_history) < 2:
        return 0.0
    total = 0.0
    for prev, curr in zip(weights_history[:-1], weights_history[1:]):
        all_syms = set(prev) | set(curr)
        total += sum(abs(curr.get(s, 0.0) - prev.get(s, 0.0)) for s in all_syms)
    return total / (len(weights_history) - 1)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with positive returns."""
    if returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


# ---------------------------------------------------------------------------
# Convenience roll-up
# ---------------------------------------------------------------------------

def compute_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute all standard metrics and return as a flat dictionary."""
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
        "hit_rate": hit_rate(returns),
    }
