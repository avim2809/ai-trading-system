"""Performance and risk metrics: Sharpe, Sortino, max drawdown, etc.

All functions accept a pandas Series of simple daily returns unless stated
otherwise.  Edge cases (empty series, zero variance, zero drawdown) return
``0.0`` rather than raising.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

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
# Trade-level metrics
# ---------------------------------------------------------------------------

def _trade_pnls(trades: list[dict], key: str = "pnl_net") -> list[float]:
    """Extract numeric per-trade P&L, preferring net then gross."""
    pnls: list[float] = []
    for t in trades:
        val = t.get(key)
        if val is None:
            val = t.get("pnl")
        if isinstance(val, (int, float)) and not pd.isna(val):
            pnls.append(float(val))
    return pnls


def profit_factor(trades: list[dict]) -> float:
    """Gross profit / gross loss across closed trades.

    ``inf`` when there are wins but no losses; ``0.0`` when empty or all flat.
    """
    pnls = _trade_pnls(trades)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def expectancy(trades: list[dict]) -> float:
    """Average net P&L per closed trade (expected value of a trade)."""
    pnls = _trade_pnls(trades)
    if not pnls:
        return 0.0
    return float(sum(pnls) / len(pnls))


def trade_win_rate(trades: list[dict]) -> float:
    """Fraction of closed trades with positive net P&L."""
    pnls = _trade_pnls(trades)
    if not pnls:
        return 0.0
    return float(sum(1 for p in pnls if p > 0) / len(pnls))


def compute_trade_metrics(trades: list[dict]) -> dict[str, float]:
    """Roll-up of trade-level metrics from a per-trade log.

    Complements the return-based metrics (which operate on the daily equity
    curve) with statistics only visible at the individual-trade level.
    """
    pnls = _trade_pnls(trades)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    result = {
        "num_trades": float(len(pnls)),
        "trade_win_rate": trade_win_rate(trades),
        "profit_factor": profit_factor(trades),
        "expectancy": expectancy(trades),
        "avg_win": float(sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "gross_profit": float(sum(wins)),
        "gross_loss": float(sum(losses)),
    }
    log.debug(
        "compute_trade_metrics: %d trades, win_rate=%.3f profit_factor=%.3f "
        "expectancy=%.4f", result["num_trades"], result["trade_win_rate"],
        result["profit_factor"], result["expectancy"],
    )
    return result


# ---------------------------------------------------------------------------
# Benchmark-relative metrics
# ---------------------------------------------------------------------------

def _align(returns: pd.Series, benchmark: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-join two return series on their (date) index, dropping NaNs."""
    df = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if df.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    return df.iloc[:, 0], df.iloc[:, 1]


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Sensitivity of strategy returns to the benchmark: cov / var(benchmark)."""
    r, b = _align(returns, benchmark)
    if len(r) < 2:
        return 0.0
    var_b = b.var(ddof=1)
    if var_b < 1e-14:
        return 0.0
    cov = r.cov(b)
    return float(cov / var_b)


def alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Jensen's alpha (CAPM intercept), arithmetic annualization."""
    r, b = _align(returns, benchmark)
    if len(r) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    bta = beta(r, b)
    daily_alpha = (r.mean() - daily_rf) - bta * (b.mean() - daily_rf)
    return float(daily_alpha * periods_per_year)


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized active return divided by tracking error."""
    r, b = _align(returns, benchmark)
    if len(r) < 2:
        return 0.0
    active = r - b
    te = active.std(ddof=1)
    if te < 1e-14:
        return 0.0
    return float(active.mean() / te * np.sqrt(periods_per_year))


def excess_return(returns: pd.Series, benchmark: pd.Series) -> float:
    """Total strategy return minus total benchmark return over the period."""
    r, b = _align(returns, benchmark)
    if r.empty:
        return 0.0
    return float(total_return(r) - total_return(b))


def compute_benchmark_metrics(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Benchmark-relative metrics: alpha, beta, information ratio, excess return."""
    _, b_aligned = _align(returns, benchmark)
    return {
        "benchmark_total_return": total_return(b_aligned),
        "alpha": alpha(returns, benchmark, risk_free_rate),
        "beta": beta(returns, benchmark),
        "information_ratio": information_ratio(returns, benchmark),
        "excess_return": excess_return(returns, benchmark),
    }


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
