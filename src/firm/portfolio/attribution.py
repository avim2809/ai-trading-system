"""Performance attribution – decomposes returns by strategy, sector, factor.

Tracks per-strategy P&L from tagged fills and daily mark-to-market, then
exposes metric roll-ups via :func:`~firm.eval.metrics.compute_all_metrics`.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pandas as pd

from firm.eval.metrics import compute_all_metrics


class PerformanceAttribution:
    """Per-strategy and per-factor P&L attribution."""

    def __init__(self) -> None:
        self._trade_log: list[dict] = []
        self._strategy_returns: dict[str, list[float]] = defaultdict(list)
        self._strategy_dates: dict[str, list[datetime]] = defaultdict(list)
        self._strategy_holdings: dict[str, dict[str, float]] = defaultdict(dict)
        self._prev_prices: dict[str, float] = {}
        self._factor_exposures: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_trades(
        self,
        fills: list[dict],
        prices: dict[str, float],
    ) -> None:
        """Record executed trades with strategy tags for attribution.

        Each fill dict: ``{"symbol", "shares", "price", "strategy"}``.
        """
        for fill in fills:
            self._trade_log.append(dict(fill))
            strategy = fill.get("strategy", "_default")
            symbol = fill["symbol"]
            shares = fill["shares"]
            cur = self._strategy_holdings[strategy].get(symbol, 0.0)
            self._strategy_holdings[strategy][symbol] = cur + shares

    def update_daily(
        self,
        date: datetime,
        prices: dict[str, float],
        nav: float,
        strategy_holdings: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Record daily return contribution by strategy using mark-to-market.

        Each strategy's daily figure is its position-weighted dollar P&L
        divided by total portfolio NAV — a "contribution to portfolio
        return" series. Dividing by NAV (not left as raw dollar P&L) matters
        because get_strategy_metrics() feeds this straight into
        compute_all_metrics(), which assumes period *percentage* returns
        (e.g. total_return does ``(1+r).prod()``) — raw dollar P&L values
        produced nonsense metrics. Strategies here don't have independently
        allocated capital (all 12 blend into one target-weight decision
        sharing one capital pool), so NAV-normalized contributions are also
        the only framing that's directly comparable and summable across
        strategies.
        """
        strat_hold = strategy_holdings or self._strategy_holdings
        for strategy, sym_shares in strat_hold.items():
            daily_pnl = 0.0
            for sym, shares in sym_shares.items():
                prev = self._prev_prices.get(sym)
                curr = prices.get(sym)
                if prev is not None and curr is not None and prev != 0:
                    daily_pnl += shares * (curr - prev)
            daily_return = daily_pnl / nav if nav > 0 else 0.0
            self._strategy_returns[strategy].append(daily_return)
            self._strategy_dates[strategy].append(date)

        self._prev_prices = dict(prices)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_strategy_returns(self, strategy: str) -> pd.Series:
        """Get daily P&L series for a specific strategy."""
        dates = self._strategy_dates.get(strategy, [])
        values = self._strategy_returns.get(strategy, [])
        if not dates:
            return pd.Series(dtype=float)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name=strategy)

    def get_strategy_metrics(self) -> dict[str, dict[str, float]]:
        """Compute metrics for each strategy using compute_all_metrics.

        Strategies with no return history are skipped.
        """
        result: dict[str, dict[str, float]] = {}
        for strategy in self._strategy_returns:
            series = self.get_strategy_returns(strategy)
            if series.empty:
                continue
            result[strategy] = compute_all_metrics(series)
        return result

    def get_factor_attribution(self) -> pd.DataFrame:
        """Factor-level P&L breakdown.

        Returns an empty DataFrame when no factor exposures have been
        registered.
        """
        if not self._factor_exposures:
            return pd.DataFrame()
        return pd.DataFrame(self._factor_exposures).T

    def set_factor_exposures(
        self,
        strategy: str,
        exposures: dict[str, float],
    ) -> None:
        """Register factor exposures for a strategy."""
        self._factor_exposures[strategy] = exposures

    def summary(self) -> pd.DataFrame:
        """Summary table: strategy × metric."""
        metrics = self.get_strategy_metrics()
        if not metrics:
            return pd.DataFrame()
        return pd.DataFrame(metrics).T

    @property
    def trade_log(self) -> list[dict]:
        return list(self._trade_log)

    @property
    def strategies(self) -> list[str]:
        return list(self._strategy_returns.keys())
