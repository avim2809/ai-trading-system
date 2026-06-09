"""Backtrader Strategy subclass that delegates to the agent pipeline.

This is the glue between backtrader's ``next()`` callback and the firm's
orchestrator.  It translates bar data into PitView calls and applies
fills from the execution agent.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

import backtrader as bt

from firm.data.pit_store import PointInTimeDataStore

if TYPE_CHECKING:
    from firm.agents.orchestrator import Orchestrator
    from firm.portfolio.attribution import PerformanceAttribution
    from firm.portfolio.state import PortfolioState
    from firm.strategies.base import PitView

log = logging.getLogger(__name__)


class PitViewAdapter:
    """Adapts :class:`PointInTimeDataStore` to the :class:`PitView` protocol.

    Constructed once per rebalance bar with the current ``asof`` datetime.
    All data queries are automatically filtered to ``<= asof``.
    """

    def __init__(
        self,
        pit_store: PointInTimeDataStore,
        asof: datetime,
        universe: list[str],
    ) -> None:
        self._pit_store = pit_store
        self._asof = asof
        self._universe = list(universe)

    @property
    def asof(self) -> datetime:
        return self._asof

    @property
    def universe(self) -> list[str]:
        return list(self._universe)

    def prices(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_prices(syms, self._asof, lookback_days)

    def fundamentals(
        self,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_fundamentals(syms, self._asof)

    def sentiment(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_sentiment(syms, self._asof, lookback_days)


class FirmStrategy(bt.Strategy):
    """Bridge between Backtrader and the multi-agent orchestrator.

    On each rebalance bar, builds a point-in-time context and calls
    ``orchestrator.step()``, then translates the resulting orders into
    Backtrader buy/sell calls.
    """

    params = (
        ("orchestrator", None),  # Orchestrator instance
        ("pit_store", None),  # PointInTimeDataStore
        ("portfolio_state", None),  # PortfolioState
        ("rebalance_frequency", "weekly"),  # 'daily', 'weekly', 'monthly'
        ("universe", None),  # list of symbols
        ("attribution", None),  # PerformanceAttribution instance (optional)
    )

    def __init__(self):
        self._last_rebalance: datetime | None = None
        self._data_map: dict[str, bt.AbstractDataBase] = {}
        for d in self.datas:
            self._data_map[d._name] = d

    def next(self):
        current_dt: datetime = self.datas[0].datetime.datetime(0)

        if not self._should_rebalance(current_dt):
            return

        pit_view = PitViewAdapter(
            self.p.pit_store, current_dt, self.p.universe
        )

        prices: dict[str, float] = {}
        for sym in self.p.universe:
            data = self._data_map.get(sym)
            if data is not None and len(data) > 0:
                prices[sym] = data.close[0]

        context = {
            "pit_view": pit_view,
            "portfolio": self.p.portfolio_state,
            "prices": prices,
        }

        try:
            orders, blackboard = self.p.orchestrator.step(context)
        except Exception:
            log.error("Orchestrator step failed at %s", current_dt, exc_info=True)
            self._last_rebalance = current_dt
            return

        for order_dict in orders:
            symbol = order_dict.get("symbol")
            if symbol is None or symbol not in self._data_map:
                continue
            data = self._data_map[symbol]
            qty = order_dict.get("shares", order_dict.get("quantity", 0))
            if qty > 0:
                self.buy(data=data, size=qty)
            elif qty < 0:
                self.sell(data=data, size=abs(qty))

        if self.p.portfolio_state is not None and orders:
            self.p.portfolio_state.update(orders, prices)

        if self.p.attribution is not None and orders:
            self.p.attribution.record_trades(orders, prices)

        self._last_rebalance = current_dt

    def _should_rebalance(self, dt: datetime) -> bool:
        """Check whether we should rebalance on this bar."""
        if self._last_rebalance is None:
            return True
        freq = self.p.rebalance_frequency
        if freq == "daily":
            return True
        elif freq == "weekly":
            return (dt - self._last_rebalance).days >= 5
        elif freq == "monthly":
            return dt.month != self._last_rebalance.month
        return True

    def notify_order(self, order):
        if order.status in [order.Completed]:
            log.debug(
                "Order filled: %s %s @ %.2f, size=%.0f",
                "BUY" if order.isbuy() else "SELL",
                order.data._name,
                order.executed.price,
                order.executed.size,
            )

    def notify_trade(self, trade):
        if trade.isclosed:
            log.debug(
                "Trade closed: %s, PnL=%.2f",
                trade.data._name,
                trade.pnl,
            )
