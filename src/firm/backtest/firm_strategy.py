"""Backtrader Strategy subclass that delegates to the agent pipeline.

This is the glue between backtrader's ``next()`` callback and the firm's
orchestrator.  It translates bar data into PitView calls and applies
fills from the execution agent.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

import backtrader as bt

from firm.data.pit_store import PointInTimeDataStore

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
        lookback_reports: int = 4,
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_fundamentals(syms, self._asof, lookback_reports)

    def sentiment(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_sentiment(syms, self._asof, lookback_days)

    def macro(self, series_id: str, lookback_days: int = 365) -> pd.Series:
        return self._pit_store.get_macro(series_id, self._asof, lookback_days)


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
        ("commission_pct", 0.001),  # per-trade commission rate
        ("slippage_pct", 0.0005),  # per-trade slippage rate
        ("memory", None),   # TradingMemoryLog instance (optional)
        ("llm_config", None),  # LLM config dict for reflection calls (optional)
        # First date real trading/evaluation may begin (ISO string, optional).
        # Data before this may still be fed in (e.g. a warmup buffer so
        # long-lookback strategies have real history to work with), but
        # nothing here may place an order, update rebalance state, or record
        # attribution before this date — see run.py's execute_backtest for
        # why the warmup buffer exists.
        ("start_date", None),
    )

    def __init__(self):
        self._last_rebalance: datetime | None = None
        self._data_map: dict[str, bt.AbstractDataBase] = {}
        for d in self.datas:
            self._data_map[d._name] = d
        # Track previous bar's NAV and date for deferred reflection.
        self._prev_rebalance_date: str | None = None
        self._prev_rebalance_nav: float | None = None
        self._eval_start = (
            datetime.fromisoformat(self.p.start_date).date() if self.p.start_date else None
        )
        self._llm_service = None

    def next(self):
        current_dt: datetime = self.datas[0].datetime.datetime(0)

        # Warmup bars (before the real evaluation window) exist purely so
        # long-lookback strategies (e.g. regime_hmm's 252-day HMM training
        # window) have real history via pit_view.prices() — no trading,
        # rebalance-state, or attribution tracking may happen yet.
        if self._eval_start is not None and current_dt.date() < self._eval_start:
            return

        prices: dict[str, float] = {}
        for sym in self.p.universe:
            data = self._data_map.get(sym)
            if data is not None and len(data) > 0:
                prices[sym] = data.close[0]

        # Mark-to-market per-strategy holdings every bar (not just rebalance
        # bars) so the attribution return series has the same daily
        # granularity as the overall portfolio's.
        if self.p.attribution is not None:
            self.p.attribution.update_daily(current_dt, prices, self.broker.getvalue())

        if not self._should_rebalance(current_dt):
            return

        pit_view = PitViewAdapter(
            self.p.pit_store, current_dt, self.p.universe
        )

        # Reflect on the previous decision now that its P&L is known.
        self._maybe_reflect(current_dt)

        context = {
            "pit_view": pit_view,
            "portfolio": self.p.portfolio_state,
            "prices": prices,
            "memory": self.p.memory,
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
            # ExecutionAgent already tags each order with its dominant
            # contributing strategy (order_dict["strategy"]), but that tag
            # was being dropped here — every trade in trades.parquet ended
            # up attributed to the "_default" fallback regardless of which
            # of the 12 strategies actually drove it. Data feeds have no
            # built-in per-order channel, so stash it on the feed's `.info`
            # (read by TradeLogAnalyzer/StrategyAttributionAnalyzer at the
            # moment a trade opens) — safe because a rebalance places at
            # most one order per symbol, and any resulting trade opens
            # before this same symbol's next rebalance could overwrite it.
            data.info = {"strategy": order_dict.get("strategy", "composite")}
            qty = order_dict.get("shares", order_dict.get("quantity", 0))
            if qty > 0:
                self.buy(data=data, size=qty)
            elif qty < 0:
                self.sell(data=data, size=abs(qty))

        if self.p.portfolio_state is not None and orders:
            # Mirror the broker's transaction costs in the secondary book so
            # attribution/NAV snapshots stay consistent with headline metrics.
            cost_rate = self.p.commission_pct + self.p.slippage_pct
            cost = sum(o.get("notional", 0.0) for o in orders) * cost_rate
            self.p.portfolio_state.update(orders, prices, cost=cost)

        if self.p.attribution is not None and orders:
            self.p.attribution.record_trades(orders, prices)

        # Phase A: store this decision for deferred reflection.
        if self.p.memory is not None:
            proposal = getattr(blackboard, "proposal", None)
            if proposal is not None:
                date_str = current_dt.strftime("%Y-%m-%d")
                nav = self.broker.getvalue()
                self.p.memory.store_decision(
                    date=date_str,
                    proposal_weights=dict(proposal.targets),
                    notes=getattr(proposal, "notes", "") or "",
                    nav_at_decision=nav,
                )
                self._prev_rebalance_date = date_str
                self._prev_rebalance_nav = nav

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

    def _maybe_reflect(self, current_dt: datetime) -> None:
        """Trigger LLM reflection on the previous rebalance decision.

        Called at the start of each rebalance bar.  Uses the change in broker
        NAV between the two bars as the realized return; benchmark is omitted
        in the backtest path to keep reflection cheap (SPY prices are available
        in the price data if the user wants to compute it themselves).
        """
        if self.p.memory is None:
            return
        if self._prev_rebalance_date is None or self._prev_rebalance_nav is None:
            return
        prev_nav = self._prev_rebalance_nav
        if prev_nav <= 0:
            return
        current_nav = self.broker.getvalue()
        raw_return = (current_nav / prev_nav) - 1.0
        llm = self._get_llm_service()
        if llm is None:
            return
        try:
            self.p.memory.reflect(
                date=self._prev_rebalance_date,
                raw_return=raw_return,
                benchmark_return=0.0,
                llm_service=llm,
            )
        except Exception:
            log.debug("Memory reflection failed for %s", self._prev_rebalance_date, exc_info=True)
        finally:
            self._prev_rebalance_date = None
            self._prev_rebalance_nav = None

    def _get_llm_service(self):
        if self._llm_service is not None:
            return self._llm_service
        llm_config = self.p.llm_config or {}
        if not llm_config:
            return None
        try:
            from firm.llm.provider import LLMService
            self._llm_service = LLMService(llm_config)
        except Exception:
            log.debug("LLM service unavailable — reflection disabled")
        return self._llm_service

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
