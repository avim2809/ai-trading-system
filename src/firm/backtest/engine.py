"""Core backtest engine.

Wraps backtrader's Cerebro and feeds it with point-in-time data,
orchestrating the agent pipeline each bar.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

import backtrader as bt

from firm.backtest.analyzers import (
    BenchmarkAnalyzer,
    DetailedReturnsAnalyzer,
    StrategyAttributionAnalyzer,
    TradeLogAnalyzer,
    TurnoverAnalyzer,
)
from firm.backtest.commissions import PercentageCommission
from firm.backtest.datafeeds import load_feeds
from firm.backtest.firm_strategy import FirmStrategy
from firm.portfolio.state import PortfolioState

if TYPE_CHECKING:
    from firm.agents.orchestrator import Orchestrator
    from firm.data.pit_store import PointInTimeDataStore
    from firm.eval.reports import BacktestReport

log = logging.getLogger(__name__)


class BacktestEngine:
    """Configures and runs a Backtrader backtest with the full agent pipeline.

    Typical usage::

        engine = BacktestEngine(config)
        engine.setup(prices_df, pit_store, orchestrator, universe)
        engine.run()
        report = engine.generate_report()
        print(report.to_text())
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.cerebro = bt.Cerebro()
        self.results: list | None = None
        self._portfolio_state: PortfolioState | None = None
        self._attribution: Any = None

    def setup(
        self,
        prices_df: pd.DataFrame,
        pit_store: PointInTimeDataStore,
        orchestrator: Orchestrator,
        universe: list[str],
        memory=None,
        llm_config: dict | None = None,
    ) -> None:
        """Wire everything together: feeds, broker, strategy, analyzers."""
        initial_capital = self.config.get("initial_capital", 10_000_000)
        self.cerebro.broker.setcash(initial_capital)

        commission_pct = self.config.get("commission_pct", 0.001)
        commission = PercentageCommission(commission=commission_pct)
        self.cerebro.broker.addcommissioninfo(commission)

        # Slippage is applied at the broker via backtrader's native API rather
        # than folded into the commission scheme. Orders are submitted in
        # next() and fill at the following bar's open, so slip_open is the
        # relevant knob.
        slippage_pct = self.config.get("slippage_pct", 0.0005)
        if slippage_pct > 0:
            self.cerebro.broker.set_slippage_perc(
                perc=slippage_pct, slip_open=True, slip_match=True, slip_out=False
            )

        feeds = load_feeds(prices_df, universe)
        for symbol, feed in feeds.items():
            self.cerebro.adddata(feed, name=symbol)

        from firm.portfolio.attribution import PerformanceAttribution

        portfolio_state = PortfolioState(initial_capital=initial_capital)
        attribution = PerformanceAttribution()

        self.cerebro.addstrategy(
            FirmStrategy,
            orchestrator=orchestrator,
            pit_store=pit_store,
            portfolio_state=portfolio_state,
            rebalance_frequency=self.config.get("rebalance_frequency", "weekly"),
            universe=universe,
            attribution=attribution,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            memory=memory,
            llm_config=llm_config,
        )

        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        self.cerebro.addanalyzer(DetailedReturnsAnalyzer, _name="detailed_returns")
        self.cerebro.addanalyzer(TurnoverAnalyzer, _name="turnover")
        self.cerebro.addanalyzer(StrategyAttributionAnalyzer, _name="strategy_attr")
        self.cerebro.addanalyzer(TradeLogAnalyzer, _name="trade_log")
        self.cerebro.addanalyzer(BenchmarkAnalyzer, _name="benchmark")

        self._portfolio_state = portfolio_state
        self._attribution = attribution
        log.info(
            "BacktestEngine set up: %d symbols, capital=%s, rebalance=%s",
            len(feeds),
            f"{initial_capital:,.0f}",
            self.config.get("rebalance_frequency", "weekly"),
        )

    def run(self) -> list:
        """Execute the backtest. Returns the raw Backtrader results."""
        log.info("Starting backtest run")
        self.results = self.cerebro.run()
        log.info("Backtest complete")
        return self.results

    def get_results(self) -> dict[str, Any]:
        """Extract structured results from the completed backtest."""
        if not self.results:
            raise RuntimeError("Must call run() before get_results()")

        strat = self.results[0]
        analysis: dict[str, Any] = {}

        analysis["final_value"] = self.cerebro.broker.getvalue()

        try:
            sharpe = strat.analyzers.sharpe.get_analysis()
            analysis["sharpe"] = sharpe.get("sharperatio")
        except Exception:
            analysis["sharpe"] = None

        try:
            dd = strat.analyzers.drawdown.get_analysis()
            analysis["max_drawdown_pct"] = dd.get("max", {}).get("drawdown")
        except Exception:
            analysis["max_drawdown_pct"] = None

        try:
            analysis["returns"] = strat.analyzers.returns.get_analysis()
        except Exception:
            analysis["returns"] = {}

        try:
            analysis["trades"] = strat.analyzers.trades.get_analysis()
        except Exception:
            analysis["trades"] = {}

        try:
            analysis["turnover"] = strat.analyzers.turnover.get_analysis()
        except Exception:
            analysis["turnover"] = {}

        try:
            analysis["strategy_attribution"] = strat.analyzers.strategy_attr.get_analysis()
        except Exception:
            analysis["strategy_attribution"] = {}

        try:
            analysis["trade_log"] = strat.analyzers.trade_log.get_analysis().get("trades", [])
        except Exception:
            analysis["trade_log"] = []

        try:
            analysis["detailed_returns"] = strat.analyzers.detailed_returns.get_analysis()
        except Exception:
            analysis["detailed_returns"] = {}

        try:
            analysis["benchmark"] = strat.analyzers.benchmark.get_analysis()
        except Exception:
            analysis["benchmark"] = {}

        return analysis

    def generate_report(self) -> BacktestReport:
        """Build a :class:`BacktestReport` from the completed run."""
        from firm.eval.reports import BacktestReport as _Report
        from firm.portfolio.attribution import PerformanceAttribution

        results = self.get_results()
        detailed = results.get("detailed_returns", {})
        values = detailed.get("values", [])
        dates = detailed.get("dates", [])

        if len(values) >= 2:
            daily_returns = pd.Series(values, index=pd.DatetimeIndex(dates))
            daily_returns = daily_returns.pct_change().dropna()
        else:
            daily_returns = pd.Series(dtype=float)

        bench = results.get("benchmark", {})
        bench_values = bench.get("values", [])
        bench_dates = bench.get("dates", [])
        if len(bench_values) >= 2:
            benchmark_returns = pd.Series(
                bench_values, index=pd.DatetimeIndex(bench_dates)
            ).pct_change().dropna()
        else:
            benchmark_returns = pd.Series(dtype=float)

        snapshots = []
        if self._portfolio_state is not None:
            snapshots = self._portfolio_state.history

        return _Report(
            returns=daily_returns,
            attribution=self._attribution or PerformanceAttribution(),
            snapshots=snapshots,
            benchmark_returns=benchmark_returns,
            trades=results.get("trade_log", []),
        )
