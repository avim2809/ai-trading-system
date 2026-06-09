"""Custom backtrader analyzers for turnover, attribution, and daily returns.

These supplement the built-in ``SharpeRatio``, ``DrawDown``, ``Returns``,
and ``TradeAnalyzer`` analyzers wired up in :mod:`firm.backtest.engine`.
"""

from __future__ import annotations

import backtrader as bt


class TurnoverAnalyzer(bt.Analyzer):
    """Track portfolio turnover on each bar.

    Turnover is the average absolute change in portfolio weights between
    consecutive bars where a position change occurred.
    """

    def start(self):
        self._prev_weights: dict[str, float] = {}
        self._turnovers: list[float] = []

    def next(self):
        portfolio_value = self.strategy.broker.getvalue()
        if portfolio_value <= 0:
            return

        weights: dict[str, float] = {}
        for data in self.strategy.datas:
            pos = self.strategy.getposition(data).size
            if pos != 0:
                weights[data._name] = (pos * data.close[0]) / portfolio_value

        if self._prev_weights:
            all_syms = set(self._prev_weights) | set(weights)
            bar_turnover = sum(
                abs(weights.get(s, 0.0) - self._prev_weights.get(s, 0.0))
                for s in all_syms
            )
            if bar_turnover > 1e-9:
                self._turnovers.append(bar_turnover)

        self._prev_weights = weights

    def get_analysis(self) -> dict:
        avg = sum(self._turnovers) / len(self._turnovers) if self._turnovers else 0.0
        return {
            "avg_turnover": avg,
            "total_turnover": sum(self._turnovers),
            "rebalance_count": len(self._turnovers),
        }


class StrategyAttributionAnalyzer(bt.Analyzer):
    """Track P&L attribution by strategy tag.

    Orders/trades placed via the bridge carry a ``strategy`` info key.
    This analyzer accumulates per-strategy realised P&L.
    """

    def start(self):
        self._strategy_pnl: dict[str, float] = {}

    def notify_trade(self, trade):
        if trade.isclosed:
            strategy = trade.data.info.get("strategy", "_default") if hasattr(trade.data, "info") else "_default"
            pnl = trade.pnl
            self._strategy_pnl[strategy] = self._strategy_pnl.get(strategy, 0.0) + pnl

    def get_analysis(self) -> dict:
        return dict(self._strategy_pnl)


class DetailedReturnsAnalyzer(bt.Analyzer):
    """Record daily portfolio values for post-hoc return computation.

    The output is a dict with ``dates`` and ``values`` lists, suitable
    for computing daily simple returns and feeding into
    :func:`firm.eval.metrics.compute_all_metrics`.
    """

    def start(self):
        self._daily_values: list[float] = []
        self._dates: list = []

    def next(self):
        dt = self.strategy.datas[0].datetime.datetime(0)
        self._dates.append(dt)
        self._daily_values.append(self.strategy.broker.getvalue())

    def get_analysis(self) -> dict:
        return {
            "dates": list(self._dates),
            "values": list(self._daily_values),
        }
