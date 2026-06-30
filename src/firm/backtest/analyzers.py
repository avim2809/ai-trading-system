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


class TradeLogAnalyzer(bt.Analyzer):
    """Record each closed trade as a structured row for persistence.

    Captures the opening size/price when a trade is first opened (Backtrader
    zeroes ``trade.size`` once a trade closes, so the entry size must be
    grabbed on the opening event) and emits one row per closed trade. The
    rows feed ``trades.parquet`` per run, which the structured-query layer
    (:class:`firm.rag.structured.RunStore`) reads via SQL — numeric trade
    facts are queried, never re-derived by an LLM.
    """

    def start(self):
        self._open: dict[int, dict] = {}
        self._trades: list[dict] = []

    def notify_trade(self, trade):
        if trade.justopened:
            # Grab entry size/price now; both are reset after the trade closes.
            self._open[trade.ref] = {"size": trade.size, "price": trade.price}
            return

        if not trade.isclosed:
            return

        opened = self._open.pop(trade.ref, {})
        size = float(opened.get("size", 0.0))
        entry_price = float(opened.get("price", trade.price))
        strategy = (
            trade.data.info.get("strategy", "_default")
            if hasattr(trade.data, "info") else "_default"
        )
        gross = float(trade.pnl)
        net = float(trade.pnlcomm)
        cost_basis = abs(entry_price * size)
        # Net P&L over entry notional; gross is used to back out the exit price.
        return_pct = (net / cost_basis) if cost_basis > 0 else 0.0
        exit_price = entry_price + (gross / size) if size else entry_price

        self._trades.append({
            "entry_dt": bt.num2date(trade.dtopen).isoformat(),
            "exit_dt": bt.num2date(trade.dtclose).isoformat(),
            "symbol": trade.data._name,
            "strategy": strategy,
            "size": size,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "pnl": gross,
            "pnl_net": net,
            "commission": gross - net,
            "return_pct": return_pct,
            "bars_held": int(trade.barlen),
        })

    def get_analysis(self) -> dict:
        return {"trades": list(self._trades)}


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


class BenchmarkAnalyzer(bt.Analyzer):
    """Track an equal-weight buy-and-hold benchmark of the traded universe.

    Records a normalized index value per bar: the mean across all data feeds
    of ``close[t] / close[first_valid]``. This is a self-contained benchmark
    (no external SPY feed required) representing "what equal-weight passive
    holding of the same universe would have returned" over the same dates,
    so the strategy can be measured for alpha/beta/information ratio against
    it via :mod:`firm.eval.metrics`.
    """

    def start(self):
        self._dates: list = []
        self._values: list[float] = []
        self._bases: dict[str, float] = {}  # first valid close per symbol

    def next(self):
        dt = self.strategy.datas[0].datetime.datetime(0)
        ratios: list[float] = []
        for data in self.strategy.datas:
            price = data.close[0]
            if price is None or price <= 0:
                continue
            base = self._bases.get(data._name)
            if base is None:
                self._bases[data._name] = price
                base = price
            ratios.append(price / base)
        if ratios:
            self._dates.append(dt)
            self._values.append(sum(ratios) / len(ratios))

    def get_analysis(self) -> dict:
        return {
            "dates": list(self._dates),
            "values": list(self._values),
        }
