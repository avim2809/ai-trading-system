"""Tests for the trade-level metrics in firm.eval.metrics."""

from __future__ import annotations

import math

from firm.eval.metrics import (
    compute_trade_metrics,
    expectancy,
    profit_factor,
    trade_win_rate,
)

TRADES = [
    {"pnl_net": 100.0},
    {"pnl_net": -50.0},
    {"pnl_net": 200.0},
    {"pnl_net": -50.0},
]


class TestTradeMetrics:
    def test_profit_factor(self):
        # gross profit 300 / gross loss 100 = 3.0
        assert profit_factor(TRADES) == 3.0

    def test_profit_factor_no_losses_is_inf(self):
        assert profit_factor([{"pnl_net": 10.0}, {"pnl_net": 5.0}]) == math.inf

    def test_profit_factor_empty_is_zero(self):
        assert profit_factor([]) == 0.0

    def test_expectancy(self):
        # (100 - 50 + 200 - 50) / 4 = 50
        assert expectancy(TRADES) == 50.0

    def test_win_rate(self):
        assert trade_win_rate(TRADES) == 0.5

    def test_falls_back_to_gross_pnl_key(self):
        trades = [{"pnl": 100.0}, {"pnl": -25.0}]
        assert profit_factor(trades) == 4.0

    def test_rollup(self):
        m = compute_trade_metrics(TRADES)
        assert m["num_trades"] == 4.0
        assert m["profit_factor"] == 3.0
        assert m["expectancy"] == 50.0
        assert m["trade_win_rate"] == 0.5
        assert m["avg_win"] == 150.0
        assert m["avg_loss"] == -50.0
        assert m["gross_profit"] == 300.0
        assert m["gross_loss"] == -100.0
