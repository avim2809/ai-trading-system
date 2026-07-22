"""Phase 1a: per-trade log capture and persistence.

Validates that a backtest captures closed trades and that the report writes a
schema-valid Parquet file, including the empty-run case.
"""

from __future__ import annotations

import pandas as pd

from firm.backtest.run import execute_backtest
from firm.eval.reports import TRADE_COLUMNS, BacktestReport


class TestTradePersistence:
    def test_empty_trades_writes_schema_valid_parquet(self, tmp_path):
        report = BacktestReport(
            returns=pd.Series(dtype=float),
            attribution=_empty_attribution(),
            snapshots=[],
            trades=[],
        )
        n = report.save_trades(str(tmp_path / "trades.parquet"))
        assert n == 0
        df = pd.read_parquet(tmp_path / "trades.parquet")
        assert list(df.columns) == TRADE_COLUMNS
        assert len(df) == 0

    def test_trades_round_trip(self, tmp_path):
        rows = [{
            "entry_dt": "2021-01-04T00:00:00", "exit_dt": "2021-01-08T00:00:00",
            "symbol": "AAPL", "strategy": "_default", "size": 100.0,
            "entry_price": 130.0, "exit_price": 135.0, "pnl": 500.0,
            "pnl_net": 498.0, "commission": 2.0, "return_pct": 0.0383,
            "bars_held": 4,
        }]
        report = BacktestReport(pd.Series(dtype=float), _empty_attribution(), [], trades=rows)
        report.save_trades(str(tmp_path / "trades.parquet"))
        df = pd.read_parquet(tmp_path / "trades.parquet")
        assert df.iloc[0]["symbol"] == "AAPL"
        assert df.iloc[0]["size"] == 100.0


class TestBacktestCapturesTrades:
    """End-to-end: a real (short) synthetic backtest produces trade rows."""

    def test_synthetic_backtest_records_trades(self, tmp_path):
        cfg = {
            "data_source": "synthetic",
            "start_date": "2021-01-01",
            "end_date": "2021-04-30",
            "seed": 7,
            "rebalance_frequency": "weekly",
        }
        report = execute_backtest(cfg)
        assert report.trades, "expected at least one closed trade"

        n = report.save_trades(str(tmp_path / "trades.parquet"))
        df = pd.read_parquet(tmp_path / "trades.parquet")
        assert n == len(df) == len(report.trades)
        assert list(df.columns) == TRADE_COLUMNS

        # Invariants on the captured rows.
        assert (df["bars_held"] >= 0).all()
        # commission == gross - net (within float tolerance).
        assert ((df["pnl"] - df["pnl_net"] - df["commission"]).abs() < 1e-6).all()
        # Shorts are representable (signed size); at least some non-zero sizes.
        assert (df["size"] != 0).any()


class TestBacktestStrategyAttribution:
    """Regression coverage for a real gap found while analyzing walk-forward
    results: every trade in trades.parquet was tagged strategy="_default"
    regardless of which of the 12 strategies actually drove it, and
    report.json had no "strategies" attribution table at all.

    Root cause was two separate bugs: (1) ExecutionAgent already tags each
    order with its real dominant strategy, but FirmStrategy.next() dropped
    that field when placing the backtrader order instead of threading it
    through to the data feed the trade analyzers read it from, and (2)
    PerformanceAttribution.update_daily() — which turns tagged holdings
    into an actual per-strategy return series — was never called anywhere,
    so even a correct tag would never have produced return metrics.
    """

    def test_trades_are_tagged_with_real_strategy_names_not_default(self):
        cfg = {
            "data_source": "synthetic",
            "start_date": "2021-01-01",
            "end_date": "2021-06-30",
            "seed": 7,
            "rebalance_frequency": "weekly",
            "strategies": ["momentum", "trend"],
        }
        report = execute_backtest(cfg)
        assert report.trades, "expected at least one closed trade"

        strategies_seen = {t["strategy"] for t in report.trades}
        assert "_default" not in strategies_seen
        assert strategies_seen <= {"momentum", "trend", "composite"}

    def test_report_includes_per_strategy_attribution_table(self):
        cfg = {
            "data_source": "synthetic",
            "start_date": "2021-01-01",
            "end_date": "2021-06-30",
            "seed": 7,
            "rebalance_frequency": "weekly",
            "strategies": ["momentum", "trend"],
        }
        report = execute_backtest(cfg)
        d = report.to_dict()
        assert "strategies" in d
        assert d["strategies"], "expected at least one strategy's metrics"
        for metrics in d["strategies"].values():
            assert "total_return" in metrics
            assert "sharpe_ratio" in metrics


def _empty_attribution():
    from firm.portfolio.attribution import PerformanceAttribution
    return PerformanceAttribution()
