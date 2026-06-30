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


def _empty_attribution():
    from firm.portfolio.attribution import PerformanceAttribution
    return PerformanceAttribution()
