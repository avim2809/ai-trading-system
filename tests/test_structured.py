"""Phase 1b: DuckDB structured-query layer over run artifacts."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from firm.rag.structured import ReadOnlyQueryError, RunStore


def _make_run(runs_dir, run_id, *, sharpe, max_dd, trades):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    report = {
        "portfolio": {
            "total_return": 0.2, "cagr": 0.1, "sharpe_ratio": sharpe,
            "max_drawdown": max_dd, "hit_rate": 0.55,
        },
        "period": {"start": "2021-01-01", "end": "2021-12-31"},
        "final_nav": 12_000_000.0,
        "data_points": 252,
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    pd.DataFrame(trades).to_parquet(run_dir / "trades.parquet", index=False)
    return run_id


@pytest.fixture
def runs_dir(tmp_path):
    base = tmp_path / "runs"
    base.mkdir()
    _make_run(base, "run_a", sharpe=1.5, max_dd=-0.10, trades=[
        {"symbol": "AAPL", "strategy": "mom", "size": 100.0,
         "pnl": 500.0, "pnl_net": 498.0, "return_pct": 0.04, "bars_held": 4},
    ])
    _make_run(base, "run_b", sharpe=0.8, max_dd=-0.25, trades=[
        {"symbol": "MSFT", "strategy": "rev", "size": -50.0,
         "pnl": -200.0, "pnl_net": -202.0, "return_pct": -0.02, "bars_held": 9},
        {"symbol": "AAPL", "strategy": "mom", "size": 80.0,
         "pnl": 300.0, "pnl_net": 299.0, "return_pct": 0.03, "bars_held": 3},
    ])
    # RunStore tags each trades.parquet with its run_id on load.
    return str(base)


class TestRunStore:
    def test_runs_view_one_row_per_run(self, runs_dir):
        rs = RunStore(runs_dir)
        df = rs.runs()
        assert set(df["run_id"]) == {"run_a", "run_b"}
        assert "sharpe_ratio" in df.columns

    def test_best_sharpe_via_sql(self, runs_dir):
        rs = RunStore(runs_dir)
        df = rs.query(
            "SELECT run_id FROM runs ORDER BY sharpe_ratio DESC LIMIT 1"
        )
        assert df.iloc[0]["run_id"] == "run_a"

    def test_trades_view_tagged_with_run_id(self, runs_dir):
        rs = RunStore(runs_dir)
        df = rs.query("SELECT run_id, symbol, size FROM trades ORDER BY symbol, size")
        assert len(df) == 3
        assert set(df["run_id"]) == {"run_a", "run_b"}
        # Short position preserved as a negative size.
        assert (df["size"] < 0).any()

    def test_aggregate_join(self, runs_dir):
        rs = RunStore(runs_dir)
        df = rs.query(
            "SELECT run_id, COUNT(*) AS n, SUM(pnl_net) AS net "
            "FROM trades GROUP BY run_id ORDER BY run_id"
        )
        by_run = {r["run_id"]: r for r in df.to_dict("records")}
        assert by_run["run_b"]["n"] == 2
        assert abs(by_run["run_b"]["net"] - 97.0) < 1e-6

    def test_read_only_guard_rejects_mutations(self, runs_dir):
        rs = RunStore(runs_dir)
        for bad in ["DROP TABLE runs", "DELETE FROM trades",
                    "SELECT 1; DROP TABLE runs", "UPDATE runs SET seed=1"]:
            with pytest.raises(ReadOnlyQueryError):
                rs.query(bad)

    def test_empty_runs_dir(self, tmp_path):
        rs = RunStore(str(tmp_path / "nope"))
        assert rs.runs().empty
        assert rs.trades().empty
        # schema() still works and names both tables.
        assert "runs(" in rs.schema() and "trades(" in rs.schema()

    def test_schema_lists_columns(self, runs_dir):
        schema = RunStore(runs_dir).schema()
        assert "sharpe_ratio" in schema
        assert "SELECT" in schema
