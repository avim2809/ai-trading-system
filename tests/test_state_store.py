"""Tests for firm.live.state_store.LiveStateStore and its two callers:
PortfolioState.restore_history and PerformanceAttribution.export_state/
restore_state.
"""

from __future__ import annotations

from datetime import datetime

from firm.contracts.models import PortfolioSnapshot
from firm.live.state_store import LiveStateStore
from firm.portfolio.attribution import PerformanceAttribution
from firm.portfolio.state import PortfolioState


class TestLiveStateStorePortfolioHistory:
    def test_round_trips_snapshots(self, tmp_path):
        store = LiveStateStore(tmp_path / "state.db")
        snaps = [
            PortfolioSnapshot(
                asof=datetime(2024, 1, 1, 15, 30),
                holdings={"AAPL": 10.0},
                weights={"AAPL": 0.5},
                cash=50_000.0,
                nav=100_000.0,
                per_strategy_pnl={"momentum": 250.0},
            ),
            PortfolioSnapshot(
                asof=datetime(2024, 1, 2, 15, 30),
                holdings={"AAPL": 12.0},
                weights={"AAPL": 0.55},
                cash=48_000.0,
                nav=101_000.0,
                per_strategy_pnl={"momentum": 300.0},
            ),
        ]
        store.save_portfolio_history(snaps)

        restored = store.load_portfolio_history()
        assert len(restored) == 2
        assert restored[0].asof == snaps[0].asof
        assert restored[1].nav == 101_000.0
        assert restored[1].per_strategy_pnl == {"momentum": 300.0}
        store.close()

    def test_load_with_nothing_persisted_returns_empty_list(self, tmp_path):
        store = LiveStateStore(tmp_path / "state.db")
        assert store.load_portfolio_history() == []
        store.close()

    def test_save_trims_to_max_snapshots(self, tmp_path):
        store = LiveStateStore(tmp_path / "state.db", max_snapshots=3)
        snaps = [
            PortfolioSnapshot(asof=datetime(2024, 1, i + 1), cash=float(i), nav=float(i))
            for i in range(5)
        ]
        store.save_portfolio_history(snaps)
        restored = store.load_portfolio_history()
        assert len(restored) == 3
        # Oldest are dropped; the most recent three survive in order.
        assert [s.nav for s in restored] == [2.0, 3.0, 4.0]
        store.close()

    def test_corrupt_blob_degrades_to_empty_list(self, tmp_path):
        store = LiveStateStore(tmp_path / "state.db")
        store._save_blob("portfolio_history", "not-a-list-of-snapshot-dicts")
        # A JSON string (not a list of dicts) round-trips through json.loads
        # fine but PortfolioSnapshot(**row) must fail per-row rather than
        # raising out of load_portfolio_history().
        assert store.load_portfolio_history() == []
        store.close()


class TestLiveStateStoreAttribution:
    def test_round_trips_attribution_state(self, tmp_path):
        store = LiveStateStore(tmp_path / "state.db")
        state = {
            "trade_log": [{"symbol": "AAPL", "shares": 10.0, "price": 150.0, "strategy": "momentum"}],
            "strategy_returns": {"momentum": [0.01, -0.005]},
            "strategy_dates": {"momentum": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"]},
            "strategy_holdings": {"momentum": {"AAPL": 10.0}},
            "prev_prices": {"AAPL": 151.0},
            "factor_exposures": {"momentum": {"beta": 1.1}},
        }
        store.save_attribution_state(state)
        restored = store.load_attribution_state()
        assert restored == state
        store.close()

    def test_load_with_nothing_persisted_returns_none(self, tmp_path):
        store = LiveStateStore(tmp_path / "state.db")
        assert store.load_attribution_state() is None
        store.close()


class TestLiveStateStoreKillSwitchMirror:
    def test_round_trips_kill_switch_blob(self, tmp_path):
        store = LiveStateStore(tmp_path / "state.db")
        store.save_kill_switch({"halted": True, "peak_equity": 123.45})
        assert store.load_kill_switch() == {"halted": True, "peak_equity": 123.45}
        store.close()


class TestPortfolioStateRestoreHistory:
    def test_restore_history_replaces_history_only(self):
        portfolio = PortfolioState(initial_capital=100_000)
        portfolio.update([{"symbol": "AAPL", "shares": 5, "price": 150.0, "strategy": "momentum"}], {"AAPL": 150.0})
        snaps = [
            PortfolioSnapshot(asof=datetime(2024, 1, 1), cash=1.0, nav=2.0),
            PortfolioSnapshot(asof=datetime(2024, 1, 2), cash=3.0, nav=4.0),
        ]
        portfolio.restore_history(snaps)

        assert portfolio.history == snaps
        # cash/holdings are untouched by restore_history — broker sync (not
        # this method) is the sole source of truth for those in live mode.
        assert portfolio.holdings == {"AAPL": 5}
        assert portfolio.cash != 1.0


class TestPerformanceAttributionExportRestore:
    def test_export_then_restore_round_trips(self):
        attr = PerformanceAttribution()
        attr.record_trades(
            [{"symbol": "AAPL", "shares": 10.0, "price": 150.0, "strategy": "momentum"}],
            {"AAPL": 150.0},
        )
        attr.update_daily(datetime(2024, 1, 1), {"AAPL": 150.0}, nav=100_000.0)
        attr.update_daily(datetime(2024, 1, 2), {"AAPL": 155.0}, nav=100_050.0)
        attr.set_factor_exposures("momentum", {"beta": 1.2})

        state = attr.export_state()

        restored = PerformanceAttribution()
        restored.restore_state(state)

        assert restored.trade_log == attr.trade_log
        assert restored.strategies == attr.strategies
        pd_series_a = attr.get_strategy_returns("momentum")
        pd_series_b = restored.get_strategy_returns("momentum")
        assert list(pd_series_a.values) == list(pd_series_b.values)
        assert list(pd_series_a.index) == list(pd_series_b.index)
        assert restored.get_factor_attribution().equals(attr.get_factor_attribution())
        assert restored.dominant_strategy_by_symbol() == attr.dominant_strategy_by_symbol()

    def test_restore_state_tolerates_missing_keys(self):
        attr = PerformanceAttribution()
        attr.restore_state({})
        assert attr.trade_log == []
        assert attr.strategies == []

    def test_restore_state_tolerates_partial_state(self):
        attr = PerformanceAttribution()
        attr.restore_state({"strategy_returns": {"momentum": [0.01]}})
        assert attr.strategies == ["momentum"]
        assert attr.trade_log == []
