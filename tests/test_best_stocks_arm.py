"""Tests for the Danelfin Best-Stocks selection logic and synthetic ledger.

Pure logic — no network. select_best_stocks is tested against a
DanelfinProvider whose get_trade_ideas is mocked per-sector.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from firm.live.best_stocks_arm import (
    SECTORS,
    TARGET_HOLDINGS,
    select_best_stocks,
)
from firm.live.best_stocks_ledger import BestStocksLedger


def _sector_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _candidate(symbol: str, aiscore: float, low_risk: float = 6, volume: float = 500_000, win_rate: float = 0.6) -> dict:
    return {
        "symbol": symbol, "aiscore": aiscore, "low_risk": low_risk,
        "average_volume_3m": volume, "win_rate_3m": win_rate,
    }


class TestSelectBestStocks:
    def test_picks_top_sectors_by_avg_aiscore(self):
        provider = MagicMock()

        def fake_get_trade_ideas(**kwargs):
            sector = kwargs["sector"]
            # information-technology: high avg aiscore, 5 candidates
            if sector == "information-technology":
                return _sector_df([_candidate(f"IT{i}", 9 - i * 0.1) for i in range(6)])
            # energy: low avg aiscore, 5 candidates
            if sector == "energy":
                return _sector_df([_candidate(f"EN{i}", 4 - i * 0.1) for i in range(6)])
            return pd.DataFrame()

        provider.get_trade_ideas = MagicMock(side_effect=fake_get_trade_ideas)
        selection = select_best_stocks(provider, top_n_sectors=1, top_n_per_sector=5)

        assert len(selection) == 5
        assert all(row["sector"] == "information-technology" for row in selection)

    def test_sector_with_too_few_candidates_is_excluded(self):
        provider = MagicMock()

        def fake_get_trade_ideas(**kwargs):
            sector = kwargs["sector"]
            if sector == "information-technology":
                return _sector_df([_candidate(f"IT{i}", 8) for i in range(2)])  # only 2, need 5
            if sector == "energy":
                return _sector_df([_candidate(f"EN{i}", 5) for i in range(6)])
            return pd.DataFrame()

        provider.get_trade_ideas = MagicMock(side_effect=fake_get_trade_ideas)
        selection = select_best_stocks(provider, top_n_sectors=1, top_n_per_sector=5)

        assert all(row["sector"] == "energy" for row in selection)

    def test_no_eligible_sectors_returns_empty(self):
        provider = MagicMock()
        provider.get_trade_ideas = MagicMock(return_value=pd.DataFrame())
        selection = select_best_stocks(provider)
        assert selection == []

    def test_full_universe_yields_25_stocks(self):
        """Every SECTORS entry has >= 5 candidates -> full 5x5 = 25 selection."""
        provider = MagicMock()

        def fake_get_trade_ideas(**kwargs):
            sector = kwargs["sector"]
            idx = SECTORS.index(sector)
            return _sector_df([_candidate(f"{sector[:3].upper()}{i}", 9 - idx * 0.1 - i * 0.01) for i in range(6)])

        provider.get_trade_ideas = MagicMock(side_effect=fake_get_trade_ideas)
        selection = select_best_stocks(provider)
        assert len(selection) == TARGET_HOLDINGS
        assert len({row["sector"] for row in selection}) == 5

    def test_within_sector_picks_highest_aiscore(self):
        provider = MagicMock()

        def fake_get_trade_ideas(**kwargs):
            if kwargs["sector"] == "energy":
                return _sector_df([
                    _candidate("LOW1", 3), _candidate("LOW2", 3.5),
                    _candidate("HIGH1", 9), _candidate("HIGH2", 8.5),
                    _candidate("HIGH3", 8), _candidate("HIGH4", 7.5), _candidate("HIGH5", 7),
                ])
            return pd.DataFrame()

        provider.get_trade_ideas = MagicMock(side_effect=fake_get_trade_ideas)
        selection = select_best_stocks(provider, top_n_sectors=1, top_n_per_sector=5)
        symbols = {row["symbol"] for row in selection}
        assert symbols == {"HIGH1", "HIGH2", "HIGH3", "HIGH4", "HIGH5"}


class TestBestStocksLedger:
    def test_full_rebalance_equal_weights(self):
        ledger = BestStocksLedger(initial_capital=100_000.0, cash=100_000.0)
        asof = datetime(2026, 1, 1, tzinfo=timezone.utc)
        selection = [
            {"symbol": "AAA", "sector": "x", "aiscore": 9, "low_risk": 6, "average_volume_3m": 1e6, "sector_avg_aiscore": 9},
            {"symbol": "BBB", "sector": "x", "aiscore": 8, "low_risk": 6, "average_volume_3m": 1e6, "sector_avg_aiscore": 9},
        ]
        prices = {"AAA": 100.0, "BBB": 50.0}
        ledger.full_rebalance(asof, selection, prices)

        assert ledger.holdings["AAA"] == pytest.approx(500.0)  # 50k / 100
        assert ledger.holdings["BBB"] == pytest.approx(1000.0)  # 50k / 50
        assert ledger.nav(prices) == pytest.approx(100_000.0)

    def test_mark_to_market_appends_nav_history(self):
        ledger = BestStocksLedger(initial_capital=10_000.0, cash=0.0, holdings={"AAA": 100.0})
        asof = datetime(2026, 1, 1, tzinfo=timezone.utc)
        nav = ledger.mark_to_market(asof, {"AAA": 110.0})
        assert nav == pytest.approx(11_000.0)
        assert len(ledger.nav_history) == 1
        assert ledger.nav_history[0]["nav"] == pytest.approx(11_000.0)

    def test_quarterly_replace_swaps_dropped_symbol(self):
        ledger = BestStocksLedger(
            initial_capital=10_000.0, cash=0.0,
            holdings={"OLD": 100.0, "KEEP": 50.0},
        )
        asof = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fresh_selection = [
            {"symbol": "KEEP", "sector": "x", "aiscore": 8, "low_risk": 6, "average_volume_3m": 1e6, "sector_avg_aiscore": 8},
            {"symbol": "NEW", "sector": "x", "aiscore": 9, "low_risk": 6, "average_volume_3m": 1e6, "sector_avg_aiscore": 8},
        ]
        prices = {"OLD": 10.0, "KEEP": 20.0, "NEW": 40.0}
        ledger.quarterly_replace(asof, fresh_selection, prices)

        assert "OLD" not in ledger.holdings
        assert "NEW" in ledger.holdings
        assert ledger.holdings["KEEP"] == pytest.approx(50.0)  # unchanged
        assert ledger.holdings["NEW"] == pytest.approx(1000.0 / 40.0)  # freed $1000 (100*10) / $40

    def test_quarterly_replace_no_changes_when_nothing_dropped(self):
        ledger = BestStocksLedger(initial_capital=10_000.0, cash=0.0, holdings={"AAA": 100.0})
        asof = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fresh_selection = [{"symbol": "AAA", "sector": "x", "aiscore": 8, "low_risk": 6, "average_volume_3m": 1e6, "sector_avg_aiscore": 8}]
        ledger.quarterly_replace(asof, fresh_selection, {"AAA": 100.0})
        assert ledger.holdings == {"AAA": 100.0}
        assert ledger.last_quarterly_replace == "2026-01-01"

    def test_annual_rebalance_resets_weights_same_symbols(self):
        ledger = BestStocksLedger(
            initial_capital=10_000.0, cash=0.0,
            holdings={"AAA": 50.0, "BBB": 200.0},  # drifted: AAA@100=5000, BBB@10=2000 -> unequal
        )
        asof = datetime(2026, 1, 1, tzinfo=timezone.utc)
        prices = {"AAA": 100.0, "BBB": 10.0}
        ledger.annual_rebalance(asof, prices)

        aaa_value = ledger.holdings["AAA"] * prices["AAA"]
        bbb_value = ledger.holdings["BBB"] * prices["BBB"]
        assert aaa_value == pytest.approx(bbb_value)
        assert set(ledger.holdings) == {"AAA", "BBB"}  # symbols unchanged

    def test_due_for_quarterly_replace(self):
        ledger = BestStocksLedger(last_quarterly_replace="2026-01-01")
        assert not ledger.due_for_quarterly_replace(datetime(2026, 2, 1, tzinfo=timezone.utc))
        assert ledger.due_for_quarterly_replace(datetime(2026, 4, 5, tzinfo=timezone.utc))

    def test_due_for_annual_rebalance(self):
        ledger = BestStocksLedger(last_full_rebalance="2025-01-01")
        assert not ledger.due_for_annual_rebalance(datetime(2025, 6, 1, tzinfo=timezone.utc))
        assert ledger.due_for_annual_rebalance(datetime(2026, 1, 5, tzinfo=timezone.utc))

    def test_save_and_load_roundtrip(self, tmp_path):
        ledger = BestStocksLedger(
            initial_capital=10_000.0, cash=123.45, holdings={"AAA": 10.0},
            last_full_rebalance="2026-01-01", last_quarterly_replace="2026-01-01",
        )
        path = tmp_path / "ledger.json"
        ledger.save(path)
        loaded = BestStocksLedger.load(path)
        assert loaded.holdings == {"AAA": 10.0}
        assert loaded.cash == pytest.approx(123.45)
        assert loaded.last_full_rebalance == "2026-01-01"

    def test_load_missing_file_returns_default(self, tmp_path):
        loaded = BestStocksLedger.load(tmp_path / "does_not_exist.json")
        assert loaded.holdings == {}
        assert loaded.cash == 0.0
