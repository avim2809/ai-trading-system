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
    select_best_stocks_historical,
    select_from_real_beststocks,
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


def _real_beststocks_df(rows: list[tuple[str, int]]) -> pd.DataFrame:
    """rows: list of (symbol, rank)."""
    return pd.DataFrame([
        {
            "date": "2026-08-02", "symbol": sym, "rank": rank,
            "ai_score": 9.5 - rank * 0.1, "ai_score_change": 0.1,
            "fundamental_score": 8.0, "technical_score": 8.0,
            "sentiment_score": 8.0, "low_risk_score": 7.0,
            "perf_ytd": 0.1, "sector": "technology", "country": "US",
        }
        for sym, rank in rows
    ])


class TestSelectFromRealBestStocks:
    def test_wraps_real_list_directly_no_reconstruction(self):
        provider = MagicMock()
        provider.get_best_stocks.return_value = _real_beststocks_df([("AAPL", 1), ("NVDA", 2)])
        selection = select_from_real_beststocks(provider)
        assert [row["symbol"] for row in selection] == ["AAPL", "NVDA"]
        assert selection[0]["rank"] == 1
        assert selection[0]["sector"] == "technology"

    def test_empty_list_returns_empty(self):
        provider = MagicMock()
        provider.get_best_stocks.return_value = pd.DataFrame()
        assert select_from_real_beststocks(provider) == []

    def test_excludes_main_engine_collisions(self):
        provider = MagicMock()
        provider.get_best_stocks.return_value = _real_beststocks_df([("AAPL", 1), ("NVDA", 2)])
        selection = select_from_real_beststocks(provider, excluded_symbols=frozenset({"AAPL"}))
        assert [row["symbol"] for row in selection] == ["NVDA"]

    def test_sorted_by_rank(self):
        provider = MagicMock()
        # Deliberately out of rank order in the raw response.
        provider.get_best_stocks.return_value = _real_beststocks_df([("NVDA", 2), ("AAPL", 1)])
        selection = select_from_real_beststocks(provider)
        assert [row["symbol"] for row in selection] == ["AAPL", "NVDA"]


def _hist_candidate(symbol: str, aiscore: float, low_risk: float = 6) -> dict:
    return {"symbol": symbol, "aiscore": aiscore, "low_risk": low_risk}


class TestSelectBestStocksHistorical:
    """select_best_stocks_historical uses get_historical_sector_scores
    (genuinely historical bulk /ranking mode) instead of /v3/trade-ideas —
    see that function's docstring for why sector ranking is deliberately
    NOT volume-filtered (only the final per-sector picks are)."""

    def test_sector_ranking_uses_full_unfiltered_pool(self):
        """Sector average must reflect ALL low_risk-qualifying candidates,
        not just the ones that would pass a volume filter — this is the
        documented THIRD deviation in select_best_stocks_historical's
        docstring."""
        provider = MagicMock()

        def fake_scan(sector, date, **kwargs):
            if sector == "information-technology":
                # High avg aiscore among 6 candidates
                return _sector_df([_hist_candidate(f"IT{i}", 9 - i * 0.1) for i in range(6)])
            if sector == "energy":
                return _sector_df([_hist_candidate(f"EN{i}", 4 - i * 0.1) for i in range(6)])
            return pd.DataFrame()

        provider.get_historical_sector_scores = MagicMock(side_effect=fake_scan)
        selection = select_best_stocks_historical(
            provider, "2024-06-03", top_n_sectors=1, top_n_per_sector=5,
        )
        assert len(selection) == 5
        assert all(row["sector"] == "information-technology" for row in selection)
        assert all(row["date"] == "2024-06-03" for row in selection)

    def test_volume_filter_only_applied_to_final_picks_not_ranking(self):
        """A sector's ranking-average must be computed from the FULL pool
        even when the volume filter would reject most of it — only the
        final top-N-per-sector picks skip volume-failing names."""
        provider = MagicMock()

        def fake_scan(sector, date, **kwargs):
            if sector == "energy":
                # 6 candidates: top 2 by aiscore fail volume, rest pass.
                return _sector_df([
                    _hist_candidate("FAIL1", 9), _hist_candidate("FAIL2", 8.5),
                    _hist_candidate("OK1", 8), _hist_candidate("OK2", 7.5),
                    _hist_candidate("OK3", 7), _hist_candidate("OK4", 6.5), _hist_candidate("OK5", 6),
                ])
            return pd.DataFrame()

        provider.get_historical_sector_scores = MagicMock(side_effect=fake_scan)
        volume_filter = lambda sym: not sym.startswith("FAIL")  # noqa: E731
        selection = select_best_stocks_historical(
            provider, "2024-06-03", top_n_sectors=1, top_n_per_sector=5, volume_filter=volume_filter,
        )
        symbols = {row["symbol"] for row in selection}
        assert symbols == {"OK1", "OK2", "OK3", "OK4", "OK5"}
        assert "FAIL1" not in symbols and "FAIL2" not in symbols

    def test_sector_underfilled_when_too_few_pass_volume_filter(self):
        """If a sector runs out of volume-passing candidates before
        filling top_n_per_sector slots, it should return however many it
        found rather than crash or silently include a failing symbol."""
        provider = MagicMock()

        def fake_scan(sector, date, **kwargs):
            if sector == "energy":
                return _sector_df([_hist_candidate(f"E{i}", 9 - i) for i in range(6)])
            return pd.DataFrame()

        provider.get_historical_sector_scores = MagicMock(side_effect=fake_scan)
        volume_filter = lambda sym: sym in ("E0", "E1")  # noqa: E731 — only 2 pass, need 5
        selection = select_best_stocks_historical(
            provider, "2024-06-03", top_n_sectors=1, top_n_per_sector=5, volume_filter=volume_filter,
        )
        assert {row["symbol"] for row in selection} == {"E0", "E1"}

    def test_sector_with_too_few_total_candidates_is_ineligible(self):
        provider = MagicMock()

        def fake_scan(sector, date, **kwargs):
            if sector == "information-technology":
                return _sector_df([_hist_candidate(f"IT{i}", 8) for i in range(2)])  # need 5
            if sector == "energy":
                return _sector_df([_hist_candidate(f"EN{i}", 5) for i in range(6)])
            return pd.DataFrame()

        provider.get_historical_sector_scores = MagicMock(side_effect=fake_scan)
        selection = select_best_stocks_historical(
            provider, "2024-06-03", top_n_sectors=1, top_n_per_sector=5,
        )
        assert all(row["sector"] == "energy" for row in selection)

    def test_no_eligible_sectors_returns_empty(self):
        provider = MagicMock()
        provider.get_historical_sector_scores = MagicMock(return_value=pd.DataFrame())
        selection = select_best_stocks_historical(provider, "2024-06-03")
        assert selection == []

    def test_no_volume_filter_selects_top_aiscore_directly(self):
        provider = MagicMock()

        def fake_scan(sector, date, **kwargs):
            if sector == "energy":
                return _sector_df([_hist_candidate(f"E{i}", 9 - i) for i in range(6)])
            return pd.DataFrame()

        provider.get_historical_sector_scores = MagicMock(side_effect=fake_scan)
        selection = select_best_stocks_historical(
            provider, "2024-06-03", top_n_sectors=1, top_n_per_sector=5,
        )
        assert {row["symbol"] for row in selection} == {"E0", "E1", "E2", "E3", "E4"}


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
