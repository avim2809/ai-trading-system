"""Tests for the FMP-sourced sector-balanced dynamic universe scanner
(firm.live.sp500_universe_sync / firm.live.sp500_sector_cache), plus the
shared FMP sector-normalization helper (firm.data.providers.fmp).

Mirrors tests/test_danelfin_universe_sync.py's structure: pure logic in
isolation (build_diversified_candidates, sector-cache merge), a thin
orchestration wrapper (sync_once) with a mocked engine, and the JSON
state/cache persistence idioms.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.data.providers.base import ProviderError
from firm.data.providers.fmp import FMPProvider, _normalize_gics_sector
from firm.live.dynamic_universe_state import save_dynamic_universe_state
from firm.live.sp500_sector_cache import (
    load_sector_cache,
    merge_sector_records,
    refresh_sector_cache,
    save_sector_cache,
)
from firm.live.sp500_universe_sync import build_diversified_candidates, sync_once


def _constituents_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows: list of (symbol, sector)."""
    return pd.DataFrame([{"symbol": sym, "sector": sector} for sym, sector in rows])


class TestBuildDiversifiedCandidates:
    def test_least_represented_sector_wins_first_slot(self):
        candidates = _constituents_df(
            [("NVDA", "technology"), ("AMD", "technology"), ("XOM", "energy")]
        )
        ranked = build_diversified_candidates(
            candidates,
            static_sector_counts={"technology": 5, "energy": 1},
            liquidity={"NVDA": 100.0, "AMD": 50.0, "XOM": 10.0},
            max_dynamic_symbols=1,
        )
        # energy (count=1) is less represented than technology (count=5) --
        # its only candidate (XOM) should win the single slot.
        assert list(ranked["symbol"]) == ["XOM"]
        assert list(ranked["sector"]) == ["energy"]

    def test_water_fill_levels_across_multiple_thin_sectors(self):
        """With 2 slots and technology starting far ahead of energy/
        utilities, both slots should go to the two currently-thinnest
        sectors rather than double-filling from a single one."""
        candidates = _constituents_df(
            [("NVDA", "technology"), ("XOM", "energy"), ("NEE", "utilities")]
        )
        ranked = build_diversified_candidates(
            candidates,
            static_sector_counts={"technology": 10, "energy": 0, "utilities": 0},
            liquidity={"NVDA": 100.0, "XOM": 10.0, "NEE": 5.0},
            max_dynamic_symbols=2,
        )
        assert set(ranked["sector"]) == {"energy", "utilities"}
        assert "NVDA" not in set(ranked["symbol"])

    def test_within_sector_liquidity_ranking(self):
        candidates = _constituents_df([("AMD", "technology"), ("NVDA", "technology")])
        ranked = build_diversified_candidates(
            candidates,
            static_sector_counts={},
            liquidity={"AMD": 10.0, "NVDA": 1000.0},
            max_dynamic_symbols=1,
        )
        assert list(ranked["symbol"]) == ["NVDA"]

    def test_ties_broken_alphabetically(self):
        candidates = _constituents_df([("ZETA", "technology"), ("ALPHA", "technology")])
        ranked = build_diversified_candidates(
            candidates,
            static_sector_counts={},
            liquidity={},
            max_dynamic_symbols=1,
        )
        assert list(ranked["symbol"]) == ["ALPHA"]

    def test_exclusions_respected(self):
        candidates = _constituents_df([("NVDA", "technology"), ("AMD", "technology")])
        ranked = build_diversified_candidates(
            candidates,
            static_sector_counts={},
            liquidity={"NVDA": 100.0, "AMD": 50.0},
            max_dynamic_symbols=2,
            exclude={"NVDA"},
        )
        assert list(ranked["symbol"]) == ["AMD"]

    def test_unknown_sector_rows_excluded(self):
        candidates = _constituents_df([("NVDA", "unknown"), ("AMD", "technology")])
        ranked = build_diversified_candidates(
            candidates,
            static_sector_counts={},
            liquidity={"NVDA": 1000.0, "AMD": 10.0},
            max_dynamic_symbols=2,
        )
        assert list(ranked["symbol"]) == ["AMD"]

    def test_empty_input_returns_empty_frame(self):
        ranked = build_diversified_candidates(
            pd.DataFrame(columns=["symbol", "sector"]),
            static_sector_counts={},
            liquidity={},
            max_dynamic_symbols=10,
        )
        assert ranked.empty
        assert list(ranked.columns) == ["symbol", "sector"]

    def test_zero_max_dynamic_symbols_returns_empty(self):
        candidates = _constituents_df([("NVDA", "technology")])
        ranked = build_diversified_candidates(
            candidates, static_sector_counts={}, liquidity={}, max_dynamic_symbols=0,
        )
        assert ranked.empty

    def test_caps_at_max_dynamic_symbols(self):
        candidates = _constituents_df(
            [("NVDA", "technology"), ("AMD", "technology"), ("XOM", "energy")]
        )
        ranked = build_diversified_candidates(
            candidates, static_sector_counts={}, liquidity={}, max_dynamic_symbols=2,
        )
        assert len(ranked) == 2

    def test_determinism(self):
        candidates = _constituents_df(
            [("NVDA", "technology"), ("AMD", "technology"), ("XOM", "energy"), ("CVX", "energy")]
        )
        kwargs = dict(
            static_sector_counts={"technology": 3, "energy": 1},
            liquidity={"NVDA": 100.0, "AMD": 90.0, "XOM": 20.0, "CVX": 15.0},
            max_dynamic_symbols=3,
        )
        first = build_diversified_candidates(candidates, **kwargs)
        second = build_diversified_candidates(candidates, **kwargs)
        assert list(first["symbol"]) == list(second["symbol"])


class TestSectorCache:
    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_sector_cache(tmp_path / "nope.json") == {}

    def test_save_then_load_round_trips(self, tmp_path):
        path = tmp_path / "sub" / "sectors.json"
        cache = {"AAPL": {"sector": "technology", "source": "static", "as_of": "2026-08-02"}}
        save_sector_cache(path, cache)
        assert path.exists()
        assert load_sector_cache(path) == cache

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json")
        assert load_sector_cache(path) == {}

    def test_load_non_dict_json_returns_empty(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text(json.dumps(["not", "a", "dict"]))
        assert load_sector_cache(path) == {}

    def test_merge_overlays_fmp_rows_on_existing(self):
        existing = {"AAPL": {"sector": "technology", "source": "static", "as_of": "2026-08-01"}}
        fmp_rows = _constituents_df([("NVDA", "technology")])
        merged = merge_sector_records(existing, fmp_rows, seed_map=None, today="2026-08-02")
        assert merged["NVDA"] == {"sector": "technology", "source": "fmp", "as_of": "2026-08-02"}
        assert merged["AAPL"]["sector"] == "technology"

    def test_merge_drops_unknown_sector_rows(self):
        fmp_rows = _constituents_df([("XYZ", "unknown")])
        merged = merge_sector_records({}, fmp_rows, seed_map=None, today="2026-08-02")
        assert "XYZ" not in merged

    def test_merge_seeds_missing_static_names(self):
        merged = merge_sector_records(
            {},
            pd.DataFrame(columns=["symbol", "sector"]),
            seed_map={"AAPL": "technology", "SPY": "unknown"},
            today="2026-08-02",
        )
        assert merged["AAPL"] == {"sector": "technology", "source": "static", "as_of": "2026-08-02"}
        assert "SPY" not in merged

    def test_merge_seed_map_never_overrides_fmp_row(self):
        fmp_rows = _constituents_df([("AAPL", "technology")])
        merged = merge_sector_records(
            {}, fmp_rows, seed_map={"AAPL": "financials"}, today="2026-08-02",
        )
        assert merged["AAPL"]["source"] == "fmp"

    def test_merge_preserves_existing_when_fmp_rows_empty(self):
        existing = {"AAPL": {"sector": "technology", "source": "fmp", "as_of": "2026-08-01"}}
        merged = merge_sector_records(
            existing, pd.DataFrame(columns=["symbol", "sector"]), seed_map=None, today="2026-08-02",
        )
        assert merged == existing

    def test_refresh_sector_cache_saves_merged_result(self, tmp_path):
        path = tmp_path / "sectors.json"
        provider = MagicMock()
        provider.get_universe_constituents_with_sectors.return_value = _constituents_df(
            [("NVDA", "technology")]
        )
        result = refresh_sector_cache(path, fmp_provider=provider, today="2026-08-02")
        assert result["NVDA"]["sector"] == "technology"
        assert load_sector_cache(path) == result

    def test_refresh_sector_cache_handles_fetch_failure_preserves_prior(self, tmp_path):
        path = tmp_path / "sectors.json"
        save_sector_cache(
            path, {"AAPL": {"sector": "technology", "source": "static", "as_of": "2026-08-01"}}
        )
        provider = MagicMock()
        provider.get_universe_constituents_with_sectors.side_effect = RuntimeError("boom")
        result = refresh_sector_cache(path, fmp_provider=provider, today="2026-08-02")
        assert result["AAPL"]["sector"] == "technology"

    def test_refresh_sector_cache_backfill_bounded(self, tmp_path):
        path = tmp_path / "sectors.json"
        provider = MagicMock()
        # Two unknown-sector candidates, only 1 allowed to backfill.
        provider.get_universe_constituents_with_sectors.return_value = _constituents_df(
            [("A", "unknown"), ("B", "unknown")]
        )
        backfill = MagicMock()
        backfill.get_sector.return_value = "technology"
        result = refresh_sector_cache(
            path, fmp_provider=provider, backfill_provider=backfill, backfill_limit=1, today="2026-08-02",
        )
        assert backfill.get_sector.call_count == 1
        assert sum(1 for sym in ("A", "B") if sym in result) == 1

    def test_refresh_sector_cache_no_backfill_without_provider(self, tmp_path):
        path = tmp_path / "sectors.json"
        provider = MagicMock()
        provider.get_universe_constituents_with_sectors.return_value = _constituents_df(
            [("A", "unknown")]
        )
        result = refresh_sector_cache(path, fmp_provider=provider, today="2026-08-02")
        assert "A" not in result


class TestFMPSectorNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Information Technology", "technology"),
            ("Technology", "technology"),
            ("Communication Services", "communication"),
            ("Consumer Cyclical", "consumer_discretionary"),
            ("Consumer Defensive", "consumer_staples"),
            ("Financial Services", "financials"),
            ("Healthcare", "healthcare"),
            ("Energy", "energy"),
            ("Real Estate", "real_estate"),
            ("Utilities", "utilities"),
            ("Basic Materials", "materials"),
        ],
    )
    def test_normalize_known_labels(self, raw, expected):
        assert _normalize_gics_sector(raw) == expected

    def test_normalize_case_insensitive(self):
        assert _normalize_gics_sector("TECHNOLOGY") == "technology"
        assert _normalize_gics_sector("information technology") == "technology"

    def test_normalize_unknown_label_returns_unknown(self):
        assert _normalize_gics_sector("Not A Real Sector") == "unknown"

    def test_normalize_empty_returns_unknown(self):
        assert _normalize_gics_sector("") == "unknown"
        assert _normalize_gics_sector(None) == "unknown"

    def test_get_universe_constituents_with_sectors_returns_symbol_and_sector(self):
        with patch("firm.data.providers.fmp.get_settings"):
            provider = FMPProvider(api_key="test-key")
        provider._client = MagicMock()
        provider._client.get_json.return_value = [
            {"symbol": "NVDA", "sector": "Information Technology"},
            {"symbol": "XOM", "sector": "Energy"},
        ]
        df = provider.get_universe_constituents_with_sectors("sp500")
        assert list(df["symbol"]) == ["NVDA", "XOM"]
        assert list(df["sector"]) == ["technology", "energy"]

    def test_get_universe_constituents_with_sectors_premium_gate_raises_not_implemented(self):
        with patch("firm.data.providers.fmp.get_settings"):
            provider = FMPProvider(api_key="test-key")
        provider._client = MagicMock()
        provider._client.get_json.side_effect = ProviderError("402 restricted")
        with pytest.raises(NotImplementedError):
            provider.get_universe_constituents_with_sectors("sp500")

    def test_get_universe_constituents_with_sectors_unknown_index_raises(self):
        with patch("firm.data.providers.fmp.get_settings"):
            provider = FMPProvider(api_key="test-key")
        with pytest.raises(NotImplementedError):
            provider.get_universe_constituents_with_sectors("russell3000")


class TestSyncOnce:
    def _make_engine(self, prices_df: pd.DataFrame | None = None):
        engine = MagicMock()
        prices_provider = MagicMock()
        prices_provider.get_prices.return_value = (
            prices_df if prices_df is not None else pd.DataFrame(columns=["symbol", "date", "close", "volume"])
        )
        engine._data_feed._providers = {"prices": prices_provider}
        return engine

    def test_no_provider_configured_skips(self, tmp_path):
        engine = MagicMock()
        engine._data_feed._providers = {}
        result = sync_once(
            engine,
            state_path=tmp_path / "state.json",
            sector_cache_path=tmp_path / "sectors.json",
            static_universe=["AAPL"],
            static_sector_map={"AAPL": "technology"},
            max_dynamic_symbols=10,
            min_dwell_days=5,
        )
        assert result == {"skipped": "no_provider"}
        engine.update_universe.assert_not_called()

    def test_fetch_failure_skips_without_touching_state(self, tmp_path):
        engine = self._make_engine()
        state_path = tmp_path / "state.json"
        with patch("firm.live.sp500_universe_sync.FMPProvider") as mock_fmp_cls:
            mock_fmp_cls.return_value.get_universe_constituents_with_sectors.side_effect = RuntimeError("boom")
            result = sync_once(
                engine,
                state_path=state_path,
                sector_cache_path=tmp_path / "sectors.json",
                static_universe=["AAPL"],
                static_sector_map={"AAPL": "technology"},
                max_dynamic_symbols=10,
                min_dwell_days=5,
            )
        assert result == {"skipped": "fetch_failed"}
        engine.update_universe.assert_not_called()
        assert not state_path.exists()

    def test_empty_sector_cache_skips(self, tmp_path):
        engine = self._make_engine()
        with patch("firm.live.sp500_universe_sync.FMPProvider") as mock_fmp_cls:
            mock_fmp_cls.return_value.get_universe_constituents_with_sectors.return_value = _constituents_df(
                [("NVDA", "technology")]
            )
            result = sync_once(
                engine,
                state_path=tmp_path / "state.json",
                sector_cache_path=tmp_path / "sectors.json",  # never created -> empty cache
                static_universe=["AAPL"],
                static_sector_map={"AAPL": "technology"},
                max_dynamic_symbols=10,
                min_dwell_days=5,
            )
        assert result == {"skipped": "empty_sector_cache"}
        engine.update_universe.assert_not_called()

    def test_additions_call_update_universe_and_sector_map(self, tmp_path):
        sector_cache_path = tmp_path / "sectors.json"
        save_sector_cache(
            sector_cache_path,
            {"NVDA": {"sector": "technology", "source": "fmp", "as_of": "2026-08-01"}},
        )
        prices_df = pd.DataFrame(
            [{"symbol": "NVDA", "date": "2026-08-01", "close": 100.0, "volume": 1000.0}]
        )
        engine = self._make_engine(prices_df)
        with patch("firm.live.sp500_universe_sync.FMPProvider") as mock_fmp_cls:
            mock_fmp_cls.return_value.get_universe_constituents_with_sectors.return_value = _constituents_df(
                [("NVDA", "technology")]
            )
            result = sync_once(
                engine,
                state_path=tmp_path / "state.json",
                sector_cache_path=sector_cache_path,
                static_universe=["AAPL"],
                static_sector_map={"AAPL": "technology"},
                max_dynamic_symbols=10,
                min_dwell_days=5,
            )
        assert result["additions"] == ["NVDA"]
        engine.update_universe.assert_called_once()
        assert set(engine.update_universe.call_args[0][0]) == {"AAPL", "NVDA"}
        engine.update_sector_map.assert_called_once_with({"NVDA": "technology"})

    def test_no_changes_does_not_call_engine_setters(self, tmp_path):
        """An already-held dynamic symbol that's still a real FMP
        constituent today must be recognized as present (no absence
        increment, no spurious removal) even though it's excluded from
        fresh water-fill selection."""
        sector_cache_path = tmp_path / "sectors.json"
        save_sector_cache(
            sector_cache_path,
            {"NVDA": {"sector": "technology", "source": "fmp", "as_of": "2026-08-01"}},
        )
        state_path = tmp_path / "state.json"
        save_dynamic_universe_state(
            state_path,
            {"NVDA": {"sector": "technology", "added_date": "2026-08-01", "consecutive_absent_days": 0}},
        )
        engine = self._make_engine()
        with patch("firm.live.sp500_universe_sync.FMPProvider") as mock_fmp_cls:
            mock_fmp_cls.return_value.get_universe_constituents_with_sectors.return_value = _constituents_df(
                [("NVDA", "technology")]
            )
            result = sync_once(
                engine,
                state_path=state_path,
                sector_cache_path=sector_cache_path,
                static_universe=["AAPL"],
                static_sector_map={"AAPL": "technology"},
                max_dynamic_symbols=10,
                min_dwell_days=5,
            )
        assert result["additions"] == []
        assert result["removals"] == []
        engine.update_universe.assert_not_called()
        engine.update_sector_map.assert_not_called()

    def test_held_symbol_absent_from_todays_fmp_list_increments_absence(self, tmp_path):
        """If an already-held symbol is genuinely no longer in today's FMP
        S&P 500 list, it must still be tracked toward dwell-based removal —
        this is real membership loss, not a selection-exclusion artifact."""
        sector_cache_path = tmp_path / "sectors.json"
        save_sector_cache(
            sector_cache_path,
            {"NVDA": {"sector": "technology", "source": "fmp", "as_of": "2026-08-01"}},
        )
        state_path = tmp_path / "state.json"
        save_dynamic_universe_state(
            state_path,
            {"NVDA": {"sector": "technology", "added_date": "2026-08-01", "consecutive_absent_days": 4}},
        )
        engine = self._make_engine()
        with patch("firm.live.sp500_universe_sync.FMPProvider") as mock_fmp_cls:
            # NVDA no longer appears in today's fresh FMP pull at all.
            mock_fmp_cls.return_value.get_universe_constituents_with_sectors.return_value = _constituents_df([])
            result = sync_once(
                engine,
                state_path=state_path,
                sector_cache_path=sector_cache_path,
                static_universe=["AAPL"],
                static_sector_map={"AAPL": "technology"},
                max_dynamic_symbols=10,
                min_dwell_days=5,
            )
        assert result["removals"] == ["NVDA"]
        engine.update_universe.assert_called_once()
