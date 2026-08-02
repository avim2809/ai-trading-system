"""Tests for dynamic universe growth (firm.live.danelfin_universe_sync).

Covers the pure logic (compute_universe_update) in isolation — no engine or
network — plus the thin orchestration wrapper (sync_once) with a mocked
engine, and the JSON state persistence idiom.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd

from firm.live.danelfin_universe_sync import compute_universe_update, sync_once
from firm.live.dynamic_universe_state import (
    load_dynamic_universe_state,
    save_dynamic_universe_state,
)


def _best_stocks_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows: list of (symbol, sector), in rank order (best first)."""
    return pd.DataFrame(
        [{"symbol": sym, "sector": sector, "rank": i + 1} for i, (sym, sector) in enumerate(rows)]
    )


class TestComputeUniverseUpdate:
    def test_additions_capped_at_max_dynamic_symbols(self):
        new_universe, new_state, additions, removals = compute_universe_update(
            static_universe=["AAPL"],
            dynamic_state={},
            today_best_stocks=_best_stocks_df([("NVDA", "tech"), ("AMD", "tech"), ("XOM", "energy")]),
            max_dynamic_symbols=2,
            min_dwell_days=5,
            today="2026-08-02",
        )
        assert additions == ["NVDA", "AMD"]
        assert removals == []
        assert set(new_universe) == {"AAPL", "NVDA", "AMD"}
        assert new_state["NVDA"] == {"sector": "tech", "added_date": "2026-08-02", "consecutive_absent_days": 0}

    def test_static_universe_never_touched_even_if_in_best_stocks(self):
        """A statically-configured symbol appearing in today's list must not
        be duplicated into dynamic_state or double-counted against the cap."""
        new_universe, new_state, additions, removals = compute_universe_update(
            static_universe=["AAPL"],
            dynamic_state={},
            today_best_stocks=_best_stocks_df([("AAPL", "tech"), ("NVDA", "tech")]),
            max_dynamic_symbols=10,
            min_dwell_days=5,
            today="2026-08-02",
        )
        assert additions == ["NVDA"]
        assert "AAPL" not in new_state
        assert new_universe.count("AAPL") == 1

    def test_absence_counter_increments_and_resets(self):
        state = {"NVDA": {"sector": "tech", "added_date": "2026-07-01", "consecutive_absent_days": 2}}
        # NVDA absent today -> increments to 3
        _, new_state, _, removals = compute_universe_update(
            static_universe=["AAPL"],
            dynamic_state=state,
            today_best_stocks=_best_stocks_df([("AMD", "tech")]),
            max_dynamic_symbols=10,
            min_dwell_days=5,
            today="2026-08-02",
        )
        assert new_state["NVDA"]["consecutive_absent_days"] == 3
        assert removals == []

        # NVDA reappears -> resets to 0
        _, new_state2, _, _ = compute_universe_update(
            static_universe=["AAPL"],
            dynamic_state=new_state,
            today_best_stocks=_best_stocks_df([("NVDA", "tech")]),
            max_dynamic_symbols=10,
            min_dwell_days=5,
            today="2026-08-03",
        )
        assert new_state2["NVDA"]["consecutive_absent_days"] == 0

    def test_removal_only_after_dwell_threshold(self):
        state = {"NVDA": {"sector": "tech", "added_date": "2026-07-01", "consecutive_absent_days": 4}}
        # One more absent day reaches min_dwell_days=5 -> removed
        new_universe, new_state, additions, removals = compute_universe_update(
            static_universe=["AAPL"],
            dynamic_state=state,
            today_best_stocks=_best_stocks_df([]),
            max_dynamic_symbols=10,
            min_dwell_days=5,
            today="2026-08-02",
        )
        assert removals == ["NVDA"]
        assert "NVDA" not in new_state
        assert "NVDA" not in new_universe

    def test_removal_does_not_fire_before_dwell_threshold(self):
        state = {"NVDA": {"sector": "tech", "added_date": "2026-07-01", "consecutive_absent_days": 1}}
        _, new_state, _, removals = compute_universe_update(
            static_universe=["AAPL"],
            dynamic_state=state,
            today_best_stocks=_best_stocks_df([]),
            max_dynamic_symbols=10,
            min_dwell_days=5,
            today="2026-08-02",
        )
        assert removals == []
        assert new_state["NVDA"]["consecutive_absent_days"] == 2

    def test_statically_configured_symbol_never_gets_absence_tracking(self):
        """Even if a static symbol is passed inside dynamic_state by mistake,
        the function should not be the thing relied on to protect it — but
        verify static symbols are simply never candidates for addition and
        the static base is always included in new_universe regardless of
        best_stocks content."""
        new_universe, _, additions, _ = compute_universe_update(
            static_universe=["AAPL", "MSFT"],
            dynamic_state={},
            today_best_stocks=_best_stocks_df([]),
            max_dynamic_symbols=10,
            min_dwell_days=5,
            today="2026-08-02",
        )
        assert additions == []
        assert set(new_universe) == {"AAPL", "MSFT"}

    def test_empty_best_stocks_increments_all_dynamic_absences(self):
        state = {
            "NVDA": {"sector": "tech", "added_date": "2026-07-01", "consecutive_absent_days": 0},
            "XOM": {"sector": "energy", "added_date": "2026-07-01", "consecutive_absent_days": 0},
        }
        _, new_state, additions, removals = compute_universe_update(
            static_universe=["AAPL"],
            dynamic_state=state,
            today_best_stocks=pd.DataFrame(),
            max_dynamic_symbols=10,
            min_dwell_days=5,
            today="2026-08-02",
        )
        assert additions == []
        assert removals == []
        assert new_state["NVDA"]["consecutive_absent_days"] == 1
        assert new_state["XOM"]["consecutive_absent_days"] == 1

    def test_no_slots_left_when_at_cap(self):
        state = {
            "NVDA": {"sector": "tech", "added_date": "2026-07-01", "consecutive_absent_days": 0},
            "AMD": {"sector": "tech", "added_date": "2026-07-01", "consecutive_absent_days": 0},
        }
        _, _, additions, _ = compute_universe_update(
            static_universe=["AAPL"],
            dynamic_state=state,
            today_best_stocks=_best_stocks_df([("XOM", "energy")]),
            max_dynamic_symbols=2,
            min_dwell_days=5,
            today="2026-08-02",
        )
        assert additions == []

    def test_rank_order_preserved_when_selecting_additions(self):
        """Given more candidates than slots, the top-ranked ones (first rows,
        best_stocks is assumed rank-ordered) should win the available slots."""
        new_universe, _, additions, _ = compute_universe_update(
            static_universe=[],
            dynamic_state={},
            today_best_stocks=_best_stocks_df([("NVDA", "tech"), ("AMD", "tech"), ("XOM", "energy")]),
            max_dynamic_symbols=1,
            min_dwell_days=5,
            today="2026-08-02",
        )
        assert additions == ["NVDA"]


class TestDynamicUniverseStatePersistence:
    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_dynamic_universe_state(tmp_path / "nope.json") == {}

    def test_save_then_load_round_trips(self, tmp_path):
        path = tmp_path / "sub" / "state.json"
        state = {"NVDA": {"sector": "tech", "added_date": "2026-08-02", "consecutive_absent_days": 0}}
        save_dynamic_universe_state(path, state)
        assert path.exists()
        assert load_dynamic_universe_state(path) == state

    def test_load_corrupt_file_returns_empty_and_does_not_raise(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json")
        assert load_dynamic_universe_state(path) == {}

    def test_load_non_dict_json_returns_empty(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text(json.dumps(["not", "a", "dict"]))
        assert load_dynamic_universe_state(path) == {}


class TestSyncOnce:
    def _make_engine(self, best_stocks_df: pd.DataFrame | None, raise_on_fetch: bool = False):
        engine = MagicMock()
        provider = MagicMock()
        if raise_on_fetch:
            provider.get_best_stocks.side_effect = RuntimeError("API down")
        else:
            provider.get_best_stocks.return_value = (
                best_stocks_df if best_stocks_df is not None else pd.DataFrame()
            )
        engine._data_feed._providers = {"best_stocks": provider}
        return engine

    def test_no_provider_configured_skips(self, tmp_path):
        engine = MagicMock()
        engine._data_feed._providers = {}
        result = sync_once(
            engine,
            state_path=tmp_path / "state.json",
            static_universe=["AAPL"],
            max_dynamic_symbols=10,
            min_dwell_days=5,
        )
        assert result == {"skipped": "no_provider"}
        engine.update_universe.assert_not_called()

    def test_fetch_failure_skips_without_touching_state(self, tmp_path):
        engine = self._make_engine(None, raise_on_fetch=True)
        state_path = tmp_path / "state.json"
        result = sync_once(
            engine,
            state_path=state_path,
            static_universe=["AAPL"],
            max_dynamic_symbols=10,
            min_dwell_days=5,
        )
        assert result == {"skipped": "fetch_failed"}
        engine.update_universe.assert_not_called()
        assert not state_path.exists()

    def test_additions_call_update_universe_and_update_sector_map(self, tmp_path):
        engine = self._make_engine(_best_stocks_df([("NVDA", "technology")]))
        state_path = tmp_path / "state.json"
        result = sync_once(
            engine,
            state_path=state_path,
            static_universe=["AAPL"],
            max_dynamic_symbols=10,
            min_dwell_days=5,
        )
        assert result["additions"] == ["NVDA"]
        engine.update_universe.assert_called_once()
        assert set(engine.update_universe.call_args[0][0]) == {"AAPL", "NVDA"}
        engine.update_sector_map.assert_called_once_with({"NVDA": "technology"})
        assert state_path.exists()

    def test_no_changes_does_not_call_engine_setters(self, tmp_path):
        state_path = tmp_path / "state.json"
        save_dynamic_universe_state(
            state_path,
            {"NVDA": {"sector": "technology", "added_date": "2026-08-01", "consecutive_absent_days": 0}},
        )
        engine = self._make_engine(_best_stocks_df([("NVDA", "technology")]))
        result = sync_once(
            engine,
            state_path=state_path,
            static_universe=["AAPL"],
            max_dynamic_symbols=10,
            min_dwell_days=5,
        )
        assert result["additions"] == []
        assert result["removals"] == []
        engine.update_universe.assert_not_called()
        engine.update_sector_map.assert_not_called()
