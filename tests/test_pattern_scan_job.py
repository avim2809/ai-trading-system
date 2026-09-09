"""Tests for the optional, off-by-default daily pattern scan job
(docs/pattern_recognition_plan.md §2a): src/firm/live/pattern_scan_job.py.

Uses data_source="synthetic" (deterministic, no real cached market data
needed) for the scan-and-persist path, and monkeypatches
firm.runtime.load_prices for the outcome-tracking path (which is specifically
about re-checking against *fresh* data, so synthetic-at-trigger-time doesn't
apply there).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from firm.live.pattern_scan_history import PatternScanHistoryStore
from firm.live.pattern_scan_job import PatternScanJob, pattern_scan_enabled


# ---------------------------------------------------------------------------
# Env-var gating
# ---------------------------------------------------------------------------


class TestPatternScanEnabled:
    def test_defaults_off(self, monkeypatch):
        monkeypatch.delenv("FIRM_ENABLE_PATTERN_SCAN", raising=False)
        assert pattern_scan_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
    def test_truthy_values_enable(self, monkeypatch, value):
        monkeypatch.setenv("FIRM_ENABLE_PATTERN_SCAN", value)
        assert pattern_scan_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "garbage"])
    def test_other_values_stay_off(self, monkeypatch, value):
        monkeypatch.setenv("FIRM_ENABLE_PATTERN_SCAN", value)
        assert pattern_scan_enabled() is False


# ---------------------------------------------------------------------------
# App wiring: the job must never start unless explicitly enabled
# ---------------------------------------------------------------------------


class TestAppWiring:
    def test_job_absent_by_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FIRM_ENABLE_PATTERN_SCAN", raising=False)
        monkeypatch.setenv("FIRM_DATA_DIR", str(tmp_path))
        from fastapi.testclient import TestClient

        from firm.api.app import create_app

        app = create_app()
        with TestClient(app) as client:
            assert app.state.pattern_scan_job is None
            assert client.get("/api/health").status_code == 200

    def test_job_starts_when_enabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FIRM_ENABLE_PATTERN_SCAN", "1")
        monkeypatch.setenv("FIRM_DATA_DIR", str(tmp_path))
        from fastapi.testclient import TestClient

        from firm.api.app import create_app

        app = create_app()
        with TestClient(app) as client:
            job = app.state.pattern_scan_job
            assert job is not None
            assert job._scheduler.running
            assert client.get("/api/health").status_code == 200
        # Context-manager exit runs the lifespan shutdown path, which stops
        # the job's scheduler -- assert it actually did, not just that stop()
        # was callable.
        assert not job._scheduler.running


# ---------------------------------------------------------------------------
# run_once(): scan + persist (synthetic data, deterministic)
# ---------------------------------------------------------------------------


class TestRunOnce:
    def _job(self, tmp_path, **kwargs):
        store = PatternScanHistoryStore(db_path=str(tmp_path / "history.db"))
        kwargs.setdefault("data_source", "synthetic")
        return PatternScanJob(symbols=["AAPL", "MSFT", "NVDA"], history_store=store, **kwargs), store

    def test_run_once_persists_matches_with_scheduled_source(self, tmp_path, monkeypatch):
        # _check_pending_outcomes() always re-checks against real cached
        # data (outcome tracking is about what actually happened, not
        # another synthetic draw) regardless of the job's own scan
        # data_source -- isolate that step here so this test is purely
        # about the scan+persist path, not coupled to whatever this repo's
        # real data/cache happens to contain.
        import firm.runtime as runtime_mod

        def _no_cache(settings):
            raise FileNotFoundError("isolated from real data/cache for this test")

        monkeypatch.setattr(runtime_mod, "load_prices", _no_cache)
        job, store = self._job(tmp_path)
        result = job.run_once(asof="2023-12-31")
        assert result["scanned"] == 3
        assert result["matches"] > 0

        history = store.list_history(limit=1000)
        assert len(history) == result["matches"]
        assert all(row["source"] == "scheduled" for row in history)
        assert all(row["outcome"] is None for row in history)
        assert all(row["confirm_date"] is not None for row in history)

    def test_run_once_handles_missing_cache_data_gracefully(self, tmp_path, monkeypatch):
        """data_source='cache' with no cached data must not raise -- mirrors
        trigger_scan's 503 case, but this runs unattended so it just logs
        and returns a zero result instead of erroring."""
        import firm.runtime as runtime_mod

        def _boom(settings):
            raise FileNotFoundError("no cached price data found")

        monkeypatch.setattr(runtime_mod, "load_prices", _boom)
        job, _store = self._job(tmp_path, data_source="cache")
        result = job.run_once(asof="2023-12-31")
        assert result == {"scanned": 0, "matches": 0, "outcomes_checked": 0}


# ---------------------------------------------------------------------------
# _check_pending_outcomes(): triple-barrier re-check against fresh data
# ---------------------------------------------------------------------------


def _fresh_prices(symbol: str, closes: list[float], start_date: str) -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=len(closes), freq="D")
    return pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "open": closes,
        "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes],
        "close": closes,
        "adj_close": closes,
        "volume": [1_000_000.0] * len(closes),
    })


class TestCheckPendingOutcomes:
    def _pending_row(self, store, *, confirm_date, direction="long", entry=100.0, stop=90.0, target=120.0):
        row = {
            "symbol": "AAPL", "asof": "2024-01-01T00:00:00", "pattern": "cup_handle",
            "direction": direction, "confirmed": True, "confirm_index": 10,
            "entry": entry, "stop": stop, "target": target, "fit_quality": 1.0,
            "geometry_tolerance_used": 1.0, "volume_ratio": 2.0, "duration_bars": 20,
            "follow_through_atr": 2.0, "risk_reward": 2.0, "quality_score": 90.0,
            "score_breakdown": {"total": 90.0}, "pivots": [], "meta": {},
        }
        store.insert_matches([row], confirm_dates=[confirm_date], source="scheduled")

    def test_resolves_target_hit(self, tmp_path, monkeypatch):
        import firm.runtime as runtime_mod

        store = PatternScanHistoryStore(db_path=str(tmp_path / "h.db"))
        self._pending_row(store, confirm_date="2024-01-01", direction="long", stop=90.0, target=120.0)

        # Fresh data starting the day after confirm_date, closing well above target.
        fresh = _fresh_prices("AAPL", [105.0, 110.0, 125.0], "2024-01-02")
        monkeypatch.setattr(runtime_mod, "load_prices", lambda settings: fresh)

        job, _ = PatternScanJob(symbols=["AAPL"], history_store=store, data_source="synthetic"), store
        checked = job._check_pending_outcomes()
        assert checked == 1
        row = store.list_history(limit=1)[0]
        assert row["outcome"] == "target_hit"

    def test_resolves_stop_hit(self, tmp_path, monkeypatch):
        import firm.runtime as runtime_mod

        store = PatternScanHistoryStore(db_path=str(tmp_path / "h.db"))
        self._pending_row(store, confirm_date="2024-01-01", direction="long", stop=90.0, target=120.0)

        fresh = _fresh_prices("AAPL", [95.0, 88.0, 80.0], "2024-01-02")
        monkeypatch.setattr(runtime_mod, "load_prices", lambda settings: fresh)

        job = PatternScanJob(symbols=["AAPL"], history_store=store, data_source="synthetic")
        checked = job._check_pending_outcomes()
        assert checked == 1
        row = store.list_history(limit=1)[0]
        assert row["outcome"] == "stop_hit"

    def test_leaves_recent_unresolved_row_pending(self, tmp_path, monkeypatch):
        import firm.runtime as runtime_mod

        store = PatternScanHistoryStore(db_path=str(tmp_path / "h.db"))
        recent = (datetime.now() - timedelta(days=2)).date().isoformat()
        self._pending_row(store, confirm_date=recent, direction="long", stop=90.0, target=120.0)

        # Price stays between stop and target -- neither barrier touched.
        fresh = _fresh_prices("AAPL", [100.0, 101.0, 99.0], recent)
        monkeypatch.setattr(runtime_mod, "load_prices", lambda settings: fresh)

        job = PatternScanJob(symbols=["AAPL"], history_store=store, data_source="synthetic")
        checked = job._check_pending_outcomes()
        assert checked == 0
        row = store.list_history(limit=1)[0]
        assert row["outcome"] is None

    def test_old_unresolved_row_times_out(self, tmp_path, monkeypatch):
        import firm.runtime as runtime_mod

        store = PatternScanHistoryStore(db_path=str(tmp_path / "h.db"))
        old_date = (datetime.now() - timedelta(days=45)).date().isoformat()
        self._pending_row(store, confirm_date=old_date, direction="long", stop=90.0, target=120.0)

        fresh = _fresh_prices("AAPL", [100.0, 101.0, 99.0], old_date)
        monkeypatch.setattr(runtime_mod, "load_prices", lambda settings: fresh)

        job = PatternScanJob(symbols=["AAPL"], history_store=store, data_source="synthetic")
        checked = job._check_pending_outcomes()
        assert checked == 1
        row = store.list_history(limit=1)[0]
        assert row["outcome"] == "timeout"

    def test_no_pending_rows_is_a_cheap_noop(self, tmp_path):
        store = PatternScanHistoryStore(db_path=str(tmp_path / "h.db"))
        job = PatternScanJob(symbols=["AAPL"], history_store=store, data_source="synthetic")
        assert job._check_pending_outcomes() == 0
