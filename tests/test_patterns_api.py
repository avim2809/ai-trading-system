"""Tests for the on-demand pattern-scan REST API (docs/pattern_recognition_plan.md
Phase 3): src/firm/api/routers/patterns.py.

Uses the same FastAPI TestClient conventions as tests/test_api.py. All
scanning here goes through data_source="synthetic" (firm.data.synthetic
.make_synthetic_prices) so the suite needs no real cached market data and
stays fully deterministic — a fixed seed/asof/symbol list always produces
the same confirmed patterns (verified directly against the real
firm.patterns.scanner.scan_symbol before writing these assertions).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from firm.api.app import create_app

# 2026-09-27 (find_confirmation edge-trigger fix -- see
# firm.patterns.confirmation's module docstring for the bug history): a
# genuine breakout confirmed within the last 3 bars is a much rarer event
# on random GBM noise than "still past some level" was, so the original
# 10-symbol universe (DEFAULT_SYMBOLS) produced far fewer real matches at
# this seed/asof under the corrected semantics -- widened to 30 large-cap
# tickers (same seed/asof) to get a real, diverse sample again (17 matches
# at min_score=30, verified directly against the real scan_symbol before
# rewriting the assertions below, exactly as this module's docstring
# already promises -- not asserted from memory).
_SYMBOLS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ",
    "XOM", "PG", "UNH", "HD", "MA", "DIS", "BAC", "KO", "PFE", "CSCO",
    "WMT", "ORCL", "ADBE", "CRM", "NFLX", "INTC", "QCOM", "TXN", "HON", "LIN",
]
_ASOF = "2023-12-31"
_SEED = 42


@pytest.fixture(autouse=True)
def _reset_pattern_cache(tmp_path, monkeypatch):
    """Isolate the router's module-level scan cache *and* history store
    between tests.

    Mirrors test_api.py's `_isolate_registry` fixture for the runs router's
    singletons — same rationale: these are process-local globals, not
    per-request state, so without an explicit reset a match cached by one
    test (or a real data/pattern_scan_history.db write) would leak into the
    next. The history store gets a fresh tmp_path-scoped DB file per test
    via FIRM_DATA_DIR, exactly like a real second firm-api instance would
    get its own separate file.
    """
    import firm.api.routers.patterns as patterns_mod

    monkeypatch.setenv("FIRM_DATA_DIR", str(tmp_path))
    patterns_mod._SCAN_CACHE = []
    patterns_mod._LAST_SCAN = None
    patterns_mod._history_store_instance = None
    yield
    patterns_mod._SCAN_CACHE = []
    patterns_mod._LAST_SCAN = None
    patterns_mod._history_store_instance = None


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


def _trigger(client, **overrides):
    body = {
        "symbols": _SYMBOLS,
        "asof": _ASOF,
        "data_source": "synthetic",
        "min_score": 30.0,
        "seed": _SEED,
    }
    body.update(overrides)
    return client.post("/api/patterns/scan/trigger", json=body)


# ------------------------------------------------------------------
# Empty state (before any trigger)
# ------------------------------------------------------------------


class TestEmptyState:
    def test_scan_empty_before_trigger(self, client):
        r = client.get("/api/patterns/scan")
        assert r.status_code == 200
        assert r.json() == []

    def test_symbol_empty_before_trigger(self, client):
        r = client.get("/api/patterns/AAPL")
        assert r.status_code == 200
        assert r.json() == []

    def test_summary_empty_before_trigger(self, client):
        r = client.get("/api/patterns/summary")
        assert r.status_code == 200
        assert r.json() == {
            "total": 0,
            "by_pattern": {},
            "by_direction": {},
            "last_scan": None,
        }


# ------------------------------------------------------------------
# Trigger validation
# ------------------------------------------------------------------


class TestTriggerValidation:
    def test_rejects_empty_symbols(self, client):
        r = _trigger(client, symbols=[])
        assert r.status_code == 422

    def test_rejects_malformed_asof(self, client):
        r = _trigger(client, asof="not-a-date")
        assert r.status_code == 422

    def test_rejects_bad_data_source(self, client):
        r = _trigger(client, data_source="live")
        assert r.status_code == 422

    def test_defaults_produce_a_scan_with_no_body_overrides(self, client):
        # Every field on PatternScanRequest has a default (data_source
        # defaults to "synthetic", like every other request schema in this
        # API), so an empty POST body must still run a real scan, not 422.
        r = client.post("/api/patterns/scan/trigger", json={})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["scanned"] == 5  # StepRequest-style default symbols
        # 0, not 4 -- re-verified directly against scan_symbol after the
        # 2026-09-27 find_confirmation edge-trigger fix (see
        # firm.patterns.confirmation's module docstring): a genuine
        # breakout confirmed within the last 3 bars is a much rarer event
        # than "still past some level" was, and none of AAPL/MSFT/GOOG/
        # AMZN/META happens to have one as of this exact seed/asof. A
        # correct, if less exciting, result -- not a fixture bug (the
        # 30-symbol universe below still finds plenty; see _SYMBOLS'
        # comment).
        assert data["matches"] == 0


# ------------------------------------------------------------------
# Trigger -> scan round trip (real scan_symbol, synthetic data)
# ------------------------------------------------------------------


class TestTriggerScanRoundTrip:
    def test_trigger_reports_scanned_and_match_counts(self, client):
        r = _trigger(client)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["scanned"] == len(_SYMBOLS)
        # Verified directly against firm.patterns.scanner.scan_symbol for
        # this exact seed/asof/min_score/symbol-list combination (post
        # 2026-09-27 find_confirmation edge-trigger fix -- see _SYMBOLS'
        # comment). Some symbols legitimately contribute more than one
        # match of the same or different pattern families, from different
        # confirmed pivot windows — see docs/pattern_recognition_plan.md
        # §6.2 — so this isn't 1-per-symbol.
        assert data["matches"] == 17
        assert data["last_scan"]["data_source"] == "synthetic"
        assert data["last_scan"]["asof"] == "2023-12-31T00:00:00"
        assert data["last_scan"]["symbols_scanned"] == len(_SYMBOLS)
        assert data["last_scan"]["symbols_missing_data"] == []
        assert data["last_scan"]["symbols_failed"] == []
        assert data["last_scan"]["match_count"] == 17

    def test_scan_populated_after_trigger(self, client):
        _trigger(client)
        r = client.get("/api/patterns/scan")
        assert r.status_code == 200
        matches = r.json()
        assert len(matches) == 17

        # Every match has the full documented shape.
        for m in matches:
            assert m["symbol"] in _SYMBOLS
            assert m["direction"] in {"long", "short"}
            assert m["confirmed"] is True
            assert isinstance(m["pattern"], str) and m["pattern"]
            assert 0.0 <= m["quality_score"] <= 100.0
            assert isinstance(m["score_breakdown"], dict)
            assert "total" in m["score_breakdown"]
            # Most families have >=3 structural pivots, but flag/pennant
            # (continuation.py) is legitimately just a 2-pivot pole pair
            # (flagpole_start, flagpole_end) -- see
            # docs/pattern_recognition_plan.md §6.2.
            assert isinstance(m["pivots"], list) and len(m["pivots"]) >= 2
            for p in m["pivots"]:
                assert set(p) == {"index", "price", "kind"}
                assert p["kind"] in {"peak", "trough"}
            assert isinstance(m["meta"], dict)

        # Best (highest quality_score) first.
        scores = [m["quality_score"] for m in matches]
        assert scores == sorted(scores, reverse=True)

        # CRM's inverse_head_shoulders (~68.4) is the single highest-scoring
        # match in this fixture (post 2026-09-27 find_confirmation
        # edge-trigger fix -- see _SYMBOLS' comment) — confirms real
        # per-symbol scan_symbol output actually reached the cache, not
        # just a placeholder.
        assert matches[0]["symbol"] == "CRM"
        assert matches[0]["pattern"] == "inverse_head_shoulders"
        assert matches[0]["quality_score"] == pytest.approx(68.4, abs=0.5)

    def test_second_trigger_replaces_rather_than_appends(self, client):
        _trigger(client)
        first_count = len(client.get("/api/patterns/scan").json())
        _trigger(client)
        second_count = len(client.get("/api/patterns/scan").json())
        assert first_count == second_count == 17

    def test_trigger_with_stricter_min_score_yields_fewer_cached_matches(self, client):
        _trigger(client, min_score=30.0)
        matches_loose = client.get("/api/patterns/scan").json()
        _trigger(client, min_score=60.0)
        matches = client.get("/api/patterns/scan").json()
        # Verified directly against scan_symbol post 2026-09-27
        # find_confirmation edge-trigger fix: nothing in this fixture
        # clears 80 any more (genuine breakouts confirmed within the last 3
        # bars just aren't that clean-scoring on random GBM noise), so this
        # uses 60 as the "stricter" floor instead -- CRM
        # (inverse_head_shoulders, ~68.4), TSLA (ascending_triangle,
        # ~67.1), and V (head_shoulders_top, ~62.9) clear it.
        assert len(matches) == 3
        assert len(matches) < len(matches_loose)
        assert all(m["quality_score"] >= 60.0 for m in matches)
        assert {m["symbol"] for m in matches} == {"CRM", "TSLA", "V"}

    def test_trigger_cache_missing_symbol_reports_no_data_not_a_crash(self, client, monkeypatch):
        """data_source="cache" loads whatever firm.runtime.load_prices
        returns -- unlike the synthetic path (which happily invents a full
        price series for any symbol name given, so it can never actually
        exercise "no data for this symbol"), a real/cached load can omit a
        requested symbol entirely. Monkeypatches load_prices (exactly where
        trigger_scan's deferred `from firm.runtime import load_prices`
        looks it up) rather than depending on real cached market data."""
        import pandas as pd

        import firm.runtime as runtime_mod

        cached = pd.DataFrame({
            "date": pd.to_datetime(["2023-12-29", "2023-12-30", "2023-12-31"]),
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [190.0, 191.0, 192.0],
            "high": [191.0, 192.0, 193.0],
            "low": [189.0, 190.0, 191.0],
            "close": [190.5, 191.5, 192.5],
            "volume": [1_000_000, 1_100_000, 1_200_000],
            "adj_close": [190.5, 191.5, 192.5],
        })
        monkeypatch.setattr(runtime_mod, "load_prices", lambda settings: cached)

        r = client.post("/api/patterns/scan/trigger", json={
            "symbols": ["AAPL", "NOT_A_REAL_SYMBOL"],
            "asof": "2023-12-31",
            "data_source": "cache",
            "min_score": 30.0,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["last_scan"]["symbols_missing_data"] == ["NOT_A_REAL_SYMBOL"]
        # AAPL has only 3 cached bars (< scan_symbol's 20-bar floor), so it's
        # still "scanned" (groupby'd and handed to scan_symbol) even though
        # it can never confirm a pattern -- 0 matches, not a crash.
        assert data["last_scan"]["symbols_scanned"] == 1
        assert data["matches"] == 0

    def test_trigger_cache_no_data_at_all_returns_503(self, client, monkeypatch):
        import firm.runtime as runtime_mod

        def _boom(settings):
            raise FileNotFoundError("no cached price data found")

        monkeypatch.setattr(runtime_mod, "load_prices", _boom)

        r = client.post("/api/patterns/scan/trigger", json={
            "symbols": ["AAPL"], "asof": "2023-12-31", "data_source": "cache",
        })
        assert r.status_code == 503


# ------------------------------------------------------------------
# GET /patterns/scan filters
# ------------------------------------------------------------------


class TestScanFilters:
    def test_filter_by_direction(self, client):
        _trigger(client)
        r = client.get("/api/patterns/scan?direction=long")
        assert r.status_code == 200
        matches = r.json()
        assert len(matches) == 9  # verified directly against scan_symbol
        assert all(m["direction"] == "long" for m in matches)

        r = client.get("/api/patterns/scan?direction=short")
        matches = r.json()
        assert len(matches) == 8
        assert all(m["direction"] == "short" for m in matches)

    def test_filter_by_pattern(self, client):
        _trigger(client)
        r = client.get("/api/patterns/scan?pattern=inverse_head_shoulders")
        assert r.status_code == 200
        matches = r.json()
        # CRM, AAPL, DIS, TXN, XOM (one each) — verified directly against
        # scan_symbol post 2026-09-27 find_confirmation edge-trigger fix.
        assert len(matches) == 5
        assert all(m["pattern"] == "inverse_head_shoulders" for m in matches)

    def test_filter_by_min_score(self, client):
        _trigger(client)
        r = client.get("/api/patterns/scan?min_score=60")
        assert r.status_code == 200
        matches = r.json()
        assert len(matches) == 3
        assert {m["symbol"] for m in matches} == {"CRM", "TSLA", "V"}

    def test_filters_combine(self, client):
        _trigger(client)
        r = client.get("/api/patterns/scan?direction=short&min_score=60")
        matches = r.json()
        assert all(m["direction"] == "short" and m["quality_score"] >= 60 for m in matches)

    def test_unmatched_filter_returns_empty_not_error(self, client):
        _trigger(client)
        r = client.get("/api/patterns/scan?pattern=cup_handle")
        assert r.status_code == 200
        assert r.json() == []


# ------------------------------------------------------------------
# GET /patterns/summary
# ------------------------------------------------------------------


class TestSummary:
    def test_summary_counts_by_pattern_and_direction(self, client):
        _trigger(client)
        r = client.get("/api/patterns/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 17
        assert sum(data["by_pattern"].values()) == 17
        assert data["by_direction"] == {"long": 9, "short": 8}
        assert data["last_scan"]["match_count"] == 17

    def test_summary_reflects_last_trigger_only(self, client):
        _trigger(client, min_score=60.0)
        data = client.get("/api/patterns/summary").json()
        assert data["total"] == 3


# ------------------------------------------------------------------
# GET /patterns/history (durable, unlike the in-memory /scan cache)
# ------------------------------------------------------------------


class TestHistoryEndpoint:
    def test_history_empty_before_any_trigger(self, client):
        r = client.get("/api/patterns/history")
        assert r.status_code == 200
        assert r.json() == []

    def test_history_populated_after_trigger_with_full_shape(self, client):
        _trigger(client)
        r = client.get("/api/patterns/history")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 17
        row = rows[0]
        # History rows carry everything a cached /scan row does, plus
        # persistence-only fields that prove this hit the real history
        # endpoint (see the route-ordering note on get_symbol_patterns) —
        # not the /{symbol} catch-all falling through to an empty result.
        for key in ("id", "confirm_date", "outcome", "outcome_checked_at", "source", "created_at"):
            assert key in row
        assert row["outcome"] is None
        assert row["source"] == "manual"
        assert isinstance(row["score_breakdown"], dict)
        assert isinstance(row["pivots"], list)

    def test_history_survives_in_memory_cache_reset(self, client):
        """Simulates a process restart's effect on the in-memory cache
        (which starts empty again) without touching the SQLite file, to
        prove history really is a separate, durable surface from
        GET /patterns/scan -- not just reading the same cache twice."""
        import firm.api.routers.patterns as patterns_mod

        _trigger(client)
        assert len(client.get("/api/patterns/scan").json()) == 17
        patterns_mod._SCAN_CACHE = []
        patterns_mod._LAST_SCAN = None

        assert client.get("/api/patterns/scan").json() == []
        assert len(client.get("/api/patterns/history").json()) == 17

    def test_history_accumulates_across_multiple_triggers(self, client):
        _trigger(client, min_score=60.0)  # 3 matches (post 2026-09-27 fix)
        _trigger(client, min_score=30.0)  # 17 matches
        history = client.get("/api/patterns/history").json()
        assert len(history) == 20
        # /scan (the in-memory cache) reflects only the *second* trigger.
        assert len(client.get("/api/patterns/scan").json()) == 17

    def test_history_filters_by_symbol_and_pattern(self, client):
        _trigger(client)
        r = client.get("/api/patterns/history?symbol=crm")
        rows = r.json()
        assert rows and all(row["symbol"] == "CRM" for row in rows)

        r = client.get("/api/patterns/history?pattern=inverse_head_shoulders")
        rows = r.json()
        assert len(rows) == 5
        assert all(row["pattern"] == "inverse_head_shoulders" for row in rows)

    def test_history_route_not_shadowed_by_symbol_catchall(self, client):
        _trigger(client)
        rows = client.get("/api/patterns/history").json()
        # If route registration order regressed, this would silently hit
        # get_symbol_patterns(symbol="history") instead and return [] (no
        # symbol is literally named "HISTORY" in this fixture).
        assert len(rows) == 17
        assert all("id" in row for row in rows)


# ------------------------------------------------------------------
# GET /patterns/{symbol}
# ------------------------------------------------------------------


class TestSymbolEndpoint:
    def test_symbol_with_matches(self, client):
        # 2026-09-27 (find_confirmation edge-trigger fix): NVDA has no
        # confirmed match at all in this fixture any more (see _SYMBOLS'
        # comment) -- DIS is the symbol with multiple confirmed matches
        # (3, from different pattern families) under the corrected
        # semantics, verified directly against scan_symbol.
        _trigger(client)
        r = client.get("/api/patterns/DIS")
        assert r.status_code == 200
        matches = r.json()
        assert len(matches) == 3
        assert all(m["symbol"] == "DIS" for m in matches)
        # Best first: inverse_head_shoulders (~53.1), bull_flag (~35.7),
        # descending_triangle (~33.7).
        assert matches[0]["pattern"] == "inverse_head_shoulders"
        assert matches[1]["pattern"] == "bull_flag"
        assert matches[2]["pattern"] == "descending_triangle"
        assert matches[1]["quality_score"] > matches[2]["quality_score"]

    def test_symbol_is_case_insensitive(self, client):
        _trigger(client)
        r = client.get("/api/patterns/dis")
        assert len(r.json()) == 3

    def test_symbol_with_no_matches_returns_empty_list(self, client):
        _trigger(client)
        r = client.get("/api/patterns/DOES_NOT_EXIST")
        assert r.status_code == 200
        assert r.json() == []

    def test_symbol_scoped_to_that_symbol_only(self, client):
        # TSLA has exactly one confirmed match (ascending_triangle, ~67.1)
        # in this fixture -- verified directly against scan_symbol.
        _trigger(client)
        r = client.get("/api/patterns/TSLA")
        matches = r.json()
        assert matches
        assert all(m["symbol"] == "TSLA" for m in matches)
