"""Tests for firm.live.news_guard — deterministic (offline / provided events)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from firm.live.news_guard import (
    Event,
    decide,
    evaluate,
    instrument_currencies,
    is_crypto,
    load_events,
    load_from_csv,
)

# A single high-impact USD event (NFP-style) at 13:30 UTC.
NFP = Event(
    title="US Non-Farm Payrolls",
    currency="USD",
    impact="High",
    when=datetime(2026, 7, 2, 13, 30, tzinfo=timezone.utc),
)


class TestDecision:
    def test_blocks_inside_window(self):
        at = datetime(2026, 7, 2, 13, 25, tzinfo=timezone.utc)  # 5 min before
        res = decide("SPY", at, [NFP])
        assert res["decision"] == "block"
        assert res["blocking_event"]["currency"] == "USD"

    def test_approves_outside_window(self):
        at = datetime(2026, 7, 2, 10, 0, tzinfo=timezone.utc)  # hours before
        res = decide("SPY", at, [NFP])
        assert res["decision"] == "approve"
        assert res["next_event"] is not None

    def test_after_window_approves(self):
        at = datetime(2026, 7, 2, 14, 0, tzinfo=timezone.utc)  # 30 min after
        res = decide("SPY", at, [NFP])
        assert res["decision"] == "approve"

    def test_unrelated_currency_not_blocked(self):
        # A pure EUR instrument should ignore a USD-only event.
        at = datetime(2026, 7, 2, 13, 30, tzinfo=timezone.utc)
        res = decide("DAX", at, [NFP])
        assert res["decision"] == "approve"


class TestInstrumentMapping:
    def test_fx_pair(self):
        assert instrument_currencies("EURUSD") == {"EUR", "USD"}

    def test_index_maps_to_usd(self):
        assert instrument_currencies("SPY") == {"USD"}

    def test_unknown_falls_back_usd(self):
        assert instrument_currencies("ZZZZZZ") == {"USD"}

    def test_crypto_detection(self):
        assert is_crypto("BTCUSD")
        assert not is_crypto("SPY")


class TestBundledCsv:
    def test_bundled_csv_loads(self):
        events = load_from_csv()
        assert events
        assert all(isinstance(e, Event) for e in events)

    def test_evaluate_offline_deterministic(self):
        res = evaluate("SPY", "2026-07-29T18:05:00Z", offline=True)
        assert res["decision"] == "block"  # inside FOMC window from bundled CSV


class TestLoadEventsSourceDispatch:
    """source="investing" ladder: investing -> forexfactory -> bundled CSV.
    Default source="forexfactory" is unchanged (forexfactory -> bundled CSV)."""

    def test_default_source_is_forexfactory_unchanged(self):
        with patch(
            "firm.live.news_guard.load_from_forexfactory", return_value=[NFP],
        ) as mock_ff, patch("firm.live.news_guard.load_from_investing") as mock_investing:
            events, source = load_events()
        assert source == "forexfactory"
        assert events == [NFP]
        mock_ff.assert_called_once()
        mock_investing.assert_not_called()

    def test_investing_source_used_when_it_succeeds(self):
        with patch("firm.live.news_guard.load_from_investing", return_value=[NFP]), \
             patch("firm.live.news_guard.load_from_forexfactory") as mock_ff:
            events, source = load_events(source="investing")
        assert source == "investing"
        assert events == [NFP]
        mock_ff.assert_not_called()

    def test_investing_failure_falls_back_to_forexfactory(self):
        with patch(
            "firm.live.news_guard.load_from_investing", side_effect=RuntimeError("blocked"),
        ), patch("firm.live.news_guard.load_from_forexfactory", return_value=[NFP]):
            events, source = load_events(source="investing")
        assert source == "forexfactory"
        assert events == [NFP]

    def test_investing_empty_falls_back_to_forexfactory(self):
        with patch("firm.live.news_guard.load_from_investing", return_value=[]), \
             patch("firm.live.news_guard.load_from_forexfactory", return_value=[NFP]):
            events, source = load_events(source="investing")
        assert source == "forexfactory"
        assert events == [NFP]

    def test_investing_and_forexfactory_both_fail_falls_back_to_csv(self):
        with patch(
            "firm.live.news_guard.load_from_investing", side_effect=RuntimeError("blocked"),
        ), patch(
            "firm.live.news_guard.load_from_forexfactory", side_effect=RuntimeError("network"),
        ):
            events, source = load_events(source="investing")
        assert source == "bundled-csv"
        assert events  # the bundled CSV always has something

    def test_offline_skips_every_live_source_including_investing(self):
        with patch("firm.live.news_guard.load_from_investing") as mock_investing, \
             patch("firm.live.news_guard.load_from_forexfactory") as mock_ff:
            events, source = load_events(offline=True, source="investing")
        assert source == "bundled-csv"
        mock_investing.assert_not_called()
        mock_ff.assert_not_called()
