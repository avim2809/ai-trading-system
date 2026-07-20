"""Tests for IBKRBroker's price resolution.

Regression coverage for a real incident found verifying a live IBKR paper
run: outside trading hours (or without a live data subscription), IBKR's
delayed ticker has bid/ask as the empty sentinel -1 (midpoint() -> NaN) and
``last`` defaults to 0.0 — a real float, not NaN, so a NaN-only check
silently accepted it as a fake zero price and every order got skipped with
"No price for X".
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("ib_async")

from firm.brokers.ibkr import IBKRBroker


def _ticker(midpoint=float("nan"), last=float("nan"), close=float("nan")):
    t = SimpleNamespace(last=last, close=close)
    t.midpoint = lambda: midpoint
    return t


class TestResolvePrice:
    def test_uses_valid_midpoint(self):
        assert IBKRBroker._resolve_price(_ticker(midpoint=100.5), "AAPL") == 100.5

    def test_falls_back_to_last_when_midpoint_nan(self):
        assert IBKRBroker._resolve_price(_ticker(last=101.0), "AAPL") == 101.0

    def test_ignores_zero_last_sentinel_falls_back_to_close(self):
        # The exact bug: last=0.0 is a real float (not NaN) that must not be
        # mistaken for a genuine last-trade price.
        ticker = _ticker(last=0.0, close=333.74)
        assert IBKRBroker._resolve_price(ticker, "AAPL") == 333.74

    def test_all_empty_returns_zero_with_warning(self, caplog):
        ticker = _ticker(last=0.0, close=float("nan"))
        with caplog.at_level("WARNING"):
            price = IBKRBroker._resolve_price(ticker, "AAPL")
        assert price == 0.0
        assert any("No midpoint, last, or close" in r.message for r in caplog.records)

    def test_negative_or_zero_midpoint_is_rejected(self):
        ticker = _ticker(midpoint=0.0, last=0.0, close=50.0)
        assert IBKRBroker._resolve_price(ticker, "AAPL") == 50.0

    def test_get_current_prices_uses_resolve_price_per_symbol(self):
        broker = IBKRBroker(host="127.0.0.1", port=4002, client_id=1)
        fake_ib = SimpleNamespace(
            isConnected=lambda: True,
            qualifyContracts=lambda *cs: None,
            reqTickers=lambda *cs: [_ticker(last=0.0, close=200.0), _ticker(midpoint=50.0)],
        )
        broker._ib = fake_ib
        with patch("firm.brokers.ibkr.Stock", side_effect=lambda sym, *_a: SimpleNamespace(symbol=sym)):
            prices = broker.get_current_prices(["AAPL", "MSFT"])
        assert prices == {"AAPL": 200.0, "MSFT": 50.0}
