"""Tests for IBKRProvider._get_prices's fail-fast handling of a broken
IBKR HMDS (historical-data) farm.

Regression coverage for a real incident (2026-09-23): a broken `ushmds`
farm made every one of 25 symbols' reqHistoricalData calls in the
per-symbol loop run out its full timeout in turn — ~8 minutes proving the
same IBKR-side outage 25 times — before the loop gave up. IBKRBroker now
tracks farm health via errorEvent (see test_ibkr_broker.py); this provider
must check that status and bail immediately instead of retrying.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("ib_async")

from firm.data.providers.ibkr import IBKRProvider
from firm.data.schemas import PRICE_COLS


class _FakeBroker:
    def __init__(self, broken: bool):
        self._broken = broken

    def is_historical_data_farm_broken(self) -> bool:
        return self._broken


class TestFailFastOnBrokenFarm:
    def test_skips_entire_loop_when_farm_already_known_broken(self):
        provider = IBKRProvider(shared_broker=_FakeBroker(broken=True))
        fake_ib = SimpleNamespace(
            reqMarketDataType=lambda *a, **k: None,
            qualifyContracts=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("qualifyContracts should never be called when farm is known broken")
            ),
        )
        result = provider._get_prices(fake_ib, ["AAPL", "MSFT", "SPY"], "2026-01-01", "2026-01-31")
        assert result.empty
        assert list(result.columns) == PRICE_COLS

    def test_proceeds_normally_when_farm_is_healthy(self):
        provider = IBKRProvider(shared_broker=_FakeBroker(broken=False))
        calls: list[str] = []

        def _qualify(contract):
            calls.append(contract.symbol)

        fake_ib = SimpleNamespace(
            reqMarketDataType=lambda *a, **k: None,
            qualifyContracts=_qualify,
            reqHistoricalData=lambda *a, **k: [],
        )
        with patch(
            "firm.data.providers.ibkr.Stock",
            side_effect=lambda sym, *_a, **_k: SimpleNamespace(symbol=sym),
        ):
            result = provider._get_prices(fake_ib, ["AAPL"], "2026-01-01", "2026-01-31")
        assert calls == ["AAPL"]
        assert result.empty  # no bars returned by the fake, but the call was attempted

    def test_no_shared_broker_means_no_farm_check_at_all(self):
        """A private (non-shared) IB() connection has no farm-status
        tracking — _get_prices must not blow up trying to call a method
        that doesn't exist on a bare connection."""
        provider = IBKRProvider(shared_broker=None)
        calls: list[str] = []
        fake_ib = SimpleNamespace(
            reqMarketDataType=lambda *a, **k: None,
            qualifyContracts=lambda contract: calls.append(contract.symbol),
            reqHistoricalData=lambda *a, **k: [],
        )
        with patch(
            "firm.data.providers.ibkr.Stock",
            side_effect=lambda sym, *_a, **_k: SimpleNamespace(symbol=sym),
        ):
            provider._get_prices(fake_ib, ["AAPL"], "2026-01-01", "2026-01-31")
        assert calls == ["AAPL"]

    def test_mid_loop_break_when_farm_reported_broken_after_first_failure(self):
        broker = _FakeBroker(broken=False)
        provider = IBKRProvider(shared_broker=broker)
        attempted: list[str] = []

        def _qualify(contract):
            attempted.append(contract.symbol)
            # Simulate the farm going broken exactly when the first
            # request fails (mirrors the real errorEvent firing mid-request).
            broker._broken = True
            raise ConnectionError("simulated HMDS timeout")

        fake_ib = SimpleNamespace(
            reqMarketDataType=lambda *a, **k: None,
            qualifyContracts=_qualify,
        )
        with patch(
            "firm.data.providers.ibkr.Stock",
            side_effect=lambda sym, *_a, **_k: SimpleNamespace(symbol=sym),
        ):
            result = provider._get_prices(
                fake_ib, ["AAPL", "MSFT", "SPY"], "2026-01-01", "2026-01-31"
            )
        assert attempted == ["AAPL"]  # bailed before trying MSFT/SPY
        assert result.empty
