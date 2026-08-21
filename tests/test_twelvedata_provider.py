"""Tests for TwelveDataProvider fundamentals plan-gate behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from firm.data.providers.base import ProviderError
from firm.data.providers.twelvedata import TwelveDataProvider


@pytest.fixture()
def provider():
    TwelveDataProvider._fundamentals_plan_blocked = False
    settings = SimpleNamespace(
        request_timeout_seconds=10,
        max_retries=1,
        require=lambda _k: "unused",
    )
    return TwelveDataProvider(api_key="test-key", settings=settings)


def test_fundamentals_403_plan_limit_fails_fast_for_remaining_symbols(provider):
    provider._client = MagicMock()
    provider._client.get_json.side_effect = ProviderError(
        "https://api.twelvedata.com/statistics returned HTTP 403: "
        '{"code":403,"message":"/statistics is available exclusively with pro plans"}'
    )

    df = provider.get_fundamentals(["AAPL", "MSFT", "GOOG"], "2020-01-01", "2027-01-01")

    assert df.empty
    # Stop after first confirmed plan-level 403 instead of per-symbol spam.
    assert provider._client.get_json.call_count == 1
    assert TwelveDataProvider._fundamentals_plan_blocked is True


def test_fundamentals_short_circuits_when_plan_already_blocked(provider):
    TwelveDataProvider._fundamentals_plan_blocked = True
    provider._client = MagicMock()

    df = provider.get_fundamentals(["AAPL", "MSFT"], "2020-01-01", "2027-01-01")

    assert df.empty
    provider._client.get_json.assert_not_called()
