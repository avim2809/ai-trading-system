"""Tests for live provider / strategy wiring helpers."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from firm.data.providers.ibkr import IBKRProvider
from firm.live.provider_utils import (
    FUNDAMENTAL_DEPENDENT_STRATEGIES,
    build_live_providers,
    filter_strategies_for_providers,
    fundamentals_available,
    load_live_yaml_defaults,
    merge_live_yaml_defaults,
)


def test_fundamentals_available_with_fmp_key(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    assert fundamentals_available() is True


def test_fundamentals_available_ibkr_only(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.setattr(
        "firm.data.fundamentals_cache.load_cached_fundamentals_df",
        lambda: None,
    )
    ibkr = MagicMock(spec=IBKRProvider)
    # SEC EDGAR is keyless — fundamentals remain available for strategy wiring.
    assert fundamentals_available({"fundamentals": ibkr}) is True


def test_filter_keeps_fundamental_strategies_with_edgar_fallback(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setattr(
        "firm.data.fundamentals_cache.load_cached_fundamentals_df",
        lambda: None,
    )
    ibkr = MagicMock(spec=IBKRProvider)

    strategies = [
        "momentum",
        "multi_factor",
        "event_driven",
        "trend",
    ]
    filtered = filter_strategies_for_providers(
        strategies,
        {"fundamentals": ibkr, "prices": ibkr},
        logger=logging.getLogger("test"),
    )
    assert filtered == strategies


def test_load_live_yaml_includes_stat_arb_params():
    data = load_live_yaml_defaults()
    assert "strategy_params" in data
    assert "stat_arb" in data["strategy_params"]
    assert data["strategy_params"]["stat_arb"].get("require_cointegration") is True


def test_merge_live_yaml_defaults_fills_strategy_params():
    merged = merge_live_yaml_defaults({})
    assert "stat_arb" in merged.get("strategy_params", {})


def test_resolve_live_startup_uses_yaml_universe_and_risk():
    from firm.live.provider_utils import resolve_live_startup

    resolved = resolve_live_startup()
    assert resolved["broker"] == "ibkr_paper"
    assert len(resolved["symbols"]) >= 20
    assert "stat_arb" in (resolved["strategies"] or [])
    assert resolved["engine_config"].get("kill_switch_drawdown") == 0.08
    assert resolved["engine_config"].get("max_position_pct") == 0.05
    assert resolved["strategy_params"]["stat_arb"]["require_cointegration"] is True


@patch("firm.data.providers.ibkr.IBKRProvider")
def test_build_live_providers_ibkr(mock_ibkr, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-fmp")
    monkeypatch.setenv("MASSIVE_API_KEY", "test-massive")
    from firm.data.providers.fallback import FallbackProvider

    providers = build_live_providers("ibkr_paper")
    assert "prices" in providers
    assert "sentiment" in providers
    assert isinstance(providers["fundamentals"], FallbackProvider)
    mock_ibkr.assert_called_once()


@patch("firm.data.providers.ibkr.IBKRProvider")
def test_build_live_providers_ibkr_no_fundamentals_without_keys(mock_ibkr, monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    providers = build_live_providers("ibkr_paper")
    assert "fundamentals" not in providers


def test_build_live_providers_fallback():
    providers = build_live_providers("alpaca_paper")
    assert providers["prices"] is providers["fundamentals"]
