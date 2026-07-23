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
    ibkr = MagicMock(spec=IBKRProvider)
    assert fundamentals_available({"fundamentals": ibkr}) is False


def test_filter_drops_fundamental_strategies_without_keys(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
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
    assert "multi_factor" not in filtered
    assert "event_driven" not in filtered
    assert filtered == ["momentum", "trend"]
    assert FUNDAMENTAL_DEPENDENT_STRATEGIES == {"multi_factor", "event_driven"}


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
def test_build_live_providers_ibkr(mock_ibkr):
    providers = build_live_providers("ibkr_paper")
    assert "prices" in providers
    assert "sentiment" in providers
    mock_ibkr.assert_called_once()


def test_build_live_providers_fallback():
    providers = build_live_providers("alpaca_paper")
    assert providers["prices"] is providers["fundamentals"]
