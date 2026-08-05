"""Tests for live provider / strategy wiring helpers."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from firm.data.providers.ibkr import IBKRProvider
from firm.live.provider_utils import (
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
    # config/live.yaml's costs: block must reach ExecutionAgent's flat
    # commission_pct/slippage_pct/spread_pct keys, not be silently dropped.
    assert resolved["engine_config"].get("commission_pct") == 0.0005
    assert resolved["engine_config"].get("slippage_pct") == 0.0005
    assert resolved["engine_config"].get("spread_pct") == 0.0002
    assert resolved["engine_config"].get("market_impact_coefficient") == 0.005
    # Broker failover: config/live.yaml's broker_disconnect_alert_threshold
    # must reach the engine config (see docs/PROJECT_CONTEXT.md "Broker &
    # host failover").
    assert resolved["engine_config"].get("broker_disconnect_alert_threshold") == 3
    # Generic per-strategy circuit breaker config must reach the engine
    # config too (disabled by default — see docs/portfolio_construction_diagnosis.md).
    cb = resolved["engine_config"].get("strategy_circuit_breaker")
    assert cb is not None
    assert cb.get("enabled") is False
    rw = resolved["engine_config"].get("strategy_regime_weights")
    assert rw is not None
    assert rw.get("enabled") is False


def test_resolve_live_startup_costs_from_yaml_are_flattened():
    from unittest.mock import patch as _patch

    from firm.live.provider_utils import resolve_live_startup

    fake_yaml = {
        "risk": {"max_position_pct": 0.05},
        "costs": {
            "commission_pct": 0.0012, "slippage_pct": 0.0007, "spread_pct": 0.0003,
            "market_impact_coefficient": 0.008,
        },
    }
    with _patch("firm.live.provider_utils.load_live_yaml_defaults", return_value=fake_yaml):
        resolved = resolve_live_startup()
    assert resolved["engine_config"]["commission_pct"] == 0.0012
    assert resolved["engine_config"]["slippage_pct"] == 0.0007
    assert resolved["engine_config"]["spread_pct"] == 0.0003
    assert resolved["engine_config"]["market_impact_coefficient"] == 0.008


def test_resolve_live_startup_costs_absent_from_yaml_leaves_execution_agent_defaults():
    from unittest.mock import patch as _patch

    from firm.live.provider_utils import resolve_live_startup

    fake_yaml = {"risk": {"max_position_pct": 0.05}}
    with _patch("firm.live.provider_utils.load_live_yaml_defaults", return_value=fake_yaml):
        resolved = resolve_live_startup()
    assert "commission_pct" not in resolved["engine_config"]
    assert "slippage_pct" not in resolved["engine_config"]
    assert "spread_pct" not in resolved["engine_config"]
    assert "market_impact_coefficient" not in resolved["engine_config"]


def test_resolve_live_startup_merges_persisted_dynamic_universe_when_enabled(tmp_path):
    from unittest.mock import patch as _patch

    from firm.live.dynamic_universe_state import save_dynamic_universe_state
    from firm.live.provider_utils import resolve_live_startup

    state_path = tmp_path / "dyn_state.json"
    save_dynamic_universe_state(
        state_path,
        {"NVDA": {"sector": "technology", "added_date": "2026-08-01", "consecutive_absent_days": 0}},
    )
    fake_yaml = {
        "universe": {"symbols": ["AAPL", "MSFT"]},
        "danelfin_dynamic_universe": {"enabled": True, "state_path": str(state_path)},
    }
    with _patch("firm.live.provider_utils.load_live_yaml_defaults", return_value=fake_yaml):
        resolved = resolve_live_startup()
    assert set(resolved["symbols"]) == {"AAPL", "MSFT", "NVDA"}
    assert resolved["engine_config"]["sector_map"] == {"NVDA": "technology"}


def test_resolve_live_startup_ignores_dynamic_state_when_disabled(tmp_path):
    from unittest.mock import patch as _patch

    from firm.live.dynamic_universe_state import save_dynamic_universe_state
    from firm.live.provider_utils import resolve_live_startup

    state_path = tmp_path / "dyn_state.json"
    save_dynamic_universe_state(
        state_path,
        {"NVDA": {"sector": "technology", "added_date": "2026-08-01", "consecutive_absent_days": 0}},
    )
    fake_yaml = {
        "universe": {"symbols": ["AAPL", "MSFT"]},
        "danelfin_dynamic_universe": {"enabled": False, "state_path": str(state_path)},
    }
    with _patch("firm.live.provider_utils.load_live_yaml_defaults", return_value=fake_yaml):
        resolved = resolve_live_startup()
    assert set(resolved["symbols"]) == {"AAPL", "MSFT"}
    assert "sector_map" not in resolved["engine_config"]


def test_resolve_live_startup_explicit_symbols_override_skips_dynamic_merge(tmp_path):
    from unittest.mock import patch as _patch

    from firm.live.dynamic_universe_state import save_dynamic_universe_state
    from firm.live.provider_utils import resolve_live_startup

    state_path = tmp_path / "dyn_state.json"
    save_dynamic_universe_state(
        state_path,
        {"NVDA": {"sector": "technology", "added_date": "2026-08-01", "consecutive_absent_days": 0}},
    )
    fake_yaml = {
        "universe": {"symbols": ["AAPL", "MSFT"]},
        "danelfin_dynamic_universe": {"enabled": True, "state_path": str(state_path)},
    }
    with _patch("firm.live.provider_utils.load_live_yaml_defaults", return_value=fake_yaml):
        resolved = resolve_live_startup(symbols=["TSLA"])
    assert resolved["symbols"] == ["TSLA"]


@patch("firm.data.providers.ibkr.IBKRProvider")
def test_build_live_providers_ibkr(mock_ibkr, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-fmp")
    monkeypatch.setenv("MASSIVE_API_KEY", "test-massive")
    from firm.data.providers.fallback import FallbackProvider

    providers = build_live_providers("ibkr_paper")
    assert "prices" in providers
    assert "sentiment" in providers
    assert isinstance(providers["sentiment"], FallbackProvider)
    assert isinstance(providers["fundamentals"], FallbackProvider)
    assert providers["sentiment"] is providers["fundamentals"]
    mock_ibkr.assert_called_once()


@patch("firm.data.providers.ibkr.IBKRProvider")
def test_build_live_providers_ibkr_no_fundamentals_without_keys(mock_ibkr, monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    providers = build_live_providers("ibkr_paper")
    assert "fundamentals" not in providers
    assert "sentiment" not in providers


@patch("firm.data.providers.ibkr.IBKRProvider")
def test_build_live_providers_ibkr_sentiment_without_fundamentals_keys(mock_ibkr, monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("MASSIVE_API_KEY", "test-massive")
    from firm.data.providers.fallback import FallbackProvider

    providers = build_live_providers("ibkr_paper")
    assert isinstance(providers["sentiment"], FallbackProvider)
    assert "fundamentals" in providers


@patch("firm.data.providers.alpaca.AlpacaProvider")
def test_build_live_providers_alpaca(mock_alpaca, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-alpaca-secret")
    monkeypatch.setenv("FMP_API_KEY", "test-fmp")
    monkeypatch.setenv("MASSIVE_API_KEY", "test-massive")
    from firm.data.providers.fallback import FallbackProvider

    providers = build_live_providers("alpaca_paper")
    assert "prices" in providers
    assert "sentiment" in providers
    assert isinstance(providers["sentiment"], FallbackProvider)
    assert isinstance(providers["fundamentals"], FallbackProvider)
    assert providers["sentiment"] is providers["fundamentals"]
    mock_alpaca.assert_called_once()


def test_build_live_providers_fallback():
    # A broker with no dedicated data-provider branch (unlike ibkr*/alpaca*)
    # falls through to one shared FallbackProvider for every capability.
    providers = build_live_providers("some_future_broker")
    assert providers["prices"] is providers["fundamentals"]
