"""Helpers for wiring live data providers to the strategy pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_LIVE_YAML = Path(__file__).resolve().parents[3] / "config" / "live.yaml"

# Strategies that need quarterly fundamentals (PE, EPS, ROE, …) to produce
# meaningful signals.  On an IBKR-only stack without FMP/Massive keys they
# silently degrade to price-only proxies or emit nothing.
FUNDAMENTAL_DEPENDENT_STRATEGIES: frozenset[str] = frozenset(
    {"multi_factor", "event_driven"}
)


def load_live_yaml_defaults() -> dict[str, Any]:
    """Return parsed ``config/live.yaml`` (empty dict if missing)."""
    if not _LIVE_YAML.exists():
        return {}
    with open(_LIVE_YAML) as f:
        return yaml.safe_load(f) or {}


def merge_live_yaml_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Fill missing live-engine keys from ``config/live.yaml`` when present."""
    resolved = resolve_live_startup(
        broker=config.get("broker"),
        symbols=config.get("symbols"),
        strategies=config.get("strategies"),
        strategy_params=config.get("strategy_params"),
        auto_approve=config.get("auto_approve"),
        initial_capital=config.get("initial_capital"),
        risk_overrides=config.get("risk_overrides"),
    )
    merged = dict(config)
    for key in ("strategies", "strategy_params", "symbols"):
        if not merged.get(key) and resolved.get(key):
            merged[key] = resolved[key]
    return merged


def resolve_live_startup(
    *,
    broker: str | None = None,
    symbols: list[str] | None = None,
    strategies: list[str] | None = None,
    strategy_params: dict[str, Any] | None = None,
    auto_approve: list[str] | None = None,
    initial_capital: float | None = None,
    risk_overrides: dict[str, Any] | None = None,
    schedule: str | None = None,
    approval_mode: str | None = None,
    kill_switch_drawdown: float | None = None,
    max_daily_trades: int | None = None,
    max_daily_turnover: float | None = None,
) -> dict[str, Any]:
    """Merge API/CLI overrides with ``config/live.yaml`` for systemd deployments.

    The ``ai-trading.service`` unit runs ``firm-api`` (not ``run_live_trading.py``),
    so ``config/live.yaml`` is the canonical source for universe, risk, strategies,
    and per-strategy params unless the caller overrides them explicitly.
    """
    yaml_cfg = load_live_yaml_defaults()
    strategies_block = yaml_cfg.get("strategies") or {}

    broker = broker or yaml_cfg.get("broker", "ibkr_paper")
    schedule = schedule or yaml_cfg.get("schedule", "market_open")
    approval_mode = approval_mode or yaml_cfg.get("approval_mode", "semi_auto")

    if not symbols:
        symbols = list((yaml_cfg.get("universe") or {}).get("symbols") or [])
    if not symbols:
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    if not strategies:
        strategies = list(strategies_block.get("enabled") or []) or None

    if not strategy_params:
        strategy_params = dict(yaml_cfg.get("strategy_params") or {})
    else:
        strategy_params = dict(strategy_params)

    if auto_approve is None:
        auto_approve = list(strategies_block.get("auto_approve") or [])

    if initial_capital is None:
        initial_capital = float(yaml_cfg.get("initial_capital", 100_000))

    engine_config: dict[str, Any] = {
        "initial_capital": initial_capital,
        "symbols": symbols,
        "strategies": strategies,
        "strategy_params": strategy_params,
    }
    risk = dict(yaml_cfg.get("risk") or {})
    if risk_overrides:
        risk.update(risk_overrides)
    engine_config.update(risk)

    # Optional behavioural knobs from live.yaml → engine/orchestrator config.
    # All default OFF/unchanged when absent (news-guard, optimal signal
    # combination, Kelly sizing), so live behaviour is unchanged until enabled.
    for key in (
        "news_guard",
        "signal_combination",
        "allocation_method",
        "kelly_fraction",
        "max_positions",
    ):
        if key in yaml_cfg:
            engine_config[key] = yaml_cfg[key]

    if kill_switch_drawdown is not None:
        engine_config["kill_switch_drawdown"] = kill_switch_drawdown
    if max_daily_trades is not None:
        engine_config["max_daily_trades"] = max_daily_trades
    if max_daily_turnover is not None:
        engine_config["max_daily_turnover"] = max_daily_turnover

    return {
        "broker": broker,
        "schedule": schedule,
        "approval_mode": approval_mode,
        "symbols": symbols,
        "strategies": strategies,
        "strategy_params": strategy_params,
        "auto_approve": auto_approve,
        "engine_config": engine_config,
    }


def build_live_providers(broker_type: str) -> dict[str, Any]:
    """Build the provider map expected by :class:`LiveDataFeed`.

    IBKR brokers use IB Gateway for prices + news sentiment; fundamentals
  come from FMP when ``FMP_API_KEY`` is set.  All other brokers use the
    multi-vendor :class:`FallbackProvider` chain.
    """
    if broker_type.startswith("ibkr"):
        from firm.data.providers.ibkr import IBKRProvider

        host = os.getenv("IBKR_HOST", "127.0.0.1")
        if broker_type in ("ibkr_paper", "ibkr"):
            port = int(os.getenv("IBKR_PAPER_PORT", "4002"))
        else:
            port = int(os.getenv("IBKR_PORT", "7496"))
        ibkr = IBKRProvider(host=host, port=port, client_id=2)
        providers: dict[str, Any] = {"prices": ibkr, "sentiment": ibkr}
        if os.getenv("FMP_API_KEY"):
            try:
                from firm.data.providers.fmp import FMPProvider

                providers["fundamentals"] = FMPProvider()
            except Exception as exc:
                log.info("FMP fundamentals provider not available: %s", exc)
        return providers

    from firm.data.providers.fallback import FallbackProvider

    market_data = FallbackProvider()
    return {
        "prices": market_data,
        "fundamentals": market_data,
        "sentiment": market_data,
    }


def fundamentals_available(providers: dict[str, Any] | None = None) -> bool:
    """Return True when a fundamentals feed is likely reachable."""
    if os.getenv("FMP_API_KEY") or os.getenv("MASSIVE_API_KEY"):
        return True
    if providers is None:
        return False
    fund_prov = providers.get("fundamentals")
    if fund_prov is None:
        return False
    from firm.data.providers.ibkr import IBKRProvider

    return not isinstance(fund_prov, IBKRProvider)


def filter_strategies_for_providers(
    strategies: list[str],
    providers: dict[str, Any] | None = None,
    *,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Drop fundamental-dependent strategies when no fundamentals provider exists."""
    if fundamentals_available(providers):
        return list(strategies)
    skipped = [s for s in strategies if s in FUNDAMENTAL_DEPENDENT_STRATEGIES]
    if not skipped:
        return list(strategies)
    remaining = [s for s in strategies if s not in FUNDAMENTAL_DEPENDENT_STRATEGIES]
    msg = (
        "No fundamentals provider configured (set FMP_API_KEY for IBKR live) — "
        f"disabling {skipped}. Active strategies: {remaining or '(none)'}"
    )
    (logger or log).warning(msg)
    return remaining
