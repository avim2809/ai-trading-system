"""Read-only meta endpoints: health, strategies, config defaults."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

log = logging.getLogger(__name__)

router = APIRouter()

STRATEGY_INFO: dict[str, dict[str, str]] = {
    "momentum": {
        "summary": "Cross-sectional momentum",
        "description": "Securities that have performed well over the past 6-12 months tend to continue outperforming. Ranks the universe by 12-1 month returns, going long top decile and short bottom decile.",
    },
    "trend": {
        "summary": "Time-series trend following",
        "description": "When an asset's price sits above its long-term moving average it is in an uptrend. Uses dual MA crossover (50/200d) scaled by inverse volatility for managed-momentum exposure.",
    },
    "mean_reversion": {
        "summary": "Short-term reversal",
        "description": "Short recent winners and buy recent losers over 1-5 day horizons, exploiting the well-documented short-term reversal effect in cross-sectional equity returns.",
    },
    "stat_arb": {
        "summary": "Statistical arbitrage (pairs trading)",
        "description": "Identifies cointegrated stock pairs, computes OLS hedge ratios, and trades mean reversion of the spread when the z-score exceeds entry thresholds.",
    },
    "multi_factor": {
        "summary": "Multi-factor composite equity",
        "description": "Combines value (1/PE, 1/PB), quality (ROE, low leverage), momentum (12-1m return), and low-volatility factors into a single composite z-score. Diversifies across alpha sources.",
    },
    "sentiment": {
        "summary": "News & sentiment driven",
        "description": "Aggregates recent news sentiment scores per symbol and computes rolling sentiment deltas to detect shifts in market mood that precede price moves.",
    },
    "event_driven": {
        "summary": "Post-earnings announcement drift (PEAD)",
        "description": "Detects earnings surprises from fundamental data and trades the well-documented post-announcement drift: positive surprises drift up, negative drift down over subsequent days.",
    },
    "ml_prediction": {
        "summary": "Machine learning return prediction",
        "description": "Walk-forward trained model (GradientBoosting or Ridge) using lagged returns, volatility, and volume features to predict forward N-day returns. Strict point-in-time training cutoff.",
    },
    "volatility_breakout": {
        "summary": "Volatility breakout",
        "description": "Markets alternate between low-volatility compression and high-volatility expansion. Enters when price breaks out of a recent range during a vol squeeze, riding the expansion.",
    },
    "seasonality": {
        "summary": "Calendar/seasonality effects",
        "description": "Exploits persistent calendar anomalies: turn-of-month effect (bullish bias in last 1 and first 3 trading days) and day-of-week return patterns.",
    },
    "gann": {
        "summary": "W.D. Gann composite",
        "description": "Quantifies five Gann techniques — geometric angles from pivot points, Square of Nine support/resistance, time cycle convergence, swing charting, and retracement levels — into a composite signal with a trend-strength filter.",
    },
    "regime_hmm": {
        "summary": "HMM market-regime detection",
        "description": "Fits a per-symbol Gaussian Hidden Markov Model on stationarised features (log returns, 5-day cumulative log return, ATR, volume-spike ratio), decodes the hidden regime via the forward posterior, and labels states Bull/Chop/Bear by mean return. Emits directional signals (long in Bull, short in Bear, damped in Chop) weighted by regime confidence. Pairs with the RiskAgent market-regime exposure overlay (Chen, Yi & Zhao, 2020).",
    },
}


@router.get("/health")
def health(request: Request):
    """Liveness/readiness probe.

    ``status`` always reflects the API process itself — it stays ``"ok"``
    even when the live engine's broker is disconnected, so a transient IBKR
    outage (e.g. IB Gateway's mandatory daily restart) can't trip an
    infra-level auto-restart of the whole API process and take down
    approvals/dashboards/backtests along with it. Broker connectivity is
    surfaced as its own field for monitoring/alerting to consume instead.
    ``broker.connected`` is a fast local state read (``is_connected()``
    never makes a network call — see ``IBKRBroker.is_connected``), so this
    endpoint stays cheap enough to poll frequently.
    """
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None or not getattr(engine, "is_running", False):
        return {
            "status": "ok",
            "broker": {"type": None, "connected": None, "live_engine_running": False},
        }

    broker = getattr(engine, "_broker", None)
    connected = None
    if broker is not None:
        try:
            connected = broker.is_connected()
        except Exception:
            log.warning("Broker is_connected() check failed", exc_info=True)
            connected = False

    broker_type = getattr(engine, "_broker_type", "")
    if connected is False:
        log.warning(
            "Health check: live engine running but broker %r is disconnected",
            broker_type,
        )
    return {
        "status": "ok",
        "broker": {
            "type": broker_type or None,
            "connected": connected,
            "live_engine_running": True,
        },
    }


@router.get("/strategies")
def strategies():
    from firm.strategies import list_strategies, get as get_strategy
    names = list_strategies()
    result = []
    for name in names:
        cls = get_strategy(name)
        default_params = {}
        if hasattr(cls, "default_params"):
            default_params = cls.default_params
        info = STRATEGY_INFO.get(name, {})
        result.append({
            "name": name,
            "default_params": default_params,
            "summary": info.get("summary", name),
            "description": info.get("description", ""),
        })
    return result


@router.get("/config/defaults")
def config_defaults():
    from firm.config import get_settings
    settings = get_settings()
    return {
        "universe": settings.universe.model_dump(),
        "backtest": settings.backtest.model_dump(),
        "risk": settings.risk.model_dump(),
        "strategy_params": settings.strategy_params,
        "allocation_method": settings.allocation_method,
        "kelly_fraction": settings.kelly_fraction,
        "signal_combination": settings.signal_combination or {"method": "confidence"},
        "strategy_circuit_breaker": settings.strategy_circuit_breaker or {"enabled": False},
        "strategy_regime_weights": settings.strategy_regime_weights or {"enabled": False},
    }
