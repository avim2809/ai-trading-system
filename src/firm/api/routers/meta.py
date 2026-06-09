"""Read-only meta endpoints: health, strategies, config defaults."""

from __future__ import annotations

from fastapi import APIRouter

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
}


@router.get("/health")
def health():
    return {"status": "ok"}


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
    }
