"""Alpha strategies – signal generation against point-in-time data."""

# Import every strategy module so that @register decorators execute.
from firm.strategies import (  # noqa: F401
    danelfin_ai_score,
    danelfin_best_stocks_signal,
    danelfin_live_signals,
    danelfin_market_percentile,
    event_driven,
    gann,
    investing_analyst_ratings,
    mean_reversion,
    ml_prediction,
    momentum,
    multi_factor,
    pattern_recognition,
    regime_hmm,
    seasonality,
    sentiment,
    stat_arb,
    trend,
    volatility_breakout,
)
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import get, list_strategies, register

__all__ = ["BaseStrategy", "PitView", "get", "list_strategies", "register"]
