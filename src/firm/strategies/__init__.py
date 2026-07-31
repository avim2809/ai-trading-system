"""Alpha strategies – signal generation against point-in-time data."""

from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import get, list_strategies, register

# Import every strategy module so that @register decorators execute.
from firm.strategies import (  # noqa: F401
    danelfin_ai_score,
    event_driven,
    gann,
    investing_analyst_ratings,
    mean_reversion,
    ml_prediction,
    momentum,
    multi_factor,
    regime_hmm,
    seasonality,
    sentiment,
    stat_arb,
    trend,
    volatility_breakout,
)

__all__ = ["BaseStrategy", "PitView", "register", "get", "list_strategies"]
