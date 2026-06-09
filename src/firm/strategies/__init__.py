"""Alpha strategies – signal generation against point-in-time data."""

from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import get, list_strategies, register

# Import every strategy module so that @register decorators execute.
from firm.strategies import (  # noqa: F401
    event_driven,
    gann,
    mean_reversion,
    ml_prediction,
    momentum,
    multi_factor,
    seasonality,
    sentiment,
    stat_arb,
    trend,
    volatility_breakout,
)

__all__ = ["BaseStrategy", "PitView", "register", "get", "list_strategies"]
