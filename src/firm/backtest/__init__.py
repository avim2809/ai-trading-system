"""Backtest engine – drives bar-by-bar simulation via backtrader.

Heavy imports are deferred to avoid circular-import chains through
``firm.eval`` ↔ ``firm.portfolio.attribution``.
"""

from firm.backtest.analyzers import (
    DetailedReturnsAnalyzer,
    StrategyAttributionAnalyzer,
    TurnoverAnalyzer,
)
from firm.backtest.commissions import PercentageCommission
from firm.backtest.datafeeds import AdjustedPandasData, dataframe_to_feed, load_feeds
from firm.backtest.sizers import TargetWeightSizer


def __getattr__(name: str):
    if name == "BacktestEngine":
        from firm.backtest.engine import BacktestEngine
        return BacktestEngine
    if name == "FirmStrategy":
        from firm.backtest.firm_strategy import FirmStrategy
        return FirmStrategy
    if name == "PitViewAdapter":
        from firm.backtest.firm_strategy import PitViewAdapter
        return PitViewAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdjustedPandasData",
    "BacktestEngine",
    "DetailedReturnsAnalyzer",
    "FirmStrategy",
    "PercentageCommission",
    "PitViewAdapter",
    "StrategyAttributionAnalyzer",
    "TargetWeightSizer",
    "TurnoverAnalyzer",
    "dataframe_to_feed",
    "load_feeds",
]
