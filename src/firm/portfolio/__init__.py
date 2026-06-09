"""Portfolio state tracking and attribution."""

from firm.portfolio.state import PortfolioState

__all__ = ["PortfolioState"]


def __getattr__(name: str):
    if name == "PerformanceAttribution":
        from firm.portfolio.attribution import PerformanceAttribution

        return PerformanceAttribution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
