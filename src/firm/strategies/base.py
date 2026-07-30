"""Base strategy interface and the PitView protocol.

Every strategy receives a :class:`PitView` – a read-only, point-in-time
data accessor – and returns a list of :class:`Signal` objects.
Phase 2 workers implement concrete strategies against this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from firm.contracts.models import Signal


@runtime_checkable
class PitView(Protocol):
    """Read-only point-in-time data accessor passed to strategies."""

    @property
    def asof(self) -> datetime:
        ...

    @property
    def universe(self) -> list[str]:
        ...

    def prices(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        ...

    def fundamentals(
        self,
        symbols: list[str] | None = None,
        lookback_reports: int = 4,
    ) -> pd.DataFrame:
        ...

    def sentiment(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        ...

    def estimates(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        ...


class BaseStrategy(ABC):
    """Abstract base for all alpha strategies."""

    def __init__(self, name: str, params: dict | None = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate(self, pit_view: PitView) -> list[Signal]:
        """Generate signals for the universe given point-in-time data."""
        ...
