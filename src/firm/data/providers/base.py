"""Abstract base class every data provider must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ProviderError(Exception):
    """Raised when a data-provider operation fails."""


class DataProvider(ABC):
    """Abstract base for all data providers."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def get_prices(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Return OHLCV dataframe with columns from PRICE_COLS."""
        ...

    @abstractmethod
    def get_fundamentals(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Return fundamentals dataframe."""
        ...

    @abstractmethod
    def get_news_sentiment(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Return sentiment dataframe."""
        ...

    @abstractmethod
    def get_corporate_actions(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Return corporate actions (splits, dividends)."""
        ...

    @abstractmethod
    def get_universe_constituents(
        self, index: str, date: str
    ) -> list[str]:
        """Return list of constituent symbols for an index as of a date."""
        ...
