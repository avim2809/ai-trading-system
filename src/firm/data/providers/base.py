"""Abstract base class every data provider must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from firm.data.schemas import (
    CORPORATE_ACTION_COLS,
    FUNDAMENTAL_COLS,
    PRICE_COLS,
    SENTIMENT_COLS,
    UNIVERSE_COLUMNS,
)


class ProviderError(Exception):
    """Raised when a data-provider operation fails."""


# Shared across every fundamentals provider (fmp.py, massive.py): vendor
# "date" fields for ratios/financials endpoints are the fiscal PERIOD-END
# date, not the actual filing/announcement date — a real look-ahead bug for
# any strategy trusting date <= asof (multi_factor's value/quality factors
# in particular). SEC deadlines are 40-45 days for a 10-Q (accelerated
# filers) and up to 90 days for a 10-K; 45 days is a conservative estimate
# covering the common case without needing to distinguish quarterly from
# annual reports. Mirrors the same fix already applied to macro data (see
# fred.py's _PUBLICATION_LAG_DAYS) — shift the date forward so the PIT
# store's date <= asof filter can't see a report before it would exist.
FUNDAMENTALS_PUBLICATION_LAG_DAYS = 45


class DataProvider(ABC):
    """Abstract base for all data providers."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Empty-frame helpers – return typed empty DataFrames for each domain.
    # ------------------------------------------------------------------

    @staticmethod
    def empty_prices() -> pd.DataFrame:
        return pd.DataFrame(columns=PRICE_COLS)

    @staticmethod
    def empty_fundamentals() -> pd.DataFrame:
        return pd.DataFrame(columns=FUNDAMENTAL_COLS)

    @staticmethod
    def empty_news() -> pd.DataFrame:
        return pd.DataFrame(columns=SENTIMENT_COLS)

    @staticmethod
    def empty_corporate_actions() -> pd.DataFrame:
        return pd.DataFrame(columns=CORPORATE_ACTION_COLS)

    @staticmethod
    def empty_universe() -> pd.DataFrame:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)

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
