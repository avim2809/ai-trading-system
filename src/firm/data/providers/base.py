"""Abstract base class every data provider must implement."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from firm.data.schemas import (
    AI_SCORE_COLS,
    ANALYST_RATINGS_COLS,
    CORPORATE_ACTION_COLS,
    FUNDAMENTAL_COLS,
    LIVE_SIGNAL_COLS,
    PRICE_COLS,
    SENTIMENT_COLS,
    UNIVERSE_COLUMNS,
)

log = logging.getLogger(__name__)


class ProviderError(Exception):
    """Raised when a data-provider operation fails."""


# Shared across every fundamentals provider (fmp.py, massive.py, finnhub.py,
# twelvedata.py, alphavantage.py): vendor "date" fields for ratios/financials
# endpoints are the fiscal PERIOD-END date, not the actual filing/announcement
# date — a real look-ahead bug for any strategy trusting date <= asof
# (multi_factor's value/quality factors in particular). SEC deadlines are
# 40-45 days for a 10-Q (accelerated filers) and up to 90 days for a 10-K; 45
# days is a conservative estimate covering the common case without needing to
# distinguish quarterly from annual reports. Mirrors the same fix already
# applied to macro data (see fred.py's _PUBLICATION_LAG_DAYS) — shift the date
# forward so the PIT store's date <= asof filter can't see a report before it
# would exist.
#
# This is only a *fallback*: providers that expose the real filing date
# (SEC EDGAR's `filed` field, FMP's `fillingDate`) should use
# `resolve_filing_date()` below to prefer the genuine date instead.
FUNDAMENTALS_PUBLICATION_LAG_DAYS = 45


def resolve_filing_date(
    period_end: Any,
    filed: Any = None,
    *,
    symbol: str = "",
) -> pd.Timestamp:
    """Point-in-time-correct timestamp for a fundamentals row.

    Prefers *filed* — the actual date the filing hit the wire (SEC EDGAR's
    ``filed`` field, FMP's ``fillingDate``, etc.) — over the
    ``period_end + FUNDAMENTALS_PUBLICATION_LAG_DAYS`` heuristic. The
    heuristic is only an *estimate* of when a filing became public: some
    accelerated filers report well under 45 days after period end, while
    some — especially small caps or late 10-K filers — take longer, so
    using the real date avoids both under- and over-estimating knowability
    when a provider actually supplies it. Falls back to the heuristic
    (logging at debug — this is routine for providers that don't expose a
    real filing date, not an error) when *filed* is missing or unparseable.
    """
    if filed:
        try:
            return pd.Timestamp(str(filed)[:10])
        except (ValueError, TypeError) as exc:
            log.debug(
                "Unparseable filing date %r for symbol=%s period=%s; "
                "falling back to %dd heuristic (%s)",
                filed, symbol, period_end, FUNDAMENTALS_PUBLICATION_LAG_DAYS, exc,
            )
    return pd.Timestamp(period_end) + pd.Timedelta(days=FUNDAMENTALS_PUBLICATION_LAG_DAYS)


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

    @staticmethod
    def empty_analyst_ratings() -> pd.DataFrame:
        return pd.DataFrame(columns=ANALYST_RATINGS_COLS)

    @staticmethod
    def empty_ai_scores() -> pd.DataFrame:
        return pd.DataFrame(columns=AI_SCORE_COLS)

    @staticmethod
    def empty_live_signals() -> pd.DataFrame:
        return pd.DataFrame(columns=LIVE_SIGNAL_COLS)

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

    @abstractmethod
    def get_analyst_ratings(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Return analyst rating-consensus dataframe with ANALYST_RATINGS_COLS."""
        ...

    @abstractmethod
    def get_ai_scores(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Return AI-driven composite score dataframe with AI_SCORE_COLS."""
        ...

    @abstractmethod
    def get_live_signals(self, symbols: list[str]) -> pd.DataFrame:
        """Return today's latest-snapshot signal dataframe with
        LIVE_SIGNAL_COLS. No historical time series exists for this
        capability — it is always "as of right now", live-only."""
        ...
