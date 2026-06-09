"""Point-in-time data store – the critical no-look-ahead component.

Every data access during backtesting MUST go through this store.  All queries
are filtered so that only rows with ``date <= asof`` are ever returned,
eliminating future-data leakage.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

log = logging.getLogger("firm.data.pit_store")


class PointInTimeDataStore:
    """Ensures no future data leaks into strategy decisions.

    All queries are filtered to return only data with timestamp <= asof.
    """

    def __init__(self) -> None:
        self._prices: pd.DataFrame = pd.DataFrame()
        self._fundamentals: pd.DataFrame = pd.DataFrame()
        self._sentiment: pd.DataFrame = pd.DataFrame()
        self._corporate_actions: pd.DataFrame = pd.DataFrame()

    def load(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        corporate_actions: pd.DataFrame | None = None,
    ) -> None:
        """Load all datasets. Called once at backtest start."""
        self._prices = self._ensure_date_col(prices)
        self._fundamentals = self._ensure_date_col(fundamentals) if fundamentals is not None else pd.DataFrame()
        self._sentiment = self._ensure_date_col(sentiment) if sentiment is not None else pd.DataFrame()
        self._corporate_actions = (
            self._ensure_date_col(corporate_actions) if corporate_actions is not None else pd.DataFrame()
        )
        log.info(
            "PIT store loaded: %d price rows, %d fundamental rows, %d sentiment rows",
            len(self._prices),
            len(self._fundamentals),
            len(self._sentiment),
        )

    @staticmethod
    def _ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "date" not in df.columns:
            return df
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_prices(
        self,
        symbols: list[str],
        asof: datetime,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """Return price data for symbols where date <= asof, up to lookback_days back."""
        if self._prices.empty:
            return pd.DataFrame()
        asof_ts = pd.Timestamp(asof)
        earliest = asof_ts - timedelta(days=lookback_days)
        mask = (
            self._prices["symbol"].isin(symbols)
            & (self._prices["date"] <= asof_ts)
            & (self._prices["date"] >= earliest)
        )
        return self._prices.loc[mask].copy()

    def get_fundamentals(
        self,
        symbols: list[str],
        asof: datetime,
    ) -> pd.DataFrame:
        """Return latest fundamental data for each symbol where date <= asof."""
        if self._fundamentals.empty:
            return pd.DataFrame()
        asof_ts = pd.Timestamp(asof)
        mask = self._fundamentals["symbol"].isin(symbols) & (self._fundamentals["date"] <= asof_ts)
        filtered = self._fundamentals.loc[mask]
        if filtered.empty:
            return pd.DataFrame()
        return filtered.sort_values("date").groupby("symbol").last().reset_index()

    def get_sentiment(
        self,
        symbols: list[str],
        asof: datetime,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        """Return sentiment data where date <= asof within the lookback window."""
        if self._sentiment.empty:
            return pd.DataFrame()
        asof_ts = pd.Timestamp(asof)
        earliest = asof_ts - timedelta(days=lookback_days)
        mask = (
            self._sentiment["symbol"].isin(symbols)
            & (self._sentiment["date"] <= asof_ts)
            & (self._sentiment["date"] >= earliest)
        )
        return self._sentiment.loc[mask].copy()

    def get_universe(self, asof: datetime) -> list[str]:
        """Return universe as of date (survivorship-aware).

        Falls back to all symbols present in price data up to *asof*.
        """
        if self._prices.empty:
            return []
        asof_ts = pd.Timestamp(asof)
        return (
            self._prices.loc[self._prices["date"] <= asof_ts, "symbol"]
            .unique()
            .tolist()
        )
