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
        # Optional callable(asof) -> list[str] giving survivorship-aware
        # index membership (e.g. firm.data.universe.UniverseResolver).
        self._universe_resolver = None

    def set_universe_resolver(self, resolver) -> None:
        """Install a survivorship-aware membership resolver for get_universe."""
        self._universe_resolver = resolver

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
        """Return price data for *symbols* where date <= asof.

        ``lookback_days`` is interpreted as a count of **trading days** (rows),
        not calendar days: the most recent ``lookback_days`` bars per symbol on
        or before *asof* are returned.  Strategies size their windows in
        trading days (e.g. a 200-day moving average), so a calendar-day filter
        would silently under-deliver history (~252 trading days span ~365
        calendar days) and could disable long-lookback strategies entirely.
        """
        if self._prices.empty:
            return pd.DataFrame()
        asof_ts = pd.Timestamp(asof)
        mask = self._prices["symbol"].isin(symbols) & (self._prices["date"] <= asof_ts)
        filtered = self._prices.loc[mask]
        if filtered.empty:
            return filtered.copy()
        # Keep the most recent ``lookback_days`` trading rows per symbol.
        filtered = filtered.sort_values("date")
        return filtered.groupby("symbol", group_keys=False).tail(lookback_days).copy()

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
        """Return the tradable universe as of *asof*.

        When a survivorship-aware ``universe_resolver`` is installed (via
        :meth:`set_universe_resolver`) it is authoritative and resolves
        point-in-time index membership.  Otherwise this falls back to every
        symbol with price data on or before *asof* — which is **not**
        survivorship-aware (it reflects whatever names happen to be loaded).
        """
        if self._universe_resolver is not None:
            return list(self._universe_resolver(asof))
        if self._prices.empty:
            return []
        asof_ts = pd.Timestamp(asof)
        return (
            self._prices.loc[self._prices["date"] <= asof_ts, "symbol"]
            .unique()
            .tolist()
        )
