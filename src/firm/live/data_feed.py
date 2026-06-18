"""Live data feed – builds a PIT store from current provider data.

Fetches real-time/recent market data via existing DataProvider classes and
loads it into a PointInTimeDataStore so the agent pipeline can consume live
data through the same PitView interface used in backtesting.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from firm.data.pit_store import PointInTimeDataStore
from firm.data.providers.base import DataProvider

log = logging.getLogger(__name__)


class LivePitViewAdapter:
    """Adapts PointInTimeDataStore to the PitView protocol for live usage.

    Identical in shape to the backtest PitViewAdapter so the agent pipeline
    works without modification.
    """

    def __init__(
        self,
        pit_store: PointInTimeDataStore,
        asof: datetime,
        universe: list[str],
    ) -> None:
        self._pit_store = pit_store
        self._asof = asof
        self._universe = list(universe)

    @property
    def asof(self) -> datetime:
        return self._asof

    @property
    def universe(self) -> list[str]:
        return list(self._universe)

    def prices(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_prices(syms, self._asof, lookback_days)

    def fundamentals(self, symbols: list[str] | None = None) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_fundamentals(syms, self._asof)

    def sentiment(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_sentiment(syms, self._asof, lookback_days)


class LiveDataFeed:
    """Fetches current market data and populates a PIT store for the pipeline."""

    def __init__(
        self,
        providers: dict[str, DataProvider],
        universe: list[str],
        lookback_days: int = 252,
        exclude_forming_bar: bool = True,
    ) -> None:
        self._providers = providers
        self._universe = list(universe)
        self._lookback_days = lookback_days
        # Drop the current (asof-day) daily bar, which is typically still
        # forming intraday; using it would be a mild look-ahead and a source
        # of live-vs-backtest skew.  Completed bars are picked up next session.
        self._exclude_forming_bar = exclude_forming_bar
        self._pit_store = PointInTimeDataStore()

    def refresh(self, asof: datetime | None = None) -> LivePitViewAdapter:
        """Fetch latest data from providers and return a PitView.

        Args:
            asof: Decision timestamp.  Defaults to now (UTC).

        Returns:
            A :class:`LivePitViewAdapter` bound to the refreshed PIT store.
        """
        asof = asof or datetime.utcnow()
        end = asof.strftime("%Y-%m-%d")
        start = (asof - timedelta(days=self._lookback_days)).strftime("%Y-%m-%d")

        prices = pd.DataFrame()
        fundamentals = pd.DataFrame()
        sentiment = pd.DataFrame()

        price_prov = self._providers.get("prices")
        if price_prov:
            try:
                prices = price_prov.get_prices(self._universe, start, end)
                if self._exclude_forming_bar and not prices.empty and "date" in prices.columns:
                    today = pd.Timestamp(asof).normalize()
                    prices = prices[pd.to_datetime(prices["date"]).dt.normalize() < today]
                log.info("Fetched %d price rows for %d symbols", len(prices), len(self._universe))
            except Exception:
                log.error("Price fetch failed", exc_info=True)

        fund_prov = self._providers.get("fundamentals")
        if fund_prov:
            try:
                fundamentals = fund_prov.get_fundamentals(self._universe, start, end)
            except (NotImplementedError, Exception):
                log.debug("Fundamental fetch skipped or failed", exc_info=True)

        sent_prov = self._providers.get("sentiment")
        if sent_prov:
            try:
                sentiment = sent_prov.get_news_sentiment(self._universe, start, end)
            except (NotImplementedError, Exception):
                log.debug("Sentiment fetch skipped or failed", exc_info=True)

        self._pit_store = PointInTimeDataStore()
        self._pit_store.load(
            prices=prices,
            fundamentals=fundamentals if not fundamentals.empty else None,
            sentiment=sentiment if not sentiment.empty else None,
        )

        return LivePitViewAdapter(self._pit_store, asof, self._universe)

    @property
    def pit_store(self) -> PointInTimeDataStore:
        return self._pit_store
