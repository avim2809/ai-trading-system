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
        self._estimates: pd.DataFrame = pd.DataFrame()
        self._corporate_actions: pd.DataFrame = pd.DataFrame()
        # Macro series keyed by FRED series ID → DataFrame[date, <series_id>]
        self._macro: dict[str, pd.DataFrame] = {}
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
        estimates: pd.DataFrame | None = None,
    ) -> None:
        """Load all datasets. Called once at backtest start."""
        self._prices = self._ensure_date_col(prices)
        self._fundamentals = self._ensure_date_col(fundamentals) if fundamentals is not None else pd.DataFrame()
        self._sentiment = self._ensure_date_col(sentiment) if sentiment is not None else pd.DataFrame()
        self._estimates = self._ensure_date_col(estimates) if estimates is not None else pd.DataFrame()
        self._corporate_actions = (
            self._ensure_date_col(corporate_actions) if corporate_actions is not None else pd.DataFrame()
        )
        log.info(
            "PIT store loaded: %d price rows, %d fundamental rows, %d sentiment rows, %d estimates rows",
            len(self._prices),
            len(self._fundamentals),
            len(self._sentiment),
            len(self._estimates),
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
        lookback_reports: int = 4,
    ) -> pd.DataFrame:
        """Return the most recent *lookback_reports* snapshots per symbol
        where date <= asof (oldest first, newest last) — not just a single
        latest row.

        Event-driven surprise detection (e.g. EPS change quarter-over-
        quarter) needs at least 2 snapshots to compute anything; this used
        to always collapse to exactly one row per symbol via
        groupby("symbol").last(), so any such surprise detection could
        never see more than one data point and silently always fell
        through to whatever fallback existed — a real incident, not a
        hypothetical. Strategies that only want the latest snapshot (e.g.
        multi_factor's point-in-time value/quality factors) already reduce
        to one row themselves via their own groupby("symbol").last(), so
        returning more history here doesn't change their behaviour.
        """
        if self._fundamentals.empty:
            return pd.DataFrame()
        asof_ts = pd.Timestamp(asof)
        mask = self._fundamentals["symbol"].isin(symbols) & (self._fundamentals["date"] <= asof_ts)
        filtered = self._fundamentals.loc[mask]
        if filtered.empty:
            return pd.DataFrame()
        return (
            filtered.sort_values("date")
            .groupby("symbol", group_keys=False)
            .tail(lookback_reports)
            .reset_index(drop=True)
        )

    def get_estimates(
        self,
        symbols: list[str],
        asof: datetime,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """Return analyst-ratings-consensus snapshots where date <= asof
        within the lookback window (e.g. ANALYST_RATINGS_COLS — FMP's
        grades-historical is monthly, so a ~365-day default keeps roughly a
        year of consensus trend per symbol, unlike get_sentiment's 5-day
        default suited to a daily-cadence series)."""
        if self._estimates.empty:
            return pd.DataFrame()
        asof_ts = pd.Timestamp(asof)
        earliest = asof_ts - timedelta(days=lookback_days)
        mask = (
            self._estimates["symbol"].isin(symbols)
            & (self._estimates["date"] <= asof_ts)
            & (self._estimates["date"] >= earliest)
        )
        return self._estimates.loc[mask].copy()

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

    def load_macro(self, bundle: dict[str, pd.DataFrame]) -> None:
        """Load a dict of FRED macro series into the store.

        Each value must be a DataFrame with columns [date, <series_id>] as
        returned by :func:`firm.data.providers.fred.fetch_macro_bundle`.
        """
        for series_id, df in bundle.items():
            if df.empty:
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            self._macro[series_id] = df.sort_values("date").reset_index(drop=True)
        log.info("PIT store: loaded %d macro series: %s", len(self._macro), list(self._macro))

    def get_macro(
        self,
        series_id: str,
        asof: datetime,
        lookback_days: int = 365,
    ) -> pd.Series:
        """Return a macro indicator series where date <= asof.

        Returns a pandas Series indexed by date with the indicator values,
        covering at most *lookback_days* calendar days before *asof*.
        Returns an empty Series if the series was not loaded.

        Args:
            series_id:     FRED series ID (e.g. "T10Y2Y", "CPIAUCSL").
            asof:          Point-in-time ceiling — no future data.
            lookback_days: How many calendar days of history to return.
        """
        df = self._macro.get(series_id)
        if df is None or df.empty:
            return pd.Series(dtype=float, name=series_id)
        asof_ts = pd.Timestamp(asof)
        earliest = asof_ts - timedelta(days=lookback_days)
        value_col = series_id if series_id in df.columns else df.columns[-1]
        mask = (df["date"] <= asof_ts) & (df["date"] >= earliest)
        subset = df.loc[mask].set_index("date")[value_col]
        return subset

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

    def get_universe_union(self, start: datetime, end: datetime) -> list[str]:
        """Union of tradable symbols across ``[start, end]`` — the superset a
        backtest needs to load data feeds for when the tradable set can
        change mid-window (a name added to the index after ``start`` still
        needs its feed loaded, even though :meth:`get_universe` at ``start``
        alone wouldn't include it yet).

        Uses the resolver's own ``symbols_between`` when it has one (e.g.
        :meth:`firm.data.universe.UniverseResolver.symbols_between`, which
        correctly captures a name that joins *and* leaves entirely within the
        window). A resolver installed as a plain callable (no
        ``symbols_between``) degrades to the union of the start/end
        snapshots — this can miss a name whose entire membership window falls
        strictly between ``start`` and ``end``, which is a much narrower gap
        than not resolving membership changes at all.
        """
        resolver = self._universe_resolver
        if resolver is not None and hasattr(resolver, "symbols_between"):
            return list(resolver.symbols_between(start, end))
        if resolver is not None:
            log.debug(
                "get_universe_union: resolver has no symbols_between; "
                "degrading to union of start/end snapshots (may miss a "
                "name whose membership window falls entirely inside "
                "[%s, %s])",
                start, end,
            )
        return sorted(set(self.get_universe(start)) | set(self.get_universe(end)))
