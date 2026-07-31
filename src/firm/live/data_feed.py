"""Live data feed – builds a PIT store from current provider data.

Fetches real-time/recent market data via existing DataProvider classes and
loads it into a PointInTimeDataStore so the agent pipeline can consume live
data through the same PitView interface used in backtesting.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from firm.data.fundamentals_cache import (
    hours_since_refresh,
    load_cached_fundamentals_df,
    merge_with_cached_fundamentals,
    symbols_missing_fundamentals,
)
from firm.data.sentiment_cache import (
    incremental_days_from_env,
    load_cached_sentiment_df,
    merge_with_cached_sentiment,
    partition_sentiment_fetch,
    save_sentiment_cache,
)
from firm.data.pit_store import PointInTimeDataStore
from firm.data.providers.base import DataProvider
from firm.time_utils import utcnow

log = logging.getLogger(__name__)

# Live cycles default to cache-only fundamentals (refresh offline via
# ``firm.live.fundamentals_refresh`` / daily APScheduler job in firm-api).
# ``FIRM_LIVE_FETCH_FUNDAMENTALS=1`` to opt back into per-cycle network
# fetches for symbols missing from cache.
_LIVE_FETCH_ENV = "FIRM_LIVE_FETCH_FUNDAMENTALS"
# When live fetch is enabled, only hit the network if the cache is older than
# this many hours (daily refresh cadence by default).
_REFRESH_MAX_AGE_HOURS_ENV = "FIRM_FUNDAMENTALS_REFRESH_MAX_AGE_HOURS"


def _live_fundamentals_fetch_enabled() -> bool:
    return os.getenv(_LIVE_FETCH_ENV, "").strip().lower() in ("1", "true", "yes")


def _refresh_max_age_hours() -> float:
    raw = os.getenv(_REFRESH_MAX_AGE_HOURS_ENV, "24")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 24.0


# Analyst-ratings consensus (FMP's grades-historical) updates monthly, so
# live cycles default to cache-only here too — same rationale as
# fundamentals, opt in with FIRM_LIVE_FETCH_ANALYST_RATINGS=1. Unlike
# sentiment's hot/cold incremental-day partitioning (built for a
# daily-cadence series), a plain per-cycle fetch of the whole universe is
# cheap enough at monthly cadence not to need that complexity.
_LIVE_FETCH_RATINGS_ENV = "FIRM_LIVE_FETCH_ANALYST_RATINGS"


def _live_analyst_ratings_fetch_enabled() -> bool:
    return os.getenv(_LIVE_FETCH_RATINGS_ENV, "").strip().lower() in ("1", "true", "yes")


# Danelfin's AI scores update daily (unlike analyst-ratings' monthly
# cadence) — still cache-only by default for consistency with every other
# optional capability here, opt in with FIRM_LIVE_FETCH_AI_SCORES=1. A live
# fetch is cheap even when enabled: get_ai_scores(start=yesterday) only
# pulls page 1 per symbol (its own pagination stops as soon as it reaches
# `start`), not a full historical re-walk.
_LIVE_FETCH_AI_SCORES_ENV = "FIRM_LIVE_FETCH_AI_SCORES"


def _live_ai_scores_fetch_enabled() -> bool:
    return os.getenv(_LIVE_FETCH_AI_SCORES_ENV, "").strip().lower() in ("1", "true", "yes")


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

    def fundamentals(
        self, symbols: list[str] | None = None, lookback_reports: int = 4
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_fundamentals(syms, self._asof, lookback_reports)

    def sentiment(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_sentiment(syms, self._asof, lookback_days)

    def estimates(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_estimates(syms, self._asof, lookback_days)

    def ai_scores(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        syms = symbols or self._universe
        return self._pit_store.get_ai_scores(syms, self._asof, lookback_days)


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
        self._universe_resolver = self._build_universe_resolver()

    def _build_universe_resolver(self):
        """Install a survivorship-aware resolver on the PIT store, when possible.

        Live trading always targets *today's* symbol list, so this is more a
        consistency/audit hook than a survivorship-bias fix (that's inherently a
        backtest concern) — but wiring it here means real membership data (once
        acquired; see the ``pit-universe-membership`` follow-up) also validates
        the live universe without further code changes.
        """
        try:
            from firm.config import get_settings
            from firm.runtime import build_universe_resolver

            return build_universe_resolver(get_settings(), self._universe)
        except Exception:
            log.warning(
                "Live universe resolver setup failed — continuing with the "
                "configured symbol list only",
                exc_info=True,
            )
            from firm.data.universe import UniverseResolver

            return UniverseResolver.from_static(self._universe)

    def refresh(self, asof: datetime | None = None) -> LivePitViewAdapter:
        """Fetch latest data from providers and return a PitView.

        Args:
            asof: Decision timestamp.  Defaults to now (UTC).

        Returns:
            A :class:`LivePitViewAdapter` bound to the refreshed PIT store.
        """
        # Naive UTC, not tz-aware: this flows straight into
        # PointInTimeDataStore.get_prices/get_fundamentals/get_sentiment,
        # which compare it against tz-naive `date` columns loaded via
        # pd.to_datetime — an aware value here would raise on that compare.
        asof = asof or utcnow()
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
                    today = pd.Timestamp(asof)
                    if today.tz is not None:
                        # Defensive: price data columns are tz-naive, so an
                        # aware asof (e.g. an explicit caller-supplied one)
                        # would otherwise raise on this comparison.
                        today = today.tz_localize(None)
                    today = today.normalize()
                    prices = prices[pd.to_datetime(prices["date"]).dt.normalize() < today]
                log.info("Fetched %d price rows for %d symbols", len(prices), len(self._universe))
            except Exception:
                log.error("Price fetch failed", exc_info=True)
        else:
            log.error(
                "No 'prices' provider configured on this LiveDataFeed — "
                "refresh() will feed the pipeline an empty price frame"
            )

        cached_fundamentals = load_cached_fundamentals_df()
        fund_prov = self._providers.get("fundamentals")
        if fund_prov and _live_fundamentals_fetch_enabled():
            from firm.config import get_settings

            age = hours_since_refresh(get_settings().data.cache_dir)
            max_age = _refresh_max_age_hours()
            if age is None or age >= max_age:
                missing_fund = symbols_missing_fundamentals(
                    self._universe, cached_fundamentals,
                )
                if missing_fund:
                    try:
                        log.info(
                            "Live fundamentals fetch for %d uncached symbol(s): %s",
                            len(missing_fund), missing_fund,
                        )
                        fundamentals = fund_prov.get_fundamentals(missing_fund, start, end)
                    except (NotImplementedError, Exception):
                        log.debug("Fundamental fetch skipped or failed", exc_info=True)
                else:
                    log.info(
                        "Fundamentals cache covers full universe (%d symbols)",
                        len(self._universe),
                    )
            else:
                log.debug(
                    "Skipping live fundamentals fetch — cache refreshed %.1fh ago (< %.0fh)",
                    age, max_age,
                )
        elif fund_prov:
            log.debug(
                "Live fundamentals: cache-only mode (set %s=1 to enable network fetch)",
                _LIVE_FETCH_ENV,
            )
        else:
            log.warning("No 'fundamentals' provider configured on this LiveDataFeed")

        fundamentals = merge_with_cached_fundamentals(fundamentals, cached_fundamentals)

        cached_sentiment = load_cached_sentiment_df()
        sent_prov = self._providers.get("sentiment")
        if sent_prov:
            try:
                plan = partition_sentiment_fetch(
                    self._universe,
                    cached_sentiment,
                    asof,
                    self._lookback_days,
                    incremental_days=incremental_days_from_env(),
                )
                live_parts: list[pd.DataFrame] = []
                cold = plan["cold_symbols"]
                warm = plan["warm_symbols"]
                if cold:
                    log.info(
                        "Sentiment fetch (full window) for %d cold symbol(s)",
                        len(cold),
                    )
                    live_parts.append(
                        sent_prov.get_news_sentiment(
                            cold, plan["full_start"], plan["end"],
                        )
                    )
                if warm:
                    log.info(
                        "Sentiment fetch (incremental) for %d warm symbol(s)",
                        len(warm),
                    )
                    live_parts.append(
                        sent_prov.get_news_sentiment(
                            warm, plan["incremental_start"], plan["end"],
                        )
                    )
                if not cold and not warm:
                    log.info(
                        "Sentiment cache fresh for full universe (%d symbols)",
                        len(self._universe),
                    )
                if live_parts:
                    non_empty = [
                        df for df in live_parts if df is not None and not df.empty
                    ]
                    if non_empty:
                        sentiment = pd.concat(non_empty, ignore_index=True)
                sentiment = merge_with_cached_sentiment(sentiment, cached_sentiment)
                if not sentiment.empty:
                    sentiment = save_sentiment_cache(
                        sentiment,
                        lookback_days=self._lookback_days,
                        asof=asof,
                    )
            except (NotImplementedError, Exception):
                log.warning("Sentiment fetch failed — using cache if available", exc_info=True)
                sentiment = merge_with_cached_sentiment(pd.DataFrame(), cached_sentiment)
        else:
            log.warning("No 'sentiment' provider configured on this LiveDataFeed")

        estimates = pd.DataFrame()
        try:
            from firm.config import get_settings
            from firm.runtime import load_analyst_ratings

            cached_estimates = load_analyst_ratings(get_settings())
            if cached_estimates is not None:
                estimates = cached_estimates
        except Exception:
            log.warning("Analyst-ratings cache load failed — continuing without it", exc_info=True)
        estimates_prov = self._providers.get("estimates")
        if estimates_prov and _live_analyst_ratings_fetch_enabled():
            try:
                fresh = estimates_prov.get_analyst_ratings(self._universe, start, end)
                if not fresh.empty:
                    estimates = (
                        pd.concat([estimates, fresh], ignore_index=True)
                        if not estimates.empty else fresh
                    )
                    estimates["date"] = pd.to_datetime(estimates["date"])
                    estimates = estimates.sort_values(["symbol", "date"]).drop_duplicates(
                        subset=["symbol", "date"], keep="last",
                    )
                log.info("Fetched %d analyst-ratings rows", len(fresh))
            except (NotImplementedError, Exception):
                log.warning("Analyst-ratings fetch failed — using cache if available", exc_info=True)
        elif estimates_prov:
            log.debug(
                "Live analyst ratings: cache-only mode (set %s=1 to enable network fetch)",
                _LIVE_FETCH_RATINGS_ENV,
            )

        ai_scores = pd.DataFrame()
        try:
            from firm.config import get_settings
            from firm.runtime import load_ai_scores

            cached_ai_scores = load_ai_scores(get_settings())
            if cached_ai_scores is not None:
                ai_scores = cached_ai_scores
        except Exception:
            log.warning("AI-scores cache load failed — continuing without it", exc_info=True)
        ai_scores_prov = self._providers.get("ai_scores")
        if ai_scores_prov and _live_ai_scores_fetch_enabled():
            try:
                fresh = ai_scores_prov.get_ai_scores(self._universe, start, end)
                if not fresh.empty:
                    ai_scores = (
                        pd.concat([ai_scores, fresh], ignore_index=True)
                        if not ai_scores.empty else fresh
                    )
                    ai_scores["date"] = pd.to_datetime(ai_scores["date"])
                    ai_scores = ai_scores.sort_values(["symbol", "date"]).drop_duplicates(
                        subset=["symbol", "date"], keep="last",
                    )
                log.info("Fetched %d AI-score rows", len(fresh))
            except (NotImplementedError, Exception):
                log.warning("AI-scores fetch failed — using cache if available", exc_info=True)
        elif ai_scores_prov:
            log.debug(
                "Live AI scores: cache-only mode (set %s=1 to enable network fetch)",
                _LIVE_FETCH_AI_SCORES_ENV,
            )

        self._pit_store = PointInTimeDataStore()
        self._pit_store.set_universe_resolver(self._universe_resolver)
        self._pit_store.load(
            prices=prices,
            fundamentals=fundamentals if not fundamentals.empty else None,
            sentiment=sentiment if not sentiment.empty else None,
            estimates=estimates if not estimates.empty else None,
            ai_scores=ai_scores if not ai_scores.empty else None,
        )

        return LivePitViewAdapter(self._pit_store, asof, self._universe)

    @property
    def pit_store(self) -> PointInTimeDataStore:
        return self._pit_store
