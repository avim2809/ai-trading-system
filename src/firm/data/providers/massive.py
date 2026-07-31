"""Massive.com data provider – prices, news/sentiment, fundamentals, corporate actions.

API docs: https://massive.com/docs

Endpoints used:
* Custom OHLC bars:    GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
* News + sentiment:   GET /v2/reference/news   (insights[] carries per-ticker sentiment)
* Financial ratios:   GET /stocks/financials/v1/ratios
* Dividends:          GET /stocks/v1/dividends
* Splits:             GET /stocks/v1/splits

Authentication: API key passed as ``?apiKey=<key>`` query parameter.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Sequence

import pandas as pd
import requests

from firm.data.providers.base import (
    FUNDAMENTALS_PUBLICATION_LAG_DAYS,
    DataProvider,
    ProviderError,
)
from firm.data.schemas import (
    CORPORATE_ACTION_COLS,
    FUNDAMENTAL_COLS,
    PRICE_COLS,
    SENTIMENT_COLS,
)

log = logging.getLogger("firm.data.providers.massive")

_BASE_URL = "https://api.massive.com"
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_DEFAULT_NEWS_MIN_INTERVAL_SEC = 12.0
# Module-level pacing for news — free-tier Massive keys are ~5 req/min.
_last_news_request_at: float = 0.0

_SENTIMENT_MAP: dict[str, float] = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


class MassiveProvider(DataProvider):
    """Massive.com REST API wrapper.

    Covers equities prices (adjusted OHLCV), news with built-in per-ticker
    sentiment scores, financial ratios, and corporate actions (dividends +
    splits).

    Args:
        api_key: Massive API key.  Falls back to ``settings.massive_api_key``
                 when not supplied directly.
        settings: Application settings instance; auto-loaded when *None*.
    """

    name = "massive"
    # Free-tier keys often lack the financial-ratios endpoint; after the
    # first plan-wide 403 we skip the rest of the batch (and future calls)
    # to avoid hammering the API and spamming logs every live cycle.
    _fundamentals_plan_blocked: bool = False

    def __init__(self, api_key: str = "", settings=None) -> None:
        if not api_key:
            if settings is None:
                from firm.config import get_settings
                settings = get_settings()
            api_key = getattr(settings, "massive_api_key", "")
        if not api_key:
            raise ProviderError(
                "MassiveProvider requires an API key. "
                "Set MASSIVE_API_KEY in .env or pass api_key= explicitly."
            )
        super().__init__(api_key)

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    @staticmethod
    def _news_min_interval_sec() -> float:
        raw = os.getenv("MASSIVE_NEWS_MIN_INTERVAL_SEC", str(_DEFAULT_NEWS_MIN_INTERVAL_SEC))
        try:
            return max(0.0, float(raw))
        except ValueError:
            return _DEFAULT_NEWS_MIN_INTERVAL_SEC

    @classmethod
    def _wait_for_news_slot(cls) -> None:
        global _last_news_request_at
        interval = cls._news_min_interval_sec()
        if interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - _last_news_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _last_news_request_at = time.monotonic()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        all_params = {**(params or {}), "apiKey": self.api_key}
        url = f"{_BASE_URL}{path}"
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            log.debug("GET %s (attempt %d)", url, attempt)
            try:
                resp = requests.get(url, params=all_params, timeout=30)
            except requests.RequestException as exc:
                last_exc = exc
                log.warning("massive_network_error url=%s attempt=%d error=%s", url, attempt, exc)
                time.sleep(min(_BACKOFF_BASE ** attempt, 10.0))
                continue
            if resp.status_code == 429:
                # Massive's rate limit is a hard account-wide per-minute
                # ceiling, not a per-request fluke — retrying the same
                # symbol with backoff won't clear it within a few seconds,
                # and FallbackProvider is already calling this once per
                # symbol in a batch. Retrying here would multiply a single
                # rate-limit event into minutes of wasted latency across a
                # multi-symbol universe. Fail fast so the caller's fallback
                # chain (Tiingo/AlphaVantage/FMP) engages immediately.
                log.warning("massive_rate_limited url=%s — failing fast, no retry", url)
                raise ProviderError(f"{url} returned HTTP 429 (rate limited)")
            if resp.status_code in (500, 502, 503, 504):
                last_exc = ProviderError(f"{url} returned HTTP {resp.status_code}")
                log.warning("massive_transient_error status=%d attempt=%d", resp.status_code, attempt)
                time.sleep(min(_BACKOFF_BASE ** attempt, 10.0))
                continue
            if not resp.ok:
                raise ProviderError(
                    f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}"
                )
            try:
                return resp.json()
            except ValueError as exc:
                raise ProviderError(f"Invalid JSON from {url}: {exc}") from exc
        raise ProviderError(
            f"Massive request failed after {_MAX_RETRIES} retries: {url}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def get_prices(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """Return daily adjusted OHLCV bars for each symbol.

        Uses ``adjusted=true`` so the ``close`` field already reflects splits;
        ``adj_close`` is set to the same value.
        """
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            try:
                data = self._get(
                    f"/v2/aggs/ticker/{sym}/range/1/day/{start_str}/{end_str}",
                    params={"adjusted": "true", "sort": "asc", "limit": 50000},
                )
                results = data.get("results") or []
                if not results:
                    log.warning("massive_no_prices symbol=%s", sym)
                    continue
                df = pd.DataFrame(results)
                frames.append(
                    pd.DataFrame(
                        {
                            "date": pd.to_datetime(df["t"], unit="ms").dt.normalize(),
                            "symbol": sym,
                            "open": df["o"].astype(float),
                            "high": df["h"].astype(float),
                            "low": df["l"].astype(float),
                            "close": df["c"].astype(float),
                            "volume": df["v"].astype(float),
                            "adj_close": df["c"].astype(float),
                        }
                    )[PRICE_COLS]
                )
            except ProviderError:
                log.exception("massive_prices_failed symbol=%s", sym)
        if not frames:
            return pd.DataFrame(columns=PRICE_COLS)
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # News sentiment
    # ------------------------------------------------------------------

    def get_news_sentiment(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """Return sentiment rows from Massive news articles.

        Each article's ``insights`` array contains per-ticker sentiment labels
        (positive / negative / neutral) mapped to scores in [-1, 1].  Articles
        with no insights are recorded with score 0.0 (neutral).
        """
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
        want = {s.upper() for s in symbols}
        rows: list[dict] = []
        # Fetch per-ticker to stay within free-tier result limits.
        seen: set[str] = set()  # deduplicate articles appearing for multiple tickers
        for sym in symbols:
            try:
                self._wait_for_news_slot()
                data = self._get(
                    "/v2/reference/news",
                    params={
                        "ticker": sym,
                        "published_utc.gte": start_str,
                        "published_utc.lte": end_str,
                        "limit": 1000,
                        "sort": "published_utc",
                        "order": "asc",
                    },
                )
                for article in data.get("results") or []:
                    article_id = article.get("id", "")
                    pub = pd.to_datetime(article.get("published_utc"), errors="coerce")
                    headline = article.get("title", "")
                    source = (article.get("publisher") or {}).get("name", "massive")
                    insights = article.get("insights") or []
                    scored: dict[str, float] = {
                        str(ins.get("ticker", "")).upper(): _SENTIMENT_MAP.get(
                            str(ins.get("sentiment", "neutral")).lower(), 0.0
                        )
                        for ins in insights
                        if str(ins.get("ticker", "")).upper() in want
                    }
                    if scored:
                        for t, score in scored.items():
                            key = f"{article_id}:{t}"
                            if key in seen:
                                continue
                            seen.add(key)
                            rows.append(
                                {
                                    "date": pub.normalize() if not pd.isna(pub) else None,
                                    "symbol": t,
                                    "sentiment_score": score,
                                    "news_volume": 1,
                                    "source": source,
                                    "headline": headline,
                                }
                            )
                    elif sym.upper() in want:
                        key = f"{article_id}:{sym.upper()}"
                        if key not in seen:
                            seen.add(key)
                            rows.append(
                                {
                                    "date": pub.normalize() if not pd.isna(pub) else None,
                                    "symbol": sym.upper(),
                                    "sentiment_score": 0.0,
                                    "news_volume": 1,
                                    "source": source,
                                    "headline": headline,
                                }
                            )
            except ProviderError as exc:
                log.warning("massive_news_failed symbol=%s (%s)", sym, exc)
                if "429" in str(exc):
                    log.warning(
                        "massive_news_rate_limited — stopping batch after %s "
                        "(remaining symbols will use cache/fallback)",
                        sym,
                    )
                    break
        if not rows:
            return pd.DataFrame(columns=SENTIMENT_COLS)
        return (
            pd.DataFrame(rows, columns=SENTIMENT_COLS)
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Fundamentals (financial ratios)
    # ------------------------------------------------------------------

    def get_fundamentals(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """Return financial ratios from ``/stocks/financials/v1/ratios``.

        Mapped fields: market_cap, pe_ratio, pb_ratio, roe, debt_to_equity,
        eps, dividend_yield.  Revenue and net_income are not provided by the
        ratios endpoint and are returned as NaN; use FMPProvider for
        income-statement metrics.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if self._fundamentals_plan_blocked:
            return pd.DataFrame(columns=FUNDAMENTAL_COLS)
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            if self._fundamentals_plan_blocked:
                break
            try:
                data = self._get(
                    "/stocks/financials/v1/ratios",
                    params={"ticker": sym, "limit": 1000},
                )
                results = data.get("results") or []
                if not results:
                    log.warning("massive_no_fundamentals symbol=%s", sym)
                    continue
                rows = []
                for rec in results:
                    date_raw = rec.get("date")
                    if not date_raw:
                        continue
                    period_end = pd.Timestamp(date_raw).normalize()
                    ts = period_end + pd.Timedelta(days=FUNDAMENTALS_PUBLICATION_LAG_DAYS)
                    if not (start_ts <= ts <= end_ts):
                        continue
                    rows.append(
                        {
                            "date": ts,
                            "symbol": sym,
                            "market_cap": rec.get("market_cap"),
                            "pe_ratio": rec.get("price_to_earnings"),
                            "pb_ratio": rec.get("price_to_book"),
                            "roe": rec.get("return_on_equity"),
                            "debt_to_equity": rec.get("debt_to_equity"),
                            "revenue": None,
                            "net_income": None,
                            "eps": rec.get("earnings_per_share"),
                            "dividend_yield": rec.get("dividend_yield"),
                        }
                    )
                if rows:
                    frames.append(pd.DataFrame(rows, columns=FUNDAMENTAL_COLS))
            except ProviderError as exc:
                msg = str(exc)
                if "403" in msg and "NOT_AUTHORIZED" in msg:
                    if not type(self)._fundamentals_plan_blocked:
                        log.warning(
                            "massive_fundamentals_unavailable (plan does not include "
                            "financial ratios — skipping Massive for fundamentals)"
                        )
                        type(self)._fundamentals_plan_blocked = True
                    continue
                log.warning("massive_fundamentals_failed symbol=%s (%s)", sym, exc)
            except Exception:
                log.exception("massive_fundamentals_failed symbol=%s", sym)
        if not frames:
            return pd.DataFrame(columns=FUNDAMENTAL_COLS)
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Corporate actions
    # ------------------------------------------------------------------

    def get_corporate_actions(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """Return dividends and splits from Massive."""
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            # dividends
            try:
                div_data = self._get(
                    "/stocks/v1/dividends",
                    params={
                        "ticker": sym,
                        "ex_dividend_date.gte": start_str,
                        "ex_dividend_date.lte": end_str,
                        "limit": 5000,
                    },
                )
                for rec in div_data.get("results") or []:
                    ex = rec.get("ex_dividend_date")
                    if not ex:
                        continue
                    cash = float(rec.get("cash_amount") or 0.0)
                    frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "date": pd.Timestamp(ex).normalize(),
                                    "symbol": sym,
                                    "action_type": "dividend",
                                    "value": cash,
                                    "description": f"dividend ${cash:.4f}",
                                }
                            ]
                        )
                    )
            except ProviderError:
                log.exception("massive_dividends_failed symbol=%s", sym)
            # splits
            try:
                split_data = self._get(
                    "/stocks/v1/splits",
                    params={
                        "ticker": sym,
                        "execution_date.gte": start_str,
                        "execution_date.lte": end_str,
                        "limit": 5000,
                    },
                )
                for rec in split_data.get("results") or []:
                    ex = rec.get("execution_date")
                    if not ex:
                        continue
                    split_from = float(rec.get("split_from") or 1)
                    split_to = float(rec.get("split_to") or 1)
                    ratio = split_to / split_from if split_from else float("nan")
                    adj_type = rec.get("adjustment_type", "split")
                    frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "date": pd.Timestamp(ex).normalize(),
                                    "symbol": sym,
                                    "action_type": "split",
                                    "value": ratio,
                                    "description": f"{int(split_from)}:{int(split_to)} {adj_type}",
                                }
                            ]
                        )
                    )
            except ProviderError:
                log.exception("massive_splits_failed symbol=%s", sym)
        if not frames:
            return pd.DataFrame(columns=CORPORATE_ACTION_COLS)
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )[CORPORATE_ACTION_COLS]

    # ------------------------------------------------------------------
    # Universe constituents
    # ------------------------------------------------------------------

    def get_universe_constituents(self, index: str, date: str) -> list[str]:
        raise NotImplementedError(
            "MassiveProvider does not supply index constituents; use FMPProvider."
        )

    def get_analyst_ratings(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError(
            "MassiveProvider does not provide analyst ratings; use FMPProvider."
        )

    def get_ai_scores(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError(
            "MassiveProvider does not provide AI scores; use DanelfinProvider."
        )

    def get_live_signals(self, symbols: list[str]) -> pd.DataFrame:
        raise NotImplementedError(
            "MassiveProvider does not provide live signals; use DanelfinProvider."
        )
