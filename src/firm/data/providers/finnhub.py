"""Finnhub adapter — company metrics and financials.

Docs: https://finnhub.io/docs/api/company-basic-financials
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Sequence

import pandas as pd

from firm.config import Settings, get_settings
from firm.data.providers._rest import RestClient
from firm.data.providers.base import (
    FUNDAMENTALS_PUBLICATION_LAG_DAYS,
    DataProvider,
    ProviderError,
)
from firm.data.providers.constants import ETF_SYMBOLS
from firm.data.providers.sentiment_lexicon import score_headline
from firm.data.schemas import FUNDAMENTAL_COLS, SENTIMENT_COLS

log = logging.getLogger(__name__)

_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubProvider(DataProvider):
    """Finnhub REST adapter (fundamentals via ``/stock/metric``)."""

    name = "finnhub"

    def __init__(self, api_key: str = "", settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._api_key = api_key or self.settings.require("finnhub_api_key")
        super().__init__(self._api_key)
        self._client = RestClient(_BASE_URL, self.settings)

    def get_prices(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("FinnhubProvider does not provide prices; use FallbackProvider.")

    def get_fundamentals(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            if symbol.upper() in ETF_SYMBOLS:
                log.debug("Skipping ETF fundamentals fetch for %s", symbol)
                continue
            try:
                payload = self._client.get_json(
                    "/stock/metric",
                    params={"symbol": symbol, "metric": "all", "token": self._api_key},
                )
            except ProviderError as exc:
                log.warning("finnhub_fundamentals_failed symbol=%s (%s)", symbol, exc)
                continue
            rows = _metric_payload_to_rows(symbol, payload, start_ts, end_ts)
            if rows:
                frames.append(pd.DataFrame(rows, columns=FUNDAMENTAL_COLS))
            else:
                log.warning("finnhub_no_fundamentals symbol=%s", symbol)
        if not frames:
            return self.empty_fundamentals()
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    def get_news_sentiment(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Return sentiment rows from Finnhub ``/company-news`` headlines."""
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
        rows: list[dict] = []
        for sym in symbols:
            try:
                articles = self._client.get_json(
                    "/company-news",
                    params={
                        "symbol": sym,
                        "from": start_str,
                        "to": end_str,
                        "token": self._api_key,
                    },
                )
            except ProviderError as exc:
                log.warning("finnhub_news_failed symbol=%s (%s)", sym, exc)
                continue
            if not isinstance(articles, list):
                log.warning("finnhub_news_unexpected_payload symbol=%s", sym)
                continue
            for art in articles:
                ts = pd.to_datetime(art.get("datetime"), unit="s", errors="coerce")
                headline = art.get("headline") or art.get("summary") or ""
                rows.append(
                    {
                        "date": ts.date() if not pd.isna(ts) else None,
                        "symbol": sym.upper(),
                        "sentiment_score": score_headline(headline),
                        "news_volume": 1,
                        "source": art.get("source") or "finnhub",
                        "headline": headline,
                    }
                )
        if not rows:
            return pd.DataFrame(columns=SENTIMENT_COLS)
        df = pd.DataFrame(rows, columns=SENTIMENT_COLS)
        df = df.dropna(subset=["date"])
        if df.empty:
            return pd.DataFrame(columns=SENTIMENT_COLS)
        return (
            df.groupby(["date", "symbol"], as_index=False)
            .agg(
                sentiment_score=("sentiment_score", "mean"),
                news_volume=("news_volume", "sum"),
                source=("source", "first"),
                headline=("headline", "first"),
            )[SENTIMENT_COLS]
        )

    def get_corporate_actions(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("FinnhubProvider corporate actions are not wired.")

    def get_universe_constituents(self, index: str, date: str = "") -> list[str]:
        raise NotImplementedError("FinnhubProvider does not supply index constituents.")

    def get_analyst_ratings(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("FinnhubProvider does not provide analyst ratings; use FMP.")

    def get_ai_scores(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("FinnhubProvider does not provide AI scores; use Danelfin.")


def _metric_payload_to_rows(
    symbol: str,
    payload: dict[str, Any],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict[str, Any]]:
    series = (payload.get("series") or {}).get("quarterly") or {}
    if not series:
        metric = payload.get("metric") or {}
        latest = metric.get("series") or {}
        period = latest.get("period") or metric.get("52WeekHighDate")
        if period:
            ts = pd.Timestamp(period) + pd.Timedelta(days=FUNDAMENTALS_PUBLICATION_LAG_DAYS)
            if start_ts <= ts <= end_ts:
                return [_snapshot_row(symbol, ts, metric)]
        return []

    metric_keys = {
        "pe_ratio": ("pe", "peTTM"),
        "pb_ratio": ("pb", "pbAnnual"),
        "roe": ("roe", "roeTTM"),
        "eps": ("eps", "epsTTM"),
        "dividend_yield": ("dividendYield", "dividendYieldIndicatedAnnual"),
    }
    periods: set[str] = set()
    for values in series.values():
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict) and item.get("period"):
                    periods.add(str(item["period"]))
    rows: list[dict[str, Any]] = []
    for period in sorted(periods):
        period_end = pd.Timestamp(period).normalize()
        ts = period_end + pd.Timedelta(days=FUNDAMENTALS_PUBLICATION_LAG_DAYS)
        if not (start_ts <= ts <= end_ts):
            continue
        row: dict[str, Any] = {
            "date": ts,
            "symbol": symbol,
            "market_cap": None,
            "pe_ratio": None,
            "pb_ratio": None,
            "roe": None,
            "debt_to_equity": None,
            "revenue": None,
            "net_income": None,
            "eps": None,
            "dividend_yield": None,
        }
        for col, keys in metric_keys.items():
            for key in keys:
                values = series.get(key)
                if not isinstance(values, list):
                    continue
                for item in values:
                    if str(item.get("period")) == period:
                        row[col] = item.get("v")
                        break
        rows.append(row)
    return rows


def _snapshot_row(symbol: str, ts: pd.Timestamp, metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": ts,
        "symbol": symbol,
        "market_cap": metric.get("marketCapitalization"),
        "pe_ratio": metric.get("peTTM") or metric.get("peNormalizedAnnual"),
        "pb_ratio": metric.get("pbAnnual"),
        "roe": metric.get("roeTTM"),
        "debt_to_equity": metric.get("totalDebt/totalEquityAnnual"),
        "revenue": None,
        "net_income": None,
        "eps": metric.get("epsTTM"),
        "dividend_yield": metric.get("dividendYieldIndicatedAnnual"),
    }
