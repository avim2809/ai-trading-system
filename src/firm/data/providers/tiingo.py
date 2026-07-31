"""Tiingo data provider – adjusted prices and news sentiment."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

from firm.config import Settings, get_settings
from firm.data.providers.base import DataProvider
from firm.data.schemas import PRICE_COLS, SENTIMENT_COLS

log = logging.getLogger("firm.data.providers.tiingo")

_BASE_URL = "https://api.tiingo.com"
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2.0


class TiingoProvider(DataProvider):
    """Tiingo REST API wrapper.

    Primary responsibilities: adjusted OHLCV prices, news sentiment.
    """

    name = "tiingo"

    def __init__(self, api_key: str = "", settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        super().__init__(api_key or self.settings.require("tiingo_api_key"))

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(1, _MAX_RETRIES + 1):
            log.debug("GET %s (attempt %d)", url, attempt)
            resp = requests.get(url, headers=self._headers(), params=params or {}, timeout=30)
            if resp.status_code == 429:
                wait = _BACKOFF_FACTOR**attempt
                log.warning("Rate-limited by Tiingo; backing off %.1fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Tiingo request failed after {_MAX_RETRIES} retries: {url}")

    def get_prices(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            try:
                data = self._get(
                    f"{_BASE_URL}/tiingo/daily/{sym}/prices",
                    params={"startDate": start, "endDate": end},
                )
                if not data:
                    log.warning("No price data for %s", sym)
                    continue
                df = pd.DataFrame(data)
                # Every other provider in the fallback chain (Massive, FMP,
                # AlphaVantage) stores "date" as a tz-naive normalized
                # Timestamp. Tiingo's raw dates carry a "Z" (UTC) suffix, so
                # a plain to_datetime() here would produce tz-*aware*
                # Timestamps — mixing those with the rest of the chain's
                # naive ones in one column breaks pyarrow's parquet writer
                # (and even a tz-agnostic normalize step, since pandas
                # refuses to compare/convert naive and aware datetimes
                # together). Parse as UTC then drop the tz to match.
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
                df["symbol"] = sym
                df = df.rename(columns={"adjClose": "adj_close"})
                for col in PRICE_COLS:
                    if col not in df.columns:
                        df[col] = None
                frames.append(df[PRICE_COLS])
            except Exception:
                log.exception("Failed to fetch Tiingo prices for %s", sym)
        if not frames:
            return pd.DataFrame(columns=PRICE_COLS)
        return pd.concat(frames, ignore_index=True)

    def get_fundamentals(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("Tiingo does not provide fundamental data; use FMP.")

    def get_news_sentiment(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            try:
                data = self._get(
                    f"{_BASE_URL}/tiingo/news",
                    params={
                        "tickers": sym,
                        "startDate": start,
                        "endDate": end,
                        "limit": 1000,
                    },
                )
                if not data:
                    log.warning("No news sentiment data for %s", sym)
                    continue
                rows = []
                for article in data:
                    rows.append(
                        {
                            "date": pd.to_datetime(article.get("publishedDate", ""), utc=True).tz_localize(None).normalize()
                            if article.get("publishedDate")
                            else None,
                            "symbol": sym,
                            "sentiment_score": 0.0,  # Tiingo IEX provides sentiment; basic news does not
                            "news_volume": 1,
                            "source": article.get("source", ""),
                            "headline": article.get("title", ""),
                        }
                    )
                frames.append(pd.DataFrame(rows, columns=SENTIMENT_COLS))
            except Exception:
                log.exception("Failed to fetch Tiingo news for %s", sym)
        if not frames:
            return pd.DataFrame(columns=SENTIMENT_COLS)
        return pd.concat(frames, ignore_index=True)

    def get_corporate_actions(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("Tiingo does not provide corporate actions; use Polygon.")

    def get_universe_constituents(self, index: str, date: str) -> list[str]:
        raise NotImplementedError("Tiingo does not provide index constituents; use Polygon or FMP.")

    def get_analyst_ratings(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("Tiingo does not provide analyst ratings; use FMP.")

    def get_ai_scores(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("Tiingo does not provide AI scores; use Danelfin.")

    def get_live_signals(self, symbols: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Tiingo does not provide live signals; use Danelfin.")
