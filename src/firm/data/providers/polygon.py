"""Polygon.io data provider – prices, corporate actions, universe constituents."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

from firm.data.providers.base import DataProvider
from firm.data.schemas import CORPORATE_ACTION_COLS, PRICE_COLS

log = logging.getLogger("firm.data.providers.polygon")

_BASE_URL = "https://api.polygon.io"
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2.0


class PolygonProvider(DataProvider):
    """Polygon.io REST API wrapper.

    Primary responsibilities: adjusted OHLCV prices, corporate actions, and
    index constituent snapshots.
    """

    name = "polygon"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        params["apiKey"] = self.api_key
        url = f"{_BASE_URL}{path}"

        for attempt in range(1, _MAX_RETRIES + 1):
            log.debug("GET %s (attempt %d)", url, attempt)
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = _BACKOFF_FACTOR**attempt
                log.warning("Rate-limited by Polygon; backing off %.1fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()

        raise RuntimeError(f"Polygon request failed after {_MAX_RETRIES} retries: {url}")

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def get_prices(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            try:
                data = self._get(
                    f"/v2/aggs/ticker/{sym}/range/1/day/{start}/{end}",
                    params={"adjusted": "true", "sort": "asc", "limit": 50000},
                )
                results = data.get("results", [])
                if not results:
                    log.warning("No price data for %s", sym)
                    continue
                df = pd.DataFrame(results)
                df = df.rename(
                    columns={
                        "t": "date",
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume",
                    }
                )
                df["date"] = pd.to_datetime(df["date"], unit="ms").dt.date
                df["symbol"] = sym
                df["adj_close"] = df["close"]
                frames.append(df[PRICE_COLS])
            except Exception:
                log.exception("Failed to fetch prices for %s", sym)
        if not frames:
            return pd.DataFrame(columns=PRICE_COLS)
        return pd.concat(frames, ignore_index=True)

    def get_fundamentals(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("Polygon does not provide fundamental data; use FMP.")

    def get_news_sentiment(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("Polygon does not provide sentiment data; use Tiingo.")

    def get_corporate_actions(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            try:
                # Dividends
                div_data = self._get(
                    f"/v3/reference/dividends",
                    params={"ticker": sym, "ex_dividend_date.gte": start, "ex_dividend_date.lte": end, "limit": 1000},
                )
                for rec in div_data.get("results", []):
                    frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "date": rec.get("ex_dividend_date"),
                                    "symbol": sym,
                                    "action_type": "dividend",
                                    "value": rec.get("cash_amount", 0),
                                    "description": f"dividend ${rec.get('cash_amount', 0)}",
                                }
                            ]
                        )
                    )

                # Splits
                split_data = self._get(
                    f"/v3/reference/splits",
                    params={"ticker": sym, "execution_date.gte": start, "execution_date.lte": end, "limit": 1000},
                )
                for rec in split_data.get("results", []):
                    ratio = rec.get("split_to", 1) / max(rec.get("split_from", 1), 1)
                    frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "date": rec.get("execution_date"),
                                    "symbol": sym,
                                    "action_type": "split",
                                    "value": ratio,
                                    "description": f"{rec.get('split_from')}:{rec.get('split_to')} split",
                                }
                            ]
                        )
                    )
            except Exception:
                log.exception("Failed to fetch corporate actions for %s", sym)

        if not frames:
            return pd.DataFrame(columns=CORPORATE_ACTION_COLS)
        return pd.concat(frames, ignore_index=True)[CORPORATE_ACTION_COLS]

    def get_universe_constituents(self, index: str, date: str) -> list[str]:
        try:
            data = self._get(
                "/v3/reference/tickers",
                params={
                    "market": "stocks",
                    "exchange": "XNYS,XNAS",
                    "active": "true",
                    "limit": 1000,
                    "sort": "ticker",
                    "date": date,
                },
            )
            return [r["ticker"] for r in data.get("results", [])]
        except Exception:
            log.exception("Failed to fetch universe constituents")
            return []
