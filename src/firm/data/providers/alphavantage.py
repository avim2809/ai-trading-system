"""Alpha Vantage adapter - news & sentiment (and EOD prices as a fallback).

Docs: https://www.alphavantage.co/documentation/. Endpoints used:

* News sentiment: ``function=NEWS_SENTIMENT``
* Daily adjusted: ``function=TIME_SERIES_DAILY_ADJUSTED``
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd

from firm.config import Settings, get_settings
from firm.data.providers._rest import RestClient
from firm.data.providers.base import DataProvider, ProviderError
from firm.logging_setup import get_logger

log = get_logger(__name__)

_BASE_URL = "https://www.alphavantage.co"


class AlphaVantageProvider(DataProvider):
    """Adapter for the Alpha Vantage REST API (news sentiment, prices)."""

    name = "alphavantage"

    def __init__(self, api_key: str = "", settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._api_key = api_key or self.settings.require("alphavantage_api_key")
        self._client = RestClient(_BASE_URL, self.settings)

    def get_prices(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        for symbol in symbols:
            payload = self._client.get_json(
                "/query",
                params={
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": symbol,
                    "outputsize": "full",
                    "apikey": self._api_key,
                },
            )
            series = payload.get("Time Series (Daily)")
            if not series:
                log.info(
                    "alphavantage_no_prices",
                    extra={"context": {"symbol": symbol, "note": payload.get("Note", "")}},
                )
                continue
            records = []
            for day, vals in series.items():
                ts = pd.Timestamp(day)
                if not (start_ts <= ts <= end_ts):
                    continue
                records.append(
                    {
                        "date": ts.normalize(),
                        "symbol": symbol,
                        "open": float(vals["1. open"]),
                        "high": float(vals["2. high"]),
                        "low": float(vals["3. low"]),
                        "close": float(vals["4. close"]),
                        "adj_close": float(vals["5. adjusted close"]),
                        "volume": float(vals["6. volume"]),
                    }
                )
            if records:
                frames.append(pd.DataFrame(records))
        if not frames:
            return self.empty_prices()
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    def get_fundamentals(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        from firm.data.providers.base import FUNDAMENTALS_PUBLICATION_LAG_DAYS
        from firm.data.providers.constants import ETF_SYMBOLS
        from firm.data.schemas import FUNDAMENTAL_COLS

        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            if symbol.upper() in ETF_SYMBOLS:
                continue
            try:
                payload = self._client.get_json(
                    "/query",
                    params={
                        "function": "OVERVIEW",
                        "symbol": symbol,
                        "apikey": self._api_key,
                    },
                )
            except ProviderError as exc:
                log.warning("alphavantage_fundamentals_failed symbol=%s (%s)", symbol, exc)
                continue
            if "Information" in payload or "Note" in payload:
                log.warning(
                    "alphavantage_fundamentals_unavailable symbol=%s (%s)",
                    symbol,
                    payload.get("Information") or payload.get("Note"),
                )
                continue
            quarter = payload.get("LatestQuarter")
            if not quarter:
                log.warning("alphavantage_no_fundamentals symbol=%s", symbol)
                continue
            period_end = pd.Timestamp(quarter)
            ts = period_end + pd.Timedelta(days=FUNDAMENTALS_PUBLICATION_LAG_DAYS)
            if not (start_ts <= ts <= end_ts):
                continue
            frames.append(pd.DataFrame([{
                "date": ts,
                "symbol": symbol,
                "market_cap": _av_float(payload.get("MarketCapitalization")),
                "pe_ratio": _av_float(payload.get("PERatio")),
                "pb_ratio": _av_float(payload.get("PriceToBookRatio")),
                "roe": _av_float(payload.get("ReturnOnEquityTTM")),
                "debt_to_equity": _av_float(payload.get("DebtToEquity")),
                "revenue": None,
                "net_income": None,
                "eps": _av_float(payload.get("EPS")),
                "dividend_yield": _av_float(payload.get("DividendYield")),
            }], columns=FUNDAMENTAL_COLS))
        if not frames:
            return self.empty_fundamentals()
        return pd.concat(frames, ignore_index=True).reset_index(drop=True)

    def get_news_sentiment(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        start_str = pd.Timestamp(start).strftime("%Y%m%dT%H%M")
        end_str = pd.Timestamp(end).strftime("%Y%m%dT%H%M")
        rows: list[dict] = []
        want = {s.upper() for s in symbols}
        # Alpha Vantage accepts up to 50 tickers per call; chunk to be safe.
        for chunk in _chunks(list(symbols), 50):
            payload = self._client.get_json(
                "/query",
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": ",".join(chunk),
                    "time_from": start_str,
                    "time_to": end_str,
                    "limit": 1000,
                    "apikey": self._api_key,
                },
            )
            if "Information" in payload or "Note" in payload:
                raise ProviderError(
                    f"Alpha Vantage rate limit/info: {payload.get('Information') or payload.get('Note')}"
                )
            for art in payload.get("feed", []) or []:
                published = _parse_av_time(art.get("time_published"))
                for ts_obj in art.get("ticker_sentiment", []) or []:
                    sym = str(ts_obj.get("ticker", "")).upper()
                    if sym not in want:
                        continue
                    rows.append(
                        {
                            "date": published,
                            "symbol": sym,
                            "sentiment_score": float(
                                ts_obj.get("ticker_sentiment_score", 0.0) or 0.0
                            ),
                            # Keep sentiment schema stable across providers;
                            # relevance and URL are provider-specific extras.
                            "news_volume": 1,
                            "source": art.get("source", "alphavantage"),
                            "headline": art.get("title", ""),
                        }
                    )
        if not rows:
            return self.empty_news()
        return (
            pd.DataFrame(rows)
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    def get_corporate_actions(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "AlphaVantageProvider does not expose discrete corporate actions; use MassiveProvider."
        )

    def get_universe_constituents(
        self, index: str, asof: datetime | None = None
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "AlphaVantageProvider does not supply index constituents; use FMPProvider."
        )

    def get_analyst_ratings(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "AlphaVantageProvider does not provide analyst ratings; use FMPProvider."
        )

    def get_ai_scores(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "AlphaVantageProvider does not provide AI scores; use DanelfinProvider."
        )

    def get_live_signals(self, symbols: Sequence[str]) -> pd.DataFrame:
        raise NotImplementedError(
            "AlphaVantageProvider does not provide live signals; use DanelfinProvider."
        )

    def get_best_stocks(self) -> pd.DataFrame:
        raise NotImplementedError(
            "AlphaVantageProvider does not provide best-stocks; use DanelfinProvider."
        )


def _parse_av_time(value: str | None) -> pd.Timestamp:
    """Parse Alpha Vantage ``YYYYMMDDTHHMMSS`` timestamps."""
    if not value:
        return pd.NaT
    try:
        return pd.Timestamp(datetime.strptime(value, "%Y%m%dT%H%M%S"))
    except ValueError:
        return pd.to_datetime(value, errors="coerce")


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _av_float(value: str | None) -> float | None:
    if value in (None, "", "None", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
