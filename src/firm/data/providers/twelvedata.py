"""Twelve Data adapter — statistics / profile fundamentals (free tier).

Docs: https://twelvedata.com/docs
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
from firm.data.schemas import FUNDAMENTAL_COLS

log = logging.getLogger(__name__)

_BASE_URL = "https://api.twelvedata.com"


class TwelveDataProvider(DataProvider):
    """Twelve Data REST adapter (fundamentals snapshot via ``/statistics``)."""

    name = "twelvedata"

    def __init__(self, api_key: str = "", settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._api_key = api_key or self.settings.require("twelvedata_api_key")
        super().__init__(self._api_key)
        self._client = RestClient(_BASE_URL, self.settings)

    def get_prices(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("TwelveDataProvider prices use FallbackProvider.")

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
                continue
            try:
                stats = self._client.get_json(
                    "/statistics",
                    params={"symbol": symbol, "apikey": self._api_key},
                )
            except ProviderError as exc:
                log.warning("twelvedata_fundamentals_failed symbol=%s (%s)", symbol, exc)
                continue
            row = _statistics_to_row(symbol, stats, start_ts, end_ts)
            if row:
                frames.append(pd.DataFrame([row], columns=FUNDAMENTAL_COLS))
            else:
                log.warning("twelvedata_no_fundamentals symbol=%s", symbol)
        if not frames:
            return self.empty_fundamentals()
        return pd.concat(frames, ignore_index=True)

    def get_news_sentiment(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("TwelveDataProvider sentiment is not wired.")

    def get_corporate_actions(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("TwelveDataProvider corporate actions are not wired.")

    def get_universe_constituents(self, index: str, date: str = "") -> list[str]:
        raise NotImplementedError("TwelveDataProvider does not supply index constituents.")

    def get_analyst_ratings(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("TwelveDataProvider does not provide analyst ratings; use FMP.")

    def get_ai_scores(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("TwelveDataProvider does not provide AI scores; use Danelfin.")


def _statistics_to_row(
    symbol: str,
    payload: dict[str, Any],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict[str, Any] | None:
    valuations = payload.get("valuations") or {}
    financials = payload.get("financials") or {}
    dividends = payload.get("dividends_and_splits") or {}
    period_raw = (
        financials.get("fiscal_year_end")
        or payload.get("meta", {}).get("fiscal_year_end")
    )
    if period_raw:
        period_end = pd.Timestamp(str(period_raw))
    else:
        period_end = pd.Timestamp.now().normalize()
    ts = period_end + pd.Timedelta(days=FUNDAMENTALS_PUBLICATION_LAG_DAYS)
    if not (start_ts <= ts <= end_ts):
        return None
    return {
        "date": ts,
        "symbol": symbol,
        "market_cap": _float_or_none(valuations.get("market_capitalization")),
        "pe_ratio": _float_or_none(valuations.get("trailing_pe")),
        "pb_ratio": _float_or_none(valuations.get("price_to_book")),
        "roe": _float_or_none(financials.get("return_on_equity_ttm")),
        "debt_to_equity": _float_or_none(financials.get("debt_to_equity")),
        "revenue": _float_or_none(financials.get("revenue_ttm")),
        "net_income": _float_or_none(financials.get("net_income_ttm")),
        "eps": _float_or_none(valuations.get("trailing_eps")),
        "dividend_yield": _float_or_none(dividends.get("forward_dividend_yield")),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
