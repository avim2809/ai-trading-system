"""Financial Modeling Prep (FMP) adapter – prices, fundamentals, constituents.

Docs: https://financialmodelingprep.com/developer/docs

Uses the ``/stable`` endpoint family (post-Aug 2025). Endpoints:
* Prices:         GET /stable/historical-price-eod/full
* Income stmt:    GET /stable/income-statement
* Key metrics:    GET /stable/key-metrics
* Ratios:         GET /stable/ratios
* SP500 members:  GET /stable/sp500-constituent  (premium plan required)
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd

from firm.config import Settings, get_settings
from firm.data.providers._rest import RestClient
from firm.data.providers.base import (
    FUNDAMENTALS_PUBLICATION_LAG_DAYS,
    DataProvider,
    ProviderError,
)
from firm.data.schemas import FUNDAMENTAL_COLS, PRICE_COLS
from firm.logging_setup import get_logger

log = get_logger(__name__)

_BASE_URL = "https://financialmodelingprep.com"


class FMPProvider(DataProvider):
    """Adapter for the FMP REST API (fundamentals, prices, constituents)."""

    name = "fmp"

    def __init__(self, api_key: str = "", settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._api_key = api_key or self.settings.require("fmp_api_key")
        self._client = RestClient(_BASE_URL, self.settings)

    def _params(self, **extra) -> dict:
        return {"apikey": self._api_key, **extra}

    # ------------------------------------------------------------------
    # prices
    # ------------------------------------------------------------------

    def get_prices(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
        for symbol in symbols:
            try:
                data = self._client.get_json(
                    "/stable/historical-price-eod/full",
                    params=self._params(symbol=symbol, **{"from": start_str, "to": end_str}),
                )
                records = data if isinstance(data, list) else (data.get("historical") or [])
                if not records:
                    log.warning("fmp_no_price_records symbol=%s start=%s end=%s", symbol, start_str, end_str)
                    continue
                df = pd.DataFrame(records)
                adj_col = "adjClose" if "adjClose" in df.columns else "close"
                frames.append(
                    pd.DataFrame(
                        {
                            "date": pd.to_datetime(df["date"]).dt.normalize(),
                            "symbol": symbol,
                            "open": df["open"].astype(float),
                            "high": df["high"].astype(float),
                            "low": df["low"].astype(float),
                            "close": df["close"].astype(float),
                            "volume": df["volume"].astype(float),
                            "adj_close": df[adj_col].astype(float),
                        }
                    )[PRICE_COLS]
                )
            except Exception:
                log.exception("fmp_prices_failed symbol=%s", symbol)
        if not frames:
            return self.empty_prices()
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # fundamentals
    # ------------------------------------------------------------------

    def get_fundamentals(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """Return fundamentals in FUNDAMENTAL_COLS wide format.

        Merges income statement (revenue, net_income, eps) with key metrics
        (market_cap, roe, debt_to_equity) and ratios (pe_ratio, pb_ratio,
        dividend_yield) by symbol + fiscal year.
        """
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            try:
                income = self._client.get_json(
                    "/stable/income-statement",
                    params=self._params(symbol=symbol, limit=5),
                )
                metrics = self._client.get_json(
                    "/stable/key-metrics",
                    params=self._params(symbol=symbol, limit=5),
                )
                ratios = self._client.get_json(
                    "/stable/ratios",
                    params=self._params(symbol=symbol, limit=5),
                )
                # Index by (fiscalYear, period) for merging
                income_map = {
                    (r.get("fiscalYear"), r.get("period")): r
                    for r in (income if isinstance(income, list) else [])
                }
                metrics_map = {
                    (r.get("fiscalYear"), r.get("period")): r
                    for r in (metrics if isinstance(metrics, list) else [])
                }
                for rec in ratios if isinstance(ratios, list) else []:
                    date_raw = rec.get("date")
                    if not date_raw:
                        continue
                    period_end = pd.Timestamp(date_raw).normalize()
                    ts = period_end + pd.Timedelta(days=FUNDAMENTALS_PUBLICATION_LAG_DAYS)
                    if not (start_ts <= ts <= end_ts):
                        continue
                    key = (rec.get("fiscalYear"), rec.get("period"))
                    inc = income_map.get(key, {})
                    met = metrics_map.get(key, {})
                    frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "date": ts,
                                    "symbol": symbol,
                                    "market_cap": met.get("marketCap"),
                                    "pe_ratio": rec.get("priceToEarningsRatio"),
                                    "pb_ratio": rec.get("priceToBookRatio"),
                                    "roe": rec.get("returnOnEquity"),
                                    "debt_to_equity": rec.get("debtToEquityRatio"),
                                    "revenue": inc.get("revenue"),
                                    "net_income": inc.get("netIncome"),
                                    "eps": inc.get("eps"),
                                    "dividend_yield": rec.get("dividendYield"),
                                }
                            ],
                            columns=FUNDAMENTAL_COLS,
                        )
                    )
            except Exception:
                log.exception("fmp_fundamentals_failed symbol=%s", symbol)
        if not frames:
            return self.empty_fundamentals()
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    def get_news_sentiment(
        self, symbols: Sequence[str], start: datetime | str, end: datetime | str
    ) -> pd.DataFrame:
        raise NotImplementedError("FMPProvider does not provide news sentiment; use MassiveProvider.")

    # ------------------------------------------------------------------
    # corporate actions
    # ------------------------------------------------------------------

    def get_corporate_actions(
        self, symbols: Sequence[str], start: datetime | str, end: datetime | str
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "FMPProvider corporate actions require a premium plan; use MassiveProvider."
        )

    # ------------------------------------------------------------------
    # universe
    # ------------------------------------------------------------------

    def get_universe_constituents(
        self, index: str, date: str | None = None
    ) -> list[str]:
        endpoint = {
            "sp500": "/stable/sp500-constituent",
            "nasdaq": "/stable/nasdaq-constituent",
            "dowjones": "/stable/dowjones-constituent",
        }.get(index.lower())
        if endpoint is None:
            raise NotImplementedError(f"FMPProvider has no constituent endpoint for '{index}'.")
        try:
            data = self._client.get_json(endpoint, params=self._params())
            return [r["symbol"] for r in (data if isinstance(data, list) else []) if r.get("symbol")]
        except ProviderError as exc:
            raise NotImplementedError(
                f"FMP constituent endpoint returned an error (may require premium plan): {exc}"
            ) from exc
