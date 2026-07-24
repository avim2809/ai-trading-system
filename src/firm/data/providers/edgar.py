"""SEC EDGAR fundamentals via the public ``companyfacts`` JSON API.

No API key — requires a descriptive User-Agent (``SEC_EDGAR_USER_AGENT``).
Docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
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

_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_CIK_CACHE_MAX_AGE_DAYS = 7
_REQUEST_PAUSE_SECONDS = 0.12

_REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
)
_NET_INCOME_TAGS = ("NetIncomeLoss",)
_EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")
_ASSETS_TAGS = ("Assets",)
_LIABILITIES_TAGS = ("Liabilities",)
_EQUITY_TAGS = ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")


class EdgarProvider(DataProvider):
    """US SEC EDGAR XBRL fundamentals (free, no API key)."""

    name = "edgar"

    def __init__(self, api_key: str = "", settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        super().__init__(api_key or "edgar")
        self._user_agent = (
            getattr(self.settings, "sec_edgar_user_agent", "")
            or "ai-trading-system research@example.com"
        )
        self._client = RestClient("https://data.sec.gov", self.settings)
        self._cik_map: dict[str, int] | None = None

    def get_prices(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("EdgarProvider does not provide prices.")

    def get_fundamentals(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        cik_map = self._load_cik_map()
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            if symbol.upper() in ETF_SYMBOLS:
                log.debug("Skipping ETF fundamentals fetch for %s", symbol)
                continue
            cik = cik_map.get(symbol.upper())
            if cik is None:
                log.warning("edgar_no_cik symbol=%s", symbol)
                continue
            try:
                time.sleep(_REQUEST_PAUSE_SECONDS)
                payload = self._client.get_json(
                    f"/api/xbrl/companyfacts/CIK{cik:010d}.json",
                    headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                )
            except ProviderError as exc:
                log.warning("edgar_fundamentals_failed symbol=%s (%s)", symbol, exc)
                continue
            rows = _companyfacts_to_rows(symbol, payload, start_ts, end_ts)
            if rows:
                frames.append(pd.DataFrame(rows, columns=FUNDAMENTAL_COLS))
            else:
                log.warning("edgar_no_fundamentals symbol=%s", symbol)
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
        raise NotImplementedError("EdgarProvider does not provide news sentiment.")

    def get_corporate_actions(
        self, symbols: list[str], start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("EdgarProvider does not provide corporate actions.")

    def get_universe_constituents(self, index: str, date: str = "") -> list[str]:
        raise NotImplementedError("EdgarProvider does not supply index constituents.")

    def _load_cik_map(self) -> dict[str, int]:
        if self._cik_map is not None:
            return self._cik_map
        cache_dir = Path(self.settings.data.cache_dir)
        cache_path = cache_dir / "sec" / "company_tickers.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stale = True
        if cache_path.exists():
            age_days = (
                time.time() - cache_path.stat().st_mtime
            ) / 86400.0
            stale = age_days > _CIK_CACHE_MAX_AGE_DAYS
        if stale or not cache_path.exists():
            try:
                log.info("Refreshing SEC ticker→CIK map")
                resp_client = RestClient("https://www.sec.gov", self.settings)
                raw = resp_client.get_json(
                    "/files/company_tickers.json",
                    headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                )
                with open(cache_path, "w") as f:
                    json.dump(raw, f)
            except ProviderError as exc:
                log.warning("SEC CIK map refresh failed (%s); using cache if present", exc)
        if cache_path.exists():
            with open(cache_path) as f:
                raw = json.load(f)
            self._cik_map = _parse_cik_map(raw)
            return self._cik_map
        self._cik_map = {}
        return self._cik_map


def _parse_cik_map(raw: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    if isinstance(raw, dict):
        for entry in raw.values():
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker", "")).upper()
            cik = entry.get("cik_str") or entry.get("cik")
            if ticker and cik is not None:
                out[ticker] = int(cik)
    return out


def _companyfacts_to_rows(
    symbol: str,
    payload: dict[str, Any],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict[str, Any]]:
    facts = (payload.get("facts") or {}).get("us-gaap") or {}
    revenue = _series_by_period(facts, _REVENUE_TAGS, unit="USD")
    net_income = _series_by_period(facts, _NET_INCOME_TAGS, unit="USD")
    eps = _series_by_period(facts, _EPS_TAGS, unit="USD/shares")
    assets = _series_by_period(facts, _ASSETS_TAGS, unit="USD")
    equity = _series_by_period(facts, _EQUITY_TAGS, unit="USD")
    liabilities = _series_by_period(facts, _LIABILITIES_TAGS, unit="USD")

    periods = sorted(
        set(revenue) | set(net_income) | set(eps) | set(assets) | set(equity)
    )
    rows: list[dict[str, Any]] = []
    for period_end in periods:
        ts = pd.Timestamp(period_end) + pd.Timedelta(days=FUNDAMENTALS_PUBLICATION_LAG_DAYS)
        if not (start_ts <= ts <= end_ts):
            continue
        rev = revenue.get(period_end)
        ni = net_income.get(period_end)
        eps_v = eps.get(period_end)
        eq = equity.get(period_end)
        liab = liabilities.get(period_end)
        debt_to_equity = None
        if eq and eq != 0 and liab is not None:
            debt_to_equity = float(liab) / float(eq)
        roe = None
        if eq and eq != 0 and ni is not None:
            roe = float(ni) / float(eq)
        rows.append({
            "date": ts,
            "symbol": symbol,
            "market_cap": None,
            "pe_ratio": None,
            "pb_ratio": None,
            "roe": roe,
            "debt_to_equity": debt_to_equity,
            "revenue": rev,
            "net_income": ni,
            "eps": eps_v,
            "dividend_yield": None,
        })
    return rows


def _series_by_period(
    facts: dict[str, Any],
    tags: tuple[str, ...],
    *,
    unit: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for tag in tags:
        units = (facts.get(tag) or {}).get("units") or {}
        entries = units.get(unit) or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            end = entry.get("end")
            val = entry.get("val")
            form = str(entry.get("form", ""))
            if not end or val is None:
                continue
            if form not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                continue
            key = str(end)[:10]
            out[key] = float(val)
        if out:
            break
    return out
