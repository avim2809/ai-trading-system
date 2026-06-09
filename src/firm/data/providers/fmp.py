"""Financial Modeling Prep (FMP) adapter - fundamentals, earnings, constituents.

Docs: https://site.financialmodelingprep.com/developer/docs. Endpoints used:

* Income statement:  ``/api/v3/income-statement/{symbol}``
* Key metrics:       ``/api/v3/key-metrics/{symbol}``
* Earnings calendar: ``/api/v3/historical/earning_calendar/{symbol}``
* Splits/dividends:  ``/api/v3/historical-price-full/stock_split/{symbol}`` etc.
* Constituents:      ``/api/v3/sp500_constituent`` and ``.../historical/...``

Fundamentals use the SEC **filing/accepted date** as the point-in-time ``asof``
column where available (falling back to the period end date) to avoid look-ahead.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd

from firm.config import Settings, get_settings
from firm.data import schemas
from firm.data.providers._rest import RestClient
from firm.data.providers.base import DataProvider
from firm.logging_setup import get_logger

log = get_logger(__name__)

_BASE_URL = "https://financialmodelingprep.com"

# Tidy/long fundamental metrics pulled from income-statement + key-metrics.
_INCOME_METRICS = ("revenue", "netIncome", "eps", "operatingIncome", "grossProfit")
_KEY_METRICS = ("peRatio", "pbRatio", "roe", "debtToEquity", "freeCashFlowPerShare")


class FMPProvider(DataProvider):
    """Adapter for the FMP REST API (fundamentals, earnings, constituents)."""

    name = "fmp"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._api_key = self.settings.require("fmp_api_key")
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
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
        for symbol in symbols:
            payload = self._client.get_json(
                f"/api/v3/historical-price-full/{symbol}",
                params={"from": start_str, "to": end_str, "apikey": self._api_key},
            )
            hist = payload.get("historical") if isinstance(payload, dict) else None
            if not hist:
                continue
            df = pd.DataFrame(hist)
            out = pd.DataFrame(
                {
                    schemas.COL_DATE: pd.to_datetime(df["date"]).dt.normalize(),
                    schemas.COL_SYMBOL: symbol,
                    schemas.COL_OPEN: df["open"].astype(float),
                    schemas.COL_HIGH: df["high"].astype(float),
                    schemas.COL_LOW: df["low"].astype(float),
                    schemas.COL_CLOSE: df["close"].astype(float),
                    schemas.COL_ADJ_CLOSE: (
                        df["adjClose"] if "adjClose" in df else df["close"]
                    ).astype(float),
                    schemas.COL_VOLUME: df["volume"].astype(float),
                }
            )
            frames.append(out)
        if not frames:
            return self.empty_prices()
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values([schemas.COL_SYMBOL, schemas.COL_DATE])
            .reset_index(drop=True)
        )

    # --- fundamentals --------------------------------------------------------
    def get_fundamentals(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        rows: list[dict] = []
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        for symbol in symbols:
            income = self._client.get_json(
                f"/api/v3/income-statement/{symbol}",
                params={"period": "quarter", "limit": 80, "apikey": self._api_key},
            )
            rows.extend(self._tidy_statement(symbol, income, _INCOME_METRICS, start_ts, end_ts))
            metrics = self._client.get_json(
                f"/api/v3/key-metrics/{symbol}",
                params={"period": "quarter", "limit": 80, "apikey": self._api_key},
            )
            rows.extend(self._tidy_statement(symbol, metrics, _KEY_METRICS, start_ts, end_ts))
        if not rows:
            return self.empty_fundamentals()
        return (
            pd.DataFrame(rows)
            .sort_values([schemas.COL_SYMBOL, schemas.COL_ASOF, schemas.COL_METRIC])
            .reset_index(drop=True)
        )

    @staticmethod
    def _tidy_statement(
        symbol: str,
        payload: object,
        metrics: tuple[str, ...],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> list[dict]:
        if not isinstance(payload, list):
            return []
        out: list[dict] = []
        for rec in payload:
            period_end = pd.Timestamp(rec.get("date"))
            # Prefer the SEC accepted/filing date as point-in-time availability.
            asof_raw = rec.get("acceptedDate") or rec.get("fillingDate") or rec.get("date")
            asof = pd.Timestamp(asof_raw)
            if pd.isna(asof) or not (start_ts <= asof <= end_ts):
                continue
            period_label = rec.get("period")
            period = (
                f"{period_end.year}-{period_label}"
                if period_label
                else (period_end.strftime("%Y-%m-%d") if not pd.isna(period_end) else "")
            )
            for metric in metrics:
                if metric not in rec or rec[metric] is None:
                    continue
                try:
                    value = float(rec[metric])
                except (TypeError, ValueError):
                    continue
                out.append(
                    {
                        schemas.COL_SYMBOL: symbol,
                        schemas.COL_PERIOD: period,
                        schemas.COL_ASOF: asof.normalize(),
                        schemas.COL_METRIC: metric,
                        schemas.COL_VALUE: value,
                    }
                )
        return out

    def get_news_sentiment(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "FMPProvider news scoring is not wired; use AlphaVantageProvider."
        )

    # --- corporate actions ---------------------------------------------------
    def get_corporate_actions(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        for symbol in symbols:
            splits = self._client.get_json(
                f"/api/v3/historical-price-full/stock_split/{symbol}",
                params={"apikey": self._api_key},
            )
            for s in (splits.get("historical", []) if isinstance(splits, dict) else []):
                ex = pd.Timestamp(s.get("date"))
                if pd.isna(ex) or not (start_ts <= ex <= end_ts):
                    continue
                num = float(s.get("numerator", 1) or 1)
                den = float(s.get("denominator", 1) or 1)
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                schemas.COL_SYMBOL: symbol,
                                schemas.COL_EX_DATE: ex.normalize(),
                                schemas.COL_ACTION_TYPE: "split",
                                schemas.COL_RATIO: num / den if den else float("nan"),
                                schemas.COL_CASH_AMOUNT: float("nan"),
                            }
                        ]
                    )
                )
            divs = self._client.get_json(
                f"/api/v3/historical-price-full/stock_dividend/{symbol}",
                params={"apikey": self._api_key},
            )
            for d in (divs.get("historical", []) if isinstance(divs, dict) else []):
                ex = pd.Timestamp(d.get("date"))
                if pd.isna(ex) or not (start_ts <= ex <= end_ts):
                    continue
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                schemas.COL_SYMBOL: symbol,
                                schemas.COL_EX_DATE: ex.normalize(),
                                schemas.COL_ACTION_TYPE: "dividend",
                                schemas.COL_RATIO: float("nan"),
                                schemas.COL_CASH_AMOUNT: float(d.get("dividend", 0.0) or 0.0),
                            }
                        ]
                    )
                )
        if not frames:
            return self.empty_corporate_actions()
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values([schemas.COL_SYMBOL, schemas.COL_EX_DATE])
            .reset_index(drop=True)
        )

    # --- universe ------------------------------------------------------------
    def get_universe_constituents(
        self, index: str, asof: datetime | None = None
    ) -> pd.DataFrame:
        endpoint = {
            "sp500": "/api/v3/sp500_constituent",
            "nasdaq": "/api/v3/nasdaq_constituent",
            "dowjones": "/api/v3/dowjones_constituent",
        }.get(index.lower())
        if endpoint is None:
            raise NotImplementedError(f"FMPProvider has no constituent endpoint for '{index}'.")

        current = self._client.get_json(endpoint, params={"apikey": self._api_key})
        rows: list[dict] = []
        for c in current if isinstance(current, list) else []:
            rows.append(
                {
                    schemas.COL_INDEX: index,
                    schemas.COL_SYMBOL: c.get("symbol"),
                    schemas.COL_ADDED_DATE: pd.Timestamp(c.get("dateFirstAdded"))
                    if c.get("dateFirstAdded")
                    else pd.NaT,
                    schemas.COL_REMOVED_DATE: pd.NaT,
                }
            )
        # Historical add/remove events restore survivorship history.
        historical = self._client.get_json(
            f"{endpoint}/historical", params={"apikey": self._api_key}
        )
        for h in historical if isinstance(historical, list) else []:
            symbol = h.get("symbol")
            change_date = pd.Timestamp(h.get("date")) if h.get("date") else pd.NaT
            if str(h.get("removedTicker")):
                rows.append(
                    {
                        schemas.COL_INDEX: index,
                        schemas.COL_SYMBOL: h.get("removedTicker") or symbol,
                        schemas.COL_ADDED_DATE: pd.NaT,
                        schemas.COL_REMOVED_DATE: change_date,
                    }
                )
        if not rows:
            return self.empty_universe()
        df = pd.DataFrame(rows)
        if asof is not None:
            asof_ts = pd.Timestamp(asof)
            df = df[
                (df[schemas.COL_ADDED_DATE].isna() | (df[schemas.COL_ADDED_DATE] <= asof_ts))
                & (df[schemas.COL_REMOVED_DATE].isna() | (df[schemas.COL_REMOVED_DATE] > asof_ts))
            ]
        return df.reset_index(drop=True)
