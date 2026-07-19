"""FRED (Federal Reserve Economic Data) macro data provider.

Fetches macroeconomic time series from the St. Louis Fed's free API.
A free API key is available at https://fred.stlouisfed.org/docs/api/api_key.html

Key series used by the trading system:
  T10Y2Y     — 10Y-2Y Treasury spread (yield curve slope), daily
  FEDFUNDS   — Effective Fed Funds Rate, monthly
  CPIAUCSL   — CPI All Urban Consumers, monthly
  VIXCLS     — CBOE Volatility Index (VIX), daily
  UNRATE     — Unemployment rate, monthly

All series are returned as a DataFrame with columns [date, value] filtered
to date <= asof by the PIT store — the same guarantee that applies to price
and fundamental data.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta

import pandas as pd
import requests

log = logging.getLogger("firm.data.providers.fred")

_BASE = "https://api.stlouisfed.org/fred"
_TIMEOUT = 30
_MAX_RETRIES = 3
_BACKOFF = 2.0

# Publication lags by frequency — how many calendar days after the period end
# before FRED releases the observation.  Used to shift observation dates
# forward so we never use data before it was actually available.
_PUBLICATION_LAG_DAYS: dict[str, int] = {
    "d": 1,    # daily: available next business day
    "w": 5,    # weekly: ~1 week
    "m": 14,   # monthly: ~2 weeks after month-end
    "q": 30,   # quarterly: ~1 month after quarter-end
    "a": 60,   # annual
}

# Human-readable aliases → FRED series IDs.
ALIASES: dict[str, str] = {
    "yield_curve": "T10Y2Y",
    "10y_2y_spread": "T10Y2Y",
    "fed_funds": "FEDFUNDS",
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "pce": "PCEPI",
    "core_pce": "PCEPILFE",
    "vix": "VIXCLS",
    "unemployment": "UNRATE",
    "unemployment_rate": "UNRATE",
    "10y_treasury": "DGS10",
    "2y_treasury": "DGS2",
    "30y_treasury": "DGS30",
    "real_gdp": "GDPC1",
    "gdp": "GDP",
    "inflation_expectations": "T10YIE",
    "nonfarm_payrolls": "PAYEMS",
    "industrial_production": "INDPRO",
}


class FREDProvider:
    """Thin wrapper around the FRED REST API.

    Args:
        api_key: FRED API key. Available for free at fred.stlouisfed.org.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def get_indicator(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch a FRED time series and return a PIT-safe DataFrame.

        Resolves human-readable aliases (e.g. "yield_curve") to FRED series
        IDs, applies a publication lag to avoid look-ahead, and returns a
        DataFrame with columns [date, series_id] — one row per observation,
        forward-filled to daily frequency so strategies can align with their
        price series.

        Args:
            series_id:  FRED series ID or a human-friendly alias.
            start_date: ISO date string, e.g. "2018-01-01".
            end_date:   ISO date string, e.g. "2024-12-31".

        Returns:
            DataFrame with columns [date, <series_id>], daily frequency.
            Empty DataFrame if the key is missing or the series is unavailable.
        """
        if not self._api_key:
            log.warning("FRED_API_KEY not set — skipping macro fetch for %s", series_id)
            return pd.DataFrame()

        series_id = ALIASES.get(series_id.lower(), series_id)

        raw = self._fetch_observations(series_id, start_date, end_date)
        if raw.empty:
            return pd.DataFrame()

        freq = self._get_frequency(series_id)
        lag_days = _PUBLICATION_LAG_DAYS.get(freq, 14)

        raw["date"] = pd.to_datetime(raw["date"]) + timedelta(days=lag_days)
        raw = raw.rename(columns={"value": series_id})

        # Forward-fill to a complete daily date range so strategies can do
        # simple date-indexed lookups without worrying about missing weekends/
        # holidays or gaps between monthly observations.
        date_range = pd.date_range(start=raw["date"].min(), end=raw["date"].max(), freq="D")
        daily = (
            raw.set_index("date")[[series_id]]
            .reindex(date_range)
            .ffill()
            .reset_index()
            .rename(columns={"index": "date"})
        )
        return daily

    def _fetch_observations(self, series_id: str, start: str, end: str) -> pd.DataFrame:
        url = f"{_BASE}/series/observations"
        params = {
            "series_id": series_id,
            "observation_start": start,
            "observation_end": end,
            "api_key": self._api_key,
            "file_type": "json",
        }
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                obs = data.get("observations", [])
                if not obs:
                    log.warning(
                        "FRED series %r returned no observations for %s..%s",
                        series_id, start, end,
                    )
                    return pd.DataFrame()
                df = pd.DataFrame(obs)[["date", "value"]]
                # FRED uses "." for missing values
                df = df[df["value"] != "."].copy()
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                return df.dropna(subset=["value"])
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 400:
                    log.warning("FRED series %r not found", series_id)
                    return pd.DataFrame()
                log.warning("FRED fetch attempt %d failed: %s", attempt + 1, exc)
            except Exception as exc:
                log.warning("FRED fetch attempt %d error: %s", attempt + 1, exc)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF ** attempt)
        return pd.DataFrame()

    def _get_frequency(self, series_id: str) -> str:
        """Return a single-char frequency code for a FRED series."""
        url = f"{_BASE}/series"
        params = {"series_id": series_id, "api_key": self._api_key, "file_type": "json"}
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            srs = resp.json().get("seriess", [{}])[0]
            freq = srs.get("frequency_short", "m").lower()[0]
            return freq
        except Exception:
            log.warning(
                "FRED frequency lookup failed for %r — defaulting to monthly "
                "(14-day) publication lag", series_id, exc_info=True,
            )
            return "m"


def fetch_macro_bundle(
    api_key: str,
    start_date: str,
    end_date: str,
    series: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch a standard bundle of macro series for backtesting.

    Returns a dict mapping each series ID to its PIT-safe daily DataFrame.
    Silently skips any series that fails (e.g. invalid key or rate limit).

    Args:
        api_key:     FRED API key.
        start_date:  Backtest start date (FRED data pulled from here).
        end_date:    Backtest end date.
        series:      List of series IDs/aliases. Defaults to the standard set.
    """
    if series is None:
        series = ["T10Y2Y", "FEDFUNDS", "CPIAUCSL", "VIXCLS", "UNRATE"]

    provider = FREDProvider(api_key)
    bundle: dict[str, pd.DataFrame] = {}
    for s in series:
        resolved = ALIASES.get(s.lower(), s)
        try:
            df = provider.get_indicator(s, start_date, end_date)
            if not df.empty:
                bundle[resolved] = df
                log.info("FRED: loaded %s (%d rows)", resolved, len(df))
        except Exception as exc:
            log.warning("FRED: could not load %s: %s", s, exc)
    return bundle
