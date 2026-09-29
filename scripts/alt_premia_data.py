"""Fetch and cache the raw inputs for the alternative-premia edge search.

See docs/edge_search_plan_2026_09.md. Every source is free:

- Tiingo daily ``adjClose`` (total return) for ETFs/ETNs, Tiingo crypto daily for BTC
- CBOE index history CSVs (VIX, VIX3M, PUT)
- FRED ``DTB3`` (3-month T-bill, secondary market, % annualised)
- federalreserve.gov FOMC calendars (scheduled meetings only)

Raw responses are written once to ``--out`` (parquet/CSV) so evaluation runs are
reproducible offline and never re-hit the network. Nothing here computes any
candidate's returns.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

log = logging.getLogger(__name__)

ETFS = ["SPY", "IEF", "EFA", "EEM", "TLT", "GLD", "DBC", "VNQ", "VXX", "SVXY"]
CBOE = ["VIX", "VIX3M", "PUT"]
# 2003-09-15 is listed as a "Meeting" on the historical page but the regular
# scheduled meeting that year was 2003-09-16 (the 8 regular meetings are Jan 29,
# Mar 18, May 6, Jun 25, Aug 12, Sep 16, Oct 28, Dec 9).
_NOT_SCHEDULED = {pd.Timestamp("2003-09-15")}
UA = {"User-Agent": "Mozilla/5.0 (research; edge-search)"}
_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}


def _env(key: str) -> str:
    val = os.environ.get(key) or dotenv_values(".env").get(key)
    if not val:
        raise SystemExit(f"{key} not set")
    return val


def fetch_tiingo(ticker: str, token: str) -> pd.DataFrame:
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices"
    r = requests.get(url, params={"startDate": "1990-01-01", "token": token}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    log.info("tiingo %s: %d rows %s..%s", ticker, len(df), df.date.min().date(), df.date.max().date())
    return df[["date", "close", "adjClose", "divCash", "splitFactor", "volume"]]


def fetch_tiingo_crypto(ticker: str, token: str) -> pd.DataFrame:
    frames = []
    for start, end in [("2014-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]:
        r = requests.get(
            "https://api.tiingo.com/tiingo/crypto/prices",
            params={"tickers": ticker, "startDate": start, "endDate": end,
                    "resampleFreq": "1day", "token": token},
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        if payload:
            frames.append(pd.DataFrame(payload[0]["priceData"]))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date")
    log.info("tiingo crypto %s: %d rows %s..%s", ticker, len(df), df.date.min().date(), df.date.max().date())
    return df[["date", "open", "high", "low", "close", "volume"]]


def fetch_cboe(symbol: str) -> pd.DataFrame:
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
    r = requests.get(url, headers=UA, timeout=60, allow_redirects=True)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    log.info("cboe %s: %d rows %s..%s", symbol, len(df), df.date.min().date(), df.date.max().date())
    return df


def fetch_fred(series: str, key: str) -> pd.DataFrame:
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": series, "api_key": key, "file_type": "json"},
        timeout=60,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    log.info("fred %s: %d rows", series, len(df))
    return df


def _parse_historical(year: int, html: str) -> list[dict]:
    out = []
    for m in re.finditer(r"<h5[^>]*>([^<]*)</h5>", html):
        text = m.group(1).strip()
        # "March 18 Meeting - 2003", "June 24-25 Meeting", "April/May 30-1 Meeting",
        # "Jan/Feb 31-1 Meeting". Conference calls / "(unscheduled)" never match.
        # Also "January 31-February 1 Meeting", "July 31-August 1  Meeting".
        hit = re.match(r"([A-Za-z/]+)\s+(\d+)(?:-(?:([A-Za-z]+)\s+)?(\d+))?\s+Meeting\s*-\s*(\d{4})", text)
        if not hit:
            continue
        months, d1, month2, d2, yr = hit.groups()
        last = (month2 or months.split("/")[-1])[:3]
        month = next(v for k, v in _MONTHS.items() if k[:3] == last)
        day = int(d2 or d1)
        stamp = pd.Timestamp(int(yr), month, day)
        if stamp in _NOT_SCHEDULED:
            log.info("fomc: dropping %s (%s): labelled Meeting but not on the regular schedule", stamp.date(), text)
            continue
        out.append({"announcement": pd.Timestamp(int(yr), month, day), "source": f"historical{year}", "raw": text})
    return out


def _parse_current(html: str) -> list[dict]:
    out = []
    for panel in re.finditer(r"(\d{4}) FOMC Meetings(.*?)(?=\d{4} FOMC Meetings|$)", html, re.S):
        yr, body = int(panel.group(1)), panel.group(2)
        months = re.findall(r'fomc-meeting__month[^>]*>\s*(?:<strong>)?([A-Za-z/]+)', body)
        dates = re.findall(r'fomc-meeting__date[^>]*>\s*([^<]+)', body)
        for mon, dt in zip(months, dates):
            dt = dt.strip()
            if "unscheduled" in dt.lower() or "notation" in dt.lower():
                continue
            dm = re.match(r"(\d+)(?:-(\d+))?", dt)
            if not dm:
                continue
            last_mon = mon.split("/")[-1]
            if last_mon[:3] not in {k[:3] for k in _MONTHS}:
                continue
            month = next(v for k, v in _MONTHS.items() if k[:3] == last_mon[:3])
            day = int(dm.group(2) or dm.group(1))
            out.append({"announcement": pd.Timestamp(yr, month, day), "source": "current", "raw": f"{mon} {dt}"})
    return out


def fetch_fomc() -> pd.DataFrame:
    rows: list[dict] = []
    cur = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=UA, timeout=60)
    cur.raise_for_status()
    rows += _parse_current(cur.text)
    covered = {r["announcement"].year for r in rows}
    for year in range(1994, 2027):
        if year in covered:
            continue
        r = requests.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm", headers=UA, timeout=60)
        if r.status_code != 200:
            log.warning("fomc historical %d: HTTP %d", year, r.status_code)
            continue
        rows += _parse_historical(year, r.text)
        time.sleep(0.5)
    df = pd.DataFrame(rows).drop_duplicates("announcement").sort_values("announcement")
    log.info("fomc: %d scheduled announcements %s..%s", len(df), df.announcement.min().date(), df.announcement.max().date())
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tiingo = _env("TIINGO_API_KEY")
    for t in ETFS:
        fetch_tiingo(t, tiingo).to_parquet(out / f"tiingo_{t}.parquet")
    fetch_tiingo_crypto("btcusd", tiingo).to_parquet(out / "tiingo_BTCUSD.parquet")
    for s in CBOE:
        fetch_cboe(s).to_parquet(out / f"cboe_{s}.parquet")
    fetch_fred("DTB3", _env("FRED_API_KEY")).to_parquet(out / "fred_DTB3.parquet")
    fetch_fomc().to_parquet(out / "fomc_scheduled.parquet")
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
