"""Download EODHD end-of-day history for the insider-cluster research universe.

One-off research download (docs/research_findings_beyond_equities_2026_09_30.md,
scripts/insider_cluster_preregistered_bars.py): daily OHLCV + adjusted close for
every ticker that appears in the SEC Form 4 cluster-event list, including
delisted companies, plus a few size-benchmark ETFs and EODHD's full US symbol
lists (active + delisted) for later ticker-reuse checks.

Writes under ``data/research/eodhd/`` (gitignored: licensed vendor data):

    prices/<TICKER>.parquet     date, open, high, low, close, adjusted_close, volume
    symbols_active.parquet      EODHD US exchange-symbol list
    symbols_delisted.parquet    EODHD US delisted-symbol list
    manifest.json               per-ticker status (ok / empty / error), first/last date, rows

Resumable: tickers that already have a parquet file are skipped. Requires
``EODHD_API_KEY`` in ``.env`` (never logged).

    python scripts/fetch_eodhd_prices.py \
        --events <scratchpad>/insider_data/cluster_events.parquet [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "data" / "research" / "eodhd"
BASE = "https://eodhd.com/api/"
BENCHMARKS = ["SPY", "IWM", "IJR", "IWC", "MDY", "IJH", "VB", "VO", "VTI"]
START = "2005-01-01"
MAX_RPS = 8.0


def _key() -> str:
    key = os.environ.get("EODHD_API_KEY") or dotenv_values(_ROOT / ".env").get("EODHD_API_KEY") or ""
    if not key.strip():
        raise SystemExit("EODHD_API_KEY not set in .env")
    return key.strip()


_PREFIXES = ("NYSEAMERICAN:", "NYSEARCA:", "NYSEMKT:", "NASDAQ:", "NYSE:", "AMEX:", "OTC:", "OTCBB:", "PINK:")


def clean_ticker(raw: str) -> str | None:
    """Normalise an as-filed SEC ISSUERTRADINGSYMBOL.

    As-filed values include quotes, brackets, exchange prefixes and lists
    ('"WM"', '( EPG )', 'NYSE: XYZ', 'ABC, ABC.PR'). Returns the first
    plausible ticker (letters/digits plus one class separator), or None for
    values like 'NONE' / 'N/A'.
    """
    import re

    t = str(raw).upper().strip()
    for ch in "\"'()[]{}*":
        t = t.replace(ch, " ")
    t = t.replace(" :", ":").replace(": ", ":")
    for pre in _PREFIXES:
        t = t.replace(pre, " ")
    tokens = [x for x in re.split(r"[\s,;&]+", t) if x]
    for tok in tokens:
        tok = tok.strip(".-/")
        if tok in ("NONE", "NA", "N/A", "NULL", "TBD", "NYSE", "NASDAQ", "AMEX", "OTC"):
            continue
        if re.fullmatch(r"[A-Z0-9]{1,6}([./-][A-Z0-9]{1,3})?", tok):
            return tok
    return None


def eodhd_symbol(ticker: str) -> str:
    """SEC as-filed ticker -> EODHD US code (class separators become '-')."""
    t = ticker.strip().upper().replace("/", "-").replace(".", "-")
    return f"{t}.US"


class Client:
    def __init__(self, key: str, max_rps: float = MAX_RPS) -> None:
        self._key = key
        self._min_gap = 1.0 / max_rps
        self._last = 0.0
        self.calls = 0

    def get(self, path: str, **params) -> tuple[int, object]:
        for attempt in range(5):
            wait = self._min_gap - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            try:
                r = requests.get(BASE + path, params={**params, "api_token": self._key, "fmt": "json"},
                                 timeout=90)
            except requests.RequestException as exc:
                log.warning("EODHD %s: %s (attempt %d)", path, exc, attempt + 1)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                log.warning("EODHD %s: HTTP %d (attempt %d), backing off", path, r.status_code, attempt + 1)
                time.sleep(2 ** attempt * 5)
                continue
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, r.text[:300]
        return 599, "retries exhausted"


def fetch_symbol_lists(client: Client) -> None:
    for name, params in (("symbols_active", {}), ("symbols_delisted", {"delisted": 1})):
        path = OUT / f"{name}.parquet"
        if path.exists():
            continue
        status, data = client.get("exchange-symbol-list/US", **params)
        if status == 200 and isinstance(data, list):
            pd.DataFrame(data).to_parquet(path)
            log.info("%s: %d symbols", name, len(data))
        else:
            log.error("%s: HTTP %s %s", name, status, str(data)[:200])


def fetch_ticker(client: Client, ticker: str) -> dict:
    path = OUT / "prices" / f"{ticker}.parquet"
    status, data = client.get("eod/" + eodhd_symbol(ticker), **{"from": START})
    if status != 200:
        return {"status": "error", "http": status, "detail": str(data)[:200]}
    if not isinstance(data, list) or not data or not isinstance(data[0], dict) or "date" not in data[0]:
        return {"status": "empty"}
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(path)
    return {"status": "ok", "rows": int(len(df)), "first": str(df["date"].min().date()),
            "last": str(df["date"].max().date())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N tickers only (smoke test)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    (OUT / "prices").mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(args.events)
    raw = sorted({t for t in events["ticker"].dropna().astype(str) if t.strip()})
    mapping = {r: clean_ticker(r) for r in raw}
    pd.DataFrame({"raw_ticker": list(mapping), "ticker": list(mapping.values())}).to_parquet(
        OUT / "ticker_map.parquet")
    unmappable = [r for r, c in mapping.items() if c is None]
    log.info("as-filed tickers: %d raw -> %d clean (%d unmappable, e.g. %s)", len(raw),
             len({c for c in mapping.values() if c}), len(unmappable), unmappable[:5])
    tickers = sorted({c for c in mapping.values() if c} | set(BENCHMARKS))
    if args.limit:
        tickers = tickers[: args.limit]
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {k: v for k, v in manifest.items() if v.get("status") != "error"}  # retry errors
    client = Client(_key())
    fetch_symbol_lists(client)
    todo = [t for t in tickers if not (OUT / "prices" / f"{t}.parquet").exists() and manifest.get(t, {}).get("status") != "empty"]
    log.info("%d tickers, %d to fetch", len(tickers), len(todo))
    for i, t in enumerate(todo, 1):
        manifest[t] = fetch_ticker(client, t)
        if i % 250 == 0 or i == len(todo):
            manifest_path.write_text(json.dumps(manifest, indent=1))
            done = [v["status"] for v in manifest.values()]
            log.info("%d/%d fetched (ok %d, empty %d, error %d), %d API calls",
                     i, len(todo), done.count("ok"), done.count("empty"), done.count("error"), client.calls)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
