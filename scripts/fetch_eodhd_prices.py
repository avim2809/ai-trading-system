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
MAX_RPS = 12.0   # EODHD allows 1,000 requests/minute (~16/s); stay well below


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
    # OTC venue suffixes are not share classes: 'AAAC.OB' is AAAC on the
    # Bulletin Board, 'ABVT.PK' is Pink Sheets (EODHD codes them plainly).
    t = re.sub(r"[.\-](OB|0B|BB|PK|OTC|PINK|QB|QX|OTCBB)\b", "", t)
    tokens = [x for x in re.split(r"[\s,;&]+", t) if x]
    for tok in tokens:
        tok = tok.strip(".-/")
        if tok in ("NONE", "NA", "N/A", "NULL", "TBD", "NYSE", "NASDAQ", "AMEX", "OTC"):
            continue
        if tok.split("/")[0].split(".")[0] in ("NYSE", "NASDAQ", "AMEX", "OTC", "NYSEAMERICAN", "ARCA"):
            continue
        if re.fullmatch(r"[A-Z0-9]{1,6}([./-][A-Z0-9]{1,3})?", tok):
            return tok
    return None


def eodhd_symbol(ticker: str) -> str:
    """SEC as-filed ticker -> EODHD US code (class separators become '-')."""
    t = ticker.strip().upper().replace("/", "-").replace(".", "-")
    return f"{t}.US"


class Client:
    """Thread-safe EODHD client: a shared minimum gap between request starts
    keeps total throughput under ``max_rps`` however many workers call it."""

    def __init__(self, key: str, max_rps: float = MAX_RPS) -> None:
        import threading

        self._key = key
        self._min_gap = 1.0 / max_rps
        self._next = time.monotonic()
        self._lock = threading.Lock()
        self._session = requests.Session()
        self.calls = 0

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._min_gap
            self.calls += 1
        if start > now:
            time.sleep(start - now)

    def get(self, path: str, **params) -> tuple[int, object]:
        for attempt in range(5):
            self._throttle()
            try:
                r = self._session.get(BASE + path, params={**params, "api_token": self._key, "fmt": "json"},
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


def price_path(ticker: str) -> Path:
    """File for a ticker; class separators can't become path separators."""
    return OUT / "prices" / (ticker.replace("/", "_") + ".parquet")


def fetch_ticker(client: Client, ticker: str) -> dict:
    path = price_path(ticker)
    status, data = client.get("eod/" + eodhd_symbol(ticker), **{"from": START})
    if status != 200:
        return {"status": "error", "http": status, "detail": str(data)[:200]}
    if not isinstance(data, list) or not data or not isinstance(data[0], dict) or "date" not in data[0]:
        return {"status": "empty"}
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    tmp.replace(path)  # atomic: an interrupted run never leaves a partial file
    return {"status": "ok", "rows": len(df), "first": str(df["date"].min().date()),
            "last": str(df["date"].max().date())}


_NAME_DROP = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED",
              "PLC", "LLC", "LP", "L", "P", "THE", "HOLDINGS", "HOLDING", "GROUP", "SA", "NV", "AG"}


def norm_name(name: str) -> str:
    """Normalise a company name for EXACT matching ('ABINGTON BANCORP, INC./PA'
    and 'Abington Bancorp Inc' both become 'ABINGTON BANCORP')."""
    import re

    n = str(name).upper()
    n = re.sub(r"/[A-Z]{2,3}/?\s*$", " ", n)          # trailing state tag: /PA, /DE/
    n = re.sub(r"[^A-Z0-9 ]", " ", n)
    return " ".join(w for w in n.split() if w not in _NAME_DROP)


def resolve_by_name(client: Client, events: pd.DataFrame, manifest: dict) -> None:
    """For tickers EODHD doesn't know (404/empty), look for a US common stock
    whose normalised name EXACTLY matches the filer's issuer name, fetch it,
    and accept it only if its history covers that ticker's event dates.
    Accepted series are saved as prices/<TICKER>.parquet with the EODHD code
    recorded in the manifest ("resolved_code")."""
    tm = pd.read_parquet(OUT / "ticker_map.parquet")
    syms = pd.concat([pd.read_parquet(OUT / "symbols_active.parquet"),
                      pd.read_parquet(OUT / "symbols_delisted.parquet")], ignore_index=True)
    syms = syms[(syms["Type"] == "Common Stock") & (syms["Country"] == "USA")].copy()
    syms["key"] = syms["Name"].map(norm_name)
    by_key = syms.groupby("key")["Code"].apply(lambda c: sorted(set(c))).to_dict()
    ev = events.merge(tm, left_on="ticker", right_on="raw_ticker", suffixes=("_raw", ""))
    missing = [t for t, v in manifest.items() if v.get("status") in ("error", "empty")]
    # Also tickers whose fetched series starts AFTER their earliest event (the
    # code now belongs to a later company): look for the older issuer by name.
    first_event = pd.to_datetime(ev.groupby("ticker")["known_date"].min())
    for t, v in manifest.items():
        if v.get("status") == "ok" and not v.get("resolved_code") and t in first_event.index:
            if pd.Timestamp(v["first"]) > first_event[t]:
                missing.append(t)
    resolved = 0
    for t in missing:
        try:
            resolved += _resolve_one(client, t, ev, by_key, manifest)
        except Exception as exc:
            log.warning("name resolution for %s failed (%s) -- left unresolved", t, exc)
    log.info("name resolution: %d of %d missing tickers resolved by exact issuer-name match",
             resolved, len(missing))


def _resolve_one(client: Client, t: str, ev: pd.DataFrame, by_key: dict, manifest: dict) -> int:
    if True:
        rows = ev[ev["ticker"] == t]
        if rows.empty:
            return 0
        dates = pd.to_datetime(rows["known_date"])
        best = None
        for key in {norm_name(n) for n in rows["issuer_name"].dropna()}:
            for code in by_key.get(key, [])[:3]:
                status, data = client.get(f"eod/{code}.US", **{"from": START})
                if status != 200 or not isinstance(data, list) or not data or "date" not in data[0]:
                    continue
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                covered = int(((dates >= df["date"].min()) & (dates <= df["date"].max())).sum())
                if covered and (best is None or covered > best[0]):
                    best = (covered, code, df)
        if best and manifest.get(t, {}).get("status") == "ok" and not manifest[t].get("resolved_code"):
            existing = pd.read_parquet(price_path(t), columns=["date"])["date"]
            already = int(((dates >= existing.min()) & (dates <= existing.max())).sum())
            if best[0] <= already:
                return 0
        if best:
            covered, code, df = best
            dest = price_path(t)
            tmp = dest.with_suffix(".parquet.tmp")
            df.to_parquet(tmp)
            tmp.replace(dest)
            manifest[t] = {"status": "ok", "resolved_code": code, "resolved_by": "exact_name",
                           "rows": len(df), "first": str(df["date"].min().date()),
                           "last": str(df["date"].max().date()), "events_covered": covered,
                           "events_total": len(dates)}
            return 1
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N tickers only (smoke test)")
    ap.add_argument("--workers", type=int, default=8, help="parallel requests (rate-limited in total)")
    ap.add_argument("--resolve-names", action="store_true",
                    help="after fetching, resolve 404/empty tickers by exact issuer-name match")
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
    current = set(tickers)
    manifest = {k: v for k, v in manifest.items() if k in current}  # drop keys from older cleaning rules
    client = Client(_key())
    fetch_symbol_lists(client)
    todo = [t for t in tickers if not price_path(t).exists()
            and manifest.get(t, {}).get("status") not in ("empty", "error")]
    retry = [t for t in tickers if manifest.get(t, {}).get("status") == "error"]
    todo += [t for t in retry if t not in todo]
    log.info("%d tickers, %d to fetch", len(tickers), len(todo))
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_ticker, client, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                manifest[t] = fut.result()
            except Exception as exc:  # keep going; the ticker is retried next run
                log.warning("%s: fetch failed (%s)", t, exc)
                manifest[t] = {"status": "error", "detail": str(exc)[:200]}
            if i % 500 == 0 or i == len(todo):
                manifest_path.write_text(json.dumps(manifest, indent=1))
                done = [v["status"] for v in manifest.values()]
                log.info("%d/%d fetched (ok %d, empty %d, error %d), %d API calls",
                         i, len(todo), done.count("ok"), done.count("empty"), done.count("error"), client.calls)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    if args.resolve_names:
        resolve_by_name(client, events, manifest)
        manifest_path.write_text(json.dumps(manifest, indent=1))
    done = [v["status"] for v in manifest.values()]
    log.info("final: ok %d, empty %d, error %d; %d API calls this run",
             done.count("ok"), done.count("empty"), done.count("error"), client.calls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
