"""Download the rest of the EODHD Historian data worth keeping after the month.

Complements ``scripts/fetch_eodhd_prices.py`` (insider-cluster stock universe).
Everything lands under ``data/research/eodhd/`` (gitignored, licensed data):

    forex/<CODE>.parquet            every FOREX pair EODHD lists (EOD)
    crypto/<CODE>.parquet           every crypto code, ACTIVE AND DELISTED (survivorship-aware)
    etfs/<SYMBOL>.parquet           country / sector / industry / bond / commodity / currency ETFs
    us_universe/<CODE>.parquet      every exchange-listed US common stock, active AND delisted
    corporate_actions/splits/<T>.parquet, .../dividends/<T>.parquet
                                    for every ticker in prices/ (the insider universe)
    extras_manifest.json            per-item status

Resumable (existing files are skipped); parallel under the shared rate limit.

    python scripts/fetch_eodhd_extras.py [--only forex,crypto,etfs,us_universe,actions] [--workers 8]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_eodhd_prices import MAX_RPS, OUT, START, Client, _key  # noqa: E402

log = logging.getLogger(__name__)

ETFS = [
    # international / country
    "EFA", "EEM", "VGK", "VEA", "VWO", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWH", "EWS", "EWZ", "EWW",
    "EWT", "EWY", "EZA", "EWP", "EWI", "EWQ", "EWL", "EWN", "EWD", "EWK", "EWO", "EIS", "FXI", "INDA",
    # US sectors
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC",
    # bonds
    "AGG", "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "BIL", "EMB",
    # commodities
    "GLD", "SLV", "DBC", "USO", "UNG", "DBA", "PDBC",
    # currencies
    "UUP", "FXE", "FXY", "FXB", "FXA", "FXC", "FXF",
    # broad US / size
    "SPY", "QQQ", "IWM", "IJR", "IWC", "IJH", "MDY", "VTI",
    # narrower US industries (for industry-momentum tests; docs/research_brief_eodhd_findings.md #1)
    "SMH", "SOXX", "XSD", "XBI", "IBB", "XPH", "IHI", "KBE", "KRE", "KIE", "IAI", "XHB", "ITB", "XRT",
    "XOP", "OIH", "XES", "XME", "GDX", "GDXJ", "IYT", "XTN", "ITA", "XAR", "IGV", "FDN", "VNQ", "IYR",
    "PBJ", "XHS", "IYZ", "TAN", "ICLN",
    # more bonds / commodities (bond & commodity trend, #3)
    "BND", "IEI", "TLH", "VGIT", "VGLT", "MUB", "BWX", "IGOV", "JNK", "SHV", "GOVT", "SCHP",
    "DBB", "DBE", "DBP", "CPER", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "USCI", "GSG", "DJP",
]


def _safe(name: str) -> str:
    return name.replace("/", "_").replace("$", "S_")


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


def fetch_eod(client: Client, code: str, path: Path) -> dict:
    status, data = client.get(f"eod/{code}", **{"from": START})
    if status != 200:
        return {"status": "error", "http": status}
    if not isinstance(data, list) or not data or not isinstance(data[0], dict) or "date" not in data[0]:
        return {"status": "empty"}
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    _save(df, path)
    return {"status": "ok", "rows": int(len(df)), "first": str(df["date"].min().date()),
            "last": str(df["date"].max().date())}


def fetch_action(client: Client, kind: str, ticker: str, path: Path) -> dict:
    code = ticker.replace(".", "-").replace("/", "-")
    status, data = client.get(f"{'div' if kind == 'dividends' else 'splits'}/{code}.US", **{"from": START})
    if status != 200:
        return {"status": "error", "http": status}
    df = pd.DataFrame(data if isinstance(data, list) else [])
    _save(df, path)  # an empty frame is a valid answer: no splits / no dividends
    return {"status": "ok", "rows": int(len(df))}


def _merge_manifest(path: Path, updates: dict) -> dict:
    """Merge this run's statuses into the on-disk manifest under an exclusive lock,
    so concurrent runs (e.g. --only forex and --only us_universe) keep each other's entries."""
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = json.loads(path.read_text()) if path.exists() else {}
        merged.update(updates)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=1))
        tmp.replace(path)
    return merged


def jobs(client: Client, only: set[str]) -> list[tuple[str, callable]]:
    out: list[tuple[str, callable]] = []
    if "forex" in only:
        s, d = client.get("exchange-symbol-list/FOREX")
        for code in sorted({x["Code"] for x in (d if isinstance(d, list) else [])}):
            p = OUT / "forex" / f"{_safe(code)}.parquet"
            out.append((f"forex:{code}", p, lambda c=code, p=p: fetch_eod(client, f"{c}.FOREX", p)))
    if "crypto" in only:
        codes: dict[str, str] = {}
        for params, flag in (({}, "active"), ({"delisted": 1}, "delisted")):
            s, d = client.get("exchange-symbol-list/CC", **params)
            for x in (d if isinstance(d, list) else []):
                codes.setdefault(x["Code"], flag)
        pd.DataFrame({"code": list(codes), "listing": list(codes.values())}).to_parquet(
            OUT / "crypto_symbols.parquet")
        for code in sorted(codes):
            p = OUT / "crypto" / f"{_safe(code)}.parquet"
            out.append((f"crypto:{code}", p, lambda c=code, p=p: fetch_eod(client, f"{c}.CC", p)))
    if "etfs" in only:
        for sym in ETFS:
            p = OUT / "etfs" / f"{sym}.parquet"
            out.append((f"etf:{sym}", p, lambda s=sym, p=p: fetch_eod(client, f"{s}.US", p)))
    if "us_universe" in only:
        # Every exchange-listed US common stock, ACTIVE AND DELISTED: a
        # survivorship-free universe for breadth / 52-week-high / cross-sectional
        # tests (no index-membership data in the Historian plan).
        listed = {"NASDAQ", "NYSE", "AMEX", "NYSE MKT", "NYSE ARCA", "BATS", "NYSEARCA"}
        frames = []
        for name in ("symbols_active", "symbols_delisted"):
            f = pd.read_parquet(OUT / f"{name}.parquet")
            f = f[(f["Type"] == "Common Stock") & (f["Exchange"].isin(listed))].copy()
            f["listing"] = "active" if name.endswith("active") else "delisted"
            frames.append(f)
        uni = pd.concat(frames, ignore_index=True).drop_duplicates("Code")
        uni.to_parquet(OUT / "us_universe_symbols.parquet")
        for code in sorted(uni["Code"]):
            p = OUT / "us_universe" / f"{_safe(code)}.parquet"
            out.append((f"us:{code}", p, lambda c=code, p=p: fetch_eod(client, f"{c}.US", p)))
    if "actions" in only:
        for f in sorted((OUT / "prices").glob("*.parquet")):
            t = f.stem
            for kind in ("splits", "dividends"):
                p = OUT / "corporate_actions" / kind / f"{t}.parquet"
                out.append((f"{kind}:{t}", p, lambda k=kind, t=t, p=p: fetch_action(client, k, t, p)))
    return [(k, p, fn) for k, p, fn in out]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="forex,crypto,etfs,actions")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-rps", type=float, default=MAX_RPS,
                    help="per-process rate; concurrent runs must sum below EODHD's ~16/s")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    manifest_path = OUT / "extras_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    client = Client(_key(), max_rps=args.max_rps)
    todo = [(k, fn) for k, p, fn in jobs(client, only)
            if not p.exists() and manifest.get(k, {}).get("status") != "empty"]
    log.info("extras: %d items to fetch (%s) at %.1f req/s", len(todo), sorted(only), args.max_rps)
    updates: dict = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn): k for k, fn in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            k = futures[fut]
            try:
                updates[k] = fut.result()
            except Exception as exc:
                log.warning("%s failed (%s)", k, exc)
                updates[k] = {"status": "error", "detail": str(exc)[:200]}
            if i % 1000 == 0 or i == len(todo):
                manifest = _merge_manifest(manifest_path, updates)
                st = [v["status"] for v in updates.values()]
                log.info("%d/%d (ok %d, empty %d, error %d), %d API calls", i, len(todo),
                         st.count("ok"), st.count("empty"), st.count("error"), client.calls)
    manifest = _merge_manifest(manifest_path, updates)
    by_kind: dict[str, dict[str, int]] = {}
    for k, v in manifest.items():
        d = by_kind.setdefault(k.split(":")[0], {})
        d[v["status"]] = d.get(v["status"], 0) + 1
    log.info("extras final: %s; %d API calls this run", by_kind, client.calls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
