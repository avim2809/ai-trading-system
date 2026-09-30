"""Download the rest of the EODHD Historian data worth keeping after the month.

Complements ``scripts/fetch_eodhd_prices.py`` (insider-cluster stock universe).
Everything lands under ``data/research/eodhd/`` (gitignored, licensed data):

    forex/<CODE>.parquet            every FOREX pair EODHD lists (EOD)
    crypto/<CODE>.parquet           every crypto code, ACTIVE AND DELISTED (survivorship-aware)
    etfs/<SYMBOL>.parquet           country / sector / bond / commodity / currency ETFs
    corporate_actions/splits/<T>.parquet, .../dividends/<T>.parquet
                                    for every ticker in prices/ (the insider universe)
    extras_manifest.json            per-item status

Resumable (existing files are skipped); parallel under the shared rate limit.

    python scripts/fetch_eodhd_extras.py [--only forex,crypto,etfs,actions] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_eodhd_prices import OUT, START, Client, _key  # noqa: E402

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
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    manifest_path = OUT / "extras_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    client = Client(_key())
    todo = [(k, fn) for k, p, fn in jobs(client, only)
            if not p.exists() and manifest.get(k, {}).get("status") != "empty"]
    log.info("extras: %d items to fetch (%s)", len(todo), sorted(only))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn): k for k, fn in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            k = futures[fut]
            try:
                manifest[k] = fut.result()
            except Exception as exc:
                log.warning("%s failed (%s)", k, exc)
                manifest[k] = {"status": "error", "detail": str(exc)[:200]}
            if i % 1000 == 0 or i == len(todo):
                manifest_path.write_text(json.dumps(manifest, indent=1))
                st = [v["status"] for v in manifest.values()]
                log.info("%d/%d (ok %d, empty %d, error %d), %d API calls", i, len(todo),
                         st.count("ok"), st.count("empty"), st.count("error"), client.calls)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    by_kind: dict[str, dict[str, int]] = {}
    for k, v in manifest.items():
        d = by_kind.setdefault(k.split(":")[0], {})
        d[v["status"]] = d.get(v["status"], 0) + 1
    log.info("extras final: %s; %d API calls this run", by_kind, client.calls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
