"""Owner-run: ingest Bank of Israel USD/ILS and CBS Israeli CPI into data/research/il_macro/ (ticket P2-02).

The research agent never runs this (no network from agent sessions). Endpoints are NOT hard-coded because they
were not verified offline: pass each source as a URL (fetched here) or a local CSV path.

  python scripts/fetch_il_macro.py --usd-ils SRC --cpi SRC [--out data/research/il_macro]

CSV expectations: usd_ils needs columns date,close ; cpi needs date (observation month),value and optionally
publish_date (else ``firm.data.etf_loader.CPI_DEFAULT_LAG_DAYS`` applies). Rows after 2026-09-30 are dropped.
Writes usd_ils.parquet, il_cpi.parquet and research/data_manifests/il_macro.json (hashes only).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger("fetch_il_macro")
LAST_USABLE = dt.date(2026, 9, 30)


def _read_csv(src: str) -> pd.DataFrame:
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=60) as r:  # noqa: S310 - owner-supplied URL, owner-run script
            return pd.read_csv(io.BytesIO(r.read()))
    return pd.read_csv(src)


def _prep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns {missing}; got {list(df.columns)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(LAST_USABLE)].sort_values("date").drop_duplicates("date")
    if "publish_date" in df.columns:
        df["publish_date"] = pd.to_datetime(df["publish_date"])
    return df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd-ils", required=True)
    ap.add_argument("--cpi", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "research" / "il_macro")
    ap.add_argument("--manifest", type=Path, default=ROOT / "research" / "data_manifests" / "il_macro.json")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    a.out.mkdir(parents=True, exist_ok=True)
    fx = _prep(_read_csv(a.usd_ils), ["date", "close"])
    cpi = _prep(_read_csv(a.cpi), ["date", "value"])
    paths = []
    for name, df in (("usd_ils", fx), ("il_cpi", cpi)):
        p = a.out / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths.append(p)
        log.info("%s: %d rows %s..%s", name, len(df), df["date"].min().date(), df["date"].max().date())
    from firm.data.manifest import build_manifest, write_manifest

    # Manifest metadata is read via firm.research.data_access, so the il_macro directory must be in the freeze allow-list.
    m = build_manifest(paths, root=a.out, dataset="il_macro", licence="Bank of Israel / CBS public series (verify terms)")
    write_manifest(m, a.manifest, extra={"note": "last observation <= 2026-09-30; EODHD USDILS is the cross-check"})
    log.info("manifest written: %s", a.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
