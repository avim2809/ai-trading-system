#!/usr/bin/env python
"""Import Sharadar (Nasdaq Data Link) bulk CSV exports into the Parquet cache.

Expects vendor files downloaded manually or via ``nasdaq-data-link`` CLI — no
API key is required at import time. Normalizes Sharadar column names into the
canonical schemas in ``firm.data.schemas`` and writes:

- ``combined/prices``
- ``combined/fundamentals`` (subset of SF1 fields mapped to FUNDAMENTAL_COLS)
- ``combined/universe_membership`` (via ``import_universe_membership`` logic)

Sharadar table reference (see docs/longer_dataset_options.md):

| Table | Typical file | Maps to |
|-------|--------------|---------|
| SEP   | ``SEP.csv``  | daily prices (``ticker`` → ``symbol``, ``closeadj`` → ``adj_close``) |
| SF1   | ``SF1.csv``  | quarterly fundamentals (``dimension=MRY`` rows) |
| TICKERS / custom | membership CSV | ``index,symbol,added_date,removed_date`` |

Usage:
    python scripts/etl_sharadar_to_cache.py --prices SEP.csv
    python scripts/etl_sharadar_to_cache.py --prices SEP.csv --fundamentals SF1.csv \\
        --membership sp500_membership.csv --index SP500 --merge
    python scripts/etl_sharadar_to_cache.py --prices SEP.csv --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.data.cache import ParquetCache  # noqa: E402
from firm.data.schemas import FUNDAMENTAL_COLS, PRICE_COLS, UNIVERSE_COLUMNS  # noqa: E402

log = logging.getLogger(__name__)

_PRICES_KEY = "combined/prices"
_FUNDAMENTALS_KEY = "combined/fundamentals"
_MEMBERSHIP_KEY = "combined/universe_membership"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalize_sharadar_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Map Sharadar SEP (or similar) rows to :data:`PRICE_COLS`."""
    rename = {
        "ticker": "symbol",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "closeadj": "adj_close",
        "closeunadj": "close",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    missing = [c for c in PRICE_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"prices missing columns after rename: {missing}")
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    return out[PRICE_COLS]


def normalize_sharadar_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Map Sharadar SF1 rows to :data:`FUNDAMENTAL_COLS` (best-effort)."""
    if "dimension" in df.columns:
        df = df[df["dimension"].astype(str).str.upper() == "MRY"]
    rename = {
        "ticker": "symbol",
        "datekey": "date",
        "calendardate": "date",
        "marketcap": "market_cap",
        "pe": "pe_ratio",
        "pb": "pb_ratio",
        "roe": "roe",
        "debt": "debt_to_equity",
        "revenue": "revenue",
        "netinc": "net_income",
        "eps": "eps",
        "divyield": "dividend_yield",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "date" not in out.columns:
        raise ValueError("fundamentals: need datekey or calendardate column")
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    for col in FUNDAMENTAL_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[FUNDAMENTAL_COLS].dropna(subset=["date", "symbol"])


def _merge_panel(cache: ParquetCache, key: str, new_df: pd.DataFrame) -> int:
    if new_df.empty:
        log.warning("empty frame for %s — skip", key)
        return 0
    merged = cache.merge_combined(key, new_df)
    cache.put(key, merged)
    return len(merged)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, help="Sharadar SEP (or compatible) CSV")
    parser.add_argument("--fundamentals", type=Path, help="Sharadar SF1 CSV")
    parser.add_argument("--membership", type=Path, help="Index membership CSV")
    parser.add_argument("--index", default="SP500", help="Index label for membership rows")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--merge", action="store_true", help="Merge into existing cache keys")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not any([args.prices, args.fundamentals, args.membership]):
        parser.error("provide at least one of --prices, --fundamentals, --membership")

    cache = ParquetCache(args.cache_dir)
    counts: dict[str, int] = {}

    if args.prices:
        raw = _read_csv(args.prices)
        prices = normalize_sharadar_prices(raw)
        log.info("prices: %d rows, %d symbols", len(prices), prices["symbol"].nunique())
        counts["prices"] = len(prices)
        if not args.dry_run:
            if args.merge:
                counts["prices"] = _merge_panel(cache, _PRICES_KEY, prices)
            else:
                cache.put(_PRICES_KEY, prices)

    if args.fundamentals:
        raw = _read_csv(args.fundamentals)
        fundamentals = normalize_sharadar_fundamentals(raw)
        log.info(
            "fundamentals: %d rows, %d symbols",
            len(fundamentals), fundamentals["symbol"].nunique(),
        )
        counts["fundamentals"] = len(fundamentals)
        if not args.dry_run:
            if args.merge:
                counts["fundamentals"] = _merge_panel(cache, _FUNDAMENTALS_KEY, fundamentals)
            else:
                cache.put(_FUNDAMENTALS_KEY, fundamentals)

    if args.membership:
        # Reuse the dedicated membership importer (subprocess keeps one code path).
        import subprocess

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "import_universe_membership.py"),
            str(args.membership),
            "--cache-dir", args.cache_dir,
            "--index", args.index,
        ]
        if args.merge:
            cmd.append("--merge")
        if args.dry_run:
            cmd.append("--dry-run")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            return proc.returncode
        if not args.dry_run:
            mem = cache.get(_MEMBERSHIP_KEY)
            counts["membership"] = len(mem) if mem is not None else 0

    if args.dry_run:
        log.info("Dry run — normalized counts: %s", counts)
    else:
        log.info("Cache write complete: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
