#!/usr/bin/env python
"""Import vendor index-membership CSV/Parquet into the combined cache layout.

Writes ``data/cache/combined/universe_membership`` (Parquet key used by
``firm.runtime.load_universe_membership``). Input must conform to
:data:`firm.data.schemas.UNIVERSE_COLUMNS`:

    index, symbol, added_date, removed_date

``removed_date`` may be empty/NaT for currently active members.

Usage:
    python scripts/import_universe_membership.py vendor/sp500_membership.csv
    python scripts/import_universe_membership.py vendor/*.parquet --index SP500
    python scripts/import_universe_membership.py a.csv b.csv --merge
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
from firm.data.schemas import (  # noqa: E402
    COL_ADDED_DATE,
    COL_INDEX,
    COL_REMOVED_DATE,
    COL_SYMBOL,
    UNIVERSE_COLUMNS,
)

log = logging.getLogger(__name__)

_CACHE_KEY = "combined/universe_membership"


def _read_input(path: Path, default_index: str | None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Normalise column names (vendor exports vary).
    rename = {
        "ticker": COL_SYMBOL,
        "symbol": COL_SYMBOL,
        "index_name": COL_INDEX,
        "index": COL_INDEX,
        "added": COL_ADDED_DATE,
        "added_date": COL_ADDED_DATE,
        "start_date": COL_ADDED_DATE,
        "removed": COL_REMOVED_DATE,
        "removed_date": COL_REMOVED_DATE,
        "end_date": COL_REMOVED_DATE,
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    missing = [c for c in UNIVERSE_COLUMNS if c not in df.columns]
    if missing and not (missing == [COL_INDEX] and default_index):
        raise ValueError(f"{path}: missing columns {missing}; have {list(df.columns)}")

    if COL_INDEX not in df.columns and default_index:
        df[COL_INDEX] = default_index

    if default_index and (df[COL_INDEX].isna() | (df[COL_INDEX].astype(str).str.strip() == "")).any():
        df[COL_INDEX] = df[COL_INDEX].fillna(default_index)

    df[COL_ADDED_DATE] = pd.to_datetime(df[COL_ADDED_DATE], errors="coerce")
    df[COL_REMOVED_DATE] = pd.to_datetime(df[COL_REMOVED_DATE], errors="coerce")
    df[COL_SYMBOL] = df[COL_SYMBOL].astype(str).str.upper().str.strip()
    df[COL_INDEX] = df[COL_INDEX].astype(str).str.strip()
    return df[UNIVERSE_COLUMNS]


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values([COL_INDEX, COL_SYMBOL, COL_ADDED_DATE])
        .drop_duplicates(subset=[COL_INDEX, COL_SYMBOL, COL_ADDED_DATE], keep="last")
        .reset_index(drop=True)
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="CSV or Parquet membership files")
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Parquet cache root (default: data/cache)",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Default index label when input rows omit the index column",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing combined/universe_membership instead of replacing",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write cache")
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for path in args.inputs:
        if not path.exists():
            log.error("Input not found: %s", path)
            return 1
        df = _read_input(path, args.index)
        log.info("Read %d rows from %s", len(df), path)
        frames.append(df)

    combined = _dedupe(pd.concat(frames, ignore_index=True))
    log.info(
        "Prepared %d rows (%d symbols, %d indices)",
        len(combined),
        combined[COL_SYMBOL].nunique(),
        combined[COL_INDEX].nunique(),
    )

    if args.dry_run:
        log.info("Dry run — no cache write")
        return 0

    cache = ParquetCache(args.cache_dir)
    if args.merge:
        existing = cache.get(_CACHE_KEY)
        if existing is not None and not existing.empty:
            combined = _dedupe(pd.concat([existing, combined], ignore_index=True))
            log.info("Merged with existing cache → %d rows", len(combined))

    cache.put(_CACHE_KEY, combined)
    log.info("Wrote cache key %s (%d rows) under %s", _CACHE_KEY, len(combined), args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
