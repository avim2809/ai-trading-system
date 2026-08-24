#!/usr/bin/env python
"""Backfill FRED macro series into ``combined/macro``.

PART 3 Phase 4 of the remediation plan: ``firm.data.providers.fred.
fetch_macro_bundle`` and ``PointInTimeDataStore.load_macro``/``get_macro``
already existed (built for ``ml_prediction.py``, which is disabled), but
nothing ever cached the result -- every backtest that had ``FRED_API_KEY``
set was instead fetching macro data live, over the network, on every single
run (see ``src/firm/runtime.py``'s macro-loading block). This script writes
one cached ``combined/macro`` panel once; ``runtime.py`` prefers the cache
over a live fetch when present.

Stored in the same long ``(date, symbol, value)`` shape
``ParquetCache.merge_combined`` already dedupes/sorts by -- ``symbol`` here
is the FRED series ID (e.g. ``T10Y2Y``), not an equity ticker (see
``firm.data.providers.fred.macro_bundle_to_long_frame``). ``runtime.py``
converts it back via that module's ``macro_bundle_from_cache``.

Usage:
    python scripts/backfill_macro.py
    python scripts/backfill_macro.py --start 2010-01-01 --end 2026-06-30
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

from firm.config import get_settings  # noqa: E402
from firm.data.cache import ParquetCache  # noqa: E402
from firm.data.providers.fred import fetch_macro_bundle, macro_bundle_to_long_frame  # noqa: E402

log = logging.getLogger(__name__)

_COMBINED_KEY = "combined/macro"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None, help="Default: today")
    parser.add_argument(
        "--series", default=None,
        help="Comma-separated FRED series IDs/aliases (default: the standard bundle)",
    )
    args = parser.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    settings = get_settings()
    api_key = settings.fred_api_key
    if not api_key:
        log.error("FRED_API_KEY not set — nothing to backfill")
        return 1

    series = args.series.split(",") if args.series else None
    bundle = fetch_macro_bundle(api_key, args.start, end, series=series)
    if not bundle:
        log.error("fetch_macro_bundle returned nothing — check the API key/network")
        return 1

    new_df = macro_bundle_to_long_frame(bundle)
    if new_df.empty:
        log.error("All fetched series were empty — nothing to write")
        return 1

    cache = ParquetCache(settings.data.cache_dir)
    merged = cache.merge_combined(_COMBINED_KEY, new_df)
    log.info(
        "combined/macro: %d rows across %d series (%s -> %s)",
        len(merged), merged["symbol"].nunique(),
        str(merged["date"].min())[:10], str(merged["date"].max())[:10],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
