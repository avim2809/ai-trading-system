#!/usr/bin/env python
"""Backfill daily prices from Tiingo into ``combined/prices``.

Fetches one symbol at a time (respecting free-tier rate limits), merging each
into the Parquet cache after every successful pull so partial progress survives
interruptions.

Default symbol list: ``universe.symbols`` from ``config/live.yaml`` (25 names).
Default range: 2010-01-01 → today.

Usage:
    python scripts/backfill_tiingo_prices.py
    python scripts/backfill_tiingo_prices.py --start 2010-01-01 --symbols AAPL,MSFT
    python scripts/backfill_tiingo_prices.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.config import get_settings  # noqa: E402
from firm.data.cache import ParquetCache  # noqa: E402
from firm.data.providers.tiingo import TiingoProvider  # noqa: E402
from firm.data.schemas import PRICE_COLS  # noqa: E402

log = logging.getLogger(__name__)

_COMBINED_KEY = "combined/prices"
_DEFAULT_LIVE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "live.yaml"


def load_live_universe_symbols(live_config: Path = _DEFAULT_LIVE_CONFIG) -> list[str]:
    """Return sorted tickers from ``config/live.yaml`` ``universe.symbols``."""
    if not live_config.exists():
        raise FileNotFoundError(f"live config not found: {live_config}")
    data = yaml.safe_load(live_config.read_text(encoding="utf-8")) or {}
    symbols = (data.get("universe") or {}).get("symbols") or []
    return sorted({str(s).upper().strip() for s in symbols if str(s).strip()})


def _cache_summary(cache: ParquetCache) -> dict:
    df = cache.get(_COMBINED_KEY)
    if df is None or df.empty:
        return {"rows": 0, "symbols": 0, "min_date": None, "max_date": None}
    dates = df["date"]
    return {
        "rows": len(df),
        "symbols": int(df["symbol"].nunique()),
        "min_date": str(dates.min())[:10],
        "max_date": str(dates.max())[:10],
    }


def backfill_symbols(
    symbols: list[str],
    *,
    start: str,
    end: str,
    cache_dir: str,
    min_interval_sec: float,
    dry_run: bool,
) -> dict:
    """Fetch Tiingo EOD for each symbol and merge into ``combined/prices``."""
    settings = get_settings()
    cache = ParquetCache(cache_dir)
    before = _cache_summary(cache)
    log.info("cache before: %s", before)

    if dry_run:
        log.info(
            "dry-run: would fetch %d symbols %s → %s (interval %.1fs)",
            len(symbols), start, end, min_interval_sec,
        )
        return {"before": before, "fetched": 0, "failed": [], "after": before}

    provider = TiingoProvider(settings=settings)
    fetched = 0
    failed: list[str] = []

    for i, sym in enumerate(symbols):
        if i > 0 and min_interval_sec > 0:
            time.sleep(min_interval_sec)
        log.info("fetching %s (%d/%d) %s → %s", sym, i + 1, len(symbols), start, end)
        try:
            frame = provider.get_prices([sym], start, end)
        except Exception as exc:
            log.warning("fetch failed symbol=%s (%s)", sym, exc)
            failed.append(sym)
            continue
        if frame is None or frame.empty:
            log.warning("no rows returned for %s", sym)
            failed.append(sym)
            continue
        merged = cache.merge_combined(_COMBINED_KEY, frame[PRICE_COLS])
        fetched += 1
        sym_rows = len(frame)
        log.info(
            "merged %s: %d rows (combined now %d rows, %d symbols)",
            sym, sym_rows, len(merged), merged["symbol"].nunique(),
        )

    after = _cache_summary(cache)
    log.info("cache after: %s", after)
    return {"before": before, "fetched": fetched, "failed": failed, "after": after}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-config",
        type=Path,
        default=_DEFAULT_LIVE_CONFIG,
        help="YAML with universe.symbols (default: config/live.yaml)",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated tickers (default: live.yaml universe)",
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--cache-dir", default=None, help="Parquet cache root")
    parser.add_argument(
        "--min-interval-sec",
        type=float,
        default=2.0,
        help="Pause between symbol requests (free tier: 50/hr ≈ 72s max; 2s is safe for 25)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.symbols:
        symbols = sorted({s.strip().upper() for s in args.symbols.split(",") if s.strip()})
    else:
        symbols = load_live_universe_symbols(args.live_config)

    if not symbols:
        log.error("empty symbol list")
        return 1

    settings = get_settings()
    cache_dir = args.cache_dir or settings.data.cache_dir

    try:
        result = backfill_symbols(
            symbols,
            start=args.start,
            end=args.end,
            cache_dir=str(cache_dir),
            min_interval_sec=args.min_interval_sec,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        log.error("backfill aborted: %s", exc)
        return 1

    print(f"symbols={len(symbols)} fetched={result['fetched']} failed={result['failed']}")
    print(f"before: {result['before']}")
    print(f"after:  {result['after']}")
    return 0 if not result["failed"] or result["fetched"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
