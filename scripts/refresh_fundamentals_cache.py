#!/usr/bin/env python3
"""CLI wrapper for fundamentals cache refresh (same logic as firm-api scheduler).

The live engine refreshes automatically at boot (when stale) and on a daily
cron job inside ``TradingScheduler``.  Use this script for manual runs only.

Example::

    .venv/bin/python scripts/refresh_fundamentals_cache.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from firm.live.fundamentals_refresh import refresh_fundamentals_cache  # noqa: E402
from firm.live.provider_utils import load_live_yaml_defaults  # noqa: E402
from firm.logging_setup import setup_logging  # noqa: E402


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="Refresh combined/fundamentals Parquet cache.")
    p.add_argument("--symbols", help="Comma-separated tickers (default: live.yaml universe)")
    p.add_argument("--lookback-days", type=int, default=365 * 6)
    args = p.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        yaml_cfg = load_live_yaml_defaults()
        symbols = list((yaml_cfg.get("universe") or {}).get("symbols") or [])
    if not symbols:
        sys.exit("No symbols to refresh")

    ok = refresh_fundamentals_cache(symbols, lookback_days=args.lookback_days)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
