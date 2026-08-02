#!/usr/bin/env python
"""One-off fetch: populate the market-percentile calibration cache.

Real API cost, spent deliberately once per invocation — NOT part of the
regular fetch-data pipeline (see firm.runtime.load_market_percentile's
docstring for why). Fetches one full cross-sectional ai_score population
snapshot (~11 sectors x ~6 low_risk values, +pagination = ~66+ Danelfin API
calls) per monthly rebalance date across the chosen window, then writes the
combined panel to ParquetCache's "combined/market_percentile" key so
firm.backtest.run.execute_backtest can load it for
scripts/calibrate_danelfin_market_percentile.py.

Scope decision (2026-08-02): the user has a 10K Danelfin API calls/month
budget, already partially consumed by danelfin_live_signals and the
best-stocks-arm daily job running continuously. A single 18-month window at
MONTHLY (not weekly) snapshot cadence was chosen as a bounded, ~12%-of-budget
validation (~18 dates x ~66+ calls =~ 1,200 calls, ~20 min of API pacing at
Danelfin's paced 1.0s/call — well under their stated 120 calls/min rate
limit too) rather than the full weekly-cadence 3-window A/B every other
Danelfin strategy here got before enabling — see docs/investing_pro_integration.md
for the full cost writeup and rationale.

Usage:
    python scripts/fetch_market_percentile_calibration_data.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.config import get_settings  # noqa: E402
from firm.data.cache import ParquetCache  # noqa: E402
from firm.data.danelfin_market_percentile import fetch_market_percentile_pool  # noqa: E402
from firm.data.providers.danelfin import DanelfinProvider  # noqa: E402

log = logging.getLogger(__name__)

WINDOW_START = "2025-01-01"
WINDOW_END = "2026-06-30"


def _monthly_dates(start: str, end: str) -> list[str]:
    """First-of-month dates from start to end, inclusive of start's month."""
    dates = pd.date_range(start=start, end=end, freq="MS")
    if dates.empty or dates[0] != pd.Timestamp(start):
        dates = pd.DatetimeIndex([pd.Timestamp(start)]).append(dates)
    return [d.strftime("%Y-%m-%d") for d in dates]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    provider = DanelfinProvider(settings=settings)
    cache = ParquetCache(settings.data.cache_dir)

    dates = _monthly_dates(WINDOW_START, WINDOW_END)
    log.info(
        "market_percentile_fetch: %d monthly snapshot dates from %s to %s "
        "(~66+ API calls each, ~1s pacing/call — estimated %.0f min total)",
        len(dates), WINDOW_START, WINDOW_END, len(dates) * 66 / 60.0,
    )

    frames: list[pd.DataFrame] = []
    t0 = time.time()
    for i, date in enumerate(dates, 1):
        log.info("market_percentile_fetch: [%d/%d] date=%s ...", i, len(dates), date)
        df = fetch_market_percentile_pool(provider, date)
        log.info(
            "market_percentile_fetch: [%d/%d] date=%s -> %d symbols (elapsed %.0fs total)",
            i, len(dates), date, len(df), time.time() - t0,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        log.error("market_percentile_fetch: no data fetched for any date — aborting cache write")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    cache.put("combined/market_percentile", combined)
    log.info(
        "market_percentile_fetch: done. %d total rows across %d dates written to "
        "combined/market_percentile (%.1f min elapsed)",
        len(combined), combined["date"].nunique(), (time.time() - t0) / 60.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
