"""Fetch a broad cross-sectional Danelfin ai_score population snapshot.

Backs ``firm.strategies.danelfin_market_percentile`` — a strategy that ranks
each universe symbol's ai_score against the *whole market*, not just this
project's own ~25-name fixed universe (see MARKET_PERCENTILE_COLS in
``firm.data.schemas`` for why that distinction needs its own data shape).

Kept as a standalone module rather than a ``DanelfinProvider`` method to
avoid a circular import: this reuses ``firm.live.best_stocks_arm.SECTORS``
(the confirmed kebab-case sector list), and ``best_stocks_arm`` already
imports ``DanelfinProvider`` — a provider-side import of ``best_stocks_arm``
would create a cycle.

Real cost, not glossed over: one full snapshot (all sectors) costs ~11
sectors x ~6 low_risk values (+pagination) = 66+ Danelfin API calls, each
paced 1.0s apart (``DanelfinProvider._REQUEST_PAUSE_SECONDS``) — roughly
a minute of pure pacing per snapshot, before request latency. This module
does NOT cache results and does NOT decide how often to call itself; that
policy (live per-cycle vs. backtest per-rebalance-date vs. not at all)
belongs to whatever wires this into a PitView, which — as of 2026-08-02 —
nothing does yet. See ``firm.strategies.danelfin_market_percentile``'s
docstring for the current "built and tested, not yet live-wired" status.
"""

from __future__ import annotations

import logging

import pandas as pd

from firm.data.providers.danelfin import DanelfinProvider
from firm.data.schemas import MARKET_PERCENTILE_COLS
from firm.live.best_stocks_arm import SECTORS

log = logging.getLogger(__name__)


def fetch_market_percentile_pool(
    provider: DanelfinProvider,
    date: str,
    sectors: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch and concatenate one historical sector-scores snapshot per
    sector for *date*, remapped to MARKET_PERCENTILE_COLS.

    Args:
        provider: DanelfinProvider instance.
        date:     ISO date string (YYYY-MM-DD) — passed straight through to
                  ``get_historical_sector_scores``'s bulk ``/ranking`` mode.
        sectors:  Override the sector list (defaults to
                  ``firm.live.best_stocks_arm.SECTORS``, the confirmed
                  kebab-case list this project already relies on elsewhere).

    Returns:
        DataFrame with MARKET_PERCENTILE_COLS columns, deduplicated by
        symbol (a symbol should only ever appear under one sector for a
        given date, but dedup defensively in case Danelfin's data disagrees
        with itself across sector-filtered calls).
    """
    frames: list[pd.DataFrame] = []
    for sector in sectors or SECTORS:
        try:
            df = provider.get_historical_sector_scores(sector, date)
        except Exception:
            log.warning(
                "market_percentile_sector_fetch_failed sector=%s date=%s",
                sector, date, exc_info=True,
            )
            continue
        if df.empty:
            continue
        frames.append(pd.DataFrame({
            "date": date,
            "symbol": df["symbol"],
            "sector": sector,
            "ai_score": df.get("aiscore"),
        }))

    if not frames:
        log.warning("market_percentile_pool_empty date=%s", date)
        return pd.DataFrame(columns=MARKET_PERCENTILE_COLS)

    pool = pd.concat(frames, ignore_index=True)
    pool = pool.drop_duplicates(subset=["symbol"])
    return pool[MARKET_PERCENTILE_COLS]
