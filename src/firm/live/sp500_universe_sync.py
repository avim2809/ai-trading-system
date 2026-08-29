"""Dynamic universe growth driven by FMP's S&P 500 constituent list,
sector-balanced against the static universe rather than ranked by any of
this system's own alpha signals (that would be circular/look-ahead-biased
against what then trades them — see the module docstring in
``firm.live.danelfin_universe_sync`` for the sibling design this reuses).

This is the FMP-sourced alternative to ``danelfin_universe_sync`` (Danelfin
is dead — account closed 2026-08-16): same capped-growth, dwell-based-removal
machinery (``compute_universe_update``, imported unchanged from
``danelfin_universe_sync``), fed by a different daily candidate list —
sector water-fill against ``config/live.yaml``'s static 25-name universe,
picking whichever sector is currently least-represented there, then the
most-liquid name (highest trailing dollar volume) within that sector.

Two data dependencies, kept deliberately separate:
  - ``FMPProvider.get_universe_constituents_with_sectors`` — today's *fresh*
    S&P 500 membership list (who is currently in the index at all).
  - ``firm.live.sp500_sector_cache`` — the curated, backfilled, fail-soft
    sector tag for each candidate (refreshed on its own weekly cadence, see
    ``firm.live.scheduler``), preferred over whatever a single day's raw FMP
    pull happened to return, since the cache degrades gracefully across an
    FMP outage while a single day's call does not.

Both a failed FMP fetch and an empty/not-yet-populated sector cache cause a
graceful no-op skip, never a crash and never a spurious absence-counter
bump on already-held dynamic symbols (mirroring the exact caution
``danelfin_universe_sync.sync_once`` already takes on its own fetch
failures).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from firm.data.providers.fmp import FMPProvider
from firm.live.danelfin_universe_sync import compute_universe_update
from firm.live.dynamic_universe_state import load_dynamic_universe_state, save_dynamic_universe_state
from firm.live.sp500_sector_cache import load_sector_cache
from firm.time_utils import utcnow

log = logging.getLogger(__name__)


def build_diversified_candidates(
    constituents_with_sector: pd.DataFrame,
    static_sector_counts: dict[str, int],
    liquidity: dict[str, float],
    max_dynamic_symbols: int,
    exclude: set[str] | list[str] | None = None,
) -> pd.DataFrame:
    """Sector water-fill candidate ranking (pure, no engine/network access).

    Repeatedly assigns the next slot to whichever *sector* currently has
    the fewest represented names — starting from ``static_sector_counts``
    (the fixed static 25-name universe's distribution, never the drifting
    dynamic set, to avoid a feedback loop where today's picks bias
    tomorrow's picks away from themselves for the wrong reason) and rising
    as slots are filled within a run. This front-loads sectors thin/absent
    in the static list (materials, industrials, utilities, staples, real
    estate) ahead of piling on more of an already-heavy sector (tech).
    Within a sector, the highest trailing average dollar volume wins
    (liquidity, not alpha), ties broken alphabetically for determinism.

    Args:
        constituents_with_sector: ``symbol``, ``sector`` columns (typically
            sourced from the persisted sp500 sector cache, not necessarily
            a single day's raw FMP pull). Rows with a missing/"unknown"/
            empty sector are dropped — never a selection candidate, since
            an unknown sector would silently bypass ``max_sector_pct``.
        static_sector_counts: sector -> count of symbols currently in the
            static universe (``config/live.yaml``'s ``risk.sector_map``).
        liquidity: symbol -> trailing average dollar volume (close*volume);
            missing/falsy values are treated as 0 (ranked last).
        max_dynamic_symbols: total slots to fill across all sectors.
        exclude: symbols never eligible (already static or already
            dynamically held).

    Returns:
        Rank-ordered (best first) ``symbol``/``sector`` frame, at most
        ``max_dynamic_symbols`` rows — feeds straight into
        ``compute_universe_update``.
    """
    if constituents_with_sector is None or constituents_with_sector.empty or max_dynamic_symbols <= 0:
        return pd.DataFrame(columns=["symbol", "sector"])

    exclude_set = set(exclude or ())
    pool = constituents_with_sector[
        constituents_with_sector["symbol"].notna()
        & ~constituents_with_sector["symbol"].isin(exclude_set)
        & constituents_with_sector["sector"].notna()
        & (constituents_with_sector["sector"] != "unknown")
        & (constituents_with_sector["sector"] != "")
    ].drop_duplicates(subset="symbol")

    if pool.empty:
        return pd.DataFrame(columns=["symbol", "sector"])

    queues: dict[str, list[str]] = {}
    for sector, group in pool.groupby("sector"):
        queues[sector] = sorted(
            group["symbol"].tolist(),
            key=lambda sym: (-float(liquidity.get(sym, 0.0) or 0.0), sym),
        )

    levels: dict[str, int] = {sector: int(static_sector_counts.get(sector, 0)) for sector in queues}

    picks: list[tuple[str, str]] = []
    while len(picks) < max_dynamic_symbols:
        active_sectors = [sector for sector, queue in queues.items() if queue]
        if not active_sectors:
            break
        next_sector = min(active_sectors, key=lambda sector: (levels[sector], sector))
        symbol = queues[next_sector].pop(0)
        picks.append((symbol, next_sector))
        levels[next_sector] += 1

    return pd.DataFrame(picks, columns=["symbol", "sector"])


def _trailing_dollar_volume(
    prices_provider: Any, symbols: list[str], lookback_days: int,
) -> dict[str, float]:
    """Mean trailing ``close*volume`` per symbol from *prices_provider*.

    Best-effort: a fetch failure logs and returns ``{}`` (every candidate
    then ranks as 0 liquidity within its sector — degraded ranking, not a
    crash of the whole sync).
    """
    if not symbols:
        return {}
    end = utcnow()
    # Calendar-day buffer so weekends/holidays don't starve the lookback of
    # actual trading days.
    start = end - pd.Timedelta(days=max(1, int(lookback_days)) * 2)
    try:
        prices = prices_provider.get_prices(list(symbols), start, end)
    except Exception:
        log.warning(
            "sp500_universe_sync: trailing dollar-volume fetch failed — "
            "liquidity ranking will treat all candidates as 0", exc_info=True,
        )
        return {}
    if prices is None or prices.empty or "close" not in prices.columns or "volume" not in prices.columns:
        return {}
    dollar_volume = prices["close"].astype(float) * prices["volume"].astype(float)
    return dollar_volume.groupby(prices["symbol"]).mean().to_dict()


def sync_once(
    engine: Any,
    *,
    state_path: str | Path,
    sector_cache_path: str | Path,
    static_universe: list[str],
    static_sector_map: dict[str, str],
    max_dynamic_symbols: int,
    min_dwell_days: int,
    liquidity_lookback_days: int = 30,
) -> dict[str, Any]:
    """Fetch today's FMP S&P 500 membership, rank sector-balanced candidates
    by trailing liquidity, and apply any dynamic universe additions/removals
    to *engine*, in place — the FMP-sourced sibling of
    ``danelfin_universe_sync.sync_once``, reusing its
    ``compute_universe_update`` unchanged.

    Reads the ``"prices"`` provider already wired onto the engine's data
    feed (see ``firm.live.provider_utils``) rather than requiring a
    separate provider handle — this function is meant to be called from a
    scheduled job that only has the engine.
    """
    data_feed = getattr(engine, "_data_feed", None)
    prices_provider = getattr(data_feed, "_providers", {}).get("prices") if data_feed is not None else None
    if prices_provider is None:
        log.warning("sp500_universe_sync: no prices provider configured — skipping")
        return {"skipped": "no_provider"}

    try:
        constituents = FMPProvider().get_universe_constituents_with_sectors("sp500")
    except Exception:
        log.warning(
            "sp500_universe_sync: failed to fetch FMP sp500 constituents "
            "(missing FMP_API_KEY, premium-gated endpoint, or transient "
            "outage) — skipping this run (avoids spuriously incrementing "
            "absence counters on a transient/degraded API)", exc_info=True,
        )
        return {"skipped": "fetch_failed"}

    sector_cache = load_sector_cache(sector_cache_path)
    if not sector_cache:
        log.warning(
            "sp500_universe_sync: sector cache is empty — skipping until "
            "it's populated (run sp500_sector_cache.refresh_sector_cache)"
        )
        return {"skipped": "empty_sector_cache"}

    state = load_dynamic_universe_state(state_path)
    static_set = set(static_universe)
    dynamic_held = set(state.keys())

    # Every current FMP constituent not in the static universe, tagged from
    # the curated sector cache (preferred over whatever a single day's raw
    # FMP pull happened to return — the cache is backfilled and fail-soft
    # across an FMP outage; a single day's call is neither).
    non_static_symbols = [sym for sym in constituents.get("symbol", []) if sym not in static_set]
    non_static_with_sector = pd.DataFrame(
        [
            {"symbol": sym, "sector": sector_cache.get(sym, {}).get("sector", "unknown")}
            for sym in non_static_symbols
        ],
        columns=["symbol", "sector"],
    )

    static_sector_counts: dict[str, int] = {}
    for sym in static_universe:
        sector = static_sector_map.get(sym, "unknown")
        if sector and sector != "unknown":
            static_sector_counts[sector] = static_sector_counts.get(sector, 0) + 1

    liquidity = _trailing_dollar_volume(prices_provider, non_static_symbols, liquidity_lookback_days)

    # Fresh candidates only — already-held dynamic symbols are excluded here
    # so a water-fill slot is never "wasted" re-picking a name already held
    # (compute_universe_update would just skip it anyway, but this keeps the
    # capped-at-max_dynamic_symbols output full of genuinely new options).
    fresh_pool = non_static_with_sector[~non_static_with_sector["symbol"].isin(dynamic_held)]
    fresh_ranked = build_diversified_candidates(
        fresh_pool, static_sector_counts, liquidity, max_dynamic_symbols,
    )

    # Already-held dynamic symbols that are STILL a real FMP constituent
    # today must still appear in what's passed to compute_universe_update,
    # purely so it can correctly reset their absence counter — otherwise,
    # since fresh_pool above deliberately excludes them from selection,
    # compute_universe_update would see every already-held symbol as
    # "absent" on every single run regardless of true index membership,
    # forcing a spurious dwell-based removal (and real-money churn) of a
    # position that never actually left the S&P 500.
    held_still_present = non_static_with_sector[non_static_with_sector["symbol"].isin(dynamic_held)]
    ranked = (
        pd.concat([fresh_ranked, held_still_present], ignore_index=True)
        if not held_still_present.empty
        else fresh_ranked
    )

    today = utcnow().date().isoformat()
    new_universe, new_state, additions, removals = compute_universe_update(
        static_universe, state, ranked, max_dynamic_symbols, min_dwell_days, today,
    )
    save_dynamic_universe_state(state_path, new_state)

    if additions or removals:
        engine.update_universe(new_universe)
        sector_updates = {sym: new_state[sym]["sector"] for sym in additions}
        if sector_updates:
            engine.update_sector_map(sector_updates)
        log.info(
            "sp500_universe_sync: +%s -%s (universe now %d symbols)",
            additions, removals, len(new_universe),
        )
    else:
        log.debug("sp500_universe_sync: no universe changes (%d dynamic symbols held)", len(new_state))

    return {"universe": new_universe, "additions": additions, "removals": removals}
