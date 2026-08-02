"""Dynamic universe growth driven by Danelfin's real /v3/beststocks list.

Closes a real gap: ``danelfin_best_stocks_signal`` (see
``firm.strategies.danelfin_best_stocks_signal``) only fires for universe
symbols that happen to also appear in Danelfin's real Top-25 — against a
small, mostly US-mega-cap fixed universe, that overlap is usually tiny.
This module lets the live universe grow to include names Danelfin itself
is currently highlighting, with explicit risk bounds the user asked for:

  - **Capped total dynamic exposure** (``max_dynamic_symbols``) — protects
    against unbounded universe growth.
  - **Dwell-based removal** (``min_dwell_days_before_removal``) — a
    dynamically-added symbol must be absent from the day's list for
    several consecutive days before being removed, so a single noisy day
    of list churn doesn't force a full liquidation (existing execution
    logic is all-at-once; this doesn't change that, it just delays when
    it triggers).
  - **Statically-configured symbols are never touched** by absence
    tracking or removal — only symbols this module itself added.

``compute_universe_update`` is pure (no engine/network access) so it's
fully unit-testable in isolation; ``sync_once`` is the thin orchestration
wrapper an APScheduler job (see ``firm.live.scheduler``) actually calls.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from firm.live.dynamic_universe_state import load_dynamic_universe_state, save_dynamic_universe_state
from firm.time_utils import utcnow

log = logging.getLogger(__name__)


def compute_universe_update(
    static_universe: list[str],
    dynamic_state: dict[str, dict[str, Any]],
    today_best_stocks: pd.DataFrame,
    max_dynamic_symbols: int,
    min_dwell_days: int,
    today: str,
) -> tuple[list[str], dict[str, dict[str, Any]], list[str], list[str]]:
    """Compute the next universe + dynamic state from today's best-stocks snapshot.

    Args:
        static_universe:    Symbols from ``config/live.yaml`` — never touched
                             by absence tracking or removal, regardless of
                             whether they also appear in ``today_best_stocks``.
        dynamic_state:      Current ``{symbol: {"sector", "added_date",
                             "consecutive_absent_days"}}`` for symbols this
                             module previously added.
        today_best_stocks:  Today's real Danelfin best_stocks() snapshot
                             (``symbol``, ``sector`` columns at minimum),
                             assumed rank-ordered (best first) — an empty
                             frame means no additions this run and every
                             existing dynamic symbol's absence counter
                             increments. Callers should skip invoking this
                             at all on a fetch *failure* (vs. a genuinely
                             empty real list) to avoid spurious increments
                             from a transient API outage.
        max_dynamic_symbols: Cap on total dynamically-held symbols at once.
        min_dwell_days:     Consecutive absent days required before removal.
        today:              ISO date string, stamped onto newly-added entries.

    Returns:
        ``(new_universe, new_dynamic_state, additions, removals)``.
    """
    static_set = set(static_universe)
    has_data = not today_best_stocks.empty and "symbol" in today_best_stocks.columns
    today_symbols = set(today_best_stocks["symbol"]) if has_data else set()
    sector_by_symbol: dict[str, str] = (
        dict(zip(today_best_stocks["symbol"], today_best_stocks.get("sector", "unknown")))
        if has_data
        else {}
    )

    new_state: dict[str, dict[str, Any]] = {sym: dict(v) for sym, v in dynamic_state.items()}
    removals: list[str] = []

    # Absence tracking + dwell-based removal — dynamic symbols only, never
    # the statically-configured base universe.
    for sym in list(new_state.keys()):
        if sym in today_symbols:
            new_state[sym]["consecutive_absent_days"] = 0
        else:
            absent = int(new_state[sym].get("consecutive_absent_days", 0)) + 1
            new_state[sym]["consecutive_absent_days"] = absent
            if absent >= min_dwell_days:
                del new_state[sym]
                removals.append(sym)

    # Additions — capped at max_dynamic_symbols total dynamic holdings,
    # preserving today_best_stocks' own rank order (best first).
    additions: list[str] = []
    if has_data:
        slots = max(0, max_dynamic_symbols - len(new_state))
        for sym in today_best_stocks["symbol"]:
            if slots <= 0:
                break
            if sym in static_set or sym in new_state:
                continue
            new_state[sym] = {
                "sector": sector_by_symbol.get(sym, "unknown"),
                "added_date": today,
                "consecutive_absent_days": 0,
            }
            additions.append(sym)
            slots -= 1

    dynamic_only = [s for s in new_state if s not in static_set]
    new_universe = list(dict.fromkeys(list(static_universe) + dynamic_only))
    return new_universe, new_state, additions, removals


def sync_once(
    engine: Any,
    *,
    state_path: str | Path,
    static_universe: list[str],
    max_dynamic_symbols: int,
    min_dwell_days: int,
) -> dict[str, Any]:
    """Fetch today's real Danelfin best-stocks list and apply any dynamic
    universe additions/removals to *engine*, in place.

    Reads the ``"best_stocks"`` provider already wired onto the engine's
    data feed (see ``firm.live.provider_utils``, gated on
    ``DANELFIN_API_KEY``) rather than requiring a separate provider handle —
    this function is meant to be called from a scheduled job that only has
    the engine.
    """
    provider = getattr(engine, "_data_feed", None)
    provider = getattr(provider, "_providers", {}).get("best_stocks") if provider is not None else None
    if provider is None:
        log.warning("danelfin_universe_sync: no best_stocks provider configured — skipping")
        return {"skipped": "no_provider"}

    try:
        best_stocks_df = provider.get_best_stocks()
    except Exception:
        log.warning(
            "danelfin_universe_sync: failed to fetch best_stocks — skipping "
            "this run (avoids spuriously incrementing absence counters on a "
            "transient API outage)", exc_info=True,
        )
        return {"skipped": "fetch_failed"}

    state = load_dynamic_universe_state(state_path)
    today = utcnow().date().isoformat()
    new_universe, new_state, additions, removals = compute_universe_update(
        static_universe, state, best_stocks_df, max_dynamic_symbols, min_dwell_days, today,
    )
    save_dynamic_universe_state(state_path, new_state)

    if additions or removals:
        engine.update_universe(new_universe)
        sector_updates = {sym: new_state[sym]["sector"] for sym in additions}
        if sector_updates:
            engine.update_sector_map(sector_updates)
        log.info(
            "danelfin_universe_sync: +%s -%s (universe now %d symbols)",
            additions, removals, len(new_universe),
        )
    else:
        log.debug("danelfin_universe_sync: no universe changes (%d dynamic symbols held)", len(new_state))

    return {"universe": new_universe, "additions": additions, "removals": removals}
