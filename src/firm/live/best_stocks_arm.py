"""Danelfin "Best Stocks Strategy" — a separate, synthetic paper-tracking arm.

This is NOT part of the main risk-managed strategy engine (see
``config/live.yaml``) — it is a standalone implementation of Danelfin's own
published methodology (per the user's verbatim summary of
https://danelfin.com/best-stock-investment-strategy, Cloudflare-blocked so
never independently fetched), run alongside the main book purely to compare
a rules-based ranking approach against this project's own multi-strategy
engine, per the user's explicit request:

    "Build it as a separate tracked arm" — NOT blended into the existing
    universe/strategies, and NOT a real broker-executed second engine
    either (see the design-decision note below).

Danelfin's published rules (as summarized by the user):
    1. Filter the full stock universe: Proven Buy Signal + Low Risk score
       >= 5/10 + average 3-month volume > 100,000 shares.
    2. Rank sectors by the average AI Score of *all* qualifying stocks in
       that sector.
    3. Select the top 5 sectors.
    4. Within each of those 5 sectors, select the 5 highest-AI-Score
       stocks — 25 stocks total, equal-weighted.
    5. Rebalance: every 3 months, replace stocks that no longer meet the
       criteria; every 12 months, rebalance back to equal weighting.
    Danelfin's own backtest (Jan 2017-Jun 2025, 10 Monte Carlo simulations)
    claims S&P 500 outperformance with smaller drawdowns — their claim, not
    independently verified here; this arm exists to form our own view.

Design decision — synthetic NAV ledger, not a second live-trading engine:
    A research pass (see docs/danelfin_best_stocks_arm.md) found this
    project's "LLM A/B" precedent is NOT a concurrent multi-arm setup — it's
    a *sequential* A/B on the same single engine/broker-account/state DB.
    Genuinely running two *simultaneous* engines would need a second
    IBKR client ID, a second systemd unit/process, and a second
    LiveStateStore db (LiveStateStore has no experiment-name column — two
    engines sharing one db file would silently overwrite each other's NAV
    history). That's a real, separate piece of production infrastructure,
    not something to stand up silently as a side effect of a comparison
    exercise. Since the user's actual goal is comparison ("compare against
    your own ranking engine"), not necessarily real order execution, this
    implements a lightweight **synthetic mark-to-market ledger**: real
    market data (Danelfin's live screener + this project's own price
    providers) drives the 25-symbol selection and daily valuation, but no
    broker order is ever placed. See ``scripts/run_best_stocks_arm.py``.

API details verified live before building this (not guessed):
    ``DanelfinProvider.get_trade_ideas`` — see its own docstring for the
    real ``/v3/trade-ideas`` response shape / filter semantics discovered
    doing this work (min-threshold ``aiscore``/``low_risk``/
    ``average_volume_3m`` filters, ``sector`` kebab-case values, 100-row
    cap with no pagination).

Update — this IS backtestable after all, via a different endpoint mode:
    The line above (and this arm's original "structurally unbacktestable"
    framing) was based only on ``/v3/trade-ideas``, which genuinely has no
    history. But ``/ranking`` — the same endpoint already used for the
    genuinely-historical ``danelfin_ai_score`` strategy — also supports a
    **bulk historical mode**: pass ``date``+``sector`` (+``low_risk``)
    with NO ``ticker``, and it returns every matching symbol for that
    historical date, not one ticker's timeline. See
    ``DanelfinProvider.get_historical_sector_scores`` and
    ``select_best_stocks_historical`` below, and
    ``scripts/backtest_best_stocks_arm.py`` for the actual walk-forward
    test this enables. The live arm (``select_best_stocks``,
    ``BestStocksLedger``) is unchanged by this — it's a genuinely separate
    code path exercised only by the backtest script.
"""

from __future__ import annotations

import logging

import pandas as pd

from firm.data.providers.danelfin import DanelfinProvider

log = logging.getLogger("firm.live.best_stocks_arm")

# Kebab-case GICS-like sector values. Only 7 of these 11 were directly
# observed in a live, unfiltered /v3/trade-ideas sample (communication-
# services, consumer-staples, energy, health-care, industrials,
# information-technology, materials); the remaining 4 (financials,
# consumer-discretionary, real-estate, utilities) follow the same
# confirmed kebab-case convention but weren't individually observed. If any
# of these is wrong for this API, that sector's filtered call just returns
# zero candidates and is silently excluded from ranking below — not a
# crash, and not distinguishable from "this sector genuinely has zero
# qualifying stocks today" without further live verification.
SECTORS: list[str] = [
    "communication-services",
    "consumer-discretionary",
    "consumer-staples",
    "energy",
    "financials",
    "health-care",
    "industrials",
    "information-technology",
    "materials",
    "real-estate",
    "utilities",
]

MIN_LOW_RISK = 5
MIN_AVG_VOLUME_3M = 100_000
TOP_N_SECTORS = 5
TOP_N_PER_SECTOR = 5
TARGET_HOLDINGS = TOP_N_SECTORS * TOP_N_PER_SECTOR  # 25


def scan_sector_candidates(
    provider: DanelfinProvider,
    sector: str,
    min_low_risk: int = MIN_LOW_RISK,
    min_avg_volume_3m: int = MIN_AVG_VOLUME_3M,
) -> pd.DataFrame:
    """One /v3/trade-ideas call for *sector*, pre-filtered server-side on
    low_risk/volume (aiscore=1 is a no-op minimum — Danelfin's own
    ``/v3/trade-ideas`` screener already implies a buy call per its
    purpose; see get_trade_ideas's docstring for the "Proven Buy Signal"
    caveat). Capped at 100 rows per sector (the API's own hard limit, no
    pagination) — see SECTORS' docstring note for what an empty result
    means."""
    try:
        df = provider.get_trade_ideas(
            sector=sector,
            aiscore=1,
            low_risk=min_low_risk,
            average_volume_3m=min_avg_volume_3m,
            limit=100,
        )
    except Exception:
        log.warning("best_stocks_sector_scan_failed sector=%s", sector, exc_info=True)
        return pd.DataFrame()
    return df


def select_best_stocks(
    provider: DanelfinProvider,
    top_n_sectors: int = TOP_N_SECTORS,
    top_n_per_sector: int = TOP_N_PER_SECTOR,
    min_low_risk: int = MIN_LOW_RISK,
    min_avg_volume_3m: int = MIN_AVG_VOLUME_3M,
    excluded_symbols: frozenset[str] = frozenset(),
) -> list[dict]:
    """Danelfin Best-Stocks selection: rank sectors by the average AI
    Score of ALL their qualifying candidates, keep the top *top_n_sectors*,
    then within each keep the *top_n_per_sector* highest-AI-Score names.

    A sector needs at least *top_n_per_sector* qualifying candidates to be
    eligible at all — otherwise it can't fill its slots, which would
    silently shrink the target portfolio below 25 names.

    ``excluded_symbols`` (e.g. the main engine's own universe — see
    best_stocks_execution.main_engine_excluded_symbols) is dropped from
    each sector's candidate pool BEFORE ranking/selection, so a colliding
    name never displaces a non-colliding one and the arm still fills its
    slots from tradable candidates wherever the sector has enough of them.

    Returns a list of dicts (one per selected stock):
    ``{symbol, sector, aiscore, low_risk, average_volume_3m,
    sector_avg_aiscore}``, sorted by sector_avg_aiscore desc, then
    aiscore desc within each sector.
    """
    per_sector: dict[str, pd.DataFrame] = {}
    for sector in SECTORS:
        df = scan_sector_candidates(provider, sector, min_low_risk, min_avg_volume_3m)
        if excluded_symbols and not df.empty:
            before = len(df)
            df = df[~df["symbol"].isin(excluded_symbols)]
            if len(df) < before:
                log.info(
                    "best_stocks_excluded_collisions sector=%s dropped=%d (main engine universe)",
                    sector, before - len(df),
                )
        if len(df) >= top_n_per_sector:
            per_sector[sector] = df
        else:
            log.debug(
                "best_stocks_sector_ineligible sector=%s n_candidates=%d (need >= %d)",
                sector, len(df), top_n_per_sector,
            )

    if not per_sector:
        log.warning("best_stocks_no_eligible_sectors")
        return []

    sector_avg = {sector: df["aiscore"].mean() for sector, df in per_sector.items()}
    ranked_sectors = sorted(sector_avg, key=lambda s: sector_avg[s], reverse=True)[:top_n_sectors]

    selected: list[dict] = []
    for sector in ranked_sectors:
        df = per_sector[sector].copy()
        # Tie-break on win_rate_3m when present (more of a real
        # differentiator among same-aiscore names than symbol order).
        sort_cols = ["aiscore"] + (["win_rate_3m"] if "win_rate_3m" in df.columns else [])
        df = df.sort_values(sort_cols, ascending=False).head(top_n_per_sector)
        for _, row in df.iterrows():
            selected.append({
                "symbol": str(row["symbol"]),
                "sector": sector,
                "aiscore": float(row["aiscore"]),
                "low_risk": float(row.get("low_risk", float("nan"))),
                "average_volume_3m": float(row.get("average_volume_3m", float("nan"))),
                "sector_avg_aiscore": float(sector_avg[sector]),
            })
    return selected


def selection_symbols(selection: list[dict]) -> list[str]:
    return [row["symbol"] for row in selection]


def select_best_stocks_historical(
    provider: DanelfinProvider,
    date: str,
    top_n_sectors: int = TOP_N_SECTORS,
    top_n_per_sector: int = TOP_N_PER_SECTOR,
    volume_filter=None,
) -> list[dict]:
    """Historical reconstruction of :func:`select_best_stocks`, using
    :meth:`DanelfinProvider.get_historical_sector_scores` (genuinely
    historical, bulk ``/ranking`` mode) instead of ``/v3/trade-ideas``
    (snapshot-only, no history — see that method's docstring for how this
    was discovered).

    Two real differences from the live version, both forced by what
    Danelfin's historical data actually exposes:

    1. No "Proven Buy Signal" filter — ``/ranking`` has no buy/hold/sell
       field at any date, historical or otherwise. This selection uses
       only low_risk (already server-filtered to >=5 via
       ``get_historical_sector_scores``'s default) + a caller-supplied
       ``volume_filter`` + aiscore ranking.
    2. ``volume_filter``, if given, is a ``symbol -> bool`` predicate the
       CALLER must supply (e.g. backed by this project's own historical
       price/volume data) — Danelfin's bulk ``/ranking`` mode has no
       ``average_volume_3m`` field at all (unlike ``/v3/trade-ideas``).

    A THIRD, deliberate difference from both the live arm and the literal
    Danelfin rule: ``volume_filter`` is only evaluated against candidates
    actually being considered to FILL a top-ranked sector's slots (top
    ``aiscore``-first, walking down only as far as needed to fill
    *top_n_per_sector*), not against every low_risk-qualifying candidate
    in every sector. Checking real historical volume for every qualifying
    candidate turned out to mean hundreds to (for a sector like
    financials) 500+ per-symbol price-history fetches per rebalance date —
    prohibitively slow for a multi-year backtest. This means the SECTOR
    RANKING step (mean aiscore across all low_risk-qualifying candidates)
    is computed WITHOUT the volume filter applied — a real, disclosed
    deviation from Danelfin's literal rule (which filters by volume
    first, then ranks), not a silent one. Document this wherever this
    function's results are reported.

    Returns the same shape as select_best_stocks (plus a ``date`` key),
    ranked by sector_avg_aiscore desc then aiscore desc within sector.
    """
    per_sector: dict[str, pd.DataFrame] = {}
    for sector in SECTORS:
        df = provider.get_historical_sector_scores(sector, date)
        if len(df) >= top_n_per_sector:
            per_sector[sector] = df
        else:
            log.debug(
                "best_stocks_historical_sector_ineligible sector=%s date=%s n=%d (need >= %d)",
                sector, date, len(df), top_n_per_sector,
            )

    if not per_sector:
        log.warning("best_stocks_historical_no_eligible_sectors date=%s", date)
        return []

    # Sector ranking is intentionally NOT volume-filtered — see the
    # docstring's "THIRD difference" above.
    sector_avg = {sector: df["aiscore"].mean() for sector, df in per_sector.items()}
    ranked_sectors = sorted(sector_avg, key=lambda s: sector_avg[s], reverse=True)[:top_n_sectors]

    selected: list[dict] = []
    for sector in ranked_sectors:
        df = per_sector[sector].sort_values("aiscore", ascending=False)
        filled = 0
        for _, row in df.iterrows():
            symbol = str(row["symbol"])
            if volume_filter is not None and not volume_filter(symbol):
                continue
            selected.append({
                "symbol": symbol,
                "sector": sector,
                "aiscore": float(row["aiscore"]),
                "low_risk": float(row["low_risk"]),
                "sector_avg_aiscore": float(sector_avg[sector]),
                "date": date,
            })
            filled += 1
            if filled >= top_n_per_sector:
                break
        if filled < top_n_per_sector:
            log.warning(
                "best_stocks_historical_sector_underfilled sector=%s date=%s filled=%d need=%d "
                "(ran out of low_risk-qualifying candidates passing the volume filter)",
                sector, date, filled, top_n_per_sector,
            )
    return selected
