"""On-demand + scheduled chart-pattern scan API (Phases 3 of
docs/pattern_recognition_plan.md).

Exposes :func:`firm.patterns.scanner.scan_symbol` over REST so a frontend can
browse pattern-scan results, and shares its core scan logic (:func:`run_scan`
below) with the optional, independently-scheduled ``firm.live
.pattern_scan_job.PatternScanJob`` — **no background/periodic/scheduled
scanning happens from this router itself**. Every ``POST
/patterns/scan/trigger`` call is synchronous, only run when a human or the
frontend calls it explicitly; nothing in this module is wired into
``firm.api.app``'s lifespan or ``firm.live.scheduler.TradingScheduler``. This
keeps the feature fully inert with respect to the two live paper-trading
engines already running on this box unless a human deliberately sets
``FIRM_ENABLE_PATTERN_SCAN`` (see ``pattern_scan_job.py``) — see CLAUDE.md and
docs/pattern_recognition_plan.md §5.1/§2a.

Two result surfaces:
- ``_SCAN_CACHE``/``_LAST_SCAN`` (module globals): the *latest* trigger's
  results only, process-local and non-persistent — reassigned wholesale
  (never mutated in place) so a concurrent GET always sees either the
  complete previous scan or the complete new one, never a partial write.
- ``PatternScanHistoryStore`` (SQLite, ``firm.live.pattern_scan_history``):
  every match ever persisted, across every trigger and every scheduled run,
  survives a process restart — read via ``GET /patterns/history``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from firm.api.schemas import PatternScanRequest
from firm.api.serializers import safe_value, serialize_dict
from firm.patterns.match import PatternMatch
from firm.time_utils import utcnow

log = logging.getLogger(__name__)

router = APIRouter(prefix="/patterns", tags=["patterns"])

# Populated only by trigger_scan()/run_scan() callers; every read endpoint
# below is a pure, side-effect-free filter over these two globals. See
# module docstring for why this is process-local/non-persistent by design,
# unlike the SQLite history store.
_SCAN_CACHE: list[dict[str, Any]] = []
_LAST_SCAN: dict[str, Any] | None = None
_history_store_instance: Any = None


def _history_store():
    """Lazily-constructed singleton — deferred import so this module (and
    the FastAPI app) still loads even if something about the store's own
    dependencies were ever unavailable, matching this codebase's other
    lazy-singleton accessors.

    Reads ``FIRM_DATA_DIR`` fresh here rather than freezing it into a
    module-level constant at import time (the more common convention
    elsewhere in this codebase, e.g. ``firm.api.app``/``firm.api.routers
    .live`` — fine there since it's only ever set once, before process
    start) specifically so tests can isolate this store per-test via
    ``monkeypatch.setenv`` without needing to reload the module.
    """
    global _history_store_instance
    if _history_store_instance is None:
        from firm.live.pattern_scan_history import PatternScanHistoryStore

        data_dir = os.environ.get("FIRM_DATA_DIR", "data")
        _history_store_instance = PatternScanHistoryStore(
            db_path=f"{data_dir}/pattern_scan_history.db"
        )
    return _history_store_instance


def _serialize_match(symbol: str, asof: datetime, match: PatternMatch) -> dict[str, Any]:
    """``PatternMatch`` -> JSON-safe dict.

    Reuses :mod:`firm.api.serializers`'s NaN/Inf-safe ``safe_value``/
    ``serialize_dict`` helpers — the same ones ``serialize_signal`` already
    uses for every other strategy's ``Signal.meta`` — rather than inventing a
    second sanitization convention.
    """
    return {
        "symbol": symbol,
        "asof": asof.isoformat(),
        "pattern": match.pattern,
        "direction": match.direction,
        "confirmed": match.confirmed,
        "confirm_index": match.confirm_index,
        "entry": safe_value(float(match.entry)),
        "stop": safe_value(float(match.stop)),
        "target": safe_value(float(match.target)),
        "fit_quality": safe_value(float(match.fit_quality)),
        "geometry_tolerance_used": safe_value(float(match.geometry_tolerance_used)),
        "volume_ratio": safe_value(float(match.volume_ratio)),
        "duration_bars": int(match.duration_bars),
        "follow_through_atr": safe_value(float(match.follow_through_atr)),
        "risk_reward": safe_value(float(match.risk_reward)),
        "quality_score": safe_value(float(match.quality_score)),
        "score_breakdown": serialize_dict(match.score_breakdown),
        "pivots": [
            {"index": int(p.index), "price": safe_value(float(p.price)), "kind": p.kind}
            for p in match.pivots
        ],
        "meta": serialize_dict(match.meta),
    }


def _confirm_date(sym_df: pd.DataFrame, confirm_index: int) -> str | None:
    """Calendar date of the confirmation bar, for later outcome-tracking
    alignment (see ``PatternScanHistoryStore.insert_matches``'s
    ``confirm_dates`` param) — distinct from ``asof``, the scan's shared
    reference date.
    """
    try:
        return sym_df["date"].iloc[confirm_index].isoformat()
    except Exception:
        return None


def _sorted_best_first(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(matches, key=lambda r: r["quality_score"] or 0.0, reverse=True)


def run_scan(
    *,
    symbols: list[str],
    asof: str,
    data_source: str,
    lookback_days: int,
    zigzag_pct: float,
    min_score: float,
    confirm_lookback_bars: int,
    stop_atr_floor: float,
    enabled_patterns: list[str] | None,
    seed: int = 42,
) -> dict[str, Any]:
    """Core scan logic, shared by the ``trigger_scan`` route below and the
    optional scheduled ``firm.live.pattern_scan_job.PatternScanJob`` — same
    real data-loading chain (``firm.runtime.load_prices`` ->
    ``PointInTimeDataStore`` -> ``PitViewAdapter``, the exact one every
    backtest and ``POST /api/agents/step`` already use — see
    ``firm.api.routers.agents.agent_step``) and the same
    ``firm.patterns.scanner.scan_symbol`` call, not a second copy of either.

    Per-symbol OHLC is adjusted for splits/dividends via the strategy's own
    ``_adjusted_ohlc`` (reused, not reimplemented). Unlike the strategy
    (which keeps only the single best match per symbol to avoid
    double-counting a symbol in the downstream cross-sectional z-score — see
    docs/pattern_recognition_plan.md deviation #4), this is a browsing/
    persistence surface, not a signal source, so *every* confirmed match
    clearing ``min_score`` is returned, not just the top one per symbol.

    Raises ``ValueError`` for empty ``symbols`` or a malformed ``asof``, and
    propagates ``FileNotFoundError`` from a ``data_source="cache"`` miss —
    callers translate those into whatever's appropriate for their context
    (an HTTP error for the route, a logged skip for the scheduled job).
    """
    from firm.backtest.firm_strategy import PitViewAdapter
    from firm.data.pit_store import PointInTimeDataStore
    from firm.patterns.scanner import scan_symbol
    from firm.strategies.pattern_recognition import _adjusted_ohlc

    if not symbols:
        raise ValueError("symbols must not be empty")

    if data_source == "synthetic":
        from firm.data.synthetic import make_synthetic_prices

        prices_df = make_synthetic_prices(symbols=symbols, end_date=asof, seed=seed)
    else:
        from firm.config import get_settings
        from firm.runtime import load_prices

        prices_df = load_prices(get_settings())

    try:
        asof_dt = datetime.fromisoformat(asof)
    except ValueError as exc:
        raise ValueError(f"invalid asof {asof!r}") from exc

    pit_store = PointInTimeDataStore()
    pit_store.load(prices=prices_df)
    pit_view = PitViewAdapter(pit_store, asof_dt, symbols)
    prices = pit_view.prices(symbols=symbols, lookback_days=lookback_days)

    enabled_set = set(enabled_patterns) if enabled_patterns else None

    got_data = set(prices["symbol"].astype(str)) if not prices.empty else set()
    missing = [s for s in symbols if s not in got_data]
    if missing:
        log.info("Pattern scan: no price data as of %s for %s", asof, missing)

    results: list[dict[str, Any]] = []
    confirm_dates: list[str | None] = []
    failed: list[str] = []
    scanned = 0
    if not prices.empty:
        for symbol, sym_df in prices.groupby("symbol"):
            scanned += 1
            sym_df = sym_df.sort_values("date")
            try:
                ohlcv = _adjusted_ohlc(sym_df)
                matches = scan_symbol(
                    ohlcv,
                    enabled_patterns=enabled_set,
                    zigzag_pct=zigzag_pct,
                    min_score=min_score,
                    confirm_lookback_bars=confirm_lookback_bars,
                    stop_atr_floor=stop_atr_floor,
                )
            except Exception:
                log.warning("Pattern scan failed for %s", symbol, exc_info=True)
                failed.append(str(symbol))
                continue
            for m in matches:
                results.append(_serialize_match(str(symbol), asof_dt, m))
                confirm_dates.append(_confirm_date(sym_df, m.confirm_index))

    return {
        "results": results,
        "confirm_dates": confirm_dates,
        "scanned": scanned,
        "missing": missing,
        "failed": failed,
        "asof_dt": asof_dt,
    }


@router.post("/scan/trigger")
def trigger_scan(req: PatternScanRequest) -> dict[str, Any]:
    """Synchronously scan ``req.symbols`` right now, replace the in-memory
    cache, and persist every match into the durable history store.
    """
    try:
        scan = run_scan(
            symbols=req.symbols,
            asof=req.asof,
            data_source=req.data_source,
            lookback_days=req.lookback_days,
            zigzag_pct=req.zigzag_pct,
            min_score=req.min_score,
            confirm_lookback_bars=req.confirm_lookback_bars,
            stop_atr_floor=req.stop_atr_floor,
            enabled_patterns=req.enabled_patterns,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        log.warning("Pattern scan trigger: no cached price data available: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    results = scan["results"]
    global _SCAN_CACHE, _LAST_SCAN
    _SCAN_CACHE = results
    _LAST_SCAN = {
        "asof": scan["asof_dt"].isoformat(),
        "data_source": req.data_source,
        "symbols_requested": req.symbols,
        "symbols_scanned": scan["scanned"],
        "symbols_missing_data": scan["missing"],
        "symbols_failed": scan["failed"],
        "triggered_at": utcnow().isoformat(),
        "match_count": len(results),
    }
    _history_store().insert_matches(results, confirm_dates=scan["confirm_dates"], source="manual")
    log.info(
        "Pattern scan triggered: %d/%d symbols scanned, %d matches "
        "(data_source=%s, asof=%s)",
        scan["scanned"], len(req.symbols), len(results), req.data_source, req.asof,
    )
    return {"scanned": scan["scanned"], "matches": len(results), "last_scan": _LAST_SCAN}


@router.get("/scan")
def get_scan(
    pattern: str | None = Query(None, description="Filter to one pattern name, e.g. 'bull_flag'"),
    min_score: float | None = Query(None, ge=0, le=100, description="Minimum quality_score"),
    direction: str | None = Query(None, description="Filter to 'long' or 'short'"),
) -> list[dict[str, Any]]:
    """Most recent cached scan results, best (highest quality_score) first.

    ``[]`` if ``POST /patterns/scan/trigger`` has never been called (or
    matched nothing) since this process started — not an error, since an
    empty scan is a perfectly normal starting state.
    """
    results = _SCAN_CACHE
    if pattern:
        results = [r for r in results if r["pattern"] == pattern]
    if direction:
        results = [r for r in results if r["direction"] == direction]
    if min_score is not None:
        results = [r for r in results if (r["quality_score"] or 0.0) >= min_score]
    return _sorted_best_first(results)


@router.get("/summary")
def get_summary() -> dict[str, Any]:
    """Counts by pattern type and direction across the cached scan."""
    by_pattern: dict[str, int] = {}
    by_direction: dict[str, int] = {}
    for r in _SCAN_CACHE:
        by_pattern[r["pattern"]] = by_pattern.get(r["pattern"], 0) + 1
        by_direction[r["direction"]] = by_direction.get(r["direction"], 0) + 1
    return {
        "total": len(_SCAN_CACHE),
        "by_pattern": by_pattern,
        "by_direction": by_direction,
        "last_scan": _LAST_SCAN,
    }


@router.get("/history")
def get_history(
    symbol: str | None = Query(None),
    pattern: str | None = Query(None),
    direction: str | None = Query(None),
    outcome: str | None = Query(
        None, description="'pending' for not-yet-resolved rows, or an exact outcome value"
    ),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    """Durable history of every match ever persisted by a trigger or a
    scheduled scan (``firm.live.pattern_scan_job``, if enabled) — unlike
    ``GET /patterns/scan``, survives a process restart. Paginated/filterable,
    newest first. Registered *before* ``/{symbol}`` below so that fixed path
    isn't shadowed by the catch-all (see that route's own docstring).
    """
    return _history_store().list_history(
        symbol=symbol, pattern=pattern, direction=direction, outcome=outcome,
        limit=limit, offset=offset,
    )


@router.get("/{symbol}")
def get_symbol_patterns(symbol: str) -> list[dict[str, Any]]:
    """All cached pattern matches for one symbol (case-insensitive), best
    first. ``[]`` (not a 404) for a symbol with no confirmed matches in the
    cache -- "no patterns found for this symbol" is a normal, valid result,
    not a missing-resource error.

    Registered after ``/scan``, ``/summary``, ``/scan/trigger`` and
    ``/history`` above so those fixed paths are matched first -- FastAPI/
    Starlette tries routes in registration order, and this catch-all
    single-segment path would otherwise shadow them (e.g. a request for
    ``/patterns/history`` resolving here with ``symbol="history"``).
    """
    sym = symbol.upper()
    results = [r for r in _SCAN_CACHE if r["symbol"].upper() == sym]
    return _sorted_best_first(results)
