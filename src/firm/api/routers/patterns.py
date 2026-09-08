"""On-demand chart-pattern scan API (Phase 3 of docs/pattern_recognition_plan.md).

Exposes :func:`firm.patterns.scanner.scan_symbol` over REST so a frontend can
browse pattern-scan results without a scheduled job. Deliberately narrow
scope (see the tracker doc's Phase 3 row): no persistence layer, no
trade-outcome-history tracking (that is a distinct, descoped follow-up), and
— most importantly — **no background/periodic/scheduled scanning of any
kind**. Every scan runs synchronously, only when a human or the frontend
calls ``POST /patterns/scan/trigger`` explicitly; nothing here is wired into
``firm.api.app``'s lifespan or ``firm.live.scheduler.TradingScheduler``. This
keeps the feature fully inert with respect to the two live paper-trading
engines already running on this box (see CLAUDE.md and this doc's §5.1) — a
future scheduled EOD scan job is a reasonable follow-up, but requires its own
deliberate, human-reviewed wiring and is intentionally not implemented here.

Results live in a process-local in-memory cache (module globals below),
populated only by the trigger endpoint and read by the three GET endpoints
below. Restarting the API process (or asking a different ``firm-api``
instance, e.g. the :8001 Alpaca one behind this same nginx host) starts from
an empty cache — there is no disk persistence, by design (see the tracker
doc's Phase 3 scope note on why trade-outcome history is a separate feature).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from firm.api.schemas import PatternScanRequest
from firm.api.serializers import safe_value, serialize_dict
from firm.patterns.match import PatternMatch
from firm.time_utils import utcnow

log = logging.getLogger(__name__)

router = APIRouter(prefix="/patterns", tags=["patterns"])

# Populated only by trigger_scan(); every read endpoint below is a pure,
# side-effect-free filter over these two globals. Process-local, in-memory,
# non-persistent by design — see module docstring. Reassigned wholesale
# (never mutated in place) so a concurrent GET always sees either the
# complete previous scan or the complete new one, never a partial write.
_SCAN_CACHE: list[dict[str, Any]] = []
_LAST_SCAN: dict[str, Any] | None = None


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


def _sorted_best_first(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(matches, key=lambda r: r["quality_score"] or 0.0, reverse=True)


@router.post("/scan/trigger")
def trigger_scan(req: PatternScanRequest) -> dict[str, Any]:
    """Synchronously scan ``req.symbols`` right now and replace the cache.

    Real data path (``data_source="cache"``): loads prices via
    ``firm.runtime.load_prices`` -> ``PointInTimeDataStore`` ->
    ``PitViewAdapter`` -- the exact same chain every backtest and
    ``POST /api/agents/step`` already use (see
    ``firm.api.routers.agents.agent_step``), not a new data-loading path.
    ``data_source="synthetic"`` (the default) uses
    ``firm.data.synthetic.make_synthetic_prices`` instead, primarily so this
    endpoint — and its tests — don't require real cached market data.

    Per-symbol OHLC is adjusted for splits/dividends via the strategy's own
    ``_adjusted_ohlc`` (reused, not reimplemented) before being handed to the
    real ``firm.patterns.scanner.scan_symbol``. Unlike the strategy (which
    keeps only the single best match per symbol to avoid double-counting a
    symbol in the downstream cross-sectional z-score — see
    docs/pattern_recognition_plan.md deviation #4), this endpoint is a
    browsing surface, not a signal source, so *every* confirmed match
    clearing ``min_score`` is cached, not just the top one per symbol.
    """
    from firm.backtest.firm_strategy import PitViewAdapter
    from firm.data.pit_store import PointInTimeDataStore
    from firm.patterns.scanner import scan_symbol
    from firm.strategies.pattern_recognition import _adjusted_ohlc

    symbols = req.symbols
    if not symbols:
        raise HTTPException(status_code=422, detail="symbols must not be empty")

    if req.data_source == "synthetic":
        from firm.data.synthetic import make_synthetic_prices

        prices_df = make_synthetic_prices(symbols=symbols, end_date=req.asof, seed=req.seed)
    else:
        from firm.config import get_settings
        from firm.runtime import load_prices

        try:
            prices_df = load_prices(get_settings())
        except FileNotFoundError as exc:
            log.warning("Pattern scan trigger: no cached price data available: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        asof_dt = datetime.fromisoformat(req.asof)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid asof {req.asof!r}") from exc

    pit_store = PointInTimeDataStore()
    pit_store.load(prices=prices_df)
    pit_view = PitViewAdapter(pit_store, asof_dt, symbols)
    prices = pit_view.prices(symbols=symbols, lookback_days=req.lookback_days)

    enabled_set = set(req.enabled_patterns) if req.enabled_patterns else None

    got_data = set(prices["symbol"].astype(str)) if not prices.empty else set()
    missing = [s for s in symbols if s not in got_data]
    if missing:
        log.info(
            "Pattern scan trigger: no price data as of %s for %s", req.asof, missing,
        )

    results: list[dict[str, Any]] = []
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
                    zigzag_pct=req.zigzag_pct,
                    min_score=req.min_score,
                    confirm_lookback_bars=req.confirm_lookback_bars,
                    stop_atr_floor=req.stop_atr_floor,
                )
            except Exception:
                log.warning("Pattern scan failed for %s", symbol, exc_info=True)
                failed.append(str(symbol))
                continue
            for m in matches:
                results.append(_serialize_match(str(symbol), asof_dt, m))

    global _SCAN_CACHE, _LAST_SCAN
    _SCAN_CACHE = results
    _LAST_SCAN = {
        "asof": asof_dt.isoformat(),
        "data_source": req.data_source,
        "symbols_requested": symbols,
        "symbols_scanned": scanned,
        "symbols_missing_data": missing,
        "symbols_failed": failed,
        "triggered_at": utcnow().isoformat(),
        "match_count": len(results),
    }
    log.info(
        "Pattern scan triggered: %d/%d symbols scanned, %d matches "
        "(data_source=%s, asof=%s)",
        scanned, len(symbols), len(results), req.data_source, req.asof,
    )
    return {"scanned": scanned, "matches": len(results), "last_scan": _LAST_SCAN}


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


@router.get("/{symbol}")
def get_symbol_patterns(symbol: str) -> list[dict[str, Any]]:
    """All cached pattern matches for one symbol (case-insensitive), best
    first. ``[]`` (not a 404) for a symbol with no confirmed matches in the
    cache -- "no patterns found for this symbol" is a normal, valid result,
    not a missing-resource error.

    Registered after ``/scan``, ``/summary`` and ``/scan/trigger`` above so
    those fixed paths are matched first -- FastAPI/Starlette tries routes in
    registration order, and this catch-all single-segment path would
    otherwise shadow them (e.g. a request for ``/patterns/scan`` resolving
    here with ``symbol="scan"``).
    """
    sym = symbol.upper()
    results = [r for r in _SCAN_CACHE if r["symbol"].upper() == sym]
    return _sorted_best_first(results)
