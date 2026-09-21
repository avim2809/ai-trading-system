"""Adaptive per-sleeve capital reweighting -- scheduled recommendation check.

Opt-in (``capital_reallocation.enabled`` in config/live*.yaml, off by
default), sleeved-mode only. Mirrors ``firm.live.fundamentals_refresh`` /
``news_ingestion_job``'s pattern: an APScheduler entrypoint that runs
unattended on a cadence and never raises out to the scheduler thread.

Deliberately never mutates live capital -- it only computes and logs the
recommendation ``firm.live.capital_reallocation.compute_reallocated_weights``
produces from ``Orchestrator.get_sleeve_metrics()``'s current numbers, for
a human to review via ``GET /api/live/capital-reallocation/recommendation``
before deciding whether to apply it via the separate, explicit
``POST /api/live/capital-reallocation/apply``. Same "visibility before
enforcement" principle as the daily-reflection recommendation queue
(``firm.llm.schemas.DailyReflectionRecommendation``), simplified here since
this recommendation is a deterministic function of current sleeve state
rather than an LLM artifact that must be captured and survive until a
human gets to it -- recomputing it fresh on every read is always correct
and needs no separate persistence or approval-queue plumbing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from firm.live.capital_reallocation import (
    DEFAULT_CAP_PCT,
    DEFAULT_FLOOR_PCT,
    DEFAULT_METRIC,
    DEFAULT_MIN_TRACK_DAYS,
    DEFAULT_TILT_STRENGTH,
    compute_reallocated_weights,
)

if TYPE_CHECKING:
    from firm.live.engine import LiveTradingEngine

log = logging.getLogger(__name__)


def reallocation_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve the ``capital_reallocation`` config block into
    :func:`~firm.live.capital_reallocation.compute_reallocated_weights`
    kwargs, falling back to its module defaults for anything unset."""
    return {
        "metric": cfg.get("metric", DEFAULT_METRIC),
        "floor_pct": float(cfg.get("floor_pct", DEFAULT_FLOOR_PCT)),
        "cap_pct": float(cfg.get("cap_pct", DEFAULT_CAP_PCT)),
        "min_track_days": int(cfg.get("min_track_days", DEFAULT_MIN_TRACK_DAYS)),
        "tilt_strength": float(cfg.get("tilt_strength", DEFAULT_TILT_STRENGTH)),
    }


def compute_recommendation(
    engine: "LiveTradingEngine",
    *,
    metric: str = DEFAULT_METRIC,
    floor_pct: float = DEFAULT_FLOOR_PCT,
    cap_pct: float = DEFAULT_CAP_PCT,
    min_track_days: int = DEFAULT_MIN_TRACK_DAYS,
    tilt_strength: float = DEFAULT_TILT_STRENGTH,
) -> dict[str, Any]:
    """Read-only: what the sleeve capital split would become right now.

    ``sleeves`` is empty (with a ``note`` explaining why) when the engine
    isn't in sleeved mode or no sleeve has ever traded -- there is nothing
    to reweight yet. Never mutates the engine or orchestrator.
    """
    orchestrator = engine._orchestrator
    if getattr(orchestrator, "capital_allocation_mode", "blended") != "sleeved":
        return {
            "sleeves": {},
            "note": "capital_allocation_mode is not 'sleeved' -- nothing to reweight",
        }
    current_weights = orchestrator.sleeve_capital_weights()
    if not current_weights:
        return {"sleeves": {}, "note": "no sleeve has traded yet -- nothing to reweight"}

    sleeve_metrics = orchestrator.get_sleeve_metrics()
    result = compute_reallocated_weights(
        sleeve_metrics,
        current_weights,
        metric=metric,
        floor_pct=floor_pct,
        cap_pct=cap_pct,
        min_track_days=min_track_days,
        tilt_strength=tilt_strength,
    )
    return {
        "sleeves": result,
        "params": {
            "metric": metric,
            "floor_pct": floor_pct,
            "cap_pct": cap_pct,
            "min_track_days": min_track_days,
            "tilt_strength": tilt_strength,
        },
    }


def run_scheduled_capital_reallocation_check(engine: "LiveTradingEngine") -> None:
    """APScheduler entrypoint -- computes and logs the current
    recommendation. No-op unless ``capital_reallocation.enabled`` is set
    on the engine's own config; never applies anything (see module
    docstring). Swallows every exception -- an unattended scheduled job
    must never take the scheduler thread down with it.
    """
    cfg = dict((getattr(engine, "_config", {}) or {}).get("capital_reallocation") or {})
    if not cfg.get("enabled", False):
        return
    try:
        recommendation = compute_recommendation(engine, **reallocation_params(cfg))
        reweighted = {
            s: round(info["new_weight"], 4)
            for s, info in recommendation["sleeves"].items()
            if info.get("reason") == "reweighted"
        }
        log.info(
            "Scheduled capital-reallocation check: %s -- review via GET "
            "/api/live/capital-reallocation/recommendation and apply via "
            "POST .../apply if it looks right; nothing is applied automatically",
            reweighted or recommendation.get("note", "no eligible sleeves"),
        )
    except Exception:
        log.error("Scheduled capital-reallocation check failed", exc_info=True)
