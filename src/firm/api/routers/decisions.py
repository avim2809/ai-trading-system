"""Read-only API for the trading decision/reflection memory log.

Exposes :class:`firm.agents.memory.TradingMemoryLog` (data/memory/decisions.jsonl)
for GUI monitoring — the same entries the live engine's deferred-reflection
loop writes, but as structured records instead of the markdown block used
for LLM prompt injection.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/decisions")
def list_decisions(limit: int = Query(50, ge=1, le=1000)) -> list[dict[str, Any]]:
    """Most recent trading decisions and their reflections, newest first."""
    from firm.agents.memory import TradingMemoryLog

    return TradingMemoryLog().list_decisions(n=limit)


@router.get("/lessons")
def get_lessons(limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    """Aggregated "lessons learned" digest: verdict counts + recent lessons.

    Pure aggregation over already-reflected decisions' structured fields
    (:meth:`firm.agents.memory.TradingMemoryLog.summarize_lessons`) — no new
    LLM call, so this is cheap to poll from the dashboard.
    """
    from firm.agents.memory import TradingMemoryLog

    return TradingMemoryLog().summarize_lessons(n=limit)


@router.get("/recommendations")
def list_recommendations(pending_only: bool = Query(True)) -> list[dict[str, Any]]:
    """Bounded, pre-enumerated recommendations from daily reflection
    rollups (2026-09-20) — see ``firm.llm.schemas.DailyReflectionRecommendation``
    for exactly what these can and can't say (never a free-form config
    edit). Read-only: applying one is a separate, explicit action via
    ``POST /api/live/recommendations/{date}/apply``, since that needs the
    running live engine, not just the on-disk log this reads."""
    from firm.agents.memory import TradingMemoryLog

    return TradingMemoryLog().list_recommendations(pending_only=pending_only)
