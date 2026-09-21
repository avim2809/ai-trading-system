"""Read-only API for the trading decision/reflection memory log.

Exposes :class:`firm.agents.memory.TradingMemoryLog` (``{FIRM_DATA_DIR}/memory/
decisions.jsonl``) for GUI monitoring — the same entries the live engine's
deferred-reflection loop writes, but as structured records instead of the
markdown block used for LLM prompt injection.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(prefix="/memory", tags=["memory"])

# FIRM_DATA_DIR lets a second firm-api instance (same checkout, e.g. the
# Alpaca instance on :8001) read its own data dir instead of the module
# default -- see firm.api.routers.live's identical _DATA_DIR/
# _MEMORY_LOG_PATH pattern, which the live engine's own TradingMemoryLog is
# built from. Only overridden when FIRM_DATA_DIR is actually *set* (never
# for the unset/"data" default) so this stays a no-op for the IBKR
# instance and for tests that isolate via
# `monkeypatch.setattr(firm.agents.memory, "_DEFAULT_PATH", ...)` -- an
# unconditional override would defeat that monkeypatch by always supplying
# an explicit ``memory_log_path``, which takes priority over
# ``_DEFAULT_PATH``.
_DATA_DIR = os.environ.get("FIRM_DATA_DIR")
_MEMORY_LOG_PATH = f"{_DATA_DIR}/memory/decisions.jsonl" if _DATA_DIR else None


def _memory_log() -> Any:
    from firm.agents.memory import TradingMemoryLog

    if _MEMORY_LOG_PATH:
        return TradingMemoryLog({"memory_log_path": _MEMORY_LOG_PATH})
    return TradingMemoryLog()


@router.get("/decisions")
def list_decisions(limit: int = Query(50, ge=1, le=1000)) -> list[dict[str, Any]]:
    """Most recent trading decisions and their reflections, newest first."""
    return _memory_log().list_decisions(n=limit)


@router.get("/lessons")
def get_lessons(limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    """Aggregated "lessons learned" digest: verdict counts + recent lessons.

    Pure aggregation over already-reflected decisions' structured fields
    (:meth:`firm.agents.memory.TradingMemoryLog.summarize_lessons`) — no new
    LLM call, so this is cheap to poll from the dashboard.
    """
    return _memory_log().summarize_lessons(n=limit)


@router.get("/recommendations")
def list_recommendations(pending_only: bool = Query(True)) -> list[dict[str, Any]]:
    """Bounded, pre-enumerated recommendations from daily reflection
    rollups (2026-09-20) — see ``firm.llm.schemas.DailyReflectionRecommendation``
    for exactly what these can and can't say (never a free-form config
    edit). Read-only: applying one is a separate, explicit action via
    ``POST /api/live/recommendations/{date}/apply``, since that needs the
    running live engine, not just the on-disk log this reads."""
    return _memory_log().list_recommendations(pending_only=pending_only)
