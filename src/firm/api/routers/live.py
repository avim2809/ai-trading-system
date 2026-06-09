"""Live trading API router.

Provides endpoints for engine control, positions, orders, approvals,
and live configuration.  Singletons (engine, approval queue) are stored
on ``request.app.state`` and initialised lazily on the first ``/start``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["live"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    broker: str = "alpaca_paper"
    schedule: str = "market_open"
    approval_mode: str = "semi_auto"
    auto_approve_strategies: list[str] = []
    symbols: list[str] = []
    initial_capital: float = 100_000


class ConfigUpdateRequest(BaseModel):
    schedule: str | None = None
    approval_mode: str | None = None
    auto_approve_strategies: list[str] | None = None
    symbols: list[str] | None = None


class RejectRequest(BaseModel):
    reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_engine(request: Request):
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        raise HTTPException(status_code=400, detail="Live engine not started")
    return engine


def _get_queue(request: Request):
    queue = getattr(request.app.state, "approval_queue", None)
    if queue is None:
        raise HTTPException(status_code=400, detail="Live engine not started")
    return queue


# ---------------------------------------------------------------------------
# Engine control
# ---------------------------------------------------------------------------

@router.get("/status")
def live_status(request: Request) -> dict[str, Any]:
    engine = getattr(request.app.state, "live_engine", None)
    scheduler = getattr(request.app.state, "live_scheduler", None)

    if engine is None:
        return {
            "state": "stopped",
            "broker": "",
            "broker_connected": False,
            "next_run": None,
            "active_strategies": [],
            "approval_mode": "",
            "uptime_seconds": None,
            "last_cycle": None,
        }

    next_run = None
    if scheduler is not None:
        nr = scheduler.next_run()
        next_run = nr.isoformat() if nr else None

    uptime = None
    if hasattr(engine, "_started_at") and engine._started_at:
        uptime = (datetime.now() - engine._started_at).total_seconds()

    last_cycle = None
    if engine.cycle_history:
        lc = engine.cycle_history[-1]
        last_cycle = {
            "cycle_id": lc.cycle_id,
            "timestamp": lc.timestamp.isoformat(),
            "orders_generated": lc.orders_generated,
        }

    universe = engine._data_feed._universe if hasattr(engine, "_data_feed") else []

    return {
        "state": "running" if engine.is_running else "stopped",
        "broker": getattr(engine, "_broker_type", ""),
        "broker_connected": engine._broker.is_connected() if engine._broker else False,
        "next_run": next_run,
        "active_strategies": list(universe),
        "approval_mode": getattr(engine, "_approval_mode", ""),
        "uptime_seconds": uptime,
        "last_cycle": last_cycle,
    }


@router.post("/start")
def live_start(body: StartRequest, request: Request) -> dict[str, Any]:
    if getattr(request.app.state, "live_engine", None) is not None:
        engine = request.app.state.live_engine
        if engine.is_running:
            raise HTTPException(status_code=409, detail="Engine already running")

    from firm.live.approval import ApprovalQueue
    from firm.live.data_feed import LiveDataFeed
    from firm.live.engine import LiveTradingEngine
    from firm.live.scheduler import TradingScheduler

    broker = _create_broker(body.broker)
    universe = body.symbols or ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    data_feed = LiveDataFeed(providers={}, universe=universe)

    approval_queue = ApprovalQueue(broker=broker, persist_path="data/approvals.json")
    request.app.state.approval_queue = approval_queue

    config = {
        "initial_capital": body.initial_capital,
        "symbols": universe,
    }
    engine = LiveTradingEngine(
        config=config,
        broker=broker,
        data_feed=data_feed,
        approval_queue=approval_queue,
        approval_mode=body.approval_mode,
        auto_approve_strategies=body.auto_approve_strategies,
    )
    engine.start()
    request.app.state.live_engine = engine

    try:
        scheduler = TradingScheduler(engine=engine, schedule=body.schedule)
        scheduler.start()
        request.app.state.live_scheduler = scheduler
    except ImportError:
        log.warning("APScheduler not installed; scheduling disabled")
        request.app.state.live_scheduler = None

    return {"status": "started", "broker": body.broker, "schedule": body.schedule}


@router.post("/stop")
def live_stop(request: Request) -> dict[str, Any]:
    scheduler = getattr(request.app.state, "live_scheduler", None)
    if scheduler is not None:
        scheduler.stop()
        request.app.state.live_scheduler = None

    engine = getattr(request.app.state, "live_engine", None)
    if engine is not None:
        engine.stop()
        request.app.state.live_engine = None

    return {"status": "stopped"}


@router.post("/trigger")
def live_trigger(request: Request) -> dict[str, Any]:
    engine = _get_engine(request)
    from firm.live.engine import CycleResult

    result = engine.run_cycle()
    return {
        "cycle_id": result.cycle_id,
        "timestamp": result.timestamp.isoformat(),
        "orders_generated": result.orders_generated,
        "orders_submitted": result.orders_submitted,
        "orders_queued": result.orders_queued,
        "error": result.error,
    }


# ---------------------------------------------------------------------------
# Portfolio / Account / Orders
# ---------------------------------------------------------------------------

@router.get("/positions")
def live_positions(request: Request) -> list[dict[str, Any]]:
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        return []
    positions = engine._broker.get_positions()
    return [
        {
            "symbol": p.symbol,
            "quantity": p.quantity,
            "avg_cost": p.avg_cost,
            "market_value": p.market_value,
            "unrealized_pnl": p.unrealized_pnl,
        }
        for p in positions
    ]


@router.get("/account")
def live_account(request: Request) -> dict[str, Any]:
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        return {"cash": 0, "equity": 0, "buying_power": 0, "currency": "USD"}
    return engine._broker.get_account()


@router.get("/orders")
def live_orders(request: Request) -> list[dict[str, Any]]:
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        return []
    result: list[dict[str, Any]] = []
    for cycle in reversed(engine.cycle_history):
        for os_dict in cycle.order_statuses:
            result.append(os_dict)
    return result


@router.get("/cycles")
def live_cycles(request: Request) -> list[dict[str, Any]]:
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        return []
    return [
        {
            "cycle_id": c.cycle_id,
            "timestamp": c.timestamp.isoformat(),
            "orders_generated": c.orders_generated,
            "orders_submitted": c.orders_submitted,
            "orders_queued": c.orders_queued,
            "error": c.error,
        }
        for c in reversed(engine.cycle_history[-50:])
    ]


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------

@router.get("/approvals")
def list_approvals(request: Request) -> list[dict[str, Any]]:
    queue = getattr(request.app.state, "approval_queue", None)
    if queue is None:
        return []
    return [_serialize_approval(a) for a in queue.get_pending()]


@router.get("/approvals/{approval_id}")
def get_approval(approval_id: str, request: Request) -> dict[str, Any]:
    queue = _get_queue(request)
    a = queue.get_by_id(approval_id)
    if a is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return _serialize_approval(a, include_blackboard=True)


@router.post("/approvals/{approval_id}/approve")
def approve_order(approval_id: str, request: Request) -> dict[str, Any]:
    queue = _get_queue(request)
    try:
        statuses = queue.approve(approval_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "approval_id": approval_id,
        "status": "approved",
        "orders_submitted": len(statuses),
    }


@router.post("/approvals/{approval_id}/reject")
def reject_order(approval_id: str, body: RejectRequest, request: Request) -> dict[str, Any]:
    queue = _get_queue(request)
    try:
        queue.reject(approval_id, body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"approval_id": approval_id, "status": "rejected"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@router.get("/config")
def get_live_config(request: Request) -> dict[str, Any]:
    engine = getattr(request.app.state, "live_engine", None)
    scheduler = getattr(request.app.state, "live_scheduler", None)

    if engine is None:
        from firm.strategies.registry import list_strategies
        all_strategies = list_strategies()
        return {
            "broker": "alpaca_paper",
            "schedule": "market_open",
            "approval_mode": "semi_auto",
            "strategies": {
                "enabled": all_strategies,
                "auto_approve": [],
                "require_approval": all_strategies,
            },
            "risk": {
                "kill_switch_drawdown": 0.10,
                "max_daily_trades": 50,
                "max_daily_turnover": 0.5,
            },
            "universe": {
                "symbols": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"],
            },
        }

    auto = sorted(engine._auto_approve) if hasattr(engine, "_auto_approve") else []
    universe = engine._data_feed._universe if hasattr(engine, "_data_feed") else []

    return {
        "broker": getattr(engine, "_broker_type", "alpaca_paper"),
        "schedule": scheduler._schedule_spec if scheduler else "market_open",
        "approval_mode": getattr(engine, "_approval_mode", "semi_auto"),
        "strategies": {
            "enabled": list(universe),
            "auto_approve": auto,
            "require_approval": [s for s in universe if s not in auto],
        },
        "risk": {
            "kill_switch_drawdown": 0.10,
            "max_daily_trades": 50,
            "max_daily_turnover": 0.5,
        },
        "universe": {
            "symbols": list(universe),
        },
    }


@router.put("/config")
def update_live_config(body: ConfigUpdateRequest, request: Request) -> dict[str, Any]:
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        raise HTTPException(status_code=400, detail="Engine not started")

    if body.approval_mode is not None:
        engine._approval_mode = body.approval_mode
    if body.auto_approve_strategies is not None:
        engine._auto_approve = set(body.auto_approve_strategies)
    if body.symbols is not None:
        engine._data_feed._universe = body.symbols

    return {"status": "updated"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_broker(broker_type: str):
    """Factory for broker instances based on config string."""
    import os

    if broker_type in ("alpaca_paper", "alpaca_live"):
        from firm.brokers.alpaca import AlpacaBroker

        return AlpacaBroker(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            paper=(broker_type == "alpaca_paper"),
        )
    elif broker_type in ("ibkr", "ibkr_live"):
        from firm.brokers.ibkr import IBKRBroker

        return IBKRBroker(
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            port=int(os.getenv("IBKR_PORT", "7496")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
        )
    elif broker_type == "ibkr_paper":
        from firm.brokers.ibkr import IBKRBroker

        return IBKRBroker(
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            port=int(os.getenv("IBKR_PAPER_PORT", "7497")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown broker type: {broker_type}")


def _serialize_approval(a, include_blackboard: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "approval_id": a.approval_id,
        "created_at": a.created_at.isoformat(),
        "expires_at": a.expires_at.isoformat(),
        "status": a.status,
        "strategy": a.strategy,
        "orders": a.orders,
        "reject_reason": a.reject_reason,
    }
    if include_blackboard:
        d["blackboard_snapshot"] = a.blackboard_snapshot
    return d
