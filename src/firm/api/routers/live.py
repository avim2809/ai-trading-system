"""Live trading API router.

Provides endpoints for engine control, positions, orders, approvals,
and live configuration.  Singletons (engine, approval queue) are stored
on ``request.app.state`` and initialised lazily on the first ``/start``.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["live"])

# Serialises engine lifecycle mutations (start/stop/config) so concurrent
# requests can't create two engines or race start against stop on the shared
# app.state singletons.
_engine_lock = threading.Lock()


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
    # Empty = all registered strategies (matches build_orchestrator's default).
    strategies: list[str] = []
    kill_switch_drawdown: float = 0.10
    max_daily_trades: int = 50
    max_daily_turnover: float = 0.5


class ConfigUpdateStrategies(BaseModel):
    enabled: list[str] | None = None
    auto_approve: list[str] | None = None


class ConfigUpdateRisk(BaseModel):
    kill_switch_drawdown: float | None = None
    max_daily_trades: int | None = None
    max_daily_turnover: float | None = None


class ConfigUpdateUniverse(BaseModel):
    symbols: list[str] | None = None


class ConfigUpdateRequest(BaseModel):
    """Mirrors the shape returned by GET /live/config for round-trip saves.

    ``broker`` is intentionally not settable here — changing the execution
    broker requires stopping and restarting the engine.
    """
    schedule: str | None = None
    approval_mode: str | None = None
    strategies: ConfigUpdateStrategies | None = None
    risk: ConfigUpdateRisk | None = None
    universe: ConfigUpdateUniverse | None = None


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

    return {
        "state": "running" if engine.is_running else "stopped",
        "broker": getattr(engine, "_broker_type", ""),
        "broker_connected": engine._broker.is_connected() if engine._broker else False,
        "next_run": next_run,
        "active_strategies": engine.enabled_strategies if hasattr(engine, "enabled_strategies") else [],
        "approval_mode": getattr(engine, "_approval_mode", ""),
        "uptime_seconds": uptime,
        "last_cycle": last_cycle,
    }


@router.post("/start")
def live_start(body: StartRequest, request: Request) -> dict[str, Any]:
    from firm.live.approval import ApprovalQueue
    from firm.live.data_feed import LiveDataFeed
    from firm.live.engine import LiveTradingEngine
    from firm.live.scheduler import TradingScheduler

    with _engine_lock:
        if getattr(request.app.state, "live_engine", None) is not None:
            engine = request.app.state.live_engine
            if engine.is_running:
                raise HTTPException(status_code=409, detail="Engine already running")

        broker = _create_broker(body.broker)
        universe = body.symbols or ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

        # Broker-agnostic market data: FallbackProvider chains
        # Massive -> Tiingo -> AlphaVantage -> FMP per capability, skipping
        # any provider whose key isn't configured. Independent of which
        # broker is used for execution.
        from firm.data.providers.fallback import FallbackProvider

        market_data = FallbackProvider()
        data_feed = LiveDataFeed(
            providers={
                "prices": market_data,
                "fundamentals": market_data,
                "sentiment": market_data,
            },
            universe=universe,
        )

        approval_queue = ApprovalQueue(broker=broker, persist_path="data/approvals.json")
        request.app.state.approval_queue = approval_queue

        from firm.llm.config import load_llm_config, provider_config

        config = {
            "initial_capital": body.initial_capital,
            "symbols": universe,
            # Empty -> build_orchestrator defaults to all registered
            # strategies; a non-empty list enables only those.
            "strategies": body.strategies or None,
            "kill_switch_drawdown": body.kill_switch_drawdown,
            "max_daily_trades": body.max_daily_trades,
            "max_daily_turnover": body.max_daily_turnover,
            # Without these, every analyst silently stays in "quant" mode and
            # the LLM layer (sentiment enhancement, bull/bear/debate, memory
            # reflection) never activates for a run started via this endpoint.
            "agent_modes": load_llm_config().get("agent_modes", {}),
            "llm_config": provider_config(),
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
    with _engine_lock:
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

    result = engine.run_cycle()
    return {
        "cycle_id": result.cycle_id,
        "timestamp": result.timestamp.isoformat(),
        "orders_generated": result.orders_generated,
        "orders_submitted": result.orders_submitted,
        "orders_queued": result.orders_queued,
        "orders_failed": result.orders_failed,
        "skipped": result.skipped,
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


@router.get("/alerts")
def live_alerts(request: Request) -> dict[str, Any]:
    """Operational alerts (drawdown breach, broker outage, degraded recon)
    plus the current kill-switch state."""
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        return {"halted": False, "alerts": []}
    return {
        "halted": engine.halted,
        "alerts": list(reversed(engine.alerts[-100:])),
    }


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
    enabled = engine.enabled_strategies if hasattr(engine, "enabled_strategies") else []
    universe = engine._data_feed._universe if hasattr(engine, "_data_feed") else []
    risk = engine.risk_config if hasattr(engine, "risk_config") else {
        "kill_switch_drawdown": 0.10, "max_daily_trades": 50, "max_daily_turnover": 0.5,
    }

    return {
        "broker": getattr(engine, "_broker_type", "alpaca_paper"),
        "schedule": scheduler._schedule_spec if scheduler else "market_open",
        "approval_mode": getattr(engine, "_approval_mode", "semi_auto"),
        "strategies": {
            "enabled": enabled,
            "auto_approve": auto,
            "require_approval": [s for s in enabled if s not in auto],
        },
        "risk": risk,
        "universe": {
            "symbols": list(universe),
        },
    }


@router.put("/config")
def update_live_config(body: ConfigUpdateRequest, request: Request) -> dict[str, Any]:
    from firm.live.scheduler import TradingScheduler

    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        raise HTTPException(status_code=400, detail="Engine not started")

    if body.approval_mode is not None:
        engine._approval_mode = body.approval_mode
    if body.strategies is not None:
        if body.strategies.auto_approve is not None:
            engine._auto_approve = set(body.strategies.auto_approve)
        if body.strategies.enabled is not None:
            engine.update_strategies(body.strategies.enabled)
    if body.risk is not None:
        engine.update_risk(
            kill_switch_drawdown=body.risk.kill_switch_drawdown,
            max_daily_trades=body.risk.max_daily_trades,
            max_daily_turnover=body.risk.max_daily_turnover,
        )
    if body.universe is not None and body.universe.symbols is not None:
        engine._data_feed._universe = body.universe.symbols

    if body.schedule is not None:
        with _engine_lock:
            old_scheduler = getattr(request.app.state, "live_scheduler", None)
            if old_scheduler is not None:
                old_scheduler.stop()
            try:
                new_scheduler = TradingScheduler(engine=engine, schedule=body.schedule)
                new_scheduler.start()
                request.app.state.live_scheduler = new_scheduler
            except ImportError:
                log.warning("APScheduler not installed; scheduling disabled")
                request.app.state.live_scheduler = None

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
