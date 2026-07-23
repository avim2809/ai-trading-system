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

# Module-level so tests can monkeypatch it to a tmp path — pointing this at
# the real production file (as a hardcoded literal previously did) means
# every test that calls /live/start reads and overwrites live approval
# history with test data.
_APPROVALS_PATH = "data/approvals.json"


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    broker: str | None = None
    schedule: str | None = None
    approval_mode: str | None = None
    auto_approve_strategies: list[str] = []
    symbols: list[str] = []
    initial_capital: float | None = None
    # Empty = all registered strategies (matches build_orchestrator's default).
    strategies: list[str] = []
    kill_switch_drawdown: float | None = None
    max_daily_trades: int | None = None
    max_daily_turnover: float | None = None
    # Skip scheduled/API-triggered cycles when the market is closed rather
    # than running the full pipeline against stale/misleading off-hours
    # quotes. Default on; disable only for deliberate off-hours testing.
    respect_market_hours: bool = True
    # Passthrough for the deeper risk-agent envelope (max_position_pct,
    # max_gross_exposure, max_net_exposure, max_sector_pct, vol_target,
    # max_drawdown_pct, regime_overlay, ...) — mirrors how
    # scripts/run_live_trading.py --config flattens a YAML risk: block
    # straight into the engine config, so the same tuned envelope can be
    # started via this endpoint instead of only via that script.
    risk_overrides: dict[str, Any] = {}
    strategy_params: dict[str, dict[str, Any]] = {}


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
    strategy_params: dict[str, dict[str, Any]] | None = None


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


def _start_live_engine(
    app,
    *,
    broker: str,
    schedule: str,
    approval_mode: str,
    symbols: list[str],
    strategies: list[str] | None,
    strategy_params: dict[str, Any],
    auto_approve: list[str],
    engine_config: dict[str, Any],
) -> dict[str, Any]:
    """Create broker, data feed, engine, and scheduler on ``app.state``."""
    from firm.live.approval import ApprovalQueue
    from firm.live.data_feed import LiveDataFeed
    from firm.live.engine import LiveTradingEngine
    from firm.live.provider_utils import build_live_providers, filter_strategies_for_providers
    from firm.live.scheduler import TradingScheduler
    from firm.llm.config import load_llm_config, provider_config

    broker_instance = _create_broker(broker)
    live_providers = build_live_providers(broker)
    data_feed = LiveDataFeed(providers=live_providers, universe=symbols)

    if strategies:
        strategies = filter_strategies_for_providers(strategies, live_providers, logger=log)
        auto_approve = [s for s in auto_approve if s in strategies]

    approval_queue = ApprovalQueue(broker=broker_instance, persist_path=_APPROVALS_PATH)
    app.state.approval_queue = approval_queue

    config = {
        **engine_config,
        "symbols": symbols,
        "strategies": strategies,
        "strategy_params": strategy_params,
        "agent_modes": load_llm_config().get("agent_modes", {}),
        "llm_config": provider_config(),
    }
    config.setdefault("respect_market_hours", True)
    engine = LiveTradingEngine(
        config=config,
        broker=broker_instance,
        data_feed=data_feed,
        approval_queue=approval_queue,
        approval_mode=approval_mode,
        auto_approve_strategies=auto_approve,
    )
    engine._broker_type = broker
    engine.start()
    app.state.live_engine = engine

    try:
        scheduler = TradingScheduler(engine=engine, schedule=schedule)
        scheduler.start()
        app.state.live_scheduler = scheduler
    except ImportError:
        log.warning("APScheduler not installed; scheduling disabled")
        app.state.live_scheduler = None

    return {"status": "started", "broker": broker, "schedule": schedule}


def bootstrap_live_from_yaml(app) -> None:
    """Start the live engine from ``config/live.yaml`` when enabled for systemd.

    Set ``FIRM_AUTO_START_LIVE=1`` in the service environment so ``firm-api``
    picks up universe, risk, strategies, and ``strategy_params`` on boot without
    using ``scripts/run_live_trading.py``.
    """
    import os

    flag = os.getenv("FIRM_AUTO_START_LIVE", "").lower()
    if flag not in ("1", "true", "yes"):
        return

    try:
        with _engine_lock:
            engine = getattr(app.state, "live_engine", None)
            if engine is not None and engine.is_running:
                log.info("Live engine already running; skipping auto-start")
                return

            from firm.live.provider_utils import resolve_live_startup

            resolved = resolve_live_startup()
            log.info(
                "Auto-starting live engine from config/live.yaml "
                "(broker=%s, %d symbols, %d strategies)",
                resolved["broker"],
                len(resolved["symbols"]),
                len(resolved["strategies"] or []),
            )
            _start_live_engine(
                app,
                broker=resolved["broker"],
                schedule=resolved["schedule"],
                approval_mode=resolved["approval_mode"],
                symbols=resolved["symbols"],
                strategies=resolved["strategies"],
                strategy_params=resolved["strategy_params"],
                auto_approve=resolved["auto_approve"],
                engine_config=resolved["engine_config"],
            )
    except Exception:
        log.exception(
            "Auto-start from config/live.yaml failed; API will continue "
            "(start live manually from the dashboard or POST /api/live/start)"
        )


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
        # Independent of the watchdog alert (which has a real incident of
        # firing failing silently) — a plain clock read an operator/GUI can
        # use to notice a stuck cycle even if the alert path itself breaks.
        "cycle_running_seconds": engine.current_cycle_running_seconds,
    }


@router.post("/start")
def live_start(body: StartRequest, request: Request) -> dict[str, Any]:
    from firm.live.provider_utils import resolve_live_startup

    with _engine_lock:
        if getattr(request.app.state, "live_engine", None) is not None:
            engine = request.app.state.live_engine
            if engine.is_running:
                raise HTTPException(status_code=409, detail="Engine already running")

        resolved = resolve_live_startup(
            broker=body.broker,
            symbols=body.symbols or None,
            strategies=body.strategies or None,
            strategy_params=body.strategy_params if body.strategy_params else None,
            auto_approve=body.auto_approve_strategies,
            initial_capital=body.initial_capital,
            risk_overrides=body.risk_overrides,
            schedule=body.schedule,
            approval_mode=body.approval_mode,
            kill_switch_drawdown=body.kill_switch_drawdown,
            max_daily_trades=body.max_daily_trades,
            max_daily_turnover=body.max_daily_turnover,
        )
        engine_config = dict(resolved["engine_config"])
        engine_config["respect_market_hours"] = body.respect_market_hours

        return _start_live_engine(
            request.app,
            broker=resolved["broker"],
            schedule=resolved["schedule"],
            approval_mode=resolved["approval_mode"],
            symbols=resolved["symbols"],
            strategies=resolved["strategies"],
            strategy_params=resolved["strategy_params"],
            auto_approve=resolved["auto_approve"],
            engine_config=engine_config,
        )


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
def live_trigger(request: Request, force: bool = False) -> dict[str, Any]:
    """Run one cycle immediately. ``force=true`` bypasses the market-hours
    check — for deliberate off-hours testing; scheduled cycles never do."""
    engine = _get_engine(request)

    result = engine.run_cycle(force=force)
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


@router.delete("/cycles")
def clear_cycles(request: Request) -> dict[str, Any]:
    """Wipe the in-memory cycle/order history.

    Used alongside clearing backtest runs when past history is known to
    be invalid (e.g. after a strategy bug fix) and shouldn't linger in
    the dashboard looking like real trading activity.
    """
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        return {"cleared": 0}
    return {"cleared": engine.clear_cycle_history()}


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


@router.delete("/approvals")
def clear_approvals(request: Request) -> dict[str, Any]:
    """Wipe every approval record (pending and historical).

    Used alongside clearing backtest runs when past approvals are known
    to be invalid (e.g. after a strategy bug fix).
    """
    queue = getattr(request.app.state, "approval_queue", None)
    if queue is None:
        return {"cleared": 0}
    return {"cleared": queue.clear()}


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
        from firm.live.provider_utils import load_live_yaml_defaults

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
            "strategy_params": load_live_yaml_defaults().get("strategy_params", {}),
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
        "strategy_params": dict(getattr(engine, "_config", {}).get("strategy_params") or {}),
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
    if body.strategy_params is not None:
        engine.update_strategy_params(body.strategy_params)

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
