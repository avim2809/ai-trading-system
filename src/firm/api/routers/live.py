"""Live trading API router.

Provides endpoints for engine control, positions, orders, approvals,
and live configuration.  Singletons (engine, approval queue) are stored
on ``request.app.state`` and initialised lazily on the first ``/start``.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from firm.live.trade_history import TradeHistoryStore

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
_TRADE_HISTORY_ORDERS_PATH = "data/order_history.json"
_TRADE_HISTORY_CYCLES_PATH = "data/cycle_history.json"
_KILL_SWITCH_STATE_PATH = "data/kill_switch_state.json"
_STATE_DB_PATH = "data/live_state.db"
_MEMORY_LOG_PATH = "data/memory/decisions.jsonl"


def _start_live_scheduler(
    app,
    engine,
    schedule: str,
    engine_config: dict[str, Any],
) -> None:
    """Start scheduler, fundamentals refresh, and catch-up after warmup."""
    schedule_tz = engine_config.get("schedule_timezone", "US/Eastern")
    universe = list(engine_config.get("symbols") or [])
    refresh_hour = int(engine_config.get("fundamentals_refresh_hour", 8))
    dynamic_universe_cfg = engine_config.get("danelfin_dynamic_universe") or {}
    try:
        from firm.live.fundamentals_refresh import maybe_refresh_fundamentals_cache_on_start
        from firm.live.pipeline_warmup import PipelineWarmupGate, warmup_wait_seconds
        from firm.live.scheduler import TradingScheduler, maybe_catch_up_session_cycle

        gate = getattr(app.state, "pipeline_warmup_gate", None)
        if gate is None:
            gate = PipelineWarmupGate()
            app.state.pipeline_warmup_gate = gate

        warmup_config = dict(getattr(engine, "_config", engine_config))
        warmup_config.setdefault("strategies", engine_config.get("strategies"))
        gate.start_background(warmup_config)

        def _boot_scheduler() -> None:
            if not gate.wait_ready(timeout=warmup_wait_seconds()):
                log.warning(
                    "Pipeline warmup did not finish in %.0fs — starting scheduler anyway",
                    warmup_wait_seconds(),
                )
            try:
                scheduler = TradingScheduler(
                    engine=engine,
                    schedule=schedule,
                    timezone=schedule_tz,
                    universe=universe,
                    fundamentals_refresh_hour=refresh_hour,
                    dynamic_universe_enabled=bool(dynamic_universe_cfg.get("enabled", False)),
                    dynamic_universe_state_path=dynamic_universe_cfg.get(
                        "state_path", "data/dynamic_universe_state.json"
                    ),
                    dynamic_universe_max_symbols=int(
                        dynamic_universe_cfg.get("max_dynamic_symbols", 10)
                    ),
                    dynamic_universe_min_dwell_days=int(
                        dynamic_universe_cfg.get("min_dwell_days_before_removal", 5)
                    ),
                )
                scheduler.start()
                app.state.live_scheduler = scheduler
                maybe_refresh_fundamentals_cache_on_start(universe)
                maybe_catch_up_session_cycle(
                    engine,
                    schedule,
                    timezone=schedule_tz,
                    warmup_gate=gate,
                )
            except Exception:
                log.error("Scheduler boot failed", exc_info=True)

        threading.Thread(
            target=_boot_scheduler,
            name="live-scheduler-boot",
            daemon=True,
        ).start()
    except ImportError:
        log.warning("APScheduler not installed; scheduling disabled")
        app.state.live_scheduler = None


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
    # Optional behavioural knobs (default OFF/unchanged): macro-event blackout,
    # research signal combination, and TraderAgent allocation method.
    news_guard: dict[str, Any] | None = None
    signal_combination: dict[str, Any] | None = None
    # Generic per-strategy rolling-Sharpe circuit breaker (off unless
    # ``enabled: true``). See firm.agents.research._circuit_breaker and
    # docs/portfolio_construction_diagnosis.md — disabled by default because
    # naive default thresholds were found to over-gate volatile-but-
    # legitimate strategies and net hurt Sharpe in the tested windows.
    strategy_circuit_breaker: dict[str, Any] | None = None
    strategy_regime_weights: dict[str, Any] | None = None
    allocation_method: str | None = None
    kelly_fraction: float | None = None


class ConfigUpdateStrategies(BaseModel):
    enabled: list[str] | None = None
    auto_approve: list[str] | None = None


class ConfigUpdateRisk(BaseModel):
    kill_switch_drawdown: float | None = None
    max_daily_trades: int | None = None
    max_daily_turnover: float | None = None


class ConfigUpdateUniverse(BaseModel):
    symbols: list[str] | None = None


class ConfigUpdateNewsGuard(BaseModel):
    enabled: bool | None = None
    before_min: int | None = None
    after_min: int | None = None
    offline: bool | None = None


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
    news_guard: ConfigUpdateNewsGuard | None = None
    signal_combination: dict[str, Any] | None = None
    strategy_circuit_breaker: dict[str, Any] | None = None
    strategy_regime_weights: dict[str, Any] | None = None
    allocation_method: str | None = None
    kelly_fraction: float | None = None


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


def _get_trade_history(app) -> "TradeHistoryStore":
    from firm.live.trade_history import TradeHistoryStore

    store = getattr(app.state, "trade_history", None)
    if store is None:
        store = TradeHistoryStore(
            orders_path=_TRADE_HISTORY_ORDERS_PATH,
            cycles_path=_TRADE_HISTORY_CYCLES_PATH,
        )
        app.state.trade_history = store
    return store


def _get_approval_queue_readonly(app):
    """Load persisted approvals without requiring a running live engine."""
    from firm.live.approval import ApprovalQueue

    queue = getattr(app.state, "approval_queue", None)
    if queue is None:
        queue = ApprovalQueue(persist_path=_APPROVALS_PATH)
        app.state.approval_queue = queue
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
    from firm.live.notifications import build_alert_callback
    from firm.live.provider_utils import build_live_providers, filter_strategies_for_providers
    from firm.llm.config import load_llm_config, provider_config

    if not engine_config.get("sector_map"):
        # RiskAgent treats a missing sector_map as "cap skipped" and merely
        # logs a warning per cycle — acceptable for a one-off backtest, but
        # live trading runs unattended for weeks, so a documented hard
        # control (max_sector_pct) silently never firing is a real capital
        # risk, not a cosmetic gap. Fail startup instead of degrading.
        log.error(
            "Refusing to start live trading: no risk.sector_map configured "
            "in config/live.yaml — the max_sector_pct concentration cap "
            "would silently never be enforced"
        )
        raise ValueError(
            "Missing risk.sector_map in config/live.yaml: the sector "
            "concentration cap (max_sector_pct) cannot be enforced without "
            "it. Add a sector_map entry for every symbol in the live "
            "universe before starting live trading."
        )

    broker_instance = _create_broker(broker)
    live_providers = build_live_providers(broker)
    data_feed = LiveDataFeed(providers=live_providers, universe=symbols)

    if strategies:
        strategies = filter_strategies_for_providers(strategies, live_providers, logger=log)
        auto_approve = [s for s in auto_approve if s in strategies]

    approval_queue = ApprovalQueue(broker=broker_instance, persist_path=_APPROVALS_PATH)
    app.state.approval_queue = approval_queue

    trade_history = _get_trade_history(app)

    config = {
        **engine_config,
        "symbols": symbols,
        "strategies": strategies,
        "strategy_params": strategy_params,
        "agent_modes": load_llm_config().get("agent_modes", {}),
        "llm_config": provider_config(),
    }
    config.setdefault("respect_market_hours", True)
    config.setdefault("memory_log_path", _MEMORY_LOG_PATH)
    engine = LiveTradingEngine(
        config=config,
        broker=broker_instance,
        data_feed=data_feed,
        approval_queue=approval_queue,
        approval_mode=approval_mode,
        auto_approve_strategies=auto_approve,
        trade_history=trade_history,
        alert_callback=build_alert_callback(),
        kill_switch_state_path=_KILL_SWITCH_STATE_PATH,
        state_db_path=_STATE_DB_PATH,
    )
    engine._broker_type = broker
    # Set synchronously so GET /live/config reflects the requested schedule
    # immediately, rather than the scheduler's own _schedule_spec, which is
    # only assigned once the background warmup-then-boot thread
    # (_start_live_scheduler) finishes — that can lag start-up by seconds.
    engine._schedule = schedule
    engine.start()
    app.state.live_engine = engine

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
        _start_live_scheduler(
            app,
            app.state.live_engine,
            resolved["schedule"],
            resolved["engine_config"],
        )
    except Exception:
        log.exception(
            "Auto-start from config/live.yaml failed; API will continue "
            "(start live manually from the dashboard or POST /api/live/start)"
        )


async def auto_start_live_with_retries(
    app, delays: tuple[float, ...] = (1, 60, 180, 300),
) -> bool:
    """Call :func:`bootstrap_live_from_yaml` with retries on boot.

    Safety net against a boot-race outage seen 2026-07-29:
    ``scripts/wait_for_ibgateway.sh`` (an ``ai-trading.service`` ExecStartPre)
    already gates process start on IB Gateway's port being open — the
    primary fix — but IBC's login can rarely still be mid-flight past that
    (2FA prompt, a slow Gateway/IBC update), and previously nothing retried
    after a single failed attempt, so the engine silently stayed stopped for
    hours with no one aware. ``bootstrap_live_from_yaml`` is safe to call
    repeatedly (it no-ops if the engine is already running).

    Stops retrying — and never tries again — the instant a start is ever
    observed to succeed, so this can never fight an operator's later,
    intentional ``POST /live/stop`` (which clears ``app.state.live_engine``).

    Returns ``True`` if the engine came up (immediately, or after a retry),
    ``False`` if ``FIRM_AUTO_START_LIVE`` isn't enabled or every retry was
    exhausted without success (in which case a ``critical`` alert is fired
    via :func:`firm.live.notifications.build_alert_callback`, if configured).
    """
    import asyncio
    import os

    flag = os.getenv("FIRM_AUTO_START_LIVE", "").lower()
    if flag not in ("1", "true", "yes"):
        return False

    for delay in delays:
        await asyncio.sleep(delay)
        await asyncio.to_thread(bootstrap_live_from_yaml, app)
        engine = getattr(app.state, "live_engine", None)
        if engine is not None and engine.is_running:
            return True

    log.warning(
        "Live engine still not running after boot + retries "
        "(FIRM_AUTO_START_LIVE=1) — start manually via POST /api/live/start "
        "once the underlying issue (e.g. IB Gateway) is resolved",
    )
    from firm.live.notifications import build_alert_callback

    alert_callback = build_alert_callback()
    if alert_callback is not None:
        alert_callback({
            "kind": "auto_start_failed",
            "severity": "critical",
            "message": (
                "Live engine failed to auto-start after boot and all "
                "retries (FIRM_AUTO_START_LIVE=1) — needs manual "
                "POST /api/live/start."
            ),
            "cycle_id": None,
        })
    return False


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
            # No broker instance exists at all yet — genuinely unknown, not
            # a guess at "closed".
            "market_open": None,
            "next_market_open": None,
            "next_market_close": None,
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

    market_open = None
    next_market_open = None
    next_market_close = None
    if engine._broker is not None:
        try:
            mh = engine._broker.market_hours()
            market_open = mh.is_open
            next_market_open = mh.next_open.isoformat() if mh.next_open else None
            next_market_close = mh.next_close.isoformat() if mh.next_close else None
        except Exception:
            log.warning("live_status: market_hours() failed", exc_info=True)

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
        "market_open": market_open,
        "next_market_open": next_market_open,
        "next_market_close": next_market_close,
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
        # Explicit request knobs override live.yaml defaults.
        if body.news_guard is not None:
            engine_config["news_guard"] = body.news_guard
        if body.signal_combination is not None:
            engine_config["signal_combination"] = body.signal_combination
        if body.strategy_circuit_breaker is not None:
            engine_config["strategy_circuit_breaker"] = body.strategy_circuit_breaker
        if body.strategy_regime_weights is not None:
            engine_config["strategy_regime_weights"] = body.strategy_regime_weights
        if body.allocation_method is not None:
            engine_config["allocation_method"] = body.allocation_method
        if body.kelly_fraction is not None:
            engine_config["kelly_fraction"] = body.kelly_fraction

        try:
            result = _start_live_engine(
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    _start_live_scheduler(
        request.app,
        request.app.state.live_engine,
        resolved["schedule"],
        engine_config,
    )
    return result


@router.post("/stop")
def live_stop(request: Request) -> dict[str, Any]:
    shutdown_live_engine(request.app)
    return {"status": "stopped"}


def shutdown_live_engine(app) -> None:
    """Stop scheduler and engine — used by ``/live/stop`` and app shutdown."""
    with _engine_lock:
        scheduler = getattr(app.state, "live_scheduler", None)
        if scheduler is not None:
            scheduler.stop()
            app.state.live_scheduler = None

        engine = getattr(app.state, "live_engine", None)
        if engine is not None:
            engine.stop()
            app.state.live_engine = None


@router.post("/trigger")
def live_trigger(request: Request, force: bool = False, sync: bool = False):
    """Queue one cycle immediately (default) or run synchronously (``sync=true``).

    ``force=true`` bypasses the market-hours check — for deliberate off-hours
    testing; scheduled cycles never do.
    """
    engine = _get_engine(request)

    if sync:
        result = engine.run_cycle(force=force)
        return {
            "status": "completed",
            "cycle_id": result.cycle_id,
            "timestamp": result.timestamp.isoformat(),
            "orders_generated": result.orders_generated,
            "orders_submitted": result.orders_submitted,
            "orders_queued": result.orders_queued,
            "orders_failed": result.orders_failed,
            "skipped": result.skipped,
            "error": result.error,
        }

    def _run() -> None:
        try:
            engine.run_cycle(force=force)
        except Exception:
            log.error("Triggered cycle failed", exc_info=True)

    threading.Thread(
        target=_run,
        name="live-cycle-trigger",
        daemon=True,
    ).start()
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "message": "Cycle queued on engine worker",
            "force": force,
        },
    )


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
            # Explicit long/short label — quantity's sign already implies
            # this, but callers (the dashboard in particular) shouldn't
            # each have to re-derive it themselves.
            "side": "short" if p.quantity < 0 else "long",
        }
        for p in positions
    ]


@router.get("/positions/summary")
def live_positions_summary(request: Request) -> dict[str, Any]:
    """Aggregate long vs. short exposure — mirrors the long/short split
    IBKR itself shows (Client Portal's "Net Asset Value" donut chart)."""
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        return {"long_value": 0.0, "short_value": 0.0, "n_long": 0, "n_short": 0}
    positions = engine._broker.get_positions()
    long_value = sum(p.market_value for p in positions if p.quantity > 0)
    short_value = sum(p.market_value for p in positions if p.quantity < 0)
    return {
        "long_value": long_value,
        "short_value": short_value,
        "n_long": sum(1 for p in positions if p.quantity > 0),
        "n_short": sum(1 for p in positions if p.quantity < 0),
    }


@router.get("/account")
def live_account(request: Request) -> dict[str, Any]:
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        return {"cash": 0, "equity": 0, "buying_power": 0, "currency": "USD"}
    return engine._broker.get_account()


@router.get("/orders")
def live_orders(request: Request) -> list[dict[str, Any]]:
    store = _get_trade_history(request.app)
    persisted = store.list_orders()
    if persisted:
        return persisted

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
    store = _get_trade_history(request.app)
    persisted = store.list_cycles()
    if persisted:
        return persisted

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


@router.get("/portfolio-history")
def live_portfolio_history(request: Request) -> dict[str, Any]:
    """NAV/equity-curve history + portfolio-level performance metrics,
    computed from the same PortfolioSnapshot history persisted every cycle
    to LiveStateStore — the live-trading equivalent of
    GET /runs/{id}/equity + the portfolio_summary section of
    GET /runs/{id}/report (see firm.backtest.run.build_equity_data and
    firm.eval.reports.BacktestReport.portfolio_summary, which this mirrors
    rather than reimplements)."""
    from firm.eval.metrics import compute_all_metrics

    engine = getattr(request.app.state, "live_engine", None)
    empty = {"dates": [], "values": [], "drawdown": [], "metrics": {}, "n_observations": 0}
    if engine is None:
        return empty
    snapshots = engine._portfolio.history
    if not snapshots:
        return empty

    dates = [s.asof.isoformat() for s in snapshots]
    values = [s.nav for s in snapshots]
    peak = 0.0
    drawdown = []
    for nav in values:
        peak = max(peak, nav)
        drawdown.append(round((nav - peak) / peak, 6) if peak > 0 else 0.0)

    navs = pd.Series(values, index=pd.DatetimeIndex([s.asof for s in snapshots]))
    returns = navs.pct_change().dropna()
    metrics = compute_all_metrics(returns) if not returns.empty else {}

    # Sharpe/CAGR/etc. from a handful of cycles are statistically
    # meaningless (e.g. a real annualized Sharpe computed from 4 daily
    # points) — surfaced as its own field so the frontend can caveat
    # rather than silently show a wild, misleading ratio as if it were
    # trustworthy.
    return {
        "dates": dates, "values": values, "drawdown": drawdown,
        "metrics": metrics, "n_observations": len(returns),
    }


@router.get("/attribution")
def live_attribution(request: Request) -> dict[str, dict[str, float]]:
    """Per-strategy performance metrics from the live engine's own
    PerformanceAttribution — the live-trading equivalent of
    GET /runs/{id}/report's ``strategies`` field
    (firm.eval.reports.BacktestReport.strategy_summary)."""
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        return {}
    return engine._attribution.get_strategy_metrics()


@router.delete("/cycles")
def clear_cycles(request: Request) -> dict[str, Any]:
    """Wipe the in-memory cycle/order history.

    Used alongside clearing backtest runs when past history is known to
    be invalid (e.g. after a strategy bug fix) and shouldn't linger in
    the dashboard looking like real trading activity.
    """
    engine = getattr(request.app.state, "live_engine", None)
    if engine is not None:
        return {"cleared": engine.clear_cycle_history()}

    store = getattr(request.app.state, "trade_history", None)
    if store is not None:
        cleared = store.clear_all()
        return {"cleared": cleared["cycles"]}
    return {"cleared": 0}


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


@router.post("/kill-switch/reset")
def reset_kill_switch(request: Request) -> dict[str, Any]:
    """Manually clear the drawdown kill switch and re-arm trading.

    Deliberately requires an explicit operator call — the engine never
    un-halts itself. Halt state (and this reset) persists across process
    restarts via ``data/kill_switch_state.json``, so this is also the only
    way to resume trading after a halt survives a restart.
    """
    engine = getattr(request.app.state, "live_engine", None)
    if engine is None:
        raise HTTPException(status_code=400, detail="Live engine is not running")
    if not engine.halted:
        return {"halted": False, "reset": False, "message": "Kill switch was not tripped"}
    result = engine.reset_kill_switch()
    return {"reset": True, **result}


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------

@router.get("/approvals")
def list_approvals(request: Request) -> list[dict[str, Any]]:
    queue = _get_approval_queue_readonly(request.app)
    return [_serialize_approval(a) for a in queue.get_all()]


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
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    queue = _get_queue(request)
    engine = _get_engine(request)
    try:
        resolved_id = queue.resolve_id(approval_id)
        # IBKR I/O must run on the same worker thread that called connect().
        statuses = engine.run_on_cycle_worker(queue.approve, resolved_id, timeout=120)
    except FuturesTimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Approval timed out waiting for broker (cycle worker busy?)",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    from firm.live.engine import LiveTradingEngine

    store = _get_trade_history(request.app)
    if statuses:
        store.record_orders(
            [LiveTradingEngine._status_to_dict(s, strategy) for s, strategy in statuses],
            source="approval",
            approval_id=resolved_id,
        )

    return {
        "approval_id": resolved_id,
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

def _execution_costs(engine: Any) -> dict[str, float]:
    """Effective commission/slippage rates on the engine's ExecutionAgent."""
    execution = getattr(getattr(engine, "_orchestrator", None), "execution", None)
    if execution is None:
        return {}
    return {
        "commission_pct": execution.commission_pct,
        "slippage_pct": execution.slippage_pct,
        "spread_pct": execution.spread_pct,
    }


@router.get("/config")
def get_live_config(request: Request) -> dict[str, Any]:
    engine = getattr(request.app.state, "live_engine", None)
    scheduler = getattr(request.app.state, "live_scheduler", None)

    if engine is None:
        from firm.strategies.registry import list_strategies
        from firm.live.provider_utils import load_live_yaml_defaults

        all_strategies = list_strategies()
        yaml_defaults = load_live_yaml_defaults()
        ng = yaml_defaults.get("news_guard") or {}
        return {
            "broker": "alpaca_paper",
            "schedule": "market_open",
            "approval_mode": "semi_auto",
            "strategies": {
                "enabled": all_strategies,
                "auto_approve": [],
                "require_approval": all_strategies,
            },
            "strategy_params": yaml_defaults.get("strategy_params", {}),
            "risk": {
                "kill_switch_drawdown": 0.10,
                "max_daily_trades": 50,
                "max_daily_turnover": 0.5,
            },
            "universe": {
                "symbols": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"],
            },
            "news_guard": {
                "enabled": bool(ng.get("enabled", False)),
                "before_min": int(ng.get("before_min", 30)),
                "after_min": int(ng.get("after_min", 15)),
                "offline": bool(ng.get("offline", False)),
            },
            "signal_combination": yaml_defaults.get(
                "signal_combination", {"method": "confidence"}
            ),
            "strategy_circuit_breaker": yaml_defaults.get(
                "strategy_circuit_breaker", {"enabled": False}
            ),
            "strategy_regime_weights": yaml_defaults.get(
                "strategy_regime_weights", {"enabled": False}
            ),
            "allocation_method": yaml_defaults.get(
                "allocation_method", "conviction_weighted"
            ),
            "kelly_fraction": float(yaml_defaults.get("kelly_fraction", 0.5)),
            "costs": dict(yaml_defaults.get("costs") or {}),
        }

    auto = sorted(engine._auto_approve) if hasattr(engine, "_auto_approve") else []
    enabled = engine.enabled_strategies if hasattr(engine, "enabled_strategies") else []
    universe = engine._data_feed._universe if hasattr(engine, "_data_feed") else []
    risk = engine.risk_config if hasattr(engine, "risk_config") else {
        "kill_switch_drawdown": 0.10, "max_daily_trades": 50, "max_daily_turnover": 0.5,
    }

    return {
        "broker": getattr(engine, "_broker_type", "alpaca_paper"),
        # Prefer the last requested schedule (set synchronously on
        # start/update) over the live scheduler's own _schedule_spec, which
        # only becomes accurate once its background boot thread finishes —
        # see the comment in _start_live_engine.
        "schedule": getattr(engine, "_schedule", None)
        or (scheduler._schedule_spec if scheduler else "market_open"),
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
        "news_guard": {
            "enabled": bool(getattr(engine, "_news_guard_enabled", False)),
            "before_min": int(getattr(engine, "_news_guard_before", 30)),
            "after_min": int(getattr(engine, "_news_guard_after", 15)),
            "offline": bool(getattr(engine, "_news_guard_offline", False)),
        },
        "signal_combination": dict(
            getattr(engine, "_config", {}).get("signal_combination")
            or {"method": "confidence"}
        ),
        "strategy_circuit_breaker": dict(
            getattr(engine, "_config", {}).get("strategy_circuit_breaker")
            or {"enabled": False}
        ),
        "strategy_regime_weights": dict(
            getattr(engine, "_config", {}).get("strategy_regime_weights")
            or {"enabled": False}
        ),
        "allocation_method": getattr(engine, "_config", {}).get(
            "allocation_method", "conviction_weighted"
        ),
        "kelly_fraction": float(
            getattr(engine, "_config", {}).get("kelly_fraction", 0.5)
        ),
        # Effective rates on the running ExecutionAgent — reflects whatever
        # config/live.yaml's costs: block resolved to at startup, not just
        # the raw file (so this stays correct even if that ever diverges).
        "costs": _execution_costs(engine),
    }


@router.put("/config")
def update_live_config(body: ConfigUpdateRequest, request: Request) -> dict[str, Any]:
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
        engine.update_universe(body.universe.symbols)
    if body.strategy_params is not None:
        engine.update_strategy_params(body.strategy_params)
    if body.news_guard is not None:
        engine.update_news_guard(
            enabled=body.news_guard.enabled,
            before_min=body.news_guard.before_min,
            after_min=body.news_guard.after_min,
            offline=body.news_guard.offline,
        )
    if body.signal_combination is not None:
        engine.update_signal_combination(body.signal_combination)
    if body.strategy_circuit_breaker is not None:
        engine.update_strategy_circuit_breaker(body.strategy_circuit_breaker)
    if body.strategy_regime_weights is not None:
        engine.update_strategy_regime_weights(body.strategy_regime_weights)
    if body.allocation_method is not None or body.kelly_fraction is not None:
        engine.update_allocation(
            allocation_method=body.allocation_method,
            kelly_fraction=body.kelly_fraction,
        )

    if body.schedule is not None:
        # Set synchronously — see _start_live_engine's comment — so GET
        # /live/config reports the new schedule right away instead of
        # racing the background scheduler-boot thread started below.
        engine._schedule = body.schedule
        with _engine_lock:
            old_scheduler = getattr(request.app.state, "live_scheduler", None)
            if old_scheduler is not None:
                old_scheduler.stop()
            request.app.state.live_scheduler = None
        try:
            _start_live_scheduler(
                request.app, engine, body.schedule, engine._config,
            )
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
