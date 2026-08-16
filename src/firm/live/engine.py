"""LiveTradingEngine – the core live/paper trading loop.

Replaces BacktestEngine for real-time operation.  Each cycle:
  1. Fetch live prices from data providers  →  PIT store
  2. Sync portfolio state from broker positions
  3. Run Orchestrator.step()
  4. Route resulting orders (auto-submit or queue for approval)
  5. Log everything
"""

from __future__ import annotations

import json
import logging
import threading
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from concurrent.futures import thread as _cf_thread
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from firm.brokers.base import Broker, BrokerError, OrderRequest, OrderStatus
from firm.live.approval import ApprovalQueue
from firm.live.data_feed import LiveDataFeed
from firm.live.portfolio_sync import sync_portfolio_from_broker
from firm.live.scheduler import DEFAULT_MARKET_TIMEZONE, trading_day_key
from firm.live.state_store import LiveStateStore
from firm.portfolio.state import PortfolioState
from firm.runtime import build_orchestrator
from firm.time_utils import utcnow

log = logging.getLogger(__name__)


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """Single-worker pool whose thread is daemon (won't block process exit)."""

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=_cf_thread._worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._create_worker_context(),
                    self._work_queue,
                ),
                daemon=True,
            )
            t.start()
            self._threads.add(t)
            _cf_thread._threads_queues[t] = self._work_queue


@dataclass
class CycleResult:
    """Summary of one engine cycle."""

    cycle_id: int
    timestamp: datetime
    orders_generated: int = 0
    orders_submitted: int = 0
    orders_queued: int = 0
    orders_failed: int = 0
    approval_ids: list[str] = field(default_factory=list)
    order_statuses: list[dict[str, Any]] = field(default_factory=list)
    failed_orders: list[dict[str, Any]] = field(default_factory=list)
    discrepancies: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    skipped: bool = False
    halted: bool = False
    error: str | None = None


class LiveTradingEngine:
    """Drives the agent pipeline against a live broker on each cycle."""

    def __init__(
        self,
        config: dict[str, Any],
        broker: Broker,
        data_feed: LiveDataFeed,
        approval_queue: ApprovalQueue,
        approval_mode: str = "semi_auto",
        auto_approve_strategies: list[str] | None = None,
        alert_callback: Any = None,
        trade_history: Any = None,
        kill_switch_state_path: str | Path | None = None,
        state_db_path: str | Path | None = None,
    ) -> None:
        self._config = config
        self._broker = broker
        self._data_feed = data_feed
        self._approval_queue = approval_queue
        self._trade_history = trade_history
        self._approval_mode = approval_mode
        self._auto_approve: set[str] = set(auto_approve_strategies or [])

        self._orchestrator = build_orchestrator(config)
        # Broker type string (e.g. "ibkr_paper", "ibkr_live"). The API router
        # overrides this after construction; default from config so the
        # execution-safety live lock works even for engines built directly.
        self._broker_type: str = str(config.get("broker", ""))
        self._enabled_strategies: list[str] = list(config.get("strategies") or self._all_strategy_names())
        initial_capital = config.get("initial_capital", 100_000)
        self._portfolio = PortfolioState(initial_capital=initial_capital)
        self._cycle_count = 0
        self._cycle_history: list[CycleResult] = []
        self._running = False
        self._started_at: datetime | None = None

        # Daily risk limits — reset each trading-session calendar day
        # (``schedule_timezone``, default US/Eastern). Unlike the drawdown
        # kill switch (which halts permanently), breaching these forces the
        # cycle's orders to manual approval rather than blocking them
        # outright, so an operator stays in control instead of trades
        # silently vanishing.
        self._max_daily_trades = int(config.get("max_daily_trades", 50))
        self._max_daily_turnover = float(config.get("max_daily_turnover", 0.5))
        # Per-order hard notional cap for the execution_safety.guard_order gate
        # below — reuses RiskAgent's own single-position cap (config/live.yaml
        # risk.max_position_pct) so the final pre-submission check can never be
        # looser than the portfolio-construction limit it's backstopping.
        # 1.0 (no key configured) makes the cap a no-op rather than fabricating
        # a number nothing upstream agreed to.
        self._max_position_pct = float(config.get("max_position_pct", 1.0))
        self._trading_day_timezone = str(
            config.get("schedule_timezone", DEFAULT_MARKET_TIMEZONE),
        )
        self._daily_date: str | None = None
        self._daily_trade_count = 0
        self._daily_turnover_value = 0.0
        # Skip cycles outright when the market is closed — avoids burning a
        # full LLM-enhanced pipeline pass (and IBKR's misleading zero/stale
        # off-hours quotes) on a cycle that can't submit anything useful
        # anyway. Scheduled cycles always respect this; a manual /live/trigger
        # can override with force=True for deliberate off-hours testing.
        self._respect_market_hours = bool(config.get("respect_market_hours", True))
        # Macro-event blackout gate (news-guard). Default OFF — when enabled it
        # holds orders whose instrument sits inside a high-impact economic-event
        # window (FOMC/NFP/CPI...). ``offline`` uses only the bundled calendar.
        ng_cfg = config.get("news_guard") or {}
        self._news_guard_enabled = bool(ng_cfg.get("enabled", False))
        self._news_guard_before = int(ng_cfg.get("before_min", 30))
        self._news_guard_after = int(ng_cfg.get("after_min", 15))
        self._news_guard_offline = bool(ng_cfg.get("offline", False))
        # "forexfactory" (default, free/keyless) or "investing" (richer
        # coverage via the opt-in, off-by-default Investing.com scraper —
        # see firm.data.investing.calendar); either way a live-fetch
        # failure falls through to forexfactory then the bundled CSV.
        self._news_guard_source = str(ng_cfg.get("source", "forexfactory"))
        # Per-strategy return history for the optimal (inverse-covariance)
        # signal combination. Maintained in-process and only updated when
        # ``signal_combination.method == 'optimal'`` — so the confidence-
        # weighted default path is completely unaffected. History resets on
        # restart; until enough cycles accumulate the combiner falls back to
        # the confidence-weighted mean per symbol.
        from firm.portfolio.attribution import PerformanceAttribution

        self._attribution = PerformanceAttribution()
        # Watchdog: a cycle that runs far longer than any normal cycle
        # should (network/broker calls have no universal timeout — e.g. a
        # stale IBKR connection after IB Gateway's mandatory daily restart
        # can hang a blocking call forever with zero error) must not fail
        # silently for hours with no visibility. This only alerts — it
        # deliberately does not try to force-abandon the stuck thread or
        # release the lock, since a thread that resumes later could then
        # race with a new cycle over shared state.
        self._cycle_watchdog_seconds = float(config.get("cycle_watchdog_seconds", 1800))
        self._cycle_hard_timeout_seconds = float(config.get("cycle_hard_timeout_seconds", 900))
        self._watchdog_timer: threading.Timer | None = None
        # Broker disconnect/reconnect tracking (see docs/PROJECT_CONTEXT.md
        # "Broker & host failover"). A single dropped socket shouldn't need a
        # human — the very first broker call each cycle (portfolio sync) is
        # attempted-reconnected inline on the same worker thread ib_async is
        # bound to. Consecutive failures despite reconnect attempts escalate
        # to a distinct, louder alert past this threshold so IB Gateway's
        # occasional multi-cycle outages (e.g. its mandatory daily restart
        # overlapping a cycle) get a human's attention instead of just
        # quietly logging "broker_unavailable" forever.
        self._consecutive_broker_failures = 0
        self._broker_disconnect_alert_threshold = int(
            config.get("broker_disconnect_alert_threshold", 3)
        )
        # Token for the in-flight cycle worker. Cleared on hard timeout so a
        # stale thread cannot submit orders after the lock is released.
        self._active_cycle_token: object | None = None
        self._shutting_down = False
        self._cycle_executor = _DaemonThreadPoolExecutor(
            max_workers=1, thread_name_prefix="live-cycle",
        )
        # Independent of the watchdog timer/alert above: a real incident
        # showed a cycle can hang for 24+ hours with the watchdog's alert
        # never firing (a threading.Timer callback failing silently is its
        # own kind of hang). This plain timestamp needs nothing more than a
        # clock read to answer "is a cycle currently stuck?", so it stays
        # observable via /live/status even if the alert path itself is
        # broken — defense in depth, not a replacement for fixing that.
        self._current_cycle_started_at: datetime | None = None

        # Decision memory — optional; enabled when memory_log_path is configured.
        # Pending decisions (and the NAV to diff against) are persisted to
        # the memory log itself rather than tracked in-memory, so a process
        # restart between the decision and its outcome doesn't silently drop
        # the reflection (see _maybe_reflect).
        from firm.agents.memory import TradingMemoryLog
        self._memory = TradingMemoryLog(config)
        self._llm_service: Any = None

        # Observability/alerting. The kill switch trips when peak-to-trough
        # drawdown breaches ``kill_switch_drawdown`` (falls back to the risk
        # ``max_drawdown_pct``; 1.0 ≈ disabled). ``alert_callback`` is an
        # optional sink (e.g. Slack/email) invoked with each alert dict.
        self._alerts: list[dict[str, Any]] = []
        self._alert_callback = alert_callback
        self._peak_equity = float(initial_capital)
        self._kill_switch_drawdown = float(
            config.get("kill_switch_drawdown", config.get("max_drawdown_pct", 1.0))
        )
        self._halted = False
        # Durable halt state: without this, a process restart after a
        # drawdown trip silently un-halts the engine (in-memory ``_halted``
        # resets to False) and a scheduled cycle can resume trading a
        # blown-up account with no operator ever having made that decision.
        # ``None`` (the default, used by all tests and any direct
        # construction) means no persistence at all — opt-in via the API
        # router, which points this at a real file for production use.
        self._kill_switch_state_path = (
            Path(kill_switch_state_path) if kill_switch_state_path else None
        )
        self._load_kill_switch_state()
        # Durable portfolio-history/attribution state (SQLite). Separate
        # opt-in from kill_switch_state_path: ``None`` (default, all tests
        # and direct construction) keeps engines fully disk-free; the API
        # router points this at a real file for production use, same as the
        # kill-switch JSON path above. Unlike the kill switch, losing this
        # state on restart is not a safety issue (the broker remains the
        # source of truth for cash/holdings), only a continuity one — the
        # equity curve and the ``optimal`` signal-combination's per-strategy
        # return history reset to empty without it.
        self._state_store = (
            LiveStateStore(state_db_path) if state_db_path else None
        )
        self._load_persisted_state()
        # Serialises cycles so a manual/API trigger cannot run concurrently
        # with a scheduled one (which, combined with no broker idempotency,
        # would double-submit real orders).
        self._cycle_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_cycle_running_seconds(self) -> float | None:
        """Seconds the in-progress cycle has been running, or None if idle."""
        if self._current_cycle_started_at is None:
            return None
        return (utcnow() - self._current_cycle_started_at).total_seconds()

    @property
    def portfolio(self) -> PortfolioState:
        return self._portfolio

    @property
    def cycle_history(self) -> list[CycleResult]:
        return list(self._cycle_history)

    def cycles_today(self, timezone: str = "US/Eastern") -> list[dict[str, Any]]:
        """Every cycle summary recorded for the session date, oldest first.

        Merges in-memory history (this process only) with the persisted
        trade-history store (survives restarts), keyed by ``cycle_id`` so a
        cycle recorded in both isn't duplicated. In-memory-only lookups
        previously let a crash-restart mid-session (e.g. a broker-disconnect
        crash loop) wipe all knowledge of a cycle that had already run and
        submitted orders — the scheduler's catch-up job would then fire a
        spurious duplicate cycle for that day.
        """
        from datetime import timezone as dt_timezone
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(timezone)
        today = datetime.now(tz).date()

        def _is_today(ts: datetime) -> bool:
            aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=dt_timezone.utc)
            return aware.astimezone(tz).date() == today

        by_cycle_id: dict[int, dict[str, Any]] = {}
        if self._trade_history is not None:
            for summary in self._trade_history.list_cycles(limit=20):
                raw_ts = summary.get("timestamp")
                if not raw_ts:
                    continue
                try:
                    ts = datetime.fromisoformat(raw_ts)
                except ValueError:
                    continue
                if _is_today(ts):
                    by_cycle_id[summary.get("cycle_id", -1)] = dict(summary)

        for result in self._cycle_history:
            if _is_today(result.timestamp):
                by_cycle_id[result.cycle_id] = {
                    "cycle_id": result.cycle_id,
                    "timestamp": result.timestamp.isoformat(),
                    "orders_generated": result.orders_generated,
                    "orders_submitted": result.orders_submitted,
                    "orders_queued": result.orders_queued,
                    "orders_failed": result.orders_failed,
                    "skipped": result.skipped,
                    "error": result.error,
                }

        return [by_cycle_id[cid] for cid in sorted(by_cycle_id)]

    def had_cycle_today(self, timezone: str = "US/Eastern") -> bool:
        """True when any cycle (including skipped) was recorded for the session date."""
        return bool(self.cycles_today(timezone=timezone))

    def reconcile_order_history(self) -> int:
        """Poll the broker for every locally non-terminal order's true status
        and correct ``order_history.json`` in place.

        See ``firm.live.order_reconciliation`` — orders are otherwise only
        ever recorded once, at submission time, so an async fill/cancel that
        happens after that never reaches the persisted record. Returns the
        number of records corrected; 0 (never raises) if there is no
        attached trade-history store or the reconciliation pass itself
        fails.
        """
        if self._trade_history is None:
            return 0
        from firm.live.order_reconciliation import reconcile_order_statuses

        try:
            return reconcile_order_statuses(self._trade_history, self._broker)
        except Exception:
            log.warning("Order-history reconciliation failed", exc_info=True)
            return 0

    def clear_cycle_history(self) -> int:
        """Wipe the in-memory cycle/order history. Returns the count removed.

        Also clears persisted trade history when a store is attached.
        """
        count = len(self._cycle_history)
        self._cycle_history = []
        if self._trade_history is not None:
            self._trade_history.clear_all()
        return count

    def _persist_cycle_result(self, result: CycleResult) -> None:
        self._persist_live_state()
        if self._trade_history is None:
            return
        summary = {
            "cycle_id": result.cycle_id,
            "timestamp": result.timestamp.isoformat(),
            "orders_generated": result.orders_generated,
            "orders_submitted": result.orders_submitted,
            "orders_queued": result.orders_queued,
            "orders_failed": result.orders_failed,
            "skipped": result.skipped,
            "error": result.error,
        }
        self._trade_history.record_cycle(summary)
        if result.order_statuses:
            self._trade_history.record_orders(
                result.order_statuses,
                cycle_id=result.cycle_id,
                source="cycle",
            )

    @property
    def alerts(self) -> list[dict[str, Any]]:
        """All alerts emitted this session (most recent last)."""
        return list(self._alerts)

    @property
    def halted(self) -> bool:
        """True once the drawdown kill switch has tripped."""
        return self._halted

    @property
    def enabled_strategies(self) -> list[str]:
        return list(self._enabled_strategies)

    @property
    def risk_config(self) -> dict[str, float]:
        return {
            "kill_switch_drawdown": self._kill_switch_drawdown,
            "max_daily_trades": self._max_daily_trades,
            "max_daily_turnover": self._max_daily_turnover,
        }

    @staticmethod
    def _all_strategy_names() -> list[str]:
        from firm.strategies import list_strategies
        return list_strategies()

    def update_strategies(self, names: list[str]) -> None:
        """Swap which strategies the orchestrator runs, effective next cycle.

        Rebuilds the orchestrator with the same config plus the new
        ``strategies`` list — cheap relative to a trading cycle, and the
        portfolio/broker/memory state is untouched.
        """
        self._enabled_strategies = list(names) if names else self._all_strategy_names()
        new_config = {**self._config, "strategies": self._enabled_strategies}
        self._orchestrator = build_orchestrator(new_config)
        self._config = new_config
        log.info("Live engine strategies updated: %s", self._enabled_strategies)

    def update_strategy_params(self, strategy_params: dict[str, Any]) -> None:
        """Replace per-strategy params and rebuild the orchestrator."""
        self._config = {**self._config, "strategy_params": strategy_params}
        self._orchestrator = build_orchestrator(self._config)
        log.info("Live engine strategy_params updated: %s", list(strategy_params))

    def update_universe(self, symbols: list[str]) -> None:
        """Replace the live trading universe, effective next cycle.

        General-purpose (not tied to any one signal source) — a static
        universe doesn't make sense in general for a system with capabilities
        like ``danelfin_universe_sync`` that can identify names worth adding.
        Keeps ``_config['symbols']`` in sync so a later orchestrator rebuild
        (e.g. ``update_strategies``) doesn't silently revert to the old list.
        """
        symbols = list(symbols)
        self._data_feed._universe = symbols
        self._config = {**self._config, "symbols": symbols}
        log.info("Live engine universe updated: %d symbols", len(symbols))

    def update_sector_map(self, sector_map: dict[str, str]) -> None:
        """Merge new symbol->sector entries into the risk agent's sector map.

        Additive (merges, never replaces) — closes the gap where a symbol
        added to the universe outside of ``config/live.yaml`` would
        otherwise fall into ``RiskAgent``'s ``sector_map.get(sym, "unknown")``
        fallback and silently bypass its true sector's concentration cap.
        Also syncs ``_config['sector_map']`` so a later orchestrator rebuild
        (any ``update_*`` that calls ``build_orchestrator`` again) doesn't
        drop the mapping.
        """
        if not sector_map:
            return
        self._orchestrator.risk.sector_map.update(sector_map)
        self._config = {
            **self._config,
            "sector_map": {**self._config.get("sector_map", {}), **sector_map},
        }
        log.info("Live engine sector_map updated: %s", sector_map)

    def update_news_guard(
        self,
        enabled: bool | None = None,
        before_min: int | None = None,
        after_min: int | None = None,
        offline: bool | None = None,
    ) -> None:
        """Update the macro-event blackout gate on a running engine.

        Takes effect on the next cycle. Keeps ``_config['news_guard']`` in sync
        so a subsequent orchestrator rebuild preserves the setting.
        """
        if enabled is not None:
            self._news_guard_enabled = bool(enabled)
        if before_min is not None:
            self._news_guard_before = int(before_min)
        if after_min is not None:
            self._news_guard_after = int(after_min)
        if offline is not None:
            self._news_guard_offline = bool(offline)
        self._config = {
            **self._config,
            "news_guard": {
                "enabled": self._news_guard_enabled,
                "before_min": self._news_guard_before,
                "after_min": self._news_guard_after,
                "offline": self._news_guard_offline,
            },
        }
        log.info(
            "Live engine news_guard updated: enabled=%s before=%d after=%d offline=%s",
            self._news_guard_enabled, self._news_guard_before,
            self._news_guard_after, self._news_guard_offline,
        )

    def update_signal_combination(self, signal_combination: dict[str, Any]) -> None:
        """Switch the research signal-combination method and rebuild the pipeline.

        ``{"method": "confidence"|"optimal"}``. The bull/bear researchers read
        this from config at construction, so the orchestrator is rebuilt.
        """
        self._config = {**self._config, "signal_combination": dict(signal_combination)}
        self._orchestrator = build_orchestrator(self._config)
        log.info("Live engine signal_combination updated: %s", signal_combination)

    def _signal_combination_method(self) -> str:
        """Current research signal-combination method ('confidence'|'optimal')."""
        combo = self._config.get("signal_combination") or {}
        return str(combo.get("method", "confidence"))

    def update_strategy_circuit_breaker(self, strategy_circuit_breaker: dict[str, Any]) -> None:
        """Update the per-strategy rolling-Sharpe circuit breaker and rebuild
        the pipeline (see ``firm.agents.research._circuit_breaker``).

        Disabled by default (``{"enabled": false}``) — see
        ``docs/portfolio_construction_diagnosis.md`` for why the naive
        default thresholds were found to net *hurt* backtested Sharpe and
        should not be enabled without further calibration/validation.
        """
        self._config = {
            **self._config,
            "strategy_circuit_breaker": dict(strategy_circuit_breaker),
        }
        self._orchestrator = build_orchestrator(self._config)
        log.info(
            "Live engine strategy_circuit_breaker updated: %s", strategy_circuit_breaker
        )

    def update_strategy_regime_weights(self, strategy_regime_weights: dict[str, Any]) -> None:
        """Update regime-conditional strategy weight multipliers and rebuild
        the pipeline (see ``firm.agents.research._regime_weights``).

        Disabled by default (``{"enabled": false}``) — calibrate on historical
        windows before enabling live.
        """
        self._config = {
            **self._config,
            "strategy_regime_weights": dict(strategy_regime_weights),
        }
        self._orchestrator = build_orchestrator(self._config)
        log.info(
            "Live engine strategy_regime_weights updated: enabled=%s",
            bool(strategy_regime_weights.get("enabled")),
        )

    def update_allocation(
        self,
        allocation_method: str | None = None,
        kelly_fraction: float | None = None,
    ) -> None:
        """Switch the TraderAgent allocation method and rebuild the pipeline."""
        updates: dict[str, Any] = {}
        if allocation_method is not None:
            updates["allocation_method"] = allocation_method
        if kelly_fraction is not None:
            updates["kelly_fraction"] = float(kelly_fraction)
        if not updates:
            return
        self._config = {**self._config, **updates}
        self._orchestrator = build_orchestrator(self._config)
        log.info("Live engine allocation updated: %s", updates)

    def update_risk(
        self,
        kill_switch_drawdown: float | None = None,
        max_daily_trades: int | None = None,
        max_daily_turnover: float | None = None,
    ) -> None:
        """Update risk limits on a running engine, effective immediately."""
        if kill_switch_drawdown is not None:
            self._kill_switch_drawdown = float(kill_switch_drawdown)
        if max_daily_trades is not None:
            self._max_daily_trades = int(max_daily_trades)
        if max_daily_turnover is not None:
            self._max_daily_turnover = float(max_daily_turnover)
        log.info("Live engine risk limits updated: %s", self.risk_config)

    @staticmethod
    def _order_notional(order: dict[str, Any], prices: dict[str, float]) -> float:
        qty = order.get("quantity", abs(order.get("shares", 0)))
        return abs(qty * prices.get(order.get("symbol", ""), 0.0))

    def _cap_orders_to_daily_budget(
        self, orders: list[dict[str, Any]], prices: dict[str, float], nav: float
    ) -> list[dict[str, Any]]:
        """Trim and pro-rata scale *orders* to fit the day's remaining budget.

        Confirmed live (2026-08-03 through 2026-08-07): every trading day's
        turnover ran 60-95% of NAV against a nominal 25% cap, because
        full_auto let a breach through unchanged — the cap was pure
        telemetry. Trade count is discrete and can't be scaled, so an excess
        order count is truncated to the largest-notional orders that fit the
        remaining slots (the trades most likely to matter for the
        rebalance). Turnover is continuous: once the count fits, every
        surviving order's quantity is scaled down by the same factor so
        total notional lands at (not simply under) the turnover cap, instead
        of some names rebalancing in full while others are dropped outright.
        """
        remaining_slots = max(0, self._max_daily_trades - self._daily_trade_count)
        kept = sorted(
            orders, key=lambda o: self._order_notional(o, prices), reverse=True
        )[:remaining_slots]

        remaining_turnover = max(
            0.0, self._max_daily_turnover * nav - self._daily_turnover_value
        )
        kept_notional = sum(self._order_notional(o, prices) for o in kept)
        scale = min(1.0, remaining_turnover / kept_notional) if kept_notional > 0 else 1.0
        if scale >= 1.0:
            return kept

        scaled = []
        for o in kept:
            qty = o.get("quantity", abs(o.get("shares", 0)))
            new_order = {**o, "quantity": qty * scale}
            if "shares" in o:
                new_order["shares"] = o["shares"] * scale
            if "notional" in o:
                new_order["notional"] = o["notional"] * scale
            scaled.append(new_order)
        return scaled

    def _check_daily_limits(
        self, now: datetime, orders: list[dict[str, Any]], prices: dict[str, float]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Return ``(force_manual, orders_to_submit)``.

        In ``full_auto`` mode the operator has opted out of the approval
        queue, so a breach can't route to manual review — but letting it
        through unchanged made the cap pure telemetry (see
        ``_cap_orders_to_daily_budget``'s docstring for the live incident).
        Instead the order list itself is capped to the day's remaining
        budget. ``semi_auto`` / manual modes still route the whole
        (uncapped) cycle to the approval queue when limits would be
        exceeded, since a human reviews it before anything is capped.
        """
        date_str = trading_day_key(now, self._trading_day_timezone)
        if self._daily_date != date_str:
            self._daily_date = date_str
            self._daily_trade_count = 0
            self._daily_turnover_value = 0.0
            log.debug(
                "Daily limit counters reset for trading day %s (%s)",
                date_str, self._trading_day_timezone,
            )

        turnover_value = sum(self._order_notional(o, prices) for o in orders)
        nav = self._portfolio.nav
        projected_trades = self._daily_trade_count + len(orders)
        projected_turnover_frac = (
            (self._daily_turnover_value + turnover_value) / nav if nav > 0 else 0.0
        )

        breached = (
            projected_trades > self._max_daily_trades
            or projected_turnover_frac > self._max_daily_turnover
        )
        result_orders = orders
        if breached:
            if self._approval_mode == "full_auto":
                msg = (
                    f"Daily limit would be breached (trades {projected_trades}/"
                    f"{self._max_daily_trades}, turnover {projected_turnover_frac:.1%}/"
                    f"{self._max_daily_turnover:.1%}); full_auto — capping to budget."
                )
                # full_auto means no human reviews this cycle — the breach is
                # now contained (capped) rather than bypassed, but a day that
                # needed capping at all is still worth surfacing loudly.
                severity = "critical"
            else:
                msg = (
                    f"Daily limit would be breached (trades {projected_trades}/"
                    f"{self._max_daily_trades}, turnover {projected_turnover_frac:.1%}/"
                    f"{self._max_daily_turnover:.1%}); routing orders to manual approval."
                )
                severity = "warning"
            self._emit_alert("daily_limit_breach", severity, msg)

        if breached and self._approval_mode == "full_auto":
            result_orders = self._cap_orders_to_daily_budget(orders, prices, nav)
            capped_turnover = sum(self._order_notional(o, prices) for o in result_orders)
            log.warning(
                "Daily limit breached but approval_mode=full_auto — capped to budget "
                "(%d/%d orders kept, turnover would've been %.1f%%, capped to %.1f%%)",
                len(result_orders), len(orders), projected_turnover_frac * 100,
                ((self._daily_turnover_value + capped_turnover) / nav * 100) if nav > 0 else 0.0,
            )
            self._daily_trade_count += len(result_orders)
            self._daily_turnover_value += capped_turnover
        elif not breached:
            self._daily_trade_count = projected_trades
            self._daily_turnover_value += turnover_value

        force_manual = breached and self._approval_mode != "full_auto"
        return force_manual, result_orders

    def _emit_alert(
        self, kind: str, severity: str, message: str, **context: Any
    ) -> dict[str, Any]:
        """Record an operational alert, log it, and forward to the callback.

        A failing callback must never break the trading loop, so it is
        guarded. Alerts are also attached to the current cycle result.
        """
        alert = {
            "timestamp": utcnow().isoformat(),
            "kind": kind,
            "severity": severity,
            "message": message,
            "cycle_id": self._cycle_count,
            **context,
        }
        self._alerts.append(alert)
        # Keep memory bounded for long-running sessions.
        if len(self._alerts) > 500:
            self._alerts = self._alerts[-500:]
        log_fn = log.critical if severity == "critical" else log.warning
        log_fn("ALERT [%s/%s] %s %s", severity, kind, message, context or "")
        if self._alert_callback is not None:
            try:
                self._alert_callback(alert)
            except Exception:
                log.warning("Alert callback failed", exc_info=True)
        return alert

    def _apply_news_guard(
        self, orders: list[dict[str, Any]], now: datetime, result: CycleResult
    ) -> list[dict[str, Any]]:
        """Hold orders whose instrument sits inside a high-impact event window.

        Returns the orders that are cleared to proceed. Held orders are dropped
        from this cycle (they are re-generated next cycle once the window
        passes) and each is surfaced as a warning alert. No-op when the gate is
        disabled (the default).

        Fails CLOSED, not open: if the calendar can't be loaded at all (live
        fetch AND the bundled CSV both failed), every order is held — a
        critical ``news_guard_calendar_unavailable`` alert — rather than
        approved blind. A live-fetch failure that still lands on the bundled
        CSV succeeds (as before) but raises a ``news_guard_stale_calendar``
        warning alert, since a static offline calendar can silently miss an
        event scheduled after it was last updated.
        """
        if not self._news_guard_enabled or not orders:
            return orders

        from firm.live import news_guard as ng

        at = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        try:
            events, source = ng.load_events(
                offline=self._news_guard_offline, source=self._news_guard_source,
            )
        except Exception as exc:
            # The whole point of this gate is "don't trade blind into a
            # macro-event window"; if we can't tell where the events are
            # (live fetch AND the bundled CSV both failed — load_events
            # itself already falls back to the CSV on a live-fetch
            # failure), the safe default is to hold every order this
            # cycle, not to silently wave them all through.
            log.error(
                "news-guard calendar totally unavailable (%s) — failing "
                "CLOSED: holding all %d order(s) this cycle rather than "
                "trading blind through a possible blackout window.",
                exc, len(orders), exc_info=True,
            )
            alert = self._emit_alert(
                "news_guard_calendar_unavailable", "critical",
                f"news-guard calendar unavailable ({exc}) — all {len(orders)} "
                "order(s) held this cycle; will retry next cycle.",
            )
            result.alerts.append(alert)
            return []

        if source == "bundled-csv" and not self._news_guard_offline:
            # Only the live-fetch-failed path lands here with offline=False
            # (a deliberately configured offline=True is expected, not a
            # degradation, and would otherwise alert every single cycle).
            age_hours = ng.bundled_csv_age_hours()
            age_desc = f"{age_hours:.1f}h old" if age_hours is not None else "age unknown"
            alert = self._emit_alert(
                "news_guard_stale_calendar", "warning",
                "news-guard live calendar fetch failed — falling back to the "
                f"bundled offline calendar ({age_desc}); blackout decisions may "
                "miss events scheduled after it was last updated.",
            )
            result.alerts.append(alert)

        allowed: list[dict[str, Any]] = []
        # Grouped by blocking event, not emitted per-symbol: a single
        # broad macro event (e.g. a US-wide indicator like ISM Manufacturing
        # PMI) can simultaneously block every symbol in the universe — one
        # alert per symbol would mean dozens of near-identical alerts (and
        # dozens of webhook/Discord posts, since _emit_alert forwards every
        # single one to the alert callback) for what is operationally one
        # event. Keyed by the event's own identity (title/currency/time),
        # not the order list's position, since different symbols can
        # legitimately be blocked by *different* relevant events.
        blocked: dict[tuple[Any, ...], dict[str, Any]] = {}
        for o in orders:
            symbol = o.get("symbol", "")
            try:
                res = ng.decide(
                    symbol, at, events,
                    self._news_guard_before, self._news_guard_after, source,
                )
            except Exception as exc:
                log.warning("news-guard decide failed for %s: %s", symbol, exc)
                allowed.append(o)
                continue
            if res.get("decision") == "block":
                event = res.get("blocking_event") or {}
                key = (event.get("title"), event.get("currency"), event.get("time"))
                bucket = blocked.setdefault(key, {
                    "reason": res.get("reason", ""),
                    "first_symbol": symbol.upper(),
                    "blocking_event": res.get("blocking_event"),
                    "symbols": [],
                })
                bucket["symbols"].append(symbol.upper())
            else:
                allowed.append(o)

        for bucket in blocked.values():
            symbols = sorted(bucket["symbols"])
            # The per-symbol reason already ends in "... for {SYMBOL}." —
            # swap that trailing clause for the full held-symbol list rather
            # than re-deriving the whole sentence (duplicating decide()'s
            # own phrasing logic here would drift out of sync with it).
            # Match against the symbol that actually produced this reason
            # string (the first one seen for this event), not symbols[0]
            # post-sort — those can differ, which would silently fail the
            # suffix match and leave a dangling double clause.
            reason = bucket["reason"]
            suffix = f" for {bucket['first_symbol']}."
            if reason.endswith(suffix):
                reason = reason[: -len(suffix)]
            reason += f" for {len(symbols)} symbol(s): {', '.join(symbols)}."
            alert = self._emit_alert(
                "news_guard_blackout", "warning", reason,
                symbols=symbols, blocking_event=bucket["blocking_event"],
            )
            result.alerts.append(alert)
        return allowed

    def _check_drawdown(self, result: CycleResult) -> None:
        """Update peak equity and trip the kill switch on a drawdown breach."""
        nav = self._portfolio.nav
        if nav <= 0:
            return
        self._peak_equity = max(self._peak_equity, nav)
        if self._peak_equity <= 0:
            return
        drawdown = (self._peak_equity - nav) / self._peak_equity
        if drawdown >= self._kill_switch_drawdown and not self._halted:
            self._halted = True
            alert = self._emit_alert(
                "drawdown_breach",
                "critical",
                f"Drawdown {drawdown:.1%} breached kill switch "
                f"{self._kill_switch_drawdown:.1%}; halting new orders.",
                drawdown=round(drawdown, 6),
                nav=round(nav, 2),
                peak_equity=round(self._peak_equity, 2),
            )
            result.alerts.append(alert)
            self._persist_kill_switch_state(
                reason=alert["message"], drawdown=drawdown, nav=nav,
            )
        elif self._kill_switch_state_path is not None and not self._halted:
            # Keep peak-equity tracking durable even while not halted, so a
            # restart mid-drawdown (but before the trip threshold) doesn't
            # reset the high-water mark and understate a subsequent breach.
            # Only reached when not halted: once tripped, self._halted stays
            # True and every later cycle must leave the persisted halt (and
            # its reason/tripped_at) alone rather than silently clearing it.
            self._persist_kill_switch_state(halted=False)

    def _load_kill_switch_state(self) -> None:
        """Restore ``_halted``/``_peak_equity`` from disk, if persisted.

        No-op when persistence is disabled (default) or the file doesn't
        exist yet (first run). Corrupt/unreadable state fails safe by
        leaving the in-memory defaults (not halted) rather than crashing
        engine construction — but logs loudly since a lost halt is a real
        safety regression, not a routine fallback.
        """
        if self._kill_switch_state_path is None:
            return
        path = self._kill_switch_state_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception:
            log.warning(
                "Failed to read kill-switch state from %s — starting "
                "un-halted; verify manually that this is safe", path,
                exc_info=True,
            )
            return
        self._halted = bool(data.get("halted", False))
        if "peak_equity" in data:
            self._peak_equity = max(self._peak_equity, float(data["peak_equity"]))
        if self._halted:
            log.warning(
                "Restored HALTED kill-switch state from %s (reason=%s, "
                "tripped_at=%s) — engine will refuse new orders until "
                "reset_kill_switch() is called", path,
                data.get("reason"), data.get("tripped_at"),
            )
        else:
            log.info("Restored kill-switch state from %s (not halted)", path)

    def _persist_kill_switch_state(
        self,
        *,
        halted: bool | None = None,
        reason: str | None = None,
        drawdown: float | None = None,
        nav: float | None = None,
    ) -> None:
        if self._kill_switch_state_path is None:
            return
        data: dict[str, Any] = {
            "halted": self._halted if halted is None else halted,
            "peak_equity": round(self._peak_equity, 2),
        }
        if reason is not None:
            data["reason"] = reason
            data["tripped_at"] = utcnow().isoformat()
        if drawdown is not None:
            data["drawdown"] = round(drawdown, 6)
        if nav is not None:
            data["nav_at_trip"] = round(nav, 2)
        try:
            self._kill_switch_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._kill_switch_state_path.write_text(json.dumps(data, indent=2))
        except Exception:
            log.warning(
                "Failed to persist kill-switch state to %s",
                self._kill_switch_state_path, exc_info=True,
            )

    def _load_persisted_state(self) -> None:
        """Restore portfolio history + attribution state from a previous run.

        No-op when persistence is disabled (default) or nothing was
        persisted yet. Best-effort: a corrupt/unreadable blob logs a warning
        and leaves the fresh in-memory defaults rather than failing engine
        construction — unlike the kill switch, this state is a continuity
        nicety (the broker remains authoritative for cash/holdings), not a
        safety control.
        """
        if self._state_store is None:
            return
        try:
            history = self._state_store.load_portfolio_history()
            if history:
                self._portfolio.restore_history(history)
                log.info(
                    "Restored %d persisted portfolio snapshots from %s",
                    len(history), self._state_store.db_path,
                )
        except Exception:
            log.warning("Failed to restore persisted portfolio history", exc_info=True)
        try:
            attribution_state = self._state_store.load_attribution_state()
            if attribution_state:
                self._attribution.restore_state(attribution_state)
                log.info(
                    "Restored persisted attribution state for strategies: %s",
                    list(attribution_state.get("strategy_returns", {}).keys()),
                )
        except Exception:
            log.warning("Failed to restore persisted attribution state", exc_info=True)
        try:
            daily_limits = self._state_store.load_daily_limits()
            if daily_limits:
                self._daily_date = daily_limits.get("daily_date")
                self._daily_trade_count = int(daily_limits.get("daily_trade_count", 0))
                self._daily_turnover_value = float(daily_limits.get("daily_turnover_value", 0.0))
                log.info(
                    "Restored daily trade/turnover counters for %s: "
                    "%d trades, $%.2f turnover",
                    self._daily_date, self._daily_trade_count, self._daily_turnover_value,
                )
        except Exception:
            log.warning("Failed to restore persisted daily trade/turnover counters", exc_info=True)
        trader = getattr(self._orchestrator, "trader", None)
        if hasattr(trader, "load_state"):
            try:
                trader_state = self._state_store.load_trader_state()
                if trader_state:
                    trader.load_state(trader_state)
                    log.info(
                        "Restored conviction-EMA state for %d symbols",
                        len(trader_state.get("conviction_ema") or {}),
                    )
            except Exception:
                log.warning("Failed to restore persisted trader state", exc_info=True)

    def _persist_live_state(self) -> None:
        """Save portfolio history + attribution state after a cycle.

        Called from :meth:`_persist_cycle_result` (i.e. after every cycle
        attempt, including skipped/errored ones) so a crash mid-cycle loses
        at most the in-flight cycle, not the accumulated history. Also
        mirrors the current kill-switch state into the same database — the
        JSON file at ``_kill_switch_state_path`` remains the mechanism this
        engine actually reads on startup (see ``_load_kill_switch_state``);
        this is purely an additional durable copy for operators/tools that
        query one database instead of scattered files.
        """
        if self._state_store is None:
            return
        try:
            self._state_store.save_portfolio_history(self._portfolio.history)
        except Exception:
            log.warning("Failed to persist portfolio history", exc_info=True)
        try:
            self._state_store.save_attribution_state(self._attribution.export_state())
        except Exception:
            log.warning("Failed to persist attribution state", exc_info=True)
        try:
            self._state_store.save_kill_switch({
                "halted": self._halted,
                "peak_equity": round(self._peak_equity, 2),
            })
        except Exception:
            log.debug("Failed to mirror kill-switch state to state store", exc_info=True)
        try:
            self._state_store.save_daily_limits({
                "daily_date": self._daily_date,
                "daily_trade_count": self._daily_trade_count,
                "daily_turnover_value": round(self._daily_turnover_value, 2),
            })
        except Exception:
            log.warning("Failed to persist daily trade/turnover counters", exc_info=True)
        trader = getattr(self._orchestrator, "trader", None)
        if hasattr(trader, "get_state"):
            try:
                self._state_store.save_trader_state(trader.get_state())
            except Exception:
                log.warning("Failed to persist trader state", exc_info=True)

    def reset_kill_switch(self) -> dict[str, Any]:
        """Clear the drawdown kill switch and re-arm trading.

        This is a deliberate operator action (e.g. via
        ``POST /api/live/kill-switch/reset``), not something the engine ever
        does on its own — the whole point of the switch is that trading stays
        halted until a human confirms it's safe to resume. Resets the
        high-water mark to current NAV so the drawdown calculation restarts
        from here rather than immediately re-tripping against the pre-halt
        peak.
        """
        was_halted = self._halted
        self._halted = False
        self._peak_equity = self._portfolio.nav
        self._persist_kill_switch_state(halted=False)
        alert = self._emit_alert(
            "kill_switch_reset",
            "warning",
            "Kill switch manually reset by operator; trading re-armed.",
            was_halted=was_halted,
            new_peak_equity=round(self._peak_equity, 2),
        )
        log.warning(
            "Kill switch reset (was_halted=%s); new peak equity=$%.2f",
            was_halted, self._peak_equity,
        )
        return {"halted": self._halted, "peak_equity": self._peak_equity, "alert": alert}

    def run_on_cycle_worker(self, fn, /, *args, timeout: float = 120, **kwargs):
        """Run ``fn`` on the dedicated live-cycle / IBKR I/O thread.

        ib_async binds to the thread that called ``connect()``; cycles and
        order submission must share that thread or qualifyContracts/placeOrder
        can hang indefinitely with no error.
        """
        future = self._cycle_executor.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout)

    def _connect_broker_on_worker(self) -> None:
        self._broker.connect()
        account = self._broker.get_account()
        log.info("Live engine started – account equity: $%.2f", account.get("equity", 0))
        self._portfolio.cash = account.get("cash", self._portfolio.cash)

    def _disconnect_broker_on_worker(self) -> None:
        self._broker.disconnect()

    def start(self) -> None:
        """Connect to the broker and validate the account."""
        try:
            self.run_on_cycle_worker(self._connect_broker_on_worker, timeout=120)
        except FuturesTimeoutError:
            raise BrokerError(
                "Broker connect timed out on cycle worker thread"
            ) from None
        self._running = True
        self._started_at = datetime.now()

    def stop(self) -> None:
        """Disconnect from the broker."""
        self._shutting_down = True
        self._running = False
        self._started_at = None
        self._active_cycle_token = None
        if self._watchdog_timer is not None:
            self._watchdog_timer.cancel()
            self._watchdog_timer = None
        try:
            self.run_on_cycle_worker(self._disconnect_broker_on_worker, timeout=5)
        except FuturesTimeoutError:
            log.warning(
                "Broker disconnect timed out — cycle worker may be stuck; "
                "abandoning IB session (service restart may be required)",
            )
        except Exception:
            log.warning("Error disconnecting broker", exc_info=True)
        # Do not block process exit on a hung cycle worker — the thread may
        # still be running but is daemon and will not pin the process.
        self._cycle_executor.shutdown(wait=False, cancel_futures=True)
        if self._state_store is not None:
            try:
                self._persist_live_state()
                self._state_store.close()
            except Exception:
                log.warning("Error closing live state store", exc_info=True)
        log.info("Live engine stopped after %d cycles", self._cycle_count)

    def run_cycle(self, force: bool = False) -> CycleResult:
        """Execute one full cycle of the agent pipeline.

        Cycles are serialised: if one is already running (e.g. a scheduled
        tick while a manual trigger is in flight) this call returns
        immediately with ``skipped=True`` rather than racing and
        double-submitting orders.

        When the market is closed, the cycle is skipped before any data
        fetch or agent pipeline work — there's nothing useful a cycle can
        submit against a closed market, and running it anyway wastes a full
        LLM-enhanced pass against IBKR's misleading off-hours quotes. Pass
        ``force=True`` (used by a manual ``/live/trigger?force=true``) to
        run anyway, e.g. for deliberate off-hours testing.

        Returns a :class:`CycleResult` summarizing what happened.
        """
        if self._shutting_down:
            log.info("run_cycle skipped: engine is shutting down")
            return CycleResult(
                cycle_id=self._cycle_count,
                timestamp=utcnow(),
                skipped=True,
                error="skipped: engine shutting down",
            )
        if not self._cycle_lock.acquire(blocking=False):
            log.warning("run_cycle skipped: a cycle is already in progress")
            return CycleResult(
                cycle_id=self._cycle_count,
                timestamp=utcnow(),
                skipped=True,
                error="cycle already in progress",
            )
        try:
            self._cycle_count += 1
            now = utcnow()
            self._current_cycle_started_at = now
            result = CycleResult(cycle_id=self._cycle_count, timestamp=now)

            self._watchdog_timer = threading.Timer(
                self._cycle_watchdog_seconds, self._on_cycle_watchdog_timeout, args=(self._cycle_count,)
            )
            self._watchdog_timer.daemon = True
            self._watchdog_timer.start()

            if self._respect_market_hours and not force:
                try:
                    market_open = self._broker.is_market_open()
                except Exception:
                    # Fail open: a broken market-hours check must never
                    # silently prevent every future cycle from running.
                    log.warning(
                        "Could not determine market hours; proceeding with cycle",
                        exc_info=True,
                    )
                    market_open = True
                if not market_open:
                    log.info("Cycle %d skipped: market is closed", self._cycle_count)
                    result.skipped = True
                    result.error = "skipped: market closed"
                    self._cycle_history.append(result)
                    self._persist_cycle_result(result)
                    return result

            cycle_token = object()
            self._active_cycle_token = cycle_token
            try:
                if self._shutting_down:
                    result.skipped = True
                    result.error = "skipped: engine shutting down"
                    return result
                future = self._cycle_executor.submit(
                    self._run_cycle_work, cycle_token, now, result,
                )
                future.result(timeout=self._cycle_hard_timeout_seconds)
            except FuturesTimeoutError:
                self._active_cycle_token = None
                msg = (
                    f"Cycle {self._cycle_count} hard-timed out after "
                    f"{self._cycle_hard_timeout_seconds:.0f}s — releasing the cycle "
                    "lock so future cycles can run. A stale worker thread may "
                    "still be running in the background."
                )
                log.error(msg)
                result.error = (
                    f"cycle hard timeout after {self._cycle_hard_timeout_seconds:.0f}s"
                )
                result.alerts.append(self._emit_alert(
                    "cycle_hard_timeout", "critical", msg,
                ))
            except Exception as exc:
                result.error = str(exc)
                log.error(
                    "Cycle %d failed: %s", self._cycle_count, exc, exc_info=True,
                )

            self._cycle_history.append(result)
            self._persist_cycle_result(result)
            return result
        finally:
            if self._watchdog_timer is not None:
                self._watchdog_timer.cancel()
                self._watchdog_timer = None
            self._current_cycle_started_at = None
            self._active_cycle_token = None
            self._cycle_lock.release()

    def _cycle_token_active(self, token: object) -> bool:
        return self._active_cycle_token is token

    def _run_cycle_work(
        self,
        token: object,
        now: datetime,
        result: CycleResult,
    ) -> None:
        """Pipeline body for one cycle (runs on the cycle worker thread)."""
        if not self._cycle_token_active(token):
            return

        try:
            if not self._ensure_broker_healthy(result):
                result.error = "broker unavailable: proactive health check failed"
                return

            pit_view = self._data_feed.refresh(asof=now)

            prices = self._resolve_cycle_prices(pit_view)

            discrepancies = sync_portfolio_from_broker(
                self._broker, self._portfolio, prices
            )
            result.discrepancies = discrepancies

            if self._consecutive_broker_failures:
                # This is the first broker call in the cycle (reconciliation) —
                # reaching here without raising proves connectivity is back,
                # whether that's thanks to _try_broker_reconnect() below or IB
                # Gateway/the network recovering on its own between cycles.
                log.warning(
                    "Broker connectivity restored after %d failed cycle(s)",
                    self._consecutive_broker_failures,
                )
                result.alerts.append(self._emit_alert(
                    "broker_reconnected", "warning",
                    f"Broker connectivity restored after "
                    f"{self._consecutive_broker_failures} failed cycle(s).",
                ))
                self._consecutive_broker_failures = 0

            if any(d.get("type") == "open_orders_unavailable" for d in discrepancies):
                result.alerts.append(self._emit_alert(
                    "reconciliation_degraded", "warning",
                    "Open orders unavailable; reconciliation may be incomplete.",
                ))

            self._check_drawdown(result)
            if self._halted:
                result.halted = True
                result.error = "halted: drawdown kill switch tripped"
                return

            self._maybe_reflect(now)

            if not self._cycle_token_active(token):
                log.warning(
                    "Cycle %d abandoned after reflection — skipping pipeline",
                    self._cycle_count,
                )
                return

            context = {
                "pit_view": pit_view,
                "portfolio": self._portfolio,
                "prices": prices,
                "memory": self._memory,
                "attribution": self._attribution,
            }

            # Mark-to-market attribution daily regardless of signal-combination
            # method — this used to run only under `optimal` (to feed its
            # inverse-covariance weighting), which meant the `confidence`
            # method (the live default) never populated per-strategy holdings,
            # so the execution agent had no fallback for attributing exit
            # trades and they collapsed into "composite". Tracking is cheap
            # relative to a trading cycle and keeps every fill traceable.
            try:
                self._attribution.update_daily(now, prices, self._portfolio.nav)
                # Fed unconditionally (not just under `optimal`): the generic
                # per-strategy circuit breaker (agents.research._circuit_breaker)
                # needs trailing return history regardless of combination
                # method, same rationale as the attribution tracking above.
                strategy_returns = self._attribution.get_all_strategy_returns()
                if strategy_returns:
                    context["strategy_returns"] = strategy_returns
            except Exception:
                # Silent attribution corruption directly undermines
                # per-strategy circuit breakers fed by
                # get_all_strategy_returns() — must be visible, not debug.
                log.warning(
                    "attribution daily update failed", exc_info=True,
                )

            orders, blackboard = self._orchestrator.step(context)
            result.orders_generated = len(orders)

            proposal = getattr(blackboard, "proposal", None)
            if proposal is not None:
                date_str = trading_day_key(now, self._trading_day_timezone)
                regime = ""
                if hasattr(blackboard, "risk_decision") and blackboard.risk_decision:
                    regime = "; ".join(getattr(blackboard.risk_decision, "actions", []))
                self._memory.store_decision(
                    date=date_str,
                    proposal_weights=dict(proposal.targets),
                    notes=f"cycle={self._cycle_count}; {regime}".strip("; "),
                    nav_at_decision=self._portfolio.nav,
                    per_strategy=dict(proposal.per_strategy),
                )

            if not orders:
                log.info("Cycle %d: no orders generated", self._cycle_count)
                return

            if not self._cycle_token_active(token):
                log.warning(
                    "Cycle %d abandoned before order routing — dropping %d orders",
                    self._cycle_count, len(orders),
                )
                return

            orders = self._apply_news_guard(orders, now, result)
            if not orders:
                log.info(
                    "Cycle %d: all orders held by news-guard", self._cycle_count
                )
                return

            alerts_before = len(self._alerts)
            force_manual, orders = self._check_daily_limits(now, orders, prices)
            if len(self._alerts) > alerts_before:
                result.alerts.append(self._alerts[-1])
            if force_manual:
                auto_orders, manual_orders = [], orders
            else:
                auto_orders, manual_orders = self._split_by_approval(orders)

            if auto_orders:
                statuses, failed = self._execute_orders(
                    auto_orders, cycle_id=self._cycle_count
                )
                result.orders_submitted = len(statuses)
                result.order_statuses = [
                    self._status_to_dict(s, strategy) for s, strategy in statuses
                ]
                result.failed_orders = failed
                result.orders_failed = len(failed)

                try:
                    self._attribution.record_trades(
                        self._orders_to_fills(auto_orders), prices,
                    )
                except Exception:
                    # Same rationale as the daily-update case above: this
                    # quietly recording nothing undermines per-strategy
                    # attribution/circuit breakers and must be visible.
                    log.warning(
                        "attribution trade recording failed", exc_info=True,
                    )

            if manual_orders:
                for strategy, group in self._group_by_strategy(manual_orders).items():
                    aid = self._approval_queue.add(
                        orders=group,
                        blackboard=blackboard,
                        strategy=strategy,
                    )
                    result.orders_queued += len(group)
                    result.approval_ids.append(aid)

            log.info(
                "Cycle %d: %d generated, %d submitted, %d queued, %d failed",
                self._cycle_count,
                result.orders_generated,
                result.orders_submitted,
                result.orders_queued,
                result.orders_failed,
            )

            # A cycle where every auto order failed to submit is a silent
            # no-trade day: the broker can report "reconnected" (a lightweight
            # check) while individual calls like qualifyContracts still time
            # out, so this doesn't go through the BrokerError path below and
            # would otherwise surface as nothing louder than the routine
            # broker_unavailable/broker_reconnected alerts.
            if auto_orders and result.orders_submitted == 0 and result.orders_failed == len(auto_orders):
                result.alerts.append(self._emit_alert(
                    "cycle_all_orders_failed",
                    "critical",
                    f"All {result.orders_failed} order(s) failed to submit this "
                    "cycle (0 submitted) — broker may report connected while "
                    "individual requests time out; check IB Gateway.",
                ))

        except Exception as exc:
            result.error = str(exc)
            log.error("Cycle %d failed: %s", self._cycle_count, exc, exc_info=True)
            if isinstance(exc, BrokerError):
                self._consecutive_broker_failures += 1
                reconnected = self._try_broker_reconnect()
                self._emit_broker_failure_alert(
                    result, f"Broker call failed during cycle: {exc}",
                    reconnected=reconnected,
                )

    def _emit_broker_failure_alert(
        self, result: CycleResult, detail: str, *, reconnected: bool
    ) -> None:
        """Emit the broker-unavailable / sustained-disconnect alert.

        Shared by the reactive path (a cycle's first broker call raised
        ``BrokerError``) and the proactive path (``_ensure_broker_healthy``'s
        health check failed and reconnect also failed) so the two don't
        duplicate the same severity/escalation logic. Assumes the caller has
        already incremented ``_consecutive_broker_failures``.

        Severity mirrors how urgent this actually is: a self-healed or
        not-yet-sustained failure is the exact "single dropped socket/blip —
        including IB Gateway's routine daily restart — shouldn't need a
        human" case ``_try_broker_reconnect``'s own docstring describes, so
        it stays "warning". Only the genuinely sustained case (crossed
        ``broker_disconnect_alert_threshold`` with reconnect still failing)
        is "critical".
        """
        severity = "warning"
        message = detail
        if reconnected:
            message += (
                " — reconnected successfully; the next cycle should "
                "resume normally."
            )
            alert_kind = "broker_unavailable"
        elif self._consecutive_broker_failures >= self._broker_disconnect_alert_threshold:
            alert_kind = "broker_disconnected_sustained"
            severity = "critical"
            message = (
                f"Broker has failed {self._consecutive_broker_failures} "
                f"consecutive cycle(s) and automatic reconnect did not "
                f"succeed: {detail}. Likely needs manual intervention "
                "(check IB Gateway is running/logged in, or restart "
                "ai-trading.service) — see docs/PROJECT_CONTEXT.md "
                "'Broker & host failover'."
            )
        else:
            alert_kind = "broker_unavailable"
        result.alerts.append(self._emit_alert(
            alert_kind, severity, message,
            consecutive_failures=self._consecutive_broker_failures,
            reconnected=reconnected,
        ))

    def _ensure_broker_healthy(self, result: CycleResult) -> bool:
        """Proactively verify (and if needed repair) the broker connection
        before the cycle's first data/broker call.

        Confirmed live: IB Gateway restarts nightly (systemd-scheduled), but
        the broker connection here is never otherwise refreshed — the first
        real request of the next day's cycle discovers the stale socket only
        after burning a 20s timeout per symbol across the whole universe,
        losing that entire day's orders. This runs on the cycle worker
        thread ib_async is bound to (called from the top of
        ``_run_cycle_work``), so broker calls are made directly.

        Returns ``True`` if the cycle may proceed — connection healthy, or a
        single reconnect succeeded. Returns ``False`` if the connection is
        down and reconnect failed; the caller aborts the cycle early rather
        than burning the per-symbol timeout storm against a dead socket.
        """
        if self._broker.health_check():
            return True

        log.warning(
            "Proactive broker health check failed before cycle %d — "
            "connection likely stale (e.g. IB Gateway restarted overnight); "
            "reconnecting",
            self._cycle_count,
        )
        reconnected = self._try_broker_reconnect()
        if reconnected:
            # Do NOT reset _consecutive_broker_failures or emit
            # broker_reconnected here: the post-reconciliation recovery block
            # further down in _run_cycle_work already does both, but only
            # when prior cycles had actually failed (counter > 0) — reusing
            # that bookkeeping avoids a duplicate alert on the routine daily
            # self-heal (counter == 0), which is this bug's normal case and
            # should stay quiet-but-logged, not paged.
            log.warning(
                "Proactive reconnect succeeded before cycle %d — proceeding "
                "on a fresh connection",
                self._cycle_count,
            )
            return True

        self._consecutive_broker_failures += 1
        self._emit_broker_failure_alert(
            result,
            "Broker unavailable: proactive health check and reconnect both failed",
            reconnected=False,
        )
        return False

    def _try_broker_reconnect(self) -> bool:
        """Attempt one inline reconnect after a cycle's broker call failed.

        Runs on the cycle worker thread (the same thread ``_run_cycle_work``
        itself executes on, which is also the thread ib_async's ``IB``
        instance is bound to — no ``run_on_cycle_worker`` hop needed). A
        single dropped socket then self-heals within the same cycle's error
        handling instead of requiring a full service restart; a still-down
        IB Gateway just fails again here and the caller's consecutive-failure
        counter keeps climbing toward the sustained-disconnect alert.
        """
        try:
            self._broker.reconnect()
        except Exception as exc:
            log.warning(
                "Broker reconnect attempt failed (consecutive failures=%d): %s",
                self._consecutive_broker_failures, exc, exc_info=True,
            )
            return False
        log.info(
            "Broker reconnected after %d consecutive failure(s)",
            self._consecutive_broker_failures,
        )
        return True

    def _on_cycle_watchdog_timeout(self, cycle_id: int) -> None:
        """Fires if a cycle is still running well past any normal duration.

        Purely observational — does not touch the lock or try to abandon
        the stuck thread (which could still resume later and race a new
        cycle over shared state). The goal is simply to never again let a
        hang go unnoticed for hours with zero error and zero alert.
        """
        self._emit_alert(
            "cycle_watchdog_timeout", "critical",
            f"Cycle {cycle_id} has been running for over "
            f"{self._cycle_watchdog_seconds:.0f}s without completing — likely a hung "
            "network call (e.g. a stale broker connection after an IB Gateway restart). "
            "No new cycles can start until this one finishes or the service is restarted.",
        )

    def _resolve_cycle_prices(self, pit_view: Any) -> dict[str, float]:
        """Prices for sizing and reconciliation within a cycle.

        IBKR's ``reqTickers`` must run on the same thread that called
        ``connect()``; cycles execute on a worker thread, so we derive
        marks from the freshly loaded PIT price panel instead (same
        completed bars the strategies already see).
        """
        universe = list(self._data_feed._universe)
        from firm.brokers.ibkr import IBKRBroker

        if isinstance(self._broker, IBKRBroker):
            prices: dict[str, float] = {}
            try:
                price_df = pit_view.prices(universe, lookback_days=5)
                if not price_df.empty and "symbol" in price_df.columns:
                    for sym in universe:
                        sym_rows = price_df[price_df["symbol"] == sym]
                        if sym_rows.empty:
                            continue
                        row = sym_rows.iloc[-1]
                        raw = row.get("close")
                        if raw is None or (isinstance(raw, float) and raw != raw):
                            raw = row.get("adj_close")
                        if raw is not None and float(raw) > 0:
                            prices[sym] = float(raw)
            except Exception:
                log.warning("Could not derive IBKR cycle prices from PIT view", exc_info=True)
            if prices:
                missing = [s for s in universe if s not in prices]
                if missing:
                    log.warning(
                        "PIT cycle prices missing %d symbol(s): %s",
                        len(missing), missing[:10],
                    )
                log.debug(
                    "Cycle prices from PIT (IBKR thread-safe): %d/%d symbols",
                    len(prices), len(universe),
                )
                return prices
            raise BrokerError(
                "No usable PIT prices for IBKR cycle — refusing reqTickers off worker thread"
            )

        return self._broker.get_current_prices(universe)

    def _maybe_reflect(self, now: datetime) -> None:
        """Trigger deferred LLM reflection on any decisions whose P&L is now known.

        Called at the start of each cycle — before this cycle's own decision
        is stored — so every entry ``find_all_pending()`` returns is
        genuinely from a previous cycle. Pending decisions (and the NAV to
        diff against) are read back from the persisted memory log rather
        than an in-memory pointer, so a process restart between the decision
        and this call doesn't silently drop the reflection.
        """
        pending = self._memory.find_all_pending()
        if not pending:
            return
        llm = self._get_llm_service()
        if llm is None:
            log.warning(
                "Skipping reflection on %d pending decision(s) (dates: %s) — "
                "no LLM service available",
                len(pending), [e["date"] for e in pending],
            )
            return

        current_nav = self._portfolio.nav
        for entry in pending:
            prev_nav = entry.get("nav_at_decision")
            if not prev_nav or prev_nav <= 0:
                log.debug(
                    "Skipping reflection for %s — no nav_at_decision recorded "
                    "(pre-dates this fix)", entry["date"],
                )
                continue
            raw_return = (current_nav / prev_nav) - 1.0
            # Use a flat 0.0 benchmark when SPY price is unavailable — the
            # reflection is still useful even without alpha decomposition.
            try:
                self._memory.reflect(
                    date=entry["date"],
                    raw_return=raw_return,
                    benchmark_return=0.0,
                    llm_service=llm,
                )
            except Exception:
                log.warning("Memory reflection failed for %s", entry["date"], exc_info=True)

    def _get_llm_service(self) -> Any:
        """Lazy-initialise the LLM service for reflection calls.

        Prefers an explicit ``config["llm_config"]``; falls back to
        ``config/llm.yaml``'s ``provider`` section (default model, fallback
        models, load-balancing) so a live engine started via the API — which
        does not currently thread ``llm_config`` through — still picks up
        the configured fallback/load-balance behaviour instead of silently
        reverting to the hardcoded Groq-only default.
        """
        if self._llm_service is not None:
            return self._llm_service
        try:
            from firm.llm.provider import LLMService
            llm_config = self._config.get("llm_config")
            if not llm_config:
                from firm.llm.config import provider_config
                llm_config = provider_config()
            self._llm_service = LLMService(llm_config)
        except Exception:
            log.warning("LLM service unavailable — memory reflection disabled", exc_info=True)
        return self._llm_service

    @staticmethod
    def _group_by_strategy(
        orders: list[dict[str, Any]]
    ) -> "OrderedDict[str, list[dict[str, Any]]]":
        """Group orders by their originating strategy (preserving order).

        Each group becomes a separate approval entry so an operator can
        approve/reject per strategy instead of one mixed basket.
        """
        grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for o in orders:
            grouped.setdefault(o.get("strategy", ""), []).append(o)
        return grouped

    def _split_by_approval(
        self, orders: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split orders into auto-execute and needs-approval buckets."""
        if self._approval_mode == "full_auto":
            return orders, []
        if self._approval_mode == "semi_auto":
            auto: list[dict[str, Any]] = []
            manual: list[dict[str, Any]] = []
            for o in orders:
                strategy = o.get("strategy", "")
                if strategy in self._auto_approve:
                    auto.append(o)
                else:
                    manual.append(o)
            return auto, manual
        # Default: everything needs approval
        return [], orders

    def _execute_orders(
        self, orders: list[dict[str, Any]], cycle_id: int = 0
    ) -> tuple[list[tuple[OrderStatus, str]], list[dict[str, Any]]]:
        """Submit orders to the broker.

        Returns ``(statuses, failed)`` where *failed* lists the orders whose
        submission raised, so a partially-executed basket is visible to the
        caller/operator rather than silently dropped.  Each order carries a
        deterministic ``client_order_id`` so the broker can deduplicate a
        re-submission of the same cycle's order.

        ``statuses`` is ``(OrderStatus, strategy)`` pairs, not bare
        ``OrderStatus`` — the originating strategy is only known here, at
        submission time, and must be threaded through explicitly for the
        persisted trade history to remain traceable back to which strategy
        caused which order (see ``_status_to_dict``).
        """
        from firm.live.execution_safety import Order, RiskProfile, guard_live_submission, guard_order

        # Final, independently-auditable hard cap right before the broker
        # call — a backstop against a bug anywhere upstream (RiskAgent,
        # ExecutionAgent sizing, a stale NAV read, ...) producing an
        # oversized or off-universe order, not a replacement for those
        # portfolio-level checks. require_stop=False: this engine rebalances
        # to target weights, it has no per-order protective-stop concept.
        # check_risk_limits compares against the *trade's* notional, not the
        # resulting position's — a full flip from -max_position_pct to
        # +max_position_pct in one cycle is a legitimate rebalance and trades
        # ~2x the single-position cap, so the cap here is doubled to avoid
        # blocking that case while still catching a genuinely runaway order.
        risk_profile = RiskProfile(
            account_equity=self._portfolio.nav,
            max_position_notional=2.0 * self._max_position_pct * self._portfolio.nav,
            symbol_allowlist=self._data_feed._universe,
            require_stop=False,
        )

        statuses: list[tuple[OrderStatus, str]] = []
        failed: list[dict[str, Any]] = []
        # Circuit breaker: confirmed live 2026-08-07 that once the broker
        # connection is in a bad state (IBKR Gateway stalled mid-cycle after
        # a reconnect), every subsequent submit_order call times out
        # independently — the loop burned ~6 minutes failing all 19
        # remaining orders one at a time at 20s each instead of recognizing
        # the pattern. After a few consecutive submission failures, treat it
        # as systemic for this cycle and stop trying the broker; the next
        # cycle's own reconnect-before-generating-orders logic is what
        # actually recovers it, not more retries here.
        _MAX_CONSECUTIVE_BROKER_FAILURES = 3
        consecutive_broker_failures = 0
        broker_circuit_open = False
        for o in orders:
            safety_order = Order(
                symbol=o["symbol"], side=o["side"],
                qty=float(o.get("quantity", abs(o.get("shares", 0)))),
                order_type=o.get("order_type", "market"),
                price=float(o.get("price", 0.0)),
            )
            # live=False deliberately: guard_order's job here is purely the
            # RiskProfile hard-cap check (symbol allowlist + max notional).
            # Live-vs-paper routing is guard_live_submission's job below —
            # doing it in both places would double-audit the same decision.
            risk_gate = guard_order(safety_order, risk_profile, live=False)
            if risk_gate["routed"] == "blocked":
                failed.append({**o, "error": risk_gate["reason"]})
                self._emit_alert(
                    "order_risk_cap_blocked", "critical", risk_gate["reason"],
                    symbol=o.get("symbol"), audit_id=risk_gate["audit_id"],
                )
                continue

            # Hard env lock: a live broker must have FIRM_ALLOW_TRADING=1 set in
            # the service environment or the order is blocked (never silently
            # downgraded) and audited. Paper brokers pass straight through.
            gate = guard_live_submission(
                self._broker_type, o, cycle_id=cycle_id
            )
            if not gate["allowed"]:
                failed.append({**o, "error": gate["reason"]})
                self._emit_alert(
                    "live_trading_locked", "critical", gate["reason"],
                    symbol=o.get("symbol"), audit_id=gate["audit_id"],
                )
                continue

            raw_qty = float(o.get("quantity", abs(o.get("shares", 0))))
            share_qty = int(round(abs(raw_qty)))
            if share_qty <= 0:
                log.debug(
                    "Skipping dust order %s %s (raw qty %.4f rounds to 0 shares)",
                    o.get("side"), o.get("symbol"), raw_qty,
                )
                continue

            req = OrderRequest(
                symbol=o["symbol"],
                side=o["side"],
                quantity=share_qty,
                order_type=o.get("order_type", "market"),
                limit_price=o.get("limit_price"),
                strategy=o.get("strategy", "composite"),
                client_order_id=f"c{cycle_id}-{o['symbol']}-{o['side']}",
            )
            if broker_circuit_open:
                failed.append({
                    **o,
                    "error": (
                        f"Skipped: broker submission circuit open after "
                        f"{consecutive_broker_failures} consecutive failures this cycle"
                    ),
                })
                continue
            try:
                status = self._broker.submit_order(req)
                consecutive_broker_failures = 0
                # Paired with req.strategy here (not read back off `status`
                # later): OrderStatus is a broker-level type with no notion
                # of which of *our* strategies caused it — this is the only
                # point where that link still exists, and losing it here
                # means the persisted trade history can never answer "which
                # strategy placed this order" (needed for reflection/
                # lessons-learned, not just display).
                statuses.append((status, req.strategy))
                log.info("Submitted: %s %s %.2f %s → %s",
                         req.side, req.symbol, req.quantity, req.order_type, status.status)
            except BrokerError as exc:
                log.error("Failed to submit order for %s", req.symbol, exc_info=True)
                failed.append({**o, "error": str(exc)})
                consecutive_broker_failures += 1
                if consecutive_broker_failures >= _MAX_CONSECUTIVE_BROKER_FAILURES:
                    broker_circuit_open = True
                    self._emit_alert(
                        "broker_submission_circuit_open", "critical",
                        f"Aborting remaining order submissions this cycle after "
                        f"{consecutive_broker_failures} consecutive broker failures "
                        f"(last: {exc})",
                    )
        return statuses, failed

    @staticmethod
    def _orders_to_fills(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize order dicts into the signed-``shares`` fill format
        :meth:`PerformanceAttribution.record_trades` expects.

        ``firm.agents.execution.ExecutionAgent`` already emits both a signed
        ``shares`` field and an unsigned ``quantity`` + ``side`` pair on
        every real order, so this is a no-op in the normal pipeline.  It
        exists as a defensive normalization for any other order source
        (a custom/mocked orchestrator, a future execution path) that only
        supplies ``side`` + ``quantity`` — previously that shape hit a
        ``KeyError`` inside ``record_trades`` on the missing ``shares`` key,
        silently swallowed by a broad ``except Exception`` and logged at
        debug, so per-strategy attribution quietly recorded nothing for that
        order.
        """
        fills = []
        for o in orders:
            qty = float(o.get("quantity", abs(o.get("shares", 0))))
            sign = -1.0 if o.get("side") == "sell" else 1.0
            fills.append({
                "symbol": o["symbol"],
                "shares": sign * qty,
                "price": float(o.get("price", 0.0)),
                "strategy": o.get("strategy", "composite"),
            })
        return fills

    @staticmethod
    def _status_to_dict(s: OrderStatus, strategy: str = "") -> dict[str, Any]:
        return {
            "order_id": s.order_id,
            "symbol": s.symbol,
            "side": s.side,
            "quantity": s.quantity,
            "filled_quantity": s.filled_quantity,
            "avg_fill_price": s.avg_fill_price,
            "status": s.status,
            "timestamp": s.timestamp.isoformat() if s.timestamp else None,
            "strategy": strategy,
        }
