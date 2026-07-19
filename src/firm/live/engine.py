"""LiveTradingEngine – the core live/paper trading loop.

Replaces BacktestEngine for real-time operation.  Each cycle:
  1. Fetch live prices from data providers  →  PIT store
  2. Sync portfolio state from broker positions
  3. Run Orchestrator.step()
  4. Route resulting orders (auto-submit or queue for approval)
  5. Log everything
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from firm.brokers.base import Broker, BrokerError, OrderRequest, OrderStatus
from firm.live.approval import ApprovalQueue
from firm.live.data_feed import LiveDataFeed
from firm.live.portfolio_sync import sync_portfolio_from_broker
from firm.portfolio.state import PortfolioState
from firm.runtime import build_orchestrator

log = logging.getLogger(__name__)


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
    ) -> None:
        self._config = config
        self._broker = broker
        self._data_feed = data_feed
        self._approval_queue = approval_queue
        self._approval_mode = approval_mode
        self._auto_approve: set[str] = set(auto_approve_strategies or [])

        self._orchestrator = build_orchestrator(config)
        self._enabled_strategies: list[str] = list(config.get("strategies") or self._all_strategy_names())
        initial_capital = config.get("initial_capital", 100_000)
        self._portfolio = PortfolioState(initial_capital=initial_capital)
        self._cycle_count = 0
        self._cycle_history: list[CycleResult] = []
        self._running = False

        # Daily risk limits — reset each calendar day. Unlike the drawdown
        # kill switch (which halts permanently), breaching these forces the
        # cycle's orders to manual approval rather than blocking them
        # outright, so an operator stays in control instead of trades
        # silently vanishing.
        self._max_daily_trades = int(config.get("max_daily_trades", 50))
        self._max_daily_turnover = float(config.get("max_daily_turnover", 0.5))
        self._daily_date: str | None = None
        self._daily_trade_count = 0
        self._daily_turnover_value = 0.0

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
        # Serialises cycles so a manual/API trigger cannot run concurrently
        # with a scheduled one (which, combined with no broker idempotency,
        # would double-submit real orders).
        self._cycle_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def portfolio(self) -> PortfolioState:
        return self._portfolio

    @property
    def cycle_history(self) -> list[CycleResult]:
        return list(self._cycle_history)

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

    def _check_daily_limits(
        self, now: datetime, orders: list[dict[str, Any]], prices: dict[str, float]
    ) -> bool:
        """Track today's trade count/turnover; return True if this cycle's
        orders would breach the daily cap and should be forced to manual
        approval instead of blocked outright (keeps an operator in the loop
        rather than silently dropping signals or silently over-trading).
        """
        date_str = now.strftime("%Y-%m-%d")
        if self._daily_date != date_str:
            self._daily_date = date_str
            self._daily_trade_count = 0
            self._daily_turnover_value = 0.0

        turnover_value = sum(
            abs(o.get("quantity", abs(o.get("shares", 0))) * prices.get(o.get("symbol", ""), 0.0))
            for o in orders
        )
        nav = self._portfolio.nav
        projected_trades = self._daily_trade_count + len(orders)
        projected_turnover_frac = (
            (self._daily_turnover_value + turnover_value) / nav if nav > 0 else 0.0
        )

        breached = (
            projected_trades > self._max_daily_trades
            or projected_turnover_frac > self._max_daily_turnover
        )
        if breached:
            self._emit_alert(
                "daily_limit_breach", "warning",
                f"Daily limit would be breached (trades {projected_trades}/"
                f"{self._max_daily_trades}, turnover {projected_turnover_frac:.1%}/"
                f"{self._max_daily_turnover:.1%}); routing orders to manual approval.",
            )
        else:
            self._daily_trade_count = projected_trades
            self._daily_turnover_value += turnover_value
        return breached

    def _emit_alert(
        self, kind: str, severity: str, message: str, **context: Any
    ) -> dict[str, Any]:
        """Record an operational alert, log it, and forward to the callback.

        A failing callback must never break the trading loop, so it is
        guarded. Alerts are also attached to the current cycle result.
        """
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
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

    def start(self) -> None:
        """Connect to the broker and validate the account."""
        self._broker.connect()
        account = self._broker.get_account()
        log.info("Live engine started – account equity: $%.2f", account.get("equity", 0))
        self._portfolio.cash = account.get("cash", self._portfolio.cash)
        self._running = True

    def stop(self) -> None:
        """Disconnect from the broker."""
        self._running = False
        try:
            self._broker.disconnect()
        except Exception:
            log.warning("Error disconnecting broker", exc_info=True)
        log.info("Live engine stopped after %d cycles", self._cycle_count)

    def run_cycle(self) -> CycleResult:
        """Execute one full cycle of the agent pipeline.

        Cycles are serialised: if one is already running (e.g. a scheduled
        tick while a manual trigger is in flight) this call returns
        immediately with ``skipped=True`` rather than racing and
        double-submitting orders.

        Returns a :class:`CycleResult` summarizing what happened.
        """
        if not self._cycle_lock.acquire(blocking=False):
            log.warning("run_cycle skipped: a cycle is already in progress")
            return CycleResult(
                cycle_id=self._cycle_count,
                timestamp=datetime.utcnow(),
                skipped=True,
                error="cycle already in progress",
            )
        try:
            self._cycle_count += 1
            now = datetime.utcnow()
            result = CycleResult(cycle_id=self._cycle_count, timestamp=now)

            try:
                pit_view = self._data_feed.refresh(asof=now)

                prices = self._broker.get_current_prices(self._data_feed._universe)

                discrepancies = sync_portfolio_from_broker(
                    self._broker, self._portfolio, prices
                )
                result.discrepancies = discrepancies

                # Surface a reconciliation that ran with an incomplete
                # in-flight view as an operational alert.
                if any(d.get("type") == "open_orders_unavailable" for d in discrepancies):
                    result.alerts.append(self._emit_alert(
                        "reconciliation_degraded", "warning",
                        "Open orders unavailable; reconciliation may be incomplete.",
                    ))

                # Drawdown kill switch: once tripped, stop submitting new orders.
                self._check_drawdown(result)
                if self._halted:
                    result.halted = True
                    result.error = "halted: drawdown kill switch tripped"
                    self._cycle_history.append(result)
                    return result

                # Phase B: reflect on the previous cycle now that its P&L is known.
                self._maybe_reflect(now)

                context = {
                    "pit_view": pit_view,
                    "portfolio": self._portfolio,
                    "prices": prices,
                    "memory": self._memory,
                }
                orders, blackboard = self._orchestrator.step(context)
                result.orders_generated = len(orders)

                # Phase A: store this cycle's decision for deferred reflection.
                proposal = getattr(blackboard, "proposal", None)
                if proposal is not None:
                    date_str = now.strftime("%Y-%m-%d")
                    regime = ""
                    if hasattr(blackboard, "risk_decision") and blackboard.risk_decision:
                        regime = "; ".join(getattr(blackboard.risk_decision, "actions", []))
                    self._memory.store_decision(
                        date=date_str,
                        proposal_weights=dict(proposal.targets),
                        notes=f"cycle={self._cycle_count}; {regime}".strip("; "),
                        nav_at_decision=self._portfolio.nav,
                    )

                if not orders:
                    log.info("Cycle %d: no orders generated", self._cycle_count)
                    self._cycle_history.append(result)
                    return result

                daily_limit_breached = self._check_daily_limits(now, orders, prices)
                if daily_limit_breached:
                    result.alerts.append(self._alerts[-1])
                    auto_orders, manual_orders = [], orders
                else:
                    auto_orders, manual_orders = self._split_by_approval(orders)

                if auto_orders:
                    statuses, failed = self._execute_orders(
                        auto_orders, cycle_id=self._cycle_count
                    )
                    result.orders_submitted = len(statuses)
                    result.order_statuses = [self._status_to_dict(s) for s in statuses]
                    result.failed_orders = failed
                    result.orders_failed = len(failed)

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

            except Exception as exc:
                result.error = str(exc)
                log.error("Cycle %d failed: %s", self._cycle_count, exc, exc_info=True)
                # A broker/connectivity failure is operationally distinct from a
                # logic error — surface it as an alert so an operator is notified.
                if isinstance(exc, BrokerError):
                    result.alerts.append(self._emit_alert(
                        "broker_unavailable", "critical",
                        f"Broker call failed during cycle: {exc}",
                    ))

            self._cycle_history.append(result)
            return result
        finally:
            self._cycle_lock.release()

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
    ) -> tuple[list[OrderStatus], list[dict[str, Any]]]:
        """Submit orders to the broker.

        Returns ``(statuses, failed)`` where *failed* lists the orders whose
        submission raised, so a partially-executed basket is visible to the
        caller/operator rather than silently dropped.  Each order carries a
        deterministic ``client_order_id`` so the broker can deduplicate a
        re-submission of the same cycle's order.
        """
        statuses: list[OrderStatus] = []
        failed: list[dict[str, Any]] = []
        for o in orders:
            req = OrderRequest(
                symbol=o["symbol"],
                side=o["side"],
                quantity=o.get("quantity", abs(o.get("shares", 0))),
                order_type=o.get("order_type", "market"),
                limit_price=o.get("limit_price"),
                strategy=o.get("strategy", "composite"),
                client_order_id=f"c{cycle_id}-{o['symbol']}-{o['side']}",
            )
            try:
                status = self._broker.submit_order(req)
                statuses.append(status)
                log.info("Submitted: %s %s %.2f %s → %s",
                         req.side, req.symbol, req.quantity, req.order_type, status.status)
            except BrokerError as exc:
                log.error("Failed to submit order for %s", req.symbol, exc_info=True)
                failed.append({**o, "error": str(exc)})
        return statuses, failed

    @staticmethod
    def _status_to_dict(s: OrderStatus) -> dict[str, Any]:
        return {
            "order_id": s.order_id,
            "symbol": s.symbol,
            "side": s.side,
            "quantity": s.quantity,
            "filled_quantity": s.filled_quantity,
            "avg_fill_price": s.avg_fill_price,
            "status": s.status,
            "timestamp": s.timestamp.isoformat() if s.timestamp else None,
        }
