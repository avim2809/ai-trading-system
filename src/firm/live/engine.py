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
    skipped: bool = False
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
    ) -> None:
        self._config = config
        self._broker = broker
        self._data_feed = data_feed
        self._approval_queue = approval_queue
        self._approval_mode = approval_mode
        self._auto_approve: set[str] = set(auto_approve_strategies or [])

        self._orchestrator = build_orchestrator(config)
        self._portfolio = PortfolioState(
            initial_capital=config.get("initial_capital", 100_000)
        )
        self._cycle_count = 0
        self._cycle_history: list[CycleResult] = []
        self._running = False
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

                context = {
                    "pit_view": pit_view,
                    "portfolio": self._portfolio,
                    "prices": prices,
                }
                orders, blackboard = self._orchestrator.step(context)
                result.orders_generated = len(orders)

                if not orders:
                    log.info("Cycle %d: no orders generated", self._cycle_count)
                    self._cycle_history.append(result)
                    return result

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

            self._cycle_history.append(result)
            return result
        finally:
            self._cycle_lock.release()

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
