"""Applies (or discards) an overnight/pre-open "planning" cycle's plan.

A "planning" cycle (see ``firm.live.scheduler``'s planning-cycle job and
``Orchestrator.step``'s ``dry_run`` param) runs the full analysis pipeline
before market open and parks its resulting orders in the approval queue
(forced there regardless of the instance's real ``approval_mode`` -- see
``LiveTradingEngine._run_cycle_work``) rather than submitting them. This
module is the other half: at the real market-open cycle, decide whether
that plan is still representative of reality and, if so, submit it instead
of running the normal fresh pipeline again.

Deliberately conservative: any single symbol moving past
``price_tolerance_pct`` discards the WHOLE plan (all-or-nothing), not just
that symbol -- a plan is one coherent portfolio-level decision, not a bag
of independent per-symbol bets that can be partially stale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from firm.brokers.base import OrderStatus
from firm.live.approval import ApprovalQueue

log = logging.getLogger(__name__)


@dataclass
class OvernightPlanOutcome:
    applied: bool
    approval_id: str | None = None
    reason: str = ""
    max_deviation_pct: float = 0.0
    statuses: list[tuple[OrderStatus, str]] = field(default_factory=list)


def _find_pending_plan(approval_queue: ApprovalQueue):
    for approval in approval_queue.get_pending():
        if approval.source_cycle_type == "planning":
            return approval
    return None


def maybe_apply_overnight_plan(
    approval_queue: ApprovalQueue,
    current_prices: dict[str, float],
    price_tolerance_pct: float,
) -> OvernightPlanOutcome:
    """Look up a pending planning-cycle approval and apply it if still fresh.

    Returns ``OvernightPlanOutcome(applied=False)`` unchanged from today's
    behavior when there is no pending planning approval (the common case
    whenever ``planning_cycle`` isn't enabled) -- this function is a pure
    no-op addition to the ``"open"`` cycle in that case.
    """
    # get_pending() (used by _find_pending_plan) already calls
    # expire_stale() internally -- an expired approval is never "pending"
    # by the time we look, so it naturally falls into the same "none
    # exists" bucket below. No separate expiry check needed here.
    plan = _find_pending_plan(approval_queue)
    if plan is None:
        return OvernightPlanOutcome(applied=False, reason="no pending planning approval")

    max_deviation = 0.0
    worst_symbol = None
    for order in plan.orders:
        symbol = order.get("symbol")
        plan_price = float(order.get("price", 0.0) or 0.0)
        now_price = current_prices.get(symbol)
        if not plan_price or now_price is None:
            continue
        deviation_pct = abs(now_price - plan_price) / plan_price * 100.0
        if deviation_pct > max_deviation:
            max_deviation, worst_symbol = deviation_pct, symbol

    if max_deviation > price_tolerance_pct:
        reason = (
            f"{worst_symbol} moved {max_deviation:.2f}% "
            f"(tolerance {price_tolerance_pct:.2f}%)"
        )
        approval_queue.reject(plan.approval_id, reason=reason)
        log.info(
            "Overnight plan %s discarded: %s", plan.approval_id, reason,
        )
        return OvernightPlanOutcome(
            applied=False, approval_id=plan.approval_id, reason=reason,
            max_deviation_pct=max_deviation,
        )

    statuses = approval_queue.approve(plan.approval_id)
    log.info(
        "Overnight plan %s applied at market open: %d order(s), max deviation %.2f%%",
        plan.approval_id, len(statuses), max_deviation,
    )
    return OvernightPlanOutcome(
        applied=True, approval_id=plan.approval_id,
        reason="within tolerance", max_deviation_pct=max_deviation,
        statuses=statuses,
    )
