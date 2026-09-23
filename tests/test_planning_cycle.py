"""Tests for firm.live.planning_cycle.maybe_apply_overnight_plan.

See its module docstring: looks up a pending "planning"-sourced approval
and either applies it (submits the real orders now) if every symbol's
current price is still within tolerance of the plan's decision-time price,
or discards it (rejects, falls through to the normal fresh pipeline)
otherwise. All-or-nothing per plan -- one symbol moving too far discards
the whole plan, not just that symbol.
"""

from __future__ import annotations

from datetime import timedelta

from firm.live.approval import ApprovalQueue
from firm.live.planning_cycle import maybe_apply_overnight_plan
from firm.time_utils import utcnow
from tests.test_brokers import MockBroker


def _blackboard_stub():
    from firm.agents.blackboard import Blackboard

    return Blackboard(asof=utcnow())


def _queue_with_plan(orders, expiry_minutes=600) -> tuple[ApprovalQueue, str]:
    broker = MockBroker()
    broker.connect()
    queue = ApprovalQueue(broker=broker, expiry_minutes=expiry_minutes)
    aid = queue.add(
        orders=orders, blackboard=_blackboard_stub(),
        strategy="momentum", source_cycle_type="planning",
    )
    return queue, aid


class TestNoPlanPresent:
    def test_no_pending_approval_returns_false_unchanged(self):
        broker = MockBroker()
        broker.connect()
        queue = ApprovalQueue(broker=broker)

        outcome = maybe_apply_overnight_plan(queue, {"AAPL": 150.0}, price_tolerance_pct=0.5)

        assert outcome.applied is False
        assert outcome.approval_id is None

    def test_pending_approval_from_a_non_planning_cycle_is_ignored(self):
        """An approval queued for some other reason (e.g. a daily-limit
        breach forcing manual review) must not be mistaken for an
        overnight plan."""
        broker = MockBroker()
        broker.connect()
        queue = ApprovalQueue(broker=broker)
        queue.add(orders=[{"symbol": "AAPL", "side": "buy", "quantity": 1, "price": 150.0}],
                  blackboard=_blackboard_stub(), strategy="momentum")

        outcome = maybe_apply_overnight_plan(queue, {"AAPL": 150.0}, price_tolerance_pct=0.5)

        assert outcome.applied is False
        assert outcome.approval_id is None


class TestFreshPlanApplies:
    def test_within_tolerance_applies_and_submits(self):
        orders = [{"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0}]
        queue, aid = _queue_with_plan(orders)

        outcome = maybe_apply_overnight_plan(queue, {"AAPL": 150.30}, price_tolerance_pct=0.5)

        assert outcome.applied is True
        assert outcome.approval_id == aid
        assert len(outcome.statuses) == 1
        assert queue.get_by_id(aid).status == "approved"

    def test_exactly_at_tolerance_boundary_still_applies(self):
        orders = [{"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 100.0}]
        queue, aid = _queue_with_plan(orders)

        # Exactly 0.5% deviation with a 0.5% tolerance -- boundary inclusive.
        outcome = maybe_apply_overnight_plan(queue, {"AAPL": 100.50}, price_tolerance_pct=0.5)

        assert outcome.applied is True


class TestStalePlanDiscarded:
    def test_single_symbol_past_tolerance_rejects_whole_plan(self):
        orders = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0},
            {"symbol": "MSFT", "side": "sell", "quantity": 5, "price": 300.0},
        ]
        queue, aid = _queue_with_plan(orders)

        # AAPL within tolerance, MSFT moved 5% -- all-or-nothing means the
        # whole plan is discarded, not just MSFT's leg.
        outcome = maybe_apply_overnight_plan(
            queue, {"AAPL": 150.30, "MSFT": 315.0}, price_tolerance_pct=0.5,
        )

        assert outcome.applied is False
        assert outcome.approval_id == aid
        assert "MSFT" in outcome.reason
        assert queue.get_by_id(aid).status == "rejected"

    def test_discarded_plan_never_reaches_broker(self):
        orders = [{"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 100.0}]
        queue, aid = _queue_with_plan(orders)
        cash_before = queue._broker._cash

        maybe_apply_overnight_plan(queue, {"AAPL": 120.0}, price_tolerance_pct=0.5)

        assert queue._broker._cash == cash_before

    def test_expired_plan_returns_false_and_marks_expired(self):
        """ApprovalQueue.get_pending() already expires stale approvals as
        a side effect of being called -- an expired plan is never seen as
        "pending" by our lookup, so it falls into the same "none exists"
        bucket as no-plan-at-all rather than a separately-reported case."""
        orders = [{"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0}]
        queue, aid = _queue_with_plan(orders, expiry_minutes=600)
        # Force it into the past rather than waiting real time.
        approval = queue.get_by_id(aid)
        approval.created_at = utcnow() - timedelta(hours=12)
        approval.expires_at = utcnow() - timedelta(hours=2)

        outcome = maybe_apply_overnight_plan(queue, {"AAPL": 150.0}, price_tolerance_pct=0.5)

        assert outcome.applied is False
        assert outcome.approval_id is None
        assert queue.get_by_id(aid).status == "expired"

    def test_missing_current_price_for_a_symbol_does_not_crash(self):
        """A symbol with no current price available (e.g. a stale/removed
        feed entry) must be skipped in the deviation check, not raise."""
        orders = [{"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0}]
        queue, aid = _queue_with_plan(orders)

        outcome = maybe_apply_overnight_plan(queue, {}, price_tolerance_pct=0.5)

        # No price to compare against -> treated as within tolerance
        # (0 deviation measured), not a reason to discard.
        assert outcome.applied is True
