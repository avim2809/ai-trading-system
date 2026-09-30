"""Regression tests for the 2026-09-30 adversarial review of allocation mode.

Each test reproduces a reviewed failure scenario (one-sided quote leverage,
off-hours / duplicate orders, crash mid-submission, unmanaged-liquidation
retry, non-full_auto approval) against the real engine with mock brokers.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import patch

import pytest

from firm.brokers.base import BrokerPosition, OrderStatus
from firm.time_utils import utcnow
from tests.test_allocation_mode import AllocBroker, _alloc_cfg, _make_engine, _run_at

TUE_OPEN = datetime(2026, 10, 6, 13, 30)     # 09:30 ET
TUE_LATER = datetime(2026, 10, 6, 14, 30)
WED_OPEN = datetime(2026, 10, 7, 13, 30)
SAT = datetime(2026, 10, 10, 15, 0)


def _gross_over_equity(br) -> float:
    acct = br.get_account()
    return sum(abs(p.market_value) for p in br.get_positions()) / acct["equity"]


class HalfQuoteBroker(AllocBroker):
    """A one-sided quote (ask=0) makes the adapter's 'mid' half the real price."""
    def get_current_prices(self, symbols):
        return {s: (250.0 if s == "SPY" else self._prices[s]) for s in symbols}


class PendingBroker(AllocBroker):
    """Accepts orders but never fills them (e.g. queued while the market is closed)."""
    def __init__(self):
        super().__init__()
        self.open: dict[str, OrderStatus] = {}

    def submit_order(self, order):
        self.submitted.append(order)
        oid = uuid.uuid4().hex[:8]
        st = OrderStatus(order_id=oid, symbol=order.symbol, side=order.side, quantity=order.quantity,
                         filled_quantity=0.0, avg_fill_price=0.0, status="pending", timestamp=utcnow())
        self.open[oid] = st
        return st

    def get_order_status(self, oid):
        return self.open[oid]

    def get_open_orders(self):
        return list(self.open.values())


def test_one_sided_quote_does_not_lever_the_account(tmp_path):
    eng, br, _, _ = _make_engine(tmp_path, HalfQuoteBroker(), allocation=_alloc_cfg(max_order_notional=70_000))
    _run_at(eng, TUE_OPEN)
    spy = [o for o in br.submitted if o.symbol == "SPY"]
    assert spy, "core should still be bought"
    # Sized from the last close (500), not the half-price quote (250).
    assert spy[0].quantity * 500.0 <= 0.6 * 100_000 * 1.01
    assert _gross_over_equity(br) <= 1.01


def test_backstop_refuses_plan_that_exceeds_gross_at_last_close(tmp_path):
    eng, br, _, _ = _make_engine(tmp_path, allocation=_alloc_cfg())
    # Force the planner itself to size off a wrong price that slips past the
    # sanity check (e.g. a stale history): patch the allocator's prices.
    orig = eng._allocation_prices
    eng._allocation_prices = lambda positions, history: {**orig(positions, history), "SPY": 250.0}
    res = _run_at(eng, TUE_OPEN)
    assert br.submitted == []
    assert any(a.get("kind") == "allocation_gross_backstop" for a in res.alerts)


def test_forced_weekend_cycle_places_no_orders(tmp_path):
    eng, br, _, _ = _make_engine(tmp_path, allocation=_alloc_cfg())
    _run_at(eng, TUE_OPEN)
    n0 = len(br.submitted)
    br._prices["SPY"] = 600.0
    for sym, p in list(br._positions.items()):
        br._positions[sym] = BrokerPosition(sym, p.quantity, p.avg_cost, p.quantity * br._prices[sym])
    br._market_open = False
    res = _run_at(eng, SAT, force=True)
    assert len(br.submitted) == n0
    assert res.skipped and "market closed" in (res.error or "")


def test_market_clock_failure_fails_closed(tmp_path):
    class ClockFail(AllocBroker):
        def is_market_open(self):
            raise RuntimeError("clock endpoint 503")
    eng, br, _, _ = _make_engine(tmp_path, ClockFail(), allocation=_alloc_cfg())
    _run_at(eng, datetime(2026, 10, 6, 23, 0))
    assert br.submitted == []


def test_orders_left_open_block_next_days_plan(tmp_path):
    eng, br, _, _ = _make_engine(tmp_path, PendingBroker(), allocation=_alloc_cfg())
    _run_at(eng, TUE_OPEN)
    n1 = len(br.submitted)
    assert n1 > 0
    _run_at(eng, TUE_LATER)
    _run_at(eng, WED_OPEN)
    assert len(br.submitted) == n1, "no duplicate orders while earlier ones are still working"


def test_open_orders_read_failure_fails_closed(tmp_path):
    class OpenFail(AllocBroker):
        def get_open_orders(self):
            raise RuntimeError("orders endpoint 500")
    eng, br, _, _ = _make_engine(tmp_path, OpenFail(), allocation=_alloc_cfg())
    _run_at(eng, TUE_OPEN)
    assert br.submitted == []


def test_crash_mid_submission_does_not_double_buy(tmp_path):
    br = PendingBroker()
    eng, br, _, _ = _make_engine(tmp_path, br, allocation=_alloc_cfg())
    real_persist = type(eng)._persist_allocation_state
    calls = {"n": 0}

    def persist_until_submit(self):
        # Let the pre-submission 'submitting' save through, then "crash":
        # nothing after it is persisted.
        if calls["n"] == 0 and self._allocation_state.get("day_status") == "submitting":
            calls["n"] += 1
            return real_persist(self)
        return None

    with patch.object(type(eng), "_persist_allocation_state", persist_until_submit):
        _run_at(eng, TUE_OPEN)
    n1 = len(br.submitted)
    eng2, _, _, _ = _make_engine(tmp_path, br, allocation=_alloc_cfg())
    assert eng2._allocation_state.get("day_status") == "submitting"
    _run_at(eng2, TUE_LATER)
    assert len(br.submitted) == n1, "orders still working -> no re-submission after restart"


def test_rejected_unmanaged_liquidation_schedules_retry(tmp_path):
    br = AllocBroker()
    br._prices["AAPL"] = 200.0
    br.seed_position("AAPL", 10)
    eng, br, _, _ = _make_engine(tmp_path, br, allocation=_alloc_cfg(liquidate_unmanaged=True))
    _run_at(eng, TUE_OPEN)
    recs = [r for r in eng._allocation_state["day_orders"] if r["symbol"] == "AAPL"]
    assert recs and recs[0]["sleeves"] == ["_unmanaged"]
    oid = recs[0]["order_id"]
    br._orders[oid] = OrderStatus(order_id=oid, symbol="AAPL", side="sell", quantity=10,
                                  filled_quantity=0, avg_fill_price=0, status="rejected",
                                  timestamp=utcnow())
    assert eng._check_allocation_orders() == "failed"


def test_allocation_mode_requires_full_auto(tmp_path):
    with pytest.raises(ValueError, match="full_auto"):
        _make_engine(tmp_path, allocation=_alloc_cfg(), approval_mode="semi_auto")


def test_cash_buffer_scales_targets_proportionally():
    from firm.allocation.allocator import build_allocator
    alloc = build_allocator({**_alloc_cfg(), "cash_buffer": 0.01,
                             "sleeves": [{"name": "core", "type": "static", "weight": 1.0,
                                          "weights": {"SPY": 0.6, "IEF": 0.4}, "rebalance": "monthly"}]})
    plan = alloc.plan(TUE_OPEN, 100_000, {}, {"SPY": 500.0, "IEF": 95.0}, {}, {"core": None})
    assert plan.targets["SPY"] == pytest.approx(0.594)
    assert plan.targets["IEF"] == pytest.approx(0.396)


# ---------------------------------------------------------------------------
# Second review round (2026-09-30)
# ---------------------------------------------------------------------------

FULL_CORE = {"name": "core", "type": "static", "weight": 1.0,
             "weights": {"SPY": 0.6, "IEF": 0.4}, "rebalance": "monthly"}
NOV2 = datetime(2026, 11, 2, 14, 30)       # monthly rebalance day


def _remark(br, prices):
    br._prices.update(prices)
    for s, p in list(br._positions.items()):
        br._positions[s] = BrokerPosition(s, p.quantity, p.avg_cost, p.quantity * br._prices[s])


@pytest.mark.parametrize("spy", [485.0, 480.0, 470.0])
def test_backstop_does_not_block_rebalance_on_down_days(tmp_path, spy):
    eng, br, _, _ = _make_engine(tmp_path, allocation=_alloc_cfg(sleeves=[dict(FULL_CORE)], cash_buffer=0.01))
    _run_at(eng, TUE_OPEN)
    _remark(br, {"SPY": spy, "IEF": 95.5})
    res = _run_at(eng, NOV2)
    assert not any(a.get("kind") == "allocation_gross_backstop" for a in res.alerts)
    assert eng._allocation_state.get("day_status") in ("complete", "settled")


def test_risk_reducing_plan_always_passes_backstop(tmp_path):
    eng, br, feed, _ = _make_engine(tmp_path, allocation=_alloc_cfg(sleeves=[dict(FULL_CORE)], cash_buffer=0.01))
    _run_at(eng, TUE_OPEN)
    _remark(br, {"SPY": 470.0, "IEF": 95.5})
    sells = [{"symbol": "SPY", "side": "sell", "quantity": 2.0, "price": 470.0}]
    ok, _ = eng._allocation_gross_ok(sells, br.get_positions(), dict(feed._history), br.get_account()["cash"])
    assert ok


def test_untraded_unpriced_position_does_not_block_backstop(tmp_path):
    eng, br, feed, _ = _make_engine(tmp_path, allocation=_alloc_cfg(sleeves=[dict(FULL_CORE)], cash_buffer=0.01))
    _run_at(eng, TUE_OPEN)
    buys = [{"symbol": "IEF", "side": "buy", "quantity": 1.0, "price": 95.0}]
    for extra in (BrokerPosition("XYZ", 10, 5.0, 0.0), BrokerPosition("OLD", 0.0, 5.0, 0.0)):
        ok, why = eng._allocation_gross_ok(buys, br.get_positions() + [extra], dict(feed._history),
                                           br.get_account()["cash"])
        assert ok, why


def test_traded_symbol_without_reference_fails_closed(tmp_path):
    eng, br, feed, _ = _make_engine(tmp_path, allocation=_alloc_cfg(sleeves=[dict(FULL_CORE)]))
    buys = [{"symbol": "NEW", "side": "buy", "quantity": 1.0, "price": 10.0}]
    ok, why = eng._allocation_gross_ok(buys, [], dict(feed._history), 100_000.0)
    assert not ok and "NEW" in why


def test_bad_held_mark_does_not_trade_the_wrong_way(tmp_path):
    class BadMark(AllocBroker):
        bad = False

        def get_positions(self):
            out = []
            for p in super().get_positions():
                if self.bad and p.symbol == "SPY":
                    p = BrokerPosition(p.symbol, p.quantity, p.avg_cost, p.market_value * 2)
                out.append(p)
            return out

        def get_account(self):
            a = super().get_account()
            if self.bad and "SPY" in self._positions:
                a["equity"] += self._positions["SPY"].market_value
            return a

    eng, br, _, _ = _make_engine(tmp_path, BadMark(), allocation=_alloc_cfg(sleeves=[dict(FULL_CORE)], cash_buffer=0.01))
    _run_at(eng, TUE_OPEN)
    br.bad = True
    n0 = len(br.submitted)
    _run_at(eng, WED_OPEN)                 # daily drift check with a 2x SPY mark
    assert br.submitted[n0:] == [], "a corrected mark must not leave a phantom drift"


def test_open_order_block_alerts_and_escalates(tmp_path):
    eng, br, _, _ = _make_engine(tmp_path, PendingBroker(), allocation=_alloc_cfg())
    _run_at(eng, TUE_OPEN)
    later = _run_at(eng, TUE_LATER)
    blocked_day1 = [a for a in later.alerts if a.get("kind") == "allocation_blocked_by_open_orders"]
    # Same-day "pending" is the normal wait; the open-orders gate fires on the next day's first run.
    nxt = _run_at(eng, WED_OPEN)
    kinds = [(a.get("kind"), a.get("severity")) for a in nxt.alerts]
    assert ("allocation_blocked_by_open_orders", "warning") in kinds or blocked_day1
    thu = _run_at(eng, datetime(2026, 10, 8, 13, 30))
    assert ("allocation_blocked_by_open_orders", "critical") in [(a.get("kind"), a.get("severity")) for a in thu.alerts]


def test_runtime_approval_mode_change_is_refused_by_cycle(tmp_path):
    eng, br, _, _ = _make_engine(tmp_path, allocation=_alloc_cfg())
    eng._approval_mode = "semi_auto"
    res = _run_at(eng, TUE_OPEN)
    assert br.submitted == []
    assert any(a.get("kind") == "allocation_approval_mode" for a in res.alerts)


def test_config_put_refuses_non_full_auto_in_allocation_mode():
    from types import SimpleNamespace

    from fastapi import HTTPException

    from firm.api.routers import live as live_router

    engine = SimpleNamespace(strategy_mode="allocation", _approval_mode="full_auto")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(live_engine=engine)))
    body = live_router.ConfigUpdateRequest(approval_mode="semi_auto")
    with pytest.raises(HTTPException) as exc:
        live_router.update_live_config(body, request)
    assert exc.value.status_code == 400
    assert engine._approval_mode == "full_auto"
