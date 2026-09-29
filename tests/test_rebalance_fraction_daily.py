"""rebalance_fraction must mean "per trading day", not "per cycle".

Validated in daily-cadence backtests (one ExecutionAgent call per day), but live
runs ~7 cycles per trading day on the same completed-bar panel: re-applying the
0.7 fraction every cycle closed ~97% of a gap within 3 cycles. See
docs/optimal_combination_fix_2026_09.md section 6.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from firm.agents.base import AgentContext
from firm.agents.execution import ExecutionAgent
from firm.contracts.models import RiskDecision
from firm.portfolio.state import PortfolioState

PRICES = {"AAPL": 100.0}
CFG = {"rebalance_band_pct": 0.05, "rebalance_fraction": 0.7, "close_dust_fraction": 0.2}
DAY1 = datetime(2026, 9, 28, 14, 30)  # naive UTC = 10:30 ET


def _book() -> PortfolioState:
    return PortfolioState(initial_capital=1_000_000)


def _cycle(agent, pf, target, now, fill=True, **inputs):
    report = agent.run(
        AgentContext(now=now),
        decision=RiskDecision(approved=True, adjusted_targets={"AAPL": target}),
        portfolio=pf, prices=PRICES, **inputs,
    )
    if fill and report.fills:
        pf.update(report.fills, PRICES)
    return report


def _w(pf) -> float:
    return pf.get_weights(PRICES).get("AAPL", 0.0)


def test_stable_target_trades_once_per_day():
    agent, pf = ExecutionAgent(config=CFG), _book()
    first = _cycle(agent, pf, 0.20, DAY1)
    assert len(first.fills) == 1
    assert _w(pf) == pytest.approx(0.14, abs=1e-3)  # 0.7 of the 0.20 gap
    for h in range(1, 7):  # 6 more intraday cycles, same target
        assert _cycle(agent, pf, 0.20, DAY1 + timedelta(hours=h)).fills == []
    assert _w(pf) == pytest.approx(0.14, abs=1e-3)


def test_next_day_takes_the_next_fractional_step():
    agent, pf = ExecutionAgent(config=CFG), _book()
    _cycle(agent, pf, 0.20, DAY1)
    _cycle(agent, pf, 0.20, DAY1 + timedelta(hours=3))
    day2 = _cycle(agent, pf, 0.20, DAY1 + timedelta(days=1))
    assert len(day2.fills) == 1
    assert _w(pf) == pytest.approx(0.14 + 0.7 * 0.06, abs=1e-3)


def test_matches_one_call_per_day_backtest_cadence():
    """With one call per day, the schedule is identical to before the fix."""
    daily, pf_d = ExecutionAgent(config=CFG), _book()
    per_cycle_ref = []
    w = 0.0
    for d in range(4):
        _cycle(daily, pf_d, 0.20, DAY1 + timedelta(days=d))
        w = w + 0.7 * (0.20 - w) if abs(0.20 - w) >= 0.05 else w
        per_cycle_ref.append(w)
        assert _w(pf_d) == pytest.approx(w, abs=1e-3)


def test_unfilled_first_order_is_retried_same_day():
    agent, pf = ExecutionAgent(config=CFG), _book()
    assert len(_cycle(agent, pf, 0.20, DAY1, fill=False).fills) == 1
    retry = _cycle(agent, pf, 0.20, DAY1 + timedelta(hours=1))
    assert len(retry.fills) == 1
    assert _w(pf) == pytest.approx(0.14, abs=1e-3)


def test_real_intraday_target_change_is_acted_on_at_damped_rate():
    agent, pf = ExecutionAgent(config=CFG), _book()
    _cycle(agent, pf, 0.20, DAY1)  # -> 0.14
    moved = _cycle(agent, pf, 0.40, DAY1 + timedelta(hours=2))
    assert len(moved.fills) == 1
    # day's fractional target from the start-of-day weight (0.0)
    assert _w(pf) == pytest.approx(0.7 * 0.40, abs=1e-3)


def test_forced_full_close_is_not_held_back():
    agent, pf = ExecutionAgent(config=CFG), _book()
    _cycle(agent, pf, 0.20, DAY1)
    close = _cycle(agent, pf, 0.0, DAY1 + timedelta(hours=1))
    assert len(close.fills) == 1
    assert _w(pf) == pytest.approx(0.0, abs=1e-6)


def test_books_are_isolated():
    """One ExecutionAgent shared across sleeves: anchors must not collide."""
    agent = ExecutionAgent(config=CFG)
    a, b = _book(), _book()
    _cycle(agent, a, 0.20, DAY1, book_key="sleeve:a")
    first_b = _cycle(agent, b, 0.20, DAY1 + timedelta(minutes=5), book_key="sleeve:b")
    assert len(first_b.fills) == 1
    assert _w(b) == pytest.approx(0.14, abs=1e-3)


def test_fraction_one_is_unaffected():
    agent, pf = ExecutionAgent(config={**CFG, "rebalance_fraction": 1.0}), _book()
    _cycle(agent, pf, 0.20, DAY1)
    assert _w(pf) == pytest.approx(0.20, abs=1e-3)
    assert _cycle(agent, pf, 0.20, DAY1 + timedelta(hours=1)).fills == []


def test_day_boundary_uses_market_timezone():
    """21:00 ET on day 1 and 09:30 ET on day 2 are different trading days even
    though 21:00 ET is already the next UTC date."""
    agent, pf = ExecutionAgent(config=CFG), _book()
    _cycle(agent, pf, 0.20, datetime(2026, 9, 28, 20, 0))  # 16:00 ET day 1
    late = _cycle(agent, pf, 0.20, datetime(2026, 9, 29, 1, 0))  # 21:00 ET, still day 1
    assert late.fills == []
    nxt = _cycle(agent, pf, 0.20, datetime(2026, 9, 29, 13, 30))  # 09:30 ET day 2
    assert len(nxt.fills) == 1
