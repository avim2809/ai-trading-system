"""P4-04: tiered drawdown and operational kill switch (pure module, fault-injection tests)."""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from firm.risk import kill_switch as ks
from firm.risk.kill_switch import Action, KillSwitchConfig, KillSwitchState, Tier

TODAY = date(2026, 3, 4)  # Wednesday
BARS = {"SPY": date(2026, 3, 4)}


def cfg(p95: float = 0.12, tau: float = 0.04, **kw) -> KillSwitchConfig:
    return KillSwitchConfig(charter_dd_bootstrap_p95=p95, tau=tau, **kw)  # survival_ref = 0.12 (2.5 tau = 0.10)


def series(values: list[float], end: date = TODAY) -> pd.Series:
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def run(values, state=None, c=None, **kw):
    args = {
        "last_bar_date": BARS, "today": TODAY, "broker_connected": True, "in_rebalance_window": False,
        "position_breaks": (),
    }
    args.update(kw)
    s = state if state is not None else KillSwitchState.new()
    return ks.evaluate(values if isinstance(values, pd.Series) else series(values), s, c or cfg(), **args)


# ---- drawdown tiers -------------------------------------------------------------------------------------------------
def test_dd_below_soft_ok():
    d = run([1.0, 1.1, 1.1 * 0.95])
    assert d.tier is Tier.OK and d.action is Action.NONE and d.triggers == () and d.alerts == ()


def test_dd_soft_boundary():
    assert run([1.0, 0.89]).tier is Tier.OK  # 11% < 12%
    d = run([1.0, 0.88])  # exactly 1.0x survival_ref
    assert d.tier is Tier.SOFT and "dd_soft" in d.triggers


def test_dd_hard_boundary():
    assert run([1.0, 0.83]).tier is Tier.SOFT  # 17% < 18%
    d = run([1.0, 0.82])  # exactly 1.5x survival_ref
    assert d.tier is Tier.HARD and d.action is Action.BLOCK_NEW_ORDERS and "dd_hard" in d.triggers


@pytest.mark.parametrize("p95,tau,expected", [(0.30, 0.04, 0.30), (0.05, 0.04, 0.10)])
def test_survival_ref_uses_max_of_p95_and_2p5_tau(p95, tau, expected):
    assert cfg(p95, tau).survival_ref == pytest.approx(expected)


def test_config_validation():
    with pytest.raises(ValueError):
        cfg(p95=-0.1)
    with pytest.raises(ValueError):
        cfg(tau=0.0)
    with pytest.raises(ValueError):
        cfg(soft_mult=2.0, hard_mult=1.5)


# ---- reset, HWM, corrections ----------------------------------------------------------------------------------------
def _tripped():
    d = run([1.0, 0.80])
    assert d.tier is Tier.HARD
    return d.state


def test_hwm_immutable_after_reset():
    st = _tripped()
    st2 = ks.record_reset(st, approver="owner", reason="reviewed", at=datetime(2026, 3, 5, 9, tzinfo=UTC), current_unit_value=0.80)
    assert st2.hwm == st.hwm == 1.0 and len(st2.resets) == 1 and len(st.resets) == 0
    with pytest.raises(ValueError):
        ks.record_reset(st, approver="", reason="x", at=datetime(2026, 3, 5, tzinfo=UTC), current_unit_value=0.8)
    with pytest.raises(ValueError):
        ks.record_reset(st, approver="owner", reason="  ", at=datetime(2026, 3, 5, tzinfo=UTC), current_unit_value=0.8)


def test_approved_reset_rearms_tier():
    st = ks.record_reset(_tripped(), approver="o", reason="r", at=datetime(2026, 3, 4, 20, tzinfo=UTC), current_unit_value=0.80)
    later = TODAY + timedelta(days=1)
    s = series([1.0, 0.80, 0.80], end=later)
    d = run(s, st, today=later, last_bar_date={"SPY": later})
    assert d.tier is Tier.OK and d.dd_from_baseline == 0.0
    # further loss of 1.5x survival_ref (18%) from the baseline re-triggers
    s2 = series([1.0, 0.80, 0.80, 0.80 * 0.82], end=later + timedelta(days=1))
    day2 = later + timedelta(days=1)
    d2 = run(s2, st, today=day2, last_bar_date={"SPY": day2})
    assert d2.tier is Tier.HARD


def test_hard_is_latched_until_reset():
    d = run([1.0, 0.80])
    d2 = run([1.0, 0.80, 1.0], d.state)  # NAV recovers, no approved reset
    assert d2.tier is Tier.HARD and d2.state.halted


def test_reset_does_not_hide_cumulative_dd_from_decommission():
    st = ks.record_reset(_tripped(), approver="o", reason="r", at=datetime(2026, 3, 4, 20, tzinfo=UTC), current_unit_value=0.80)
    d = run([1.0, 0.80], st)
    assert d.dd_immutable == pytest.approx(0.20) and d.dd_from_baseline == pytest.approx(0.0)


def test_nav_correction_audited_and_excludes_spike():
    s = series([1.0, 1.5, 1.0, 0.99])  # spurious spike at index 1
    bad = s.index[1].date()
    spiked = run(s)
    assert spiked.tier is Tier.HARD  # false trip from the spike
    st = ks.record_nav_correction(
        spiked.state, bad_date=bad, approver="o", reason="vendor print error", at=datetime(2026, 3, 4, 20, tzinfo=UTC)
    )
    assert len(st.nav_corrections) == 1 and st.nav_corrections[0]["bad_date"] == bad
    assert len(s) == 4  # nothing deleted
    d = run(s, st)
    assert d.state.hwm == pytest.approx(1.0) and d.tier is Tier.OK
    assert len(d.state.nav_corrections) == 1
    with pytest.raises(ValueError):
        ks.record_nav_correction(st, bad_date=bad, approver="", reason="r", at=datetime(2026, 3, 4, tzinfo=UTC))


def test_deposit_does_not_move_hwm():
    # a 50% deposit is not a return: unit value is built from returns only
    u = ks.unit_value(pd.Series([0.0, 0.01, -0.01], index=pd.bdate_range("2026-03-02", periods=3)))
    assert u.max() == pytest.approx(1.01) and u.iloc[0] == pytest.approx(1.0)
    assert run(u).state.hwm == pytest.approx(1.01)


def test_withdrawal_creates_no_drawdown():
    u = ks.unit_value(pd.Series([0.01, 0.0, 0.0, 0.0], index=pd.bdate_range("2026-03-02", periods=4)))
    assert run(u).dd_immutable == 0.0


def test_drawdown_function():
    assert ks.drawdown(pd.Series([1.0, 0.9]), 1.2) == pytest.approx(0.25)


# ---- actions ---------------------------------------------------------------------------------------------------------
def test_hard_tier_does_not_emit_liquidation_by_default():
    d = run([1.0, 0.5])
    assert d.action is Action.BLOCK_NEW_ORDERS and d.exposure_factor == 1.0


def test_flatten_forbidden_during_stale_or_disconnect():
    c = cfg(flatten_on_hard=True)
    assert run([1.0, 0.5], c=c).action is Action.FLATTEN and run([1.0, 0.5], c=c).exposure_factor == 0.0
    stale = run([1.0, 0.5], c=c, last_bar_date={"SPY": date(2026, 2, 20)})
    assert stale.tier is Tier.HARD and stale.action is Action.BLOCK_NEW_ORDERS and stale.exposure_factor == 1.0
    disc = run([1.0, 0.5], c=c, broker_connected=False, in_rebalance_window=True)
    assert disc.action is Action.BLOCK_NEW_ORDERS and disc.exposure_factor == 1.0


def test_soft_is_alert_only():
    d = run([1.0, 0.87])
    assert d.tier is Tier.SOFT and d.action is Action.NONE and d.exposure_factor == 1.0
    assert [a["kind"] for a in d.alerts] == ["kill_soft_warning"]


# ---- operational faults ----------------------------------------------------------------------------------------------
def test_stale_data_triggers_after_2_trading_days_not_weekend():
    fri = date(2026, 2, 27)
    mon, tue, wed = date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)
    assert fri.weekday() == 4
    for today, stale in ((mon, False), (tue, False), (wed, True)):
        d = run([1.0, 1.0], today=today, last_bar_date={"SPY": fri})
        assert (d.tier is Tier.HALT) is stale, today
        if stale:
            assert "stale_data:SPY" in d.triggers and d.action is Action.BLOCK_NEW_ORDERS
            assert d.alerts[0]["kind"] == "kill_stale_data"


def test_special_closure_handled_via_p2_03_table():
    last, today = date(2025, 1, 8), date(2025, 1, 13)  # 2025-01-09 national day of mourning
    assert date(2025, 1, 9) in ks.SPECIAL_CLOSURES
    assert run([1.0], today=today, last_bar_date={"SPY": last}).tier is Tier.OK


def test_known_false_trigger_for_closure_absent_from_table():
    last, today = date(2025, 1, 8), date(2025, 1, 13)
    d = run([1.0], today=today, last_bar_date={"SPY": last}, special_closures=frozenset())
    assert d.tier is Tier.HALT and "stale_data:SPY" in d.triggers  # documented limitation until the table is complete


def test_broker_disconnect_only_in_rebalance_window():
    assert run([1.0], broker_connected=False, in_rebalance_window=False).tier is Tier.OK
    d = run([1.0], broker_connected=False, in_rebalance_window=True)
    assert d.tier is Tier.HALT and "broker_disconnect" in d.triggers and d.action is Action.BLOCK_NEW_ORDERS


def test_position_break_trigger():
    d = run([1.0], position_breaks=[{"symbol": "SPY", "diff": 3}])
    assert d.tier is Tier.HALT and "position_break" in d.triggers


def test_precedence_hard_over_halt_over_soft():
    soft = run([1.0, 0.87])
    halt_soft = run([1.0, 0.87], position_breaks=[{"x": 1}])
    hard_halt = run([1.0, 0.80], position_breaks=[{"x": 1}])
    assert (soft.tier, halt_soft.tier, hard_halt.tier) == (Tier.SOFT, Tier.HALT, Tier.HARD)
    assert {"dd_soft", "position_break"} <= set(halt_soft.triggers)
    assert {"dd_hard", "position_break"} <= set(hard_halt.triggers)


def test_legacy_halt_recorded_not_lowering_tier():
    d = run([1.0, 0.80], engine_halted=True, breaker_active=True)
    assert d.tier is Tier.HARD and d.already_halted
    assert "engine_halted" in d.triggers and "breaker_active" in d.triggers
    ok = run([1.0, 1.0], engine_halted=True)
    assert ok.tier is Tier.OK and ok.already_halted and not run([1.0, 1.0]).already_halted


# ---- alerts ----------------------------------------------------------------------------------------------------------
def test_persistent_hard_alerts_once():
    d1 = run([1.0, 0.80])
    assert [a["kind"] for a in d1.alerts] == ["kill_hard_stop"]
    d2 = run([1.0, 0.80, 0.79], d1.state)
    assert d2.tier is Tier.HARD and d2.alerts == ()


def test_escalation_alerts_even_when_engine_halted():
    d1 = run([1.0, 0.87], engine_halted=True)
    assert d1.already_halted and len(d1.alerts) == 1
    d2 = run([1.0, 0.87, 0.80], d1.state, engine_halted=True)
    assert d2.tier is Tier.HARD and [a["kind"] for a in d2.alerts] == ["kill_hard_stop"]


def test_non_ok_decision_logs_warning(caplog):
    with caplog.at_level("WARNING", logger="firm.risk.kill_switch"):
        run([1.0, 0.87])
    assert any(r.levelname == "WARNING" for r in caplog.records)


# ---- purity ----------------------------------------------------------------------------------------------------------
def test_pure_no_io(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("I/O attempted")

    monkeypatch.setattr(builtins, "open", boom)
    monkeypatch.setattr(Path, "open", boom)
    run([1.0, 0.80], position_breaks=[{"x": 1}])
    ks.record_reset(KillSwitchState.new(), approver="o", reason="r", at=datetime(2026, 3, 4, tzinfo=UTC), current_unit_value=1.0)


def test_no_live_imports():
    src = Path(ks.__file__).resolve().parents[2]
    code = (
        "import sys, firm.risk.kill_switch\n"
        "bad=[m for m in sys.modules if m in ('firm.live.engine','firm.api.app','firm.runtime')"
        " or m.startswith(('firm.live.','firm.api.','firm.agents.'))]\n"
        "print('BAD:'+','.join(sorted(bad)))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120, check=False)
    assert out.stdout.strip().endswith("BAD:"), out.stdout + out.stderr


def test_draft_config_loader_rejects_nulls(tmp_path):
    p = tmp_path / "k.yaml"
    p.write_text("charter_dd_bootstrap_p95: null\ntau: null\n")
    with pytest.raises(ValueError):
        ks.load_kill_switch_config(p)
    p.write_text("charter_dd_bootstrap_p95: 0.12\ntau: 0.04\n")
    assert ks.load_kill_switch_config(p) == replace(cfg(), stale_data_days=2)
    assert ks.load_kill_switch_config(Path(__file__).resolve().parents[1] / "config" / "kill_switch.yaml", allow_null=True) is None
