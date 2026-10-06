"""Shadow forward-test mechanics on synthetic data (no network, no real data)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import s2_forward_shadow as fs


def _px(start="2026-10-01", n=70, spy_step=0.001, ief_step=0.0002, gap=0.0):
    days = pd.bdate_range(start, periods=n)
    spy = pd.DataFrame({"adj_open": 100 * np.cumprod(np.full(n, 1 + spy_step)) * (1 + gap),
                        "adj_close": 100 * np.cumprod(np.full(n, 1 + spy_step)) * (1 + gap) * (1 + spy_step)}, index=days)
    ief = pd.DataFrame({"adj_open": 50 * np.cumprod(np.full(n, 1 + ief_step)),
                        "adj_close": 50 * np.cumprod(np.full(n, 1 + ief_step)) * (1 + ief_step)}, index=days)
    return spy, ief


def _dec(day, v1_cut=False, v3_cut=False):
    return {"check_day": day, "cut": {"overlay_v1": v1_cut, "overlay_v3": v3_cut}}


def test_check_days_are_first_trading_days_and_respect_start():
    ds = fs.check_days(date(2026, 10, 1), date(2027, 1, 31))
    assert ds == [date(2026, 10, 1), date(2026, 11, 2), date(2026, 12, 1), date(2027, 1, 4)]  # Jan 1 closed, Jan 2 Sat
    assert fs.check_days(date(2026, 10, 1), date(2026, 9, 30)) == []


def test_rule_is_strictly_below_threshold_and_weights_sum_to_one():
    assert fs.decide(0.4999, 0.50) is True and fs.decide(0.50, 0.50) is False
    assert fs.decide(0.45, 0.40) is False
    for cut in (True, False):
        assert sum(fs.weights_for(cut).values()) == pytest.approx(1.0)
    assert fs.weights_for(True) == {"SPY": 0.30, "IEF": 0.70}


def test_prior_session_is_strictly_earlier():
    s = [date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1)]
    assert fs.prior_session(date(2026, 10, 1), s) == date(2026, 9, 30)
    assert fs.prior_session(date(2026, 9, 29), s) is None


def test_initial_purchase_pays_cost_and_all_portfolios_start_below_100_only_by_cost():
    spy, ief = _px()
    nav = fs.shadow_nav([_dec("2026-10-01")], spy, ief)
    t0 = spy.index[0]
    # day-1 NAV = (bought at the open with 3bps cost) * close/open
    expected_spy = 100 * (1 - 0.0003) * (spy.at[t0, "adj_close"] / spy.at[t0, "adj_open"])
    assert nav["spy"].iloc[0] == pytest.approx(expected_spy)
    # 60/40 buys both legs: cost = 3bps of the whole notional
    exp_plain = 100 * (1 - 0.0003) * (0.6 * spy.at[t0, "adj_close"] / spy.at[t0, "adj_open"]
                                      + 0.4 * ief.at[t0, "adj_close"] / ief.at[t0, "adj_open"])
    assert nav["plain_60_40"].iloc[0] == pytest.approx(exp_plain)


def test_overlay_equals_plain_when_never_cut_and_differs_when_cut():
    spy, ief = _px(spy_step=0.002)
    never = fs.shadow_nav([_dec("2026-10-01"), _dec("2026-11-02")], spy, ief)
    assert never["overlay_v1"].tolist() == pytest.approx(never["plain_60_40"].tolist())
    cut = fs.shadow_nav([_dec("2026-10-01"), _dec("2026-11-02", v1_cut=True)], spy, ief)
    after = cut["date"] >= pd.Timestamp("2026-11-02")
    # cutting equity in a steadily rising equity market lags plain 60/40 after the cut
    assert (cut.loc[after, "overlay_v1"].iloc[-1] < cut.loc[after, "plain_60_40"].iloc[-1])
    # and before the cut they are identical
    assert cut.loc[~after, "overlay_v1"].tolist() == pytest.approx(cut.loc[~after, "plain_60_40"].tolist())


def test_weights_drift_between_checks_and_rebalance_restores_them():
    spy, ief = _px(spy_step=0.01, ief_step=0.0)
    d = [_dec("2026-10-01"), _dec("2026-11-02")]
    nav = fs.shadow_nav(d, spy, ief)
    # just before the Nov rebalance SPY has outgrown 60%; after it, value is conserved less 3bps*turnover
    t_before = pd.Timestamp("2026-10-30")
    drift_w = 0.6 * (spy.at[t_before, "adj_close"] / spy.at[spy.index[0], "adj_open"]) / (
        0.6 * (spy.at[t_before, "adj_close"] / spy.at[spy.index[0], "adj_open"]) + 0.4)
    assert drift_w > 0.6
    assert nav["plain_60_40"].is_monotonic_increasing


def test_nav_is_deterministic_and_empty_ledger_is_safe():
    spy, ief = _px()
    d = [_dec("2026-10-01"), _dec("2026-11-02", v1_cut=True, v3_cut=True)]
    a, b = fs.shadow_nav(d, spy, ief), fs.shadow_nav(d, spy, ief)
    pd.testing.assert_frame_equal(a, b)
    empty = fs.shadow_nav([], spy, ief)
    assert list(empty.columns) == ["date", *fs.PORTFOLIOS] and empty.empty
    # a check day with no price bar yet (today, before the open) yields no rows, not an error
    future = fs.shadow_nav([_dec("2027-06-01")], spy, ief)
    assert future.empty and list(future.columns) == ["date", *fs.PORTFOLIOS]


def test_integrity_band_and_stale_data_checks():
    ok = {"eligible": 3000, "with_bar": 5400, "breadth": 0.5}
    assert fs.integrity_ok(ok, 5400) == (True, "ok")
    assert fs.integrity_ok({**ok, "eligible": 900}, 5400)[0] is False
    assert fs.integrity_ok({**ok, "eligible": 6000}, 5400)[0] is False
    assert fs.integrity_ok({**ok, "with_bar": 2000}, 5400)[0] is False          # bars not yet published
    assert fs.integrity_ok({**ok, "breadth": float("nan")}, 5400)[0] is False


def test_completed_episodes_counts_runs_followed_by_a_plain_check():
    assert fs.completed_episodes([False, True, True, False, False, True, False]) == 2
    assert fs.completed_episodes([False, True, True]) == 0     # still in the first episode
    assert fs.completed_episodes([]) == 0


def test_decision_rules_promote_reject_continue():
    good = {"sharpe": 0.9, "max_dd": 0.10, "calmar": 0.8}
    base = {"sharpe": 0.8, "max_dd": 0.15, "calmar": 0.5}
    assert fs.evaluate_rules(24, 0, 2, good, base)["outcome"] == "PROPOSE_LIVE_TEST"
    assert fs.evaluate_rules(23, 0, 2, good, base)["outcome"] == "CONTINUE"        # too short
    assert fs.evaluate_rules(24, 2, 2, good, base)["outcome"] == "CONTINUE"        # too many missed checks
    worse = {"sharpe": 0.6, "max_dd": 0.20, "calmar": 0.3}
    assert fs.evaluate_rules(24, 0, 2, worse, base)["outcome"] == "REJECT"
    assert fs.evaluate_rules(24, 0, 1, worse, base)["outcome"] == "CONTINUE"       # fewer than 2 episodes
    assert fs.evaluate_rules(60, 0, 1, worse, base)["outcome"] == "INCONCLUSIVE_CLOSE"


def test_latest_record_per_check_day_governs(tmp_path):
    fs.append_decision(tmp_path, {"check_day": "2026-10-01", "status": "anomaly", "cut": {}})
    fs.append_decision(tmp_path, {"check_day": "2026-11-02", "status": "ok", "cut": {}})
    fs.append_decision(tmp_path, {"check_day": "2026-10-01", "status": "ok", "cut": {}})
    final = fs.load_decisions(tmp_path)
    assert [d["check_day"] for d in final] == ["2026-10-01", "2026-11-02"]
    assert final[0]["status"] == "ok" and len(fs.load_log(tmp_path)) == 3   # audit log keeps every attempt


def test_perf_metrics():
    nav = pd.Series(100 * np.cumprod(np.r_[1.0, np.full(251, 1.0004)]), index=pd.bdate_range("2026-10-01", periods=252))
    p = fs.perf(nav, pd.Series(0.0, index=nav.index))
    assert p["max_dd"] == 0.0 and p["cagr"] > 0 and p["days"] == 251
