"""Fidelity monitor (P5-04). Synthetic data only; no network, no broker, no real state."""

from __future__ import annotations

import ast
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.costs.model import CostBreakdown
from firm.monitoring import fidelity as F
from firm.monitoring import shadow_loader as SL
from firm.monitoring.fidelity import (
    FidelityDay,
    Fill,
    GatesConfig,
    ReviewEvent,
    evaluate_fidelity,
    implementation_shortfall,
    record_missed_reviews,
    record_position_breaks,
)

REPO = Path(__file__).resolve().parents[1]
GATES = GatesConfig()


def _days(live, shadow, start=date(2026, 10, 5)):
    d = pd.bdate_range(start, periods=len(live))
    return [FidelityDay(x.date(), float(a), float(b)) for x, a, b in zip(d, live, shadow, strict=True)]


def _pair(n, rho, seed, vol=0.01):
    rng = np.random.default_rng(seed)
    shadow = rng.normal(0, vol, n)
    noise_sd = vol * math.sqrt(1 / rho**2 - 1)
    return shadow + rng.normal(0, noise_sd, n), shadow


def bd(commission=0.0, fees=0.0, half=0.0, impact=0.0):
    return CostBreakdown(commission, fees, half, impact, 0.0, commission + fees + half + impact)


# ------------------------------------------------------------------ gates config

def test_thresholds_read_from_gates_yaml():
    g = F.load_gates_config()
    assert (g.corr_min, g.te_max_fraction_of_tau, g.window_trading_days) == (0.95, 0.25, 63)
    assert (g.cost_max_multiple, g.cost_min_fills, g.cost_min_instruments) == (1.5, 30, 10)
    assert g.max_unreconciled_trading_days == 1
    assert g.zero_exposure_rule.startswith("exclude_from_corr_and_tracking_error")


# ------------------------------------------------------------------ corr / TE

def test_identical_series():
    x = np.random.default_rng(0).normal(0, 0.01, 60)
    r = evaluate_fidelity(_days(x, x), [], [], 0.10, GATES)
    assert r.corr == pytest.approx(1.0) and r.te_annualised == 0.0
    assert r.corr_pass is True and r.te_pass is True and not r.breach


def test_noise_gives_expected_te():
    shadow = np.random.default_rng(1).normal(0, 0.01, 63)
    noise = np.random.default_rng(2).normal(0, 0.002, 63)
    r = evaluate_fidelity(_days(shadow + noise, shadow), [], [], 0.10, GATES)
    assert r.te_annualised == pytest.approx(np.std(noise, ddof=1) * math.sqrt(252))
    assert r.te_vs_tau == pytest.approx(r.te_annualised / 0.10)


def test_threshold_edges_corr_and_te():
    live, shadow = _pair(63, 0.97, 3)
    r0 = evaluate_fidelity(_days(live, shadow), [], [], 0.10, GATES)
    # corr gate exactly at the observed corr passes; one ulp-ish above fails
    assert evaluate_fidelity(_days(live, shadow), [], [], 0.10, GatesConfig(corr_min=r0.corr)).corr_pass is True
    assert evaluate_fidelity(_days(live, shadow), [], [], 0.10, GatesConfig(corr_min=r0.corr + 1e-9)).corr_pass is False
    # TE <= 25% of tau: tau just above / below te/0.25
    tau_edge = r0.te_annualised / 0.25
    assert evaluate_fidelity(_days(live, shadow), [], [], tau_edge * (1 + 1e-9), GATES).te_pass is True
    assert evaluate_fidelity(_days(live, shadow), [], [], tau_edge * (1 - 1e-9), GATES).te_pass is False


def test_uses_trailing_window_only():
    live, shadow = _pair(63, 0.99, 4)
    junk_l, junk_s = _pair(40, 0.2, 5)
    r = evaluate_fidelity(_days(np.r_[junk_l, live], np.r_[junk_s, shadow]), [], [], 0.10, GATES)
    assert r.n_days == 63
    assert r.corr == pytest.approx(np.corrcoef(live, shadow)[0, 1])


def test_zero_exposure_days_excluded_and_counted():
    live, shadow = _pair(40, 0.99, 6)
    live = np.r_[live, np.zeros(10)]
    shadow = np.r_[shadow, np.zeros(10)]
    r = evaluate_fidelity(_days(live, shadow), [], [], 0.10, GATES)
    assert r.n_zero_exposure_excluded == 10 and r.n_days == 40
    assert r.corr == pytest.approx(np.corrcoef(live[:40], shadow[:40])[0, 1])


def test_all_flat_is_unevaluable_not_breach():
    r = evaluate_fidelity(_days(np.zeros(20), np.zeros(20)), [], [], 0.10, GATES)
    assert r.corr is None and r.te_annualised is None and not r.breach


def test_monthly_breach_false_alarm_rate():
    """True corr 0.98, tau 0.20 (TE gate 5% vs true TE ~3.2%): rate of two consecutive monthly breaches over a 6-month
    stretch (126 days), 400 seeded simulations. Stated bound: <= 2% (measured ~0%): the 63-day trailing window, with
    breaches counted only once the window is complete, keeps sample corr from straying below 0.95."""
    n_sims, hits = 400, 0
    for s in range(n_sims):
        live, shadow = _pair(126, 0.98, 1000 + s)
        flags = F.monthly_breaches(_days(live, shadow, date(2026, 10, 1)), 0.20, GATES)
        hits += F.consecutive_breach_months(flags)
    assert hits / n_sims <= 0.02, hits / n_sims


def test_partial_window_is_informational_not_a_breach():
    """Documents why: a 21-point sample corr (true 0.97, the ticket's example) falls below 0.95 in >5% of draws, so a partial window cannot breach."""
    n_sims = 400
    low = 0
    for s in range(n_sims):
        live, shadow = _pair(21, 0.97, 5000 + s)
        r = evaluate_fidelity(_days(live, shadow), [], [], 0.20, GATES)
        assert not r.window_complete and not r.breach
        low += r.corr_pass is False
    assert low / n_sims > 0.05


def test_consecutive_breach_months_logic():
    assert F.consecutive_breach_months([("a", True), ("b", True)])
    assert not F.consecutive_breach_months([("a", True), ("b", False), ("c", True)])


# ------------------------------------------------------------------ implementation shortfall

def _fill(side=1, px=100.0, qty=100.0, commission=1.0, fees=0.2, oid="o1", sym="SPY"):
    return Fill(oid, sym, side, qty, px, commission, fees)


def test_cost_ratio_exact_zero_slippage():
    f = _fill(px=100.0)
    m = bd(commission=1.0, fees=0.2, half=3.0, impact=2.0)
    sf = implementation_shortfall([f], {"o1": 100.0}, [m])
    notional = 100.0 * 100.0
    realised = (1.0 + 0.2) / notional * 1e4
    assert sf.loc[0, "realised_bps"] == realised
    assert sf.loc[0, "ratio"] == realised / (m.total / notional * 1e4)
    assert sf.loc[0, "ratio"] < 1


def test_shortfall_sign_for_sells_and_buys():
    buy = implementation_shortfall([_fill(1, 100.1)], {"o1": 100.0}, [bd(1.0)]).loc[0, "slippage_bps"]
    sell = implementation_shortfall([_fill(-1, 99.9)], {"o1": 100.0}, [bd(1.0)]).loc[0, "slippage_bps"]
    assert buy == pytest.approx(10.0) and sell == pytest.approx(10.0)


def test_cost_ratio_changes_if_commission_omitted():
    m = bd(1.2, 0.0, 3.0, 2.0)
    full = implementation_shortfall([_fill()], {"o1": 100.0}, [m]).loc[0, "ratio"]
    no_comm = implementation_shortfall([_fill(commission=0.0)], {"o1": 100.0}, [m]).loc[0, "ratio"]
    assert full != no_comm


def test_cost_ratio_changes_if_fees_omitted():
    m = bd(1.0, 0.2, 3.0, 2.0)
    full = implementation_shortfall([_fill()], {"o1": 100.0}, [m]).loc[0, "ratio"]
    no_fee = implementation_shortfall([_fill(fees=0.0)], {"o1": 100.0}, [m]).loc[0, "ratio"]
    assert full != no_fee


def test_modelled_includes_spread_and_impact():
    """Modelled cost is CostBreakdown.total: dropping half-spread or impact changes modelled_bps and hence the ratio."""
    f = [_fill(1, 100.05)]
    base = implementation_shortfall(f, {"o1": 100.0}, [bd(1.0, 0.2, 3.0, 2.0)]).loc[0]
    no_half = implementation_shortfall(f, {"o1": 100.0}, [bd(1.0, 0.2, 0.0, 2.0)]).loc[0]
    no_imp = implementation_shortfall(f, {"o1": 100.0}, [bd(1.0, 0.2, 3.0, 0.0)]).loc[0]
    assert base["modelled_bps"] == pytest.approx((1.0 + 0.2 + 3.0 + 2.0) / (100 * 100.05) * 1e4)
    assert no_half["ratio"] > base["ratio"] and no_imp["ratio"] > base["ratio"]


def test_real_cost_model_total_is_used():
    from firm.costs.model import cost, spec_from_config
    b = cost(spec_from_config("etf_alpaca"), 100.0, 100.0, 1e6, None, 0.01)
    assert b.total == pytest.approx(b.commission + b.exchange_fees + b.half_spread + b.impact + b.roll)
    assert b.half_spread > 0 and b.impact > 0


def test_arrival_mid_required():
    with pytest.raises(ValueError):
        implementation_shortfall([_fill()], {}, [bd(1.0)])


def _many_fills(n, n_sym, slip_px=0.0):
    fills, mids, mods = [], {}, []
    for i in range(n):
        oid = f"o{i}"
        fills.append(Fill(oid, f"S{i % n_sym}", 1, 100.0, 100.0 + slip_px, 1.0, 0.0))
        mids[oid] = 100.0
        mods.append(bd(1.0, 0.0, 2.0, 1.0))
    return fills, mids, mods


def test_g_paper_3_insufficient_below_30_fills_or_10_instruments():
    d = _days(*_pair(30, 0.99, 7))
    for n, k in ((29, 12), (40, 9)):
        fills, mids, mods = _many_fills(n, k)
        r = evaluate_fidelity(d, fills, mods, 0.1, GATES, arrival_mids=mids)
        assert not r.g_paper_3_evaluable and r.g_paper_3_status == "insufficient_data"
        assert r.cost_ratio_median is None and r.cost_pass is None and not r.breach
    fills, mids, mods = _many_fills(30, 10)
    r = evaluate_fidelity(d, fills, mods, 0.1, GATES, arrival_mids=mids)
    assert r.g_paper_3_evaluable and r.cost_ratio_median is not None


def test_g_paper_3_breach_when_cost_above_1_5x():
    d = _days(*_pair(30, 0.99, 8))
    fills, mids, mods = _many_fills(30, 10, slip_px=0.2)   # 20 bps slippage vs 4 bps modelled
    r = evaluate_fidelity(d, fills, mods, 0.1, GATES, arrival_mids=mids)
    assert r.cost_pass is False and r.breach
    assert r.cost_ratio_aggregate > 1.5 and r.cost_ratio_median > 1.5


# ------------------------------------------------------------------ position breaks

def _utc(y, m, d):
    return datetime(y, m, d, 21, 0, tzinfo=UTC)


def test_position_break_duration():
    disc = [{"type": "position_mismatch", "symbol": "SPY", "expected": 10, "actual": 9}]
    day1 = _utc(2026, 10, 5)  # Monday
    b1 = record_position_breaks(disc, [], day1)
    assert len(b1) == 1 and b1[0].closed_at is None and not b1[0].breach
    b2 = record_position_breaks(disc, b1, _utc(2026, 10, 6))          # one trading day: not yet a breach
    assert b2[0].trading_days_open == 1 and not b2[0].breach
    b3 = record_position_breaks(disc, b2, _utc(2026, 10, 7))          # two trading days: breach
    assert b3[0].trading_days_open == 2 and b3[0].breach and b3[0].closed_at is None
    b4 = record_position_breaks([], b3, _utc(2026, 10, 8))            # clears
    assert b4[0].closed_at is not None and b4[0].breach               # breach history kept
    b5 = record_position_breaks(disc, b4, _utc(2026, 10, 9))          # reappears: a NEW record
    assert len(b5) == 2 and b5[1].closed_at is None and not b5[1].breach


def test_position_break_over_weekend_counts_trading_days():
    disc = [{"type": "cash_mismatch", "symbol": ""}]
    b = record_position_breaks(disc, [], _utc(2026, 10, 9))           # Friday
    b = record_position_breaks(disc, b, _utc(2026, 10, 12))           # Monday: 1 trading day
    assert b[0].trading_days_open == 1 and not b[0].breach


def test_break_records_round_trip_json(tmp_path):
    b = record_position_breaks([{"type": "x", "symbol": "A"}], [], _utc(2026, 10, 5))
    p = tmp_path / "pb.json"
    from dataclasses import asdict
    F._write_json(p, [asdict(x) for x in b])
    assert F._load_breaks(p) == b


# ------------------------------------------------------------------ missed reviews

def test_missed_review_stuck_order_counts_unless_explained():
    d1, d2, d3, d4 = (date(2026, 10, 5) + timedelta(days=7 * i) for i in range(4))
    working = [{"order_id": "w1", "since": d2, "until": d3}]
    events = [ReviewEvent(d1, True), ReviewEvent(d3, False, "owner: broker outage, documented")]
    missed = record_missed_reviews([d1, d2, d3, d4], events, working)
    assert [(m.date, m.reason) for m in missed] == [(d2, "blocked_by_working_order"), (d4, "no_decision")]


def test_decided_review_with_working_order_is_not_missed():
    d = date(2026, 10, 5)
    assert record_missed_reviews([d], [ReviewEvent(d, True)], [{"since": d, "until": None}]) == []


# ------------------------------------------------------------------ determinism guard

def test_only_deterministic_candidates():
    ok = {"name": "c", "deterministic": True, "broker": "alpaca"}
    F.assert_deterministic(ok)
    for bad in ({**ok, "broker": "ibkr"}, {**ok, "llm_arm": "B"}, {**ok, "sleeves": [{"llm_enabled": True}]},
                {"name": "c", "broker": "alpaca"}):
        with pytest.raises(ValueError):
            F.assert_deterministic(bad)


# ------------------------------------------------------------------ shadow replay and daily job

def _write_snapshot(root: Path, cand: str, snap_id: str, n=30, start=date(2026, 10, 5)):
    d = SL.snapshot_dir(cand, snap_id, root)
    d.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(11)
    prices = pd.DataFrame({"AAA": 100 * np.cumprod(1 + rng.normal(0, 0.01, n)),
                           "BBB": 50 * np.cumprod(1 + rng.normal(0, 0.01, n))}, index=idx)
    targets = pd.DataFrame({"AAA": 10.0, "BBB": 20.0}, index=idx)
    targets.iloc[n // 2:, 0] = 5.0                                            # one rebalance
    for name, fr in {"prices": prices, "targets": targets, "adv": prices * 0 + 1e6,
                     "vol_pct": prices * 0 + 0.01}.items():
        fr.to_parquet(d / f"{name}.parquet")
    (d / "meta.json").write_text(json.dumps({"multipliers": {"AAA": 1.0, "BBB": 1.0}, "initial_capital": 100000.0}))
    return prices, targets


CAND = {"name": "cand1", "deterministic": True, "broker": "alpaca",
        "cost_specs": {"AAA": "etf_alpaca", "BBB": "etf_alpaca"}}


def test_shadow_replay_post_seal_fixture_without_guard(tmp_path):
    """Fixture dates are after the 2026-10-01 seal; no seal guard is installed and data_access is not involved."""
    import sys
    _write_snapshot(tmp_path, "cand1", "snap1")
    before = set(sys.modules)
    s = F.shadow_returns(CAND, "snap1", date(2026, 10, 5), date(2026, 11, 30), snapshot_root=tmp_path)
    assert len(s) == 30 and s.index.min() >= pd.Timestamp("2026-10-05")
    assert "firm.research.data_access" not in (set(sys.modules) - before)
    # sealed-tree location of the default snapshot root
    assert SL.SEALED_DIR_NAME in SL.snapshot_dir("cand1", "snap1").parts
    assert SL.snapshot_dir("cand1", "snap1").parts[-6:-3] == ("research", SL.SEALED_DIR_NAME, "candidates")


def test_shadow_replay_hand_computed_first_trade_cost(tmp_path):
    """Day 0 targets are filled at bar 1 (lag 1); the cost-model charge for that bar reduces the return exactly."""
    prices, _ = _write_snapshot(tmp_path, "cand1", "snap1")
    s = F.shadow_returns(CAND, "snap1", date(2026, 10, 5), date(2026, 11, 30), snapshot_root=tmp_path)
    from firm.costs.model import cost, spec_from_config
    sp = spec_from_config("etf_alpaca")
    c = sum(cost(sp, q, float(prices[k].iloc[1]), 1e6, None, 0.01).total for k, q in (("AAA", 10.0), ("BBB", 20.0)))
    assert s.iloc[0] == 0.0
    assert s.iloc[1] == pytest.approx(-c / 100000.0)


def test_shadow_replay_does_not_touch_research_ledger(tmp_path, monkeypatch):
    from firm.research import ledger as L
    sentinel = tmp_path / "research_ledger"
    sentinel.mkdir()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(sentinel))
    _write_snapshot(tmp_path, "cand1", "snap1")
    F.shadow_returns(CAND, "snap1", date(2026, 10, 5), date(2026, 11, 30), snapshot_root=tmp_path)
    assert list(sentinel.iterdir()) == []
    assert __import__("os").environ[L.LEDGER_ROOT_ENV] == str(sentinel)


def test_shadow_loader_rejects_path_tricks(tmp_path):
    for bad in ("../x", "a/b", ""):
        with pytest.raises(ValueError):
            SL.snapshot_dir(bad, "s", tmp_path)


def _fetch(tmp_path, live=None, **extra):
    _write_snapshot(tmp_path / "snaps", "cand1", "snap1")
    shadow = F.shadow_returns(CAND, "snap1", date(2026, 10, 5), date(2026, 11, 30), snapshot_root=tmp_path / "snaps")
    payload = {"candidate_cfg": CAND, "snapshot_id": "snap1", "tau": 0.10, "start": date(2026, 10, 5),
               "end": date(2026, 11, 30), "live_returns": shadow if live is None else live,
               "fills": [], "arrival_mids": {}, "modelled": [], "discrepancies": [], "scheduled_reviews": [],
               "review_events": [], "working_orders": [], **extra}
    return lambda: payload


def test_daily_job_runs_and_alerts_nothing_when_identical(tmp_path):
    alerts = []
    state = tmp_path / "state"
    r = F.daily_job("cand1", state, fetch_live=_fetch(tmp_path), notify=lambda **k: alerts.append(k),
                    snapshot_root=tmp_path / "snaps", now=_utc(2026, 11, 2))
    assert r["report"]["corr"] == pytest.approx(1.0) and r["report"]["te_annualised"] == pytest.approx(0.0, abs=1e-12)
    assert alerts == [] and not r["breach"]


def test_daily_job_alerts_on_breach_break_and_missed_review(tmp_path):
    alerts = []
    fl = _fetch(tmp_path)
    shadow = fl()["live_returns"]
    live = shadow + np.random.default_rng(5).normal(0, 0.02, len(shadow))
    fl = _fetch(tmp_path, live=live, discrepancies=[{"type": "position_mismatch", "symbol": "AAA"}],
                scheduled_reviews=[date(2026, 10, 12)])
    state = tmp_path / "state"
    kw = {"fetch_live": fl, "notify": lambda **k: alerts.append(k["kind"]), "snapshot_root": tmp_path / "snaps",
          "gates": GatesConfig(window_trading_days=20)}
    F.daily_job("cand1", state, now=_utc(2026, 11, 2), **kw)
    F.daily_job("cand1", state, now=_utc(2026, 11, 3), **kw)
    F.daily_job("cand1", state, now=_utc(2026, 11, 4), **kw)          # open 2 trading days -> overdue
    assert {"fidelity_breach", "missed_review", "position_break_overdue"} <= set(alerts)
    assert alerts.count("position_break_overdue") == 1
    st = json.loads((state / "state.json").read_text())
    assert st["report"]["position_breaks"][0]["breach"] is True


def test_daily_state_paths(tmp_path):
    state, root = tmp_path / "state", tmp_path / "sealed_root"
    r = F.daily_job("cand1", state, fetch_live=_fetch(tmp_path), notify=lambda **k: None,
                    snapshot_root=tmp_path / "snaps", now=_utc(2026, 11, 2))
    produced = {p for p in tmp_path.rglob("*") if p.is_file() and "snaps" not in p.parts}
    assert produced and all(state in p.parents for p in produced), produced
    assert not root.exists()
    # in-repo state dirs other than the gitignored monitor root are refused
    with pytest.raises(ValueError):
        F.daily_job("cand1", REPO / "docs" / "x", fetch_live=_fetch(tmp_path), notify=lambda **k: None)
    # owner-run export writes only under <root>/<candidate>/
    dest = F.export_report("cand1", state, root)
    assert dest == SL.candidate_dir("cand1", root) / "reports" / "2026-11-02.json"
    assert [p for p in root.rglob("*") if p.is_file()] == [dest]
    assert json.loads(dest.read_text())["candidate"] == r["candidate"]


def test_daily_job_rejects_ibkr_candidate(tmp_path):
    fl = _fetch(tmp_path)
    p = fl()
    bad = {**p, "candidate_cfg": {**CAND, "broker": "ibkr"}}
    with pytest.raises(ValueError):
        F.daily_job("cand1", tmp_path / "s", fetch_live=lambda: bad, notify=lambda **k: None, snapshot_root=tmp_path / "snaps")


def test_no_live_side_effects(tmp_path, monkeypatch):
    """Broker, engine and socket write paths all raise; the job still completes."""
    import socket

    from firm.brokers import alpaca, base, ibkr

    def boom(*a, **k):
        raise AssertionError("live side effect attempted")

    for mod in (base, alpaca, ibkr):
        for cls in (c for c in vars(mod).values() if isinstance(c, type)):
            for meth in ("submit_order", "cancel_order"):
                if meth in vars(cls):
                    monkeypatch.setattr(cls, meth, boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    F.daily_job("cand1", tmp_path / "state", fetch_live=_fetch(tmp_path), notify=lambda **k: None,
                snapshot_root=tmp_path / "snaps", now=_utc(2026, 11, 2))


# ------------------------------------------------------------------ isolation

def test_shadow_loader_not_importable_from_research_or_live():
    for sub in ("research", "live", "api"):
        for f in (REPO / "src" / "firm" / sub).rglob("*.py") if sub != "api" else (REPO / "src" / "firm" / "api").rglob("*.py"):
            tree = ast.parse(f.read_text())
            for n in ast.walk(tree):
                names = []
                if isinstance(n, ast.ImportFrom):
                    names = [n.module or ""] + [(n.module or "") + "." + a.name for a in n.names]
                elif isinstance(n, ast.Import):
                    names = [a.name for a in n.names]
                assert not any(x.startswith("firm.monitoring") for x in names), f


def test_fidelity_modules_do_not_use_data_access():
    for name in ("fidelity", "shadow_loader"):
        src = (REPO / "src" / "firm" / "monitoring" / f"{name}.py").read_text()
        assert "import data_access" not in src and "firm.research.data_access" not in src.replace("``firm.research.data_access``", "")


def test_cli_export_and_report(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("daily_reconcile", REPO / "scripts" / "daily_reconcile.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    state = tmp_path / "state"
    F.daily_job("cand1", state, fetch_live=_fetch(tmp_path), notify=lambda **k: None, snapshot_root=tmp_path / "snaps",
                now=_utc(2026, 11, 2))
    assert m.main(["report", "--candidate", "cand1", "--state-dir", str(state)]) == 0
    assert m.main(["export", "--candidate", "cand1", "--state-dir", str(state), "--out-root", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "cand1" / "reports" / "2026-11-02.json").is_file()
