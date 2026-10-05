"""P5-01: lifecycle state machine, G-PAPER evaluator, gated sleeve and the H5 dry-run Allocator test (synthetic only)."""

from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from firm.allocation.allocator import Allocator
from firm.allocation.btc_trend import BtcTrendSleeve
from firm.allocation.sleeves import StaticSleeve
from firm.lifecycle import decommission as dc
from firm.lifecycle import gates as gt
from firm.lifecycle import state_machine as sm
from firm.lifecycle.gated_sleeve import LifecycleGatedSleeve
from firm.lifecycle.state_machine import (
    ApprovalRequiredError,
    FamilyRecord,
    GateNotMetError,
    Registry,
    State,
    TransitionError,
)

REPO = Path(__file__).resolve().parents[1]
GATES = gt.load_gates_config(REPO / "config" / "gates.yaml")
NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)
FAM = "core_v1"


def now_fn():
    return NOW


def rec(state: State, *, age_days: float = 400, prior: State | None = None, kind: str = "candidate", **kw):
    base = {
        "family": FAM,
        "state": state,
        "prior_state": prior,
        "entered_at_utc": NOW - timedelta(days=age_days),
        "code_commit": "abc123",
        "config_hash": "h1",
        "kind": kind,
    }
    base.update(kw)
    return FamilyRecord(**base)


def registry(tmp_path: Path, *recs: FamilyRecord) -> Registry:
    d = tmp_path / "approvals"
    d.mkdir(exist_ok=True)
    return Registry(families=tuple(recs), approvals_dir=d)


def approve(reg: Registry, family: str, src: str, dst: str, approver="avi.owner", sig="ssh:SHA256:abc"):
    doc = {
        "family": family,
        "from_state": src,
        "to_state": dst,
        "approver": approver,
        "approved_at_utc": NOW.isoformat(),
        "signature_ref": sig,
    }
    (reg.approvals_dir / f"{family}__{src}__{dst}.yaml").write_text(yaml.safe_dump(doc))


def evidence(tmp_path: Path, src: State, dst: State, **over) -> Path:
    doc = {
        "family": FAM,
        "edge": f"{src.value}->{dst.value}",
        "prereg_hash": "h1",
        "tier": "A",
        "trial_ids": ["t1", "t2"],
        "code_commit": "abc123",
        "config_hash": "h1",
        "gpaper_overall": "pass",
        "live_step_criteria_hold": True,
    }
    doc.update(over)
    p = tmp_path / f"ev_{src.value}_{dst.value}.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


CHAIN = list(sm._CHAIN)
EDGES = list(itertools.pairwise(CHAIN))


# ---- structure -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("src", "dst"), [(State.IDEA, State.PAPER), (State.RESEARCH, State.LIVE_FULL)])
def test_illegal_transitions_raise(tmp_path, src, dst):
    reg = registry(tmp_path, rec(src))
    assert not sm.allowed(src, dst)
    with pytest.raises(TransitionError):
        sm.transition(FAM, dst, evidence(tmp_path, src, dst), "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)


@pytest.mark.parametrize(("src", "dst"), EDGES)
def test_forward_edge_without_approval_raises_approval_required(tmp_path, src, dst):
    reg = registry(tmp_path, rec(src, age_days=400))
    ev = evidence(tmp_path, src, dst)
    with pytest.raises(ApprovalRequiredError):
        sm.transition(FAM, dst, ev, "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)


@pytest.mark.parametrize(("src", "dst"), EDGES)
def test_forward_edge_with_evidence_and_approval_succeeds(tmp_path, src, dst):
    reg = registry(tmp_path, rec(src, age_days=400))
    approve(reg, FAM, src.value, dst.value)
    out = sm.transition(
        FAM, dst, evidence(tmp_path, src, dst), "avi.owner", registry=reg, gates=GATES, now_fn=now_fn
    )
    assert out.get(FAM).state is dst and out.history[-1]["identity_status"] == "unverifiable"
    assert reg.get(FAM).state is src  # input registry untouched


def test_failed_evidence_raises_gate_not_met_not_approval(tmp_path):
    src, dst = State.RESEARCH, State.RESEARCH_PASS
    reg = registry(tmp_path, rec(src))
    approve(reg, FAM, src.value, dst.value)  # approval present: only the evidence fails
    for bad in ({"tier": "C"}, {"tier": "D"}, {"trial_ids": []}, {"prereg_hash": "other"}):
        with pytest.raises(GateNotMetError):
            sm.transition(
                FAM, dst, evidence(tmp_path, src, dst, **bad), "avi.owner", registry=reg, gates=GATES, now_fn=now_fn
            )
    with pytest.raises(GateNotMetError):
        sm.transition(FAM, dst, None, "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)


def test_non_candidate_has_no_promotion_path(tmp_path):
    src, dst = State.PAPER, State.LIVE_STEP_1
    reg = registry(tmp_path, rec(src, kind="forward_test"))
    approve(reg, FAM, src.value, dst.value)
    with pytest.raises(TransitionError):
        sm.transition(FAM, dst, evidence(tmp_path, src, dst), "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)


def test_embargo_shorter_than_14_days_raises(tmp_path):
    src, dst = State.EMBARGO, State.PAPER
    reg = registry(tmp_path, rec(src, age_days=13))
    approve(reg, FAM, src.value, dst.value)
    with pytest.raises(GateNotMetError):
        sm.transition(FAM, dst, evidence(tmp_path, src, dst), "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)
    ok = registry(tmp_path, rec(src, age_days=14))
    out = sm.transition(
        FAM, dst, evidence(tmp_path, src, dst), "avi.owner", registry=ok, gates=GATES, now_fn=now_fn
    )
    assert out.get(FAM).state is State.PAPER


@pytest.mark.parametrize("changed", [{"code_commit": "zzz"}, {"config_hash": "h2"}])
def test_embargo_with_changed_code_commit_or_config_hash_raises(tmp_path, changed):
    src, dst = State.EMBARGO, State.PAPER
    reg = registry(tmp_path, rec(src, age_days=30))
    approve(reg, FAM, src.value, dst.value)
    with pytest.raises(GateNotMetError):
        sm.transition(
            FAM, dst, evidence(tmp_path, src, dst, **changed), "avi.owner", registry=reg, gates=GATES, now_fn=now_fn
        )


def test_embargo_entry_freezes_code_and_config(tmp_path):
    src, dst = State.RESEARCH_PASS, State.EMBARGO
    reg = registry(tmp_path, rec(src))
    approve(reg, FAM, src.value, dst.value)
    out = sm.transition(
        FAM,
        dst,
        evidence(tmp_path, src, dst, code_commit="new9", config_hash="h9"),
        "avi.owner",
        registry=reg,
        gates=GATES,
        now_fn=now_fn,
    )
    assert (out.get(FAM).code_commit, out.get(FAM).config_hash) == ("new9", "h9")
    assert out.get(FAM).entered_at_utc == NOW


@pytest.mark.parametrize(("src", "dst"), [(State.LIVE_STEP_1, State.LIVE_STEP_2), (State.LIVE_STEP_2, State.LIVE_FULL)])
def test_live_step_dwell_under_3_months_raises(tmp_path, src, dst):
    approve_reg = registry(tmp_path, rec(src, age_days=80))  # < 3 calendar months before 2026-10-05
    approve(approve_reg, FAM, src.value, dst.value)
    with pytest.raises(GateNotMetError):
        sm.transition(
            FAM, dst, evidence(tmp_path, src, dst), "avi.owner", registry=approve_reg, gates=GATES, now_fn=now_fn
        )
    with pytest.raises(GateNotMetError):
        long_reg = registry(tmp_path, rec(src, age_days=200))
        approve(long_reg, FAM, src.value, dst.value)
        sm.transition(
            FAM,
            dst,
            evidence(tmp_path, src, dst, live_step_criteria_hold=False),
            "avi.owner",
            registry=long_reg,
            gates=GATES,
            now_fn=now_fn,
        )


def test_probation_exit_requires_approval(tmp_path):
    reg = registry(tmp_path, rec(State.PROBATION, prior=State.PAPER))
    with pytest.raises(ApprovalRequiredError):
        sm.transition(FAM, State.PAPER, None, "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)
    with pytest.raises(TransitionError):  # only back to the prior state
        sm.transition(FAM, State.LIVE_FULL, None, "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)
    approve(reg, FAM, "PROBATION", "PAPER")
    out = sm.transition(FAM, State.PAPER, None, "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)
    assert out.get(FAM).state is State.PAPER and out.get(FAM).prior_state is None
    assert out.get(FAM).entered_at_utc == NOW


def test_entering_probation_needs_no_approval_and_records_prior(tmp_path):
    reg = registry(tmp_path, rec(State.PAPER))
    out = sm.transition(FAM, State.PROBATION, None, None, registry=reg, gates=GATES, now_fn=now_fn)
    assert out.get(FAM).state is State.PROBATION and out.get(FAM).prior_state is State.PAPER


def test_approval_by_agent_identity_rejected(tmp_path):
    src, dst = State.RESEARCH, State.RESEARCH_PASS
    for who in ("claude", "research-agent", "ci-bot"):
        reg = registry(tmp_path, rec(src))
        approve(reg, FAM, src.value, dst.value, approver=who)
        with pytest.raises(ApprovalRequiredError):
            sm.transition(FAM, dst, evidence(tmp_path, src, dst), who, registry=reg, gates=GATES, now_fn=now_fn)
    assert sm.identity_status("claude", "sig") == "rejected"


def test_identity_check_unverifiable_without_od06(tmp_path):
    assert sm.identity_status("avi.owner", "ssh:sig") == "unverifiable"  # never "pass" without OD-06 option b
    assert sm.identity_status("avi.owner", "ssh:sig", distinct_agent_identity_configured=True) == "pass"
    src, dst = State.RESEARCH, State.RESEARCH_PASS
    reg = registry(tmp_path, rec(src))
    approve(reg, FAM, src.value, dst.value, sig=None)  # unverifiable AND unsigned -> refused
    with pytest.raises(ApprovalRequiredError):
        sm.transition(FAM, dst, evidence(tmp_path, src, dst), "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)


def test_paper_state_weight_zero_on_live_account():
    r = rec(State.PAPER)
    assert sm.target_weight_allowed(r, account_type="paper", zero_weight=False)
    assert not sm.target_weight_allowed(r, account_type="live", zero_weight=False)
    for s in (State.LIVE_STEP_1, State.LIVE_STEP_2, State.LIVE_FULL):
        assert sm.target_weight_allowed(rec(s), account_type="live", zero_weight=False)
    for s in (State.IDEA, State.SPEC, State.PREREG, State.RESEARCH, State.RESEARCH_PASS, State.EMBARGO):
        assert not sm.target_weight_allowed(rec(s), account_type="paper", zero_weight=False)
    for s in (State.DECOMMISSIONED, State.ARCHIVED):
        assert not sm.target_weight_allowed(rec(s), account_type="paper", zero_weight=False)
    assert not sm.target_weight_allowed(rec(State.LIVE_FULL), account_type="live", zero_weight=True)


def test_registry_round_trip(tmp_path):
    reg = registry(tmp_path, rec(State.PAPER), rec(State.PROBATION, prior=State.PAPER, family="x"))
    p = tmp_path / "research" / "lifecycle" / "registry.yaml"
    p.parent.mkdir(parents=True)
    sm.save_registry(reg, p)
    back = sm.load_registry(p)
    assert back.families == reg.families and back.approvals_dir == reg.approvals_dir


def test_seed_registry_states():
    reg = sm.load_registry(REPO / "research" / "lifecycle" / "registry.yaml")
    alloc, s2 = reg.get("alpaca_allocation_92_8"), reg.get("s2_shadow_ledger")
    assert alloc.kind == "forward_test" and s2.kind == "shadow_ledger"
    assert all(r.kind != "candidate" for r in reg.families)
    for r in (alloc, s2):  # neither can be promoted
        nxt = sm._CHAIN[sm._CHAIN.index(r.state) + 1]
        with pytest.raises(TransitionError):
            sm.transition(r.family, nxt, None, "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)
    from firm.strategies import registry as strat

    legacy_live = {r.family for r in reg.families if r.state is State.PAPER and r.kind == "legacy_pipeline"}
    assert legacy_live == {n for n, s in strat._STATUS.items() if s is strat.StrategyStatus.LEGACY_LIVE}


# ---- G-PAPER ---------------------------------------------------------------------------------------------------------
TAU = 0.09


def events(n=30, start=date(2026, 1, 5), **kw):
    return [gt.ReviewEvent(start + timedelta(weeks=i), **kw) for i in range(n)]


def fills(n=40, n_inst=12, ratio=1.0):
    return [gt.Fill(date(2026, 3, 1), f"S{i % n_inst}", ratio * 1.0, 1.0) for i in range(n)]


def daily(n=63, noise=0.0001, seed=3, exposure=1.0):
    rng = np.random.default_rng(seed)
    model = rng.normal(0, 0.01, n)
    paper = model + rng.normal(0, noise, n)
    return pd.DataFrame(
        {"ret_paper": paper, "ret_model": model, "exposure": exposure}, index=pd.bdate_range("2026-06-01", periods=n)
    )


DRILLS = dict.fromkeys(gt.REQUIRED_DRILLS, True)


def run_gpaper(gates=GATES, **over):
    args = {"events": events(), "fills": fills(), "daily": daily(), "tau": TAU, "fault_drills": DRILLS}
    args.update(over)
    return gt.evaluate_gpaper(
        args.pop("events"), args.pop("fills"), args.pop("daily"), gates, **args
    )


def test_gpaper_all_pass():
    r = run_gpaper()
    assert r.overall == "pass", r


def test_gpaper2_corr_te_thresholds():
    assert run_gpaper(daily=daily(noise=0.0001)).criteria["2_fidelity"] == "pass"
    noisy = run_gpaper(daily=daily(noise=0.01))  # corr ~0.7
    assert noisy.criteria["2_fidelity"] == "fail" and noisy.overall == "fail"
    # high correlation but TE above 25% of a tiny tau
    assert run_gpaper(daily=daily(noise=0.0005), tau=0.005).criteria["2_fidelity"] == "fail"


def test_gpaper2_zero_exposure_days_excluded_and_reported():
    d = daily()
    d.iloc[:10, d.columns.get_loc("exposure")] = 0.0
    d.iloc[:10, d.columns.get_loc("ret_paper")] = 0.5  # would wreck the correlation if included
    r = run_gpaper(daily=d)
    assert r.criteria["2_fidelity"] == "pass" and r.details["2"]["zero_exposure_days_excluded"] == 10


def test_gpaper4_break_over_1_day_or_unexplained_missed_review_fails():
    # 2026-03-02 (Mon): resolved next trading day is OK, two trading days later fails
    ok = run_gpaper(position_breaks=[gt.PositionBreak(date(2026, 3, 2), date(2026, 3, 3))])
    assert ok.criteria["4_breaks_reviews"] == "pass"
    bad = run_gpaper(position_breaks=[gt.PositionBreak(date(2026, 3, 2), date(2026, 3, 4))])
    assert bad.criteria["4_breaks_reviews"] == "fail"
    open_break = run_gpaper(position_breaks=[gt.PositionBreak(date(2026, 3, 2), None)])
    assert open_break.criteria["4_breaks_reviews"] == "fail"
    evs = events()
    evs[3] = replace(evs[3], missed=True)
    assert run_gpaper(events=evs).criteria["4_breaks_reviews"] == "fail"
    evs[3] = replace(evs[3], missed=True, outage_documented=True)
    assert run_gpaper(events=evs).criteria["4_breaks_reviews"] == "pass"


def test_gpaper5_missing_fault_drill_record_fails():
    assert run_gpaper(fault_drills=None).criteria["5_drills"] == "fail"
    partial = {**DRILLS, "hard": False}
    assert run_gpaper(fault_drills=partial).criteria["5_drills"] == "fail"
    assert run_gpaper(fault_drills={k: v for k, v in DRILLS.items() if k != "halt"}).overall == "fail"


def test_gpaper_needs_26_events_and_6_months():
    assert run_gpaper(events=events(25)).criteria["1_period"] == "insufficient_data"
    # 30 weekly events but spanning < 6 calendar months (daily-ish cadence): 26 events in 26 days
    dense = [gt.ReviewEvent(date(2026, 1, 5) + timedelta(days=i)) for i in range(30)]
    assert run_gpaper(events=dense).criteria["1_period"] == "insufficient_data"
    # 26 weekly events span only 25 weeks (< 6 calendar months): count is met, period is not
    assert run_gpaper(events=events(26)).criteria["1_period"] == "insufficient_data"
    assert run_gpaper(events=events(27)).criteria["1_period"] == "pass"
    missed = [replace(e, missed=True) if i % 2 else e for i, e in enumerate(events(40))]
    assert run_gpaper(events=missed).criteria["1_period"] == "insufficient_data"  # missed reviews do not count


def test_gpaper3_insufficient_until_30_fills_10_instruments():
    assert run_gpaper(fills=fills(29)).criteria["3_cost"] == "insufficient_data"
    assert run_gpaper(fills=fills(40, n_inst=9)).criteria["3_cost"] == "insufficient_data"
    assert run_gpaper(fills=fills(30, n_inst=10)).criteria["3_cost"] == "pass"
    assert run_gpaper(fills=fills(40, ratio=1.6)).criteria["3_cost"] == "fail"
    # aggregate OK but median per-trade over the limit
    mixed = fills(40, ratio=1.6)[:25] + [gt.Fill(date(2026, 3, 1), "S1", 0.5, 1.0)] * 15
    r = run_gpaper(fills=mixed)
    assert r.details["3"]["median_ratio"] > 1.5 and r.criteria["3_cost"] == "fail"
    assert run_gpaper(fills=fills(40, ratio=1.6)).overall == "fail"
    assert run_gpaper(fills=fills(29)).overall in {"insufficient_data", "fail"}


def test_gpaper_thresholds_from_gates_yaml(tmp_path):
    raw = yaml.safe_load((REPO / "config" / "gates.yaml").read_text())
    raw["g_paper"]["min_review_events"] = 40
    raw["g_paper"]["realised_cost_max_multiple_of_modelled"] = 3.0
    p = tmp_path / "gates.yaml"
    p.write_text(yaml.safe_dump(raw))
    alt = gt.load_gates_config(p)
    assert run_gpaper(events=events(30)).criteria["1_period"] == "pass"
    assert run_gpaper(gates=alt, events=events(30)).criteria["1_period"] == "insufficient_data"
    assert run_gpaper(fills=fills(40, ratio=2.0)).criteria["3_cost"] == "fail"
    assert run_gpaper(gates=alt, fills=fills(40, ratio=2.0)).criteria["3_cost"] == "pass"
    # embargo and live-step dwell also come from the file
    raw["g_paper"]["embargo_weeks_before_start"] = 6
    p.write_text(yaml.safe_dump(raw))
    src, dst = State.EMBARGO, State.PAPER
    reg = registry(tmp_path, rec(src, age_days=20))
    approve(reg, FAM, src.value, dst.value)
    ev = evidence(tmp_path, src, dst)
    sm.transition(FAM, dst, ev, "avi.owner", registry=reg, gates=GATES, now_fn=now_fn)
    with pytest.raises(GateNotMetError):
        sm.transition(FAM, dst, ev, "avi.owner", registry=reg, gates=gt.load_gates_config(p), now_fn=now_fn)


def test_gates_config_missing_key_raises(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("g_paper: {}\n")
    with pytest.raises(gt.GatesConfigError):
        run_gpaper(gates=gt.load_gates_config(p))


# ---- gated sleeve / H5 -----------------------------------------------------------------------------------------------
ASOF = datetime(2026, 10, 5, 15, tzinfo=UTC)
PRICES = {"SPY": 500.0, "IEF": 100.0}


def gated(tmp_path: Path, inner, family: str, reg: Registry, account="paper", sub=""):
    rp = tmp_path / f"registry{sub}.yaml"
    sm.save_registry(reg, rp)
    return LifecycleGatedSleeve(inner, family, rp, tmp_path / "lifecycle_overrides.json", account, now_fn)


def two_family_registry(tmp_path, s1: State, s2: State, prior2=None):
    return registry(tmp_path, rec(s1, family="fam_a"), rec(s2, family="fam_b", prior_state=prior2))


def plan_for(sleeves):
    return Allocator(sleeves).plan(ASOF, 100_000.0, {}, PRICES, {}, {})


def test_h5_dry_run_allocator_plan(tmp_path):
    reg = two_family_registry(tmp_path, State.PAPER, State.RESEARCH)
    a = gated(tmp_path, StaticSleeve("a", 0.5, {"SPY": 1.0}), "fam_a", reg)
    b = gated(tmp_path, StaticSleeve("b", 0.5, {"IEF": 1.0}), "fam_b", reg)
    plan = plan_for([a, b])
    assert plan.targets == {"SPY": 0.5, "IEF": 0.0}  # RESEARCH family gets no weight (Allocator zero-fills its symbols)
    assert {t for t, w in plan.targets.items() if w > 0} == {"SPY"}


@pytest.mark.parametrize(
    ("state", "prior", "zero", "expect"),
    [
        (State.PROBATION, State.PAPER, False, True),  # review state keeps the weight
        (State.PROBATION, State.PAPER, True, False),  # two triggers
        (State.DECOMMISSIONED, None, False, False),
        (State.ARCHIVED, None, False, False),
        (State.EMBARGO, None, False, False),
        (State.PAPER, None, False, True),
    ],
)
def test_h5_probation_decommission_weights(tmp_path, state, prior, zero, expect):
    reg = registry(tmp_path, rec(state, family="fam_a", prior_state=prior))
    if zero:
        dc.write_zero_weight(tmp_path / "lifecycle_overrides.json", "fam_a", ["cusum_alarm", "x"], now_fn)
    s = gated(tmp_path, StaticSleeve("a", 0.5, {"SPY": 1.0}), "fam_a", reg)
    plan = plan_for([s])
    assert plan.targets == {"SPY": 0.5 if expect else 0.0}


def test_gated_wrapper_forwards_attributes(tmp_path):
    idx = pd.date_range(end="2026-10-04", periods=150, freq="D", tz="UTC")
    px = pd.Series(30_000 * np.exp(np.cumsum(np.full(150, 0.004) + 0.01 * np.sin(np.arange(150)))), index=idx)
    hist = {"BTC/USD": px}
    prices = {"BTC/USD": float(px.iloc[-1])}
    reg = registry(tmp_path, rec(State.PAPER, family="fam_a"))
    bare = BtcTrendSleeve(0.08)
    wrapped = gated(tmp_path, BtcTrendSleeve(0.08), "fam_a", reg)
    for attr in ("name", "weight", "fractional", "time_in_force", "drift_check", "band_within"):
        assert getattr(wrapped, attr) == getattr(bare, attr), attr
    assert wrapped.drift_check is False and wrapped.time_in_force == "gtc" and wrapped.fractional is True
    assert wrapped.symbols() == bare.symbols()
    assert wrapped.is_rebalance_due(ASOF, None) == bare.is_rebalance_due(ASOF, None)
    # an attribute the inner sleeve lacks stays absent (Allocator getattr defaults apply)
    plain = gated(tmp_path, StaticSleeve("a", 0.5, {"SPY": 1.0}), "fam_a", reg, sub="2")
    assert getattr(plain, "drift_check", "dflt") == "dflt" and getattr(plain, "band_within", None) is None
    p_bare = Allocator([bare]).plan(ASOF, 100_000.0, {}, prices, hist, {})
    p_wrap = Allocator([wrapped]).plan(ASOF, 100_000.0, {}, prices, hist, {})
    assert p_bare.to_dict() == p_wrap.to_dict()
    assert p_bare.targets  # not vacuous


def test_unknown_family_and_unreadable_registry_fail_closed(tmp_path):
    reg = registry(tmp_path, rec(State.PAPER, family="fam_a"))
    s = gated(tmp_path, StaticSleeve("a", 0.5, {"SPY": 1.0}), "nope", reg)
    assert s.target_weights(ASOF, {}) == {}
    s2 = LifecycleGatedSleeve(
        StaticSleeve("a", 0.5, {"SPY": 1.0}), "fam_a", tmp_path / "missing.yaml", tmp_path / "o.json", "paper", now_fn
    )
    assert s2.target_weights(ASOF, {}) == {}


# ---- combiner -> override store -> gated sleeve ----------------------------------------------------------------------
def inputs(**kw):
    base = {"dd": 0.0, "survival_ref": 0.12, "cusum_alarm": False, "fidelity_breach_months": 0, "mechanism_invalid": False}
    base.update(kw)
    return dc.DecommissionInputs(**base)


def test_one_trigger_keeps_weight_two_triggers_zero(tmp_path):
    reg = registry(tmp_path, rec(State.PAPER, family="fam_a"))
    ov = tmp_path / "lifecycle_overrides.json"
    verdict, reg1 = dc.apply_decommission(reg, ov, "fam_a", inputs(cusum_alarm=True), GATES, now_fn)
    assert verdict == "probation" and reg1.get("fam_a").state is State.PROBATION
    assert reg1.get("fam_a").prior_state is State.PAPER and not ov.exists()
    s1 = gated(tmp_path, StaticSleeve("a", 0.5, {"SPY": 1.0}), "fam_a", reg1)
    assert plan_for([s1]).targets == {"SPY": 0.5}  # one trigger: weight unchanged
    verdict2, reg2 = dc.apply_decommission(
        reg1, ov, "fam_a", inputs(cusum_alarm=True, mechanism_invalid=True), GATES, now_fn
    )
    assert verdict2 == "zero_weight" and ov.exists()
    s2 = gated(tmp_path, StaticSleeve("a", 0.5, {"SPY": 1.0}), "fam_a", reg2, sub="2")
    assert plan_for([s2]).targets == {"SPY": 0.0}
    # direct 2-trigger from PAPER also lands in PROBATION with zero weight
    verdict3, reg3 = dc.apply_decommission(
        reg, tmp_path / "o3.json", "fam_a", inputs(dd=0.2, cusum_alarm=True), GATES, now_fn
    )
    assert verdict3 == "zero_weight" and reg3.get("fam_a").state is State.PROBATION


def test_zero_weight_override_survives_restart(tmp_path):
    reg = registry(tmp_path, rec(State.PAPER, family="fam_a"))
    ov = tmp_path / "lifecycle_overrides.json"
    _, reg2 = dc.apply_decommission(reg, ov, "fam_a", inputs(cusum_alarm=True, dd=0.5), GATES, now_fn)
    rp = tmp_path / "registry.yaml"
    sm.save_registry(reg2, rp)
    # "restart": brand-new objects built from disk only
    for _ in range(2):
        s = LifecycleGatedSleeve(StaticSleeve("a", 0.5, {"SPY": 1.0}), "fam_a", rp, ov, "paper", now_fn)
        assert s.target_weights(ASOF, {}) == {}
    # a registry edit alone cannot restore the weight (registry says PAPER again): override still wins
    sm.save_registry(reg, rp)
    s = LifecycleGatedSleeve(StaticSleeve("a", 0.5, {"SPY": 1.0}), "fam_a", rp, ov, "paper", now_fn)
    assert s.target_weights(ASOF, {}) == {}
    # only an owner approval file dated after the decision clears it
    stale = {"family": "fam_a", "approver": "avi.owner", "clears_zero_weight": True,
             "approved_at_utc": (NOW - timedelta(days=1)).isoformat()}  # fmt: skip
    (reg.approvals_dir / "fam_a__clear_zero_weight.yaml").write_text(yaml.safe_dump(stale))
    assert s.target_weights(ASOF, {}) == {}
    fresh = {**stale, "approved_at_utc": (NOW + timedelta(days=1)).isoformat()}
    (reg.approvals_dir / "fam_a__clear_zero_weight.yaml").write_text(yaml.safe_dump(fresh))
    assert s.target_weights(ASOF, {}) == {"SPY": 1.0}


def test_override_write_is_atomic_and_keeps_other_families(tmp_path):
    ov = tmp_path / "o.json"
    dc.write_zero_weight(ov, "a", ["x"], now_fn)
    dc.write_zero_weight(ov, "b", ["y"], now_fn)
    assert set(dc.read_overrides(ov)) == {"a", "b"}
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_no_sleeve_registry_registration():
    src = (REPO / "src" / "firm" / "lifecycle").rglob("*.py")
    assert not [p for p in src if "SLEEVE_REGISTRY[" in p.read_text() or "SLEEVE_REGISTRY.update" in p.read_text()]
