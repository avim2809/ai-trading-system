"""core_v1 shared pipeline (signals -> forecasts -> sizing -> engine) on a SYNTHETIC panel. No data, no host ledger."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from firm.research import core_v1_pipeline as P
from firm.research import ledger as L

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import core_v1_preregistered as pre

GATES = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())


@pytest.fixture(autouse=True)
def tmp_ledger(tmp_path, monkeypatch):
    r = tmp_path / "ledger"
    r.mkdir()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(r))
    monkeypatch.setenv("FIRM_LEDGER_ALLOW_DIRTY", "1")
    return r


@pytest.fixture(scope="module")
def panel():
    return P.synthetic_panel(seed=3, n_days=1500, symbols=("AAA", "BBB", "CCC", "DDD"))


@pytest.fixture(scope="module")
def base():
    return P.params_from_gates(GATES, tau=0.09)


# ---- parameters --------------------------------------------------------------------------------------------------------------
def test_params_come_from_gates_not_literals(base):
    rp = GATES["robustness_parameters"]
    assert base.vol_span == rp["vol_span"]["value"] and base.forecast_cap == rp["forecast_cap"]["value"]
    assert base.ewmac_fast_spans == tuple(rp["ewmac_fast_spans"]["value"])
    assert base.breakout_lookbacks == tuple(rp["breakout_lookbacks_N"]["value"])
    assert base.tau == 0.09 and base.gross_cap == rp["gross_cap"]["value"]
    assert base.speed_cost_max_fraction == pytest.approx(1 / 3)


def test_perturb_integer_rounding_half_up_min_two(base):
    assert P.perturb(base, "vol_span", 0.75).vol_span == 26          # 35 * 0.75 = 26.25
    assert P.perturb(base, "vol_span", 1.25).vol_span == 44          # 43.75
    assert P.perturb(base, "vol_long_window_days", 0.75).vol_long_window_days == 1890
    p = P.perturb(base, "ewmac_fast_spans[0]", 0.75)                 # 2 * 0.75 = 1.5 -> 2 (half up)
    assert p.ewmac_fast_spans[0] == 2 and p.ewmac_fast_spans[1:] == base.ewmac_fast_spans[1:]
    assert P.perturb(base, "ewmac_fast_spans[0]", 1.25).ewmac_fast_spans[0] == 3   # 2.5 -> 3
    assert P.perturb(base, "breakout_lookbacks_N[0]", 0.75).breakout_lookbacks[0] == 15
    assert P.perturb(base, "ewmac_slow_to_fast_ratio", 0.75).ewmac_slow_to_fast_ratio == 3
    assert P.perturb(base, "forecast_cap", 0.75).forecast_cap == pytest.approx(15.0)
    assert P.perturb(base, "tau", 1.25).tau == pytest.approx(0.1125)
    with pytest.raises(KeyError):
        P.perturb(base, "no_such_parameter", 1.25)


def test_perturbable_list_matches_gates(base):
    names = P.perturbable_names(GATES)
    assert names["assessed"] and "vol_ewma_span" in names["unassessed"]          # null value in gates: cannot be perturbed
    assert {"max_vol_scale", "instrument_risk_cap_multiple"} <= set(names["unassessed"])  # P4-03 layer is not in the sizing path
    assert sum(n.startswith("ewmac_fast_spans[") for n in names["assessed"]) == 6
    assert sum(n.startswith("breakout_lookbacks_N[") for n in names["assessed"]) == 5
    assert "tau" in names["assessed"] and "vol_ewma_span" not in names["assessed"]


def test_speed_subset_selects_base_rule_ids(base):
    defs = P.rule_defs(base, base)
    ids = [d.rule_id for d in defs]
    assert ids == ["ewmac_2", "ewmac_4", "ewmac_8", "ewmac_16", "ewmac_32", "ewmac_64",
                   "breakout_20", "breakout_40", "breakout_80", "breakout_160", "breakout_320"]
    sub = P.subset_ids(base, pre.SPEED_SUBSETS["drop_fastest"])
    assert sub == ["ewmac_8", "ewmac_16", "ewmac_32", "ewmac_64", "breakout_40", "breakout_80", "breakout_160", "breakout_320"]
    pert = P.perturb(base, "ewmac_fast_spans[2]", 0.75)                       # 8 -> 6; rule id stays ewmac_8
    d = {x.rule_id: x for x in P.rule_defs(base, pert)}
    assert d["ewmac_8"].span == 6 and d["ewmac_8"].slow == 24 and d["ewmac_16"].span == 16


# ---- panel, vol, signals -------------------------------------------------------------------------------------------------------
def test_synthetic_panel_shape_and_entry_gate(panel, base):
    assert list(panel.symbols) == ["AAA", "BBB", "CCC", "DDD"] and panel.close.notna().all().all() and (panel.close > 0).all().all()
    vol = P.vol_annual(panel, base)
    first = vol.apply(lambda s: s.first_valid_index())
    assert all(panel.ret[s].dropna().index.get_loc(first[s]) == pre.ENTRY_GATE_DAYS - 1 for s in panel.symbols)


def test_no_lookahead_in_raw_forecasts(panel, base):
    defs = P.rule_defs(base, base)
    full = P.raw_forecasts(panel, base, defs)
    cut = 1100
    short = P.Panel(**{**panel.__dict__, "close": panel.close.iloc[:cut], "ret": panel.ret.iloc[:cut], "raw_close": panel.raw_close.iloc[:cut],
                       "adv": panel.adv.iloc[:cut]})
    part = P.raw_forecasts(short, base, defs)
    for rid in full:
        pd.testing.assert_frame_equal(full[rid].iloc[:cut], part[rid], check_exact=False, rtol=1e-12, atol=1e-12)


# ---- scalars -----------------------------------------------------------------------------------------------------------------------
def test_pooled_scalar_makes_mean_abs_ten_on_uncapped_raw(panel, base):
    defs = P.rule_defs(base, base)
    raw = P.raw_forecasts(panel, base, defs)
    window = (str(panel.close.index[0].date()), str(panel.close.index[-1].date()))
    scalars = P.estimate_pooled_scalars({r: {s: raw[r][s] for s in panel.symbols} for r in raw}, window)
    assert set(scalars) == set(raw) and all(v > 0 for v in scalars.values())
    for r, sc in scalars.items():
        pooled = pd.concat([raw[r][s].loc[window[0]:window[1]].dropna().abs() for s in panel.symbols])
        assert (pooled * sc).mean() == pytest.approx(10.0, abs=1e-9)
    chk = P.scalar_diagnostics(raw, scalars, window, cap=base.forecast_cap)
    assert all(abs(v["mean_abs_uncapped"] - 10.0) < 1e-9 for v in chk.values())
    assert all(v["mean_abs_capped"] <= v["mean_abs_uncapped"] + 1e-12 for v in chk.values())   # the post-cap mean is reported separately


def test_scalars_estimated_per_rule_not_per_instrument(panel, base):
    defs = P.rule_defs(base, base)
    raw = P.raw_forecasts(panel, base, defs)
    window = (str(panel.close.index[0].date()), str(panel.close.index[-1].date()))
    sc = P.estimate_pooled_scalars({r: {s: raw[r][s] for s in panel.symbols} for r in raw}, window)
    assert all(isinstance(v, float) for v in sc.values())      # one float per rule


# ---- speed filter ------------------------------------------------------------------------------------------------------------------
def test_speed_filter_uses_cost_and_turnover_only_and_differs_by_instrument():
    turn = {"CHEAP": {"ewmac_2": 40.0, "ewmac_4": 20.0, "breakout_20": 15.0}, "DEAR": {"ewmac_2": 40.0, "ewmac_4": 20.0, "breakout_20": 15.0}}
    cost = {"CHEAP": 0.0002, "DEAR": 0.0030}
    sigma = {"CHEAP": 0.16, "DEAR": 0.16}
    out = P.apply_cost_speed_filter(turn, cost, sigma, expected_rule_sharpe=0.30)
    # limit = 0.3 / 3 = 0.1 Sharpe units; cost = turnover * cost / sigma
    assert out["CHEAP"] == {"ewmac": [2, 4], "breakout": [20]}       # 40*0.0002/0.16 = 0.05 ok
    assert out["DEAR"] == {"ewmac": [], "breakout": []}              # 15*0.003/0.16 = 0.28 > 0.1: everything dropped
    mid = P.apply_cost_speed_filter(turn, {"CHEAP": 0.0002, "DEAR": 0.0006}, sigma, expected_rule_sharpe=0.30)
    assert mid["DEAR"] == {"ewmac": [4], "breakout": [20]}           # 40*0.0006/0.16 = 0.15 > 0.1 drops ewmac_2 only
    assert mid["CHEAP"] != mid["DEAR"]
    import inspect

    assert not {"returns", "pnl", "sharpe"} & set(inspect.signature(P.apply_cost_speed_filter).parameters)


def test_dropping_one_of_six_speeds_redistributes_the_group_weight():
    surv = {"ewmac": [2, 4, 8, 16, 32], "breakout": [20, 40, 80, 160, 320]}     # ewmac_64 dropped
    w = P.rule_weights(surv, {"ewmac": 0.5, "breakout": 0.5})
    assert w["ewmac_2"] == pytest.approx(0.1) and w["breakout_20"] == pytest.approx(0.1) and "ewmac_64" not in w
    assert sum(w.values()) == pytest.approx(1.0)
    only_breakout = P.rule_weights({"ewmac": [], "breakout": [20, 40]}, {"ewmac": 0.5, "breakout": 0.5})
    assert only_breakout == {"breakout_20": 0.5, "breakout_40": 0.5}              # an empty group's weight goes to the other group
    assert P.rule_weights({"ewmac": [], "breakout": []}, {"ewmac": 0.5, "breakout": 0.5}) == {}


# ---- FDM / IDM ---------------------------------------------------------------------------------------------------------------------
def test_fdm_and_idm_obey_caps_and_floor():
    rho = np.array([[1.0, 0.0], [0.0, 1.0]])
    w = np.array([0.5, 0.5])
    fdm = P.fdm_from_rho(w, rho, cap=2.5)
    assert fdm == pytest.approx(np.sqrt(2))
    assert P.fdm_from_rho(np.ones(10) / 10, np.eye(10), cap=4.0) == pytest.approx(np.sqrt(10))
    assert P.fdm_from_rho(np.ones(10) / 10, np.eye(10), cap=2.0) == 2.0               # sqrt(10) = 3.16 capped at the cap
    neg = np.array([[1.0, -0.9], [-0.9, 1.0]])
    assert P.fdm_from_rho(w, neg, cap=2.5) == pytest.approx(np.sqrt(2))                # negative correlation is floored at 0
    rets = pd.DataFrame(np.random.default_rng(1).normal(0, 0.01, (600, 3)), columns=list("xyz"))
    H, idm = P.estimate_idm_H(rets, np.array([1 / 3] * 3), ("", ""))
    assert H.shape == (3, 3) and 1.0 <= idm <= 2.5


def test_fdm_rho_estimator_returns_matrix_and_value(panel, base):
    defs = P.rule_defs(base, base)
    raw = P.raw_forecasts(panel, base, defs)
    window = (str(panel.close.index[0].date()), str(panel.close.index[-1].date()))
    sc = P.estimate_pooled_scalars({r: {s: raw[r][s] for s in panel.symbols} for r in raw}, window)
    fc = P.signed_forecasts(raw, sc, base)
    rho_df = P.pooled_rho(fc, window, panel.symbols)
    assert rho_df.shape == (11, 11) and np.allclose(np.diag(rho_df.values), 1.0) and np.allclose(rho_df.values, rho_df.values.T)
    w = P.rule_weights({"ewmac": list(base.ewmac_fast_spans), "breakout": list(base.breakout_lookbacks)}, {"ewmac": 0.5, "breakout": 0.5})
    rho, fdm = P.estimate_fdm_rho({s: fc[s] for s in panel.symbols}, window, w)
    assert rho.shape == (11, 11) and 1.0 <= fdm <= base.fdm_cap


# ---- perturbation re-estimates the scalar --------------------------------------------------------------------------------------------
def test_perturbed_speed_has_pooled_abs_forecast_ten(panel, base):
    pert = P.perturb(base, "ewmac_fast_spans[3]", 0.75)                       # 16 -> 12
    defs = P.rule_defs(base, pert)
    raw = P.raw_forecasts(panel, pert, defs)
    window = (str(panel.close.index[0].date()), str(panel.close.index[-1].date()))
    sc = P.estimate_pooled_scalars({r: {s: raw[r][s] for s in panel.symbols} for r in raw}, window)
    pooled = pd.concat([raw["ewmac_16"][s].loc[window[0]:window[1]].dropna().abs() for s in panel.symbols])
    assert (pooled * sc["ewmac_16"]).mean() == pytest.approx(10.0, abs=1e-9)
    d = {x.rule_id: x for x in defs}
    assert d["ewmac_16"].span == 12 and d["ewmac_16"].slow == 48


# ---- end to end sizing / engine ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def constants(panel, base):
    defs = P.rule_defs(base, base)
    raw = P.raw_forecasts(panel, base, defs)
    window = (str(panel.close.index[0].date()), str(panel.close.index[-1].date()))
    sc = P.estimate_pooled_scalars({r: {s: raw[r][s] for s in panel.symbols} for r in raw}, window)
    fc = P.signed_forecasts(raw, sc, base)
    rho = P.pooled_rho(fc, window, panel.symbols)
    surv = {s: {"ewmac": list(base.ewmac_fast_spans), "breakout": list(base.breakout_lookbacks)} for s in panel.symbols}
    w = {s: 1.0 / len(panel.symbols) for s in panel.symbols}
    return P.ConstantsBundle(scalars=sc, survivors=surv, rho=rho, idm=1.2, instrument_weights=w, group_weights={"ewmac": 0.5, "breakout": 0.5},
                             window=window)


def test_run_config_end_to_end(panel, base, constants):
    res = P.run_config(panel, constants, base, subset_ids=None, ledger_mode="exploratory", label="smoke")
    assert len(res.engine.returns) == len(panel.close) and res.engine.returns.notna().all()
    assert (res.engine.positions >= -1e-12).all().all()                                  # long/flat
    assert res.engine.gross_exposure.max() <= base.gross_cap + 1e-6
    assert 0.0 <= res.gross_cap_bound_share <= 1.0
    assert (res.engine.positions.abs().sum(axis=1) > 0).any()
    assert res.engine.costs["total"].sum() > 0


def test_stress_multiplier_scales_costs(panel, base, constants):
    r1 = P.run_config(panel, constants, base, subset_ids=None, ledger_mode="exploratory", label="1x")
    r2 = P.run_config(panel, constants, base, subset_ids=None, ledger_mode="exploratory", label="2x", stress=2.0)
    assert r2.engine.costs["total"].sum() == pytest.approx(2.0 * r1.engine.costs["total"].sum(), rel=1e-9)
    assert r2.engine.positions.equals(r1.engine.positions)


def test_flat_bps_comparison_cost_is_not_the_gate_cost(panel, base, constants):
    r1 = P.run_config(panel, constants, base, subset_ids=None, ledger_mode="exploratory", label="gate")
    rf = P.run_config(panel, constants, base, subset_ids=None, ledger_mode="exploratory", label="flat", flat_bps=5.0)
    assert rf.engine.costs["total"].sum() != pytest.approx(r1.engine.costs["total"].sum())


# ---- reference drawdown ------------------------------------------------------------------------------------------------------------------
def test_bootstrap_max_dd_is_seeded_and_scales_with_horizon():
    r = np.random.default_rng(0).normal(0.0003, 0.01, 3000)
    a = P.bootstrap_max_dd_p95(r, path_len=250, draws=2000, seed=5)
    b = P.bootstrap_max_dd_p95(r, path_len=250, draws=2000, seed=5)
    long = P.bootstrap_max_dd_p95(r, path_len=2520, draws=500, seed=5)
    assert a == b and 0.0 < a < long < 1.0
    assert P.bootstrap_max_dd_p95(r, path_len=250, draws=2000, seed=6) != a


# ---- tax-model trades ----------------------------------------------------------------------------------------------------------------------
def test_tax_trades_exclude_dividend_reinvestment():
    idx = pd.bdate_range("2020-01-01", periods=60)
    raw = pd.DataFrame({"AAA": 100.0}, index=idx)
    div = pd.DataFrame({"date": [idx[30]], "symbol": ["AAA"], "amount": [1.0]})
    f = pd.Series(1.0, index=idx)
    f.loc[: idx[29]] = 1.0 / 1.01                # adjusted close lower before the ex-date by 1%
    adj = raw.mul(f, axis=0)
    pos_adj = pd.DataFrame({"AAA": 10.0}, index=idx)   # engine units constant in adjusted shares
    # the engine holds a constant number of ADJUSTED shares: in raw shares that is 10 * f_t / ... -> grows 1% at the ex-date
    trades = P.tax_model_trades(pos_adj, adj, raw, div, symbols=["AAA"])
    first = trades.iloc[0]
    assert first["symbol"] == "AAA" and first["qty"] > 0 and first["date"] == idx[0]
    assert len(trades) == 1                                                    # no discretionary trade after the first buy: the dividend
    #                                                                          reinvestment the tax model does itself is not a trade


def test_core_only_100_targets_rebalance_on_month_start_and_drift():
    idx = pd.bdate_range("2021-01-04", periods=80)
    px = pd.DataFrame({"SPY": np.linspace(100, 130, 80), "IEF": 100.0}, index=idx)
    tg = P.core_only_100_targets(px, capital=100_000.0, band_abs=0.02)
    assert tg.iloc[0].mul(px.iloc[0]).sum() == pytest.approx(100_000.0, rel=1e-9)
    assert tg.iloc[0]["SPY"] * px.iloc[0]["SPY"] == pytest.approx(60_000.0, rel=1e-9)
    assert (tg.diff().abs().sum(axis=1) > 0).sum() >= 2                          # at least the first day and a later rebalance


# ---- hygiene ------------------------------------------------------------------------------------------------------------------------------
def test_pipeline_never_reads_files_directly():
    src = (ROOT / "src" / "firm" / "research" / "core_v1_pipeline.py").read_text()
    tree = ast.parse(src)
    bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in {"read_parquet", "read_csv", "read_feather"}]
    assert not bad, f"direct file reads at lines {bad}; use firm.research.data_access via firm.data.etf_loader"
