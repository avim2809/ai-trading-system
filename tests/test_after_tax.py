"""After-tax, after-cost benchmark reporting (P2-05). Synthetic data only; INFORMATION ONLY, not tax advice."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.costs.model import load_cost_config
from firm.reporting import after_tax as at
from firm.reporting.after_tax import TaxConfig

COSTS = load_cost_config()


def cfg(**kw) -> TaxConfig:
    base = {"rate_real_gain": 0.25, "surtax_rate_1": 0.03, "surtax_rate_2": 0.02, "surtax_threshold_ils": 1e12,
            "inflation_adjust": False, "loss_carryforward": False, "withholding": {"US": 0.0, "IE": 0.0}, "assumptions": {}}
    base.update(kw)
    return TaxConfig(**base)


def flat(idx, v):
    return pd.Series(float(v), index=idx)


def px(idx, **cols):
    return pd.DataFrame({k: pd.Series(v, index=idx, dtype=float) if np.ndim(v) else flat(idx, v) for k, v in cols.items()})


def trade(date, sym, qty, price=None):
    return {"date": pd.Timestamp(date), "symbol": sym, "qty": qty, **({} if price is None else {"price": price})}


NO_DIV = pd.DataFrame({"date": pd.to_datetime([]), "symbol": [], "amount": []})
US = {"X": "US", "Y": "US"}


def run(trades, prices, fx, cpi, c, *, div=NO_DIV, cash=10_000.0, **kw):
    return at.apply_tax(pd.DataFrame(trades), prices, fx, cpi, div, US, c, initial_cash_usd=cash, **kw)


def idx_days(n=10, start="2021-03-01"):
    return pd.bdate_range(start, periods=n)


def test_single_trade_hand_computed():
    i = idx_days()
    p = px(i, X=[100] * 5 + [150] * 5)
    out = run([trade(i[0], "X", 100), trade(i[6], "X", -100)], p, flat(i, 1), flat(i, 100), cfg())
    assert out.attrs["annual_tax"]["total"].sum() == pytest.approx(1250.0, abs=1e-9)
    assert out["nav_pre_tax_ils"].iloc[-1] == pytest.approx(15_000.0)
    assert out["nav_after_tax_ils"].iloc[-1] == pytest.approx(13_750.0)


def test_inflation_adjusted_basis():
    i = idx_days()
    p = px(i, X=[100] * 5 + [150] * 5)
    cpi = pd.Series(np.linspace(100, 120, len(i)), index=i)
    cpi.iloc[6:] = 120.0
    cpi.iloc[:1] = 100.0
    cpi.iloc[1:6] = 100.0
    out = run([trade(i[0], "X", 100), trade(i[6], "X", -100)], p, flat(i, 1), cpi, cfg(inflation_adjust=True))
    row = out.attrs["annual_tax"].iloc[0]
    assert row["net_real_gain"] == pytest.approx(5000 - 2000)  # gain falls by the basis uplift 10000 * 20%
    assert row["total"] == pytest.approx(750.0)


def _two_year_idx():
    return pd.to_datetime(["2021-12-20", "2021-12-21", "2021-12-22", "2022-01-03", "2022-01-04", "2022-01-05"])


def test_loss_offset_same_year():
    i = idx_days()
    p = px(i, X=[100] * 5 + [150] * 5, Y=[100] * 5 + [80] * 5)
    t = [trade(i[0], "X", 100), trade(i[0], "Y", 100), trade(i[6], "X", -100), trade(i[6], "Y", -100)]
    out = run(t, p, flat(i, 1), flat(i, 100), cfg(), cash=20_000.0)
    assert out.attrs["annual_tax"]["total"].sum() == pytest.approx(0.25 * 3000)  # 5000 - 2000


@pytest.mark.parametrize("carry,expected", [(False, 1250.0), (True, 750.0)])
def test_loss_carryforward(carry, expected):
    i = _two_year_idx()
    p = px(i, X=[100, 100, 80, 80, 80, 130], Y=[100] * 3 + [100, 100, 150])
    t = [trade(i[0], "X", 100), trade(i[2], "X", -100),  # 2021: loss of 2000
         trade(i[3], "Y", 100), trade(i[5], "Y", -100)]  # 2022: gain of 5000
    out = run(t, p, flat(i, 1), flat(i, 100), cfg(loss_carryforward=carry), cash=20_000.0)
    tax = out.attrs["annual_tax"]
    assert tax.loc[2021, "total"] == 0.0
    assert tax.loc[2022, "total"] == pytest.approx(expected)


def test_surtax_only_above_threshold():
    i = idx_days()
    p = px(i, X=[100] * 5 + [150] * 5)
    t = [trade(i[0], "X", 100), trade(i[6], "X", -100)]
    below = run(t, p, flat(i, 1), flat(i, 100), cfg(surtax_threshold_ils=5000.0))
    assert below.attrs["annual_tax"]["surtax"].sum() == 0.0  # income == threshold: nothing above
    above = run(t, p, flat(i, 1), flat(i, 100), cfg(surtax_threshold_ils=4000.0))
    assert above.attrs["annual_tax"]["surtax"].sum() == pytest.approx((0.03 + 0.02) * 1000.0)
    assert above.attrs["annual_tax"]["total"].sum() == pytest.approx(1250 + 50)


def test_fx_translation_gain_in_ils_when_usd_rises():
    i = idx_days()
    p = px(i, X=[100.0] * 10)  # flat in USD
    fx = pd.Series([3.5] * 5 + [4.0] * 5, index=i)
    out = run([trade(i[0], "X", 100), trade(i[6], "X", -100)], p, fx, flat(i, 100), cfg())
    row = out.attrs["annual_tax"].iloc[0]
    assert row["net_real_gain"] == pytest.approx(10_000 * 4.0 - 10_000 * 3.5)
    assert row["total"] == pytest.approx(1250.0)


def test_withholding_us_vs_ie_domicile_differs():
    i = idx_days()
    p = px(i, X=100.0)
    div = pd.DataFrame({"date": [i[3]], "symbol": ["X"], "amount": [2.0]})
    c = cfg(withholding={"US": 0.15, "IE": 0.0})
    t = [trade(i[0], "X", 100)]
    us = at.apply_tax(pd.DataFrame(t), p, flat(i, 1), flat(i, 100), div, {"X": "US"}, c, initial_cash_usd=10_000.0)
    ie = at.apply_tax(pd.DataFrame(t), p, flat(i, 1), flat(i, 100), div, {"X": "IE"}, c, initial_cash_usd=10_000.0)
    assert us["nav_pre_tax_ils"].iloc[-1] < ie["nav_pre_tax_ils"].iloc[-1]  # withholding leaves the fund (cash reinvested net)
    assert us["nav_pre_tax_ils"].iloc[-1] == pytest.approx(10_000 + 200 * 0.85)
    assert ie["nav_pre_tax_ils"].iloc[-1] == pytest.approx(10_000 + 200)


def test_after_tax_never_exceeds_pre_tax_total_return():
    i = pd.bdate_range("2019-01-02", periods=900)
    rng = np.random.default_rng(1)
    p = px(i, X=100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 900)), Y=100 * np.cumprod(1 + rng.normal(0.0002, 0.006, 900)))
    t = [trade(i[0], "X", 60), trade(i[0], "Y", 40), trade(i[300], "X", -30), trade(i[300], "Y", 10), trade(i[600], "Y", -20)]
    fx = pd.Series(3.5 + 0.2 * np.sin(np.arange(900) / 90), index=i)
    cpi = pd.Series(np.linspace(100, 108, 900), index=i)
    out = run(t, p, fx, cpi, cfg(inflation_adjust=True), cash=10_000.0)
    off = run(t, p, fx, cpi, cfg(inflation_adjust=True), cash=10_000.0, taxes=False)
    assert (out["nav_after_tax_ils"] <= off["nav_after_tax_ils"] + 1e-9).all()
    assert out["nav_after_tax_ils"].iloc[-1] / out["nav_after_tax_ils"].iloc[0] <= off["nav_after_tax_ils"].iloc[-1] / off["nav_after_tax_ils"].iloc[0]


def _bm2_inputs(n=800, seed=2):
    i = pd.bdate_range("2019-06-03", periods=n)
    rng = np.random.default_rng(seed)
    prices = px(i, SPY=300 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)), IEF=100 * np.cumprod(1 + rng.normal(0.0001, 0.003, n)))
    dd = i[63::63]
    div = pd.DataFrame({"date": list(dd) * 2, "symbol": ["SPY"] * len(dd) + ["IEF"] * len(dd),
                        "amount": [1.5] * len(dd) + [0.3] * len(dd)})
    return i, prices, pd.Series(3.6 + 0.0003 * np.arange(n), index=i), pd.Series(np.linspace(100, 106, n), index=i), div


def _bm2(rebalance, **kw):
    _i, prices, fx, cpi, div = _bm2_inputs()
    return at.benchmark_bm2(prices, fx, cpi, div, cfg(inflation_adjust=True), COSTS, rebalance=rebalance, **kw)


def test_benchmark_bm2_annual_rebalance_weights():
    out = _bm2("annual")
    tr = out.attrs["trades"]
    i, prices, *_ = _bm2_inputs()
    first = tr[tr["date"] == i[0]]
    vals = first["qty"].to_numpy() * prices.loc[i[0], first["symbol"]].to_numpy()
    assert dict(zip(first["symbol"], vals / vals.sum(), strict=True)) == pytest.approx({"SPY": 0.6, "IEF": 0.4})
    expected = {i[0]} | {i[i.year == y][0] for y in sorted(set(i.year)) if y != i[0].year}  # initial + first trading day of each later year
    assert set(tr["date"]) == expected


def test_benchmark_bm2_monthly_rebalance_weights():
    out = _bm2("monthly")
    tr = out.attrs["trades"]
    days = sorted(set(tr["date"]))
    assert len(days) > 20
    assert all(a.month != b.month or a.year != b.year for a, b in itertools.pairwise(days))


@pytest.mark.parametrize("rebalance", ["annual", "monthly"])
def test_cost_deducted_in_both(rebalance):
    out = _bm2(rebalance)
    assert out["after_cost"].iloc[-1] < out["gross"].iloc[-1]
    assert out["after_tax"].iloc[-1] < out["after_cost"].iloc[-1]
    assert set(out.columns) >= {"gross", "after_cost", "after_tax"}


def _gates(**bench):
    return {"g_research": {"benchmark": {"benchmark": "BM2_60_40_SPY_IEF", "rebalance": "annual", "sensitivity": "monthly", **bench}}}


def _gate_run(gates):
    _i, prices, fx, cpi, div = _bm2_inputs()
    return at.gate_benchmark_bm2(prices, fx, cpi, div, cfg(inflation_adjust=True), COSTS, gates=gates)


@pytest.mark.parametrize("variant", ["annual", "monthly"])
def test_gate_benchmark_is_pinned_variant(variant):
    name, df, both = _gate_run(_gates(uses_higher_after_tax_sharpe_of_the_two=False, gate_variant=variant))
    assert name == variant and set(both) == {"annual", "monthly"}
    pd.testing.assert_frame_equal(df, both[variant])


def test_gate_benchmark_reads_the_real_gates_file_pinned_annual():
    name, _df, _both = at.gate_benchmark_bm2(*_bm2_inputs()[1:], cfg(inflation_adjust=True), COSTS)
    assert name == "annual"
    assert at.gate_variant_rule() == (False, "annual")


def test_gate_benchmark_legacy_higher_after_tax_variant_when_flag_true():
    name, df, both = _gate_run(_gates(uses_higher_after_tax_sharpe_of_the_two=True, gate_variant="annual"))
    sr = {k: at.summarise(v["after_tax"]).sharpe for k, v in both.items()}
    assert name == max(sr, key=sr.get)
    pd.testing.assert_frame_equal(df, both[name])


@pytest.mark.parametrize("bench", [
    {"gate_variant": "annual"},                                                              # flag missing
    {"uses_higher_after_tax_sharpe_of_the_two": False},                                      # variant missing
    {"uses_higher_after_tax_sharpe_of_the_two": False, "gate_variant": "weekly"},            # unknown variant
    {"uses_higher_after_tax_sharpe_of_the_two": "no", "gate_variant": "annual"},             # not a bool
])
def test_gate_benchmark_fails_closed(bench):
    with pytest.raises(ValueError):
        at.gate_variant_rule(_gates(**bench))
    with pytest.raises(ValueError):
        _gate_run(_gates(**bench))


def test_gate_variant_rule_needs_a_benchmark_block():
    with pytest.raises(ValueError):
        at.gate_variant_rule({"x": {"y": 1}})


def test_tax_conventions_identical_for_system_and_bm2():
    _i, prices, fx, cpi, div = _bm2_inputs()
    c = cfg(inflation_adjust=True, loss_carryforward=True)
    b = at.benchmark_bm2(prices, fx, cpi, div, c, COSTS, rebalance="annual")
    sys_nav = at.apply_tax(b.attrs["trades"], prices, fx, cpi, div, {"SPY": "US", "IEF": "US"}, c,
                           initial_cash_usd=b.attrs["initial_usd"])
    pd.testing.assert_series_equal(sys_nav["nav_after_tax_ils"], b["after_tax"], check_names=False, rtol=1e-9)


def test_buy_and_hold_end_nav_includes_deferred_liability():
    i = idx_days()
    p = px(i, X=[100] * 5 + [150] * 5)
    out = run([trade(i[0], "X", 100)], p, flat(i, 1), flat(i, 100), cfg())
    assert out["dtl_ils"].iloc[-1] == pytest.approx(1250.0)
    assert out["nav_after_tax_ils"].iloc[-1] == pytest.approx(15_000 - 1250)
    assert out.attrs["annual_tax"]["total"].sum() == 0.0  # nothing realised
    liq = run([trade(i[0], "X", 100)], p, flat(i, 1), flat(i, 100), cfg(), convention="terminal_liquidation")
    assert liq["nav_after_tax_ils"].iloc[5] == pytest.approx(liq["nav_pre_tax_ils"].iloc[5])  # no accrual before the end
    assert liq["nav_after_tax_ils"].iloc[-1] == pytest.approx(13_750.0)


def test_no_double_count():
    i = pd.bdate_range("2020-01-02", periods=300)
    rng = np.random.default_rng(5)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.008, 300)), index=i)
    div = pd.DataFrame({"date": i[[60, 130, 200, 270]], "symbol": "X", "amount": [0.7, 0.8, 0.9, 1.0]})
    divs = pd.Series(0.0, index=i)
    divs[div["date"].to_numpy()] = div["amount"].to_numpy()
    adj_ret = (close + divs) / close.shift(1) - 1  # total return from raw close plus dividends
    adj_nav = 10_000.0 * (1 + adj_ret.fillna(0)).cumprod()
    out = at.apply_tax(pd.DataFrame([trade(i[0], "X", 10_000.0 / close.iloc[0], close.iloc[0])]), close.to_frame("X"), flat(i, 1),
                       flat(i, 100), div, {"X": "US"}, cfg(withholding={"US": 0.3, "IE": 0.0}), initial_cash_usd=10_000.0, taxes=False)
    # taxes off: no withholding either; fully reinvested, so NAV matches the adjusted-close total-return NAV
    pd.testing.assert_series_equal(out["nav_pre_tax_ils"], adj_nav, check_names=False, atol=1e-6, rtol=1e-6)


def _perf_inputs(n=1500, seed=0, phi=0.0, gap=0.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0004, 0.01, (n, 2))
    r = np.zeros((n, 2))
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    r[:, 0] += gap
    i = pd.bdate_range("2015-01-02", periods=n)
    return pd.Series(np.cumprod(1 + r[:, 0]), index=i), pd.Series(np.cumprod(1 + r[:, 1]), index=i)


def _states(a, b=None):
    return {k: a for k in ("gross", "after_cost", "after_tax")}


def test_bootstrap_ci_reproducible_with_seed():
    a, b = _perf_inputs()
    r1 = at.compare(_states(a), _states(b), n_boot=300, seed=7)
    r2 = at.compare(_states(a), _states(b), n_boot=300, seed=7)
    r3 = at.compare(_states(a), _states(b), n_boot=300, seed=8)
    assert r1 == r2
    assert r1["diff"]["after_tax"]["sharpe"] != r3["diff"]["after_tax"]["sharpe"]
    assert r1["seed"] == 7 and r1["n_boot"] == 300


def test_ci_uses_block_not_iid():
    a, b = _perf_inputs(phi=0.6, seed=3)
    res = at.compare(_states(a), _states(b), n_boot=400, seed=1)
    d = res["diff"]["after_tax"]["sharpe"]
    block_w = d["ci_high"] - d["ci_low"]
    ra, rb = a.pct_change().dropna().to_numpy(), b.pct_change().dropna().to_numpy()
    rng = np.random.default_rng(1)
    iid = []
    for _ in range(400):
        k = rng.integers(0, len(ra), len(ra))
        iid.append(ra[k].mean() / ra[k].std(ddof=1) * np.sqrt(252) - rb[k].mean() / rb[k].std(ddof=1) * np.sqrt(252))
    iid_w = np.percentile(iid, 97.5) - np.percentile(iid, 2.5)
    assert block_w > 1.3 * iid_w
    assert res["block_length"] > 1.5
    with pytest.raises(ValueError):
        at.compare(_states(a), _states(b), n_boot=10, seed=1, block=1.0)  # iid resampling is refused


def test_paired_resample_uses_same_indices():
    a, _ = _perf_inputs()
    res = at.compare(_states(a), _states(a), n_boot=200, seed=2)  # identical legs: any paired draw gives a zero gap
    for k in ("sharpe", "cagr", "max_dd"):
        d = res["diff"]["after_tax"][k]
        assert d["ci_low"] == pytest.approx(0, abs=1e-12) and d["ci_high"] == pytest.approx(0, abs=1e-12)


def test_ci_centred_on_sharpe_gap_not_ir():
    rng = np.random.default_rng(11)
    n = 2000
    rb = rng.normal(0.0005, 0.01, n)
    ra = rb + 0.0002 + rng.normal(0, 0.0005, n)  # nearly the same series: SR gap small, IR of the difference huge
    i = pd.bdate_range("2012-01-02", periods=n)
    a, b = pd.Series(np.cumprod(1 + ra), index=i), pd.Series(np.cumprod(1 + rb), index=i)
    res = at.compare(_states(a), _states(b), n_boot=600, seed=4)
    d = res["diff"]["after_tax"]["sharpe"]
    ir = (ra - rb).mean() / (ra - rb).std(ddof=1) * np.sqrt(252)
    gap = at.summarise(a).sharpe - at.summarise(b).sharpe
    assert d["estimate"] == pytest.approx(gap)
    assert ir > 5 * abs(gap)
    mid = 0.5 * (d["ci_low"] + d["ci_high"])
    assert d["ci_low"] <= gap <= d["ci_high"]
    assert abs(mid - gap) < 0.2 * abs(ir)


def test_compare_returns_gross_after_cost_after_tax_states():
    a, b = _perf_inputs(seed=9)
    res = at.compare({"gross": a, "after_cost": a * 0.999 ** np.arange(len(a)), "after_tax": a * 0.998 ** np.arange(len(a))},
                     _states(b), n_boot=100, seed=1)
    assert set(res["system"]) == set(res["bench"]) == set(res["diff"]) == {"gross", "after_cost", "after_tax"}
    assert isinstance(res["system"]["gross"], at.PerfSummary)
    assert set(res["diff"]["gross"]) == {"sharpe", "cagr", "max_dd"}
    assert set(res["diff"]["gross"]["sharpe"]) == {"estimate", "ci_low", "ci_high"}
    assert res["system"]["after_tax"].cagr < res["system"]["gross"].cagr


def test_summarise_known_values():
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    nav = pd.Series(np.cumprod(1 + np.r_[0.0, r]), index=pd.bdate_range("2020-01-01", periods=6))
    s = at.summarise(nav)
    assert s.vol == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert s.sharpe == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert s.max_dd == pytest.approx(0.02)
    assert s.cagr == pytest.approx(np.prod(1 + r) ** (252 / 5) - 1)


def test_load_tax_config_tags_and_placeholders():
    c = at.load_tax_config(Path(__file__).resolve().parents[1] / "config" / "tax_il.yaml")
    assert (c.rate_real_gain, c.surtax_rate_1, c.surtax_rate_2, c.surtax_threshold_ils) == (0.25, 0.03, 0.02, 721560)
    assert c.withholding == {"US": 0.0, "IE": 0.0}
    assert "UNSET" in c.assumptions["withholding.US"] and "UNSET" in c.assumptions["withholding.IE"]
    assert "INFORMATION ONLY" in at.describe_assumptions(c)


def test_no_hardcoded_tax_threshold_in_code():
    src = (Path(at.__file__)).read_text()
    assert "721560" not in src and "721_560" not in src


@pytest.mark.real_data
@pytest.mark.skipif(not Path("data/research/eodhd/etfs_full/SPY.parquet").exists(), reason="licensed data absent")
def test_core_only_100_sanity():
    """Owner-triggered only (deselected by default): pre-tax BM2 CAGR within 1 pp of the allocation replay's 8.66%."""
    import datetime as dt

    from firm.data.etf_loader import load_dividends, load_etf_universe, total_return  # noqa: F401

    asof = dt.date(2026, 9, 28)
    u = Path("config/universe_etf.yaml")
    s = load_etf_universe(u, asof=asof, include_delisted=False)
    tr = pd.concat({k: total_return(s[k]) for k in ("SPY", "IEF")}, axis=1).loc["2015-02-02":asof.isoformat()].dropna()
    nav = (1 + (0.6 * tr["SPY"] + 0.4 * tr["IEF"])).cumprod()  # daily-rebalanced pre-tax reference
    assert abs(at.summarise(nav).cagr - 0.0866) < 0.01


def _states_from_bm2(df):
    return {s: df[s] for s in at.STATES}  # raw columns: they inherit the frame's DataFrame-valued attrs


def test_compare_accepts_bm2_columns_carrying_dataframe_attrs():
    a, m = _bm2("annual"), _bm2("monthly")
    assert isinstance(a["after_tax"].attrs.get("trades"), pd.DataFrame)  # the precondition that used to break pd.concat
    out = at.compare(_states_from_bm2(m), _states_from_bm2(a), n_boot=50, seed=1)
    assert out["n_obs"] > 100


def test_compare_does_not_mutate_input_attrs():
    a, m = _bm2("annual"), _bm2("monthly")
    before = set(a["after_tax"].attrs)
    at.compare(_states_from_bm2(m), _states_from_bm2(a), n_boot=20, seed=1)
    assert set(a["after_tax"].attrs) == before
