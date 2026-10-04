"""P2-04 tests for firm.costs.model."""
from __future__ import annotations

import inspect
import math
import sys

import numpy as np
import pandas as pd
import pytest

from firm.costs import model
from firm.costs.model import (
    CostBreakdown,
    InstrumentCostSpec,
    annual_cost_drag,
    cost,
    empirical_half_spread,
    load_cost_config,
    roll_cost,
    spec_from_config,
)

CFG = load_cost_config()
ETF_IBKR = spec_from_config("etf_ibkr", CFG)
ETF_ALP = spec_from_config("etf_alpaca", CFG)
FUT = spec_from_config("mes_ibkr", CFG)


def _c(spec, qty, **kw):
    args = {"price": 100.0, "adv": 1e6, "spread": 0.0004, "vol": 0.01}
    args.update(kw)
    return cost(spec, qty, args.pop("price"), args.pop("adv"), args.pop("spread"), args.pop("vol"), cfg=CFG, **args)


@pytest.mark.parametrize("spec,qty,roll", [(ETF_IBKR, 500, False), (ETF_IBKR, -500, False), (FUT, 3, True)])
def test_multiplier_scales_every_component(spec, qty, roll):
    base = _c(spec, qty, is_roll=roll)
    for m in (1, 2, 3):
        got = _c(spec, qty, is_roll=roll, multiplier=m)
        for f in ("commission", "exchange_fees", "half_spread", "impact", "roll", "total"):
            assert getattr(got, f) == pytest.approx(m * getattr(base, f))
        assert got.total == pytest.approx(got.commission + got.exchange_fees + got.half_spread + got.impact + got.roll)
    assert base.scaled(2).total == pytest.approx(2 * base.total)


def test_ibkr_tiered_schedule_matches_published_examples():
    sch = CFG["schedules"]["ibkr_etf"]["commission"]
    per, lo, cap = sch["per_share"], sch["min_per_order"], sch["max_pct_of_value"]
    # 100 shares: 0.35 exactly equals the per-order minimum
    assert _c(ETF_IBKR, 100).commission == pytest.approx(max(100 * per, lo))
    # 1000 shares: 3.50
    assert _c(ETF_IBKR, 1000).commission == pytest.approx(1000 * per)
    # 100000 shares: per-share 350 (cap 1% of 10M value is far above)
    assert _c(ETF_IBKR, 100_000, adv=1e9).commission == pytest.approx(100_000 * per)
    # minimum binds for 10 shares; the 1%-of-value cap binds for a very cheap stock
    assert _c(ETF_IBKR, 10).commission == pytest.approx(lo)
    assert _c(ETF_IBKR, 1000, price=0.10).commission == pytest.approx(cap * 1000 * 0.10)
    # futures: flat per contract
    assert _c(FUT, 3).commission == pytest.approx(3 * CFG["schedules"]["ibkr_future"]["commission"]["per_contract"])


def test_alpaca_etf_zero_commission_with_regulatory_fees_on_sells_only():
    buy, sell = _c(ETF_ALP, 1000), _c(ETF_ALP, -1000)
    assert buy.commission == 0 and sell.commission == 0
    assert buy.exchange_fees == 0
    reg = CFG["schedules"]["alpaca_etf"]["regulatory_on_sells"]
    expected = reg["sec_rate_of_value"] * 1000 * 100.0 + min(reg["taf_per_share"] * 1000, reg["taf_max_per_order"])
    assert sell.exchange_fees == pytest.approx(expected) and expected > 0


def test_impact_sqrt_scaling():
    a = _c(ETF_ALP, 1000)
    assert _c(ETF_ALP, 4000).impact / (4000 * 100.0) == pytest.approx(2 * a.impact / (1000 * 100.0))
    assert _c(ETF_ALP, 1000, vol=0.02).impact == pytest.approx(2 * a.impact)
    assert a.impact == pytest.approx(ETF_ALP.impact_k * 0.01 * math.sqrt(1000 / 1e6) * 1000 * 100.0)


def test_roll_is_two_crossings():
    one = _c(FUT, 2).half_spread
    assert _c(FUT, 2, is_roll=True).roll == pytest.approx(2 * one)
    assert _c(FUT, 2).roll == 0.0
    assert roll_cost(FUT, 2, 100.0, 0.0004, CFG) == pytest.approx(2 * one)


def test_empirical_half_spread_recovers_known_spread():
    rng = np.random.default_rng(12345)
    # Regime note: CS is accurate when the spread is comparable to daily vol (here 1% vs 1%); when daily vol >> spread (liquid ETFs) the
    # estimator is biased UP (clipping noise at 0), i.e. conservative for cost purposes. See report.
    days, ticks, spread = 2500, 300, 0.01
    sigma_tick = 0.01 / math.sqrt(ticks)
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, sigma_tick, size=(days, ticks)).ravel())).reshape(days, ticks)
    ask, bid = mid * (1 + spread / 2), mid * (1 - spread / 2)
    high = pd.Series(ask.max(axis=1))
    low = pd.Series(bid.min(axis=1))
    close = pd.Series(mid[:, -1])
    est = empirical_half_spread(high, low, close)
    assert est == pytest.approx(spread / 2, rel=0.15)


def test_cs_negative_estimates_clipped_to_zero():
    # zero-spread, tiny-range days produce negative raw estimates for some pairs; the mean must stay >= 0
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 0.5, 500))
    rng_ = rng.uniform(0.001, 0.5, 500)
    est = empirical_half_spread(pd.Series(c + rng_), pd.Series(c - rng_), pd.Series(c))
    assert est >= 0.0
    flat = pd.Series(np.full(50, 100.0))
    assert empirical_half_spread(flat, flat, flat) == 0.0


def test_spread_units_fraction_rejects_ge_1():
    with pytest.raises(ValueError):
        _c(ETF_IBKR, 100, spread=1.0)
    with pytest.raises(ValueError):
        _c(ETF_IBKR, 100, spread=5.0)


@pytest.mark.skip(reason="open acceptance item for P5-04/P6-02: reconcile 10 broker-statement trades once fills exist (P6; candidate account only)")
def test_reconcile_10_broker_statement_trades():
    raise AssertionError("unreachable until P6 fills exist")


def test_no_flat_bps_path():
    public = [(n, f) for n, f in inspect.getmembers(model, inspect.isfunction) if not n.startswith("_") and f.__module__ == model.__name__]
    for name, fn in public:
        params = [p for p in inspect.signature(fn).parameters if p != "cfg"]
        assert not (len(params) == 1 and "bps" in params[0]), name
    assert "bps" not in inspect.signature(cost).parameters


def test_zero_qty_zero_cost():
    z = _c(ETF_IBKR, 0, multiplier=3)
    assert z == CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("adv", [0.0, -5.0])
def test_nonpositive_adv_raises(adv):
    with pytest.raises(ValueError):
        _c(ETF_IBKR, 100, adv=adv)
    with pytest.raises(ValueError):
        _c(ETF_IBKR, 0, adv=adv)


def test_default_spread_used_when_none_and_spec_bps():
    d = _c(ETF_IBKR, 100, spread=None)
    assert d.half_spread == pytest.approx(0.5 * CFG["default_spread_fraction"] * 100.0 * 100)
    spec = InstrumentCostSpec("etf", "ibkr", half_spread_bps=1.0)
    assert _c(spec, 100, spread=None).half_spread == pytest.approx(1e-4 * 100.0 * 100)


def test_annual_cost_drag():
    nav = pd.Series([1e6, 1e6], index=pd.to_datetime(["2020-01-01", "2021-01-01"]))
    trades = pd.DataFrame({"symbol": ["SPY"], "qty": [1000], "price": [100.0], "adv": [1e6], "vol": [0.01]})
    d1 = annual_cost_drag(trades, nav, specs={"SPY": ETF_IBKR}, cfg=CFG)
    d2 = annual_cost_drag(trades, nav, specs={"SPY": ETF_IBKR}, multiplier=2, cfg=CFG)
    assert d1 > 0 and d2 == pytest.approx(2 * d1)


def test_legacy_unaffected():
    import subprocess

    code = (
        "import sys, importlib\n"
        "import firm.agents._liquidity as L\n"
        "before = {k: v for k, v in vars(L).items() if not k.startswith('__')}\n"
        "importlib.import_module('firm.costs.model')\n"
        "after = {k: v for k, v in vars(L).items() if not k.startswith('__')}\n"
        "assert before == after\n"
        "fresh = 'firm.agents._liquidity' in sys.modules\n"
        "print('OK')\n"
        "m = importlib.import_module('firm.costs.model')\n"
        "assert not hasattr(m, '_liquidity') and not any(k.endswith('_liquidity') for k in vars(m))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "OK" in out.stdout
    src = inspect.getsource(model)
    assert "firm.agents" not in src.replace("firm.agents._liquidity` ", "").split('"""', 2)[2]
