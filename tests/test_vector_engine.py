"""Signed, multiplier-aware, cost-model-driven vector engine (P3-09). Synthetic data and fake cost functions only."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.backtest import vector_engine as VE
from firm.backtest.vector_engine import (
    EngineConfig,
    LedgerContext,
    LedgerRequiredError,
    run_vector_backtest,
)
from firm.costs.model import CostBreakdown
from firm.research import ledger as L

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _bd(total=0.0, roll=0.0, commission=0.0, impact=0.0):
    return CostBreakdown(commission, 0.0, 0.0, impact, roll, total + roll)


def zero_cost(symbol, date, qty_delta, price, adv, vol_pct, multiplier, is_roll, stress):
    return _bd()


class FlatFraction:
    """Currency cost = bps * |qty| * price * mult * stress; records stress and calls."""

    def __init__(self, bps=10.0, roll_bps=0.0, mult=None):
        self.bps, self.roll_bps, self.mult = bps, roll_bps, mult or {}
        self.stresses: list[float] = []
        self.calls: list[tuple] = []

    def __call__(self, symbol, date, qty_delta, price, adv, vol_pct, multiplier, is_roll, stress):
        self.stresses.append(stress)
        self.calls.append((symbol, date, qty_delta, is_roll))
        notional = abs(qty_delta) * price * multiplier
        if is_roll:
            return _bd(roll=self.roll_bps / 1e4 * notional * stress)
        return _bd(total=self.bps / 1e4 * notional * stress, commission=self.bps / 1e4 * notional * stress)


def idx(n):
    return pd.bdate_range("2024-01-01", periods=n)


def frame(cols, n=None):
    arr = np.asarray(cols, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return pd.DataFrame(arr, index=idx(arr.shape[0]), columns=[f"S{i}" for i in range(arr.shape[1])])


def run(prices, targets, mult=None, cost_fn=zero_cost, cfg=None, **kw):
    mult = mult or {c: 1.0 for c in prices.columns}
    cfg = cfg or EngineConfig(initial_capital=1000.0, fill_lag_bars=kw.pop("lag", 1))
    kw.setdefault("exploratory", True)
    return run_vector_backtest(prices, targets, mult, cost_fn, cfg, **kw)


@pytest.fixture(autouse=True)
def tmp_ledger(tmp_path, monkeypatch):
    r = tmp_path / "ledger"
    r.mkdir()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(r))
    monkeypatch.setenv("FIRM_LEDGER_ALLOW_DIRTY", "1")
    return r


def test_zero_cost_equals_hand_pnl():
    prices = frame([100.0, 102.0, 101.0])
    targets = frame([2.0, 0.0, 0.0])  # decided at close 0: long 2 units held over bar 1, flat after bar 2 fill
    res = run(prices, targets, mult={"S0": 5.0})
    # lag 1: held = [0, 2, 0]; pnl bar1 = 0 units (pos_{t-1}=0), bar2 = 2*5*(101-102) = -10
    # position entered at bar 1 close (fill) -> earns bar 2
    assert res.returns.iloc[0] == 0.0
    assert res.returns.iloc[1] == 0.0
    assert res.returns.iloc[2] == pytest.approx(-10.0 / 1000.0, abs=1e-12)
    # lag 0: held = [2,0,0]: earns bar1 only: 2*5*(2) = 20
    res0 = run(prices, targets, mult={"S0": 5.0}, lag=0)
    assert res0.returns.iloc[1] == pytest.approx(20.0 / 1000.0, abs=1e-12)
    assert res0.returns.iloc[2] == 0.0


def test_short_position_pnl_sign():
    prices = frame([100.0, 100.0, 90.0])
    targets = frame([-1.0, -1.0, -1.0])
    res = run(prices, targets)
    assert res.returns.iloc[2] == pytest.approx(10.0 / 1000.0, abs=1e-12)
    assert res.returns.iloc[2] > 0


def test_allow_short_false_rejects_negative_targets():
    prices = frame([100.0, 100.0, 90.0])
    cfg = EngineConfig(initial_capital=1000.0, allow_short=False)
    with pytest.raises(ValueError, match="short"):
        run(prices, frame([-1.0, -1.0, -1.0]), cfg=cfg)


def test_no_lookahead_via_target_shift():
    rng = np.random.default_rng(0)
    prices = frame(100 + np.cumsum(rng.normal(size=(30, 2)), axis=0))
    t0 = frame(rng.normal(size=(30, 2)))
    t = 12
    t1 = t0.copy()
    t1.iloc[t] = t0.iloc[t] * -3 + 1
    a, b = run(prices, t0), run(prices, t1)
    # target at t is filled at t+1, first earns at t+2
    pd.testing.assert_series_equal(a.returns.iloc[: t + 2], b.returns.iloc[: t + 2])
    assert a.returns.iloc[t + 2] != b.returns.iloc[t + 2]
    # and trades only differ from t+1
    assert (a.positions.iloc[: t + 1] == b.positions.iloc[: t + 1]).all().all()
    assert not (a.positions.iloc[t + 1] == b.positions.iloc[t + 1]).all()


def test_multiplier_scaling():
    rng = np.random.default_rng(1)
    prices = frame(100 + np.cumsum(rng.normal(size=(20, 1)), axis=0))
    tg = frame(rng.normal(size=(20, 1)) * 4)
    a = run(prices, tg, mult={"S0": 1.0})
    b = run(prices, tg / 2.0, mult={"S0": 2.0})
    np.testing.assert_allclose(a.returns.to_numpy(), b.returns.to_numpy(), atol=1e-12)


def test_costs_use_cost_fn_and_stress():
    rng = np.random.default_rng(2)
    prices = frame(100 + np.cumsum(rng.normal(size=(15, 2)), axis=0))
    tg = frame(rng.normal(size=(15, 2)) * 3)
    totals = {}
    for s in (1.0, 2.0, 3.0):
        fn = FlatFraction(bps=7.0)
        res = run(prices, tg, cost_fn=fn, cfg=EngineConfig(initial_capital=1000.0, stress_multiplier=s))
        assert set(fn.stresses) == {s}
        totals[s] = res.costs["total"].sum()
    assert totals[1.0] > 0
    assert totals[2.0] == pytest.approx(2 * totals[1.0], rel=1e-12)
    assert totals[3.0] == pytest.approx(3 * totals[1.0], rel=1e-12)


def test_roll_cost_charged_on_roll_dates_only():
    prices = frame([100.0] * 8)
    tg = frame([3.0] * 8)
    flags = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    flags.iloc[4, 0] = True
    fn = FlatFraction(bps=0.0, roll_bps=5.0)
    res = run(prices, tg, cost_fn=fn, roll_flags=flags)
    assert res.costs["roll"].sum() == pytest.approx(5.0 / 1e4 * 3 * 100.0, rel=1e-12)
    assert (res.costs["roll"].drop(prices.index[4]) == 0).all()
    assert res.costs.loc[prices.index[4], "roll"] > 0
    # a roll on a flat day (no position held) costs nothing
    flags2 = flags.copy()
    flags2.iloc[0, 0] = True
    res2 = run(prices, tg, cost_fn=FlatFraction(bps=0.0, roll_bps=5.0), roll_flags=flags2)
    assert res2.costs.loc[prices.index[0], "roll"] == 0


def test_gross_returns_minus_costs_equals_net():
    rng = np.random.default_rng(3)
    prices = frame(100 + np.cumsum(rng.normal(size=(40, 3)), axis=0))
    tg = frame(rng.normal(size=(40, 3)) * 2)
    res = run(prices, tg, cost_fn=FlatFraction(bps=12.0), cfg=EngineConfig(initial_capital=1000.0, borrow_rate_annual=0.02))
    eq_prev = 1000.0 * (1 + res.returns).cumprod().shift(1).fillna(1.0)
    implied = res.gross_returns - (res.costs["total"] + res.costs["borrow"]) / eq_prev
    np.testing.assert_allclose(implied.to_numpy(), res.returns.to_numpy(), atol=1e-12)
    assert res.costs["borrow"].sum() > 0


def test_ledger_required(tmp_ledger):
    prices, tg = frame([100.0, 101.0, 102.0]), frame([1.0, 1.0, 1.0])
    with pytest.raises(LedgerRequiredError):
        run_vector_backtest(prices, tg, {"S0": 1.0}, zero_cost, EngineConfig(initial_capital=1000.0))
    assert not (tmp_ledger / L.LEDGER_FILE).exists()
    res = run_vector_backtest(prices, tg, {"S0": 1.0}, zero_cost, EngineConfig(initial_capital=1000.0), exploratory=True)
    df = L.trials()
    assert len(df) == 1 and df.iloc[0]["mode"] == "exploratory"
    assert res.meta["trial_id"] == df.iloc[0]["trial_id"]


def test_ledger_ctx_records_registered_blocked_without_approval():
    prices, tg = frame([100.0, 101.0, 102.0]), frame([1.0, 1.0, 1.0])
    ctx = LedgerContext(family="fam", mode="registered", prereg="nope", data_snapshot_id="snap1", seed=7)
    with pytest.raises(L.UnapprovedPreregError):
        run_vector_backtest(prices, tg, {"S0": 1.0}, zero_cost, EngineConfig(initial_capital=1000.0), ledger_ctx=ctx)
    ctx2 = LedgerContext(family="fam", mode="exploratory", data_snapshot_id="snap1", seed=7)
    res = run_vector_backtest(prices, tg, {"S0": 1.0}, zero_cost, EngineConfig(initial_capital=1000.0), ledger_ctx=ctx2)
    row = L.trials().iloc[0]
    assert row["data_snapshot_id"] == "snap1" and row["family"] == "fam"
    assert res.meta["data_snapshot_id"] == "snap1"
    with pytest.raises(ValueError):  # exploratory flag cannot downgrade a registered context
        run_vector_backtest(prices, tg, {"S0": 1.0}, zero_cost, EngineConfig(initial_capital=1000.0), ledger_ctx=ctx, exploratory=True)


def test_engine_does_not_import_backtrader_or_live():
    path = SRC / "firm" / "backtest" / "vector_engine.py"
    banned = ("backtrader", "firm.live", "firm.api", "firm.runtime", "scripts")
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for n in names:
            assert not any(n == b or n.startswith(b + ".") for b in banned), n
    code = (
        f"import sys; sys.path.insert(0, {str(SRC)!r}); import firm.backtest.vector_engine; "
        "bad=[m for m in ('firm.live','firm.api','firm.runtime') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, check=False).returncode == 0
    init = (SRC / "firm" / "backtest" / "__init__.py").read_text()
    assert "vector_engine" not in init


def test_rf_credited_on_cash():
    n = 6
    prices = frame(np.linspace(100, 110, n))
    rf = pd.Series(0.0002, index=prices.index)
    cash = run(prices, frame([0.0] * n), rf=rf)
    np.testing.assert_allclose(cash.excess_returns.to_numpy(), 0.0, atol=1e-15)
    np.testing.assert_allclose(cash.returns.to_numpy(), 0.0002, atol=1e-15)
    # 50% invested from the start: units such that MV = 0.5 * equity at held start (lag 0, target decided at close 0)
    tg = frame([5.0] * n)  # 5 units * 100 = 500 of 1000
    inv = run(prices, tg, rf=rf, lag=0)
    r1 = prices.iloc[1, 0] / prices.iloc[0, 0] - 1
    w = 500.0 / (1000.0 * 1.0002)  # bar 0 already credited rf on the all-cash book
    assert inv.excess_returns.iloc[1] == pytest.approx(w * (r1 - 0.0002), abs=1e-12)
    # exactly 50% invested at the start of the bar when rf is credited from zero equity growth (rf=0 on bar 0)
    rf0 = rf.copy()
    rf0.iloc[0] = 0.0
    inv0 = run(prices, tg, rf=rf0, lag=0)
    assert inv0.excess_returns.iloc[1] == pytest.approx(0.5 * (r1 - 0.0002), abs=1e-12)
    assert inv.meta["rf_used"] is True
    assert run(prices, tg).meta["rf_used"] is False


def test_gross_cap_bound_share():
    prices, tg = frame(np.linspace(100, 101, 10)), frame([1.0] * 10)
    flags = pd.Series([True] * 3 + [False] * 7, index=prices.index)
    assert run(prices, tg, gross_cap_bound=flags).gross_cap_bound_days == pytest.approx(0.3)
    assert np.isnan(run(prices, tg).gross_cap_bound_days)


def test_fx_scales_pnl():
    prices = frame([100.0, 100.0, 110.0])
    fx = frame([1.0, 1.0, 1.5])
    res = run(prices, frame([1.0, 1.0, 1.0]), fx=fx)
    assert res.returns.iloc[2] == pytest.approx(10 * 1.5 / 1000.0, abs=1e-12)


def test_fractional_false_truncates_units():
    prices = frame([100.0] * 4)
    res = run(prices, frame([2.7, -2.7, 2.7, 2.7]), cfg=EngineConfig(initial_capital=1000.0, fractional={"S0": False}))
    assert list(res.positions["S0"]) == [0.0, 2.0, -2.0, 2.0]


def test_input_validation():
    prices = frame([100.0, np.nan, 101.0])
    with pytest.raises(ValueError, match="NaN"):
        run(prices, frame([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="multiplier"):
        run(frame([100.0] * 3), frame([1.0] * 3), mult={"X": 1.0})
    with pytest.raises(ValueError, match="fill_lag_bars"):
        run(frame([100.0] * 3), frame([1.0] * 3), cfg=EngineConfig(initial_capital=1000.0, fill_lag_bars=-1))


# ------------------------------------------------------------------ legacy equivalence anchor

def _legacy():
    spec = importlib.util.spec_from_file_location("legacy_alt", REPO / "scripts" / "run_alt_premia_evaluation.py")
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def _legacy_case(bps, rf_level, cash):
    legacy = _legacy()
    rng = np.random.default_rng(5)
    T, N = 60, 3
    rets = rng.normal(0.0004, 0.01, size=(T, N))
    rf = np.full(T, rf_level)
    tw = {i: rng.dirichlet(np.ones(N + 1))[:N] for i in range(0, T, 7)}  # long-only, gross <= 1
    net, _ = legacy.simulate(rets, rf, lambda i, w: tw.get(i), np.full(N, bps), cash=cash)
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=idx(T), columns=list("ABC"))
    return rets, rf, tw, net, prices


def _units_from_weights(tw, prices, rf, with_rf, bps):
    """Unit targets (indexed by decision date) that realise the legacy target WEIGHTS of post-return, pre-cost equity.
    Equity is tracked with the engine's own accounting (currency cost on post-return equity) so the weights line up."""
    P = prices.to_numpy()
    T, N = P.shape
    equity, units = 1000.0, np.zeros(N)
    out = np.zeros((T, N))
    for i in range(T):
        pnl = float(units @ (P[i] - P[i - 1])) if i else 0.0
        mv = float(units @ P[i - 1]) if i else 0.0
        equity += pnl + (rf[i] * (equity - mv) if with_rf else 0.0)
        if i in tw:
            new = tw[i] * equity / P[i]
            equity -= bps / 1e4 * float(np.abs(new - units) @ P[i])
            units = new
        out[i] = units
    return pd.DataFrame(out, index=prices.index, columns=prices.columns)


def _flat_cost(bps):
    def fn(symbol, date, qty_delta, price, adv, vol_pct, multiplier, is_roll, stress):
        return _bd(total=bps / 1e4 * abs(qty_delta) * price * multiplier)

    return fn


@pytest.mark.parametrize("with_rf", [False, True])
def test_equivalence_with_legacy_simulate_on_long_only(with_rf):
    _, rf, tw, net, prices = _legacy_case(0.0, 0.0002 if with_rf else 0.0, cash=with_rf)
    tg = _units_from_weights(tw, prices, rf, with_rf, 0.0)
    res = run(prices, tg, cfg=EngineConfig(initial_capital=1000.0, fill_lag_bars=0), rf=pd.Series(rf, index=prices.index) if with_rf else None)
    np.testing.assert_allclose(res.returns.to_numpy(), net, atol=1e-12)
    np.testing.assert_allclose(res.gross_returns.to_numpy(), net, atol=1e-12)


def test_equivalence_with_legacy_simulate_flat_bps_accounting_difference():
    """Known, documented accounting difference (not exact equivalence): (i) the engine charges currency cost on post-return equity,
    so on a rebalance day its cost fraction is the legacy one times (1 + rp); (ii) after a costed rebalance the engine keeps its
    UNITS while legacy re-bases the weights on the cost-reduced equity, a persistent O(cost * |r|) difference."""
    bps = 2.0
    _, rf, tw, net_c, prices = _legacy_case(bps, 0.0002, cash=True)
    net0 = _legacy_case(0.0, 0.0002, cash=True)[3]
    tg = _units_from_weights(tw, prices, rf, True, bps)
    res = run(prices, tg, cost_fn=_flat_cost(bps), cfg=EngineConfig(initial_capital=1000.0, fill_lag_bars=0), rf=pd.Series(rf, index=prices.index))
    assert np.max(np.abs(res.returns.to_numpy() - net_c)) < 1e-5
    # day 0 is the one day with no earlier cost history: legacy cost fraction * (1 + rp_0), to 1e-10
    rp0 = net0[0]  # zero-cost legacy return on day 0 == rp (cash earns rf)
    leg_cost0 = net0[0] - net_c[0]
    eng_cost0 = res.gross_returns.iloc[0] - res.returns.iloc[0]
    assert eng_cost0 == pytest.approx(leg_cost0 * (1.0 + rp0), abs=1e-10)
    assert eng_cost0 != pytest.approx(leg_cost0, abs=1e-12)


def test_targets_from_forecasts_wrapper():
    n = 12
    prices = frame(np.full((n, 2), 100.0))
    fc = frame(np.full((n, 2), 10.0))
    vol = frame(np.full((n, 2), 0.2))
    tg, flags = VE.targets_from_forecasts(
        fc, prices, vol, weights={"S0": 0.5, "S1": 0.5}, idm=1.0, tau=0.1, capital=1000.0,
        multipliers={"S0": 1.0, "S1": 1.0}, buffer_fraction=0.0, long_only=True, gross_cap=1.0,
    )
    # full position = 1000*1*0.5*0.1/(100*0.2) = 2.5 units each; gross = 500 / 1000 <= 1 -> not bound
    assert tg.iloc[0].tolist() == pytest.approx([2.5, 2.5])
    assert not flags.any()
    tg2, flags2 = VE.targets_from_forecasts(
        fc, prices, vol, weights={"S0": 0.5, "S1": 0.5}, idm=1.0, tau=0.1, capital=1000.0,
        multipliers={"S0": 1.0, "S1": 1.0}, buffer_fraction=0.0, long_only=True, gross_cap=0.25,
    )
    assert flags2.all()
    assert (tg2 * prices).abs().sum(axis=1).iloc[0] == pytest.approx(250.0)
