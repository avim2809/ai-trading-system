"""P3-02 EWMAC forecast tests (synthetic data only)."""

from __future__ import annotations

import inspect
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.signals.ewmac import (
    FORECAST_CAP,
    SPEEDS,
    estimate_pooled_scalar,
    ewmac_forecast,
    ewmac_raw,
    forecast_turnover,
    select_speeds,
    speed_cost_sharpe,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _walk(seed: int, n: int = 1500, vol: float = 1.0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    price = pd.Series(1000 + np.cumsum(rng.normal(0, vol, n)),
                      index=pd.bdate_range("2010-01-01", periods=n))
    sigma = pd.Series(vol, index=price.index)
    return price, sigma


def test_constants():
    assert SPEEDS == ((2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256))
    assert all(s == 4 * f for f, s in SPEEDS)
    assert FORECAST_CAP == 20.0


def test_sign_on_synthetic_trend():
    idx = pd.bdate_range("2020-01-01", periods=400)
    sigma = pd.Series(1.0, index=idx)
    up = pd.Series(100 + 0.5 * np.arange(400), index=idx)
    assert (ewmac_forecast(up, sigma, 16, 1.0).iloc[100:] > 0).all()
    assert (ewmac_forecast(-up, sigma, 16, 1.0).iloc[100:] < 0).all()
    flat = pd.Series(100.0, index=idx)
    assert ewmac_forecast(flat, sigma, 16, 1.0).abs().max() < 1e-9


def test_cap():
    idx = pd.bdate_range("2020-01-01", periods=300)
    price = pd.Series(100.0, index=idx)
    price.iloc[150:] = 1e4
    sigma = pd.Series(0.5, index=idx)
    f = ewmac_forecast(price, sigma, 8, 5.0)
    assert f.abs().max() <= 20.0 and f.max() == 20.0
    down = ewmac_forecast(-price, sigma, 8, 5.0)
    assert down.min() == -20.0
    g = ewmac_forecast(-price, sigma, 8, 5.0, floor=0.0)
    assert g.min() >= 0.0


def test_slow_is_4x_fast_default():
    price, sigma = _walk(1)
    pd.testing.assert_series_equal(ewmac_raw(price, sigma, 8), ewmac_raw(price, sigma, 8, slow=32))
    assert not ewmac_raw(price, sigma, 8, slow=16).equals(ewmac_raw(price, sigma, 8))


def test_pooled_scalar_gives_avg_abs_10():
    raws = {}
    for i in range(8):
        price, sigma = _walk(i, vol=0.5 + i)
        raws[f"I{i}"] = ewmac_raw(price, sigma * (1 + 0.3 * i), 16)
    start, end = pd.Timestamp("2011-01-01"), pd.Timestamp("2015-06-30")
    s = estimate_pooled_scalar(raws, 10.0, start, end)
    pooled = pd.concat([r.loc[start:end] for r in raws.values()]).dropna().abs()
    assert abs(float((pooled * s).mean()) - 10.0) < 1e-9
    capped = pd.concat([(r.loc[start:end] * s).clip(-20, 20) for r in raws.values()]).dropna().abs()
    assert float(capped.mean()) <= 10.0 + 1e-12
    per_inst = [float((r.loc[start:end].dropna().abs() * s).mean()) for r in raws.values()]
    assert max(per_inst) - min(per_inst) > 0.1


def test_scalar_uses_only_research_window():
    price, sigma = _walk(3)
    raws = {"A": ewmac_raw(price, sigma, 16)}
    start, end = price.index[300], price.index[900]
    s0 = estimate_pooled_scalar(raws, 10.0, start, end)
    idx2 = pd.bdate_range(price.index[-1] + pd.Timedelta(days=1), periods=500)
    ext = pd.concat([raws["A"], pd.Series(1e6, index=idx2)])
    assert estimate_pooled_scalar({"A": ext}, 10.0, start, end) == s0
    with pytest.raises(ValueError):
        estimate_pooled_scalar(raws, 10.0, start, price.index[-1] + pd.Timedelta(days=30))
    with pytest.raises(ValueError):
        estimate_pooled_scalar(raws, 10.0, None, end)
    with pytest.raises(ValueError):
        estimate_pooled_scalar(raws, 10.0, end, start)


def test_scalar_guards_against_sealed_end():
    price, sigma = _walk(3)
    raws = {"A": ewmac_raw(price, sigma, 16)}
    with pytest.raises(ValueError):
        estimate_pooled_scalar(raws, 10.0, price.index[300], price.index[900],
                               max_research_date=price.index[500])


def test_no_lookahead():
    price, sigma = _walk(4, n=700)
    full = ewmac_forecast(price, sigma, 8, 7.0)
    for k in (120, 400, 650):
        part = ewmac_forecast(price.iloc[: k + 1], sigma.iloc[: k + 1], 8, 7.0)
        assert part.iloc[-1] == full.iloc[k]


def test_nan_price_skipped_not_filled():
    price, sigma = _walk(5, n=300)
    price.iloc[100] = np.nan
    f = ewmac_raw(price, sigma, 8)
    assert np.isnan(f.iloc[100]) and np.isfinite(f.iloc[101])


def test_speed_selection_per_instrument():
    cand = {
        "SPY": {sp: {"turnover": t, "cost_per_trade": 0.0002, "sigma_pct": 0.16}
                for sp, t in zip(SPEEDS, (30, 20, 12, 7, 4, 2))},
        "THIN": {sp: {"turnover": t, "cost_per_trade": 0.003, "sigma_pct": 0.08}
                 for sp, t in zip(SPEEDS, (30, 20, 12, 7, 4, 2))},
    }
    out = select_speeds(cand, expected_rule_sharpe=0.3)
    assert set(out) == {"SPY", "THIN"}
    assert out["SPY"] != out["THIN"]
    assert len(out["SPY"]) > len(out["THIN"])


def test_speed_selection_by_cost_only():
    sp = list(SPEEDS)
    turnovers = [20.0, 12.0, 8.0, 5.0, 3.0, 2.0]
    cand = {"X": {s: {"turnover": t, "cost_per_trade": 0.001, "sigma_pct": 0.1}
                  for s, t in zip(sp, turnovers)}}
    # annual cost in Sharpe units = turnover * 0.001 / 0.1 = turnover / 100
    out = select_speeds(cand, expected_rule_sharpe=0.3)  # limit 0.1 -> turnover <= 10
    assert out["X"] == sp[2:]
    prev = -1
    for sh in (0.1, 0.2, 0.3, 0.6, 1.2):
        n = len(select_speeds(cand, sh)["X"])
        assert n >= prev
        prev = n
    params = set(inspect.signature(select_speeds).parameters)
    assert params == {"candidates", "expected_rule_sharpe", "max_cost_fraction"}
    assert speed_cost_sharpe(10.0, 0.001, 0.1) == pytest.approx(0.1)


def test_speed_selection_bad_inputs():
    with pytest.raises(ValueError):
        speed_cost_sharpe(1.0, 0.001, 0.0)
    with pytest.raises(ValueError):
        select_speeds({}, expected_rule_sharpe=0.0)


def test_turnover_monotone_in_speed():
    price, sigma = _walk(7, n=4000)
    ts = []
    for fast, slow in SPEEDS:
        f = ewmac_forecast(price, sigma, fast, 1.0, slow=slow)
        ts.append(forecast_turnover(f))
    assert all(a > b for a, b in itertools.pairwise(ts)), ts


def test_forecast_turnover_formula():
    f = pd.Series([0.0, 10.0, 0.0, 10.0])
    assert forecast_turnover(f) == pytest.approx(256 * 10.0 / (20.0 / 4))


def test_signatures_have_no_pnl_input():
    for fn in (ewmac_raw, ewmac_forecast, estimate_pooled_scalar, select_speeds):
        names = " ".join(inspect.signature(fn).parameters)
        assert "return" not in names and "pnl" not in names and "sharpe_realised" not in names


def test_import_light():
    code = (
        "import sys, firm.signals.ewmac;"
        "bad=[m for m in sys.modules if m.split('.')[:2] in "
        "(['firm','live'],['firm','api'],['firm','runtime'],['firm','research'])];"
        "print('BAD:'+','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env={"PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "BAD:" in out.stdout.splitlines(), out.stdout
