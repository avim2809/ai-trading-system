"""P3-01 blended EWMA vol estimator: synthetic and hand-computed fixtures only."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.signals.vol import (
    TRADING_DAYS_PER_YEAR,
    ewma_std,
    ewma_vol,
    expanding_percentile_floor,
    long_run_mean,
    price_unit_vol,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _rets(n, sigma=0.01, seed=0, start="2000-01-03"):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, sigma, n), index=pd.bdate_range(start, periods=n))


def test_constant():
    assert TRADING_DAYS_PER_YEAR == 256


def test_hand_computed_small_case():
    # span=3 -> alpha = 2/(3+1) = 0.5, adjust=False, zero-mean:
    #   s0 = r0^2 = 1e-4
    #   s1 = .5*1e-4 + .5*4e-4 = 2.5e-4
    #   s2 = .5*2.5e-4 + .5*9e-4 = 5.75e-4
    #   s3 = .5*5.75e-4 + .5*1.6e-3 = 1.0875e-3
    #   s4 = .5*1.0875e-3 + .5*2.5e-3 = 1.79375e-3
    #   s5 = .5*1.79375e-3 + .5*0 = 8.96875e-4
    r = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.0], index=pd.bdate_range("2020-01-01", periods=6))
    expect = np.sqrt([1e-4, 2.5e-4, 5.75e-4, 1.0875e-3, 1.79375e-3, 8.96875e-4])
    got = ewma_std(r, span=3)
    np.testing.assert_allclose(got.to_numpy(), expect, atol=1e-12, rtol=0)
    v = ewma_vol(r, span=3, blend_long_weight=0.0, floor_percentile=0.0, min_obs=1, annualise=False)
    np.testing.assert_allclose(v.to_numpy(), expect, atol=1e-12, rtol=0)
    va = ewma_vol(r, span=3, blend_long_weight=0.0, floor_percentile=0.0, min_obs=1)
    np.testing.assert_allclose(va.to_numpy(), expect * 16, atol=1e-12, rtol=0)


def test_blend_70_30():
    r = pd.concat([_rets(400, 0.005, 1), _rets(400, 0.02, 2, start="2002-01-01")], ignore_index=True)
    r.index = pd.bdate_range("2000-01-03", periods=len(r))
    fast = np.sqrt(r.pow(2).ewm(span=35, adjust=False).mean())
    long_ = fast.expanding().mean()  # n < 2520 -> expanding
    blend = 0.7 * fast + 0.3 * long_
    floor = blend.expanding(min_periods=256).quantile(0.05)
    expect = np.maximum(blend, floor) * 16
    got = ewma_vol(r)
    assert got.iloc[:255].isna().all() and got.iloc[255:].notna().all()
    np.testing.assert_allclose(got.iloc[255:], expect.iloc[255:], atol=1e-12, rtol=0)
    # no floor binding: pure blend check on the part where blend > floor
    mask = (blend > floor).to_numpy() & (np.arange(len(r)) >= 255)
    np.testing.assert_allclose(got[mask], (blend * 16)[mask], atol=1e-12, rtol=0)


def test_no_lookahead():
    r = _rets(700, 0.01, 3)
    r.iloc[600:610] = 0.15  # late spike
    r.iloc[300:380] *= 0.1  # quiet regime, moves the floor
    full = ewma_vol(r)
    rng = np.random.default_rng(7)
    for k in rng.integers(256, 699, size=20):
        part = ewma_vol(r.iloc[: k + 1])
        assert part.iloc[k] == full.iloc[k]
        pd.testing.assert_series_equal(part, full.iloc[: k + 1])


def test_floor_is_expanding_not_global():
    base = _rets(500, 0.01, 4)
    early = ewma_vol(base)
    low = _rets(500, 0.0005, 5, start="2002-01-01")
    low.index = pd.bdate_range(base.index[-1] + pd.Timedelta(days=1), periods=500)
    ext = ewma_vol(pd.concat([base, low]))
    pd.testing.assert_series_equal(ext.iloc[:500], early)
    s = pd.Series(np.r_[np.full(300, 0.01), np.full(300, 0.001)], index=pd.bdate_range("2000-01-03", periods=600))
    fl = expanding_percentile_floor(s, 0.05, 256)
    assert fl.iloc[:255].isna().all()
    assert (fl.iloc[255:300] == 0.01).all()
    assert fl.iloc[-1] < 0.01


def test_floor_matches_numpy_quantile_prefix():
    s = pd.Series(np.abs(np.random.default_rng(1).normal(0.01, 0.003, 400)), index=pd.bdate_range("2000-01-03", periods=400))
    fl = expanding_percentile_floor(s, 0.05, 256)
    for k in (255, 300, 399):
        assert fl.iloc[k] == pytest.approx(np.quantile(s.iloc[: k + 1], 0.05), abs=1e-15)


def test_warmup_expanding_until_2520():
    n = 2700
    r = _rets(n, 0.01, 6)
    fast = ewma_std(r, 35)
    lr = long_run_mean(fast, 2520, "expanding")
    exp = fast.expanding().mean()
    np.testing.assert_allclose(lr.iloc[:2519], exp.iloc[:2519], atol=1e-15, rtol=0)
    # at n == 2520 the window is exactly full: expanding == rolling there, and afterwards rolling takes over
    assert lr.iloc[2519] == pytest.approx(fast.iloc[:2520].mean(), abs=1e-15)
    assert lr.iloc[2600] == pytest.approx(fast.iloc[81:2601].mean(), abs=1e-15)
    assert lr.iloc[2600] != pytest.approx(exp.iloc[2600], abs=1e-12)
    ln = long_run_mean(fast, 2520, "nan")
    assert ln.iloc[:2519].isna().all() and ln.iloc[2519] == lr.iloc[2519]
    v = ewma_vol(r, warmup="nan")
    assert v.iloc[:2519].isna().all() and v.iloc[2519:].notna().all()
    ve = ewma_vol(r)
    assert ve.iloc[:255].isna().all() and ve.iloc[255:].notna().all()


def test_counts_valid_observations_not_rows():
    r = _rets(400, 0.01, 8)
    r.iloc[::2] = np.nan  # 200 valid obs in 400 rows
    assert ewma_vol(r).isna().all()
    r2 = _rets(520, 0.01, 8)
    r2.iloc[::2] = np.nan  # 260 valid
    v = ewma_vol(r2)
    valid_pos = np.flatnonzero(r2.notna().to_numpy())
    assert v.iloc[valid_pos[:255]].isna().all()
    assert v.iloc[valid_pos[255]] == v.iloc[valid_pos[255]]
    # long window counts valid obs too
    f = pd.Series(np.arange(1.0, 11.0))
    f.iloc[[2, 3]] = np.nan
    lr = long_run_mean(f, 3, "expanding")
    assert lr.iloc[4] == pytest.approx(np.mean([1.0, 2.0, 5.0]))


def test_nan_returns_skipped_not_zeroed():
    r = _rets(800, 0.01, 9)
    holes = r.index[[10, 50, 51, 300, 301, 302, 500, 700]]
    rn = r.copy()
    rn.loc[holes] = np.nan
    a = ewma_vol(rn)
    b = ewma_vol(rn.dropna())
    common = b.index
    np.testing.assert_allclose(a.loc[common].to_numpy(), b.to_numpy(), atol=1e-12, rtol=0, equal_nan=True)
    # zero-filling or ffill must differ
    z = ewma_vol(rn.fillna(0.0))
    both = z.loc[common].notna() & b.notna()
    assert not np.allclose(z.loc[common][both].to_numpy(), b[both].to_numpy(), atol=1e-12)
    # NaN return rows get NaN vol (never forward-filled)
    assert a.loc[holes].isna().all()


def test_scale_annualisation():
    r = _rets(20000, 0.01, 10)
    v = ewma_vol(r)
    assert v.dropna().mean() == pytest.approx(0.16, rel=0.05)
    assert ewma_vol(r, annualise=False).dropna().mean() == pytest.approx(0.01, rel=0.05)


def test_price_unit_vol():
    p = pd.Series([100.0, 101.0, 99.0], index=pd.bdate_range("2020-01-01", periods=3))
    v = pd.Series([0.16, 0.32, np.nan], index=p.index)
    out = price_unit_vol(p, v)
    assert out.iloc[0] == pytest.approx(1.0) and out.iloc[1] == pytest.approx(2.02)
    assert np.isnan(out.iloc[2])


@pytest.mark.parametrize(
    "kw",
    [
        {"span": 1},
        {"blend_long_weight": 1.5},
        {"floor_percentile": 1.0},
        {"min_obs": 0},
        {"warmup": "ffill"},
        {"long_window_days": 0},
    ],
)
def test_bad_args_raise(kw):
    with pytest.raises(ValueError):
        ewma_vol(_rets(300), **kw)


def test_infinite_returns_raise():
    r = _rets(300)
    r.iloc[5] = np.inf
    with pytest.raises(ValueError):
        ewma_vol(r)


def test_import_light():
    code = (
        "import sys, firm.signals.vol;"
        "bad=[m for m in sys.modules if m.split('.')[:2] in (['firm','live'],['firm','api'],['firm','runtime'])];"
        "print('BAD:'+','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env={"PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "BAD:" in out.stdout.splitlines(), out.stdout
