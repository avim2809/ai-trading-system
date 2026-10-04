"""P3-03 breakout forecast tests (synthetic data only)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.signals.breakout import (
    BREAKOUT_LOOKBACKS,
    FORECAST_CAP,
    breakout_forecast,
    breakout_raw,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _walk(seed: int, n: int = 600, vol: float = 1.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(rng.normal(0, vol, n)))


def _pooled_scalar(raws: dict[str, pd.Series], target_abs: float = 10.0) -> float:
    # Closed form target / pooled mean|raw|. Same maths as ewmac.estimate_pooled_scalar (P3-02, not yet
    # merged on this base); kept local so this ticket does not depend on an unmerged module.
    pooled = pd.concat(list(raws.values())).dropna().abs()
    return target_abs / float(pooled.mean())


def test_lookbacks_are_the_preregistered_set():
    assert BREAKOUT_LOOKBACKS == (20, 40, 80, 160, 320)


def test_hand_computed():
    p = pd.Series([10.0, 14.0, 11.0, 12.0, 13.0, 9.0])
    raw = breakout_raw(p, 4, smooth_span=1)
    assert raw.iloc[:3].isna().all()
    # t=3 [10,14,11,12]: mid 12 -> 0 ; t=4 [14,11,12,13]: mid 12.5, range 3 -> 40*0.5/3 ; t=5 [11,12,13,9]: mid 11, range 4 -> -20
    assert raw.iloc[3] == pytest.approx(0.0, abs=1e-12)
    assert raw.iloc[4] == pytest.approx(20.0 / 3.0, abs=1e-12)
    assert raw.iloc[5] == pytest.approx(-20.0, abs=1e-12)


@pytest.mark.parametrize("n", BREAKOUT_LOOKBACKS)
def test_range_bounds(n):
    for seed in range(100):
        raw = breakout_raw(_walk(seed, 400), n, smooth_span=1).dropna()
        assert raw.min() >= -20.0 - 1e-9 and raw.max() <= 20.0 + 1e-9


def test_sign_trend_and_reversal():
    up = pd.Series(np.linspace(100, 200, 400))
    dn = pd.Series(np.linspace(200, 100, 400))
    lags = []
    for n in BREAKOUT_LOOKBACKS:
        assert (breakout_raw(up, n).dropna() > 0).all()
        assert (breakout_raw(dn, n).dropna() < 0).all()
        rev = pd.Series(np.r_[np.linspace(100, 200, 400), np.linspace(200, 100, 400)])
        f = breakout_raw(rev, n)
        after = f.iloc[400:]
        assert f.iloc[399] > 0
        lags.append(int((after > 0).sum()))  # bars the sign stays positive after the peak
    assert lags == sorted(lags) and lags[0] < lags[-1]
    assert all(lag > 0 for lag in lags)


@pytest.mark.parametrize("n", BREAKOUT_LOOKBACKS)
def test_zero_range_is_zero_not_nan(n):
    flat = pd.Series(np.full(n + 50, 50.0))
    raw = breakout_raw(flat, n)
    assert raw.iloc[: n - 1].isna().all()
    tail = raw.iloc[n - 1 :]
    assert np.isfinite(tail).all() and (tail == 0.0).all()


@pytest.mark.parametrize("n", BREAKOUT_LOOKBACKS)
def test_smoothing_span_is_n_over_4(n):
    p = _walk(7, 500)
    unsmoothed = breakout_raw(p, n, smooth_span=1)
    expected = unsmoothed.ewm(span=max(1, n // 4), adjust=False).mean()
    pd.testing.assert_series_equal(breakout_raw(p, n), expected, atol=1e-12, rtol=0)
    # impulse response of the smoother: alpha = 2 / (span + 1)
    span = n // 4
    imp = pd.Series(np.r_[np.zeros(5), 1.0, np.zeros(30)]).ewm(span=span, adjust=False).mean()
    assert imp.iloc[6] == pytest.approx((1 - 2 / (span + 1)) * 2 / (span + 1), abs=1e-12)


def test_pooled_scalar_gives_avg_abs_10():
    for n in BREAKOUT_LOOKBACKS:
        raws = {f"i{k}": breakout_raw(_walk(k, 800, vol=0.5 + k), n) for k in range(8)}
        s = _pooled_scalar(raws)
        pooled = pd.concat([r * s for r in raws.values()]).dropna().abs()
        assert pooled.mean() == pytest.approx(10.0, abs=1e-9)


def test_no_lookahead():
    p = _walk(3, 400)
    for n in BREAKOUT_LOOKBACKS:
        full = breakout_forecast(p, n, 1.3)
        for k in (n, n + 17, 399):
            part = breakout_forecast(p.iloc[: k + 1], n, 1.3)
            assert part.iloc[-1] == full.iloc[k]


def test_cap_and_floor():
    p = pd.Series(np.r_[np.full(60, 100.0), np.linspace(100, 300, 60)])
    assert breakout_forecast(p, 20, 5.0).dropna().max() == FORECAST_CAP
    q = pd.Series(np.r_[np.full(60, 100.0), np.linspace(100, 0, 60)])
    f = breakout_forecast(q, 20, 5.0)
    assert f.min() == -FORECAST_CAP and f.max() <= FORECAST_CAP
    assert breakout_forecast(q, 20, 5.0, floor=0.0).dropna().min() == 0.0
    assert breakout_forecast(q, 20, 5.0, floor=None, cap=1e9).min() < -FORECAST_CAP


def test_unrelated_to_volatility_breakout():
    code = (
        "import sys, firm.signals.breakout\n"
        "bad = [m for m in sys.modules if m == 'firm.strategies' or m.startswith('firm.strategies.')]\n"
        "assert not bad, bad\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
