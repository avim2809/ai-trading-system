"""P3-04 carry forecast tests (synthetic fixtures only; no data files)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.signals import carry
from firm.signals.carry import (
    CARRY_SMOOTH_SPANS,
    carry_forecast,
    raw_carry,
    years_between_contracts,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _s(vals):
    return pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=len(vals)), dtype=float)


def test_backwardation_positive():
    near, far = _s([105.0] * 30), _s([100.0] * 30)
    sig = _s([10.0] * 30)
    assert (carry_forecast(near, far, 0.25, sig, span=5, scalar=1.0) > 0).all()
    assert (carry_forecast(far, near, 0.25, sig, span=5, scalar=1.0) < 0).all()


def test_hand_computed():
    r = raw_carry(_s([105.0]), _s([100.0]), 0.25, _s([10.0]))
    assert r.iloc[0] == pytest.approx(2.0, abs=1e-12)


def test_longer_gap_lower_carry():
    near, far, sig = _s([105.0] * 5), _s([100.0] * 5), _s([10.0] * 5)
    short = raw_carry(near, far, 0.25, sig).abs()
    long = raw_carry(near, far, 1.0, sig).abs()
    assert (long < short).all()


def test_years_series_supported():
    near, far, sig = _s([105.0, 105.0]), _s([100.0, 100.0]), _s([10.0, 10.0])
    r = raw_carry(near, far, _s([0.25, 0.5]), sig)
    assert r.iloc[0] == pytest.approx(2.0) and r.iloc[1] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [0.0, -0.1, np.inf])
def test_rejects_nonpositive_years(bad):
    s = _s([1.0, 1.0])
    with pytest.raises(ValueError):
        raw_carry(s, s, bad, s)
    with pytest.raises(ValueError):
        raw_carry(s, s, _s([0.25, bad]), s)


def test_years_between_contracts():
    e1 = pd.Series(pd.to_datetime(["2020-03-20", "2020-03-20"]))
    e2 = pd.Series(pd.to_datetime(["2020-06-19", "2021-03-19"]))
    y = years_between_contracts(e1, e2)
    assert y.iloc[0] == pytest.approx(91 / 365.25)
    assert y.iloc[1] == pytest.approx(364 / 365.25)
    with pytest.raises(ValueError):
        years_between_contracts(e2, e1)
    with pytest.raises(ValueError):
        years_between_contracts(e1, e1)


def test_smoothing_spans():
    n = 300
    spread = np.zeros(n)
    spread[10] = 1.0  # impulse in (near - far)
    near, far, sig = _s(spread), _s(np.zeros(n)), _s(np.ones(n))
    assert CARRY_SMOOTH_SPANS == (5, 20, 60, 120)
    for s in CARRY_SMOOTH_SPANS:
        f = carry_forecast(near, far, 1.0, sig, span=s, scalar=1.0, cap=1e9, floor=None)
        expected = near.ewm(span=s, adjust=False).mean()
        np.testing.assert_allclose(f.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_cap_floor_and_scalar():
    near = _s(100.0 + 50 * np.sign(np.sin(np.arange(200) / 30)))
    far, sig = _s([100.0] * 200), _s([1.0] * 200)
    f = carry_forecast(near, far, 0.25, sig, span=5, scalar=10.0)
    assert f.max() == 20.0 and f.min() == -20.0
    flat = carry_forecast(near, far, 0.25, sig, span=5, scalar=10.0, floor=0.0)
    assert flat.min() == 0.0 and flat.max() == 20.0
    small = carry_forecast(_s([100.5] * 50), far.iloc[:50], 1.0, sig.iloc[:50], span=5, scalar=3.0)
    assert small.iloc[-1] == pytest.approx(1.5)


def test_nan_is_skipped_not_filled():
    near = _s([105.0, np.nan, 105.0, 105.0])
    far, sig = _s([100.0] * 4), _s([10.0] * 4)
    f = carry_forecast(near, far, 1.0, sig, span=5, scalar=1.0)
    assert np.isnan(f.iloc[1]) and f.iloc[[0, 2, 3]].notna().all()
    z = carry_forecast(near, far, 1.0, _s([10.0, 0.0, 10.0, 10.0]), span=5, scalar=1.0)
    assert np.isnan(z.iloc[1])


def test_no_lookahead():
    rng = np.random.default_rng(1)
    n, k = 400, 250
    near = _s(100 + rng.normal(size=n).cumsum())
    far = near - _s(rng.normal(size=n))
    sig = _s(np.abs(rng.normal(size=n)) + 1.0)
    full = carry_forecast(near, far, 0.25, sig, span=20, scalar=2.0)
    trunc = carry_forecast(near.iloc[: k + 1], far.iloc[: k + 1], 0.25, sig.iloc[: k + 1], span=20, scalar=2.0)
    assert full.iloc[k] == trunc.iloc[k]
    pd.testing.assert_series_equal(full.iloc[: k + 1], trunc)


def test_import_light():
    code = (
        "import sys, firm.signals.carry;"
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


def test_no_etf_proxy_symbols():
    public = [n for n in dir(carry) if not n.startswith("_")]
    assert not [n for n in public if any(w in n.lower() for w in ("etf", "proxy", "yield", "dividend"))]
    assert set(public) >= {"raw_carry", "carry_forecast", "years_between_contracts"}
