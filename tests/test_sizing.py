"""P3-06 position sizing, buffering and IDM tests (synthetic data only)."""

from __future__ import annotations

import inspect
import itertools
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from firm.portfolio import sizing
from firm.portfolio.sizing import (
    DEFAULT_BUFFER_FRACTION,
    IDM_CAP,
    buffer_width,
    buffered_trade,
    full_position,
    gross_cap_scale,
    idm,
    round_position,
    target_position,
)

SRC = Path(__file__).resolve().parents[1] / "src"
BASE = {"capital": 100000.0, "idm": 1.0, "weight": 0.1, "tau": 0.10, "multiplier": 1.0, "price": 100.0, "fx": 1.0, "vol_pct": 0.20}


def test_constants():
    assert IDM_CAP == 2.5 and DEFAULT_BUFFER_FRACTION == 0.10


def test_target_position_hand_computed():
    assert target_position(10.0, **BASE) == pytest.approx(50.0, abs=1e-12)
    assert target_position(20.0, **BASE) == pytest.approx(100.0, abs=1e-12)
    assert target_position(0.0, **BASE) == 0.0
    assert full_position(**BASE) == pytest.approx(50.0, abs=1e-12)


def test_buffer_is_10pct_of_full_position():
    assert abs(buffer_width(**BASE) - 0.1 * full_position(**BASE)) < 1e-12
    assert abs(buffer_width(**BASE) - 0.1 * target_position(10.0, **BASE)) < 1e-12
    assert buffer_width(**BASE, fraction=0.2) == pytest.approx(10.0)


def test_buffered_trade_edges():
    # N = 50, B = 5 -> band [45, 55]
    assert buffered_trade(50.0, 50.0, 5.0) == 50.0
    assert buffered_trade(45.0, 50.0, 5.0) == 45.0
    assert buffered_trade(55.0, 50.0, 5.0) == 55.0
    assert buffered_trade(60.0, 50.0, 5.0) == 55.0
    assert buffered_trade(30.0, 50.0, 5.0) == 45.0
    # long/flat: lower edge clamped at 0
    assert buffered_trade(-3.0, 2.0, 5.0) == 0.0
    assert buffered_trade(0.0, 2.0, 5.0) == 0.0
    assert buffered_trade(3.0, 2.0, 5.0) == 3.0
    # signed variant keeps the negative lower edge
    assert buffered_trade(-9.0, 2.0, 5.0, long_only=False) == -3.0
    with pytest.raises(ValueError):
        buffered_trade(0.0, 1.0, -1.0)


def test_turnover_falls_monotonically_with_buffer():
    rng = np.random.default_rng(11)
    f = np.clip(10 + np.cumsum(rng.normal(0, 2.0, 2000)), 0, 20)
    totals = []
    for frac in (0.0, 0.05, 0.1, 0.2):
        pos, traded = 0.0, 0.0
        for fc in f:
            n = target_position(fc, **BASE)
            b = buffer_width(**BASE, fraction=frac)
            new = buffered_trade(pos, n, b)
            traded += abs(new - pos)
            pos = new
        totals.append(traded)
    assert all(a > b for a, b in itertools.pairwise(totals)), totals


def test_integer_rounding_futures():
    kw = {**BASE, "multiplier": 50.0, "price": 4000.0, "vol_pct": 0.15, "capital": 1_000_000.0, "weight": 0.2, "tau": 0.12}
    n = target_position(10.0, **kw)
    b = buffer_width(**kw)
    assert n == pytest.approx(1_000_000 * 0.2 * 0.12 / (50 * 4000 * 0.15))  # 0.8 contracts
    pos = round_position(buffered_trade(0.0, 3.4 * 1.0, 0.2), False, 50.0, "nearest")
    assert pos == 3.0 and isinstance(pos, float)
    assert round_position(buffered_trade(0.0, n, b), False, 50.0, "nearest") == 1.0
    # rounding after buffering: current 3 (rounded), target 3.45 B 0.5 -> inside band, no trade
    cur = 3.0
    assert round_position(buffered_trade(cur, 3.45, 0.5), False, 50.0, "nearest") == cur


def test_etf_rounds_toward_zero():
    # literal table mirroring Allocator._qty_toward_zero (floor |q| to 6 dp / floor(|q|+1e-9))
    frac = [(1.23456789, 1.234567), (0.9999999, 0.999999), (2.0, 2.0), (0.0000004, 0.0), (-1.23456789, -1.234567)]
    for q, want in frac:
        assert round_position(q, True) == want
    whole = [(3.99, 3.0), (3.9999999999, 4.0), (0.99, 0.0), (7.0, 7.0), (-3.99, -3.0)]
    for q, want in whole:
        assert round_position(q, False) == want
    assert round_position(2.7, False, 1.0, "toward_zero") == 2.0
    with pytest.raises(ValueError):
        round_position(2.7, False, 1.0, "nearest")  # nearest is futures only (multiplier > 1)
    with pytest.raises(ValueError):
        round_position(2.7, False, 1.0, "bogus")
    assert round_position(2.7, False, 1.0, "none") == 2.7


def test_fractional_etf_allowed():
    assert round_position(1.5, True) == 1.5
    assert round_position(1.5, False) == 1.0
    with pytest.raises(ValueError):
        round_position(1.5, True, 50.0)


def test_toward_zero_matches_allocator_helper():
    # the same arithmetic as src/firm/allocation/allocator.py _qty_toward_zero, spelled out
    for q in np.random.default_rng(3).uniform(0, 50, 200):
        assert round_position(q, True) == math.floor(q * 1e6) / 1e6
        assert round_position(q, False) == float(math.floor(q + 1e-9))


def test_idm_hand_computed_and_cap():
    h = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert abs(idm(np.array([0.5, 0.5]), h) - 1 / math.sqrt(0.75)) < 1e-12
    assert idm(np.full(20, 0.05), np.eye(20)) == IDM_CAP
    neg = np.array([[1.0, -0.5], [-0.5, 1.0]])
    assert idm(np.array([0.5, 0.5]), neg) == pytest.approx(math.sqrt(2))


def test_gross_cap_scale_flag():
    out, bound = gross_cap_scale({"A": 0.7, "B": 0.6})
    assert bound is True and sum(out.values()) == pytest.approx(1.0)
    assert out["A"] / out["B"] == pytest.approx(0.7 / 0.6)
    out2, bound2 = gross_cap_scale({"A": 0.5, "B": 0.3})
    assert bound2 is False and out2 == {"A": 0.5, "B": 0.3}
    _, b3 = gross_cap_scale({"A": 1.0})
    assert b3 is False
    with pytest.raises(ValueError):
        gross_cap_scale({"A": 1.0}, cap=0.0)


@pytest.mark.parametrize("key", ["price", "vol_pct", "multiplier", "fx"])
def test_zero_vol_or_price_raises(key):
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            target_position(10.0, **{**BASE, key: bad})
        with pytest.raises(ValueError):
            buffer_width(**{**BASE, key: bad})


def test_no_alpha_input_and_tau_has_no_default():
    for name in ("full_position", "target_position", "buffer_width", "buffered_trade", "round_position", "idm",
                 "gross_cap_scale"):
        params = inspect.signature(getattr(sizing, name)).parameters
        assert not any(k in p for p in params for k in ("return", "pnl", "sharpe"))
    for name in ("full_position", "target_position", "buffer_width"):
        assert inspect.signature(getattr(sizing, name)).parameters["tau"].default is inspect.Parameter.empty


def test_import_light():
    code = (
        "import sys, firm.portfolio.sizing;"
        "bad=[m for m in sys.modules if m.split('.')[:2] in "
        "(['firm','live'],['firm','api'],['firm','runtime'])];"
        "print('BAD:'+','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env={"PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "BAD:" in out.stdout.splitlines(), out.stdout
    assert "sizing" not in (SRC / "firm" / "portfolio" / "__init__.py").read_text()
