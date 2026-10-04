"""P4-05: fractional-Kelly sanity bound."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from firm.risk.kelly import KellyCheck, check_tau, kelly_vol, max_tau, planning_sharpe
from firm.validation.sharpe_stats import PerPeriodSharpe, SharpeUnitsError

SRC = Path(__file__).resolve().parents[1] / "src"


def test_known_example():
    assert planning_sharpe(0.4, 0.5) == 0.2
    assert kelly_vol(0.2) == 0.2
    assert max_tau(0.4, 0.5) == 0.10
    chk = check_tau(0.12, 0.4, 0.5)
    assert isinstance(chk, KellyCheck)
    assert (chk.tau, chk.max_tau, chk.ok, chk.planning_sharpe) == (0.12, 0.10, False, 0.2)
    assert check_tau(0.08, 0.4, 0.5).ok


def test_negative_sharpe_gives_zero_bound():
    assert planning_sharpe(-0.3, 0.5) == 0.0
    assert max_tau(-0.3, 0.5) == 0.0
    assert max_tau(0.0, 0.5) == 0.0
    assert not check_tau(0.01, -0.3, 0.5).ok
    assert check_tau(0.0, -0.3, 0.5).ok


def test_per_period_sharpe_rejected():
    with pytest.raises(SharpeUnitsError):
        planning_sharpe(PerPeriodSharpe(0.03), 0.5)
    with pytest.raises(SharpeUnitsError):
        max_tau(PerPeriodSharpe(0.03), 0.5)
    with pytest.raises(SharpeUnitsError):
        check_tau(0.1, PerPeriodSharpe(0.03), 0.5)
    with pytest.raises(SharpeUnitsError):  # implausibly large even for an annual Sharpe
        max_tau(40.0, 0.5)


def test_not_double_deflated():
    for sr in (0.1, 0.4, 0.9, 1.7):
        for h in (0.5, 0.6, 0.9):
            for f in (0.25, 0.5, 1.0):
                assert max_tau(sr, h, f) == f * (1 - h) * sr


def test_monotone_in_sharpe_and_haircut():
    srs = np.linspace(-0.5, 2.0, 26)
    hs = np.linspace(0.5, 0.95, 10)
    for h in hs:
        vals = [max_tau(s, h) for s in srs]
        assert all(a <= b for a, b in itertools.pairwise(vals))
    for s in srs:
        vals = [max_tau(s, h) for h in hs]
        assert all(a >= b for a, b in itertools.pairwise(vals))


def test_haircut_below_charter_minimum_raises():
    with pytest.raises(ValueError, match="haircut"):
        planning_sharpe(0.4, 0.49)
    with pytest.raises(ValueError):
        max_tau(0.4, 0.3)
    with pytest.raises(ValueError):
        check_tau(0.1, 0.4, 0.0)
    assert planning_sharpe(0.4, 0.3, min_haircut=0.25) == pytest.approx(0.28)
    for bad in (-0.1, 1.0, 1.5, float("nan")):
        with pytest.raises(ValueError):
            planning_sharpe(0.4, bad, min_haircut=0.0)


def test_kelly_fraction_scales_linearly():
    base = max_tau(0.8, 0.5, 0.5)
    assert max_tau(0.8, 0.5, 0.25) == pytest.approx(base / 2)
    assert max_tau(0.8, 0.5, 1.0) == pytest.approx(base * 2)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            max_tau(0.8, 0.5, bad)


def test_check_tau_boundary_inclusive():
    m = max_tau(0.4, 0.5)
    assert check_tau(m, 0.4, 0.5).ok
    assert not check_tau(np.nextafter(m, 1.0), 0.4, 0.5).ok
    with pytest.raises(ValueError):
        check_tau(-0.01, 0.4, 0.5)


def test_no_pipeline_kelly_import(tmp_path):
    code = (
        "import sys, firm.risk.kelly\n"
        "bad = [m for m in sys.modules if m == 'firm.agents' or m.startswith('firm.agents.')"
        " or m == 'firm.portfolio.optimizer']\n"
        "print('BAD:' + ','.join(bad))\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "PYTHONDONTWRITEBYTECODE": "1",
        "FIRM_DATA_DIR": str(tmp_path),
    }
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "BAD:" in out.stdout.splitlines(), out.stdout
