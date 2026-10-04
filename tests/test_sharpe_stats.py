"""P1-02: PSR / DSR / MinTRL summary-statistic API."""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from firm.eval.overfitting import deflated_sharpe, probabilistic_sharpe
from firm.validation.sharpe_stats import (
    DegenerateInputError,
    SharpeUnitsError,
    check_per_period,
    dsr,
    expected_max_sr,
    length_adjusted_var_sr,
    min_trl,
    moments,
    psr,
    sr_sampling_var,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _series(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(60, 2001))
    df = float(rng.choice([3.5, 5.0, 8.0, 1000.0]))
    x = rng.standard_t(df, n) * 0.01
    if seed % 3 == 0:  # negative skew: occasional crashes
        x = x - (rng.random(n) < 0.02) * 0.05
    return x + float(rng.uniform(-0.0003, 0.0012))


def test_psr_matches_legacy_probabilistic_sharpe():
    for seed in range(50):
        r = _series(seed)
        sr, skew, kurt, n = moments(r)
        assert n == len(r)
        for star in (0.0, 0.5 * sr, -0.02):
            assert psr(sr, star, n, skew, kurt) == pytest.approx(
                probabilistic_sharpe(r, star), abs=1e-12
            )


def test_dsr_matches_legacy_deflated_sharpe():
    rng = np.random.default_rng(7)
    for seed in range(50):
        r = _series(seed)
        sr, skew, kurt, n = moments(r)
        n_trials = int(rng.integers(2, 301))
        trials = rng.normal(0.0, 0.05, n_trials)
        var = float(np.var(trials, ddof=1))
        assert dsr(sr, n, skew, kurt, n_trials, var) == pytest.approx(
            deflated_sharpe(r, trials), abs=1e-12
        )
        k = int(rng.integers(1, 200))  # legacy prior_trials path
        assert dsr(sr, n, skew, kurt, n_trials + k, var) == pytest.approx(
            deflated_sharpe(r, trials, prior_trials=k), abs=1e-12
        )


def test_paper_examples_1e_3():
    # Bailey and Lopez de Prado (2014): N=100, var of SR 0.5/250, T=1250, annual SR 2.5,
    # skew -3, raw kurtosis 10. Hard-coded published values.
    assert expected_max_sr(100, 0.5 / 250) == pytest.approx(0.1132, abs=1e-3)
    assert dsr(2.5 / math.sqrt(250), 1250, -3.0, 10.0, 100, 0.5 / 250) == pytest.approx(
        0.9004, abs=1e-3
    )


@pytest.mark.parametrize("n_trials", [10, 100, 1000])
def test_expected_max_sr_monte_carlo(n_trials):
    var = 0.0004
    rng = np.random.default_rng(1234 + n_trials)
    reps, chunk = 20000, 2000
    maxima = np.concatenate(
        [
            rng.normal(0.0, math.sqrt(var), (chunk, n_trials)).max(axis=1)
            for _ in range(reps // chunk)
        ]
    )
    assert maxima.mean() == pytest.approx(expected_max_sr(n_trials, var), rel=0.05)


def test_min_trl():
    sr, star, skew, kurt = 0.5 / math.sqrt(252), 0.0, -0.4, 5.0
    rad = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    expected = 1 + rad * (1.6448536269514722 / (sr - star)) ** 2
    assert min_trl(sr, star, skew, kurt) == pytest.approx(expected, rel=1e-6)
    # about 11 years at annual SR 0.5, normal returns, alpha 0.05
    years = min_trl(sr, 0.0, 0.0, 3.0) / 252
    assert years == pytest.approx(11.0, rel=0.15)
    grid = [0.02, 0.03, 0.04, 0.06, 0.1]
    vals = [min_trl(s, 0.0, -0.5, 5.0) for s in grid]
    assert vals == sorted(vals, reverse=True)
    sk = [min_trl(0.05, 0.0, s, 5.0) for s in (-3.0, -2.0, -1.0, 0.0)]
    assert sk == sorted(sk, reverse=True)  # more negative skew needs a longer record
    with pytest.raises(DegenerateInputError):
        min_trl(0.05, 0.05, 0.0, 3.0)
    with pytest.raises(DegenerateInputError):
        min_trl(0.03, 0.05, 0.0, 3.0)


def test_length_adjusted_var_sr():
    rng = np.random.default_rng(99)
    k = 400
    t = np.exp(rng.uniform(math.log(50), math.log(2000), k)).astype(int)
    srs = np.array([moments(rng.normal(0.0, 0.01, ti))[0] for ti in t])
    n_cand = 1000
    cand = sr_sampling_var(0.0, n_cand)
    samp_mean = float(np.mean(1.0 / (t - 1)))
    se = math.sqrt(2.0 / (k - 1)) * samp_mean  # Monte Carlo error of the cross-trial variance
    adj = length_adjusted_var_sr(srs, t, n_cand)
    assert abs(adj - cand) < 4 * se
    assert float(np.var(srs, ddof=1)) > 3 * cand  # unadjusted variance inflated by short series
    assert length_adjusted_var_sr(srs, t, n_cand, floor=0.5) == 0.5
    # Mertens formula by hand
    sr, skew, kurt, n = 0.05, -0.7, 6.0, 501
    assert sr_sampling_var(sr, n, skew, kurt) == pytest.approx(
        (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1), abs=1e-12
    )
    assert sr_sampling_var(0.0, 101) == pytest.approx(1 / 100, abs=1e-12)
    # no cross-trial excess -> candidate sampling variance exactly
    assert length_adjusted_var_sr([0.01, 0.01], 10_000, 1000) == pytest.approx(
        sr_sampling_var(0.01, 1000), abs=1e-12
    )


def test_dsr_raises_on_degenerate():
    with pytest.raises(DegenerateInputError):
        dsr(0.05, 500, 0.0, 3.0, 1, 0.001)
    with pytest.raises(DegenerateInputError):
        dsr(0.05, 500, 0.0, 3.0, 50, 0.0)
    with pytest.raises(DegenerateInputError):
        dsr(0.05, 1, 0.0, 3.0, 50, 0.001)
    with pytest.raises(DegenerateInputError):  # radicand <= 0 is not clamped
        psr(0.2, 0.0, 500, 8.0, 1.0)
    with pytest.raises(DegenerateInputError):
        moments([0.01] * 20)
    with pytest.raises(DegenerateInputError):
        moments([0.01])


def test_annualised_guard(caplog):
    with pytest.raises(SharpeUnitsError):
        psr(1.5, 0.0, 500, 0.0, 3.0)
    with pytest.raises(SharpeUnitsError):
        psr(0.1, 1.5, 500, 0.0, 3.0)
    with pytest.raises(SharpeUnitsError):
        sr_sampling_var(1.5, 500)
    with caplog.at_level(logging.WARNING):
        psr(0.3, 0.0, 500, 0.0, 3.0)
    assert any("per-period" in rec.getMessage() for rec in caplog.records)
    check_per_period(1.5, max_abs=2.0)  # explicit keyword override
    with pytest.raises(SharpeUnitsError):
        check_per_period(0.6, max_abs=0.5, name="x")


def test_small_sample_warns(caplog):
    with caplog.at_level(logging.WARNING):
        psr(0.01, 0.0, 5, 0.0, 3.0)
    assert any("unreliable" in rec.getMessage() for rec in caplog.records)


def test_import_has_no_live_modules(tmp_path):
    code = (
        "import sys, firm.validation.sharpe_stats\n"
        "bad = [m for m in sys.modules if m in ('firm.live','firm.api','firm.runtime')"
        " or m.startswith(('firm.live.','firm.api.'))]\n"
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
