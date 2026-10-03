"""Tests for firm.validation.bootstrap (P1-07)."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from firm.validation.bootstrap import (
    block_bootstrap_returns,
    paired_sharpe_gap_bootstrap,
    politis_white_block_length,
    stationary_bootstrap_indices,
)

ROOT = Path(__file__).resolve().parents[1]


def ar1(n, rho, seed, burn=200):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = rho * x[t - 1] + e[t]
    return x[burn:]


def lag1(x):
    xc = x - x.mean()
    return float((xc[1:] * xc[:-1]).sum() / (xc * xc).sum())


def test_indices_in_range_and_shape():
    idx = stationary_bootstrap_indices(100, 50, 5.0, seed=1)
    assert idx.shape == (50, 100) and idx.dtype.kind == "i"
    assert idx.min() >= 0 and idx.max() < 100
    assert stationary_bootstrap_indices(100, 7, 5.0, seed=1, n_periods=33).shape == (7, 33)
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(100, 5, 0.5, seed=1)


def test_mean_block_length_geometric():
    n, mb = 5000, 20.0
    idx = stationary_bootstrap_indices(n, 200, mb, seed=3)
    breaks = (idx[:, 1:] != (idx[:, :-1] + 1) % n).sum() + idx.shape[0]  # + first block of each path
    mean_run = idx.size / breaks
    assert abs(mean_run - mb) / mb < 0.05


def test_preserves_autocorrelation():
    x = ar1(2000, 0.5, 11)
    target = lag1(x)
    boot = block_bootstrap_returns(x, 20.0, 100, seed=2)
    got = np.mean([lag1(row) for row in boot])
    assert abs(got - target) < 0.05
    iid = x[np.random.default_rng(5).integers(0, len(x), size=(100, len(x)))]
    ctrl = np.mean([lag1(row) for row in iid])
    assert abs(ctrl) < 0.05 and abs(target - ctrl) > 0.3  # the control discriminates


@pytest.mark.parametrize("rho,tol", [(0.2, 0.30), (0.5, 0.15), (0.8, 0.20)])
def test_politis_white_known_ar1(rho, tol):
    """Median over 20 simulated AR(1) series (n=5000) vs b = n^(1/3) (2rho/(1-rho^2))^(2/3).

    Tolerances are relative; rho=0.2 is wider because the flat-top window keeps only
    a few lags (m_hat is 1-2), which biases G down (observed median about -22%).
    """
    n = 5000
    analytic = n ** (1 / 3) * (2 * rho / (1 - rho**2)) ** (2 / 3)
    est = float(np.median([politis_white_block_length(ar1(n, rho, s)) for s in range(20)]))
    assert abs(est - analytic) / analytic < tol, (est, analytic)


def test_block_length_capped_and_floored():
    n = 5000
    assert politis_white_block_length(np.random.default_rng(1).normal(size=n)) == 1.0
    assert politis_white_block_length(np.ones(100)) == 1.0
    cap = np.ceil(min(3 * np.sqrt(n), n / 3))
    assert politis_white_block_length(ar1(n, 0.995, 1)) <= cap
    # strongly persistent series never exceed the cap (it rarely binds: max_lag ~ sqrt(n) bounds b first)
    for m in (60, 90, 5000):
        rw = np.cumsum(np.random.default_rng(2).normal(size=m))
        assert 1.0 <= politis_white_block_length(rw) <= np.ceil(min(3 * np.sqrt(m), m / 3))
    # multi-series: maximum over columns
    two = np.column_stack([np.random.default_rng(1).normal(size=n), ar1(n, 0.5, 4)])
    assert politis_white_block_length(two) == politis_white_block_length(two[:, 1])
    with pytest.raises(ValueError):
        politis_white_block_length(np.arange(5.0))


def test_seed_deterministic_and_paired_gap():
    a, b = ar1(500, 0.2, 1), ar1(500, 0.2, 2)
    g1 = paired_sharpe_gap_bootstrap(a, b, 300, seed=4)
    g2 = paired_sharpe_gap_bootstrap(a, b, 300, seed=4)
    assert np.array_equal(g1, g2) and g1.shape == (300,)
    assert not np.array_equal(g1, paired_sharpe_gap_bootstrap(a, b, 300, seed=5))
    # identical series -> gap is exactly zero in every sim (pairing uses common indices)
    assert np.allclose(paired_sharpe_gap_bootstrap(a, a, 50, seed=1), 0.0)


def _load_script_stationary_indices():
    path = ROOT / "scripts" / "run_alt_premia_evaluation.py"
    spec = importlib.util.spec_from_file_location("_alt_premia_readonly", path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)  # module-level only; __main__ guard keeps main() from running
    finally:
        sys.path[:] = saved
    return mod.stationary_indices


def test_equivalence_with_script_generators_in_distribution():
    legacy = _load_script_stationary_indices()
    x = ar1(1000, 0.5, 9)
    n, mb, sims = len(x), 10, 200
    old = legacy(n, sims, mb, np.random.default_rng(1))
    new = stationary_bootstrap_indices(n, sims, float(mb), seed=1)
    for idx in (old, new):
        breaks = (idx[:, 1:] != (idx[:, :-1] + 1) % n).sum() + sims
        assert abs(idx.size / breaks - mb) / mb < 0.08
    ac_old = np.mean([lag1(x[r]) for r in old])
    ac_new = np.mean([lag1(x[r]) for r in new])
    assert abs(ac_old - ac_new) < 0.05
    m_old, m_new = x[old].mean(1), x[new].mean(1)
    assert abs(m_old.std() - m_new.std()) / m_old.std() < 0.25


def test_vectorised_faster_than_python_loop():
    n, sims, mb = 1000, 100, 10.0

    def python_loop(seed):
        rng = np.random.default_rng(seed)
        out = np.empty((sims, n), dtype=np.int64)
        for s in range(sims):
            i = rng.integers(0, n)
            for t in range(n):
                if t > 0 and rng.random() < 1 / mb:
                    i = rng.integers(0, n)
                elif t > 0:
                    i = (i + 1) % n
                out[s, t] = i
        return out

    t0 = time.perf_counter()
    python_loop(0)
    slow = time.perf_counter() - t0
    t0 = time.perf_counter()
    stationary_bootstrap_indices(n, sims, mb, seed=0)
    fast = time.perf_counter() - t0
    assert fast < slow / 3
