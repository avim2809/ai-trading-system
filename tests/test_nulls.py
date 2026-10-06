"""Tests for firm.validation.nulls (P1-07).

The uniformity tests are the actual safeguard against a Gann-style broken null
(p = 1.000): each generator is run on data drawn under ITS OWN H0 and the
resulting p-values must be uniform and rarely exactly 1.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy import stats

from firm.patterns.significance import benjamini_hochberg_accept, pattern_p_value
from firm.validation import nulls
from firm.validation.bootstrap import block_bootstrap_returns, politis_white_block_length
from firm.validation.nulls import (
    benjamini_hochberg,
    p_value,
    phase_randomised_surrogate,
    random_entry_null,
    sign_randomisation_null,
)

N_RUNS = 500
N_NULL = 499  # p lives on the grid {1/500, ..., 1}; KS vs continuous uniform is conservative-ish, see below


def ar1(n, rho, rng, burn=100):
    e = rng.normal(size=n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = rho * x[t - 1] + e[t]
    return x[burn:]


def test_p_value_matches_legacy_pattern_p_value():
    rng = np.random.default_rng(0)
    for _ in range(100):
        null = np.round(rng.normal(size=int(rng.integers(1, 80))), 1)  # ties on purpose
        obs = float(np.round(rng.normal(), 1))
        assert p_value(obs, null, "right") == pattern_p_value(obs, null)


def test_p_value_tails():
    null = np.arange(10.0)
    assert p_value(5.0, null, "right") == (1 + 5) / 11
    assert p_value(5.0, null, "left") == (1 + 6) / 11
    assert p_value(100.0, null, "right") == 1 / 11
    assert p_value(0.0, null, "two") == min(1.0, 2 * min(2 / 11, 11 / 11))
    assert 0 < p_value(1e9, null, "two") <= 1


def test_bh_matches_legacy():
    rng = np.random.default_rng(1)
    for i in range(200):
        m = int(rng.integers(0, 40))
        p = np.round(rng.uniform(size=m) ** 2, 2 if i % 2 else 6)  # coarse rounding -> ties
        q = float(rng.choice([0.05, 0.10, 0.2]))
        assert np.array_equal(benjamini_hochberg(p, q), benjamini_hochberg_accept(p, q))


def test_p_value_uniform_under_null():
    """Rank formula only. n=999 null draws per run: p lives on the grid {k/1000}; the
    discretisation error vs a continuous uniform (<= 1/1000) is far below KS resolution
    at 500 runs."""
    rng = np.random.default_rng(2)
    ps = [p_value(rng.normal(), rng.normal(size=999)) for _ in range(500)]
    assert stats.kstest(ps, "uniform").pvalue > 0.01


def _h0_sign(rng):
    pnl = ar1(300, 0.3, rng)  # symmetric zero-mean autocorrelated PnL
    null = sign_randomisation_null(np.ones(300), pnl, N_NULL, seed=int(rng.integers(1 << 30)))
    return p_value(pnl.mean(), null)


def _h0_entry(rng):
    data = rng.normal(size=300)  # drift-free returns
    pos = np.zeros(300)
    starts = np.sort(rng.choice(280, size=6, replace=False) // 20 * 20 + rng.integers(0, 10, size=6) * 0)
    for s in starts:
        pos[s : s + int(rng.integers(5, 15))] = 1.0  # random positions, independent of returns
    null = random_entry_null(lambda d: pos, data, N_NULL, seed=int(rng.integers(1 << 30)))
    return p_value(float(np.mean(pos * data)), null)


def _stat_asym(x):
    d = np.diff(x)
    return float(np.mean(d**3))  # time-reversal asymmetry: zero for stationary Gaussian series


def _h0_surrogate(rng):
    x = ar1(256, 0.5, rng)  # stationary Gaussian
    null = np.array([_stat_asym(s) for s in phase_randomised_surrogate(x, N_NULL, seed=int(rng.integers(1 << 30)))])
    return p_value(_stat_asym(x), null)


def _h0_bootstrap(rng):
    x = ar1(500, 0.3, rng)  # zero-mean stationary
    boot = block_bootstrap_returns(x - x.mean(), None, N_NULL, seed=int(rng.integers(1 << 30)))
    return p_value(float(x.mean()), boot.mean(axis=1))


@pytest.mark.parametrize("gen", [_h0_sign, _h0_entry, _h0_surrogate, _h0_bootstrap], ids=lambda f: f.__name__)
def test_null_generators_uniform_under_H0(gen):
    rng = np.random.default_rng(12345)
    ps = np.array([gen(rng) for _ in range(N_RUNS)])
    assert stats.kstest(ps, "uniform").pvalue > 0.01, "p-values not uniform under H0"
    assert np.mean(ps == 1.0) < 0.02, "too many p == 1 (Gann-style broken null)"


def test_block_sign_vs_iid_sign_on_ar1():
    rng = np.random.default_rng(7)
    rej_block = rej_iid = 0
    runs = 300
    for _ in range(runs):
        pnl = ar1(400, 0.5, rng)
        s = int(rng.integers(1 << 30))
        pb = p_value(pnl.mean(), sign_randomisation_null(np.ones(400), pnl, 199, s))
        pi = p_value(pnl.mean(), sign_randomisation_null(np.ones(400), pnl, 199, s, block_len=1))
        rej_block += pb < 0.05
        rej_iid += pi < 0.05
    assert 0.02 <= rej_block / runs <= 0.09, rej_block / runs  # about alpha=0.05
    assert rej_iid / runs > 0.12  # theory for rho=0.5: about 17% one-sided


def test_iid_sign_warns(caplog):
    with caplog.at_level(logging.WARNING):
        sign_randomisation_null(np.ones(50), np.random.default_rng(0).normal(size=50), 10, 1, block_len=1)
    assert any("i.i.d. sign flip" in r.message for r in caplog.records)


def test_power_p_below_005_most_runs():
    """T=250, true mean 0.5 sd per period (pinned)."""
    rng = np.random.default_rng(99)
    hits_sign = hits_entry = 0
    regime = np.tile(np.repeat([1.0, -1.0], 25), 5)  # 10 blocks of 25 periods
    for _ in range(N_RUNS):
        r = rng.normal(0.5, 1.0, 250)
        null = sign_randomisation_null(np.ones(250), r, 199, int(rng.integers(1 << 30)))
        hits_sign += p_value(r.mean(), null) < 0.05
        # skilled timing: long during +regime blocks where the mean is +0.5 sd, -0.5 otherwise
        d = rng.normal(0, 1.0, 250) + 0.5 * regime
        strat = lambda x: (regime > 0).astype(float)
        null = random_entry_null(strat, d, 199, int(rng.integers(1 << 30)))
        hits_entry += p_value(float(np.mean((regime > 0) * d)), null) < 0.05
    assert hits_sign / N_RUNS >= 0.80, hits_sign / N_RUNS
    assert hits_entry / N_RUNS >= 0.80, hits_entry / N_RUNS


def test_random_entry_preserves_exposure():
    pos = np.zeros(100)
    pos[10:20], pos[40:45], pos[60:80] = 1, -1, 1
    data = np.random.default_rng(0).normal(size=100)
    out = random_entry_null(lambda d: pos, data, 50, 3, stat=lambda x: float(np.count_nonzero(x)))
    assert (out == 35).all()
    # lengths resampled from the empirical set when keep_holding_periods=False
    out2 = random_entry_null(lambda d: pos, data, 50, 3, keep_holding_periods=False, stat=lambda x: float(np.count_nonzero(x)))
    assert out2.min() >= 3 * 5 and out2.max() <= 3 * 20
    assert len(set(out2)) > 1
    with pytest.raises(ValueError):
        random_entry_null(lambda d: np.zeros_like(d), data, 5, 1)


def test_phase_surrogate_preserves_spectrum_mean_var():
    rng = np.random.default_rng(4)
    for n in (256, 257):
        x = ar1(n, 0.6, rng) + 0.3
        s = phase_randomised_surrogate(x, 20, seed=1)
        assert s.shape == (20, n)
        assert np.allclose(s.mean(1), x.mean(), atol=1e-10)
        assert np.allclose(s.var(1), x.var(), rtol=1e-9)
        assert np.allclose(np.abs(np.fft.rfft(s, axis=1)), np.abs(np.fft.rfft(x)), atol=1e-8)
        assert not np.allclose(s[0], x)
        assert not np.allclose(s[0], s[1])


def test_seed_reproducible_and_different_seeds_differ():
    rng = np.random.default_rng(0)
    r = ar1(200, 0.3, rng)
    pos = np.ones(200)
    for f in (
        lambda s: sign_randomisation_null(pos, r, 50, s),
        lambda s: phase_randomised_surrogate(r, 5, s),
        lambda s: random_entry_null(lambda d: (np.arange(200) % 40 < 10).astype(float), r, 20, s),
        lambda s: block_bootstrap_returns(r, 5.0, 10, s),
    ):
        assert np.array_equal(f(1), f(1))
        assert not np.array_equal(f(1), f(2))


def test_empty_null_raises():
    with pytest.raises(ValueError):
        p_value(1.0, np.array([]))
    with pytest.raises(ValueError):
        p_value(1.0, np.array([0.0, np.nan]))


def test_no_global_rng_state_touched():
    r = np.random.default_rng(0).normal(size=120)
    before = np.random.get_state()
    sign_randomisation_null(np.ones(120), r, 20, 1)
    phase_randomised_surrogate(r, 5, 1)
    random_entry_null(lambda d: (np.arange(120) % 30 < 8).astype(float), r, 5, 1)
    block_bootstrap_returns(r, None, 5, 1)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_reexport_and_no_live_imports():
    assert nulls.block_bootstrap_returns is block_bootstrap_returns
    import sys

    assert politis_white_block_length  # bootstrap module importable standalone
    src = open(nulls.__file__).read() + open(sys.modules["firm.validation.bootstrap"].__file__).read()
    assert "firm.live" not in src and "firm.api" not in src
