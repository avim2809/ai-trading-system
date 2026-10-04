"""P1-06: White RC, Hansen SPA, Romano-Wolf. Small simulations; full size/power is P1-08."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from firm.validation.multiple_testing import reality_check, romano_wolf, spa_test

SR2_DRIFT = 2.0 / np.sqrt(252)  # per-period drift in sd units for annualised SR 2.0


def _null(rng, T, K):
    return rng.normal(0, 1, (T, K))


def test_rc_p_value_uniform_ish_under_null():
    rng = np.random.default_rng(10)
    rej = np.mean(
        [reality_check(_null(rng, 500, 5), B=299, seed=s).p_value <= 0.05 for s in range(200)]
    )
    assert 0.01 <= rej <= 0.10


def test_spa_rejects_strong_signal():
    rng = np.random.default_rng(2024)  # pinned; do not change
    d = _null(rng, 2520, 10)
    d[:, 0] += SR2_DRIFT
    r = spa_test(d, B=999, seed=0)
    assert r.p_consistent < 0.05 and r.best_index == 0


def test_spa_less_conservative_than_rc_with_irrelevant_alternatives():
    rng = np.random.default_rng(7)
    spa_p, rc_p = [], []
    for s in range(25):
        d = np.hstack([_null(rng, 500, 10), _null(rng, 500, 40) - 0.5])
        spa_p.append(spa_test(d, B=299, seed=s).p_consistent)
        rc_p.append(reality_check(d, B=299, seed=s).p_value)
    assert np.mean(spa_p) <= np.mean(rc_p)


def test_spa_pvalue_ordering():
    rng = np.random.default_rng(3)
    d = np.hstack([_null(rng, 400, 5), _null(rng, 400, 5) - 0.4])
    r = spa_test(d, B=499, seed=1)
    assert r.p_lower <= r.p_consistent <= r.p_upper


def test_romano_wolf_stepdown_monotone_and_fwer():
    rng = np.random.default_rng(11)
    anyrej = 0
    for s in range(100):
        df = romano_wolf(_null(rng, 300, 10), B=199, seed=s)
        srt = df.sort_values("t_stat", ascending=False)
        assert (np.diff(srt["p_adj"].to_numpy()) >= -1e-15).all()
        anyrej += bool(df["reject"].any())
    assert anyrej / 100 <= 0.10


def test_romano_wolf_rejects_true_effects():
    rng = np.random.default_rng(12)
    hits = 0
    for s in range(50):
        d = _null(rng, 500, 20)
        d[:, :3] += 0.25
        df = romano_wolf(d, B=299, seed=s)
        hits += bool(df["reject"].iloc[:3].all())
    assert hits / 50 >= 0.8


def _ar1(rng, T, K, rho):
    e = rng.normal(0, 1, (T, K))
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    return x


def test_autocorrelated_losses_not_iid(caplog):
    rng = np.random.default_rng(13)
    auto = iid = 0
    n = 200
    for s in range(n):
        d = _ar1(rng, 250, 3, 0.5)
        auto += reality_check(d, B=199, seed=s).p_value <= 0.05
        with caplog.at_level(logging.WARNING):
            iid += reality_check(d, B=199, block_len=1, seed=s).p_value <= 0.05
    assert iid / n > 0.10  # forced-iid control over-rejects
    assert abs(auto / n - 0.05) < abs(iid / n - 0.05)
    assert any("iid bootstrap" in r.message for r in caplog.records)


def test_zero_variance_column_never_rejected():
    rng = np.random.default_rng(14)
    d = _null(rng, 300, 4)
    d[:, 2] = 0.3  # constant: zero bootstrap variance
    df = romano_wolf(d, B=199, seed=0)
    assert not bool(df["reject"].iloc[2]) and df["t_stat"].iloc[2] == 0.0
    assert np.isfinite(df["p_adj"]).all()
    assert np.isfinite(reality_check(d, B=199).p_value) and np.isfinite(
        spa_test(d, B=199).p_consistent
    )


def test_seed_reproducible():
    d = _null(np.random.default_rng(15), 300, 5)
    assert reality_check(d, B=199, seed=4) == reality_check(d, B=199, seed=4)
    assert spa_test(d, B=199, seed=4) == spa_test(d, B=199, seed=4)
    a, b = romano_wolf(d, B=199, seed=4), romano_wolf(d, B=199, seed=4)
    assert a.equals(b)


def test_sign_convention_documented():
    rng = np.random.default_rng(16)
    col = np.abs(rng.normal(0.5, 0.2, 400))  # strictly positive diffs
    d = np.column_stack([col, rng.normal(0, 1, 400)])
    assert reality_check(d, B=499, seed=0).p_value < 0.05
    neg = np.column_stack([-col, rng.normal(0, 1, 400)])
    assert reality_check(neg, B=499, seed=0).p_value > 0.05
    assert (
        "LARGER IS BETTER" in __import__("firm.validation.multiple_testing", fromlist=["x"]).__doc__
    )


def test_matches_hand_computed_tiny_case():
    d = np.array([[1.0, 0.0], [2.0, -1.0], [0.0, 1.0], [3.0, 0.0], [1.0, 2.0], [-1.0, 0.0]])
    idx = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 0, 0, 3, 3, 3],
            [1, 3, 3, 4, 0, 0],
            [5, 5, 2, 2, 4, 4],
            [3, 3, 3, 1, 1, 1],
        ]
    )
    T = 6
    dbar = d.mean(axis=0)
    dstar = np.array([d[i].mean(axis=0) for i in idx])
    omega = (np.sqrt(T) * dstar).std(axis=0, ddof=1)
    t = np.sqrt(T) * dbar / omega
    tstar = np.sqrt(T) * (dstar - dbar) / omega
    expected = (1 + (tstar.max(axis=1) >= t.max()).sum()) / (1 + len(idx))
    r = reality_check(d, B=5, block_len=2.0, _indices=idx)
    assert r.p_value == pytest.approx(expected) and r.best_index == int(np.argmax(t))
    assert r.t_max == pytest.approx(t.max()) and r.n_boot == 5
    # SPA upper == RC with max(0, .) statistic
    s = spa_test(d, B=5, block_len=2.0, _indices=idx)
    exp_u = (1 + (np.maximum(0, tstar.max(axis=1)) > max(0, t.max())).sum()) / 6
    assert s.p_upper == pytest.approx(exp_u)


@pytest.mark.slow
@pytest.mark.skip(reason="run via scripts/validate_stats_pipeline.py (P1-08)")
def test_size_power_full_in_p1_08():
    raise AssertionError
