"""Diversification primitives (P1-03)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from firm.validation import diversification as D


def _block_corr(k: int, size: int = 6, within: float = 0.9) -> np.ndarray:
    n = k * size
    c = np.zeros((n, n))
    for b in range(k):
        c[b * size:(b + 1) * size, b * size:(b + 1) * size] = within
    np.fill_diagonal(c, 1.0)
    return c


def test_participation_ratio_bounds():
    assert D.participation_ratio(np.eye(7)) == pytest.approx(7.0)
    assert D.participation_ratio(np.ones((5, 5))) == pytest.approx(1.0)
    pr = D.participation_ratio(_block_corr(3))
    assert 1.0 < pr < 18.0


def test_effective_rank_equals_n_for_identity_and_1_for_rank1():
    assert D.effective_rank(np.eye(9)) == pytest.approx(9.0)
    assert D.effective_rank(np.ones((6, 6))) == pytest.approx(1.0)
    assert D.effective_number_of_bets(np.eye(8), np.full(8, 1 / 8)) == pytest.approx(8.0, rel=1e-9)


def test_enb_concentration_lowers_enb_with_cov_fixed():
    cov = _block_corr(4, size=3, within=0.8) * 0.0004
    n = cov.shape[0]
    w_div = np.full(n, 1 / n)
    w_conc = w_div.copy()
    last = []
    for tilt in (0.0, 0.3, 0.6, 0.9):
        w = (1 - tilt) * w_div
        w[:3] += tilt / 3  # pile into one block (asset class)
        last.append(D.effective_number_of_bets(cov, w))
    assert all(a > b for a, b in zip(last, last[1:]))
    assert w_conc.sum() == pytest.approx(1.0)


def test_enb_scale_invariant_in_weights():
    cov = _block_corr(2) * 0.001
    w = np.linspace(1, 2, 12)
    assert D.effective_number_of_bets(cov, w) == pytest.approx(D.effective_number_of_bets(cov, 5 * w))


def test_enb_min_torsion_not_implemented():
    with pytest.raises(NotImplementedError):
        D.effective_number_of_bets(np.eye(3), np.ones(3), method="min_torsion")


def test_mp_edge_formula():
    assert D.marchenko_pastur_edge(100, 400) == pytest.approx((1 + math.sqrt(0.25)) ** 2) == pytest.approx(2.25)
    assert D.marchenko_pastur_edge(100, 400, sigma2=2.0) == pytest.approx(4.5)
    r = np.random.default_rng(0).standard_normal((400, 100))
    top = np.linalg.eigvalsh(np.corrcoef(r, rowvar=False)).max()
    assert top < 1.15 * D.marchenko_pastur_edge(100, 400)


def test_n_signal_eigenvalues_with_planted_factors():
    rng = np.random.default_rng(1)
    T, k, size = 1000, 4, 8
    f = rng.standard_normal((T, k))
    x = np.repeat(f, size, axis=1) * 0.8 + 0.6 * rng.standard_normal((T, k * size))
    assert D.n_signal_eigenvalues(np.corrcoef(x, rowvar=False), T) == k


def test_diversification_ratio_matches_hand():
    w = np.array([0.5, 0.5])
    vols = np.array([0.2, 0.1])
    corr = np.array([[1, 0.5], [0.5, 1]])
    port_vol = math.sqrt(0.25 * 0.04 + 0.25 * 0.01 + 2 * 0.25 * 0.5 * 0.2 * 0.1)
    assert D.diversification_ratio(w, vols, corr) == pytest.approx(0.15 / port_vol)
    assert D.diversification_ratio(w, vols, np.ones((2, 2))) == pytest.approx(1.0)


def test_fisher_z_known_values():
    z, p = D.fisher_z_corr_diff(0.5, 100, 0.3, 100)
    exp = (math.atanh(0.5) - math.atanh(0.3)) / math.sqrt(2 / 97)
    assert z == pytest.approx(exp) and z == pytest.approx(1.670, abs=1e-3)
    assert p == pytest.approx(0.0950, abs=1e-3)
    assert D.fisher_z_corr_diff(0.4, 50, 0.4, 80)[1] == pytest.approx(1.0)


def test_bh_adjust_matches_hand():
    p = np.array([0.001, 0.2, 0.012, 0.9, 0.03])
    # sorted: .001 .012 .03 .2 .9 vs thresholds .02 .04 .06 .08 .10 -> first three pass
    assert D.bh_adjust(p, q=0.10).tolist() == [True, False, True, False, True]
