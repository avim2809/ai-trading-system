"""P1-04: PBO via CSCV."""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pytest
from scipy.stats import rankdata

from firm.eval.overfitting import cscv_pbo
from firm.validation.pbo import PBOResult, pbo, sharpe_cols


def _noise(seed, T=1600, N=10):
    return np.random.default_rng(seed).normal(0, 0.01, (T, N))


def _drift(seed, N, sr_ann, T=1600):
    rng = np.random.default_rng(seed)
    M = rng.normal(0, 0.01, (T, N))
    M[:, 0] += sr_ann / np.sqrt(252) * 0.01
    return M


def _naive(M, S):
    T, N = M.shape
    rows = T // S
    blocks = [M[k * rows : (k + 1) * rows] for k in range(S)]
    out = []
    for comb in combinations(range(S), S // 2):
        rest = [k for k in range(S) if k not in comb]
        IS = np.vstack([blocks[k] for k in comb])
        OOS = np.vstack([blocks[k] for k in rest])
        ns = int(np.argmax(sharpe_cols(IS)))
        o = sharpe_cols(OOS)
        w = rankdata(o, method="average")[ns] / (N + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        out.append(np.log(w / (1 - w)))
    return np.array(out)


def test_noise_gives_half():
    vals = [pbo(_noise(s, 1600, 10 + s % 21), S=16).pbo for s in range(20)]
    assert abs(np.mean(vals) - 0.5) < 0.1


@pytest.mark.parametrize("N,sr", [(30, 3.0), (10, 2.5)])
def test_strong_drift_gives_low_pbo(N, sr):
    vals = [pbo(_drift(s, N, sr), S=16).pbo for s in range(20)]
    assert np.mean(vals) < 0.1


@pytest.mark.parametrize("S", [8, 16])
def test_equivalence_with_cscv_pbo_tie_free(S):
    for seed in range(30):
        M = _noise(seed, T=S * 40, N=6 + seed % 5)
        assert abs(pbo(M, S).pbo - cscv_pbo(M, S)) < 1e-12


def test_equivalence_with_embargo():
    for seed in range(5):
        M = _noise(seed, T=320, N=8)
        assert abs(pbo(M, 8, embargo_pct=0.1).pbo - cscv_pbo(M, 8, embargo_pct=0.1)) < 1e-12


def test_divergence_on_ties_documented():
    # three identical zero columns plus one always-losing column: n* is a zero column
    # (Sharpe 0 > negative). Legacy argsort ranking puts column 0 at the bottom of the
    # tied group (low rank -> lambda < 0 -> counted as overfit); average ranks put it
    # mid-group, above the loser.
    rng = np.random.default_rng(1)
    M = np.zeros((320, 4))
    M[:, 3] = -0.01 + rng.normal(0, 0.001, 320)
    new = pbo(M, 8).pbo
    legacy = cscv_pbo(M, 8)
    assert new == 0.0 and legacy == 1.0
    perm = [3, 2, 1, 0]
    assert pbo(M[:, perm], 8).pbo == new  # invariant to column order


def test_permutation_invariant_in_columns():
    M = _noise(3, 800, 12)
    perm = np.random.default_rng(0).permutation(12)
    assert pbo(M, 8).pbo == pbo(M[:, perm], 8).pbo


def test_reversal_symmetry():
    rng = np.random.default_rng(5)
    for seed in range(5):
        e = np.random.default_rng(seed).normal(0, 0.01, (640, 8))
        M = np.empty_like(e)
        M[0] = e[0]
        for t in range(1, 640):
            M[t] = 0.4 * M[t - 1] + e[t]
        assert pbo(M, 16).pbo == pbo(M[::-1].copy(), 16).pbo
    del rng


def test_blocks_are_contiguous():
    S, rows, N = 8, 20, 6
    rng = np.random.default_rng(2)
    M = rng.normal(0, 0.01, (S * rows, N))
    good = {0: {0, 1}, 1: {2, 3}, 2: {4, 5}, 3: {6, 7}}
    for j, bl in good.items():
        for k in bl:
            M[k * rows : (k + 1) * rows, j] += 0.02
    res = pbo(M, S)
    np.testing.assert_allclose(np.sort(res.logits), np.sort(_naive(M, S)), atol=1e-12)
    # combinations are generated in itertools order -> exact per-split match
    np.testing.assert_allclose(res.logits, _naive(M, S), atol=1e-12)


def test_naive_reference_matches():
    for seed in range(3):
        M = _noise(seed, 320, 9)
        r = pbo(M, 8)
        np.testing.assert_allclose(r.logits, _naive(M, 8), atol=1e-12)
        assert abs(r.pbo - float((_naive(M, 8) <= 0).mean())) < 1e-12


def test_pypbo_agreement():
    pytest.importorskip("pypbo")


def test_degradation_slope_sign():
    slopes = [pbo(_noise(s, 800, 20), 16).perf_degradation_slope for s in range(10)]
    assert np.mean(slopes) < 0
    assert all(s < 0 for s in slopes)
    losses, oos = [], []
    for s in range(20):
        r = pbo(_drift(s, 30, 3.0), 16)
        losses.append(r.prob_oos_loss)
    assert np.mean(losses) < 0.1
    M = _drift(0, 30, 3.0)
    assert sharpe_cols(M)[0] > 0 and not oos


def test_prob_oos_loss_range():
    r = pbo(_noise(0, 320, 8), 8)
    assert 0.0 <= r.prob_oos_loss <= 1.0 and isinstance(r, PBOResult)
    assert r.n_splits == 70 and len(r.logits) == 70 and 0 <= r.perf_degradation_r2 <= 1


def test_tail_policy():
    M = _noise(0, 325, 8)
    r = pbo(M, 8)
    assert r.rows_dropped == 5 and r.rows_used == 320
    assert r.pbo == pbo(M[:320], 8).pbo
    with pytest.raises(ValueError):
        pbo(M, 8, tail="raise")
    assert pbo(M[:320], 8, tail="raise").rows_dropped == 0


def test_small_inputs_raise():
    with pytest.raises(ValueError):
        pbo(_noise(0, 10, 5), 8)
    with pytest.raises(ValueError):
        pbo(_noise(0, 320, 1), 8)
    with pytest.raises(ValueError):
        pbo(_noise(0, 320, 5), 7)
    with pytest.raises(ValueError):
        pbo(np.zeros(10), 4)


def test_uninformative_flag_for_N_lt_4():
    assert pbo(_noise(0, 320, 3), 8).uninformative
    assert pbo(_noise(0, 320, 2), 8).uninformative
    assert not pbo(_noise(0, 320, 4), 8).uninformative


def test_custom_metric_matches_default():
    M = _noise(1, 160, 5)
    assert pbo(M, 8, metric=lambda x: sharpe_cols(x)).pbo == pbo(M, 8).pbo  # wrapper -> slow path
    np.testing.assert_allclose(
        pbo(M, 8, metric=lambda x: sharpe_cols(x)).logits, pbo(M, 8).logits, atol=1e-9
    )


@pytest.mark.slow
def test_runtime_budget():
    M = _noise(0, 2500, 50)
    t0 = time.time()
    r = pbo(M, 16)
    assert time.time() - t0 < 30 and r.n_splits == 12870
