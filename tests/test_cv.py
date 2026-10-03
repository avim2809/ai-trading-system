"""Tests for firm.validation.cv (P1-05)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from firm.patterns.ml.purged_cv import purged_kfold_splits
from firm.validation.cv import CombinatorialPurgedCV, PurgedKFold, legacy_day_embargo_splits


def make_labels(T=200, max_len=5, seed=0, irregular=False):
    rng = np.random.default_rng(seed)
    if irregular:
        gaps = rng.integers(1, 15, size=T)
        t0 = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.cumsum(gaps), unit="D")
    else:
        t0 = pd.date_range("2020-01-01", periods=T, freq="D")
    lens = rng.integers(0, max_len + 1, size=T)
    t1 = pd.DatetimeIndex(t0) + pd.to_timedelta(lens, unit="D")
    return pd.Series(t1, index=pd.DatetimeIndex(t0))


def _assert_no_overlap(labels, train, test):
    t0 = labels.index.to_numpy()
    t1 = labels.to_numpy()
    assert len(np.intersect1d(train, test)) == 0
    if len(train) == 0:
        return
    lo, hi = t0[train][:, None], t1[train][:, None]
    # train [lo,hi] intersects test [a,b] iff hi >= a and lo <= b
    inter = (hi >= t0[test][None, :]) & (lo <= t1[test][None, :])
    assert not inter.any()


@pytest.mark.parametrize("N,k,phi", [(10, 2, 9), (6, 2, 5), (8, 2, 7), (10, 3, 36), (5, 1, 1)])
def test_cpcv_path_count_n10_k2_is_9(N, k, phi):
    cv = CombinatorialPurgedCV(N, k, make_labels(300))
    assert cv.n_paths == phi == math.comb(N, k) * k // N
    assert cv.get_n_splits() == math.comb(N, k)
    assert len(list(cv.split())) == math.comb(N, k)
    assert cv.n_paths == 9 or N != 10 or k != 2


def _identity_preds(cv, T):
    vals = np.arange(T)
    return {sid: vals[test] for sid, (_, test) in enumerate(cv.split())}


def test_paths_cover_sample_exactly_once():
    T = 203
    cv = CombinatorialPurgedCV(10, 2, make_labels(T))
    paths = cv.backtest_paths(_identity_preds(cv, T))
    assert len(paths) == 9
    for p in paths:
        assert len(p) == T
        assert np.array_equal(p, np.arange(T))  # each sample once, in time order


def test_each_group_in_expected_number_of_test_splits():
    for N, k in [(10, 2), (6, 3), (8, 2)]:
        cv = CombinatorialPurgedCV(N, k, make_labels(160))
        counts = np.zeros(N, int)
        for combo in cv.split_groups():
            for g in combo:
                counts[g] += 1
        assert (counts == math.comb(N - 1, k - 1)).all()


def test_no_train_test_label_overlap_random():
    for seed in range(40):
        rng = np.random.default_rng(seed)
        T = int(rng.integers(40, 300))
        labels = make_labels(T, max_len=int(rng.integers(0, 30)), seed=seed, irregular=bool(seed % 2))
        pct = float(rng.choice([0.0, 0.01, 0.05]))
        for train, test in CombinatorialPurgedCV(6, 2, labels, pct).split():
            _assert_no_overlap(labels, train, test)
        for train, test in PurgedKFold(5, labels, pct).split():
            _assert_no_overlap(labels, train, test)


def test_no_train_test_label_overlap_hypothesis():
    hyp = pytest.importorskip("hypothesis", reason="hypothesis not installed (dev extra)")
    st = pytest.importorskip("hypothesis.strategies")

    @hyp.settings(max_examples=60, deadline=None)
    @hyp.given(T=st.integers(30, 250), max_len=st.integers(0, 40), seed=st.integers(0, 10_000))
    def prop(T, max_len, seed):
        labels = make_labels(T, max_len, seed)
        for train, test in CombinatorialPurgedCV(5, 2, labels, 0.02).split():
            _assert_no_overlap(labels, train, test)

    prop()


def test_hypothesis_available():
    pytest.importorskip("hypothesis", reason="hypothesis is a dev extra; install dev extras in the research venv")


def test_embargo_removes_expected_samples():
    T = 1000
    labels = make_labels(T, max_len=0)  # zero-length labels -> purge removes nothing
    cv = PurgedKFold(5, labels, embargo_pct=0.05)
    splits = list(cv.split())
    for i, (train, test) in enumerate(splits):
        end = test[-1] + 1
        banned = np.arange(end, min(end + 50, T))
        assert not np.isin(banned, train).any()
        # nothing else removed from train
        assert len(train) == T - len(test) - len(banned)
    # CPCV: block followed by train group is embargoed; adjacent test groups merge
    cp = CombinatorialPurgedCV(10, 2, labels, 0.05)
    for combo, (train, test) in zip(cp.split_groups(), cp.split()):
        ends = [cp._groups[g][-1] + 1 for g in combo]
        for g, end in zip(combo, ends):
            if g + 1 in combo:
                continue
            banned = np.arange(end, min(end + 50, T))
            banned = banned[~np.isin(banned, test)]
            assert not np.isin(banned, train).any()
        assert len(train) == T - len(test) - sum(
            len(np.setdiff1d(np.arange(e, min(e + 50, T)), test)) for g, e in zip(combo, ends) if g + 1 not in combo
        )


def test_purged_kfold_matches_legacy_when_no_embargo():
    labels = make_labels(240, max_len=12, seed=3, irregular=True)
    new = list(PurgedKFold(6, labels, embargo_pct=0).split())
    old = purged_kfold_splits(labels.index, labels.to_numpy(), n_splits=6, embargo_days=0)
    assert len(new) == len(old) == 6
    for (a, b), (c, d) in zip(new, old):
        assert set(a) == set(c) and set(b) == set(d)


@pytest.mark.parametrize("days", [0, 3, 20])
def test_legacy_day_embargo_adapter_equivalence(days):
    labels = make_labels(180, max_len=8, seed=4, irregular=True)
    # unsorted input to exercise original-order positional indices
    perm = np.random.default_rng(1).permutation(len(labels))
    t0, t1 = labels.index[perm], labels.to_numpy()[perm]
    new = legacy_day_embargo_splits(t0, t1, 5, days)
    old = purged_kfold_splits(t0, t1, n_splits=5, embargo_days=days)
    for (a, b), (c, d) in zip(new, old):
        assert np.array_equal(a, c) and np.array_equal(b, d)


def test_pct_vs_day_embargo_documented():
    # Irregular sampling: a dense burst (1/day) then, after a ~265 day gap, a sparse
    # region. Test fold = the burst. 10% of T embargoes the next 20 samples; a
    # 20-day embargo after the burst's end reaches none of them (gap > 20 days).
    t0 = list(pd.date_range("2020-01-01", periods=100, freq="D")) + list(
        pd.date_range("2021-01-01", periods=100, freq="10D")
    )
    labels = pd.Series(pd.DatetimeIndex(t0), index=pd.DatetimeIndex(t0))  # zero-length labels
    pct_train = list(PurgedKFold(2, labels, embargo_pct=0.10).split())[0][0]  # test = first half
    day_train = purged_kfold_splits(labels.index, labels.to_numpy(), n_splits=2, embargo_days=20)[0][0]
    assert len(pct_train) == 100 - 20
    assert len(day_train) == 100
    assert len(pct_train) != len(day_train)


def test_backtest_paths_validates_shapes():
    T = 100
    cv = CombinatorialPurgedCV(5, 2, make_labels(T))
    good = _identity_preds(cv, T)
    bad = dict(good)
    bad[0] = bad[0][:-1]
    with pytest.raises(ValueError):
        cv.backtest_paths(bad)
    miss = dict(good)
    del miss[3]
    with pytest.raises(ValueError):
        cv.backtest_paths(miss)
    with pytest.raises(ValueError):
        cv.backtest_paths({k: v.reshape(-1, 1) for k, v in good.items()})
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(5, 2, None)


def test_deterministic_split_order():
    labels = make_labels(150, seed=2)
    a = [(t.tolist(), s.tolist()) for t, s in CombinatorialPurgedCV(6, 2, labels).split()]
    b = [(t.tolist(), s.tolist()) for t, s in CombinatorialPurgedCV(6, 2, labels).split()]
    assert a == b
    import itertools

    assert CombinatorialPurgedCV(6, 2, labels).split_groups() == list(itertools.combinations(range(6), 2))


def test_paths_differ_when_in_sample_step_differs():
    T = 400
    labels = make_labels(T, max_len=1)
    rng = np.random.default_rng(0)
    # config A is better in the first half, B in the second half
    ra = rng.normal(0, 1, T) + np.where(np.arange(T) < T // 2, 0.5, -0.5)
    rb = rng.normal(0, 1, T) + np.where(np.arange(T) < T // 2, -0.5, 0.5)
    cv = CombinatorialPurgedCV(10, 2, labels, 0.01)

    def assemble(configs):
        preds = {}
        for sid, (train, test) in enumerate(cv.split()):
            best = max(configs, key=lambda c: c[train].sum())  # in-sample argmax
            preds[sid] = best[test]
        return cv.backtest_paths(preds)

    paths = assemble([ra, rb])
    assert len({p.tobytes() for p in paths}) > 1
    single = assemble([ra])
    for p in single:
        assert np.array_equal(p, ra)
