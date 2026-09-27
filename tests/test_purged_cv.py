"""Tests for firm.patterns.ml.purged_cv -- Purged K-Fold cross-validation
(Part B item 4, 2026-09-27)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from firm.patterns.ml.purged_cv import purged_kfold_splits


def _dates(n: int, start: str = "2024-01-01"):
    return pd.date_range(start, periods=n, freq="D")


class TestPurgedKfoldSplitsBasics:
    def test_rejects_fewer_than_two_splits(self):
        with pytest.raises(ValueError):
            purged_kfold_splits(_dates(10), _dates(10), n_splits=1)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            purged_kfold_splits(_dates(10), _dates(5), n_splits=2)

    def test_rejects_more_splits_than_samples(self):
        with pytest.raises(ValueError):
            purged_kfold_splits(_dates(3), _dates(3), n_splits=5)

    def test_returns_n_splits_folds(self):
        t0 = _dates(20)
        splits = purged_kfold_splits(t0, t0, n_splits=4)
        assert len(splits) == 4

    def test_every_sample_used_as_test_exactly_once(self):
        t0 = _dates(20)
        splits = purged_kfold_splits(t0, t0, n_splits=5)
        all_test = np.concatenate([test_idx for _train, test_idx in splits])
        assert sorted(all_test.tolist()) == list(range(20))
        assert len(set(all_test.tolist())) == 20  # no duplicates

    def test_train_and_test_never_overlap_within_a_fold(self):
        t0 = _dates(30)
        splits = purged_kfold_splits(t0, t0, n_splits=6)
        for train_idx, test_idx in splits:
            assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))

    def test_indices_are_sorted_ascending(self):
        t0 = _dates(20)
        splits = purged_kfold_splits(t0, t0, n_splits=4)
        for train_idx, test_idx in splits:
            assert list(train_idx) == sorted(train_idx.tolist())
            assert list(test_idx) == sorted(test_idx.tolist())

    def test_does_not_require_pre_sorted_input(self):
        n = 20
        t0_sorted = _dates(n)
        rng = np.random.RandomState(0)
        perm = rng.permutation(n)
        t0_shuffled = pd.Series(t0_sorted.to_numpy()[perm])

        splits_sorted = purged_kfold_splits(t0_sorted, t0_sorted, n_splits=4)
        splits_shuffled = purged_kfold_splits(t0_shuffled, t0_shuffled, n_splits=4)

        # Same fold structure, just relabeled through the permutation:
        # t0_shuffled[i] == t0_sorted[perm[i]], so a shuffled-space index
        # array maps back to sorted-space via perm[...].
        for (train_s, test_s), (train_u, test_u) in zip(splits_sorted, splits_shuffled):
            assert set(perm[train_u].tolist()) == set(train_s.tolist())
            assert set(perm[test_u].tolist()) == set(test_s.tolist())


class TestPurgedKfoldSplitsPurging:
    def test_zero_length_spans_with_no_embargo_purge_only_exact_overlap(self):
        # Point-in-time events (t0 == t1): only the test fold's own members
        # "overlap" themselves -- with embargo_days=0, nothing else should
        # be purged from train beyond the test fold itself.
        t0 = _dates(20)
        splits = purged_kfold_splits(t0, t0, n_splits=4, embargo_days=0)
        for train_idx, test_idx in splits:
            assert len(train_idx) + len(test_idx) == 20

    def test_overlapping_spans_are_purged_from_train(self):
        # 10 samples, each spanning 3 days -- neighboring events' spans
        # overlap by construction, so purging must remove some real rows.
        t0 = _dates(10)
        t1 = t0 + pd.Timedelta(days=3)
        splits = purged_kfold_splits(t0, t1, n_splits=5, embargo_days=0)
        for train_idx, test_idx in splits:
            # With overlapping spans, at least one non-test row must be purged.
            assert len(train_idx) + len(test_idx) < 10

    def test_embargo_purges_additional_trailing_train_rows(self):
        t0 = _dates(20)
        no_embargo = purged_kfold_splits(t0, t0, n_splits=4, embargo_days=0)
        with_embargo = purged_kfold_splits(t0, t0, n_splits=4, embargo_days=3)
        for (train_a, _test_a), (train_b, _test_b) in zip(no_embargo, with_embargo):
            assert len(train_b) <= len(train_a)

    def test_embargo_removes_samples_immediately_after_test_fold(self):
        # A single test point at day 10; a lone train-adjacent sample the
        # very next day must be purged with a >=1-day embargo, and survive
        # (with embargo_days=0 and no span overlap) without one.
        dates = pd.to_datetime(["2024-01-10", "2024-01-11"])
        t0 = pd.Series(dates)
        # Force day 10 alone into its own fold by using n_splits == n (each
        # sample its own fold) -- day 10 is sorted first, so fold 0 is it.
        splits_no_embargo = purged_kfold_splits(t0, t0, n_splits=2, embargo_days=0)
        splits_with_embargo = purged_kfold_splits(t0, t0, n_splits=2, embargo_days=2)

        train_no_embargo, test_fold0 = splits_no_embargo[0]
        train_with_embargo, _ = splits_with_embargo[0]
        assert test_fold0.tolist() == [0]  # day-10 row is the test fold
        assert 1 in train_no_embargo.tolist()  # day-11 row untouched without embargo
        assert 1 not in train_with_embargo.tolist()  # purged once embargo covers day 11

    def test_purely_disjoint_events_are_never_purged(self):
        # Widely-spaced point events -- no overlap possible, purge should
        # remove nothing beyond the test fold itself.
        t0 = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"])
        splits = purged_kfold_splits(t0, t0, n_splits=2, embargo_days=5)
        for train_idx, test_idx in splits:
            assert len(train_idx) + len(test_idx) == 4
