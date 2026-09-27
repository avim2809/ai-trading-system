"""Purged K-Fold cross-validation (Lopez de Prado, *Advances in Financial
Machine Learning*, ch. 7) for the pattern-confirmation classifiers.

Why this exists (Part B item 4, 2026-09-27): ``scripts/train_pattern_ml.py``
only ever produces ONE trailing time-ordered train/test split
(``time_ordered_split``) with a simple calendar-day embargo dropped from the
train period immediately before the test cutoff. That is a reasonable FINAL
holdout for the number ultimately reported, but it is a single split -- one
sample of "how well does this generalize," noisy by construction, and it
purges only at one boundary (the trailing edge of train, immediately before
test). It is not itself a cross-validation scheme, and this codebase had
none for the classifier's own training (the existing walk-forward/PBO
machinery in :mod:`firm.eval.overfitting` operates at the PORTFOLIO level,
across whole backtest runs -- a different, already-adequate tool for a
different job; this module is specifically for the ML classifier).

De Prado's fix (ch. 7.4, "PurgedKFold"): split the time-ordered samples into
``n_splits`` contiguous folds, each used once as a test set; for every fold,
purge from the corresponding TRAINING set any sample whose own label-
determination window ``[t0, t1]`` overlaps that test fold's date range (a
label is not "resolved" until ``t1``, so a training sample whose outcome
was only just becoming known during the test period leaks test-period
information into training), plus an additional embargo window of
``embargo_days`` immediately after the test fold ends (a sample starting
right after test may still be influenced by serial correlation/volatility
clustering carried over from the test period).

Scope note: this implements standard (non-combinatorial) Purged K-Fold --
each fold serves as the test set exactly once. The plan this was scoped
against explicitly allows this as the minimum acceptable bar ("implement
combinatorial purged CV, OR AT MINIMUM proper per-label purging using each
sample's own t1/barrier-touch time"); full Combinatorial Purged CV (ch. 12
-- every C(N, k) combination of test groups, with a backtest-path
reconstruction) is a real, further enhancement, not implemented here to
keep this change focused and testable.

Fold assignment and purging both operate on **calendar dates**
(``confirm_date``/``exit_date``), not raw per-symbol bar positions --
unlike :mod:`firm.patterns.ml.sample_weights`'s uniqueness weighting (which
is inherently about same-symbol scanner-window overlap and only meaningful
within one symbol's own bar timeline), fold assignment for cross-validated
model evaluation should reflect genuine calendar time across the WHOLE
training set (multiple symbols at once) -- a training row for one symbol
whose label window overlaps a test fold's calendar period is a real
leakage risk even though its raw bar position (into its own symbol's
array) has nothing to do with another symbol's.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def purged_kfold_splits(
    t0,
    t1,
    *,
    n_splits: int,
    embargo_days: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """De Prado ch. 7.4 Purged K-Fold: ``n_splits`` contiguous, time-ordered
    folds, each used once as the test set, with train purged of any sample
    overlapping (or within ``embargo_days`` after) that fold's own date range.

    Args:
        t0: Per-sample label-window START (e.g. ``confirm_date``) --
            anything :func:`pandas.to_datetime` accepts.
        t1: Per-sample label-window END (e.g. ``exit_date`` -- see
            :func:`firm.patterns.ml.labeling.label_triple_barrier_with_exit`),
            same length as ``t0``. ``t1 >= t0`` for every sample.
        n_splits: Number of folds (``>= 2``).
        embargo_days: Calendar days of additional purge immediately after
            each test fold's own ``t1`` (see module docstring). ``0``
            (default) disables the embargo, leaving pure overlap-based
            purging.

    Returns:
        A list of ``n_splits`` ``(train_idx, test_idx)`` pairs -- integer
        positional index arrays into the ORIGINAL (not time-sorted) input,
        each sorted ascending. Fold boundaries are contiguous in *sorted*
        time (samples don't need to arrive pre-sorted -- this function
        sorts internally, exactly like
        :func:`scripts.train_pattern_ml.time_ordered_split` re-sorts by
        ``confirm_date`` rather than trusting input order).
    """
    if n_splits < 2:
        raise ValueError(f"purged_kfold_splits: n_splits must be >= 2, got {n_splits}")

    t0_arr = pd.to_datetime(pd.Series(t0)).reset_index(drop=True)
    t1_arr = pd.to_datetime(pd.Series(t1)).reset_index(drop=True)
    n = len(t0_arr)
    if n != len(t1_arr):
        raise ValueError(f"purged_kfold_splits: t0 (n={n}) and t1 (n={len(t1_arr)}) must be the same length")
    if n < n_splits:
        raise ValueError(f"purged_kfold_splits: n_splits={n_splits} exceeds sample count n={n}")

    order = np.argsort(t0_arr.to_numpy(), kind="stable")
    fold_positions = np.array_split(np.arange(n), n_splits)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in fold_positions:
        test_idx = np.sort(order[fold])
        test_t0 = t0_arr.iloc[test_idx].min()
        test_t1 = t1_arr.iloc[test_idx].max()
        purge_end = test_t1 + pd.Timedelta(days=embargo_days)

        # Purge: any sample (train or not) whose own span overlaps
        # [test_t0, purge_end] -- standard interval-overlap test, extended
        # past the test fold's own end by the embargo.
        overlaps = (t1_arr >= test_t0) & (t0_arr <= purge_end)

        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_mask &= ~overlaps.to_numpy()
        train_idx = np.flatnonzero(train_mask)
        splits.append((train_idx, test_idx))

        log.debug(
            "purged_kfold_splits: fold test=[%s, %s] (n_test=%d) -- purged %d overlapping/"
            "embargoed sample(s) from train (n_train=%d)",
            test_t0, test_t1, len(test_idx),
            int(np.sum(overlaps.to_numpy() & ~np.isin(np.arange(n), test_idx))), len(train_idx),
        )

    return splits
