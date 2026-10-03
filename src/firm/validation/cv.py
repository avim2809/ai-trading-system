"""Purged K-fold and combinatorial purged cross-validation.

References: Lopez de Prado, *Advances in Financial Machine Learning* (2018),
ch. 7 (purging and embargo) and ch. 12 (combinatorial purged CV).

Conventions
-----------
* ``label_end`` is a ``pd.Series`` indexed by the sample start time ``t0``
  (sorted ascending, ties allowed) whose values are the label end time ``t1``
  (``>= t0``). Index arrays returned by ``split`` are POSITIONAL into that
  series.
* Purge: a train sample is dropped if its label interval ``[t0, t1]`` overlaps
  the ``[min t0, max t1]`` hull of a test block. For CPCV every test group is
  purged against separately (the groups are not contiguous).
* Embargo: after each contiguous test block the next ``ceil(embargo_pct * T)``
  samples (by position in time order) are also dropped from train. Adjacent
  test groups are merged into one block first, so the embargo starts at the end
  of the merged block (an embargo inside a test group would be a no-op).
* ``embargo_pct`` is a fraction of the sample count ``T``. The legacy
  ``firm.patterns.ml.purged_cv.purged_kfold_splits`` instead embargoes a number
  of CALENDAR DAYS after the test fold's ``t1``; with irregular sampling the
  two differ (see ``legacy_day_embargo_splits``). The legacy function is left
  untouched because ``scripts/train_pattern_ml.py`` depends on it.
* No randomness anywhere; CPCV split order is ``itertools.combinations`` order.

CPCV is intended for ML components and robustness reporting on the small
pre-registered grid. It is not a search tool for trend parameters.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd


def _validate_label_end(label_end: pd.Series | None) -> tuple[np.ndarray, np.ndarray]:
    if label_end is None:
        raise ValueError("label_end is required")
    t0 = pd.DatetimeIndex(pd.to_datetime(label_end.index)).to_numpy("datetime64[ns]").astype(np.int64)
    t1 = pd.to_datetime(pd.Series(label_end.to_numpy())).to_numpy("datetime64[ns]").astype(np.int64)
    if len(t0) == 0:
        raise ValueError("label_end is empty")
    if np.any(np.diff(t0) < 0):
        raise ValueError("label_end index (t0) must be sorted ascending")
    if np.any(t1 < t0):
        raise ValueError("label_end values (t1) must be >= their t0")
    return t0, t1


def _embargo_count(embargo_pct: float, n: int) -> int:
    if not (0.0 <= embargo_pct < 1.0):
        raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")
    return int(math.ceil(embargo_pct * n - 1e-9))


def _train_indices(
    t0: np.ndarray,
    t1: np.ndarray,
    test_groups: list[np.ndarray],
    embargo_n: int,
    embargo_ns: int = 0,
) -> np.ndarray:
    """Train positions for a test set made of ``test_groups`` (sorted position arrays)."""
    n = len(t0)
    mask = np.ones(n, dtype=bool)
    for g in test_groups:
        mask[g] = False
    for g in test_groups:
        g_t0, g_t1 = t0[g].min(), t1[g].max()
        mask &= ~((t1 >= g_t0) & (t0 <= g_t1 + embargo_ns))
    if embargo_n > 0:
        # merge adjacent groups into blocks; embargo after each block's last position
        for g in test_groups:
            end = int(g[-1]) + 1
            nxt = [h for h in test_groups if h[0] == end]
            if nxt:
                continue  # followed directly by another test group: embargo starts after that one
            mask[end : end + embargo_n] = False
        for g in test_groups:  # test samples themselves are never train
            mask[g] = False
    return np.flatnonzero(mask)


class PurgedKFold:
    """Time-ordered K-fold with purging and a percentage embargo."""

    def __init__(self, n_splits: int, label_end: pd.Series, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = int(n_splits)
        self._t0, self._t1 = _validate_label_end(label_end)
        if len(self._t0) < n_splits:
            raise ValueError(f"n_splits={n_splits} exceeds sample count {len(self._t0)}")
        self.embargo_pct = embargo_pct
        self._embargo_n = _embargo_count(embargo_pct, len(self._t0))

    def split(self, X=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        folds = np.array_split(np.arange(len(self._t0)), self.n_splits)
        for test_idx in folds:
            yield _train_indices(self._t0, self._t1, [test_idx], self._embargo_n), test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class CombinatorialPurgedCV:
    """CPCV with ``n_groups`` contiguous groups, ``n_test_groups`` of which are test per split.

    Yields ``C(N, k)`` splits and ``phi = C(N, k) * k / N`` backtest paths.
    """

    def __init__(
        self,
        n_groups: int = 10,
        n_test_groups: int = 2,
        label_end: pd.Series | None = None,
        embargo_pct: float = 0.01,
    ) -> None:
        if not (1 <= n_test_groups < n_groups):
            raise ValueError("need 1 <= n_test_groups < n_groups")
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self._t0, self._t1 = _validate_label_end(label_end)
        if len(self._t0) < n_groups:
            raise ValueError(f"n_groups={n_groups} exceeds sample count {len(self._t0)}")
        self.embargo_pct = embargo_pct
        self._embargo_n = _embargo_count(embargo_pct, len(self._t0))
        self._groups = np.array_split(np.arange(len(self._t0)), self.n_groups)
        self._combos = list(itertools.combinations(range(self.n_groups), self.n_test_groups))

    def split(self, X=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for combo in self._combos:
            test_groups = [self._groups[g] for g in combo]
            test_idx = np.concatenate(test_groups)
            yield _train_indices(self._t0, self._t1, test_groups, self._embargo_n), test_idx

    def get_n_splits(self) -> int:
        return math.comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        return math.comb(self.n_groups, self.n_test_groups) * self.n_test_groups // self.n_groups

    def split_groups(self) -> list[tuple[int, ...]]:
        return list(self._combos)

    def backtest_paths(self, fold_predictions: Mapping[int, pd.Series | np.ndarray]) -> list[np.ndarray]:
        """Assemble ``n_paths`` complete out-of-sample series.

        ``fold_predictions[split_id]`` holds the OOS values for that split's test
        samples (test groups concatenated in time order). For each group, its
        ``phi`` blocks (one per containing split, in split order) go to paths
        ``0..phi-1``; blocks are then concatenated in group (time) order.
        """
        n_splits = self.get_n_splits()
        missing = set(range(n_splits)) - set(fold_predictions)
        if missing:
            raise ValueError(f"fold_predictions missing split ids {sorted(missing)[:5]}")
        group_blocks: list[list[np.ndarray]] = [[] for _ in range(self.n_groups)]
        for sid, combo in enumerate(self._combos):
            pred = np.asarray(fold_predictions[sid])
            sizes = [len(self._groups[g]) for g in combo]
            if pred.ndim != 1 or len(pred) != sum(sizes):
                raise ValueError(
                    f"split {sid}: expected 1-d predictions of length {sum(sizes)}, got shape {pred.shape}"
                )
            off = 0
            for g, sz in zip(combo, sizes):
                group_blocks[g].append(pred[off : off + sz])
                off += sz
        phi = self.n_paths
        paths = []
        for j in range(phi):
            paths.append(np.concatenate([group_blocks[g][j] for g in range(self.n_groups)]))
        total = len(self._t0)
        if any(len(g) != phi for g in group_blocks) or any(len(p) != total for p in paths):
            raise ValueError("backtest paths are not complete and of equal length")
        return paths


def legacy_day_embargo_splits(t0, t1, n_splits: int, embargo_days: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Day-based embargo with the legacy semantics, built on this module's purge core.

    Reproduces ``firm.patterns.ml.purged_cv.purged_kfold_splits(..., embargo_days=...)``
    (positional indices into the ORIGINAL order, folds cut over the t0-sorted order).
    Used only by the equivalence test; new work should use ``embargo_pct``.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    a = pd.to_datetime(pd.Series(t0)).reset_index(drop=True).to_numpy("datetime64[ns]").astype(np.int64)
    b = pd.to_datetime(pd.Series(t1)).reset_index(drop=True).to_numpy("datetime64[ns]").astype(np.int64)
    if len(a) != len(b):
        raise ValueError("t0 and t1 must be the same length")
    if len(a) < n_splits:
        raise ValueError("n_splits exceeds sample count")
    order = np.argsort(a, kind="stable")
    out = []
    for fold in np.array_split(np.arange(len(a)), n_splits):
        test_idx = np.sort(order[fold])
        train = _train_indices(a, b, [test_idx], 0, int(embargo_days) * 86_400 * 10**9)
        out.append((train, test_idx))
    return out
