"""Tests for scripts/train_cnn_validator.py's time-ordered train/test split.

Mirrors tests/test_train_pattern_ml.py's TestTimeOrderedSplit -- the two
scripts duplicate the same split logic (X here is an image ndarray, not a
DataFrame) rather than sharing it, per this repo's no-cross-script-import
convention. The module imports cleanly in the main venv (torch is present;
only cnn_validator.train/encode_gaf's actual torch/pyts calls are gated),
so this needs no .venv-ml.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from train_cnn_validator import time_ordered_split


class TestTimeOrderedSplit:
    def _fixture(self, n=20):
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        rng = np.random.RandomState(0)
        perm = rng.permutation(n)
        X = np.arange(n)[perm].reshape(n, 1, 1).astype(float)
        y = np.arange(n)[perm]
        meta = pd.DataFrame({"confirm_date": dates.to_numpy()[perm]})
        return X, y, meta

    def test_test_set_is_strictly_later_than_train_set(self):
        X, y, meta = self._fixture(n=20)
        X_train, X_test, y_train, y_test = time_ordered_split(X, y, meta, test_size=0.25)

        assert len(X_test) == 5
        assert len(X_train) == 15
        assert max(y_train) < min(y_test)

    def test_embargo_drops_trailing_train_rows_near_cutoff(self):
        X, y, meta = self._fixture(n=20)
        no_embargo = time_ordered_split(X, y, meta, test_size=0.25, embargo_bars=0)
        with_embargo = time_ordered_split(X, y, meta, test_size=0.25, embargo_bars=5)

        assert len(with_embargo[0]) < len(no_embargo[0])
        assert len(with_embargo[1]) == len(no_embargo[1])

    def test_empty_input_returns_empty_splits(self):
        X = np.empty((0, 1, 1))
        y = np.array([], dtype=int)
        meta = pd.DataFrame({"confirm_date": pd.Series([], dtype="datetime64[ns]")})
        X_train, X_test, y_train, y_test = time_ordered_split(X, y, meta, test_size=0.25)
        assert len(X_train) == 0
        assert len(X_test) == 0

    def test_single_row_input_all_goes_to_train(self):
        X = np.zeros((1, 1, 1))
        y = np.array([0])
        meta = pd.DataFrame({"confirm_date": [datetime(2024, 1, 1)]})
        X_train, X_test, y_train, y_test = time_ordered_split(X, y, meta, test_size=0.25)
        assert len(X_train) == 1
        assert len(X_test) == 0
