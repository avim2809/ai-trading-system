"""Tests for scripts/train_pattern_ml.py's dataset-building and
train/test-split helpers.

No test previously imported this script at all (only the underlying
firm.patterns.ml primitives it calls were tested) -- this closes that gap,
specifically for the time-ordered split that replaced a random
train_test_split on data the module's own docstring calls "highly
autocorrelated." Imports the script module directly by adding scripts/ to
sys.path, mirroring tests/test_pattern_recognition_rollout_gate.py
(scripts/ is not a package).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from train_pattern_ml import build_dataset, time_ordered_split  # noqa: E402

from firm.data.synthetic import make_synthetic_prices  # noqa: E402


def _panel(symbols, n_days=300, seed=7):
    return make_synthetic_prices(symbols, n_days=n_days, seed=seed)


class TestBuildDatasetConfirmDate:
    def test_meta_includes_confirm_date_aligned_to_source_dates(self):
        panel = _panel(["AAPL", "MSFT"])
        X, y, meta = build_dataset(
            panel, zigzag_pct=0.03, min_score=0.0, confirm_lookback_bars=3,
            stop_atr_floor=1.5, timeout_bars=20, min_window_bars=60, step_bars=10,
        )
        if meta.empty:
            pytest.skip("fixture produced zero confirmed matches -- nothing to check")

        assert "confirm_date" in meta.columns
        for _, row in meta.iterrows():
            sym_df = panel[panel["symbol"] == row["symbol"]].sort_values("date").reset_index(drop=True)
            expected_date = sym_df["date"].iloc[row["confirm_index"]]
            assert pd.Timestamp(row["confirm_date"]) == pd.Timestamp(expected_date)


class TestTimeOrderedSplit:
    def _fixture(self, n=20):
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # Shuffle row order on input to prove the split re-sorts by date
        # itself rather than trusting input order.
        rng = np.random.RandomState(0)
        perm = rng.permutation(n)
        X = pd.DataFrame({"f": np.arange(n)[perm]})
        y = np.arange(n)[perm]  # label == original chronological rank, for easy assertions
        meta = pd.DataFrame({"confirm_date": dates.to_numpy()[perm]})
        return X, y, meta

    def test_test_set_is_strictly_later_than_train_set(self):
        X, y, meta = self._fixture(n=20)
        X_train, X_test, y_train, y_test = time_ordered_split(X, y, meta, test_size=0.25)

        assert len(X_test) == 5
        assert len(X_train) == 15
        # y encodes chronological rank -- every train label must precede
        # every test label.
        assert max(y_train) < min(y_test)

    def test_embargo_drops_trailing_train_rows_near_cutoff(self):
        X, y, meta = self._fixture(n=20)
        no_embargo = time_ordered_split(X, y, meta, test_size=0.25, embargo_bars=0)
        with_embargo = time_ordered_split(X, y, meta, test_size=0.25, embargo_bars=5)

        assert len(with_embargo[0]) < len(no_embargo[0])  # X_train shrank
        assert len(with_embargo[1]) == len(no_embargo[1])  # X_test unchanged

    def test_empty_input_returns_empty_splits(self):
        X = pd.DataFrame({"f": []})
        y = np.array([], dtype=int)
        meta = pd.DataFrame({"confirm_date": pd.Series([], dtype="datetime64[ns]")})
        X_train, X_test, y_train, y_test = time_ordered_split(X, y, meta, test_size=0.25)
        assert len(X_train) == 0
        assert len(X_test) == 0

    def test_single_row_input_all_goes_to_train(self):
        X = pd.DataFrame({"f": [1.0]})
        y = np.array([0])
        meta = pd.DataFrame({"confirm_date": [datetime(2024, 1, 1)]})
        X_train, X_test, y_train, y_test = time_ordered_split(X, y, meta, test_size=0.25)
        assert len(X_train) == 1
        assert len(X_test) == 0
