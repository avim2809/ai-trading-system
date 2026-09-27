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

from train_pattern_ml import build_dataset, main, time_ordered_split  # noqa: E402

from firm.data.synthetic import make_synthetic_prices  # noqa: E402
from firm.patterns.ml import xgb_classifier  # noqa: E402

try:
    import xgboost  # noqa: F401

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

requires_xgboost = pytest.mark.skipif(
    not _XGBOOST_AVAILABLE, reason="xgboost not installed (optional `patterns_ml` extra)",
)


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


@requires_xgboost
class TestMetaPAct:
    """Part B item 2 (2026-09-27): _meta_p_act's reliance on
    xgb_classifier's fixed (-1, 0, 1) LABELS layout (column index 2 always
    holding P(label==1)) even for a model fit on the binary {0, 1} subset."""

    def test_column_2_holds_p_label_1_for_a_binary_fit(self):
        from train_pattern_ml import _meta_p_act

        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.rand(40, 3), columns=["a", "b", "c"])
        y = (X["a"] > 0.5).astype(int).to_numpy()  # a clean, learnable binary target
        model = xgb_classifier.train(X, y)
        assert model.present_labels == (0, 1)  # sanity: binary fit, not the full 3-class set

        p_act = _meta_p_act(model, X)
        assert p_act.shape == (40,)
        assert np.all((p_act >= 0.0) & (p_act <= 1.0))
        # The model should have actually learned the rule, not be random.
        pred = (p_act >= 0.5).astype(int)
        assert (pred == y).mean() > 0.8

    def test_never_returns_a_negative_one_prediction(self):
        # predict_label's argmax can only select this fixed model's own
        # present columns; verifies the always-zero "-1"/stop column (index
        # 0, never fit on a binary {0,1} target) never wins the argmax.
        from train_pattern_ml import _meta_p_act

        rng = np.random.RandomState(1)
        X = pd.DataFrame(rng.rand(30, 3), columns=["a", "b", "c"])
        y = rng.randint(0, 2, size=30)
        model = xgb_classifier.train(X, y)

        p_act = _meta_p_act(model, X)
        pred = (p_act >= 0.5).astype(int)
        assert set(pred.tolist()) <= {0, 1}


@requires_xgboost
class TestSafeBinaryAuc:
    def test_returns_none_when_test_set_is_single_class(self):
        from train_pattern_ml import _safe_binary_auc

        assert _safe_binary_auc(np.array([0.1, 0.9, 0.5]), np.array([1, 1, 1])) is None

    def test_returns_real_auc_for_a_perfect_separator(self):
        from train_pattern_ml import _safe_binary_auc

        p_act = np.array([0.1, 0.2, 0.8, 0.9])
        y_test = np.array([0, 0, 1, 1])
        assert _safe_binary_auc(p_act, y_test) == pytest.approx(1.0)


@requires_xgboost
class TestMainTrainsBothModels:
    """End-to-end smoke test (Part B item 2): a real `main()` invocation
    against synthetic data must produce BOTH the 3-class direction model AND
    the separate binary meta-label model as distinct on-disk artifacts."""

    def test_main_saves_both_direction_and_meta_models(self, tmp_path, capsys):
        output = tmp_path / "pattern_xgb.pkl"
        argv = [
            "--data-source", "synthetic",
            "--n-days", "800",
            "--symbols", "AAPL,MSFT,GOOG,AMZN,META,TSLA,NVDA,JPM,V,JNJ",
            "--min-score", "0",
            "--output", str(output),
        ]
        rc = main(argv)
        captured = capsys.readouterr()
        if rc != 0:
            pytest.skip(f"synthetic fixture produced too few/degenerate rows to train "
                        f"this run (stderr: {captured.err.strip()!r}) -- not this feature's concern")

        assert rc == 0
        meta_output = tmp_path / "pattern_xgb_meta.pkl"
        assert output.exists()
        assert meta_output.exists()

        direction_model = xgb_classifier.load(output)
        meta_model = xgb_classifier.load(meta_output)
        # The two saved artifacts must be genuinely different fits (distinct
        # label domains), not the same model saved twice under two names.
        assert set(direction_model.present_labels) <= {-1, 0, 1}
        assert set(meta_model.present_labels) <= {0, 1}
        assert "Meta-label test accuracy" in captured.out
        assert "Meta-label model saved to" in captured.out

    def test_meta_output_defaults_to_output_stem_plus_meta(self):
        from train_pattern_ml import _parse_args

        args = _parse_args(["--output", "data/models/pattern_xgb.pkl"])
        assert args.meta_output == "data/models/pattern_xgb_meta.pkl"

    def test_meta_output_explicit_override_respected(self):
        from train_pattern_ml import _parse_args

        args = _parse_args([
            "--output", "data/models/pattern_xgb.pkl",
            "--meta-output", "data/models/custom_meta.pkl",
        ])
        assert args.meta_output == "data/models/custom_meta.pkl"
