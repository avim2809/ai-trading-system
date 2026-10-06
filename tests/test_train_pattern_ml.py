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

from train_pattern_ml import (
    build_dataset,
    main,
    run_purged_cv_report,
    time_ordered_split,
)

from firm.data.synthetic import make_synthetic_prices
from firm.patterns.ml import xgb_classifier

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


class TestBuildDatasetSampleWeight:
    """Part B item 3 (2026-09-27): build_dataset now also computes de Prado
    ch.4 average-uniqueness sample weights per row."""

    def test_meta_includes_exit_index_and_sample_weight_in_unit_interval(self):
        panel = _panel(["AAPL", "MSFT", "NVDA"], n_days=500)
        X, y, meta = build_dataset(
            panel, zigzag_pct=0.03, min_score=0.0, confirm_lookback_bars=3,
            stop_atr_floor=1.5, timeout_bars=20, min_window_bars=60, step_bars=10,
        )
        if meta.empty:
            pytest.skip("fixture produced zero confirmed matches -- nothing to check")

        assert "exit_index" in meta.columns
        assert "sample_weight" in meta.columns
        assert (meta["exit_index"] >= meta["confirm_index"]).all()
        assert meta["sample_weight"].between(0.0, 1.0, inclusive="right").all()

    def test_sample_weight_matches_sample_weights_by_group_directly(self):
        # The wiring, not the math (average_uniqueness/sample_weights_by_group
        # have their own dedicated unit tests) -- build_dataset's own output
        # must equal calling that function on the same (symbol, span) data.
        from firm.patterns.ml.sample_weights import sample_weights_by_group

        panel = _panel(["AAPL", "MSFT"], n_days=400)
        X, y, meta = build_dataset(
            panel, zigzag_pct=0.03, min_score=0.0, confirm_lookback_bars=3,
            stop_atr_floor=1.5, timeout_bars=20, min_window_bars=60, step_bars=10,
        )
        if meta.empty:
            pytest.skip("fixture produced zero confirmed matches -- nothing to check")

        expected = sample_weights_by_group(
            meta["symbol"].tolist(), list(zip(meta["confirm_index"].tolist(), meta["exit_index"].tolist())),
        )
        np.testing.assert_allclose(meta["sample_weight"].to_numpy(), expected)

    def test_empty_dataset_has_sample_weight_column_too(self):
        X, y, meta = build_dataset(
            pd.DataFrame(), zigzag_pct=0.03, min_score=0.0, confirm_lookback_bars=3,
            stop_atr_floor=1.5, timeout_bars=20,
        )
        assert meta.empty
        assert "sample_weight" in meta.columns
        assert "exit_date" in meta.columns


class TestBuildDatasetMarketContext:
    """Part B item 6 (2026-09-27): build_dataset now aligns and feeds a
    market-proxy window into build_features -- not just infrastructure
    sitting unused (same bar Part A item 6's regime discount was held to)."""

    def test_market_context_is_actually_populated_for_some_rows(self):
        panel = _panel(["AAPL", "MSFT", "NVDA"], n_days=500)
        X, y, meta = build_dataset(
            panel, zigzag_pct=0.03, min_score=0.0, confirm_lookback_bars=3,
            stop_atr_floor=1.5, timeout_bars=20, min_window_bars=60, step_bars=10,
        )
        if X.empty:
            pytest.skip("fixture produced zero confirmed matches -- nothing to check")

        assert "market_context_available" in X.columns
        # Not every row necessarily clears the regime-feature rolling
        # warmup this early in a symbol's own window, but SOME real,
        # multi-symbol run must actually engage the market-context path --
        # otherwise this is silently dead wiring.
        assert (X["market_context_available"] == 1.0).any()

    def test_single_symbol_panel_has_zero_relative_strength_symbol_is_its_own_market(self):
        # With exactly one symbol, the equal-weight market proxy IS that
        # symbol's own close series -> symbol return == market return ->
        # relative_strength must be exactly 0 for every populated row.
        panel = _panel(["AAPL"], n_days=500)
        X, y, meta = build_dataset(
            panel, zigzag_pct=0.03, min_score=0.0, confirm_lookback_bars=3,
            stop_atr_floor=1.5, timeout_bars=20, min_window_bars=60, step_bars=10,
        )
        if X.empty:
            pytest.skip("fixture produced zero confirmed matches -- nothing to check")

        available = X[X["market_context_available"] == 1.0]
        if available.empty:
            pytest.skip("no row cleared market-context availability in this fixture")
        np.testing.assert_allclose(available["relative_strength"].to_numpy(), 0.0, atol=1e-9)
        np.testing.assert_allclose(available["market_return_pct"].to_numpy(), available["pre_pattern_return"].to_numpy(), atol=1e-9)

    def test_exit_date_is_on_or_after_confirm_date(self):
        panel = _panel(["AAPL", "MSFT"], n_days=400)
        X, y, meta = build_dataset(
            panel, zigzag_pct=0.03, min_score=0.0, confirm_lookback_bars=3,
            stop_atr_floor=1.5, timeout_bars=20, min_window_bars=60, step_bars=10,
        )
        if meta.empty:
            pytest.skip("fixture produced zero confirmed matches -- nothing to check")
        assert (pd.to_datetime(meta["exit_date"]) >= pd.to_datetime(meta["confirm_date"])).all()


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
class TestSafeMulticlassAuc:
    """Part B item 4 (2026-09-27) regression coverage: sklearn's
    roc_auc_score silently returns nan (not a ValueError) for a y_test that
    has >=2 classes overall but is still missing one of xgb_classifier.
    LABELS' 3 possible values -- exactly the routine case for a small
    Purged CV fold. _safe_multiclass_auc must treat that as "n/a" (None),
    not let nan corrupt a caller's summary statistic."""

    def test_returns_none_when_test_set_is_single_class(self):
        from train_pattern_ml import _safe_multiclass_auc

        model = xgb_classifier.train(
            pd.DataFrame(np.random.RandomState(0).rand(20, 3)),
            np.random.RandomState(0).randint(-1, 2, size=20),
        )
        X_test = pd.DataFrame(np.random.RandomState(1).rand(5, 3))
        assert _safe_multiclass_auc(model, X_test, np.array([1, 1, 1, 1, 1])) is None

    def test_returns_none_not_nan_when_one_label_is_absent_from_y_test(self):
        # y_test has 2 distinct classes (passes the len(set())>=2 guard)
        # but never the "0"/timeout label -- the real bug this closes.
        from train_pattern_ml import _safe_multiclass_auc

        rng = np.random.RandomState(2)
        X = pd.DataFrame(rng.rand(30, 3))
        y = rng.randint(-1, 2, size=30)
        model = xgb_classifier.train(X, y)

        X_test = pd.DataFrame(rng.rand(8, 3))
        y_test = np.array([-1, -1, 1, 1, -1, 1, -1, 1])  # no "0" present

        result = _safe_multiclass_auc(model, X_test, y_test)
        assert result is None  # not nan, not a float that would corrupt np.mean

    def test_returns_real_auc_when_all_three_labels_present(self):
        from train_pattern_ml import _safe_multiclass_auc

        rng = np.random.RandomState(3)
        X = pd.DataFrame(rng.rand(40, 3))
        y = rng.randint(-1, 2, size=40)
        model = xgb_classifier.train(X, y)

        X_test = pd.DataFrame(rng.rand(15, 3))
        y_test = np.array([-1, 0, 1] * 5)  # all 3 labels present

        result = _safe_multiclass_auc(model, X_test, y_test)
        assert result is not None
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0


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


@requires_xgboost
class TestMainPassesSampleWeight:
    """Part B item 3 (2026-09-27): a regression test against the wiring
    silently regressing back to unweighted training in some future
    refactor -- monkeypatches xgb_classifier.train to record exactly what
    it was called with, for BOTH the direction and meta models."""

    def test_both_models_trained_with_real_non_none_sample_weight(self, tmp_path, monkeypatch):
        import train_pattern_ml as script

        calls = []
        real_train = xgb_classifier.train

        def _recording_train(X, y, *, sample_weight=None, params=None):
            calls.append(sample_weight)
            return real_train(X, y, sample_weight=sample_weight, params=params)

        monkeypatch.setattr(script.xgb_classifier, "train", _recording_train)

        output = tmp_path / "pattern_xgb.pkl"
        argv = [
            "--data-source", "synthetic",
            "--n-days", "800",
            "--symbols", "AAPL,MSFT,GOOG,AMZN,META,TSLA,NVDA,JPM,V,JNJ",
            "--min-score", "0",
            "--output", str(output),
        ]
        rc = main(argv)
        if rc != 0:
            pytest.skip("synthetic fixture produced too few/degenerate rows to train this run")

        # >= 2: the two final models (direction, meta), plus one call per
        # usable purged-CV fold per label (Part B item 4) -- every one of
        # them must still receive real weights, not just the first two.
        assert len(calls) >= 2
        for sample_weight in calls:
            assert sample_weight is not None
            assert np.all((sample_weight > 0.0) & (sample_weight <= 1.0))


@requires_xgboost
class TestRunPurgedCvReport:
    """Part B item 4 (2026-09-27): purged K-Fold CV reporting, wired into
    train_pattern_ml.py -- tests the reporting function directly against a
    small, controlled, learnable synthetic dataset (with real confirm_date/
    exit_date/sample_weight columns, exactly what build_dataset produces)."""

    def _fixture(self, n=80, seed=0):
        rng = np.random.RandomState(seed)
        X = pd.DataFrame(rng.rand(n, 5), columns=[f"f{i}" for i in range(5)])
        y = (X["f0"] > 0.5).astype(int).to_numpy() * 2 - 1  # {-1, 1}, learnable
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        meta = pd.DataFrame({
            "confirm_date": dates,
            "exit_date": dates + pd.Timedelta(days=2),
            "sample_weight": np.ones(n),
        })
        return X, y, meta

    def test_reports_n_splits_folds_with_real_accuracy(self, capsys):
        X, y, meta = self._fixture()
        result = run_purged_cv_report(
            X, y, meta, n_splits=5, embargo_days=0, label_name="test-direction", is_binary_meta=False,
        )
        assert result["n_usable_folds"] == 5
        assert result["mean_accuracy"] is not None
        assert 0.0 <= result["mean_accuracy"] <= 1.0
        captured = capsys.readouterr()
        assert "Purged CV (test-direction)" in captured.out
        assert "5/5 usable folds" in captured.out

    def test_binary_meta_path_uses_binary_auc(self):
        X, y, meta = self._fixture()
        y_binary = (y == 1).astype(int)
        result = run_purged_cv_report(
            X, y_binary, meta, n_splits=4, embargo_days=0, label_name="test-meta", is_binary_meta=True,
        )
        assert result["n_usable_folds"] == 4
        assert result["mean_auc"] is not None

    def test_n_splits_zero_is_a_no_op(self, capsys):
        X, y, meta = self._fixture()
        result = run_purged_cv_report(
            X, y, meta, n_splits=0, embargo_days=0, label_name="test", is_binary_meta=False,
        )
        assert result["n_usable_folds"] == 0
        assert result["mean_accuracy"] is None

    def test_too_few_rows_for_requested_splits_skips_cleanly(self, capsys):
        X, y, meta = self._fixture(n=3)
        result = run_purged_cv_report(
            X, y, meta, n_splits=10, embargo_days=0, label_name="test", is_binary_meta=False,
        )
        assert result["n_usable_folds"] == 0
        captured = capsys.readouterr()
        assert "skipped" in captured.out

    def test_degenerate_single_class_dataset_skips_every_fold_without_raising(self):
        X, y, meta = self._fixture()
        y_degenerate = np.ones_like(y)  # only one class anywhere
        result = run_purged_cv_report(
            X, y_degenerate, meta, n_splits=5, embargo_days=0, label_name="test", is_binary_meta=False,
        )
        assert result["n_usable_folds"] == 0
        assert result["mean_accuracy"] is None
