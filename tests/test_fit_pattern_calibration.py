"""Tests for scripts/fit_pattern_calibration.py.

Pure-function tests for build_calibration (the fit/refuse decision) plus
the history-DB sample-extraction helper against a real, tmp-path
PatternScanHistoryStore. Imports the script module directly by adding
scripts/ to sys.path, mirroring tests/test_pattern_recognition_rollout_gate.py
(scripts/ is not a package).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from fit_pattern_calibration import (  # noqa: E402
    _MIN_RESOLVED_FOR_CALIBRATION,
    _resolved_sample_from_history,
    _resolved_sample_from_synthetic_rule_based,
    _resolved_sample_from_synthetic_xgboost,
    build_calibration,
    main,
)

from firm.live.pattern_scan_history import PatternScanHistoryStore  # noqa: E402


def _xgb_model_is_stale_relative_to_current_feature_schema() -> bool:
    """True if the real on-disk pattern_xgb.onnx artifact's expected input
    width no longer matches firm.patterns.ml.feature_engineering.build_features'
    current column count -- i.e. the model needs retraining (see Part B item
    7) before any test that scores real synthetic matches through it can
    assert anything meaningful. Self-healing: once the model is retrained
    against the current schema, this returns False again and the skip above
    stops firing, with no further test maintenance needed.
    """
    from firm.patterns.ml import xgb_inference

    if not xgb_inference.DEFAULT_MODEL_PATH.exists():
        return False  # "not available" is a separate, already-handled skip
    try:
        import onnxruntime as ort

        from firm.patterns.match import PatternMatch
        from firm.patterns.ml.feature_engineering import build_features

        session = ort.InferenceSession(
            str(xgb_inference.DEFAULT_MODEL_PATH), providers=["CPUExecutionProvider"],
        )
        expected = session.get_inputs()[0].shape[1]
        dummy = PatternMatch(
            pattern="bull_flag", direction="long", pivots=(), confirm_index=1,
            entry=100.0, stop=95.0, target=110.0, fit_quality=1.0,
            geometry_tolerance_used=1.0, volume_ratio=1.0, duration_bars=1,
            follow_through_atr=1.0, risk_reward=1.0, quality_score=1.0,
            score_breakdown={},
        )
        actual = len(build_features(dummy))
        return isinstance(expected, int) and expected != actual
    except Exception:
        return False  # can't tell -- let the real test run and surface any issue


class TestBuildCalibration:
    def test_refuses_below_minimum_sample_size(self):
        n = _MIN_RESOLVED_FOR_CALIBRATION - 1
        scores = np.linspace(0.1, 0.9, n)
        labels = np.array([1.0 if s > 0.5 else 0.0 for s in scores])
        assert build_calibration(scores, labels, model="rule_based") is None

    def test_fits_at_minimum_sample_size(self):
        rng = np.random.RandomState(0)
        n = _MIN_RESOLVED_FOR_CALIBRATION
        scores = rng.uniform(0.0, 1.0, n)
        labels = (scores + rng.normal(0, 0.1, n) > 0.5).astype(float)
        result = build_calibration(scores, labels, model="rule_based")
        assert result is not None
        assert result["type"] == "sigmoid"
        assert result["model"] == "rule_based"
        assert result["n_samples"] == n
        assert "a" in result and "b" in result

    def test_rejects_unknown_model(self):
        rng = np.random.RandomState(0)
        n = _MIN_RESOLVED_FOR_CALIBRATION
        scores = rng.uniform(0.0, 1.0, n)
        labels = (scores > 0.5).astype(float)
        with pytest.raises(ValueError, match="model"):
            build_calibration(scores, labels, model="not_a_real_model")

    def test_higher_scores_predict_higher_calibrated_probability(self):
        # A well-separated sample: low scores lose, high scores win.
        rng = np.random.RandomState(1)
        n = 200
        scores = rng.uniform(0.0, 1.0, n)
        labels = (scores > 0.5).astype(float)
        result = build_calibration(scores, labels, model="xgboost")
        assert result is not None
        assert result["model"] == "xgboost"

        from firm.patterns.ml.calibration import apply_sigmoid_calibration

        low = apply_sigmoid_calibration(np.array([0.05]), result["a"], result["b"])[0]
        high = apply_sigmoid_calibration(np.array([0.95]), result["a"], result["b"])[0]
        assert high > low


class TestResolvedSampleFromHistory:
    def _row(self, store: PatternScanHistoryStore, *, symbol: str, quality_score: float, outcome: str | None):
        store.insert_matches(
            [
                {
                    "symbol": symbol,
                    "asof": "2024-01-01",
                    "pattern": "double_top",
                    "direction": "short",
                    "confirmed": True,
                    "confirm_index": 10,
                    "entry": 100.0,
                    "stop": 105.0,
                    "target": 90.0,
                    "fit_quality": 0.9,
                    "geometry_tolerance_used": 0.05,
                    "volume_ratio": 1.5,
                    "duration_bars": 20,
                    "follow_through_atr": 1.0,
                    "risk_reward": 2.0,
                    "quality_score": quality_score,
                    "score_breakdown": {},
                    "pivots": [],
                    "meta": {},
                }
            ],
            confirm_dates=["2024-01-01"],
            source="test",
        )
        rows = store.list_history(symbol=symbol, limit=1)
        if outcome is not None:
            store.update_outcome(rows[0]["id"], outcome)

    def test_excludes_pending_and_timeout_rows(self, tmp_path):
        store = PatternScanHistoryStore(db_path=str(tmp_path / "history.db"))
        self._row(store, symbol="AAA", quality_score=80.0, outcome="target_hit")
        self._row(store, symbol="BBB", quality_score=40.0, outcome="stop_hit")
        self._row(store, symbol="CCC", quality_score=60.0, outcome=None)  # pending
        self._row(store, symbol="DDD", quality_score=60.0, outcome="timeout")

        scores, labels = _resolved_sample_from_history(store, score_field="quality_score")

        assert len(scores) == 2
        assert set(labels.tolist()) == {0.0, 1.0}
        assert np.max(scores) <= 1.0  # quality_score/100 scaling applied

    def test_empty_store_returns_empty_arrays(self, tmp_path):
        store = PatternScanHistoryStore(db_path=str(tmp_path / "history.db"))
        scores, labels = _resolved_sample_from_history(store, score_field="quality_score")
        assert len(scores) == 0
        assert len(labels) == 0


class TestSyntheticSamplers:
    """Regression coverage for the 2026-09-27 model-mismatch fix: the
    xgboost sampler must score matches with the REAL trained model's
    p_target, not reuse quality_score, and both samplers should draw from
    the same underlying match set (same seed) so a rule_based vs. xgboost
    comparison is a fair one."""

    def test_rule_based_sampler_uses_quality_score(self):
        scores, labels = _resolved_sample_from_synthetic_rule_based(n_symbols=40, seed=42, timeout_bars=40)
        assert len(scores) > 0
        assert np.all((scores >= 0.0) & (scores <= 1.0))
        assert set(labels.tolist()) <= {0.0, 1.0}

    def test_xgboost_sampler_uses_real_model_not_quality_score(self):
        from firm.patterns.ml import xgb_inference
        if not xgb_inference.is_available():
            pytest.skip("XGBoost ONNX model not available on this host")
        if _xgb_model_is_stale_relative_to_current_feature_schema():
            pytest.skip(
                "on-disk data/models/pattern_xgb.onnx was trained against an older "
                "firm.patterns.ml.feature_engineering.build_features schema (Part B "
                "item 5, 2026-09-27, removed entry/stop/target/confirm_index, added "
                "bars_since_confirm) -- xgb_inference now correctly detects this "
                "count mismatch and returns None for every sample, so this real-"
                "model comparison has nothing to assert until the model is retrained "
                "(Part B item 7, gated on items 1-6 landing first)"
            )

        rule_based_scores, rule_based_labels = _resolved_sample_from_synthetic_rule_based(
            n_symbols=40, seed=42, timeout_bars=40,
        )
        xgb_scores, xgb_labels = _resolved_sample_from_synthetic_xgboost(
            n_symbols=40, seed=42, timeout_bars=40,
        )
        assert len(xgb_scores) > 0
        assert np.all((xgb_scores >= 0.0) & (xgb_scores <= 1.0))
        # Same underlying match set (same seed/labeling) -> same sample
        # count and labels, but genuinely different raw scores (the real
        # model's p_target, not quality_score/100) -- this is exactly the
        # thing the bug this fix closes would NOT have distinguished.
        assert len(xgb_scores) == len(rule_based_scores)
        assert xgb_labels.tolist() == rule_based_labels.tolist()
        assert not np.allclose(xgb_scores, rule_based_scores)

    def test_xgboost_sampler_raises_clearly_when_model_unavailable(self, monkeypatch):
        from firm.patterns.ml import xgb_inference

        monkeypatch.setattr(xgb_inference, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="xgboost"):
            _resolved_sample_from_synthetic_xgboost(n_symbols=5, seed=42, timeout_bars=40)


class TestMainCliGuards:
    def test_xgboost_history_combo_refused(self, tmp_path, capsys):
        # Call via argv injection instead of Namespace construction, since
        # main() owns arg parsing end to end.
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = [
                "fit_pattern_calibration.py", "--model", "xgboost", "--source", "history",
                "--db-path", str(tmp_path / "history.db"),
            ]
            exit_code = main()
        finally:
            _sys.argv = old_argv
        assert exit_code == 1
