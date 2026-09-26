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

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from fit_pattern_calibration import (  # noqa: E402
    _MIN_RESOLVED_FOR_CALIBRATION,
    _resolved_sample_from_history,
    build_calibration,
)

from firm.live.pattern_scan_history import PatternScanHistoryStore  # noqa: E402


class TestBuildCalibration:
    def test_refuses_below_minimum_sample_size(self):
        n = _MIN_RESOLVED_FOR_CALIBRATION - 1
        scores = np.linspace(0.1, 0.9, n)
        labels = np.array([1.0 if s > 0.5 else 0.0 for s in scores])
        assert build_calibration(scores, labels) is None

    def test_fits_at_minimum_sample_size(self):
        rng = np.random.RandomState(0)
        n = _MIN_RESOLVED_FOR_CALIBRATION
        scores = rng.uniform(0.0, 1.0, n)
        labels = (scores + rng.normal(0, 0.1, n) > 0.5).astype(float)
        result = build_calibration(scores, labels)
        assert result is not None
        assert result["type"] == "sigmoid"
        assert result["n_samples"] == n
        assert "a" in result and "b" in result

    def test_higher_scores_predict_higher_calibrated_probability(self):
        # A well-separated sample: low scores lose, high scores win.
        rng = np.random.RandomState(1)
        n = 200
        scores = rng.uniform(0.0, 1.0, n)
        labels = (scores > 0.5).astype(float)
        result = build_calibration(scores, labels)
        assert result is not None

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
