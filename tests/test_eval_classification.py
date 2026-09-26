"""Tests for firm.eval.classification -- precision/recall/F1/PR-AUC/
Brier/reliability, the classification-metric helpers
scripts/benchmark_pattern_detectors.py builds on.
"""

from __future__ import annotations

import numpy as np
import pytest

from firm.eval.classification import (
    binary_classification_report,
    brier_score,
    pr_auc,
    reliability_diagram_bins,
)


class TestBinaryClassificationReport:
    def test_perfect_predictions(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        report = binary_classification_report(y_true, y_pred)
        assert report["precision"] == pytest.approx(1.0)
        assert report["recall"] == pytest.approx(1.0)
        assert report["f1"] == pytest.approx(1.0)

    def test_never_fires_reports_zero_not_error(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        report = binary_classification_report(y_true, y_pred)
        assert report["precision"] == pytest.approx(0.0)
        assert report["recall"] == pytest.approx(0.0)
        assert report["f1"] == pytest.approx(0.0)

    def test_confusion_matrix_shape_and_counts(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 1]
        report = binary_classification_report(y_true, y_pred)
        cm = report["confusion_matrix"]
        assert sum(sum(row) for row in cm) == 4
        assert report["n"] == 4
        assert report["n_positive"] == 2


class TestPrAuc:
    def test_perfect_ranking_scores_one(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        assert pr_auc(y_true, y_score) == pytest.approx(1.0)

    def test_single_class_returns_none(self):
        assert pr_auc([1, 1, 1], [0.9, 0.8, 0.7]) is None
        assert pr_auc([0, 0, 0], [0.1, 0.2, 0.3]) is None


class TestBrierScore:
    def test_perfect_forecaster_scores_zero(self):
        assert brier_score([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]) == pytest.approx(0.0)

    def test_uninformative_forecaster_on_balanced_sample(self):
        assert brier_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.25)

    def test_confidently_wrong_scores_worse_than_uninformative(self):
        confidently_wrong = brier_score([1, 0, 1, 0], [0.0, 1.0, 0.0, 1.0])
        uninformative = brier_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        assert confidently_wrong > uninformative


class TestReliabilityDiagramBins:
    def test_well_calibrated_predictions_track_observed_rate(self):
        rng = np.random.RandomState(0)
        n = 2000
        y_prob = rng.uniform(0.0, 1.0, n)
        y_true = (rng.uniform(0.0, 1.0, n) < y_prob).astype(float)
        bins = reliability_diagram_bins(y_true, y_prob, bins=(0.0, 0.5, 1.0))
        for band in bins:
            assert band["n"] > 0
            assert abs(band["mean_predicted"] - band["observed_rate"]) < 0.1

    def test_empty_bucket_reports_none_not_omitted(self):
        y_true = [1, 1]
        y_prob = [0.95, 0.99]
        bins = reliability_diagram_bins(y_true, y_prob, bins=(0.0, 0.5, 1.0))
        low_band = bins[0]
        assert low_band["n"] == 0
        assert low_band["mean_predicted"] is None
        assert low_band["observed_rate"] is None

    def test_boundary_values_included_in_exactly_one_bin(self):
        y_true = [1, 1]
        y_prob = [0.5, 1.0]
        bins = reliability_diagram_bins(y_true, y_prob, bins=(0.0, 0.5, 1.0))
        assert sum(b["n"] for b in bins) == 2
