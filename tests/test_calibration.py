"""Tests for firm.patterns.ml.calibration -- temperature scaling (CNN) and
Platt/sigmoid scaling (XGBoost) fit against synthetic data with a known,
deliberately-introduced miscalibration, verifying each fit routine recovers
parameters that measurably improve calibration (log-loss) over the
uncalibrated baseline. Pure array math, no I/O, no model artifacts needed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import expit

from firm.patterns.ml import calibration as cal

LABEL_ORDER = (-1, 0, 1)


def _overconfident_logits_and_labels(n: int = 3000, acc: float = 0.7, seed: int = 0):
    """A synthetic 3-class 'model' that is only ``acc``-accurate but assigns
    near-certainty to whichever class it predicts (right or wrong) --
    genuinely overconfident, the exact failure mode temperature scaling
    (T > 1) is meant to fix.
    """
    rng = np.random.default_rng(seed)
    true_idx = rng.integers(0, 3, size=n)
    labels = np.array([LABEL_ORDER[i] for i in true_idx])
    correct_mask = rng.random(n) < acc
    offset = rng.integers(1, 3, size=n)
    pred_idx = np.where(correct_mask, true_idx, (true_idx + offset) % 3)
    logits = np.full((n, 3), -5.0)
    logits[np.arange(n), pred_idx] = 10.0
    return logits, labels, true_idx


def _multiclass_nll(logits: np.ndarray, true_idx: np.ndarray, temperature: float) -> float:
    probs = cal.apply_temperature(logits, temperature)
    p_true = probs[np.arange(len(true_idx)), true_idx]
    return float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))


class TestApplyTemperature:
    def test_rows_sum_to_one(self):
        logits = np.array([[1.0, 2.0, -1.0], [0.5, 0.5, 0.5]])
        probs = cal.apply_temperature(logits, 1.0)
        np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0])

    def test_temperature_one_matches_plain_softmax(self):
        logits = np.array([2.0, -1.0, 0.5])
        probs = cal.apply_temperature(logits, 1.0)
        shifted = logits - logits.max()
        expected = np.exp(shifted) / np.exp(shifted).sum()
        np.testing.assert_allclose(probs, expected)

    def test_high_temperature_flattens_distribution(self):
        logits = np.array([5.0, 0.0, -5.0])
        cool = cal.apply_temperature(logits, 1.0)
        warm = cal.apply_temperature(logits, 50.0)
        # Softening toward uniform: the spread across classes shrinks.
        assert warm.std() < cool.std()
        np.testing.assert_allclose(warm, np.full(3, 1.0 / 3.0), atol=0.05)

    def test_low_temperature_sharpens_distribution(self):
        logits = np.array([1.0, 0.5, 0.0])
        base = cal.apply_temperature(logits, 1.0)
        sharp = cal.apply_temperature(logits, 0.1)
        assert sharp.max() > base.max()

    def test_numerically_stable_for_large_logits(self):
        logits = np.array([1000.0, -1000.0, 0.0])
        probs = cal.apply_temperature(logits, 1.0)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(probs.sum(), 1.0)

    def test_rejects_non_positive_temperature(self):
        with pytest.raises(ValueError):
            cal.apply_temperature(np.array([1.0, 2.0, 3.0]), 0.0)
        with pytest.raises(ValueError):
            cal.apply_temperature(np.array([1.0, 2.0, 3.0]), -1.0)


class TestFitTemperature:
    def test_recovers_softening_temperature_for_overconfident_model(self):
        logits, labels, _true_idx = _overconfident_logits_and_labels()
        T = cal.fit_temperature(logits, labels, label_order=LABEL_ORDER)
        assert T > 1.0

    def test_fitted_temperature_improves_log_loss_vs_uncalibrated(self):
        logits, labels, true_idx = _overconfident_logits_and_labels()
        T = cal.fit_temperature(logits, labels, label_order=LABEL_ORDER)
        nll_uncalibrated = _multiclass_nll(logits, true_idx, 1.0)
        nll_calibrated = _multiclass_nll(logits, true_idx, T)
        assert nll_calibrated < nll_uncalibrated
        # This synthetic model is dramatically overconfident (near-certain
        # at ~70% actual accuracy) -- calibration should help substantially,
        # not just marginally.
        assert nll_calibrated < 0.5 * nll_uncalibrated

    def test_well_calibrated_logits_fit_temperature_near_one(self):
        # Labels sampled directly from softmax(logits) -- by construction
        # these logits *are* the true generating process, so no temperature
        # correction should be needed (T close to 1, up to sampling noise).
        rng = np.random.default_rng(1)
        n = 8000
        logits = rng.normal(scale=1.5, size=(n, 3))
        true_probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        true_idx = np.array([rng.choice(3, p=p) for p in true_probs])
        labels = np.array([LABEL_ORDER[i] for i in true_idx])
        T = cal.fit_temperature(logits, labels, label_order=LABEL_ORDER)
        assert 0.7 < T < 1.4

    def test_rejects_shape_mismatch(self):
        logits = np.zeros((10, 2))  # only 2 columns, label_order has 3
        labels = np.array(LABEL_ORDER[0:1] * 10)
        with pytest.raises(ValueError):
            cal.fit_temperature(logits, labels, label_order=LABEL_ORDER)

    def test_rejects_label_length_mismatch(self):
        logits = np.zeros((10, 3))
        labels = np.array([-1, 0, 1])  # wrong length
        with pytest.raises(ValueError):
            cal.fit_temperature(logits, labels, label_order=LABEL_ORDER)

    def test_rejects_unknown_label_value(self):
        logits = np.zeros((3, 3))
        labels = np.array([-1, 0, 7])  # 7 not in label_order
        with pytest.raises(ValueError):
            cal.fit_temperature(logits, labels, label_order=LABEL_ORDER)


def _distorted_sigmoid_dataset(n: int = 4000, seed: int = 1):
    """``raw_scores`` are a scale+shift-distorted version of a latent
    variable that genuinely drives the binary outcome -- i.e. positively
    correlated with the true probability, but not itself a calibrated
    probability (or even a calibrated logit).
    """
    rng = np.random.default_rng(seed)
    true_x = rng.normal(size=n)
    true_logit = 2.0 * true_x + 0.5
    true_p = expit(true_logit)
    labels_binary = rng.binomial(1, true_p)
    raw_scores = 0.3 * true_x - 1.0  # compressed scale + shifted
    return raw_scores, labels_binary, true_p


def _binary_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1.0 - p_clipped)))


class TestFitSigmoidCalibration:
    def test_recovers_positive_correlation_sign(self):
        raw_scores, labels_binary, _ = _distorted_sigmoid_dataset()
        a, b = cal.fit_sigmoid_calibration(raw_scores, labels_binary)
        # raw_score increases with true_p by construction (see
        # _distorted_sigmoid_dataset) -- calibrated = expit(-(a*x+b)) is
        # increasing in x iff a < 0.
        assert a < 0

    def test_calibration_improves_log_loss_vs_naive_uncalibrated(self):
        raw_scores, labels_binary, _ = _distorted_sigmoid_dataset()
        a, b = cal.fit_sigmoid_calibration(raw_scores, labels_binary)
        calibrated = cal.apply_sigmoid_calibration(raw_scores, a, b)

        naive = expit(raw_scores)  # treating the raw (uncalibrated) score as if it were a logit
        loss_calibrated = _binary_log_loss(calibrated, labels_binary)
        loss_naive = _binary_log_loss(naive, labels_binary)
        assert loss_calibrated < loss_naive
        assert loss_calibrated < 0.7 * loss_naive

    def test_calibrated_probabilities_track_true_probabilities(self):
        raw_scores, labels_binary, true_p = _distorted_sigmoid_dataset()
        a, b = cal.fit_sigmoid_calibration(raw_scores, labels_binary)
        calibrated = cal.apply_sigmoid_calibration(raw_scores, a, b)
        # Reliability check: correlation with the (unobservable in practice,
        # known here since synthetic) true probability should be very high.
        corr = np.corrcoef(calibrated, true_p)[0, 1]
        assert corr > 0.95

    def test_binned_reliability_close_to_ideal(self):
        raw_scores, labels_binary, _ = _distorted_sigmoid_dataset(n=8000)
        a, b = cal.fit_sigmoid_calibration(raw_scores, labels_binary)
        calibrated = cal.apply_sigmoid_calibration(raw_scores, a, b)

        bins = np.linspace(0.0, 1.0, 6)
        bin_idx = np.digitize(calibrated, bins) - 1
        max_gap = 0.0
        for b_idx in range(len(bins) - 1):
            mask = bin_idx == b_idx
            if mask.sum() < 30:
                continue
            predicted_mean = calibrated[mask].mean()
            actual_freq = labels_binary[mask].mean()
            max_gap = max(max_gap, abs(predicted_mean - actual_freq))
        assert max_gap < 0.1

    def test_apply_sigmoid_calibration_matches_formula(self):
        raw_scores = np.array([-1.0, 0.0, 1.0, 2.0])
        a, b = 1.5, -0.5
        result = cal.apply_sigmoid_calibration(raw_scores, a, b)
        expected = 1.0 / (1.0 + np.exp(a * raw_scores + b))
        np.testing.assert_allclose(result, expected)

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            cal.fit_sigmoid_calibration(np.array([1.0, 2.0, 3.0]), np.array([0, 1]))

    def test_rejects_single_class_labels(self):
        with pytest.raises(ValueError):
            cal.fit_sigmoid_calibration(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))

    def test_rejects_non_binary_labels(self):
        with pytest.raises(ValueError):
            cal.fit_sigmoid_calibration(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2]))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            cal.fit_sigmoid_calibration(np.array([]), np.array([]))


class TestSaveLoadCalibrationRoundtrip:
    def test_temperature_roundtrip(self, tmp_path):
        path = tmp_path / "pattern_cnn.calibration.json"
        params = {"type": "temperature", "temperature": 1.34}
        cal.save_calibration(params, path)
        loaded = cal.load_calibration(path)
        assert loaded == params

    def test_sigmoid_roundtrip(self, tmp_path):
        path = tmp_path / "pattern_xgb.calibration.json"
        params = {"type": "sigmoid", "a": -3.21, "b": 0.87}
        cal.save_calibration(params, path)
        loaded = cal.load_calibration(path)
        assert loaded == params

    def test_save_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "calib.json"
        cal.save_calibration({"type": "temperature", "temperature": 1.0}, path)
        assert path.exists()

    def test_load_missing_file_returns_none(self, tmp_path):
        result = cal.load_calibration(tmp_path / "does_not_exist.json")
        assert result is None

    def test_load_corrupt_json_returns_none(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json")
        result = cal.load_calibration(path)
        assert result is None

    def test_load_missing_file_never_raises_and_logs_once(self, tmp_path, caplog):
        import logging

        path = tmp_path / "missing_for_warn_test.json"
        with caplog.at_level(logging.WARNING, logger="firm.patterns.ml.calibration"):
            assert cal.load_calibration(path) is None
            assert cal.load_calibration(path) is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        # First call warns, second call (same path) degrades to debug --
        # exactly one WARNING should have been emitted for this path.
        assert len(warnings) == 1


class TestJsonSerializationSanity:
    def test_saved_file_is_plain_json(self, tmp_path):
        path = tmp_path / "calib.json"
        params = {"type": "sigmoid", "a": 1.0, "b": -2.0}
        cal.save_calibration(params, path)
        with open(path) as f:
            raw = json.load(f)
        assert raw == params
