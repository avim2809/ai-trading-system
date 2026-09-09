"""Tests for the CNN/GAF pattern-image validator (docs/
pattern_recognition_plan.md §4b): firm.patterns.ml.cnn_validator.

**Run these via the isolated ML environment, not the main venv**::

    .venv-ml/bin/pytest tests/test_cnn_validator.py -q

torch/pyts have no Python 3.14 wheels yet (see
docs/pattern_recognition_plan.md §4a for how .venv-ml was built) -- this
file skips everything under the main venv rather than failing collection,
mirroring tests/test_pattern_ml_onnx.py's gating.
"""

from __future__ import annotations

import numpy as np
import pytest

from firm.patterns.ml import cnn_validator

try:
    import onnxruntime  # noqa: F401
    import pyts  # noqa: F401
    import torch  # noqa: F401

    _TORCH_STACK_AVAILABLE = True
except ImportError:
    _TORCH_STACK_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TORCH_STACK_AVAILABLE,
    reason="torch/pyts/onnxruntime not installed -- run via .venv-ml/bin/pytest",
)


class TestEncodeGaf:
    def test_output_shape_and_range(self):
        series = np.linspace(0.0, 1.0, 40)
        image = cnn_validator.encode_gaf(series, image_size=16)
        assert image.shape == (16, 16)
        # GAF values are cosine-of-angle-sum differences -> bounded [-1, 1].
        assert image.min() >= -1.0 - 1e-6
        assert image.max() <= 1.0 + 1e-6


class TestExtractWindow:
    def test_returns_none_when_not_enough_history(self):
        close = np.arange(10, dtype=float)
        assert cnn_validator.extract_window(close, confirm_index=5, window_bars=32) is None

    def test_returns_none_for_a_flat_window(self):
        close = np.full(50, 100.0)
        assert cnn_validator.extract_window(close, confirm_index=40, window_bars=32) is None

    def test_normalizes_to_unit_range(self):
        close = np.linspace(50.0, 150.0, 50)
        window = cnn_validator.extract_window(close, confirm_index=40, window_bars=32)
        assert window is not None
        assert window.shape == (32,)
        assert np.isclose(window.min(), 0.0)
        assert np.isclose(window.max(), 1.0)


def _toy_images_and_labels(n: int = 60, image_size: int = 16, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, image_size, image_size).astype(np.float32)
    y = rng.randint(-1, 2, size=n)
    return X, y


class TestTrainAndPredict:
    def test_train_predict_proba_bounds_and_shape(self):
        X, y = _toy_images_and_labels()
        model = cnn_validator.train(X, y, image_size=16, epochs=2)
        proba = cnn_validator.predict_proba(model, X)
        assert proba.shape == (len(X), 3)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    def test_predict_label_matches_argmax_of_proba(self):
        X, y = _toy_images_and_labels(seed=1)
        model = cnn_validator.train(X, y, image_size=16, epochs=2)
        proba = cnn_validator.predict_proba(model, X)
        expected = np.array([cnn_validator.LABELS[i] for i in np.argmax(proba, axis=1)])
        np.testing.assert_array_equal(cnn_validator.predict_label(model, X), expected)

    def test_handles_class_missing_from_training_data(self):
        rng = np.random.RandomState(2)
        X = rng.rand(20, 16, 16).astype(np.float32)
        y = np.array([-1, 1] * 10)  # label 0 never seen
        model = cnn_validator.train(X, y, image_size=16, epochs=2)
        assert model.present_labels == (-1, 1)
        proba = cnn_validator.predict_proba(model, X)
        assert proba.shape == (20, 3)
        assert np.allclose(proba[:, 1], 0.0)

    def test_rejects_labels_outside_domain(self):
        X, _y = _toy_images_and_labels(n=5)
        with pytest.raises(ValueError):
            cnn_validator.train(X, np.array([-1, 0, 1, 2, -1]), image_size=16, epochs=1)

    def test_rejects_empty_labels(self):
        X, _y = _toy_images_and_labels(n=0)
        with pytest.raises(ValueError):
            cnn_validator.train(X, np.array([], dtype=int), image_size=16, epochs=1)


class TestSaveLoadAndOnnx:
    def test_save_and_load_roundtrip(self, tmp_path):
        X, y = _toy_images_and_labels(seed=3)
        model = cnn_validator.train(X, y, image_size=16, epochs=2)
        path = tmp_path / "model.pt"
        cnn_validator.save(model, path)
        loaded = cnn_validator.load(path)
        assert loaded.present_labels == model.present_labels
        assert loaded.image_size == model.image_size
        np.testing.assert_allclose(
            cnn_validator.predict_proba(loaded, X), cnn_validator.predict_proba(model, X), atol=1e-6,
        )

    def test_onnx_export_matches_native_predictions(self, tmp_path):
        import json

        import onnxruntime as ort

        X, y = _toy_images_and_labels(seed=4)
        model = cnn_validator.train(X, y, image_size=16, epochs=2)
        proba_native = cnn_validator.predict_proba(model, X)

        onnx_path = tmp_path / "model.onnx"
        cnn_validator.export_onnx(model, onnx_path)
        assert onnx_path.exists()
        sidecar = onnx_path.with_suffix(onnx_path.suffix + ".labels.json")
        assert sidecar.exists()
        with open(sidecar) as f:
            present_labels = tuple(json.load(f))
        assert present_labels == model.present_labels

        session = ort.InferenceSession(str(onnx_path))
        logits = session.run(None, {"input": X.reshape(-1, 1, 16, 16)})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba_compact = exp / exp.sum(axis=1, keepdims=True)
        proba_onnx = np.zeros((len(X), 3))
        for col, label in enumerate(present_labels):
            proba_onnx[:, cnn_validator.LABELS.index(label)] = proba_compact[:, col]

        np.testing.assert_allclose(proba_onnx, proba_native, atol=1e-4)
