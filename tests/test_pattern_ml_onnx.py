"""Tests for the ONNX export/inference path of the XGBoost pattern
classifier (docs/pattern_recognition_plan.md §3b/§4):
firm.patterns.ml.xgb_classifier's export_onnx/load_onnx/predict_proba_onnx.

**Run these via the isolated ML environment, not the main venv**::

    .venv-ml/bin/pytest tests/test_pattern_ml_onnx.py -q

onnxmltools/onnxruntime have no Python 3.14 wheels yet (verified against
PyPI when this environment was set up), so this file cannot even be
collected under the main .venv — see docs/pattern_recognition_plan.md §4a
for how .venv-ml was built (uv, CPU-only torch, pandas/xgboost added
alongside so firm.patterns.ml.* is importable there too).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from firm.patterns.ml import xgb_classifier

try:
    import onnxmltools  # noqa: F401
    import onnxruntime  # noqa: F401
    import xgboost  # noqa: F401

    _ONNX_STACK_AVAILABLE = True
except ImportError:
    _ONNX_STACK_AVAILABLE = False

# This whole file needs xgboost (to fit a model at all) *and*
# onnxmltools/onnxruntime (neither of which has a Python 3.14 wheel yet —
# see docs/pattern_recognition_plan.md §4a) -- skip everything rather than
# fail collection when run under the main .venv, exactly like
# tests/test_pattern_ml.py's requires_xgboost gate does for xgboost alone.
pytestmark = pytest.mark.skipif(
    not _ONNX_STACK_AVAILABLE,
    reason="onnxmltools/onnxruntime/xgboost not installed -- run via .venv-ml/bin/pytest",
)


def _toy_xy(n: int = 60, n_features: int = 6, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Same convention as tests/test_pattern_ml.py's _toy_xy — a named-column
    DataFrame is deliberate here (not a plain ndarray): the whole point of
    xgb_classifier.train()'s "fit on X.to_numpy(), not the DataFrame itself"
    behavior (see its docstring) is that ONNX export works *even when* the
    caller passes named columns, exactly like the real training script does.
    """
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.rand(n, n_features), columns=[f"score_{i}" for i in range(n_features)])
    y = rng.randint(-1, 2, size=n)
    return X, y


class TestOnnxExportRoundTrip:
    def test_onnx_predictions_match_pickled_model(self, tmp_path):
        X, y = _toy_xy()
        model = xgb_classifier.train(X, y)
        proba_native = xgb_classifier.predict_proba(model, X)

        onnx_path = tmp_path / "model.onnx"
        xgb_classifier.export_onnx(model, onnx_path, n_features=X.shape[1])
        assert onnx_path.exists()
        assert onnx_path.with_suffix(onnx_path.suffix + ".labels.json").exists()

        session, present_labels = xgb_classifier.load_onnx(onnx_path)
        assert present_labels == model.present_labels
        proba_onnx = xgb_classifier.predict_proba_onnx(session, present_labels, X)

        assert proba_onnx.shape == proba_native.shape
        np.testing.assert_allclose(proba_onnx, proba_native, atol=1e-4)

    def test_onnx_export_survives_a_class_missing_from_training_data(self, tmp_path):
        # Same scenario as test_pattern_ml.py's
        # test_predict_proba_handles_class_missing_from_training_data --
        # only 2 of 3 labels present at fit time. The exported ONNX graph
        # only ever knows about those 2 compact classes; predict_proba_onnx
        # must still decode back to the full 3-column (-1, 0, +1) layout,
        # exactly like the native predict_proba does.
        X = pd.DataFrame(np.random.RandomState(1).rand(20, 4), columns=list("abcd"))
        y = np.array([-1, 1] * 10)
        model = xgb_classifier.train(X, y)
        assert model.present_labels == (-1, 1)

        onnx_path = tmp_path / "model.onnx"
        xgb_classifier.export_onnx(model, onnx_path, n_features=4)
        session, present_labels = xgb_classifier.load_onnx(onnx_path)
        proba_onnx = xgb_classifier.predict_proba_onnx(session, present_labels, X)

        assert proba_onnx.shape == (20, 3)
        assert np.allclose(proba_onnx[:, 1], 0.0)  # label 0's column, untrained
        np.testing.assert_allclose(
            proba_onnx, xgb_classifier.predict_proba(model, X), atol=1e-4,
        )

    def test_onnx_predict_label_matches_native_argmax(self, tmp_path):
        X, y = _toy_xy(n=40, seed=2)
        model = xgb_classifier.train(X, y)
        onnx_path = tmp_path / "model.onnx"
        xgb_classifier.export_onnx(model, onnx_path, n_features=X.shape[1])
        session, present_labels = xgb_classifier.load_onnx(onnx_path)

        proba_onnx = xgb_classifier.predict_proba_onnx(session, present_labels, X)
        onnx_labels = np.array([xgb_classifier.LABELS[i] for i in np.argmax(proba_onnx, axis=1)])
        np.testing.assert_array_equal(onnx_labels, xgb_classifier.predict_label(model, X))
