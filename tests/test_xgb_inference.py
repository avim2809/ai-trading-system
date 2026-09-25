"""Tests for firm.patterns.ml.xgb_inference -- the main-venv, ONNX-only
XGBoost pattern-confirmation scoring path. Mirrors
tests/test_pattern_ml_inference.py's style (the CNN's equivalent module):
fail-soft fallback paths are exercised with a mocked/monkeypatched session
so these tests run under the main venv/CI regardless of whether a trained
artifact is present on disk, plus a "real artifact if present" suite that
exercises data/models/pattern_xgb.onnx end-to-end when it exists.

Unlike tests/test_pattern_ml_onnx.py (needs xgboost + onnxmltools to fit and
export a *new* toy model -- neither has a Python 3.14 wheel, isolated
.venv-ml only, see that file's docstring), this module only ever needs
onnxruntime to *load* an already-exported .onnx file, which does have a
cp314 wheel and is confirmed importable in the main venv (see
xgb_inference.py's module docstring) -- so most of this file's tests run
here directly. The one true "train + export + round-trip" test needs the
xgboost/onnxmltools training/export stack and is skipped (not failed) when
that isn't installed, exactly mirroring test_pattern_ml_onnx.py's
skip-gating convention.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from firm.patterns.ml import calibration as cal
from firm.patterns.ml import xgb_inference as xinf

try:
    import onnxmltools  # noqa: F401
    import onnxruntime  # noqa: F401
    import xgboost  # noqa: F401

    _ONNX_TRAIN_STACK_AVAILABLE = True
except ImportError:
    _ONNX_TRAIN_STACK_AVAILABLE = False


N_FEATURES = 47  # matches the real on-disk data/models/pattern_xgb.onnx artifact


@pytest.fixture(autouse=True)
def _reset_inference_cache():
    """Every test gets a clean lazy-singleton/warn-once state -- otherwise
    whichever test runs first would permanently decide _load_session's
    cached result for the rest of the session (see
    tests/test_pattern_ml_inference.py's identical fixture for the CNN
    module -- same rationale applies verbatim here).
    """
    xinf.reset_cache()
    yield
    xinf.reset_cache()


class TestScorePatternConfirmationFallbacks:
    def test_returns_none_when_onnxruntime_not_installed(self, monkeypatch):
        real_import = __import__

        def _fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("no onnxruntime here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(features, model_path="/tmp/whatever.onnx")
        assert result is None

    def test_returns_none_when_model_file_missing(self, tmp_path):
        features = np.zeros(N_FEATURES)
        missing_path = tmp_path / "does_not_exist.onnx"
        result = xinf.score_pattern_confirmation(features, model_path=str(missing_path))
        assert result is None

    def test_returns_none_when_model_file_corrupt(self, tmp_path):
        bad_path = tmp_path / "corrupt.onnx"
        bad_path.write_bytes(b"not a real onnx file")
        sidecar = tmp_path / "corrupt.onnx.labels.json"
        sidecar.write_text("[-1, 0, 1]")

        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(features, model_path=str(bad_path))
        assert result is None

    def test_returns_none_when_sidecar_missing(self, tmp_path):
        bad_path = tmp_path / "no_sidecar.onnx"
        bad_path.write_bytes(b"\x00\x01")
        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(features, model_path=str(bad_path))
        assert result is None

    def test_returns_none_when_sidecar_corrupt(self, tmp_path):
        bad_path = tmp_path / "model.onnx"
        bad_path.write_bytes(b"\x00\x01")
        sidecar = tmp_path / "model.onnx.labels.json"
        sidecar.write_text("{not valid json")
        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(features, model_path=str(bad_path))
        assert result is None

    def test_never_raises_on_unexpected_session_error(self, monkeypatch):
        fake_session = MagicMock()
        fake_session.run.side_effect = RuntimeError("boom")
        monkeypatch.setattr(xinf, "_load_session", lambda path: (fake_session, (-1, 0, 1)))

        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(features, model_path="/tmp/whatever.onnx")
        assert result is None

    def test_never_raises_on_malformed_feature_shape(self, monkeypatch):
        session = MagicMock()
        input_stub = MagicMock()
        input_stub.name = "input"
        session.get_inputs.return_value = [input_stub]
        monkeypatch.setattr(xinf, "_load_session", lambda path: (session, (-1, 0, 1)))

        # A batch of 2 rows is not "one pattern match" -- must degrade to
        # None, not raise, even though the underlying array is well-formed.
        bad_features = np.zeros((2, N_FEATURES))
        result = xinf.score_pattern_confirmation(bad_features, model_path="/tmp/whatever.onnx")
        assert result is None


class TestScorePatternConfirmationWithMockedSession:
    def _make_session(self, proba_row: np.ndarray, input_name: str = "input"):
        session = MagicMock()
        input_stub = MagicMock()
        input_stub.name = input_name
        session.get_inputs.return_value = [input_stub]
        # sklearn-onnx classifier graphs emit [labels, probabilities].
        labels_out = np.array([int(np.argmax(proba_row))])
        session.run.return_value = [labels_out, proba_row[None, :]]
        return session

    def test_full_label_set_mapped_correctly(self, monkeypatch):
        # present_labels == LABELS exactly -- straight passthrough.
        proba = np.array([0.2, 0.3, 0.5])  # (-1, 0, 1) order
        session = self._make_session(proba)
        monkeypatch.setattr(xinf, "_load_session", lambda path: (session, (-1, 0, 1)))

        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(features, model_path="/tmp/whatever.onnx")

        assert result is not None
        p_stop, p_timeout, p_target = result
        assert p_stop == pytest.approx(0.2)
        assert p_timeout == pytest.approx(0.3)
        assert p_target == pytest.approx(0.5)

    def test_present_labels_subset_handled_correctly(self, monkeypatch):
        # Model only ever saw 2 of 3 classes at training time -- the missing
        # column (here, "0"/timeout) must come back as 0.0, not crash.
        proba = np.array([0.35, 0.65])  # only (-1, 1) present, no 0 class
        session = self._make_session(proba)
        monkeypatch.setattr(xinf, "_load_session", lambda path: (session, (-1, 1)))

        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(features, model_path="/tmp/whatever.onnx")

        assert result is not None
        p_stop, p_timeout, p_target = result
        assert p_stop == pytest.approx(0.35)
        assert p_timeout == 0.0
        assert p_target == pytest.approx(0.65)

    def test_accepts_2d_single_row_input(self, monkeypatch):
        proba = np.array([0.1, 0.1, 0.8])
        session = self._make_session(proba)
        monkeypatch.setattr(xinf, "_load_session", lambda path: (session, (-1, 0, 1)))

        features = np.zeros((1, N_FEATURES))
        result = xinf.score_pattern_confirmation(features, model_path="/tmp/whatever.onnx")
        assert result is not None
        assert result[2] == pytest.approx(0.8)

    def test_probabilities_sum_to_one_and_in_unit_interval(self, monkeypatch):
        proba = np.array([0.15, 0.55, 0.30])
        session = self._make_session(proba)
        monkeypatch.setattr(xinf, "_load_session", lambda path: (session, (-1, 0, 1)))

        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(features, model_path="/tmp/whatever.onnx")
        assert result is not None
        assert all(0.0 <= v <= 1.0 for v in result)
        assert sum(result) == pytest.approx(1.0)


class TestApplyCalibrationParameter:
    def _make_session(self, proba_row: np.ndarray):
        session = MagicMock()
        input_stub = MagicMock()
        input_stub.name = "input"
        session.get_inputs.return_value = [input_stub]
        labels_out = np.array([int(np.argmax(proba_row))])
        session.run.return_value = [labels_out, proba_row[None, :]]
        return session

    def test_no_calibration_returns_raw_target_probability(self, monkeypatch):
        proba = np.array([0.2, 0.3, 0.5])
        session = self._make_session(proba)
        monkeypatch.setattr(xinf, "_load_session", lambda path: (session, (-1, 0, 1)))

        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(
            features, model_path="/tmp/whatever.onnx", apply_calibration=None,
        )
        assert result[2] == pytest.approx(0.5)

    def test_sigmoid_calibration_applied_to_target_probability_only(self, monkeypatch):
        proba = np.array([0.2, 0.3, 0.5])
        session = self._make_session(proba)
        monkeypatch.setattr(xinf, "_load_session", lambda path: (session, (-1, 0, 1)))

        calibration_params = {"type": "sigmoid", "a": -2.0, "b": 0.5}
        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(
            features, model_path="/tmp/whatever.onnx", apply_calibration=calibration_params,
        )
        assert result is not None
        p_stop, p_timeout, p_target = result

        expected_target = cal.apply_sigmoid_calibration(
            np.array([0.5]), calibration_params["a"], calibration_params["b"],
        )[0]
        assert p_target == pytest.approx(float(expected_target))
        # p_stop/p_timeout are untouched by calibration (see docstring).
        assert p_stop == pytest.approx(0.2)
        assert p_timeout == pytest.approx(0.3)
        # Sanity: calibration must have actually changed the value in this
        # case (not silently a no-op).
        assert p_target != pytest.approx(0.5)

    def test_unrecognized_calibration_type_skipped_gracefully(self, monkeypatch):
        proba = np.array([0.2, 0.3, 0.5])
        session = self._make_session(proba)
        monkeypatch.setattr(xinf, "_load_session", lambda path: (session, (-1, 0, 1)))

        features = np.zeros(N_FEATURES)
        result = xinf.score_pattern_confirmation(
            features, model_path="/tmp/whatever.onnx",
            apply_calibration={"type": "temperature", "temperature": 2.0},
        )
        assert result is not None
        assert result[2] == pytest.approx(0.5)  # unchanged -- not applicable to this output

    def test_malformed_calibration_dict_degrades_to_raw_value(self, monkeypatch):
        proba = np.array([0.2, 0.3, 0.5])
        session = self._make_session(proba)
        monkeypatch.setattr(xinf, "_load_session", lambda path: (session, (-1, 0, 1)))

        features = np.zeros(N_FEATURES)
        # Missing "a"/"b" keys -- must not raise, just skip calibration.
        result = xinf.score_pattern_confirmation(
            features, model_path="/tmp/whatever.onnx", apply_calibration={"type": "sigmoid"},
        )
        assert result is not None
        assert result[2] == pytest.approx(0.5)


class TestIsAvailable:
    def test_false_when_model_missing(self, tmp_path):
        assert xinf.is_available(tmp_path / "nope.onnx") is False

    def test_true_when_real_model_present_and_onnxruntime_installed(self):
        pytest.importorskip("onnxruntime")
        if not xinf.DEFAULT_MODEL_PATH.exists():
            pytest.skip("no trained data/models/pattern_xgb.onnx artifact on disk")
        assert xinf.is_available(xinf.DEFAULT_MODEL_PATH) is True


class TestRealArtifactIfPresent:
    """Exercises the real on-disk artifact end-to-end when present -- skips
    cleanly (rather than failing) when it isn't, so this suite is green
    whether or not a trained model has been committed to this checkout.
    """

    def test_score_pattern_confirmation_against_real_model(self):
        pytest.importorskip("onnxruntime")
        if not xinf.DEFAULT_MODEL_PATH.exists():
            pytest.skip("no trained data/models/pattern_xgb.onnx artifact on disk")

        rng = np.random.RandomState(42)
        features = rng.rand(N_FEATURES).astype(np.float32)
        result = xinf.score_pattern_confirmation(features)

        assert result is not None
        p_stop, p_timeout, p_target = result
        for p in (p_stop, p_timeout, p_target):
            assert 0.0 <= p <= 1.0
        assert (p_stop + p_timeout + p_target) == pytest.approx(1.0, abs=1e-4)

    def test_real_model_with_calibration_applied(self):
        pytest.importorskip("onnxruntime")
        if not xinf.DEFAULT_MODEL_PATH.exists():
            pytest.skip("no trained data/models/pattern_xgb.onnx artifact on disk")

        rng = np.random.RandomState(7)
        features = rng.rand(N_FEATURES).astype(np.float32)
        raw = xinf.score_pattern_confirmation(features)
        assert raw is not None

        calibration_params = {"type": "sigmoid", "a": -1.0, "b": 0.0}
        calibrated = xinf.score_pattern_confirmation(features, apply_calibration=calibration_params)
        assert calibrated is not None
        # Calibration only ever touches p_target.
        assert calibrated[0] == pytest.approx(raw[0])
        assert calibrated[1] == pytest.approx(raw[1])


@pytest.mark.skipif(
    not _ONNX_TRAIN_STACK_AVAILABLE,
    reason="xgboost/onnxmltools/onnxruntime not installed -- run via .venv-ml/bin/pytest",
)
class TestOnnxExportRoundTrip:
    """A genuine train -> export -> load -> infer round trip through this
    module's own score_pattern_confirmation, using a freshly-fit toy
    XGBoost model (not the production artifact) -- needs the full
    xgboost + onnxmltools training/export stack, which (per
    tests/test_pattern_ml_onnx.py's identical gate) has no Python 3.14
    wheel and is therefore only ever available under .venv-ml.
    """

    def test_freshly_trained_model_round_trips_through_this_module(self, tmp_path):
        import pandas as pd

        from firm.patterns.ml import xgb_classifier

        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.rand(80, N_FEATURES), columns=[f"f{i}" for i in range(N_FEATURES)])
        y = rng.randint(-1, 2, size=80)
        model = xgb_classifier.train(X, y)

        onnx_path = tmp_path / "toy_pattern_xgb.onnx"
        xgb_classifier.export_onnx(model, onnx_path, n_features=N_FEATURES)

        row = X.iloc[0].to_numpy(dtype=np.float32)
        result = xinf.score_pattern_confirmation(row, model_path=str(onnx_path))

        assert result is not None
        p_stop, p_timeout, p_target = result
        for p in (p_stop, p_timeout, p_target):
            assert 0.0 <= p <= 1.0
        assert (p_stop + p_timeout + p_target) == pytest.approx(1.0, abs=1e-4)

        # Cross-check against xgb_classifier's own native predict_proba.
        native_proba = xgb_classifier.predict_proba(model, X.iloc[[0]])[0]
        np.testing.assert_allclose([p_stop, p_timeout, p_target], native_proba, atol=1e-3)
