"""Tests for firm.patterns.ml.inference -- the main-venv, ONNX-only CNN
scoring path used by firm.strategies.pattern_recognition. Unlike
tests/test_cnn_validator.py / tests/test_pattern_ml_onnx.py (isolated
.venv-ml only, need torch/pyts/onnxmltools), this module deliberately has no
hard torch/pyts dependency, so these tests run under the main venv/CI same
as everything else -- the ONNX session is mocked, not loaded from a real
model file, so these pass with or without a trained artifact on disk.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from firm.patterns.ml import inference as inf


@pytest.fixture(autouse=True)
def _reset_inference_cache():
    """Every test gets a clean lazy-singleton/warn-once state -- otherwise
    whichever test runs first would permanently decide _load_session's
    cached result for the rest of the session (lru_cache is keyed on the
    literal path string, so distinct tmp paths avoid collisions between
    tests, but the real DEFAULT_MODEL_PATH cache entry must still be reset
    so tests don't leak into each other via that shared key).
    """
    inf.reset_cache()
    yield
    inf.reset_cache()


class TestEncodeGasf:
    def test_matches_pyts_gasf_reference_values(self):
        # Hand-computed reference for a tiny window, verified independently
        # (during development) against a real pyts.image.GramianAngularField
        # run under .venv-ml (max abs diff 0.0). Re-derived here from the
        # closed-form formula so this test has no isolated-env dependency.
        window = np.array([0.0, 0.5, 1.0, 0.25])
        x = 2.0 * window - 1.0  # [-1, 0, 1, -0.5]
        sin_part = np.sqrt(np.clip(1.0 - x**2, 0.0, 1.0))
        expected = np.outer(x, x) - np.outer(sin_part, sin_part)
        result = inf._encode_gasf(window)
        np.testing.assert_allclose(result, expected)

    def test_output_is_square_and_in_range(self):
        rng = np.random.RandomState(0)
        window = rng.rand(32)
        out = inf._encode_gasf(window)
        assert out.shape == (32, 32)
        assert np.all(out >= -1.0 - 1e-9) and np.all(out <= 1.0 + 1e-9)


class TestScorePatternQualityFallbacks:
    def test_returns_none_when_onnxruntime_not_installed(self, monkeypatch):
        real_import = __import__

        def _fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("no onnxruntime here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        result = inf.score_pattern_quality(close, 80, model_path="/tmp/whatever.onnx")
        assert result is None

    def test_returns_none_when_model_file_missing(self, tmp_path):
        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        missing_path = tmp_path / "does_not_exist.onnx"
        result = inf.score_pattern_quality(close, 80, model_path=str(missing_path))
        assert result is None

    def test_returns_none_when_model_file_corrupt(self, tmp_path):
        bad_path = tmp_path / "corrupt.onnx"
        bad_path.write_bytes(b"not a real onnx file")
        sidecar = tmp_path / "corrupt.onnx.labels.json"
        sidecar.write_text("[-1, 0, 1]")

        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        result = inf.score_pattern_quality(close, 80, model_path=str(bad_path))
        assert result is None

    def test_returns_none_when_sidecar_missing(self, tmp_path):
        # A structurally-valid-looking path but no .labels.json sidecar --
        # exercises the same "corrupt/incomplete artifact" degrade path.
        bad_path = tmp_path / "no_sidecar.onnx"
        bad_path.write_bytes(b"\x00\x01")
        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        result = inf.score_pattern_quality(close, 80, model_path=str(bad_path))
        assert result is None

    def test_returns_none_on_insufficient_history(self, tmp_path):
        # Even with a model file present, too little history before
        # confirm_index for a full window must degrade gracefully, not raise.
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"\x00")
        sidecar = tmp_path / "model.onnx.labels.json"
        sidecar.write_text("[-1, 0, 1]")

        close = np.array([100.0, 101.0, 99.0])  # far fewer than window_bars=32
        result = inf.score_pattern_quality(close, 2, model_path=str(model_path))
        assert result is None

    def test_returns_none_when_window_bars_ne_image_size(self):
        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        result = inf.score_pattern_quality(
            close, 80, model_path="/tmp/whatever.onnx", window_bars=16, image_size=32,
        )
        assert result is None

    def test_never_raises_on_unexpected_session_error(self, tmp_path, monkeypatch):
        """A model file that loads fine but blows up at .run() time (e.g. a
        shape mismatch) must still degrade to None, never propagate.
        """
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"\x00")
        sidecar = tmp_path / "model.onnx.labels.json"
        sidecar.write_text("[-1, 0, 1]")

        fake_session = MagicMock()
        fake_session.run.side_effect = RuntimeError("boom")
        monkeypatch.setattr(inf, "_load_session", lambda path: (fake_session, (-1, 0, 1)))

        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        result = inf.score_pattern_quality(close, 80, model_path=str(model_path))
        assert result is None


class TestScorePatternQualityWithMockedSession:
    def _make_session(self, logits: np.ndarray, input_name: str = "input"):
        session = MagicMock()
        input_stub = MagicMock()
        input_stub.name = input_name
        session.get_inputs.return_value = [input_stub]
        session.run.return_value = [logits[None, :]]
        return session

    def test_high_target_probability_yields_high_quality_score(self, monkeypatch):
        # logits strongly favoring the +1 (target) class over -1 (stop)
        logits = np.array([-5.0, 0.0, 5.0])  # order matches present_labels below
        session = self._make_session(logits)
        monkeypatch.setattr(inf, "_load_session", lambda path: (session, (-1, 0, 1)))

        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        result = inf.score_pattern_quality(close, 80, model_path="/tmp/whatever.onnx")

        assert result is not None
        assert result > 0.9  # near-certain target hit, near-zero stop -> near 1.0

    def test_high_stop_probability_yields_low_quality_score(self, monkeypatch):
        logits = np.array([5.0, 0.0, -5.0])  # -1 (stop) dominant
        session = self._make_session(logits)
        monkeypatch.setattr(inf, "_load_session", lambda path: (session, (-1, 0, 1)))

        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        result = inf.score_pattern_quality(close, 80, model_path="/tmp/whatever.onnx")

        assert result is not None
        assert result < 0.1

    def test_balanced_logits_yield_neutral_score(self, monkeypatch):
        logits = np.array([0.0, 10.0, 0.0])  # timeout class dominant, target==stop
        session = self._make_session(logits)
        monkeypatch.setattr(inf, "_load_session", lambda path: (session, (-1, 0, 1)))

        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        result = inf.score_pattern_quality(close, 80, model_path="/tmp/whatever.onnx")

        assert result == pytest.approx(0.5, abs=1e-6)

    def test_result_always_in_unit_interval(self, monkeypatch):
        logits = np.array([100.0, 0.0, -100.0])
        session = self._make_session(logits)
        monkeypatch.setattr(inf, "_load_session", lambda path: (session, (-1, 0, 1)))

        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        result = inf.score_pattern_quality(close, 80, model_path="/tmp/whatever.onnx")

        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_present_labels_subset_handled_correctly(self, monkeypatch):
        # A model whose training slice only ever saw 2 of the 3 classes --
        # present_labels omits one; predict_proba-style remapping must still
        # produce a sane 3-way split, not crash on the missing column.
        logits = np.array([3.0, -3.0])  # only (-1, 1) present, no 0 class
        session = self._make_session(logits)
        monkeypatch.setattr(inf, "_load_session", lambda path: (session, (-1, 1)))

        close = np.cumsum(np.random.RandomState(0).randn(100)) + 100
        result = inf.score_pattern_quality(close, 80, model_path="/tmp/whatever.onnx")

        assert result is not None
        assert result < 0.1  # -1 (stop) logit dominant


class TestIsAvailable:
    def test_false_when_model_missing(self, tmp_path):
        assert inf.is_available(tmp_path / "nope.onnx") is False

    def test_true_when_real_model_present_and_onnxruntime_installed(self):
        pytest.importorskip("onnxruntime")
        if not inf.DEFAULT_MODEL_PATH.exists():
            pytest.skip("no trained data/models/pattern_cnn.onnx artifact on disk")
        assert inf.is_available(inf.DEFAULT_MODEL_PATH) is True


class TestRealArtifactIfPresent:
    """Exercises the real on-disk artifact end-to-end when present -- skips
    cleanly (rather than failing) when it isn't, so this suite is green
    whether or not a trained model has been committed to this checkout.
    """

    def test_score_pattern_quality_against_real_model(self):
        pytest.importorskip("onnxruntime")
        if not inf.DEFAULT_MODEL_PATH.exists():
            pytest.skip("no trained data/models/pattern_cnn.onnx artifact on disk")

        close = np.cumsum(np.random.RandomState(42).randn(100)) + 100
        result = inf.score_pattern_quality(close, 80)
        assert result is not None
        assert 0.0 <= result <= 1.0
