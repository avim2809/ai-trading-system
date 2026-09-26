"""Tests for firm.patterns.ml.cnn_validator's pure-numpy GASF encoder path
(_encode_gasf_numpy / encode_gaf's image_size == len(series) dispatch).

Deliberately NOT gated behind tests/test_cnn_validator.py's torch/pyts
skip-marker: the whole point of this path is that it needs neither
dependency, so this file must actually run on a host (like this one) where
pyts has no wheel and .venv-ml doesn't exist -- that's the only way to
catch encode_gaf silently falling through to _require_pyts() and raising
ImportError instead of using the numpy path.
"""

from __future__ import annotations

import numpy as np
import pytest

from firm.patterns.ml.cnn_validator import _encode_gasf_numpy, encode_gaf


class TestEncodeGasfNumpy:
    def test_output_shape(self):
        series = np.linspace(0.0, 1.0, 32)
        image = _encode_gasf_numpy(series)
        assert image.shape == (32, 32)

    def test_output_bounded(self):
        rng = np.random.RandomState(0)
        series = rng.uniform(0.0, 1.0, 32)
        image = _encode_gasf_numpy(series)
        assert image.min() >= -1.0 - 1e-9
        assert image.max() <= 1.0 + 1e-9

    def test_constant_series_is_uniform(self):
        # x = 2*0.5-1 = 0 for every element -> phi = pi/2 everywhere ->
        # cos(phi_i+phi_j) = cos(pi) = -1 everywhere.
        series = np.full(10, 0.5)
        image = _encode_gasf_numpy(series)
        np.testing.assert_allclose(image, -1.0, atol=1e-9)

    def test_diagonal_equals_cos_of_double_angle(self):
        # GASF's diagonal is cos(2*phi_i) = 2*x_i^2 - 1 (x=cos(phi)).
        series = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        image = _encode_gasf_numpy(series)
        x = np.clip(2.0 * series - 1.0, -1.0, 1.0)
        expected_diag = 2.0 * x**2 - 1.0
        np.testing.assert_allclose(np.diag(image), expected_diag, atol=1e-9)


class TestEncodeGafDispatch:
    def test_matching_size_uses_numpy_path_no_pyts_required(self):
        # This is the regression this file exists to catch: without pyts
        # installed (true on this host), encode_gaf must NOT raise
        # ImportError when image_size == len(series).
        series = np.linspace(0.0, 1.0, 32)
        image = encode_gaf(series, image_size=32)
        assert image.shape == (32, 32)
        np.testing.assert_allclose(image, _encode_gasf_numpy(series))

    def test_mismatched_size_requires_pyts(self):
        series = np.linspace(0.0, 1.0, 40)
        with pytest.raises(ImportError, match="pyts"):
            encode_gaf(series, image_size=16)
