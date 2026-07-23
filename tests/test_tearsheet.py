"""Tests for firm.eval.tearsheet (skipped when quantstats is not installed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

quantstats = pytest.importorskip("quantstats")

from firm.eval.tearsheet import render_tearsheet  # noqa: E402


def _returns(n: int = 400) -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, size=n), index=idx)


class TestTearsheet:
    def test_renders_nonempty_html(self, tmp_path):
        out = render_tearsheet(_returns(), out_html=tmp_path / "ts.html")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_returns_raises(self, tmp_path):
        with pytest.raises(ValueError):
            render_tearsheet(pd.Series([], dtype=float), out_html=tmp_path / "x.html")
