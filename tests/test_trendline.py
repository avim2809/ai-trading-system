"""Tests for firm.patterns.trendline.

Covers the degenerate-flat-input boundary case for both OLS fits: a series
with zero (or near-zero) variance around its own mean has nothing for a
trendline to explain, so it must score r2=0.0 (no fit-quality signal),
never the 1.0 ("perfect fit") a naive ss_tot==0 guard would otherwise
produce -- see the review finding this closes.
"""

from __future__ import annotations

import numpy as np
import pytest

from firm.patterns.trendline import fit_poly2, fit_trendline, line_through


def test_fit_trendline_real_slope_has_perfect_r2():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, 3.0, 5.0, 7.0]  # y = 2x + 1, exact
    line = fit_trendline(xs, ys)
    assert line is not None
    assert line.slope == pytest.approx(2.0)
    assert line.r2 == pytest.approx(1.0)


def test_fit_trendline_degenerate_flat_input_scores_zero_not_perfect():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [5.0, 5.0, 5.0, 5.0]  # flat -- no variance to explain
    line = fit_trendline(xs, ys)
    assert line is not None
    assert line.r2 == pytest.approx(0.0)


def test_fit_trendline_fewer_than_two_points_returns_none():
    assert fit_trendline([1.0], [1.0]) is None
    assert fit_trendline([], []) is None


def test_fit_poly2_degenerate_flat_input_scores_zero_not_perfect():
    ys = np.full(6, 3.0)
    a, b, c, r2 = fit_poly2(ys)
    assert r2 == pytest.approx(0.0)


def test_fit_poly2_real_curve_has_high_r2():
    x = np.arange(6, dtype=float)
    ys = 0.5 * x**2 - x + 2.0  # exact concave-up parabola
    a, b, c, r2 = fit_poly2(ys)
    assert a > 0
    assert r2 == pytest.approx(1.0)


def test_line_through_two_points_is_always_perfect_fit():
    line = line_through(0.0, 0.0, 2.0, 4.0)
    assert line.slope == pytest.approx(2.0)
    assert line.r2 == pytest.approx(1.0)


def test_line_through_vertical_points_falls_back_to_flat():
    line = line_through(1.0, 3.0, 1.0, 7.0)
    assert line.slope == pytest.approx(0.0)
    assert line.r2 == pytest.approx(1.0)
