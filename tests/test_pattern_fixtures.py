"""Sanity checks on tests/pattern_fixtures.py itself -- the golden
benchmark corpus must stay in sync with the live pattern list and must
actually produce the OHLCV shape scan_symbol expects.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from pattern_fixtures import (
    AMBIGUOUS_FIXTURES,
    BOUNDARY_FIXTURES,
    NEGATIVE_FIXTURES,
    POSITIVE_FIXTURES,
    assert_full_pattern_name_coverage,
    build_frame,
    synthetic_negative_frames,
)

from firm.patterns.ml.feature_engineering import PATTERN_NAMES


def test_every_pattern_name_is_covered_by_the_corpus():
    assert_full_pattern_name_coverage()


def test_build_frame_produces_expected_columns_and_length():
    for fx in POSITIVE_FIXTURES + AMBIGUOUS_FIXTURES + NEGATIVE_FIXTURES + BOUNDARY_FIXTURES:
        df = build_frame(fx)
        assert list(df.columns) == ["high", "low", "close", "volume"]
        assert len(df) == fx.total_bars


def test_positive_and_ambiguous_fixtures_have_nonempty_expectations():
    for fx in POSITIVE_FIXTURES + AMBIGUOUS_FIXTURES:
        assert fx.expected_patterns, f"{fx.name} should have >=1 expected pattern"


def test_negative_and_boundary_fixtures_expect_nothing():
    for fx in NEGATIVE_FIXTURES + BOUNDARY_FIXTURES:
        assert fx.expected_patterns == frozenset()


def test_synthetic_negative_frames_shape():
    frames = synthetic_negative_frames(n_symbols=3, seed=1)
    assert len(frames) == 3
    for df in frames:
        assert list(df.columns) == ["high", "low", "close", "volume"]
        assert len(df) > 0

    # Deterministic given the same seed.
    frames_again = synthetic_negative_frames(n_symbols=3, seed=1)
    assert frames[0]["close"].tolist() == frames_again[0]["close"].tolist()


def test_pattern_names_constant_has_17_entries():
    # Guards against this test file's own assumptions silently drifting if
    # the live pattern list ever changes shape.
    assert len(PATTERN_NAMES) == 17
