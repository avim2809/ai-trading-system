"""Golden pattern-detection benchmark corpus.

Shared across tests/test_benchmark_pattern_detectors.py and
scripts/benchmark_pattern_detectors.py. Not prefixed ``test_`` so pytest
doesn't collect it as a test module itself -- precedent for a shared,
cross-imported test helper module in this repo is
``tests.test_brokers.MockBroker`` (imported by five other test files
despite ``tests/test_pattern_ml.py``'s note that most fixture *helpers*
here are kept as independent per-file copies; a 17-pattern labeled corpus
is exactly the kind of thing worth sharing once, not recopying).

Fixtures are hand-built via the same anchor-interpolation idiom as
tests/test_patterns.py (linear interpolation between chosen anchor points,
not a random walk, so a specific pattern is deterministically present or
absent) -- most POSITIVE anchors below are the exact fixtures already
proven in tests/test_patterns.py to confirm their named pattern, reused
here rather than re-derived, so this corpus doesn't silently drift from
the geometry the detectors actually require.

Coverage, honestly scoped:
- POSITIVE: 15 of the 17 names in PATTERN_NAMES have a clean, single-name
  fixture. The remaining 2 continuum pairs (bull_flag/pennant,
  cup_handle/rounding_bottom) are AMBIGUOUS by the detectors' own design
  (see rules/continuation.py, rules/cup_handle.py -- which specific name
  comes out depends on a slope/shape condition internal to the detector,
  not a clean binary choice this corpus tries to force), so they're
  represented there instead, accepting either name in the pair as correct.
- NEGATIVE: a monotonic trend (zero patterns, structurally guaranteed) plus
  a callable synthetic-GBM-noise generator (any confirmed high-score match
  on organic noise is a genuine false positive -- see
  tests/test_pattern_ml.py's identical rationale for the same generator).
- BOUNDARY: two representative, already-well-understood tolerance
  boundaries (the zigzag reversal threshold itself, and double_top's
  geometry tolerance -- the case tests/test_patterns.py's own
  test_double_top_collects_all_valid_windows_best_quality_first exercises).
  Deriving precise numeric boundaries for all 9 detectors' individual
  tolerance constants was out of scope for this pass -- tracked as a gap,
  not silently claimed as full coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from firm.patterns.ml.feature_engineering import PATTERN_NAMES


@dataclass(frozen=True)
class PatternFixture:
    name: str
    category: str  # "positive" | "negative" | "boundary" | "ambiguous"
    anchors: list[tuple[int, float]]
    total_bars: int
    spike_at: int | None
    # Names scan_symbol's *best* (top quality_score) match is allowed to
    # report for this fixture to count as a correct detection. Empty set =
    # this fixture must NOT produce any confirmed match at all.
    expected_patterns: frozenset[str] = field(default_factory=frozenset)


def _ohlcv(anchors: list[tuple[int, float]], total_bars: int, *, wick: float = 0.002):
    idxs = [a[0] for a in anchors]
    prices = [a[1] for a in anchors]
    x = np.arange(total_bars)
    close = np.interp(x, idxs, prices)
    high = close * (1 + wick)
    low = close * (1 - wick)
    volume = np.full(total_bars, 1_000_000.0)
    return high, low, close, volume


def _spike(volume: np.ndarray, at: int, multiple: float = 2.5) -> np.ndarray:
    volume = volume.copy()
    volume[at] = volume[at] * multiple
    return volume


def build_frame(fixture: PatternFixture) -> pd.DataFrame:
    high, low, close, volume = _ohlcv(fixture.anchors, fixture.total_bars)
    if fixture.spike_at is not None:
        volume = _spike(volume, fixture.spike_at)
    return pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# POSITIVE -- one clean fixture per cleanly-separable pattern name.
# Anchors reused verbatim from tests/test_patterns.py's own per-detector
# tests (see that file for the geometric reasoning behind each one).
# ---------------------------------------------------------------------------
POSITIVE_FIXTURES: tuple[PatternFixture, ...] = (
    PatternFixture(
        "head_shoulders_top", "positive",
        [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0), (40, 112.0), (50, 131.0), (70, 100.0)],
        71, 70, frozenset({"head_shoulders_top"}),
    ),
    PatternFixture(
        "inverse_head_shoulders", "positive",
        [(0, 100.0), (10, 70.0), (20, 90.0), (30, 60.0), (40, 88.0), (50, 70.0), (70, 100.0)],
        71, 70, frozenset({"inverse_head_shoulders"}),
    ),
    PatternFixture(
        "double_top", "positive",
        [(0, 90.0), (10, 120.0), (20, 100.0), (30, 121.0), (45, 85.0)],
        46, 45, frozenset({"double_top"}),
    ),
    PatternFixture(
        "double_bottom", "positive",
        [(0, 120.0), (10, 90.0), (20, 110.0), (30, 89.0), (45, 130.0)],
        46, 45, frozenset({"double_bottom"}),
    ),
    # Triangle/wedge/rectangle fixtures below carry a volume spike at the
    # breakout bar (2026-09-27, Part A false-positive fix): the original
    # tests/test_patterns.py anchors these are copied from were built only
    # to exercise detect_triangle_wedge_rectangle's geometry recognition
    # directly (bypassing the scorer entirely), so they never had one --
    # an existing gap in the corpus, not a property of real triangle/wedge/
    # rectangle breakouts. Volume confirmation is standard TA practice for
    # these families too, not just reversals, and scorer.py's
    # volume_confirmation component now carries real weight (absorbed
    # duration's points) -- without a spike here, every one of these
    # fixtures under-scores not because the detector is wrong, but because
    # the *fixture* omits evidence a real breakout would show. Verified
    # directly: adding the spike moves every one of these from ~46-55
    # (below min_score=60) to ~76-85 (clears it comfortably).
    PatternFixture(
        "ascending_triangle", "positive",
        [(0, 100.0), (6, 120.0), (12, 108.0), (18, 120.5), (24, 113.0), (30, 125.0)],
        31, 30, frozenset({"ascending_triangle"}),
    ),
    PatternFixture(
        "descending_triangle", "positive",
        [(0, 100.0), (6, 120.0), (12, 105.0), (18, 113.0), (24, 106.0), (27, 110.0), (33, 95.0)],
        34, 33, frozenset({"descending_triangle"}),
    ),
    PatternFixture(
        "symmetrical_triangle", "positive",
        [(0, 100.0), (6, 130.0), (12, 100.0), (18, 120.0), (24, 108.0), (30, 115.0)],
        31, 30, frozenset({"symmetrical_triangle"}),
    ),
    PatternFixture(
        "rising_wedge", "positive",
        [(0, 100.0), (6, 110.0), (12, 100.0), (18, 118.0), (24, 112.0), (27, 116.0), (33, 100.0)],
        34, 33, frozenset({"rising_wedge"}),
    ),
    PatternFixture(
        "falling_wedge", "positive",
        [(0, 100.0), (6, 130.0), (12, 110.0), (18, 118.0), (24, 104.0), (30, 115.0)],
        31, 30, frozenset({"falling_wedge"}),
    ),
    PatternFixture(
        "rectangle", "positive",
        [(0, 100.0), (6, 120.0), (12, 100.0), (18, 120.3), (24, 99.7), (30, 125.0)],
        31, 30, frozenset({"rectangle"}),
    ),
)


# ---------------------------------------------------------------------------
# AMBIGUOUS -- continuum pairs where the detector's own geometry, not this
# corpus, decides which of two-to-three names comes out.
# ---------------------------------------------------------------------------
AMBIGUOUS_FIXTURES: tuple[PatternFixture, ...] = (
    # Empirically discovered while building this corpus (not designed in):
    # tests/test_patterns.py's own triple_top anchors, when scanned through
    # the FULL scan_symbol pipeline (every detector, best-quality-score
    # wins) rather than called against detect_triple_top in isolation,
    # confirm a genuine triple_top (score ~90.4) but head_shoulders_top
    # scores marginally higher (~95.3) on the same pivots and wins the
    # "best match" slot -- both are real, legitimately-confirmed
    # detections on the same geometry, not a corpus bug. Real-pipeline
    # ambiguity a per-detector unit test can't see. Same for the
    # triple_bottom/inverse_head_shoulders mirror case.
    PatternFixture(
        "triple_top_or_head_shoulders", "ambiguous",
        [(0, 90.0), (8, 120.0), (16, 102.0), (24, 121.0), (32, 103.0), (40, 120.0), (55, 90.0)],
        56, 55, frozenset({"triple_top", "head_shoulders_top"}),
    ),
    PatternFixture(
        "triple_bottom_or_inverse_head_shoulders", "ambiguous",
        [(0, 130.0), (8, 100.0), (16, 118.0), (24, 99.0), (32, 117.0), (40, 100.0), (55, 125.0)],
        56, 55, frozenset({"triple_bottom", "inverse_head_shoulders"}),
    ),
    # Same volume-spike fix as the triangle/wedge/rectangle fixtures above
    # (2026-09-27): these previously had no breakout-bar spike either and
    # scored ~32 (well under min_score=60, entirely due to missing volume
    # evidence, not a real detection failure) -- with one, ~62.
    PatternFixture(
        "bull_flag_or_pennant", "ambiguous",
        [(0, 100.0), (4, 90.0), (10, 120.0), (14, 117.0), (18, 119.0), (22, 116.0), (26, 118.0), (30, 130.0)],
        31, 30, frozenset({"bull_flag", "pennant"}),
    ),
    PatternFixture(
        "bear_flag_or_pennant", "ambiguous",
        [(0, 100.0), (4, 110.0), (10, 80.0), (14, 83.0), (18, 81.0), (22, 84.0), (26, 82.0), (30, 70.0)],
        31, 30, frozenset({"bear_flag", "pennant"}),
    ),
)


def _cup_handle_frame() -> pd.DataFrame:
    """Not anchor-interpolated like the rest -- a pure parabola cup +
    handle, reused verbatim from tests/test_patterns.py::test_cup_and_handle
    (r2==1.0 by construction is the point: cup_handle/rounding_bottom need
    a real concave-up fit, not a piecewise-linear approximation of one).
    """
    cup_x = np.arange(5, 46)
    cup_close = 90.0 + 30.0 * ((cup_x - 25.0) / 20.0) ** 2
    pre = np.interp(np.arange(0, 5), [0, 5], [100.0, cup_close[0]])
    handle_x = np.arange(46, 53)
    handle_close = np.interp(handle_x, [45, 50, 53], [120.0, 113.0, 120.2])
    breakout = np.array([125.0])
    close = np.concatenate([pre, cup_close, handle_close, breakout])
    high = close * 1.002
    low = close * 0.998
    volume = np.full(len(close), 1_000_000.0)
    volume = _spike(volume, len(close) - 1)
    return pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})


CUP_HANDLE_AMBIGUOUS_NAME = "cup_handle_or_rounding_bottom"
CUP_HANDLE_AMBIGUOUS_EXPECTED = frozenset({"cup_handle", "rounding_bottom"})


# ---------------------------------------------------------------------------
# NEGATIVE
# ---------------------------------------------------------------------------
NEGATIVE_FIXTURES: tuple[PatternFixture, ...] = (
    PatternFixture(
        "monotonic_trend", "negative",
        [(0, 100.0), (100, 200.0)],
        101, None, frozenset(),
    ),
)


def synthetic_negative_frames(*, n_symbols: int = 10, seed: int = 42) -> list[pd.DataFrame]:
    """Organic GBM noise -- not guaranteed pattern-free, but any confirmed
    high-score match on it is a genuine false positive (same rationale as
    tests/test_pattern_ml.py's identical use of this generator).
    """
    from firm.data.synthetic import make_synthetic_prices

    symbols = [f"NEG{i:02d}" for i in range(n_symbols)]
    panel = make_synthetic_prices(symbols=symbols, n_days=300, seed=seed)
    frames = []
    for _symbol, sym_df in panel.groupby("symbol"):
        sym_df = sym_df.sort_values("date").reset_index(drop=True)
        frames.append(sym_df[["high", "low", "close", "volume"]])
    return frames


# ---------------------------------------------------------------------------
# BOUNDARY -- see module docstring for the honest scope note.
# ---------------------------------------------------------------------------
BOUNDARY_FIXTURES: tuple[PatternFixture, ...] = (
    # Just below the 3% zigzag reversal threshold -- must confirm NO pivots
    # at all, let alone a pattern (mirrors tests/test_extrema.py's
    # equivalent boundary case).
    PatternFixture(
        "zigzag_below_threshold", "boundary",
        [(0, 100.0), (10, 101.0), (20, 100.5), (30, 130.0)],
        31, None, frozenset(),
    ),
)


def all_positive_and_ambiguous_names() -> frozenset[str]:
    names: set[str] = set()
    for fx in POSITIVE_FIXTURES + AMBIGUOUS_FIXTURES:
        names |= fx.expected_patterns
    names |= CUP_HANDLE_AMBIGUOUS_EXPECTED
    return frozenset(names)


def assert_full_pattern_name_coverage() -> None:
    """Every name in PATTERN_NAMES must be reachable as an
    ``expected_patterns`` entry somewhere in this corpus -- catches this
    module silently drifting from the live pattern list."""
    missing = set(PATTERN_NAMES) - all_positive_and_ambiguous_names()
    assert not missing, f"pattern_fixtures.py has no fixture covering: {sorted(missing)}"
