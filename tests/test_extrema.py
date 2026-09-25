"""Tests for firm.patterns.extrema.zigzag_pivots, including the optional
per-bar ``threshold_fn`` hook (ATR-scaled thresholds).

Fixtures follow tests/test_patterns.py's convention: anchors are (index,
price) pairs linearly interpolated into OHLC arrays, with each leg strictly
monotonic and exceeding the reversal threshold so confirmed pivots land
exactly on the chosen anchors. Per firm.patterns.extrema.zigzag_pivots's own
docstring, bar 0 is never emitted as a pivot (it's the bootstrap anchor, not
a confirmed reversal), so a fixture's first confirmed pivot is anchor[1].
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from firm.patterns.extrema import zigzag_pivots


def _ohlc(anchors: list[tuple[int, float]], total_bars: int):
    idxs = [a[0] for a in anchors]
    prices = [a[1] for a in anchors]
    x = np.arange(total_bars)
    close = np.interp(x, idxs, prices)
    high = close * 1.002
    low = close * 0.998
    return high, low


# ---------------------------------------------------------------------------
# Baseline (fixed pct, threshold_fn=None) — mirrors test_patterns.py's
# existing zigzag coverage, kept here so this module's regression tests have
# a same-file baseline to diff ``threshold_fn`` behaviour against.
# ---------------------------------------------------------------------------


def test_zigzag_finds_alternating_pivots_at_exact_anchors():
    anchors = [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0)]
    high, low = _ohlc(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    assert [p.index for p in pivots] == [10, 20]
    assert [p.kind for p in pivots] == ["peak", "trough"]
    assert pivots[0].price == pytest.approx(130.0, rel=0.01)
    assert pivots[1].price == pytest.approx(110.0, rel=0.01)


def test_zigzag_ignores_moves_below_threshold():
    anchors = [(0, 100.0), (10, 101.0), (20, 100.5), (30, 130.0)]
    high, low = _ohlc(anchors, 31)
    pivots = zigzag_pivots(high, low, pct=0.03)
    assert pivots == []  # still tracking the initial up-move; nothing confirmed yet


# ---------------------------------------------------------------------------
# threshold_fn=None regression safety: identical to the pre-change fixed-pct
# call for existing fixture cases.
# ---------------------------------------------------------------------------


class TestThresholdFnDefaultIsIdentical:
    @pytest.mark.parametrize(
        "anchors,total_bars,pct",
        [
            ([(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0)], 31, 0.03),
            ([(0, 100.0), (10, 101.0), (20, 100.5), (30, 130.0)], 31, 0.03),
            (
                [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0), (40, 112.0), (50, 131.0), (70, 100.0)],
                71,
                0.03,
            ),
            ([(0, 90.0), (10, 120.0), (20, 100.0), (30, 121.0), (45, 85.0)], 46, 0.05),
        ],
    )
    def test_matches_fixed_pct_baseline(self, anchors, total_bars, pct):
        high, low = _ohlc(anchors, total_bars)
        baseline = zigzag_pivots(high, low, pct=pct)
        with_none = zigzag_pivots(high, low, pct=pct, threshold_fn=None)
        assert with_none == baseline

    def test_omitting_threshold_fn_kwarg_entirely_matches_baseline(self):
        anchors = [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0)]
        high, low = _ohlc(anchors, 31)
        baseline = zigzag_pivots(high, low, pct=0.03)
        explicit_default = zigzag_pivots(high, low, 0.03)
        assert baseline == explicit_default


# ---------------------------------------------------------------------------
# threshold_fn actually changes reversal sensitivity, sub-range by sub-range.
# ---------------------------------------------------------------------------


class TestThresholdFnChangesSensitivity:
    def _choppy_series(self, total_bars: int = 80, amplitude: float = 0.02):
        # A clean sine oscillation with a ~4%-ish peak-to-trough swing
        # everywhere: below a 5% fixed pct (confirms nothing) but well
        # above a 1% tight threshold (confirms every quarter-period).
        t = np.arange(total_bars)
        close = 100.0 * (1 + amplitude * np.sin(t * (np.pi / 4)))
        high = close * 1.001
        low = close * 0.999
        return high, low

    def test_tight_sub_range_yields_more_pivots_than_loose_baseline(self):
        high, low = self._choppy_series(80)
        split = 40

        baseline_loose = zigzag_pivots(high, low, pct=0.05)
        baseline_loose_in_range = [p for p in baseline_loose if p.index < split]
        assert baseline_loose_in_range == []  # sanity: fixed 5% pct finds nothing here

        def threshold_fn(i: int) -> float:
            return 0.01 if i < split else 0.05

        scaled = zigzag_pivots(high, low, pct=0.05, threshold_fn=threshold_fn)
        scaled_in_range = [p for p in scaled if p.index < split]

        assert len(scaled_in_range) > len(baseline_loose_in_range)
        assert len(scaled_in_range) > 0

    def test_loose_sub_range_yields_fewer_pivots_than_tight_baseline(self):
        high, low = self._choppy_series(80)
        split = 40

        baseline_tight = zigzag_pivots(high, low, pct=0.01)
        baseline_tight_after = [p for p in baseline_tight if p.index >= split]
        assert len(baseline_tight_after) > 0  # sanity: fixed 1% pct finds several here

        def threshold_fn(i: int) -> float:
            return 0.01 if i < split else 0.05

        scaled = zigzag_pivots(high, low, pct=0.01, threshold_fn=threshold_fn)
        scaled_after = [p for p in scaled if p.index >= split]

        assert len(scaled_after) < len(baseline_tight_after)


# ---------------------------------------------------------------------------
# Per-bar fallback: None/0/negative/NaN threshold_fn results fall back to
# pct for those bars specifically, not the whole run.
# ---------------------------------------------------------------------------


class TestThresholdFnFallback:
    def _anchors_series(self):
        anchors = [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0), (40, 118.0)]
        return _ohlc(anchors, 41)

    @pytest.mark.parametrize("bad_value", [None, 0, -0.02, float("nan")])
    def test_bad_value_at_every_bar_falls_back_to_uniform_pct(self, bad_value):
        high, low = self._anchors_series()

        def always_bad(i: int) -> float:
            return bad_value

        result = zigzag_pivots(high, low, pct=0.03, threshold_fn=always_bad)
        baseline = zigzag_pivots(high, low, pct=0.03)
        assert result == baseline

    def test_bad_value_only_on_specific_bars_falls_back_only_there(self):
        # Bars 0-4: a clean run-up to a peak at 130. Bars 5-8 ("the bad
        # window"): a choppy pullback+recovery whose individual legs are
        # each >1% (so a *tight* 1% threshold would confirm two extra,
        # noisy intermediate pivots there) but every leg individually is
        # <3% (so the fixed pct=0.03 fallback rides straight through it,
        # confirming only the eventual deeper trough at bar 8). Bars 9-13
        # (outside the bad window): another shallow dip+recovery that only
        # the *real* 1% threshold_fn value (not the pct=0.03 fallback)
        # picks up, proving the fallback is scoped to the bad bars only.
        prefix = [100.0, 105.0, 110.0, 120.0, 130.0]
        window = [127.5, 130.0, 126.5, 124.0, 135.0]
        tail = [133.7, 136.0, 132.0, 145.0]
        values = np.array(prefix + window + tail)
        high = low = values

        tight = zigzag_pivots(high, low, pct=0.01)
        loose = zigzag_pivots(high, low, pct=0.03)

        bad_bars = set(range(5, 9))

        def threshold_fn(i: int) -> float:
            return math.nan if i in bad_bars else 0.01

        mixed = zigzag_pivots(high, low, pct=0.03, threshold_fn=threshold_fn)

        # Inside the bad window (indices < 9): behaves like the pct=0.03
        # fallback (misses the two tight-only intermediate pivots)...
        mixed_before_9 = [p for p in mixed if p.index < 9]
        loose_before_9 = [p for p in loose if p.index < 9]
        tight_before_9 = [p for p in tight if p.index < 9]
        assert mixed_before_9 == loose_before_9
        assert mixed_before_9 != tight_before_9
        assert len(tight_before_9) > len(loose_before_9)

        # ...while outside the bad window (indices >= 9): behaves like the
        # real tight threshold_fn value, not the pct fallback.
        mixed_from_9 = [p for p in mixed if p.index >= 9]
        tight_from_9 = [p for p in tight if p.index >= 9]
        loose_from_9 = [p for p in loose if p.index >= 9]
        assert mixed_from_9 == tight_from_9
        assert mixed_from_9 != loose_from_9

    def test_mixed_bad_values_across_bars_never_raises(self):
        high, low = self._anchors_series()
        bad_cycle = [None, 0, -1.0, float("nan"), 0.02]

        def cycling(i: int) -> float:
            return bad_cycle[i % len(bad_cycle)]

        # Must not raise, and every non-fallback (0.02) bar should behave
        # like a real threshold while bad ones silently use pct=0.03.
        result = zigzag_pivots(high, low, pct=0.03, threshold_fn=cycling)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# threshold_fn raising propagates (fail-loud, consistent with this module's
# no-fail-soft-internally philosophy — a buggy caller-supplied callback
# should surface immediately rather than being swallowed).
# ---------------------------------------------------------------------------


class TestThresholdFnRaises:
    def test_exception_in_threshold_fn_propagates(self):
        anchors = [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0)]
        high, low = _ohlc(anchors, 31)

        def boom(i: int) -> float:
            raise RuntimeError("caller bug")

        with pytest.raises(RuntimeError, match="caller bug"):
            zigzag_pivots(high, low, pct=0.03, threshold_fn=boom)
