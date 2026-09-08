"""Turning-point extraction for chart-pattern detection.

Every classical chart pattern (Lo, Mamaysky & Wang 2000) is defined as a
sequence of alternating price extrema. The standard, well-proven way to pull
those out of noisy daily bars is a percentage-reversal ZigZag: track the
running high/low of the current swing and confirm (and flip) a pivot once
price reverses by more than ``pct`` from that extreme. This is simpler and
faster than recursive Perceptually-Important-Points extraction and is what
most production chart-pattern scanners (e.g. the open-source `stock-pattern`
and `zigzag` libraries) use in practice — it guarantees strict alternation by
construction, so no separate merge/dedup pass is needed.

Cup & Handle / Rounding Bottom don't use pivots at all — they fit a smooth
curve to the whole window, so :func:`gaussian_smooth` is provided separately
for that use case.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

PivotKind = Literal["peak", "trough"]


class Pivot(NamedTuple):
    index: int  # position within the input arrays (0-based)
    price: float
    kind: PivotKind


def zigzag_pivots(high: np.ndarray, low: np.ndarray, pct: float = 0.03) -> list[Pivot]:
    """Confirmed alternating peak/trough pivots, oldest first.

    Only *confirmed* reversals are returned — the still-forming swing at the
    end of the series (which could still extend) is deliberately dropped, so
    every pivot returned is a stable anchor for pattern geometry. ``pct`` is
    the minimum retracement (as a fraction of the swing extreme) required to
    confirm a reversal; smaller values yield more, noisier pivots.
    """
    n = len(high)
    if n < 3:
        return []

    pivots: list[Pivot] = []
    direction = 0  # 0 = undetermined, 1 = tracking a swing high, -1 = tracking a swing low
    # During bootstrap we don't yet know which side will confirm first, so
    # both the running high and running low are tracked from bar 0 (bar 0
    # itself is never emitted as a pivot — it's an arbitrary window edge,
    # not a confirmed reversal with data on both sides of it).
    max_idx, max_price = 0, float(high[0])
    min_idx, min_price = 0, float(low[0])
    extreme_idx = 0
    extreme_price = float(high[0])

    for i in range(1, n):
        if direction == 0:
            if float(high[i]) > max_price:
                max_idx, max_price = i, float(high[i])
            if float(low[i]) < min_price:
                min_idx, min_price = i, float(low[i])
            # A confirmed downswing means the running high was a peak; a
            # confirmed upswing means the running low was a trough — note
            # each check compares against the *opposite* running extreme.
            if low[i] <= max_price * (1 - pct):
                if max_idx != 0:
                    pivots.append(Pivot(max_idx, max_price, "peak"))
                direction = -1
                extreme_idx, extreme_price = i, float(low[i])
            elif high[i] >= min_price * (1 + pct):
                if min_idx != 0:
                    pivots.append(Pivot(min_idx, min_price, "trough"))
                direction = 1
                extreme_idx, extreme_price = i, float(high[i])
            continue

        if direction == 1:
            if high[i] > extreme_price:
                extreme_idx, extreme_price = i, float(high[i])
            elif low[i] <= extreme_price * (1 - pct):
                pivots.append(Pivot(extreme_idx, extreme_price, "peak"))
                direction = -1
                extreme_idx, extreme_price = i, float(low[i])
        else:
            if low[i] < extreme_price:
                extreme_idx, extreme_price = i, float(low[i])
            elif high[i] >= extreme_price * (1 + pct):
                pivots.append(Pivot(extreme_idx, extreme_price, "trough"))
                direction = 1
                extreme_idx, extreme_price = i, float(high[i])

    return pivots


def recent_pivot_windows(
    pivots: list[Pivot], size: int, max_lookback: int = 3
) -> list[list[Pivot]]:
    """Candidate windows of *size* consecutive pivots, most recent first.

    A pattern's true last structural pivot (e.g. a Head & Shoulders' right
    shoulder) is not always ``pivots[-1]``: if the subsequent move away from
    it — the breakout itself, a post-breakout bounce, a handle's pullback —
    is large enough to trigger its own zigzag confirmation, *that* becomes
    the newest pivot instead, pushing the real pattern one or more pivots
    back in the list. Trying a few trailing offsets (most recent first, so
    the freshest valid match still wins) lets a detector see past that noise
    instead of only ever checking the single most recent window.
    """
    n = len(pivots)
    if n < size:
        return []
    max_offset = min(max_lookback, n - size)
    return [pivots[n - size - offset : n - offset] for offset in range(max_offset + 1)]


def gaussian_smooth(values: np.ndarray, bandwidth_fraction: float = 0.1) -> np.ndarray:
    """Nadaraya-Watson-style Gaussian kernel smoother, vectorised.

    ``bandwidth_fraction`` follows Lo/Mamaysky/Wang's convention of scaling
    the kernel bandwidth to the window length rather than a fixed bar count,
    so the same fraction works for both a 60-bar and a 250-bar window.
    """
    n = len(values)
    if n == 0:
        return values.astype(float)
    h = max(1.0, bandwidth_fraction * n)
    x = np.arange(n, dtype=float)
    # weights[t, i] = kernel distance between bar t and bar i
    d = (x[:, None] - x[None, :]) / h
    weights = np.exp(-0.5 * d * d)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ values
