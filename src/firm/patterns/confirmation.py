"""Shared breakout-confirmation search used by every pattern-rule module.

A detected geometric structure (neckline, trendline, handle-high, ...) only
becomes a tradeable pattern once price actually closes through the relevant
level — see the "Confirmation" column in the pattern-rule design. This finds
the most recent such close within a lookback window, so a formation that
completed weeks ago but never broke out simply doesn't match.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

Direction = Literal["above", "below"]


def find_confirmation(
    close: np.ndarray,
    level_at: Callable[[int], float],
    direction: Direction,
    lookback_bars: int,
    min_index: int,
) -> int:
    """Most recent bar index in the last *lookback_bars* where ``close``
    crossed ``level_at(i)`` in *direction*, searching newest-first so the
    freshest breakout wins. Returns -1 if none found (pattern not yet, or no
    longer, confirmed). Never considers bars at or before *min_index* (the
    pattern's own last structural pivot) — a "breakout" before the pattern
    even finished forming isn't one.
    """
    n = len(close)
    start = max(min_index + 1, n - lookback_bars)
    for i in range(n - 1, start - 1, -1):
        level = level_at(i)
        if direction == "above" and close[i] > level:
            return i
        if direction == "below" and close[i] < level:
            return i
    return -1
