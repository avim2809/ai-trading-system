"""Small numeric helpers shared across pattern-detection modules.

Kept dependency-free (numpy only) since these run inside a per-symbol scan
loop that may cover thousands of symbols.
"""

from __future__ import annotations

import numpy as np


def atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's Average True Range, same length as the inputs (NaN-padded head).

    Used to floor stop-loss distances so a pattern's structural stop never
    ends up tighter than the symbol's recent volatility would make noise-prone.
    """
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    out = np.full(n, np.nan)
    if n <= period:
        return out
    out[period] = tr[1 : period + 1].mean()
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def volume_ratio(volume: np.ndarray, at: int, lookback: int = 20) -> float:
    """Volume at bar *at* relative to the trailing *lookback*-bar average
    (excluding *at* itself). Returns ``nan`` when there isn't enough history.
    """
    start = at - lookback
    if start < 0 or at < 0 or at >= len(volume):
        return float("nan")
    baseline = volume[start:at]
    if len(baseline) == 0:
        return float("nan")
    avg = float(np.mean(baseline))
    if avg <= 0:
        return float("nan")
    return float(volume[at]) / avg
