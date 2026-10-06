"""Shared null/permutation library.

Nulls live here, tested for uniform p-values under their own H0, and never ad
hoc in strategy code (the Gann null that returned p = 1.000 was a broken
one-off). Methods: permutation p-value ``(1 + count) / (1 + n)`` (Phipson and
Smyth 2010); stationary bootstrap (Politis and Romano 1994) with automatic block
length (Politis and White 2004, Patton, Politis and White 2009), see
``firm.validation.bootstrap``; phase-randomised surrogates (Theiler et al. 1992).

Every function takes an explicit ``seed`` (``np.random.default_rng``; no global
RNG). ``firm.patterns.significance`` is the live pattern-specific counterpart and
is not touched; equivalence of ``p_value`` and ``benjamini_hochberg`` with it is
pinned by tests.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

import numpy as np

from firm.validation.bootstrap import (
    block_bootstrap_returns,
    politis_white_block_length,
)

log = logging.getLogger(__name__)

__all__ = [
    "benjamini_hochberg",
    "block_bootstrap_returns",
    "mean_return",
    "p_value",
    "phase_randomised_surrogate",
    "random_entry_null",
    "sign_randomisation_null",
]


def mean_return(x: np.ndarray) -> float:
    return float(np.mean(x))


def p_value(stat_obs: float, stat_null: np.ndarray, tail: Literal["right", "left", "two"] = "right") -> float:
    """``(1 + #{null >= obs}) / (1 + n)`` (right); ``<=`` for left; two-sided
    ``min(1, 2*min(p_left, p_right))``. Never 0; an empty null raises."""
    null = np.asarray(stat_null, dtype=float)
    if null.size == 0:
        raise ValueError("empty null distribution")
    if np.isnan(null).any() or np.isnan(stat_obs):
        raise ValueError("NaN in observed statistic or null distribution")
    n = null.size
    p_right = (1.0 + float(np.sum(null >= stat_obs))) / (1.0 + n)
    p_left = (1.0 + float(np.sum(null <= stat_obs))) / (1.0 + n)
    if tail == "right":
        return p_right
    if tail == "left":
        return p_left
    if tail == "two":
        return min(1.0, 2.0 * min(p_left, p_right))
    raise ValueError(f"unknown tail {tail!r}")


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.10) -> np.ndarray:
    """BH step-up: boolean accept mask, same order as ``p_values``."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    passing = p[order] <= (np.arange(1, m + 1) / m) * q
    accept = np.zeros(m, dtype=bool)
    if passing.any():
        accept[order[: int(np.max(np.where(passing)[0])) + 1]] = True
    return accept


def _apply_stat(stat: Callable[[np.ndarray], float], mat: np.ndarray) -> np.ndarray:
    if stat is mean_return:
        return mat.mean(axis=1)
    return np.fromiter((stat(row) for row in mat), dtype=float, count=len(mat))


def sign_randomisation_null(
    positions: np.ndarray,
    returns: np.ndarray,
    n: int,
    seed: int,
    stat: Callable[[np.ndarray], float] = mean_return,
    block_len: float | None = None,
) -> np.ndarray:
    """Block sign-randomisation null of ``stat`` of the per-period PnL ``positions * returns``.

    One random sign per stationary-bootstrap block (mean ``block_len``; default
    Politis-White on the PnL) multiplies the PnL; the statistic is recomputed.
    Valid under a symmetric, zero-mean return conditional on positions and robust
    to autocorrelated PnL. ``block_len=1`` is the i.i.d. sign flip: it understates
    the variance of the mean for autocorrelated PnL (null too narrow, over-rejects)
    and is allowed only when passed explicitly (logs a WARNING).
    """
    pos = np.asarray(positions, dtype=float)
    ret = np.asarray(returns, dtype=float)
    if pos.shape != ret.shape or pos.ndim != 1:
        raise ValueError("positions and returns must be 1-d and equal length")
    if n < 1:
        raise ValueError("n must be >= 1")
    pnl = pos * ret
    if block_len is None:
        block_len = politis_white_block_length(pnl)
    elif block_len <= 1:
        log.warning("sign_randomisation_null: block_len=%s is an i.i.d. sign flip; null is too narrow for autocorrelated PnL", block_len)
    if not block_len >= 1:
        raise ValueError("block_len must be >= 1")
    rng = np.random.default_rng(seed)
    t = len(pnl)
    new = rng.random((n, t)) < 1.0 / float(block_len)
    new[:, 0] = True
    block_id = np.cumsum(new, axis=1) - 1  # block index per period
    signs = rng.integers(0, 2, size=(n, t)) * 2.0 - 1.0  # one candidate sign per block id
    path_signs = np.take_along_axis(signs, block_id, axis=1)
    return _apply_stat(stat, path_signs * pnl)


def phase_randomised_surrogate(r: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``(n, len(r))`` surrogates with the power spectrum of ``r`` and random phases.

    DC (and Nyquist for even length) keep their values, so the mean is preserved
    and, by Parseval, so is the variance.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or len(r) < 4:
        raise ValueError("r must be 1-d with at least 4 points")
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    f = np.fft.rfft(r)
    mag = np.abs(f)
    phase = np.angle(f)
    free = slice(1, len(f) - 1) if len(r) % 2 == 0 else slice(1, len(f))
    new_phase = np.tile(phase, (n, 1))
    new_phase[:, free] = rng.uniform(0.0, 2.0 * np.pi, size=(n, new_phase[:, free].shape[1]))
    return np.fft.irfft(mag * np.exp(1j * new_phase), n=len(r), axis=1)


def _extract_trades(pos: np.ndarray) -> list[tuple[float, int]]:
    """Maximal runs of constant non-zero position as (value, length)."""
    trades: list[tuple[float, int]] = []
    start = None
    for i, v in enumerate(pos):
        if start is not None and v != pos[start]:
            trades.append((float(pos[start]), i - start))
            start = None
        if start is None and v != 0:
            start = i
    if start is not None:
        trades.append((float(pos[start]), len(pos) - start))
    return trades


def random_entry_null(
    strategy: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    n: int,
    seed: int,
    keep_holding_periods: bool = True,
    stat: Callable[[np.ndarray], float] = mean_return,
) -> np.ndarray:
    """Null of ``stat`` of ``positions * data`` with the trades re-placed at random.

    ``strategy(data)`` returns per-period positions aligned with the 1-d return
    series ``data``. Trades are maximal runs of constant non-zero position. The
    null keeps the number of trades and (``keep_holding_periods=True``) their
    holding periods and values, draws the non-overlapping placement uniformly
    (random trade order, uniform random composition of the idle periods into gaps).
    With ``keep_holding_periods=False`` lengths are resampled with replacement from
    the empirical distribution (redrawn if they do not fit; falls back to the
    observed lengths after 100 failed tries). Raises if the strategy never trades.
    """
    data = np.asarray(data, dtype=float)
    pos = np.asarray(strategy(data), dtype=float)
    if pos.shape != data.shape or data.ndim != 1:
        raise ValueError("strategy must return positions with the same 1-d shape as data")
    trades = _extract_trades(pos)
    if not trades:
        raise ValueError("strategy produced no trades; random-entry null is undefined")
    values = np.array([v for v, _ in trades])
    lengths = np.array([ln for _, ln in trades])
    t, k = len(data), len(trades)
    rng = np.random.default_rng(seed)
    out = np.empty((n, t))
    for i in range(n):
        ln = lengths
        if not keep_holding_periods:
            for _ in range(100):
                cand = rng.choice(lengths, size=k, replace=True)
                if cand.sum() <= t:
                    ln = cand
                    break
        order = rng.permutation(k)
        slack = t - int(ln.sum())
        # uniform random composition of `slack` idle periods into k+1 gaps
        cuts = np.sort(rng.integers(0, slack + 1, size=k))
        gaps = np.diff(np.concatenate([[0], cuts, [slack]]))
        row = np.zeros(t)
        cur = int(gaps[0])
        for j, o in enumerate(order):
            row[cur : cur + ln[o]] = values[o]
            cur += int(ln[o]) + int(gaps[j + 1])
        out[i] = row
    return _apply_stat(stat, out * data)
