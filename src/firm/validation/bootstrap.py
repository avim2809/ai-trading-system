"""Stationary bootstrap and automatic block-length selection.

Methods
-------
* Politis and Romano (1994), the stationary bootstrap: blocks of Geometric(p)
  length, ``p = 1 / mean_block``, series treated as circular.
* Politis and White (2004) automatic block length, with the Patton, Politis and
  White (2009) correction (``D_SB = 2 g(0)^2``), implemented in numpy because
  ``arch`` is not a dependency. Flat-top bandwidth ``M = 2 * m_hat`` as in the
  paper and in ``arch.bootstrap.optimal_block_length``.

Every function takes an explicit ``seed`` and uses ``np.random.default_rng``;
no global RNG state is read or written. This module replaces the copy-pasted
generators in the frozen evaluation scripts FOR NEW WORK ONLY.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


def _autocov(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocovariances R(0..max_lag) (divide by n)."""
    n = len(x)
    xc = x - x.mean()
    f = np.fft.rfft(xc, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f))[: max_lag + 1] / n
    return acov


def _block_length_1d(x: np.ndarray, kind: str) -> float:
    n = len(x)
    if n < 8:
        raise ValueError(f"series too short for block-length selection (n={n})")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains non-finite values")
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    max_lag = min(math.ceil(math.sqrt(n)) + k_n, n - 1)
    r = _autocov(x, max_lag)
    if r[0] <= 0:
        return 1.0  # constant series
    rho = r / r[0]
    crit = 2.0 * math.sqrt(math.log10(n) / n)
    insig = np.abs(rho[1:]) < crit  # insig[j-1] <-> lag j
    # m_hat: smallest m with |rho(m+j)| < crit for j = 1..K_n
    m_hat = None
    for m in range(max_lag - k_n + 1):
        if insig[m : m + k_n].all():
            m_hat = m
            break
    if m_hat is None:
        m_hat = max_lag // 2
    big_m = min(2 * m_hat, max_lag)
    if big_m == 0:
        return 1.0
    k = np.arange(1, big_m + 1)
    s = k / big_m
    lam = np.where(s <= 0.5, 1.0, 2.0 * (1.0 - s))
    g_big = 2.0 * np.sum(lam * k * r[1 : big_m + 1])  # sum over |k| <= M of lambda*|k|*R(k)
    g0 = r[0] + 2.0 * np.sum(lam * r[1 : big_m + 1])
    if kind == "stationary":
        d = 2.0 * g0**2
    elif kind == "circular":
        d = (4.0 / 3.0) * g0**2
    else:
        raise ValueError(f"unknown kind {kind!r}")
    if d <= 0:
        return 1.0
    b = (2.0 * g_big**2 / d) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    cap = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    return float(max(1.0, min(b, cap)))


def politis_white_block_length(x: np.ndarray, *, kind: Literal["stationary", "circular"] = "stationary") -> float:
    """Automatic (mean) block length; for 2-d ``(n, p)`` input the maximum over columns."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return _block_length_1d(x, kind)
    if x.ndim == 2:
        return float(max(_block_length_1d(x[:, j], kind) for j in range(x.shape[1])))
    raise ValueError("x must be 1-d or 2-d")


def stationary_bootstrap_indices(
    n: int, n_sims: int, mean_block: float, seed: int, n_periods: int | None = None
) -> np.ndarray:
    """``(n_sims, n_periods)`` int indices into a length-``n`` series (Politis-Romano).

    ``idx[0]`` is uniform; at each later step a new block starts at a fresh
    uniform index with probability ``1/mean_block``, else ``idx[t] = (idx[t-1]+1) mod n``.
    """
    if n < 1 or n_sims < 1:
        raise ValueError("n and n_sims must be >= 1")
    if not mean_block >= 1:
        raise ValueError(f"mean_block must be >= 1, got {mean_block}")
    t = n if n_periods is None else int(n_periods)
    rng = np.random.default_rng(seed)
    p = 1.0 / float(mean_block)
    new = rng.random((n_sims, t)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_sims, t))
    pos = np.broadcast_to(np.arange(t), (n_sims, t))
    block_start_pos = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    start_val = np.take_along_axis(starts, block_start_pos, axis=1)
    return ((start_val + pos - block_start_pos) % n).astype(np.int64)


def block_bootstrap_returns(
    r: np.ndarray, block_len: float | None, n: int, seed: int, n_periods: int | None = None
) -> np.ndarray:
    """``(n, n_periods)`` stationary-bootstrap resamples of ``r``; ``block_len=None`` -> Politis-White."""
    r = np.asarray(r, dtype=float)
    if r.ndim != 1:
        raise ValueError("r must be 1-d")
    b = politis_white_block_length(r) if block_len is None else float(block_len)
    idx = stationary_bootstrap_indices(len(r), n, b, seed, n_periods)
    return r[idx]


def paired_sharpe_gap_bootstrap(
    a: np.ndarray, b: np.ndarray, n: int, seed: int, block_len: float | None = None
) -> np.ndarray:
    """Per-simulation ``Sharpe(a) - Sharpe(b)`` (per-period, ddof=1) with common resampled indices.

    Block length ``None`` -> Politis-White on both series (maximum).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-d and equal length")
    bl = politis_white_block_length(np.column_stack([a, b])) if block_len is None else float(block_len)
    rng_seed = np.random.SeedSequence(seed)
    out = np.empty(n)
    chunk = max(1, min(n, 2_000_000 // max(len(a), 1)))
    child_seeds = rng_seed.spawn(math.ceil(n / chunk))
    for ci, k in enumerate(range(0, n, chunk)):
        m = min(chunk, n - k)
        idx = stationary_bootstrap_indices(len(a), m, bl, int(child_seeds[ci].generate_state(1)[0]))
        sa, sb = a[idx], b[idx]
        out[k : k + m] = sa.mean(1) / sa.std(1, ddof=1) - sb.mean(1) / sb.std(1, ddof=1)
    return out
