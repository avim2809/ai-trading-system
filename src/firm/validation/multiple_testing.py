"""White Reality Check, Hansen SPA and Romano-Wolf stepdown (credibility plan P1-06).

References: White (2000) Econometrica 68(5); Hansen (2005) JBES 23(4);
Romano and Wolf (2005) Econometrica 73(4); Politis and Romano (1994) stationary bootstrap.

SIGN CONVENTION: ``perf_diffs`` is a ``T x K`` array of per-period performance
differentials ``d[t, k] = f[t, k] - f_bench[t]`` where LARGER IS BETTER (loss = -performance).
The null is ``E[d_k] <= 0`` for every k; a column of strictly positive differentials is
significant, its negation is not.

Resampling uses ONLY ``firm.validation.bootstrap`` (stationary bootstrap, automatic
Politis-White block length = max over columns). An iid bootstrap is never used
implicitly; ``block_len=1`` is allowed only when passed explicitly and logs a WARNING.
Size/power acceptance thresholds live in ``config/gates.yaml`` and are exercised by P1-08,
not here.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from firm.validation.bootstrap import politis_white_block_length, stationary_bootstrap_indices

logger = logging.getLogger(__name__)

_OMEGA_FLOOR = 1e-12
_CHUNK_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class RCResult:
    p_value: float
    t_max: float
    best_index: int
    block_len: float
    n_boot: int


@dataclass(frozen=True)
class SPAResult:
    p_lower: float
    p_consistent: float
    p_upper: float
    t_max: float
    best_index: int
    block_len: float


@dataclass(frozen=True)
class _Boot:
    T: int
    dbar: np.ndarray  # (K,)
    dstar: np.ndarray  # (B, K) bootstrap means
    omega: np.ndarray  # (K,) std of sqrt(T)*dstar, floored
    valid: np.ndarray  # (K,) bool: column has non-zero bootstrap variance
    block_len: float

    @property
    def t(self) -> np.ndarray:
        return np.where(self.valid, math.sqrt(self.T) * self.dbar / self.omega, 0.0)

    def tstar(self, mu: np.ndarray | None = None) -> np.ndarray:
        """(B, K) studentised bootstrap stats ``sqrt(T)(d*-dbar+mu)/omega`` (0 for zero-variance columns)."""
        z = self.dstar - self.dbar
        if mu is not None:
            z = z + mu
        return np.where(self.valid, math.sqrt(self.T) * z / self.omega, 0.0)


def _validate(perf_diffs: np.ndarray) -> np.ndarray:
    d = np.asarray(perf_diffs, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    if d.ndim != 2:
        raise ValueError("perf_diffs must be T x K")
    if d.shape[0] < 3 or d.shape[1] < 1:
        raise ValueError(f"need T >= 3 rows and K >= 1 columns, got {d.shape}")
    if not np.all(np.isfinite(d)):
        raise ValueError("perf_diffs contains non-finite values")
    return d


def _prepare(
    perf_diffs: np.ndarray,
    B: int,
    block_len: float | None,
    seed: int,
    indices: np.ndarray | None = None,
) -> _Boot:
    d = _validate(perf_diffs)
    T, K = d.shape
    if block_len is None:
        bl = politis_white_block_length(d)
    else:
        bl = float(block_len)
        if bl <= 1.0:
            logger.warning("block_len=%s is an iid bootstrap; invalid for autocorrelated data", bl)
    if indices is None:
        idx = stationary_bootstrap_indices(T, B, bl, seed)
    else:
        idx = np.asarray(indices)
        if idx.ndim != 2 or idx.shape[1] != T:
            raise ValueError("indices must be (B, T)")
        B = idx.shape[0]
    dstar = np.empty((B, K))
    chunk = max(1, int(_CHUNK_BYTES // (T * K * 8)))
    for s in range(0, B, chunk):
        dstar[s : s + chunk] = d[idx[s : s + chunk]].mean(axis=1)
    omega_raw = (math.sqrt(T) * dstar).std(axis=0, ddof=1) if B > 1 else np.zeros(K)
    valid = omega_raw > _OMEGA_FLOOR
    return _Boot(T, d.mean(axis=0), dstar, np.maximum(omega_raw, _OMEGA_FLOOR), valid, bl)


def reality_check(
    perf_diffs: np.ndarray,
    B: int = 2000,
    block_len: float | None = None,
    seed: int = 0,
    *,
    studentise: bool = True,
    _indices: np.ndarray | None = None,
) -> RCResult:
    """White (2000) Reality Check: does the best of K beat the benchmark after the search?

    ``V = max_k t_k`` (``studentise=True``, default) or White's raw ``max_k sqrt(T) dbar_k``;
    bootstrap ``V*_b = max_k sqrt(T)(dbar*_kb - dbar_k)/omega_k`` (recentred at the sample mean);
    ``p = (1 + #{V*_b >= V}) / (1 + B)``. Conservative when many alternatives are poor (see SPA).
    """
    bt = _prepare(perf_diffs, B, block_len, seed, _indices)
    root_t = math.sqrt(bt.T)
    if studentise:
        stat, boot = bt.t, bt.tstar()
    else:
        stat = root_t * bt.dbar
        boot = root_t * (bt.dstar - bt.dbar)
    best = int(np.argmax(stat))
    v = float(stat[best])
    nb = boot.shape[0]
    p = (1 + int((boot.max(axis=1) >= v).sum())) / (1 + nb)
    return RCResult(p_value=float(p), t_max=v, best_index=best, block_len=bt.block_len, n_boot=nb)


def spa_test(
    perf_diffs: np.ndarray,
    B: int = 2000,
    block_len: float | None = None,
    seed: int = 0,
    *,
    _indices: np.ndarray | None = None,
) -> SPAResult:
    """Hansen (2005) Superior Predictive Ability test (studentised, with recentring).

    ``T_SPA = max(0, max_k t_k)``. Recentring ``mu_k``: lower ``min(dbar_k, 0)``;
    consistent ``dbar_k * 1{t_k <= -sqrt(2 log log T)}``; upper ``0`` (equals RC).
    ``T*_b = max(0, max_k sqrt(T)(dbar*_kb - dbar_k + mu_k)/omega_k)``,
    ``p = (1 + #{T*_b > T_SPA}) / (1 + B)``. ``p_consistent`` is the headline.
    """
    bt = _prepare(perf_diffs, B, block_len, seed, _indices)
    t = bt.t
    best = int(np.argmax(t))
    t_spa = max(0.0, float(t[best]))
    thresh = -math.sqrt(2.0 * math.log(math.log(max(bt.T, 3))))
    mus = {
        "lower": np.minimum(bt.dbar, 0.0),
        "consistent": np.where(t <= thresh, bt.dbar, 0.0),
        "upper": np.zeros_like(bt.dbar),
    }
    nb = bt.dstar.shape[0]
    p = {}
    for name, mu in mus.items():
        tstar = np.maximum(0.0, bt.tstar(mu).max(axis=1))
        p[name] = (1 + int((tstar > t_spa).sum())) / (1 + nb)
    return SPAResult(
        p_lower=float(p["lower"]),
        p_consistent=float(p["consistent"]),
        p_upper=float(p["upper"]),
        t_max=t_spa,
        best_index=best,
        block_len=bt.block_len,
    )


def romano_wolf(
    perf_diffs: np.ndarray,
    names: Sequence[str] | None = None,
    alpha: float = 0.05,
    B: int = 2000,
    block_len: float | None = None,
    seed: int = 0,
    *,
    two_sided: bool = False,
    _indices: np.ndarray | None = None,
) -> pd.DataFrame:
    """Romano-Wolf (2005) stepdown on max t; strong FWER control at ``alpha``.

    One-sided by default (performance positive; sort by signed ``t`` descending);
    ``two_sided=True`` sorts and tests ``|t|``. Both ``t_k`` and the bootstrap
    ``t*_kb = sqrt(T)(dbar*_kb - dbar_k)/omega_k`` use the same full-sample ``omega_k``.
    Step j: ``c_j`` = (1-alpha) quantile of ``max_{k remaining} t*_kb``; reject (j) if
    ``t_(j) > c_j`` else stop. ``p_adj,(j) = max(p_adj,(j-1), (1 + #{max_remaining t*_b >= t_(j)})/(1+B))``.
    Returns columns ``strategy, t_stat, p_adj, reject`` in input column order.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    bt = _prepare(perf_diffs, B, block_len, seed, _indices)
    K = bt.dbar.shape[0]
    if names is None:
        names = [f"s{k}" for k in range(K)]
    if len(names) != K:
        raise ValueError("names length must equal number of columns")
    t = bt.t
    ts = bt.tstar()
    if two_sided:
        t_eff, ts_eff = np.abs(t), np.abs(ts)
    else:
        t_eff, ts_eff = t, ts
    order = np.argsort(-t_eff, kind="stable")
    nb = ts.shape[0]
    p_adj = np.empty(K)
    reject = np.zeros(K, dtype=bool)
    running, stopped = 0.0, False
    for j, k in enumerate(order):
        rem = order[j:]
        mx = ts_eff[:, rem].max(axis=1)
        p_j = (1 + int((mx >= t_eff[k]).sum())) / (1 + nb)
        running = max(running, p_j)
        p_adj[k] = running
        if not stopped:
            crit = float(np.quantile(mx, 1.0 - alpha, method="higher"))
            if t_eff[k] > crit and bt.valid[k]:
                reject[k] = True
            else:
                stopped = True
    return pd.DataFrame({"strategy": list(names), "t_stat": t, "p_adj": p_adj, "reject": reject})
