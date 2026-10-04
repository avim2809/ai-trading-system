"""Probability of Backtest Overfitting via CSCV (credibility plan P1-04).

Bailey, Borwein, Lopez de Prado and Zhu (2017), "The Probability of Backtest
Overfitting". Research-only; the legacy ``firm.eval.overfitting.cscv_pbo`` is
left untouched (frozen scripts call it with S=8).

Deliberate differences from the legacy function:

* ties are ranked with AVERAGE ranks (legacy uses ``argsort`` order);
* ties in the in-sample argmax go to the lowest column index;
* no 0.5 sentinel: ``N < 2`` or ``T < 2*S`` raises ``ValueError``; ``N < 4``
  sets ``uninformative=True`` (the gate must not accept it as a pass);
* tail policy: ``truncate_tail`` drops the LAST ``T % S`` rows (same as legacy)
  and reports ``rows_dropped``; ``raise`` refuses a non-divisible ``T``.
  Rows are never shuffled: blocks are contiguous time slices.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from scipy.stats import rankdata

_CLAMP = 1e-6


def sharpe_cols(M: np.ndarray) -> np.ndarray:
    """Per-period (non-annualised) Sharpe per column, ddof=1; sd == 0 -> 0.0."""
    M = np.asarray(M, dtype=float)
    mean = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)


@dataclass(frozen=True)
class PBOResult:
    pbo: float  # share of splits with lambda <= 0
    logits: np.ndarray  # lambda per split
    perf_degradation_slope: float  # OLS slope of OOS metric on IS metric of n*
    perf_degradation_r2: float
    prob_oos_loss: float  # share of splits where n*'s OOS metric < 0
    n_splits: int
    n_blocks: int  # S
    n_trials: int  # N
    rows_used: int  # T after tail policy
    rows_dropped: int
    uninformative: bool  # N < 4 (T < 2*S raises)


def _fast_sharpe_all(bs: np.ndarray, bss: np.ndarray, n: int, sel: np.ndarray) -> np.ndarray:
    """Sharpe (ddof=1) for each combination of blocks from per-block sums; ``n`` rows per combo."""
    s = bs[sel].sum(axis=1)
    ss = bss[sel].sum(axis=1)
    mean = s / n
    var = (ss - n * mean**2) / (n - 1)
    # cancellation floor: constant columns have exactly zero variance
    var = np.where(var <= 1e-14 * ss / n, 0.0, var)
    sd = np.sqrt(var)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)


def pbo(
    M: np.ndarray,
    S: int = 16,
    metric: Callable[[np.ndarray], np.ndarray] = sharpe_cols,
    tail: Literal["truncate_tail", "raise"] = "truncate_tail",
    embargo_pct: float = 0.0,
) -> PBOResult:
    """CSCV probability of backtest overfitting.

    ``M`` is ``T x N`` (periods x trials) of per-period returns in time order.
    ``metric`` maps a ``(rows, N)`` array to ``(N,)`` scores (default Sharpe).
    Algorithm: S contiguous blocks; for each of C(S, S/2) combinations IS = those
    blocks, OOS = the rest; ``n* = argmax IS metric`` (lowest index on ties);
    ``omega = avg_rank(OOS metric of n*) / (N+1)`` clipped to [1e-6, 1-1e-6];
    ``lambda = ln(omega/(1-omega))``; ``PBO = P(lambda <= 0)``.
    ``embargo_pct`` trims OOS rows adjacent to IS blocks, as in the legacy function.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError("M must be 2-D (periods x trials)")
    if S < 2 or S % 2:
        raise ValueError("S must be an even integer >= 2")
    if not 0.0 <= embargo_pct < 1.0:
        raise ValueError("embargo_pct must be in [0, 1)")
    if tail not in ("truncate_tail", "raise"):
        raise ValueError(f"unknown tail policy {tail!r}")
    T, N = M.shape
    if N < 2:
        raise ValueError(f"need N >= 2 trials, got {N}")
    if T < 2 * S:
        raise ValueError(f"need T >= 2*S rows, got T={T}, S={S}")
    if not np.all(np.isfinite(M)):
        raise ValueError("M contains non-finite values")
    rows = T // S
    dropped = T - rows * S
    if dropped and tail == "raise":
        raise ValueError(f"T={T} not divisible by S={S}; use tail='truncate_tail'")
    M = M[: rows * S]
    combs = np.array(list(combinations(range(S), S // 2)), dtype=np.intp)
    embargo_rows = math.ceil(embargo_pct * rows) if embargo_pct > 0 else 0
    n_comb = len(combs)

    if metric is sharpe_cols and embargo_rows == 0:
        blocks = M.reshape(S, rows, N)
        bs = blocks.sum(axis=1)
        bss = (blocks**2).sum(axis=1)
        in_mask = np.zeros((n_comb, S), dtype=bool)
        np.put_along_axis(in_mask, combs, True, axis=1)
        oos_sel = np.array([np.flatnonzero(~r) for r in in_mask], dtype=np.intp)
        n_half = rows * (S // 2)
        is_m = _fast_sharpe_all(bs, bss, n_half, combs)
        oos_m = _fast_sharpe_all(bs, bss, n_half, oos_sel)
    else:
        chunks = [M[k * rows : (k + 1) * rows] for k in range(S)]
        is_list, oos_list = [], []
        for comb in combs:
            cs = {int(c) for c in comb}
            oos_blocks = []
            for k in range(S):
                if k in cs:
                    continue
                block = chunks[k]
                if embargo_rows:
                    start = embargo_rows if (k - 1) in cs else 0
                    end = embargo_rows if (k + 1) in cs else 0
                    block = (
                        block[start : len(block) - end] if start + end < len(block) else block[0:0]
                    )
                if len(block):
                    oos_blocks.append(block)
            if not oos_blocks:
                continue  # embargo purged the whole OOS set; skip as legacy does
            is_list.append(
                np.asarray(metric(np.vstack([chunks[int(k)] for k in comb])), dtype=float)
            )
            oos_list.append(np.asarray(metric(np.vstack(oos_blocks)), dtype=float))
        if not is_list:
            raise ValueError("embargo_pct purged every out-of-sample set")
        is_m, oos_m = np.vstack(is_list), np.vstack(oos_list)
        if is_m.shape[1] != N:
            raise ValueError("metric must return one score per column")

    nstar = np.argmax(is_m, axis=1)  # first max -> lowest index on ties
    ranks = rankdata(oos_m, axis=1, method="average")
    idx = np.arange(len(nstar))
    omega = np.clip(ranks[idx, nstar] / (N + 1), _CLAMP, 1 - _CLAMP)
    logits = np.log(omega / (1 - omega))
    is_star, oos_star = is_m[idx, nstar], oos_m[idx, nstar]

    sx = is_star - is_star.mean()
    sxx = float((sx**2).sum())
    if sxx > 0:
        slope = float((sx * (oos_star - oos_star.mean())).sum() / sxx)
        syy = float(((oos_star - oos_star.mean()) ** 2).sum())
        r2 = float((sx * (oos_star - oos_star.mean())).sum() ** 2 / (sxx * syy)) if syy > 0 else 0.0
    else:
        slope, r2 = 0.0, 0.0

    return PBOResult(
        pbo=float((logits <= 0).mean()),
        logits=logits,
        perf_degradation_slope=slope,
        perf_degradation_r2=r2,
        prob_oos_loss=float((oos_star < 0).mean()),
        n_splits=len(logits),
        n_blocks=S,
        n_trials=N,
        rows_used=rows * S,
        rows_dropped=dropped,
        uninformative=N < 4,
    )
