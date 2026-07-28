"""Overfitting statistics: PBO (CSCV), Deflated & Probabilistic Sharpe.

Clean-room implementations of the public Bailey / López de Prado papers
("The Probability of Backtest Overfitting" and "The Deflated Sharpe Ratio").
No scipy — the normal CDF and its inverse are computed by hand, so this module
depends only on numpy and the standard library.

The three checks answer one question: is a backtest's edge real, or an artefact
of trying many configurations?

- :func:`cscv_pbo` — feed the full grid of per-period trial returns (one column
  per parameter/strategy config) and it estimates how often the best in-sample
  config drops below the out-of-sample median (combinatorially-symmetric CV).
- :func:`deflated_sharpe` — the chosen strategy's returns plus every trial's
  Sharpe → probability the true Sharpe is above zero *after* penalising for the
  number of trials.
- :func:`probabilistic_sharpe` — the same probability without the trial penalty.

Ported from the external trading-suite ``pbo-deflated-sharpe`` skill; the six
core functions are used verbatim, adapted only for firm's package layout.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations

import numpy as np

log = logging.getLogger(__name__)

# Euler-Mascheroni constant used by the deflated-Sharpe expected-maximum estimate.
EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam's inverse normal CDF approximation."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [
        -39.69683028665376,
        220.9460984245205,
        -275.9285104469687,
        138.3577518672690,
        -30.66479806614716,
        2.506628277459239,
    ]
    b = [
        -54.47609879822406,
        161.5858368580409,
        -155.6989798598866,
        66.80131188771972,
        -13.28068155288572,
    ]
    c = [
        -0.007784894002430293,
        -0.3223964580411365,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _sharpe_cols(matrix: np.ndarray) -> np.ndarray:
    """Per-column (non-annualised) Sharpe of a returns matrix."""
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)


def cscv_pbo(matrix: np.ndarray, n_partitions: int = 8) -> float:
    """Probability of Backtest Overfitting via combinatorially-symmetric CV.

    Parameters
    ----------
    matrix:
        ``T`` periods (rows) x ``N`` trials (columns) of per-period returns.
    n_partitions:
        Number of balanced blocks to split the timeline into (``S`` in the
        paper). Needs ``T >= 2 * n_partitions`` rows and ``N >= 2`` columns.

    Returns
    -------
    float in ``[0, 1]`` — the fraction of balanced splits where the best
    in-sample config lands below the out-of-sample median. ``>= 0.5`` = overfit.
    Returns ``0.5`` (uninformative) when the input is too small.
    """
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("matrix must be 2-D (periods x trials)")
    T, N = M.shape
    S = n_partitions
    if N < 2 or T < 2 * S:
        return 0.5
    rows = T // S
    M = M[: rows * S]
    chunks = [M[k * rows : (k + 1) * rows] for k in range(S)]
    allset = set(range(S))
    lambdas = []
    for comb in combinations(range(S), S // 2):
        IS = np.vstack([chunks[k] for k in comb])
        OOS = np.vstack([chunks[k] for k in allset - set(comb)])
        nstar = int(np.argmax(_sharpe_cols(IS)))
        oos = _sharpe_cols(OOS)
        order = oos.argsort()
        ranks = np.empty(N)
        ranks[order] = np.arange(1, N + 1)
        w = min(max(ranks[nstar] / (N + 1), 1e-6), 1 - 1e-6)
        lambdas.append(math.log(w / (1 - w)))
    return float((np.array(lambdas) <= 0).mean())


def probabilistic_sharpe(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true Sharpe > ``sr_benchmark``).

    Corrects a raw Sharpe for track-record length and for skew / fat tails.
    Needs at least 8 observations.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 8:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = r.mean() / sd
    m = r - r.mean()
    skew = float((m**3).mean() / sd**3)
    kurt = float((m**4).mean() / sd**4)  # non-excess
    denom = math.sqrt(max(1e-12, 1 - skew * sr + ((kurt - 1) / 4) * sr**2))
    return _norm_cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom)


def deflated_sharpe(returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """Deflated Sharpe Ratio: PSR against a trials-aware benchmark.

    ``trial_sharpes`` is the Sharpe of every configuration that was tried. The
    more trials (and the wider their spread), the higher the benchmark and the
    harder the DSR is to pass — that is the point.
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    n_trials = max(len(trials), 1)
    var_sr = float(np.var(trials, ddof=1)) if n_trials > 1 else 0.0
    if var_sr <= 0 or n_trials < 2:
        return probabilistic_sharpe(returns, 0.0)
    sr_star = math.sqrt(var_sr) * (
        (1 - EULER_GAMMA) * _norm_ppf(1 - 1 / n_trials)
        + EULER_GAMMA * _norm_ppf(1 - 1 / (n_trials * math.e))
    )
    return probabilistic_sharpe(returns, sr_star)


def verdict(pbo: float | None = None, dsr: float | None = None) -> dict:
    """Turn raw PBO / DSR numbers into a plain-English pass/fail read.

    ``PBO < 0.50`` → not overfit (the winning config still wins out-of-sample
    more often than not). ``DSR > 0.95`` → the Sharpe is very likely real after
    accounting for how many configurations were tried. The overall verdict is
    ``pass`` only when every check that was run passes.
    """
    lines: list[str] = []
    checks: list[bool] = []

    if pbo is not None:
        pbo_pass = pbo < 0.50
        checks.append(pbo_pass)
        lines.append(
            f"PBO {pbo * 100:.0f}% — "
            + (
                "PASS. The best-looking config keeps winning out-of-sample more "
                "often than not, so the backtest is probably not just curve-fit."
                if pbo_pass
                else "FAIL. The best-looking config flips to below-median "
                "out-of-sample most of the time — this looks overfit. Treat the "
                "headline result as noise."
            )
        )

    if dsr is not None:
        dsr_strong = dsr > 0.95
        checks.append(dsr_strong)
        lines.append(
            f"DSR {dsr:.2f} — "
            + (
                "STRONG. After deflating for the number of trials, there is a "
                f"{dsr * 100:.0f}% chance the true Sharpe is above zero."
                if dsr_strong
                else "WEAK. After deflating for the number of trials, the chance "
                f"the true Sharpe is above zero is only {dsr * 100:.0f}%. The edge "
                "may be a product of trying many configurations."
            )
        )

    overall = "pass" if checks and all(checks) else "fail"
    return {"verdict": overall, "lines": lines}


def _period_sharpe(returns: np.ndarray) -> float:
    """Non-annualised per-period Sharpe (mean/std), matching PSR's internal sr."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd)


def walk_forward_overfitting(
    fold_returns: list[np.ndarray],
    fold_trial_returns: list[list[np.ndarray]] | None = None,
) -> dict:
    """Overfitting read across walk-forward out-of-sample folds.

    Each element of ``fold_returns`` is one fold's OOS (test-window) per-period
    return series for the config that was actually selected and traded that
    fold — this always feeds ``probabilistic_sharpe`` (a plain PSR read of the
    realised OOS track record needs no notion of "trials" at all).

    ``fold_trial_returns``, when given, is the *genuine* trial data: for each
    fold, the per-period **train-window** return series of every candidate
    configuration that was actually backtested and compared before selecting
    the winner (e.g. a walk-forward parameter grid). This is what makes
    ``pbo``/``deflated_sharpe`` meaningful in the Bailey/López de Prado sense —
    CSCV and the trial-count penalty are both defined over *competing
    configurations evaluated on the same period*, not over sequential
    out-of-sample folds of a single fixed config. Consequently:

    - ``pbo`` is the mean, across folds with >=2 candidates and enough
      train-window rows, of a CSCV-PBO computed on that fold's own
      (candidates x train-periods) returns matrix. Omitted entirely when no
      fold has genuine multi-candidate trial data — a fabricated PBO from
      folds-as-trials would silently misrepresent what was actually tested.
    - ``deflated_sharpe`` benchmarks the pooled OOS Sharpe against every
      candidate's train Sharpe, across every fold — the true count and spread
      of configurations tried. With no genuine trial data (``fold_trial_returns``
      omitted, or every fold ran a single candidate), there is truly only one
      trial and DSR correctly degrades to plain PSR rather than penalising for
      trials that were never tried.
    - ``verdict`` — plain-English pass/fail combining PBO (when available) and DSR.

    Returns an empty dict when there is not enough OOS data to say anything.
    """
    series = [np.asarray(f, dtype=float) for f in fold_returns]
    series = [s[np.isfinite(s)] for s in series if len(s) >= 2]
    log.debug(
        "walk_forward_overfitting: %d/%d folds usable",
        len(series), len(fold_returns),
    )
    if len(series) < 2:
        log.info(
            "walk_forward_overfitting: insufficient data (%d usable folds); "
            "skipping overfitting stats",
            len(series),
        )
        return {}

    pooled = np.concatenate(series)

    trial_sharpes: list[float] = []
    fold_pbos: list[float] = []
    n_trial_folds = 0
    for fold_trials in (fold_trial_returns or []):
        trial_series = [np.asarray(t, dtype=float) for t in fold_trials]
        trial_series = [t[np.isfinite(t)] for t in trial_series if len(t) >= 2]
        if not trial_series:
            continue
        n_trial_folds += 1
        # Every candidate genuinely backtested on this fold's train window is
        # a real trial and counts toward DSR's penalty, even a lone one (no
        # selection took place, but it was still one configuration tried).
        trial_sharpes.extend(_period_sharpe(t) for t in trial_series)
        # PBO strictly needs >=2 competing candidates to rank against each
        # other within a CSCV split.
        if len(trial_series) < 2:
            continue
        min_len = min(len(t) for t in trial_series)
        if min_len < 8:
            continue
        n_partitions = max(4, min(8, min_len // 2))
        matrix = np.column_stack([t[:min_len] for t in trial_series])
        fold_pbos.append(cscv_pbo(matrix, n_partitions=n_partitions))

    result: dict = {
        "n_folds": len(series),
        "probabilistic_sharpe": probabilistic_sharpe(pooled, 0.0),
        "deflated_sharpe": deflated_sharpe(
            pooled, np.array(trial_sharpes, dtype=float)
        ),
    }
    if fold_pbos:
        result["pbo"] = float(np.mean(fold_pbos))
        result["pbo_n_folds"] = len(fold_pbos)
    log.debug(
        "walk_forward_overfitting: %d/%d folds had genuine multi-candidate "
        "trial data (%d contributed a within-fold PBO); %d trial Sharpes feed DSR",
        n_trial_folds, len(fold_returns), len(fold_pbos), len(trial_sharpes),
    )

    result["verdict"] = verdict(
        pbo=result.get("pbo"), dsr=result.get("deflated_sharpe")
    )["verdict"]
    log.info(
        "walk_forward_overfitting: n_folds=%d psr=%.3f dsr=%.3f pbo=%s verdict=%s",
        result["n_folds"], result["probabilistic_sharpe"],
        result["deflated_sharpe"],
        f"{result['pbo']:.3f}" if "pbo" in result else "n/a",
        result["verdict"],
    )
    return result
