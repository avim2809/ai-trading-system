"""Sharpe-ratio summary statistics: PSR, DSR, MinTRL (credibility plan P1-02).

Formulas follow Bailey and Lopez de Prado, "The Sharpe Ratio Efficient Frontier"
(2012) and "The Deflated Sharpe Ratio" (2014), and are numerically equivalent to
the legacy raw-array functions in ``firm.eval.overfitting`` wherever those are
well defined (``_norm_cdf`` / ``_norm_ppf`` are reused, not forked).

Conventions
-----------
* Every Sharpe argument is PER-PERIOD, in the same frequency as the returns the
  moments came from. Annualised Sharpe (``firm.eval.metrics.sharpe_ratio``) must
  never be passed in. ``check_per_period`` raises ``SharpeUnitsError`` when
  ``abs(sr) > 0.5`` (about 8 annualised on daily data) and logs a WARNING when
  ``0.25 < abs(sr) <= 0.5``. Weekly/monthly users can relax ``max_abs`` only by
  calling ``check_per_period`` themselves; the public functions use the default.
* Kurtosis is RAW (normal = 3). ``moments`` uses the legacy estimators:
  population skew and raw kurtosis divided by the ddof=1 standard deviation.
* No sentinels and no silent fallbacks (differences from the legacy code):
  ``n_obs < 2``, a non-positive variance-estimator radicand
  ``1 - skew*SR + (kurt-1)/4*SR^2``, ``n_trials < 2`` or ``var_sr <= 0`` raise
  ``DegenerateInputError``. Legacy clamped the radicand at 1e-12, returned 0.0
  for ``n < 8`` and degraded DSR to PSR(0) for a degenerate grid. ``n_obs < 8``
  only logs a WARNING (moments are unreliable).
* This module never chooses N or ``var_sr``: the gate config and the caller do.
  ``length_adjusted_var_sr`` only supplies the pure arithmetic.
* ``min_trl`` is in observations of the input frequency; convert to years outside.

This package is additive research code and must not be imported by any live
module (see ``tests/test_live_import_isolation.py``). Importing it loads
``firm.eval`` (pandas, sklearn) through that package's ``__init__``.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from firm.eval.overfitting import EULER_GAMMA, _norm_cdf, _norm_ppf

log = logging.getLogger(__name__)

__all__ = [
    "EULER_GAMMA",
    "DegenerateInputError",
    "SharpeUnitsError",
    "check_per_period",
    "dsr",
    "expected_max_sr",
    "length_adjusted_var_sr",
    "min_trl",
    "moments",
    "psr",
    "sr_sampling_var",
]

_WARN_ABS = 0.25
_MIN_RELIABLE_OBS = 8


class SharpeUnitsError(ValueError):
    """Raised when a Sharpe looks annualised where a per-period value is required."""


class DegenerateInputError(ValueError):
    """Raised on inputs for which the statistic is undefined."""


def check_per_period(sr: float, *, max_abs: float = 0.5, name: str = "sr") -> None:
    """Reject (or warn about) a Sharpe that looks annualised."""
    if not math.isfinite(sr):
        raise DegenerateInputError(f"{name} is not finite: {sr!r}")
    if abs(sr) > max_abs:
        raise SharpeUnitsError(
            f"{name}={sr:.4f} looks annualised (|sr| > {max_abs}); pass the per-period Sharpe"
        )
    if abs(sr) > _WARN_ABS:
        log.warning("%s=%.4f is large for a per-period Sharpe; check units", name, sr)


def _radicand(sr: float, skew: float, kurt: float) -> float:
    rad = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if not rad > 0.0:
        raise DegenerateInputError(
            f"non-positive SR variance radicand {rad!r} (sr={sr}, skew={skew}, kurt={kurt})"
        )
    return rad


def _check_n_obs(n_obs: int) -> None:
    if n_obs < 2:
        raise DegenerateInputError(f"n_obs must be >= 2, got {n_obs}")
    if n_obs < _MIN_RELIABLE_OBS:
        log.warning("n_obs=%d < %d: skew/kurtosis are unreliable", n_obs, _MIN_RELIABLE_OBS)


def moments(returns) -> tuple[float, float, float, int]:
    """Return ``(per-period SR, skew, raw kurtosis, n_obs)`` of finite returns.

    Same estimators as ``firm.eval.overfitting.probabilistic_sharpe``.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        raise DegenerateInputError(f"need at least 2 finite returns, got {n}")
    sd = float(r.std(ddof=1))
    mean = float(r.mean())
    if sd <= 1e-12 * abs(mean):  # constant series up to float rounding
        sd = 0.0
    if sd == 0.0:
        raise DegenerateInputError("zero-variance returns")
    m = r - mean
    sr = mean / sd
    skew = float((m**3).mean() / sd**3)
    kurt = float((m**4).mean() / sd**4)
    return sr, skew, kurt, n


def psr(sr: float, sr_star: float, n_obs: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: ``Phi((SR - SR*) / SE)``, per-period inputs."""
    check_per_period(sr, name="sr")
    check_per_period(sr_star, name="sr_star")
    _check_n_obs(n_obs)
    rad = _radicand(sr, skew, kurt)
    return _norm_cdf((sr - sr_star) * math.sqrt(n_obs - 1) / math.sqrt(rad))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """Expected maximum per-period Sharpe of ``n_trials`` null trials (B&LdP 2014)."""
    if n_trials < 2:
        raise DegenerateInputError(f"n_trials must be >= 2, got {n_trials}")
    if not var_sr > 0.0:
        raise DegenerateInputError(f"var_sr must be > 0, got {var_sr!r}")
    return math.sqrt(var_sr) * (
        (1.0 - EULER_GAMMA) * _norm_ppf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def dsr(sr: float, n_obs: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> float:
    """Deflated Sharpe Ratio: ``PSR(SR* = E[max SR])``. Raises on degenerate N/var."""
    return psr(sr, expected_max_sr(n_trials, var_sr), n_obs, skew, kurt)


def min_trl(sr: float, sr_star: float, skew: float, kurt: float, alpha: float = 0.05) -> float:
    """Minimum track-record length (observations) for ``PSR(SR*) >= 1 - alpha``.

    ``1 + (1 - skew*SR + (kurt-1)/4*SR^2) * (Phi^-1(1-alpha) / (SR - SR*))^2``.
    Undefined (raises) unless ``sr > sr_star``: the formula is symmetric in the
    sign of ``SR - SR*`` and would otherwise return a finite length for a
    strategy that is estimated to be worse than the benchmark.
    """
    check_per_period(sr, name="sr")
    check_per_period(sr_star, name="sr_star")
    if not 0.0 < alpha < 1.0:
        raise DegenerateInputError(f"alpha must be in (0, 1), got {alpha}")
    if not sr > sr_star:
        raise DegenerateInputError(f"min_trl needs sr > sr_star, got {sr} <= {sr_star}")
    rad = _radicand(sr, skew, kurt)
    return 1.0 + rad * (_norm_ppf(1.0 - alpha) / (sr - sr_star)) ** 2


def sr_sampling_var(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Lo/Mertens sampling variance of a per-period Sharpe estimate."""
    check_per_period(sr, name="sr")
    if n_obs < 2:
        raise DegenerateInputError(f"n_obs must be >= 2, got {n_obs}")
    return _radicand(sr, skew, kurt) / (n_obs - 1)


def length_adjusted_var_sr(
    srs, n_obs, n_obs_cand: int, *, skews=None, kurts=None, floor: float = 0.0
) -> float:
    """Length-adjusted cross-trial variance of Sharpe ratios (gates.yaml G-RESEARCH 1).

    ``max(floor, max(0, var(srs, ddof=1) - mean(sampling_var_i)) + sampling_var(mean(srs),
    n_obs_cand))``. ``n_obs`` is a scalar or one T_i per trial. Per-period inputs
    only. Choosing the family rows and the floor is the caller's job.
    """
    s = np.asarray(srs, dtype=float)
    k = len(s)
    if k < 2:
        raise DegenerateInputError(f"need at least 2 trials, got {k}")
    if not np.all(np.isfinite(s)):
        raise DegenerateInputError("non-finite trial Sharpe")
    ns = np.broadcast_to(np.asarray(n_obs), (k,)) if np.ndim(n_obs) == 0 else np.asarray(n_obs)
    sk = np.zeros(k) if skews is None else np.broadcast_to(np.asarray(skews, dtype=float), (k,))
    ku = np.full(k, 3.0) if kurts is None else np.broadcast_to(np.asarray(kurts, dtype=float), (k,))
    if len(ns) != k:
        raise DegenerateInputError("n_obs length does not match srs")
    samp = [sr_sampling_var(float(s[i]), int(ns[i]), float(sk[i]), float(ku[i])) for i in range(k)]
    excess = max(0.0, float(np.var(s, ddof=1)) - float(np.mean(samp)))
    cand = sr_sampling_var(float(s.mean()), int(n_obs_cand))
    return max(floor, excess + cand)
