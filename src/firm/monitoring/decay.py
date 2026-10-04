"""CUSUM decay monitor (credibility plan P5-03).

Lower-side Page (1954) CUSUM detecting a drop of the per-period mean of a
standardised excess-return series from ``mu0`` to ``mu1``::

    z_t = (x_t - mu0) / sigma
    S_0 = 0,  S_t = max(0, S_{t-1} - z_t - k),  k = (mu0 - mu1) / (2 sigma)
    alarm when S_t > h

Pure module: no I/O, no live wiring (P5-04 / P5-01 consume it). Alerts are the
caller's job and must read ``ALERT_WEBHOOK_URL`` from ``os.environ`` only.

Units
-----
``mu0``, ``mu1`` and ``sigma`` are per-period. ``mu / sigma`` is a per-period
Sharpe and is passed through ``firm.validation.sharpe_stats.check_per_period``,
so an annualised Sharpe raises ``SharpeUnitsError``.

Calibration procedure (``calibrate_h``)
---------------------------------------
``h`` is never hard-coded. It is found by bisection so that the MEAN run length
to first alarm of in-control series equals ``arl0_periods`` (not the median: run
lengths are roughly geometric and the median is about 0.7 x the mean). Each
bisection step re-simulates ``n_sims`` paths from the same ``seed`` (common random
numbers, so the run length is monotone in ``h``); the Siegmund (1985) approximation
only seeds the bracket. The in-control series is iid N(0, 1) in standardised
units by default, or, when ``sample`` is supplied, a Politis-Romano stationary
bootstrap of that sample (demeaned, scaled by ``sigma``) with mean block length
``mean_block``. With fat tails and autocorrelation the normal-theory ARL is
optimistic, so ``calibrate_h_gate`` returns the larger of the two ``h``.
``calibrate_h_detailed`` records the seed, simulation size, achieved mean ARL
and its standard error (``CalibrationResult``). Bisection on fixed draws hits the
target by construction, so verify on an independent seed (see tests). Runs not
alarmed by ``max_periods`` are censored at that value (negligible for the default
60 x ARL0 under the geometric tail).

Behaviour to know
-----------------
* An alarm is latched: ``cusum_step`` keeps ``alarm`` and ``first_alarm_index``
  after it fires, and there is no reset function. Clearing is an owner approval
  in P5-01.
* ``first_alarm_index`` is the 0-based index of the observation that fired.
* CUSUM is a slow decommission trigger by construction: at daily scale with
  mu0 = SR 0.25 (after haircut), mu1 = SR -0.3 and ARL0 = 1260, the mean detection
  delay from a zero start is about 2.5 years (the 36-month t-test is only a
  secondary check and is never used alone).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from firm.validation.sharpe_stats import check_per_period

__all__ = [
    "CalibrationResult",
    "CusumConfig",
    "CusumState",
    "calibrate_h",
    "calibrate_h_detailed",
    "calibrate_h_gate",
    "cusum_run",
    "cusum_step",
    "siegmund_arl",
    "simulate_run_lengths",
    "t_test_36m",
]

_SIEGMUND_SHIFT = 1.166  # continuous-to-discrete correction for unit-variance normal steps


@dataclass(frozen=True)
class CusumConfig:
    mu0: float  # expected per-period mean (after >= 50% haircut, from the charter)
    mu1: float  # alternative to detect (e.g. 0 = edge gone)
    sigma: float  # per-period vol of the series
    h: float  # decision threshold (from calibrate_h)
    k: float | None = None  # reference value in sigma units; default (mu0 - mu1) / (2 sigma)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.sigma) and self.sigma > 0.0):
            raise ValueError(f"sigma must be finite and > 0, got {self.sigma!r}")
        if not (math.isfinite(self.mu0) and math.isfinite(self.mu1)):
            raise ValueError("mu0 and mu1 must be finite")
        check_per_period(self.mu0 / self.sigma, name="mu0/sigma")
        check_per_period(self.mu1 / self.sigma, name="mu1/sigma")
        if not self.mu1 < self.mu0:
            raise ValueError(f"decay monitor needs mu1 < mu0, got mu1={self.mu1}, mu0={self.mu0}")
        if not (math.isfinite(self.h) and self.h >= 0.0):
            raise ValueError(f"h must be finite and >= 0, got {self.h!r}")
        if self.k is not None and not (math.isfinite(self.k) and self.k >= 0.0):
            raise ValueError(f"k must be finite and >= 0, got {self.k!r}")

    @property
    def k_eff(self) -> float:
        return (self.mu0 - self.mu1) / (2.0 * self.sigma) if self.k is None else self.k


@dataclass(frozen=True)
class CusumState:
    s: float
    n: int
    alarm: bool
    first_alarm_index: int | None


@dataclass(frozen=True)
class CalibrationResult:
    h: float
    arl0_target: int
    arl_achieved: float  # mean run length on the calibration draws
    arl_se: float  # its standard error
    seed: int
    n_sims: int
    method: str  # "normal" | "stationary_bootstrap"
    k: float
    max_periods: int


def cusum_step(state: CusumState, x: float, cfg: CusumConfig) -> CusumState:
    """Advance one observation. A fired alarm stays latched."""
    if not math.isfinite(x):
        raise ValueError(f"non-finite observation: {x!r}")
    z = (x - cfg.mu0) / cfg.sigma
    s = max(0.0, state.s - z - cfg.k_eff)
    n = state.n + 1
    if state.alarm:
        return CusumState(s, n, True, state.first_alarm_index)
    if s > cfg.h:
        return CusumState(s, n, True, n - 1)
    return CusumState(s, n, False, None)


def cusum_run(x: Sequence[float], cfg: CusumConfig, state: CusumState | None = None) -> CusumState:
    """Run the CUSUM over ``x`` (streaming ``cusum_step`` applied in order)."""
    st = state if state is not None else CusumState(0.0, 0, False, None)
    for v in x:
        st = cusum_step(st, float(v), cfg)
    return st


def t_test_36m(monthly_excess: Sequence[float], mu0: float) -> tuple[float, float]:
    """Rolling last-36-month one-sided t-test, H1: mean < ``mu0``. Returns ``(t, p)``.

    Secondary check only. Needs at least 36 finite months (raises otherwise);
    ``mu0`` is in the units of the monthly series.
    """
    from scipy import stats

    arr = np.asarray(monthly_excess, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite monthly excess return")
    if len(arr) < 36:
        raise ValueError(f"need at least 36 months, got {len(arr)}")
    last = arr[-36:]
    sd = float(last.std(ddof=1))
    if sd == 0.0:
        raise ValueError("zero-variance monthly series")
    t = (float(last.mean()) - mu0) / (sd / 6.0)
    return float(t), float(stats.t.cdf(t, df=35))


def siegmund_arl(cfg: CusumConfig, *, true_mean: float | None = None) -> float:
    """Siegmund (1985) ARL approximation for iid normal data with the given true mean.

    ``true_mean`` defaults to ``cfg.mu0`` (in control). Zero-start ARL.
    """
    m = cfg.mu0 if true_mean is None else true_mean
    drift = (cfg.mu0 - m) / cfg.sigma - cfg.k_eff  # mean increment of S per period
    return _siegmund(cfg.h, cfg.k_eff, drift)


def _siegmund(h: float, k: float, drift: float) -> float:
    b = h + _SIEGMUND_SHIFT
    if abs(drift) < 1e-12:
        return b * b
    return (math.exp(-2.0 * drift * b) + 2.0 * drift * b - 1.0) / (2.0 * drift**2)


def _siegmund_h(k: float, arl0: float) -> float:
    lo, hi = 0.0, 1.0
    while _siegmund(hi, k, -k) < arl0 and hi < 1e4:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _siegmund(mid, k, -k) < arl0:
            lo = mid
        else:
            hi = mid
    return hi


def simulate_run_lengths(
    cfg: CusumConfig,
    *,
    n_sims: int,
    seed: int,
    true_mean: float | None = None,
    sample: Sequence[float] | None = None,
    mean_block: int = 20,
    max_periods: int = 100_000,
) -> np.ndarray:
    """Simulate ``n_sims`` run lengths to first alarm (1 = alarm on the first observation).

    ``true_mean`` is the true per-period mean (default ``cfg.mu0``: in control).
    Runs with no alarm by ``max_periods`` are censored at ``max_periods``.
    """
    if n_sims < 1 or max_periods < 1:
        raise ValueError("n_sims and max_periods must be >= 1")
    rng = np.random.default_rng(seed)
    m = cfg.mu0 if true_mean is None else true_mean
    shift = (m - cfg.mu0) / cfg.sigma
    k = cfg.k_eff
    resid = None
    if sample is not None:
        arr = np.asarray(sample, dtype=float)
        if len(arr) < 2 or not np.all(np.isfinite(arr)):
            raise ValueError("bootstrap sample needs >= 2 finite values")
        resid = (arr - arr.mean()) / cfg.sigma
        p_restart = 1.0 / max(mean_block, 1)
        idx = rng.integers(0, len(resid), n_sims)
    s = np.zeros(n_sims)
    rl = np.full(n_sims, max_periods, dtype=np.int64)
    alive = np.ones(n_sims, dtype=bool)
    for t in range(1, max_periods + 1):
        if resid is None:
            z = rng.standard_normal(n_sims) + shift
        else:
            z = resid[idx] + shift
            restart = rng.random(n_sims) < p_restart
            idx = np.where(restart, rng.integers(0, len(resid), n_sims), (idx + 1) % len(resid))
        s = np.maximum(0.0, s - z - k)
        hit = alive & (s > cfg.h)
        if hit.any():
            rl[hit] = t
            alive &= ~hit
            if not alive.any():
                break
    return rl


def _cfg_with_h(cfg: CusumConfig, h: float) -> CusumConfig:
    return CusumConfig(mu0=cfg.mu0, mu1=cfg.mu1, sigma=cfg.sigma, h=h, k=cfg.k)


def calibrate_h_detailed(
    cfg_without_h: CusumConfig,
    *,
    arl0_periods: int,
    n_sims: int = 5000,
    seed: int,
    block: str = "stationary",
    sample: Sequence[float] | None = None,
    mean_block: int = 20,
    max_periods: int | None = None,
) -> CalibrationResult:
    """Bisect ``h`` so the MEAN in-control run length equals ``arl0_periods``.

    ``cfg_without_h.h`` is ignored. ``block`` names the bootstrap scheme
    (``"stationary"``, the only one implemented) and only matters when ``sample`` is
    given; without a sample the baseline iid-normal series is used.
    """
    if block != "stationary":
        raise ValueError(f"unsupported block scheme {block!r}")
    if arl0_periods < 2:
        raise ValueError("arl0_periods must be >= 2")
    cap = max_periods if max_periods is not None else 60 * arl0_periods
    k = cfg_without_h.k_eff

    def mean_rl(h: float) -> tuple[float, float]:
        rl = simulate_run_lengths(
            _cfg_with_h(cfg_without_h, h),
            n_sims=n_sims,
            seed=seed,
            sample=sample,
            mean_block=mean_block,
            max_periods=cap,
        )
        se = float(rl.std(ddof=1) / math.sqrt(len(rl))) if len(rl) > 1 else float("nan")
        return float(rl.mean()), se

    guess = _siegmund_h(k, float(arl0_periods))
    lo, hi = 0.5 * guess, 1.5 * guess
    for _ in range(40):  # widen the bracket until it straddles the target
        if mean_rl(lo)[0] <= arl0_periods or lo == 0.0:
            break
        lo = 0.5 * lo if lo > 1e-6 else 0.0
    for _ in range(40):
        if mean_rl(hi)[0] >= arl0_periods:
            break
        hi *= 1.5
    else:
        raise RuntimeError("could not bracket h; ARL0 unreachable within max_periods")
    for _ in range(60):
        if hi - lo <= 1e-3 * hi:
            break
        mid = 0.5 * (lo + hi)
        if mean_rl(mid)[0] < arl0_periods:
            lo = mid
        else:
            hi = mid
    arl, se = mean_rl(hi)
    return CalibrationResult(
        h=hi,
        arl0_target=arl0_periods,
        arl_achieved=arl,
        arl_se=se,
        seed=seed,
        n_sims=n_sims,
        method="normal" if sample is None else "stationary_bootstrap",
        k=k,
        max_periods=cap,
    )


def calibrate_h(
    cfg_without_h: CusumConfig,
    *,
    arl0_periods: int,
    n_sims: int = 5000,
    seed: int,
    block: str = "stationary",
    sample: Sequence[float] | None = None,
    mean_block: int = 20,
) -> float:
    """Threshold ``h`` for the target mean in-control run length (see module docstring)."""
    return calibrate_h_detailed(
        cfg_without_h,
        arl0_periods=arl0_periods,
        n_sims=n_sims,
        seed=seed,
        block=block,
        sample=sample,
        mean_block=mean_block,
    ).h


def calibrate_h_gate(
    cfg_without_h: CusumConfig,
    *,
    arl0_periods: int,
    sample: Sequence[float],
    n_sims: int = 5000,
    seed: int,
    mean_block: int = 20,
) -> CalibrationResult:
    """Larger (more conservative) of the normal-theory and bootstrap-calibrated ``h``."""
    normal = calibrate_h_detailed(
        cfg_without_h, arl0_periods=arl0_periods, n_sims=n_sims, seed=seed
    )
    boot = calibrate_h_detailed(
        cfg_without_h,
        arl0_periods=arl0_periods,
        n_sims=n_sims,
        seed=seed,
        sample=sample,
        mean_block=mean_block,
    )
    return boot if boot.h > normal.h else normal
