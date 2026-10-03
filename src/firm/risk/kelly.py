"""Fractional-Kelly sanity bound on target volatility (credibility plan P4-05).

This is a SANITY BOUND, not a sizing rule: the charter's ex-ante target
volatility ``tau`` must not exceed ``kelly_fraction`` (default 0.5) times the
Kelly-implied volatility of a haircut planning Sharpe. It must never be used to
size positions or to raise ``tau``, and it is unrelated to the pipeline's
``allocation_method: kelly`` (``TraderAgent._kelly``), which is neither used nor
imported here.

For a Gaussian continuous-time strategy the growth-optimal (full-Kelly)
portfolio volatility equals the annual Sharpe ratio (leverage f* = S / sigma).
With ``S_plan = (1 - haircut) * max(SR_backtest_net_annual, 0)``::

    max_tau = kelly_fraction * S_plan

Example: SR_backtest 0.4, haircut 0.5 -> S_plan 0.2, max_tau 0.10, so a 12-15%
target volatility fails for a modest Sharpe.

Selection bias is controlled separately by DSR >= 0.95 (G-RESEARCH 1) and is
deliberately NOT stacked here: no ``expected_max_sr`` subtraction on top of the
haircut.

Units: Sharpe arguments are ANNUALISED net backtest Sharpes (hence
``sharpe_annual``). A value tagged ``firm.validation.sharpe_stats.PerPeriodSharpe``
or with ``|SR| > 10`` raises ``SharpeUnitsError``. An untagged small per-period
number cannot be detected, so callers must convert (or tag) it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from firm.validation.sharpe_stats import PerPeriodSharpe, SharpeUnitsError

__all__ = ["KellyCheck", "check_tau", "kelly_vol", "max_tau", "planning_sharpe"]

_MAX_ANNUAL_ABS = 10.0
CHARTER_MIN_HAIRCUT = 0.5


def _check_annual(sr: float, name: str) -> None:
    if isinstance(sr, PerPeriodSharpe):
        raise SharpeUnitsError(f"{name} is tagged per-period; Kelly bound needs annual Sharpe")
    if not math.isfinite(sr):
        raise ValueError(f"{name} is not finite: {sr!r}")
    if abs(sr) > _MAX_ANNUAL_ABS:
        raise SharpeUnitsError(f"{name}={sr} is implausible as an annual Sharpe")


def _check_haircut(haircut: float, min_haircut: float) -> None:
    if not 0.0 <= min_haircut < 1.0:
        raise ValueError(f"min_haircut must be in [0, 1), got {min_haircut!r}")
    if not (0.0 <= haircut < 1.0):  # also rejects NaN
        raise ValueError(f"haircut must be in [0, 1), got {haircut!r}")
    if haircut < min_haircut:
        raise ValueError(f"haircut {haircut} is below the charter minimum {min_haircut}")


def _check_fraction(kelly_fraction: float) -> None:
    if not 0.0 < kelly_fraction <= 1.0:
        raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction!r}")


def planning_sharpe(
    sharpe_backtest_annual: float, haircut: float, *, min_haircut: float = CHARTER_MIN_HAIRCUT
) -> float:
    """``(1 - haircut) * max(sharpe_backtest_annual, 0)``; haircut in [min_haircut, 1.0)."""
    _check_annual(sharpe_backtest_annual, "sharpe_backtest_annual")
    _check_haircut(haircut, min_haircut)
    return (1.0 - haircut) * max(sharpe_backtest_annual, 0.0)


def kelly_vol(sharpe_planning_annual: float) -> float:
    """Full-Kelly portfolio volatility (annual) = planning Sharpe (continuous-time, Gaussian)."""
    _check_annual(sharpe_planning_annual, "sharpe_planning_annual")
    return max(sharpe_planning_annual, 0.0)


def max_tau(
    sharpe_backtest_annual: float,
    haircut: float,
    kelly_fraction: float = 0.5,
    *,
    min_haircut: float = CHARTER_MIN_HAIRCUT,
) -> float:
    """``kelly_fraction * kelly_vol(planning_sharpe(...))``."""
    _check_fraction(kelly_fraction)
    s = planning_sharpe(sharpe_backtest_annual, haircut, min_haircut=min_haircut)
    return kelly_fraction * kelly_vol(s)


@dataclass(frozen=True)
class KellyCheck:
    tau: float
    max_tau: float
    ok: bool
    planning_sharpe: float


def check_tau(
    tau: float,
    sharpe_backtest_annual: float,
    haircut: float,
    kelly_fraction: float = 0.5,
    *,
    min_haircut: float = CHARTER_MIN_HAIRCUT,
) -> KellyCheck:
    """Is ``tau <= max_tau`` (inclusive)? A no-edge candidate only passes with tau == 0."""
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError(f"tau must be finite and >= 0, got {tau!r}")
    s = planning_sharpe(sharpe_backtest_annual, haircut, min_haircut=min_haircut)
    bound = max_tau(sharpe_backtest_annual, haircut, kelly_fraction, min_haircut=min_haircut)
    return KellyCheck(tau=tau, max_tau=bound, ok=tau <= bound, planning_sharpe=s)
