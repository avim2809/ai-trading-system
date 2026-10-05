"""Tiered drawdown and operational kill switch (ticket P4-04; candidate-only, pure functions).

NOT wired into any live path: ``firm.live``, ``firm.api`` and ``firm.runtime`` never import it (P0-06 isolation test lists
``firm.risk``). The legacy mechanisms (``LiveTradingEngine._check_drawdown`` with the IBKR 8% / Alpaca 25% thresholds and
``RiskAgent._drawdown_breaker``) are untouched and are consulted only as inputs (``engine_halted``, ``breaker_active``).
Engine integration is opt-in at P6-01. The module returns alert dicts; it emits nothing and performs no I/O in ``evaluate``.

Precedence (highest first): HARD > HALT > SOFT > OK.

* SOFT: drawdown >= ``soft_mult * survival_ref``. Alert only (``Action.NONE``, exposure factor 1.0).
* HALT: operational fault (stale data, broker disconnect inside the rebalance window, position break).
  ``Action.BLOCK_NEW_ORDERS``; never liquidates.
* HARD: drawdown >= ``hard_mult * survival_ref``. ``Action.BLOCK_NEW_ORDERS``; latched until an audited ``record_reset``.
  FLATTEN is emitted only when ``cfg.flatten_on_hard`` is set (needs an owner-signed OD-16 rule) and never while stale data
  or a broker disconnect is active.

``survival_ref = max(bootstrap p95, 2.5 tau)`` (OD-16 split rule); the 2.5 tau figure is a plan rule of thumb.
Drawdown is measured on a cash-flow-adjusted unit-value series (``unit_value``), never raw NAV. The immutable high-water mark
(``dd_immutable``, feeds decommission) is never rewritten by a reset; a reset records an audit entry and sets a re-arm
baseline from which the tier is measured. A NAV correction excludes a bad observation date (audited, nothing deleted).

Implementation choices recorded for the OD-16 register (P0-08):

* The P2-03 special-closures table does not exist yet; ``SPECIAL_CLOSURES`` below is a provisional stand-in (``evaluate``
  accepts an override). A closure missing from it false-triggers stale data (documented by a test).
* HARD is latched in ``KillSwitchState.halted`` (like the engine's persisted halt); only ``record_reset`` clears it, or a
  ``record_nav_correction`` of the very date that set the HWM (the trip was derived from a bad observation; it re-trips if
  real).
* Boundary comparisons carry a 1e-12 tolerance so a path crossing exactly 1.0x / 1.5x trips despite float error.
* Alerts fire when the tier changes to a non-OK tier (escalations always; engine_halted does not suppress them).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from firm.allocation.calendar import is_us_trading_day

log = logging.getLogger(__name__)

__all__ = [
    "SPECIAL_CLOSURES",
    "Action",
    "Decision",
    "KillSwitchConfig",
    "KillSwitchState",
    "Tier",
    "drawdown",
    "evaluate",
    "load_kill_switch_config",
    "record_nav_correction",
    "record_reset",
    "unit_value",
]

_TOL = 1e-12

# Provisional special NYSE closures absent from firm.allocation.calendar (stand-in for the P2-03 table).
SPECIAL_CLOSURES: frozenset[date] = frozenset(
    {
        date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),  # 9/11
        date(2004, 6, 11),  # Reagan mourning
        date(2007, 1, 2),  # Ford mourning
        date(2012, 10, 29), date(2012, 10, 30),  # Sandy
        date(2018, 12, 5),  # Bush mourning
        date(2025, 1, 9),  # Carter mourning
    }
)


class Tier(str, Enum):
    OK = "ok"
    SOFT = "soft"
    HALT = "halt"
    HARD = "hard"


class Action(str, Enum):
    NONE = "none"
    BLOCK_NEW_ORDERS = "block_new_orders"
    FLATTEN = "flatten"


_RANK = {Tier.OK: 0, Tier.SOFT: 1, Tier.HALT: 2, Tier.HARD: 3}
_ALERT_KIND = {Tier.SOFT: "kill_soft_warning", Tier.HALT: "kill_stale_data", Tier.HARD: "kill_hard_stop"}


@dataclass(frozen=True)
class KillSwitchConfig:
    charter_dd_bootstrap_p95: float  # ledger-derived artefact, not risk.yaml
    tau: float
    soft_mult: float = 1.0
    hard_mult: float = 1.5
    stale_data_days: int = 2  # trading days
    flatten_on_hard: bool = False  # needs an owner-signed OD-16 rule

    def __post_init__(self) -> None:
        if not self.charter_dd_bootstrap_p95 >= 0:
            raise ValueError("charter_dd_bootstrap_p95 must be >= 0")
        if not self.tau > 0:
            raise ValueError("tau must be > 0")
        if not 0 < self.soft_mult <= self.hard_mult:
            raise ValueError("require 0 < soft_mult <= hard_mult")
        if self.stale_data_days < 0:
            raise ValueError("stale_data_days must be >= 0")

    @property
    def survival_ref(self) -> float:
        return max(self.charter_dd_bootstrap_p95, 2.5 * self.tau)


@dataclass(frozen=True)
class KillSwitchState:
    hwm: float
    hwm_date: date | None
    halted: bool
    tier: Tier
    resume_baseline: float | None
    resume_hwm: float | None
    resets: tuple[dict, ...]
    nav_corrections: tuple[dict, ...]

    @classmethod
    def new(cls) -> KillSwitchState:
        return cls(0.0, None, False, Tier.OK, None, None, (), ())


@dataclass(frozen=True)
class Decision:
    tier: Tier
    action: Action
    triggers: tuple[str, ...]
    exposure_factor: float
    already_halted: bool
    dd_immutable: float
    dd_from_baseline: float
    alerts: tuple[dict, ...]
    state: KillSwitchState


def unit_value(returns: pd.Series) -> pd.Series:
    """Time-weighted unit value from daily returns (cash flows are not returns, so they cannot move it)."""
    return (1.0 + returns.astype(float)).cumprod()


def drawdown(unit_values: pd.Series, hwm: float) -> float:
    """Drawdown of the last observation versus ``max(hwm, series max)``."""
    peak = max(float(hwm), float(unit_values.max()))
    return (peak - float(unit_values.iloc[-1])) / peak


def _to_date(ts: Any) -> date:
    return pd.Timestamp(ts).date()


def _trading_days_since(last: date, today: date, closures: frozenset[date]) -> int:
    n, d = 0, last + timedelta(days=1)
    while d <= today:
        if is_us_trading_day(d) and d not in closures:
            n += 1
        d += timedelta(days=1)
    return n


def evaluate(
    unit_values: pd.Series,
    state: KillSwitchState,
    cfg: KillSwitchConfig,
    *,
    last_bar_date: Mapping[str, date],
    today: date,
    broker_connected: bool,
    in_rebalance_window: bool,
    position_breaks: Sequence[Mapping],
    engine_halted: bool = False,
    breaker_active: bool = False,
    special_closures: frozenset[date] | None = None,
) -> Decision:
    closures = SPECIAL_CLOSURES if special_closures is None else special_closures
    bad_dates = {c["bad_date"] for c in state.nav_corrections}
    keep = [_to_date(i) not in bad_dates for i in unit_values.index]
    u = unit_values[keep]
    if u.empty:
        raise ValueError("unit_values has no usable observations")
    last_u = float(u.iloc[-1])

    prior = state.hwm
    if state.hwm_date is not None and state.hwm_date in bad_dates:
        prior = 0.0  # the stored HWM came from an audited-bad observation
    hwm = max(prior, float(u.max()))
    hwm_date = state.hwm_date if hwm == prior and prior > 0 else _to_date(u.idxmax())
    dd_immutable = (hwm - last_u) / hwm

    resume_hwm = state.resume_hwm
    if state.resets and resume_hwm is not None:
        since = state.resets[-1]["at"].date()
        post = u[[_to_date(i) > since for i in u.index]]
        resume_hwm = max(resume_hwm, float(post.max())) if len(post) else resume_hwm
        resume_hwm = max(resume_hwm, last_u) if len(post) else resume_hwm
        dd_base = (resume_hwm - last_u) / resume_hwm
    else:
        dd_base = dd_immutable

    soft_ref, hard_ref = cfg.soft_mult * cfg.survival_ref, cfg.hard_mult * cfg.survival_ref
    triggers: list[str] = []
    tier = Tier.OK
    if state.halted or dd_base >= hard_ref - _TOL:
        tier = Tier.HARD
        triggers.append("dd_hard")
    elif dd_base >= soft_ref - _TOL:
        tier = Tier.SOFT
        triggers.append("dd_soft")

    stale = [
        f"stale_data:{sym}"
        for sym, last in sorted(last_bar_date.items())
        if _trading_days_since(last, today, closures) > cfg.stale_data_days
    ]
    disconnect = in_rebalance_window and not broker_connected
    ops = bool(stale) or disconnect or bool(position_breaks)
    triggers += stale
    if disconnect:
        triggers.append("broker_disconnect")
    if position_breaks:
        triggers.append("position_break")
    if ops and _RANK[tier] < _RANK[Tier.HALT]:
        tier = Tier.HALT
    if engine_halted:
        triggers.append("engine_halted")
    if breaker_active:
        triggers.append("breaker_active")

    action, exposure = Action.NONE, 1.0
    if tier in (Tier.HALT, Tier.HARD):
        action = Action.BLOCK_NEW_ORDERS
    if tier is Tier.HARD and cfg.flatten_on_hard and not (stale or disconnect):
        action, exposure = Action.FLATTEN, 0.0

    alerts: tuple[dict, ...] = ()
    if tier is not state.tier and tier is not Tier.OK:
        kind = _ALERT_KIND[tier]
        if tier is Tier.HALT and not stale:
            kind = "kill_operational_fault"
        alerts = (
            {
                "kind": kind,
                "severity": "critical" if tier in (Tier.HALT, Tier.HARD) else "warning",
                "tier": tier.value,
                "triggers": tuple(triggers),
                "dd_immutable": dd_immutable,
                "dd_from_baseline": dd_base,
                "survival_ref": cfg.survival_ref,
                "date": today.isoformat(),
            },
        )

    new_state = replace(
        state,
        hwm=hwm,
        hwm_date=hwm_date,
        halted=tier is Tier.HARD,
        tier=tier,
        resume_hwm=resume_hwm,
    )
    decision = Decision(
        tier, action, tuple(triggers), exposure, engine_halted or breaker_active,
        dd_immutable, dd_base, alerts, new_state,
    )
    if tier is not Tier.OK:
        log.warning(
            "kill_switch %s action=%s triggers=%s dd_immutable=%.4f dd_from_baseline=%.4f survival_ref=%.4f",
            tier.value, action.value, list(triggers), dd_immutable, dd_base, cfg.survival_ref,
        )
    return decision


def _require(approver: str, reason: str) -> None:
    if not approver or not approver.strip():
        raise ValueError("approver is required")
    if not reason or not reason.strip():
        raise ValueError("reason is required")


def record_reset(
    state: KillSwitchState, *, approver: str, reason: str, at: datetime, current_unit_value: float
) -> KillSwitchState:
    """Audited, approved re-arm. Leaves the immutable HWM untouched; the tier is measured from the new baseline."""
    _require(approver, reason)
    if not current_unit_value > 0:
        raise ValueError("current_unit_value must be > 0")
    rec = {
        "approver": approver, "reason": reason, "at": at, "unit_value": float(current_unit_value),
        "tier_before": state.tier.value, "hwm": state.hwm,
    }
    log.warning("kill_switch reset by %s at %s baseline=%.6f reason=%s", approver, at, current_unit_value, reason)
    return replace(
        state, halted=False, tier=Tier.OK, resume_baseline=float(current_unit_value),
        resume_hwm=float(current_unit_value), resets=(*state.resets, rec),
    )


def record_nav_correction(
    state: KillSwitchState, *, bad_date: date, approver: str, reason: str, at: datetime
) -> KillSwitchState:
    """Audited exclusion of a bad observation date (data is never deleted)."""
    _require(approver, reason)
    rec = {"bad_date": bad_date, "approver": approver, "reason": reason, "at": at}
    log.warning("kill_switch nav correction %s by %s: %s", bad_date, approver, reason)
    out = replace(state, nav_corrections=(*state.nav_corrections, rec))
    if state.hwm_date == bad_date:  # the trip, if any, rested on the bad HWM; evaluate re-trips if the loss is real
        out = replace(out, halted=False, tier=Tier.OK)
    return out


def load_kill_switch_config(path: str | Path, *, allow_null: bool = False) -> KillSwitchConfig | None:
    """Load the candidate config; raises while ``tau`` / ``charter_dd_bootstrap_p95`` are null (``allow_null`` -> None)."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if raw.get("tau") is None or raw.get("charter_dd_bootstrap_p95") is None:
        if allow_null:
            return None
        raise ValueError(f"{path}: tau and charter_dd_bootstrap_p95 must be set (owner/charter-derived)")
    keys = ("charter_dd_bootstrap_p95", "tau", "soft_mult", "hard_mult", "stale_data_days", "flatten_on_hard")
    return KillSwitchConfig(**{k: raw[k] for k in keys if k in raw})
