"""Gates config loader and the G-PAPER evaluator (ticket P5-01).

Every threshold is read from ``config/gates.yaml`` (``load_gates_config``); nothing numeric is hard-coded here except the
conventions noted below (252 trading days per year for the annualised tracking error; ``MIN_FIDELITY_OBS``).

Implementation choices recorded for the OD-16 register:

* G-PAPER 2 is evaluated on the trailing ``fidelity.window_trading_days`` rows of ``daily`` after excluding zero-exposure
  days (the count is reported); fewer than ``MIN_FIDELITY_OBS`` usable days gives ``insufficient_data``.
* Tracking error is the annualised standard deviation of (paper return - model return), compared with
  ``tracking_error_max_fraction_of_tau * tau``.
* G-PAPER 1 below either threshold is ``insufficient_data`` (the paper period is extended, never waived), not ``fail``.
* G-PAPER 3 needs the fills and instruments minimum before any verdict; then BOTH the median per-trade ratio and the
  aggregate ratio of realised to modelled cost (commission included in both) must be <= the multiple.
* G-PAPER 4: a position break is unreconciled for ``trading days in (opened, resolved or asof]``; more than the maximum fails.
  Any missed review not marked as a documented outage fails (allowed = 0).
* G-PAPER 5: every required drill name must be present and true.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from firm.allocation.calendar import is_us_trading_day

log = logging.getLogger(__name__)

__all__ = [
    "REQUIRED_DRILLS",
    "Fill",
    "GPaperResult",
    "GatesConfig",
    "GatesConfigError",
    "PositionBreak",
    "ReviewEvent",
    "add_months",
    "evaluate_gpaper",
    "load_gates_config",
]

Verdict = Literal["pass", "fail", "insufficient_data"]
MIN_FIDELITY_OBS = 21  # implementation choice: about a third of the 63-day window
TRADING_DAYS_PER_YEAR = 252
# G-PAPER 5: kill-switch tiers (P4-04) and the decommission combiner (G-DECOMMISSION) must each have a fault-drill record.
REQUIRED_DRILLS = ("soft", "halt", "hard", "decommission_combiner")


class GatesConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GatesConfig:
    raw: Mapping[str, Any]

    def get(self, *path: str) -> Any:
        node: Any = self.raw
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                raise GatesConfigError(f"gates config is missing {'.'.join(path)}")
            node = node[key]
        return node

    @property
    def paper_eligible_tiers(self) -> tuple[str, ...]:
        return tuple(self.get("verdict", "paper_eligible_tiers"))


def load_gates_config(path: str | Path = "config/gates.yaml") -> GatesConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise GatesConfigError(f"{path}: not a mapping")
    return GatesConfig(raw)


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    for day in range(min(d.day, 31), 0, -1):
        try:
            return date(year, month, day)
        except ValueError:
            continue
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class ReviewEvent:
    review_date: date
    held: bool = True  # the scheduled review took place (traded or not)
    missed: bool = False
    outage_documented: bool = False


@dataclass(frozen=True)
class Fill:
    trade_date: date
    symbol: str
    realised_cost: float  # commission + slippage, currency or bps (same unit as modelled_cost)
    modelled_cost: float


@dataclass(frozen=True)
class PositionBreak:
    opened: date
    resolved: date | None = None


@dataclass(frozen=True)
class GPaperResult:
    overall: Verdict
    criteria: dict[str, Verdict]
    details: dict[str, Any] = field(default_factory=dict)


def _trading_days_between(start: date, end: date) -> int:
    n, d = 0, start + timedelta(days=1)
    while d <= end:
        if is_us_trading_day(d):
            n += 1
        d += timedelta(days=1)
    return n


def _crit1(events: Sequence[ReviewEvent], gates: GatesConfig) -> tuple[Verdict, dict]:
    need = int(gates.get("g_paper", "min_review_events"))
    months = int(gates.get("g_paper", "min_calendar_months"))
    held = sorted(e.review_date for e in events if e.held and not e.missed)
    det: dict[str, Any] = {"events": len(held), "min_events": need, "min_months": months}
    if not held:
        return "insufficient_data", det
    span_ok = add_months(held[0], months) <= held[-1]
    det["first"], det["last"] = held[0].isoformat(), held[-1].isoformat()
    return ("pass" if len(held) >= need and span_ok else "insufficient_data"), det


def _crit2(daily: pd.DataFrame, tau: float, gates: GatesConfig) -> tuple[Verdict, dict]:
    cfg = gates.get("g_paper", "fidelity")
    window = int(cfg["window_trading_days"])
    corr_min = float(cfg["daily_return_corr_min"])
    te_max = float(cfg["tracking_error_max_fraction_of_tau"]) * tau
    tail = daily.tail(window)
    active = tail[tail["exposure"].abs() > 0]
    det: dict[str, Any] = {
        "zero_exposure_days_excluded": int(len(tail) - len(active)),
        "obs": len(active),
        "corr_min": corr_min,
        "te_max": te_max,
    }
    if len(active) < MIN_FIDELITY_OBS:
        return "insufficient_data", det
    paper, model = active["ret_paper"].astype(float), active["ret_model"].astype(float)
    if paper.std() == 0 or model.std() == 0:
        return "insufficient_data", det
    corr = float(np.corrcoef(paper, model)[0, 1])
    te = float((paper - model).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    det.update(corr=corr, tracking_error=te)
    return ("pass" if corr >= corr_min and te <= te_max else "fail"), det


def _crit3(fills: Sequence[Fill], gates: GatesConfig) -> tuple[Verdict, dict]:
    g = gates.get("g_paper")
    min_fills, min_inst = int(g["cost_test_min_fills"]), int(g["cost_test_min_instruments"])
    mult = float(g["realised_cost_max_multiple_of_modelled"])
    inst = {f.symbol for f in fills}
    det: dict[str, Any] = {"fills": len(fills), "instruments": len(inst), "max_multiple": mult}
    if len(fills) < min_fills or len(inst) < min_inst:
        return "insufficient_data", det
    modelled = np.array([f.modelled_cost for f in fills], dtype=float)
    realised = np.array([f.realised_cost for f in fills], dtype=float)
    if not (modelled > 0).all():
        return "insufficient_data", det  # a ratio is undefined; the cost model must price every fill
    agg = float(realised.sum() / modelled.sum())
    med = float(np.median(realised / modelled))
    det.update(aggregate_ratio=agg, median_ratio=med)
    return ("pass" if agg <= mult and med <= mult else "fail"), det


def _crit4(
    events: Sequence[ReviewEvent], breaks: Sequence[PositionBreak], asof: date, gates: GatesConfig
) -> tuple[Verdict, dict]:
    g = gates.get("g_paper")
    max_days = int(g["position_breaks"]["max_unreconciled_trading_days"])
    allowed_missed = int(g["missed_reviews"]["allowed"])
    long_breaks = [
        b.opened.isoformat()
        for b in breaks
        if _trading_days_between(b.opened, b.resolved or asof) > max_days
    ]
    missed = [e.review_date.isoformat() for e in events if e.missed and not e.outage_documented]
    det = {"long_breaks": long_breaks, "unexplained_missed_reviews": missed}
    return ("fail" if long_breaks or len(missed) > allowed_missed else "pass"), det


def _crit5(drills: Mapping[str, bool] | None) -> tuple[Verdict, dict]:
    drills = drills or {}
    missing = [d for d in REQUIRED_DRILLS if not drills.get(d)]
    return ("fail" if missing else "pass"), {"missing_drills": missing}


def evaluate_gpaper(
    events: Sequence[ReviewEvent],
    fills: Sequence[Fill],
    daily: pd.DataFrame,
    gates: GatesConfig,
    *,
    tau: float,
    position_breaks: Sequence[PositionBreak] = (),
    fault_drills: Mapping[str, bool] | None = None,
    asof: date | None = None,
) -> GPaperResult:
    """Evaluate the five G-PAPER criteria. ``daily`` needs columns ``ret_paper``, ``ret_model``, ``exposure``."""
    if not tau > 0:
        raise ValueError("tau must be > 0")
    asof = asof or max((e.review_date for e in events), default=date.min)
    c1, d1 = _crit1(events, gates)
    c2, d2 = _crit2(daily, tau, gates)
    c3, d3 = _crit3(fills, gates)
    c4, d4 = _crit4(events, position_breaks, asof, gates)
    c5, d5 = _crit5(fault_drills)
    crit: dict[str, Verdict] = {"1_period": c1, "2_fidelity": c2, "3_cost": c3, "4_breaks_reviews": c4, "5_drills": c5}
    vals = set(crit.values())
    overall: Verdict = "fail" if "fail" in vals else ("insufficient_data" if "insufficient_data" in vals else "pass")
    log.info("G-PAPER verdict=%s criteria=%s", overall, crit)
    return GPaperResult(overall, crit, {"1": d1, "2": d2, "3": d3, "4": d4, "5": d5})
