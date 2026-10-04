"""Named stress-period suite (P3-07, source plan G-RESEARCH 5).

A pure reporting function over a returns series (plus an optional positions frame for
active-instrument counts). No optimiser, no parameter search: any change after viewing a
result is a new trial and a new prereg. All ten named periods are pre-seal and therefore
in-sample and burned; the suite is a reporting device, not an out-of-sample test.

Conventions:
- ``max_drawdown`` is a positive loss on the episode equity curve starting at 1.0 at the
  episode start (earlier equity does not enter).
- ``realised_vol = std(r, ddof=1) * sqrt(256)`` (same annualisation as P3-01).
- Active instruments are those with non-NaN positions on the FIRST day of the episode
  (the conservative reading: instruments that enter mid-window are not counted).
- ``reference_max_dd`` and ``min_active_fraction`` have no defaults (pre-registered).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

__all__ = ["EpisodeResult", "StressPeriod", "load_stress_periods", "run_stress_suite", "suite_summary"]

ANNUALISATION_DAYS = 256


@dataclass(frozen=True)
class StressPeriod:
    name: str
    start: date
    end: date
    note: str = ""


@dataclass
class EpisodeResult:
    name: str
    start: date
    end: date
    total_return: float
    max_drawdown: float
    worst_day: float
    realised_vol: float
    target_vol: float
    vol_ratio: float
    n_days: int
    n_active_instruments: int
    active_instruments: list[str] = field(default_factory=list)
    breach: bool = False
    status: str = "ok"  # "ok" | "unusable" | "not_applicable" | "low_coverage"


def load_stress_periods(path: str = "config/stress_periods.yaml") -> list[StressPeriod]:
    raw = yaml.safe_load(Path(path).read_text())
    out = [
        StressPeriod(str(p["name"]), p["start"], p["end"], str(p.get("note", "")))
        for p in raw["periods"]
    ]
    for p in out:
        if p.end < p.start:
            raise ValueError(f"stress period {p.name!r} ends before it starts")
    return out


def _max_drawdown(r: np.ndarray) -> float:
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def run_stress_suite(
    returns: pd.Series,
    positions: pd.DataFrame | None,
    periods: list[StressPeriod],
    target_vol: float,
    reference_max_dd: float | Callable[[int], float],
    breach_multiple: float = 1.5,
    min_days: int = 5,
    *,
    min_active_fraction: float,
) -> list[EpisodeResult]:
    """Score every named period; every period yields a visible row (never dropped).

    ``reference_max_dd`` is a positive loss, or a function of the episode's ``n_days``
    returning the episode-length bootstrap p95 (OD-16 split rule, P5-02).
    """
    idx = pd.DatetimeIndex(returns.index)
    results: list[EpisodeResult] = []
    for p in periods:
        mask = (idx >= pd.Timestamp(p.start)) & (idx <= pd.Timestamp(p.end))
        r_ser = returns[mask].dropna()
        r = r_ser.to_numpy(dtype=float)
        n = len(r)

        n_active, active = -1, []
        if positions is not None:
            pidx = pd.DatetimeIndex(positions.index)
            pwin = positions[(pidx >= pd.Timestamp(p.start)) & (pidx <= pd.Timestamp(p.end))]
            if len(pwin):
                first = pwin.iloc[0]
                active = [str(c) for c in positions.columns[first.notna().to_numpy()]]
            n_active = len(active)

        if n >= 1:
            total = float(np.prod(1.0 + r) - 1.0)
            mdd = _max_drawdown(r)
            worst = float(r.min())
        else:
            total = mdd = worst = math.nan
        vol = float(np.std(r, ddof=1) * math.sqrt(ANNUALISATION_DAYS)) if n >= 2 else math.nan
        ratio = vol / target_vol if target_vol > 0 and not math.isnan(vol) else math.nan

        if n < min_days:
            status = "unusable"
        elif positions is not None and n_active == 0:
            status = "not_applicable"
        elif positions is not None and n_active / max(len(positions.columns), 1) < min_active_fraction:
            status = "low_coverage"
        else:
            status = "ok"

        breach = False
        if n >= min_days:
            ref = reference_max_dd(n) if callable(reference_max_dd) else reference_max_dd
            breach = bool(mdd > breach_multiple * ref)
        if status != "ok":
            log.warning("stress episode %s status=%s n_days=%d n_active=%d", p.name, status, n, n_active)
        results.append(
            EpisodeResult(
                name=p.name, start=p.start, end=p.end, total_return=total, max_drawdown=mdd,
                worst_day=worst, realised_vol=vol, target_vol=target_vol, vol_ratio=ratio,
                n_days=n, n_active_instruments=n_active, active_instruments=active,
                breach=breach, status=status,
            )
        )
    return results


def suite_summary(results: list[EpisodeResult]) -> dict[str, Any]:
    scored = [r for r in results if not math.isnan(r.max_drawdown)]
    worst = max(scored, key=lambda r: r.max_drawdown) if scored else None
    n_breached = sum(r.breach for r in results)
    return {
        "worst_episode": worst.name if worst else None,
        "worst_max_drawdown": worst.max_drawdown if worst else None,
        "any_breach": n_breached > 0,
        "n_breached": n_breached,
        "n_unusable": sum(r.status == "unusable" for r in results),
        "n_not_applicable": sum(r.status == "not_applicable" for r in results),
        "n_low_coverage": sum(r.status == "low_coverage" for r in results),
        "all_ok": bool(results) and all(r.status == "ok" for r in results),
    }
