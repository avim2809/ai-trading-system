"""Real-capital allocation gate — automated status for the paper-track-record
promotion checklist documented in ``docs/PROJECT_CONTEXT.md`` (~L670-694).

That checklist has always been real policy, but until now nobody computed
it — an operator had to manually check duration, trade count, drawdown, etc.
against the live dashboard. ``compute_capital_gate`` is a pure function over
already-fetched engine state (no engine/broker dependency), so it's directly
unit-testable with synthetic data; ``firm.api.routers.live``'s
``GET /live/capital-gate`` wires it to a running engine.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from firm.contracts.models import PortfolioSnapshot
from firm.eval.metrics import max_drawdown
from firm.eval.robustness import MonteCarloAnalyzer

log = logging.getLogger(__name__)

# Promotion-gate thresholds (docs/PROJECT_CONTEXT.md ~L670-694).
MIN_TRADING_DAYS = 60
MIN_EXECUTED_ORDERS = 100
MAX_DRAWDOWN_LIMIT = 0.15
SHARPE_CI_CONFIDENCE = 0.90
MIN_OBS_FOR_SHARPE = 20


def _daily_nav_series(snapshots: list[PortfolioSnapshot]) -> pd.Series:
    """Collapse (possibly intra-day) snapshots to one NAV per calendar date.

    The engine snapshots the portfolio every cycle, which can run more than
    once a day (e.g. an ``hourly`` schedule) — this keeps the *last* NAV
    observed on each date so the criteria below reason over a genuine daily
    series, not double-counted intra-day noise.
    """
    if not snapshots:
        return pd.Series(dtype=float)
    by_date: dict[Any, float] = {}
    for snap in sorted(snapshots, key=lambda s: s.asof):
        by_date[snap.asof.date()] = snap.nav
    dates = sorted(by_date)
    return pd.Series([by_date[d] for d in dates], index=pd.DatetimeIndex(dates))


def compute_capital_gate(
    *,
    snapshots: list[PortfolioSnapshot],
    executed_order_count: int,
    alerts: list[dict[str, Any]],
    halted: bool,
    broker: str | None = None,
) -> dict[str, Any]:
    """Evaluate every real-capital allocation gate criterion.

    Pure computation — callers fetch ``snapshots``/``executed_order_count``/
    ``alerts``/``halted`` from a live engine (or synthesize them in tests).
    """
    nav_series = _daily_nav_series(snapshots)
    n_days = int(nav_series.size)
    returns = nav_series.pct_change().dropna()
    n_obs = int(returns.size)

    duration = {
        "label": "Duration",
        "threshold": f">= {MIN_TRADING_DAYS} trading days",
        "value": n_days,
        "passing": n_days >= MIN_TRADING_DAYS,
    }

    trade_count = {
        "label": "Trade count",
        "threshold": f">= {MIN_EXECUTED_ORDERS} executed",
        "value": executed_order_count,
        "passing": executed_order_count >= MIN_EXECUTED_ORDERS,
    }

    if n_obs < MIN_OBS_FOR_SHARPE:
        realized_sharpe = {
            "label": "Realized Sharpe (daily)",
            "threshold": f"{int(SHARPE_CI_CONFIDENCE * 100)}% CI lower bound > 0",
            "value": None,
            "passing": None,
            "n_observations": n_obs,
            "caveat": (
                f"insufficient data (<{MIN_OBS_FOR_SHARPE} daily return "
                "observations) to compute a meaningful bootstrap CI"
            ),
        }
    else:
        ci = MonteCarloAnalyzer(confidence=SHARPE_CI_CONFIDENCE, seed=42) \
            .sharpe_confidence_interval(returns)
        realized_sharpe = {
            "label": "Realized Sharpe (daily)",
            "threshold": f"{int(SHARPE_CI_CONFIDENCE * 100)}% CI lower bound > 0",
            "value": ci["lower_bound"],
            "passing": ci["lower_bound"] > 0,
            "point_estimate": ci["point_estimate"],
            "n_observations": n_obs,
        }

    dd = float(max_drawdown(returns))
    max_drawdown_criterion = {
        "label": "Max drawdown",
        "threshold": f"<= {int(MAX_DRAWDOWN_LIMIT * 100)}%",
        "value": dd,
        "passing": dd <= MAX_DRAWDOWN_LIMIT,
    }

    # Best-effort from in-memory alerts only — trip history isn't persisted
    # across restarts today (only the current halted/peak-equity blob is,
    # via state_store.py/engine.py), so this is explicitly non-durable.
    # A currently-halted engine is an unambiguous fail; otherwise a session
    # with zero observed trips passes, and one or more trips is marked
    # ambiguous (None) rather than a hard fail — this endpoint has no way to
    # tell an "explained"/reviewed trip from an unexplained one; durable
    # trip-history persistence (a separate, heavier follow-up) is what would
    # let this become a real unexplained-trip count.
    trips = sum(1 for a in alerts if a.get("kind") == "drawdown_breach")
    if halted:
        kill_switch_passing = False
    elif trips == 0:
        kill_switch_passing = True
    else:
        kill_switch_passing = None
    kill_switch_trips = {
        "label": "Kill-switch trips",
        "threshold": "0 unexplained auto trips",
        "value": trips,
        "passing": kill_switch_passing,
        "durable": False,
        "currently_halted": bool(halted),
        "caveat": "session-scoped, not persisted across restarts",
    }

    # Manual runbook item — never derivable from engine state.
    llm_ab = {"label": "LLM A/B", "passing": None, "applicable": False}

    criteria = {
        "duration": duration,
        "trade_count": trade_count,
        "realized_sharpe": realized_sharpe,
        "max_drawdown": max_drawdown_criterion,
        "kill_switch_trips": kill_switch_trips,
        "llm_ab": llm_ab,
    }
    gating_keys = (
        "duration", "trade_count", "realized_sharpe", "max_drawdown", "kill_switch_trips",
    )
    n_passing = sum(1 for k in gating_keys if criteria[k]["passing"] is True)
    blocking = [k for k in gating_keys if criteria[k]["passing"] is not True]

    return {
        "broker": broker,
        "overall_passing": not blocking,
        "n_passing": n_passing,
        "blocking": blocking,
        "criteria": criteria,
    }
