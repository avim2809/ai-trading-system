"""Wiring test: ``ctx.strategy_returns`` actually drives optimal combination.

The optimal (inverse-covariance) signal combination is only useful if the
per-strategy return history reaches the researchers. These tests exercise the
full ``AgentContext -> net_scores_for_blackboard`` path (what the bull/bear
researchers call) and confirm that:

  * with history present + ``method='optimal'`` the optimal weighting is used;
  * without history (``strategy_returns=None``) it degrades to the
    confidence-weighted mean — i.e. enabling ``optimal`` is a safe no-op until
    history is wired in.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from firm.agents.base import AgentContext
from firm.agents.blackboard import Blackboard
from firm.agents.research._combine import net_scores_for_blackboard
from firm.contracts.models import Signal, SignalSet

NOW = datetime(2026, 1, 2)
OPTIMAL = {"signal_combination": {"method": "optimal"}}


def _bb() -> Blackboard:
    """One symbol, two redundant bullish strategies + one independent bearish."""
    signals = [
        Signal("AAPL", "momentum", 2.0, 1.0, "21d", NOW),
        Signal("AAPL", "trend", 2.0, 1.0, "21d", NOW),  # redundant w/ momentum
        Signal("AAPL", "mean_reversion", -1.0, 1.0, "21d", NOW),  # independent
    ]
    bb = Blackboard(asof=NOW)
    bb.signal_sets.append(SignalSet(domain="technical", asof=NOW, signals=signals))
    return bb


def _history() -> dict[str, pd.Series]:
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 0.01, 200))  # momentum
    return {
        "momentum": base,
        "trend": base.copy(),  # perfectly correlated → should share weight
        "mean_reversion": pd.Series(rng.normal(0, 0.01, 200)),  # independent
    }


def test_optimal_used_when_history_present():
    bb = _bb()
    ctx = AgentContext(now=NOW, strategy_returns=_history())
    scores = net_scores_for_blackboard(bb, ctx, OPTIMAL)

    # Confidence-weighted mean would be (2 + 2 - 1) / 3 = 1.0. Optimal weighting
    # makes momentum+trend share one "independent" slot while mean_reversion
    # keeps its own, so the -1 carries relatively more weight → net < 1.0.
    assert scores["AAPL"] < 0.99


def test_falls_back_without_history():
    bb = _bb()
    ctx = AgentContext(now=NOW, strategy_returns=None)
    scores = net_scores_for_blackboard(bb, ctx, OPTIMAL)
    # No history → confidence-weighted mean: (2 + 2 - 1) / 3 = 1.0.
    assert scores["AAPL"] == 1.0


def test_history_changes_the_score():
    """The two paths must differ — proof the history is genuinely consumed."""
    bb = _bb()
    with_hist = net_scores_for_blackboard(
        bb, AgentContext(now=NOW, strategy_returns=_history()), OPTIMAL
    )
    without = net_scores_for_blackboard(
        bb, AgentContext(now=NOW, strategy_returns=None), OPTIMAL
    )
    assert with_hist["AAPL"] != without["AAPL"]
