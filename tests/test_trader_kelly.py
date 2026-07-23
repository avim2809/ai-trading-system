"""Tests for the Kelly allocation method in firm.agents.trader.TraderAgent."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from firm.agents.base import AgentContext
from firm.agents.trader import TraderAgent
from firm.contracts.models import DebateResult

NOW = datetime(2026, 1, 2)


def _prices_from_returns(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.cumprod(1.0 + returns)


class _PitView:
    """Returns a positive-edge series for WIN and a negative-edge one for LOSE."""

    asof = NOW

    def __init__(self):
        rng = np.random.default_rng(0)
        n = 252
        # Positive edge: p_win high and avg win > avg loss.
        win_ret = rng.choice([0.02, -0.01], size=n, p=[0.6, 0.4])
        # Negative edge: mostly losses.
        lose_ret = rng.choice([0.01, -0.02], size=n, p=[0.4, 0.6])
        dates = pd.date_range("2025-01-01", periods=n, freq="D")
        frames = []
        for sym, ret in (("WIN", win_ret), ("LOSE", lose_ret)):
            px = _prices_from_returns(ret)
            frames.append(pd.DataFrame({
                "date": dates, "symbol": sym, "close": px, "adj_close": px,
            }))
        self._df = pd.concat(frames, ignore_index=True)

    def prices(self, symbols=None, lookback_days=252):
        return self._df[self._df["symbol"].isin(symbols)]


class TestKellyAllocation:
    def test_positive_edge_gets_weight_negative_edge_zero(self):
        trader = TraderAgent(config={"allocation_method": "kelly", "kelly_fraction": 0.5})
        ctx = AgentContext(now=NOW, pit_view=_PitView())
        results = [
            DebateResult(symbol="WIN", net_conviction=0.5),
            DebateResult(symbol="LOSE", net_conviction=0.5),
        ]
        proposal = trader.run(ctx, debate_results=results)
        assert proposal.targets.get("WIN", 0.0) > 0.0
        assert proposal.targets.get("LOSE", 0.0) == 0.0

    def test_conviction_sign_respected(self):
        trader = TraderAgent(config={"allocation_method": "kelly"})
        ctx = AgentContext(now=NOW, pit_view=_PitView())
        results = [DebateResult(symbol="WIN", net_conviction=-0.5)]
        proposal = trader.run(ctx, debate_results=results)
        # A short conviction on a positive-edge name → negative weight.
        assert proposal.targets["WIN"] < 0.0

    def test_full_kelly_edge_value(self):
        # p=0.6, b=2 → f = (0.6*2 - 0.4)/2 = 0.4
        pv = _PitView()
        edge = TraderAgent._kelly_edge(pv, "WIN")
        assert edge is not None
        assert edge > 0.0

    def test_no_history_falls_back_to_conviction(self):
        trader = TraderAgent(config={"allocation_method": "kelly"})
        ctx = AgentContext(now=NOW, pit_view=None)
        results = [
            DebateResult(symbol="AAPL", net_conviction=0.6),
            DebateResult(symbol="GOOG", net_conviction=-0.4),
        ]
        proposal = trader.run(ctx, debate_results=results)
        # Falls back to conviction weighting → both names allocated.
        assert proposal.targets["AAPL"] > 0.0
        assert proposal.targets["GOOG"] < 0.0
